"""Fault trace instrumentation for SWE dry-run mode.

Instruments callables identified from the golden patch so we capture structured
call chains from test (symptom) to buggy callable (root cause) when running
the test harness on unmodified code.

Supports both SWE-Bench Verified (unified diff ``patch`` field) and R2E-Gym
(structured ``parsed_commit_content`` with ``file_diffs``).

Callable detection uses **AST comparison**: parse both the pre-patch (old) and
post-patch (new) source with ``ast``, then identify callables that appear in
both versions but with different source — these are the *in-place* modified
callables worth tracing.  Pure additions (only in new) and pure deletions
(only in old) are excluded.
"""

from __future__ import annotations

import ast
import base64
import io
import json
import logging
import re
import textwrap
import threading
import tokenize
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# CPython's ast.parse() uses a global recursion depth counter that is not
# thread-safe.  Under heavy ThreadPoolExecutor concurrency (512 workers)
# this causes sporadic "AST constructor recursion depth mismatch" errors.
_ast_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Patch extraction — unified entry point for both datasets
# ---------------------------------------------------------------------------


def extract_non_test_patch(task: dict) -> str:
    """Return a unified-diff string of non-test changes for the given task.

    Works with both dataset formats:
      - **SWE-Bench Verified**: ``task["patch"]`` already contains only non-test
        code changes (test changes live in ``task["test_patch"]``).
      - **R2E-Gym**: no ``patch`` field; we reconstruct a unified diff from
        ``parsed_commit_content.file_diffs``, keeping only files listed in
        ``relevant_files`` (which excludes tests and docs).
    """
    # SWE-Bench path
    if "patch" in task and task["patch"]:
        return task["patch"]

    # R2E-Gym path
    pcc_raw = task.get("parsed_commit_content")
    if not pcc_raw:
        return ""
    return _reconstruct_patch_from_r2e(pcc_raw, task)


def _reconstruct_patch_from_r2e(pcc_raw: str | dict, task: dict) -> str:
    """Reconstruct a unified diff from R2E-Gym structured commit data.

    Filters to non-test ``.py`` files using ``relevant_files``.  Falls back to
    a path-based heuristic when ``relevant_files`` is absent.
    """
    if isinstance(pcc_raw, str):
        try:
            pcc = json.loads(pcc_raw)
        except (json.JSONDecodeError, TypeError):
            return ""
    else:
        pcc = pcc_raw

    file_diffs = pcc.get("file_diffs", [])
    if not file_diffs:
        return ""

    # Build allow-set of non-test .py files
    relevant = task.get("relevant_files")
    if relevant is not None:
        # relevant_files is typically a list/ndarray of non-test .py paths
        allow_set: set[str] | None = set(relevant)
    else:
        allow_set = None

    parts: list[str] = []

    for fd in file_diffs:
        header = fd.get("header", {})
        file_path = header.get("file", {}).get("path", "")
        if not file_path or not file_path.endswith(".py"):
            continue

        # Filter: use allow-set if available, otherwise path heuristic
        if allow_set is not None:
            if file_path not in allow_set:
                continue
        else:
            if _is_test_file(file_path):
                continue

        minus = fd.get("minus_file", {}).get("path", f"a/{file_path}")
        plus = fd.get("plus_file", {}).get("path", f"b/{file_path}")
        hunks = fd.get("hunks", [])
        if not hunks:
            continue

        diff_lines: list[str] = []
        diff_lines.append(f"diff --git {minus} {plus}")
        diff_lines.append(f"--- {minus}")
        diff_lines.append(f"+++ {plus}")

        for hunk in hunks:
            desc = hunk.get("descriptor", {})
            old_r = desc.get("old_range", {})
            new_r = desc.get("new_range", {})
            old_start = old_r.get("start", 0)
            old_len = old_r.get("length", 0)
            new_start = new_r.get("start", 0)
            new_len = new_r.get("length", 0)
            section = desc.get("section", "")
            header_line = f"@@ -{old_start},{old_len} +{new_start},{new_len} @@"
            if section:
                header_line += f" {section}"
            diff_lines.append(header_line)

            for line_info in hunk.get("line_group", {}).get("all_lines", []):
                content = line_info.get("content", "")
                line_type = line_info.get("type", "context")
                if line_type == "context":
                    diff_lines.append(f" {content}")
                elif line_type == "deleted":
                    diff_lines.append(f"-{content}")
                elif line_type == "added":
                    diff_lines.append(f"+{content}")

        parts.append("\n".join(diff_lines))

    return "\n".join(parts)


def _is_test_file(path: str) -> bool:
    """Heuristic: does this path look like a test file?"""
    normalized = path.replace("\\", "/")
    parts = normalized.split("/")
    filename = parts[-1] if parts else normalized

    # R2E test helper/runner files are test infrastructure even when their
    # basename is not test_*.py.  Keeping these out of the bonus path prevents
    # agents from receiving process reward for reading the harness itself.
    if normalized.startswith("r2e_tests/"):
        if filename in {"test.py", "tests.py"}:
            return True
        if filename.startswith("test_") or filename.endswith("_test.py"):
            return True
        if filename in {"helper.py", "conftest.py", "pytest_plugin.py", "pytest_plugins.py"}:
            return True
        if filename.endswith("_runner.py"):
            return True

    if filename in {"test.py", "tests.py", "conftest.py", "pytest_plugin.py", "pytest_plugins.py"}:
        return True
    if filename.endswith("_runner.py"):
        return True

    for p in parts:
        if p in ("tests", "test", "testing"):
            return True
        if p.startswith("test_") or p.endswith("_test.py"):
            return True
    return False


def _test_file_reason(path: str) -> str | None:
    """Return the test-suite/harness pattern matched by *path*."""
    normalized = path.replace("\\", "/").lstrip("./")
    parts = normalized.split("/")
    filename = parts[-1] if parts else normalized

    if normalized.startswith("r2e_tests/"):
        return "r2e_tests/**"
    if normalized.startswith("django/test/"):
        return "django/test/**"

    for idx, part in enumerate(parts):
        if part in ("tests", "test", "testing"):
            return "/".join(parts[: idx + 1]) + "/**"
        if part.startswith("test_"):
            return "/".join(parts[:idx] + ["test_*.py"]) if idx else "test_*.py"
        if part.endswith("_test.py"):
            return "/".join(parts[:idx] + ["*_test.py"]) if idx else "*_test.py"

    if filename in {"test.py", "tests.py", "conftest.py", "pytest_plugin.py", "pytest_plugins.py"}:
        return filename
    if filename.endswith("_runner.py"):
        return "*_runner.py"
    return None


# ---------------------------------------------------------------------------
# Task normalisation — flatten SWE-bench verl wrapper
# ---------------------------------------------------------------------------


def normalize_task(task: dict) -> dict:
    """Ensure all dataset fields are top-level keys.

    SWE-Bench Verified data (loaded via ``Dataset.load_data``) wraps the real
    fields inside a JSON string called ``extra_info``.  R2E-Gym data is
    already flat.  This function detects and unpacks the wrapper so that
    downstream code can use ``task["docker_image"]``, ``task["patch"]``, etc.
    regardless of the source dataset.
    """
    extra = task.get("extra_info")
    if extra is None:
        return task

    if isinstance(extra, str):
        try:
            extra = json.loads(extra)
        except (json.JSONDecodeError, TypeError):
            return task
    if isinstance(extra, dict):
        merged = {**extra, **task}
        merged["extra_info"] = extra
        return merged
    return task


def make_instance_id(task: dict) -> str:
    """Generate a human-readable instance ID from a task dict.

    Handles both dataset formats:
      - **SWE-Bench Verified**: already has ``instance_id`` like
        ``astropy__astropy-12907`` — returned as-is.
      - **R2E-Gym**: has ``repo_name`` + ``commit_hash`` — combined as
        ``{repo_name}__{commit_hash[:8]}``, e.g. ``orange3__2d9617bd``.
    """
    # SWE-Bench: instance_id already readable
    iid = task.get("instance_id")
    if iid and not _looks_like_bare_hash(iid):
        return iid

    # R2E-Gym: repo_name + short commit hash
    repo = task.get("repo_name", "")
    commit = task.get("commit_hash", "")

    if not repo:
        # Fallback: try to extract from docker_image
        # e.g. "namanjain12/orange3_final:2d9617bd..." → "orange3"
        docker = task.get("docker_image", "")
        if "/" in docker:
            image_part = docker.split("/", 1)[1]  # "orange3_final:hash"
            image_name = image_part.split(":")[0]  # "orange3_final"
            if image_name.endswith("_final"):
                repo = image_name[: -len("_final")]
            else:
                repo = image_name

    if repo and commit:
        return f"{repo}__{commit[:8]}"
    if commit:
        return commit[:12]
    if iid:
        return iid
    return "unknown"


def _looks_like_bare_hash(s: str) -> bool:
    """Return True if *s* looks like a bare hex hash (no separators)."""
    return len(s) >= 20 and all(c in "0123456789abcdef" for c in s)


# ---------------------------------------------------------------------------
# 1. AST-based callable extraction
# ---------------------------------------------------------------------------


@dataclass
class CallableInfo:
    """Metadata for a single callable extracted from AST."""

    name: str
    qualified_name: str
    file_path: str
    start_line: int  # ``def`` line (1-based)
    end_line: int  # last line of the callable body
    source: str  # verbatim source text for equality comparison

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
        }


_TOLERANT_CLASS_RE = re.compile(r"^([ \t]*)class\s+([A-Za-z_][A-Za-z0-9_]*)\b")
_TOLERANT_DEF_RE = re.compile(r"^([ \t]*)(?:async\s+def|def)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def _line_indent(line: str) -> int:
    return len(line.expandtabs(8)) - len(line.lstrip(" \t").expandtabs(8))


def _assemble_qualified_name(
    stack: list[tuple[str, str, int]],
    leaf_name: str,
) -> str:
    parts: list[str] = []
    for kind, name, _ in stack:
        parts.append(name)
        if kind == "func":
            parts.append("<locals>")
    parts.append(leaf_name)
    return ".".join(parts)


def _find_tolerant_block_end(lines: list[str], start_line: int, indent: int) -> int:
    end = len(lines)
    for line_no in range(start_line + 1, len(lines) + 1):
        line = lines[line_no - 1]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if _line_indent(line) <= indent:
            end = line_no - 1
            break
    return max(start_line, end)


def extract_callables_tolerant(
    source: str,
    file_path: str,
) -> dict[str, CallableInfo]:
    """Extract callable definitions without parsing full Python syntax."""
    lines = source.splitlines()
    callables: dict[str, CallableInfo] = {}
    stack: list[tuple[str, str, int]] = []

    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("@"):
            continue
        indent = _line_indent(line)
        while stack and indent <= stack[-1][2]:
            stack.pop()

        class_match = _TOLERANT_CLASS_RE.match(line)
        if class_match:
            stack.append(("class", class_match.group(2), indent))
            continue

        def_match = _TOLERANT_DEF_RE.match(line)
        if not def_match:
            continue
        name = def_match.group(2)
        qualified = _assemble_qualified_name(stack, name)
        end_line = _find_tolerant_block_end(lines, line_no, indent)
        snippet = "\n".join(lines[line_no - 1 : end_line])
        callables[qualified] = CallableInfo(
            name=name,
            qualified_name=qualified,
            file_path=file_path,
            start_line=line_no,
            end_line=end_line,
            source=snippet,
        )
        stack.append(("func", name, indent))
    return callables


def extract_callables_with_tolerant_fallback(
    source: str,
    file_path: str,
) -> dict[str, CallableInfo]:
    """Use AST extraction, falling back to indentation-based callable discovery."""
    callables = extract_callables_from_ast(source, file_path)
    if callables:
        return callables
    return extract_callables_tolerant(source, file_path)


def extract_callables_from_ast(
    source: str,
    file_path: str,
) -> dict[str, CallableInfo]:
    """Parse *source* with :mod:`ast` and return every callable definition.

    Returns ``{qualified_name: CallableInfo}``.  Nested functions and class
    methods use Python-style lexical ``__qualname__`` strings, including
    ``.<locals>.`` separators for function scopes.

    Possible errors:

    * **SyntaxError** — the source uses syntax unsupported by the running
      Python version, or is simply broken.  Returns ``{}`` in that case
      (same behaviour as the old ``find_modified_callables``).
    """
    try:
        with _ast_lock:
            tree = ast.parse(source)
    except SyntaxError:
        logger.warning("SyntaxError while parsing %s — skipping", file_path)
        return {}

    lines = source.splitlines()
    callables: dict[str, CallableInfo] = {}

    def _visit(node: ast.AST, stack: list[tuple[str, str, int]]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                _visit(child, stack + [("class", child.name, -1)])
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                qualified = _assemble_qualified_name(stack, child.name)
                start = child.lineno
                end = child.end_lineno or child.lineno
                snippet = "\n".join(lines[start - 1 : end])
                callables[qualified] = CallableInfo(
                    name=child.name,
                    qualified_name=qualified,
                    file_path=file_path,
                    start_line=start,
                    end_line=end,
                    source=snippet,
                )
                _visit(child, stack + [("func", child.name, -1)])
            else:
                _visit(child, stack)

    _visit(tree, [])
    return callables


# ---------------------------------------------------------------------------
# 2. AST-diff: find in-place modified callables
# ---------------------------------------------------------------------------


def find_modified_callables_from_sources(
    old_source: str,
    new_source: str,
    file_path: str,
) -> list[dict]:
    """Compare *old* and *new* ASTs to find callables modified **in-place**.

    A callable is "modified in-place" when it appears in **both** versions
    (matched by qualified name) but its source text differs.  This naturally
    excludes:

    * **Pure additions** — callable only in new → not instrumentable on the
      old (pre-fix) code.
    * **Pure deletions** — callable only in old → removed by the fix, not a
      target for tracing.

    Returns a list of dicts (``name``, ``file_path``, ``start_line``,
    ``end_line``, ``qualified_name``) referring to the **old** version —
    because that is the version present in the sandbox when we instrument.
    """
    old_callables = extract_callables_from_ast(old_source, file_path)
    new_callables = extract_callables_from_ast(new_source, file_path)
    old_lines = old_source.splitlines()
    new_lines = new_source.splitlines()

    def _nested_children(
        info: CallableInfo,
        callables: dict[str, CallableInfo],
    ) -> list[CallableInfo]:
        return [child for child in callables.values() if info.start_line < child.start_line and child.end_line <= info.end_line]

    def _own_source(
        info: CallableInfo,
        children: list[CallableInfo],
        lines: list[str],
    ) -> str:
        own_lines = set(range(info.start_line, info.end_line + 1))
        for child in children:
            own_lines.difference_update(range(child.start_line, child.end_line + 1))
        return "\n".join(lines[i - 1] for i in sorted(own_lines) if 1 <= i <= len(lines))

    modified: list[dict] = []
    for qname, old_info in old_callables.items():
        new_info = new_callables.get(qname)
        if new_info is None:
            continue
        old_own_source = _own_source(
            old_info,
            _nested_children(old_info, old_callables),
            old_lines,
        )
        new_own_source = _own_source(
            new_info,
            _nested_children(new_info, new_callables),
            new_lines,
        )
        if old_own_source != new_own_source:
            entry = old_info.to_dict()
            entry["source"] = old_info.source
            modified.append(entry)
    return modified


def find_modified_callables_from_task(task: dict) -> list[dict]:
    """Extract modified callables from a task dict's file_diffs.

    Convenience wrapper that parses ``parsed_commit_content`` (R2E-Gym) or
    ``parsed_commit`` (SWE-Bench), compares old/new ASTs for each non-test
    ``.py`` file, and returns all in-place modified callables.
    """
    for key in ("parsed_commit_content", "parsed_commit"):
        raw = task.get(key)
        if not raw:
            continue
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
        if not isinstance(raw, dict):
            continue
        file_diffs = raw.get("file_diffs", [])
        if not file_diffs:
            continue

        relevant = task.get("relevant_files")
        allow_set = set(relevant) if relevant is not None else None

        all_modified: list[dict] = []
        for fd in file_diffs:
            path = fd.get("header", {}).get("file", {}).get("path", "")
            if not path or not path.endswith(".py"):
                continue
            if allow_set is not None:
                if path not in allow_set:
                    continue
            else:
                if _is_test_file(path):
                    continue

            old_src = fd.get("old_file_content") or ""
            new_src = fd.get("new_file_content") or ""
            if old_src and new_src:
                all_modified.extend(find_modified_callables_from_sources(old_src, new_src, path))
        return all_modified

    return []


# ---------------------------------------------------------------------------
# 3. generate_tracer_module
# ---------------------------------------------------------------------------


def generate_tracer_module(repo_path: str, alt_path: str = "") -> str:
    """Generate _swe_fault_tracer.py source to be deployed in sandbox site-packages.

    The tracer captures call stacks when instrumented callables are invoked,
    filters to repo-internal frames, deduplicates, and writes structured JSONL
    to ``/tmp/_swe_fault_traces.jsonl``.

    Writing to a file bypasses pytest's fd-level capture of stderr, which was
    swallowing all trace output in 31/32 tested instances.

    Args:
        repo_path: Primary repo path in sandbox (e.g. ``/testbed``).
        alt_path: Alternate root where test files may live (e.g. ``/root``
            for R2E-Gym, where ``r2e_tests/`` is moved to ``/root/r2e_tests``
            and symlinked back). Needed because Python's ``co_filename``
            resolves symlinks, so test frames show the real path under
            ``alt_path`` rather than the symlink under ``repo_path``.
    """
    # Build the list of allowed path prefixes.
    # alt_path (e.g. "/root") is too broad on its own — it would match
    # stdlib under /root/.local/share/uv/python/.  Only add the specific
    # subdirectory that holds symlinked test files.
    prefixes = [repo_path]
    if alt_path and alt_path != repo_path:
        prefixes.append(alt_path + "/r2e_tests")
    prefixes_tuple = repr(tuple(prefixes))

    return textwrap.dedent(f"""\
        import json
        import os
        import sys
        import threading

        _lock = threading.Lock()
        _seen = set()
        _REPO_PATH = "{repo_path}"
        _PATH_PREFIXES = {prefixes_tuple}
        _TRACE_FILE = "{TRACE_FILE_PATH}"
        def _env_int(name, default):
            try:
                value = int(os.environ.get(name, str(default)))
            except ValueError:
                return default
            return value if value >= 0 else default

        _MAX_EVENTS = _env_int("P2A_TRACE_MAX_EVENTS", 10000)
        _MAX_FRAMES = _env_int("P2A_TRACE_MAX_FRAMES", 80)

        def _is_test_file_path(path):
            normalized = path.replace("\\\\", "/")
            parts = normalized.split("/")
            filename = parts[-1] if parts else normalized
            if "/r2e_tests/" in normalized:
                return True
            if filename in {{"conftest.py", "pytest_plugin.py", "pytest_plugins.py"}}:
                return True
            if filename.startswith("test_") or filename.endswith("_test.py"):
                return True
            return any(part in ("tests", "test", "testing") for part in parts)

        def _resolve_qualname(frame):
            code = frame.f_code
            self_obj = frame.f_locals.get("self")
            if self_obj is not None:
                return type(self_obj).__qualname__ + "." + code.co_name
            cls_obj = frame.f_locals.get("cls")
            if cls_obj is not None:
                cls = cls_obj if isinstance(cls_obj, type) else type(cls_obj)
                return cls.__qualname__ + "." + code.co_name

            # Static methods and nested free functions have no self/cls.  Use
            # the lexical qualname from the code object instead of walking the
            # runtime caller chain; the latter pollutes names with pytest hook
            # frames and can inflate hop distances by hundreds of frames.
            qualname = getattr(code, "co_qualname", None)
            if qualname and qualname != "<module>":
                return qualname
            if code.co_name and code.co_name != "<module>":
                return code.co_name
            return "<unknown>"

        def trace(callable_name, file_path, def_lineno):
            try:
                # INVARIANT: frame list is OUTER -> INNER. frames[0] is the outermost caller (test entry).
                # frames[-1] is the innermost callee (the instrumented patched callable). build_call_graph_from_traces depends on this.
                frames = []
                frame = sys._getframe(1)
                while frame is not None:
                    frames.append(frame)
                    frame = frame.f_back
                frames.reverse()

                repo_frames = [
                    f for f in frames
                    if any(f.f_code.co_filename.startswith(p) for p in _PATH_PREFIXES)
                    and f.f_code.co_name != "<module>"
                    and "/.venv/" not in f.f_code.co_filename
                    and "/site-packages/" not in f.f_code.co_filename
                ]
                if not repo_frames:
                    return
                test_frame_indexes = [
                    i for i, f in enumerate(repo_frames)
                    if _is_test_file_path(f.f_code.co_filename)
                ]
                if test_frame_indexes:
                    repo_frames = repo_frames[test_frame_indexes[0]:]
                if _MAX_FRAMES > 0 and len(repo_frames) > _MAX_FRAMES:
                    repo_frames = repo_frames[:1] + repo_frames[-(_MAX_FRAMES - 1):]

                frame_entries = [
                    {{
                        "file": f.f_code.co_filename,
                        "line": f.f_lineno,
                        "name": _resolve_qualname(f),
                    }}
                    for f in repo_frames
                ]
                key = (callable_name, file_path, def_lineno,
                       tuple((f["file"], f["line"], f["name"]) for f in frame_entries))
                with _lock:
                    if key in _seen:
                        return
                    if len(_seen) >= _MAX_EVENTS:
                        return
                    _seen.add(key)
                    entry = {{
                        "callable": callable_name,
                        "file": file_path,
                        "lineno": def_lineno,
                        "frames": frame_entries,
                    }}
                    with open(_TRACE_FILE, "a") as fh:
                        fh.write(json.dumps(entry) + "\\n")
            except Exception:
                pass  # never crash the instrumented function
    """)


# ---------------------------------------------------------------------------
# 4. instrument_source
# ---------------------------------------------------------------------------


def _find_signature_colon(line_text: str) -> int:
    """Find the colon that terminates a one-line function signature."""
    paren_depth = 0
    for idx, ch in enumerate(line_text):
        if ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth -= 1
        elif ch == "#":
            break
        elif ch == ":" and paren_depth <= 0:
            return idx
    return -1


def _line_text_and_ending(line: str) -> tuple[str, str]:
    """Split one physical source line into text and its original line ending."""
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n"):
        return line[:-1], "\n"
    if line.endswith("\r"):
        return line[:-1], "\r"
    return line, "\n"


def _split_oneline_simple_suite(suite: str) -> list[str]:
    """Split a one-line simple suite into semicolon-separated statements."""
    statements: list[str] = []
    start_col = 0
    depth = 0

    try:
        tokens = tokenize.generate_tokens(io.StringIO(suite).readline)
        for token in tokens:
            if token.type != tokenize.OP:
                continue
            if token.string in "([{":
                depth += 1
            elif token.string in ")]}":
                depth = max(0, depth - 1)
            elif token.string == ";" and depth == 0:
                statement = suite[start_col : token.start[1]].strip()
                if statement:
                    statements.append(statement)
                start_col = token.end[1]
    except tokenize.TokenError:
        return [part.strip() for part in suite.split(";") if part.strip()]

    tail = suite[start_col:].strip()
    if tail:
        statements.append(tail)
    return statements


def _rewrite_oneline_suite(lines: list[str], func_lineno: int) -> int:
    """Expand ``def f(): stmt`` into a multi-line suite in-place.

    Returns the number of additional physical lines inserted.
    """
    line_idx = func_lineno - 1
    if line_idx < 0 or line_idx >= len(lines):
        return 0

    line_text, line_ending = _line_text_and_ending(lines[line_idx])
    colon_idx = _find_signature_colon(line_text)
    if colon_idx < 0:
        return 0

    suite = line_text[colon_idx + 1 :].strip()
    if not suite:
        return 0

    signature = line_text[: colon_idx + 1].rstrip()
    def_indent = line_text[: len(line_text) - len(line_text.lstrip())]
    body_indent = def_indent + "    "
    statements = _split_oneline_simple_suite(suite)
    if not statements:
        return 0

    rewritten = [signature + line_ending]
    rewritten.extend(f"{body_indent}{statement}{line_ending}" for statement in statements)
    lines[line_idx : line_idx + 1] = rewritten
    return len(rewritten) - 1


def _find_function_node_at_line(source: str, callable_info: dict) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Find the function node matching the callable metadata."""
    try:
        with _ast_lock:
            tree = ast.parse(source)
    except SyntaxError:
        return None

    target_name = callable_info.get("name")
    target_lineno = callable_info.get("start_line")
    target_qualified_name = callable_info.get("qualified_name")

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if node.lineno != target_lineno:
            continue
        if node.name == target_name:
            return node
        if target_qualified_name and target_qualified_name.split(".")[-1] == node.name:
            return node
    return None


def instrument_source(source: str, callables: list[dict]) -> str:
    """Insert trace calls into source after each callable's def line.

    Sorts by line descending to preserve line numbers during insertion.
    """
    if not callables:
        return source

    lines = source.splitlines(keepends=True)

    try:
        with _ast_lock:
            tree = ast.parse(source)
    except SyntaxError:
        logger.warning("SyntaxError while parsing source for instrumentation")
        return source

    nodes_by_key: dict[tuple[int, str], ast.FunctionDef | ast.AsyncFunctionDef] = {}
    nodes_by_line: dict[int, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}

    def _visit(node: ast.AST, class_name: str | None = None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                _visit(child, class_name=child.name)
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                qualified = f"{class_name}.{child.name}" if class_name else child.name
                nodes_by_key[(child.lineno, qualified)] = child
                nodes_by_key[(child.lineno, child.name)] = child
                nodes_by_line.setdefault(child.lineno, []).append(child)
                _visit(child, class_name=class_name)

    def _is_docstring_stmt(node: ast.AST) -> bool:
        return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)

    def _line_ending(line: str) -> str:
        if line.endswith("\r\n"):
            return "\r\n"
        if line.endswith("\n"):
            return "\n"
        return ""

    def _module_prelude_insert_idx() -> int:
        """Return a safe insertion point after docstring/future imports."""
        idx = 0
        if lines and lines[0].startswith("#!"):
            idx = 1
        coding_re = re.compile(r"coding[:=]\s*[-\w.]+")
        if idx < len(lines) and coding_re.search(lines[idx]):
            idx += 1

        body_idx = 0
        if tree.body and _is_docstring_stmt(tree.body[0]):
            idx = max(idx, tree.body[0].end_lineno or tree.body[0].lineno)
            body_idx = 1

        while body_idx < len(tree.body):
            node = tree.body[body_idx]
            if not (isinstance(node, ast.ImportFrom) and node.module == "__future__"):
                break
            idx = max(idx, node.end_lineno or node.lineno)
            body_idx += 1

        return idx

    def _module_trace_prelude(ending: str) -> str:
        return (
            f"{ending}"
            f"try:{ending}"
            f"    import _swe_fault_tracer as _p2a_ft{ending}"
            f"except Exception:{ending}"
            f"    _p2a_ft = None{ending}"
        )

    _visit(tree)

    # Sort by start_line descending so insertions don't shift earlier lines
    sorted_callables = sorted(callables, key=lambda c: c["start_line"], reverse=True)
    inserted_trace = False

    for c in sorted_callables:
        function_node = _find_function_node_at_line("".join(lines), c)
        if function_node and function_node.body and function_node.body[0].lineno == function_node.lineno:
            _rewrite_oneline_suite(lines, c["start_line"])

        func_line_idx = c["start_line"] - 1  # 0-based index

        if func_line_idx >= len(lines):
            continue

        key = (c["start_line"], c.get("qualified_name") or c.get("name", ""))
        func_node = nodes_by_key.get(key)
        if func_node is None:
            line_matches = nodes_by_line.get(c["start_line"], [])
            if len(line_matches) != 1:
                continue
            func_node = line_matches[0]
        if not func_node.body:
            continue

        first_stmt = func_node.body[0]
        def_line = lines[func_line_idx]
        def_indent = def_line[: len(def_line) - len(def_line.lstrip())]
        body_indent = def_indent + "    "
        insert_idx = first_stmt.lineno - 1

        if first_stmt.lineno == func_node.lineno:
            line = lines[func_line_idx]
            ending = _line_ending(line)
            rewritten_ending = ending or "\n"
            line_without_ending = line[: -len(ending)] if ending else line
            header = line_without_ending[: first_stmt.col_offset].rstrip()
            body_text = line_without_ending[first_stmt.col_offset :].lstrip()
            lines[func_line_idx] = f"{header}{rewritten_ending}"
            insert_idx = func_line_idx + 1

            if _is_docstring_stmt(first_stmt):
                doc_end_col = first_stmt.end_col_offset or first_stmt.col_offset
                doc_text = line_without_ending[first_stmt.col_offset : doc_end_col].lstrip()
                remainder = line_without_ending[doc_end_col:].lstrip()
                if remainder.startswith(";"):
                    remainder = remainder[1:].lstrip()
                lines.insert(insert_idx, f"{body_indent}{doc_text}{rewritten_ending}")
                insert_idx += 1
                if remainder:
                    lines.insert(insert_idx, f"{body_indent}{remainder}{rewritten_ending}")
            else:
                lines.insert(insert_idx, f"{body_indent}{body_text}{ending}")
        else:
            body_line_idx = first_stmt.lineno - 1
            if body_line_idx < len(lines):
                body_line = lines[body_line_idx]
                body_indent = body_line[: first_stmt.col_offset]

            if _is_docstring_stmt(first_stmt):
                insert_idx = first_stmt.end_lineno or first_stmt.lineno

        # Check if already instrumented
        if insert_idx < len(lines) and "_swe_fault_tracer" in lines[insert_idx]:
            continue

        # Wrap in try/except so import or trace failures don't crash the
        # instrumented function — the function must still behave normally
        # (tests should fail for the original bug, not our instrumentation).
        trace_line = f'{body_indent}try:\n{body_indent}    _ft = globals().get("_p2a_ft") or __import__("_swe_fault_tracer"); _ft.trace("{c["qualified_name"]}", "{c["file_path"]}", {c["start_line"]})\n{body_indent}except Exception:\n{body_indent}    pass\n'

        lines.insert(insert_idx, trace_line)
        inserted_trace = True

    if inserted_trace and "_p2a_ft" not in source:
        ending = "\n"
        for line in lines:
            ending = _line_ending(line) or ending
            if ending:
                break
        lines.insert(_module_prelude_insert_idx(), _module_trace_prelude(ending))

    return "".join(lines)


# ---------------------------------------------------------------------------
# 5. instrument_sandbox
# ---------------------------------------------------------------------------


def _read_sandbox_file(env: Any, full_path: str, b64_chunk_lines: int = 50) -> tuple[str, int]:
    """Read a file from the sandbox without ARL gateway output truncation.

    Plain ``cat`` of large files is truncated by the gateway response size
    limit.  The gateway intermittently enforces a ~4097 byte stdout cap, so
    each chunk must stay well below that.  At 50 base64 lines × 77 bytes
    (76 chars + newline) = 3850 bytes per chunk, safely under the limit.

    Returns ``(content, exit_code)`` where exit_code=0 on success.
    """
    B64_TMP = "/tmp/_swe_b64_tmp.txt"

    # Encode file as base64 into a temp file (stdout → file, no output)
    _, _, enc_exit = env._execute_raw(f"base64 {full_path} > {B64_TMP} 2>/dev/null")
    if enc_exit != 0:
        return "", enc_exit

    # Get expected file size for post-read verification
    wc_bytes_out, _, _ = env._execute_raw(f"wc -c {full_path}")
    try:
        expected_bytes = int(wc_bytes_out.strip().split()[0])
    except (ValueError, IndexError):
        expected_bytes = -1

    # Number of base64 lines (each is 76 chars; ~57 bytes decoded per line)
    wc_out, _, _ = env._execute_raw(f"wc -l {B64_TMP}")
    try:
        b64_total = int(wc_out.strip().split()[0])
    except (ValueError, IndexError):
        return "", 1

    # Read base64 in chunks and decode each chunk
    decoded_chunks: list[bytes] = []
    for start in range(1, b64_total + 2, b64_chunk_lines):
        end = start + b64_chunk_lines - 1
        expected_lines = min(b64_chunk_lines, b64_total - start + 1)
        chunk_b64, _, chunk_exit = env._execute_raw(f"sed -n '{start},{end}p' {B64_TMP}")
        if chunk_exit not in (0, 1):
            return "", chunk_exit
        clean = chunk_b64.replace("\n", "").replace("\r", "").replace(" ", "")
        if not clean:
            continue

        # Detect gateway truncation: fewer lines than expected
        actual_lines = chunk_b64.count("\n") + (1 if chunk_b64 and not chunk_b64.endswith("\n") else 0)
        if actual_lines < expected_lines:
            logger.warning(
                "Gateway truncation at lines %d-%d of %s: got %d/%d lines (%d bytes). Retrying with smaller chunks.",
                start,
                end,
                full_path,
                actual_lines,
                expected_lines,
                len(chunk_b64),
            )
            # Retry the entire file with half-sized chunks
            env._run(f"rm -f {B64_TMP}")
            if b64_chunk_lines > 10:
                return _read_sandbox_file(env, full_path, b64_chunk_lines=b64_chunk_lines // 2)
            else:
                logger.warning("Chunk size too small, giving up on %s", full_path)
                return "", 1

        try:
            padding = (4 - len(clean) % 4) % 4
            decoded_chunks.append(base64.b64decode(clean + "=" * padding))
        except Exception as exc:
            logger.warning("base64 decode error at lines %d-%d of %s: %s", start, end, full_path, exc)
            return "", 1

    env._run(f"rm -f {B64_TMP}")
    content = b"".join(decoded_chunks)

    # Verify completeness
    if expected_bytes >= 0 and len(content) != expected_bytes:
        logger.warning(
            "Size mismatch for %s: expected %d bytes, got %d. Retrying with smaller chunks.",
            full_path,
            expected_bytes,
            len(content),
        )
        if b64_chunk_lines > 10:
            return _read_sandbox_file(env, full_path, b64_chunk_lines=b64_chunk_lines // 2)
        return "", 1

    return content.decode("utf-8", errors="replace"), 0


def _get_patched_py_files(patch_text: str) -> set[str]:
    """Return the set of ``.py`` files touched by *patch_text*.

    Includes files from both ``--- a/`` and ``+++ b/`` headers so that pure
    additions (old is ``/dev/null``) and pure deletions (new is ``/dev/null``)
    are captured.  The caller decides which to instrument.
    """
    files: set[str] = set()
    for m in re.finditer(r"^--- a/(.+\.py)\s*$", patch_text, re.MULTILINE):
        p = m.group(1)
        if p != "/dev/null":
            files.add(p)
    for m in re.finditer(r"^\+\+\+ b/(.+\.py)\s*$", patch_text, re.MULTILINE):
        p = m.group(1)
        if p != "/dev/null":
            files.add(p)
    return files


def _get_new_contents_from_task(
    entry: dict,
    files: set[str],
) -> dict[str, str]:
    """Try to obtain post-patch file contents from the task dict.

    Works with both dataset formats:

    * **R2E-Gym**: ``parsed_commit_content`` → ``file_diffs`` → ``new_file_content``
    * **SWE-Bench Verified**: ``parsed_commit`` → ``file_diffs`` → ``new_file_content``
    """
    # Try both field names
    pcc_raw = entry.get("parsed_commit_content") or entry.get("parsed_commit")
    if not pcc_raw:
        return {}

    if isinstance(pcc_raw, str):
        try:
            pcc = json.loads(pcc_raw)
        except (json.JSONDecodeError, TypeError):
            return {}
    else:
        pcc = pcc_raw

    result: dict[str, str] = {}
    for fd in pcc.get("file_diffs", []):
        header = fd.get("header", {})
        path = header.get("file", {}).get("path", "")
        if path in files:
            content = fd.get("new_file_content")
            if content:
                result[path] = content
    return result


def _get_new_contents_via_sandbox(
    env: Any,
    patch_text: str,
    files: set[str],
) -> dict[str, str]:
    """Apply the golden patch in the sandbox, read new sources, then revert.

    Fallback for when post-patch content is not available in the task dict
    (e.g. older SWE-Bench snapshots without ``parsed_commit``).
    """
    patch_b64 = base64.b64encode(patch_text.encode()).decode()
    env._run(f"printf '%s' '{patch_b64}' | base64 -d > /tmp/_golden_patch.diff")

    _, err = env._run(f"cd {env.repo_path} && git apply /tmp/_golden_patch.diff")
    if "Error" in str(err):
        logger.warning("Failed to apply golden patch in sandbox: %s", err)
        env._run(f"cd {env.repo_path} && git checkout -- .")
        return {}

    result: dict[str, str] = {}
    for file_path in files:
        full_path = f"{env.repo_path}/{file_path}"
        content, exit_code = _read_sandbox_file(env, full_path)
        if exit_code == 0:
            result[file_path] = content

    env._run(f"cd {env.repo_path} && git checkout -- .")
    return result


def instrument_sandbox(
    env: Any,
    modified_callables: list[dict],
) -> list[dict]:
    """Orchestrate fault tracing instrumentation in the sandbox.

    Called after ``env.reset()`` with the modified-callable list already
    computed by the caller (via ``find_modified_callables_from_sources``).
    Deploys the tracer module and injects trace calls into the sandbox
    source files — **no redundant AST comparison** is performed here.

    Args:
        env: Any with an active sandbox session.
        modified_callables: Pre-computed list of callable dicts (``name``,
            ``file_path``, ``start_line``, ``end_line``, ``qualified_name``).

    Returns the subset of *modified_callables* that were successfully
    instrumented in the sandbox.
    """
    if not modified_callables:
        logger.info("No callables to instrument")
        return []

    # Group callables by file
    callables_by_file: dict[str, list[dict]] = {}
    for c in modified_callables:
        callables_by_file.setdefault(c["file_path"], []).append(c)

    # 1. Find site-packages path & deploy tracer module. The command is reliable
    # on a clean boot but can return empty output under a transient ARL websocket
    # hiccup at high concurrency; retry a few times and never index an empty list.
    site_packages = ""
    site_output = site_err = ""
    for _ in range(3):
        site_output, site_err = env._run('python -c "import site; print(site.getsitepackages()[0])"')
        lines = site_output.strip().splitlines()
        if lines and lines[0].strip():
            site_packages = lines[0].strip()
            break
    if not site_packages or "Error" in str(site_err):
        logger.warning("Failed to find site-packages: %s %s", site_output, site_err)
        return []

    tracer_source = generate_tracer_module(env.repo_path, getattr(env, "alt_path", ""))
    tracer_b64 = base64.b64encode(tracer_source.encode()).decode()
    tracer_dest = f"{site_packages}/_swe_fault_tracer.py"
    env._run(f"printf '%s' '{tracer_b64}' | base64 -d > {tracer_dest}")

    # Verify tracer is importable in the sandbox execution context
    verify_out, verify_err = env._run("python -c \"import _swe_fault_tracer; print('OK')\"")
    if "OK" not in verify_out:
        logger.warning(
            "Tracer module not importable (deployed to %s): %s %s",
            tracer_dest,
            verify_out,
            verify_err,
        )
        return []

    # 2. For each file: read old source from sandbox, inject trace calls
    instrumented_callables: list[dict] = []

    for file_path, callables in callables_by_file.items():
        full_path = f"{env.repo_path}/{file_path}"

        old_source, exit_code = _read_sandbox_file(env, full_path)
        if exit_code != 0:
            logger.warning("Failed to read %s (exit %d)", full_path, exit_code)
            continue

        # Re-locate callables by qualified_name in the actual sandbox source.
        # The line numbers from find_modified_callables_from_task() come from
        # the dataset's old_file_content, which may differ from the sandbox
        # (e.g. docker image built from a different commit).
        sandbox_callables = extract_callables_from_ast(old_source, file_path)
        relocated = []
        for c in callables:
            sb_info = sandbox_callables.get(c["qualified_name"])
            if sb_info is not None:
                relocated.append(
                    {
                        **c,
                        "start_line": sb_info.start_line,
                        "end_line": sb_info.end_line,
                        "source": sb_info.source,
                    }
                )
            else:
                logger.warning(
                    "Callable %s not found in sandbox source %s (sandbox has %d lines, expected line %d). Skipping.",
                    c["qualified_name"],
                    file_path,
                    len(old_source.splitlines()),
                    c["start_line"],
                )
        if not relocated:
            continue
        callables = relocated

        # Instrument the old (pre-fix) source
        instrumented = instrument_source(old_source, callables)
        if instrumented == old_source:
            continue

        # Re-parse instrumented source to get post-instrumentation line ranges.
        # Instrumentation inserts 4 lines (try/import/except/pass) per callable,
        # shifting all subsequent line numbers. Traceback frames will report the
        # shifted (instrumented) line numbers, so we need to match against those.
        instr_callables = extract_callables_from_ast(instrumented, file_path)
        for c in callables:
            instr_info = instr_callables.get(c["qualified_name"])
            if instr_info is not None:
                c["instr_start_line"] = instr_info.start_line
                c["instr_end_line"] = instr_info.end_line
            else:
                # Fallback: estimate shift (+4 lines for the try/import/except/pass)
                c["instr_start_line"] = c["start_line"]
                c["instr_end_line"] = c["end_line"] + 4

        instrumented_callables.extend(callables)

        # Write back via base64
        instr_b64 = base64.b64encode(instrumented.encode()).decode()
        chunk_size = 65536
        if len(instr_b64) <= chunk_size:
            env._run(f"printf '%s' '{instr_b64}' | base64 -d > {full_path}")
        else:
            env._run(f": > {full_path}")
            for i in range(0, len(instr_b64), chunk_size):
                chunk = instr_b64[i : i + chunk_size]
                env._run(f"printf '%s' '{chunk}' | base64 -d >> {full_path}")

    logger.info(
        "Instrumented %d callables across %d files",
        len(instrumented_callables),
        len(callables_by_file),
    )
    return instrumented_callables


# ---------------------------------------------------------------------------
# 6. parse_fault_traces
# ---------------------------------------------------------------------------


def parse_fault_traces(
    raw_output: str,
    modified_callables: list[dict],
    repo_path: str,
) -> list[list[dict]]:
    """Parse <<<FAULT_TRACE_BEGIN/END>>> blocks from raw test output.

    .. deprecated::
        Use :func:`parse_fault_traces_from_file` instead — the tracer now
        writes JSONL to a file rather than stderr markers.

    Each trace is a list of frame dicts with keys: file_path, line_no,
    func_name, line_content, is_patched.

    Only keeps traces that contain at least one patched frame.
    """
    if not raw_output or not modified_callables:
        return []

    # Build line-range lookup for patched detection
    patched_ranges: dict[str, list[tuple[int, int, str]]] = {}
    for c in modified_callables:
        patched_ranges.setdefault(c["file_path"], []).append((c["start_line"], c["end_line"], c["qualified_name"]))

    traces: list[list[dict]] = []

    # Find all FAULT_TRACE blocks
    pattern = re.compile(
        r"<<<FAULT_TRACE_BEGIN:([^:]+):([^:]+):(\d+)>>>\n"
        r"(.*?)"
        r"<<<FAULT_TRACE_END>>>",
        re.DOTALL,
    )

    for match in pattern.finditer(raw_output):
        match.group(1)
        # groups 2, 3 (file, lineno) available for future use
        frames_text = match.group(4).strip()

        if not frames_text:
            continue

        frames: list[dict] = []
        for frame_line in frames_text.splitlines():
            frame_line = frame_line.strip()
            if not frame_line:
                continue

            # Parse: file_path:line_no:func_name:line_content
            parts = frame_line.split(":", 3)
            if len(parts) < 4:
                continue

            frame_file = parts[0]
            try:
                frame_lineno = int(parts[1])
            except ValueError:
                continue
            frame_func = parts[2]
            frame_content = parts[3]

            # Make file_path relative to repo for matching
            rel_path = frame_file
            if frame_file.startswith(repo_path + "/"):
                rel_path = frame_file[len(repo_path) + 1 :]

            # Skip .venv / site-packages frames
            if ".venv/" in rel_path or "site-packages/" in rel_path:
                continue

            # Check if this frame is in a patched callable (line-range matching)
            is_patched = False
            qualified = frame_func  # default to bare name
            if rel_path in patched_ranges:
                for start, end, qname in patched_ranges[rel_path]:
                    if start <= frame_lineno <= end:
                        is_patched = True
                        qualified = qname
                        break

            frames.append(
                {
                    "file_path": rel_path,
                    "line_no": frame_lineno,
                    "func_name": frame_func,
                    "qualified_name": qualified,
                    "line_content": frame_content,
                    "is_patched": is_patched,
                }
            )

        if frames:
            has_patched = any(f["is_patched"] for f in frames)

            if has_patched:
                traces.append(frames)

    return traces


# ---------------------------------------------------------------------------
# 6b. parse_fault_traces_from_file — JSONL-based (preferred)
# ---------------------------------------------------------------------------

TRACE_FILE_PATH = "/root/_p2a_swe_fault_traces.jsonl"


def parse_fault_traces_from_file(
    env: Any,
    modified_callables: list[dict],
    repo_path: str,
    alt_path: str = "",
    *,
    require_patched: bool = True,
    trace_file_path: str = TRACE_FILE_PATH,
) -> list[list[dict]]:
    """Read ``/tmp/_swe_fault_traces.jsonl`` from the sandbox and parse traces.

    Each JSONL line is a dict with keys: ``callable``, ``file``, ``lineno``,
    ``frames`` (list of ``{file, line, name, code}``).

    Returns a list of traces (each trace = list of frame dicts) in the same
    format as :func:`parse_fault_traces`, for compatibility with
    :func:`aggregate_traces` and :func:`build_call_graph_from_traces`.

    By default only traces containing at least one patched frame are returned.
    Debug callers can set ``require_patched=False`` to inspect all raw traces
    emitted by the tracer before the GT/F2P filters are applied.
    """
    if not modified_callables:
        return []

    stdout, exit_code = _read_sandbox_file(env, trace_file_path)
    if exit_code != 0 or not stdout.strip():
        logger.info("No trace file found or empty: exit_code=%d", exit_code)
        return []

    # Build line-range lookup for patched detection.
    # Use instrumented line ranges (instr_start_line / instr_end_line) when
    # available — traceback frames report post-instrumentation line numbers,
    # which are shifted by the 4-line try/import/except/pass blocks inserted
    # by instrument_source().  Fall back to original ranges if the callable
    # was not instrumented (shouldn't happen, but defensive).
    patched_ranges: dict[str, list[tuple[int, int, str]]] = {}
    patched_by_entry: dict[tuple[str, str], dict] = {}
    for c in modified_callables:
        start = c.get("instr_start_line", c["start_line"])
        end = c.get("instr_end_line", c["end_line"])
        patched_ranges.setdefault(c["file_path"], []).append((start, end, c["qualified_name"]))
        patched_by_entry[(c["file_path"], c["qualified_name"])] = c
        patched_by_entry[(c["file_path"], c["name"])] = c

    traces: list[list[dict]] = []

    for line in stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue

        raw_frames = entry.get("frames", [])
        if not raw_frames:
            continue

        frames: list[dict] = []
        for f in raw_frames:
            frame_file = f.get("file", "")
            frame_lineno = f.get("line", 0)
            frame_func = f.get("name", "")
            frame_code = f.get("code", "")

            # Make file_path relative to repo (or alt_path/r2e_tests) for matching
            rel_path = frame_file
            if frame_file.startswith(repo_path + "/"):
                rel_path = frame_file[len(repo_path) + 1 :]
            elif alt_path and frame_file.startswith(alt_path + "/r2e_tests/"):
                rel_path = frame_file[len(alt_path) + 1 :]
                rel_path = frame_file[len(alt_path) + 1 :]

            # Check if this frame is in a patched callable
            is_patched = False
            qualified = frame_func
            if rel_path in patched_ranges:
                for start, end, _qname in patched_ranges[rel_path]:
                    if start <= frame_lineno <= end:
                        is_patched = True
                        break

            frames.append(
                {
                    "file_path": rel_path,
                    "line_no": frame_lineno,
                    "func_name": frame_func,
                    "qualified_name": qualified,
                    "line_content": frame_code,
                    "is_patched": is_patched,
                }
            )

        if not any(fr["is_patched"] for fr in frames):
            entry_file = entry.get("file", "")
            entry_callable = entry.get("callable", "")
            patched_callable = patched_by_entry.get((entry_file, entry_callable))
            if patched_callable is not None:
                frames.append(
                    {
                        "file_path": entry_file,
                        "line_no": patched_callable.get("instr_start_line", patched_callable["start_line"]),
                        "func_name": patched_callable.get("name", entry_callable),
                        "qualified_name": patched_callable.get("qualified_name", entry_callable),
                        "line_content": "",
                        "is_patched": True,
                    }
                )

        if frames and (not require_patched or any(fr["is_patched"] for fr in frames)):
            traces.append(frames)

    return traces


# ---------------------------------------------------------------------------
# 7. aggregate_traces
# ---------------------------------------------------------------------------


def aggregate_traces(traces: list[list[dict]]) -> list[list[dict]]:
    """Deduplicate and keep only maximal traces.

    - Remove exact duplicates
    - Remove subchains: if trace A is a subsequence of trace B, drop A
    """
    if not traces:
        return []

    def _trace_key(trace: list[dict]) -> tuple:
        return tuple((f["file_path"], f["line_no"], f["func_name"]) for f in trace)

    # Deduplicate exact matches
    seen_keys: set[tuple] = set()
    unique_traces: list[list[dict]] = []
    for trace in traces:
        key = _trace_key(trace)
        if key not in seen_keys:
            seen_keys.add(key)
            unique_traces.append(trace)

    if len(unique_traces) <= 1:
        return unique_traces

    # Remove subchains: trace A is a subsequence of trace B if all frames
    # of A appear in B in order
    def _is_subsequence(shorter: list[dict], longer: list[dict]) -> bool:
        if len(shorter) >= len(longer):
            return False
        s_key = _trace_key(shorter)
        l_key = _trace_key(longer)
        it = iter(l_key)
        return all(frame in it for frame in s_key)

    # Sort by length descending so we check shorter against longer
    unique_traces.sort(key=len, reverse=True)
    maximal: list[list[dict]] = []

    for trace in unique_traces:
        is_subchain = False
        for existing in maximal:
            if _is_subsequence(trace, existing):
                is_subchain = True
                break
        if not is_subchain:
            maximal.append(trace)

    return maximal


# ---------------------------------------------------------------------------
# 8. build_call_graph_from_traces
# ---------------------------------------------------------------------------


def _source_snippet(source: str, start_line: int, end_line: int) -> str:
    if not source:
        return ""
    lines = source.splitlines()
    start = max(int(start_line), 1)
    end = max(int(end_line), start)
    return "\n".join(lines[start - 1 : end])


def _select_enclosing_callable(
    node_key: str,
    call_site: int,
    callables: dict[str, CallableInfo],
) -> CallableInfo | None:
    _, _, qualified_name = node_key.partition("::")
    exact = callables.get(qualified_name)
    if exact and exact.start_line <= call_site <= exact.end_line:
        return exact

    candidates = [ci for ci in callables.values() if ci.start_line <= call_site <= ci.end_line]
    if not candidates:
        return None
    return min(candidates, key=lambda ci: (ci.end_line - ci.start_line, -ci.start_line))


_GENERIC_ISSUE_ANCHOR_NAMES = {
    "__call__",
    "__new__",
    "call",
    "delete",
    "dispatch",
    "func",
    "function",
    "get",
    "handle",
    "head",
    "inner",
    "main",
    "method",
    "options",
    "patch",
    "post",
    "put",
    "run",
    "setup",
    "teardown",
    "test",
    "view",
    "wrapper",
}


@dataclass(frozen=True)
class IssueAnchorCandidate:
    """A conservative code-like mention extracted from issue text."""

    raw: str
    kind: str
    value: str
    file_path: str | None = None
    qualified_name: str | None = None
    name: str | None = None

    def to_dict(self) -> dict:
        result = {
            "raw": self.raw,
            "kind": self.kind,
            "value": self.value,
        }
        if self.file_path:
            result["file_path"] = self.file_path
        if self.qualified_name:
            result["qualified_name"] = self.qualified_name
        if self.name:
            result["name"] = self.name
        return result


def _normalize_anchor_path(path: str) -> str:
    value = path.replace("\\", "/").strip().strip("\"'")
    value = value.split("#", 1)[0]
    value = re.sub(r":\d+(?::\d+)?$", "", value)
    return value.lstrip("./")


def _clean_issue_anchor_value(raw: str) -> str:
    value = raw.strip().strip("`\"'")
    value = value.strip(" \t\r\n.,;:)]}")
    value = value.removesuffix("()")
    return value


def _anchor_leaf(value: str) -> str:
    cleaned = value.removesuffix("()")
    if "::" in cleaned:
        cleaned = cleaned.rsplit("::", 1)[-1]
    return cleaned.rsplit(".", 1)[-1]


def _is_generic_issue_anchor(name: str) -> bool:
    return name.lower() in _GENERIC_ISSUE_ANCHOR_NAMES


def _make_issue_anchor_candidate(
    raw: str,
    kind: str,
    *,
    file_path: str | None = None,
    qualified_name: str | None = None,
) -> IssueAnchorCandidate | None:
    value = _clean_issue_anchor_value(raw)
    if not value:
        return None
    if "\n" in value or "\t" in value:
        return None
    if any(ch.isspace() for ch in value):
        return None

    parsed_file_path = _normalize_anchor_path(file_path) if file_path else None
    parsed_qualified_name = qualified_name.removesuffix("()") if qualified_name else None
    parsed_name: str | None = None

    if "::" in value and not parsed_qualified_name:
        left, right = value.split("::", 1)
        parsed_qualified_name = right.removesuffix("()")
        if ".py" in left or "/" in left:
            parsed_file_path = _normalize_anchor_path(left)
    elif not parsed_file_path and (".py" in value or "/" in value):
        parsed_file_path = _normalize_anchor_path(value)
    elif not parsed_qualified_name and "." in value:
        parsed_qualified_name = value.removesuffix("()")
    elif not parsed_qualified_name:
        parsed_name = value.removesuffix("()")

    if parsed_qualified_name:
        parsed_name = _anchor_leaf(parsed_qualified_name)
    elif parsed_file_path:
        parsed_name = None

    disambiguated = bool(parsed_file_path or (parsed_qualified_name and "." in parsed_qualified_name))
    if parsed_name and _is_generic_issue_anchor(parsed_name) and not disambiguated:
        return None
    if parsed_name and len(parsed_name) < 3 and not parsed_name.startswith("__"):
        return None

    return IssueAnchorCandidate(
        raw=raw,
        kind=kind,
        value=value,
        file_path=parsed_file_path,
        qualified_name=parsed_qualified_name,
        name=parsed_name,
    )


def _dedupe_issue_anchor_candidate(
    candidates: list[IssueAnchorCandidate],
    seen: set[tuple[str, str, str, str]],
    candidate: IssueAnchorCandidate | None,
) -> None:
    if candidate is None:
        return
    key = (
        candidate.file_path or "",
        candidate.qualified_name or "",
        candidate.name or "",
        candidate.kind,
    )
    if key in seen:
        return
    seen.add(key)
    candidates.append(candidate)


def _extract_issue_anchor_candidates(issue_text: str | None) -> list[IssueAnchorCandidate]:
    if not isinstance(issue_text, str) or not issue_text.strip():
        return []

    candidates: list[IssueAnchorCandidate] = []
    seen: set[tuple[str, str, str, str]] = set()

    def add(raw: str, kind: str, *, file_path: str | None = None, qualified_name: str | None = None) -> None:
        _dedupe_issue_anchor_candidate(
            candidates,
            seen,
            _make_issue_anchor_candidate(raw, kind, file_path=file_path, qualified_name=qualified_name),
        )

    for match in re.finditer(r'File "([^"\n]+\.py)", line \d+, in ([A-Za-z_]\w*|<[^>]+>)', issue_text):
        func_name = match.group(2)
        if func_name.startswith("<"):
            continue
        file_path = _normalize_anchor_path(match.group(1))
        add(f"{file_path}::{func_name}", "traceback_frame", file_path=file_path, qualified_name=func_name)

    for match in re.finditer(r"`([^`\n]{1,200})`", issue_text):
        add(match.group(1), "backtick")

    for match in re.finditer(r"(?<![\w/.-])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.py(?::\d+)?", issue_text):
        add(match.group(0), "file_path")

    for match in re.finditer(r"(?<![\w.])([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)(?![\w.])", issue_text):
        add(match.group(1), "dotted_name")

    for match in re.finditer(r"(?<![\w.])([A-Za-z_]\w*)\s*\(", issue_text):
        add(match.group(1), "call")

    return candidates


def _normalize_qualname(value: str) -> str:
    return value.replace(".<locals>.", ".").replace("<locals>.", "")


def _path_suffix_matches(frame_path: str, candidate_path: str) -> bool:
    frame_norm = _normalize_anchor_path(frame_path)
    candidate_norm = _normalize_anchor_path(candidate_path)
    return (
        frame_norm == candidate_norm
        or frame_norm.endswith(f"/{candidate_norm}")
        or candidate_norm.endswith(f"/{frame_norm}")
    )


def _qualname_suffix_matches(frame_qualname: str, candidate_qualname: str) -> bool:
    frame_norm = _normalize_qualname(frame_qualname)
    candidate_norm = _normalize_qualname(candidate_qualname.removesuffix("()"))
    return frame_norm == candidate_norm or frame_norm.endswith(f".{candidate_norm}")


def _qualified_leaf_matches_frame(frame_qualname: str, candidate_qualname: str) -> bool:
    candidate_name = _anchor_leaf(candidate_qualname)
    if _is_generic_issue_anchor(candidate_name):
        return False
    return _name_matches_frame(frame_qualname, candidate_name)


def _qualified_class_matches_frame(frame_qualname: str, candidate_qualname: str) -> bool:
    candidate_name = _anchor_leaf(candidate_qualname)
    if not candidate_name[:1].isupper():
        return False
    frame_norm = _normalize_qualname(frame_qualname)
    return f".{candidate_name}." in f".{frame_norm}."


def _name_matches_frame(frame_qualname: str, candidate_name: str) -> bool:
    frame_norm = _normalize_qualname(frame_qualname)
    frame_name = frame_norm.rsplit(".", 1)[-1]
    if frame_name == candidate_name:
        return True
    if _is_generic_issue_anchor(candidate_name):
        return False
    return any(
        frame_name == f"{candidate_name}{suffix}"
        for suffix in ("_value", "_values")
    )


def _issue_anchor_match_score(candidate: IssueAnchorCandidate, frame: dict) -> int | None:
    frame_path = str(frame.get("file_path") or "")
    frame_qualname = str(frame.get("qualified_name") or frame.get("func_name") or "")

    if candidate.file_path:
        if "/" not in _normalize_anchor_path(candidate.file_path) and not candidate.qualified_name and not candidate.name:
            return None
        if not _path_suffix_matches(frame_path, candidate.file_path):
            return None
        if candidate.qualified_name:
            return 100 if _qualname_suffix_matches(frame_qualname, candidate.qualified_name) else None
        if candidate.name:
            return 80 if _name_matches_frame(frame_qualname, candidate.name) else None
        return 30

    if candidate.qualified_name:
        if _qualname_suffix_matches(frame_qualname, candidate.qualified_name):
            return 90
        if _qualified_class_matches_frame(frame_qualname, candidate.qualified_name):
            return 70
        if _qualified_leaf_matches_frame(frame_qualname, candidate.qualified_name):
            return 60
        return None

    if candidate.name:
        return 50 if _name_matches_frame(frame_qualname, candidate.name) else None

    return None


def _issue_anchor_matches_frame(candidate: IssueAnchorCandidate, frame: dict) -> bool:
    return _issue_anchor_match_score(candidate, frame) is not None


def _select_issue_anchor(
    trace: list[dict],
    root_idx: int,
    candidates: list[IssueAnchorCandidate],
) -> tuple[int, list[IssueAnchorCandidate]] | None:
    selected: tuple[int, list[IssueAnchorCandidate]] | None = None
    selected_rank: tuple[int, int] | None = None
    for idx, frame in enumerate(trace[: root_idx + 1]):
        if _test_file_reason(str(frame.get("file_path") or "")):
            continue
        scored_matches = [
            (score, candidate)
            for candidate in candidates
            if (score := _issue_anchor_match_score(candidate, frame)) is not None
        ]
        matches = [candidate for _score, candidate in scored_matches]
        if matches:
            rank = (max(score for score, _candidate in scored_matches), idx)
            if selected_rank is None or rank > selected_rank:
                selected = (idx, matches)
                selected_rank = rank
    return selected


def _frame_node_key(frame: dict) -> str:
    return f"{frame['file_path']}::{frame.get('qualified_name', frame['func_name'])}"


def _callable_node_key(callable_info: dict) -> str:
    return f"{callable_info['file_path']}::{callable_info['qualified_name']}"


def _terminal_patched_root_keys(
    observed_keys: set[str],
    dependency_edges: set[tuple[str, str]],
) -> tuple[set[str], list[list[str]]]:
    index = 0
    stack: list[str] = []
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    on_stack: set[str] = set()
    outgoing: dict[str, set[str]] = {key: set() for key in observed_keys}
    for upstream, downstream in dependency_edges:
        if upstream in observed_keys and downstream in observed_keys:
            outgoing.setdefault(upstream, set()).add(downstream)
            outgoing.setdefault(downstream, set())

    components: list[list[str]] = []

    def strongconnect(node_key: str) -> None:
        nonlocal index
        indices[node_key] = index
        lowlinks[node_key] = index
        index += 1
        stack.append(node_key)
        on_stack.add(node_key)

        for downstream_key in outgoing.get(node_key, ()):
            if downstream_key not in indices:
                strongconnect(downstream_key)
                lowlinks[node_key] = min(lowlinks[node_key], lowlinks[downstream_key])
            elif downstream_key in on_stack:
                lowlinks[node_key] = min(lowlinks[node_key], indices[downstream_key])

        if lowlinks[node_key] != indices[node_key]:
            return

        component: list[str] = []
        while True:
            member = stack.pop()
            on_stack.remove(member)
            component.append(member)
            if member == node_key:
                break
        components.append(sorted(component))

    for node_key in sorted(observed_keys):
        if node_key not in indices:
            strongconnect(node_key)

    component_index = {
        node_key: component_id
        for component_id, component in enumerate(components)
        for node_key in component
    }
    terminal_components: list[list[str]] = []
    for component_id, component in enumerate(components):
        has_external_downstream = any(
            component_index[downstream_key] != component_id
            for node_key in component
            for downstream_key in outgoing.get(node_key, ())
        )
        if not has_external_downstream:
            terminal_components.append(component)

    terminal_roots = {node_key for component in terminal_components for node_key in component}
    return terminal_roots, sorted(terminal_components, key=lambda component: (component[0], len(component)))


def build_call_graph_from_traces(
    traces: list[list[dict]],
    modified_callables: list[dict],
    file_reader: Callable[[str], str] | None = None,
    issue_text: str | None = None,
) -> dict:
    """Build a call graph with hop distances from aggregated traces.

    Each trace is a call chain from test → ... → patched callable.
    Terminal patched frames in the trace-induced patched-callable dependency
    graph get hop_distance=0. From each terminal patched frame, we walk
    backward through the trace to assign hop distances to callers.

    For callables appearing in multiple traces, the minimum hop distance is kept.

    Args:
        traces: Output of aggregate_traces() — list of traces, each a list of
                frame dicts with keys: file_path, line_no, func_name, is_patched.
        modified_callables: Output of find_modified_callables_from_sources() —
                list of dicts with keys: qualified_name, file_path, start_line, end_line.
        issue_text: Optional task issue/problem text used to mark a diagnostic
                symptom frame for Path visualization and metrics. Reward starts
                at the first non-test frame after the test harness.

    Returns:
        Dict with keys:
            - call_graph_nodes: {node_key: {file_path, start_line, end_line,
              hop_distance, normalized_distance, source}}
            - call_graph_edges: [[caller_key, callee_key], ...]
            - hop_max: maximum hop distance observed
            - patched_callables: the input modified_callables
            - traceable: True
    """
    # PRECONDITION: each trace in `traces` is ordered outer -> inner. See generate_tracer_module.

    # Strip internal instrumentation keys from output
    clean_callables = [
        {k: v for k, v in mc.items() if not k.startswith("instr_") and k != "source"}
        for mc in modified_callables
    ]

    # Identify patched-callable reachability induced by the observed traces.
    observed_patched_keys: set[str] = set()
    patched_frames_by_trace: list[dict[str, Any]] = []
    patched_dependency_edges: set[tuple[str, str]] = set()
    for trace_idx, trace in enumerate(traces):
        patched_occurrences: list[tuple[int, str]] = []
        for frame_idx, frame in enumerate(trace):
            if frame.get("is_patched"):
                node_key = _frame_node_key(frame)
                patched_occurrences.append((frame_idx, node_key))
                observed_patched_keys.add(node_key)

        patched_frame_items: dict[str, dict[str, Any]] = {}
        for frame_idx, node_key in patched_occurrences:
            frame_item = patched_frame_items.setdefault(
                node_key,
                {
                    "frame_index": frame_idx,
                    "frame_indices": [],
                    "node_key": node_key,
                    "downstream_patched_frame_keys": set(),
                },
            )
            frame_item["frame_indices"].append(frame_idx)

        for upstream_pos, (_frame_idx, upstream_key) in enumerate(patched_occurrences):
            downstream_keys = {
                downstream_key
                for _downstream_idx, downstream_key in patched_occurrences[upstream_pos + 1 :]
                if downstream_key != upstream_key
            }
            patched_frame_items[upstream_key]["downstream_patched_frame_keys"].update(downstream_keys)
            for downstream_key in downstream_keys:
                patched_dependency_edges.add((upstream_key, downstream_key))

        patched_frames: list[dict[str, Any]] = []
        for frame_item in sorted(patched_frame_items.values(), key=lambda item: item["frame_index"]):
            frame_item["downstream_patched_frame_keys"] = sorted(frame_item["downstream_patched_frame_keys"])
            patched_frames.append(frame_item)

        patched_frames_by_trace.append(
            {
                "trace_index": trace_idx,
                "patched_frames": patched_frames,
            }
        )

    terminal_root_keys, terminal_root_components = _terminal_patched_root_keys(
        observed_patched_keys,
        patched_dependency_edges,
    )
    upstream_adapter_patched_keys = observed_patched_keys - terminal_root_keys
    for trace_item in patched_frames_by_trace:
        for frame_item in trace_item["patched_frames"]:
            frame_item["trace_terminal"] = not frame_item["downstream_patched_frame_keys"]
            frame_item["selected_root_seed"] = frame_item["node_key"] in terminal_root_keys
            frame_item["upstream_adapter"] = frame_item["node_key"] in upstream_adapter_patched_keys

    unobserved_patched_callables: list[dict] = []
    issue_anchor_candidates = _extract_issue_anchor_candidates(issue_text)

    for mc in modified_callables:
        if _callable_node_key(mc) not in observed_patched_keys:
            unobserved_patched_callables.append({k: v for k, v in mc.items() if not k.startswith("instr_") and k != "source"})

    def collect_node_distances(
        root_indices_by_trace: list[list[int]],
        *,
        include_reward_start_records: bool = False,
    ) -> tuple[dict[str, dict], set[str], dict[str, set[str]], set[str], set[str], set[str], list[dict[str, Any]]]:
        collected_info: dict[str, dict] = {}
        collected_rewardable: set[str] = set()
        collected_excluded: dict[str, set[str]] = {}
        selected_anchor_keys: set[str] = set()
        ground_truth_anchor_keys: set[str] = set()
        test_adapter_keys: set[str] = set()
        reward_start_records: list[dict[str, Any]] = []

        def mark_excluded(node_key: str, reason: str) -> None:
            collected_excluded.setdefault(node_key, set()).add(reason)

        def mark_rewardable(node_key: str) -> None:
            collected_rewardable.add(node_key)

        for trace_idx, (trace, root_indices) in enumerate(zip(traces, root_indices_by_trace, strict=False)):
            for root_idx in root_indices:
                selected_anchor = _select_issue_anchor(trace, root_idx, issue_anchor_candidates)
                if selected_anchor is None:
                    reward_start_idx: int | None = None
                    reward_start_source = "test_filtered_fallback"
                    selected_anchor_key = None
                    selected_candidates: list[IssueAnchorCandidate] = []
                else:
                    reward_start_idx, selected_candidates = selected_anchor
                    selected_anchor_key = _frame_node_key(trace[reward_start_idx])
                    selected_anchor_keys.add(selected_anchor_key)
                    reward_start_source = "issue_anchor"

                ground_truth_anchor_idx: int | None = None
                for candidate_idx in range(0, root_idx + 1):
                    candidate_frame = trace[candidate_idx]
                    if not _test_file_reason(str(candidate_frame.get("file_path") or "")):
                        ground_truth_anchor_idx = candidate_idx
                        ground_truth_anchor_keys.add(_frame_node_key(candidate_frame))
                        break

                if include_reward_start_records:
                    reward_start_records.append(
                        {
                            "trace_index": trace_idx,
                            "root_frame_index": root_idx,
                            "root_node_key": _frame_node_key(trace[root_idx]),
                            "reward_start_source": "first_non_test_after_test",
                            "ground_truth_anchor_frame_index": ground_truth_anchor_idx,
                            "ground_truth_anchor_node_key": (
                                _frame_node_key(trace[ground_truth_anchor_idx])
                                if ground_truth_anchor_idx is not None
                                else None
                            ),
                            "issue_anchor_source": reward_start_source,
                            "selected_anchor_frame_index": reward_start_idx,
                            "selected_anchor_node_key": selected_anchor_key,
                            "matched_issue_anchor_candidates": [
                                candidate.to_dict()
                                for candidate in selected_candidates
                            ],
                        }
                    )

                for hop, j in enumerate(range(root_idx, -1, -1)):
                    frame = trace[j]
                    node_key = _frame_node_key(frame)
                    test_reason = _test_file_reason(str(frame.get("file_path") or ""))
                    if test_reason:
                        mark_excluded(node_key, f"test_suite_or_harness:{test_reason}")
                    else:
                        mark_rewardable(node_key)
                        if reward_start_idx is not None and j < reward_start_idx:
                            test_adapter_keys.add(node_key)

                    if node_key not in collected_info:
                        collected_info[node_key] = {
                            "file_path": frame["file_path"],
                            "func_name": frame.get("qualified_name", frame["func_name"]),
                            "line_no": frame["line_no"],
                            "hop_distance": hop,
                            "observed_in_trace": True,
                        }
                    else:
                        collected_info[node_key]["hop_distance"] = min(collected_info[node_key]["hop_distance"], hop)
                        collected_info[node_key]["observed_in_trace"] = True
        return (
            collected_info,
            collected_rewardable,
            collected_excluded,
            selected_anchor_keys,
            ground_truth_anchor_keys,
            test_adapter_keys,
            reward_start_records,
        )

    def rewardable_keys_for(info_by_key: dict[str, dict], rewardable: set[str], excluded: dict[str, set[str]]) -> set[str]:
        return {
            key
            for key in info_by_key
            if key in rewardable and not any(reason.startswith("test_suite_or_harness:") for reason in excluded.get(key, ()))
        }

    def diagnostic_hop_max(info_by_key: dict[str, dict], rewardable: set[str], excluded: dict[str, set[str]]) -> int:
        keys = rewardable_keys_for(info_by_key, rewardable, excluded)
        hops = [info_by_key[key]["hop_distance"] for key in keys]
        return max(max(hops, default=0), 1) if info_by_key else 0

    legacy_root_indices_by_trace = [
        [frame["frame_index"] for frame in item["patched_frames"]]
        for item in patched_frames_by_trace
    ]
    legacy_info, legacy_rewardable, legacy_excluded, _, _, _, _ = collect_node_distances(legacy_root_indices_by_trace)
    legacy_rewardable_keys = rewardable_keys_for(legacy_info, legacy_rewardable, legacy_excluded)
    legacy_zero_node_count = sum(
        1
        for key in legacy_rewardable_keys
        if legacy_info[key]["hop_distance"] == 0
    )
    legacy_hop_max = diagnostic_hop_max(legacy_info, legacy_rewardable, legacy_excluded)

    root_indices_by_trace = [
        [
            frame["frame_index"]
            for frame in item["patched_frames"]
            if frame["node_key"] in terminal_root_keys
        ]
        for item in patched_frames_by_trace
    ]
    (
        node_info,
        rewardable_seen,
        excluded_reasons,
        selected_anchor_keys,
        ground_truth_anchor_keys,
        test_adapter_keys,
        reward_start_records,
    ) = collect_node_distances(
        root_indices_by_trace,
        include_reward_start_records=True,
    )
    reward_start_source = "first_non_test_after_test"
    issue_anchor_source = "issue_anchor" if selected_anchor_keys else "unmatched_issue_anchor"
    issue_anchor_candidate_dicts = [candidate.to_dict() for candidate in issue_anchor_candidates]

    if not node_info:
        patched_root_selection = {
            "observed_patched_frames_by_trace": patched_frames_by_trace,
            "terminal_root_seeds": sorted(terminal_root_keys),
            "terminal_root_components": terminal_root_components,
            "upstream_adapter_patched_callables": sorted(upstream_adapter_patched_keys),
            "patched_dependency_edges": [[upstream, downstream] for upstream, downstream in sorted(patched_dependency_edges)],
            "legacy_distance_zero_node_count": legacy_zero_node_count,
            "distance_zero_node_count": 0,
            "legacy_hop_max": legacy_hop_max,
        }
        return {
            "call_graph_nodes": {},
            "call_graph_edges": [],
            "hop_max": 0,
            "raw_hop_max": 0,
            "rewardable_node_count": 0,
            "excluded_non_rewardable_node_count": 0,
            "excluded_test_harness_node_count": 0,
            "excluded_test_harness_nodes": [],
            "excluded_test_adapter_node_count": 0,
            "excluded_test_adapter_nodes": [],
            "excluded_pre_symptom_node_count": 0,
            "excluded_pre_symptom_nodes": [],
            "test_harness_file_patterns": [],
            "reward_start_source": reward_start_source,
            "issue_anchor_source": issue_anchor_source,
            "reward_start_by_trace": reward_start_records,
            "ground_truth_anchor_nodes": [],
            "selected_issue_anchor_nodes": [],
            "symptom_nodes": [],
            "test_adapter_nodes": [],
            "root_cause_nodes": [],
            "reward_path_edges": [],
            "direct_symptom_to_root_cause_edges": [],
            "fix_adapter_nodes": [],
            "call_graph_edge_metadata": [],
            "issue_anchor_candidates": issue_anchor_candidate_dicts,
            "patched_root_selection": patched_root_selection,
            "patched_callables": clean_callables,
            "unobserved_patched_callables": unobserved_patched_callables,
            "traceable": False,
        }

    raw_hop_max = max(n["hop_distance"] for n in node_info.values())
    rewardable_keys = rewardable_keys_for(node_info, rewardable_seen, excluded_reasons)
    rewardable_hops = [node_info[key]["hop_distance"] for key in rewardable_keys]
    hop_max = max(max(rewardable_hops, default=0), 1)  # avoid division by zero

    # Build final nodes with normalized distance
    call_graph_nodes = {}
    for node_key, info in node_info.items():
        rewardable = node_key in rewardable_keys
        raw_hop = info["hop_distance"]
        effective_hop = min(raw_hop, hop_max) if rewardable else hop_max
        reasons = sorted(excluded_reasons.get(node_key, ()))
        if rewardable:
            if node_key in terminal_root_keys:
                node_role = "root_cause"
            elif node_key in selected_anchor_keys:
                node_role = "symptom"
            elif node_key in upstream_adapter_patched_keys:
                node_role = "fix_adapter"
            elif node_key in test_adapter_keys:
                node_role = "test_adapter"
            else:
                node_role = "intermediate"
        elif any(reason.startswith("test_suite_or_harness:") for reason in reasons):
            node_role = "test_harness"
        else:
            node_role = "test_adapter"
        call_graph_nodes[node_key] = {
            "file_path": info["file_path"],
            "start_line": info["line_no"],
            "end_line": info["line_no"],  # will be enriched below
            "hop_distance": effective_hop,
            "raw_hop_distance": raw_hop,
            "normalized_distance": effective_hop / hop_max,
            "observed_in_trace": info["observed_in_trace"],
            "rewardable": rewardable,
            "node_role": node_role,
            "excluded_from_hop_max": not rewardable,
        }
        if node_key in observed_patched_keys:
            call_graph_nodes[node_key]["patched_callable"] = True
            call_graph_nodes[node_key]["patch_role"] = (
                "root_cause" if node_key in terminal_root_keys else "fix_adapter"
            )
        if reasons and not rewardable:
            call_graph_nodes[node_key]["exclusion_reason"] = ";".join(reasons)

    # Enrich patched callable nodes with full line ranges from static AST analysis
    patched_keys: set[str] = set()
    patched_sources: dict[str, str] = {}
    for mc in modified_callables:
        key = f"{mc['file_path']}::{mc['qualified_name']}"
        if key in call_graph_nodes:
            call_graph_nodes[key]["start_line"] = mc["start_line"]
            call_graph_nodes[key]["end_line"] = mc["end_line"]
            patched_keys.add(key)
            if isinstance(mc.get("source"), str):
                patched_sources[key] = mc["source"]

    # Enrich non-patched nodes with AST-derived line ranges.
    # Each frame's line_no is the call-site execution line, which lies inside
    # the function body.  We resolve the callable by qualified name when the
    # tracer and AST agree, otherwise by the innermost callable containing that
    # execution line.
    if file_reader:
        files_needed = {node["file_path"] for node in call_graph_nodes.values()}
        file_sources: dict[str, str] = {}
        file_callables: dict[str, dict[str, CallableInfo]] = {}
        for fp in files_needed:
            source = file_reader(fp)
            if source:
                file_sources[fp] = source
                file_callables[fp] = extract_callables_from_ast(source, fp)

        for node_key, node in call_graph_nodes.items():
            if node_key not in patched_keys:
                fp = node["file_path"]
                call_site = node["start_line"]  # pre-enrichment = frame's line_no
                ci = _select_enclosing_callable(node_key, call_site, file_callables.get(fp, {}))
                if ci:
                    node["start_line"] = ci.start_line
                    node["end_line"] = ci.end_line
            source = patched_sources.get(node_key)
            if source is None:
                source = _source_snippet(file_sources.get(node["file_path"], ""), node["start_line"], node["end_line"])
            if source and node.get("rewardable"):
                node["source"] = source

    edge_set: set[tuple[str, str]] = set()
    for trace in traces:
        patched_indices = [j for j, frame in enumerate(trace) if frame.get("is_patched", False)]
        for patched_idx in patched_indices:
            for j in range(0, patched_idx):
                caller = trace[j]
                callee = trace[j + 1]
                caller_key = _frame_node_key(caller)
                callee_key = _frame_node_key(callee)
                if (
                    caller_key != callee_key
                    and caller_key in call_graph_nodes
                    and callee_key in call_graph_nodes
                ):
                    edge_set.add((caller_key, callee_key))
    call_graph_edges = [[caller, callee] for caller, callee in sorted(edge_set)]
    symptom_nodes = sorted(
        key
        for key, node in call_graph_nodes.items()
        if node.get("node_role") == "symptom"
    )
    root_cause_nodes = sorted(
        key
        for key, node in call_graph_nodes.items()
        if node.get("node_role") == "root_cause"
    )
    fix_adapter_nodes = sorted(
        key
        for key, node in call_graph_nodes.items()
        if node.get("node_role") == "fix_adapter"
    )
    test_adapter_nodes = sorted(
        key
        for key, node in call_graph_nodes.items()
        if node.get("node_role") == "test_adapter"
    )
    reward_path_roles = {"symptom", "intermediate", "fix_adapter", "root_cause"}
    visual_path_nodes: set[str] = set()
    if selected_anchor_keys and root_cause_nodes:
        outgoing: dict[str, set[str]] = {}
        incoming: dict[str, set[str]] = {}
        for caller, callee in call_graph_edges:
            outgoing.setdefault(caller, set()).add(callee)
            incoming.setdefault(callee, set()).add(caller)

        forward = set(selected_anchor_keys)
        frontier = list(selected_anchor_keys)
        while frontier:
            current = frontier.pop()
            for downstream in outgoing.get(current, ()):
                if downstream not in forward:
                    forward.add(downstream)
                    frontier.append(downstream)

        backward = set(root_cause_nodes)
        frontier = list(root_cause_nodes)
        while frontier:
            current = frontier.pop()
            for upstream in incoming.get(current, ()):
                if upstream not in backward:
                    backward.add(upstream)
                    frontier.append(upstream)

        visual_path_nodes = {
            key
            for key in forward & backward
            if call_graph_nodes.get(key, {}).get("node_role") in reward_path_roles
        }
    call_graph_edge_metadata: list[dict[str, Any]] = []
    reward_path_edges: list[list[str]] = []
    direct_symptom_to_root_cause_edges: list[list[str]] = []
    for caller, callee in call_graph_edges:
        caller_role = call_graph_nodes[caller].get("node_role")
        callee_role = call_graph_nodes[callee].get("node_role")
        edge = [caller, callee]
        reward_path_edge = caller in visual_path_nodes and callee in visual_path_nodes
        direct_symptom_to_root_cause = caller_role == "symptom" and callee_role == "root_cause"
        if reward_path_edge:
            reward_path_edges.append(edge)
        if direct_symptom_to_root_cause:
            direct_symptom_to_root_cause_edges.append(edge)
        call_graph_edge_metadata.append(
            {
                "caller": caller,
                "callee": callee,
                "caller_role": caller_role,
                "callee_role": callee_role,
                "role_transition": f"{caller_role}->{callee_role}",
                "reward_path_edge": reward_path_edge,
                "direct_symptom_to_root_cause": direct_symptom_to_root_cause,
            }
        )

    traceable = any(node["rewardable"] and node["hop_distance"] == 0 for node in call_graph_nodes.values())
    distance_zero_node_count = sum(
        1
        for node in call_graph_nodes.values()
        if node.get("rewardable") and node.get("normalized_distance") == 0.0
    )
    patched_root_selection = {
        "observed_patched_frames_by_trace": patched_frames_by_trace,
        "terminal_root_seeds": sorted(terminal_root_keys),
        "terminal_root_components": terminal_root_components,
        "upstream_adapter_patched_callables": sorted(upstream_adapter_patched_keys),
        "patched_dependency_edges": [[upstream, downstream] for upstream, downstream in sorted(patched_dependency_edges)],
        "legacy_distance_zero_node_count": legacy_zero_node_count,
        "distance_zero_node_count": distance_zero_node_count,
        "legacy_hop_max": legacy_hop_max,
    }
    excluded_nodes = sorted(key for key, node in call_graph_nodes.items() if not node.get("rewardable"))
    excluded_test_nodes = sorted(
        key
        for key, node in call_graph_nodes.items()
        if node.get("node_role") == "test_harness"
    )
    excluded_test_adapter_nodes: list[str] = []

    return {
        "call_graph_nodes": call_graph_nodes,
        "call_graph_edges": call_graph_edges,
        "hop_max": hop_max,
        "raw_hop_max": raw_hop_max,
        "rewardable_node_count": sum(1 for node in call_graph_nodes.values() if node.get("rewardable")),
        "excluded_non_rewardable_node_count": len(excluded_nodes),
        "excluded_test_harness_node_count": len(excluded_test_nodes),
        "excluded_test_harness_nodes": excluded_test_nodes,
        "excluded_test_adapter_node_count": len(excluded_test_adapter_nodes),
        "excluded_test_adapter_nodes": excluded_test_adapter_nodes,
        "excluded_pre_symptom_node_count": len(excluded_test_adapter_nodes),
        "excluded_pre_symptom_nodes": excluded_test_adapter_nodes,
        "test_harness_file_patterns": sorted(
            {
                reason.split(":", 1)[1]
                for reasons in excluded_reasons.values()
                for reason in reasons
                if reason.startswith("test_suite_or_harness:")
            }
        ),
        "reward_start_source": reward_start_source,
        "issue_anchor_source": issue_anchor_source,
        "reward_start_by_trace": reward_start_records,
        "ground_truth_anchor_nodes": sorted(ground_truth_anchor_keys),
        "selected_issue_anchor_nodes": sorted(selected_anchor_keys),
        "symptom_nodes": symptom_nodes,
        "test_adapter_nodes": test_adapter_nodes,
        "root_cause_nodes": root_cause_nodes,
        "fix_adapter_nodes": fix_adapter_nodes,
        "reward_path_edges": reward_path_edges,
        "direct_symptom_to_root_cause_edges": direct_symptom_to_root_cause_edges,
        "call_graph_edge_metadata": call_graph_edge_metadata,
        "issue_anchor_candidates": issue_anchor_candidate_dicts,
        "patched_root_selection": patched_root_selection,
        "patched_callables": clean_callables,
        "unobserved_patched_callables": unobserved_patched_callables,
        "traceable": traceable,
    }
