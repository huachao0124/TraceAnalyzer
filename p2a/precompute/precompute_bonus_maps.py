#!/usr/bin/env python3
"""Precompute P2A bonus maps for SWE dataset instances.

Static mode (no sandbox needed):
    Extracts patched callables from AST diff, assigns d=0 to all.

Dynamic mode (requires sandbox):
    Runs full trace pipeline → builds call graph → assigns hop distances.

Classification decision tree (evaluated top-to-bottom, first match wins):

    Static layer (AST diff of old vs new content):
      newly_created  – all GT callables only exist in new_file_content
      no_callable    – patch has no callable-level changes
      signature_mismatch – F2P call fails before entering the callable body

    Dynamic layer (instrument → run tests → parse traces):
      instrumentation_failed – instrumentation produced no instrumented callable (error=True)
      all_pass       – buggy F2P run exited cleanly before trace-volume classification (error=True)
      no_trace       – 0 traces captured after instrumentation (error=True)
      no_gt          – traces exist but none contain a GT callable (error=True)
      no_f2p         – GT traces exist, tests fail, but F2P filter removed all (error=True)
      trace_cap_inconclusive – a trace cap was reached before no_f2p could be proven (error=True)
      latent         – F2P→GT call chain with intermediate nodes and at least
                       one root cause not exposed as the selected symptom anchor
                       (traceable=True)
      exposed        – F2P→GT call chain with intermediate nodes where every
                       root cause is already exposed as the selected symptom
                       anchor (traceable=True)
      direct         – F2P→GT call chain, test calls GT directly (traceable=True)

Usage:
    python -m p2a.precompute.precompute_bonus_maps \\
        data/swe/R2E_Gym_Subset.parquet \\
        --output_dir data/swe/bonus_maps --mode dynamic --n_parallel 50
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import gzip
import json
import os
import re
import shlex
import sys
import subprocess
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from p2a.bonus_map_scope import (
    DIRECT_CASE,
    EXPOSED_CASE,
    LATENT_CASE,
    bonus_map_pattern_computable,
    enrich_bonus_map_case_metadata,
)
from p2a.datasets import parse_string_list, selector_files
from p2a.trace import (
    TRACE_FILE_PATH,
    _is_test_file,
    extract_callables_with_tolerant_fallback,
    find_modified_callables_from_task,
    make_instance_id as _trace_make_instance_id,
    normalize_task as _trace_normalize_task,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FIXTURE_NAMES = frozenset(
    {
        "setUp",
        "tearDown",
        "setUpClass",
        "tearDownClass",
        "asyncSetUp",
        "asyncTearDown",
    }
)

BONUS_MAP_SCHEMA_VERSION = 5
TRACE_SIDECAR_FORMAT = "p2a_trace_sidecar_v1"
FAILURE_MANIFEST_NAME = "precompute_failures.jsonl"
DEFAULT_TRACE_MAX_EVENTS = 10_000
DEFAULT_TRACE_MAX_FRAMES = 80
DEFAULT_TRACE_PARSE_CHUNK_LINES = 1_000
TEST_STDOUT_PATH = "/root/_p2a_swe_test_stdout.txt"
TEST_STDERR_PATH = "/root/_p2a_swe_test_stderr.txt"
TEST_EXIT_PATH = "/root/_p2a_swe_test_exit.txt"
TRACE_PARSE_PATH = "/root/_p2a_swe_fault_traces.parse.jsonl"


_PRODUCER_METADATA: dict | None = None


def _empty_graph_metadata() -> dict:
    return {
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
        "reward_start_source": "first_non_test_after_test",
        "issue_anchor_source": "unmatched_issue_anchor",
        "reward_start_by_trace": [],
        "ground_truth_anchor_nodes": [],
        "selected_issue_anchor_nodes": [],
        "symptom_nodes": [],
        "test_adapter_nodes": [],
        "root_cause_nodes": [],
        "fix_adapter_nodes": [],
        "reward_path_edges": [],
        "direct_symptom_to_root_cause_edges": [],
        "call_graph_edge_metadata": [],
        "issue_anchor_candidates": [],
    }


def _task_issue_text(task: dict) -> str | None:
    fields = (
        "problem_statement",
        "issue",
        "issue_text",
        "description",
        "problem",
        "title",
    )
    for field in fields:
        value = task.get(field)
        if isinstance(value, str) and value.strip():
            return value

    extra = task.get("extra_info")
    if isinstance(extra, dict):
        for field in fields:
            value = extra.get(field)
            if isinstance(value, str) and value.strip():
                return value

    return None


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        print(f"[WARN] ignoring invalid {name}={raw!r}; using {default}", file=sys.stderr)
        return default
    if minimum is not None and value < minimum:
        print(f"[WARN] ignoring {name}={value}; expected >= {minimum}, using {default}", file=sys.stderr)
        return default
    return value


def _env_positive_int_or_none(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError:
        print(f"[WARN] ignoring invalid {name}={raw!r}; parsing all captured trace lines", file=sys.stderr)
        return None
    if value <= 0:
        return None
    return value


def _looks_like_bare_hash(value: str) -> bool:
    return len(value) >= 20 and all(ch in "0123456789abcdef" for ch in value)


def normalize_task(task: dict) -> dict:
    """Normalize raw dataset rows and Uni-Agent ``extra_info.tools_kwargs`` rows."""
    task = _trace_normalize_task(task)
    extra = task.get("extra_info")
    if isinstance(extra, str) and extra.strip():
        try:
            extra = json.loads(extra)
        except (json.JSONDecodeError, TypeError):
            extra = None
    if not isinstance(extra, dict):
        extra = {}

    tools_kwargs = task.get("tools_kwargs")
    if not isinstance(tools_kwargs, dict):
        maybe_tools = extra.get("tools_kwargs")
        tools_kwargs = maybe_tools if isinstance(maybe_tools, dict) else {}

    reward = tools_kwargs.get("reward") if isinstance(tools_kwargs.get("reward"), dict) else {}
    metadata = reward.get("metadata") if isinstance(reward.get("metadata"), dict) else {}

    merged = {**metadata, **task}
    if tools_kwargs:
        merged["tools_kwargs"] = tools_kwargs
    if extra:
        merged["extra_info"] = extra
    if "repo" in merged and "repo_name" not in merged:
        merged["repo_name"] = merged["repo"]
    # ``relevant_files`` is carried as a JSON string by the enrich adapter (a
    # native list column re-introduces a nested-chunk parquet read error at
    # scale); the static callable detector expects the decoded list.
    rel = merged.get("relevant_files")
    if isinstance(rel, str) and rel.strip():
        try:
            merged["relevant_files"] = json.loads(rel)
        except (json.JSONDecodeError, TypeError):
            pass
    return merged


def make_instance_id(task: dict) -> str:
    """Return the instance id used by Uni-Agent R2E parquet rows.

    The old rLLM helper used an 8-char R2E commit prefix. Uni-Agent's
    ``r2e_gym_subset_filtered.py`` uses 10 chars, and trainer lookup keys must
    match the parquet metadata exactly.
    """
    task = normalize_task(task)
    iid = task.get("instance_id")
    if isinstance(iid, str) and iid and not _looks_like_bare_hash(iid):
        return iid

    repo = task.get("repo_name") or task.get("repo")
    commit = task.get("new_commit_hash") or task.get("commit_hash") or task.get("base_commit")
    if isinstance(repo, str) and repo and isinstance(commit, str) and commit:
        return f"{repo}__{commit[:10]}"

    base = _trace_make_instance_id(task)
    if isinstance(base, str) and base:
        return base
    return "unknown"


def _producer_metadata() -> dict:
    """Return stable metadata for cache provenance."""
    global _PRODUCER_METADATA
    if _PRODUCER_METADATA is not None:
        return dict(_PRODUCER_METADATA)

    repo_root = Path(__file__).resolve().parents[2]
    commit = None
    branch = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip() or None
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip() or None
    except OSError:
        pass

    _PRODUCER_METADATA = {
        "schema_version": BONUS_MAP_SCHEMA_VERSION,
        "producer": "p2a.precompute.precompute_bonus_maps",
        "producer_commit": commit,
        "producer_branch": branch,
    }
    return dict(_PRODUCER_METADATA)


def _debug_progress(instance_id: str, step: str) -> None:
    if os.getenv("P2A_DEBUG_PROGRESS"):
        print(f"  [{instance_id}] {step}", flush=True)


def _with_metadata(result: dict, *, reason_code: str | None = None, diagnostics: dict | None = None) -> dict:
    """Attach schema/provenance and compact debug diagnostics."""
    enriched = {**_producer_metadata(), **result}
    enriched["reason_code"] = reason_code or result.get("case_type")
    if diagnostics:
        enriched.update(diagnostics)
    return enriched


def _base_diagnostics(
    *,
    test_exit: int | None = None,
    total_trace_entries: int | None = None,
    raw_gt_trace_count: int | None = None,
    f2p_test_funcs: set[str] | None = None,
    f2p_trace_count: int | None = None,
    instrumented_callables_count: int | None = None,
    stdout: str | None = None,
    stderr: str | None = None,
    test_output_capture: str | None = None,
    import_targets: list[dict] | None = None,
) -> dict:
    return {
        "test_exit": test_exit,
        "total_trace_entries": total_trace_entries,
        "raw_gt_trace_count": raw_gt_trace_count,
        "f2p_test_funcs": sorted(f2p_test_funcs) if f2p_test_funcs is not None else None,
        "f2p_trace_count": f2p_trace_count,
        "instrumented_callables_count": instrumented_callables_count,
        "stdout_chars": len(stdout) if stdout is not None else None,
        "stderr_chars": len(stderr) if stderr is not None else None,
        "test_output_capture": test_output_capture,
        "import_targets": import_targets,
    }


def _tail_text(value: str | None, max_chars: int) -> str | None:
    if value is None:
        return None
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[-max_chars:]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _classify_precompute_exception(exc: BaseException) -> str:
    text = str(exc)
    if (
        "gRPC Execute failed" in text
        or "code = Unavailable" in text
        or "transport: Error while dialing" in text
        or "connection refused" in text.lower()
    ):
        return "arl_grpc_unavailable"
    if "sandbox post-setup failed" in text:
        return "sandbox_post_setup_failed"
    return "exception"


def _is_retryable_precompute_failure(data: dict | None) -> bool:
    if not isinstance(data, dict):
        return False
    return bool(
        data.get("precompute_failure")
        or data.get("case_type") == "precompute_failed"
        or data.get("reason_code") in {"exception", "precompute_exception"}
    )


def _existing_bonus_map_is_complete(path: str | os.PathLike[str]) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return not _is_retryable_precompute_failure(data)


def _remove_retryable_bonus_map(path: str | os.PathLike[str]) -> None:
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    if _is_retryable_precompute_failure(data):
        os.remove(path)


def _failure_record(
    *,
    idx,
    instance_id: str,
    output_path: str | None = None,
    result: dict | None = None,
    error: BaseException | str | None = None,
) -> dict:
    result = result or {}
    message = result.get("exception_message")
    error_type = result.get("exception_type")
    traceback_tail = result.get("exception_traceback_tail")
    failure_kind = result.get("failure_kind")
    if error is not None:
        message = str(error)
        error_type = type(error).__name__ if isinstance(error, BaseException) else "Error"
        if isinstance(error, BaseException):
            failure_kind = _classify_precompute_exception(error)
    return {
        "timestamp": _utc_now_iso(),
        "idx": str(idx),
        "instance_id": instance_id,
        "case_type": result.get("case_type") or "precompute_failed",
        "reason_code": result.get("reason_code") or "precompute_exception",
        "failure_kind": failure_kind or "exception",
        "error_type": error_type,
        "message": _tail_text(message, 2000),
        "traceback_tail": _tail_text(traceback_tail, 8000),
        "output_path": output_path,
    }


def _append_failure_manifest(path: str | os.PathLike[str], records: list[dict]) -> None:
    if not records:
        return
    os.makedirs(os.path.dirname(os.fspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True))
            f.write("\n")


def _slice_traces(traces: list[list[dict]] | None, max_traces: int | None) -> tuple[list[list[dict]], bool]:
    if not traces:
        return [], False
    if max_traces is None or max_traces < 0 or len(traces) <= max_traces:
        return traces, False
    return traces[:max_traces], True


def _write_trace_sidecar(
    *,
    instance_id: str,
    sidecar_dir: str | None,
    raw_traces: list[list[dict]] | None,
    raw_gt_traces: list[list[dict]] | None,
    f2p_traces: list[list[dict]] | None = None,
    aggregated_f2p_traces: list[list[dict]] | None = None,
    diagnostics: dict | None = None,
    stdout: str | None = None,
    stderr: str | None = None,
    max_sidecar_traces: int | None = None,
    max_output_chars: int = 100_000,
) -> dict:
    """Write bulky trace/debug data to a gzip sidecar, not the training JSON."""
    if not sidecar_dir:
        return {}

    sidecar_root = Path(sidecar_dir)
    sidecar_root.mkdir(parents=True, exist_ok=True)
    sidecar_path = sidecar_root / f"{instance_id}.traces.json.gz"

    trace_sets = {
        "raw_traces": raw_traces or [],
        "raw_gt_traces": raw_gt_traces or [],
        "f2p_traces": f2p_traces or [],
        "aggregated_f2p_traces": aggregated_f2p_traces or [],
    }
    saved_trace_sets = {}
    counts = {}
    for name, traces in trace_sets.items():
        saved, truncated = _slice_traces(traces, max_sidecar_traces)
        saved_trace_sets[name] = saved
        counts[name] = {
            "total": len(traces),
            "saved": len(saved),
            "truncated": truncated,
        }

    payload = {
        "schema_version": BONUS_MAP_SCHEMA_VERSION,
        "sidecar_format": TRACE_SIDECAR_FORMAT,
        "instance_id": instance_id,
        "counts": counts,
        "diagnostics": diagnostics or {},
        "stdout_chars": len(stdout) if stdout is not None else None,
        "stderr_chars": len(stderr) if stderr is not None else None,
        "stdout_tail": _tail_text(stdout, max_output_chars),
        "stderr_tail": _tail_text(stderr, max_output_chars),
        **saved_trace_sets,
    }
    with gzip.open(sidecar_path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)

    return {
        "trace_sidecar_path": str(sidecar_path),
        "trace_sidecar_format": TRACE_SIDECAR_FORMAT,
        "trace_sidecar_counts": counts,
    }


def find_newly_created_callables(task: dict) -> list[dict]:
    """Find callables that only exist in new_file_content (added by the fix).

    These are pure additions — the callable doesn't exist in old_file_content
    at all — so they cannot be instrumented on the pre-fix (buggy) code.
    """
    task = normalize_task(task)
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

        newly_created: list[dict] = []
        for fd in file_diffs:
            path = fd.get("header", {}).get("file", {}).get("path", "")
            if not path or not path.endswith(".py"):
                continue
            old_src = fd.get("old_file_content") or ""
            new_src = fd.get("new_file_content") or ""
            if not new_src:
                continue

            new_callables = extract_callables_with_tolerant_fallback(new_src, path)
            if old_src:
                old_callables = extract_callables_with_tolerant_fallback(old_src, path)
            else:
                old_callables = {}

            for qname, info in new_callables.items():
                if qname not in old_callables:
                    newly_created.append(info.to_dict())
        return newly_created

    return []


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
_PARAMETRIZE_SUFFIX_RE = re.compile(r"\[.*\]$")
_TEST_FUNC_RE = re.compile(r"(?<![A-Za-z0-9_])(test[A-Za-z0-9_]*)(?=$|[^A-Za-z0-9_])")
_TEST_DEF_RE = re.compile(r"^\s*(?:async\s+)?def\s+(test[A-Za-z0-9_]*)\s*\(")
_PYTEST_STATUS_RE = re.compile(r"\b(PASSED|FAILED|ERROR)\b")
_UNITTEST_FAILURE_HEADER_RE = re.compile(r"^(?:FAIL|ERROR):\s+([A-Za-z_][A-Za-z0-9_]*)\s+\(([^)]+)\)\s*$")


def _strip_parametrize(name: str) -> str:
    """Strip pytest parametrize suffix: ``test_foo[True]`` → ``test_foo``."""
    return _PARAMETRIZE_SUFFIX_RE.sub("", name)


def _normalize_test_func_name(name: str) -> str:
    """Return the bare pytest test function name from a nodeid/qualname."""
    cleaned = _ANSI_ESCAPE_RE.sub("", str(name or "")).strip()
    unittest_display = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s+\([^)]+\)$", cleaned)
    if unittest_display:
        return unittest_display.group(1)
    cleaned = cleaned.split(" - ", 1)[0]
    matches = _TEST_FUNC_RE.findall(cleaned)
    if matches:
        return _strip_parametrize(matches[-1])
    bare = cleaned.rsplit(".", 1)[-1].rsplit("::", 1)[-1]
    return _strip_parametrize(bare)


def _parse_pytest_status_lines(raw_output: str) -> dict[str, str]:
    """Parse pytest ``-vv`` status lines when the short summary is absent."""
    statuses: dict[str, str] = {}
    for raw_line in raw_output.splitlines():
        line = _ANSI_ESCAPE_RE.sub("", raw_line).strip()
        if not line or "::" not in line:
            continue

        # Verbose progress: ``path.py::test_x[param] FAILED [ 10%]``.
        match = _PYTEST_STATUS_RE.search(line)
        if match:
            status = match.group(1)
            if line.startswith(status):
                remainder = line[len(status) :].strip()
                node = remainder.split(" - ", 1)[0].strip()
            else:
                node = line[: match.start()].strip()
            if "::" in node:
                statuses[node] = status
    return statuses


def _r2e_fixed_passed_test_funcs(task: dict) -> set[str] | None:
    """Return fixed-run passing R2E tests from ``expected_output_json``."""
    expected_raw = task.get("expected_output_json")
    if expected_raw is None:
        return None

    if isinstance(expected_raw, str):
        try:
            expected_status = json.loads(expected_raw)
        except (json.JSONDecodeError, TypeError):
            return None
    elif isinstance(expected_raw, dict):
        expected_status = expected_raw
    else:
        return None

    if not isinstance(expected_status, dict):
        return None

    passed_funcs: set[str] = set()
    for name, status in expected_status.items():
        if str(status or "").strip().upper() != "PASSED":
            continue
        bare = _normalize_test_func_name(str(name).split(" - ", 1)[0])
        if bare:
            passed_funcs.add(bare)
    return passed_funcs


def _looks_like_test_func(name: str) -> bool:
    return bool(name and (name.startswith("test") or name in _FIXTURE_NAMES))


def _swebench_unittest_failure_funcs_from_descriptions(raw_output: str, f2p_nodeids: list[str]) -> set[str]:
    """Map SWE-bench unittest failure descriptions back to method names."""
    selectors = [str(item or "").strip() for item in f2p_nodeids if str(item or "").strip()]
    if not raw_output or not selectors:
        return set()

    lines = [_ANSI_ESCAPE_RE.sub("", line).strip() for line in raw_output.splitlines()]
    funcs: set[str] = set()
    for idx, line in enumerate(lines):
        match = _UNITTEST_FAILURE_HEADER_RE.match(line)
        if not match:
            continue
        method, qualname = match.groups()
        header_display = f"{method} ({qualname})"
        context = "\n".join(lines[idx : min(idx + 5, len(lines))])
        for selector in selectors:
            bare = _normalize_test_func_name(selector)
            if selector == header_display or bare == method or selector in context:
                funcs.add(method)
                break
    return funcs


def _get_f2p_test_funcs(task: dict, raw_output: str, swebench_verified: bool) -> set[str] | None:
    """Identify fail-to-pass (F2P) test function names.

    F2P = tests that FAIL on buggy code and PASS after the developer's fix.

    For SWE-Bench Verified: uses the ``FAIL_TO_PASS`` field from the task.
    For R2E-Gym: intersects buggy pytest failures with fixed-run passes from
    ``expected_output_json``.

    Returns:
        set[str]: bare test function names (may be empty if no tests failed).
        None: only when we genuinely cannot parse the test output.

    Note: parametrize suffixes (``[param1-param2]``) are stripped so that
    ``test_foo[True]`` matches trace frames that only contain ``test_foo``.
    """
    task = normalize_task(task)
    if swebench_verified:
        f2p_raw = task.get("FAIL_TO_PASS")
        if f2p_raw:
            if isinstance(f2p_raw, str):
                try:
                    f2p_list = json.loads(f2p_raw)
                except (json.JSONDecodeError, TypeError):
                    f2p_list = [f2p_raw]
            elif isinstance(f2p_raw, list):
                f2p_list = f2p_raw
            else:
                return None
            funcs = set()
            for t in f2p_list:
                bare = _normalize_test_func_name(str(t))
                if _looks_like_test_func(bare):
                    funcs.add(bare)
            funcs.update(_swebench_unittest_failure_funcs_from_descriptions(raw_output, f2p_list))
            return funcs  # may be empty
        return None
    else:
        from r2egym.repo_analysis.execution_log_parser import decolor_dict_keys, parse_log_pytest

        test_status = decolor_dict_keys(parse_log_pytest(raw_output))
        if not test_status:
            test_status = _parse_pytest_status_lines(raw_output)
        if not test_status:
            return None  # genuinely can't parse
        fixed_passed_funcs = _r2e_fixed_passed_test_funcs(task)
        if fixed_passed_funcs is None:
            return None
        failed_funcs = set()
        for name, status in test_status.items():
            if str(status or "").strip().upper() in ("FAILED", "ERROR"):
                bare = _normalize_test_func_name(str(name).split(" - ", 1)[0])
                if bare:
                    failed_funcs.add(bare)
        return failed_funcs & fixed_passed_funcs  # empty set = parsed OK but no true F2P


def _filter_traces_to_f2p(traces: list[list[dict]], f2p_test_funcs: set[str]) -> list[list[dict]]:
    """Keep only traces whose call chain originates from an F2P test function.

    A trace originates from an F2P test if ANY test-file frame has a
    func_name in *f2p_test_funcs*, or is a fixture (setUp/tearDown) that
    runs for every test including F2P ones.
    """
    if not f2p_test_funcs:
        return []

    filtered = []
    for trace in traces:
        keep = False
        for frame in trace:
            file_path = frame.get("file_path", "")
            if not _is_test_file(file_path):
                continue
            func_name = frame.get("func_name", "")
            bare_func_name = _normalize_test_func_name(func_name)
            if bare_func_name in f2p_test_funcs:
                keep = True
                break
            if bare_func_name in _FIXTURE_NAMES:
                keep = True
                break
        if keep:
            filtered.append(trace)
    return filtered


def _test_func_names_from_callables(callables: list[dict]) -> set[str]:
    """Extract bare test function names from callable metadata."""
    funcs = set()
    for item in callables:
        if not _is_test_file(item.get("file_path", "")):
            continue
        qname = str(item.get("qualified_name") or item.get("name") or "")
        if not qname:
            continue
        bare = _normalize_test_func_name(qname)
        if bare and (bare.startswith("test") or bare in _FIXTURE_NAMES):
            funcs.add(bare)
    return funcs


def _test_func_names_from_traces(traces: list[list[dict]]) -> list[str]:
    """Return sorted bare test-frame names observed in traces."""
    funcs = set()
    for trace in traces:
        for frame in trace:
            if not _is_test_file(frame.get("file_path", "")):
                continue
            name = str(frame.get("func_name") or frame.get("qualified_name") or "")
            bare = _normalize_test_func_name(name)
            if bare:
                funcs.add(bare)
    return sorted(funcs)


def _run_tests_with_file_capture(env, test_script: str, timeout: int = 300) -> tuple[str, str, int | None, dict]:
    """Run the test script while avoiding ARL gateway stdout truncation.

    The ARL gateway can truncate large command stdout.  Pytest's short summary
    lives near the end of output, so dynamic bonus-map construction must capture
    stdout/stderr to sandbox files and read them back with the chunked helper.
    """
    from p2a.trace import _read_sandbox_file

    env._run(f"rm -f {TEST_STDOUT_PATH} {TEST_STDERR_PATH} {TEST_EXIT_PATH}")
    quoted_script = shlex.quote(test_script)
    cmd = (
        "set +e; "
        "export PY_COLORS=0 NO_COLOR=1 TERM=dumb; "
        f'export P2A_TRACE_MAX_EVENTS="${{P2A_TRACE_MAX_EVENTS:-{DEFAULT_TRACE_MAX_EVENTS}}}"; '
        f'export P2A_TRACE_MAX_FRAMES="${{P2A_TRACE_MAX_FRAMES:-{DEFAULT_TRACE_MAX_FRAMES}}}"; '
        'export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} -rA --color=no -vv -W ignore::pytest.PytestWarning"; '
        f"bash {quoted_script} > {TEST_STDOUT_PATH} 2> {TEST_STDERR_PATH}; "
        f"code=$?; printf '%s\\n' \"$code\" > {TEST_EXIT_PATH}; true"
    )
    wrapper_stdout, wrapper_stderr, wrapper_exit = env._execute_raw(cmd, timeout=timeout)

    stdout, stdout_exit = _read_sandbox_file(env, TEST_STDOUT_PATH)
    stderr, stderr_exit = _read_sandbox_file(env, TEST_STDERR_PATH)
    exit_text, exit_read_code = _read_sandbox_file(env, TEST_EXIT_PATH)
    exit_parse_failed = False
    try:
        test_exit = int(exit_text.strip().splitlines()[-1])
    except (ValueError, IndexError):
        exit_parse_failed = True
        test_exit = wrapper_exit

    if stdout_exit != 0 and wrapper_stdout:
        stdout = wrapper_stdout
    if stderr_exit != 0 and wrapper_stderr:
        stderr = wrapper_stderr

    all_three_read_failed = stdout_exit != 0 and stderr_exit != 0 and exit_read_code != 0
    if all_three_read_failed and exit_parse_failed and wrapper_exit == 0:
        # A successful wrapper exit only means the outer shell step returned,
        # not that the test process passed.  Treat this as unknown so the
        # decision tree cannot misclassify it as all_pass/no_trace_exit_zero.
        test_exit = None

    capture = {
        "stdout_read_exit": stdout_exit,
        "stderr_read_exit": stderr_exit,
        "exit_read_exit": exit_read_code,
        "wrapper_exit": wrapper_exit,
        "exit_parse_failed": exit_parse_failed,
        "all_three_read_failed": all_three_read_failed,
        "trusted_test_exit": test_exit is not None and not all_three_read_failed,
    }
    return stdout, stderr, test_exit, capture


def _parse_fault_traces_in_chunks(
    env,
    parse_fault_traces_from_file,
    instrumented_callables: list[dict],
    repo_path: str,
    alt_path: str,
    *,
    trace_file_line_count: int,
    line_cap: int | None,
    chunk_lines: int,
) -> list[list[dict]]:
    """Parse captured trace JSONL without pre-dropping valid evidence.

    The sandbox file reader returns a complete file as one string.  Keep each
    host read bounded by slicing the sandbox JSONL file into deterministic
    chunks first.  When ``P2A_TRACE_PARSE_MAX_LINES`` is unset, every captured
    line is parsed.
    """
    if trace_file_line_count <= 0:
        return []

    parse_line_count = trace_file_line_count
    if line_cap is not None:
        parse_line_count = min(trace_file_line_count, line_cap)
    if parse_line_count <= 0:
        return []

    traces: list[list[dict]] = []
    for start in range(1, parse_line_count + 1, chunk_lines):
        end = min(start + chunk_lines - 1, parse_line_count)
        env._execute_raw(
            f"sed -n '{start},{end}p' {TRACE_FILE_PATH} > {TRACE_PARSE_PATH}",
            timeout=60,
        )
        traces.extend(
            parse_fault_traces_from_file(
                env,
                instrumented_callables,
                repo_path,
                alt_path,
                require_patched=False,
                trace_file_path=TRACE_PARSE_PATH,
            )
        )

    env._execute_raw(f"rm -f {TRACE_PARSE_PATH}", timeout=60)
    return traces


def _swebench_f2p_nodeids(task: dict) -> list[str]:
    """Return raw SWE-bench FAIL_TO_PASS selectors in dataset order."""
    f2p_raw = task.get("FAIL_TO_PASS")
    if not f2p_raw:
        return []
    if isinstance(f2p_raw, str):
        try:
            f2p_list = json.loads(f2p_raw)
        except (json.JSONDecodeError, TypeError):
            f2p_list = [f2p_raw]
    elif isinstance(f2p_raw, list):
        f2p_list = f2p_raw
    else:
        return []
    out = []
    seen = set()
    for item in f2p_list:
        nodeid = str(item).strip()
        if nodeid and nodeid not in seen:
            seen.add(nodeid)
            out.append(nodeid)
    return out


def _django_labels_from_f2p(nodeids: list[str]) -> list[str]:
    """Convert unittest display names to Django runtests labels."""
    labels = []
    seen = set()
    for nodeid in nodeids:
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s+\(([^)]+)\)$", nodeid.strip())
        if not match:
            continue
        method, qual = match.groups()
        label = qual if qual.rsplit(".", 1)[-1] == method else f"{qual}.{method}"
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


def _swebench_output_has_f2p_failure(raw_output: str, f2p_nodeids: list[str]) -> bool:
    """Detect F2P failures when run_tests cleanup masks pytest's exit code."""
    if not raw_output or not f2p_nodeids:
        return False
    lines = [_ANSI_ESCAPE_RE.sub("", line) for line in raw_output.splitlines()]
    for nodeid in f2p_nodeids:
        bare = _normalize_test_func_name(nodeid)
        for line in lines:
            if _swebench_line_matches_f2p_selector(line, nodeid, bare) and re.search(r"\b(FAILED|ERROR|FAIL)\b", line):
                return True
    return False


_TEST_RESULT_LINE_RE = re.compile(r"\b(PASSED|FAILED|ERROR|FAIL|ok)\b|\.{3}\s*ok\b")


def _swebench_f2p_collection_observation(raw_output: str, f2p_nodeids: list[str]) -> dict[str, list[str]]:
    """Report which SWE-bench F2P selectors appear in test result lines."""
    observed = []
    missing = []
    if not f2p_nodeids:
        return {"observed": observed, "missing": missing}
    lines = [_ANSI_ESCAPE_RE.sub("", line) for line in (raw_output or "").splitlines()]
    result_lines = [line for line in lines if _TEST_RESULT_LINE_RE.search(line)]
    for nodeid in f2p_nodeids:
        bare = _normalize_test_func_name(nodeid)
        if any(_swebench_line_matches_f2p_selector(line, nodeid, bare) for line in result_lines):
            observed.append(nodeid)
        else:
            missing.append(nodeid)
    return {"observed": observed, "missing": missing}


def _swebench_line_matches_f2p_selector(line: str, nodeid: str, bare: str) -> bool:
    selector = str(nodeid or "").strip()
    bare = str(bare or "").strip()
    if not selector and not bare:
        return False

    # Full selectors and unittest display names should match exactly. Bare
    # names need token boundaries so ``test_foo`` does not match
    # ``test_foo_bar`` in pytest summaries.
    if selector and selector != bare:
        start = line.find(selector)
        if start >= 0:
            end = start + len(selector)
            if end == len(line) or line[end] not in "_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
                return True
    if bare:
        return re.search(rf"(?<![A-Za-z0-9_]){re.escape(bare)}(?:\[[^\]]+\])?(?![A-Za-z0-9_])", line) is not None
    return selector in line


_SIGNATURE_ENTRY_FAILURE_RE = re.compile(
    r"TypeError: .*(got an unexpected keyword argument|missing .*required positional argument|takes .*positional arguments?)"
)


def _swebench_output_has_signature_entry_failure(raw_output: str) -> bool:
    """Detect failures raised by Python argument binding before body entry."""
    if not raw_output:
        return False
    return any(_SIGNATURE_ENTRY_FAILURE_RE.search(_ANSI_ESCAPE_RE.sub("", line)) for line in raw_output.splitlines())


def _swebench_output_has_zero_tests(raw_output: str) -> bool:
    """Detect successful-looking test runs that collected no tests."""
    if not raw_output:
        return False
    text = _ANSI_ESCAPE_RE.sub("", raw_output)
    return bool(
        re.search(r"\bno tests? (?:ran|run|collected)\b", text, re.IGNORECASE)
        or re.search(r"\b0 (?:tests? )?passed\b", text, re.IGNORECASE)
        or re.search(r"\btests finished:\s*0 passed\b", text, re.IGNORECASE)
    )


def _repo_relative_test_file(value: str) -> str | None:
    path = str(value or "").strip().strip("'\"")
    if not path:
        return None
    if path.startswith("a/") or path.startswith("b/"):
        path = path[2:]
    if path.startswith("./"):
        path = path[2:]
    if path.endswith(".py") and _is_test_file(path):
        return path
    return None


def _selector_test_file(selector: str) -> str | None:
    head = str(selector or "").strip().split("::", 1)[0].split(" ", 1)[0]
    return _repo_relative_test_file(head)


def _test_patch_function_files(patch_text: str | None) -> dict[str, set[str]]:
    current_file: str | None = None
    in_hunk = False
    by_func: dict[str, set[str]] = {}
    for line in str(patch_text or "").splitlines():
        diff_match = re.match(r"^diff --git a/(.+?) b/(.+)$", line)
        if diff_match:
            current_file = _repo_relative_test_file(diff_match.group(2))
            in_hunk = False
            continue
        if line.startswith("+++ "):
            target = line[4:].strip()
            if target != "/dev/null":
                current_file = _repo_relative_test_file(target)
            in_hunk = False
            continue
        if not current_file:
            continue

        search_text = ""
        if line.startswith("@@"):
            in_hunk = True
            parts = line.split("@@", 2)
            if len(parts) >= 3:
                search_text = parts[2]
        elif not in_hunk:
            continue
        elif line.startswith("+") and not line.startswith("+++"):
            search_text = line[1:]
        elif line.startswith(" "):
            search_text = line[1:]

        if not search_text:
            continue
        match = _TEST_DEF_RE.match(search_text)
        if match:
            by_func.setdefault(_strip_parametrize(match.group(1)), set()).add(current_file)
    return by_func


def _test_patch_files(patch_text: str | None) -> list[str]:
    files: set[str] = set()
    for line in str(patch_text or "").splitlines():
        diff_match = re.match(r"^diff --git a/(.+?) b/(.+)$", line)
        if diff_match:
            path = _repo_relative_test_file(diff_match.group(2))
            if path:
                files.add(path)
            continue
        if line.startswith("+++ "):
            target = line[4:].strip()
            if target != "/dev/null":
                path = _repo_relative_test_file(target)
                if path:
                    files.add(path)
    return sorted(files)


def _swebench_f2p_file_candidates(task: dict, f2p_nodeids: list[str]) -> dict[str, list[str]]:
    patch_func_files = _test_patch_function_files(task.get("test_patch"))
    candidates: dict[str, list[str]] = {}
    for nodeid in f2p_nodeids:
        selector = str(nodeid or "").strip()
        files: set[str] = set()
        selector_file = _selector_test_file(selector)
        if selector_file:
            files.add(selector_file)
        bare = _normalize_test_func_name(selector)
        if bare:
            files.update(patch_func_files.get(bare, set()))
        candidates[selector] = sorted(files)
    return candidates


_TEST_OUTPUT_FAILURE_RE = re.compile(
    r"Traceback \(most recent call last\)|"
    r"\b(?:FAILED|ERROR|FAILURES|INTERNALERROR)\b|"
    r"\b(?:ImportError|ModuleNotFoundError|SyntaxError):",
    re.IGNORECASE,
)
_TEST_OUTPUT_PASS_RE = re.compile(
    r"\btests finished:\s*[1-9][0-9]*\s+passed\b|"
    r"\b[1-9][0-9]*\s+passed\b|"
    r"\bOK\b",
    re.IGNORECASE,
)


def _swebench_output_known_clean(raw_output: str) -> bool:
    if not raw_output:
        return False
    text = _ANSI_ESCAPE_RE.sub("", raw_output)
    if _TEST_OUTPUT_FAILURE_RE.search(text):
        return False
    return bool(_TEST_OUTPUT_PASS_RE.search(text))


def _prepare_swebench_test_script(
    env,
    task: dict,
    test_script: str,
    patched_callables: list[dict] | None = None,
) -> dict:
    """Prepare the SWE-bench eval script for preinstalled image environments."""
    diag: dict[str, str | int] = {}
    run_tests = task.get("run_tests")
    if isinstance(run_tests, str) and run_tests.strip():
        env.write_file(test_script, run_tests if run_tests.endswith("\n") else f"{run_tests}\n")
        diag["swebench_run_tests_source"] = "metadata"
    else:
        diag["swebench_run_tests_source"] = "image"

    repo = str(task.get("repo") or task.get("repo_name") or "")
    f2p_nodeids = _swebench_f2p_nodeids(task)
    f2p_django_labels = _django_labels_from_f2p(f2p_nodeids)
    f2p_bare_funcs = sorted({_normalize_test_func_name(nodeid) for nodeid in f2p_nodeids if _normalize_test_func_name(nodeid)})
    f2p_files = sorted({str(nodeid).split("::", 1)[0] for nodeid in f2p_nodeids if ".py" in str(nodeid)})
    f2p_file_candidates = _swebench_f2p_file_candidates(task, f2p_nodeids)
    test_patch_files = _test_patch_files(task.get("test_patch"))
    test_patch_test_def_files = sorted(
        {file_path for files in _test_patch_function_files(task.get("test_patch")).values() for file_path in files}
    )
    diag["swebench_f2p_nodeids"] = f2p_nodeids
    diag["swebench_f2p_django_labels"] = f2p_django_labels
    diag["swebench_f2p_bare_funcs"] = f2p_bare_funcs
    diag["swebench_f2p_files"] = f2p_files
    diag["swebench_f2p_file_candidates"] = f2p_file_candidates
    diag["swebench_test_patch_files"] = test_patch_files
    diag["swebench_test_patch_test_def_files"] = test_patch_test_def_files
    skip_editable_module = {
        "scikit-learn": "sklearn",
        "scikit-learn/scikit-learn": "sklearn",
    }.get(repo)
    skip_editable_reason = "repo_override" if skip_editable_module else None
    import_targets = _detect_import_targets(env, patched_callables or [])
    diag["swebench_import_targets_before_install"] = import_targets
    if not skip_editable_module:
        source_target = next((item for item in import_targets if item.get("matches_repo_path")), None)
        if source_target:
            skip_editable_module = source_target.get("module")
            skip_editable_reason = "preinstalled_source_tree"
    skip_probe_code = None
    if skip_editable_module:
        skip_probe_code = (
            "import importlib.util; "
            f"spec = importlib.util.find_spec({skip_editable_module!r}); "
            "origin = getattr(spec, 'origin', None) if spec else None; "
            f"print('skip editable install; {skip_editable_module}=' + str(origin) + "
            f"' reason={skip_editable_reason}')"
        )
    replace_tox_current_env = repo in {"sphinx", "sphinx-doc/sphinx"}

    setup_stdout, setup_stderr = env._run(
        "ln -sfn /opt/miniconda3/envs/testbed /root/.venv && "
        "python -c 'import chardet' >/dev/null 2>&1 || "
        "PIP_ROOT_USER_ACTION=ignore python -m pip install -q chardet",
        timeout=120,
    )
    diag["swebench_prepare_stdout"] = setup_stdout
    diag["swebench_prepare_stderr"] = setup_stderr

    patch_script = f"""python - <<'PY'
import json
import shlex
from pathlib import Path

path = Path({test_script!r})
skip_module = {skip_editable_module!r}
skip_probe_code = {skip_probe_code!r}
replace_tox_current_env = {replace_tox_current_env!r}
f2p_nodeids = {f2p_nodeids!r}
f2p_django_labels = {f2p_django_labels!r}
f2p_bare_funcs = {f2p_bare_funcs!r}
f2p_files = {f2p_files!r}
f2p_file_candidates = {f2p_file_candidates!r}
test_patch_files = {test_patch_files!r}
test_patch_test_def_files = {test_patch_test_def_files!r}
before = path.read_text()
lines = before.splitlines()
changed = False
skipped_editable = 0
replaced_tox = 0
targeted_pytest = 0
targeted_django = 0
targeted_sympy = 0
sympy_file_level_selection = 0
sympy_selected_files = set()

def quote_join(items):
    return " ".join(shlex.quote(str(item)) for item in items)

def clean_path(value):
    path = str(value or "").strip().strip("'\\\"")
    if path.startswith("./"):
        path = path[2:]
    return path

for i, line in enumerate(lines):
    stripped = line.strip()
    if replace_tox_current_env and stripped.startswith("tox --current-env") and " -- " in stripped:
        test_args = quote_join(f2p_nodeids) if f2p_nodeids else stripped.split(" -- ", 1)[1].strip()
        lines[i] = "python -m pytest -rA --color=no -vv " + test_args
        changed = True
        replaced_tox += 1
        targeted_pytest += int(bool(f2p_nodeids))
        continue
    if f2p_nodeids and (stripped.startswith("pytest ") or stripped.startswith("python -m pytest ")):
        lines[i] = "python -m pytest -rA --color=no -vv " + quote_join(f2p_nodeids)
        changed = True
        targeted_pytest += 1
        continue
    if f2p_django_labels and (
        stripped.startswith("./tests/runtests.py")
        or stripped.startswith("python tests/runtests.py")
        or stripped.startswith("python ./tests/runtests.py")
    ):
        # Keep Django's SWE-bench module-level selection. Some F2P entries are
        # docstring descriptions or context-sensitive tests; narrowing to the
        # display-name subset can skip the real failing check.
        continue
    if f2p_bare_funcs and "bin/test" in stripped:
        # SymPy's historical bin/test runners do not consistently implement
        # pytest-style -k expressions; the SWE-bench script already narrows the
        # run to patched test files.
        try:
            parts = shlex.split(stripped)
        except ValueError:
            parts = stripped.split()
        selected_here = set()
        for part in parts:
            selected = clean_path(part)
            if selected.endswith(".py"):
                selected_here.add(selected)
        sympy_selected_files.update(selected_here)
        if selected_here:
            sympy_file_level_selection += 1
        continue
    if "python -m pip install" not in line or " -e ." not in line:
        continue
    if skip_module:
        lines[i] = "python -c " + repr(skip_probe_code)
        changed = True
        skipped_editable += 1
        continue
    if "--no-build-isolation" in line:
        continue
    lines[i] = line.replace("python -m pip install", "python -m pip install --no-build-isolation", 1)
    changed = True
text = "\\n".join(lines) + "\\n"
if changed and text != before:
    path.write_text(text)
print("changed=" + str(changed))
print("skipped_editable=" + str(skipped_editable))
print("replaced_tox=" + str(replaced_tox))
print("targeted_pytest=" + str(targeted_pytest))
print("targeted_django=" + str(targeted_django))
print("targeted_sympy=" + str(targeted_sympy))
print("sympy_file_level_selection=" + str(sympy_file_level_selection))
sympy_f2p_file_coverage = {{}}
sympy_selected_patch_files = sympy_selected_files & {{clean_path(item) for item in test_patch_files}}
sympy_helper_fallback_files = sympy_selected_patch_files - {{clean_path(item) for item in test_patch_test_def_files}}
sympy_f2p_helper_file_fallback = {{}}
for selector in f2p_nodeids:
    candidates = [clean_path(item) for item in f2p_file_candidates.get(str(selector), [])]
    coverage = sorted(set(candidates) & sympy_selected_files)
    fallback = []
    if not coverage and not candidates and len(sympy_helper_fallback_files) == 1:
        fallback = sorted(sympy_helper_fallback_files)
        coverage = fallback
    sympy_f2p_file_coverage[str(selector)] = coverage
    sympy_f2p_helper_file_fallback[str(selector)] = fallback
sympy_f2p_uncovered_nodeids = [
    str(selector)
    for selector in f2p_nodeids
    if not sympy_f2p_file_coverage.get(str(selector))
]
print("sympy_selected_files=" + json.dumps(sorted(sympy_selected_files)))
print("sympy_selected_patch_files=" + json.dumps(sorted(sympy_selected_patch_files)))
print("sympy_helper_fallback_files=" + json.dumps(sorted(sympy_helper_fallback_files)))
print("sympy_f2p_file_coverage=" + json.dumps(sympy_f2p_file_coverage, sort_keys=True))
print("sympy_f2p_helper_file_fallback=" + json.dumps(sympy_f2p_helper_file_fallback, sort_keys=True))
print("sympy_f2p_uncovered_nodeids=" + json.dumps(sympy_f2p_uncovered_nodeids))
print("sympy_f2p_file_coverage_complete=" + str(int(bool(f2p_nodeids) and not sympy_f2p_uncovered_nodeids)))
PY"""
    stdout, stderr = env._run(f"chmod +x {shlex.quote(test_script)} && {patch_script}", timeout=60)
    diag.update({
        "swebench_test_script_patch_stdout": stdout,
        "swebench_test_script_patch_stderr": stderr,
    })
    return diag


def _prepare_swebench_pro_test_script(env, task: dict, test_script: str) -> dict:
    """Prepare a SWE-Bench-Pro runner that follows the official file-selector contract."""

    diag: dict[str, object] = {}
    run_tests = task.get("run_tests")
    f2p_nodeids = _swebench_f2p_nodeids(task)
    selected_files = selector_files(parse_string_list(task.get("selected_test_files_to_run")))
    if not selected_files:
        selected_files = selector_files(f2p_nodeids)
    diag["swebench_pro_f2p_nodeids"] = f2p_nodeids
    diag["swebench_pro_selected_files"] = selected_files
    diag["swebench_f2p_nodeids"] = f2p_nodeids
    diag["swebench_test_script_patch_stdout"] = "targeted_pytest=1\n" if f2p_nodeids else "targeted_pytest=0\n"
    if not isinstance(run_tests, str) or not run_tests.strip():
        raise ValueError("SWE-Bench-Pro run_tests is required for dynamic precompute")

    official_script = "/tmp/p2a_swebench_pro_official_run.sh"
    selected_arg = shlex.quote(",".join(selected_files)) if selected_files else ""
    env.write_file(official_script, run_tests if run_tests.endswith("\n") else f"{run_tests}\n")
    wrapper = "\n".join(
        [
            "#!/bin/bash",
            "set -uo pipefail",
            f"cd {shlex.quote(env.repo_path)} || exit 101",
            f"chmod +x {shlex.quote(official_script)}",
            f"bash {shlex.quote(official_script)} {selected_arg}",
            "",
        ]
    )
    env.write_file(test_script, wrapper)
    stdout, stderr = env._run(
        f"chmod +x {shlex.quote(test_script)} {shlex.quote(official_script)}",
        timeout=60,
    )
    diag["swebench_pro_run_tests_source"] = "metadata"
    diag["swebench_pro_prepare_stdout"] = stdout
    diag["swebench_pro_prepare_stderr"] = stderr
    return diag


def _missing_swebench_pro_script_fields(task: dict) -> list[str]:
    missing = []
    for field in ("run_tests", "swebench_pro_parser"):
        value = task.get(field)
        if not isinstance(value, str) or not value.strip():
            missing.append(field)
    return missing


def _module_name_from_file_path(file_path: str) -> str | None:
    """Best-effort Python module name for import-target diagnostics."""
    if not file_path.endswith(".py"):
        return None
    module_path = file_path[:-3]
    for prefix in ("lib/", "src/"):
        if module_path.startswith(prefix):
            module_path = module_path[len(prefix) :]
            break
    module = module_path.replace("/", ".")
    if module.endswith(".__init__"):
        module = module[: -len(".__init__")]
    return module or None


def _detect_import_targets(env, patched_callables: list[dict]) -> list[dict]:
    """Compare patched file paths with Python's actual import targets."""
    modules = []
    seen = set()
    for item in patched_callables:
        file_path = item.get("file_path", "")
        module = _module_name_from_file_path(file_path)
        if module and module not in seen:
            seen.add(module)
            modules.append({"module": module, "file_path": file_path})
    if not modules:
        return []

    payload = json.dumps(modules)
    script = (
        "python - <<'PY'\n"
        "import importlib.util, json\n"
        f"mods = json.loads({payload!r})\n"
        "out = []\n"
        "for item in mods:\n"
        "    mod = item['module']\n"
        "    try:\n"
        "        spec = importlib.util.find_spec(mod)\n"
        "        origin = getattr(spec, 'origin', None) if spec else None\n"
        "    except Exception as exc:\n"
        "        origin = f'<error:{type(exc).__name__}:{exc}>'\n"
        "    out.append({**item, 'import_path': origin})\n"
        "print(json.dumps(out))\n"
        "PY"
    )
    stdout, _stderr, exit_code = env._execute_raw(script, timeout=60)
    if exit_code != 0:
        return [{"module": item["module"], "file_path": item["file_path"], "import_path": None, "import_probe_error": exit_code} for item in modules]
    try:
        targets = json.loads(stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return [{"module": item["module"], "file_path": item["file_path"], "import_path": None, "import_probe_error": "parse_failed"} for item in modules]

    repo_prefix = env.repo_path.rstrip("/") + "/"
    for item in targets:
        import_path = item.get("import_path")
        expected = repo_prefix + item.get("file_path", "")
        item["matches_repo_path"] = bool(import_path and import_path == expected)
        item["expected_repo_path"] = expected
    return targets


def _capture_failed(capture_diag: dict | None) -> bool:
    return bool(capture_diag and capture_diag.get("all_three_read_failed"))


def _no_trace_reason_code(test_exit: int | None, capture_diag: dict | None = None) -> str:
    if _capture_failed(capture_diag):
        return "test_setup_failure"
    if test_exit in {2, 4, 5}:
        return "test_setup_failure"
    if test_exit == 0:
        return "no_trace_exit_zero"
    if test_exit == 1:
        return "no_trace_exit_one"
    return "no_trace"


def _all_pass_reason_code(
    test_exit: int | None,
    capture_diag: dict | None = None,
    *,
    swebench_f2p_collection_missing: bool = False,
    allow_missing_f2p_collection: bool = False,
) -> str | None:
    if test_exit != 0:
        return None
    if _capture_failed(capture_diag):
        return None
    if swebench_f2p_collection_missing and not allow_missing_f2p_collection:
        return None
    return "buggy_version_passes"


def _trace_cap_inconclusive_reason(diagnostics: dict, suffix: str) -> str | None:
    if diagnostics.get("trace_event_cap_reached"):
        return f"trace_event_cap_{suffix}_inconclusive"
    if diagnostics.get("trace_parse_line_cap_reached"):
        return f"trace_parse_line_cap_{suffix}_inconclusive"
    return None


def _mark_trace_cap_inconclusive(
    diagnostics: dict,
    *,
    would_be_case_type: str,
    would_be_reason_code: str,
) -> dict:
    updated = dict(diagnostics)
    updated["trace_cap_inconclusive"] = True
    updated["would_be_case_type"] = would_be_case_type
    updated["would_be_reason_code"] = would_be_reason_code
    return updated


# ---------------------------------------------------------------------------
# Static bonus map
# ---------------------------------------------------------------------------


def _clean_callables(callables: list[dict]) -> list[dict]:
    return [{k: v for k, v in item.items() if k != "source"} for item in callables]


def compute_static_bonus_map(task: dict) -> dict:
    """Compute a static bonus map.

    Without runtime traces there is no patched-to-patched reachability signal,
    so static mode keeps every patched callable as a root.
    """
    task = normalize_task(task)
    instance_id = make_instance_id(task)
    all_modified = find_modified_callables_from_task(task)
    newly_created = find_newly_created_callables(task)

    # --- Decision tree: static layer ---
    if not all_modified:
        if newly_created:
            case_type = "newly_created"
        else:
            case_type = "no_callable"
        return _with_metadata(
            {
                "instance_id": instance_id,
                "case_type": case_type,
                "traceable": False,
                "error": False,
                "patched_callables": [],
                "newly_created_callables": newly_created,
                "call_graph_nodes": {},
                "call_graph_edges": [],
                "hop_max": 0,
                **_empty_graph_metadata(),
            },
            reason_code=case_type,
        )

    # Static mode has no trace signal for terminal-root inference.
    call_graph_nodes = {}
    for mc in all_modified:
        node_key = f"{mc['file_path']}::{mc['qualified_name']}"
        call_graph_nodes[node_key] = {
            "file_path": mc["file_path"],
            "start_line": mc["start_line"],
            "end_line": mc["end_line"],
            "hop_distance": 0,
            "raw_hop_distance": 0,
            "normalized_distance": 0.0,
            "observed_in_trace": False,
            "rewardable": True,
            "node_role": "root_cause",
            "excluded_from_hop_max": False,
        }
        if isinstance(mc.get("source"), str) and mc["source"]:
            call_graph_nodes[node_key]["source"] = mc["source"]

    return _with_metadata(
        {
            "instance_id": instance_id,
            "case_type": "static",
            "traceable": False,
            "error": False,
            "patched_callables": _clean_callables(all_modified),
            "newly_created_callables": newly_created,
            "call_graph_nodes": call_graph_nodes,
            "call_graph_edges": [],
            "hop_max": 0,
            "raw_hop_max": 0,
            "rewardable_node_count": len(call_graph_nodes),
            "excluded_non_rewardable_node_count": 0,
            "excluded_test_harness_node_count": 0,
            "excluded_test_harness_nodes": [],
            "excluded_test_adapter_node_count": 0,
            "excluded_test_adapter_nodes": [],
            "excluded_pre_symptom_node_count": 0,
            "excluded_pre_symptom_nodes": [],
            "test_harness_file_patterns": [],
            "reward_start_source": "first_non_test_after_test",
            "issue_anchor_source": "unmatched_issue_anchor",
            "reward_start_by_trace": [],
            "ground_truth_anchor_nodes": [],
            "selected_issue_anchor_nodes": [],
            "symptom_nodes": [],
            "test_adapter_nodes": [],
            "root_cause_nodes": sorted(call_graph_nodes),
            "fix_adapter_nodes": [],
            "reward_path_edges": [],
            "direct_symptom_to_root_cause_edges": [],
            "call_graph_edge_metadata": [],
            "issue_anchor_candidates": [],
        },
        reason_code="static_mode",
    )


# ---------------------------------------------------------------------------
# Dynamic bonus map — decision tree
# ---------------------------------------------------------------------------


def compute_dynamic_bonus_map(
    task: dict,
    *,
    trace_sidecar_dir: str | None = None,
    max_sidecar_traces: int | None = None,
    max_sidecar_output_chars: int = 100_000,
    sandbox_backend: str | None = None,
) -> dict:
    """Compute a dynamic bonus map using the full trace pipeline.

    Implements the decision tree:
      1. newly_created / no_callable  (static layer)
      2. instrumentation_failed -> all_pass -> no_trace -> no_gt -> no_f2p/trace_cap_inconclusive -> latent / exposed / direct
    """
    from p2a.test_setup import parse_fixups, startup_fixup_command
    from p2a.trace import (
        aggregate_traces,
        build_call_graph_from_traces,
        instrument_sandbox,
        parse_fault_traces_from_file,
    )

    task = normalize_task(task)
    instance_id = make_instance_id(task)

    all_modified = find_modified_callables_from_task(task)
    newly_created = find_newly_created_callables(task)
    backend = (sandbox_backend or os.environ.get("P2A_SANDBOX_BACKEND") or "uni_agent").lower()
    env_diag: dict = {"sandbox_backend": backend}
    env = None
    has_structured_callable_source = bool(task.get("parsed_commit_content") or task.get("parsed_commit"))

    if not all_modified and (newly_created or has_structured_callable_source or not task.get("patch")):
        if newly_created:
            case_type = "newly_created"
        else:
            case_type = "no_callable"
        return _make_result(
            instance_id,
            case_type,
            [],
            newly_created,
            error=False,
            reason_code=case_type,
            diagnostics=env_diag,
        )

    try:
        if backend == "uni_agent":
            from p2a.precompute.uni_agent_sandbox import (
                create_uni_agent_sandbox,
                find_changed_callables_via_patch,
            )

            env = create_uni_agent_sandbox(task, instance_id=instance_id)
            _debug_progress(instance_id, "sandbox_start")
            env.start()
            _debug_progress(instance_id, "sandbox_started")
            _debug_progress(instance_id, "checkout_buggy")
            env_diag.update(env.checkout_buggy_commit(task, instance_id=instance_id))
            _debug_progress(instance_id, "checkout_done")
            if env_diag.get("sandbox_code_state_mismatch"):
                return _make_result(
                    instance_id,
                    "no_trace",
                    all_modified,
                    newly_created,
                    error=True,
                    reason_code="sandbox_code_state_mismatch",
                    diagnostics=env_diag,
                )

            if getattr(env, "swebench_pro", False):
                env_diag["startup_fixups_applied"] = []
            else:
                # Repo startup fixups (source patches + dep pins) run AFTER buggy
                # checkout and BEFORE instrumentation, so the instrumented + tested
                # source is the fixed-up source (same adapter the gate uses).
                _repo = instance_id.split("__", 1)[0] if instance_id and "__" in instance_id else ""
                _debug_progress(instance_id, "startup_fixups")
                _fix_out, _ = env._run(startup_fixup_command(_repo), timeout=300)
                _fixups = parse_fixups(_fix_out)
                env_diag["startup_fixups_applied"] = _fixups
                if _fixups:
                    print(f"  [{instance_id}] startup_fixups_applied={_fixups}")

            # Uni-Agent R2E parquet keeps reward.metadata.patch but drops the
            # parsed old/new file contents. Recover callable diffs from the
            # actual buggy sandbox plus the golden patch.
            if not all_modified and task.get("patch"):
                _debug_progress(instance_id, "recover_patch_callables")
                patch_modified, patch_newly_created, patch_diag = find_changed_callables_via_patch(env, task)
                all_modified = patch_modified
                newly_created = patch_newly_created or newly_created
                env_diag.update(patch_diag)
        else:
            raise ValueError(f"Unsupported sandbox_backend={backend!r}; expected uni_agent")

        # ── Static layer ──────────────────────────────────────────────────
        if not all_modified:
            if newly_created:
                case_type = "newly_created"
            else:
                case_type = "no_callable"
            return _make_result(
                instance_id,
                case_type,
                [],
                newly_created,
                error=False,
                reason_code=case_type,
                diagnostics=env_diag,
            )

        # ── Dynamic layer: instrument → run → parse ───────────────────────
        if getattr(env, "swebench_pro", False) and not _swebench_f2p_nodeids(task):
            return _make_result(
                instance_id,
                "no_f2p",
                all_modified,
                newly_created,
                error=True,
                reason_code="missing_fail_to_pass",
                diagnostics={**env_diag, "swebench_pro": True, "swebench_pro_f2p_nodeids": []},
            )
        if getattr(env, "swebench_pro", False):
            missing_script_fields = _missing_swebench_pro_script_fields(task)
            if missing_script_fields:
                return _make_result(
                    instance_id,
                    "precompute_failed",
                    all_modified,
                    newly_created,
                    error=True,
                    reason_code="missing_swebench_pro_scripts",
                    diagnostics={
                        **env_diag,
                        "precompute_failure": True,
                        "failure_kind": "missing_swebench_pro_scripts",
                        "swebench_pro": True,
                        "missing_swebench_pro_fields": missing_script_fields,
                    },
                )

        _debug_progress(instance_id, "instrument")
        instrumented_callables = instrument_sandbox(env, all_modified)
        if not instrumented_callables:
            print(f"  [{instance_id}] instrumentation_failed: instrumentation produced 0 callables")
            return _make_result(
                instance_id,
                "instrumentation_failed",
                all_modified,
                newly_created,
                error=True,
                reason_code="instrumentation_empty",
                diagnostics={**env_diag, **_base_diagnostics(instrumented_callables_count=0)},
            )

        # Clear stale trace file, run tests
        env._run(f"rm -f {TRACE_FILE_PATH}")

        swebench_like = bool(env.swebench_verified or getattr(env, "swebench_pro", False))
        if getattr(env, "swebench_pro", False):
            test_script = "/tmp/p2a_swebench_pro_run_tests.sh"
        elif env.swebench_verified:
            test_script = "/run_tests.sh"
        else:
            test_script = f"{env.alt_path}/run_tests.sh"
        test_script_diag = {}
        if getattr(env, "swebench_pro", False):
            _debug_progress(instance_id, "prepare_swebench_pro_test_script")
            test_script_diag = _prepare_swebench_pro_test_script(env, task, test_script)
        elif env.swebench_verified:
            _debug_progress(instance_id, "prepare_swebench_test_script")
            test_script_diag = _prepare_swebench_test_script(env, task, test_script, all_modified)
        if not swebench_like:
            env._run(f"sed -i '/pytest/{{/-rA/!s/pytest/pytest -rA/}}' {test_script}")
        # Repo fixups already ran after buggy checkout (see startup_fixup_command);
        # no separate normalization shim needed here.
        timeout_env = "P2A_SWEBENCH_TEST_TIMEOUT" if swebench_like else "P2A_TEST_TIMEOUT"
        default_timeout = 900
        test_timeout = _env_int(timeout_env, default_timeout, minimum=1)
        _debug_progress(instance_id, f"run_tests timeout={test_timeout}")
        stdout, stderr, test_exit, capture_diag = _run_tests_with_file_capture(env, test_script, timeout=test_timeout)
        _debug_progress(instance_id, f"run_tests_done exit={test_exit}")
        raw_output = f"{stdout}\n{stderr}" if stderr else stdout
        f2p_nodeids = _swebench_f2p_nodeids(task)
        swebench_f2p_failure_observed = (
            swebench_like and _swebench_output_has_f2p_failure(raw_output, f2p_nodeids)
        )
        swebench_f2p_observation = (
            _swebench_f2p_collection_observation(raw_output, f2p_nodeids)
            if swebench_like
            else {"observed": [], "missing": []}
        )
        swebench_f2p_collection_missing = bool(
            swebench_like and f2p_nodeids and swebench_f2p_observation["missing"]
        )
        if test_exit == 0 and swebench_like and _swebench_output_has_zero_tests(raw_output):
            capture_diag = dict(capture_diag)
            capture_diag["test_exit_overridden_from_zero_tests"] = True
            test_exit = 5
        if test_exit == 0 and swebench_f2p_failure_observed:
            capture_diag = dict(capture_diag)
            capture_diag["test_exit_overridden_from_output"] = True
            test_exit = 1

        patch_stdout = str(test_script_diag.get("swebench_test_script_patch_stdout") or "")
        swebench_targeted_f2p = bool(re.search(r"targeted_(pytest|django|sympy)=[1-9]", patch_stdout))
        swebench_sympy_file_level_selection = bool(re.search(r"sympy_file_level_selection=[1-9]", patch_stdout))
        swebench_sympy_f2p_file_coverage_complete = bool(
            re.search(r"sympy_f2p_file_coverage_complete=1", patch_stdout)
        )
        swebench_output_known_clean = _swebench_output_known_clean(raw_output)
        allow_missing_f2p_collection = bool(
            swebench_f2p_collection_missing
            and swebench_sympy_f2p_file_coverage_complete
            and swebench_output_known_clean
        )
        all_pass_reason = _all_pass_reason_code(
            test_exit,
            capture_diag,
            swebench_f2p_collection_missing=swebench_f2p_collection_missing,
            allow_missing_f2p_collection=allow_missing_f2p_collection,
        )
        if all_pass_reason:
            all_pass_diag = _base_diagnostics(
                test_exit=test_exit,
                total_trace_entries=None,
                raw_gt_trace_count=None,
                instrumented_callables_count=len(instrumented_callables),
                stdout=stdout,
                stderr=stderr,
                test_output_capture="file",
            )
            all_pass_diag.update(env_diag)
            all_pass_diag.update(test_script_diag)
            all_pass_diag["swebench_targeted_f2p"] = swebench_targeted_f2p
            all_pass_diag["swebench_sympy_file_level_selection"] = swebench_sympy_file_level_selection
            all_pass_diag["swebench_sympy_f2p_file_coverage_complete"] = (
                swebench_sympy_f2p_file_coverage_complete
            )
            all_pass_diag["swebench_output_known_clean"] = swebench_output_known_clean
            all_pass_diag["swebench_verified"] = env.swebench_verified
            all_pass_diag["swebench_pro"] = bool(getattr(env, "swebench_pro", False))
            all_pass_diag["test_timeout"] = test_timeout
            all_pass_diag["trace_event_cap"] = _env_int("P2A_TRACE_MAX_EVENTS", DEFAULT_TRACE_MAX_EVENTS, minimum=0)
            all_pass_diag["trace_frame_cap"] = _env_int("P2A_TRACE_MAX_FRAMES", DEFAULT_TRACE_MAX_FRAMES, minimum=0)
            all_pass_diag["trace_parse_line_cap"] = _env_positive_int_or_none("P2A_TRACE_PARSE_MAX_LINES")
            all_pass_diag["trace_parse_chunk_lines"] = None
            all_pass_diag["trace_file_line_count"] = None
            all_pass_diag["trace_event_cap_reached"] = False
            all_pass_diag["trace_parse_line_cap_reached"] = False
            all_pass_diag["trace_parse_skipped"] = True
            all_pass_diag["trace_parse_skip_reason"] = "all_pass"
            all_pass_diag["test_output_capture_detail"] = capture_diag
            all_pass_diag["parsed_trace_count"] = 0
            all_pass_diag["swebench_f2p_failure_observed"] = swebench_f2p_failure_observed
            all_pass_diag["swebench_f2p_observed_nodeids"] = swebench_f2p_observation["observed"]
            all_pass_diag["swebench_f2p_missing_nodeids"] = swebench_f2p_observation["missing"]
            all_pass_diag["swebench_f2p_collection_missing"] = swebench_f2p_collection_missing
            all_pass_diag["swebench_f2p_collection_missing_allowed"] = allow_missing_f2p_collection
            all_pass_diag["raw_gt_test_funcs"] = []
            print(f"  [{instance_id}] all_pass: buggy F2P run exited cleanly. test_exit={test_exit}")
            return _make_result(
                instance_id,
                "all_pass",
                all_modified,
                newly_created,
                error=True,
                reason_code=all_pass_reason,
                diagnostics=all_pass_diag,
            )

        # ── Decision node: NO_TRACE ──────────────────────────────────
        # Also check the raw trace file for total entry count (including
        # traces without GT) to distinguish no_trace from no_gt.
        trace_file_out, _, tf_exit = env._execute_raw(f"wc -l < {TRACE_FILE_PATH} 2>/dev/null || echo 0")
        try:
            trace_file_line_count = int(trace_file_out.strip())
        except (ValueError, AttributeError):
            trace_file_line_count = 0

        # Parse captured trace lines in bounded chunks.  By default there is no
        # parser cap; P2A_TRACE_PARSE_MAX_LINES is an explicit debugging knob.
        trace_parse_line_cap = _env_positive_int_or_none("P2A_TRACE_PARSE_MAX_LINES")
        trace_parse_chunk_lines = _env_int(
            "P2A_TRACE_PARSE_CHUNK_LINES",
            DEFAULT_TRACE_PARSE_CHUNK_LINES,
            minimum=1,
        )
        parse_line_count = trace_file_line_count
        if trace_parse_line_cap is not None:
            parse_line_count = min(trace_file_line_count, trace_parse_line_cap)
        _debug_progress(
            instance_id,
            f"parse_traces lines={parse_line_count}/{trace_file_line_count} "
            f"cap={trace_parse_line_cap} chunk={trace_parse_chunk_lines}",
        )
        raw_traces_all = _parse_fault_traces_in_chunks(
            env,
            parse_fault_traces_from_file,
            instrumented_callables,
            env.repo_path,
            env.alt_path,
            trace_file_line_count=trace_file_line_count,
            line_cap=trace_parse_line_cap,
            chunk_lines=trace_parse_chunk_lines,
        )
        _debug_progress(instance_id, f"parse_traces_done parsed={len(raw_traces_all)}")
        raw_traces = [trace for trace in raw_traces_all if any(frame.get("is_patched") for frame in trace)]
        parsed_trace_count = len(raw_traces_all)
        total_trace_entries = max(trace_file_line_count, parsed_trace_count)
        common_diag = _base_diagnostics(
            test_exit=test_exit,
            total_trace_entries=total_trace_entries,
            raw_gt_trace_count=len(raw_traces),
            instrumented_callables_count=len(instrumented_callables),
            stdout=stdout,
            stderr=stderr,
            test_output_capture="file",
        )
        common_diag.update(env_diag)
        common_diag.update(test_script_diag)
        common_diag["swebench_targeted_f2p"] = swebench_targeted_f2p
        common_diag["swebench_sympy_file_level_selection"] = swebench_sympy_file_level_selection
        common_diag["swebench_sympy_f2p_file_coverage_complete"] = swebench_sympy_f2p_file_coverage_complete
        common_diag["swebench_output_known_clean"] = swebench_output_known_clean
        common_diag["swebench_verified"] = env.swebench_verified
        common_diag["swebench_pro"] = bool(getattr(env, "swebench_pro", False))
        common_diag["test_timeout"] = test_timeout
        trace_event_cap = _env_int("P2A_TRACE_MAX_EVENTS", DEFAULT_TRACE_MAX_EVENTS, minimum=0)
        trace_frame_cap = _env_int("P2A_TRACE_MAX_FRAMES", DEFAULT_TRACE_MAX_FRAMES, minimum=0)
        common_diag["trace_event_cap"] = trace_event_cap
        common_diag["trace_frame_cap"] = trace_frame_cap
        common_diag["trace_parse_line_cap"] = trace_parse_line_cap
        common_diag["trace_parse_chunk_lines"] = trace_parse_chunk_lines
        common_diag["trace_file_line_count"] = trace_file_line_count
        common_diag["trace_event_cap_reached"] = trace_event_cap > 0 and trace_file_line_count >= trace_event_cap
        common_diag["trace_parse_line_cap_reached"] = (
            trace_parse_line_cap is not None and trace_file_line_count > trace_parse_line_cap
        )
        common_diag["test_output_capture_detail"] = capture_diag
        common_diag["parsed_trace_count"] = parsed_trace_count
        common_diag["swebench_f2p_failure_observed"] = swebench_f2p_failure_observed
        common_diag["swebench_f2p_observed_nodeids"] = swebench_f2p_observation["observed"]
        common_diag["swebench_f2p_missing_nodeids"] = swebench_f2p_observation["missing"]
        common_diag["swebench_f2p_collection_missing"] = swebench_f2p_collection_missing
        common_diag["swebench_f2p_collection_missing_allowed"] = allow_missing_f2p_collection
        common_diag["raw_gt_test_funcs"] = _test_func_names_from_traces(raw_traces)
        common_diag.update(
            _write_trace_sidecar(
                instance_id=instance_id,
                sidecar_dir=trace_sidecar_dir,
                raw_traces=raw_traces_all,
                raw_gt_traces=raw_traces,
                diagnostics=common_diag,
                stdout=stdout,
                stderr=stderr,
                max_sidecar_traces=max_sidecar_traces,
                max_output_chars=max_sidecar_output_chars,
            )
        )

        if parsed_trace_count == 0:
            import_targets = _detect_import_targets(env, all_modified)
            common_diag["import_targets"] = import_targets
            imports_match = bool(import_targets) and all(item.get("matches_repo_path") for item in import_targets)
            if swebench_like and swebench_f2p_failure_observed and imports_match and _swebench_output_has_signature_entry_failure(raw_output):
                print(f"  [{instance_id}] signature_mismatch: F2P failed before instrumented callable body entry. instrumented={len(instrumented_callables)}")
                return _make_result(
                    instance_id,
                    "signature_mismatch",
                    all_modified,
                    newly_created,
                    error=False,
                    reason_code="signature_mismatch_before_entry",
                    diagnostics=common_diag,
                )
            if newly_created and swebench_like and swebench_f2p_failure_observed and imports_match:
                print(f"  [{instance_id}] newly_created: F2P failed but only added callables were exercised. instrumented={len(instrumented_callables)}")
                return _make_result(
                    instance_id,
                    "newly_created",
                    all_modified,
                    newly_created,
                    error=False,
                    reason_code="newly_created_not_traceable",
                    diagnostics=common_diag,
                )
            if test_exit == 0 and swebench_f2p_collection_missing:
                print(f"  [{instance_id}] no_trace: F2P selectors missing from test output. instrumented={len(instrumented_callables)}")
                return _make_result(
                    instance_id,
                    "no_trace",
                    all_modified,
                    newly_created,
                    error=True,
                    reason_code="f2p_collection_missing",
                    diagnostics=common_diag,
                )
            no_trace_reason = _no_trace_reason_code(test_exit, capture_diag)
            print(f"  [{instance_id}] no_trace: 0 trace entries. test_exit={test_exit}, instrumented={len(instrumented_callables)}")
            return _make_result(
                instance_id,
                "no_trace",
                all_modified,
                newly_created,
                error=True,
                reason_code=no_trace_reason,
                diagnostics=common_diag,
            )

        # ── Decision node: NO_GT ─────────────────────────────────────
        # total_trace_entries > 0 but raw_traces (filtered to GT) == 0
        if not raw_traces:
            print(f"  [{instance_id}] no_gt: {total_trace_entries} trace entries but 0 contain GT callables. test_exit={test_exit}")
            return _make_result(
                instance_id,
                "no_gt",
                all_modified,
                newly_created,
                error=True,
                reason_code="patched_never_invoked",
                diagnostics=common_diag,
            )

        # ── Decision node: NO_F2P ────────────────────────────────────
        f2p_test_funcs = _get_f2p_test_funcs(task, raw_output, swebench_like)

        if f2p_test_funcs is None:
            # Can't parse test output at all
            parse_diag = dict(common_diag)
            parse_diag["f2p_test_funcs"] = None
            if test_exit == 0 and swebench_f2p_collection_missing:
                case_type = "no_f2p"
                reason_code = "f2p_collection_missing"
            else:
                case_type = "no_f2p"
                reason_code = "f2p_parse_failed"
            cap_reason = _trace_cap_inconclusive_reason(parse_diag, "f2p_parse")
            if cap_reason:
                case_type = "trace_cap_inconclusive"
                parse_diag = _mark_trace_cap_inconclusive(
                    parse_diag,
                    would_be_case_type="no_f2p",
                    would_be_reason_code=reason_code,
                )
            print(f"  [{instance_id}] {case_type}: f2p_test_funcs=None (parse failed). Dropping all {len(raw_traces)} traces. test_exit={test_exit}")
            parse_diag.update(
                _write_trace_sidecar(
                    instance_id=instance_id,
                    sidecar_dir=trace_sidecar_dir,
                    raw_traces=raw_traces_all,
                    raw_gt_traces=raw_traces,
                    diagnostics=parse_diag,
                    stdout=stdout,
                    stderr=stderr,
                    max_sidecar_traces=max_sidecar_traces,
                    max_output_chars=max_sidecar_output_chars,
                )
            )
            return _make_result(
                instance_id,
                case_type,
                all_modified,
                newly_created,
                error=True,
                reason_code=cap_reason or reason_code,
                diagnostics=parse_diag,
            )

        if len(f2p_test_funcs) == 0:
            no_f2p_diag = dict(common_diag)
            no_f2p_diag["f2p_test_funcs"] = []
            if test_exit == 0 and swebench_f2p_collection_missing:
                case_type = "no_f2p"
                reason_code = "f2p_collection_missing"
            else:
                case_type = "no_f2p"
                reason_code = "test_collection_error_no_failed_nodeids"
            cap_reason = _trace_cap_inconclusive_reason(no_f2p_diag, "no_failed_nodeids")
            if cap_reason:
                case_type = "trace_cap_inconclusive"
                no_f2p_diag = _mark_trace_cap_inconclusive(
                    no_f2p_diag,
                    would_be_case_type="no_f2p",
                    would_be_reason_code=reason_code,
                )
            print(f"  [{instance_id}] {case_type}: 0 parsed test failures. test_exit={test_exit}")
            no_f2p_diag.update(
                _write_trace_sidecar(
                    instance_id=instance_id,
                    sidecar_dir=trace_sidecar_dir,
                    raw_traces=raw_traces_all,
                    raw_gt_traces=raw_traces,
                    diagnostics=no_f2p_diag,
                    stdout=stdout,
                    stderr=stderr,
                    max_sidecar_traces=max_sidecar_traces,
                    max_output_chars=max_sidecar_output_chars,
                )
            )
            return _make_result(
                instance_id,
                case_type,
                all_modified,
                newly_created,
                error=True,
                reason_code=cap_reason or reason_code,
                diagnostics=no_f2p_diag,
            )

        f2p_traces = _filter_traces_to_f2p(raw_traces, f2p_test_funcs)
        f2p_diag = dict(common_diag)
        f2p_diag["f2p_test_funcs"] = sorted(f2p_test_funcs)
        f2p_diag["f2p_trace_count"] = len(f2p_traces)
        f2p_diag["f2p_recovery_source"] = None
        f2p_diag["f2p_recovery_test_funcs"] = None
        f2p_diag["raw_gt_test_funcs_before_f2p"] = f2p_diag.get("raw_gt_test_funcs", [])
        print(f"  [{instance_id}] F2P filter: {len(raw_traces)} → {len(f2p_traces)} (F2P funcs: {f2p_test_funcs})")

        if not f2p_traces and not swebench_like:
            added_test_funcs = _test_func_names_from_callables(newly_created)
            recovered_traces = _filter_traces_to_f2p(raw_traces, added_test_funcs)
            if recovered_traces:
                print(f"  [{instance_id}] F2P recovery via added tests: {len(raw_traces)} → {len(recovered_traces)} (added tests: {added_test_funcs})")
                f2p_traces = recovered_traces
                f2p_diag["f2p_trace_count"] = len(f2p_traces)
                f2p_diag["f2p_recovery_source"] = "newly_created_tests"
                f2p_diag["f2p_recovery_test_funcs"] = sorted(added_test_funcs)

        if not f2p_traces and swebench_like and common_diag.get("swebench_targeted_f2p"):
            print(f"  [{instance_id}] F2P recovery via targeted SWE-bench run: {len(raw_traces)} traces")
            f2p_traces = raw_traces
            f2p_diag["f2p_trace_count"] = len(f2p_traces)
            f2p_diag["f2p_recovery_source"] = "targeted_swebench_run"
            f2p_diag["f2p_recovery_test_funcs"] = sorted(f2p_test_funcs)

        f2p_diag["raw_gt_test_funcs"] = _test_func_names_from_traces(f2p_traces)

        if not f2p_traces:
            case_type = "no_f2p"
            reason_code = "f2p_filter_dropped"
            cap_reason = _trace_cap_inconclusive_reason(f2p_diag, "no_f2p")
            if cap_reason:
                case_type = "trace_cap_inconclusive"
                f2p_diag = _mark_trace_cap_inconclusive(
                    f2p_diag,
                    would_be_case_type="no_f2p",
                    would_be_reason_code=reason_code,
                )
            print(f"  [{instance_id}] {case_type}: F2P filter removed all traces")
            f2p_diag.update(
                _write_trace_sidecar(
                    instance_id=instance_id,
                    sidecar_dir=trace_sidecar_dir,
                    raw_traces=raw_traces_all,
                    raw_gt_traces=raw_traces,
                    f2p_traces=f2p_traces,
                    diagnostics=f2p_diag,
                    stdout=stdout,
                    stderr=stderr,
                    max_sidecar_traces=max_sidecar_traces,
                    max_output_chars=max_sidecar_output_chars,
                )
            )
            return _make_result(
                instance_id,
                case_type,
                all_modified,
                newly_created,
                error=True,
                reason_code=cap_reason or reason_code,
                diagnostics=f2p_diag,
            )

        # ── Build call graph from F2P+GT traces ──────────────────────
        traces = aggregate_traces(f2p_traces)

        def _read_file(rel_path: str) -> str:
            from p2a.trace import _read_sandbox_file

            content, exit_code = _read_sandbox_file(env, f"{env.repo_path}/{rel_path}")
            return content if exit_code == 0 else ""

        result = build_call_graph_from_traces(
            traces,
            all_modified,
            file_reader=_read_file,
            issue_text=_task_issue_text(task),
        )

        # ── Decision node: STANDARD vs DIRECT ────────────────────────
        nodes = result.get("call_graph_nodes", {})
        n_intermediate = sum(
            1
            for v in nodes.values()
            if v.get("rewardable", True) and v.get("normalized_distance", 0) > 0
        )

        result["instance_id"] = instance_id
        if n_intermediate > 0:
            result["case_type"] = LATENT_CASE
            case_type = LATENT_CASE if bonus_map_pattern_computable(result) else EXPOSED_CASE
        else:
            case_type = DIRECT_CASE
        result["case_type"] = case_type
        result["traceable"] = True
        result["error"] = False
        result["newly_created_callables"] = newly_created
        f2p_diag.update(
            _write_trace_sidecar(
                instance_id=instance_id,
                sidecar_dir=trace_sidecar_dir,
                raw_traces=raw_traces_all,
                raw_gt_traces=raw_traces,
                f2p_traces=f2p_traces,
                aggregated_f2p_traces=traces,
                diagnostics=f2p_diag,
                stdout=stdout,
                stderr=stderr,
                max_sidecar_traces=max_sidecar_traces,
                max_output_chars=max_sidecar_output_chars,
            )
        )
        enrich_bonus_map_case_metadata(result)
        return _with_metadata(result, reason_code=result.get("case_type") or case_type, diagnostics=f2p_diag)

    except Exception as e:
        exception_traceback = traceback.format_exc()
        print(f"  [WARN] Dynamic tracing failed for {instance_id}: {e}")
        print(exception_traceback, end="")
        exception_diag = _base_diagnostics()
        exception_diag.update(env_diag)
        exception_diag.update(
            {
                "precompute_failure": True,
                "failure_kind": _classify_precompute_exception(e),
                "exception_type": type(e).__name__,
                "exception_message": str(e),
                "exception_traceback_tail": _tail_text(exception_traceback, 8000),
            }
        )
        return _make_result(
            instance_id,
            "precompute_failed",
            all_modified,
            newly_created,
            error=True,
            reason_code="precompute_exception",
            diagnostics=exception_diag,
        )
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _make_result(
    instance_id: str,
    case_type: str,
    patched_callables: list[dict],
    newly_created_callables: list[dict],
    *,
    error: bool = False,
    reason_code: str | None = None,
    diagnostics: dict | None = None,
) -> dict:
    """Build a result dict for untraceable cases."""
    return _with_metadata(
        {
            "instance_id": instance_id,
            "case_type": case_type,
            "traceable": False,
            "error": error,
            "patched_callables": _clean_callables(patched_callables),
            "newly_created_callables": newly_created_callables,
            "call_graph_nodes": {},
            "call_graph_edges": [],
            "hop_max": 0,
            **_empty_graph_metadata(),
        },
        reason_code=reason_code or case_type,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Parallel processing & CLI
# ---------------------------------------------------------------------------


def _process_one(args):
    """Worker function for parallel processing."""
    (
        idx,
        task_json,
        output_dir,
        mode,
        trace_sidecar_dir,
        max_sidecar_traces,
        max_sidecar_output_chars,
        sandbox_backend,
    ) = args
    instance_id = "unknown"
    output_path = None
    try:
        task = json.loads(task_json) if isinstance(task_json, str) else dict(task_json)
        try:
            instance_id = make_instance_id(normalize_task(task))
        except Exception:
            instance_id = "unknown"

        if mode == "static":
            result = compute_static_bonus_map(task)
        else:
            result = compute_dynamic_bonus_map(
                task,
                trace_sidecar_dir=trace_sidecar_dir,
                max_sidecar_traces=max_sidecar_traces,
                max_sidecar_output_chars=max_sidecar_output_chars,
                sandbox_backend=sandbox_backend,
            )

        instance_id = result["instance_id"]
        output_path = os.path.join(output_dir, f"{instance_id}.json")
        case_type = result.get("case_type", "unknown")

        if _is_retryable_precompute_failure(result):
            _remove_retryable_bonus_map(output_path)
            return {
                "idx": idx,
                "instance_id": instance_id,
                "case_type": case_type,
                "error": None,
                "failure": _failure_record(
                    idx=idx,
                    instance_id=instance_id,
                    output_path=output_path,
                    result=result,
                ),
            }

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        return {
            "idx": idx,
            "instance_id": instance_id,
            "case_type": case_type,
            "error": None,
            "failure": None,
        }
    except Exception as e:
        return {
            "idx": idx,
            "instance_id": instance_id,
            "case_type": "error",
            "error": str(e),
            "failure": _failure_record(
                idx=idx,
                instance_id=instance_id,
                output_path=output_path,
                error=e,
            ),
        }


def main() -> int:
    from p2a.hf_assets import shared_bonus_maps_dir

    parser = argparse.ArgumentParser(description="Precompute P2A bonus maps")
    parser.add_argument("parquet_path", help="Path to dataset parquet file")
    parser.add_argument(
        "--output_dir",
        default=str(shared_bonus_maps_dir()),
        help="Output directory for bonus map JSONs (default: data/bonus_maps)",
    )
    parser.add_argument("--mode", choices=["static", "dynamic"], default="static", help="static: AST diff only. dynamic: full trace pipeline")
    parser.add_argument(
        "--sandbox_backend",
        choices=["uni_agent"],
        default=os.environ.get("P2A_SANDBOX_BACKEND", "uni_agent"),
        help="Sandbox backend for dynamic mode (uni_agent, backed by the ARL deployment).",
    )
    parser.add_argument("--n_parallel", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N instances")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N rows before applying --limit")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="(default) skip rows whose <instance_id>.json already exists in --output_dir",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="rebuild maps even if <instance_id>.json already exists (overrides the default skip)",
    )
    parser.add_argument(
        "--no_skip_filter",
        action="store_true",
        help="Do NOT exclude instances listed in src/config/bad_instances.json "
        "(default: exclude them, matching the training-data skip filter)",
    )
    parser.add_argument(
        "--save_trace_sidecars",
        action="store_true",
        help="Write raw/f2p/aggregated trace debug sidecars as .json.gz files",
    )
    parser.add_argument(
        "--trace_sidecar_dir",
        default=None,
        help="Directory for trace sidecars. Defaults to <output_dir>/traces when --save_trace_sidecars is set.",
    )
    parser.add_argument(
        "--max_sidecar_traces",
        type=int,
        default=None,
        help="Optional max traces saved per trace set in each sidecar. Unset saves all traces.",
    )
    parser.add_argument(
        "--max_sidecar_output_chars",
        type=int,
        default=100_000,
        help="Max stdout/stderr tail chars saved in each trace sidecar.",
    )
    args = parser.parse_args()
    # Skipping already-built maps is the default; --rebuild forces a full recompute.
    skip_existing = not args.rebuild

    os.makedirs(args.output_dir, exist_ok=True)
    trace_sidecar_dir = None
    if args.save_trace_sidecars:
        trace_sidecar_dir = args.trace_sidecar_dir or os.path.join(args.output_dir, "traces")
        os.makedirs(trace_sidecar_dir, exist_ok=True)

    df = pd.read_parquet(args.parquet_path)
    if args.offset:
        if args.offset < 0:
            raise ValueError("--offset must be >= 0")
        df = df.iloc[args.offset :]
    if args.limit:
        df = df.head(args.limit)

    print(f"Loaded {len(df)} instances from {args.parquet_path}")
    print(f"Mode: {args.mode}, Output: {args.output_dir}, Workers: {args.n_parallel}")
    if args.mode == "dynamic":
        print(f"Dynamic sandbox backend: {args.sandbox_backend}")
    if trace_sidecar_dir:
        print(f"Trace sidecars: {trace_sidecar_dir}")

    # Same skip-case registry consumed by the training-data filter, so the
    # bonus-map precompute and training paths never diverge (src/config/bad_instances.json).
    if args.no_skip_filter:
        skip_ids: set[str] = set()
    else:
        from p2a.skip_cases import load_skip_ids

        skip_ids = load_skip_ids()
        print(f"Skip-case registry: {len(skip_ids)} bad instances will be excluded")
    n_skipped_bad = 0
    n_skipped_language = 0

    work_items = []
    n_skipped_existing = 0
    n_retryable_existing = 0
    for idx, row in df.iterrows():
        task_payload = row.to_dict()
        try:
            normalized_payload = normalize_task(task_payload)
            instance_id = make_instance_id(normalized_payload)
        except Exception:
            instance_id = None
            normalized_payload = task_payload
        if (
            normalized_payload.get("data_source") == "swebench-pro"
            and str(normalized_payload.get("repo_language") or "").lower() != "python"
        ):
            n_skipped_language += 1
            continue
        if instance_id and instance_id in skip_ids:
            n_skipped_bad += 1
            continue
        if skip_existing:
            if instance_id:
                existing_path = os.path.join(args.output_dir, f"{instance_id}.json")
                if os.path.exists(existing_path):
                    if _existing_bonus_map_is_complete(existing_path):
                        n_skipped_existing += 1
                        continue
                    n_retryable_existing += 1
        work_items.append(
            (
                idx,
                task_payload,
                args.output_dir,
                args.mode,
                trace_sidecar_dir,
                args.max_sidecar_traces,
                args.max_sidecar_output_chars,
                args.sandbox_backend,
            )
        )

    print(
        f"Excluded {n_skipped_bad} bad instances (skip registry), "
        f"{n_skipped_language} unsupported-language instances; processing {len(work_items)} work items"
    )
    if skip_existing:
        print(f"Skipped {n_skipped_existing} complete existing maps; retrying {n_retryable_existing} retryable/incomplete maps")

    if not work_items:
        print("No work items to process after applying offset/limit/skip_existing.")
        print(f"Bonus maps saved to: {args.output_dir}")
        return 0

    from collections import Counter

    case_counts = Counter()
    error_count = 0
    failure_records = []
    failure_manifest_path = os.path.join(args.output_dir, FAILURE_MANIFEST_NAME)
    total = len(work_items)

    if args.n_parallel <= 1:
        for item in work_items:
            outcome = _process_one(item)
            idx = outcome["idx"]
            case_type = outcome["case_type"]
            error = outcome["error"]
            failure = outcome["failure"]
            if error:
                error_count += 1
                print(f"  [{idx}] ERROR: {error}")
            if failure:
                failure_records.append(failure)
                _append_failure_manifest(failure_manifest_path, [failure])
            case_counts[case_type] += 1
            done = sum(case_counts.values())
            if done % 100 == 0:
                traceable = case_counts[DIRECT_CASE] + case_counts[LATENT_CASE] + case_counts[EXPOSED_CASE]
                print(f"  Progress: {done}/{total} (traceable: {traceable}, errors: {error_count})")
    else:
        with ThreadPoolExecutor(max_workers=args.n_parallel) as executor:
            futures = {executor.submit(_process_one, item): item for item in work_items}
            done_count = 0
            for future in as_completed(futures):
                outcome = future.result()
                case_type = outcome["case_type"]
                error = outcome["error"]
                failure = outcome["failure"]
                done_count += 1
                if error:
                    error_count += 1
                if failure:
                    failure_records.append(failure)
                    _append_failure_manifest(failure_manifest_path, [failure])
                case_counts[case_type] += 1
                if done_count % 100 == 0:
                    traceable = case_counts[DIRECT_CASE] + case_counts[LATENT_CASE] + case_counts[EXPOSED_CASE]
                    print(f"  Progress: {done_count}/{total} (traceable: {traceable}, errors: {error_count})")

    # Summary table
    traceable = case_counts[DIRECT_CASE] + case_counts[LATENT_CASE] + case_counts[EXPOSED_CASE]
    sum(case_counts[ct] for ct in ("no_trace", "no_gt", "no_f2p"))

    print(f"\n{'=' * 50}")
    print(f"Summary: {total} instances from {args.parquet_path}")
    print(f"{'=' * 50}")

    print(f"\ntraceable/          {traceable:5d}  ({100 * traceable / total:.1f}%)")
    print(f"  direct             {case_counts[DIRECT_CASE]:5d}")
    print(f"  latent             {case_counts[LATENT_CASE]:5d}")
    print(f"  exposed            {case_counts[EXPOSED_CASE]:5d}")

    print(f"\nuntraceable/        {total - traceable:5d}  ({100 * (total - traceable) / total:.1f}%)")
    print(f"  newly_created      {case_counts['newly_created']:5d}")
    print(f"  no_callable        {case_counts['no_callable']:5d}")
    if case_counts["static"] or case_counts["instrumentation_failed"]:
        print(f"  static             {case_counts['static']:5d}")
        print(f"  instrumentation_failed {case_counts['instrumentation_failed']:5d}")
    if case_counts["signature_mismatch"]:
        print(f"  signature_mismatch {case_counts['signature_mismatch']:5d}")
    print(f"  all_pass  (error)  {case_counts['all_pass']:5d}")
    print(f"  no_trace  (error)  {case_counts['no_trace']:5d}")
    print(f"  no_gt     (error)  {case_counts['no_gt']:5d}")
    print(f"  no_f2p    (error)  {case_counts['no_f2p']:5d}")
    if case_counts["trace_cap_inconclusive"]:
        print(f"  trace_cap_inconclusive (error) {case_counts['trace_cap_inconclusive']:5d}")
    if case_counts["precompute_failed"]:
        print(f"  precompute_failed (retryable) {case_counts['precompute_failed']:5d}")

    if error_count:
        print(f"\nprocess errors       {error_count:5d}")
    if failure_records:
        print(f"\nretryable precompute failures {len(failure_records):5d}")
        print(f"  failure manifest: {failure_manifest_path}")
        for record in failure_records[:20]:
            message = record.get("message") or ""
            if len(message) > 160:
                message = message[-160:]
            print(f"  - {record['instance_id']}: {record.get('failure_kind')} {message}")
        if len(failure_records) > 20:
            print(f"  ... {len(failure_records) - 20} more in manifest")

    print(f"\nBonus maps saved to: {args.output_dir}")
    if failure_records or error_count:
        print("Bonus map precompute incomplete; rerun after resolving retryable failures.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
