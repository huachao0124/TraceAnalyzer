"""
P2A (Process-to-Advantage) core module.

Implements the V2 multiplicative advantage reshaping based on Graph distance:
  A_token = A_seq * m(d)^sign(A)
  where m(d) = m_max^(1-d), d = normalized hop distance in [0,1].

Components:
  - BonusMapStore: loads precomputed bonus maps (per instance_id)
  - parse_read_actions: extracts file viewing actions from agent responses
  - parse_write_actions_from_tool_calls: extracts file writing/editing actions
  - match_reads_to_graph: matches Read actions to Graph nodes
  - compute_p2a_multiplier: V2 multiply/divide formula

Tracking modes:
  - "view_only": only track file_editor view commands
  - "view_and_bash": also track cat/grep/head/tail/sed in execute_bash
"""

import json
import os
import re
import shlex
from typing import Any


class BonusMapStore:
    """Loads and caches precomputed bonus maps from disk.

    Each bonus map is a JSON file at {bonus_map_dir}/{instance_id}.json.
    Older R2E maps on disk used 8-char commit prefixes, while Uni-Agent
    R2E parquet rows use 10-char prefixes. Exact matches win; a 10-char R2E
    key falls back to the 8-char filename.
    """

    def __init__(self, bonus_map_dir: str):
        self.bonus_map_dir = bonus_map_dir
        self._cache: dict[str, dict | None] = {}

    def get(self, instance_id: str) -> dict | None:
        if instance_id in self._cache:
            return self._cache[instance_id]

        for candidate_id in _bonus_map_candidate_ids(instance_id):
            path = os.path.join(self.bonus_map_dir, f"{candidate_id}.json")
            if not os.path.exists(path):
                continue

            with open(path) as f:
                data = json.load(f)
            self._cache[instance_id] = data
            self._cache[candidate_id] = data
            return data

        self._cache[instance_id] = None
        return None


def _bonus_map_candidate_ids(instance_id: str) -> list[str]:
    candidates = [instance_id]
    if "__" not in instance_id:
        return candidates

    repo, suffix = instance_id.rsplit("__", 1)
    if len(suffix) > 8 and all(ch in "0123456789abcdef" for ch in suffix.lower()):
        candidates.append(f"{repo}__{suffix[:8]}")
    return candidates


# ---------------------------------------------------------------------------
# Path normalization
# ---------------------------------------------------------------------------

_SANDBOX_PREFIXES = ["/testbed/", "/app/", "/workspace/", "/repo/"]


def _normalize_path(path: str) -> str:
    """Strip common sandbox prefixes and leading ./ from file paths."""
    path = path.strip()
    for prefix in _SANDBOX_PREFIXES:
        if path.startswith(prefix):
            return path[len(prefix) :]
    if path.startswith("./"):
        return path[2:]
    return path


# ---------------------------------------------------------------------------
# file_editor view parsing
# ---------------------------------------------------------------------------

# JSON-style: <function=file_editor>{"command": "view", "path": "...", ...}</function>
_FILE_EDITOR_JSON_PATTERN = re.compile(
    r'<function=file_editor>\s*(\{[^}]*"command"\s*:\s*"view"[^}]*\})',
    re.DOTALL,
)

# XML-style: <function=file_editor><parameter name="command">view</parameter>...
_FILE_EDITOR_XML_PATTERN = re.compile(
    r'<function=file_editor>.*?<parameter name="command">view</parameter>'
    r'.*?<parameter name="path">([^<]+)</parameter>'
    r'(?:.*?<parameter name="view_range">\[(\d+),\s*(\d+)\]</parameter>)?',
    re.DOTALL,
)

# Old-style with = instead of name=: <parameter=command>view</parameter>
_FILE_EDITOR_XML_EQ_PATTERN = re.compile(
    r"<function=file_editor>.*?<parameter=command>view</parameter>"
    r".*?<parameter=path>([^<]+)</parameter>"
    r"(?:.*?<parameter=view_range>\[(\d+),\s*(\d+)\]</parameter>)?",
    re.DOTALL,
)
_EDITOR_WRITE_JSON_PATTERN = re.compile(
    r"<function=(?:file_editor|str_replace_editor)>\s*(\{.*?\})",
    re.DOTALL,
)
_EDITOR_WRITE_XML_PATTERN = re.compile(
    r"<function=(?:file_editor|str_replace_editor)>.*?<parameter\s*(?:name=)?\"?command\"?>([^<]+)</parameter>"
    r".*?<parameter\s*(?:name=)?\"?path\"?>([^<]+)</parameter>",
    re.DOTALL,
)
_EDITOR_WRITE_XML_EQ_PATTERN = re.compile(
    r"<function=(?:file_editor|str_replace_editor)>.*?<parameter=command>([^<]+)</parameter>"
    r".*?<parameter=path>([^<]+)</parameter>",
    re.DOTALL,
)


def _parse_file_editor_views(text: str) -> list[dict]:
    """Parse file_editor view commands from agent response text."""
    reads = []

    # JSON-style
    for match in _FILE_EDITOR_JSON_PATTERN.finditer(text):
        try:
            payload = json.loads(match.group(1))
            if payload.get("command") != "view":
                continue
            path = _normalize_path(payload.get("path", ""))
            view_range = payload.get("view_range")
            if view_range and len(view_range) == 2:
                start_line, end_line = int(view_range[0]), int(view_range[1])
            else:
                start_line, end_line = 1, 999999
            reads.append({"file_path": path, "start_line": start_line, "end_line": end_line})
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    # XML-style (name="...")
    for match in _FILE_EDITOR_XML_PATTERN.finditer(text):
        path = _normalize_path(match.group(1))
        if match.group(2) and match.group(3):
            start_line, end_line = int(match.group(2)), int(match.group(3))
        else:
            start_line, end_line = 1, 999999
        reads.append({"file_path": path, "start_line": start_line, "end_line": end_line})

    # XML-style (=...)
    for match in _FILE_EDITOR_XML_EQ_PATTERN.finditer(text):
        path = _normalize_path(match.group(1))
        if match.group(2) and match.group(3):
            start_line, end_line = int(match.group(2)), int(match.group(3))
        else:
            start_line, end_line = 1, 999999
        reads.append({"file_path": path, "start_line": start_line, "end_line": end_line})

    return reads


# ---------------------------------------------------------------------------
# execute_bash file-viewing command parsing
# ---------------------------------------------------------------------------

# Extract the command parameter from execute_bash calls
# Supports: <function=execute_bash><parameter name="command">...</parameter>
#           <function=execute_bash><parameter=command>...</parameter>
#           <function=execute_bash>{"command": "..."}
_BASH_CMD_XML_PATTERN = re.compile(
    r'<function=execute_bash>.*?<parameter\s*(?:name=)?"?command"?>([^<]+)</parameter>',
    re.DOTALL,
)
_BASH_CMD_XML_EQ_PATTERN = re.compile(
    r"<function=execute_bash>.*?<parameter=command>([^<]+)</parameter>",
    re.DOTALL,
)
_BASH_CMD_JSON_PATTERN = re.compile(
    r'<function=execute_bash>\s*(\{[^}]*"command"\s*:[^}]*\})',
    re.DOTALL,
)

# --- Patterns for specific bash commands that view files ---

# cat <path>
# cat -n <path>
# Avoid matching cat with redirection (cat > file) or pipe-only usage
_CAT_PATTERN = re.compile(
    r"\bcat\s+(?:-[nAbeEstTv]+\s+)*"  # optional flags
    r"([^\s|><;`$&]+\.py\b)",  # file path (must end in .py to reduce false positives)
)

# head/tail [-n N] <path>
_HEAD_TAIL_PATTERN = re.compile(
    r"\b(head|tail)\s+"
    r"(?:-(?:n\s*)?(\d+)\s+)?"  # optional -n N or -N
    r"([^\s|><;`$&]+\.py\b)",
)

# sed -n 'START,ENDp' <path>
_SED_N_PATTERN = re.compile(
    r"\bsed\s+-n\s+['\"]?(\d+),(\d+)p['\"]?\s+"
    r"([^\s|><;`$&]+\.py\b)",
)

# grep -n <pattern> <path>  (with -n means showing line numbers → viewing context)
# Also: grep -rn, grep -Hn, etc.
_GREP_N_PATTERN = re.compile(
    r"\bgrep\s+(?:-[A-Za-z]*n[A-Za-z]*\s+)"  # flags must include -n
    r"(?:['\"][^'\"]*['\"]\s+|[^\s]+\s+)"  # pattern arg
    r"([^\s|><;`$&]+\.py\b)",  # file path
)

# grep <pattern> <path>  (without -n, still viewing file content)
_GREP_PATTERN = re.compile(
    r"\bgrep\s+(?:-[A-Za-z]+\s+)*"  # optional flags
    r"(?:['\"][^'\"]*['\"]\s+|[^\s]+\s+)"  # pattern arg
    r"([^\s|><;`$&]+\.py\b)",  # file path
)

_REDIRECT_WRITE_PATTERN = re.compile(r"(?<![0-9])(?:>>|>)\s*([^\s|;&]+\.py\b)")
_TEE_WRITE_PATTERN = re.compile(r"\btee\s+(?:-[A-Za-z]+\s+)*([^\s|;&]+\.py\b)")
_SED_I_WRITE_PATTERN = re.compile(
    r"\bsed\s+(?:-[A-Za-z]*i[A-Za-z]*\s+)+(?:['\"]?(\d+),(\d+)[^'\"]*['\"]?\s+)?"
    r"([^\s|;&]+\.py\b)"
)
_SED_EXPR_RANGE_PATTERN = re.compile(r"^\s*(\d+)\s*,\s*(\d+)")
_SHELL_BOUNDARY_TOKENS = {";", "&&", "||", "|"}
_PERL_PI_WRITE_PATTERN = re.compile(
    r"\bperl\s+(?:-[A-Za-z]*p[A-Za-z]*i[A-Za-z]*|-[A-Za-z]*i[A-Za-z]*p[A-Za-z]*)"
    r"(?:\s+-e\s+['\"][^'\"]*['\"])?\s+([^\s|;&]+\.py\b)"
)
_PYTHON_OPEN_WRITE_PATTERN = re.compile(
    r"\bopen\(\s*['\"]([^'\"]+\.py)['\"]\s*,\s*['\"][^'\"]*[wa][^'\"]*['\"]"
)

# python -c "..." is NOT a file view → skip
# awk/python scripts that read files are too complex → skip

# Default context window for commands that don't specify line ranges
_DEFAULT_CONTEXT_LINES = 50  # head/tail default context


def _extract_reads_from_command(cmd: str) -> list[dict]:
    """Extract file-view reads (cat / head / tail / sed -n / grep) from one bash command.

    Dedups by the resulting (path, start_line, end_line) so the same view is not
    emitted twice (e.g. a ``grep -n`` that matches both grep patterns).
    """
    reads: list[dict] = []
    seen: set = set()

    def add(file_path: str, start_line: int, end_line: int) -> None:
        key = (file_path, start_line, end_line)
        if key in seen:
            return
        seen.add(key)
        reads.append({"file_path": file_path, "start_line": start_line, "end_line": end_line})

    for match in _CAT_PATTERN.finditer(cmd):
        add(_normalize_path(match.group(1)), 1, 999999)
    for match in _HEAD_TAIL_PATTERN.finditer(cmd):
        n_lines = int(match.group(2)) if match.group(2) else _DEFAULT_CONTEXT_LINES
        path = _normalize_path(match.group(3))
        # tail length is unknown, so use the full range
        add(path, 1, n_lines if match.group(1) == "head" else 999999)
    for match in _SED_N_PATTERN.finditer(cmd):
        add(_normalize_path(match.group(3)), int(match.group(1)), int(match.group(2)))
    for match in _GREP_N_PATTERN.finditer(cmd):
        add(_normalize_path(match.group(1)), 1, 999999)
    for match in _GREP_PATTERN.finditer(cmd):
        add(_normalize_path(match.group(1)), 1, 999999)
    return reads


def _parse_bash_read_commands(text: str) -> list[dict]:
    """Parse execute_bash file-view commands (cat/head/tail/sed -n/grep) from raw text.

    Pulls the bash command strings out of execute_bash tool calls (XML or JSON),
    falling back to treating the whole text as inline bash, then runs each through
    :func:`_extract_reads_from_command`. Reads are deduped by (path, start, end).
    """
    bash_commands = []
    for match in _BASH_CMD_XML_PATTERN.finditer(text):
        bash_commands.append(match.group(1).strip())
    for match in _BASH_CMD_XML_EQ_PATTERN.finditer(text):
        bash_commands.append(match.group(1).strip())
    for match in _BASH_CMD_JSON_PATTERN.finditer(text):
        try:
            payload = json.loads(match.group(1))
            cmd = payload.get("command", "")
            if cmd:
                bash_commands.append(cmd.strip())
        except (json.JSONDecodeError, ValueError):
            continue

    if not bash_commands:
        bash_commands = [text]

    reads: list[dict] = []
    seen: set = set()
    for cmd in bash_commands:
        for read in _extract_reads_from_command(cmd):
            key = (read["file_path"], read["start_line"], read["end_line"])
            if key not in seen:
                seen.add(key)
                reads.append(read)
    return reads


# ---------------------------------------------------------------------------
# Uni-Agent sweagent tool format parsing (JSON tool calls)
# ---------------------------------------------------------------------------

# str_replace_editor view in JSON format:
# {"name": "str_replace_editor", "arguments": {"command": "view", "path": "...", "view_range": [start, end]}}
_STR_REPLACE_EDITOR_JSON = re.compile(
    r'"name"\s*:\s*"str_replace_editor".*?"arguments"\s*:\s*(\{[^}]*"command"\s*:\s*"view"[^}]*\})',
    re.DOTALL,
)

# Also match the text-based tool call format used by some models:
# <tool_call>str_replace_editor(command="view", path="/testbed/foo.py", view_range=[1, 50])</tool_call>
_STR_REPLACE_EDITOR_TEXT = re.compile(
    r'str_replace_editor\s*\(\s*command\s*=\s*["\']view["\']'
    r'.*?path\s*=\s*["\']([^"\']+)["\']'
    r'(?:.*?view_range\s*=\s*\[(\d+)\s*,\s*(-?\d+)\])?',
    re.DOTALL,
)
_STR_REPLACE_EDITOR_WRITE_TEXT = re.compile(
    r'str_replace_editor\s*\(\s*command\s*=\s*["\']([^"\']+)["\']'
    r'.*?path\s*=\s*["\']([^"\']+)["\']',
    re.DOTALL,
)


def _parse_sweagent_views(text: str) -> list[dict]:
    """Parse str_replace_editor view commands from Uni-Agent sweagent format."""
    reads = []

    for match in _STR_REPLACE_EDITOR_JSON.finditer(text):
        try:
            payload = json.loads(match.group(1))
            if payload.get("command") != "view":
                continue
            path = _normalize_path(payload.get("path", ""))
            view_range = payload.get("view_range")
            if view_range and len(view_range) == 2:
                start_line = int(view_range[0])
                end_line = int(view_range[1]) if int(view_range[1]) > 0 else 999999
            else:
                start_line, end_line = 1, 999999
            reads.append({"file_path": path, "start_line": start_line, "end_line": end_line})
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    for match in _STR_REPLACE_EDITOR_TEXT.finditer(text):
        path = _normalize_path(match.group(1))
        if match.group(2) and match.group(3):
            start_line = int(match.group(2))
            end_line = int(match.group(3)) if int(match.group(3)) > 0 else 999999
        else:
            start_line, end_line = 1, 999999
        reads.append({"file_path": path, "start_line": start_line, "end_line": end_line})

    return reads


def _coerce_text_wrapped_value(value: object) -> object:
    if isinstance(value, dict) and isinstance(value.get("$text"), str):
        return value["$text"]
    return value


def _command_arg(args: dict) -> str:
    command = _coerce_text_wrapped_value(args.get("command", ""))
    return command if isinstance(command, str) else ""


def parse_read_actions_from_tool_calls(tool_calls: list[dict]) -> list[dict]:
    """Parse read actions from structured tool_calls data (Uni-Agent native format).

    Args:
        tool_calls: List of tool call dicts with "function" -> {"name", "arguments"}.

    Returns:
        List of read action dicts.
    """
    reads = []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "")
        args_raw = func.get("arguments", {})
        if isinstance(args_raw, str):
            try:
                args_raw = json.loads(args_raw)
            except (json.JSONDecodeError, ValueError):
                continue
        if not isinstance(args_raw, dict):
            continue

        if name == "str_replace_editor" and _command_arg(args_raw) == "view":
            path_value = _coerce_text_wrapped_value(args_raw.get("path", ""))
            path = _normalize_path(path_value if isinstance(path_value, str) else "")
            view_range = args_raw.get("view_range")
            if view_range and len(view_range) == 2:
                start_line = int(view_range[0])
                end_line = int(view_range[1]) if int(view_range[1]) > 0 else 999999
            else:
                start_line, end_line = 1, 999999
            reads.append({"file_path": path, "start_line": start_line, "end_line": end_line})

        elif name == "execute_bash":
            cmd = _command_arg(args_raw)
            if cmd:
                reads.extend(_parse_bash_read_commands_from_str(cmd))

    return reads


def _tool_call_function(tool_call: Any) -> tuple[str, dict]:
    if not isinstance(tool_call, dict):
        return "", {}
    func = tool_call.get("function", {})
    if not isinstance(func, dict):
        return "", {}
    name = str(func.get("name", "") or "")
    args_raw = func.get("arguments", {})
    if isinstance(args_raw, str):
        try:
            args_raw = json.loads(args_raw)
        except (json.JSONDecodeError, ValueError):
            args_raw = {}
    if not isinstance(args_raw, dict):
        return name, {}
    args = dict(args_raw)
    for key in ("command", "path", "file", "old_str", "new_str"):
        if key in args:
            args[key] = _coerce_text_wrapped_value(args[key])
    return name, args


def _write_action(file_path: str, *, start_line: int = 1, end_line: int = 999999, command: str | None = None) -> dict:
    return {
        "file_path": _normalize_path(file_path),
        "start_line": int(start_line),
        "end_line": int(end_line),
        "command": command or "write",
    }


def _shell_tokens(cmd: str) -> list[str]:
    lexer = shlex.shlex(cmd, posix=True, punctuation_chars=";&|")
    lexer.whitespace_split = True
    lexer.commenters = ""
    return list(lexer)


def _sed_expression_range(expr: str) -> tuple[int, int] | None:
    match = _SED_EXPR_RANGE_PATTERN.match(expr)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _extract_sed_i_write_matches(cmd: str) -> list[tuple[str, int, int]]:
    try:
        tokens = _shell_tokens(cmd)
    except ValueError:
        return []

    writes: list[tuple[str, int, int]] = []
    index = 0
    while index < len(tokens):
        if tokens[index] != "sed":
            index += 1
            continue

        cursor = index + 1
        in_place = False
        active_range: tuple[int, int] | None = None
        while cursor < len(tokens) and tokens[cursor] not in _SHELL_BOUNDARY_TOKENS:
            token = tokens[cursor]
            if token in {"-e", "--expression"}:
                if cursor + 1 < len(tokens) and tokens[cursor + 1] not in _SHELL_BOUNDARY_TOKENS:
                    active_range = _sed_expression_range(tokens[cursor + 1]) or active_range
                    cursor += 2
                    continue
            if token in {"-f", "--file"}:
                cursor += 2
                continue
            if token.startswith("--in-place"):
                in_place = True
                cursor += 1
                continue
            if token.startswith("-") and token != "-" and not token.endswith(".py"):
                if "i" in token[1:]:
                    in_place = True
                cursor += 1
                continue
            if token == "":
                cursor += 1
                continue

            if in_place and token.endswith(".py"):
                start_line, end_line = active_range or (1, 999999)
                writes.append((token, start_line, end_line))
            elif in_place:
                active_range = _sed_expression_range(token) or active_range
            cursor += 1
        index = max(cursor, index + 1)
    return writes


def _extract_writes_from_command(cmd: str) -> list[dict]:
    writes: list[dict] = []
    seen: set[tuple[str, int, int, str]] = set()

    def add(file_path: str, start_line: int = 1, end_line: int = 999999, command: str = "write") -> None:
        action = _write_action(file_path, start_line=start_line, end_line=end_line, command=command)
        key = (action["file_path"], action["start_line"], action["end_line"], action["command"])
        if key in seen:
            return
        seen.add(key)
        writes.append(action)

    for match in _REDIRECT_WRITE_PATTERN.finditer(cmd):
        add(match.group(1), command="redirect")
    for match in _TEE_WRITE_PATTERN.finditer(cmd):
        add(match.group(1), command="tee")
    for file_path, start_line, end_line in _extract_sed_i_write_matches(cmd):
        add(file_path, start_line, end_line, "sed -i")
    for match in _SED_I_WRITE_PATTERN.finditer(cmd):
        if match.group(1) and match.group(2):
            add(match.group(3), int(match.group(1)), int(match.group(2)), "sed -i")
        else:
            add(match.group(3), command="sed -i")
    for match in _PERL_PI_WRITE_PATTERN.finditer(cmd):
        add(match.group(1), command="perl -pi")
    for match in _PYTHON_OPEN_WRITE_PATTERN.finditer(cmd):
        add(match.group(1), command="python open")
    return writes


def _parse_editor_write_actions(text: str) -> list[dict]:
    writes: list[dict] = []
    for match in _EDITOR_WRITE_JSON_PATTERN.finditer(text):
        try:
            payload = json.loads(match.group(1))
        except (json.JSONDecodeError, ValueError):
            continue
        command = str(payload.get("command", "") or "")
        path = payload.get("path") or payload.get("file")
        if command and command != "view" and isinstance(path, str) and path:
            writes.append(_write_action(path, command=command))
    for match in _EDITOR_WRITE_XML_PATTERN.finditer(text):
        command = match.group(1).strip()
        if command and command != "view":
            writes.append(_write_action(match.group(2), command=command))
    for match in _EDITOR_WRITE_XML_EQ_PATTERN.finditer(text):
        command = match.group(1).strip()
        if command and command != "view":
            writes.append(_write_action(match.group(2), command=command))
    for match in _STR_REPLACE_EDITOR_WRITE_TEXT.finditer(text):
        command = match.group(1).strip()
        if command and command != "view":
            writes.append(_write_action(match.group(2), command=command))
    return writes


def parse_write_actions(response_text: str) -> list[dict]:
    """Parse file writing/editing actions from raw tool-call text."""
    writes = _parse_editor_write_actions(response_text)
    writes.extend(_extract_writes_from_command(response_text))
    return writes


def parse_write_actions_from_tool_calls(tool_calls: list[dict]) -> list[dict]:
    """Parse write/edit actions from structured tool calls."""
    writes: list[dict] = []
    for tc in tool_calls:
        name, args = _tool_call_function(tc)
        command = _command_arg(args)
        if name in {"str_replace_editor", "file_editor"} and command and command != "view":
            path = args.get("path") or args.get("file")
            if isinstance(path, str) and path:
                writes.append(_write_action(path, command=command))
        elif name == "execute_bash":
            cmd = _command_arg(args)
            if cmd:
                writes.extend(_extract_writes_from_command(cmd))
    return writes


def _primary_tool_call(step_trace: dict) -> dict | None:
    tool_calls = step_trace.get("tool_calls") or []
    if not isinstance(tool_calls, list) or not tool_calls:
        return None
    first = tool_calls[0]
    return first if isinstance(first, dict) else None


def reads_from_step_trace(step_trace: dict, tracking_mode: str = "view_and_bash") -> list[dict]:
    reads: list[dict] = []
    tool_calls = step_trace.get("tool_calls") or []
    if isinstance(tool_calls, list) and tool_calls:
        reads.extend(parse_read_actions_from_tool_calls(tool_calls))
    response_text = step_trace.get("response_text") or step_trace.get("response") or step_trace.get("assistant_response")
    if isinstance(response_text, str) and response_text:
        reads.extend(parse_read_actions(response_text, tracking_mode=tracking_mode))
    return reads


def writes_from_step_trace(step_trace: dict) -> list[dict]:
    writes: list[dict] = []
    tool_calls = step_trace.get("tool_calls") or []
    if isinstance(tool_calls, list) and tool_calls:
        writes.extend(parse_write_actions_from_tool_calls(tool_calls))
    response_text = step_trace.get("response_text") or step_trace.get("response") or step_trace.get("assistant_response")
    if isinstance(response_text, str) and response_text:
        writes.extend(parse_write_actions(response_text))
    return writes


def normalize_action(step_trace: dict, tracking_mode: str = "view_and_bash") -> dict:
    """Normalize one step trace to a tool-agnostic action family and target."""
    if not isinstance(step_trace, dict):
        return {"family": "other", "target_path": None}

    tool_call = _primary_tool_call(step_trace)
    if tool_call is None:
        return {"family": "other", "target_path": None}

    name, args = _tool_call_function(tool_call)
    command = _command_arg(args)
    path = args.get("path")
    if isinstance(path, str) and path:
        target_path = _normalize_path(path)
    else:
        target_path = None

    if name in {"str_replace_editor", "file_editor"}:
        if command == "view":
            return {"family": "read", "target_path": target_path}
        if command:
            return {"family": "edit", "target_path": target_path}

    if name == "execute_bash":
        writes = parse_write_actions_from_tool_calls([tool_call])
        if writes:
            targets = {write["file_path"] for write in writes}
            return {"family": "edit", "target_path": sorted(targets)[0] if len(targets) == 1 else None}
        reads = parse_read_actions_from_tool_calls([tool_call]) if tracking_mode == "view_and_bash" else []
        if reads:
            targets = {read["file_path"] for read in reads}
            return {"family": "read", "target_path": sorted(targets)[0] if len(targets) == 1 else None}
        return {"family": "exec", "target_path": None}

    if parse_read_actions_from_tool_calls([tool_call]):
        return {"family": "read", "target_path": target_path}
    return {"family": "other", "target_path": target_path}


def segment_purpose_blocks(step_traces: list[dict], tracking_mode: str = "view_and_bash") -> list[dict]:
    """Segment consecutive same-family/same-target steps into purpose blocks."""
    blocks: list[dict] = []
    current: dict | None = None
    for trace_idx, trace in enumerate(step_traces):
        action = normalize_action(trace, tracking_mode=tracking_mode)
        family = action["family"]
        target_path = action["target_path"]
        step_idx = trace.get("step_idx", trace_idx) if isinstance(trace, dict) else trace_idx
        if (
            current is None
            or current["family"] != family
            or current["target_path"] != target_path
        ):
            current = {
                "family": family,
                "target_path": target_path,
                "step_indices": [step_idx],
                "trace_indices": [trace_idx],
            }
            blocks.append(current)
        else:
            current["step_indices"].append(step_idx)
            current["trace_indices"].append(trace_idx)
    return blocks


def _parse_bash_read_commands_from_str(cmd: str) -> list[dict]:
    """Parse file-viewing bash commands from a single command string."""
    if not isinstance(cmd, str):
        return []
    return _extract_reads_from_command(cmd)


# ---------------------------------------------------------------------------
# Public API: parse_read_actions (unified entry point)
# ---------------------------------------------------------------------------

TRACKING_MODES = ("view_only", "view_and_bash")


def parse_read_actions(response_text: str, tracking_mode: str = "view_only") -> list[dict]:
    """Parse file viewing actions from agent response text.

    Handles both old rLLM XML format (file_editor) and Uni-Agent sweagent
    format (str_replace_editor). Both are tried; results are merged.

    Args:
        response_text: The agent's response text (tool calls + reasoning).
        tracking_mode: One of:
            - "view_only": only view/file_editor commands
            - "view_and_bash": also cat/grep/head/tail/sed in execute_bash

    Returns:
        List of dicts with keys: file_path, start_line, end_line.
        Lines are 1-indexed. If no range, start_line=1, end_line=999999.
    """
    reads = _parse_file_editor_views(response_text)
    reads.extend(_parse_sweagent_views(response_text))

    if tracking_mode == "view_and_bash":
        reads.extend(_parse_bash_read_commands(response_text))

    return reads


# ---------------------------------------------------------------------------
# Graph matching
# ---------------------------------------------------------------------------


def is_rewardable_graph_node(node: dict) -> bool:
    """Whether a Graph node participates in P2A reward matching."""
    return bool(node.get("rewardable", True))


def match_reads_to_graph(reads: list[dict], bonus_map: dict) -> float:
    """Match Read actions against captured Graph nodes.

    For each read action, check if its file_path and line range overlap with any
    Graph node. Among all matches, return the minimum normalized distance
    (i.e., max bonus).

    Args:
        reads: List of read actions from parse_read_actions().
        bonus_map: A bonus map dict with legacy "call_graph_nodes" storage key.

    Returns:
        Minimum normalized distance in [0, 1] if any match found, -1.0 otherwise.
    """
    if not reads or not bonus_map:
        return -1.0

    nodes = bonus_map.get("call_graph_nodes", {})
    if not nodes:
        return -1.0

    min_distance = float("inf")

    for read in reads:
        read_path = read["file_path"]
        read_start = read["start_line"]
        read_end = read["end_line"]

        for _node_key, node in nodes.items():
            if not is_rewardable_graph_node(node):
                continue
            node_path = node["file_path"]
            node_start = node["start_line"]
            node_end = node["end_line"]

            # Check file path match
            if read_path != node_path:
                continue

            # Check line range overlap
            if read_start <= node_end and read_end >= node_start:
                d = node["normalized_distance"]
                min_distance = min(min_distance, d)

    if min_distance == float("inf"):
        return -1.0

    return min_distance


def is_rewardable_call_graph_node(node: dict) -> bool:
    """Compatibility alias for the legacy call-graph storage vocabulary."""
    return is_rewardable_graph_node(node)


def match_reads_to_callgraph(reads: list[dict], bonus_map: dict) -> float:
    """Compatibility alias for legacy callers; new code should use Graph."""
    return match_reads_to_graph(reads, bonus_map)


# ---------------------------------------------------------------------------
# V2 multiplier
# ---------------------------------------------------------------------------


def compute_p2a_multiplier(distance: float, m_max: float, advantage_sign: int) -> float:
    """Compute the V2 multiplicative advantage reshape factor.

    Formula:
        m(d) = m_max^(1-d)
        - A > 0 (positive advantage): multiply by m(d)  → amplify
        - A < 0 (negative advantage): multiply by 1/m(d) → shrink
        - A = 0: return 1.0 (no change)
        - off-graph (distance < 0): return 1.0 (no change)

    This preserves the sign of advantage unconditionally.

    Args:
        distance: Normalized distance in [0, 1], or < 0 if off-graph.
        m_max: Maximum multiplier (hyperparameter, typically 2-5).
        advantage_sign: Sign of the advantage (+1, -1, or 0).

    Returns:
        Multiplier to apply to the advantage.
    """
    # Off-graph: no modification
    if distance < 0:
        return 1.0

    # Zero advantage: no modification
    if advantage_sign == 0:
        return 1.0

    # m(d) = m_max^(1-d)
    m_d = m_max ** (1.0 - distance)

    if advantage_sign > 0:
        return m_d  # amplify good actions
    else:
        return 1.0 / m_d  # shrink bad actions
