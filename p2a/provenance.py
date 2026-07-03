"""Licensed / unlicensed reference scoring for model actions.

A *reference* is a provenance marker on the first mention of each entity in a
model action: the file/directory paths and code symbols the model types as
read or search targets (``str_replace_editor`` view paths; cat/sed/head/tail/
ls/find/grep targets in ``execute_bash``; identifier-shaped tokens in grep
patterns). The reference is *licensed* when the entity's literal (full path or
basename for paths; whole-word match for symbols) appears in the initial
context (task prompt including the issue) or in a strictly earlier step's
observations; otherwise it is *unlicensed* -- the model had no visible clue
and acted purely on its own prior. Steps are the ordering unit: tool calls
within one step are batched and their observations return together, so
same-step observations cannot license. Files the model itself created are
exempt; natural-language stopwords are not counted.

The scorer is independent of the bonus map and evaluable on every trace with
step traces. It is an observational metric -- NOT a reward signal and NOT a
cheating detector.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

from p2a.core import (
    _BASH_CMD_JSON_PATTERN,
    _BASH_CMD_XML_EQ_PATTERN,
    _BASH_CMD_XML_PATTERN,
    _command_arg,
    _normalize_path,
    _parse_editor_write_actions,
    _parse_file_editor_views,
    _parse_sweagent_views,
    _shell_tokens,
    _tool_call_function,
    parse_write_actions_from_tool_calls,
)

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{3,}$")
_NON_IDENTIFIER_SPLIT = re.compile(r"[^A-Za-z0-9_]+")
_HEREDOC_PATTERN = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?")
_EMBEDDED_REDIRECT_PATTERN = re.compile(r"^(?:\d|&)?>{1,2}(.+)$")
_SHELL_BOUNDARY_CHARS = set(";&|")
_GLOB_CHARS = set("*?[")

# Common programming / natural-language words that never count as symbol
# references even when they are identifier-shaped.
_SYMBOL_STOPWORDS = frozenset(
    {
        "args",
        "assert",
        "break",
        "case",
        "catch",
        "class",
        "const",
        "continue",
        "create",
        "default",
        "define",
        "delete",
        "elif",
        "else",
        "error",
        "errors",
        "except",
        "exception",
        "exit",
        "false",
        "file",
        "files",
        "final",
        "finally",
        "function",
        "import",
        "index",
        "input",
        "kwargs",
        "lambda",
        "line",
        "lines",
        "list",
        "local",
        "main",
        "module",
        "name",
        "names",
        "none",
        "null",
        "number",
        "object",
        "output",
        "pass",
        "print",
        "private",
        "public",
        "raise",
        "range",
        "return",
        "self",
        "static",
        "string",
        "super",
        "test",
        "tests",
        "this",
        "true",
        "type",
        "value",
        "values",
        "void",
        "while",
        "with",
        "yield",
    }
)

_READ_PATH_COMMANDS = {"cat", "head", "tail", "ls"}
_GREP_COMMANDS = {"grep", "egrep", "fgrep", "rg", "zgrep"}
_COMMAND_WRAPPERS = {"command", "env", "nohup", "sudo", "time", "xargs"}
_ENV_ASSIGNMENT_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# grep-family flags that consume the following token as a non-target value.
_GREP_VALUE_FLAGS = {
    "-A",
    "-B",
    "-C",
    "-D",
    "-d",
    "-f",
    "-g",
    "-m",
    "-t",
    "--color",
    "--colour",
    "--exclude",
    "--exclude-dir",
    "--file",
    "--glob",
    "--include",
    "--label",
    "--max-count",
    "--type",
}
_GREP_PATTERN_FLAGS = {"-e", "--regexp"}
_HEAD_TAIL_VALUE_FLAGS = {"-n", "-c"}
_FIND_NAME_FLAGS = {"-name", "-iname", "-path", "-ipath", "-wholename", "-iwholename"}


def _maybe_json(value: Any) -> Any:
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return value
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return value


_TEXT_FIELDS = (
    "response_text",
    "assistant_response",
    "completion",
    "response",
    "output",
    "text",
)


def _step_text(trace: Any) -> str:
    """Action text of a step for artifacts that carry text-format tool calls."""
    trace = _maybe_json(trace)
    if isinstance(trace, str):
        return trace
    if not isinstance(trace, dict):
        return ""
    parts = [trace[field] for field in _TEXT_FIELDS if isinstance(trace.get(field), str) and trace[field]]
    return "\n".join(parts)


def _step_tool_calls(trace: Any) -> list[dict]:
    trace = _maybe_json(trace)
    if not isinstance(trace, dict):
        return []
    raw = _maybe_json(trace.get("tool_calls"))
    if not isinstance(raw, list):
        return []
    calls: list[dict] = []
    for item in raw:
        item = _maybe_json(item)
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("function"), dict):
            calls.append(item)
        elif "name" in item or "tool_name" in item:
            calls.append(
                {
                    "function": {
                        "name": item.get("name", item.get("tool_name")),
                        "arguments": item.get("arguments", item.get("args", {})),
                    }
                }
            )
    return calls


def _step_observation_text(trace: Any) -> str:
    trace = _maybe_json(trace)
    if not isinstance(trace, dict):
        return ""
    parts: list[str] = []
    for result_value in trace.get("tool_results") or []:
        result = _maybe_json(result_value)
        if isinstance(result, str):
            parts.append(result)
            continue
        if not isinstance(result, dict):
            continue
        for key in ("observation", "content", "result", "output", "stderr", "stdout", "error"):
            value = result.get(key)
            if isinstance(value, str) and value:
                parts.append(value)
            elif value not in (None, ""):
                parts.append(json.dumps(value, ensure_ascii=False, default=str))
    return "\n".join(parts)


def _path_basename(path: str) -> str:
    return path.rstrip("/").rsplit("/", 1)[-1]


def _is_boundary_token(token: str) -> bool:
    return bool(token) and all(ch in _SHELL_BOUNDARY_CHARS for ch in token)


def _command_lines(cmd: str) -> list[str]:
    """Split a bash command into processable lines, dropping heredoc bodies."""
    lines: list[str] = []
    heredoc_end: str | None = None
    for line in cmd.splitlines():
        if heredoc_end is not None:
            if line.strip() == heredoc_end:
                heredoc_end = None
            continue
        match = _HEREDOC_PATTERN.search(line)
        if match:
            heredoc_end = match.group(1)
        lines.append(line)
    return lines


def _simple_commands(cmd: str) -> list[list[str]]:
    """Tokenize a bash command and split it into simple-command argv lists."""
    commands: list[list[str]] = []
    for line in _command_lines(cmd):
        try:
            tokens = _shell_tokens(line)
        except ValueError:
            continue
        current: list[str] = []
        for token in tokens:
            if _is_boundary_token(token):
                if current:
                    commands.append(current)
                current = []
            else:
                current.append(token)
        if current:
            commands.append(current)
    return commands


def _strip_redirections(argv: list[str]) -> list[str]:
    out: list[str] = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token in {">", ">>", "<", "<<", "<<<"} or re.fullmatch(r"(?:\d|&)>{1,2}", token):
            skip_next = True
            continue
        if token.startswith((">", "<")) or _EMBEDDED_REDIRECT_PATTERN.match(token):
            continue
        out.append(token)
    return out


def _strip_command_prefix(argv: list[str]) -> list[str]:
    index = 0
    while index < len(argv):
        token = argv[index]
        if _ENV_ASSIGNMENT_PATTERN.match(token):
            index += 1
            continue
        if _path_basename(token) in _COMMAND_WRAPPERS:
            index += 1
            continue
        break
    return argv[index:]


def _is_path_target(token: str) -> bool:
    if not token or token == "-" or token.startswith("-"):
        return False
    if any(ch in _GLOB_CHARS for ch in token):
        return False
    if "$(" in token or "`" in token or token.startswith("$"):
        return False
    if re.fullmatch(r"\d+(?:,\d+)?", token):
        return False
    return True


def _path_reference(token: str, action: str) -> dict:
    return {"kind": "path", "entity": _normalize_path(token), "action": action}


def symbols_from_pattern(pattern: str) -> list[str]:
    """Identifier-shaped tokens in a grep/search pattern, stopwords removed."""
    text = pattern.replace("\\", " ")
    symbols: list[str] = []
    seen: set[str] = set()
    for token in _NON_IDENTIFIER_SPLIT.split(text):
        if not _IDENTIFIER_PATTERN.match(token):
            continue
        if token.lower() in _SYMBOL_STOPWORDS:
            continue
        if token not in seen:
            seen.add(token)
            symbols.append(token)
    return symbols


def _references_from_path_command(argv: list[str], action: str) -> list[dict]:
    name = _path_basename(argv[0])
    refs: list[dict] = []
    skip_next = False
    for token in argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if name in {"head", "tail"} and token in _HEAD_TAIL_VALUE_FLAGS:
            skip_next = True
            continue
        if _is_path_target(token):
            refs.append(_path_reference(token, action))
    return refs


def _references_from_find(argv: list[str], action: str) -> list[dict]:
    refs: list[dict] = []
    index = 1
    while index < len(argv) and not argv[index].startswith(("-", "(", "!")):
        if _is_path_target(argv[index]):
            refs.append(_path_reference(argv[index], action))
        index += 1
    while index < len(argv):
        token = argv[index]
        if token in _FIND_NAME_FLAGS and index + 1 < len(argv):
            value = argv[index + 1]
            if _is_path_target(value):
                refs.append(_path_reference(value, action))
            index += 2
            continue
        index += 1
    return refs


def _references_from_sed(argv: list[str], action: str) -> list[dict]:
    refs: list[dict] = []
    script_seen = False
    skip_next = False
    for token in argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if token in {"-e", "--expression", "-f", "--file"}:
            skip_next = True
            script_seen = True
            continue
        if token.startswith("-"):
            continue
        if not script_seen:
            script_seen = True
            continue
        if _is_path_target(token):
            refs.append(_path_reference(token, action))
    return refs


def _references_from_grep(argv: list[str], action: str) -> list[dict]:
    refs: list[dict] = []
    patterns: list[str] = []
    positional_pattern_seen = False
    skip_value: str | None = None
    for token in argv[1:]:
        if skip_value is not None:
            if skip_value == "pattern":
                patterns.append(token)
            skip_value = None
            continue
        if token in _GREP_PATTERN_FLAGS:
            skip_value = "pattern"
            positional_pattern_seen = True
            continue
        if token in _GREP_VALUE_FLAGS:
            skip_value = "value"
            continue
        if token.startswith("-"):
            continue
        if not positional_pattern_seen:
            positional_pattern_seen = True
            patterns.append(token)
            continue
        if _is_path_target(token):
            refs.append(_path_reference(token, action))
    for pattern in patterns:
        for symbol in symbols_from_pattern(pattern):
            refs.append({"kind": "symbol", "entity": symbol, "action": action})
    return refs


def _references_from_bash(cmd: str) -> list[dict]:
    refs: list[dict] = []
    for raw_argv in _simple_commands(cmd):
        argv = _strip_command_prefix(_strip_redirections(raw_argv))
        if not argv:
            continue
        action = " ".join(raw_argv)
        name = _path_basename(argv[0])
        if name in _READ_PATH_COMMANDS:
            refs.extend(_references_from_path_command(argv, action))
        elif name == "find":
            refs.extend(_references_from_find(argv, action))
        elif name == "sed":
            refs.extend(_references_from_sed(argv, action))
        elif name in _GREP_COMMANDS:
            refs.extend(_references_from_grep(argv, action))
    return refs


def extract_entity_references(tool_calls: list[dict]) -> list[dict]:
    """Parse entity references from action arguments (never from reasoning).

    Returns dicts with ``kind`` (``path`` or ``symbol``), ``entity`` (the
    normalized literal), and ``action`` (the raw command snippet).
    """
    refs: list[dict] = []
    for tool_call in tool_calls or []:
        name, args = _tool_call_function(tool_call)
        command = _command_arg(args)
        if name in {"str_replace_editor", "file_editor"} and command == "view":
            path = args.get("path") or args.get("file")
            if isinstance(path, str) and path.strip():
                path = path.strip()
                refs.append(_path_reference(path, f"{name} view {path}"))
        elif name == "execute_bash" and command:
            refs.extend(_references_from_bash(command))
    return refs


def _bash_commands_from_text(text: str) -> list[str]:
    commands: list[str] = []
    for pattern in (_BASH_CMD_XML_PATTERN, _BASH_CMD_XML_EQ_PATTERN):
        for match in pattern.finditer(text):
            commands.append(match.group(1).strip())
    for match in _BASH_CMD_JSON_PATTERN.finditer(text):
        try:
            payload = json.loads(match.group(1))
        except (json.JSONDecodeError, ValueError):
            continue
        command = payload.get("command", "")
        if isinstance(command, str) and command.strip():
            commands.append(command.strip())
    return commands


def extract_entity_references_from_text(text: str) -> list[dict]:
    """Parse entity references from text-format tool calls (XML/JSON/inline).

    Fallback channel for artifacts whose steps carry the tool calls only in
    the response text; the shared core parsers pull out editor view paths and
    bash command strings. Text with no tool-call markers is treated as one
    inline bash command, matching the core read-scoring fallback.
    """
    if not text:
        return []
    refs: list[dict] = []
    for view in _parse_file_editor_views(text) + _parse_sweagent_views(text):
        path = view.get("file_path")
        if isinstance(path, str) and path:
            refs.append(_path_reference(path, f"view {path}"))
    commands = _bash_commands_from_text(text)
    if not commands and "<function=" not in text:
        commands = [text]
    for command in commands:
        refs.extend(_references_from_bash(command))
    return refs


def _bash_created_paths(cmd: str) -> set[str]:
    """Paths written via shell redirection or tee (heredocs write through them)."""
    created: set[str] = set()

    def add(target: str) -> None:
        if not target or target.startswith(("&", "$", "/dev/")):
            return
        if any(ch in _GLOB_CHARS for ch in target):
            return
        created.add(_normalize_path(target))

    for argv in _simple_commands(cmd):
        index = 0
        while index < len(argv):
            token = argv[index]
            if token in {">", ">>"} or re.fullmatch(r"\d>{1,2}", token):
                if index + 1 < len(argv):
                    add(argv[index + 1])
                    index += 2
                    continue
            else:
                match = _EMBEDDED_REDIRECT_PATTERN.match(token)
                if match:
                    add(match.group(1))
                elif _path_basename(token) == "tee":
                    for arg in argv[index + 1 :]:
                        if not arg.startswith("-"):
                            add(arg)
                    index = len(argv)
                    continue
            index += 1
    return created


def _created_paths(tool_calls: list[dict]) -> set[str]:
    """Files the model itself created: editor create targets + shell writes."""
    created: set[str] = set()
    for tool_call in tool_calls or []:
        name, args = _tool_call_function(tool_call)
        if name == "execute_bash":
            command = _command_arg(args)
            if command:
                created.update(_bash_created_paths(command))
    for write in parse_write_actions_from_tool_calls(tool_calls or []):
        if write.get("command") in {"create", "redirect", "tee"}:
            created.add(write["file_path"])
    return created


def _created_paths_from_text(text: str) -> set[str]:
    """Model-created files parsed from text-format tool calls.

    Redirect/tee targets are read only from the extracted bash command
    strings, never from the surrounding markup: raw XML like
    ``<parameter=path>/x.py`` would otherwise false-match as a redirect.
    """
    if not text:
        return set()
    created: set[str] = set()
    for write in _parse_editor_write_actions(text):
        if write.get("command") == "create":
            created.add(write["file_path"])
    commands = _bash_commands_from_text(text)
    if not commands and "<function=" not in text:
        commands = [text]
    for command in commands:
        created.update(_bash_created_paths(command))
    return created


@lru_cache(maxsize=4096)
def _whole_word_pattern(symbol: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])")


def _reference_is_licensed(ref: dict, licensed_chunks: list[str]) -> bool:
    entity = ref["entity"]
    if ref["kind"] == "path":
        needles = {entity, _path_basename(entity)}
        needles.discard("")
        return any(needle in chunk for chunk in licensed_chunks for needle in needles)
    pattern = _whole_word_pattern(entity)
    return any(pattern.search(chunk) for chunk in licensed_chunks)


def license_references(
    step_items: list[Any],
    initial_context_text: str,
    *,
    step_indices: list[int] | None = None,
) -> dict:
    """Classify the first reference to each entity as licensed or unlicensed.

    Single pass over the step traces. The licensed text starts as the initial
    context (task prompt including the issue) and accumulates each completed
    step's observations, so a reference can only be licensed by strictly
    earlier steps. References are deduped per trace by (kind, normalized
    entity); references to model-created files are exempt.
    """
    summary: dict[str, Any] = {
        "license_evaluable": bool(step_items),
        "n_entity_references": 0,
        "n_path_references": 0,
        "n_symbol_references": 0,
        "n_unlicensed_references": 0,
        "unlicensed_reference_rate": None,
        "unlicensed_references": [],
    }
    if not step_items:
        return summary

    licensed_chunks: list[str] = [initial_context_text] if initial_context_text else []
    created_paths: set[str] = set()
    created_basenames: set[str] = set()
    seen: set[tuple[str, str]] = set()
    for idx, trace in enumerate(step_items):
        tool_calls = _step_tool_calls(trace)
        # Structured tool calls are the preferred channel; steps without them
        # fall back to text-format tool calls so a step is never counted twice.
        if tool_calls:
            step_created = _created_paths(tool_calls)
            step_refs = extract_entity_references(tool_calls)
        else:
            text = _step_text(trace)
            step_created = _created_paths_from_text(text)
            step_refs = extract_entity_references_from_text(text)
        # A step's tool calls are batched: files created in this step exempt
        # this step's own references to them.
        for created in step_created:
            created_paths.add(created)
            basename = _path_basename(created)
            if basename:
                created_basenames.add(basename)
        for ref in step_refs:
            key = (ref["kind"], ref["entity"])
            if not ref["entity"] or key in seen:
                continue
            seen.add(key)
            if ref["kind"] == "path" and (
                ref["entity"] in created_paths or _path_basename(ref["entity"]) in created_basenames
            ):
                continue
            summary["n_entity_references"] += 1
            if ref["kind"] == "path":
                summary["n_path_references"] += 1
            else:
                summary["n_symbol_references"] += 1
            if not _reference_is_licensed(ref, licensed_chunks):
                summary["n_unlicensed_references"] += 1
                if len(summary["unlicensed_references"]) < 20:
                    step_label = (
                        step_indices[idx] if step_indices and idx < len(step_indices) else idx + 1
                    )
                    summary["unlicensed_references"].append(
                        {
                            "entity": ref["entity"],
                            "kind": ref["kind"],
                            "step": step_label,
                            "action": ref["action"][:120],
                        }
                    )
        observation = _step_observation_text(trace)
        if observation:
            licensed_chunks.append(observation)

    if summary["n_entity_references"]:
        summary["unlicensed_reference_rate"] = (
            summary["n_unlicensed_references"] / summary["n_entity_references"]
        )
    return summary
