"""Score eval rollouts against P2A bonus maps.

This is an offline diagnostic.  The eval-set bonus maps are not used to reshape
training advantages; they provide a reference graph for measuring whether a
model's eval rollout reads files/functions near the eventual fault.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from p2a.core import (
    BonusMapStore,
    compute_p2a_multiplier,
    is_rewardable_graph_node,
    match_reads_to_graph,
    normalize_action,
    parse_read_actions,
    parse_read_actions_from_tool_calls,
    reads_from_step_trace,
    segment_purpose_blocks,
    writes_from_step_trace,
)
from p2a.bonus_map_scope import (
    LATENT_CASE,
    PATH_CASE_TYPES,
    bonus_map_pattern_computable,
    bonus_map_root_anchor_overlap,
    canonical_bonus_case_type,
    canonical_detail_case_type,
    traceable_case_family,
)
from p2a.provenance import license_references

SUMMARY_SCHEMA_VERSION = "p2a_eval_fault_localization_v1"
TEXT_FIELDS = (
    "response_text",
    "assistant_response",
    "completion",
    "response",
    "output",
    "text",
)
STEP_FIELDS = ("global_step", "trainer_step", "validation_step", "training_step", "step", "iteration")
EXECUTION_ERROR_LINE_PATTERN = re.compile(
    r"^\s*(traceback\b|error:|exception:|command failed\b|failed:|failure:|no such file\b|bash:|sh:)",
    re.IGNORECASE,
)
EXECUTION_ERROR_TEXT_PATTERN = re.compile(
    r"\b(exit status|exit code|returned non-zero|command not found|segmentation fault)\b",
    re.IGNORECASE,
)
# Persisted eval artifacts historically used `chain_*` keys for what the
# dashboard now calls Path. New code should prefer the `path_*` aliases; the
# legacy keys remain readable/writable for old DB rows and detail JSON files.
LEGACY_NODE_ROLE_ALIASES = {
    "pre_symptom": "test_adapter",
}
PATH_NODE_ROLES = {"symptom", "intermediate", "fix_adapter", "root_cause"}
PATH_CONTEXT_ROLES = {"test_harness", "test_adapter", "pre_symptom"}
PATH_PATTERN_KEYS = (
    "missed_anchor",
    "missed_root_after_anchor",
    "root_before_anchor",
    "path_stall",
    "path_read_loop",
    "off_path_read_spree",
    "error_spiral_on_path",
)
LEGACY_PATH_PATTERN_KEYS = (
    "missed_anchor",
    "missed_root_after_anchor",
    "root_before_anchor",
    "chain_stall",
    "chain_read_loop",
    "off_chain_read_spree",
    "error_spiral_on_chain",
)
CHAIN_CASE_TYPES = PATH_CASE_TYPES
CHAIN_NODE_ROLES = PATH_NODE_ROLES
CHAIN_CONTEXT_ROLES = PATH_CONTEXT_ROLES
CHAIN_BAD_PATTERN_KEYS = LEGACY_PATH_PATTERN_KEYS
LEGACY_PATH_PATTERN_ALIASES = {
    "chain_stall": "path_stall",
    "chain_read_loop": "path_read_loop",
    "off_chain_read_spree": "off_path_read_spree",
    "error_spiral_on_chain": "error_spiral_on_path",
}

PATH_DETAIL_ALIASES = {
    "chain_evaluable": "path_evaluable",
    "not_chain_evaluable_reason": "not_path_evaluable_reason",
    "chain_case_kind": "path_case_kind",
    "chain_graph_covered": "path_covered",
    "chain_projection": "path_projection",
    "chain_hit": "path_hit",
    "chain_node_recall": "path_node_recall",
    "chain_read_precision": "path_read_precision",
    "n_chain_nodes": "n_path_nodes",
    "n_hit_chain_nodes": "n_hit_path_nodes",
    "chain_bad_patterns": "path_pattern_flags",
}
LEGACY_DETAIL_DEFAULTS = {
    "chain_evaluable": False,
    "not_chain_evaluable_reason": "missing_bonus_map",
    "chain_case_kind": None,
    "chain_graph_covered": False,
    "chain_projection": None,
    "chain_hit": False,
    "chain_node_recall": None,
    "chain_read_precision": None,
    "n_chain_nodes": 0,
    "n_hit_chain_nodes": 0,
    "chain_bad_patterns": {},
}


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


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)


def _sync_path_aliases(detail: dict[str, Any]) -> dict[str, Any]:
    """Mirror legacy `chain_*` detail keys to current Path terminology."""
    for legacy_key, path_key in PATH_DETAIL_ALIASES.items():
        legacy_is_default = (
            legacy_key in detail
            and path_key in detail
            and detail.get(legacy_key) == LEGACY_DETAIL_DEFAULTS.get(legacy_key, object())
            and detail.get(path_key) != detail.get(legacy_key)
        )
        if path_key in detail and (legacy_key not in detail or legacy_is_default):
            detail[legacy_key] = detail[path_key]
        elif legacy_key in detail and path_key not in detail:
            detail[path_key] = detail[legacy_key]
    projection = detail.get("path_projection")
    if isinstance(projection, dict):
        projection.setdefault("path_nodes", projection.get("chain_nodes") or [])
        projection.setdefault("path_edges", projection.get("chain_edges") or [])
        projection.setdefault("graph_context_nodes", projection.get("context_nodes") or [])
        projection.setdefault("graph_context_edges", projection.get("context_edges") or [])
        projection.setdefault("chain_nodes", projection.get("path_nodes") or [])
        projection.setdefault("chain_edges", projection.get("path_edges") or [])
        projection.setdefault("context_nodes", projection.get("graph_context_nodes") or [])
        projection.setdefault("context_edges", projection.get("graph_context_edges") or [])
    for key in ("path_pattern_flags", "chain_bad_patterns"):
        patterns = detail.get(key)
        if isinstance(patterns, dict):
            _sync_path_pattern_aliases(patterns)
    return detail


def _sync_path_pattern_aliases(patterns: dict[str, Any]) -> dict[str, Any]:
    for legacy_key, path_key in LEGACY_PATH_PATTERN_ALIASES.items():
        if path_key in patterns and legacy_key not in patterns:
            patterns[legacy_key] = patterns[path_key]
        elif legacy_key in patterns and path_key not in patterns:
            patterns[path_key] = patterns[legacy_key]
    return patterns


def _records_from_json_payload(payload: Any) -> Iterable[dict]:
    payload = _maybe_json(payload)
    if isinstance(payload, list):
        for item in payload:
            item = _maybe_json(item)
            if isinstance(item, dict):
                yield item
        return
    if isinstance(payload, dict):
        for key in ("records", "rollouts", "data", "items", "samples"):
            nested = _maybe_json(payload.get(key))
            if isinstance(nested, list):
                yield from _records_from_json_payload(nested)
                return
        yield payload


def iter_records(path: Path) -> Iterable[dict]:
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file() and child.suffix.lower() in {".jsonl", ".json", ".parquet"}:
                yield from iter_records(child)
        return

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pandas as pd

        df = pd.read_parquet(path)
        for record in df.to_dict(orient="records"):
            yield record
        return

    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSONL record: {exc}") from exc
                yield from _records_from_json_payload(payload)
        return

    with path.open(encoding="utf-8") as f:
        yield from _records_from_json_payload(json.load(f))


def _get_nested(mapping: dict, *path: str) -> Any:
    value: Any = mapping
    for key in path:
        value = _maybe_json(value)
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return _maybe_json(value)


def _as_str(value: Any) -> str | None:
    value = _maybe_json(value)
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return None


def extract_instance_id(record: dict) -> str | None:
    direct_keys = (
        "instance_id",
        "uid",
        "task_id",
        "sample_id",
        "id",
    )
    for key in direct_keys:
        value = _as_str(record.get(key))
        if value and ("__" in value or key == "instance_id"):
            return value

    nested_paths = (
        ("metadata", "instance_id"),
        ("extra_fields", "instance_id"),
        ("extra_info", "instance_id"),
        ("extra_info", "tools_kwargs", "reward", "metadata", "instance_id"),
        ("tools_kwargs", "reward", "metadata", "instance_id"),
        ("reward", "metadata", "instance_id"),
    )
    for path in nested_paths:
        value = _as_str(_get_nested(record, *path))
        if value:
            return value
    return None


def extract_data_source(record: dict) -> str:
    for container in _candidate_containers(record):
        value = _as_str(container.get("data_source"))
        if value:
            return value
    return "unknown"


def extract_run_step(record: dict) -> int | None:
    for container in _candidate_containers(record):
        for key in STEP_FIELDS:
            value = _maybe_json(container.get(key))
            if isinstance(value, bool) or value is None:
                continue
            if isinstance(value, int):
                return value
            if isinstance(value, float) and math.isfinite(value):
                return int(value)
            if isinstance(value, str):
                try:
                    return int(float(value.strip()))
                except ValueError:
                    continue
    return None


def extract_rollout_index(record: dict) -> int:
    for container in _candidate_containers(record):
        value = _maybe_json(container.get("rollout_index"))
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float) and math.isfinite(value):
            return int(value)
        if isinstance(value, str):
            try:
                return int(float(value.strip()))
            except ValueError:
                continue
    return 0


def extract_rollout_id(record: dict) -> str | None:
    for container in _candidate_containers(record):
        value = _as_str(container.get("rollout_id"))
        if value:
            return value
    return None


def _normalize_tool_call(tool_call: Any) -> dict | None:
    tool_call = _maybe_json(tool_call)
    if not isinstance(tool_call, dict):
        return None
    if isinstance(tool_call.get("function"), dict):
        return tool_call
    if "name" in tool_call and ("arguments" in tool_call or "args" in tool_call):
        return {
            "function": {
                "name": tool_call.get("name"),
                "arguments": tool_call.get("arguments", tool_call.get("args", {})),
            }
        }
    if "tool_name" in tool_call:
        return {
            "function": {
                "name": tool_call.get("tool_name"),
                "arguments": tool_call.get("arguments", tool_call.get("args", {})),
            }
        }
    return None


def _normalize_tool_calls(value: Any) -> list[dict]:
    value = _maybe_json(value)
    if value is None:
        return []
    if isinstance(value, dict):
        nested = value.get("tool_calls")
        if nested is not None:
            return _normalize_tool_calls(nested)
        normalized = _normalize_tool_call(value)
        return [normalized] if normalized else []
    if isinstance(value, list):
        calls = []
        for item in value:
            normalized = _normalize_tool_call(item)
            if normalized:
                calls.append(normalized)
        return calls
    return []


def _content_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return ""
    value = _maybe_json(value)
    if isinstance(value, list):
        parts = []
        for item in value:
            item = _maybe_json(item)
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return ""


def _reads_from_tool_calls(tool_calls: Any) -> list[dict]:
    normalized = _normalize_tool_calls(tool_calls)
    if not normalized:
        return []
    return parse_read_actions_from_tool_calls(normalized)


def _dedupe_reads(reads: Iterable[dict]) -> list[dict]:
    deduped = []
    seen = set()
    for read in reads:
        if not isinstance(read, dict):
            continue
        file_path = read.get("file_path")
        if not file_path:
            continue
        start = int(read.get("start_line", 1))
        end = int(read.get("end_line", 999999))
        key = (file_path, start, end)
        if key in seen:
            continue
        seen.add(key)
        deduped.append({"file_path": file_path, "start_line": start, "end_line": end})
    return deduped


def _extract_reads_from_text(value: Any, tracking_mode: str) -> list[dict]:
    text = _content_to_text(value)
    if not text:
        return []
    return parse_read_actions(text, tracking_mode=tracking_mode)


def _messages_reads(messages: Any, tracking_mode: str) -> list[dict]:
    messages = _maybe_json(messages)
    if not isinstance(messages, list):
        return []

    reads = []
    for message in messages:
        message = _maybe_json(message)
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).lower()
        is_assistant = role in {"assistant", "agent", "model"} or "tool_calls" in message
        if not is_assistant:
            continue
        reads.extend(_reads_from_tool_calls(message.get("tool_calls")))
        reads.extend(_extract_reads_from_text(message.get("content"), tracking_mode))
    return reads


def _candidate_containers(record: dict) -> list[dict]:
    containers = [record]
    for key in ("extra_fields", "extra_info", "metadata"):
        value = _maybe_json(record.get(key))
        if isinstance(value, dict):
            containers.append(value)
    return containers


def _step_trace_items(record: dict) -> list[Any]:
    for container in _candidate_containers(record):
        value = _maybe_json(container.get("p2a_step_traces"))
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            for key in ("p2a_step_traces", "steps", "traces"):
                nested = _maybe_json(value.get(key))
                if isinstance(nested, list):
                    return nested
    return []


def _reads_from_step(trace: Any, tracking_mode: str) -> list[dict]:
    trace = _maybe_json(trace)
    reads = []
    if isinstance(trace, dict):
        reads.extend(_reads_from_tool_calls(trace.get("tool_calls")))
        for field in TEXT_FIELDS:
            reads.extend(_extract_reads_from_text(trace.get(field), tracking_mode))
        reads.extend(_messages_reads(trace.get("messages"), tracking_mode))
    elif isinstance(trace, list):
        reads.extend(_reads_from_tool_calls(trace))
        for item in trace:
            reads.extend(_extract_reads_from_text(item, tracking_mode))
    elif isinstance(trace, str):
        reads.extend(_extract_reads_from_text(trace, tracking_mode))
    return _dedupe_reads(reads)


def extract_record_reads(
    record: dict,
    tracking_mode: str,
    step_items: list[Any] | None = None,
) -> tuple[list[list[dict]], list[dict]]:
    if step_items is None:
        step_items = _step_trace_items(record)
    step_reads = [_reads_from_step(trace, tracking_mode) for trace in step_items]

    reads = []
    for container in _candidate_containers(record):
        reads.extend(_reads_from_tool_calls(container.get("tool_calls")))
        reads.extend(_messages_reads(container.get("messages"), tracking_mode))
        for field in TEXT_FIELDS:
            reads.extend(_extract_reads_from_text(container.get(field), tracking_mode))

    all_reads = _dedupe_reads([read for reads_for_step in step_reads for read in reads_for_step] + reads)
    return step_reads, all_reads


def _first_matching_step(step_reads: list[list[dict]], bonus_map: dict, *, require_gt: bool) -> int | None:
    return _first_matching_step_with_indices(step_reads, bonus_map, require_gt=require_gt)


def _first_matching_step_with_indices(
    step_reads: list[list[dict]],
    bonus_map: dict,
    *,
    require_gt: bool,
    step_indices: list[int] | None = None,
) -> int | None:
    for idx, reads in enumerate(step_reads):
        distance = match_reads_to_graph(reads, bonus_map)
        if distance < 0:
            continue
        if require_gt and distance != 0.0:
            continue
        if step_indices and idx < len(step_indices):
            return step_indices[idx]
        return idx
    return None


def _read_hit_nodes(read: dict, bonus_map: dict, *, rewardable_only: bool = True) -> set[str]:
    nodes = bonus_map.get("call_graph_nodes", {}) if bonus_map else {}
    hits = set()
    for node_key, node in nodes.items():
        if rewardable_only and not is_rewardable_graph_node(node):
            continue
        if read["file_path"] != node["file_path"]:
            continue
        if read["start_line"] <= node["end_line"] and read["end_line"] >= node["start_line"]:
            hits.add(node_key)
    return hits


def _write_hit_nodes(write: dict, bonus_map: dict, *, rewardable_only: bool = True) -> set[str]:
    return _read_hit_nodes(write, bonus_map, rewardable_only=rewardable_only)


def _step_node_first_hits(
    step_reads: list[list[dict]],
    bonus_map: dict,
    *,
    rewardable_only: bool = True,
    step_indices: list[int] | None = None,
) -> dict[str, int]:
    first_hits: dict[str, int] = {}
    for step_idx, reads in enumerate(step_reads):
        step_label = step_indices[step_idx] if step_indices and step_idx < len(step_indices) else step_idx
        for read in reads:
            for node_key in _read_hit_nodes(read, bonus_map, rewardable_only=rewardable_only):
                first_hits.setdefault(node_key, step_label)
    return first_hits


def _trace_step_indices(step_items: list[Any]) -> list[int]:
    explicit_values: list[int] = []
    for trace in step_items:
        normalized = _normalized_trace(trace)
        value = normalized.get("step_idx", normalized.get("step_index"))
        if isinstance(value, int | float) and not isinstance(value, bool):
            explicit_values.append(int(value))
    offset = 1 if explicit_values and min(explicit_values) == 0 else 0
    indices: list[int] = []
    for index, trace in enumerate(step_items):
        normalized = _normalized_trace(trace)
        value = normalized.get("step_idx", normalized.get("step_index"))
        if isinstance(value, int | float) and not isinstance(value, bool):
            indices.append(int(value) + offset)
        else:
            indices.append(index + 1)
    return indices


def _all_read_hit_nodes(reads: list[dict], bonus_map: dict, *, rewardable_only: bool = True) -> set[str]:
    hits = set()
    for read in reads:
        hits.update(_read_hit_nodes(read, bonus_map, rewardable_only=rewardable_only))
    return hits


def _record_issue_text(record: dict) -> str:
    for container in _candidate_containers(record):
        for key in ("problem_statement", "issue_description", "issue_text", "issue", "description", "problem"):
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return ""


def _leading_prompt_texts(messages: Any) -> list[str]:
    """System-prompt and first-user-message texts from a chat message list."""
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    messages = _maybe_json(messages)
    if not isinstance(messages, list):
        return []
    parts: list[str] = []
    for message in messages:
        message = _maybe_json(message)
        if not isinstance(message, dict):
            break
        role = str(message.get("role", "")).lower()
        if role in {"system", "developer"}:
            text = _content_to_text(message.get("content"))
            if text:
                parts.append(text)
            continue
        if role == "user":
            text = _content_to_text(message.get("content"))
            if text:
                parts.append(text)
        break
    return parts


# Keys that carry the task prompt in scored records: dashboard DB records
# store the conversation under `messages`; live-validation records copy the
# dataset's `raw_prompt`/`prompt` chat messages from the batch.
_PROMPT_KEYS = ("messages", "raw_prompt", "prompt")


def _initial_context_text(record: dict) -> str:
    """Text visible to the model before its first step: task prompt + issue.

    The first prompt-bearing key wins: the system prompt and first user
    message form the task prompt (a plain-string prompt is used verbatim).
    The stored issue description is appended so records without any prompt
    payload still license from the issue.
    """
    parts: list[str] = []
    for key in _PROMPT_KEYS:
        for container in _candidate_containers(record):
            value = container.get(key)
            if value is None:
                continue
            if isinstance(value, str) and value.strip() and value.strip()[0] not in "[{":
                parts.append(value)
            else:
                parts.extend(_leading_prompt_texts(value))
            if parts:
                break
        if parts:
            break
    issue = _record_issue_text(record)
    if issue:
        parts.append(issue)
    return "\n\n".join(parts)


def _kendall_order(first_hits: dict[str, int], bonus_map: dict) -> tuple[float | None, bool]:
    nodes = bonus_map.get("call_graph_nodes", {})
    ordered = [
        (step, 1.0 - float(nodes[node_key]["normalized_distance"]))
        for node_key, step in first_hits.items()
        if node_key in nodes
    ]
    if len({value for _step, value in ordered}) < 2:
        return None, False

    concordant = 0
    discordant = 0
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            step_delta = ordered[i][0] - ordered[j][0]
            target_delta = ordered[i][1] - ordered[j][1]
            if step_delta == 0 or target_delta == 0:
                continue
            if step_delta * target_delta > 0:
                concordant += 1
            else:
                discordant += 1
    denom = concordant + discordant
    if denom == 0:
        return None, False
    tau = (concordant - discordant) / denom
    return tau, True


def _first_hit_for_nodes(first_hits: dict[str, int], node_keys: Iterable[str]) -> int | None:
    steps = [first_hits[node_key] for node_key in node_keys if node_key in first_hits]
    return min(steps) if steps else None


def _miracle_stats(
    first_hits: dict[str, int],
    bonus_map: dict,
    *,
    anchor_nodes: Iterable[str] | None = None,
    root_nodes: Iterable[str] | None = None,
) -> tuple[bool | None, int | None]:
    nodes = bonus_map.get("call_graph_nodes", {})
    intermediate = {
        node_key: node
        for node_key, node in nodes.items()
        if _node_role(node_key, nodes) in {"intermediate", "fix_adapter"}
        or 0.0 < float(node.get("normalized_distance", -1.0)) < 1.0
    }
    root_candidates = set(root_nodes or [])
    if not root_candidates:
        root_candidates = {
            node_key
            for node_key, node in nodes.items()
            if float(node.get("normalized_distance", -1.0)) == 0.0
        }
    gt_steps = [
        first_hits[node_key]
        for node_key in root_candidates
        if node_key in first_hits
    ]
    if not gt_steps:
        return False, 0
    first_gt = min(gt_steps)
    anchor_candidates = set(anchor_nodes or [])
    first_anchor = _first_hit_for_nodes(first_hits, anchor_candidates) if anchor_candidates else None
    if first_anchor is not None and first_anchor == first_gt:
        return False, 0
    if not intermediate:
        skipped_anchor = bool(anchor_candidates) and (first_anchor is None or first_gt < first_anchor)
        return (True, 1) if skipped_anchor else (None, None)
    visited_intermediate_levels = {
        float(intermediate[node_key]["normalized_distance"])
        for node_key, step in first_hits.items()
        if node_key in intermediate and step <= first_gt
    }
    all_intermediate_levels = {float(node["normalized_distance"]) for node in intermediate.values()}
    missing_levels = all_intermediate_levels - visited_intermediate_levels
    skipped_anchor = bool(anchor_candidates) and (first_anchor is None or first_gt < first_anchor)
    is_miracle = skipped_anchor or not visited_intermediate_levels
    return is_miracle, len(missing_levels) if is_miracle else 0


def _block_step_reads(record: dict, tracking_mode: str) -> list[list[dict]]:
    traces = [trace for trace in _step_trace_items(record) if isinstance(_maybe_json(trace), dict)]
    normalized_traces = [_maybe_json(trace) for trace in traces]
    blocks = segment_purpose_blocks(normalized_traces, tracking_mode=tracking_mode)
    block_reads = []
    for block in blocks:
        reads = []
        for idx in block["trace_indices"]:
            reads.extend(reads_from_step_trace(normalized_traces[idx], tracking_mode=tracking_mode))
        block_reads.append(_dedupe_reads(reads))
    return [reads for reads in block_reads if reads]


def _source_preview(node: dict, max_chars: int = 4000) -> str | None:
    source = node.get("source")
    if not isinstance(source, str) or not source:
        return None
    if len(source) <= max_chars:
        return source
    return source[:max_chars].rstrip() + "\n..."


def _source_text(node: dict) -> str | None:
    source = node.get("source")
    return source if isinstance(source, str) and source else None


def _node_role(node_key: str, nodes: dict[str, dict]) -> str | None:
    role = nodes.get(node_key, {}).get("node_role")
    if role is None:
        return None
    return LEGACY_NODE_ROLE_ALIASES.get(str(role), str(role))


def _string_list(value: Any) -> list[str]:
    value = _maybe_json(value)
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


def _edge_pairs(value: Any) -> list[tuple[str, str]]:
    value = _maybe_json(value)
    if not isinstance(value, list):
        return []
    pairs = []
    for edge in value:
        edge = _maybe_json(edge)
        if isinstance(edge, dict):
            caller = edge.get("caller")
            callee = edge.get("callee")
            if isinstance(caller, str) and isinstance(callee, str):
                pairs.append((caller, callee))
            continue
        if isinstance(edge, (list, tuple)) and len(edge) == 2 and all(isinstance(part, str) for part in edge):
            pairs.append((str(edge[0]), str(edge[1])))
    return pairs


def _edge_dict(caller: str, callee: str, edge_type: str, nodes: dict[str, dict]) -> dict[str, Any]:
    caller_role = _node_role(caller, nodes)
    callee_role = _node_role(callee, nodes)
    return {
        "caller": caller,
        "callee": callee,
        "source": caller,
        "target": callee,
        "edge_type": edge_type,
        "caller_role": caller_role,
        "callee_role": callee_role,
        "role_transition": f"{caller_role}->{callee_role}",
    }


def _path_edges(bonus_map: dict) -> list[tuple[str, str]]:
    explicit = _edge_pairs(bonus_map.get("reward_path_edges"))
    if explicit:
        return explicit
    metadata = _maybe_json(bonus_map.get("call_graph_edge_metadata"))
    if not isinstance(metadata, list):
        return []
    return [
        (str(item["caller"]), str(item["callee"]))
        for item in metadata
        if isinstance(item, dict)
        and item.get("reward_path_edge")
        and isinstance(item.get("caller"), str)
        and isinstance(item.get("callee"), str)
    ]


def _graph_edges(bonus_map: dict) -> list[tuple[str, str]]:
    explicit = _edge_pairs(bonus_map.get("call_graph_edges"))
    if explicit:
        return explicit
    metadata = _maybe_json(bonus_map.get("call_graph_edge_metadata"))
    if not isinstance(metadata, list):
        return []
    return [
        (str(item["caller"]), str(item["callee"]))
        for item in metadata
        if isinstance(item, dict)
        and isinstance(item.get("caller"), str)
        and isinstance(item.get("callee"), str)
    ]


def _call_graph_edges(bonus_map: dict) -> list[tuple[str, str]]:
    return _graph_edges(bonus_map)


def _reachable_from(starts: set[str], edges: list[tuple[str, str]]) -> set[str]:
    outgoing: dict[str, set[str]] = defaultdict(set)
    for caller, callee in edges:
        outgoing[caller].add(callee)
    seen = set(starts)
    frontier = list(starts)
    while frontier:
        current = frontier.pop()
        for nxt in outgoing.get(current, set()):
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return seen


def _can_reach(starts: set[str], edges: list[tuple[str, str]]) -> set[str]:
    reverse: dict[str, set[str]] = defaultdict(set)
    for caller, callee in edges:
        reverse[callee].add(caller)
    seen = set(starts)
    frontier = list(starts)
    while frontier:
        current = frontier.pop()
        for prev in reverse.get(current, set()):
            if prev not in seen:
                seen.add(prev)
                frontier.append(prev)
    return seen


def _path_evaluability(bonus_map: dict | None) -> tuple[bool, str | None]:
    if not bonus_map:
        return False, "missing_bonus_map"
    case_type = canonical_bonus_case_type(bonus_map)
    anchors = _string_list(bonus_map.get("selected_issue_anchor_nodes"))
    if case_type not in PATH_CASE_TYPES:
        return False, str(bonus_map.get("reason_code") or case_type or "missing_case_type")
    if not anchors:
        return False, "traceable_no_anchor"
    return True, None


def _path_projection(
    bonus_map: dict,
    hit_nodes: set[str],
    first_hits: dict[str, int],
) -> dict[str, Any]:
    nodes = bonus_map.get("call_graph_nodes", {}) if bonus_map else {}
    if not isinstance(nodes, dict):
        nodes = {}
    anchors = {key for key in _string_list(bonus_map.get("selected_issue_anchor_nodes")) if key in nodes}
    roots = {key for key in _string_list(bonus_map.get("root_cause_nodes")) if key in nodes}
    if not roots:
        roots = {key for key, node in nodes.items() if node.get("node_role") == "root_cause"}

    path_edges = [(caller, callee) for caller, callee in _path_edges(bonus_map) if caller in nodes and callee in nodes]
    forward = _reachable_from(anchors, path_edges) if anchors else set()
    backward = _can_reach(roots, path_edges) if roots else set()
    selected_path_edges = [
        (caller, callee)
        for caller, callee in path_edges
        if caller in forward and callee in forward and caller in backward and callee in backward
    ]
    path_node_keys = set(anchors) | set(roots)
    for caller, callee in selected_path_edges:
        path_node_keys.add(caller)
        path_node_keys.add(callee)
    if not selected_path_edges:
        path_node_keys.update(
            key
            for key, node in nodes.items()
            if _node_role(key, nodes) in PATH_NODE_ROLES and (key in anchors or key in roots)
        )

    call_edges = [(caller, callee) for caller, callee in _graph_edges(bonus_map) if caller in nodes and callee in nodes]
    reverse_call_edges: dict[str, set[str]] = defaultdict(set)
    for caller, callee in call_edges:
        reverse_call_edges[callee].add(caller)
    context_node_keys: set[str] = set()
    frontier = list(path_node_keys)
    while frontier:
        current = frontier.pop()
        for prev in reverse_call_edges.get(current, set()):
            if prev in context_node_keys or _node_role(prev, nodes) not in PATH_CONTEXT_ROLES:
                continue
            context_node_keys.add(prev)
            frontier.append(prev)

    context_edge_keys = [
        (caller, callee)
        for caller, callee in call_edges
        if caller in context_node_keys and callee in (context_node_keys | path_node_keys)
    ]

    def node_summary(node_key: str, *, group: str) -> dict[str, Any]:
        node = nodes.get(node_key, {})
        role = _node_role(node_key, nodes)
        summary = {
            "key": node_key,
            "file_path": node.get("file_path"),
            "start_line": node.get("start_line"),
            "end_line": node.get("end_line"),
            "normalized_distance": node.get("normalized_distance"),
            "rewardable": node.get("rewardable"),
            "excluded_from_hop_max": node.get("excluded_from_hop_max"),
            "exclusion_reason": node.get("exclusion_reason"),
            "node_role": role,
            "group": group,
            "selected_issue_anchor": node_key in anchors,
            "root_cause": node_key in roots,
            "patched_callable": node.get("patched_callable"),
            "patch_role": node.get("patch_role"),
            "hit": node_key in hit_nodes,
            "first_step": first_hits.get(node_key),
        }
        if role in PATH_NODE_ROLES or node.get("rewardable"):
            summary["source"] = _source_text(node)
            summary["source_preview"] = _source_preview(node)
        return summary

    def node_sort_key(node_key: str) -> tuple[float, str, int, str]:
        node = nodes.get(node_key, {})
        return (
            float(node.get("normalized_distance", 1.0)),
            str(node.get("file_path", "")),
            int(node.get("start_line", 0)),
            node_key,
        )

    path_nodes = [node_summary(key, group="path") for key in sorted(path_node_keys, key=node_sort_key)]
    context_nodes = [node_summary(key, group="context") for key in sorted(context_node_keys, key=node_sort_key)]
    return {
        "anchors": sorted(anchors),
        "roots": sorted(roots),
        "path_nodes": path_nodes,
        "path_edges": [_edge_dict(caller, callee, "path", nodes) for caller, callee in selected_path_edges],
        "context_nodes": context_nodes,
        "context_edges": [_edge_dict(caller, callee, "context", nodes) for caller, callee in context_edge_keys],
        "chain_nodes": path_nodes,
        "chain_edges": [_edge_dict(caller, callee, "chain", nodes) for caller, callee in selected_path_edges],
    }


def _path_case_kind(item: dict[str, Any]) -> str | None:
    value = item.get("path_case_kind", item.get("chain_case_kind"))
    if value is None:
        value = item.get("bonus_case_type")
    if value is None:
        return None
    return canonical_detail_case_type(item)


def _is_path_metric_evaluable(item: dict[str, Any]) -> bool:
    case_kind = _path_case_kind(item) or item.get("bonus_case_type")
    return item.get("path_evaluable", item.get("chain_evaluable")) is True and case_kind in PATH_CASE_TYPES


def _is_order_metric_evaluable(item: dict[str, Any]) -> bool:
    if not _is_path_metric_evaluable(item):
        return False
    if canonical_detail_case_type(item) != LATENT_CASE:
        return False
    projection = item.get("path_projection") or item.get("chain_projection") or {}
    return bool(projection.get("path_edges") or projection.get("chain_edges") or [])


def _path_step_hits(
    step_reads: list[list[dict]],
    bonus_map: dict,
    path_nodes: set[str],
    anchor_nodes: set[str],
    root_nodes: set[str],
    *,
    step_indices: list[int] | None = None,
) -> dict[str, Any]:
    path_hit_steps: list[int] = []
    anchor_hit_steps: list[int] = []
    root_hit_steps: list[int] = []
    hit_path_nodes: set[str] = set()
    read_hits_path = 0
    read_hits_off_path = 0
    node_step_counts: Counter[str] = Counter()
    for step_idx, reads in enumerate(step_reads):
        step_label = step_indices[step_idx] if step_indices and step_idx < len(step_indices) else step_idx
        step_path_hits: set[str] = set()
        for read in reads:
            hits = _read_hit_nodes(read, bonus_map, rewardable_only=False)
            path_hits = hits & path_nodes
            if path_hits:
                read_hits_path += 1
                step_path_hits.update(path_hits)
                hit_path_nodes.update(path_hits)
            else:
                read_hits_off_path += 1
        if step_path_hits:
            path_hit_steps.append(step_label)
            for node_key in step_path_hits:
                node_step_counts[node_key] += 1
            if step_path_hits & anchor_nodes:
                anchor_hit_steps.append(step_label)
            if step_path_hits & root_nodes:
                root_hit_steps.append(step_label)
    return {
        "path_hit_steps": path_hit_steps,
        "anchor_hit_steps": anchor_hit_steps,
        "root_hit_steps": root_hit_steps,
        "hit_path_nodes": hit_path_nodes,
        "read_hits_path": read_hits_path,
        "read_hits_off_path": read_hits_off_path,
        "chain_hit_steps": path_hit_steps,
        "hit_chain_nodes": hit_path_nodes,
        "read_hits_chain": read_hits_path,
        "read_hits_off_chain": read_hits_off_path,
        "node_step_counts": node_step_counts,
    }


def _path_pattern_flags(
    *,
    path_evaluable: bool,
    path_case_kind: str | None,
    first_anchor_step: int | None,
    first_root_step: int | None,
    root_hit_steps: list[int],
    path_hit_steps: list[int],
    read_hits_path: int,
    read_hits_off_path: int,
    node_step_counts: Counter[str],
    step_items: list[Any],
) -> dict[str, Any]:
    if not path_evaluable:
        return {key: False for key in PATH_PATTERN_KEYS}
    root_after_anchor = (
        first_anchor_step is not None
        and any(step >= first_anchor_step for step in root_hit_steps)
    )
    root_before_anchor = (
        path_case_kind == LATENT_CASE
        and first_anchor_step is not None
        and first_root_step is not None
        and first_root_step < first_anchor_step
    )
    path_steps_after_anchor = [
        step
        for step in path_hit_steps
        if first_anchor_step is not None and step > first_anchor_step
    ]
    flags = {
        "missed_anchor": first_anchor_step is None,
        "missed_root_after_anchor": first_anchor_step is not None and not root_after_anchor,
        "root_before_anchor": root_before_anchor,
        "path_stall": first_anchor_step is not None and not root_after_anchor and len(path_steps_after_anchor) >= 2,
        "path_read_loop": any(count >= 3 for count in node_step_counts.values()),
        "off_path_read_spree": read_hits_off_path >= 3 and read_hits_off_path > read_hits_path,
        "error_spiral_on_path": _max_error_run_on_steps(step_items, set(path_hit_steps)) >= 3,
        "n_path_read_steps": len(path_hit_steps),
        "n_path_read_hits": read_hits_path,
        "n_off_path_reads": read_hits_off_path,
    }
    flags.update(
        {
            "n_chain_read_steps": flags["n_path_read_steps"],
            "n_chain_read_hits": flags["n_path_read_hits"],
            "n_off_chain_reads": flags["n_off_path_reads"],
        }
    )
    return _sync_path_pattern_aliases(flags)


def _graph_topology(bonus_map: dict, hit_nodes: set[str], first_hits: dict[str, int]) -> dict:
    nodes = bonus_map.get("call_graph_nodes", {}) if bonus_map else {}
    node_items = []
    for node_key, node in sorted(
        nodes.items(),
        key=lambda item: (
            float(item[1].get("normalized_distance", 1.0)),
            str(item[1].get("file_path", "")),
            int(item[1].get("start_line", 0)),
            item[0],
        ),
    ):
        node_items.append(
            {
                "key": node_key,
                "file_path": node.get("file_path"),
                "start_line": node.get("start_line"),
                "end_line": node.get("end_line"),
                "normalized_distance": node.get("normalized_distance"),
                "rewardable": node.get("rewardable", True),
                "node_role": _node_role(node_key, nodes),
                "excluded_from_hop_max": node.get("excluded_from_hop_max"),
                "exclusion_reason": node.get("exclusion_reason"),
                "patched_callable": node.get("patched_callable"),
                "patch_role": node.get("patch_role"),
                "hit": node_key in hit_nodes,
                "first_step": first_hits.get(node_key),
                "source": _source_text(node),
                "source_preview": _source_preview(node),
            }
        )
    edges = [[caller, callee] for caller, callee in _graph_edges(bonus_map)]
    return {"nodes": node_items, "edges": edges}


def _node_summaries(node_keys: Iterable[str], bonus_map: dict) -> list[dict]:
    nodes = bonus_map.get("call_graph_nodes", {}) if bonus_map else {}
    summaries = []
    for node_key in sorted(node_keys, key=lambda key: (float(nodes.get(key, {}).get("normalized_distance", 1.0)), key)):
        node = nodes.get(node_key, {})
        summaries.append(
            {
                "key": node_key,
                "file_path": node.get("file_path"),
                "start_line": node.get("start_line"),
                "end_line": node.get("end_line"),
                "normalized_distance": node.get("normalized_distance"),
                "rewardable": node.get("rewardable", True),
                "node_role": _node_role(node_key, nodes),
                "patched_callable": node.get("patched_callable"),
                "patch_role": node.get("patch_role"),
            }
        )
    return summaries


def _min_node_distance(node_keys: Iterable[str], bonus_map: dict) -> float | None:
    nodes = bonus_map.get("call_graph_nodes", {}) if bonus_map else {}
    distances = [
        float(nodes[node_key].get("normalized_distance", 1.0))
        for node_key in node_keys
        if node_key in nodes
    ]
    return min(distances) if distances else None


def _normalized_trace(trace: Any) -> dict:
    trace = _maybe_json(trace)
    return trace if isinstance(trace, dict) else {}


def _step_details(
    step_items: list[Any],
    step_reads: list[list[dict]],
    bonus_map: dict,
    tracking_mode: str,
    *,
    step_indices: list[int] | None = None,
) -> list[dict]:
    details = []
    root_nodes = set(bonus_map.get("root_cause_nodes") or [])
    for step_idx, trace in enumerate(step_items):
        trace = _normalized_trace(trace)
        reads = step_reads[step_idx] if step_idx < len(step_reads) else []
        writes = writes_from_step_trace(trace)
        hit_nodes = _all_read_hit_nodes(reads, bonus_map, rewardable_only=False)
        write_hit_nodes: set[str] = set()
        for write in writes:
            write_hit_nodes.update(_write_hit_nodes(write, bonus_map, rewardable_only=False))
        action = normalize_action(trace, tracking_mode=tracking_mode)
        details.append(
            {
                "step_index": step_indices[step_idx] if step_indices and step_idx < len(step_indices) else step_idx + 1,
                "trace_index": step_idx,
                "family": action.get("family"),
                "target_path": action.get("target_path"),
                "n_reads": len(reads),
                "reads": reads,
                "writes": writes,
                "hit_nodes": _node_summaries(hit_nodes, bonus_map),
                "write_hit_nodes": _node_summaries(write_hit_nodes, bonus_map),
                "edited_root_cause": bool(write_hit_nodes & root_nodes),
                "execution_error": _trace_has_error(trace),
                "min_distance": _min_node_distance(hit_nodes, bonus_map),
            }
        )
    return details


def _tool_signature(trace: dict, tracking_mode: str) -> str:
    action = normalize_action(trace, tracking_mode=tracking_mode)
    tool_calls = trace.get("tool_calls") or []
    name = ""
    args: Any = {}
    if isinstance(tool_calls, list) and tool_calls and isinstance(tool_calls[0], dict):
        func = tool_calls[0].get("function", {})
        if isinstance(func, dict):
            name = str(func.get("name", "") or "")
            args = _maybe_json(func.get("arguments", {}))
    if isinstance(args, dict):
        compact_args = {
            key: str(value)[:120]
            for key, value in args.items()
            if key in {"command", "path", "view_range", "old_str", "new_str"}
        }
    else:
        compact_args = {}
    return json.dumps(
        {
            "family": action.get("family"),
            "target_path": action.get("target_path"),
            "tool": name,
            "args": compact_args,
        },
        sort_keys=True,
    )


def _trace_has_error(trace: Any) -> bool:
    trace = _maybe_json(trace)
    if not trace:
        return False
    if isinstance(trace, dict):
        if trace.get("error") or trace.get("system_error"):
            return True
        for result_value in trace.get("tool_results") or []:
            result = _maybe_json(result_value)
            if not isinstance(result, dict):
                continue
            if result.get("error"):
                return True
            status = str(result.get("status") or result.get("state") or "").lower()
            if status in {"error", "failed", "failure"}:
                return True
            exit_code = result.get("exit_code", result.get("returncode"))
            if isinstance(exit_code, (int, float)) and int(exit_code) != 0:
                return True
            for key in ("stderr", "observation", "content", "result", "output"):
                value = result.get(key)
                if isinstance(value, str) and _looks_like_execution_error(value):
                    return True
        return False
    return isinstance(trace, str) and _looks_like_execution_error(trace)


def _looks_like_execution_error(text: str) -> bool:
    sample = str(text or "")[:6000]
    if EXECUTION_ERROR_TEXT_PATTERN.search(sample):
        return True
    return any(EXECUTION_ERROR_LINE_PATTERN.search(line) for line in sample.splitlines()[:80])


def _max_error_run(step_items: list[Any]) -> int:
    longest = 0
    current = 0
    for trace in step_items:
        if _trace_has_error(trace):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _max_error_run_on_steps(step_items: list[Any], step_indices: set[int]) -> int:
    longest = 0
    current = 0
    for idx, trace in enumerate(step_items):
        if idx in step_indices and _trace_has_error(trace):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _purpose_blocks(
    step_items: list[Any],
    step_reads: list[list[dict]],
    bonus_map: dict,
    tracking_mode: str,
    *,
    step_indices: list[int] | None = None,
) -> list[dict]:
    normalized_traces = [_normalized_trace(trace) for trace in step_items]
    display_step_indices = step_indices or _trace_step_indices(step_items)
    has_graph = bool((bonus_map or {}).get("call_graph_nodes"))
    blocks = []
    for block_index, block in enumerate(segment_purpose_blocks(normalized_traces, tracking_mode=tracking_mode)):
        block_step_indices = [
            display_step_indices[idx] if idx < len(display_step_indices) else idx + 1
            for idx in block["trace_indices"]
        ]
        reads = []
        hit_nodes: set[str] = set()
        hit_trace_offsets = []
        for offset, trace_idx in enumerate(block["trace_indices"]):
            reads_for_step = step_reads[trace_idx] if trace_idx < len(step_reads) else []
            reads.extend(reads_for_step)
            step_hits = _all_read_hit_nodes(reads_for_step, bonus_map)
            if step_hits:
                hit_trace_offsets.append(offset)
                hit_nodes.update(step_hits)

        signatures = [_tool_signature(normalized_traces[idx], tracking_mode) for idx in block["trace_indices"]]
        is_loop = len(signatures) >= 3 and len(set(signatures)) == 1
        distance = match_reads_to_graph(reads, bonus_map)
        outcome_defined = has_graph and block["family"] == "read"
        achieved = outcome_defined and distance >= 0
        wasted = outcome_defined and not achieved
        first_hit_offset = min(hit_trace_offsets) if hit_trace_offsets else None
        first_hit_step = (
            block_step_indices[first_hit_offset]
            if first_hit_offset is not None and first_hit_offset < len(block_step_indices)
            else None
        )
        steps_to_achievement = first_hit_offset + 1 if first_hit_offset is not None else None
        blocks.append(
            {
                "block_index": block_index + 1,
                "family": block["family"],
                "target_path": block["target_path"],
                "step_indices": block_step_indices,
                "trace_indices": block["trace_indices"],
                "n_steps": len(block["trace_indices"]),
                "n_reads": len(reads),
                "outcome_defined": outcome_defined,
                "achieved": achieved,
                "wasted": wasted,
                "loop": is_loop,
                "first_hit_step": first_hit_step,
                "steps_to_achievement": steps_to_achievement,
                "min_distance": distance if distance >= 0 else None,
                "hit_nodes": _node_summaries(hit_nodes, bonus_map),
            }
        )
    return blocks


def _block_stats(purpose_blocks: list[dict], n_trace_steps: int) -> dict:
    scored = [block for block in purpose_blocks if block.get("outcome_defined")]
    achieving = [block for block in purpose_blocks if block["achieved"]]
    wasted = [block for block in purpose_blocks if block["wasted"]]
    loops = [block for block in purpose_blocks if block["loop"]]
    efficiencies = [
        block["steps_to_achievement"]
        for block in achieving
        if isinstance(block.get("steps_to_achievement"), int)
    ]
    return {
        "n_blocks": len(purpose_blocks),
        "n_scored_read_blocks": len(scored),
        "n_achieving_blocks": len(achieving),
        "n_wasted_blocks": len(wasted),
        "n_loop_blocks": len(loops),
        "n_block_steps": sum(int(block["n_steps"]) for block in purpose_blocks),
        "n_scored_read_block_steps": sum(int(block["n_steps"]) for block in scored),
        "n_achieving_block_steps": sum(int(block["n_steps"]) for block in achieving),
        "n_wasted_block_steps": sum(int(block["n_steps"]) for block in wasted),
        "n_loop_block_steps": sum(int(block["n_steps"]) for block in loops),
        "block_efficiency": sum(efficiencies) / len(efficiencies) if efficiencies else None,
        "n_trace_steps": n_trace_steps,
    }


def _bad_patterns(step_items: list[Any], purpose_blocks: list[dict]) -> dict:
    loop_blocks = [block["block_index"] for block in purpose_blocks if block["loop"]]
    max_error_run = _max_error_run(step_items)
    return {
        "loop_block_indices": loop_blocks,
        "n_loop_blocks": len(loop_blocks),
        "has_loop": bool(loop_blocks),
        "max_error_run": max_error_run,
        "error_spiral": max_error_run >= 3,
    }


def score_record(
    record: dict,
    *,
    index: int,
    bonus_maps: BonusMapStore,
    tracking_mode: str,
    near_threshold: float,
    m_max: float,
) -> dict:
    step_items = _step_trace_items(record)
    instance_id = extract_instance_id(record)
    data_source = extract_data_source(record)
    run_step = extract_run_step(record)
    rollout_index = extract_rollout_index(record)
    rollout_id = extract_rollout_id(record)
    bonus_map = bonus_maps.get(instance_id) if instance_id else None
    step_reads, reads = extract_record_reads(record, tracking_mode, step_items=step_items)
    display_step_indices = _trace_step_indices(step_items)
    step_details = _step_details(
        step_items,
        step_reads,
        bonus_map or {},
        tracking_mode,
        step_indices=display_step_indices,
    )
    purpose_blocks = _purpose_blocks(
        step_items,
        step_reads,
        bonus_map or {},
        tracking_mode,
        step_indices=display_step_indices,
    )
    block_stats = _block_stats(purpose_blocks, n_trace_steps=len(step_items))
    bad_patterns = _bad_patterns(step_items, purpose_blocks)

    result = {
        "record_index": index,
        "instance_id": instance_id,
        "data_source": data_source,
        "run_step": run_step,
        "rollout_index": rollout_index,
        "rollout_id": rollout_id,
        "step_index_origin": "one_based_display",
        "has_bonus_map": bonus_map is not None,
        "has_step_traces": bool(step_reads),
        "n_steps_with_reads": sum(1 for reads_for_step in step_reads if reads_for_step),
        "n_reads": len(reads),
        "read_files": sorted({read["file_path"] for read in reads}),
        "bonus_case_type": None,
        "bonus_traceable": None,
        "has_call_graph": False,
        "hit_call_graph": False,
        "hit_ground_truth": False,
        "hit_near": False,
        "min_distance": None,
        "best_positive_multiplier": 1.0,
        "first_hit_step": None,
        "first_ground_truth_step": None,
        "hit_precision": None,
        "hit_recall": None,
        "hit_f1": None,
        "n_hit_nodes": 0,
        "n_call_graph_nodes": 0,
        "order_score": None,
        "order_defined": False,
        "miracle_step": None,
        "miracle_severity": None,
        "block_order_score": None,
        "block_order_defined": False,
        "block_miracle_step": None,
        "block_miracle_severity": None,
        "graph_topology": None,
        "path_evaluable": False,
        "not_path_evaluable_reason": "missing_bonus_map",
        "path_case_kind": None,
        "path_covered": False,
        "path_projection": None,
        "path_hit": False,
        "anchor_hit": False,
        "root_hit": False,
        "path_node_recall": None,
        "path_read_precision": None,
        "first_anchor_step": None,
        "first_root_step": None,
        "steps_anchor_to_root": None,
        "anchor_before_root": None,
        "n_path_nodes": 0,
        "n_context_nodes": 0,
        "n_hit_path_nodes": 0,
        "path_pattern_flags": {key: False for key in PATH_PATTERN_KEYS},
        "step_details": step_details,
        "edited_root_cause": any(bool(step.get("edited_root_cause")) for step in step_details),
        "purpose_blocks": purpose_blocks,
        "bad_patterns": bad_patterns,
        **block_stats,
    }
    result.update(
        license_references(
            step_items,
            _initial_context_text(record),
            step_indices=display_step_indices,
        )
    )
    if not bonus_map:
        return _sync_path_aliases(result)

    raw_case_type = bonus_map.get("case_type")
    canonical_case_type = canonical_bonus_case_type(bonus_map)
    result["bonus_case_type"] = canonical_case_type
    result["bonus_case_type_raw"] = raw_case_type
    result["traceable_case_family"] = traceable_case_family(canonical_case_type)
    result["pattern_computable"] = bonus_map_pattern_computable(bonus_map)
    result["root_anchor_overlap"] = bonus_map.get("root_anchor_overlap") or bonus_map_root_anchor_overlap(bonus_map)
    result["bonus_traceable"] = bool(bonus_map.get("traceable"))
    nodes = bonus_map.get("call_graph_nodes", {})
    rewardable_nodes = {key: node for key, node in nodes.items() if is_rewardable_graph_node(node)}
    result["has_call_graph"] = bool(nodes)
    result["n_call_graph_nodes"] = len(nodes)
    result["n_rewardable_call_graph_nodes"] = len(rewardable_nodes)
    scoring_step_reads = step_reads if any(reads_for_step for reads_for_step in step_reads) else ([reads] if reads else [])
    scoring_display_step_indices = display_step_indices if scoring_step_reads is step_reads else ([1] if reads else [])
    step_first_hits = _step_node_first_hits(step_reads, bonus_map) if step_reads else {}
    display_step_first_hits = (
        _step_node_first_hits(step_reads, bonus_map, step_indices=display_step_indices) if step_reads else {}
    )
    display_scoring_first_hits_all = (
        _step_node_first_hits(
            scoring_step_reads,
            bonus_map,
            rewardable_only=False,
            step_indices=scoring_display_step_indices,
        )
        if scoring_step_reads
        else {}
    )
    result["graph_topology"] = _graph_topology(bonus_map, set(), display_step_first_hits)
    path_evaluable, not_path_reason = _path_evaluability(bonus_map)
    result["path_evaluable"] = path_evaluable
    result["not_path_evaluable_reason"] = not_path_reason
    result["path_case_kind"] = result["bonus_case_type"] if result["bonus_case_type"] in PATH_CASE_TYPES else None

    all_hit_nodes = _all_read_hit_nodes(reads, bonus_map, rewardable_only=False)
    path_projection = _path_projection(bonus_map, all_hit_nodes, display_scoring_first_hits_all)
    result["path_projection"] = path_projection
    path_node_keys = {node["key"] for node in path_projection.get("path_nodes", path_projection.get("chain_nodes", []))}
    context_node_keys = {node["key"] for node in path_projection.get("context_nodes", [])}
    anchor_nodes = set(path_projection.get("anchors") or [])
    root_nodes = set(path_projection.get("roots") or [])
    result["n_path_nodes"] = len(path_node_keys)
    result["n_context_nodes"] = len(context_node_keys)
    result["path_covered"] = path_evaluable and bool(path_node_keys) and bool(anchor_nodes) and bool(root_nodes)
    if path_evaluable and path_node_keys:
        path_stats = _path_step_hits(scoring_step_reads, bonus_map, path_node_keys, anchor_nodes, root_nodes)
        display_path_stats = _path_step_hits(
            scoring_step_reads,
            bonus_map,
            path_node_keys,
            anchor_nodes,
            root_nodes,
            step_indices=scoring_display_step_indices,
        )
        hit_path_nodes = path_stats["hit_path_nodes"]
        first_anchor = min(path_stats["anchor_hit_steps"]) if path_stats["anchor_hit_steps"] else None
        first_root = min(path_stats["root_hit_steps"]) if path_stats["root_hit_steps"] else None
        display_first_anchor = (
            min(display_path_stats["anchor_hit_steps"]) if display_path_stats["anchor_hit_steps"] else None
        )
        display_first_root = min(display_path_stats["root_hit_steps"]) if display_path_stats["root_hit_steps"] else None
        result["n_hit_path_nodes"] = len(hit_path_nodes)
        result["path_hit"] = bool(hit_path_nodes)
        result["anchor_hit"] = first_anchor is not None
        result["root_hit"] = first_root is not None
        result["first_anchor_step"] = display_first_anchor
        result["first_root_step"] = display_first_root
        result["path_node_recall"] = len(hit_path_nodes) / len(path_node_keys) if path_node_keys else None
        n_path_scored_reads = path_stats["read_hits_path"] + path_stats["read_hits_off_path"]
        result["path_read_precision"] = (
            path_stats["read_hits_path"] / n_path_scored_reads if n_path_scored_reads else None
        )
        if result["path_case_kind"] == LATENT_CASE and first_anchor is not None and first_root is not None:
            result["steps_anchor_to_root"] = first_root - first_anchor
            result["anchor_before_root"] = first_anchor <= first_root
        result["path_pattern_flags"] = _path_pattern_flags(
            path_evaluable=path_evaluable,
            path_case_kind=result["path_case_kind"],
            first_anchor_step=first_anchor,
            first_root_step=first_root,
            root_hit_steps=path_stats["root_hit_steps"],
            path_hit_steps=path_stats["path_hit_steps"],
            read_hits_path=path_stats["read_hits_path"],
            read_hits_off_path=path_stats["read_hits_off_path"],
            node_step_counts=path_stats["node_step_counts"],
            step_items=step_items,
        )

    distance = match_reads_to_graph(reads, bonus_map)
    if distance < 0:
        return _sync_path_aliases(result)

    hit_nodes = _all_read_hit_nodes(reads, bonus_map)
    result["n_hit_nodes"] = len(hit_nodes)
    result["graph_topology"] = _graph_topology(bonus_map, hit_nodes, display_step_first_hits)
    if rewardable_nodes:
        result["hit_recall"] = len(hit_nodes) / len(rewardable_nodes)
        hit_read_count = sum(1 for read in reads if _read_hit_nodes(read, bonus_map))
        result["hit_precision"] = hit_read_count / len(reads) if reads else None
        if result["hit_precision"] is not None and result["hit_recall"] is not None:
            denom = result["hit_precision"] + result["hit_recall"]
            result["hit_f1"] = 2 * result["hit_precision"] * result["hit_recall"] / denom if denom else 0.0

    result["hit_call_graph"] = True
    result["hit_ground_truth"] = distance == 0.0
    result["hit_near"] = distance <= near_threshold
    result["min_distance"] = distance
    result["best_positive_multiplier"] = compute_p2a_multiplier(distance, m_max, 1)
    if step_reads:
        result["first_hit_step"] = _first_matching_step_with_indices(
            step_reads,
            bonus_map,
            require_gt=False,
            step_indices=display_step_indices,
        )
        result["first_ground_truth_step"] = _first_matching_step_with_indices(
            step_reads,
            bonus_map,
            require_gt=True,
            step_indices=display_step_indices,
        )
        if _is_order_metric_evaluable(result):
            result["order_score"], result["order_defined"] = _kendall_order(step_first_hits, bonus_map)
            result["miracle_step"], result["miracle_severity"] = _miracle_stats(
                step_first_hits,
                bonus_map,
                anchor_nodes=anchor_nodes,
                root_nodes=root_nodes,
            )
        block_reads = _block_step_reads(record, tracking_mode)
        if block_reads and _is_order_metric_evaluable(result):
            block_hits = _step_node_first_hits(block_reads, bonus_map)
            result["block_order_score"], result["block_order_defined"] = _kendall_order(block_hits, bonus_map)
            result["block_miracle_step"], result["block_miracle_severity"] = _miracle_stats(
                block_hits,
                bonus_map,
                anchor_nodes=anchor_nodes,
                root_nodes=root_nodes,
            )
    else:
        result["first_hit_step"] = 1
        if distance == 0.0:
            result["first_ground_truth_step"] = 1
    return _sync_path_aliases(result)


def _rate(num: int | float, denom: int | float) -> float | None:
    if not denom:
        return None
    return num / denom


def summarize(details: list[dict], *, source: Path, bonus_map_dir: Path, tracking_mode: str, near_threshold: float, m_max: float) -> dict:
    counts = Counter()
    by_case: dict[str, Counter] = defaultdict(Counter)
    distances = []
    multipliers = []
    recalls = []
    precisions = []
    f1s = []
    order_scores = []
    block_order_scores = []
    block_efficiencies = []
    path_recalls = []
    path_read_precisions = []
    times_to_anchor = []
    times_to_root = []
    steps_anchor_to_root = []
    recall_histogram = Counter()
    hop_coverage = Counter()
    not_path_reasons = Counter()
    path_pattern_counts = Counter()
    legacy_path_pattern_counts = Counter()

    for item in details:
        _sync_path_aliases(item)
        counts["n_records"] += 1
        if item["instance_id"]:
            counts["n_with_instance_id"] += 1
        if item["has_bonus_map"]:
            counts["n_with_bonus_map"] += 1
        if item["has_call_graph"]:
            counts["n_with_call_graph"] += 1
        if item["bonus_traceable"]:
            counts["n_traceable_bonus"] += 1
        if item["n_reads"]:
            counts["n_with_reads"] += 1
        if item.get("license_evaluable"):
            counts["n_license_evaluable"] += 1
            counts["n_entity_references"] += int(item.get("n_entity_references") or 0)
            counts["n_unlicensed_references"] += int(item.get("n_unlicensed_references") or 0)
            if (item.get("n_unlicensed_references") or 0) > 0:
                counts["n_unlicensed_traces"] += 1
        counts["n_blocks"] += int(item.get("n_blocks") or 0)
        counts["n_scored_read_blocks"] += int(item.get("n_scored_read_blocks") or 0)
        counts["n_achieving_blocks"] += int(item.get("n_achieving_blocks") or 0)
        counts["n_wasted_blocks"] += int(item.get("n_wasted_blocks") or 0)
        counts["n_loop_blocks"] += int(item.get("n_loop_blocks") or 0)
        counts["n_block_steps"] += int(item.get("n_block_steps") or 0)
        counts["n_scored_read_block_steps"] += int(item.get("n_scored_read_block_steps") or 0)
        counts["n_achieving_block_steps"] += int(item.get("n_achieving_block_steps") or 0)
        counts["n_wasted_block_steps"] += int(item.get("n_wasted_block_steps") or 0)
        counts["n_loop_block_steps"] += int(item.get("n_loop_block_steps") or 0)
        if item.get("block_efficiency") is not None:
            block_efficiencies.append(item["block_efficiency"])
        bad_patterns = item.get("bad_patterns") or {}
        if bad_patterns.get("has_loop"):
            counts["n_traces_with_loop"] += 1
        if bad_patterns.get("error_spiral"):
            counts["n_traces_with_error_spiral"] += 1
        if item.get("path_covered"):
            counts["n_chain_graph_covered"] += 1
        if item.get("path_evaluable"):
            counts["n_path_evaluable"] += 1
            counts["n_chain_evaluable"] += 1
            if item.get("path_hit"):
                counts["n_chain_hit"] += 1
            if item.get("anchor_hit"):
                counts["n_anchor_hit"] += 1
            if item.get("root_hit"):
                counts["n_root_hit"] += 1
            if item.get("path_node_recall") is not None:
                path_recalls.append(item["path_node_recall"])
            if item.get("path_read_precision") is not None:
                path_read_precisions.append(item["path_read_precision"])
            if item.get("first_anchor_step") is not None:
                times_to_anchor.append(item["first_anchor_step"])
            if item.get("first_root_step") is not None:
                times_to_root.append(item["first_root_step"])
            if _is_order_metric_evaluable(item):
                counts["n_chain_order_candidates"] += 1
                if item.get("anchor_before_root") is not None:
                    counts["n_chain_order_defined"] += 1
                    steps_anchor_to_root.append(item["steps_anchor_to_root"])
                    if item.get("anchor_before_root"):
                        counts["n_anchor_before_root"] += 1
            pattern_flags = _sync_path_pattern_aliases(item.get("path_pattern_flags") or {})
            for name in PATH_PATTERN_KEYS:
                if pattern_flags.get(name):
                    path_pattern_counts[name] += 1
                    counts[f"n_{name}"] += 1
            for name in CHAIN_BAD_PATTERN_KEYS:
                if pattern_flags.get(name):
                    legacy_path_pattern_counts[name] += 1
                    if name not in PATH_PATTERN_KEYS:
                        counts[f"n_{name}"] += 1
        else:
            counts["n_not_path_evaluable"] += 1
            counts["n_not_chain_evaluable"] += 1
            not_path_reasons[str(item.get("not_path_evaluable_reason") or "unknown")] += 1
        if item["hit_call_graph"]:
            counts["n_hit_call_graph"] += 1
            distances.append(item["min_distance"])
            multipliers.append(item["best_positive_multiplier"])
        if item["hit_recall"] is not None:
            recalls.append(item["hit_recall"])
        if item["hit_precision"] is not None:
            precisions.append(item["hit_precision"])
        if item["hit_f1"] is not None:
            f1s.append(item["hit_f1"])
        if item["hit_recall"] is not None:
            recall_bucket = int(math.floor(float(item["hit_recall"]) * 10)) / 10
            recall_histogram[f"{recall_bucket:.1f}"] += 1
        for node in (item.get("graph_topology") or {}).get("nodes", []):
            if node.get("hit") and node.get("normalized_distance") is not None:
                hop_coverage[f"{float(node['normalized_distance']):.3f}"] += 1
        if _is_order_metric_evaluable(item) and item["order_defined"]:
            counts["n_order_defined"] += 1
            order_scores.append(item["order_score"])
            if item["order_score"] is not None and item["order_score"] < 0:
                counts["n_reverse_order"] += 1
        if _is_order_metric_evaluable(item) and item["block_order_defined"]:
            counts["n_block_order_defined"] += 1
            block_order_scores.append(item["block_order_score"])
            if item["block_order_score"] is not None and item["block_order_score"] < 0:
                counts["n_block_reverse_order"] += 1
        if item["hit_ground_truth"] and _is_order_metric_evaluable(item):
            counts["n_hit_ground_truth"] += 1
            if item["miracle_step"] is not None:
                counts["n_miracle_denominator"] += 1
            if item["block_miracle_step"] is not None:
                counts["n_block_miracle_denominator"] += 1
        elif item["hit_ground_truth"]:
            counts["n_hit_ground_truth"] += 1
        if item["hit_near"]:
            counts["n_hit_near"] += 1
        if _is_order_metric_evaluable(item) and item["miracle_step"] is True:
            counts["n_miracle"] += 1
        if _is_order_metric_evaluable(item) and item["block_miracle_step"] is True:
            counts["n_block_miracle"] += 1

        case_type = canonical_detail_case_type(item) or "missing_bonus_map"
        bucket = by_case[case_type]
        bucket["n"] += 1
        if item["has_call_graph"]:
            bucket["n_with_call_graph"] += 1
        if item["n_reads"]:
            bucket["n_with_reads"] += 1
        if item["hit_call_graph"]:
            bucket["n_hit_call_graph"] += 1
        if item["hit_ground_truth"]:
            bucket["n_hit_ground_truth"] += 1
        if item["hit_near"]:
            bucket["n_hit_near"] += 1
        if item.get("path_evaluable"):
            bucket["n_chain_evaluable"] += 1
        else:
            bucket["n_not_chain_evaluable"] += 1
        if item.get("path_hit"):
            bucket["n_chain_hit"] += 1
        if item.get("anchor_hit"):
            bucket["n_anchor_hit"] += 1
        if item.get("root_hit"):
            bucket["n_root_hit"] += 1

    n_records = counts["n_records"]
    n_with_bonus = counts["n_with_bonus_map"]
    n_with_graph = counts["n_with_call_graph"]
    n_path_evaluable = counts["n_chain_evaluable"]
    summary_by_case = {}
    for case_type, bucket in sorted(by_case.items()):
        denom = bucket["n_with_call_graph"] or bucket["n"]
        path_denom = bucket["n_chain_evaluable"]
        summary_by_case[case_type] = {
            **dict(bucket),
            "read_rate": _rate(bucket["n_with_reads"], bucket["n"]),
            "graph_hit_rate": _rate(bucket["n_hit_call_graph"], denom),
            "ground_truth_hit_rate": _rate(bucket["n_hit_ground_truth"], denom),
            "near_hit_rate": _rate(bucket["n_hit_near"], denom),
            "path_hit_rate": _rate(bucket["n_chain_hit"], path_denom),
            "chain_hit_rate": _rate(bucket["n_chain_hit"], path_denom),
            "anchor_hit_rate": _rate(bucket["n_anchor_hit"], path_denom),
            "root_hit_rate": _rate(bucket["n_root_hit"], path_denom),
        }

    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "source": str(source),
        "bonus_map_dir": str(bonus_map_dir),
        "tracking_mode": tracking_mode,
        "near_threshold": near_threshold,
        "m_max": m_max,
        "step_index_origin": "one_based_display",
        "counts": dict(counts),
        "rates": {
            "instance_id_rate": _rate(counts["n_with_instance_id"], n_records),
            "bonus_map_coverage": _rate(n_with_bonus, n_records),
            "call_graph_coverage": _rate(n_with_graph, n_records),
            "read_rate": _rate(counts["n_with_reads"], n_records),
            "path_coverage": _rate(counts["n_chain_graph_covered"], n_records),
            "chain_graph_coverage": _rate(counts["n_chain_graph_covered"], n_records),
            "anchor_hit_rate": _rate(counts["n_anchor_hit"], n_path_evaluable),
            "root_hit_rate": _rate(counts["n_root_hit"], n_path_evaluable),
            "path_hit_rate": _rate(counts["n_chain_hit"], n_path_evaluable),
            "chain_hit_rate": _rate(counts["n_chain_hit"], n_path_evaluable),
            "path_node_recall": sum(path_recalls) / len(path_recalls) if path_recalls else None,
            "path_read_precision": sum(path_read_precisions) / len(path_read_precisions) if path_read_precisions else None,
            "chain_node_recall": sum(path_recalls) / len(path_recalls) if path_recalls else None,
            "chain_read_precision": sum(path_read_precisions) / len(path_read_precisions) if path_read_precisions else None,
            "anchor_before_root_rate": _rate(counts["n_anchor_before_root"], counts["n_chain_order_defined"]),
            "missed_anchor_rate": _rate(counts["n_missed_anchor"], n_path_evaluable),
            "missed_root_after_anchor_rate": _rate(counts["n_missed_root_after_anchor"], n_path_evaluable),
            "root_before_anchor_rate": _rate(counts["n_root_before_anchor"], counts["n_chain_order_defined"]),
            "path_stall_rate": _rate(counts["n_path_stall"], n_path_evaluable),
            "path_read_loop_rate": _rate(counts["n_path_read_loop"], n_path_evaluable),
            "off_path_read_spree_rate": _rate(counts["n_off_path_read_spree"], n_path_evaluable),
            "error_spiral_on_path_rate": _rate(counts["n_error_spiral_on_path"], n_path_evaluable),
            "chain_stall_rate": _rate(counts["n_chain_stall"], n_path_evaluable),
            "chain_read_loop_rate": _rate(counts["n_chain_read_loop"], n_path_evaluable),
            "off_chain_read_spree_rate": _rate(counts["n_off_chain_read_spree"], n_path_evaluable),
            "error_spiral_on_chain_rate": _rate(counts["n_error_spiral_on_chain"], n_path_evaluable),
            "graph_hit_rate_over_bonus_maps": _rate(counts["n_hit_call_graph"], n_with_bonus),
            "graph_hit_rate_over_call_graphs": _rate(counts["n_hit_call_graph"], n_with_graph),
            "ground_truth_hit_rate_over_call_graphs": _rate(counts["n_hit_ground_truth"], n_with_graph),
            "near_hit_rate_over_call_graphs": _rate(counts["n_hit_near"], n_with_graph),
            "avg_node_recall": sum(recalls) / len(recalls) if recalls else None,
            "avg_read_precision": sum(precisions) / len(precisions) if precisions else None,
            "avg_hit_f1": sum(f1s) / len(f1s) if f1s else None,
            "order_defined_rate": _rate(counts["n_order_defined"], n_with_graph),
            "reverse_order_rate": _rate(counts["n_reverse_order"], counts["n_order_defined"]),
            "miracle_rate_over_gt_hits": _rate(counts["n_miracle"], counts["n_miracle_denominator"]),
            "block_order_defined_rate": _rate(counts["n_block_order_defined"], n_with_graph),
            "block_reverse_order_rate": _rate(counts["n_block_reverse_order"], counts["n_block_order_defined"]),
            "block_miracle_rate_over_gt_hits": _rate(counts["n_block_miracle"], counts["n_block_miracle_denominator"]),
            "avg_blocks_per_trace": _rate(counts["n_blocks"], n_records),
            "block_achieve_rate": _rate(counts["n_achieving_blocks"], counts["n_scored_read_blocks"]),
            "block_waste_rate": _rate(counts["n_wasted_blocks"], counts["n_scored_read_blocks"]),
            "block_loop_rate": _rate(counts["n_loop_blocks"], counts["n_blocks"]),
            "achieving_block_step_share": _rate(counts["n_achieving_block_steps"], counts["n_scored_read_block_steps"]),
            "wasted_block_step_share": _rate(counts["n_wasted_block_steps"], counts["n_scored_read_block_steps"]),
            "loop_block_step_share": _rate(counts["n_loop_block_steps"], counts["n_block_steps"]),
            "bad_pattern_trace_rate": _rate(counts["n_traces_with_loop"], n_records),
            "error_spiral_rate": _rate(counts["n_traces_with_error_spiral"], n_records),
            # Pooled over all first entity references (micro-average): of first
            # references across evaluable traces, the share with no antecedent.
            "unlicensed_reference_rate": _rate(counts["n_unlicensed_references"], counts["n_entity_references"]),
            "unlicensed_trace_rate": _rate(counts["n_unlicensed_traces"], counts["n_license_evaluable"]),
        },
        "averages": {
            "avg_entity_references": _rate(counts["n_entity_references"], counts["n_license_evaluable"]),
            "avg_unlicensed_references": _rate(counts["n_unlicensed_references"], counts["n_license_evaluable"]),
            "time_to_anchor": sum(times_to_anchor) / len(times_to_anchor) if times_to_anchor else None,
            "time_to_root": sum(times_to_root) / len(times_to_root) if times_to_root else None,
            "steps_anchor_to_root": sum(steps_anchor_to_root) / len(steps_anchor_to_root) if steps_anchor_to_root else None,
            "avg_min_distance_on_hits": sum(distances) / len(distances) if distances else None,
            "avg_best_positive_multiplier_on_hits": sum(multipliers) / len(multipliers) if multipliers else None,
            "avg_order_score": sum(order_scores) / len(order_scores) if order_scores else None,
            "avg_block_order_score": sum(block_order_scores) / len(block_order_scores) if block_order_scores else None,
            "avg_block_efficiency_steps": sum(block_efficiencies) / len(block_efficiencies) if block_efficiencies else None,
        },
        "distributions": {
            "recall_histogram": dict(sorted(recall_histogram.items())),
            "hop_coverage": dict(sorted(hop_coverage.items(), key=lambda item: float(item[0]))),
            "not_path_evaluable_reasons": dict(sorted(not_path_reasons.items())),
            "not_chain_evaluable_reasons": dict(sorted(not_path_reasons.items())),
            "path_pattern_flags": dict(sorted(path_pattern_counts.items())),
            "chain_bad_patterns": dict(sorted(legacy_path_pattern_counts.items())),
        },
        "by_case_type": summary_by_case,
    }


def summarize_trends(
    details: list[dict],
    *,
    tracking_mode: str,
    near_threshold: float,
    m_max: float,
) -> list[dict]:
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for item in details:
        run_step = item.get("run_step")
        if run_step is None:
            continue
        data_source = str(item.get("data_source") or "unknown")
        groups[(data_source, int(run_step))].append(item)

    rows = []
    for (data_source, run_step), group in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        summary = summarize(
            group,
            source=Path(f"trend/{data_source}/{run_step}"),
            bonus_map_dir=Path("."),
            tracking_mode=tracking_mode,
            near_threshold=near_threshold,
            m_max=m_max,
        )
        rows.append(
            {
                "data_source": data_source,
                "run_step": run_step,
                "n_records": summary["counts"].get("n_records", 0),
                "rates": {
                    key: summary["rates"].get(key)
                    for key in (
                        "path_coverage",
                        "chain_graph_coverage",
                        "anchor_hit_rate",
                        "root_hit_rate",
                        "path_hit_rate",
                        "chain_hit_rate",
                        "path_node_recall",
                        "path_read_precision",
                        "chain_node_recall",
                        "chain_read_precision",
                        "anchor_before_root_rate",
                        "graph_hit_rate_over_call_graphs",
                        "ground_truth_hit_rate_over_call_graphs",
                        "avg_node_recall",
                        "avg_read_precision",
                        "miracle_rate_over_gt_hits",
                        "block_achieve_rate",
                        "block_loop_rate",
                    )
                },
                "averages": {
                    key: summary["averages"].get(key)
                    for key in (
                        "time_to_anchor",
                        "time_to_root",
                        "steps_anchor_to_root",
                        "avg_order_score",
                        "avg_block_order_score",
                        "avg_block_efficiency_steps",
                    )
                },
            }
        )
    return rows


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, default=_json_default, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Score eval rollout fault-localization reads against P2A bonus maps.")
    parser.add_argument("rollouts", type=Path, help="Rollout dump file or directory (.jsonl, .json, or .parquet)")
    parser.add_argument("--bonus-map-dir", type=Path, required=True, help="Directory containing <instance_id>.json bonus maps")
    parser.add_argument("--tracking-mode", choices=["view_only", "view_and_bash"], default="view_and_bash")
    parser.add_argument("--near-threshold", type=float, default=0.5, help="Distance threshold for near-fault hits")
    parser.add_argument("--m-max", type=float, default=3.0, help="Multiplier used only for diagnostic best_positive_multiplier")
    parser.add_argument("--summary-out", type=Path, default=None, help="Optional path for summary JSON")
    parser.add_argument("--details-out", type=Path, default=None, help="Optional path for per-record JSONL details")
    args = parser.parse_args()

    if not args.rollouts.exists():
        raise FileNotFoundError(args.rollouts)
    if not args.bonus_map_dir.is_dir():
        raise NotADirectoryError(args.bonus_map_dir)

    bonus_maps = BonusMapStore(str(args.bonus_map_dir))
    details = [
        score_record(
            record,
            index=index,
            bonus_maps=bonus_maps,
            tracking_mode=args.tracking_mode,
            near_threshold=args.near_threshold,
            m_max=args.m_max,
        )
        for index, record in enumerate(iter_records(args.rollouts))
    ]
    summary = summarize(
        details,
        source=args.rollouts,
        bonus_map_dir=args.bonus_map_dir,
        tracking_mode=args.tracking_mode,
        near_threshold=args.near_threshold,
        m_max=args.m_max,
    )

    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2, default=_json_default) + "\n", encoding="utf-8")
    if args.details_out:
        write_jsonl(args.details_out, details)

    print(json.dumps(summary, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
