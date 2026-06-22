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
    is_rewardable_call_graph_node,
    match_reads_to_callgraph,
    normalize_action,
    parse_read_actions,
    parse_read_actions_from_tool_calls,
    reads_from_step_trace,
    segment_purpose_blocks,
)

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
ERROR_PATTERN = re.compile(r"\b(traceback|exception|error|failed|failure|cannot|invalid)\b", re.IGNORECASE)


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
    for idx, reads in enumerate(step_reads):
        distance = match_reads_to_callgraph(reads, bonus_map)
        if distance < 0:
            continue
        if require_gt and distance != 0.0:
            continue
        return idx
    return None


def _read_hit_nodes(read: dict, bonus_map: dict) -> set[str]:
    nodes = bonus_map.get("call_graph_nodes", {}) if bonus_map else {}
    hits = set()
    for node_key, node in nodes.items():
        if not is_rewardable_call_graph_node(node):
            continue
        if read["file_path"] != node["file_path"]:
            continue
        if read["start_line"] <= node["end_line"] and read["end_line"] >= node["start_line"]:
            hits.add(node_key)
    return hits


def _step_node_first_hits(step_reads: list[list[dict]], bonus_map: dict) -> dict[str, int]:
    first_hits: dict[str, int] = {}
    for step_idx, reads in enumerate(step_reads):
        for read in reads:
            for node_key in _read_hit_nodes(read, bonus_map):
                first_hits.setdefault(node_key, step_idx)
    return first_hits


def _all_read_hit_nodes(reads: list[dict], bonus_map: dict) -> set[str]:
    hits = set()
    for read in reads:
        hits.update(_read_hit_nodes(read, bonus_map))
    return hits


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


def _miracle_stats(first_hits: dict[str, int], bonus_map: dict) -> tuple[bool | None, int | None]:
    nodes = bonus_map.get("call_graph_nodes", {})
    intermediate = {
        node_key: node
        for node_key, node in nodes.items()
        if 0.0 < float(node.get("normalized_distance", -1.0)) < 1.0
    }
    if not intermediate:
        return None, None
    gt_steps = [
        first_hits[node_key]
        for node_key, node in nodes.items()
        if float(node.get("normalized_distance", -1.0)) == 0.0 and node_key in first_hits
    ]
    if not gt_steps:
        return False, 0
    first_gt = min(gt_steps)
    visited_intermediate_levels = {
        float(intermediate[node_key]["normalized_distance"])
        for node_key, step in first_hits.items()
        if node_key in intermediate and step <= first_gt
    }
    all_intermediate_levels = {float(node["normalized_distance"]) for node in intermediate.values()}
    missing_levels = all_intermediate_levels - visited_intermediate_levels
    is_miracle = not visited_intermediate_levels
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


def _source_preview(node: dict, max_chars: int = 500) -> str | None:
    source = node.get("source")
    if not isinstance(source, str) or not source:
        return None
    if len(source) <= max_chars:
        return source
    return source[:max_chars].rstrip() + "\n..."


def _graph_topology(bonus_map: dict, hit_nodes: set[str], first_hits: dict[str, int]) -> dict:
    nodes = bonus_map.get("call_graph_nodes", {}) if bonus_map else {}
    edge_keys = bonus_map.get("call_graph_edges", []) if bonus_map else []
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
                "node_role": node.get("node_role"),
                "excluded_from_hop_max": node.get("excluded_from_hop_max"),
                "exclusion_reason": node.get("exclusion_reason"),
                "hit": node_key in hit_nodes,
                "first_step": first_hits.get(node_key),
                "source_preview": _source_preview(node),
            }
        )
    edges = [edge for edge in edge_keys if isinstance(edge, list) and len(edge) == 2]
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
                "node_role": node.get("node_role"),
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


def _step_details(step_items: list[Any], step_reads: list[list[dict]], bonus_map: dict, tracking_mode: str) -> list[dict]:
    details = []
    for step_idx, trace in enumerate(step_items):
        trace = _normalized_trace(trace)
        reads = step_reads[step_idx] if step_idx < len(step_reads) else []
        hit_nodes = _all_read_hit_nodes(reads, bonus_map)
        action = normalize_action(trace, tracking_mode=tracking_mode)
        details.append(
            {
                "step_index": trace.get("step_idx", step_idx),
                "trace_index": step_idx,
                "family": action.get("family"),
                "target_path": action.get("target_path"),
                "n_reads": len(reads),
                "reads": reads,
                "hit_nodes": _node_summaries(hit_nodes, bonus_map),
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
    try:
        text = json.dumps(trace, default=str)[:6000]
    except TypeError:
        text = str(trace)[:6000]
    return ERROR_PATTERN.search(text) is not None


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


def _purpose_blocks(
    step_items: list[Any],
    step_reads: list[list[dict]],
    bonus_map: dict,
    tracking_mode: str,
) -> list[dict]:
    normalized_traces = [_normalized_trace(trace) for trace in step_items]
    has_graph = bool((bonus_map or {}).get("call_graph_nodes"))
    blocks = []
    for block_index, block in enumerate(segment_purpose_blocks(normalized_traces, tracking_mode=tracking_mode)):
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
        distance = match_reads_to_callgraph(reads, bonus_map)
        outcome_defined = has_graph and block["family"] == "read"
        achieved = outcome_defined and distance >= 0
        wasted = outcome_defined and not achieved
        first_hit_offset = min(hit_trace_offsets) if hit_trace_offsets else None
        first_hit_step = (
            block["step_indices"][first_hit_offset]
            if first_hit_offset is not None and first_hit_offset < len(block["step_indices"])
            else None
        )
        steps_to_achievement = first_hit_offset + 1 if first_hit_offset is not None else None
        blocks.append(
            {
                "block_index": block_index,
                "family": block["family"],
                "target_path": block["target_path"],
                "step_indices": block["step_indices"],
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
    bonus_map = bonus_maps.get(instance_id) if instance_id else None
    step_reads, reads = extract_record_reads(record, tracking_mode, step_items=step_items)
    step_details = _step_details(step_items, step_reads, bonus_map or {}, tracking_mode)
    purpose_blocks = _purpose_blocks(step_items, step_reads, bonus_map or {}, tracking_mode)
    block_stats = _block_stats(purpose_blocks, n_trace_steps=len(step_items))
    bad_patterns = _bad_patterns(step_items, purpose_blocks)

    result = {
        "record_index": index,
        "instance_id": instance_id,
        "data_source": data_source,
        "run_step": run_step,
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
        "step_details": step_details,
        "purpose_blocks": purpose_blocks,
        "bad_patterns": bad_patterns,
        **block_stats,
    }
    if not bonus_map:
        return result

    result["bonus_case_type"] = bonus_map.get("case_type")
    result["bonus_traceable"] = bool(bonus_map.get("traceable"))
    nodes = bonus_map.get("call_graph_nodes", {})
    rewardable_nodes = {key: node for key, node in nodes.items() if is_rewardable_call_graph_node(node)}
    result["has_call_graph"] = bool(nodes)
    result["n_call_graph_nodes"] = len(nodes)
    result["n_rewardable_call_graph_nodes"] = len(rewardable_nodes)
    step_first_hits = _step_node_first_hits(step_reads, bonus_map) if step_reads else {}
    result["graph_topology"] = _graph_topology(bonus_map, set(), step_first_hits)

    distance = match_reads_to_callgraph(reads, bonus_map)
    if distance < 0:
        return result

    hit_nodes = _all_read_hit_nodes(reads, bonus_map)
    result["n_hit_nodes"] = len(hit_nodes)
    result["graph_topology"] = _graph_topology(bonus_map, hit_nodes, step_first_hits)
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
        result["first_hit_step"] = _first_matching_step(step_reads, bonus_map, require_gt=False)
        result["first_ground_truth_step"] = _first_matching_step(step_reads, bonus_map, require_gt=True)
        result["order_score"], result["order_defined"] = _kendall_order(step_first_hits, bonus_map)
        result["miracle_step"], result["miracle_severity"] = _miracle_stats(step_first_hits, bonus_map)
        block_reads = _block_step_reads(record, tracking_mode)
        if block_reads:
            block_hits = _step_node_first_hits(block_reads, bonus_map)
            result["block_order_score"], result["block_order_defined"] = _kendall_order(block_hits, bonus_map)
            result["block_miracle_step"], result["block_miracle_severity"] = _miracle_stats(block_hits, bonus_map)
    else:
        result["first_hit_step"] = 0
        if distance == 0.0:
            result["first_ground_truth_step"] = 0
    return result


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
    recall_histogram = Counter()
    hop_coverage = Counter()

    for item in details:
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
        if item["order_defined"]:
            counts["n_order_defined"] += 1
            order_scores.append(item["order_score"])
            if item["order_score"] is not None and item["order_score"] < 0:
                counts["n_reverse_order"] += 1
        if item["block_order_defined"]:
            counts["n_block_order_defined"] += 1
            block_order_scores.append(item["block_order_score"])
            if item["block_order_score"] is not None and item["block_order_score"] < 0:
                counts["n_block_reverse_order"] += 1
        if item["hit_ground_truth"]:
            counts["n_hit_ground_truth"] += 1
            if item["miracle_step"] is not None:
                counts["n_miracle_denominator"] += 1
            if item["block_miracle_step"] is not None:
                counts["n_block_miracle_denominator"] += 1
        if item["hit_near"]:
            counts["n_hit_near"] += 1
        if item["miracle_step"] is True:
            counts["n_miracle"] += 1
        if item["block_miracle_step"] is True:
            counts["n_block_miracle"] += 1

        case_type = item["bonus_case_type"] or "missing_bonus_map"
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

    n_records = counts["n_records"]
    n_with_bonus = counts["n_with_bonus_map"]
    n_with_graph = counts["n_with_call_graph"]
    summary_by_case = {}
    for case_type, bucket in sorted(by_case.items()):
        denom = bucket["n_with_call_graph"] or bucket["n"]
        summary_by_case[case_type] = {
            **dict(bucket),
            "read_rate": _rate(bucket["n_with_reads"], bucket["n"]),
            "graph_hit_rate": _rate(bucket["n_hit_call_graph"], denom),
            "ground_truth_hit_rate": _rate(bucket["n_hit_ground_truth"], denom),
            "near_hit_rate": _rate(bucket["n_hit_near"], denom),
        }

    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "source": str(source),
        "bonus_map_dir": str(bonus_map_dir),
        "tracking_mode": tracking_mode,
        "near_threshold": near_threshold,
        "m_max": m_max,
        "step_index_origin": "zero_based",
        "counts": dict(counts),
        "rates": {
            "instance_id_rate": _rate(counts["n_with_instance_id"], n_records),
            "bonus_map_coverage": _rate(n_with_bonus, n_records),
            "call_graph_coverage": _rate(n_with_graph, n_records),
            "read_rate": _rate(counts["n_with_reads"], n_records),
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
        },
        "averages": {
            "avg_min_distance_on_hits": sum(distances) / len(distances) if distances else None,
            "avg_best_positive_multiplier_on_hits": sum(multipliers) / len(multipliers) if multipliers else None,
            "avg_order_score": sum(order_scores) / len(order_scores) if order_scores else None,
            "avg_block_order_score": sum(block_order_scores) / len(block_order_scores) if block_order_scores else None,
            "avg_block_efficiency_steps": sum(block_efficiencies) / len(block_efficiencies) if block_efficiencies else None,
        },
        "distributions": {
            "recall_histogram": dict(sorted(recall_histogram.items())),
            "hop_coverage": dict(sorted(hop_coverage.items(), key=lambda item: float(item[0]))),
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
