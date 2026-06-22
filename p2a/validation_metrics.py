"""Live P2A validation metrics for SWE-bench dashboard logging."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from p2a.core import BonusMapStore
from p2a.eval_fault_localization import score_record, summarize, write_jsonl

P2A_VALIDATION_METRICS = (
    "bonus_map_coverage",
    "call_graph_coverage",
    "read_rate",
    "graph_hit_rate_over_call_graphs",
    "ground_truth_hit_rate_over_call_graphs",
    "near_hit_rate_over_call_graphs",
    "avg_node_recall",
    "avg_read_precision",
    "avg_hit_f1",
    "order_defined_rate",
    "reverse_order_rate",
    "miracle_rate_over_gt_hits",
    "block_order_defined_rate",
    "block_reverse_order_rate",
    "block_miracle_rate_over_gt_hits",
    "avg_blocks_per_trace",
    "block_achieve_rate",
    "block_waste_rate",
    "block_loop_rate",
    "achieving_block_step_share",
    "wasted_block_step_share",
    "loop_block_step_share",
    "bad_pattern_trace_rate",
    "error_spiral_rate",
    "avg_min_distance_on_hits",
    "avg_best_positive_multiplier_on_hits",
    "avg_order_score",
    "avg_block_order_score",
    "avg_block_efficiency_steps",
)


def _as_python(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value
    return value


def _maybe_json(value: Any) -> Any:
    value = _as_python(value)
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return value


def _as_dict(value: Any) -> dict[str, Any]:
    value = _maybe_json(value)
    return value if isinstance(value, dict) else {}


def _row_value(values: Any, idx: int, default: Any = None) -> Any:
    if values is None:
        return default
    try:
        return _as_python(values[idx])
    except (IndexError, KeyError, TypeError):
        return default


def _get_nested(mapping: dict[str, Any], *path: str) -> Any:
    current: Any = mapping
    for key in path:
        current = _maybe_json(current)
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return _maybe_json(current)


def _record_instance_id(
    record: dict[str, Any],
    extra_fields: dict[str, Any],
    extra_info: dict[str, Any],
) -> str | None:
    for key in ("instance_id", "uid", "task_id"):
        value = record.get(key)
        if isinstance(value, str) and value and ("__" in value or key == "instance_id"):
            return value
    for value in (
        extra_fields.get("instance_id"),
        extra_info.get("instance_id"),
        _get_nested(extra_info, "tools_kwargs", "reward", "metadata", "instance_id"),
    ):
        if isinstance(value, str) and value:
            return value
    return None


def _record_data_source(
    record: dict[str, Any],
    extra_fields: dict[str, Any],
    extra_info: dict[str, Any],
) -> str:
    for value in (
        record.get("data_source"),
        extra_fields.get("data_source"),
        extra_info.get("data_source"),
    ):
        if isinstance(value, str) and value:
            return value
    return "unknown"


def _record_step_traces(record: dict[str, Any], extra_fields: dict[str, Any], extra_info: dict[str, Any]) -> Any:
    for value in (
        record.get("p2a_step_traces"),
        extra_fields.get("p2a_step_traces"),
        extra_info.get("p2a_step_traces"),
    ):
        value = _maybe_json(value)
        if value is not None:
            return value
    return None


def validation_records_from_batch(
    batch: Any,
    *,
    output_texts: list[str],
    scores: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Build scorer records from a validation DataProto after generation."""
    non_tensor = getattr(batch, "non_tensor_batch", {})
    records = []
    for idx, response_text in enumerate(output_texts):
        extra_fields = _as_dict(_row_value(non_tensor.get("extra_fields"), idx, {}))
        extra_info = _as_dict(_row_value(non_tensor.get("extra_info"), idx, {}))

        record = {
            "uid": _row_value(non_tensor.get("uid"), idx),
            "instance_id": _row_value(non_tensor.get("instance_id"), idx),
            "data_source": _row_value(non_tensor.get("data_source"), idx),
            "response_text": _row_value(non_tensor.get("response_text"), idx, response_text) or response_text,
            "p2a_step_traces": _row_value(non_tensor.get("p2a_step_traces"), idx),
            "extra_fields": dict(extra_fields),
            "extra_info": dict(extra_info),
            "score": scores[idx] if scores and idx < len(scores) else None,
        }
        record["data_source"] = _record_data_source(record, record["extra_fields"], record["extra_info"])
        record["p2a_step_traces"] = _record_step_traces(record, record["extra_fields"], record["extra_info"])
        if record["p2a_step_traces"] is None:
            record.pop("p2a_step_traces")
        instance_id = _record_instance_id(record, record["extra_fields"], record["extra_info"])
        if instance_id:
            record["instance_id"] = instance_id
            record["extra_fields"].setdefault("instance_id", instance_id)
            record["extra_info"].setdefault("instance_id", instance_id)
        record["extra_fields"].setdefault("data_source", record["data_source"])
        record["extra_info"].setdefault("data_source", record["data_source"])
        records.append(record)
    return records


def score_validation_records(
    records: Iterable[dict[str, Any]],
    *,
    bonus_maps: BonusMapStore,
    tracking_mode: str,
    near_threshold: float,
    m_max: float,
) -> list[dict[str, Any]]:
    details = []
    for index, record in enumerate(records):
        detail = score_record(
            record,
            index=index,
            bonus_maps=bonus_maps,
            tracking_mode=tracking_mode,
            near_threshold=near_threshold,
            m_max=m_max,
        )
        detail["data_source"] = record.get("data_source") or "unknown"
        details.append(detail)
    return details


def flatten_validation_metrics(
    details: list[dict[str, Any]],
    *,
    bonus_map_dir: str,
    tracking_mode: str,
    near_threshold: float,
    m_max: float,
) -> dict[str, float]:
    """Flatten scored details into val-p2a/<data_source>/<metric> keys."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detail in details:
        grouped[str(detail.get("data_source") or "unknown")].append(detail)

    metrics: dict[str, float] = {}
    for data_source, group in sorted(grouped.items()):
        summary = summarize(
            group,
            source=Path("live_validation"),
            bonus_map_dir=Path(bonus_map_dir),
            tracking_mode=tracking_mode,
            near_threshold=near_threshold,
            m_max=m_max,
        )
        values = {**summary["rates"], **summary["averages"]}
        prefix = f"val-p2a/{data_source}"
        for name in P2A_VALIDATION_METRICS:
            value = values.get(name)
            if value is not None:
                metrics[f"{prefix}/{name}"] = float(value)
        metrics[f"{prefix}/n_records"] = float(summary["counts"].get("n_records", 0))
        metrics[f"{prefix}/n_with_bonus_map"] = float(summary["counts"].get("n_with_bonus_map", 0))
        metrics[f"{prefix}/n_with_call_graph"] = float(summary["counts"].get("n_with_call_graph", 0))
    return metrics


def compute_validation_p2a_metrics(
    records: list[dict[str, Any]],
    *,
    bonus_map_dir: str,
    tracking_mode: str,
    near_threshold: float,
    m_max: float,
    details_out: str | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    bonus_maps = BonusMapStore(bonus_map_dir)
    details = score_validation_records(
        records,
        bonus_maps=bonus_maps,
        tracking_mode=tracking_mode,
        near_threshold=near_threshold,
        m_max=m_max,
    )
    metrics = flatten_validation_metrics(
        details,
        bonus_map_dir=bonus_map_dir,
        tracking_mode=tracking_mode,
        near_threshold=near_threshold,
        m_max=m_max,
    )
    if details_out:
        write_jsonl(Path(details_out), details)
    return metrics, details
