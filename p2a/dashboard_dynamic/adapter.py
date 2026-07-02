"""Normalized data adapter for the unified P2A HTML dashboard."""

from __future__ import annotations

import json
import os
import pickle
import re
import hashlib
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from p2a.bonus_map_scope import (
    CASE_FILTER_BUCKETS,
    LATENT_CASE,
    PATH_CASE_TYPES,
    canonical_detail_case_type,
)
from p2a.core import BonusMapStore, _bonus_map_candidate_ids, normalize_action, reads_from_step_trace, writes_from_step_trace
from p2a.eval_cache import (
    DONE_STATUS,
    ERROR_STATUS,
    aggregate_model_metrics,
    connection_raw_db_identity,
    connect,
    connect_readonly,
    default_dashboard_build_db_path,
    ensure_dashboard_build_db,
    init_db,
    json_dumps,
    json_loads,
    raw_db_identity,
    slim_rollout_payload,
    utc_now,
    write_dashboard_build_detail,
    write_dashboard_build_model_metrics,
)
from p2a.eval_fault_localization import (
    _json_default,
    _kendall_order,
    _miracle_stats,
    _sync_path_aliases,
    iter_records,
    score_record,
    summarize,
    summarize_trends,
)
from p2a.hf_assets import shared_p2a_data_dir


DASHBOARD_SCHEMA_VERSION = "p2a_unified_dashboard_v1"
DASHBOARD_DETAIL_CACHE_VERSION = "dashboard_detail_cache_v1"
DASHBOARD_DETAIL_CACHE_METADATA_KEY = "dashboard_detail_cache"
THIRD_PARTY_PROVIDER_SOURCES = {
    "internal_api",
    "openai_compatible",
    "third_party_api",
    "third_party",
    "api",
}
API_RUNTIME_ERROR_RE = re.compile(
    r"("
    r"insufficient\s+(?:balance|quota|credit|credits|funds)|"
    r"(?:balance|credit|credits|funds)\s+(?:insufficient|exhausted|depleted)|"
    r"quota\s+(?:exceeded|exhausted|depleted|reached)|"
    r"(?:billing|payment)\s+(?:hard\s+)?(?:limit|required|failed)|"
    r"\bHTTP\s*429\b|\b429\s+Too\s+Many\s+Requests\b|"
    r"rate\s+limit(?:ed|ing)?|too\s+many\s+requests|toomanyrequests|"
    r"余额不足|额度不足|配额不足|余额已用尽|额度已用尽|配额已用尽|账户欠费|欠费|请充值"
    r")",
    re.IGNORECASE,
)
# Legacy DB/detail JSON uses `chain_*` keys for Path concepts. Dashboard code
# should use Path helpers and only touch legacy keys through compatibility
# accessors.
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
CHAIN_BAD_PATTERN_KEYS = LEGACY_PATH_PATTERN_KEYS
PATH_METRIC_CASE_TYPES = PATH_CASE_TYPES
DYNAMIC_TRACEABLE_CASE_TYPES = PATH_METRIC_CASE_TYPES
DATASET_PARQUET_FILENAMES = {
    "swebench-hard": ("swe_bench_verified_hard.parquet",),
    "swebench-verified": ("swe_bench_verified.parquet",),
    "swebench-pro": ("swe_bench_pro.parquet",),
    "r2e-gym-subset": ("r2e_gym_subset_p2a.parquet", "r2e_gym_subset_p2a.train.parquet"),
}
THINK_BLOCK_RE = re.compile(r"<think>([\s\S]*?)(?:</think>|\Z)", re.IGNORECASE)
XML_FUNCTION_RE = re.compile(
    r"<function(?:=([A-Za-z0-9_.:-]+)|\s+name=\"([^\"]+)\")>([\s\S]*?)(?:</function>|\Z)",
    re.MULTILINE,
)
XML_PARAMETER_RE = re.compile(
    r"<parameter(?:=([A-Za-z0-9_.:-]+)|\s+name=\"([^\"]+)\")>([\s\S]*?)(?:</parameter>|\Z)",
    re.MULTILINE,
)
VERIFY_MARKERS = (
    "reward_spec",
    "Eval report:",
    "Beginning environment shutdown",
    "Environment shutdown completed",
    "num_turns:",
)
RUNNING_MARKERS = (
    "Beginning environment startup",
    "Runtime initialized",
    "STEP ",
    "MODEL INPUT",
    "ACTION:",
)


@dataclass(frozen=True)
class DashboardRequest:
    rollouts: tuple[Path, ...] = ()
    details: tuple[Path, ...] = ()
    db_path: Path | None = None
    build_db_path: Path | None = None
    log_dir: Path | None = None
    bonus_map_dir: Path | None = None
    data_file: Path | None = None
    experiment_id: str | None = None
    provider_source: str | None = None
    dataset: str | None = None
    model_api_name: str | None = None
    model_label: str | None = None
    include_db_raw_details: bool = True
    detail_offset: int = 0
    tracking_mode: str = "view_and_bash"
    near_threshold: float = 0.5
    m_max: float = 3.0
    detail_limit: int = 500
    defer_db_scoring: bool = False
    force_db_detail_rebuild: bool = False


def _safe_json_loads(value: str | None, default: Any) -> Any:
    loaded = json_loads(value, default)
    return loaded if loaded is not None else default


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_json_default, ensure_ascii=False))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _as_mapping(value: Any) -> dict[str, Any]:
    value = _safe_json_loads(value, value) if isinstance(value, str) else value
    return value if isinstance(value, dict) else {}


def _as_sequence(value: Any) -> list[Any]:
    value = _safe_json_loads(value, value) if isinstance(value, str) else value
    return value if isinstance(value, list) else []


def _tool_name(tool_call: Any) -> str:
    call = _as_mapping(tool_call)
    function = _as_mapping(call.get("function"))
    if function.get("name"):
        return str(function.get("name"))
    return str(call.get("name") or call.get("tool_name") or "unknown")


def _tool_args(tool_call: Any) -> Any:
    call = _as_mapping(tool_call)
    function = _as_mapping(call.get("function"))
    args = function.get("arguments") if function else call.get("arguments", call.get("args", {}))
    args = _safe_json_loads(args, args) if isinstance(args, str) else args
    if isinstance(args, dict):
        normalized = dict(args)
        for key in ("command", "path", "file", "old_str", "new_str"):
            value = normalized.get(key)
            if isinstance(value, dict) and isinstance(value.get("$text"), str):
                normalized[key] = value["$text"]
        return normalized
    return args


def _text_value(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=_json_default)


def _first_text(mapping: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = _text_value(mapping.get(key))
        if value:
            return value
    return ""


def _coerce_xml_value(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return ""
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def _argument_pairs(args: Any) -> list[dict[str, Any]]:
    if isinstance(args, dict):
        return [{"key": str(key), "value": _jsonable(value)} for key, value in args.items()]
    if isinstance(args, list):
        return [{"key": str(index), "value": _jsonable(value)} for index, value in enumerate(args)]
    if args in (None, ""):
        return []
    return [{"key": "value", "value": _jsonable(args)}]


def _parsed_tool_call(tool_call: Any) -> dict[str, Any]:
    name = _tool_name(tool_call)
    args = _tool_args(tool_call)
    return {"name": name, "arguments": _argument_pairs(args)}


def _tool_call_from_parsed(parsed: dict[str, Any]) -> dict[str, Any]:
    args = {pair.get("key"): pair.get("value") for pair in parsed.get("arguments", []) if pair.get("key")}
    return {"type": "function", "function": {"name": parsed.get("name") or "", "arguments": args}}


def _xml_tool_calls(text: str) -> list[dict[str, Any]]:
    calls = []
    for match in XML_FUNCTION_RE.finditer(text or ""):
        name = (match.group(1) or match.group(2) or "").strip()
        body = match.group(3)
        args = {}
        for param in XML_PARAMETER_RE.finditer(body):
            param_name = (param.group(1) or param.group(2) or "").strip()
            if param_name:
                args[param_name] = _coerce_xml_value(param.group(3))
        if name:
            calls.append({"name": name, "arguments": _argument_pairs(args)})
    return calls


def _strip_xml_tool_calls(text: str) -> str:
    return XML_FUNCTION_RE.sub("", text or "").strip()


def _block_values(value: Any, *, block_type: str | None = None) -> list[str]:
    value = _safe_json_loads(value, value) if isinstance(value, str) else value
    if not isinstance(value, list):
        return []
    parts: list[str] = []
    for block in value:
        block = _safe_json_loads(block, block) if isinstance(block, str) else block
        if isinstance(block, str) and block_type is None:
            parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        if block_type is not None and block.get("type") != block_type:
            continue
        for key in ("value", "text", "content"):
            text = block.get(key)
            if isinstance(text, str) and text:
                parts.append(text)
                break
    return parts


def _content_block_text(value: Any, *, types: set[str]) -> str:
    value = _safe_json_loads(value, value) if isinstance(value, str) else value
    if isinstance(value, str):
        return value if "text" in types else ""
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for block in value:
        block = _safe_json_loads(block, block) if isinstance(block, str) else block
        if isinstance(block, str):
            if "text" in types:
                parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type") or "")
        if block_type not in types:
            continue
        for key in ("value", "text", "content"):
            text = block.get(key)
            if isinstance(text, str) and text:
                parts.append(text)
                break
    return "".join(parts)


def _join_unique_text(parts: Iterable[str | None]) -> str:
    selected: list[str] = []
    seen_blocks: set[str] = set()
    for part in parts:
        if not part:
            continue
        text = str(part).strip()
        if not text:
            continue
        blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
        block_keys = blocks or [text]
        if all(block in seen_blocks for block in block_keys):
            continue
        selected.append(text)
        seen_blocks.update(block_keys)
    return "\n\n".join(selected)


def _split_reasoning_and_chat(trace: dict[str, Any], tool_calls: list[Any]) -> tuple[str, str, list[dict[str, Any]]]:
    response_text = _first_text(trace, ("completion", "response_text", "response", "assistant_response", "content"))
    reasoning = _first_text(trace, ("reasoning", "reasoning_content", "reasoning_text"))
    reasoning_parts = _block_values(trace.get("reasoning_blocks"), block_type="reasoning")
    if not reasoning_parts:
        reasoning_parts = _block_values(trace.get("reasoning_blocks"))
    content_reasoning = _content_block_text(trace.get("content"), types={"reasoning"})
    if content_reasoning:
        reasoning_parts.append(content_reasoning)
    if reasoning_parts:
        reasoning = _join_unique_text([reasoning, *reasoning_parts])
    text_block_chat = "\n\n".join(_block_values(trace.get("text_blocks")))
    if not response_text:
        response_text = text_block_chat or _content_block_text(trace.get("content"), types={"text", "output_text", "message"})
    think_parts = [match.group(1).strip() for match in THINK_BLOCK_RE.finditer(response_text) if match.group(1).strip()]
    if think_parts:
        reasoning = _join_unique_text([reasoning, *think_parts])
    chat = THINK_BLOCK_RE.sub("", response_text).strip()
    parsed_calls = [_parsed_tool_call(call) for call in tool_calls]
    if not parsed_calls:
        parsed_calls = _xml_tool_calls(chat)
    chat = _strip_xml_tool_calls(chat)
    if not chat:
        chat = _first_text(trace, ("chat", "chat_text", "message", "thought")) or text_block_chat
    return reasoning, chat, parsed_calls


def _source_kind(*, provider_source: str | None, schema_version: str | None, run_step: Any, log_dir: bool = False) -> str:
    source = (provider_source or "").lower()
    schema = (schema_version or "").lower()
    if source in THIRD_PARTY_PROVIDER_SOURCES or "third_party" in schema or "api" in source:
        return "third_party_api"
    if log_dir or "uni_agent" in schema:
        return "local_inference"
    if run_step is not None:
        return "local_training"
    return "offline_artifact"


def _experiment_key(parts: dict[str, Any]) -> str:
    fields = [
        parts.get("source_kind") or "unknown",
        parts.get("experiment_id") or "adhoc",
        parts.get("provider_source") or "unknown-provider",
        parts.get("dataset") or parts.get("data_source") or "unknown-dataset",
        parts.get("model_api_name") or parts.get("model_label") or "unknown-model",
        parts.get("model_label") or parts.get("model_api_name") or "unknown-label",
    ]
    if parts.get("source_kind") == "local_training":
        fields.append(parts.get("run_step") if parts.get("run_step") is not None else "unknown-step")
    return "::".join(str(field) for field in fields)


def _dataset_name(item: dict[str, Any]) -> str:
    return str(item.get("dataset") or item.get("data_source") or "unknown-dataset")


def _eval_cell_key(parts: dict[str, Any]) -> str:
    return str(parts.get("eval_cell_key") or parts.get("experiment_key") or _experiment_key(parts))


def _record_metadata(record: dict[str, Any], request: DashboardRequest, *, log_dir: bool = False) -> dict[str, Any]:
    extra = _as_mapping(record.get("extra_fields")) or _as_mapping(record.get("extra_info")) or _as_mapping(record.get("metadata"))
    provider_source = (
        request.provider_source
        or record.get("provider_source")
        or extra.get("provider_source")
        or ("local" if log_dir else None)
    )
    dataset = request.dataset or record.get("dataset") or record.get("data_source") or extra.get("data_source")
    run_step = record.get("run_step") or record.get("global_step") or record.get("trainer_step") or extra.get("run_step")
    schema_version = str(record.get("schema_version") or "")
    kind = _source_kind(
        provider_source=str(provider_source) if provider_source else None,
        schema_version=schema_version,
        run_step=run_step,
        log_dir=log_dir,
    )
    run_id = record.get("run_id") or extra.get("run_id")
    rollout_index = record.get("rollout_index", extra.get("rollout_index"))
    model_label = record.get("model_label") or record.get("model") or extra.get("model_label") or extra.get("model")
    model_api_name = record.get("model_api_name") or extra.get("model_api_name") or model_label
    experiment_id = (
        request.experiment_id
        or record.get("experiment_id")
        or extra.get("experiment_id")
        or (run_id if kind == "local_inference" and run_id else None)
        or dataset
        or "adhoc"
    )
    metadata = {
        "experiment_id": str(experiment_id) if experiment_id is not None else None,
        "source_kind": kind,
        "provider_source": str(provider_source) if provider_source is not None else None,
        "dataset": str(dataset) if dataset is not None else None,
        "model_api_name": str(model_api_name) if model_api_name is not None else None,
        "model_label": str(model_label) if model_label is not None else None,
        "run_id": str(run_id) if run_id is not None else None,
        "rollout_index": int(rollout_index) if isinstance(rollout_index, int | float) and not isinstance(rollout_index, bool) else 0,
        "rollout_id": str(record.get("rollout_id") or extra.get("rollout_id") or "") or None,
        "run_step": run_step,
        "artifact_root": str(record.get("artifact_root") or record.get("artifact_rollouts") or ""),
        "schema_version": schema_version or None,
    }
    metadata["experiment_key"] = _experiment_key(metadata)
    metadata["eval_cell_key"] = metadata["experiment_key"]
    return metadata


def _nested_mappings(record: dict[str, Any]) -> list[dict[str, Any]]:
    extra = record.get("extra_info") if isinstance(record.get("extra_info"), dict) else {}
    tools = extra.get("tools_kwargs") if isinstance(extra.get("tools_kwargs"), dict) else {}
    reward = tools.get("reward") if isinstance(tools.get("reward"), dict) else {}
    metadata = reward.get("metadata") if isinstance(reward.get("metadata"), dict) else {}
    return [record, extra, tools, reward, metadata]


def _first_text_field(record: dict[str, Any], fields: Iterable[str]) -> str | None:
    for mapping in _nested_mappings(record):
        for field in fields:
            value = mapping.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _issue_description(record: dict[str, Any]) -> str | None:
    return _first_text_field(record, ("problem_statement", "issue_description", "issue_text", "issue", "description", "problem", "title"))


def _golden_patch(record: dict[str, Any]) -> str | None:
    return _first_text_field(record, ("golden_patch", "patch", "base_patch", "fix_patch"))


def _row_instance_id(record: dict[str, Any]) -> str | None:
    direct = record.get("instance_id")
    if isinstance(direct, str) and direct:
        return direct
    for mapping in _nested_mappings(record):
        value = mapping.get("instance_id")
        if isinstance(value, str) and value:
            return value
    uid = record.get("uid")
    return uid if isinstance(uid, str) and uid else None


def _dataset_file_candidates(request: DashboardRequest, dataset: str) -> list[Path]:
    candidates: list[Path] = []

    def add(path: Path | None) -> None:
        if path is None:
            return
        expanded = path.expanduser()
        if expanded not in candidates:
            candidates.append(expanded)

    add(request.data_file)
    for filename in DATASET_PARQUET_FILENAMES.get(dataset, ()):
        try:
            add(shared_p2a_data_dir() / filename)
        except OSError:
            pass
        env_data = os.environ.get("DATA")
        if env_data:
            add(Path(env_data) / filename)
        env_datasets = os.environ.get("P2A_DATASETS_DIR")
        if env_datasets:
            add(Path(env_datasets) / "p2a" / filename)
            add(Path(env_datasets) / filename)
        add(Path.cwd() / "../../datasets/p2a" / filename)
    return candidates


def _load_dataset_lookup(request: DashboardRequest, datasets: Iterable[str]) -> dict[tuple[str, str], dict[str, str]]:
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    for dataset in sorted({str(name) for name in datasets if name}):
        for path in _dataset_file_candidates(request, dataset):
            if not path.exists():
                continue
            try:
                import pandas as pd

                records = pd.read_parquet(path).to_dict(orient="records")
            except Exception:  # noqa: BLE001 - dataset fallback must not break dashboard rendering
                continue
            for raw in records:
                if not isinstance(raw, dict):
                    continue
                record = _jsonable(raw)
                instance_id = _row_instance_id(record)
                if not instance_id:
                    continue
                item: dict[str, str] = {}
                issue = _issue_description(record)
                patch = _golden_patch(record)
                if issue:
                    item["issue_description"] = issue
                if patch:
                    item["golden_patch"] = patch
                if item:
                    lookup[(dataset, instance_id)] = item
            if lookup:
                break
    return lookup


def _enrich_details_from_dataset_parquet(details: list[dict[str, Any]], request: DashboardRequest) -> None:
    datasets = {str(detail.get("dataset") or detail.get("data_source") or "") for detail in details}
    lookup = _load_dataset_lookup(request, datasets)
    if not lookup:
        return
    for detail in details:
        dataset = str(detail.get("dataset") or detail.get("data_source") or "")
        instance_id = str(detail.get("instance_id") or "")
        values = lookup.get((dataset, instance_id))
        if not values:
            continue
        if not detail.get("issue_description") and values.get("issue_description"):
            detail["issue_description"] = values["issue_description"]
        if not detail.get("golden_patch") and values.get("golden_patch"):
            detail["golden_patch"] = values["golden_patch"]


def _source_preview_from_source(source: str, max_chars: int = 4000) -> str:
    if len(source) <= max_chars:
        return source
    return source[:max_chars].rstrip() + "\n..."


def _copy_full_node_source(node: dict[str, Any], source_node: dict[str, Any]) -> None:
    source = source_node.get("source")
    if isinstance(source, str) and source:
        node["source"] = source
        node["source_preview"] = _source_preview_from_source(source)


def _enrich_node_list_sources(nodes: Any, source_nodes: dict[str, Any]) -> None:
    if not isinstance(nodes, list):
        return
    for node in nodes:
        if not isinstance(node, dict):
            continue
        key = node.get("key")
        source_node = source_nodes.get(str(key)) if key else None
        if isinstance(source_node, dict):
            _copy_full_node_source(node, source_node)


def _enrich_details_from_bonus_maps(details: list[dict[str, Any]], bonus_map_dir: Path | None) -> None:
    if bonus_map_dir is None:
        return
    bonus_maps = BonusMapStore(str(bonus_map_dir))
    loaded: dict[str, dict[str, Any]] = {}
    for detail in details:
        instance_id = str(detail.get("instance_id") or "")
        if not instance_id:
            continue
        if instance_id not in loaded:
            bonus_map = bonus_maps.get(instance_id)
            nodes = bonus_map.get("call_graph_nodes") if isinstance(bonus_map, dict) else {}
            loaded[instance_id] = nodes if isinstance(nodes, dict) else {}
        source_nodes = loaded[instance_id]
        if not source_nodes:
            continue
        projection = _path_projection(detail)
        if projection:
            _enrich_node_list_sources(projection.get("path_nodes"), source_nodes)
            _enrich_node_list_sources(projection.get("chain_nodes"), source_nodes)
            _enrich_node_list_sources(projection.get("context_nodes"), source_nodes)
        topology = detail.get("graph_topology")
        if isinstance(topology, dict):
            _enrich_node_list_sources(topology.get("nodes"), source_nodes)


def _enrich_details_from_bonus_map_dirs(details: list[dict[str, Any]], bonus_map_dirs: dict[str, Path]) -> None:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fallback_dir = bonus_map_dirs.get("")
    for detail in details:
        dataset = _dataset_name(detail)
        bonus_dir = bonus_map_dirs.get(dataset) or fallback_dir
        if bonus_dir is None:
            continue
        by_dataset[str(bonus_dir)].append(detail)
    for path, items in by_dataset.items():
        _enrich_details_from_bonus_maps(items, Path(path))


def _step_inspection(record: dict[str, Any], step_details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    traces = _as_sequence(record.get("p2a_step_traces"))
    explicit_step_values: list[int] = []
    for trace_value in traces:
        trace = _as_mapping(trace_value)
        value = trace.get("step_idx", trace.get("step_index"))
        if isinstance(value, int | float) and not isinstance(value, bool):
            explicit_step_values.append(int(value))
    explicit_step_offset = 1 if explicit_step_values and min(explicit_step_values) == 0 else 0
    by_trace_index = {
        int(detail.get("trace_index", index)): detail
        for index, detail in enumerate(step_details or [])
        if isinstance(detail, dict)
    }
    out = []
    for index, trace_value in enumerate(traces):
        trace = _as_mapping(trace_value)
        tool_calls = _as_sequence(trace.get("tool_calls"))
        reasoning_text, chat_text, parsed_tool_calls = _split_reasoning_and_chat(trace, tool_calls)
        tool_names = [call["name"] for call in parsed_tool_calls if call.get("name")] or ["no-tool"]
        tool_args = [{pair["key"]: pair.get("value") for pair in call.get("arguments", [])} for call in parsed_tool_calls]
        primary_args = _as_mapping(tool_args[0]) if tool_args else {}
        tool_results = _as_sequence(trace.get("tool_results"))
        observation = _tool_observation(tool_results)
        scored = by_trace_index.get(index, {})
        action_trace = trace
        if parsed_tool_calls and not tool_calls:
            action_trace = {**trace, "tool_calls": [_tool_call_from_parsed(call) for call in parsed_tool_calls]}
        action = normalize_action(action_trace, tracking_mode="view_and_bash")
        recovered_reads = scored.get("reads") or reads_from_step_trace(action_trace, tracking_mode="view_and_bash")
        write_actions = scored.get("writes") or writes_from_step_trace(action_trace)
        computed_execution_error = _step_execution_error(trace, tool_results, observation)
        stale_scored_error = bool(scored.get("execution_error")) and not tool_results and not observation
        execution_error = bool(stale_scored_error or computed_execution_error)
        raw_step_index = trace.get("step_idx", trace.get("step_index"))
        if isinstance(scored.get("step_index"), int | float) and not isinstance(scored.get("step_index"), bool):
            display_step_index = int(scored["step_index"])
        elif isinstance(raw_step_index, int | float) and not isinstance(raw_step_index, bool):
            display_step_index = int(raw_step_index) + explicit_step_offset
        else:
            display_step_index = index + 1
        out.append(
            {
                "trace_index": index,
                "step_index": display_step_index,
                "tool_names": tool_names,
                "tool_name": tool_names[0] if tool_names else "no-tool",
                "tool_args": _jsonable(tool_args),
                "action_family": action.get("family"),
                "target_path": action.get("target_path"),
                "command": primary_args.get("command"),
                "path": primary_args.get("path") or primary_args.get("file"),
                "view_range": primary_args.get("view_range"),
                "old_str": primary_args.get("old_str"),
                "new_str": primary_args.get("new_str"),
                "write_actions": _jsonable(write_actions),
                "edited_root_cause": bool(scored.get("edited_root_cause")),
                "reasoning_text": reasoning_text,
                "chat_text": chat_text,
                "thought": trace.get("thought") or "",
                "think": reasoning_text,
                "response_text": _first_text(trace, ("response_text", "response", "assistant_response", "completion", "content")),
                "tool_calls": _jsonable(tool_calls),
                "parsed_tool_calls": _jsonable(parsed_tool_calls),
                "tool_results": _jsonable(tool_results),
                "observation": observation,
                "raw_action": _jsonable(tool_calls[0]) if tool_calls else "",
                "status": "error" if execution_error else "ok",
                "execution_error": execution_error,
                "recovered_reads": _jsonable(recovered_reads),
                "exit_reason": trace.get("exit_reason"),
                "parse_error": trace.get("parse_error"),
                "scored": scored,
            }
        )
    return out


def _tool_observation(tool_results: list[Any]) -> str:
    chunks = []
    for result_value in tool_results:
        result = _as_mapping(result_value)
        for key in ("observation", "content", "result", "output", "stderr", "stdout", "error"):
            value = result.get(key)
            if value not in (None, ""):
                chunks.append(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=_json_default))
                break
        if not result and isinstance(result_value, str):
            chunks.append(result_value)
    return "\n\n".join(chunks)


def _looks_like_error(text: str) -> bool:
    sample = str(text or "")[:6000]
    if API_RUNTIME_ERROR_RE.search(sample):
        return True
    if re.search(r"\b(exit status|exit code|returned non-zero|command not found|segmentation fault)\b", sample, re.IGNORECASE):
        return True
    return any(
        re.search(r"^\s*(traceback\b|error:|exception:|command failed\b|failed:|failure:|no such file\b|bash:|sh:)", line, re.IGNORECASE)
        for line in sample.splitlines()[:80]
    )


def _step_execution_error(trace: dict[str, Any], tool_results: list[Any], observation: str) -> bool:
    if _looks_like_error(observation):
        return True
    if trace.get("error") or trace.get("system_error"):
        return True
    for result_value in tool_results:
        result = _as_mapping(result_value)
        if result.get("error"):
            return True
        status = str(result.get("status") or result.get("state") or "").lower()
        if status in {"error", "failed", "failure"}:
            return True
        exit_code = result.get("exit_code", result.get("returncode"))
        if isinstance(exit_code, (int, float)) and int(exit_code) != 0:
            return True
    return False


def _enrich_detail_from_record(
    detail: dict[str, Any],
    record: dict[str, Any],
    request: DashboardRequest,
    *,
    log_dir: bool = False,
) -> dict[str, Any]:
    metadata = _record_metadata(record, request, log_dir=log_dir)
    enriched = {**metadata, **detail}
    if enriched.get("run_step") is None:
        enriched["run_step"] = metadata.get("run_step")
    if metadata.get("dataset") and enriched.get("data_source") in {None, "unknown"}:
        enriched["data_source"] = metadata["dataset"]
    enriched["experiment_key"] = metadata["experiment_key"]
    enriched["eval_cell_key"] = metadata["eval_cell_key"]
    enriched["model_label"] = metadata.get("model_label") or enriched.get("model_label")
    enriched["model_api_name"] = metadata.get("model_api_name") or enriched.get("model_api_name")
    enriched["run_id"] = metadata.get("run_id") or enriched.get("run_id")
    enriched["raw_available"] = bool(record.get("p2a_step_traces") or record.get("messages") or record.get("trajectory"))
    enriched["messages"] = _jsonable(record.get("messages") or [])
    enriched["trajectory"] = _jsonable(record.get("trajectory") or [])
    enriched["raw_response_text"] = record.get("response_text") or ""
    enriched["issue_description"] = _issue_description(record)
    enriched["golden_patch"] = _golden_patch(record)
    enriched["reward"] = record.get("reward")
    enriched["resolved"] = record.get("resolved")
    enriched["termination_reason"] = record.get("termination_reason")
    enriched["cell_status"] = record.get("status")
    enriched["error"] = record.get("error")
    enriched["system_error"] = record.get("system_error")
    token_usage = record.get("token_usage") if isinstance(record.get("token_usage"), dict) else {}
    step_traces = _as_sequence(record.get("p2a_step_traces"))
    trajectory = _as_sequence(record.get("trajectory"))
    has_trace_steps = bool(step_traces or trajectory)
    enriched["turns"] = len(step_traces or trajectory) if has_trace_steps else _number(record.get("turns"))
    enriched["tool_calls"] = _record_tool_call_count(record) if has_trace_steps else _number(record.get("tool_calls"))
    enriched["wall_time"] = _number(record.get("wall_time") or record.get("execution_time"))
    enriched["input_tokens"] = _number(token_usage.get("input_tokens"))
    enriched["output_tokens"] = _number(token_usage.get("output_tokens"))
    enriched["reasoning_tokens"] = _number(token_usage.get("reasoning_tokens"))
    enriched["cache_hit_tokens"] = _number(token_usage.get("cache_hit_tokens"))
    enriched["cache_write_tokens"] = _number(token_usage.get("cache_write_tokens"))
    enriched["cost"] = _number(token_usage.get("cost"))
    enriched["step_inspection"] = _step_inspection(record, enriched.get("step_details") or [])
    _sync_path_aliases(enriched)
    return enriched


def _is_scored_detail(record: dict[str, Any]) -> bool:
    return "record_index" in record and (
        "path_evaluable" in record or "chain_evaluable" in record or "hit_call_graph" in record or "step_details" in record
    )


def _detail_bonus_map(detail: dict[str, Any]) -> dict[str, Any] | None:
    projection = _path_projection(detail)
    nodes: dict[str, dict[str, Any]] = {}
    for node in _path_context_nodes(detail):
        if isinstance(node, dict) and node.get("key"):
            nodes[str(node["key"])] = {**node, "rewardable": node.get("rewardable", False)}
    for node in _path_nodes(detail):
        if isinstance(node, dict) and node.get("key"):
            nodes[str(node["key"])] = {**node, "rewardable": node.get("rewardable", True)}
    topology = detail.get("graph_topology") if isinstance(detail.get("graph_topology"), dict) else {}
    for node in topology.get("nodes") or []:
        if isinstance(node, dict) and node.get("key"):
            current = nodes.get(str(node["key"]), {})
            nodes[str(node["key"])] = {**node, **current, "rewardable": current.get("rewardable", node.get("rewardable", True))}
    if not nodes:
        return None
    return {
        "call_graph_nodes": nodes,
        "selected_issue_anchor_nodes": projection.get("anchors") or [],
        "root_cause_nodes": projection.get("roots") or [],
        "reward_path_edges": _path_edges(detail),
    }


def _stored_step_hit_nodes(step: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = step.get("hit_nodes")
    if not nodes and isinstance(step.get("scored"), dict):
        nodes = step["scored"].get("hit_nodes")
    return [node for node in (nodes or []) if isinstance(node, dict) and node.get("key")]


def _stored_step_order(step: dict[str, Any], fallback: int) -> int:
    for key in ("trace_index", "step_index"):
        value = step.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return int(value)
    return fallback


def _stored_step_label_offset(detail: dict[str, Any]) -> int:
    labels: list[int] = []
    for step in detail.get("step_details") or []:
        if not isinstance(step, dict):
            continue
        value = step.get("step_index")
        if isinstance(value, int | float) and not isinstance(value, bool):
            labels.append(int(value))
    return 1 if labels and min(labels) == 0 else 0


def _stored_step_display_order(step: dict[str, Any], fallback: int, offset: int) -> int:
    value = step.get("step_index")
    if isinstance(value, int | float) and not isinstance(value, bool):
        return int(value) + offset
    value = step.get("trace_index")
    if isinstance(value, int | float) and not isinstance(value, bool):
        return int(value) + 1
    return fallback + 1


def _stored_step_first_hits(detail: dict[str, Any], *, display: bool = False) -> dict[str, int]:
    first_hits: dict[str, int] = {}
    offset = _stored_step_label_offset(detail) if display else 0
    for fallback, step in enumerate(detail.get("step_details") or []):
        if not isinstance(step, dict):
            continue
        order = (
            _stored_step_display_order(step, fallback, offset)
            if display
            else _stored_step_order(step, fallback)
        )
        for node in _stored_step_hit_nodes(step):
            first_hits.setdefault(str(node["key"]), order)
    return first_hits


def _stored_block_first_hits(detail: dict[str, Any]) -> dict[str, int]:
    steps_by_trace = {
        _stored_step_order(step, fallback): step
        for fallback, step in enumerate(detail.get("step_details") or [])
        if isinstance(step, dict)
    }
    first_hits: dict[str, int] = {}
    for fallback, block in enumerate(detail.get("purpose_blocks") or []):
        if not isinstance(block, dict):
            continue
        block_order = int(block.get("block_index") if isinstance(block.get("block_index"), int) else fallback)
        step_refs = block.get("trace_indices") or block.get("step_indices") or []
        for ref in step_refs:
            if not isinstance(ref, int | float) or isinstance(ref, bool):
                continue
            step = steps_by_trace.get(int(ref))
            if not step:
                continue
            for node in _stored_step_hit_nodes(step):
                first_hits.setdefault(str(node["key"]), block_order)
    return first_hits


def _first_hit(first_hits: dict[str, int], node_keys: Iterable[str]) -> int | None:
    values = [first_hits[key] for key in node_keys if key in first_hits]
    return min(values) if values else None


def _repair_order_semantics(detail: dict[str, Any]) -> None:
    if not _is_order_metric_detail(detail):
        return
    bonus_map = _detail_bonus_map(detail)
    if bonus_map is None:
        return
    anchors = set(bonus_map.get("selected_issue_anchor_nodes") or [])
    roots = set(bonus_map.get("root_cause_nodes") or [])
    first_hits = _stored_step_first_hits(detail)
    if first_hits:
        display_first_hits = _stored_step_first_hits(detail, display=True)
        detail["order_score"], detail["order_defined"] = _kendall_order(first_hits, bonus_map)
        detail["miracle_step"], detail["miracle_severity"] = _miracle_stats(
            first_hits,
            bonus_map,
            anchor_nodes=anchors,
            root_nodes=roots,
        )
        first_anchor = _first_hit(first_hits, anchors)
        first_root = _first_hit(first_hits, roots)
        display_first_anchor = _first_hit(display_first_hits, anchors)
        display_first_root = _first_hit(display_first_hits, roots)
        detail["first_anchor_step"] = display_first_anchor
        detail["first_root_step"] = display_first_root
        if first_anchor is not None and first_root is not None:
            detail["anchor_before_root"] = first_anchor <= first_root
            detail["steps_anchor_to_root"] = first_root - first_anchor
        else:
            detail["anchor_before_root"] = None
            detail["steps_anchor_to_root"] = None
    block_hits = _stored_block_first_hits(detail)
    if block_hits:
        detail["block_order_score"], detail["block_order_defined"] = _kendall_order(block_hits, bonus_map)
        detail["block_miracle_step"], detail["block_miracle_severity"] = _miracle_stats(
            block_hits,
            bonus_map,
            anchor_nodes=anchors,
            root_nodes=roots,
        )


def _normalize_detail(detail: dict[str, Any], index: int | None = None) -> dict[str, Any]:
    normalized = {
        "record_index": index if index is not None else detail.get("record_index", 0),
        "instance_id": None,
        "data_source": "unknown",
        "run_step": None,
        "has_bonus_map": False,
        "has_step_traces": False,
        "n_steps_with_reads": 0,
        "n_reads": 0,
        "read_files": [],
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
        "chain_evaluable": False,
        "not_chain_evaluable_reason": "missing_bonus_map",
        "chain_case_kind": None,
        "chain_graph_covered": False,
        "chain_projection": None,
        "chain_hit": False,
        "anchor_hit": False,
        "root_hit": False,
        "chain_node_recall": None,
        "chain_read_precision": None,
        "first_anchor_step": None,
        "first_root_step": None,
        "steps_anchor_to_root": None,
        "anchor_before_root": None,
        "n_chain_nodes": 0,
        "n_context_nodes": 0,
        "n_hit_chain_nodes": 0,
        "chain_bad_patterns": {},
        "step_details": [],
        "edited_root_cause": False,
        "purpose_blocks": [],
        "bad_patterns": {},
        "n_blocks": 0,
        "n_scored_read_blocks": 0,
        "n_achieving_blocks": 0,
        "n_wasted_blocks": 0,
        "n_loop_blocks": 0,
        "n_block_steps": 0,
        "n_scored_read_block_steps": 0,
        "n_achieving_block_steps": 0,
        "n_wasted_block_steps": 0,
        "n_loop_block_steps": 0,
        "block_efficiency": None,
    }
    normalized.update(detail)
    _sync_path_aliases(normalized)
    _repair_order_semantics(normalized)
    _sync_path_aliases(normalized)
    return normalized


def _normalize_details(details: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for index, detail in enumerate(details):
        item = _normalize_detail(detail, index=index)
        item.setdefault("source_kind", _source_kind(
            provider_source=item.get("provider_source"),
            schema_version=item.get("schema_version"),
            run_step=item.get("run_step"),
        ))
        item.setdefault("experiment_key", _experiment_key(item))
        item.setdefault("eval_cell_key", item.get("experiment_key"))
        normalized.append(item)
    return normalized


def _finalize_summary(summary: dict[str, Any]) -> dict[str, Any]:
    counts = summary.setdefault("counts", {})
    for key in (
        "n_records",
        "n_with_instance_id",
        "n_with_bonus_map",
        "n_with_call_graph",
        "n_with_reads",
        "n_path_evaluable",
        "n_not_path_evaluable",
        "n_chain_evaluable",
        "n_not_chain_evaluable",
    ):
        counts.setdefault(key, 0)
    summary.setdefault("rates", {})
    summary.setdefault("averages", {})
    summary.setdefault("distributions", {})
    summary.setdefault("by_case_type", {})
    return summary


def _empty_summary(request: DashboardRequest) -> dict[str, Any]:
    bonus_map_dir = request.bonus_map_dir or Path(".")
    return _finalize_summary(summarize(
        [],
        source=Path("empty"),
        bonus_map_dir=bonus_map_dir,
        tracking_mode=request.tracking_mode,
        near_threshold=request.near_threshold,
        m_max=request.m_max,
    ))


def _record_dashboard_cache(record: dict[str, Any]) -> dict[str, Any]:
    cache = record.get("_dashboard_cache")
    return cache if isinstance(cache, dict) else {}


def _dashboard_build_db_path(request: DashboardRequest) -> Path | None:
    if request.build_db_path is not None:
        return request.build_db_path
    if request.db_path is None:
        return None
    return default_dashboard_build_db_path(request.db_path)


def _bonus_map_file_for_instance(bonus_map_dir: Path | None, instance_id: str) -> Path | None:
    if bonus_map_dir is None or not instance_id:
        return None
    for candidate_id in _bonus_map_candidate_ids(instance_id):
        path = bonus_map_dir / f"{candidate_id}.json"
        if path.exists():
            return path
    return None


def _dashboard_detail_fingerprint_payload(record: dict[str, Any], *, request: DashboardRequest) -> dict[str, Any] | None:
    if request.bonus_map_dir is None:
        return None
    cache = _record_dashboard_cache(record)
    raw_hash = cache.get("raw_rollout_sha256")
    if not raw_hash:
        raw_hash = _sha256_text(json_dumps(_scoreable_record(record)))
    instance_id = str(record.get("instance_id") or "")
    bonus_map_file = _bonus_map_file_for_instance(request.bonus_map_dir, instance_id)
    payload = {
        "version": DASHBOARD_DETAIL_CACHE_VERSION,
        "tracking_mode": request.tracking_mode,
        "near_threshold": request.near_threshold,
        "m_max": request.m_max,
        "raw_rollout_sha256": raw_hash,
        "bonus_map_path": str(bonus_map_file) if bonus_map_file else None,
        "bonus_map_sha256": _sha256_file(bonus_map_file) if bonus_map_file else None,
    }
    return payload


def _dashboard_detail_fingerprint_from_payload(payload: dict[str, Any] | None) -> str | None:
    if not payload:
        return None
    return _sha256_text(json_dumps(payload))


def _dashboard_detail_fingerprint(record: dict[str, Any], *, request: DashboardRequest) -> str | None:
    return _dashboard_detail_fingerprint_from_payload(_dashboard_detail_fingerprint_payload(record, request=request))


def _cached_dashboard_detail(record: dict[str, Any], fingerprint: str | None) -> dict[str, Any] | None:
    if not fingerprint:
        return None
    cache = _record_dashboard_cache(record)
    if cache.get("fingerprint") != fingerprint:
        return None
    detail = cache.get("detail")
    return dict(detail) if isinstance(detail, dict) and detail else None


def _scoreable_record(record: dict[str, Any]) -> dict[str, Any]:
    if "_dashboard_cache" not in record:
        return record
    return {key: value for key, value in record.items() if key != "_dashboard_cache"}


def _write_pending_dashboard_detail_cache(
    request: DashboardRequest,
    pending_writes: list[dict[str, Any]],
) -> None:
    build_db_path = _dashboard_build_db_path(request)
    if not request.db_path or build_db_path is None or not pending_writes:
        return
    conn: sqlite3.Connection | None = None
    wrote = False
    try:
        conn = ensure_dashboard_build_db(build_db_path)
        for item in pending_writes:
            write_dashboard_build_detail(
                conn,
                raw_db_path=request.db_path,
                raw_cell_id=int(item["raw_cell_id"]),
                experiment_id=str(item.get("experiment_id") or ""),
                provider_source=str(item.get("provider_source") or ""),
                dataset=str(item.get("dataset") or ""),
                model_api_name=str(item.get("model_api_name") or ""),
                model_label=str(item.get("model_label") or ""),
                instance_id=str(item.get("instance_id") or ""),
                rollout_index=int(item.get("rollout_index") or 0),
                rollout_id=str(item.get("rollout_id")) if item.get("rollout_id") is not None else None,
                run_id=str(item.get("run_id")) if item.get("run_id") is not None else None,
                status=str(item.get("status") or DONE_STATUS),
                raw_rollout_sha256=str(item["raw_rollout_sha256"]),
                fingerprint=str(item["fingerprint"]),
                detail=item["detail"],
                cache_metadata=item["cache_metadata"],
            )
        conn.commit()
        wrote = True
    except (OSError, sqlite3.Error):
        if conn is not None:
            try:
                conn.rollback()
            except sqlite3.Error:
                pass
    finally:
        if conn is not None:
            conn.close()
    if wrote:
        _refresh_dashboard_build_model_metrics(request)


def _deferred_dashboard_detail(record: dict[str, Any], *, index: int) -> dict[str, Any]:
    return {
        "record_index": index,
        "instance_id": _row_instance_id(record),
        "data_source": _record_dataset(record) or "unknown",
        "has_step_traces": bool(record.get("p2a_step_traces")),
        "not_path_evaluable_reason": "dashboard_cache_pending",
        "not_chain_evaluable_reason": "dashboard_cache_pending",
        "dashboard_cache_pending": True,
    }


def _score_records(
    records: Iterable[dict[str, Any]],
    *,
    request: DashboardRequest,
    start_index: int = 0,
) -> list[dict[str, Any]]:
    if request.bonus_map_dir is None:
        return []
    bonus_maps = BonusMapStore(str(request.bonus_map_dir))
    scored = []
    pending_writes: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        cache_metadata = _dashboard_detail_fingerprint_payload(record, request=request)
        fingerprint = _dashboard_detail_fingerprint_from_payload(cache_metadata)
        cached = _cached_dashboard_detail(record, fingerprint)
        if cached is not None:
            detail = cached
        elif request.defer_db_scoring and _record_dashboard_cache(record).get("cell_id"):
            detail = _deferred_dashboard_detail(record, index=start_index + index)
        else:
            detail = score_record(
                _scoreable_record(record),
                index=start_index + index,
                bonus_maps=bonus_maps,
                tracking_mode=request.tracking_mode,
                near_threshold=request.near_threshold,
                m_max=request.m_max,
            )
            cache = _record_dashboard_cache(record)
            if fingerprint and cache.get("cell_id"):
                pending_writes.append(
                    {
                        "raw_cell_id": int(cache["cell_id"]),
                        "experiment_id": record.get("experiment_id") or cache.get("experiment_id"),
                        "provider_source": record.get("provider_source") or cache.get("provider_source"),
                        "dataset": _record_dataset(record) or cache.get("dataset"),
                        "model_api_name": record.get("model_api_name") or cache.get("model_api_name"),
                        "model_label": record.get("model_label") or record.get("model") or cache.get("model_label"),
                        "instance_id": _row_instance_id(record),
                        "rollout_index": record.get("rollout_index") or cache.get("rollout_index") or 0,
                        "rollout_id": record.get("rollout_id") or cache.get("rollout_id"),
                        "run_id": record.get("run_id") or cache.get("run_id"),
                        "status": record.get("status") or cache.get("status") or DONE_STATUS,
                        "raw_rollout_sha256": (cache_metadata or {}).get("raw_rollout_sha256"),
                        "fingerprint": fingerprint,
                        "detail": detail,
                        "cache_metadata": {**(cache_metadata or {}), "fingerprint": fingerprint},
                    }
                )
        scored.append(_enrich_detail_from_record(detail, record, request))
    _write_pending_dashboard_detail_cache(request, pending_writes)
    return scored


def write_dashboard_detail_cache_for_record(
    conn: sqlite3.Connection,
    *,
    cell_id: int,
    record: dict[str, Any],
    bonus_map_dir: Path,
    build_db_path: Path | None = None,
    tracking_mode: str = "view_and_bash",
    near_threshold: float = 0.5,
    m_max: float = 3.0,
    index: int = 0,
) -> dict[str, Any]:
    raw_db_value = connection_raw_db_identity(conn)
    if not raw_db_value:
        return {"ok": False, "reason": "raw_db_path_unavailable"}
    cell_row = conn.execute(
        """
        SELECT
          c.experiment_id,
          c.provider_source,
          c.dataset,
          c.model_api_name,
          c.model_label,
          c.instance_id,
          c.rollout_index,
          c.rollout_id,
          c.run_id,
          c.status,
          r.rollout_sha256
        FROM run_cells c
        LEFT JOIN raw_rollouts r ON r.cell_id = c.id
        WHERE c.id = ?
        """,
        (int(cell_id),),
    ).fetchone()
    if cell_row is None:
        return {"ok": False, "reason": "raw_cell_missing"}
    raw_hash = cell_row["rollout_sha256"] or _sha256_text(json_dumps(_scoreable_record(record)))
    request = DashboardRequest(
        db_path=Path(raw_db_value),
        build_db_path=build_db_path,
        bonus_map_dir=bonus_map_dir,
        experiment_id=cell_row["experiment_id"],
        provider_source=cell_row["provider_source"],
        dataset=cell_row["dataset"],
        model_api_name=cell_row["model_api_name"],
        model_label=cell_row["model_label"],
        tracking_mode=tracking_mode,
        near_threshold=near_threshold,
        m_max=m_max,
    )
    cache_record = {
        **record,
        "_dashboard_cache": {
            "cell_id": int(cell_id),
            "raw_rollout_sha256": raw_hash,
        },
    }
    cache_metadata = _dashboard_detail_fingerprint_payload(cache_record, request=request)
    fingerprint = _dashboard_detail_fingerprint_from_payload(cache_metadata)
    if not fingerprint:
        return {"ok": False, "reason": "fingerprint_unavailable"}
    bonus_maps = BonusMapStore(str(bonus_map_dir))
    detail = score_record(
        _scoreable_record(cache_record),
        index=index,
        bonus_maps=bonus_maps,
        tracking_mode=tracking_mode,
        near_threshold=near_threshold,
        m_max=m_max,
    )
    build_path = _dashboard_build_db_path(request)
    if build_path is None:
        return {"ok": False, "reason": "build_db_path_unavailable"}
    with ensure_dashboard_build_db(build_path) as build_conn:
        write_dashboard_build_detail(
            build_conn,
            raw_db_path=Path(raw_db_value),
            raw_cell_id=int(cell_id),
            experiment_id=str(cell_row["experiment_id"] or ""),
            provider_source=str(cell_row["provider_source"] or ""),
            dataset=str(cell_row["dataset"] or ""),
            model_api_name=str(cell_row["model_api_name"] or ""),
            model_label=str(cell_row["model_label"] or ""),
            instance_id=str(cell_row["instance_id"] or _row_instance_id(record)),
            rollout_index=int(cell_row["rollout_index"] or 0),
            rollout_id=str(cell_row["rollout_id"]) if cell_row["rollout_id"] is not None else None,
            run_id=str(cell_row["run_id"]) if cell_row["run_id"] is not None else None,
            status=str(cell_row["status"] or DONE_STATUS),
            raw_rollout_sha256=raw_hash,
            fingerprint=fingerprint,
            detail=detail,
            cache_metadata={**(cache_metadata or {}), "fingerprint": fingerprint},
        )
        for row in _materialized_model_metrics_rows(conn, request, build_conn=build_conn, include_avg_at=True):
            write_dashboard_build_model_metrics(build_conn, raw_db_path=Path(raw_db_value), row=row)
        build_conn.commit()
    return {"ok": True, "fingerprint": fingerprint, "detail": detail}


def _raw_record_details(
    records: Iterable[dict[str, Any]],
    *,
    request: DashboardRequest,
    start_index: int = 0,
) -> list[dict[str, Any]]:
    details = []
    for index, record in enumerate(records):
        detail = {
            "record_index": start_index + index,
            "instance_id": _row_instance_id(record),
            "data_source": _record_dataset(record) or "unknown",
            "has_step_traces": bool(record.get("p2a_step_traces")),
            "not_chain_evaluable_reason": "missing_bonus_map",
        }
        details.append(_enrich_detail_from_record(detail, record, request))
    return details


def _load_record_path(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_records: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for record in iter_records(path):
        if _is_scored_detail(record):
            details.append(record)
        else:
            raw_records.append(record)
    return raw_records, details


def _load_rollout_paths(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        raw, _details = _load_record_path(path)
        records.extend(raw)
    return records


def _load_detail_paths(paths: Iterable[Path]) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for path in paths:
        raw, scored = _load_record_path(path)
        details.extend(scored)
        details.extend(record for record in raw if _is_scored_detail(record))
    return _normalize_details(details)


def _read_pickle_dict(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            value = pickle.load(handle)  # noqa: S301 - local trusted Uni-Agent artifact.
    except (OSError, pickle.PickleError, EOFError, AttributeError, ImportError):
        return {}
    return value if isinstance(value, dict) else {}


def _record_from_uni_agent_run(run_dir: Path) -> dict[str, Any] | None:
    result_path = run_dir / "interaction_result.json"
    if not result_path.exists():
        return None
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(result, dict):
        return None

    rollout_cache = _read_pickle_dict(run_dir / "rollout_cache.pkl")
    extra_fields = rollout_cache.get("extra_fields") if isinstance(rollout_cache.get("extra_fields"), dict) else {}
    record = {
        "schema_version": "p2a_uni_agent_run_v1",
        "run_id": run_dir.name,
        "instance_id": extra_fields.get("instance_id") or result.get("instance_id") or run_dir.name,
        "data_source": extra_fields.get("data_source") or result.get("data_source") or "local-uni-agent",
        "messages": result.get("messages") or [],
        "trajectory": result.get("trajectory") or [],
        "p2a_step_traces": extra_fields.get("p2a_step_traces") or result.get("p2a_step_traces") or [],
        "response_text": extra_fields.get("response_text") or result.get("response_text") or "",
        "reward": result.get("reward_score"),
        "resolved": result.get("resolved"),
        "metrics": result.get("metrics") or rollout_cache.get("metrics") or {},
        "token_usage": rollout_cache.get("token_usage") or {},
        "execution_time": result.get("execution_time"),
        "extra_fields": dict(extra_fields),
    }
    return record


def _load_uni_agent_records(log_dir: Path | None) -> list[dict[str, Any]]:
    if log_dir is None or not log_dir.exists():
        return []
    records = []
    for child in sorted(log_dir.iterdir()):
        if child.is_dir():
            record = _record_from_uni_agent_run(child)
            if record is not None:
                records.append(record)
    return records


def _db_where(
    *,
    experiment_id: str | None,
    provider_source: str | None,
    dataset: str | None,
    model_api_name: str | None = None,
    model_label: str | None = None,
) -> tuple[str, list[Any]]:
    where = []
    params: list[Any] = []
    if experiment_id:
        where.append("c.experiment_id = ?")
        params.append(experiment_id)
    if provider_source:
        where.append("c.provider_source = ?")
        params.append(provider_source)
    if dataset:
        where.append("c.dataset = ?")
        params.append(dataset)
    if model_api_name:
        where.append("c.model_api_name = ?")
        params.append(model_api_name)
    if model_label:
        where.append("c.model_label = ?")
        params.append(model_label)
    return ("WHERE " + " AND ".join(where) if where else ""), params


def _load_db_records(
    conn: sqlite3.Connection,
    request: DashboardRequest,
    *,
    build_conn: sqlite3.Connection | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    where_sql, params = _db_where(
        experiment_id=request.experiment_id,
        provider_source=request.provider_source,
        dataset=request.dataset,
        model_api_name=request.model_api_name,
        model_label=request.model_label,
    )
    raw_columns = _sqlite_columns(conn, "raw_rollouts")
    cell_columns = _sqlite_columns(conn, "run_cells")
    issue_sql = "r.issue_description" if "issue_description" in raw_columns else "NULL AS issue_description"
    patch_sql = "r.golden_patch" if "golden_patch" in raw_columns else "NULL AS golden_patch"
    rollout_hash_sql = "r.rollout_sha256 AS raw_rollout_sha256" if "rollout_sha256" in raw_columns else "NULL AS raw_rollout_sha256"
    raw_json_sql = "r.rollout_json" if "rollout_json" in raw_columns else "NULL AS rollout_json"
    raw_select_sql = (
        """
          r.messages_json,
          r.trajectory_json,
          r.p2a_step_traces_json,
          r.final_response,
          r.reward_json,
          r.resolved,
          r.token_usage_json,
          r.cache_metrics_json,
        """
        if raw_columns
        else """
          NULL AS messages_json,
          NULL AS trajectory_json,
          NULL AS p2a_step_traces_json,
          NULL AS final_response,
          NULL AS reward_json,
          NULL AS resolved,
          NULL AS token_usage_json,
          NULL AS cache_metrics_json,
        """
    )
    rollout_index_sql = "c.rollout_index" if "rollout_index" in cell_columns else "0 AS rollout_index"
    rollout_id_sql = "c.rollout_id" if "rollout_id" in cell_columns else "NULL AS rollout_id"
    rows = conn.execute(
        f"""
        SELECT
          c.id AS cell_id,
          c.experiment_id,
          c.provider_source,
          c.dataset,
          c.model_api_name,
          c.model_label,
          c.instance_id,
          {rollout_index_sql},
          {rollout_id_sql},
          c.status,
          c.error,
          c.artifact_rollouts,
          c.artifact_details,
          c.run_id,
          {issue_sql},
          {patch_sql},
          {raw_select_sql}
          {raw_json_sql},
          {rollout_hash_sql}
        FROM run_cells c
        LEFT JOIN raw_rollouts r ON r.cell_id = c.id
        {where_sql}
        ORDER BY c.model_label, c.instance_id
        """,
        params,
    ).fetchall()

    records: list[dict[str, Any]] = []
    stored_details: list[dict[str, Any]] = []
    build_rows_by_cell = {} if request.force_db_detail_rebuild else _build_detail_rows_by_cell(build_conn, request)
    bonus_file_cache: dict[tuple[str, str], Path | None] = {}
    bonus_hash_cache: dict[Path, str | None] = {}
    for row in rows:
        record = _safe_json_loads(row["rollout_json"], {})
        if not isinstance(record, dict):
            record = {}
        raw_available = bool(row["run_id"]) or any(
            row[key] is not None
            for key in (
                "messages_json",
                "trajectory_json",
                "p2a_step_traces_json",
                "final_response",
                "reward_json",
                "token_usage_json",
                "cache_metrics_json",
            )
        )
        if raw_available:
            raw_hash = row["raw_rollout_sha256"] or _sha256_text(str(row["rollout_json"] or ""))
            cache = {
                "cell_id": row["cell_id"],
                "experiment_id": row["experiment_id"],
                "provider_source": row["provider_source"],
                "dataset": row["dataset"],
                "model_api_name": row["model_api_name"],
                "model_label": row["model_label"],
                "status": row["status"],
                "run_id": row["run_id"],
                "rollout_index": row["rollout_index"],
                "rollout_id": row["rollout_id"],
                "raw_rollout_sha256": raw_hash,
            }
            validation_record = {
                **record,
                "dataset": row["dataset"],
                "data_source": row["dataset"],
                "instance_id": record.get("instance_id") or row["instance_id"],
                "raw_rollout_sha256": raw_hash,
            }
            cached_detail = _validated_build_cached_detail(
                build_rows_by_cell.get(int(row["cell_id"])),
                validation_record,
                request,
                bonus_file_cache=bonus_file_cache,
                bonus_hash_cache=bonus_hash_cache,
            )
            if cached_detail is not None:
                build_row = build_rows_by_cell[int(row["cell_id"])]
                cache["fingerprint"] = build_row["fingerprint"]
                cache["detail"] = cached_detail
            record["_dashboard_cache"] = cache
            record.setdefault("experiment_id", row["experiment_id"])
            record.setdefault("provider_source", row["provider_source"])
            record.setdefault("data_source", row["dataset"])
            record.setdefault("model", row["model_label"])
            record.setdefault("model_label", row["model_label"])
            record.setdefault("model_api_name", row["model_api_name"])
            record.setdefault("instance_id", row["instance_id"])
            record.setdefault("run_id", row["run_id"])
            record.setdefault("rollout_index", row["rollout_index"])
            record.setdefault("rollout_id", row["rollout_id"])
            record.setdefault("status", row["status"])
            record.setdefault("artifact_rollouts", row["artifact_rollouts"])
            record.setdefault("messages", _safe_json_loads(row["messages_json"], []))
            record.setdefault("trajectory", _safe_json_loads(row["trajectory_json"], []))
            record.setdefault("p2a_step_traces", _safe_json_loads(row["p2a_step_traces_json"], []))
            record.setdefault("response_text", row["final_response"] or "")
            record.setdefault("reward", _safe_json_loads(row["reward_json"], None))
            record.setdefault("resolved", bool(row["resolved"]) if row["resolved"] is not None else None)
            record.setdefault("token_usage", _safe_json_loads(row["token_usage_json"], {}))
            record.setdefault("metrics", _safe_json_loads(row["cache_metrics_json"], {}))
            if row["issue_description"] and not _issue_description(record):
                record["issue_description"] = row["issue_description"]
            if row["golden_patch"] and not _golden_patch(record):
                record["golden_patch"] = row["golden_patch"]
            records.append(record)
    return records, stored_details


def _db_bonus_map_file_for_instance(
    request: DashboardRequest,
    *,
    dataset: str | None,
    instance_id: str,
    cache: dict[tuple[str, str], Path | None],
) -> Path | None:
    key = (dataset or "", instance_id)
    if key in cache:
        return cache[key]
    if request.bonus_map_dir is not None:
        path = _bonus_map_file_for_instance(request.bonus_map_dir, instance_id)
        cache[key] = path
        return path
    path = None
    for root in _artifact_root_candidates(request):
        candidate = root / "bonus_maps" / str(dataset or "")
        if not candidate.is_dir():
            continue
        path = _bonus_map_file_for_instance(candidate, instance_id)
        if path is not None:
            break
    cache[key] = path
    return path


def _validated_db_cached_detail(
    row: sqlite3.Row,
    record: dict[str, Any],
    metrics: dict[str, Any],
    request: DashboardRequest,
    *,
    bonus_file_cache: dict[tuple[str, str], Path | None],
    bonus_hash_cache: dict[Path, str | None],
) -> dict[str, Any] | None:
    detail = metrics.get("detail")
    metadata = metrics.get(DASHBOARD_DETAIL_CACHE_METADATA_KEY)
    if not isinstance(detail, dict) or not detail or not isinstance(metadata, dict):
        return None
    if metadata.get("version") != DASHBOARD_DETAIL_CACHE_VERSION:
        return None
    raw_hash = metadata.get("raw_rollout_sha256")
    if not raw_hash:
        return None
    bonus_map_file = _db_bonus_map_file_for_instance(
        request,
        dataset=str(record.get("dataset") or record.get("data_source") or "") or None,
        instance_id=str(record.get("instance_id") or ""),
        cache=bonus_file_cache,
    )
    if bonus_map_file is not None and bonus_map_file not in bonus_hash_cache:
        bonus_hash_cache[bonus_map_file] = _sha256_file(bonus_map_file)
    payload = {
        "version": DASHBOARD_DETAIL_CACHE_VERSION,
        "tracking_mode": request.tracking_mode,
        "near_threshold": request.near_threshold,
        "m_max": request.m_max,
        "raw_rollout_sha256": raw_hash,
        "bonus_map_path": str(bonus_map_file) if bonus_map_file else None,
        "bonus_map_sha256": bonus_hash_cache.get(bonus_map_file) if bonus_map_file else None,
    }
    fingerprint = _dashboard_detail_fingerprint_from_payload(payload)
    if not fingerprint or row["fingerprint"] != fingerprint:
        return None
    if metadata.get("fingerprint") and metadata.get("fingerprint") != fingerprint:
        return None
    return dict(detail)


def _build_db_where(
    *,
    experiment_id: str | None = None,
    provider_source: str | None = None,
    dataset: str | None = None,
    model_api_name: str | None = None,
    model_label: str | None = None,
) -> tuple[str, list[Any]]:
    where = []
    params: list[Any] = []
    for field, value in (
        ("experiment_id", experiment_id),
        ("provider_source", provider_source),
        ("dataset", dataset),
        ("model_api_name", model_api_name),
        ("model_label", model_label),
    ):
        if value:
            where.append(f"{field} = ?")
            params.append(value)
    return ("WHERE " + " AND ".join(where) if where else ""), params


def _build_detail_rows_by_cell(
    build_conn: sqlite3.Connection | None,
    request: DashboardRequest,
) -> dict[int, sqlite3.Row]:
    if build_conn is None or request.db_path is None:
        return {}
    where_sql, params = _build_db_where(
        experiment_id=request.experiment_id,
        provider_source=request.provider_source,
        dataset=request.dataset,
        model_api_name=request.model_api_name,
        model_label=request.model_label,
    )
    prefix = "WHERE" if not where_sql else f"{where_sql} AND"
    try:
        rows = build_conn.execute(
            f"""
            SELECT *
            FROM dashboard_rollout_details
            {prefix} raw_db_path = ?
            """,
            [*params, raw_db_identity(request.db_path)],
        ).fetchall()
    except sqlite3.Error:
        return {}
    return {int(row["raw_cell_id"]): row for row in rows}


def _validated_build_cached_detail(
    build_row: sqlite3.Row | None,
    record: dict[str, Any],
    request: DashboardRequest,
    *,
    bonus_file_cache: dict[tuple[str, str], Path | None],
    bonus_hash_cache: dict[Path, str | None],
) -> dict[str, Any] | None:
    if build_row is None:
        return None
    detail = _safe_json_loads(build_row["detail_json"], {})
    metadata = _safe_json_loads(build_row["cache_metadata_json"], {})
    if not isinstance(detail, dict) or not detail or not isinstance(metadata, dict):
        return None
    if metadata.get("version") != DASHBOARD_DETAIL_CACHE_VERSION:
        return None
    raw_hash = metadata.get("raw_rollout_sha256")
    current_raw_hash = record.get("raw_rollout_sha256") or build_row["raw_rollout_sha256"]
    if not raw_hash or not current_raw_hash or raw_hash != current_raw_hash or build_row["raw_rollout_sha256"] != current_raw_hash:
        return None
    if metadata.get("tracking_mode") != request.tracking_mode:
        return None
    if metadata.get("near_threshold") != request.near_threshold:
        return None
    if metadata.get("m_max") != request.m_max:
        return None
    metadata_fingerprint = metadata.get("fingerprint")
    if metadata_fingerprint and metadata_fingerprint != build_row["fingerprint"]:
        return None
    bonus_map_file = _db_bonus_map_file_for_instance(
        request,
        dataset=str(record.get("dataset") or record.get("data_source") or "") or None,
        instance_id=str(record.get("instance_id") or ""),
        cache=bonus_file_cache,
    )
    if bonus_map_file is None and request.bonus_map_dir is None:
        return dict(detail) if metadata_fingerprint == build_row["fingerprint"] else None
    if bonus_map_file is not None and bonus_map_file not in bonus_hash_cache:
        bonus_hash_cache[bonus_map_file] = _sha256_file(bonus_map_file)
    payload = {
        "version": DASHBOARD_DETAIL_CACHE_VERSION,
        "tracking_mode": request.tracking_mode,
        "near_threshold": request.near_threshold,
        "m_max": request.m_max,
        "raw_rollout_sha256": raw_hash,
        "bonus_map_path": str(bonus_map_file) if bonus_map_file else None,
        "bonus_map_sha256": bonus_hash_cache.get(bonus_map_file) if bonus_map_file else None,
    }
    fingerprint = _dashboard_detail_fingerprint_from_payload(payload)
    if not fingerprint or build_row["fingerprint"] != fingerprint:
        return None
    if metadata_fingerprint and metadata_fingerprint != fingerprint:
        return None
    return dict(detail)


def _load_db_cached_details(
    conn: sqlite3.Connection,
    request: DashboardRequest,
    *,
    build_conn: sqlite3.Connection | None = None,
) -> list[dict[str, Any]]:
    where_sql, params = _db_where(
        experiment_id=request.experiment_id,
        provider_source=request.provider_source,
        dataset=request.dataset,
        model_api_name=request.model_api_name,
        model_label=request.model_label,
    )
    cell_columns = _sqlite_columns(conn, "run_cells")
    rollout_index_sql = "c.rollout_index" if "rollout_index" in cell_columns else "0 AS rollout_index"
    rollout_index_order_sql = "c.rollout_index" if "rollout_index" in cell_columns else "rollout_index"
    rollout_id_sql = "c.rollout_id" if "rollout_id" in cell_columns else "NULL AS rollout_id"
    metric_columns = _sqlite_columns(conn, "quantitative_metrics")
    fingerprint_sql = "q.fingerprint" if "fingerprint" in metric_columns else "NULL AS fingerprint"
    include_raw = request.include_db_raw_details
    raw_columns = _sqlite_columns(conn, "raw_rollouts")
    raw_join_sql = "LEFT JOIN raw_rollouts r ON r.cell_id = c.id" if raw_columns else ""
    issue_sql = "r.issue_description" if include_raw and "issue_description" in raw_columns else "NULL AS issue_description"
    patch_sql = "r.golden_patch" if include_raw and "golden_patch" in raw_columns else "NULL AS golden_patch"
    raw_hash_sql = "r.rollout_sha256" if "rollout_sha256" in raw_columns else "NULL"
    raw_select_sql = (
        f"""
          r.messages_json,
          r.trajectory_json,
          r.p2a_step_traces_json,
          r.final_response,
          r.reward_json,
          r.resolved,
          r.token_usage_json,
          r.cache_metrics_json,
          {raw_hash_sql} AS raw_rollout_sha256,
          NULL AS rollout_json,
        """
        if include_raw and raw_columns
        else """
          NULL AS messages_json,
          NULL AS trajectory_json,
          NULL AS p2a_step_traces_json,
          NULL AS final_response,
          NULL AS reward_json,
          NULL AS resolved,
          NULL AS token_usage_json,
          NULL AS cache_metrics_json,
          {raw_hash_sql} AS raw_rollout_sha256,
          NULL AS rollout_json,
        """.format(raw_hash_sql=raw_hash_sql)
    )
    limit_sql = ""
    query_params = list(params)
    detail_where_sql = where_sql
    if include_raw:
        detail_where_sql = f"{detail_where_sql} AND" if detail_where_sql else "WHERE"
        detail_where_sql = f"{detail_where_sql} c.status IN (?, ?)"
        query_params.extend([DONE_STATUS, ERROR_STATUS])
        if raw_columns:
            detail_where_sql = f"{detail_where_sql} AND r.cell_id IS NOT NULL"
    if include_raw and request.detail_limit > 0:
        limit_sql = "LIMIT ? OFFSET ?"
        query_params.extend([int(request.detail_limit), max(0, int(request.detail_offset or 0))])
    order_sql = "c.id" if include_raw else f"c.model_label, c.instance_id, {rollout_index_order_sql}"
    rows = conn.execute(
        f"""
        SELECT
          c.id AS cell_id,
          c.experiment_id,
          c.provider_source,
          c.dataset,
          c.model_api_name,
          c.model_label,
          c.instance_id,
          {rollout_index_sql},
          {rollout_id_sql},
          c.status,
          c.error,
          c.artifact_rollouts,
          c.artifact_details,
          c.run_id,
          {issue_sql},
          {patch_sql},
          {raw_select_sql}
          q.reward AS metric_reward,
          q.resolved AS metric_resolved,
          q.turns AS metric_turns,
          q.tool_calls AS metric_tool_calls,
          q.wall_time AS metric_wall_time,
          q.input_tokens AS metric_input_tokens,
          q.output_tokens AS metric_output_tokens,
          q.reasoning_tokens AS metric_reasoning_tokens,
          q.cache_hit_tokens AS metric_cache_hit_tokens,
          q.cache_write_tokens AS metric_cache_write_tokens,
          q.cost AS metric_cost,
          {fingerprint_sql}
        FROM run_cells c
        {raw_join_sql}
        LEFT JOIN quantitative_metrics q ON q.cell_id = c.id
        {detail_where_sql}
        ORDER BY {order_sql}
        {limit_sql}
        """,
        query_params,
    ).fetchall()

    details: list[dict[str, Any]] = []
    bonus_file_cache: dict[tuple[str, str], Path | None] = {}
    bonus_hash_cache: dict[Path, str | None] = {}
    build_rows_by_cell = _build_detail_rows_by_cell(build_conn, request)
    for index, row in enumerate(rows):
        record = _safe_json_loads(row["rollout_json"], {})
        if not isinstance(record, dict):
            record = {}
        record = {
            **record,
            "cell_id": row["cell_id"],
            "experiment_id": row["experiment_id"],
            "provider_source": row["provider_source"],
            "dataset": row["dataset"],
            "data_source": row["dataset"],
            "model": row["model_label"],
            "model_label": row["model_label"],
            "model_api_name": row["model_api_name"],
            "instance_id": row["instance_id"],
            "run_id": row["run_id"],
            "rollout_index": row["rollout_index"],
            "rollout_id": row["rollout_id"],
            "artifact_rollouts": row["artifact_rollouts"],
            "artifact_details": row["artifact_details"],
            "status": row["status"],
            "error": row["error"],
            "issue_description": row["issue_description"],
            "golden_patch": row["golden_patch"],
            "raw_rollout_sha256": row["raw_rollout_sha256"],
        }
        record.setdefault("messages", _safe_json_loads(row["messages_json"], []))
        record.setdefault("trajectory", _safe_json_loads(row["trajectory_json"], []))
        record.setdefault("p2a_step_traces", _safe_json_loads(row["p2a_step_traces_json"], []))
        record.setdefault("response_text", row["final_response"] or "")
        record.setdefault("reward", _safe_json_loads(row["reward_json"], None))
        record.setdefault("resolved", bool(row["resolved"]) if row["resolved"] is not None else None)
        if record.get("reward") is None and row["metric_reward"] is not None:
            record["reward"] = row["metric_reward"]
        if record.get("resolved") is None and row["metric_resolved"] is not None:
            record["resolved"] = bool(row["metric_resolved"])
        record.setdefault("turns", row["metric_turns"])
        record.setdefault("tool_calls", row["metric_tool_calls"])
        record.setdefault("wall_time", row["metric_wall_time"])
        token_usage = record.get("token_usage") if isinstance(record.get("token_usage"), dict) else _safe_json_loads(row["token_usage_json"], {})
        if not isinstance(token_usage, dict):
            token_usage = {}
        for token_key, column_key in (
            ("input_tokens", "metric_input_tokens"),
            ("output_tokens", "metric_output_tokens"),
            ("reasoning_tokens", "metric_reasoning_tokens"),
            ("cache_hit_tokens", "metric_cache_hit_tokens"),
            ("cache_write_tokens", "metric_cache_write_tokens"),
            ("cost", "metric_cost"),
        ):
            if token_usage.get(token_key) is None and row[column_key] is not None:
                token_usage[token_key] = row[column_key]
        record["token_usage"] = token_usage
        record.setdefault("metrics", _safe_json_loads(row["cache_metrics_json"], {}))
        detail = (
            _validated_build_cached_detail(
                build_rows_by_cell.get(int(row["cell_id"])),
                record,
                request,
                bonus_file_cache=bonus_file_cache,
                bonus_hash_cache=bonus_hash_cache,
            )
        )
        if detail is not None:
            details.append(_enrich_detail_from_record(detail, record, request))
        else:
            details.append(_enrich_detail_from_record(_deferred_dashboard_detail(record, index=index), record, request))
    return details


def _tail(path: Path, max_chars: int = 60_000) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(0, size - max_chars))
            payload = handle.read()
    except OSError:
        return ""
    return payload.decode("utf-8", errors="replace")


def _infer_status(file_names: set[str], log_excerpt: str) -> str:
    if "interaction_result.json" in file_names:
        return "completed"
    if "run.log" not in file_names:
        return "queued"
    if any(marker in log_excerpt for marker in VERIFY_MARKERS):
        return "verify"
    if any(marker in log_excerpt for marker in RUNNING_MARKERS):
        return "running"
    return "queued"


def _run_snapshot(run_dir: Path) -> dict[str, Any]:
    file_names = []
    latest = run_dir.stat().st_mtime
    for child in sorted(run_dir.rglob("*")):
        if child.is_file():
            file_names.append(child.relative_to(run_dir).as_posix())
            try:
                latest = max(latest, child.stat().st_mtime)
            except OSError:
                pass
    run_log = run_dir / "run.log"
    log_excerpt = _tail(run_log) if run_log.exists() else ""
    file_set = set(file_names)
    return {
        "run_id": run_dir.name,
        "path": str(run_dir),
        "status": _infer_status(file_set, log_excerpt),
        "last_update": latest,
        "files": file_names,
        "log_sources": [
            {"key": rel, "label": "Run log" if rel == "run.log" else Path(rel).name}
            for rel in file_names
            if rel == "run.log" or rel.endswith((".log", ".txt"))
        ],
        "log_excerpt": log_excerpt,
    }


def _scan_log_dir(log_dir: Path | None) -> list[dict[str, Any]]:
    if log_dir is None or not log_dir.exists():
        return []
    return [_run_snapshot(child) for child in sorted(log_dir.iterdir()) if child.is_dir()]


def _scan_db_artifact_runs(conn: sqlite3.Connection, request: DashboardRequest) -> list[dict[str, Any]]:
    where_sql, params = _db_where(
        experiment_id=request.experiment_id,
        provider_source=request.provider_source,
        dataset=request.dataset,
        model_api_name=request.model_api_name,
        model_label=request.model_label,
    )
    rows = conn.execute(
        f"""
        SELECT DISTINCT
          experiment_id,
          provider_source,
          dataset,
          model_api_name,
          model_label,
          artifact_rollouts
        FROM run_cells c
        {where_sql}
        """,
        params,
    ).fetchall()
    run_meta: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not row["artifact_rollouts"]:
            continue
        path = Path(str(row["artifact_rollouts"])).expanduser()
        if path.name == "rollouts.jsonl":
            path = path.parent
        metadata = {
            "source_kind": _source_kind(
                provider_source=row["provider_source"],
                schema_version=None,
                run_step=None,
            ),
            "experiment_id": row["experiment_id"],
            "provider_source": row["provider_source"],
            "dataset": row["dataset"],
            "model_api_name": row["model_api_name"],
            "model_label": row["model_label"],
        }
        metadata["experiment_key"] = _experiment_key(metadata)
        metadata["eval_cell_key"] = metadata["experiment_key"]
        item = run_meta.setdefault(
            str(path),
            {
                "path": path,
                "eval_cell_keys": set(),
                "datasets": set(),
                "model_labels": set(),
                "experiment_ids": set(),
            },
        )
        item["eval_cell_keys"].add(metadata["eval_cell_key"])
        item["datasets"].add(str(row["dataset"]))
        item["model_labels"].add(str(row["model_label"]))
        item["experiment_ids"].add(str(row["experiment_id"]))
    runs = []
    for item in sorted(run_meta.values(), key=lambda value: str(value["path"])):
        path = item["path"]
        if not path.exists():
            continue
        run = _run_snapshot(path)
        run["eval_cell_keys"] = sorted(item["eval_cell_keys"])
        run["datasets"] = sorted(item["datasets"])
        run["model_labels"] = sorted(item["model_labels"])
        run["experiment_ids"] = sorted(item["experiment_ids"])
        runs.append(run)
    return runs


def _read_log_from_runs(runs: Iterable[dict[str, Any]], run_id: str, source: str) -> dict[str, Any]:
    for run in runs:
        if run.get("run_id") != run_id:
            continue
        allowed_sources = {
            str(item.get("key"))
            for item in run.get("log_sources") or []
            if isinstance(item, dict) and item.get("key")
        }
        if source not in allowed_sources:
            raise FileNotFoundError(f"Run {run_id!r} log source {source!r} is not an enumerated log file")
        base = Path(str(run.get("path") or "")).resolve()
        path = (base / source).resolve()
        if base != path and base not in path.parents:
            raise FileNotFoundError(f"Run {run_id!r} log source {source!r} escapes the run directory")
        if not path.is_file():
            raise FileNotFoundError(f"Run {run_id!r} log source {source!r} not found")
        text = _tail(path, max_chars=120_000)
        return {
            "run_id": run_id,
            "source": source,
            "text": text,
            "file_size": path.stat().st_size,
        }
    raise FileNotFoundError(f"Run {run_id!r} log source {source!r} not found")


def _summary_source(request: DashboardRequest) -> Path:
    if request.db_path:
        return request.db_path
    if request.rollouts:
        return request.rollouts[0] if len(request.rollouts) == 1 else Path("multiple_rollout_sources")
    if request.details:
        return request.details[0] if len(request.details) == 1 else Path("multiple_detail_sources")
    if request.log_dir:
        return request.log_dir
    return Path("empty")


def _source_list(request: DashboardRequest) -> list[dict[str, str]]:
    sources = []
    for path in request.rollouts:
        sources.append({"kind": "rollouts", "path": str(path)})
    for path in request.details:
        sources.append({"kind": "details", "path": str(path)})
    if request.db_path:
        sources.append({"kind": "db", "path": str(request.db_path)})
    if request.log_dir:
        sources.append({"kind": "log_dir", "path": str(request.log_dir)})
    if request.bonus_map_dir:
        sources.append({"kind": "bonus_map_dir", "path": str(request.bonus_map_dir)})
    return sources


def _artifact_root_candidates(request: DashboardRequest) -> list[Path]:
    candidates: list[Path] = []

    def add(path: Path | None) -> None:
        if path is None:
            return
        resolved = path.expanduser()
        if resolved not in candidates:
            candidates.append(resolved)

    env_root = os.environ.get("P2A_ARTIFACTS_DIR") or os.environ.get("P2A_PROJECT_DATA_DIR")
    if env_root:
        add(Path(env_root))
    if request.db_path:
        add(request.db_path.parent.parent)
    if request.log_dir:
        add(request.log_dir)
        add(request.log_dir.parent)
    add(Path(__file__).resolve().parents[1] / "data")
    add(Path.cwd() / "data")
    return candidates


def _record_dataset(record: dict[str, Any]) -> str | None:
    extra = record.get("extra_info") if isinstance(record.get("extra_info"), dict) else {}
    value = record.get("dataset") or record.get("data_source") or extra.get("data_source")
    return str(value) if value else None


def _detail_dataset(detail: dict[str, Any]) -> str | None:
    value = detail.get("dataset") or detail.get("data_source")
    return str(value) if value else None


def _dataset_instance_ids(
    raw_records: Iterable[dict[str, Any]],
    details: Iterable[dict[str, Any]],
) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for record in raw_records:
        dataset = _record_dataset(record) or "unknown-dataset"
        instance_id = record.get("instance_id")
        if instance_id:
            out[dataset].add(str(instance_id))
    for detail in details:
        dataset = _detail_dataset(detail) or "unknown-dataset"
        instance_id = detail.get("instance_id")
        if instance_id:
            out[dataset].add(str(instance_id))
    return out


def _bonus_map_dir_has_instance(candidate: Path, instance_id: str) -> bool:
    return any((candidate / f"{candidate_id}.json").exists() for candidate_id in _bonus_map_candidate_ids(instance_id))


_BONUS_MAP_DIR_CACHE: dict[tuple[Any, ...], dict[str, Path]] = {}


def _source_version(path: Path | None) -> tuple[str, int, int] | None:
    if path is None:
        return None
    try:
        stat = path.stat()
    except OSError:
        return (str(path), -1, -1)
    return (str(path), int(stat.st_mtime_ns), int(stat.st_size))


def _records_version(raw_records: Iterable[dict[str, Any]], details: Iterable[dict[str, Any]]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    record_bits = tuple(sorted(str(record.get("instance_id") or "") for record in raw_records if record.get("instance_id")))
    detail_bits = tuple(sorted(str(detail.get("instance_id") or "") for detail in details if detail.get("instance_id")))
    return record_bits, detail_bits


def _effective_bonus_map_dirs(
    request: DashboardRequest,
    raw_records: Iterable[dict[str, Any]],
    details: Iterable[dict[str, Any]],
) -> dict[str, Path]:
    raw_records = list(raw_records)
    details = list(details)
    cache_key = (
        str(request.bonus_map_dir) if request.bonus_map_dir is not None else None,
        request.dataset,
        _source_version(request.db_path),
        _source_version(request.log_dir),
        _source_version(request.data_file),
        _records_version(raw_records, details),
        tuple((str(path), _source_version(path)) for path in request.rollouts),
        tuple((str(path), _source_version(path)) for path in request.details),
    )
    cached = _BONUS_MAP_DIR_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)
    datasets = {name for name in [_record_dataset(record) for record in raw_records] if name}
    datasets.update(name for name in [_detail_dataset(detail) for detail in details] if name)
    if request.dataset:
        datasets = {request.dataset}
    if request.bonus_map_dir is not None:
        if not datasets:
            result = {"": request.bonus_map_dir}
        else:
            result = {dataset: request.bonus_map_dir for dataset in datasets}
        _BONUS_MAP_DIR_CACHE[cache_key] = dict(result)
        return result
    if not datasets:
        return {}
    instance_ids_by_dataset = _dataset_instance_ids(raw_records, details)
    bonus_map_dirs: dict[str, Path] = {}
    for dataset in sorted(datasets):
        instance_ids = instance_ids_by_dataset.get(dataset, set())
        for root in _artifact_root_candidates(request):
            candidate = root / "bonus_maps" / dataset
            if candidate.is_dir() and (
                not instance_ids or any(_bonus_map_dir_has_instance(candidate, instance_id) for instance_id in instance_ids)
            ):
                bonus_map_dirs[dataset] = candidate
                break
    _BONUS_MAP_DIR_CACHE[cache_key] = dict(bonus_map_dirs)
    return bonus_map_dirs


def _effective_bonus_map_dir(
    request: DashboardRequest,
    raw_records: Iterable[dict[str, Any]],
    details: Iterable[dict[str, Any]],
) -> Path | None:
    bonus_map_dirs = _effective_bonus_map_dirs(request, raw_records, details)
    unique_paths = {path for path in bonus_map_dirs.values()}
    if len(unique_paths) == 1:
        return next(iter(unique_paths))
    return None


def _bonus_map_summary_dir(bonus_map_dirs: dict[str, Path]) -> Path:
    unique_paths = {path for path in bonus_map_dirs.values()}
    if len(unique_paths) == 1:
        return next(iter(unique_paths))
    return Path("multiple_bonus_map_dirs") if unique_paths else Path(".")


def _score_records_by_bonus_map_dir(
    records: list[dict[str, Any]],
    *,
    request: DashboardRequest,
    bonus_map_dirs: dict[str, Path],
    start_index: int = 0,
) -> tuple[list[dict[str, Any]], set[str]]:
    scored: list[dict[str, Any]] = []
    scored_datasets: set[str] = set()
    fallback_dir = bonus_map_dirs.get("")
    records_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_dataset[_record_dataset(record) or "unknown-dataset"].append(record)
    for dataset, dataset_records in sorted(records_by_dataset.items()):
        bonus_dir = bonus_map_dirs.get(dataset) or fallback_dir
        if bonus_dir is None:
            continue
        score_request = replace(request, bonus_map_dir=bonus_dir, dataset=dataset if request.dataset is None else request.dataset)
        scored.extend(_score_records(dataset_records, request=score_request, start_index=start_index + len(scored)))
        scored_datasets.add(dataset)
    return scored, scored_datasets


def _records_for_unscored_datasets(records: Iterable[dict[str, Any]], scored_datasets: set[str]) -> list[dict[str, Any]]:
    return [record for record in records if (_record_dataset(record) or "unknown-dataset") not in scored_datasets]


def _source_list_with_bonus(request: DashboardRequest, bonus_map_dirs: dict[str, Path] | Path | None) -> list[dict[str, str]]:
    sources = _source_list(request)
    if request.bonus_map_dir is None:
        if isinstance(bonus_map_dirs, Path):
            sources.append({"kind": "bonus_map_dir", "path": str(bonus_map_dirs), "mode": "inferred"})
        elif isinstance(bonus_map_dirs, dict):
            for dataset, bonus_map_dir in sorted(bonus_map_dirs.items()):
                sources.append({"kind": "bonus_map_dir", "path": str(bonus_map_dir), "dataset": dataset, "mode": "inferred"})
    return sources


def _has_eval_cache_schema(conn: sqlite3.Connection) -> bool:
    rows = conn.execute(
        """
        SELECT name FROM sqlite_master
        WHERE type = 'table' AND name IN ('run_cells', 'raw_rollouts', 'quantitative_metrics')
        """
    ).fetchall()
    return {str(row["name"]) for row in rows} == {"run_cells", "raw_rollouts", "quantitative_metrics"}


def _open_readonly_eval_cache(db_path: Path) -> sqlite3.Connection | None:
    try:
        conn = connect_readonly(db_path)
    except FileNotFoundError:
        return None
    if _has_eval_cache_schema(conn):
        return conn
    conn.close()
    return None


def _open_readonly_dashboard_build_db(request: DashboardRequest) -> sqlite3.Connection | None:
    build_db_path = _dashboard_build_db_path(request)
    if build_db_path is None:
        return None
    try:
        return connect_readonly(build_db_path)
    except (FileNotFoundError, sqlite3.Error):
        return None


def _sqlite_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _avg(values: Iterable[Any]) -> float | None:
    real = [float(value) for value in values if isinstance(value, int | float) and not isinstance(value, bool)]
    return sum(real) / len(real) if real else None


def _std(values: Iterable[Any]) -> float | None:
    real = [float(value) for value in values if isinstance(value, int | float) and not isinstance(value, bool)]
    if not real:
        return None
    mean = sum(real) / len(real)
    return (sum((value - mean) ** 2 for value in real) / len(real)) ** 0.5


def _rate(values: Iterable[Any]) -> float | None:
    real = [1 if bool(value) else 0 for value in values if value is not None]
    return sum(real) / len(real) if real else None


def _negative(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and value < 0


def _combined_reverse_marker(item: dict[str, Any]) -> bool | None:
    has_order = isinstance(item.get("order_score"), int | float) and not isinstance(item.get("order_score"), bool)
    if not has_order:
        return None
    return _negative(item.get("order_score"))


def _combined_miracle_marker(item: dict[str, Any]) -> bool | None:
    if item.get("miracle_step") is None:
        return None
    return bool(item.get("miracle_step"))


def _number(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _record_tool_call_count(record: dict[str, Any]) -> int:
    total = 0
    for trace in _as_sequence(record.get("p2a_step_traces")):
        if isinstance(trace, dict):
            total += len(_as_sequence(trace.get("tool_calls")))
    if total:
        return total
    for message in _as_sequence(record.get("messages")):
        if isinstance(message, dict):
            total += len(_as_sequence(message.get("tool_calls")))
    return total


def _sum_int(items: Iterable[dict[str, Any]], key: str) -> int:
    total = 0
    for item in items:
        value = item.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            total += int(value)
    return total


def _path_projection(detail: dict[str, Any]) -> dict[str, Any]:
    _sync_path_aliases(detail)
    projection = detail.get("path_projection") or detail.get("chain_projection") or {}
    return projection if isinstance(projection, dict) else {}


def _path_nodes(detail: dict[str, Any]) -> list[Any]:
    projection = _path_projection(detail)
    nodes = projection.get("path_nodes", projection.get("chain_nodes", []))
    return nodes if isinstance(nodes, list) else []


def _path_context_nodes(detail: dict[str, Any]) -> list[Any]:
    projection = _path_projection(detail)
    context_nodes = projection.get("context_nodes") or []
    return context_nodes if isinstance(context_nodes, list) else []


def _path_edges(detail: dict[str, Any]) -> list[Any]:
    projection = _path_projection(detail)
    edges = projection.get("path_edges", projection.get("chain_edges", []))
    return edges if isinstance(edges, list) else []


def _path_value(detail: dict[str, Any], path_key: str, legacy_key: str, default: Any = None) -> Any:
    _sync_path_aliases(detail)
    return detail.get(path_key, detail.get(legacy_key, default))


def _path_node_precision(detail: dict[str, Any]) -> float | None:
    path_nodes = _path_nodes(detail)
    context_nodes = _path_context_nodes(detail)
    hit_chain = sum(1 for node in path_nodes if isinstance(node, dict) and node.get("hit"))
    hit_context = sum(1 for node in context_nodes if isinstance(node, dict) and node.get("hit"))
    denom = hit_chain + hit_context
    return (hit_chain / denom) if denom else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    denom = precision + recall
    return 2 * precision * recall / denom if denom else 0.0


def _path_node_f1(detail: dict[str, Any]) -> float | None:
    return _f1(_path_node_precision(detail), _number(_path_value(detail, "path_node_recall", "chain_node_recall")))


def _distribution(items: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for item in items:
        value = item.get(key)
        if value is not None:
            counts[str(value)] += 1
    return dict(sorted(counts.items()))


def _detail_case_type(detail: dict[str, Any]) -> str:
    return canonical_detail_case_type(detail)


def _case_filter_bucket(detail: dict[str, Any]) -> str:
    case_type = _detail_case_type(detail)
    if _path_value(detail, "path_evaluable", "chain_evaluable") is True and case_type in CASE_FILTER_BUCKETS:
        return case_type
    return "others"


def _is_path_metric_detail(detail: dict[str, Any]) -> bool:
    return _path_value(detail, "path_evaluable", "chain_evaluable") is True and _detail_case_type(detail) in PATH_METRIC_CASE_TYPES


def _is_dynamic_traceable_detail(detail: dict[str, Any]) -> bool:
    """Compatibility wrapper for old code paths; prefer _is_path_metric_detail."""
    return _is_path_metric_detail(detail)


def _has_dual_symptom_root(detail: dict[str, Any]) -> bool:
    projection = _path_projection(detail)
    anchors = set(projection.get("anchors") or [])
    roots = set(projection.get("roots") or [])
    return bool(anchors & roots)


def _has_path_edges(detail: dict[str, Any]) -> bool:
    return bool(_path_edges(detail))


def _is_order_metric_detail(detail: dict[str, Any]) -> bool:
    return _is_path_metric_detail(detail) and _detail_case_type(detail) == LATENT_CASE and _has_path_edges(detail)


def _instance_key(detail: dict[str, Any]) -> str:
    return str(detail.get("instance_id") or f"record-{detail.get('record_index', 0)}")


def _unique_dataset_details(details: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_dataset: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for detail in details:
        dataset = _dataset_name(detail)
        instance = _instance_key(detail)
        current = by_dataset[dataset].get(instance)
        if current is None:
            by_dataset[dataset][instance] = detail
            continue
        current_score = int(bool(current.get("has_bonus_map"))) + int(bool(_path_value(current, "path_evaluable", "chain_evaluable")))
        new_score = int(bool(detail.get("has_bonus_map"))) + int(bool(_path_value(detail, "path_evaluable", "chain_evaluable")))
        if new_score > current_score:
            by_dataset[dataset][instance] = detail
    return {dataset: list(items.values()) for dataset, items in sorted(by_dataset.items())}


def _dataset_distributions(details: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for dataset, items in _unique_dataset_details(details).items():
        case_types: dict[str, int] = defaultdict(int)
        not_path: dict[str, int] = defaultdict(int)
        availability = {
            "with_bonus_map": 0,
            "with_call_graph": 0,
            "path_evaluable": 0,
            "not_path_evaluable": 0,
            "chain_evaluable": 0,
            "not_chain_evaluable": 0,
        }
        for item in items:
            case_types[_detail_case_type(item) or "missing_bonus_map"] += 1
            if item.get("has_bonus_map"):
                availability["with_bonus_map"] += 1
            if item.get("has_call_graph"):
                availability["with_call_graph"] += 1
            if _path_value(item, "path_evaluable", "chain_evaluable"):
                availability["path_evaluable"] += 1
                availability["chain_evaluable"] += 1
            else:
                availability["not_path_evaluable"] += 1
                availability["not_chain_evaluable"] += 1
                not_path[str(_path_value(item, "not_path_evaluable_reason", "not_chain_evaluable_reason", "unknown") or "unknown")] += 1
        out[dataset] = {
            "dataset": dataset,
            "n_instances": len(items),
            "distributions": {
                "case_types": dict(sorted(case_types.items())),
                "not_path_evaluable_reasons": dict(sorted(not_path.items())),
                "not_chain_evaluable_reasons": dict(sorted(not_path.items())),
                "availability": availability,
            },
        }
    return out


def _path_pattern_distribution(details: Iterable[dict[str, Any]], keys: Iterable[str] = PATH_PATTERN_KEYS) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for detail in details:
        patterns = _path_value(detail, "path_pattern_flags", "chain_bad_patterns")
        if not isinstance(patterns, dict):
            continue
        for key in keys:
            if patterns.get(key):
                counts[key] += 1
    return dict(sorted(counts.items()))


AVG_AT_METRIC_KEYS = (
    "resolved_rate",
    "reward_rate",
    "p2a_read_rate",
    "call_graph_hit_rate",
    "ground_truth_hit_rate",
    "near_hit_rate",
    "avg_min_distance",
    "avg_read_precision",
    "avg_node_recall",
    "avg_hit_f1",
    "path_coverage",
    "chain_graph_coverage",
    "path_hit_rate",
    "chain_hit_rate",
    "anchor_hit_rate",
    "root_hit_rate",
    "avg_path_node_recall",
    "avg_chain_node_recall",
    "avg_path_node_precision",
    "avg_chain_node_precision",
    "avg_path_node_f1",
    "avg_chain_node_f1",
    "avg_path_read_precision",
    "avg_chain_read_precision",
    "avg_first_anchor_step",
    "avg_first_root_step",
    "avg_steps_anchor_to_root",
    "anchor_before_root_rate",
    "avg_order_score",
    "reverse_order_rate",
    "miracle_rate",
    "avg_miracle_severity",
    "avg_block_order_score",
    "block_reverse_order_rate",
    "block_miracle_rate",
    "avg_block_efficiency",
    "avg_blocks_per_trace",
    "block_achieve_rate",
    "block_waste_rate",
    "block_loop_rate",
    "achieving_block_step_share",
    "wasted_block_step_share",
    "loop_block_step_share",
    "loop_trace_rate",
    "error_spiral_rate",
    "avg_turns",
    "avg_tool_calls",
    "avg_wall_time",
    "avg_input_tokens",
    "avg_output_tokens",
    "avg_reasoning_tokens",
    "cache_hit_rate",
    "cache_write_rate",
)


def _ratio(item: dict[str, Any], numerator: str, denominator: str) -> float | None:
    num = _number(item.get(numerator))
    den = _number(item.get(denominator))
    return (num / den) if num is not None and den else None


def _bool_number(value: Any) -> int | None:
    return None if value is None else 1 if bool(value) else 0


def _detail_metric_values(item: dict[str, Any]) -> dict[str, Any]:
    path_metric = _is_path_metric_detail(item)
    order_metric = _is_order_metric_detail(item)
    order_score = item.get("order_score") if order_metric and item.get("order_defined") is True else None
    block_order_score = item.get("block_order_score") if order_metric and item.get("block_order_defined") is True else None
    cache_hit = float(item.get("cache_hit_tokens") or 0)
    cache_write = float(item.get("cache_write_tokens") or 0)
    input_tokens = float(item.get("input_tokens") or 0)
    return {
        "resolved_rate": _bool_number(item.get("resolved")),
        "reward_rate": item.get("reward"),
        "p2a_read_rate": _bool_number((item.get("n_reads") or 0) > 0),
        "call_graph_hit_rate": _bool_number(item.get("hit_call_graph")),
        "ground_truth_hit_rate": _bool_number(item.get("hit_ground_truth")),
        "near_hit_rate": _bool_number(item.get("hit_near")),
        "avg_min_distance": item.get("min_distance"),
        "avg_read_precision": item.get("hit_precision") if path_metric else None,
        "avg_node_recall": item.get("hit_recall") if path_metric else None,
        "avg_hit_f1": item.get("hit_f1") if path_metric else None,
        "path_coverage": _bool_number(_path_value(item, "path_covered", "chain_graph_covered")) if path_metric else None,
        "chain_graph_coverage": _bool_number(_path_value(item, "path_covered", "chain_graph_covered")) if path_metric else None,
        "path_hit_rate": _bool_number(_path_value(item, "path_hit", "chain_hit")) if path_metric else None,
        "chain_hit_rate": _bool_number(_path_value(item, "path_hit", "chain_hit")) if path_metric else None,
        "anchor_hit_rate": _bool_number(item.get("anchor_hit")) if path_metric else None,
        "root_hit_rate": _bool_number(item.get("root_hit")) if path_metric else None,
        "avg_path_node_recall": _path_value(item, "path_node_recall", "chain_node_recall") if path_metric else None,
        "avg_chain_node_recall": _path_value(item, "path_node_recall", "chain_node_recall") if path_metric else None,
        "avg_path_node_precision": _path_node_precision(item) if path_metric else None,
        "avg_chain_node_precision": _path_node_precision(item) if path_metric else None,
        "avg_path_node_f1": _path_node_f1(item) if path_metric else None,
        "avg_chain_node_f1": _path_node_f1(item) if path_metric else None,
        "avg_path_read_precision": _path_value(item, "path_read_precision", "chain_read_precision") if path_metric else None,
        "avg_chain_read_precision": _path_value(item, "path_read_precision", "chain_read_precision") if path_metric else None,
        "avg_first_anchor_step": item.get("first_anchor_step") if path_metric else None,
        "avg_first_root_step": item.get("first_root_step") if path_metric else None,
        "avg_steps_anchor_to_root": item.get("steps_anchor_to_root") if path_metric else None,
        "anchor_before_root_rate": _bool_number(item.get("anchor_before_root")) if path_metric else None,
        "avg_order_score": order_score,
        "reverse_order_rate": _bool_number(_negative(order_score)) if order_score is not None else None,
        "miracle_rate": _bool_number(_combined_miracle_marker(item)) if order_metric else None,
        "avg_miracle_severity": item.get("miracle_severity") if order_metric else None,
        "avg_block_order_score": block_order_score,
        "block_reverse_order_rate": _bool_number(_negative(block_order_score)) if block_order_score is not None else None,
        "block_miracle_rate": _bool_number(item.get("block_miracle_step")) if order_metric else None,
        "avg_block_efficiency": item.get("block_efficiency") if path_metric else None,
        "avg_blocks_per_trace": item.get("n_blocks") if path_metric else None,
        "block_achieve_rate": _ratio(item, "n_achieving_blocks", "n_scored_read_blocks") if path_metric else None,
        "block_waste_rate": _ratio(item, "n_wasted_blocks", "n_scored_read_blocks") if path_metric else None,
        "block_loop_rate": _ratio(item, "n_loop_blocks", "n_blocks") if path_metric else None,
        "achieving_block_step_share": _ratio(item, "n_achieving_block_steps", "n_scored_read_block_steps") if path_metric else None,
        "wasted_block_step_share": _ratio(item, "n_wasted_block_steps", "n_scored_read_block_steps") if path_metric else None,
        "loop_block_step_share": _ratio(item, "n_loop_block_steps", "n_block_steps") if path_metric else None,
        "loop_trace_rate": _bool_number((item.get("bad_patterns") or {}).get("has_loop")),
        "error_spiral_rate": _bool_number((item.get("bad_patterns") or {}).get("error_spiral")),
        "avg_turns": item.get("turns"),
        "avg_tool_calls": item.get("tool_calls"),
        "avg_wall_time": item.get("wall_time"),
        "avg_input_tokens": item.get("input_tokens"),
        "avg_output_tokens": item.get("output_tokens"),
        "avg_reasoning_tokens": item.get("reasoning_tokens"),
        "cache_hit_rate": (cache_hit / (input_tokens + cache_hit)) if cache_hit and (input_tokens + cache_hit) else None,
        "cache_write_rate": (cache_write / (input_tokens + cache_write)) if cache_write and (input_tokens + cache_write) else None,
    }


def _rollout_index(item: dict[str, Any]) -> int:
    value = item.get("rollout_index")
    return int(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0


def _items_by_instance(items: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        out[str(item.get("instance_id") or item.get("uid") or "")].append(item)
    for values in out.values():
        values.sort(key=_rollout_index)
    return out


def _apply_rollout_count_fields(row: dict[str, Any], items: list[dict[str, Any]]) -> None:
    if not items:
        return
    by_instance = _items_by_instance(items)
    success_by_instance = _items_by_instance([item for item in items if not (item.get("error") or item.get("system_error"))])
    row["rollouts_per_instance"] = max(1, max(_rollout_index(item) for item in items) + 1)
    row["target"] = len(by_instance)
    row["target_rollouts"] = len(items)
    row["done"] = len(success_by_instance)
    row["done_rollouts"] = sum(len(values) for values in success_by_instance.values())


def _apply_avg_at_metrics(row: dict[str, Any], items: list[dict[str, Any]]) -> None:
    if not items:
        return
    rollout_n = max(1, max(_rollout_index(item) for item in items) + 1)
    by_instance = _items_by_instance(items)
    pass_at = {}
    for k in range(1, rollout_n + 1):
        pass_at[str(k)] = _rate(any(item.get("resolved") for item in values[:k]) for values in by_instance.values() if values[:k])
    row["rollouts_per_instance"] = rollout_n
    row["pass_at"] = pass_at
    row["pass_at_n"] = pass_at.get(str(rollout_n))
    row["avg_at"] = {}
    row["avg_at_std"] = {}
    row["std_scale"] = 1
    for k in range(1, rollout_n + 1):
        row["avg_at"][str(k)] = {}
        row["avg_at_std"][str(k)] = {}
        for key in AVG_AT_METRIC_KEYS:
            instance_values = []
            for values in by_instance.values():
                selected = values[:k]
                instance_values.append(_avg(_detail_metric_values(item).get(key) for item in selected))
            row["avg_at"][str(k)][key] = _avg(instance_values)
            row["avg_at_std"][str(k)][key] = _std(instance_values)
    row["avg_at_n"] = row["avg_at"].get(str(rollout_n), {})
    row["avg_at_n_std"] = row["avg_at_std"].get(str(rollout_n), {})
    for key, mean in row["avg_at_n"].items():
        row[key] = mean
        std = row["avg_at_n_std"].get(key)
        if std is not None:
            row[f"{key}_std"] = std


def _detail_model_metrics(details: list[dict[str, Any]], *, include_avg_at: bool = True) -> list[dict[str, Any]]:
    for detail in details:
        _sync_path_aliases(detail)
    groups: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for detail in details:
        key = (
            str(detail.get("experiment_key") or _experiment_key(detail)),
            str(detail.get("source_kind") or "offline_artifact"),
            str(detail.get("experiment_id") or "adhoc"),
            str(detail.get("provider_source") or "unknown-provider"),
            str(detail.get("dataset") or detail.get("data_source") or "unknown-dataset"),
            str(detail.get("model_label") or detail.get("model_api_name") or "unknown-model"),
        )
        groups[key].append(detail)

    rows = []
    for (experiment_key, source_kind, experiment_id, provider_source, dataset, model_label), items in sorted(groups.items()):
        error_count = sum(1 for item in items if item.get("error") or item.get("system_error"))
        success_items = [item for item in items if not (item.get("error") or item.get("system_error"))]
        path_metric_items = [item for item in items if _is_path_metric_detail(item)]
        order_metric_items = [item for item in items if _is_order_metric_detail(item)]
        order_items = [item for item in order_metric_items if item.get("order_defined") is True]
        block_order_items = [item for item in order_metric_items if item.get("block_order_defined") is True]
        scored_blocks = _sum_int(path_metric_items, "n_scored_read_blocks")
        total_blocks = _sum_int(path_metric_items, "n_blocks")
        scored_block_steps = _sum_int(path_metric_items, "n_scored_read_block_steps")
        block_steps = _sum_int(path_metric_items, "n_block_steps")
        cache_hit = sum(float(item.get("cache_hit_tokens") or 0) for item in items)
        cache_write = sum(float(item.get("cache_write_tokens") or 0) for item in items)
        input_tokens = sum(float(item.get("input_tokens") or 0) for item in items)
        row = {
            "experiment_key": experiment_key,
            "eval_cell_key": experiment_key,
            "source_kind": source_kind,
            "experiment_id": experiment_id,
            "provider_source": provider_source,
            "dataset": dataset,
            "model_api_name": str(items[0].get("model_api_name") or model_label),
            "model_label": model_label,
            "run_step": items[0].get("run_step"),
            "target": len(items),
            "target_rollouts": len(items),
            "done": len(success_items),
            "done_rollouts": len(success_items),
            "errors": error_count,
            "pending": 0,
            "resolved_rate": _rate(item.get("resolved") for item in items),
            "reward_rate": _avg(item.get("reward") for item in items),
            "p2a_read_rate": _rate((item.get("n_reads") or 0) > 0 for item in items),
            "call_graph_hit_rate": _rate(item.get("hit_call_graph") for item in items),
            "ground_truth_hit_rate": _rate(item.get("hit_ground_truth") for item in items),
            "near_hit_rate": _rate(item.get("hit_near") for item in items),
            "avg_min_distance": _avg(item.get("min_distance") for item in items),
            "avg_read_precision": _avg(item.get("hit_precision") for item in path_metric_items),
            "avg_node_recall": _avg(item.get("hit_recall") for item in path_metric_items),
            "avg_hit_f1": _avg(item.get("hit_f1") for item in path_metric_items),
            "path_coverage": _rate(_path_value(item, "path_covered", "chain_graph_covered") for item in path_metric_items),
            "chain_graph_coverage": _rate(_path_value(item, "path_covered", "chain_graph_covered") for item in path_metric_items),
            "path_hit_rate": _rate(_path_value(item, "path_hit", "chain_hit") for item in path_metric_items),
            "chain_hit_rate": _rate(_path_value(item, "path_hit", "chain_hit") for item in path_metric_items),
            "anchor_hit_rate": _rate(item.get("anchor_hit") for item in path_metric_items),
            "root_hit_rate": _rate(item.get("root_hit") for item in path_metric_items),
            "avg_path_node_recall": _avg(_path_value(item, "path_node_recall", "chain_node_recall") for item in path_metric_items),
            "avg_chain_node_recall": _avg(_path_value(item, "path_node_recall", "chain_node_recall") for item in path_metric_items),
            "avg_path_node_precision": _avg(_path_node_precision(item) for item in path_metric_items),
            "avg_chain_node_precision": _avg(_path_node_precision(item) for item in path_metric_items),
            "avg_path_node_f1": _avg(_path_node_f1(item) for item in path_metric_items),
            "avg_chain_node_f1": _avg(_path_node_f1(item) for item in path_metric_items),
            "avg_path_read_precision": _avg(
                _path_value(item, "path_read_precision", "chain_read_precision") for item in path_metric_items
            ),
            "avg_chain_read_precision": _avg(
                _path_value(item, "path_read_precision", "chain_read_precision") for item in path_metric_items
            ),
            "avg_first_anchor_step": _avg(item.get("first_anchor_step") for item in path_metric_items),
            "avg_first_root_step": _avg(item.get("first_root_step") for item in path_metric_items),
            "avg_steps_anchor_to_root": _avg(item.get("steps_anchor_to_root") for item in path_metric_items),
            "anchor_before_root_rate": _rate(item.get("anchor_before_root") for item in path_metric_items),
            "avg_order_score": _avg(item.get("order_score") for item in order_items),
            "reverse_order_rate": _rate(_combined_reverse_marker(item) for item in order_metric_items),
            "miracle_rate": _rate(_combined_miracle_marker(item) for item in order_metric_items),
            "avg_miracle_severity": _avg(item.get("miracle_severity") for item in order_metric_items),
            "avg_block_order_score": _avg(item.get("block_order_score") for item in block_order_items),
            "block_reverse_order_rate": _rate(
                _negative(item.get("block_order_score"))
                if isinstance(item.get("block_order_score"), int | float) and not isinstance(item.get("block_order_score"), bool)
                else None
                for item in order_metric_items
            ),
            "block_miracle_rate": _rate(
                None if item.get("block_miracle_step") is None else bool(item.get("block_miracle_step"))
                for item in order_metric_items
            ),
            "avg_block_efficiency": _avg(item.get("block_efficiency") for item in path_metric_items),
            "avg_blocks_per_trace": (total_blocks / len(path_metric_items)) if path_metric_items else None,
            "block_achieve_rate": (_sum_int(path_metric_items, "n_achieving_blocks") / scored_blocks) if scored_blocks else None,
            "block_waste_rate": (_sum_int(path_metric_items, "n_wasted_blocks") / scored_blocks) if scored_blocks else None,
            "block_loop_rate": (_sum_int(path_metric_items, "n_loop_blocks") / total_blocks) if total_blocks else None,
            "achieving_block_step_share": (_sum_int(path_metric_items, "n_achieving_block_steps") / scored_block_steps)
            if scored_block_steps
            else None,
            "wasted_block_step_share": (_sum_int(path_metric_items, "n_wasted_block_steps") / scored_block_steps)
            if scored_block_steps
            else None,
            "loop_block_step_share": (_sum_int(path_metric_items, "n_loop_block_steps") / block_steps) if block_steps else None,
            "loop_trace_rate": _rate((item.get("bad_patterns") or {}).get("has_loop") for item in items),
            "error_spiral_rate": _rate((item.get("bad_patterns") or {}).get("error_spiral") for item in items),
            "avg_turns": _avg(item.get("turns") for item in items),
            "avg_tool_calls": _avg(item.get("tool_calls") for item in items),
            "avg_wall_time": _avg(item.get("wall_time") for item in items),
            "avg_input_tokens": _avg(item.get("input_tokens") for item in items),
            "avg_output_tokens": _avg(item.get("output_tokens") for item in items),
            "avg_reasoning_tokens": _avg(item.get("reasoning_tokens") for item in items),
            "cache_hit_rate": (cache_hit / (input_tokens + cache_hit)) if cache_hit and (input_tokens + cache_hit) else None,
            "cache_write_rate": (cache_write / (input_tokens + cache_write)) if cache_write and (input_tokens + cache_write) else None,
            "total_cache_write_tokens": cache_write if cache_write else None,
            "total_cost": sum(float(item.get("cost") or 0) for item in items) if items else None,
            "not_chain_evaluable_reasons": _distribution(
                [item for item in items if not _path_value(item, "path_evaluable", "chain_evaluable")],
                "not_chain_evaluable_reason",
            ),
            "not_path_evaluable_reasons": _distribution(
                [item for item in items if not _path_value(item, "path_evaluable", "chain_evaluable")],
                "not_path_evaluable_reason",
            ),
            "path_pattern_flags": _path_pattern_distribution(items, PATH_PATTERN_KEYS),
            "chain_bad_patterns": _path_pattern_distribution(items, CHAIN_BAD_PATTERN_KEYS),
        }
        _apply_rollout_count_fields(row, items)
        if include_avg_at:
            _apply_avg_at_metrics(row, items)
        rows.append(row)
    return rows


def _case_filter_model_metrics(details: list[dict[str, Any]], *, include_avg_at: bool = True) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for mask in range(1, 1 << len(CASE_FILTER_BUCKETS)):
        buckets = tuple(bucket for index, bucket in enumerate(CASE_FILTER_BUCKETS) if mask & (1 << index))
        key = ",".join(buckets)
        filtered = [detail for detail in details if _case_filter_bucket(detail) in buckets]
        out[key] = _detail_model_metrics(filtered, include_avg_at=include_avg_at) if filtered else []
    return out


def _normalize_model_row(row: dict[str, Any]) -> dict[str, Any]:
    source_kind = _source_kind(
        provider_source=row.get("provider_source"),
        schema_version=None,
        run_step=row.get("run_step"),
    )
    normalized = {**row, "source_kind": row.get("source_kind") or source_kind}
    for legacy_key, path_key in (
        ("avg_chain_node_recall", "avg_path_node_recall"),
        ("avg_chain_node_precision", "avg_path_node_precision"),
        ("avg_chain_node_f1", "avg_path_node_f1"),
        ("avg_chain_read_precision", "avg_path_read_precision"),
        ("chain_graph_coverage", "path_coverage"),
        ("chain_hit_rate", "path_hit_rate"),
    ):
        if path_key not in normalized and legacy_key in normalized:
            normalized[path_key] = normalized[legacy_key]
        elif legacy_key not in normalized and path_key in normalized:
            normalized[legacy_key] = normalized[path_key]
    normalized["experiment_key"] = row.get("experiment_key") or _experiment_key(normalized)
    normalized["eval_cell_key"] = row.get("eval_cell_key") or normalized["experiment_key"]
    return normalized


def _merge_model_metrics(base_rows: list[dict[str, Any]], detail_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = {_normalize_model_row(row)["eval_cell_key"]: _normalize_model_row(row) for row in base_rows}
    for row in detail_rows:
        current = merged.get(row["eval_cell_key"], {})
        merged_row = dict(row)
        for key, value in current.items():
            if merged_row.get(key) is None:
                merged_row[key] = value
        for key in ("target", "target_rollouts", "done", "done_rollouts", "errors", "pending", "selected_scope", "rollouts_per_instance"):
            if current.get(key) is not None:
                merged_row[key] = current[key]
        for key in ("detail_cache_ready_rollouts", "detail_cache_pending_rollouts"):
            if current.get(key) is not None:
                merged_row[key] = current[key]
        merged[row["eval_cell_key"]] = _normalize_model_row(merged_row)
    return sorted(merged.values(), key=lambda item: (str(item.get("experiment_id")), str(item.get("model_label"))))


def _base_model_metrics_from_raw(conn: sqlite3.Connection, request: DashboardRequest) -> list[dict[str, Any]]:
    rows = aggregate_model_metrics(
        conn,
        experiment_id=request.experiment_id,
        provider_source=request.provider_source,
        dataset=request.dataset,
        model_api_name=request.model_api_name,
        model_label=request.model_label,
        include_detail_metrics=False,
        include_raw_trace_fallback=False,
    )
    return [_normalize_model_row(row) for row in rows]


def _materialized_model_metrics_rows(
    conn: sqlite3.Connection,
    request: DashboardRequest,
    *,
    build_conn: sqlite3.Connection | None,
    include_avg_at: bool = True,
) -> list[dict[str, Any]]:
    base_rows = _base_model_metrics_from_raw(conn, request)
    cache_request = replace(request, defer_db_scoring=True, include_db_raw_details=False)
    cached_details = _load_db_cached_details(conn, cache_request, build_conn=build_conn)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"ready": 0, "pending": 0})
    valid_details: list[dict[str, Any]] = []
    for detail in cached_details:
        if detail.get("cell_status") not in {DONE_STATUS, ERROR_STATUS}:
            continue
        key = _eval_cell_key(detail)
        if detail.get("dashboard_cache_pending"):
            counts[key]["pending"] += 1
        else:
            counts[key]["ready"] += 1
            valid_details.append(detail)

    complete_details = [
        detail
        for detail in valid_details
        if counts[_eval_cell_key(detail)]["pending"] == 0
    ]
    detail_rows = _detail_model_metrics(complete_details, include_avg_at=include_avg_at) if complete_details else []
    merged = _merge_model_metrics(base_rows, detail_rows)
    for row in merged:
        key = _eval_cell_key(row)
        if key in counts:
            row["detail_cache_ready_rollouts"] = counts[key]["ready"]
            row["detail_cache_pending_rollouts"] = counts[key]["pending"]
    return merged


def _load_dashboard_build_model_metrics(
    build_conn: sqlite3.Connection,
    request: DashboardRequest,
) -> list[dict[str, Any]]:
    if request.db_path is None:
        return []
    where_sql, params = _build_db_where(
        experiment_id=request.experiment_id,
        provider_source=request.provider_source,
        dataset=request.dataset,
        model_api_name=request.model_api_name,
        model_label=request.model_label,
    )
    prefix = "WHERE" if not where_sql else f"{where_sql} AND"
    try:
        rows = build_conn.execute(
            f"""
            SELECT metrics_json, updated_at
            FROM dashboard_model_metrics
            {prefix} raw_db_path = ?
            ORDER BY model_label
            """,
            [*params, raw_db_identity(request.db_path)],
        ).fetchall()
    except sqlite3.Error:
        return []
    out = []
    for row in rows:
        payload = _safe_json_loads(row["metrics_json"], {})
        if isinstance(payload, dict) and payload:
            payload["_build_updated_at"] = row["updated_at"]
            out.append(_normalize_model_row(payload))
    return out


def _build_cache_counts_by_key(
    conn: sqlite3.Connection,
    build_conn: sqlite3.Connection,
    request: DashboardRequest,
) -> dict[str, dict[str, int]]:
    cache_request = replace(request, defer_db_scoring=True, include_db_raw_details=False)
    cached_details = _load_db_cached_details(conn, cache_request, build_conn=build_conn)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"ready": 0, "pending": 0})
    for detail in cached_details:
        if detail.get("cell_status") not in {DONE_STATUS, ERROR_STATUS}:
            continue
        key = _eval_cell_key(detail)
        if detail.get("dashboard_cache_pending"):
            counts[key]["pending"] += 1
        else:
            counts[key]["ready"] += 1
    return counts


def _migrate_legacy_raw_dashboard_cache(conn: sqlite3.Connection, request: DashboardRequest) -> int:
    build_db_path = _dashboard_build_db_path(request)
    if request.db_path is None or build_db_path is None:
        return 0
    metric_columns = _sqlite_columns(conn, "quantitative_metrics")
    if "fingerprint" not in metric_columns or "metrics_json" not in metric_columns:
        return 0
    where_sql, params = _db_where(
        experiment_id=request.experiment_id,
        provider_source=request.provider_source,
        dataset=request.dataset,
        model_api_name=request.model_api_name,
        model_label=request.model_label,
    )
    cell_columns = _sqlite_columns(conn, "run_cells")
    raw_columns = _sqlite_columns(conn, "raw_rollouts")
    rollout_index_sql = "c.rollout_index" if "rollout_index" in cell_columns else "0 AS rollout_index"
    rollout_id_sql = "c.rollout_id" if "rollout_id" in cell_columns else "NULL AS rollout_id"
    raw_hash_sql = "r.rollout_sha256" if "rollout_sha256" in raw_columns else "NULL"
    raw_join_sql = "LEFT JOIN raw_rollouts r ON r.cell_id = c.id" if raw_columns else ""
    legacy_where = f"{where_sql} AND" if where_sql else "WHERE"
    rows = conn.execute(
        f"""
        SELECT
          c.id AS cell_id,
          c.experiment_id,
          c.provider_source,
          c.dataset,
          c.model_api_name,
          c.model_label,
          c.instance_id,
          {rollout_index_sql},
          {rollout_id_sql},
          c.status,
          c.run_id,
          q.fingerprint,
          q.metrics_json,
          {raw_hash_sql} AS raw_rollout_sha256
        FROM run_cells c
        LEFT JOIN quantitative_metrics q ON q.cell_id = c.id
        {raw_join_sql}
        {legacy_where} q.metrics_json LIKE '%"dashboard_detail_cache"%'
        """,
        params,
    ).fetchall()
    if not rows:
        return 0

    build_conn: sqlite3.Connection | None = None
    migrated = 0
    bonus_file_cache: dict[tuple[str, str], Path | None] = {}
    bonus_hash_cache: dict[Path, str | None] = {}
    try:
        for row in rows:
            metrics = _safe_json_loads(row["metrics_json"], {})
            if not isinstance(metrics, dict):
                continue
            metadata = metrics.get(DASHBOARD_DETAIL_CACHE_METADATA_KEY)
            if not isinstance(metadata, dict):
                continue
            record = {
                "cell_id": row["cell_id"],
                "experiment_id": row["experiment_id"],
                "provider_source": row["provider_source"],
                "dataset": row["dataset"],
                "data_source": row["dataset"],
                "model": row["model_label"],
                "model_label": row["model_label"],
                "model_api_name": row["model_api_name"],
                "instance_id": row["instance_id"],
                "run_id": row["run_id"],
                "rollout_index": row["rollout_index"],
                "rollout_id": row["rollout_id"],
                "status": row["status"],
                "raw_rollout_sha256": row["raw_rollout_sha256"] or metadata.get("raw_rollout_sha256"),
            }
            detail = _validated_db_cached_detail(
                row,
                record,
                metrics,
                request,
                bonus_file_cache=bonus_file_cache,
                bonus_hash_cache=bonus_hash_cache,
            )
            if detail is None:
                continue
            raw_hash = metadata.get("raw_rollout_sha256") or row["raw_rollout_sha256"]
            if not raw_hash:
                continue
            if build_conn is None:
                build_conn = ensure_dashboard_build_db(build_db_path)
            write_dashboard_build_detail(
                build_conn,
                raw_db_path=request.db_path,
                raw_cell_id=int(row["cell_id"]),
                experiment_id=str(row["experiment_id"] or ""),
                provider_source=str(row["provider_source"] or ""),
                dataset=str(row["dataset"] or ""),
                model_api_name=str(row["model_api_name"] or ""),
                model_label=str(row["model_label"] or ""),
                instance_id=str(row["instance_id"] or ""),
                rollout_index=int(row["rollout_index"] or 0),
                rollout_id=str(row["rollout_id"]) if row["rollout_id"] is not None else None,
                run_id=str(row["run_id"]) if row["run_id"] is not None else None,
                status=str(row["status"] or DONE_STATUS),
                raw_rollout_sha256=str(raw_hash),
                fingerprint=str(row["fingerprint"]),
                detail=detail,
                cache_metadata={**metadata, "fingerprint": row["fingerprint"]},
            )
            migrated += 1
        if build_conn is not None and migrated:
            for metric_row in _materialized_model_metrics_rows(conn, request, build_conn=build_conn, include_avg_at=True):
                write_dashboard_build_model_metrics(build_conn, raw_db_path=request.db_path, row=metric_row)
            build_conn.commit()
    except (OSError, sqlite3.Error):
        if build_conn is not None:
            try:
                build_conn.rollback()
            except sqlite3.Error:
                pass
    finally:
        if build_conn is not None:
            build_conn.close()
    return migrated


def migrate_dashboard_databases(request: DashboardRequest) -> dict[str, int]:
    if request.db_path is None:
        return {"legacy_build_rows": 0}
    conn: sqlite3.Connection | None = None
    try:
        conn = connect(request.db_path, timeout=30.0)
        init_db(conn)
        conn.commit()
        migrated = _migrate_legacy_raw_dashboard_cache(conn, request)
        return {"legacy_build_rows": migrated}
    finally:
        if conn is not None:
            conn.close()


def _migrated_build_detail_cell_ids(request: DashboardRequest) -> set[int]:
    if request.db_path is None:
        return set()
    build_db_path = _dashboard_build_db_path(request)
    if build_db_path is None or not build_db_path.exists():
        return set()
    where_sql, params = _build_db_where(
        experiment_id=request.experiment_id,
        provider_source=request.provider_source,
        dataset=request.dataset,
        model_api_name=request.model_api_name,
        model_label=request.model_label,
    )
    prefix = "WHERE" if not where_sql else f"{where_sql} AND"
    conn: sqlite3.Connection | None = None
    try:
        conn = connect_readonly(build_db_path, timeout=10.0)
        rows = conn.execute(
            f"""
            SELECT raw_cell_id
            FROM dashboard_rollout_details
            {prefix} raw_db_path = ?
            """,
            [*params, raw_db_identity(request.db_path)],
        ).fetchall()
    except (OSError, sqlite3.Error):
        return set()
    finally:
        if conn is not None:
            conn.close()
    return {int(row["raw_cell_id"]) for row in rows}


def slim_dashboard_raw_db(request: DashboardRequest, *, vacuum: bool = False) -> dict[str, Any]:
    if request.db_path is None:
        return {
            "legacy_metric_rows_slimmed": 0,
            "legacy_metric_rows_skipped_unbuilt": 0,
            "rollout_json_rows_slimmed": 0,
            "metric_bytes_removed": 0,
            "rollout_json_bytes_removed": 0,
            "vacuum": False,
        }
    migrated_cell_ids = _migrated_build_detail_cell_ids(request)
    conn: sqlite3.Connection | None = None
    metric_rows_slimmed = 0
    metric_rows_skipped = 0
    metric_bytes_removed = 0
    rollout_rows_slimmed = 0
    rollout_bytes_removed = 0
    try:
        conn = connect(request.db_path, timeout=60.0)
        init_db(conn)
        where_sql, params = _db_where(
            experiment_id=request.experiment_id,
            provider_source=request.provider_source,
            dataset=request.dataset,
            model_api_name=request.model_api_name,
            model_label=request.model_label,
        )
        metric_rows = conn.execute(
            f"""
            SELECT q.cell_id, q.metrics_json
            FROM quantitative_metrics q
            JOIN run_cells c ON c.id = q.cell_id
            {where_sql}
            """,
            params,
        ).fetchall()
        now = utc_now()
        for row in metric_rows:
            metrics = _safe_json_loads(row["metrics_json"], {})
            if not isinstance(metrics, dict):
                continue
            if "detail" not in metrics and DASHBOARD_DETAIL_CACHE_METADATA_KEY not in metrics:
                continue
            if int(row["cell_id"]) not in migrated_cell_ids:
                metric_rows_skipped += 1
                continue
            before = len(row["metrics_json"] or "")
            metrics.pop("detail", None)
            metrics.pop(DASHBOARD_DETAIL_CACHE_METADATA_KEY, None)
            after_payload = json_dumps(metrics)
            conn.execute(
                """
                UPDATE quantitative_metrics
                SET fingerprint = NULL,
                    metrics_json = ?,
                    updated_at = ?
                WHERE cell_id = ?
                """,
                (after_payload, now, row["cell_id"]),
            )
            metric_rows_slimmed += 1
            metric_bytes_removed += max(0, before - len(after_payload))

        rollout_rows = conn.execute(
            f"""
            SELECT
              c.id AS cell_id,
              c.experiment_id,
              c.provider_source,
              c.dataset,
              c.model_api_name,
              c.model_label,
              c.instance_id,
              c.rollout_index,
              c.rollout_id,
              c.status,
              c.run_id,
              r.rollout_json
            FROM raw_rollouts r
            JOIN run_cells c ON c.id = r.cell_id
            {where_sql}
            """,
            params,
        ).fetchall()
        for row in rollout_rows:
            before_payload = row["rollout_json"] or ""
            record = _safe_json_loads(before_payload, {})
            record = record if isinstance(record, dict) else {}
            slim = slim_rollout_payload(record)
            for key in (
                "experiment_id",
                "provider_source",
                "dataset",
                "model_api_name",
                "model_label",
                "instance_id",
                "rollout_index",
                "rollout_id",
                "status",
                "run_id",
            ):
                if key not in slim and row[key] is not None:
                    slim[key] = row[key]
            after_payload = json_dumps(slim)
            if len(after_payload) >= len(before_payload):
                continue
            conn.execute("UPDATE raw_rollouts SET rollout_json = ? WHERE cell_id = ?", (after_payload, row["cell_id"]))
            rollout_rows_slimmed += 1
            rollout_bytes_removed += len(before_payload) - len(after_payload)
        conn.commit()
        if vacuum:
            conn.execute("VACUUM")
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return {
            "legacy_metric_rows_slimmed": metric_rows_slimmed,
            "legacy_metric_rows_skipped_unbuilt": metric_rows_skipped,
            "rollout_json_rows_slimmed": rollout_rows_slimmed,
            "metric_bytes_removed": metric_bytes_removed,
            "rollout_json_bytes_removed": rollout_bytes_removed,
            "vacuum": bool(vacuum),
        }
    finally:
        if conn is not None:
            conn.close()


def _merge_build_model_rows(
    base_rows: list[dict[str, Any]],
    build_rows: list[dict[str, Any]],
    *,
    build_cache_counts: dict[str, dict[str, int]],
) -> list[dict[str, Any]]:
    normalized_base_rows = [_normalize_model_row(row) for row in base_rows]
    build_by_key = {_normalize_model_row(row)["eval_cell_key"]: _normalize_model_row(row) for row in build_rows}
    merged = []
    for base in normalized_base_rows:
        key = base["eval_cell_key"]
        row = dict(base)
        build_row = build_by_key.get(key)
        completed_rollouts = int(base.get("done_rollouts") or 0) + int(base.get("errors") or 0)
        counts = build_cache_counts.get(key, {"ready": 0, "pending": 0})
        build_coverage = int(counts.get("ready") or 0) + int(counts.get("pending") or 0)
        build_counts_match = bool(build_row) and int(build_row.get("detail_cache_ready_rollouts") or 0) == int(
            counts.get("ready") or 0
        ) and int(build_row.get("detail_cache_pending_rollouts") or 0) == int(counts.get("pending") or 0)
        if build_row and build_counts_match and build_coverage == completed_rollouts:
            row.update({item_key: item_value for item_key, item_value in build_row.items() if not item_key.startswith("_")})
            for progress_key in ("target", "target_rollouts", "done", "done_rollouts", "errors", "pending", "selected_scope", "rollouts_per_instance"):
                row[progress_key] = base.get(progress_key)
        merged.append(_normalize_model_row(row))
    return sorted(merged, key=lambda item: (str(item.get("experiment_id")), str(item.get("model_label"))))


def _build_model_metrics_are_stale(
    base_rows: list[dict[str, Any]],
    build_rows: list[dict[str, Any]],
    build_cache_counts: dict[str, dict[str, int]],
) -> bool:
    build_by_key = {_normalize_model_row(row)["eval_cell_key"]: _normalize_model_row(row) for row in build_rows}
    for base in [_normalize_model_row(row) for row in base_rows]:
        key = base["eval_cell_key"]
        build_row = build_by_key.get(key)
        counts = build_cache_counts.get(key, {"ready": 0, "pending": 0})
        build_coverage = int(counts.get("ready") or 0) + int(counts.get("pending") or 0)
        if not build_row:
            if build_coverage:
                return True
            continue
        completed_rollouts = int(base.get("done_rollouts") or 0) + int(base.get("errors") or 0)
        build_counts_match = int(build_row.get("detail_cache_ready_rollouts") or 0) == int(counts.get("ready") or 0) and int(
            build_row.get("detail_cache_pending_rollouts") or 0
        ) == int(counts.get("pending") or 0)
        if not build_counts_match or build_coverage != completed_rollouts:
            return True
    return False


def _refresh_dashboard_build_model_metrics(request: DashboardRequest) -> None:
    build_db_path = _dashboard_build_db_path(request)
    if request.db_path is None or build_db_path is None:
        return
    raw_conn: sqlite3.Connection | None = None
    build_conn: sqlite3.Connection | None = None
    try:
        raw_conn = connect_readonly(request.db_path, timeout=2.0)
        build_conn = ensure_dashboard_build_db(build_db_path)
        rows = _materialized_model_metrics_rows(raw_conn, request, build_conn=build_conn, include_avg_at=True)
        for row in rows:
            write_dashboard_build_model_metrics(build_conn, raw_db_path=request.db_path, row=row)
        build_conn.commit()
    except (OSError, sqlite3.Error):
        if build_conn is not None:
            try:
                build_conn.rollback()
            except sqlite3.Error:
                pass
    finally:
        if raw_conn is not None:
            raw_conn.close()
        if build_conn is not None:
            build_conn.close()


def validated_cached_model_metrics(
    conn: sqlite3.Connection,
    request: DashboardRequest,
    *,
    include_avg_at: bool = False,
    ) -> list[dict[str, Any]]:
    base_rows = _base_model_metrics_from_raw(conn, request)
    build_db_path = _dashboard_build_db_path(request)
    if build_db_path is None or not build_db_path.exists():
        return base_rows
    build_conn: sqlite3.Connection | None = None
    try:
        build_conn = connect_readonly(build_db_path, timeout=2.0)
        build_rows = _load_dashboard_build_model_metrics(build_conn, request)
        build_cache_counts = _build_cache_counts_by_key(conn, build_conn, request)
        if _build_model_metrics_are_stale(base_rows, build_rows, build_cache_counts):
            return _materialized_model_metrics_rows(conn, request, build_conn=build_conn, include_avg_at=True)
    except (OSError, sqlite3.Error):
        build_rows = []
        build_cache_counts = {}
    finally:
        if build_conn is not None:
            build_conn.close()
    if not build_rows:
        return base_rows
    return _merge_build_model_rows(
        base_rows,
        build_rows,
        build_cache_counts=build_cache_counts,
    )


def _eval_cell_registry(model_metrics: list[dict[str, Any]], details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    detail_counts: dict[str, int] = defaultdict(int)
    trace_counts: dict[str, int] = defaultdict(int)
    detail_error_counts: dict[str, int] = defaultdict(int)
    cache_pending_counts: dict[str, int] = defaultdict(int)
    for detail in details:
        key = _eval_cell_key(detail)
        detail_counts[key] += 1
        if detail.get("error") or detail.get("system_error"):
            detail_error_counts[key] += 1
        if detail.get("dashboard_cache_pending"):
            cache_pending_counts[key] += 1
        if detail.get("raw_available") or detail.get("step_details"):
            trace_counts[key] += 1

    eval_cells = []
    for row in model_metrics:
        key = _eval_cell_key(row)
        detail_count = detail_counts.get(key, 0)
        errors = int(row.get("errors") or detail_error_counts.get(key, 0) or 0)
        target_rollouts = row.get("target_rollouts") if row.get("target_rollouts") is not None else detail_count
        target_instances = row.get("target")
        rollouts_per_instance = row.get("rollouts_per_instance")
        if rollouts_per_instance is None and target_instances:
            rollouts_per_instance = max(1, int(round(float(target_rollouts or 0) / float(target_instances))))
        done_rollouts = row.get("done_rollouts")
        if done_rollouts is None:
            done_rollouts = max(0, detail_count - errors - int(row.get("pending") or 0))
        cache_pending = row.get("detail_cache_pending_rollouts")
        if cache_pending is None:
            cache_pending = cache_pending_counts.get(key, 0)
        eval_cells.append(
            {
                "eval_cell_key": key,
                "experiment_key": key,
                "source_kind": row.get("source_kind"),
                "experiment_id": row.get("experiment_id"),
                "provider_source": row.get("provider_source"),
                "dataset": row.get("dataset"),
                "model_api_name": row.get("model_api_name"),
                "model_label": row.get("model_label"),
                "run_step": row.get("run_step"),
                "target": row.get("target"),
                "target_rollouts": target_rollouts,
                "rollouts_per_instance": rollouts_per_instance,
                "done": row.get("done"),
                "done_rollouts": done_rollouts,
                "errors": errors,
                "pending": row.get("pending"),
                "cache_ready": row.get("detail_cache_ready_rollouts"),
                "cache_pending": cache_pending,
                "selected_scope": row.get("selected_scope"),
                "detail_count": detail_count,
                "trajectory_count": trace_counts.get(key, 0),
                "resolved_rate": row.get("resolved_rate"),
                "root_hit_rate": row.get("root_hit_rate") or row.get("ground_truth_hit_rate"),
                "path_node_recall": row.get("avg_path_node_recall") or row.get("avg_node_recall"),
                "chain_node_recall": row.get("avg_chain_node_recall") or row.get("avg_node_recall"),
                "read_precision": row.get("avg_path_read_precision") or row.get("avg_read_precision"),
            }
        )
    return eval_cells


def _dataset_registry(
    details: list[dict[str, Any]],
    eval_cells: list[dict[str, Any]],
    distributions_by_dataset: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    trajectories: dict[str, int] = defaultdict(int)
    cells: dict[str, set[str]] = defaultdict(set)
    models: dict[str, set[str]] = defaultdict(set)
    source_kinds: dict[str, set[str]] = defaultdict(set)
    target_instances: dict[str, int] = defaultdict(int)
    target_rollouts: dict[str, int] = defaultdict(int)
    for detail in details:
        dataset = _dataset_name(detail)
        trajectories[dataset] += 1
        cells[dataset].add(_eval_cell_key(detail))
        if detail.get("model_label"):
            models[dataset].add(str(detail.get("model_label")))
        if detail.get("source_kind"):
            source_kinds[dataset].add(str(detail.get("source_kind")))
    for cell in eval_cells:
        dataset = _dataset_name(cell)
        cells[dataset].add(_eval_cell_key(cell))
        if cell.get("model_label"):
            models[dataset].add(str(cell.get("model_label")))
        if cell.get("source_kind"):
            source_kinds[dataset].add(str(cell.get("source_kind")))
        target = _number(cell.get("target"))
        if target is not None:
            target_instances[dataset] = max(target_instances[dataset], int(target))
        rollout_total = _number(cell.get("target_rollouts"))
        if rollout_total is not None:
            target_rollouts[dataset] += int(rollout_total)
    dataset_names = sorted(set(distributions_by_dataset) | set(cells) | set(trajectories))
    return [
        {
            "dataset": dataset,
            "n_instances": (distributions_by_dataset.get(dataset) or {}).get("n_instances", 0) or target_instances.get(dataset, 0),
            "n_eval_cells": len(cells.get(dataset, set())),
            "n_trajectories": trajectories.get(dataset, 0) or target_rollouts.get(dataset, 0),
            "models": sorted(models.get(dataset, set())),
            "source_kinds": sorted(source_kinds.get(dataset, set())),
        }
        for dataset in dataset_names
    ]


def _dataset_stats_from_details(details: list[dict[str, Any]], model_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    trajectories: dict[str, int] = defaultdict(int)
    instances: dict[str, set[str]] = defaultdict(set)
    cells: dict[str, set[str]] = defaultdict(set)
    models: dict[str, set[str]] = defaultdict(set)
    source_kinds: dict[str, set[str]] = defaultdict(set)
    for detail in details:
        dataset = _dataset_name(detail)
        trajectories[dataset] += 1
        instances[dataset].add(_instance_key(detail))
        cells[dataset].add(_eval_cell_key(detail))
        if detail.get("model_label"):
            models[dataset].add(str(detail.get("model_label")))
        if detail.get("source_kind"):
            source_kinds[dataset].add(str(detail.get("source_kind")))
    for row in model_metrics:
        dataset = _dataset_name(row)
        cells[dataset].add(_eval_cell_key(row))
        if row.get("model_label"):
            models[dataset].add(str(row.get("model_label")))
        if row.get("source_kind"):
            source_kinds[dataset].add(str(row.get("source_kind")))
    dataset_names = sorted(set(trajectories) | set(instances) | set(cells))
    rows = [
        {
            "dataset": dataset,
            "n_instances": len(instances.get(dataset, set())),
            "n_eval_cells": len(cells.get(dataset, set())),
            "n_trajectories": trajectories.get(dataset, 0),
            "models": sorted(models.get(dataset, set())),
            "source_kinds": sorted(source_kinds.get(dataset, set())),
        }
        for dataset in dataset_names
    ]
    return {
        "datasets": rows,
        "totals": {
            "n_datasets": len(rows),
            "n_instances": sum(row["n_instances"] for row in rows),
            "n_eval_cells": sum(row["n_eval_cells"] for row in rows),
            "n_trajectories": sum(row["n_trajectories"] for row in rows),
        },
    }


def _case_filter_dataset_stats(
    details: list[dict[str, Any]],
    case_filter_model_metrics: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for mask in range(1, 1 << len(CASE_FILTER_BUCKETS)):
        buckets = tuple(bucket for index, bucket in enumerate(CASE_FILTER_BUCKETS) if mask & (1 << index))
        key = ",".join(buckets)
        filtered = [detail for detail in details if _case_filter_bucket(detail) in buckets]
        out[key] = _dataset_stats_from_details(filtered, case_filter_model_metrics.get(key, []))
    return out


def build_dashboard_snapshot(request: DashboardRequest) -> dict[str, Any]:
    raw_records = _load_rollout_paths(request.rollouts)
    details = _load_detail_paths(request.details)
    stored_db_details: list[dict[str, Any]] = []
    runs = _scan_log_dir(request.log_dir)
    raw_records.extend(_load_uni_agent_records(request.log_dir))

    base_model_metrics: list[dict[str, Any]] = []
    if request.db_path:
        conn = _open_readonly_eval_cache(request.db_path)
        if conn is not None:
            build_conn = _open_readonly_dashboard_build_db(request)
            try:
                if request.defer_db_scoring:
                    base_model_metrics = validated_cached_model_metrics(conn, request)
                else:
                    base_model_metrics = aggregate_model_metrics(
                        conn,
                        experiment_id=request.experiment_id,
                        provider_source=request.provider_source,
                        dataset=request.dataset,
                        model_api_name=request.model_api_name,
                        model_label=request.model_label,
                        include_detail_metrics=False,
                        include_raw_trace_fallback=False,
                    )
                if request.defer_db_scoring:
                    if request.include_db_raw_details:
                        stored_db_details.extend(_load_db_cached_details(conn, request, build_conn=build_conn))
                else:
                    db_records, db_details = _load_db_records(conn, request, build_conn=build_conn)
                    raw_records.extend(db_records)
                    stored_db_details.extend(db_details)
                runs.extend(_scan_db_artifact_runs(conn, request))
            finally:
                if build_conn is not None:
                    build_conn.close()
                conn.close()

    if request.defer_db_scoring and request.db_path:
        bonus_map_dirs = {}
    else:
        bonus_map_dirs = _effective_bonus_map_dirs(request, raw_records, [*details, *stored_db_details])
    scored_datasets: set[str] = set()
    if raw_records:
        scored_details, scored_datasets = _score_records_by_bonus_map_dir(
            raw_records,
            request=request,
            bonus_map_dirs=bonus_map_dirs,
            start_index=len(details),
        )
        details.extend(scored_details)
        stored_detail_datasets = {_dataset_name(detail) for detail in stored_db_details}
        raw_fallback_records = [
            record
            for record in _records_for_unscored_datasets(raw_records, scored_datasets)
            if (_record_dataset(record) or "unknown-dataset") not in stored_detail_datasets
        ]
        details.extend(_raw_record_details(raw_fallback_records, request=request, start_index=len(details)))
    details.extend(detail for detail in stored_db_details if _dataset_name(detail) not in scored_datasets)

    if details:
        details = _normalize_details(details)
        if not request.defer_db_scoring:
            _enrich_details_from_dataset_parquet(details, request)
            _enrich_details_from_bonus_map_dirs(details, bonus_map_dirs)
        summary = _finalize_summary(summarize(
            details,
            source=_summary_source(request),
            bonus_map_dir=_bonus_map_summary_dir(bonus_map_dirs),
            tracking_mode=request.tracking_mode,
            near_threshold=request.near_threshold,
            m_max=request.m_max,
        ))
        summary["trends"] = summarize_trends(
            details,
            tracking_mode=request.tracking_mode,
            near_threshold=request.near_threshold,
            m_max=request.m_max,
        )
    else:
        summary = _empty_summary(request)
        summary["trends"] = []

    path_metric_details = [detail for detail in details if _is_path_metric_detail(detail)]
    include_avg_at = not (request.defer_db_scoring and request.db_path)
    if request.defer_db_scoring and request.db_path:
        detail_model_metrics = []
        model_metrics = base_model_metrics
        path_metric_model_metrics = []
        case_filter_model_metrics = {}
    else:
        detail_model_metrics = _detail_model_metrics(details, include_avg_at=include_avg_at)
        model_metrics = _merge_model_metrics(base_model_metrics, detail_model_metrics)
        path_metric_model_metrics = _detail_model_metrics(path_metric_details, include_avg_at=include_avg_at)
        case_filter_model_metrics = _case_filter_model_metrics(details, include_avg_at=include_avg_at)
    case_filter_dataset_stats = _case_filter_dataset_stats(details, case_filter_model_metrics)
    eval_cells = _eval_cell_registry(model_metrics, details)
    distributions_by_dataset = _dataset_distributions(details)
    datasets = _dataset_registry(details, eval_cells, distributions_by_dataset)
    summary["by_dataset"] = {
        dataset: {
            "counts": {"n_instances": payload["n_instances"]},
            "distributions": payload["distributions"],
        }
        for dataset, payload in distributions_by_dataset.items()
    }
    summary["distributions_by_dataset"] = distributions_by_dataset

    deduped_runs = {str(run.get("path") or run.get("run_id")): run for run in runs}
    return {
        "schema_version": DASHBOARD_SCHEMA_VERSION,
        "generated_at": time.time(),
        "filters": {
            "experiment_id": request.experiment_id,
            "provider_source": request.provider_source,
            "dataset": request.dataset,
            "model_api_name": request.model_api_name,
            "model_label": request.model_label,
        },
        "sources": _source_list_with_bonus(request, bonus_map_dirs),
        "datasets": datasets,
        "eval_cells": eval_cells,
        "experiments": eval_cells,
        "summary": summary,
        "model_metrics": model_metrics,
        "path_metric_model_metrics": path_metric_model_metrics,
        "dynamic_traceable_model_metrics": path_metric_model_metrics,
        "case_filter_model_metrics": case_filter_model_metrics,
        "case_filter_dataset_stats": case_filter_dataset_stats,
        "runs": sorted(deduped_runs.values(), key=lambda run: (str(run.get("status")), str(run.get("run_id")))),
        "details": details[: request.detail_limit],
        "detail_count": len(details),
        "path_metric_detail_count": len(path_metric_details),
        "dynamic_traceable_detail_count": len(path_metric_details),
        "raw_record_count": len(raw_records),
    }


def read_dashboard_log(request: DashboardRequest, run_id: str, source: str = "run.log") -> dict[str, Any]:
    runs = _scan_log_dir(request.log_dir)
    if request.db_path:
        conn = _open_readonly_eval_cache(request.db_path)
        if conn is not None:
            try:
                runs.extend(_scan_db_artifact_runs(conn, request))
            finally:
                conn.close()
    return _read_log_from_runs(runs, run_id=run_id, source=source)


def snapshot_to_json(snapshot: dict[str, Any], *, indent: int | None = None) -> str:
    return json.dumps(snapshot, ensure_ascii=False, default=_json_default, indent=indent)
