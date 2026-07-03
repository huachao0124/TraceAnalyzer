"""Batch orchestration for API-backed Uni-Agent/P2A evaluation."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from p2a.api_providers import check_provider_available, normalize_provider_config, provider_source
from p2a.bonus_map_scope import (
    BonusMapInstanceFilter,
    parse_bonus_map_instance_filter,
    select_rows_by_bonus_map_scope,
)
from p2a.dashboard_adapter import refresh_dashboard_model_metrics, write_dashboard_detail_cache_for_record
from p2a.datasets import SUPPORTED_EVAL_DATASETS, canonical_dataset
from p2a.eval_cache import (
    DONE_STATUS,
    ERROR_STATUS,
    completed_rollout_keys,
    ensure_db,
    json_dumps,
    mark_cells_running,
    rollout_record_error,
    sha256_text,
    upsert_experiment,
    upsert_planned_cells,
    upsert_rollout_record,
    utc_now,
)
from p2a.eval_fault_localization import iter_records
from p2a.hf_assets import project_artifacts_dir
from p2a.third_party_eval import _instance_id, _load_rows, is_system_error_kind, parse_limit_arg


SUPPORTED_DATASETS = set(SUPPORTED_EVAL_DATASETS)
REDACT_KEYS = ("api_key", "apikey", "token", "secret", "password", "authorization")
SYSTEM_ERROR_STATUS = "system_error"


@dataclass(frozen=True)
class BatchModel:
    api_name: str
    label: str
    overrides: dict[str, Any]


@dataclass(frozen=True)
class BatchConfig:
    path: Path
    raw: dict[str, Any]
    provider: dict[str, Any]
    dataset_name: str
    dataset_file: Path | None
    experiment_id: str
    stage: str
    limit: int | None
    offset: int
    max_turns: int
    rollouts_per_instance: int
    per_instance_parallelism: int
    run_timeout: str | None
    per_model_concurrency: int
    model_parallelism: int
    db_path: Path
    artifacts_dir: Path
    precompute_maps: bool
    bonus_map_dir: Path | None
    bonus_map_instance_filter: BonusMapInstanceFilter
    models: list[BatchModel]


def _as_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    return value


def _parse_limit(value: Any, *, default: int | None) -> int | None:
    if value is None:
        return default
    if isinstance(value, int):
        if value < 0:
            raise ValueError("experiment.limit must be non-negative, or use 'all'")
        return value
    if isinstance(value, str):
        return parse_limit_arg(value)
    raise ValueError("experiment.limit must be an integer or 'all'")


def _positive_int(value: Any, *, default: int, name: str) -> int:
    if value is None:
        return default
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be >= 1")
    return parsed


def _canonical_dataset(name: str) -> str:
    return canonical_dataset(name)


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in REDACT_KEYS):
                redacted[key] = "<redacted>"
            else:
                redacted[key] = _redact(item)
        return redacted
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def sanitized_config_snapshot(config: BatchConfig, *, scope: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema": "p2a_third_party_batch_v1",
        "source_config": str(config.path),
        "captured_at": utc_now(),
        "config": _redact(config.raw),
        "selected_scope": scope,
    }


def load_batch_config(path: Path) -> BatchConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")

    provider_cfg = normalize_provider_config(_as_mapping(payload.get("provider"), name="provider"))
    dataset_cfg = _as_mapping(payload.get("dataset"), name="dataset")
    experiment_cfg = _as_mapping(payload.get("experiment"), name="experiment")
    storage_cfg = _as_mapping(payload.get("storage"), name="storage")
    filter_cfg = payload.get("bonus_map_instance_filter", experiment_cfg.get("bonus_map_instance_filter"))

    dataset_name = _canonical_dataset(str(dataset_cfg.get("name") or "swebench-hard"))
    dataset_file = Path(dataset_cfg["file"]).expanduser() if dataset_cfg.get("file") else None
    stage = str(experiment_cfg.get("stage") or "smoke")
    if stage not in {"smoke", "full", "both"}:
        raise ValueError("experiment.stage must be smoke, full, or both")

    models_payload = payload.get("models")
    if not isinstance(models_payload, list) or not models_payload:
        raise ValueError("models must be a non-empty list")
    models = []
    for index, item in enumerate(models_payload):
        if not isinstance(item, dict):
            raise ValueError(f"models[{index}] must be a mapping")
        api_name = str(item.get("api_name") or item.get("model") or "").strip()
        if not api_name:
            raise ValueError(f"models[{index}].api_name is required")
        label = str(item.get("label") or api_name).strip()
        overrides = {k: copy.deepcopy(v) for k, v in item.items() if k not in {"api_name", "model", "label"}}
        models.append(BatchModel(api_name=api_name, label=label, overrides=overrides))

    artifacts_dir = _resolve_storage_path(storage_cfg.get("artifacts_dir"), default_relative="third_party")
    return BatchConfig(
        path=path,
        raw=payload,
        provider=provider_cfg,
        dataset_name=dataset_name,
        dataset_file=dataset_file,
        experiment_id=str(experiment_cfg.get("id") or path.stem),
        stage=stage,
        limit=_parse_limit(experiment_cfg.get("limit"), default=500),
        offset=int(experiment_cfg.get("offset") or 0),
        max_turns=int(experiment_cfg.get("max_turns") or 20),
        rollouts_per_instance=_positive_int(
            experiment_cfg.get("rollouts_per_instance", experiment_cfg.get("rollout_n")),
            default=1,
            name="experiment.rollouts_per_instance",
        ),
        per_instance_parallelism=_positive_int(
            experiment_cfg.get("per_instance_parallelism", experiment_cfg.get("rollout_parallelism")),
            default=1,
            name="experiment.per_instance_parallelism",
        ),
        run_timeout=str(experiment_cfg["run_timeout"]) if experiment_cfg.get("run_timeout") else None,
        per_model_concurrency=max(1, int(experiment_cfg.get("per_model_concurrency") or 1)),
        model_parallelism=max(1, int(experiment_cfg.get("model_parallelism") or len(models))),
        db_path=_resolve_storage_path(storage_cfg.get("db"), default_relative="evals/traces.sqlite"),
        artifacts_dir=artifacts_dir,
        precompute_maps=bool(storage_cfg.get("precompute_maps", True)),
        bonus_map_dir=(
            _resolve_storage_path(
                storage_cfg["bonus_map_dir"],
                default_relative=f"eval_bonus_maps/{dataset_name}",
            )
            if storage_cfg.get("bonus_map_dir")
            else None
        ),
        bonus_map_instance_filter=parse_bonus_map_instance_filter(filter_cfg),
        models=models,
    )


def _phase_specs(config: BatchConfig) -> list[tuple[str, int | None]]:
    if config.stage == "smoke":
        return [("smoke", 1)]
    if config.stage == "full":
        return [("full", config.limit)]
    return [("smoke", 1), ("full", config.limit)]


def _duration_seconds(value: str | None) -> float | None:
    if not value or value == "0":
        return None
    stripped = value.strip().lower()
    unit = stripped[-1]
    if unit in {"s", "m", "h"}:
        amount = float(stripped[:-1])
        return amount * {"s": 1, "m": 60, "h": 3600}[unit]
    return float(stripped)


def _safe_slug(value: str) -> str:
    slug = value.replace("/", "_").replace(":", "_").replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in slug)


def _resolve_storage_path(value: Any, *, default_relative: str) -> Path:
    raw = default_relative if value is None else str(value)
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == "data":
        path = Path(*parts[1:]) if len(parts) > 1 else Path(".")
    return project_artifacts_dir() / path


def _run_setup(args: list[str], *, env: dict[str, str]) -> str:
    import subprocess

    proc = subprocess.run(
        ["bash", "scripts/setup.sh", *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout.strip() or f"scripts/setup.sh {' '.join(args)} failed")
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def resolve_data_file(config: BatchConfig, *, env: dict[str, str]) -> Path:
    if config.dataset_file is not None:
        return config.dataset_file
    return Path(_run_setup(["data", config.dataset_name], env=env))


def resolve_bonus_map_dir(config: BatchConfig, data_file: Path, *, env: dict[str, str]) -> Path | None:
    if not config.precompute_maps:
        return config.bonus_map_dir
    if config.bonus_map_dir is not None:
        output_dir = config.bonus_map_dir
    else:
        output_dir = project_artifacts_dir() / "bonus_maps" / config.dataset_name
    setup_env = dict(env)
    if config.bonus_map_instance_filter.active:
        setup_env.setdefault("P2A_SETUP_BONUS_OFFSET", "0")
    elif config.limit is not None:
        setup_env.setdefault("P2A_SETUP_BONUS_LIMIT", str(config.limit))
        setup_env.setdefault("P2A_SETUP_BONUS_OFFSET", str(config.offset))
    else:
        setup_env.setdefault("P2A_SETUP_BONUS_OFFSET", str(config.offset))
    return Path(_run_setup(["maps", config.dataset_name, str(data_file), str(output_dir)], env=setup_env))


def selected_instance_scope(
    data_file: Path,
    *,
    limit: int | None,
    offset: int,
    bonus_map_dir: Path | None,
    scope_filter: BonusMapInstanceFilter,
) -> tuple[list[str], dict[str, Any]]:
    if scope_filter.active and bonus_map_dir is None:
        raise ValueError("bonus_map_instance_filter requires storage.bonus_map_dir or precompute_maps: true")
    rows = _load_rows(data_file)
    scoped = select_rows_by_bonus_map_scope(
        rows,
        bonus_map_dir=bonus_map_dir,
        instance_id=_instance_id,
        scope_filter=scope_filter,
        limit=limit,
        offset=offset,
    )
    instance_ids = []
    for row in scoped.rows:
        instance_id = _instance_id(row)
        if not instance_id:
            raise ValueError(f"selected row at offset {offset} has no instance_id")
        instance_ids.append(instance_id)
    if scope_filter.active and not instance_ids:
        raise ValueError("bonus_map_instance_filter selected zero rows")
    return instance_ids, scoped.metadata


def selected_instance_ids(data_file: Path, *, limit: int | None, offset: int) -> list[str]:
    instance_ids, _scope = selected_instance_scope(
        data_file,
        limit=limit,
        offset=offset,
        bonus_map_dir=None,
        scope_filter=BonusMapInstanceFilter(),
    )
    return instance_ids


def _rollout_jobs(instance_ids: list[str], rollouts_per_instance: int) -> list[tuple[str, int]]:
    return [(instance_id, rollout_index) for instance_id in instance_ids for rollout_index in range(rollouts_per_instance)]


def _model_eval_config(config: BatchConfig, model: BatchModel, run_dir: Path) -> Path:
    payload: dict[str, Any] = {}
    for key in ("agent", "analysis"):
        if key in config.raw:
            payload[key] = copy.deepcopy(config.raw[key])
    if config.bonus_map_instance_filter.active:
        payload["bonus_map_instance_filter"] = config.bonus_map_instance_filter.metadata()
    payload["provider"] = copy.deepcopy(config.provider)

    model_cfg = copy.deepcopy(config.raw.get("model") or {})
    model_cfg.update(copy.deepcopy(model.overrides.get("model", {})))
    for key, value in model.overrides.items():
        if key != "model":
            model_cfg[key] = copy.deepcopy(value)
    model_cfg["model_name"] = model.api_name
    model_cfg["api_name"] = model.api_name
    payload["model"] = model_cfg

    config_path = run_dir / "third_party_eval.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def _base_command(
    *,
    eval_config: Path,
    data_file: Path,
    run_dir: Path,
    missing_jobs: list[tuple[str, int]],
    config: BatchConfig,
    model: BatchModel,
    bonus_map_dir: Path | None,
) -> list[str]:
    command = [
        "bash",
        "scripts/third_party_eval.sh",
        "--config",
        str(eval_config),
        "--data",
        str(data_file),
        "--out",
        str(run_dir / "rollouts.jsonl"),
        "--limit",
        "all",
        "--offset",
        "0",
        "--n-parallel",
        str(config.per_model_concurrency),
        "--rollouts-per-instance",
        str(config.rollouts_per_instance),
        "--per-instance-parallelism",
        str(config.per_instance_parallelism),
        "--max-turns",
        str(config.max_turns),
        "--summary-out",
        str(run_dir / "summary.json"),
        "--details-out",
        str(run_dir / "details.jsonl"),
        "--report-out",
        str(run_dir / "report.md"),
        "--experiment-id",
        config.experiment_id,
        "--dataset-name",
        config.dataset_name,
        "--model-label",
        model.label,
    ]
    if bonus_map_dir is not None:
        command.extend(["--bonus-map-dir", str(bonus_map_dir)])
    for instance_id, rollout_index in missing_jobs:
        command.extend(["--rollout-job", f"{instance_id}:{rollout_index}"])
    return command


async def _run_subprocess(command: list[str], *, env: dict[str, str], timeout_s: float | None) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        stdout, _ = await proc.communicate()
        return 124, stdout.decode("utf-8", errors="replace")
    return proc.returncode or 0, stdout.decode("utf-8", errors="replace")


def _system_error_summary(rollouts_path: Path) -> dict[str, Any] | None:
    if not rollouts_path.exists():
        return None
    records = list(iter_records(rollouts_path))
    if not records:
        return None
    error_records = [record for record in records if record.get("error")]
    if len(error_records) != len(records):
        return None
    system_records = [
        record
        for record in error_records
        if bool(record.get("system_error")) or is_system_error_kind(str(record.get("error_kind") or ""))
    ]
    if len(system_records) != len(records):
        return None
    kinds: dict[str, int] = {}
    stages: dict[str, int] = {}
    for record in system_records:
        kind = str(record.get("error_kind") or "unknown")
        stage = str(record.get("error_stage") or "unknown")
        kinds[kind] = kinds.get(kind, 0) + 1
        stages[stage] = stages.get(stage, 0) + 1
    example = str(system_records[0].get("error") or "")
    return {
        "n_records": len(records),
        "error_kinds": kinds,
        "error_stages": stages,
        "example_error": example[:500],
    }


def _mark_missing_error(
    db_path: Path,
    *,
    config: BatchConfig,
    model: BatchModel,
    rollout_jobs: list[tuple[str, int]],
    error: str,
) -> None:
    with ensure_db(db_path) as conn:
        now = utc_now()
        conn.executemany(
            """
            UPDATE run_cells
            SET status = ?, attempts = attempts + 1, ended_at = ?, error = ?, updated_at = ?
            WHERE experiment_id = ?
              AND provider_source = ?
              AND model_api_name = ?
              AND dataset = ?
              AND instance_id = ?
              AND rollout_index = ?
            """,
            [
                (
                    ERROR_STATUS,
                    now,
                    error[-4000:],
                    now,
                    config.experiment_id,
                    provider_source(config.provider),
                    model.api_name,
                    config.dataset_name,
                    instance_id,
                    rollout_index,
                )
                for instance_id, rollout_index in rollout_jobs
            ],
        )
        conn.commit()


def _remaining_unfinished_jobs(
    db_path: Path,
    *,
    config: BatchConfig,
    model: BatchModel,
    rollout_jobs: list[tuple[str, int]],
) -> list[tuple[str, int]]:
    if not rollout_jobs:
        return []
    source = provider_source(config.provider)
    with ensure_db(db_path) as conn:
        rows = conn.execute(
            """
            SELECT instance_id, rollout_index, status
            FROM run_cells
            WHERE experiment_id = ?
              AND provider_source = ?
              AND model_api_name = ?
              AND dataset = ?
            """,
            (config.experiment_id, source, model.api_name, config.dataset_name),
        ).fetchall()
    status_by_job = {(str(row["instance_id"]), int(row["rollout_index"] or 0)): str(row["status"]) for row in rows}
    terminal = {DONE_STATUS, ERROR_STATUS}
    return [job for job in rollout_jobs if status_by_job.get(job) not in terminal]


def _count_rollout_records(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _record in iter_records(path))


class RolloutIngestor:
    """Single-writer JSONL-to-DB ingest running inside the orchestrator process.

    Model subprocesses persist rollouts to their JSONL artifact only; the
    orchestrator tails those files and is the sole raw-DB/build-DB writer, so
    eval concurrency never multiplies SQLite write-lock contention.
    """

    def __init__(
        self,
        *,
        config: BatchConfig,
        model: BatchModel,
        rollouts_path: Path,
        bonus_map_dir: Path | None,
        db_lock: asyncio.Lock,
        db_executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.rollouts_path = rollouts_path
        self.bonus_map_dir = bonus_map_dir
        self.db_lock = db_lock
        self.db_executor = db_executor
        self.analysis_cfg = dict(config.raw.get("analysis") or {})
        self.source = provider_source(config.provider)
        self.n_ingested = 0
        self.n_skipped = 0
        self.n_detail_cache_failed = 0
        self._offset = 0
        self._metrics_dirty = False
        self._stop_event = asyncio.Event()

    def _read_new_records(self) -> tuple[list[dict[str, Any]], int]:
        """Read complete new JSONL lines; the caller commits the returned offset
        only after those records were ingested, so a failed ingest re-reads them."""
        if not self.rollouts_path.exists():
            return [], self._offset
        records: list[dict[str, Any]] = []
        offset = self._offset
        with self.rollouts_path.open("rb") as handle:
            handle.seek(offset)
            while True:
                line = handle.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    # Incomplete tail line; re-read it once the writer finishes.
                    break
                offset += len(line)
                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                try:
                    record = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
        return records, offset

    def _already_ingested(
        self,
        conn: sqlite3.Connection,
        record: dict[str, Any],
        record_sha: str,
        record_error: str | None,
    ) -> bool:
        instance_id = str(record.get("instance_id") or "")
        if not instance_id:
            return False
        row = conn.execute(
            """
            SELECT c.status, c.run_id, r.rollout_sha256
            FROM run_cells c
            LEFT JOIN raw_rollouts r ON r.cell_id = c.id
            WHERE c.experiment_id = ?
              AND c.provider_source = ?
              AND c.model_api_name = ?
              AND c.dataset = ?
              AND c.instance_id = ?
              AND c.rollout_index = ?
            """,
            (
                self.config.experiment_id,
                self.source,
                self.model.api_name,
                self.config.dataset_name,
                instance_id,
                int(record.get("rollout_index") or 0),
            ),
        ).fetchone()
        if row is None:
            return False
        if str(row["status"]) == DONE_STATUS and row["rollout_sha256"] == record_sha:
            return True
        if str(row["status"]) == ERROR_STATUS and record_error:
            # Error cells keep no raw row/sha, so identify the attempt by its
            # unique run_id: a rerun that errors again must still be ingested
            # to bump attempts and surface the latest error.
            record_run_id = str(record.get("run_id") or "")
            return bool(record_run_id) and str(row["run_id"] or "") == record_run_id
        return False

    def _ingest_records_sync(self, records: list[dict[str, Any]]) -> None:
        # Replays (phase-start recovery reads the file from offset 0) can carry
        # several attempts for the same cell; only the newest one matters, and
        # upserting superseded attempts would inflate the attempts counter.
        latest_by_cell: dict[tuple[str, int], dict[str, Any]] = {}
        for record in records:
            instance_id = str(record.get("instance_id") or "")
            if not instance_id:
                self.n_skipped += 1
                continue
            latest_by_cell[(instance_id, int(record.get("rollout_index") or 0))] = record
        with closing(ensure_db(self.config.db_path)) as conn:
            pending_details: list[tuple[int, dict[str, Any], int]] = []
            for record in latest_by_cell.values():
                record_sha = sha256_text(json_dumps(record))
                record_error = rollout_record_error(record)
                if self._already_ingested(conn, record, record_sha, record_error):
                    self.n_skipped += 1
                    continue
                cell_id = upsert_rollout_record(
                    conn,
                    experiment_id=self.config.experiment_id,
                    provider_source=self.source,
                    model_api_name=self.model.api_name,
                    model_label=self.model.label,
                    dataset=self.config.dataset_name,
                    record=record,
                    artifact_rollouts=self.rollouts_path,
                )
                conn.commit()
                if self.bonus_map_dir is not None:
                    self._metrics_dirty = True
                    if record_error is None:
                        pending_details.append((cell_id, record, self.n_ingested))
                self.n_ingested += 1
            for cell_id, record, index in pending_details:
                try:
                    write_dashboard_detail_cache_for_record(
                        conn,
                        cell_id=cell_id,
                        record=record,
                        bonus_map_dir=self.bonus_map_dir,
                        tracking_mode=self.analysis_cfg.get("tracking_mode", "view_and_bash"),
                        near_threshold=float(self.analysis_cfg.get("near_threshold", 0.5)),
                        m_max=float(self.analysis_cfg.get("m_max", 3.0)),
                        index=index,
                        refresh_model_metrics=False,
                    )
                except Exception:
                    self.n_detail_cache_failed += 1

    def _refresh_model_metrics_sync(self) -> None:
        refresh_dashboard_model_metrics(
            self.config.db_path,
            experiment_id=self.config.experiment_id,
            provider_source=self.source,
            dataset=self.config.dataset_name,
            model_api_name=self.model.api_name,
            model_label=self.model.label,
            bonus_map_dir=self.bonus_map_dir,
            tracking_mode=self.analysis_cfg.get("tracking_mode", "view_and_bash"),
            near_threshold=float(self.analysis_cfg.get("near_threshold", 0.5)),
            m_max=float(self.analysis_cfg.get("m_max", 3.0)),
        )

    async def _run_db_work(self, fn: Any, *args: Any) -> Any:
        # DB work runs on the batch's dedicated writer executor (explicitly
        # shut down by run_batch) rather than the loop's default executor, and
        # off the event loop so CPU-heavy scoring cannot back-pressure the
        # model subprocess pipes.
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.db_executor, fn, *args)

    async def ingest_new(self, *, refresh_metrics: bool = False) -> int:
        records, new_offset = self._read_new_records()
        if not records and not (refresh_metrics and self._metrics_dirty):
            return 0
        async with self.db_lock:
            if records:
                await self._run_db_work(self._ingest_records_sync, records)
                self._offset = new_offset
            if refresh_metrics and self._metrics_dirty:
                await self._run_db_work(self._refresh_model_metrics_sync)
                self._metrics_dirty = False
        return len(records)

    def stop(self) -> None:
        self._stop_event.set()

    async def run_periodic(self, interval_s: float = 30.0) -> None:
        # Shut down via stop(), never task.cancel(): cancelling while an ingest
        # runs inside asyncio.to_thread would release db_lock while the writer
        # thread is still alive, letting the final ingest start a second
        # concurrent writer.
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval_s)
                break
            except asyncio.TimeoutError:
                pass
            try:
                await self.ingest_new(refresh_metrics=True)
            except (OSError, sqlite3.Error) as exc:
                print(f"[batch] ingest for {self.model.label} failed (will retry): {exc}", flush=True)


async def run_model_phase(
    *,
    config: BatchConfig,
    model: BatchModel,
    phase: str,
    phase_limit: int | None,
    data_file: Path,
    bonus_map_dir: Path | None,
    env: dict[str, str],
    db_lock: asyncio.Lock,
    db_executor: ThreadPoolExecutor | None = None,
) -> dict[str, Any]:
    source = provider_source(config.provider)
    target_ids, scope_metadata = selected_instance_scope(
        data_file,
        limit=phase_limit,
        offset=config.offset,
        bonus_map_dir=bonus_map_dir,
        scope_filter=config.bonus_map_instance_filter,
    )
    target_jobs = _rollout_jobs(target_ids, config.rollouts_per_instance)
    run_dir = config.artifacts_dir / config.experiment_id / phase / config.dataset_name / _safe_slug(model.label)
    rollouts_path = run_dir / "rollouts.jsonl"
    ingestor = RolloutIngestor(
        config=config,
        model=model,
        rollouts_path=rollouts_path,
        bonus_map_dir=bonus_map_dir,
        db_lock=db_lock,
        db_executor=db_executor,
    )
    # Rollouts persisted to JSONL by a previous (possibly crashed) run may not
    # have reached the DB; recover them before selecting missing jobs.
    await ingestor.ingest_new()
    async with db_lock:
        with ensure_db(config.db_path) as conn:
            upsert_experiment(
                conn,
                experiment_id=config.experiment_id,
                provider_source=source,
                dataset=config.dataset_name,
                config_snapshot=sanitized_config_snapshot(config, scope=scope_metadata),
            )
            upsert_planned_cells(
                conn,
                experiment_id=config.experiment_id,
                provider_source=source,
                model_api_name=model.api_name,
                model_label=model.label,
                dataset=config.dataset_name,
                instance_ids=target_ids,
                rollouts_per_instance=config.rollouts_per_instance,
            )
            done_jobs = completed_rollout_keys(
                conn,
                experiment_id=config.experiment_id,
                provider_source=source,
                model_api_name=model.api_name,
                dataset=config.dataset_name,
            )
            missing_jobs = [job for job in target_jobs if job not in done_jobs]
            mark_cells_running(
                conn,
                experiment_id=config.experiment_id,
                provider_source=source,
                model_api_name=model.api_name,
                dataset=config.dataset_name,
                rollout_jobs=missing_jobs,
            )
            conn.commit()

    if not missing_jobs:
        return {"model": model.label, "phase": phase, "status": "skipped", "n_missing": 0}

    run_dir.mkdir(parents=True, exist_ok=True)
    eval_config = _model_eval_config(config, model, run_dir)
    command = _base_command(
        eval_config=eval_config,
        data_file=data_file,
        run_dir=run_dir,
        missing_jobs=missing_jobs,
        config=config,
        model=model,
        bonus_map_dir=bonus_map_dir,
    )
    model_env = dict(env)
    model_env["P2A_THIRD_PARTY_MODEL"] = model.api_name

    ingest_task = asyncio.create_task(ingestor.run_periodic())
    try:
        returncode, output = await _run_subprocess(command, env=model_env, timeout_s=_duration_seconds(config.run_timeout))
    finally:
        # Signal instead of cancel so an in-flight ingest thread finishes under
        # the lock before the final ingest runs (single-writer guarantee).
        ingestor.stop()
        try:
            await ingest_task
        except (OSError, sqlite3.Error) as exc:
            print(f"[batch] periodic ingest for {model.label} ended with error: {exc}", flush=True)
        await ingestor.ingest_new(refresh_metrics=True)
    log_path = run_dir / "run.log"
    log_path.write_text(output, encoding="utf-8")
    if returncode != 0:
        unfinished_jobs = _remaining_unfinished_jobs(
            config.db_path,
            config=config,
            model=model,
            rollout_jobs=missing_jobs,
        )
        async with db_lock:
            _mark_missing_error(
                config.db_path,
                config=config,
                model=model,
                rollout_jobs=unfinished_jobs,
                error=f"third_party_eval exited {returncode}; see {log_path}",
            )
        return {
            "model": model.label,
            "phase": phase,
            "status": "error",
            "returncode": returncode,
            "log": str(log_path),
            "n_missing": len(missing_jobs),
            "n_persisted": len(missing_jobs) - len(unfinished_jobs),
        }

    n_records = _count_rollout_records(rollouts_path)
    system_error = _system_error_summary(rollouts_path)
    if system_error is not None:
        return {
            "model": model.label,
            "phase": phase,
            "status": SYSTEM_ERROR_STATUS,
            "n_missing": len(missing_jobs),
            "n_ingested": ingestor.n_ingested,
            "n_records": n_records,
            "run_dir": str(run_dir),
            **system_error,
        }
    return {
        "model": model.label,
        "phase": phase,
        "status": DONE_STATUS,
        "n_missing": len(missing_jobs),
        "n_ingested": ingestor.n_ingested,
        "n_records": n_records,
        "run_dir": str(run_dir),
    }


async def run_batch(config: BatchConfig, *, env: dict[str, str] | None = None) -> list[dict[str, Any]]:
    run_env = dict(os.environ if env is None else env)
    check_provider_available(config.provider, repo_root=Path.cwd())
    data_file = resolve_data_file(config, env=run_env)
    bonus_map_dir = resolve_bonus_map_dir(config, data_file, env=run_env)
    results = []
    semaphore = asyncio.Semaphore(config.model_parallelism)
    # All DB writes across models funnel through this lock and one dedicated
    # writer thread: the orchestrator is the single writer regardless of
    # model/rollout concurrency.
    db_lock = asyncio.Lock()
    db_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="p2a-db-ingest")

    async def guarded(model: BatchModel, phase: str, limit: int | None) -> dict[str, Any]:
        async with semaphore:
            print(f"[batch] {phase}: {model.label} ({model.api_name})", flush=True)
            result = await run_model_phase(
                config=config,
                model=model,
                phase=phase,
                phase_limit=limit,
                data_file=data_file,
                bonus_map_dir=bonus_map_dir,
                env=run_env,
                db_lock=db_lock,
                db_executor=db_executor,
            )
            print(f"[batch] {phase}: {model.label} -> {result['status']}", flush=True)
            return result

    try:
        for phase, limit in _phase_specs(config):
            phase_results = await asyncio.gather(*(guarded(model, phase, limit) for model in config.models))
            results.extend(phase_results)
            if phase == "smoke" and any(result.get("status") == SYSTEM_ERROR_STATUS for result in phase_results):
                print("[batch] smoke phase hit a system error; skipping later phases", flush=True)
                break
    finally:
        db_executor.shutdown(wait=True)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "--batch", dest="config", type=Path, required=True)
    parser.add_argument("--db", type=Path, help="Override storage.db from the batch config")
    args = parser.parse_args()

    config = load_batch_config(args.config)
    if args.db is not None:
        config = BatchConfig(**{**config.__dict__, "db_path": args.db})
    results = asyncio.run(run_batch(config))
    print(json.dumps({"db": str(config.db_path), "results": results}, indent=2))
    return 0 if all(result["status"] in {DONE_STATUS, "skipped"} for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
