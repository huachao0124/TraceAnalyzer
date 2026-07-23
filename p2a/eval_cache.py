"""SQLite cache for API/local evaluation rollouts and trace-quality metrics."""

from __future__ import annotations

import json
import hashlib
import re
import sqlite3
from collections.abc import Mapping
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

from p2a.bonus_map_scope import LATENT_CASE, PATH_CASE_TYPES, canonical_detail_case_type
from p2a.eval_fault_localization import _json_default, _sync_path_aliases, iter_records


SCHEMA_VERSION = 3
DASHBOARD_BUILD_SCHEMA_VERSION = 1
DONE_STATUS = "done"
ERROR_STATUS = "error"
PENDING_STATUS = "pending"
RUNNING_STATUS = "running"
EMPTY_ROLLOUT_ERROR = "Empty rollout trace: no model response, tool call, or observation was captured."
EMPTY_ROLLOUT_ERROR_KIND = "empty_trace"
EMPTY_ROLLOUT_FAILURE_REASONS = {"unknown", "unknown_error", "error"}
RAW_ROLLOUT_METADATA_KEYS = {
    "base_url",
    "data_source",
    "dataset",
    "error",
    "error_kind",
    "error_stage",
    "execution_time",
    "experiment_id",
    "instance_id",
    "model",
    "model_api_name",
    "model_label",
    "provider_source",
    "rollout_id",
    "rollout_index",
    "run_id",
    "schema_version",
    "status",
    "system_error",
    "termination_reason",
    "wall_time",
}
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_dumps(value: Any) -> str:
    return json.dumps(value, default=_json_default, ensure_ascii=False, sort_keys=True)


def json_loads(value: str | None, default: Any = None) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _nonempty_sequence(value: Any) -> bool:
    return isinstance(value, list | tuple) and bool(value)


def _step_has_content(step: Any) -> bool:
    if not isinstance(step, Mapping):
        return False
    if any(
        _nonempty_text(step.get(key))
        for key in (
            "response_text",
            "response",
            "thought",
            "reasoning",
            "action",
            "tool_name",
            "command",
        )
    ):
        return True
    return any(_nonempty_sequence(step.get(key)) for key in ("tool_calls", "tool_results", "tool_args"))


def rollout_record_error(record: Mapping[str, Any]) -> str | None:
    explicit_error = record.get("error")
    if explicit_error:
        return str(explicit_error)
    if _nonempty_text(record.get("response_text")):
        return None
    trajectory = record.get("trajectory") if isinstance(record.get("trajectory"), list) else []
    step_traces = record.get("p2a_step_traces") if isinstance(record.get("p2a_step_traces"), list) else []
    if any(_step_has_content(step) for step in [*trajectory, *step_traces]):
        return None
    termination = str(record.get("termination_reason") or "").lower()
    step_exit_reasons = {str(step.get("exit_reason") or "").lower() for step in [*trajectory, *step_traces] if isinstance(step, Mapping) and step.get("exit_reason")}
    has_failure_shell = termination in EMPTY_ROLLOUT_FAILURE_REASONS or bool(step_exit_reasons & EMPTY_ROLLOUT_FAILURE_REASONS)
    if has_failure_shell:
        return EMPTY_ROLLOUT_ERROR
    return None


def slim_rollout_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in record.items() if key in RAW_ROLLOUT_METADATA_KEYS}


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
    return _first_text_field(
        record,
        (
            "problem_statement",
            "issue_description",
            "issue_text",
            "issue",
            "description",
            "problem",
            "title",
        ),
    )


def _golden_patch(record: dict[str, Any]) -> str | None:
    return _first_text_field(record, ("golden_patch", "patch", "base_patch", "fix_patch"))


def _final_patch(record: dict[str, Any]) -> str | None:
    value = record.get("final_patch")
    return value if isinstance(value, str) and value else None


def connect(db_path: Path | str, *, timeout: float = 30.0) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # BEGIN IMMEDIATE acquires the write lock up front; a deferred read-to-write
    # upgrade can fail with SQLITE_BUSY without waiting for busy_timeout when
    # another writer holds the lock.
    conn = sqlite3.connect(path, timeout=timeout, isolation_level="IMMEDIATE")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        # Under WAL, NORMAL keeps the database consistent on crash and only
        # relaxes power-loss durability of the latest commits; FULL fsyncs on
        # every commit and dominates write latency at high rollout throughput.
        conn.execute("PRAGMA synchronous = NORMAL")
    except sqlite3.DatabaseError:
        pass
    return conn


def connect_readonly(db_path: Path | str, *, timeout: float = 2.0) -> sqlite3.Connection:
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(path)
    uri_path = quote(str(path.resolve()), safe="/:")
    conn = sqlite3.connect(f"file:{uri_path}?mode=ro", uri=True, timeout=timeout)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA query_only = ON")
    conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, declaration: str) -> None:
    columns = {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)).fetchone()
    return row is not None


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    if not _table_exists(conn, table):
        return set()
    return {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _create_eval_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS experiments (
          experiment_id TEXT NOT NULL,
          provider_source TEXT NOT NULL,
          dataset TEXT NOT NULL,
          config_snapshot TEXT NOT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          PRIMARY KEY (experiment_id, provider_source, dataset)
        );

        CREATE TABLE IF NOT EXISTS run_cells (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          experiment_id TEXT NOT NULL,
          provider_source TEXT NOT NULL,
          model_api_name TEXT NOT NULL,
          model_label TEXT NOT NULL,
          dataset TEXT NOT NULL,
          instance_id TEXT NOT NULL,
          rollout_index INTEGER NOT NULL DEFAULT 0,
          rollout_id TEXT,
          status TEXT NOT NULL,
          attempts INTEGER NOT NULL DEFAULT 0,
          run_id TEXT,
          artifact_rollouts TEXT,
          artifact_details TEXT,
          started_at TEXT,
          ended_at TEXT,
          error TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          UNIQUE (experiment_id, provider_source, model_api_name, dataset, instance_id, rollout_index)
        );

        CREATE TABLE IF NOT EXISTS raw_rollouts (
          run_id TEXT PRIMARY KEY,
          cell_id INTEGER NOT NULL UNIQUE REFERENCES run_cells(id) ON DELETE CASCADE,
          messages_json TEXT,
          trajectory_json TEXT,
          p2a_step_traces_json TEXT,
          final_response TEXT,
          reward_json TEXT,
          resolved INTEGER,
          token_usage_json TEXT,
          cache_metrics_json TEXT,
          issue_description TEXT,
          golden_patch TEXT,
          final_patch TEXT,
          rollout_json TEXT NOT NULL,
          rollout_sha256 TEXT,
          created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS quantitative_metrics (
          cell_id INTEGER PRIMARY KEY REFERENCES run_cells(id) ON DELETE CASCADE,
          reward REAL,
          resolved INTEGER,
          p2a_read INTEGER,
          call_graph_hit INTEGER,
          ground_truth_hit INTEGER,
          near_hit INTEGER,
          min_distance REAL,
          turns INTEGER,
          tool_calls INTEGER,
          wall_time REAL,
          input_tokens REAL,
          output_tokens REAL,
          reasoning_tokens REAL,
          cache_hit_tokens REAL,
          cache_write_tokens REAL,
          cost REAL,
          fingerprint TEXT,
          metrics_json TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_run_cells_exp_model
          ON run_cells (experiment_id, provider_source, dataset, model_label);
        CREATE INDEX IF NOT EXISTS idx_run_cells_status
          ON run_cells (experiment_id, provider_source, dataset, status);
        CREATE INDEX IF NOT EXISTS idx_run_cells_instance_rollout
          ON run_cells (experiment_id, provider_source, model_api_name, dataset, instance_id, rollout_index);
        """
    )


def _insert_rows_with_common_columns(conn: sqlite3.Connection, table: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = [column for column in rows[0] if column in _table_columns(conn, table)]
    if not columns:
        return
    placeholders = ", ".join("?" for _ in columns)
    conn.executemany(
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
        [[row.get(column) for column in columns] for row in rows],
    )


def _migrate_rollout_index_schema(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "run_cells") or "rollout_index" in _table_columns(conn, "run_cells"):
        return
    run_rows = [dict(row) | {"rollout_index": 0, "rollout_id": None} for row in conn.execute("SELECT * FROM run_cells")]
    raw_rows = [dict(row) for row in conn.execute("SELECT * FROM raw_rollouts")] if _table_exists(conn, "raw_rollouts") else []
    metric_rows = [dict(row) for row in conn.execute("SELECT * FROM quantitative_metrics")] if _table_exists(conn, "quantitative_metrics") else []
    conn.commit()
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute("DROP TABLE IF EXISTS quantitative_metrics")
        conn.execute("DROP TABLE IF EXISTS raw_rollouts")
        conn.execute("DROP TABLE IF EXISTS run_cells")
        _create_eval_tables(conn)
        _insert_rows_with_common_columns(conn, "run_cells", run_rows)
        _insert_rows_with_common_columns(conn, "raw_rollouts", raw_rows)
        _insert_rows_with_common_columns(conn, "quantitative_metrics", metric_rows)
        conn.commit()
    finally:
        conn.execute("PRAGMA foreign_keys = ON")


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
          version INTEGER PRIMARY KEY,
          applied_at TEXT NOT NULL
        );
        """
    )
    _migrate_rollout_index_schema(conn)
    _create_eval_tables(conn)
    _ensure_column(conn, "raw_rollouts", "issue_description", "TEXT")
    _ensure_column(conn, "raw_rollouts", "golden_patch", "TEXT")
    _ensure_column(conn, "raw_rollouts", "final_patch", "TEXT")
    had_rollout_sha = "rollout_sha256" in _table_columns(conn, "raw_rollouts")
    _ensure_column(conn, "raw_rollouts", "rollout_sha256", "TEXT")
    if not had_rollout_sha:
        # Full-table backfill only when the column was just created; steady-state
        # init must stay cheap because writers call it on connection setup.
        backfill_raw_rollout_sha256(conn)
    _ensure_column(conn, "quantitative_metrics", "fingerprint", "TEXT")
    conn.execute(
        "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
        (SCHEMA_VERSION, utc_now()),
    )
    conn.commit()


def ensure_db(db_path: Path | str) -> sqlite3.Connection:
    conn = connect(db_path)
    init_db(conn)
    return conn


def backfill_raw_rollout_sha256(conn: sqlite3.Connection) -> int:
    if not _table_exists(conn, "raw_rollouts"):
        return 0
    columns = _table_columns(conn, "raw_rollouts")
    if "rollout_sha256" not in columns or "rollout_json" not in columns:
        return 0
    rows = conn.execute(
        """
        SELECT cell_id, rollout_json
        FROM raw_rollouts
        WHERE rollout_sha256 IS NULL OR rollout_sha256 = ''
        """
    ).fetchall()
    if not rows:
        return 0
    conn.executemany(
        "UPDATE raw_rollouts SET rollout_sha256 = ? WHERE cell_id = ?",
        [(sha256_text(str(row["rollout_json"] or "")), row["cell_id"]) for row in rows],
    )
    return len(rows)


def default_dashboard_build_db_path(raw_db_path: Path | str) -> Path:
    path = Path(raw_db_path)
    return path.with_name(f"{path.stem}.dashboard.sqlite")


def init_dashboard_build_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dashboard_schema_migrations (
          version INTEGER PRIMARY KEY,
          applied_at TEXT NOT NULL
        );
        """
    )
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS dashboard_rollout_details (
          raw_db_path TEXT NOT NULL,
          raw_cell_id INTEGER NOT NULL,
          experiment_id TEXT NOT NULL,
          provider_source TEXT NOT NULL,
          dataset TEXT NOT NULL,
          model_api_name TEXT NOT NULL,
          model_label TEXT NOT NULL,
          instance_id TEXT NOT NULL,
          rollout_index INTEGER NOT NULL DEFAULT 0,
          rollout_id TEXT,
          run_id TEXT,
          status TEXT NOT NULL,
          raw_rollout_sha256 TEXT NOT NULL,
          fingerprint TEXT NOT NULL,
          cache_metadata_json TEXT NOT NULL,
          detail_json TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          PRIMARY KEY (raw_db_path, raw_cell_id)
        );

        CREATE INDEX IF NOT EXISTS idx_dashboard_details_scope
          ON dashboard_rollout_details (
            raw_db_path, experiment_id, provider_source, dataset,
            model_api_name, model_label, instance_id, rollout_index
          );

        CREATE TABLE IF NOT EXISTS dashboard_model_metrics (
          raw_db_path TEXT NOT NULL,
          experiment_id TEXT NOT NULL,
          provider_source TEXT NOT NULL,
          dataset TEXT NOT NULL,
          model_api_name TEXT NOT NULL,
          model_label TEXT NOT NULL,
          metrics_json TEXT NOT NULL,
          detail_cache_ready_rollouts INTEGER NOT NULL DEFAULT 0,
          detail_cache_pending_rollouts INTEGER NOT NULL DEFAULT 0,
          updated_at TEXT NOT NULL,
          PRIMARY KEY (
            raw_db_path, experiment_id, provider_source, dataset,
            model_api_name, model_label
          )
        );
        """
    )
    conn.execute(
        "INSERT OR IGNORE INTO dashboard_schema_migrations(version, applied_at) VALUES (?, ?)",
        (DASHBOARD_BUILD_SCHEMA_VERSION, utc_now()),
    )
    conn.commit()


def ensure_dashboard_build_db(build_db_path: Path | str) -> sqlite3.Connection:
    conn = connect(build_db_path)
    init_dashboard_build_db(conn)
    return conn


def _database_path(conn: sqlite3.Connection) -> Path | None:
    row = conn.execute("PRAGMA database_list").fetchone()
    if row is None:
        return None
    value = row["file"] if isinstance(row, sqlite3.Row) and "file" in row.keys() else row[2]
    return Path(value).resolve() if value else None


def raw_db_identity(raw_db_path: Path | str) -> str:
    return str(Path(raw_db_path).resolve())


def connection_raw_db_identity(conn: sqlite3.Connection) -> str | None:
    path = _database_path(conn)
    return raw_db_identity(path) if path is not None else None


def write_dashboard_build_detail(
    conn: sqlite3.Connection,
    *,
    raw_db_path: Path | str,
    raw_cell_id: int,
    experiment_id: str,
    provider_source: str,
    dataset: str,
    model_api_name: str,
    model_label: str,
    instance_id: str,
    rollout_index: int,
    rollout_id: str | None,
    run_id: str | None,
    status: str,
    raw_rollout_sha256: str,
    fingerprint: str,
    detail: Mapping[str, Any],
    cache_metadata: Mapping[str, Any],
) -> None:
    now = utc_now()
    conn.execute(
        """
        INSERT INTO dashboard_rollout_details(
          raw_db_path, raw_cell_id, experiment_id, provider_source, dataset,
          model_api_name, model_label, instance_id, rollout_index, rollout_id,
          run_id, status, raw_rollout_sha256, fingerprint, cache_metadata_json,
          detail_json, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(raw_db_path, raw_cell_id) DO UPDATE SET
          experiment_id = excluded.experiment_id,
          provider_source = excluded.provider_source,
          dataset = excluded.dataset,
          model_api_name = excluded.model_api_name,
          model_label = excluded.model_label,
          instance_id = excluded.instance_id,
          rollout_index = excluded.rollout_index,
          rollout_id = excluded.rollout_id,
          run_id = excluded.run_id,
          status = excluded.status,
          raw_rollout_sha256 = excluded.raw_rollout_sha256,
          fingerprint = excluded.fingerprint,
          cache_metadata_json = excluded.cache_metadata_json,
          detail_json = excluded.detail_json,
          updated_at = excluded.updated_at
        """,
        (
            raw_db_identity(raw_db_path),
            int(raw_cell_id),
            experiment_id,
            provider_source,
            dataset,
            model_api_name,
            model_label,
            instance_id,
            int(rollout_index or 0),
            rollout_id,
            run_id,
            status,
            raw_rollout_sha256,
            fingerprint,
            json_dumps(dict(cache_metadata)),
            json_dumps(dict(detail)),
            now,
        ),
    )


def write_dashboard_build_model_metrics(
    conn: sqlite3.Connection,
    *,
    raw_db_path: Path | str,
    row: Mapping[str, Any],
) -> None:
    now = utc_now()
    conn.execute(
        """
        INSERT INTO dashboard_model_metrics(
          raw_db_path, experiment_id, provider_source, dataset, model_api_name,
          model_label, metrics_json, detail_cache_ready_rollouts,
          detail_cache_pending_rollouts, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(raw_db_path, experiment_id, provider_source, dataset, model_api_name, model_label)
        DO UPDATE SET
          metrics_json = excluded.metrics_json,
          detail_cache_ready_rollouts = excluded.detail_cache_ready_rollouts,
          detail_cache_pending_rollouts = excluded.detail_cache_pending_rollouts,
          updated_at = excluded.updated_at
        """,
        (
            raw_db_identity(raw_db_path),
            str(row.get("experiment_id") or ""),
            str(row.get("provider_source") or ""),
            str(row.get("dataset") or ""),
            str(row.get("model_api_name") or ""),
            str(row.get("model_label") or ""),
            json_dumps(dict(row)),
            int(row.get("detail_cache_ready_rollouts") or 0),
            int(row.get("detail_cache_pending_rollouts") or 0),
            now,
        ),
    )


def _clean_delete_target(
    *,
    experiment_id: str | None = None,
    provider_source: str | None = None,
    dataset: str | None = None,
) -> dict[str, str]:
    target = {
        "experiment_id": str(experiment_id).strip() if experiment_id else "",
        "provider_source": str(provider_source).strip() if provider_source else "",
        "dataset": str(dataset).strip() if dataset else "",
    }
    return {key: value for key, value in target.items() if value}


def _normalize_delete_targets(targets: Iterable[Mapping[str, Any]], *, allow_all: bool = False) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for item in targets:
        target = _clean_delete_target(
            experiment_id=item.get("experiment_id"),
            provider_source=item.get("provider_source"),
            dataset=item.get("dataset"),
        )
        if not target and not allow_all:
            raise ValueError("delete target must include experiment_id, provider_source, or dataset")
        key = tuple(sorted(target.items()))
        if key not in seen:
            normalized.append(target)
            seen.add(key)
    if not normalized and not allow_all:
        raise ValueError("at least one delete target is required")
    return normalized


def _target_where(alias: str, targets: Iterable[Mapping[str, Any]], *, allow_all: bool = False) -> tuple[str, list[Any]]:
    normalized = _normalize_delete_targets(targets, allow_all=allow_all)
    if not normalized and allow_all:
        return "1 = 1", []
    clauses = []
    params: list[Any] = []
    for target in normalized:
        if not target:
            clauses.append("1 = 1")
            continue
        parts = []
        for key in ("experiment_id", "provider_source", "dataset"):
            value = target.get(key)
            if value:
                parts.append(f"{alias}.{key} = ?")
                params.append(value)
        clauses.append("(" + " AND ".join(parts) + ")")
    return " OR ".join(clauses), params


def _single_target(
    *,
    experiment_id: str | None = None,
    provider_source: str | None = None,
    dataset: str | None = None,
) -> list[dict[str, str]]:
    return [
        _clean_delete_target(
            experiment_id=experiment_id,
            provider_source=provider_source,
            dataset=dataset,
        )
    ]


def count_run_data_targets(
    conn: sqlite3.Connection,
    targets: Iterable[Mapping[str, Any]],
    *,
    allow_all: bool = False,
) -> dict[str, int]:
    where_cells, cell_params = _target_where("c", targets, allow_all=allow_all)
    where_exp, exp_params = _target_where("e", targets, allow_all=allow_all)
    run_cells = int(conn.execute(f"SELECT COUNT(*) FROM run_cells c WHERE {where_cells}", cell_params).fetchone()[0])
    raw_rollouts = int(
        conn.execute(
            f"""
            SELECT COUNT(*)
            FROM raw_rollouts r
            JOIN run_cells c ON c.id = r.cell_id
            WHERE {where_cells}
            """,
            cell_params,
        ).fetchone()[0]
    )
    quantitative_metrics = int(
        conn.execute(
            f"""
            SELECT COUNT(*)
            FROM quantitative_metrics q
            JOIN run_cells c ON c.id = q.cell_id
            WHERE {where_cells}
            """,
            cell_params,
        ).fetchone()[0]
    )
    experiments = int(conn.execute(f"SELECT COUNT(*) FROM experiments e WHERE {where_exp}", exp_params).fetchone()[0])
    return {
        "experiments": experiments,
        "run_cells": run_cells,
        "raw_rollouts": raw_rollouts,
        "quantitative_metrics": quantitative_metrics,
    }


def count_run_data(
    conn: sqlite3.Connection,
    *,
    experiment_id: str | None = None,
    provider_source: str | None = None,
    dataset: str | None = None,
    allow_all: bool = False,
) -> dict[str, int]:
    return count_run_data_targets(
        conn,
        _single_target(
            experiment_id=experiment_id,
            provider_source=provider_source,
            dataset=dataset,
        ),
        allow_all=allow_all,
    )


def delete_run_data_targets(
    conn: sqlite3.Connection,
    targets: Iterable[Mapping[str, Any]],
    *,
    allow_all: bool = False,
) -> dict[str, int]:
    normalized = _normalize_delete_targets(targets, allow_all=allow_all)
    counts = count_run_data_targets(conn, normalized, allow_all=allow_all)
    where_cells, cell_params = _target_where("c", normalized, allow_all=allow_all)
    where_exp, exp_params = _target_where("e", normalized, allow_all=allow_all)
    with conn:
        conn.execute(
            f"DELETE FROM run_cells WHERE id IN (SELECT c.id FROM run_cells c WHERE {where_cells})",
            cell_params,
        )
        conn.execute(
            f"DELETE FROM experiments WHERE rowid IN (SELECT e.rowid FROM experiments e WHERE {where_exp})",
            exp_params,
        )
    return counts


def delete_run_data(
    conn: sqlite3.Connection,
    *,
    experiment_id: str | None = None,
    provider_source: str | None = None,
    dataset: str | None = None,
    allow_all: bool = False,
) -> dict[str, int]:
    return delete_run_data_targets(
        conn,
        _single_target(
            experiment_id=experiment_id,
            provider_source=provider_source,
            dataset=dataset,
        ),
        allow_all=allow_all,
    )


def delete_confirmation_phrase(counts: Mapping[str, Any]) -> str:
    run_cells = int(counts.get("run_cells") or 0)
    experiments = int(counts.get("experiments") or 0)
    return f"delete {run_cells} run cells and {experiments} experiments"


def backup_path_for_delete(db_path: Path | str, *, timestamp: str | None = None) -> Path:
    path = Path(db_path)
    stamp = timestamp or utc_now()
    safe_stamp = re.sub(r"[^0-9A-Za-z_.-]+", "-", stamp).strip("-")
    return path.with_name(f"{path.name}.backup-{safe_stamp}")


def upsert_experiment(
    conn: sqlite3.Connection,
    *,
    experiment_id: str,
    provider_source: str,
    dataset: str,
    config_snapshot: dict[str, Any] | str,
) -> None:
    now = utc_now()
    snapshot = config_snapshot if isinstance(config_snapshot, str) else json_dumps(config_snapshot)
    conn.execute(
        """
        INSERT INTO experiments(experiment_id, provider_source, dataset, config_snapshot, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(experiment_id, provider_source, dataset) DO UPDATE SET
          config_snapshot = excluded.config_snapshot,
          updated_at = excluded.updated_at
        """,
        (experiment_id, provider_source, dataset, snapshot, now, now),
    )


def upsert_planned_cells(
    conn: sqlite3.Connection,
    *,
    experiment_id: str,
    provider_source: str,
    model_api_name: str,
    model_label: str,
    dataset: str,
    instance_ids: Iterable[str],
    rollouts_per_instance: int = 1,
) -> None:
    now = utc_now()
    rollouts_per_instance = max(1, int(rollouts_per_instance or 1))
    rows = [
        (
            experiment_id,
            provider_source,
            model_api_name,
            model_label,
            dataset,
            instance_id,
            rollout_index,
            PENDING_STATUS,
            now,
            now,
        )
        for instance_id in instance_ids
        for rollout_index in range(rollouts_per_instance)
    ]
    conn.executemany(
        """
        INSERT INTO run_cells(
          experiment_id, provider_source, model_api_name, model_label, dataset,
          instance_id, rollout_index, status, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(experiment_id, provider_source, model_api_name, dataset, instance_id, rollout_index) DO UPDATE SET
          model_label = excluded.model_label,
          updated_at = excluded.updated_at
        """,
        rows,
    )


def mark_cells_running(
    conn: sqlite3.Connection,
    *,
    experiment_id: str,
    provider_source: str,
    model_api_name: str,
    dataset: str,
    instance_ids: Iterable[str] | None = None,
    rollout_jobs: Iterable[tuple[str, int]] | None = None,
) -> None:
    now = utc_now()
    jobs = list(rollout_jobs or [])
    if not jobs:
        jobs = [(instance_id, 0) for instance_id in (instance_ids or [])]
    conn.executemany(
        """
        UPDATE run_cells
        SET status = ?, started_at = COALESCE(started_at, ?), updated_at = ?
        WHERE experiment_id = ?
          AND provider_source = ?
          AND model_api_name = ?
          AND dataset = ?
          AND instance_id = ?
          AND rollout_index = ?
          AND status != ?
        """,
        [
            (
                RUNNING_STATUS,
                now,
                now,
                experiment_id,
                provider_source,
                model_api_name,
                dataset,
                instance_id,
                rollout_index,
                DONE_STATUS,
            )
            for instance_id, rollout_index in jobs
        ],
    )


EMPTY_ROLLOUT_SQL = """
  r.cell_id IS NOT NULL
  AND COALESCE(r.final_response, '') = ''
  AND json_array_length(COALESCE(r.trajectory_json, '[]')) <= 1
  AND json_array_length(COALESCE(r.p2a_step_traces_json, '[]')) <= 1
  AND lower(COALESCE(json_extract(r.rollout_json, '$.termination_reason'), '')) IN ('unknown', 'unknown_error', 'error')
  AND COALESCE(json_extract(r.p2a_step_traces_json, '$[0].response_text'), '') = ''
  AND COALESCE(json_extract(r.p2a_step_traces_json, '$[0].thought'), '') = ''
  AND COALESCE(json_extract(r.p2a_step_traces_json, '$[0].action'), '') = ''
  AND COALESCE(json_extract(r.p2a_step_traces_json, '$[0].tool_name'), '') = ''
  AND COALESCE(json_array_length(json_extract(r.p2a_step_traces_json, '$[0].tool_calls')), 0) = 0
  AND COALESCE(json_array_length(json_extract(r.p2a_step_traces_json, '$[0].tool_results')), 0) = 0
"""


def completed_instance_ids(
    conn: sqlite3.Connection,
    *,
    experiment_id: str,
    provider_source: str,
    model_api_name: str,
    dataset: str,
) -> set[str]:
    rows = conn.execute(
        f"""
        SELECT c.instance_id
        FROM run_cells c
        LEFT JOIN raw_rollouts r ON r.cell_id = c.id
        WHERE c.experiment_id = ?
          AND c.provider_source = ?
          AND c.model_api_name = ?
          AND c.dataset = ?
          AND c.status = ?
          AND r.cell_id IS NOT NULL
          AND NOT ({EMPTY_ROLLOUT_SQL})
        """,
        (experiment_id, provider_source, model_api_name, dataset, DONE_STATUS),
    ).fetchall()
    return {str(row["instance_id"]) for row in rows}


def completed_rollout_keys(
    conn: sqlite3.Connection,
    *,
    experiment_id: str,
    provider_source: str,
    model_api_name: str,
    dataset: str,
) -> set[tuple[str, int]]:
    rows = conn.execute(
        f"""
        SELECT c.instance_id, c.rollout_index
        FROM run_cells c
        LEFT JOIN raw_rollouts r ON r.cell_id = c.id
        WHERE c.experiment_id = ?
          AND c.provider_source = ?
          AND c.model_api_name = ?
          AND c.dataset = ?
          AND c.status = ?
          AND r.cell_id IS NOT NULL
          AND NOT ({EMPTY_ROLLOUT_SQL})
        """,
        (experiment_id, provider_source, model_api_name, dataset, DONE_STATUS),
    ).fetchall()
    return {(str(row["instance_id"]), int(row["rollout_index"] or 0)) for row in rows}


def _bool_int(value: Any) -> int | None:
    if value is None:
        return None
    return 1 if bool(value) else 0


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _tool_call_count(record: dict[str, Any]) -> int:
    total = 0
    for trace in record.get("p2a_step_traces") or []:
        if isinstance(trace, dict):
            calls = trace.get("tool_calls") or []
            if isinstance(calls, list):
                total += len(calls)
    return total


def _raw_turn_count(*values: Any) -> int | None:
    for value in values:
        parsed = json_loads(value, value) if isinstance(value, str) else value
        if isinstance(parsed, list):
            return len(parsed)
    return None


def _raw_tool_call_count(value: Any) -> int | None:
    traces = json_loads(value, value) if isinstance(value, str) else value
    if not isinstance(traces, list):
        return None
    return _tool_call_count({"p2a_step_traces": traces})


def _reward_number(record: dict[str, Any]) -> float | None:
    value = record.get("reward")
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return _number(value)


def _cell_id(
    conn: sqlite3.Connection,
    *,
    experiment_id: str,
    provider_source: str,
    model_api_name: str,
    model_label: str,
    dataset: str,
    instance_id: str,
    rollout_index: int = 0,
) -> int:
    now = utc_now()
    conn.execute(
        """
        INSERT INTO run_cells(
          experiment_id, provider_source, model_api_name, model_label, dataset,
          instance_id, rollout_index, status, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(experiment_id, provider_source, model_api_name, dataset, instance_id, rollout_index) DO UPDATE SET
          model_label = excluded.model_label,
          updated_at = excluded.updated_at
        """,
        (
            experiment_id,
            provider_source,
            model_api_name,
            model_label,
            dataset,
            instance_id,
            rollout_index,
            PENDING_STATUS,
            now,
            now,
        ),
    )
    row = conn.execute(
        """
        SELECT id FROM run_cells
        WHERE experiment_id = ?
          AND provider_source = ?
          AND model_api_name = ?
          AND dataset = ?
          AND instance_id = ?
          AND rollout_index = ?
        """,
        (
            experiment_id,
            provider_source,
            model_api_name,
            dataset,
            instance_id,
            rollout_index,
        ),
    ).fetchone()
    if row is None:
        raise RuntimeError("failed to create or load run cell")
    return int(row["id"])


def _unique_raw_run_id(conn: sqlite3.Connection, requested_run_id: str, cell_id: int) -> str:
    candidate = requested_run_id
    suffix = 1
    while True:
        row = conn.execute(
            "SELECT cell_id FROM raw_rollouts WHERE run_id = ? AND cell_id != ?",
            (candidate, cell_id),
        ).fetchone()
        if row is None:
            return candidate
        suffix += 1
        candidate = f"{requested_run_id}:{cell_id}:{suffix}"


def upsert_rollout_record(
    conn: sqlite3.Connection,
    *,
    experiment_id: str,
    provider_source: str,
    model_api_name: str,
    model_label: str,
    dataset: str,
    record: dict[str, Any],
    detail: dict[str, Any] | None = None,
    artifact_rollouts: Path | str | None = None,
    artifact_details: Path | str | None = None,
) -> int:
    instance_id = str(record.get("instance_id") or (detail or {}).get("instance_id") or "")
    if not instance_id:
        raise ValueError("rollout record has no instance_id; cannot key DB cell")
    rollout_index = int(record.get("rollout_index") or (detail or {}).get("rollout_index") or 0)

    now = utc_now()
    requested_run_id = str(record.get("run_id") or f"{experiment_id}:{provider_source}:{model_api_name}:{dataset}:{instance_id}:{rollout_index}")
    record_error = rollout_record_error(record)
    if record_error and not record.get("error"):
        record = {
            **record,
            "error": record_error,
            "error_kind": record.get("error_kind") or EMPTY_ROLLOUT_ERROR_KIND,
            "error_stage": record.get("error_stage") or "interaction",
            "system_error": True,
        }
    status = ERROR_STATUS if record_error else DONE_STATUS
    # Serialize the large payloads before the first DML statement: the write
    # transaction (and with it the database write lock) opens at the first
    # INSERT/UPDATE, and CPU-bound json work must not extend the lock window.
    if not record_error:
        token_usage = record.get("token_usage") if isinstance(record.get("token_usage"), dict) else {}
        cache_metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
        issue_description = _issue_description(record)
        golden_patch = _golden_patch(record)
        final_patch = _final_patch(record)
        rollout_hash = sha256_text(json_dumps(record))
        stored_rollout_payload = json_dumps(slim_rollout_payload(record))
        messages_payload = json_dumps(record.get("messages") or [])
        trajectory_payload = json_dumps(record.get("trajectory") or [])
        step_traces_payload = json_dumps(record.get("p2a_step_traces") or [])
        reward_payload = json_dumps(record.get("reward"))
        token_usage_payload = json_dumps(token_usage)
        cache_metrics_payload = json_dumps(cache_metrics)
        metrics_payload = json_dumps({"token_usage": token_usage, "cache_metrics": cache_metrics})

    cell_id = _cell_id(
        conn,
        experiment_id=experiment_id,
        provider_source=provider_source,
        model_api_name=model_api_name,
        model_label=model_label,
        dataset=dataset,
        instance_id=instance_id,
        rollout_index=rollout_index,
    )
    run_id = _unique_raw_run_id(conn, requested_run_id, cell_id)
    conn.execute(
        """
        UPDATE run_cells
        SET status = ?,
            attempts = attempts + 1,
            rollout_id = ?,
            run_id = ?,
            artifact_rollouts = ?,
            artifact_details = ?,
            ended_at = ?,
            error = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (
            status,
            str(record.get("rollout_id") or rollout_index),
            run_id,
            str(artifact_rollouts) if artifact_rollouts else None,
            str(artifact_details) if artifact_details else None,
            now,
            record_error,
            now,
            cell_id,
        ),
    )

    if record_error:
        conn.execute("DELETE FROM raw_rollouts WHERE cell_id = ?", (cell_id,))
        conn.execute("DELETE FROM quantitative_metrics WHERE cell_id = ?", (cell_id,))
        return cell_id

    conn.execute("DELETE FROM raw_rollouts WHERE cell_id = ?", (cell_id,))
    conn.execute(
        """
        INSERT INTO raw_rollouts(
          run_id, cell_id, messages_json, trajectory_json, p2a_step_traces_json,
          final_response, reward_json, resolved, token_usage_json, cache_metrics_json,
          issue_description, golden_patch, final_patch, rollout_json, rollout_sha256, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            cell_id,
            messages_payload,
            trajectory_payload,
            step_traces_payload,
            record.get("response_text") or "",
            reward_payload,
            _bool_int(record.get("resolved")),
            token_usage_payload,
            cache_metrics_payload,
            issue_description,
            golden_patch,
            final_patch,
            stored_rollout_payload,
            rollout_hash,
            now,
        ),
    )

    turns = len(record.get("p2a_step_traces") or record.get("trajectory") or [])
    conn.execute(
        """
        INSERT INTO quantitative_metrics(
          cell_id, reward, resolved, p2a_read, call_graph_hit, ground_truth_hit,
          near_hit, min_distance, turns, tool_calls, wall_time, input_tokens,
          output_tokens, reasoning_tokens, cache_hit_tokens, cache_write_tokens,
          cost, fingerprint, metrics_json, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(cell_id) DO UPDATE SET
          reward = excluded.reward,
          resolved = excluded.resolved,
          p2a_read = excluded.p2a_read,
          call_graph_hit = excluded.call_graph_hit,
          ground_truth_hit = excluded.ground_truth_hit,
          near_hit = excluded.near_hit,
          min_distance = excluded.min_distance,
          turns = excluded.turns,
          tool_calls = excluded.tool_calls,
          wall_time = excluded.wall_time,
          input_tokens = excluded.input_tokens,
          output_tokens = excluded.output_tokens,
          reasoning_tokens = excluded.reasoning_tokens,
          cache_hit_tokens = excluded.cache_hit_tokens,
          cache_write_tokens = excluded.cache_write_tokens,
          cost = excluded.cost,
          fingerprint = excluded.fingerprint,
          metrics_json = excluded.metrics_json,
          updated_at = excluded.updated_at
        """,
        (
            cell_id,
            _reward_number(record),
            _bool_int(record.get("resolved")),
            None,
            None,
            None,
            None,
            None,
            turns,
            _tool_call_count(record),
            _number(record.get("wall_time") or record.get("execution_time")),
            _number(token_usage.get("input_tokens")),
            _number(token_usage.get("output_tokens")),
            _number(token_usage.get("reasoning_tokens")),
            _number(token_usage.get("cache_hit_tokens")),
            _number(token_usage.get("cache_write_tokens")),
            _number(token_usage.get("cost")),
            None,
            metrics_payload,
            now,
        ),
    )
    return cell_id


def ingest_artifacts(
    conn: sqlite3.Connection,
    *,
    experiment_id: str,
    provider_source: str,
    model_api_name: str,
    model_label: str,
    dataset: str,
    rollouts_path: Path,
    details_path: Path | None = None,
) -> int:
    n_records = 0
    for record in iter_records(rollouts_path):
        upsert_rollout_record(
            conn,
            experiment_id=experiment_id,
            provider_source=provider_source,
            model_api_name=model_api_name,
            model_label=model_label,
            dataset=dataset,
            record=record,
            artifact_rollouts=rollouts_path,
            artifact_details=details_path,
        )
        n_records += 1
    return n_records


def _avg(values: list[float | None]) -> float | None:
    real = [value for value in values if value is not None]
    return sum(real) / len(real) if real else None


def _rate(values: list[int | None]) -> float | None:
    real = [value for value in values if value is not None]
    return sum(real) / len(real) if real else None


def _std(values: list[float | int | None]) -> float | None:
    real = [float(value) for value in values if value is not None]
    if not real:
        return None
    mean = sum(real) / len(real)
    return (sum((value - mean) ** 2 for value in real) / len(real)) ** 0.5


def _detail_from_metric_row(row: Mapping[str, Any]) -> dict[str, Any]:
    # Hydrated rows carry the parsed detail so avg@k passes do not re-parse the
    # same metrics_json blob once per k.
    if isinstance(row, dict) and "_parsed_detail" in row:
        return row["_parsed_detail"]
    metrics = json_loads(row["metrics_json"], {})
    detail = metrics.get("detail") if isinstance(metrics, dict) else None
    return _sync_path_aliases(detail) if isinstance(detail, dict) else {}


def _detail_number(detail: dict[str, Any], key: str) -> float | None:
    return _number(detail.get(key))


def _detail_bool(detail: dict[str, Any], key: str) -> int | None:
    return _bool_int(detail.get(key))


def _avg_detail(details: list[dict[str, Any]], key: str) -> float | None:
    return _avg([_detail_number(detail, key) for detail in details])


def _rate_detail(details: list[dict[str, Any]], key: str) -> float | None:
    return _rate([_detail_bool(detail, key) for detail in details])


def _sum_detail(details: list[dict[str, Any]], key: str) -> int:
    total = 0
    for detail in details:
        value = detail.get(key)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int | float):
            total += int(value)
    return total


def _path_value(detail: dict[str, Any], path_key: str, legacy_key: str, default: Any = None) -> Any:
    _sync_path_aliases(detail)
    return detail.get(path_key, detail.get(legacy_key, default))


def _path_projection(detail: dict[str, Any]) -> dict[str, Any]:
    projection = _path_value(detail, "path_projection", "chain_projection", {})
    return projection if isinstance(projection, dict) else {}


def _path_node_precision(detail: dict[str, Any]) -> float | None:
    projection = _path_projection(detail)
    path_nodes = projection.get("path_nodes", projection.get("chain_nodes", []))
    context_nodes = projection.get("context_nodes") or []
    if not isinstance(path_nodes, list) or not isinstance(context_nodes, list):
        return None
    hit_path = sum(1 for node in path_nodes if isinstance(node, dict) and node.get("hit"))
    hit_context = sum(1 for node in context_nodes if isinstance(node, dict) and node.get("hit"))
    denom = hit_path + hit_context
    return (hit_path / denom) if denom else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    denom = precision + recall
    return 2 * precision * recall / denom if denom else 0.0


def _path_node_f1(detail: dict[str, Any]) -> float | None:
    return _f1(
        _path_node_precision(detail),
        _number(_path_value(detail, "path_node_recall", "chain_node_recall")),
    )


def _path_read_f1(detail: dict[str, Any]) -> float | None:
    stored = _detail_number(detail, "path_read_f1")
    if stored is not None:
        return stored
    return _f1(
        _number(_path_value(detail, "path_read_precision", "chain_read_precision")),
        _number(_path_value(detail, "path_node_recall", "chain_node_recall")),
    )


def _unique_graph_precision(detail: dict[str, Any]) -> float | None:
    stored = _detail_number(detail, "unique_graph_precision")
    if stored is not None:
        return stored
    return _detail_ratio(detail, "n_unique_rewardable_graph_hit_nodes", "n_unique_graph_hit_nodes")


def _unique_graph_f1(detail: dict[str, Any]) -> float | None:
    stored = _detail_number(detail, "unique_graph_f1")
    if stored is not None:
        return stored
    return _f1(_unique_graph_precision(detail), _detail_number(detail, "hit_recall"))


def _detail_distribution(details: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for detail in details:
        value = detail.get(key)
        if value is not None:
            counts[str(value)] += 1
    return dict(sorted(counts.items()))


def _detail_case_type(detail: dict[str, Any]) -> str:
    return canonical_detail_case_type(detail)


def _is_path_metric_detail(detail: dict[str, Any]) -> bool:
    return _path_value(detail, "path_evaluable", "chain_evaluable") is True and _detail_case_type(detail) in PATH_METRIC_CASE_TYPES


def _is_order_metric_detail(detail: dict[str, Any]) -> bool:
    if not _is_path_metric_detail(detail) or _detail_case_type(detail) != LATENT_CASE:
        return False
    projection = detail.get("path_projection") or detail.get("chain_projection") or {}
    return bool(projection.get("path_edges") or projection.get("chain_edges") or [])


def _is_dynamic_traceable_detail(detail: dict[str, Any]) -> bool:
    """Compatibility wrapper for old code paths; prefer _is_path_metric_detail."""
    return _is_path_metric_detail(detail)


def _path_pattern_distribution(details: list[dict[str, Any]], keys: Iterable[str] = PATH_PATTERN_KEYS) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for detail in details:
        patterns = _path_value(detail, "path_pattern_flags", "chain_bad_patterns")
        if not isinstance(patterns, dict):
            continue
        for key in keys:
            if patterns.get(key):
                counts[key] += 1
    return dict(sorted(counts.items()))


def _detail_ratio(detail: dict[str, Any], numerator: str, denominator: str) -> float | None:
    num = _detail_number(detail, numerator)
    den = _detail_number(detail, denominator)
    return (num / den) if num is not None and den else None


AVG_AT_METRIC_KEYS = (
    "resolved_rate",
    "reward_rate",
    "p2a_read_rate",
    "call_graph_hit_rate",
    "ground_truth_hit_rate",
    "near_hit_rate",
    "avg_min_distance",
    "avg_read_precision",
    "avg_unique_graph_precision",
    "avg_node_recall",
    "avg_unique_graph_f1",
    "avg_hit_f1",
    "obs_recall",
    "obs_graph_focus",
    "obs_graph_steps_per_turn",
    "obs_unique_nodes_per_read",
    "obs_unique_nodes_per_turn",
    "obs_unique_node_coverage_per_read",
    "obs_unique_node_coverage_per_turn",
    "obs_new_graph_step_per_read",
    "obs_repeat_only_step_per_read",
    "obs_new_graph_step_rate",
    "obs_repeat_only_step_rate",
    "obs_new_node_exposure_rate",
    "obs_repeat_node_exposure_rate",
    "obs_repeat_node_exposure_per_read_node",
    "obs_repeat_node_exposure_per_turn_node",
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
    "avg_path_read_f1",
    "avg_chain_read_f1",
    "avg_first_anchor_step",
    "avg_first_root_step",
    "avg_steps_anchor_to_root",
    "anchor_before_root_rate",
    "avg_order_score",
    "reverse_order_rate",
    "miracle_rate",
    "avg_miracle_severity",
    "unlicensed_reference_rate",
    "unlicensed_step_rate",
    "unlicensed_trace_rate",
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


def _rollout_metric_values(row: Mapping[str, Any]) -> dict[str, float | int | None]:
    detail = _detail_from_metric_row(row)
    path_metric = _is_path_metric_detail(detail) if detail else False
    order_metric = _is_order_metric_detail(detail) if detail else False
    order_defined = order_metric and detail.get("order_defined") is True
    block_order_defined = order_metric and detail.get("block_order_defined") is True
    bad_patterns = detail.get("bad_patterns") if isinstance(detail.get("bad_patterns"), dict) else {}
    cache_hit = float(row["cache_hit_tokens"] or 0)
    cache_write = float(row["cache_write_tokens"] or 0)
    input_tokens = float(row["input_tokens"] or 0)
    total_blocks = _detail_number(detail, "n_blocks") if path_metric else None
    order_score = _detail_number(detail, "order_score") if order_defined else None
    block_order_score = _detail_number(detail, "block_order_score") if block_order_defined else None
    return {
        "resolved_rate": _bool_int(row["resolved"]),
        "reward_rate": _number(row["reward"]),
        "p2a_read_rate": _bool_int(row["p2a_read"]),
        "call_graph_hit_rate": _bool_int(row["call_graph_hit"]),
        "ground_truth_hit_rate": _bool_int(row["ground_truth_hit"]),
        "near_hit_rate": _bool_int(row["near_hit"]),
        "avg_min_distance": _number(row["min_distance"]),
        "avg_read_precision": _detail_number(detail, "hit_precision") if path_metric else None,
        "avg_unique_graph_precision": _unique_graph_precision(detail) if path_metric else None,
        "avg_node_recall": _detail_number(detail, "hit_recall") if path_metric else None,
        "avg_unique_graph_f1": _unique_graph_f1(detail) if path_metric else None,
        "avg_hit_f1": _detail_number(detail, "hit_f1") if path_metric else None,
        "obs_recall": _detail_number(detail, "obs_recall") if detail.get("obs_graph_evaluable") else None,
        "obs_graph_focus": _detail_number(detail, "obs_graph_focus") if detail.get("obs_graph_evaluable") else None,
        "obs_graph_steps_per_turn": _detail_number(detail, "obs_graph_steps_per_turn") if detail.get("obs_graph_evaluable") else None,
        "obs_unique_nodes_per_read": _detail_number(detail, "obs_unique_nodes_per_read") if detail.get("obs_graph_evaluable") else None,
        "obs_unique_nodes_per_turn": _detail_number(detail, "obs_unique_nodes_per_turn") if detail.get("obs_graph_evaluable") else None,
        "obs_unique_node_coverage_per_read": _detail_number(detail, "obs_unique_node_coverage_per_read") if detail.get("obs_graph_evaluable") else None,
        "obs_unique_node_coverage_per_turn": _detail_number(detail, "obs_unique_node_coverage_per_turn") if detail.get("obs_graph_evaluable") else None,
        "obs_new_graph_step_per_read": _detail_number(detail, "obs_new_graph_step_per_read") if detail.get("obs_graph_evaluable") else None,
        "obs_repeat_only_step_per_read": _detail_number(detail, "obs_repeat_only_step_per_read") if detail.get("obs_graph_evaluable") else None,
        "obs_new_graph_step_rate": _detail_number(detail, "obs_new_graph_step_rate") if detail.get("obs_graph_evaluable") else None,
        "obs_repeat_only_step_rate": _detail_number(detail, "obs_repeat_only_step_rate") if detail.get("obs_graph_evaluable") else None,
        "obs_new_node_exposure_rate": _detail_number(detail, "obs_new_node_exposure_rate") if detail.get("obs_graph_evaluable") else None,
        "obs_repeat_node_exposure_rate": _detail_number(detail, "obs_repeat_node_exposure_rate") if detail.get("obs_graph_evaluable") else None,
        "obs_repeat_node_exposure_per_read_node": _detail_number(detail, "obs_repeat_node_exposure_per_read_node") if detail.get("obs_graph_evaluable") else None,
        "obs_repeat_node_exposure_per_turn_node": _detail_number(detail, "obs_repeat_node_exposure_per_turn_node") if detail.get("obs_graph_evaluable") else None,
        "path_coverage": _bool_int(_path_value(detail, "path_covered", "chain_graph_covered")) if path_metric else None,
        "chain_graph_coverage": _bool_int(_path_value(detail, "path_covered", "chain_graph_covered")) if path_metric else None,
        "path_hit_rate": _bool_int(_path_value(detail, "path_hit", "chain_hit")) if path_metric else None,
        "chain_hit_rate": _bool_int(_path_value(detail, "path_hit", "chain_hit")) if path_metric else None,
        "anchor_hit_rate": _detail_bool(detail, "anchor_hit") if path_metric else None,
        "root_hit_rate": _detail_bool(detail, "root_hit") if path_metric else None,
        "avg_path_node_recall": _number(_path_value(detail, "path_node_recall", "chain_node_recall")) if path_metric else None,
        "avg_chain_node_recall": _number(_path_value(detail, "path_node_recall", "chain_node_recall")) if path_metric else None,
        "avg_path_node_precision": _path_node_precision(detail) if path_metric else None,
        "avg_chain_node_precision": _path_node_precision(detail) if path_metric else None,
        "avg_path_node_f1": _path_node_f1(detail) if path_metric else None,
        "avg_chain_node_f1": _path_node_f1(detail) if path_metric else None,
        "avg_path_read_precision": _number(_path_value(detail, "path_read_precision", "chain_read_precision")) if path_metric else None,
        "avg_chain_read_precision": _number(_path_value(detail, "path_read_precision", "chain_read_precision")) if path_metric else None,
        "avg_path_read_f1": _path_read_f1(detail) if path_metric else None,
        "avg_chain_read_f1": _path_read_f1(detail) if path_metric else None,
        "avg_first_anchor_step": _detail_number(detail, "first_anchor_step") if path_metric else None,
        "avg_first_root_step": _detail_number(detail, "first_root_step") if path_metric else None,
        "avg_steps_anchor_to_root": _detail_number(detail, "steps_anchor_to_root") if path_metric else None,
        "anchor_before_root_rate": _detail_bool(detail, "anchor_before_root") if path_metric else None,
        "avg_order_score": order_score,
        "reverse_order_rate": _bool_int(order_score < 0) if order_score is not None else None,
        "miracle_rate": _bool_int(bool(detail.get("miracle_step")) or bool(detail.get("block_miracle_step"))) if order_metric else None,
        "avg_miracle_severity": _detail_number(detail, "miracle_severity") if order_metric else None,
        "unlicensed_reference_rate": _detail_number(detail, "unlicensed_reference_rate") if detail.get("license_evaluable") else None,
        "unlicensed_step_rate": _detail_number(detail, "unlicensed_step_rate") if detail.get("license_evaluable") else None,
        "unlicensed_trace_rate": _bool_int((detail.get("n_unlicensed_references") or 0) > 0) if detail.get("license_evaluable") else None,
        "avg_block_order_score": block_order_score,
        "block_reverse_order_rate": _bool_int(block_order_score < 0) if block_order_score is not None else None,
        "block_miracle_rate": _bool_int(detail.get("block_miracle_step")) if order_metric and detail.get("block_miracle_step") is not None else None,
        "avg_block_efficiency": _detail_number(detail, "block_efficiency") if path_metric else None,
        "avg_blocks_per_trace": total_blocks if path_metric else None,
        "block_achieve_rate": _detail_ratio(detail, "n_achieving_blocks", "n_scored_read_blocks") if path_metric else None,
        "block_waste_rate": _detail_ratio(detail, "n_wasted_blocks", "n_scored_read_blocks") if path_metric else None,
        "block_loop_rate": _detail_ratio(detail, "n_loop_blocks", "n_blocks") if path_metric else None,
        "achieving_block_step_share": _detail_ratio(detail, "n_achieving_block_steps", "n_scored_read_block_steps") if path_metric else None,
        "wasted_block_step_share": _detail_ratio(detail, "n_wasted_block_steps", "n_scored_read_block_steps") if path_metric else None,
        "loop_block_step_share": _detail_ratio(detail, "n_loop_block_steps", "n_block_steps") if path_metric else None,
        "loop_trace_rate": _bool_int(bad_patterns.get("has_loop")) if bad_patterns else None,
        "error_spiral_rate": _bool_int(bad_patterns.get("error_spiral")) if bad_patterns else None,
        "avg_turns": _number(row["turns"]),
        "avg_tool_calls": _number(row["tool_calls"]),
        "avg_wall_time": _number(row["wall_time"]),
        "avg_input_tokens": _number(row["input_tokens"]),
        "avg_output_tokens": _number(row["output_tokens"]),
        "avg_reasoning_tokens": _number(row["reasoning_tokens"]),
        "cache_hit_rate": (cache_hit / (input_tokens + cache_hit)) if cache_hit and (input_tokens + cache_hit) else None,
        "cache_write_rate": (cache_write / (input_tokens + cache_write)) if cache_write and (input_tokens + cache_write) else None,
    }


def _rollout_n(group: list[Mapping[str, Any]]) -> int:
    if not group:
        return 1
    return max(1, max(int(row["rollout_index"] or 0) for row in group) + 1)


def _rows_by_instance(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    out: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row["instance_id"])].append(row)
    for values in out.values():
        values.sort(key=lambda item: int(item["rollout_index"] or 0))
    return out


def _first_k_rollout_rows(rows: Iterable[Mapping[str, Any]], *, k: int) -> list[Mapping[str, Any]]:
    return [row for row in rows if int(row["rollout_index"] or 0) < k]


def _avg_at_payload(metric_rows: list[Mapping[str, Any]], *, n: int) -> tuple[dict[str, float | None], dict[str, float | None]]:
    by_instance = _rows_by_instance(metric_rows)
    values_by_key: dict[str, list[float | None]] = {key: [] for key in AVG_AT_METRIC_KEYS}
    for rows in by_instance.values():
        selected = _first_k_rollout_rows(rows, k=n)
        rollout_values = [_rollout_metric_values(row) for row in selected]
        for key in AVG_AT_METRIC_KEYS:
            values_by_key[key].append(_avg([values.get(key) for values in rollout_values]))
    means = {key: _avg(values) for key, values in values_by_key.items()}
    stds = {key: _std(values) for key, values in values_by_key.items()}
    return means, stds


def _avg_at_payloads(
    metric_rows: list[Mapping[str, Any]],
    *,
    rollout_n: int,
) -> tuple[dict[str, dict[str, float | None]], dict[str, dict[str, float | None]]]:
    avg_at: dict[str, dict[str, float | None]] = {}
    avg_at_std: dict[str, dict[str, float | None]] = {}
    for k in range(1, rollout_n + 1):
        means, stds = _avg_at_payload(metric_rows, n=k)
        avg_at[str(k)] = means
        avg_at_std[str(k)] = stds
    return avg_at, avg_at_std


def _pass_at_payload(metric_rows: list[Mapping[str, Any]], *, rollout_n: int) -> dict[str, float | None]:
    by_instance = _rows_by_instance(metric_rows)
    out: dict[str, float | None] = {}
    for k in range(1, rollout_n + 1):
        values = []
        for rows in by_instance.values():
            selected = _first_k_rollout_rows(rows, k=k)
            if selected:
                values.append(1 if any(_bool_int(row["resolved"]) == 1 for row in selected) else 0)
        out[str(k)] = _rate(values)
    return out


def _k_metric_row(row: sqlite3.Row) -> Mapping[str, Any]:
    if row["status"] == DONE_STATUS:
        return row
    data = {key: row[key] for key in row.keys()}
    data["reward"] = 0.0
    data["resolved"] = 0
    return data


def _hydrate_metric_row(row: sqlite3.Row) -> dict[str, Any]:
    data = {key: row[key] for key in row.keys()}
    if data.get("status") == DONE_STATUS:
        rollout_meta = json_loads(data.get("raw_rollout_json"), {})
        if not isinstance(rollout_meta, dict):
            rollout_meta = {}
        derived_error = rollout_record_error(
            {
                **rollout_meta,
                "error": data.get("error"),
                "response_text": data.get("raw_final_response"),
                "trajectory": json_loads(data.get("raw_trajectory_json"), []),
                "p2a_step_traces": json_loads(data.get("raw_p2a_step_traces_json"), []),
            }
        )
        if derived_error:
            data["status"] = ERROR_STATUS
            data["error"] = derived_error
            data["reward"] = 0.0
            data["resolved"] = 0
    if data.get("reward") is None:
        data["reward"] = _reward_number({"reward": json_loads(data.get("raw_reward_json"), None)})
    if data.get("resolved") is None and data.get("raw_resolved") is not None:
        data["resolved"] = _bool_int(data.get("raw_resolved"))
    if data.get("turns") is None:
        data["turns"] = _raw_turn_count(data.get("raw_p2a_step_traces_json"), data.get("raw_trajectory_json"))
    if data.get("tool_calls") is None:
        data["tool_calls"] = _raw_tool_call_count(data.get("raw_p2a_step_traces_json"))
    token_usage = json_loads(data.get("raw_token_usage_json"), {})
    if not isinstance(token_usage, dict):
        token_usage = {}
    for column, key in (
        ("input_tokens", "input_tokens"),
        ("output_tokens", "output_tokens"),
        ("reasoning_tokens", "reasoning_tokens"),
        ("cache_hit_tokens", "cache_hit_tokens"),
        ("cache_write_tokens", "cache_write_tokens"),
        ("cost", "cost"),
    ):
        if data.get(column) is None:
            data[column] = _number(token_usage.get(key))
    return data


def _selected_scope_from_snapshot(value: str | None) -> dict[str, Any] | None:
    snapshot = json_loads(value, {})
    if not isinstance(snapshot, dict):
        return None
    for candidate in (
        snapshot.get("selected_scope"),
        snapshot.get("scope"),
        (snapshot.get("experiment") or {}).get("scope") if isinstance(snapshot.get("experiment"), dict) else None,
        ((snapshot.get("config") or {}).get("experiment") or {}).get("scope") if isinstance(snapshot.get("config"), dict) else None,
    ):
        if isinstance(candidate, dict) and candidate:
            return candidate
    return None


def aggregate_model_metrics(
    conn: sqlite3.Connection,
    *,
    experiment_id: str | None = None,
    provider_source: str | None = None,
    dataset: str | None = None,
    model_api_name: str | None = None,
    model_label: str | None = None,
    include_detail_metrics: bool = True,
    include_raw_trace_fallback: bool = True,
) -> list[dict[str, Any]]:
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
    where_sql = "WHERE " + " AND ".join(where) if where else ""
    cell_columns = _table_columns(conn, "run_cells")
    metric_columns = _table_columns(conn, "quantitative_metrics")
    rollout_index_sql = "c.rollout_index" if "rollout_index" in cell_columns else "0 AS rollout_index"
    rollout_index_order_sql = "c.rollout_index" if "rollout_index" in cell_columns else "rollout_index"
    rollout_id_sql = "c.rollout_id" if "rollout_id" in cell_columns else "NULL AS rollout_id"
    fingerprint_sql = "q.fingerprint" if "fingerprint" in metric_columns else "NULL"
    metrics_json_sql = "q.metrics_json" if include_detail_metrics else "NULL AS metrics_json"
    raw_exists = _table_exists(conn, "raw_rollouts")
    raw_step_traces_sql = "r.p2a_step_traces_json" if include_raw_trace_fallback and raw_exists else "NULL"
    raw_trajectory_sql = "r.trajectory_json" if include_raw_trace_fallback and raw_exists else "NULL"
    raw_final_response_sql = "r.final_response" if include_raw_trace_fallback and raw_exists else "NULL"
    raw_rollout_json_sql = "r.rollout_json" if include_raw_trace_fallback and raw_exists else "NULL"
    raw_reward_sql = "r.reward_json" if raw_exists else "NULL"
    raw_resolved_sql = "r.resolved" if raw_exists else "NULL"
    raw_token_usage_sql = "r.token_usage_json" if raw_exists else "NULL"
    raw_join_sql = "LEFT JOIN raw_rollouts r ON r.cell_id = c.id" if raw_exists else ""

    rows = conn.execute(
        f"""
        SELECT
          c.instance_id,
          {rollout_index_sql},
          {rollout_id_sql},
          c.experiment_id,
          c.provider_source,
          c.dataset,
          c.model_api_name,
          c.model_label,
          c.status,
          c.error,
          e.config_snapshot,
          q.reward,
          q.resolved,
          q.p2a_read,
          q.call_graph_hit,
          q.ground_truth_hit,
          q.near_hit,
          q.min_distance,
          q.turns,
          q.tool_calls,
          q.wall_time,
          q.input_tokens,
          q.output_tokens,
          q.reasoning_tokens,
          q.cache_hit_tokens,
          q.cache_write_tokens,
          q.cost,
          {fingerprint_sql} AS fingerprint,
          {metrics_json_sql},
          {raw_reward_sql} AS raw_reward_json,
          {raw_resolved_sql} AS raw_resolved,
          {raw_step_traces_sql} AS raw_p2a_step_traces_json,
          {raw_trajectory_sql} AS raw_trajectory_json,
          {raw_final_response_sql} AS raw_final_response,
          {raw_rollout_json_sql} AS raw_rollout_json,
          {raw_token_usage_sql} AS raw_token_usage_json
        FROM run_cells c
        LEFT JOIN experiments e
          ON e.experiment_id = c.experiment_id
         AND e.provider_source = c.provider_source
         AND e.dataset = c.dataset
        LEFT JOIN quantitative_metrics q ON q.cell_id = c.id
        {raw_join_sql}
        {where_sql}
        ORDER BY c.model_label, c.instance_id, {rollout_index_order_sql}
        """,
        params,
    ).fetchall()
    rows = [_hydrate_metric_row(row) for row in rows]
    for row in rows:
        row["_parsed_detail"] = _detail_from_metric_row(row)

    groups: dict[tuple[str, str, str, str, str], list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["experiment_id"]),
            str(row["provider_source"]),
            str(row["dataset"]),
            str(row["model_api_name"]),
            str(row["model_label"]),
        )
        groups[key].append(row)

    out = []
    for (exp_id, source, ds, api_name, label), group in sorted(groups.items(), key=lambda item: item[0][-1]):
        done = [row for row in group if row["status"] == DONE_STATUS]
        errors = [row for row in group if row["status"] == ERROR_STATUS]
        metric_rows = done
        k_metric_rows = [_k_metric_row(row) for row in group if row["status"] in {DONE_STATUS, ERROR_STATUS}]
        cache_scope_rows = [row for row in group if row["status"] == DONE_STATUS]
        detail_cache_ready = sum(1 for row in cache_scope_rows if row.get("fingerprint"))
        detail_cache_pending = max(0, len(cache_scope_rows) - detail_cache_ready)
        rollout_n = _rollout_n(group)
        avg_at, avg_at_std = _avg_at_payloads(k_metric_rows, rollout_n=rollout_n)
        avg_at_n = avg_at.get(str(rollout_n), {})
        avg_at_n_std = avg_at_std.get(str(rollout_n), {})
        pass_at = _pass_at_payload(k_metric_rows, rollout_n=rollout_n)
        distances = [row["min_distance"] for row in metric_rows if row["min_distance"] is not None]
        cache_hit = sum(float(row["cache_hit_tokens"] or 0) for row in metric_rows)
        cache_write = sum(float(row["cache_write_tokens"] or 0) for row in metric_rows)
        input_tokens = sum(float(row["input_tokens"] or 0) for row in metric_rows)
        planned_instances = _rows_by_instance(group)
        done_instances = _rows_by_instance(metric_rows)
        details = [_detail_from_metric_row(row) for row in metric_rows]
        details = [detail for detail in details if detail]
        path_metric_details = [detail for detail in details if _is_path_metric_detail(detail)]
        order_metric_details = [detail for detail in details if _is_order_metric_detail(detail)]
        order_details = [detail for detail in order_metric_details if detail.get("order_defined") is True]
        block_order_details = [detail for detail in order_metric_details if detail.get("block_order_defined") is True]
        obs_graph_details = [detail for detail in details if detail.get("obs_graph_evaluable")]
        license_details = [detail for detail in details if detail.get("license_evaluable")]
        entity_references = _sum_detail(license_details, "n_entity_references")
        unlicensed_references = _sum_detail(license_details, "n_unlicensed_references")
        trace_steps = _sum_detail(license_details, "n_trace_steps")
        unlicensed_reference_steps = _sum_detail(license_details, "n_unlicensed_reference_steps")
        scored_read_blocks = _sum_detail(path_metric_details, "n_scored_read_blocks")
        total_blocks = _sum_detail(path_metric_details, "n_blocks")
        block_steps = _sum_detail(path_metric_details, "n_block_steps")
        scored_block_steps = _sum_detail(path_metric_details, "n_scored_read_block_steps")
        row_payload = {
            "experiment_id": exp_id,
            "provider_source": source,
            "dataset": ds,
            "model_api_name": api_name,
            "model_label": label,
            "target": len(planned_instances),
            "target_rollouts": len(group),
            "done": len(done_instances),
            "done_rollouts": len(done),
            "errors": len(errors),
            "pending": sum(1 for row in group if row["status"] in {PENDING_STATUS, RUNNING_STATUS}),
            "detail_cache_ready_rollouts": detail_cache_ready,
            "detail_cache_pending_rollouts": detail_cache_pending,
            "rollouts_per_instance": rollout_n,
            "selected_scope": _selected_scope_from_snapshot(group[0]["config_snapshot"]),
            "pass_at_n": pass_at.get(str(rollout_n)),
            "pass_at": pass_at,
            "avg_at": avg_at,
            "avg_at_std": avg_at_std,
            "avg_at_n": avg_at_n,
            "avg_at_n_std": avg_at_n_std,
            "std_scale": 1,
            "resolved_rate": _rate([row["resolved"] for row in metric_rows]),
            "reward_rate": _avg([row["reward"] for row in metric_rows]),
            "p2a_read_rate": _rate([row["p2a_read"] for row in metric_rows]),
            "call_graph_hit_rate": _rate([row["call_graph_hit"] for row in metric_rows]),
            "ground_truth_hit_rate": _rate([row["ground_truth_hit"] for row in metric_rows]),
            "near_hit_rate": _rate([row["near_hit"] for row in metric_rows]),
            "avg_min_distance": (sum(float(v) for v in distances) / len(distances)) if distances else None,
            "avg_read_precision": _avg_detail(path_metric_details, "hit_precision"),
            "avg_unique_graph_precision": _avg([_unique_graph_precision(detail) for detail in path_metric_details]),
            "avg_node_recall": _avg_detail(path_metric_details, "hit_recall"),
            "avg_unique_graph_f1": _avg([_unique_graph_f1(detail) for detail in path_metric_details]),
            "avg_hit_f1": _avg_detail(path_metric_details, "hit_f1"),
            "obs_recall": _avg_detail(obs_graph_details, "obs_recall"),
            "obs_graph_focus": _avg_detail(obs_graph_details, "obs_graph_focus"),
            "obs_graph_steps_per_turn": _avg_detail(obs_graph_details, "obs_graph_steps_per_turn"),
            "obs_unique_nodes_per_read": _avg_detail(obs_graph_details, "obs_unique_nodes_per_read"),
            "obs_unique_nodes_per_turn": _avg_detail(obs_graph_details, "obs_unique_nodes_per_turn"),
            "obs_unique_node_coverage_per_read": _avg_detail(obs_graph_details, "obs_unique_node_coverage_per_read"),
            "obs_unique_node_coverage_per_turn": _avg_detail(obs_graph_details, "obs_unique_node_coverage_per_turn"),
            "obs_new_graph_step_per_read": _avg_detail(obs_graph_details, "obs_new_graph_step_per_read"),
            "obs_repeat_only_step_per_read": _avg_detail(obs_graph_details, "obs_repeat_only_step_per_read"),
            "obs_new_graph_step_rate": _avg_detail(obs_graph_details, "obs_new_graph_step_rate"),
            "obs_repeat_only_step_rate": _avg_detail(obs_graph_details, "obs_repeat_only_step_rate"),
            "obs_new_node_exposure_rate": _avg_detail(obs_graph_details, "obs_new_node_exposure_rate"),
            "obs_repeat_node_exposure_rate": _avg_detail(obs_graph_details, "obs_repeat_node_exposure_rate"),
            "obs_repeat_node_exposure_per_read_node": _avg_detail(obs_graph_details, "obs_repeat_node_exposure_per_read_node"),
            "obs_repeat_node_exposure_per_turn_node": _avg_detail(obs_graph_details, "obs_repeat_node_exposure_per_turn_node"),
            "path_coverage": _rate([_bool_int(_path_value(detail, "path_covered", "chain_graph_covered")) for detail in path_metric_details]),
            "chain_graph_coverage": _rate([_bool_int(_path_value(detail, "path_covered", "chain_graph_covered")) for detail in path_metric_details]),
            "path_hit_rate": _rate([_bool_int(_path_value(detail, "path_hit", "chain_hit")) for detail in path_metric_details]),
            "chain_hit_rate": _rate([_bool_int(_path_value(detail, "path_hit", "chain_hit")) for detail in path_metric_details]),
            "anchor_hit_rate": _rate_detail(path_metric_details, "anchor_hit"),
            "root_hit_rate": _rate_detail(path_metric_details, "root_hit"),
            "avg_path_node_recall": _avg([_number(_path_value(detail, "path_node_recall", "chain_node_recall")) for detail in path_metric_details]),
            "avg_chain_node_recall": _avg([_number(_path_value(detail, "path_node_recall", "chain_node_recall")) for detail in path_metric_details]),
            "avg_path_node_precision": _avg([_path_node_precision(detail) for detail in path_metric_details]),
            "avg_chain_node_precision": _avg([_path_node_precision(detail) for detail in path_metric_details]),
            "avg_path_node_f1": _avg([_path_node_f1(detail) for detail in path_metric_details]),
            "avg_chain_node_f1": _avg([_path_node_f1(detail) for detail in path_metric_details]),
            "avg_path_read_precision": _avg([_number(_path_value(detail, "path_read_precision", "chain_read_precision")) for detail in path_metric_details]),
            "avg_chain_read_precision": _avg([_number(_path_value(detail, "path_read_precision", "chain_read_precision")) for detail in path_metric_details]),
            "avg_path_read_f1": _avg([_path_read_f1(detail) for detail in path_metric_details]),
            "avg_chain_read_f1": _avg([_path_read_f1(detail) for detail in path_metric_details]),
            "avg_first_anchor_step": _avg_detail(path_metric_details, "first_anchor_step"),
            "avg_first_root_step": _avg_detail(path_metric_details, "first_root_step"),
            "avg_steps_anchor_to_root": _avg_detail(path_metric_details, "steps_anchor_to_root"),
            "anchor_before_root_rate": _rate_detail(path_metric_details, "anchor_before_root"),
            "avg_order_score": _avg_detail(order_details, "order_score"),
            "reverse_order_rate": _rate([_bool_int((_detail_number(detail, "order_score") or 0) < 0) for detail in order_details if _detail_number(detail, "order_score") is not None]),
            "miracle_rate": _rate([_bool_int(bool(detail.get("miracle_step")) or bool(detail.get("block_miracle_step"))) for detail in order_metric_details if detail.get("miracle_step") is not None or detail.get("block_miracle_step") is not None]),
            "avg_miracle_severity": _avg_detail(order_metric_details, "miracle_severity"),
            "unlicensed_reference_rate": (unlicensed_references / entity_references) if entity_references else None,
            "unlicensed_step_rate": (unlicensed_reference_steps / trace_steps) if trace_steps else None,
            "unlicensed_trace_rate": _rate([_bool_int((detail.get("n_unlicensed_references") or 0) > 0) for detail in license_details]),
            "avg_block_order_score": _avg_detail(block_order_details, "block_order_score"),
            "block_reverse_order_rate": _rate([_bool_int((_detail_number(detail, "block_order_score") or 0) < 0) for detail in block_order_details if _detail_number(detail, "block_order_score") is not None]),
            "block_miracle_rate": _rate([_bool_int(detail.get("block_miracle_step")) for detail in order_metric_details if detail.get("block_miracle_step") is not None]),
            "avg_block_efficiency": _avg_detail(path_metric_details, "block_efficiency"),
            "avg_blocks_per_trace": (total_blocks / len(path_metric_details)) if path_metric_details else None,
            "block_achieve_rate": (_sum_detail(path_metric_details, "n_achieving_blocks") / scored_read_blocks) if scored_read_blocks else None,
            "block_waste_rate": (_sum_detail(path_metric_details, "n_wasted_blocks") / scored_read_blocks) if scored_read_blocks else None,
            "block_loop_rate": (_sum_detail(path_metric_details, "n_loop_blocks") / total_blocks) if total_blocks else None,
            "achieving_block_step_share": (_sum_detail(path_metric_details, "n_achieving_block_steps") / scored_block_steps) if scored_block_steps else None,
            "wasted_block_step_share": (_sum_detail(path_metric_details, "n_wasted_block_steps") / scored_block_steps) if scored_block_steps else None,
            "loop_block_step_share": (_sum_detail(path_metric_details, "n_loop_block_steps") / block_steps) if block_steps else None,
            "loop_trace_rate": _rate_detail([detail.get("bad_patterns") or {} for detail in details], "has_loop"),
            "error_spiral_rate": _rate_detail([detail.get("bad_patterns") or {} for detail in details], "error_spiral"),
            "not_path_evaluable_reasons": _detail_distribution(
                [detail for detail in details if not _path_value(detail, "path_evaluable", "chain_evaluable")],
                "not_path_evaluable_reason",
            ),
            "not_chain_evaluable_reasons": _detail_distribution(
                [detail for detail in details if not _path_value(detail, "path_evaluable", "chain_evaluable")],
                "not_chain_evaluable_reason",
            ),
            "path_pattern_flags": _path_pattern_distribution(details, PATH_PATTERN_KEYS),
            "chain_bad_patterns": _path_pattern_distribution(details, CHAIN_BAD_PATTERN_KEYS),
            "avg_turns": _avg([row["turns"] for row in metric_rows]),
            "avg_tool_calls": _avg([row["tool_calls"] for row in metric_rows]),
            "avg_wall_time": _avg([row["wall_time"] for row in metric_rows]),
            "avg_input_tokens": _avg([row["input_tokens"] for row in metric_rows]),
            "avg_output_tokens": _avg([row["output_tokens"] for row in metric_rows]),
            "avg_reasoning_tokens": _avg([row["reasoning_tokens"] for row in metric_rows]),
            "cache_hit_rate": (cache_hit / (input_tokens + cache_hit)) if cache_hit and (input_tokens + cache_hit) else None,
            "cache_write_rate": (cache_write / (input_tokens + cache_write)) if cache_write and (input_tokens + cache_write) else None,
            "total_cache_write_tokens": cache_write if cache_write else None,
            "total_cost": sum(float(row["cost"] or 0) for row in metric_rows) if metric_rows else None,
        }
        for metric_key, metric_value in avg_at_n.items():
            row_payload[metric_key] = metric_value
            std_value = avg_at_n_std.get(metric_key)
            if std_value is not None:
                row_payload[f"{metric_key}_std"] = std_value
        out.append(row_payload)
    return out
