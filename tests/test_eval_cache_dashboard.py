import json
import io
import pickle
import sqlite3
import time
from http import HTTPStatus

import pytest

from p2a.dashboard_adapter import (
    DashboardRequest,
    _case_filter_model_metrics,
    _looks_like_error,
    _normalize_detail,
    _row_instance_id,
    build_dashboard_snapshot,
    migrate_dashboard_databases,
    read_dashboard_log,
    slim_dashboard_raw_db,
    validated_cached_model_metrics,
    write_dashboard_detail_cache_for_record,
)
from p2a import dashboard_server
from p2a.dashboard_server import write_static_dashboard
from p2a.eval_cache import (
    count_run_data,
    count_run_data_targets,
    default_dashboard_build_db_path,
    delete_run_data,
    ensure_db,
    aggregate_model_metrics,
    json_dumps,
    json_loads,
    mark_cells_running,
    upsert_experiment,
    upsert_planned_cells,
    upsert_rollout_record,
    utc_now,
)


def _write_legacy_raw_detail_cache(conn, *, cell_id, detail, fingerprint, cache_metadata=None):
    """Fabricate the retired raw-DB embedded detail cache to exercise migration paths."""
    row = conn.execute("SELECT metrics_json FROM quantitative_metrics WHERE cell_id = ?", (cell_id,)).fetchone()
    if row is None:
        return
    metrics = json_loads(row["metrics_json"], {})
    if not isinstance(metrics, dict):
        metrics = {}
    metrics["detail"] = dict(detail)
    if cache_metadata is not None:
        metrics["dashboard_detail_cache"] = dict(cache_metadata)
    conn.execute(
        "UPDATE quantitative_metrics SET fingerprint = ?, metrics_json = ?, updated_at = ? WHERE cell_id = ?",
        (fingerprint, json_dumps(metrics), utc_now(), cell_id),
    )


def _rollout(instance_id: str, *, resolved: bool = True):
    return {
        "run_id": f"run-{instance_id}",
        "instance_id": instance_id,
        "data_source": "swebench-hard",
        "model": "dummy-model",
        "extra_info": {
            "data_source": "swebench-hard",
            "tools_kwargs": {
                "reward": {
                    "metadata": {
                        "problem_statement": f"Problem for {instance_id}",
                        "patch": f"diff --git a/a.py b/a.py\n+fixed {instance_id}\n",
                    }
                }
            },
        },
        "messages": [{"role": "user", "content": "fix"}],
        "trajectory": [{"step_idx": 1, "exit_reason": "finished"}],
        "p2a_step_traces": [
            {
                "step_idx": 1,
                "response_text": "inspect",
                "tool_calls": [{"function": {"name": "execute_bash", "arguments": {"command": "cat /testbed/a.py"}}}],
                "tool_results": [],
            }
        ],
        "response_text": "done",
        "reward": 1.0 if resolved else 0.0,
        "resolved": resolved,
        "wall_time": 2.5,
        "token_usage": {
            "input_tokens": 100,
            "output_tokens": 20,
            "reasoning_tokens": 5,
            "cache_hit_tokens": 50,
            "cache_write_tokens": 10,
            "cost": 0.01,
        },
    }


def _set_dataset(record: dict, dataset: str) -> dict:
    record["data_source"] = dataset
    record["dataset"] = dataset
    if isinstance(record.get("extra_info"), dict):
        record["extra_info"]["data_source"] = dataset
    return record


def test_native_task_metadata_is_available_to_eval_cache_and_dashboard(tmp_path):
    db = tmp_path / "traces.sqlite"
    record = _rollout("case-native", resolved=True)
    record.pop("instance_id")
    record["extra_info"]["tools_kwargs"] = {
        "task": {
            "name": "swe_bench",
            "metadata": {
                "instance_id": "case-native",
                "problem_statement": "Native Task Config issue",
                "patch": "diff --git a/a.py b/a.py\n+native\n",
            },
        }
    }

    assert _row_instance_id(record) == "case-native"
    with ensure_db(db) as conn:
        upsert_experiment(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            dataset="swebench-hard",
            config_snapshot={"ok": True},
        )
        upsert_rollout_record(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            record=record,
            detail=_detail("case-native", case_type="direct"),
        )
        conn.commit()
        raw_row = conn.execute(
            "SELECT issue_description, golden_patch FROM raw_rollouts"
        ).fetchone()

    assert raw_row["issue_description"] == "Native Task Config issue"
    assert raw_row["golden_patch"] == "diff --git a/a.py b/a.py\n+native"


def test_dashboard_marks_api_resource_failures_as_errors():
    assert _looks_like_error("RuntimeError: insufficient balance, please recharge")
    assert _looks_like_error("HTTP 429 Too Many Requests")
    assert _looks_like_error("余额不足，请充值")
    assert not _looks_like_error("The code mentions a quota argument in documentation.")


def _detail(instance_id: str, *, case_type: str | None = None):
    detail = {
        "record_index": 0,
        "instance_id": instance_id,
        "data_source": "swebench-hard",
        "has_bonus_map": True,
        "has_step_traces": True,
        "n_reads": 1,
        "hit_call_graph": True,
        "hit_ground_truth": True,
        "hit_near": True,
        "min_distance": 0.0,
        "hit_precision": 1.0,
        "hit_recall": 1.0,
        "hit_f1": 1.0,
        "order_score": 1.0,
        "order_defined": True,
        "miracle_step": False,
        "miracle_severity": 0,
        "block_order_score": 1.0,
        "block_order_defined": True,
        "block_miracle_step": False,
        "block_miracle_severity": 0,
        "path_evaluable": True,
        "chain_evaluable": True,
        "not_path_evaluable_reason": None,
        "not_chain_evaluable_reason": None,
        "path_covered": True,
        "chain_graph_covered": True,
        "path_hit": True,
        "chain_hit": True,
        "anchor_hit": True,
        "root_hit": True,
        "path_node_recall": 1.0,
        "chain_node_recall": 1.0,
        "path_read_precision": 1.0,
        "chain_read_precision": 1.0,
        "first_anchor_step": 0,
        "first_root_step": 0,
        "steps_anchor_to_root": 0,
        "anchor_before_root": True,
        "bad_patterns": {"has_loop": False, "error_spiral": False},
        "path_pattern_flags": {
            "missed_anchor": False,
            "missed_root_after_anchor": False,
            "root_before_anchor": False,
            "chain_stall": False,
            "chain_read_loop": False,
            "off_chain_read_spree": False,
            "error_spiral_on_chain": False,
        },
        "step_details": [
            {
                "step_index": 1,
                "trace_index": 0,
                "family": "view",
                "target_path": "a.py",
                "n_reads": 1,
                "reads": [{"file_path": "a.py", "start_line": 1, "end_line": 999999}],
                "hit_nodes": [{"key": "a.py::root", "node_role": "root_cause"}],
                "min_distance": 0.0,
            }
        ],
        "purpose_blocks": [
            {
                "block_index": 0,
                "family": "view",
                "target_path": "a.py",
                "step_indices": [0],
                "achieved": True,
                "wasted": False,
                "loop": False,
                "n_steps": 1,
                "outcome_defined": True,
                "first_hit_step": 0,
                "min_distance": 0.0,
            }
        ],
        "n_blocks": 1,
        "n_scored_read_blocks": 1,
        "n_achieving_blocks": 1,
        "n_wasted_blocks": 0,
        "n_loop_blocks": 0,
        "n_block_steps": 1,
        "n_scored_read_block_steps": 1,
        "n_achieving_block_steps": 1,
        "n_wasted_block_steps": 0,
        "n_loop_block_steps": 0,
        "block_efficiency": 1.0,
        "path_projection": {
            "anchors": ["a.py::root"],
            "roots": ["a.py::root"],
            "context_nodes": [],
            "graph_context_nodes": [],
            "path_nodes": [
                {
                    "key": "a.py::root",
                    "file_path": "a.py",
                    "start_line": 1,
                    "end_line": 20,
                    "normalized_distance": 0.0,
                    "node_role": "root_cause",
                    "hit": True,
                    "first_step": 0,
                }
            ],
            "path_edges": [],
            "context_edges": [],
            "graph_context_edges": [],
        },
        "chain_bad_patterns": {
            "missed_anchor": False,
            "missed_root_after_anchor": False,
            "root_before_anchor": False,
            "chain_stall": False,
            "chain_read_loop": False,
            "off_chain_read_spree": False,
            "error_spiral_on_chain": False,
        },
        "chain_projection": {
            "anchors": ["a.py::root"],
            "roots": ["a.py::root"],
            "context_nodes": [],
            "chain_nodes": [
                {
                    "key": "a.py::root",
                    "file_path": "a.py",
                    "start_line": 1,
                    "end_line": 20,
                    "normalized_distance": 0.0,
                    "node_role": "root_cause",
                    "hit": True,
                    "first_step": 0,
                }
            ],
            "chain_edges": [],
            "context_edges": [],
        },
    }
    if case_type is not None:
        detail["bonus_case_type"] = case_type
        detail["path_case_kind"] = case_type
        detail["chain_case_kind"] = case_type
    return detail


def _bonus_map(instance_id: str):
    return {
        "instance_id": instance_id,
        "case_type": "direct",
        "traceable": True,
        "selected_issue_anchor_nodes": ["a.py::root"],
        "root_cause_nodes": ["a.py::root"],
        "reward_path_edges": [],
        "call_graph_edges": [],
        "call_graph_nodes": {
            "a.py::root": {
                "file_path": "a.py",
                "start_line": 1,
                "end_line": 20,
                "normalized_distance": 0.0,
                "rewardable": True,
                "node_role": "root_cause",
                "source": "def root():\n    return 1",
            }
        },
    }


def _standard_order_detail(instance_id: str):
    detail = _detail(instance_id, case_type="standard")
    path_projection = {
        "anchors": ["a.py::symptom"],
        "roots": ["a.py::root"],
        "context_nodes": [],
        "graph_context_nodes": [],
        "path_nodes": [
            {
                "key": "a.py::symptom",
                "file_path": "a.py",
                "start_line": 1,
                "end_line": 10,
                "normalized_distance": 1.0,
                "node_role": "symptom",
                "hit": True,
                "first_step": 0,
            },
            {
                "key": "a.py::root",
                "file_path": "a.py",
                "start_line": 11,
                "end_line": 20,
                "normalized_distance": 0.0,
                "node_role": "root_cause",
                "hit": True,
                "first_step": 1,
            },
        ],
        "path_edges": [{"caller": "a.py::symptom", "callee": "a.py::root"}],
        "context_edges": [],
        "graph_context_edges": [],
    }
    detail["path_projection"] = path_projection
    detail["chain_projection"] = {
        "anchors": path_projection["anchors"],
        "roots": path_projection["roots"],
        "context_nodes": path_projection["context_nodes"],
        "chain_nodes": path_projection["path_nodes"],
        "chain_edges": [{"caller": "a.py::symptom", "callee": "a.py::root"}],
        "context_edges": path_projection["context_edges"],
    }
    return detail


def test_normalize_detail_mirrors_path_fields_over_legacy_defaults():
    detail = {
        "instance_id": "case-path-only",
        "path_evaluable": True,
        "not_path_evaluable_reason": None,
        "path_case_kind": "direct",
        "path_covered": True,
        "path_hit": True,
        "path_node_recall": 1.0,
        "path_read_precision": 1.0,
        "n_path_nodes": 1,
        "n_hit_path_nodes": 1,
        "path_pattern_flags": {"missed_anchor": False, "path_read_loop": True},
        "path_projection": {
            "anchors": ["a.py::root"],
            "roots": ["a.py::root"],
            "path_nodes": [{"key": "a.py::root", "hit": True}],
            "path_edges": [],
            "context_nodes": [],
            "context_edges": [],
        },
    }

    normalized = _normalize_detail(detail)

    assert normalized["chain_evaluable"] is True
    assert normalized["not_chain_evaluable_reason"] is None
    assert normalized["chain_case_kind"] == "direct"
    assert normalized["chain_graph_covered"] is True
    assert normalized["chain_hit"] is True
    assert normalized["chain_node_recall"] == 1.0
    assert normalized["chain_read_precision"] == 1.0
    assert normalized["n_chain_nodes"] == 1
    assert normalized["n_hit_chain_nodes"] == 1
    assert normalized["chain_projection"] == normalized["path_projection"]
    assert normalized["chain_projection"]["chain_nodes"] == [{"key": "a.py::root", "hit": True}]
    assert normalized["chain_bad_patterns"]["chain_read_loop"] is True


def test_eval_cache_upserts_cells_without_duplicates(tmp_path):
    db = tmp_path / "traces.sqlite"
    with ensure_db(db) as conn:
        upsert_experiment(
            conn,
            experiment_id="exp",
            provider_source="openai_compatible",
            dataset="swebench-hard",
            config_snapshot={"ok": True},
        )
        upsert_planned_cells(
            conn,
            experiment_id="exp",
            provider_source="openai_compatible",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            instance_ids=["case-1"],
        )
        first = _rollout("case-1")
        second = _rollout("case-1")
        second["run_id"] = "run-case-1-retry"
        for record in (first, second):
            upsert_rollout_record(
                conn,
                experiment_id="exp",
                provider_source="openai_compatible",
                model_api_name="dummy-model",
                model_label="dummy",
                dataset="swebench-hard",
                record=record,
                detail=_detail("case-1"),
            )
        conn.commit()

        cell_count = conn.execute("SELECT COUNT(*) FROM run_cells").fetchone()[0]
        raw_count = conn.execute("SELECT COUNT(*) FROM raw_rollouts").fetchone()[0]
        metrics_count = conn.execute("SELECT COUNT(*) FROM quantitative_metrics").fetchone()[0]

    assert cell_count == 1
    assert raw_count == 1
    assert metrics_count == 1


def test_eval_cache_migrates_v1_cells_to_rollout_schema(tmp_path):
    db = tmp_path / "traces.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE run_cells (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          experiment_id TEXT NOT NULL,
          provider_source TEXT NOT NULL,
          model_api_name TEXT NOT NULL,
          model_label TEXT NOT NULL,
          dataset TEXT NOT NULL,
          instance_id TEXT NOT NULL,
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
          UNIQUE (experiment_id, provider_source, model_api_name, dataset, instance_id)
        );
        INSERT INTO run_cells(
          experiment_id, provider_source, model_api_name, model_label, dataset,
          instance_id, status, created_at, updated_at
        )
        VALUES ('exp', 'internal_api', 'dummy-model', 'dummy', 'swebench-hard', 'case-1', 'done', 'now', 'now');
        """
    )
    conn.commit()
    conn.close()

    with ensure_db(db) as migrated:
        columns = {row["name"] for row in migrated.execute("PRAGMA table_info(run_cells)").fetchall()}
        assert {"rollout_index", "rollout_id"} <= columns
        row = migrated.execute("SELECT instance_id, rollout_index, status FROM run_cells").fetchone()
        assert dict(row) == {"instance_id": "case-1", "rollout_index": 0, "status": "done"}
        upsert_planned_cells(
            migrated,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            instance_ids=["case-1"],
            rollouts_per_instance=2,
        )
        migrated.commit()
        rows = migrated.execute("SELECT rollout_index, status FROM run_cells ORDER BY rollout_index").fetchall()
        assert [(row["rollout_index"], row["status"]) for row in rows] == [(0, "done"), (1, "pending")]


def test_unified_dashboard_snapshot_includes_db_model_metrics(tmp_path):
    db = tmp_path / "traces.sqlite"
    bonus_dir = tmp_path / "bonus"
    bonus_dir.mkdir()
    (bonus_dir / "case-1.json").write_text(json.dumps(_bonus_map("case-1")), encoding="utf-8")
    with ensure_db(db) as conn:
        upsert_experiment(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            dataset="swebench-hard",
            config_snapshot={"ok": True},
        )
        upsert_planned_cells(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            instance_ids=["case-1", "case-2"],
        )
        upsert_rollout_record(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            record=_rollout("case-1", resolved=True),
            detail=_detail("case-1", case_type="direct"),
        )
        conn.commit()

        rows = aggregate_model_metrics(conn, experiment_id="exp")
        raw_row = conn.execute("SELECT issue_description, golden_patch FROM raw_rollouts").fetchone()
        metrics_row = conn.execute(
            """
            SELECT p2a_read, call_graph_hit, ground_truth_hit, near_hit, min_distance, metrics_json
            FROM quantitative_metrics
            """
        ).fetchone()

    assert rows[0]["model_label"] == "dummy"
    assert raw_row["issue_description"] == "Problem for case-1"
    assert raw_row["golden_patch"] == "diff --git a/a.py b/a.py\n+fixed case-1"
    assert metrics_row["p2a_read"] is None
    assert metrics_row["call_graph_hit"] is None
    assert metrics_row["ground_truth_hit"] is None
    assert metrics_row["near_hit"] is None
    assert metrics_row["min_distance"] is None
    assert "detail" not in json.loads(metrics_row["metrics_json"])
    assert rows[0]["target"] == 2
    assert rows[0]["done"] == 1
    assert rows[0]["resolved_rate"] == 1.0
    assert rows[0]["p2a_read_rate"] is None
    assert rows[0]["avg_read_precision"] is None
    assert rows[0]["avg_node_recall"] is None
    assert rows[0]["avg_path_node_precision"] is None
    assert rows[0]["avg_chain_node_precision"] is None
    assert rows[0]["avg_hit_f1"] is None
    assert rows[0]["anchor_hit_rate"] is None
    assert rows[0]["root_hit_rate"] is None
    assert rows[0]["block_achieve_rate"] is None
    assert rows[0]["cache_hit_rate"] == 50 / 150

    snapshot = build_dashboard_snapshot(DashboardRequest(db_path=db, experiment_id="exp", bonus_map_dir=bonus_dir))
    assert snapshot["schema_version"] == "p2a_unified_dashboard_v1"
    assert snapshot["datasets"][0]["dataset"] == "swebench-hard"
    assert snapshot["eval_cells"][0]["experiment_id"] == "exp"
    assert snapshot["experiments"][0]["experiment_id"] == "exp"
    assert snapshot["experiments"][0]["source_kind"] == "third_party_api"
    assert snapshot["model_metrics"][0]["model_label"] == "dummy"
    assert snapshot["model_metrics"][0]["target"] == 2
    assert snapshot["model_metrics"][0]["avg_read_precision"] == 1.0
    assert snapshot["model_metrics"][0]["avg_path_node_precision"] == 1.0
    assert snapshot["model_metrics"][0]["avg_chain_node_precision"] == 1.0
    assert snapshot["path_metric_detail_count"] == 1
    assert snapshot["dynamic_traceable_detail_count"] == 1
    assert snapshot["path_metric_model_metrics"][0]["model_label"] == "dummy"
    assert snapshot["path_metric_model_metrics"][0]["target"] == 1
    assert snapshot["path_metric_model_metrics"][0]["avg_tool_calls"] == 1.0
    assert snapshot["path_metric_model_metrics"][0]["cache_hit_rate"] == 50 / 150
    assert snapshot["dynamic_traceable_model_metrics"][0]["model_label"] == "dummy"
    assert snapshot["dynamic_traceable_model_metrics"][0]["target"] == 1
    assert snapshot["dynamic_traceable_model_metrics"][0]["avg_tool_calls"] == 1.0
    assert snapshot["dynamic_traceable_model_metrics"][0]["cache_hit_rate"] == 50 / 150
    assert snapshot["summary"]["counts"]["n_records"] == 1
    assert snapshot["details"][0]["instance_id"] == "case-1"
    assert snapshot["details"][0]["issue_description"] == "Problem for case-1"
    assert snapshot["details"][0]["golden_patch"] == "diff --git a/a.py b/a.py\n+fixed case-1"
    assert snapshot["details"][0]["experiment_key"] == snapshot["experiments"][0]["experiment_key"]
    assert snapshot["details"][0]["step_inspection"][0]["tool_names"] == ["execute_bash"]
    assert snapshot["details"][0]["step_inspection"][0]["action_family"] == "read"
    assert snapshot["details"][0]["step_inspection"][0]["chat_text"] == "inspect"
    assert snapshot["details"][0]["step_inspection"][0]["parsed_tool_calls"] == [
        {"name": "execute_bash", "arguments": [{"key": "command", "value": "cat /testbed/a.py"}]}
    ]
    assert snapshot["details"][0]["step_inspection"][0]["recovered_reads"][0]["file_path"] == "a.py"
    with ensure_db(db) as conn:
        row = conn.execute("SELECT fingerprint, metrics_json FROM quantitative_metrics").fetchone()
        metrics = json.loads(row["metrics_json"])
        assert row["fingerprint"] is None
        assert "detail" not in metrics
    with sqlite3.connect(default_dashboard_build_db_path(db)) as build_conn:
        build_conn.row_factory = sqlite3.Row
        row = build_conn.execute("SELECT fingerprint, detail_json, cache_metadata_json FROM dashboard_rollout_details").fetchone()
        detail = json.loads(row["detail_json"])
        metadata = json.loads(row["cache_metadata_json"])
        assert detail["root_hit"] is True
        assert metadata["fingerprint"] == row["fingerprint"]


def test_write_time_dashboard_detail_cache_feeds_light_snapshot(tmp_path):
    db = tmp_path / "traces.sqlite"
    bonus_dir = tmp_path / "bonus"
    bonus_dir.mkdir()
    (bonus_dir / "case-1.json").write_text(json.dumps(_bonus_map("case-1")), encoding="utf-8")
    record = _rollout("case-1")

    with ensure_db(db) as conn:
        upsert_experiment(conn, experiment_id="exp", provider_source="internal_api", dataset="swebench-hard", config_snapshot={})
        cell_id = upsert_rollout_record(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            record=record,
        )
        result = write_dashboard_detail_cache_for_record(conn, cell_id=cell_id, record=record, bonus_map_dir=bonus_dir)
        conn.commit()

    assert result["ok"] is True
    with ensure_db(db) as conn:
        row = conn.execute("SELECT fingerprint, metrics_json FROM quantitative_metrics WHERE cell_id = ?", (cell_id,)).fetchone()
        metrics = json.loads(row["metrics_json"])
        assert row["fingerprint"] is None
        assert "detail" not in metrics
        model_row = aggregate_model_metrics(conn, experiment_id="exp")[0]
    assert model_row["root_hit_rate"] is None
    with sqlite3.connect(default_dashboard_build_db_path(db)) as build_conn:
        build_conn.row_factory = sqlite3.Row
        build_detail = build_conn.execute("SELECT fingerprint, detail_json, cache_metadata_json FROM dashboard_rollout_details").fetchone()
        build_metric = build_conn.execute("SELECT metrics_json FROM dashboard_model_metrics").fetchone()
    assert json.loads(build_detail["detail_json"])["root_hit"] is True
    assert json.loads(build_detail["cache_metadata_json"])["fingerprint"] == build_detail["fingerprint"]
    assert json.loads(build_metric["metrics_json"])["root_hit_rate"] == 1.0

    snapshot = build_dashboard_snapshot(
        DashboardRequest(
            db_path=db,
            experiment_id="exp",
            bonus_map_dir=bonus_dir,
            defer_db_scoring=True,
            include_db_raw_details=False,
        )
    )
    assert snapshot["details"] == []
    assert snapshot["eval_cells"][0]["cache_ready"] == 1
    assert snapshot["eval_cells"][0]["cache_pending"] == 0
    assert snapshot["model_metrics"][0]["root_hit_rate"] == 1.0
    with ensure_db(db) as conn:
        cached_metrics = validated_cached_model_metrics(
            conn,
            DashboardRequest(db_path=db, experiment_id="exp", bonus_map_dir=bonus_dir),
        )
    assert cached_metrics[0]["root_hit_rate"] == 1.0


def test_build_db_model_metrics_preserve_avg_at_k_detail_payloads(tmp_path):
    db = tmp_path / "traces.sqlite"
    bonus_dir = tmp_path / "bonus"
    bonus_dir.mkdir()
    (bonus_dir / "case-1.json").write_text(json.dumps(_bonus_map("case-1")), encoding="utf-8")

    miss_record = _rollout("case-1", resolved=False)
    miss_record["run_id"] = "run-case-1-0"
    miss_record["rollout_index"] = 0
    miss_record["p2a_step_traces"][0]["tool_calls"][0]["function"]["arguments"]["command"] = "cat /testbed/b.py"
    hit_record = _rollout("case-1", resolved=True)
    hit_record["run_id"] = "run-case-1-1"
    hit_record["rollout_index"] = 1

    with ensure_db(db) as conn:
        upsert_experiment(conn, experiment_id="exp", provider_source="internal_api", dataset="swebench-hard", config_snapshot={})
        upsert_planned_cells(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            instance_ids=["case-1"],
            rollouts_per_instance=2,
        )
        for record in (miss_record, hit_record):
            cell_id = upsert_rollout_record(
                conn,
                experiment_id="exp",
                provider_source="internal_api",
                model_api_name="dummy-model",
                model_label="dummy",
                dataset="swebench-hard",
                record=record,
            )
            result = write_dashboard_detail_cache_for_record(conn, cell_id=cell_id, record=record, bonus_map_dir=bonus_dir)
            assert result["ok"] is True
        conn.commit()

    with sqlite3.connect(default_dashboard_build_db_path(db)) as build_conn:
        build_conn.row_factory = sqlite3.Row
        build_metric = build_conn.execute("SELECT metrics_json FROM dashboard_model_metrics").fetchone()
    metric = json.loads(build_metric["metrics_json"])
    assert metric["pass_at"] == {"1": 0.0, "2": 1.0}
    assert metric["avg_at"]["1"]["root_hit_rate"] == 0.0
    assert metric["avg_at"]["2"]["root_hit_rate"] == 0.5

    snapshot = build_dashboard_snapshot(
        DashboardRequest(db_path=db, experiment_id="exp", defer_db_scoring=True, include_db_raw_details=False)
    )
    assert snapshot["details"] == []
    assert snapshot["model_metrics"][0]["avg_at"]["1"]["root_hit_rate"] == 0.0
    assert snapshot["model_metrics"][0]["avg_at"]["2"]["root_hit_rate"] == 0.5


def test_deferred_metrics_keep_cached_completed_rollouts_when_one_rollout_is_rerunning(tmp_path):
    db = tmp_path / "traces.sqlite"
    bonus_dir = tmp_path / "bonus"
    bonus_dir.mkdir()
    for instance_id in ("case-1", "case-2"):
        (bonus_dir / f"{instance_id}.json").write_text(json.dumps(_bonus_map(instance_id)), encoding="utf-8")

    with ensure_db(db) as conn:
        upsert_experiment(conn, experiment_id="exp", provider_source="internal_api", dataset="swebench-hard", config_snapshot={})
        for instance_id in ("case-1", "case-2"):
            record = _rollout(instance_id)
            cell_id = upsert_rollout_record(
                conn,
                experiment_id="exp",
                provider_source="internal_api",
                model_api_name="dummy-model",
                model_label="dummy",
                dataset="swebench-hard",
                record=record,
            )
            assert write_dashboard_detail_cache_for_record(conn, cell_id=cell_id, record=record, bonus_map_dir=bonus_dir)["ok"] is True
        conn.execute("UPDATE run_cells SET status = 'running' WHERE instance_id = 'case-2'")
        conn.commit()

    snapshot = build_dashboard_snapshot(
        DashboardRequest(
            db_path=db,
            experiment_id="exp",
            provider_source="internal_api",
            dataset="swebench-hard",
            model_api_name="dummy-model",
            model_label="dummy",
            defer_db_scoring=True,
            include_db_raw_details=False,
        )
    )

    metric = snapshot["model_metrics"][0]
    cell = snapshot["eval_cells"][0]
    assert metric["detail_cache_ready_rollouts"] == 1
    assert metric["detail_cache_pending_rollouts"] == 0
    assert metric["root_hit_rate"] == 1.0
    assert cell["cache_ready"] == 1
    assert cell["cache_pending"] == 0
    assert cell["done_rollouts"] == 1
    assert cell["pending"] == 1


def test_raw_db_stores_one_structured_trace_copy_and_can_slim_legacy_cache(tmp_path):
    db = tmp_path / "traces.sqlite"
    bonus_dir = tmp_path / "bonus"
    bonus_dir.mkdir()
    (bonus_dir / "case-1.json").write_text(json.dumps(_bonus_map("case-1")), encoding="utf-8")
    record = _rollout("case-1")
    with ensure_db(db) as conn:
        upsert_experiment(conn, experiment_id="exp", provider_source="internal_api", dataset="swebench-hard", config_snapshot={})
        cell_id = upsert_rollout_record(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            record=record,
        )
        result = write_dashboard_detail_cache_for_record(conn, cell_id=cell_id, record=record, bonus_map_dir=bonus_dir)
        build_db = default_dashboard_build_db_path(db)
        with sqlite3.connect(build_db) as build_conn:
            build_conn.row_factory = sqlite3.Row
            metadata = json.loads(
                build_conn.execute("SELECT cache_metadata_json FROM dashboard_rollout_details").fetchone()["cache_metadata_json"]
            )
        _write_legacy_raw_detail_cache(
            conn,
            cell_id=cell_id,
            fingerprint=result["fingerprint"],
            detail=result["detail"],
            cache_metadata=metadata,
        )
        row = conn.execute("SELECT rollout_json, messages_json FROM raw_rollouts WHERE cell_id = ?", (cell_id,)).fetchone()
        slim_payload = json.loads(row["rollout_json"])
        assert slim_payload["run_id"] == record["run_id"]
        assert "messages" not in slim_payload
        assert json.loads(row["messages_json"])[0]["content"] == "fix"
        conn.commit()

    summary = slim_dashboard_raw_db(DashboardRequest(db_path=db, experiment_id="exp", bonus_map_dir=bonus_dir))

    assert summary["legacy_metric_rows_slimmed"] == 1
    assert summary["legacy_metric_rows_skipped_unbuilt"] == 0
    assert summary["rollout_json_rows_slimmed"] == 0
    assert summary["metric_bytes_removed"] > 0
    with ensure_db(db) as conn:
        metrics = json.loads(conn.execute("SELECT metrics_json FROM quantitative_metrics WHERE cell_id = ?", (cell_id,)).fetchone()["metrics_json"])
        raw = conn.execute("SELECT fingerprint, rollout_json, messages_json FROM raw_rollouts JOIN quantitative_metrics USING(cell_id)").fetchone()
    assert "detail" not in metrics
    assert "dashboard_detail_cache" not in metrics
    assert raw["fingerprint"] is None
    assert "messages" not in json.loads(raw["rollout_json"])
    assert json.loads(raw["messages_json"])

    snapshot = build_dashboard_snapshot(
        DashboardRequest(db_path=db, experiment_id="exp", bonus_map_dir=bonus_dir, defer_db_scoring=True, include_db_raw_details=True)
    )
    assert snapshot["details"][0]["instance_id"] == "case-1"
    assert snapshot["details"][0]["messages"]
    assert snapshot["details"][0]["root_hit"] is True


def test_dashboard_reads_do_not_migrate_legacy_raw_detail_cache(tmp_path):
    db = tmp_path / "traces.sqlite"
    bonus_dir = tmp_path / "bonus"
    bonus_dir.mkdir()
    (bonus_dir / "case-1.json").write_text(json.dumps(_bonus_map("case-1")), encoding="utf-8")
    record = _rollout("case-1")

    with ensure_db(db) as conn:
        upsert_experiment(conn, experiment_id="exp", provider_source="internal_api", dataset="swebench-hard", config_snapshot={})
        cell_id = upsert_rollout_record(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            record=record,
        )
        result = write_dashboard_detail_cache_for_record(conn, cell_id=cell_id, record=record, bonus_map_dir=bonus_dir)
        conn.commit()

    build_db = default_dashboard_build_db_path(db)
    with sqlite3.connect(build_db) as build_conn:
        build_conn.row_factory = sqlite3.Row
        metadata = json.loads(build_conn.execute("SELECT cache_metadata_json FROM dashboard_rollout_details").fetchone()["cache_metadata_json"])
    with ensure_db(db) as conn:
        _write_legacy_raw_detail_cache(
            conn,
            cell_id=cell_id,
            fingerprint=result["fingerprint"],
            detail=result["detail"],
            cache_metadata=metadata,
        )
        conn.commit()
    build_db.unlink()

    snapshot = build_dashboard_snapshot(
        DashboardRequest(
            db_path=db,
            experiment_id="exp",
            bonus_map_dir=bonus_dir,
            defer_db_scoring=True,
            include_db_raw_details=False,
        )
    )

    assert snapshot["details"] == []
    assert snapshot["model_metrics"][0]["root_hit_rate"] is None
    assert not build_db.exists()

    result = migrate_dashboard_databases(
        DashboardRequest(
            db_path=db,
            experiment_id="exp",
            bonus_map_dir=bonus_dir,
        )
    )

    assert result["legacy_build_rows"] == 1
    with sqlite3.connect(build_db) as build_conn:
        assert build_conn.execute("SELECT COUNT(*) FROM dashboard_rollout_details").fetchone()[0] == 1
    migrated = build_dashboard_snapshot(
        DashboardRequest(
            db_path=db,
            experiment_id="exp",
            bonus_map_dir=bonus_dir,
            defer_db_scoring=True,
            include_db_raw_details=False,
        )
    )
    assert migrated["model_metrics"][0]["root_hit_rate"] == 1.0
    assert migrated["eval_cells"][0]["cache_ready"] == 1


def test_dashboard_deferred_db_snapshot_uses_only_validated_detail_cache(monkeypatch, tmp_path):
    db = tmp_path / "traces.sqlite"
    bonus_dir = tmp_path / "bonus"
    bonus_dir.mkdir()
    bonus_path = bonus_dir / "case-1.json"
    bonus_path.write_text(json.dumps(_bonus_map("case-1")), encoding="utf-8")
    with ensure_db(db) as conn:
        upsert_experiment(conn, experiment_id="exp", provider_source="internal_api", dataset="swebench-hard", config_snapshot={})
        upsert_rollout_record(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            record=_rollout("case-1"),
        )
        conn.commit()

    build_dashboard_snapshot(DashboardRequest(db_path=db, experiment_id="exp", bonus_map_dir=bonus_dir))

    def fail_score(*_args, **_kwargs):
        raise AssertionError("deferred snapshot should never score DB rows")

    monkeypatch.setattr("p2a.dashboard_adapter.score_record", fail_score)
    cached = build_dashboard_snapshot(
        DashboardRequest(db_path=db, experiment_id="exp", bonus_map_dir=bonus_dir, defer_db_scoring=True)
    )
    assert cached["details"][0]["instance_id"] == "case-1"
    assert cached["details"][0].get("dashboard_cache_pending") is not True

    bonus = _bonus_map("case-1")
    bonus["call_graph_nodes"]["a.py::root"]["source"] = "def root():\n    return 2"
    bonus_path.write_text(json.dumps(bonus), encoding="utf-8")
    stale = build_dashboard_snapshot(
        DashboardRequest(db_path=db, experiment_id="exp", bonus_map_dir=bonus_dir, defer_db_scoring=True)
    )
    assert stale["details"][0]["dashboard_cache_pending"] is True


def test_dashboard_metrics_endpoint_rejects_stale_detail_cache(tmp_path):
    db = tmp_path / "traces.sqlite"
    bonus_dir = tmp_path / "bonus"
    bonus_dir.mkdir()
    bonus_path = bonus_dir / "case-1.json"
    bonus_path.write_text(json.dumps(_bonus_map("case-1")), encoding="utf-8")
    record = _rollout("case-1")
    with ensure_db(db) as conn:
        upsert_experiment(conn, experiment_id="exp", provider_source="internal_api", dataset="swebench-hard", config_snapshot={})
        cell_id = upsert_rollout_record(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            record=record,
        )
        write_dashboard_detail_cache_for_record(conn, cell_id=cell_id, record=record, bonus_map_dir=bonus_dir)
        conn.commit()

    handler_type = dashboard_server.make_handler(DashboardRequest(db_path=db, bonus_map_dir=bonus_dir))
    handler, sent = _invoke_get_handler(handler_type, "/api/metrics?dataset=swebench-hard")
    handler.do_GET()

    row = sent["payload"]["model_metrics"][0]
    assert row["root_hit_rate"] == 1.0
    assert row["detail_cache_ready_rollouts"] == 1
    assert row["detail_cache_pending_rollouts"] == 0

    bonus = _bonus_map("case-1")
    bonus["call_graph_nodes"]["a.py::root"]["source"] = "def root():\n    return 2"
    bonus_path.write_text(json.dumps(bonus), encoding="utf-8")

    handler, sent = _invoke_get_handler(handler_type, "/api/metrics?dataset=swebench-hard")
    handler.do_GET()

    stale_row = sent["payload"]["model_metrics"][0]
    assert stale_row["root_hit_rate"] is None
    assert stale_row["detail_cache_ready_rollouts"] == 0
    assert stale_row["detail_cache_pending_rollouts"] == 1


def test_dashboard_deferred_db_snapshot_skips_uncached_scoring(monkeypatch, tmp_path):
    db = tmp_path / "traces.sqlite"
    bonus_dir = tmp_path / "bonus"
    bonus_dir.mkdir()
    (bonus_dir / "case-1.json").write_text(json.dumps(_bonus_map("case-1")), encoding="utf-8")
    with ensure_db(db) as conn:
        upsert_experiment(conn, experiment_id="exp", provider_source="internal_api", dataset="swebench-hard", config_snapshot={})
        upsert_rollout_record(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            record=_rollout("case-1"),
        )
        conn.commit()

    def fail_score(*_args, **_kwargs):
        raise AssertionError("deferred snapshot should not score uncached DB rows")

    monkeypatch.setattr("p2a.dashboard_adapter.score_record", fail_score)
    snapshot = build_dashboard_snapshot(DashboardRequest(db_path=db, experiment_id="exp", bonus_map_dir=bonus_dir, defer_db_scoring=True))

    assert snapshot["details"][0]["instance_id"] == "case-1"
    assert snapshot["details"][0]["dashboard_cache_pending"] is True
    assert snapshot["details"][0]["not_path_evaluable_reason"] == "dashboard_cache_pending"
    assert snapshot["details"][0]["raw_available"] is True
    assert snapshot["details"][0]["messages"]
    assert snapshot["details"][0]["step_inspection"]
    assert snapshot["details"][0]["resolved"] is True
    assert snapshot["eval_cells"][0]["cache_pending"] == 1
    assert snapshot["eval_cells"][0]["trajectory_count"] == 1


def test_dashboard_deferred_db_snapshot_can_skip_raw_trace_payload(monkeypatch, tmp_path):
    db = tmp_path / "traces.sqlite"
    bonus_dir = tmp_path / "bonus"
    bonus_dir.mkdir()
    (bonus_dir / "case-1.json").write_text(json.dumps(_bonus_map("case-1")), encoding="utf-8")
    with ensure_db(db) as conn:
        upsert_experiment(conn, experiment_id="exp", provider_source="internal_api", dataset="swebench-hard", config_snapshot={})
        upsert_rollout_record(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            record=_rollout("case-1"),
        )
        conn.commit()

    def fail_score(*_args, **_kwargs):
        raise AssertionError("deferred snapshot should not score uncached DB rows")

    monkeypatch.setattr("p2a.dashboard_adapter.score_record", fail_score)
    snapshot = build_dashboard_snapshot(
        DashboardRequest(
            db_path=db,
            experiment_id="exp",
            bonus_map_dir=bonus_dir,
            defer_db_scoring=True,
            include_db_raw_details=False,
        )
    )

    assert snapshot["details"] == []
    assert snapshot["model_metrics"][0]["resolved_rate"] == 1.0
    assert snapshot["model_metrics"][0]["pass_at_n"] == 1.0
    assert snapshot["eval_cells"][0]["cache_ready"] == 0
    assert snapshot["eval_cells"][0]["cache_pending"] == 1
    assert snapshot["eval_cells"][0]["target_rollouts"] == 1
    assert snapshot["eval_cells"][0]["done_rollouts"] == 1


def test_dashboard_deferred_db_snapshot_keeps_run_status_counts(tmp_path):
    db = tmp_path / "traces.sqlite"
    with ensure_db(db) as conn:
        upsert_experiment(conn, experiment_id="exp", provider_source="internal_api", dataset="swebench-hard", config_snapshot={})
        upsert_planned_cells(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            instance_ids=["case-1", "case-2", "case-3"],
        )
        upsert_rollout_record(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            record=_rollout("case-1"),
        )
        mark_cells_running(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            dataset="swebench-hard",
            instance_ids=["case-2", "case-3"],
        )
        conn.commit()

    snapshot = build_dashboard_snapshot(
        DashboardRequest(
            db_path=db,
            experiment_id="exp",
            provider_source="internal_api",
            dataset="swebench-hard",
            model_api_name="dummy-model",
            model_label="dummy",
            defer_db_scoring=True,
        )
    )

    row = snapshot["eval_cells"][0]
    assert row["target_rollouts"] == 3
    assert row["done_rollouts"] == 1
    assert row["pending"] == 2


def test_eval_cache_delete_counts_and_cascades(tmp_path):
    db = tmp_path / "traces.sqlite"
    with ensure_db(db) as conn:
        for exp, provider, dataset, instance in [
            ("exp-a", "internal_api", "swebench-hard", "case-1"),
            ("exp-b", "openai_compatible", "swebench-hard", "case-2"),
            ("exp-c", "internal_api", "swebench-pro", "case-3"),
        ]:
            upsert_experiment(conn, experiment_id=exp, provider_source=provider, dataset=dataset, config_snapshot={})
            upsert_rollout_record(
                conn,
                experiment_id=exp,
                provider_source=provider,
                model_api_name="dummy-model",
                model_label="dummy",
                dataset=dataset,
                record=_set_dataset(_rollout(instance), dataset),
            )
        conn.commit()

        assert count_run_data(conn, provider_source="internal_api") == {
            "experiments": 2,
            "run_cells": 2,
            "raw_rollouts": 2,
            "quantitative_metrics": 2,
        }
        counts = delete_run_data(conn, provider_source="internal_api", dataset="swebench-hard")
        assert counts["run_cells"] == 1
        assert conn.execute("SELECT COUNT(*) FROM run_cells").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM raw_rollouts").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM quantitative_metrics").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 2
        with pytest.raises(ValueError):
            count_run_data(conn)


def _invoke_handler(handler_type, path: str, body: dict | None = None, *, cookie: str | None = None):
    handler = object.__new__(handler_type)
    payload = json.dumps(body or {}).encode("utf-8")
    handler.path = path
    handler.headers = {"Content-Length": str(len(payload))}
    if cookie:
        handler.headers["Cookie"] = cookie
    handler.rfile = io.BytesIO(payload)
    sent = {}

    def capture(out_payload, content_type, **kwargs):
        sent["payload"] = json.loads(out_payload.decode("utf-8"))
        sent["content_type"] = content_type
        sent["status"] = kwargs.get("status", HTTPStatus.OK)
        sent["headers"] = kwargs.get("headers") or {}

    handler._send_bytes = capture
    handler.send_error = lambda status, *args: sent.update({"status": status, "payload": {"error": args[0] if args else ""}})
    return handler, sent


def _invoke_get_handler(handler_type, path: str, *, cookie: str | None = None):
    handler = object.__new__(handler_type)
    handler.path = path
    handler.headers = {}
    if cookie:
        handler.headers["Cookie"] = cookie
    sent = {}

    def capture(out_payload, content_type, **kwargs):
        sent["payload"] = json.loads(out_payload.decode("utf-8"))
        sent["content_type"] = content_type
        sent["status"] = kwargs.get("status", HTTPStatus.OK)
        sent["headers"] = kwargs.get("headers") or {}

    handler._send_bytes = capture
    handler.send_error = lambda status, *args: sent.update({"status": status, "payload": {"error": args[0] if args else ""}})
    return handler, sent


def test_dashboard_admin_auth_and_delete_endpoint(tmp_path):
    db = tmp_path / "traces.sqlite"
    writer = ensure_db(db)
    try:
        upsert_experiment(writer, experiment_id="smoke", provider_source="internal_api", dataset="swebench-hard", config_snapshot={})
        upsert_rollout_record(
            writer,
            experiment_id="smoke",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            record=_rollout("case-1"),
        )
        writer.commit()
        assert db.with_name(f"{db.name}-wal").exists()

        handler_type = dashboard_server.make_handler(DashboardRequest(db_path=db), admin_password="pw")
        handler, sent = _invoke_handler(handler_type, "/api/delete/preview", {"targets": [{"experiment_id": "smoke"}]})
        handler.do_POST()
        assert sent["status"] == HTTPStatus.FORBIDDEN

        handler, sent = _invoke_handler(handler_type, "/api/auth/login", {"password": "pw"})
        handler.do_POST()
        assert sent["payload"]["admin"] is True
        cookie = sent["headers"]["Set-Cookie"].split(";", 1)[0]

        handler, sent = _invoke_handler(handler_type, "/api/delete/preview", {"targets": [{"experiment_id": "smoke"}]}, cookie=cookie)
        handler.do_POST()
        assert sent["payload"]["counts"]["run_cells"] == 1
        assert "confirmation_phrase" not in sent["payload"]

        handler, sent = _invoke_handler(handler_type, "/api/delete", {"targets": [{"experiment_id": "smoke"}]}, cookie=cookie)
        handler.do_POST()
        assert sent["payload"]["ok"] is True
        assert "backup_path" not in sent["payload"]
    finally:
        writer.close()
    with ensure_db(db) as conn:
        assert count_run_data_targets(conn, [{"experiment_id": "smoke"}])["run_cells"] == 0


def test_dashboard_admin_rebuild_endpoint_clears_target_detail_cache(monkeypatch, tmp_path):
    db = tmp_path / "traces.sqlite"
    bonus_dir = tmp_path / "bonus"
    bonus_dir.mkdir()
    (bonus_dir / "case-1.json").write_text(json.dumps(_bonus_map("case-1")), encoding="utf-8")
    (bonus_dir / "case-2.json").write_text(json.dumps(_bonus_map("case-2")), encoding="utf-8")
    with ensure_db(db) as conn:
        upsert_experiment(conn, experiment_id="smoke", provider_source="internal_api", dataset="swebench-hard", config_snapshot={})
        upsert_rollout_record(
            conn,
            experiment_id="smoke",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            record=_rollout("case-1"),
        )
        upsert_rollout_record(
            conn,
            experiment_id="smoke",
            provider_source="internal_api",
            model_api_name="other-model",
            model_label="other",
            dataset="swebench-hard",
            record=_rollout("case-2"),
        )
        conn.commit()

    build_dashboard_snapshot(DashboardRequest(db_path=db, experiment_id="smoke", bonus_map_dir=bonus_dir))
    with ensure_db(db) as conn:
        rows = conn.execute("SELECT fingerprint, metrics_json FROM quantitative_metrics ORDER BY cell_id").fetchall()
        assert len(rows) == 2
        assert all(row["fingerprint"] is None for row in rows)
        assert all("detail" not in json.loads(row["metrics_json"]) for row in rows)
    with sqlite3.connect(default_dashboard_build_db_path(db)) as build_conn:
        assert build_conn.execute("SELECT COUNT(*) FROM dashboard_rollout_details").fetchone()[0] == 2

    monkeypatch.setattr(
        dashboard_server,
        "build_dashboard_snapshot",
        lambda _request: {"schema_version": "p2a_unified_dashboard_v1", "details": []},
    )
    handler_type = dashboard_server.make_handler(DashboardRequest(db_path=db, bonus_map_dir=bonus_dir), admin_password="pw")
    handler, sent = _invoke_handler(handler_type, "/api/auth/login", {"password": "pw"})
    handler.do_POST()
    cookie = sent["headers"]["Set-Cookie"].split(";", 1)[0]

    handler, sent = _invoke_handler(
        handler_type,
        "/api/rebuild",
        {"targets": [{"experiment_id": "smoke", "model_api_name": "dummy-model", "model_label": "dummy"}]},
        cookie=cookie,
    )
    handler.do_POST()

    assert sent["payload"]["ok"] is True
    assert sent["payload"]["queued"] is True
    deadline = time.monotonic() + 2.0
    rebuilt_count = untouched_count = None
    while time.monotonic() < deadline:
        with sqlite3.connect(default_dashboard_build_db_path(db)) as build_conn:
            rebuilt_count = build_conn.execute(
                """
                SELECT COUNT(*)
                FROM dashboard_rollout_details
                WHERE model_api_name = 'dummy-model'
                """
            ).fetchone()[0]
            untouched_count = build_conn.execute(
                """
                SELECT COUNT(*)
                FROM dashboard_rollout_details
                WHERE model_api_name = 'other-model'
                """
            ).fetchone()[0]
        if rebuilt_count == 0:
            break
        time.sleep(0.01)
    assert rebuilt_count == 0
    assert untouched_count == 1
    with ensure_db(db) as conn:
        rows = conn.execute("SELECT fingerprint, metrics_json FROM quantitative_metrics ORDER BY cell_id").fetchall()
    assert all(row["fingerprint"] is None for row in rows)
    assert all("detail" not in json.loads(row["metrics_json"]) for row in rows)


def test_dashboard_admin_rebuild_materializes_build_db_metrics(tmp_path):
    db = tmp_path / "traces.sqlite"
    bonus_dir = tmp_path / "bonus"
    bonus_dir.mkdir()
    (bonus_dir / "case-1.json").write_text(json.dumps(_bonus_map("case-1")), encoding="utf-8")
    with ensure_db(db) as conn:
        upsert_experiment(conn, experiment_id="smoke", provider_source="internal_api", dataset="swebench-hard", config_snapshot={})
        upsert_rollout_record(
            conn,
            experiment_id="smoke",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            record=_rollout("case-1"),
        )
        conn.commit()

    handler_type = dashboard_server.make_handler(DashboardRequest(db_path=db, bonus_map_dir=bonus_dir), admin_password="pw")
    handler, sent = _invoke_handler(handler_type, "/api/auth/login", {"password": "pw"})
    handler.do_POST()
    cookie = sent["headers"]["Set-Cookie"].split(";", 1)[0]

    handler, sent = _invoke_handler(handler_type, "/api/rebuild", {}, cookie=cookie)
    handler.do_POST()
    assert sent["payload"]["ok"] is True

    deadline = time.monotonic() + 4.0
    status = {}
    while time.monotonic() < deadline:
        handler, sent = _invoke_get_handler(handler_type, "/api/rebuild/status", cookie=cookie)
        handler.do_GET()
        status = sent["payload"]["status"]
        if not status["active"]:
            break
        time.sleep(0.02)
    assert status["phase"] == "idle"
    assert status["last_error"] is None

    with ensure_db(db) as conn:
        raw_metrics = json.loads(conn.execute("SELECT metrics_json FROM quantitative_metrics").fetchone()["metrics_json"])
    assert "detail" not in raw_metrics
    with sqlite3.connect(default_dashboard_build_db_path(db)) as build_conn:
        build_conn.row_factory = sqlite3.Row
        build_detail = build_conn.execute("SELECT detail_json FROM dashboard_rollout_details").fetchone()
        build_metric = build_conn.execute("SELECT metrics_json FROM dashboard_model_metrics").fetchone()
    assert json.loads(build_detail["detail_json"])["root_hit"] is True
    assert json.loads(build_metric["metrics_json"])["root_hit_rate"] == 1.0

    snapshot = build_dashboard_snapshot(
        DashboardRequest(db_path=db, bonus_map_dir=bonus_dir, defer_db_scoring=True, include_db_raw_details=False)
    )
    assert snapshot["details"] == []
    assert snapshot["model_metrics"][0]["root_hit_rate"] == 1.0

    handler, sent = _invoke_get_handler(handler_type, "/api/metrics?dataset=swebench-hard")
    handler.do_GET()
    assert sent["payload"]["model_metrics"][0]["root_hit_rate"] == 1.0


def test_dashboard_admin_rebuild_endpoint_queues_each_target(monkeypatch, tmp_path):
    db = tmp_path / "traces.sqlite"
    with ensure_db(db) as conn:
        upsert_experiment(conn, experiment_id="smoke", provider_source="internal_api", dataset="swebench-hard", config_snapshot={})
        upsert_rollout_record(
            conn,
            experiment_id="smoke",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            record=_rollout("case-1"),
        )
        upsert_rollout_record(
            conn,
            experiment_id="smoke",
            provider_source="internal_api",
            model_api_name="other-model",
            model_label="other",
            dataset="swebench-hard",
            record=_rollout("case-2"),
        )
        conn.commit()

    calls = []

    def fake_build(request):
        calls.append((request.model_api_name, request.model_label))
        return {"schema_version": "p2a_unified_dashboard_v1", "details": []}

    monkeypatch.setattr(dashboard_server, "build_dashboard_snapshot", fake_build)
    handler_type = dashboard_server.make_handler(DashboardRequest(db_path=db), admin_password="pw")
    handler, sent = _invoke_handler(handler_type, "/api/auth/login", {"password": "pw"})
    handler.do_POST()
    cookie = sent["headers"]["Set-Cookie"].split(";", 1)[0]

    handler, sent = _invoke_handler(
        handler_type,
        "/api/rebuild",
        {
            "targets": [
                {
                    "experiment_id": "smoke",
                    "provider_source": "internal_api",
                    "dataset": "swebench-hard",
                    "model_api_name": "dummy-model",
                    "model_label": "dummy",
                },
                {
                    "experiment_id": "smoke",
                    "provider_source": "internal_api",
                    "dataset": "swebench-hard",
                    "model_api_name": "other-model",
                    "model_label": "other",
                },
            ]
        },
        cookie=cookie,
    )
    handler.do_POST()

    assert sent["payload"]["queued_jobs"] == 2
    deadline = time.monotonic() + 2.0
    while len(calls) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert calls == [("dummy-model", "dummy"), ("other-model", "other")]


def test_model_metrics_read_direct_run_scope_from_experiment_snapshot(tmp_path):
    db = tmp_path / "traces.sqlite"
    scope = {
        "source_size": 500,
        "selected_size_before_window": 17,
        "selected_size": 3,
        "filter": {"case_types": ["latent"], "pattern_computable": True},
    }
    with ensure_db(db) as conn:
        upsert_experiment(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            dataset="swebench-hard",
            config_snapshot={"experiment": {"scope": scope}},
        )
        upsert_planned_cells(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            instance_ids=["case-1"],
        )
        upsert_rollout_record(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            record=_rollout("case-1"),
            detail=_detail("case-1", case_type="direct"),
        )
        conn.commit()

        rows = aggregate_model_metrics(conn, experiment_id="exp")

    assert rows[0]["selected_scope"] == scope


def test_eval_cache_aggregates_pass_at_n_and_avg_at_n(tmp_path):
    db = tmp_path / "traces.sqlite"
    with ensure_db(db) as conn:
        upsert_experiment(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            dataset="swebench-hard",
            config_snapshot={"ok": True},
        )
        upsert_planned_cells(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            instance_ids=["case-1", "case-2"],
            rollouts_per_instance=2,
        )
        for instance_id, resolved_values in {"case-1": [False, True], "case-2": [False, False]}.items():
            for rollout_index, resolved in enumerate(resolved_values):
                record = _rollout(instance_id, resolved=resolved)
                record["run_id"] = f"run-{instance_id}-{rollout_index}"
                record["rollout_index"] = rollout_index
                record["wall_time"] = 1.0 + rollout_index
                upsert_rollout_record(
                    conn,
                    experiment_id="exp",
                    provider_source="internal_api",
                    model_api_name="dummy-model",
                    model_label="dummy",
                    dataset="swebench-hard",
                    record=record,
                )
        conn.commit()

        rows = aggregate_model_metrics(conn, experiment_id="exp")

    row = rows[0]
    assert row["target"] == 2
    assert row["target_rollouts"] == 4
    assert row["done"] == 2
    assert row["done_rollouts"] == 4
    assert row["rollouts_per_instance"] == 2
    assert row["pass_at"] == {"1": 0.0, "2": 0.5}
    assert row["pass_at_n"] == 0.5
    assert row["avg_at"]["1"]["resolved_rate"] == 0.0
    assert row["avg_at"]["2"]["resolved_rate"] == 0.25
    assert row["resolved_rate"] == 0.25
    assert row["resolved_rate_std"] == 0.25
    assert row["avg_wall_time"] == 1.5
    assert row["avg_wall_time_std"] == 0.0

    snapshot = build_dashboard_snapshot(DashboardRequest(db_path=db, experiment_id="exp"))

    metric = snapshot["model_metrics"][0]
    assert metric["pass_at_n"] == 0.5
    assert metric["avg_at"]["1"]["resolved_rate"] == 0.0
    assert metric["resolved_rate"] == 0.25
    assert metric["rollouts_per_instance"] == 2
    assert snapshot["eval_cells"][0]["rollouts_per_instance"] == 2
    details = sorted((detail["instance_id"], detail["rollout_index"]) for detail in snapshot["details"])
    assert details == [("case-1", 0), ("case-1", 1), ("case-2", 0), ("case-2", 1)]

    deferred = build_dashboard_snapshot(
        DashboardRequest(db_path=db, experiment_id="exp", defer_db_scoring=True, include_db_raw_details=False)
    )
    assert deferred["model_metrics"][0]["resolved_rate"] == 0.25
    assert deferred["model_metrics"][0]["pass_at_n"] == 0.5
    assert deferred["eval_cells"][0]["rollouts_per_instance"] == 2


def test_eval_cache_aggregates_from_raw_rollout_when_metric_row_is_hollow(tmp_path):
    db = tmp_path / "traces.sqlite"
    with ensure_db(db) as conn:
        upsert_experiment(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            dataset="swebench-hard",
            config_snapshot={"ok": True},
        )
        upsert_planned_cells(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            instance_ids=["case-1", "case-2"],
        )
        upsert_rollout_record(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            record=_rollout("case-1", resolved=True),
        )
        upsert_rollout_record(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            record=_rollout("case-2", resolved=False),
        )
        conn.execute("UPDATE quantitative_metrics SET resolved = NULL, turns = NULL, tool_calls = NULL")
        conn.commit()

        row = aggregate_model_metrics(conn, experiment_id="exp")[0]

    assert row["resolved_rate"] == 0.5
    assert row["pass_at"] == {"1": 0.5}
    assert row["avg_tool_calls"] == 1.0


def test_dashboard_detail_cache_does_not_create_hollow_metric_row(tmp_path):
    db = tmp_path / "traces.sqlite"
    with ensure_db(db) as conn:
        upsert_experiment(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            dataset="swebench-hard",
            config_snapshot={"ok": True},
        )
        upsert_planned_cells(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            instance_ids=["case-1"],
        )
        cell_id = conn.execute("SELECT id FROM run_cells").fetchone()["id"]
        _write_legacy_raw_detail_cache(
            conn,
            cell_id=cell_id,
            detail={"instance_id": "case-1"},
            fingerprint="fp",
            cache_metadata={"fingerprint": "fp"},
        )
        conn.commit()

        row = conn.execute("SELECT * FROM quantitative_metrics WHERE cell_id = ?", (cell_id,)).fetchone()

    assert row is None


def test_multi_rollout_k_metrics_count_error_attempts(tmp_path):
    db = tmp_path / "traces.sqlite"
    with ensure_db(db) as conn:
        upsert_experiment(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            dataset="swebench-hard",
            config_snapshot={"ok": True},
        )
        upsert_planned_cells(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            instance_ids=["case-1"],
            rollouts_per_instance=2,
        )
        error_record = _rollout("case-1", resolved=False)
        error_record["run_id"] = "run-case-1-0"
        error_record["rollout_index"] = 0
        error_record["error"] = "RuntimeError: shell dropped"
        upsert_rollout_record(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            record=error_record,
        )
        success_record = _rollout("case-1", resolved=True)
        success_record["run_id"] = "run-case-1-1"
        success_record["rollout_index"] = 1
        upsert_rollout_record(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            record=success_record,
        )
        conn.commit()

        row = aggregate_model_metrics(conn, experiment_id="exp")[0]

    assert row["pass_at"] == {"1": 0.0, "2": 1.0}
    assert row["avg_at"]["1"]["resolved_rate"] == 0.0
    assert row["avg_at"]["2"]["resolved_rate"] == 0.5
    assert row["resolved_rate"] == 0.5


def test_aggregate_model_metrics_loads_read_only_v1_cache_without_rollout_index(tmp_path):
    db = tmp_path / "v1.sqlite"
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE experiments(
          experiment_id TEXT,
          provider_source TEXT,
          dataset TEXT,
          config_snapshot TEXT
        );
        CREATE TABLE run_cells(
          id INTEGER PRIMARY KEY,
          experiment_id TEXT,
          provider_source TEXT,
          model_api_name TEXT,
          model_label TEXT,
          dataset TEXT,
          instance_id TEXT,
          status TEXT,
          error TEXT
        );
        CREATE TABLE quantitative_metrics(
          cell_id INTEGER PRIMARY KEY,
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
          metrics_json TEXT
        );
        """
    )
    conn.execute("INSERT INTO experiments VALUES (?, ?, ?, ?)", ("exp", "internal_api", "swebench-hard", "{}"))
    conn.execute(
        "INSERT INTO run_cells VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (1, "exp", "internal_api", "dummy-model", "dummy", "swebench-hard", "case-1", "done", None),
    )
    conn.execute(
        """
        INSERT INTO quantitative_metrics VALUES (
          ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        (1, 1.0, 1, None, None, None, None, None, 1, 1, 1.0, 10, 1, 0, 0, 0, 0.0, "{}"),
    )
    conn.commit()
    conn.close()

    ro = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    ro.row_factory = sqlite3.Row
    try:
        rows = aggregate_model_metrics(ro, experiment_id="exp")
    finally:
        ro.close()

    assert rows[0]["rollouts_per_instance"] == 1
    assert rows[0]["pass_at"] == {"1": 1.0}


def test_dashboard_details_load_legacy_raw_rollouts_without_rollout_sha256(tmp_path):
    db = tmp_path / "legacy.sqlite"
    record = _rollout("case-1")
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE experiments(
          experiment_id TEXT,
          provider_source TEXT,
          dataset TEXT,
          config_snapshot TEXT
        );
        CREATE TABLE run_cells(
          id INTEGER PRIMARY KEY,
          experiment_id TEXT,
          provider_source TEXT,
          model_api_name TEXT,
          model_label TEXT,
          dataset TEXT,
          instance_id TEXT,
          rollout_index INTEGER,
          rollout_id TEXT,
          status TEXT,
          attempts INTEGER,
          run_id TEXT,
          artifact_rollouts TEXT,
          artifact_details TEXT,
          started_at TEXT,
          ended_at TEXT,
          error TEXT,
          created_at TEXT,
          updated_at TEXT
        );
        CREATE TABLE raw_rollouts(
          run_id TEXT PRIMARY KEY,
          cell_id INTEGER,
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
          rollout_json TEXT NOT NULL,
          created_at TEXT
        );
        CREATE TABLE quantitative_metrics(
          cell_id INTEGER PRIMARY KEY,
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
          metrics_json TEXT
        );
        """
    )
    conn.execute("INSERT INTO experiments VALUES (?, ?, ?, ?)", ("exp", "internal_api", "swebench-hard", "{}"))
    conn.execute(
        "INSERT INTO run_cells VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            1,
            "exp",
            "internal_api",
            "dummy-model",
            "dummy",
            "swebench-hard",
            "case-1",
            0,
            None,
            "done",
            1,
            record["run_id"],
            None,
            None,
            None,
            "2026-07-02T00:00:00+00:00",
            None,
            "2026-07-02T00:00:00+00:00",
            "2026-07-02T00:00:00+00:00",
        ),
    )
    conn.execute(
        "INSERT INTO raw_rollouts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            record["run_id"],
            1,
            json.dumps(record["messages"]),
            json.dumps(record["trajectory"]),
            json.dumps(record["p2a_step_traces"]),
            record["response_text"],
            json.dumps(record["reward"]),
            1,
            json.dumps(record["token_usage"]),
            "{}",
            "legacy issue",
            "legacy patch",
            json.dumps(record),
            "2026-07-02T00:00:00+00:00",
        ),
    )
    conn.execute(
        "INSERT INTO quantitative_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (1, 1.0, 1, None, None, None, None, None, 1, 1, 2.5, 100, 20, 5, 50, 10, 0.01, "{}"),
    )
    conn.commit()
    conn.close()

    snapshot = build_dashboard_snapshot(
        DashboardRequest(db_path=db, experiment_id="exp", defer_db_scoring=True, include_db_raw_details=True)
    )

    assert snapshot["details"][0]["instance_id"] == "case-1"
    assert snapshot["details"][0]["dashboard_cache_pending"] is True
    assert snapshot["details"][0]["raw_available"] is True
    assert snapshot["details"][0]["messages"]


def test_migrate_dashboard_databases_backfills_raw_rollout_sha256(tmp_path):
    db = tmp_path / "traces.sqlite"
    with ensure_db(db) as conn:
        upsert_experiment(conn, experiment_id="exp", provider_source="internal_api", dataset="swebench-hard", config_snapshot={})
        upsert_rollout_record(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            record=_rollout("case-1"),
        )
        conn.execute("UPDATE raw_rollouts SET rollout_sha256 = NULL")
        conn.commit()

    # Steady-state init must not rescan the table; the explicit migration path repairs hashes.
    with ensure_db(db) as conn:
        row = conn.execute("SELECT rollout_sha256 FROM raw_rollouts").fetchone()
    assert row["rollout_sha256"] is None

    migrate_dashboard_databases(DashboardRequest(db_path=db, experiment_id="exp"))

    with ensure_db(db) as conn:
        row = conn.execute("SELECT rollout_json, rollout_sha256 FROM raw_rollouts").fetchone()

    assert row["rollout_sha256"]


def test_swebench_pro_dashboard_mixes_p2a_and_resolution_only_cells(tmp_path):
    db = tmp_path / "traces.sqlite"
    bonus_dir = tmp_path / "bonus"
    bonus_dir.mkdir()
    (bonus_dir / "case-python.json").write_text(json.dumps(_bonus_map("case-python")), encoding="utf-8")

    with ensure_db(db) as conn:
        upsert_experiment(
            conn,
            experiment_id="exp-pro",
            provider_source="internal_api",
            dataset="swebench-pro",
            config_snapshot={"dataset": "swebench-pro"},
        )
        upsert_planned_cells(
            conn,
            experiment_id="exp-pro",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-pro",
            instance_ids=["case-python", "case-go"],
        )

        py_record = _set_dataset(_rollout("case-python", resolved=True), "swebench-pro")
        py_record["repo_language"] = "python"
        upsert_rollout_record(
            conn,
            experiment_id="exp-pro",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-pro",
            record=py_record,
        )

        go_record = _set_dataset(_rollout("case-go", resolved=False), "swebench-pro")
        go_record["repo_language"] = "go"
        upsert_rollout_record(
            conn,
            experiment_id="exp-pro",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-pro",
            record=go_record,
        )
        conn.commit()

    snapshot = build_dashboard_snapshot(
        DashboardRequest(
            db_path=db,
            experiment_id="exp-pro",
            dataset="swebench-pro",
            bonus_map_dir=bonus_dir,
        )
    )
    details = {detail["instance_id"]: detail for detail in snapshot["details"]}

    assert snapshot["datasets"][0]["dataset"] == "swebench-pro"
    assert snapshot["model_metrics"][0]["target"] == 2
    assert snapshot["model_metrics"][0]["resolved_rate"] == 0.5
    assert snapshot["path_metric_detail_count"] == 1
    assert snapshot["dynamic_traceable_detail_count"] == 1

    assert details["case-python"]["has_bonus_map"] is True
    assert details["case-python"]["chain_case_kind"] == "direct"
    assert details["case-python"]["hit_ground_truth"] is True
    assert details["case-python"]["resolved"] is True

    assert details["case-go"]["has_bonus_map"] is False
    assert details["case-go"]["not_chain_evaluable_reason"] == "missing_bonus_map"
    assert details["case-go"]["resolved"] is False
    assert details["case-go"]["data_source"] == "swebench-pro"

    paths = write_static_dashboard(tmp_path / "swebench-pro-dashboard", snapshot)
    html = paths["html"].read_text(encoding="utf-8")
    assert "window.__P2A_DASHBOARD_SNAPSHOT__" in html
    assert "swebench-pro" in html


def test_dashboard_fills_issue_and_patch_from_dataset_parquet_when_db_raw_lacks_metadata(tmp_path):
    pd = pytest.importorskip("pandas")
    data_file = tmp_path / "swe_bench_verified_hard.parquet"
    frame = pd.DataFrame(
        [
            {
                "instance_id": "case-1",
                "problem_statement": "Issue from parquet",
                "patch": "diff --git a/a.py b/a.py\n+from parquet\n",
            }
        ]
    )
    try:
        frame.to_parquet(data_file)
    except Exception as exc:  # noqa: BLE001 - optional parquet engines vary by environment
        pytest.skip(f"parquet engine unavailable: {exc}")

    db = tmp_path / "traces.sqlite"
    record = _rollout("case-1", resolved=True)
    record.pop("extra_info", None)
    with ensure_db(db) as conn:
        upsert_experiment(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            dataset="swebench-hard",
            config_snapshot={"ok": True},
        )
        upsert_planned_cells(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            instance_ids=["case-1"],
        )
        upsert_rollout_record(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            record=record,
            detail=_detail("case-1", case_type="direct"),
        )
        conn.commit()

    snapshot = build_dashboard_snapshot(DashboardRequest(db_path=db, dataset="swebench-hard", data_file=data_file))

    assert snapshot["details"][0]["issue_description"] == "Issue from parquet"
    assert snapshot["details"][0]["golden_patch"] == "diff --git a/a.py b/a.py\n+from parquet"


def test_model_metrics_ignore_other_case_bonus_map_fields(tmp_path):
    db = tmp_path / "traces.sqlite"
    with ensure_db(db) as conn:
        upsert_experiment(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            dataset="swebench-hard",
            config_snapshot={"ok": True},
        )
        upsert_planned_cells(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            instance_ids=["case-1", "case-2"],
        )
        upsert_rollout_record(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            record=_rollout("case-1", resolved=True),
            detail=_standard_order_detail("case-1"),
        )
        other = _standard_order_detail("case-2")
        other["bonus_case_type"] = "missing_bonus_map"
        other["path_case_kind"] = "missing_bonus_map"
        other["chain_case_kind"] = "missing_bonus_map"
        other["path_evaluable"] = False
        other["chain_evaluable"] = False
        other["hit_precision"] = 0.0
        other["hit_recall"] = 0.0
        other["hit_f1"] = 0.0
        other["order_score"] = -1.0
        other["miracle_step"] = True
        other["block_order_score"] = -1.0
        other["block_miracle_step"] = True
        upsert_rollout_record(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            record=_rollout("case-2", resolved=False),
            detail=other,
        )
        conn.commit()

        rows = aggregate_model_metrics(conn, experiment_id="exp")

    assert rows[0]["done"] == 2
    assert rows[0]["avg_read_precision"] is None
    assert rows[0]["avg_node_recall"] is None
    assert rows[0]["avg_order_score"] is None
    assert rows[0]["reverse_order_rate"] is None
    assert rows[0]["miracle_rate"] is None


def test_case_filter_metrics_bucket_non_evaluable_legacy_standard_as_others():
    detail = _standard_order_detail("case-1")
    detail["path_evaluable"] = False
    detail["chain_evaluable"] = False
    detail["not_path_evaluable_reason"] = "missing_anchor"
    detail["not_chain_evaluable_reason"] = "missing_anchor"

    rows = _case_filter_model_metrics([detail])

    assert rows["latent"] == []
    assert rows["direct,latent,exposed"] == []
    assert rows["others"][0]["target"] == 1
    assert rows["others"][0]["not_path_evaluable_reasons"] == {"missing_anchor": 1}
    assert rows["others"][0]["not_chain_evaluable_reasons"] == {"missing_anchor": 1}


def test_case_filter_metrics_respect_explicit_latent_detail_without_trace_path_edges():
    detail = _detail("case-1", case_type="latent")
    detail["path_projection"]["anchors"] = ["a.py::symptom"]
    detail["path_projection"]["roots"] = ["a.py::root"]
    detail["path_projection"]["path_edges"] = []

    rows = _case_filter_model_metrics([detail])

    assert rows["latent"][0]["target"] == 1
    assert rows["exposed"] == []


def test_dashboard_does_not_reinfer_miracle_from_stored_first_hit_steps(tmp_path):
    db = tmp_path / "traces.sqlite"
    bonus_dir = tmp_path / "bonus"
    bonus_dir.mkdir()
    (bonus_dir / "case-1.json").write_text(json.dumps(_bonus_map("case-1")), encoding="utf-8")
    detail = _standard_order_detail("case-1")
    detail["first_root_step"] = 1
    detail["first_anchor_step"] = 3
    detail["miracle_step"] = False
    detail["miracle_severity"] = 0
    detail["block_miracle_step"] = False
    detail["block_miracle_severity"] = 0
    detail["step_details"] = [
        {
            "step_index": 2,
            "trace_index": 1,
            "family": "read",
            "target_path": "a.py",
            "n_reads": 1,
            "hit_nodes": [
                {"key": "a.py::symptom", "node_role": "symptom"},
                {"key": "a.py::root", "node_role": "root_cause"},
            ],
        }
    ]
    detail["purpose_blocks"] = [
        {
            "block_index": 0,
            "family": "read",
            "trace_indices": [1],
            "step_indices": [2],
            "achieved": True,
            "outcome_defined": True,
            "n_steps": 1,
        }
    ]

    with ensure_db(db) as conn:
        upsert_experiment(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            dataset="swebench-hard",
            config_snapshot={"ok": True},
        )
        upsert_planned_cells(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            instance_ids=["case-1"],
        )
        cell_id = upsert_rollout_record(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            record=_rollout("case-1", resolved=True),
            detail=detail,
        )
        write_dashboard_detail_cache_for_record(conn, cell_id=cell_id, record=_rollout("case-1", resolved=True), bonus_map_dir=bonus_dir)
        conn.commit()
    with sqlite3.connect(default_dashboard_build_db_path(db)) as build_conn:
        build_conn.execute("UPDATE dashboard_rollout_details SET detail_json = ?", (json.dumps(detail),))
        build_conn.commit()

    snapshot = build_dashboard_snapshot(DashboardRequest(db_path=db, experiment_id="exp", bonus_map_dir=bonus_dir))

    assert snapshot["details"][0]["miracle_step"] is False
    assert snapshot["details"][0]["block_miracle_step"] is False
    assert snapshot["details"][0]["first_anchor_step"] == 2
    assert snapshot["details"][0]["first_root_step"] == 2
    assert snapshot["model_metrics"][0]["miracle_rate"] == 0.0
    assert snapshot["path_metric_model_metrics"][0]["miracle_rate"] == 0.0
    assert snapshot["dynamic_traceable_model_metrics"][0]["miracle_rate"] == 0.0


def test_dashboard_infers_default_bonus_map_dir_and_rescores_db_raw_rollouts(tmp_path):
    artifact_root = tmp_path / "data"
    db = artifact_root / "evals" / "traces.sqlite"
    bonus_dir = artifact_root / "bonus_maps" / "swebench-hard"
    bonus_dir.mkdir(parents=True)
    (bonus_dir / "case-1.json").write_text(json.dumps(_bonus_map("case-1")), encoding="utf-8")
    stale_detail = _detail("case-1", case_type="direct")
    stale_detail["path_projection"]["path_nodes"][0].pop("source", None)
    stale_detail["path_projection"]["path_nodes"][0]["source_preview"] = "truncated\n..."
    stale_detail["chain_projection"]["chain_nodes"][0].pop("source", None)
    stale_detail["chain_projection"]["chain_nodes"][0]["source_preview"] = "truncated\n..."

    with ensure_db(db) as conn:
        upsert_experiment(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            dataset="swebench-hard",
            config_snapshot={"ok": True},
        )
        upsert_planned_cells(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            instance_ids=["case-1"],
        )
        upsert_rollout_record(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            record=_rollout("case-1", resolved=True),
            detail=stale_detail,
        )
        conn.commit()

    snapshot = build_dashboard_snapshot(DashboardRequest(db_path=db, dataset="swebench-hard"))

    assert {"kind": "bonus_map_dir", "path": str(bonus_dir), "dataset": "swebench-hard", "mode": "inferred"} in snapshot["sources"]
    root = snapshot["details"][0]["path_projection"]["path_nodes"][0]
    assert root["source"] == "def root():\n    return 1"
    assert not root["source"].endswith("...")


def test_dashboard_reads_node_source_from_bonus_map_for_stored_details(tmp_path):
    bonus_dir = tmp_path / "bonus"
    bonus_dir.mkdir()
    (bonus_dir / "case-1.json").write_text(json.dumps(_bonus_map("case-1")), encoding="utf-8")
    stale_detail = _detail("case-1", case_type="direct")
    stale_detail["path_projection"]["path_nodes"][0].pop("source", None)
    stale_detail["path_projection"]["path_nodes"][0]["source_preview"] = "truncated\n..."
    stale_detail["chain_projection"]["chain_nodes"][0].pop("source", None)
    stale_detail["chain_projection"]["chain_nodes"][0]["source_preview"] = "truncated\n..."
    details_file = tmp_path / "details.jsonl"
    details_file.write_text(json.dumps(stale_detail) + "\n", encoding="utf-8")

    snapshot = build_dashboard_snapshot(DashboardRequest(details=(details_file,), bonus_map_dir=bonus_dir))

    root = snapshot["details"][0]["path_projection"]["path_nodes"][0]
    assert root["source"] == "def root():\n    return 1"
    assert root["source_preview"] == "def root():\n    return 1"


def test_dashboard_step_inspection_splits_local_think_and_xml_tool_call(tmp_path):
    bonus_dir = tmp_path / "bonus"
    bonus_dir.mkdir()
    (bonus_dir / "case-1.json").write_text(json.dumps(_bonus_map("case-1")), encoding="utf-8")
    record = _rollout("case-1")
    record["p2a_step_traces"] = [
        {
            "step_idx": 1,
            "response_text": (
                "<think>inspect the suspicious file</think>\n"
                "I will open the target.\n"
                "<function=file_editor>\n"
                "<parameter=command>view</parameter>\n"
                "<parameter=path>/testbed/a.py</parameter>\n"
                "<parameter=view_range>[1, 20]</parameter>\n"
                "<parameter=concise>false</parameter>\n"
                "</function>"
            ),
            "tool_calls": [],
            "tool_results": [],
        }
    ]
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text(json.dumps(record) + "\n", encoding="utf-8")

    snapshot = build_dashboard_snapshot(DashboardRequest(rollouts=(rollouts,), bonus_map_dir=bonus_dir))
    step = snapshot["details"][0]["step_inspection"][0]

    assert step["reasoning_text"] == "inspect the suspicious file"
    assert step["chat_text"] == "I will open the target."
    assert step["tool_names"] == ["file_editor"]
    assert step["action_family"] == "read"
    assert step["parsed_tool_calls"] == [
        {
            "name": "file_editor",
            "arguments": [
                {"key": "command", "value": "view"},
                {"key": "path", "value": "/testbed/a.py"},
                {"key": "view_range", "value": [1, 20]},
                {"key": "concise", "value": False},
            ],
        }
    ]


def test_dashboard_step_inspection_marks_root_edits_and_execution_errors(tmp_path):
    bonus_dir = tmp_path / "bonus"
    bonus_dir.mkdir()
    (bonus_dir / "case-1.json").write_text(json.dumps(_bonus_map("case-1")), encoding="utf-8")
    record = _rollout("case-1")
    record["p2a_step_traces"] = [
        {
            "step_idx": 1,
            "response_text": "patch root",
            "tool_calls": [
                {
                    "function": {
                        "name": "str_replace_editor",
                        "arguments": {
                            "command": "str_replace",
                            "path": "/testbed/a.py",
                            "old_str": "return 0",
                            "new_str": "return 1",
                        },
                    }
                }
            ],
            "tool_results": [{"observation": "Traceback: command failed"}],
        }
    ]
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text(json.dumps(record) + "\n", encoding="utf-8")

    snapshot = build_dashboard_snapshot(DashboardRequest(rollouts=(rollouts,), bonus_map_dir=bonus_dir))
    detail = snapshot["details"][0]
    step = detail["step_inspection"][0]

    assert detail["edited_root_cause"] is True
    assert step["action_family"] == "edit"
    assert step["write_actions"] == [{"file_path": "a.py", "start_line": 1, "end_line": 999999, "command": "str_replace"}]
    assert step["edited_root_cause"] is True
    assert step["execution_error"] is True
    assert step["status"] == "error"


def test_dashboard_snapshot_uses_readonly_db_connection_under_writer_lock(tmp_path):
    db = tmp_path / "traces.sqlite"
    with ensure_db(db) as conn:
        upsert_experiment(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            dataset="swebench-hard",
            config_snapshot={"ok": True},
        )
        upsert_planned_cells(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            instance_ids=["case-1"],
        )
        upsert_rollout_record(
            conn,
            experiment_id="exp",
            provider_source="internal_api",
            model_api_name="dummy-model",
            model_label="dummy",
            dataset="swebench-hard",
            record=_rollout("case-1"),
            detail=_detail("case-1"),
        )
        conn.commit()

    writer = sqlite3.connect(db, timeout=0.1)
    try:
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("UPDATE run_cells SET updated_at = updated_at")
        snapshot = build_dashboard_snapshot(DashboardRequest(db_path=db, experiment_id="exp"))
    finally:
        writer.rollback()
        writer.close()

    assert snapshot["model_metrics"][0]["model_label"] == "dummy"
    assert snapshot["details"][0]["instance_id"] == "case-1"


def test_unified_dashboard_keeps_experiments_separate_in_overview(tmp_path):
    db = tmp_path / "traces.sqlite"
    with ensure_db(db) as conn:
        for exp_id, model, case_id in (("exp-a", "model-a", "case-a"), ("exp-b", "model-b", "case-b")):
            upsert_experiment(
                conn,
                experiment_id=exp_id,
                provider_source="internal_api",
                dataset="swebench-hard",
                config_snapshot={"experiment": exp_id},
            )
            upsert_planned_cells(
                conn,
                experiment_id=exp_id,
                provider_source="internal_api",
                model_api_name=model,
                model_label=model,
                dataset="swebench-hard",
                instance_ids=[case_id],
            )
            record = _rollout(case_id)
            record["model"] = model
            upsert_rollout_record(
                conn,
                experiment_id=exp_id,
                provider_source="internal_api",
                model_api_name=model,
                model_label=model,
                dataset="swebench-hard",
                record=record,
                detail=_detail(case_id),
            )
        conn.commit()

    snapshot = build_dashboard_snapshot(DashboardRequest(db_path=db))

    experiment_keys = {row["experiment_key"] for row in snapshot["experiments"]}
    detail_keys = {row["experiment_key"] for row in snapshot["details"]}
    assert len(snapshot["experiments"]) == 2
    assert len(experiment_keys) == 2
    assert detail_keys == experiment_keys
    assert {row["experiment_id"] for row in snapshot["model_metrics"]} == {"exp-a", "exp-b"}


def test_local_training_eval_cells_include_run_step(tmp_path):
    bonus_dir = tmp_path / "bonus"
    bonus_dir.mkdir()
    for instance_id in ("case-1", "case-2"):
        (bonus_dir / f"{instance_id}.json").write_text(json.dumps(_bonus_map(instance_id)), encoding="utf-8")
    rollouts = tmp_path / "rollouts.jsonl"
    records = []
    for step, instance_id in ((10, "case-1"), (20, "case-2")):
        record = _rollout(instance_id)
        record["run_step"] = step
        record["model"] = "trainer-model"
        record["model_label"] = "trainer-model"
        records.append(record)
    rollouts.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    snapshot = build_dashboard_snapshot(DashboardRequest(rollouts=(rollouts,), bonus_map_dir=bonus_dir))

    assert len(snapshot["eval_cells"]) == 2
    assert {cell["run_step"] for cell in snapshot["eval_cells"]} == {10, 20}
    assert len({cell["eval_cell_key"] for cell in snapshot["eval_cells"]}) == 2


def test_dashboard_dataset_distributions_deduplicate_instances_across_models(tmp_path):
    db = tmp_path / "traces.sqlite"
    instance_ids = [f"case-{index:02d}" for index in range(3)]
    with ensure_db(db) as conn:
        for model_index in range(2):
            model = f"model-{model_index}"
            upsert_experiment(
                conn,
                experiment_id="exp",
                provider_source="internal_api",
                dataset="swebench-hard",
                config_snapshot={"experiment": "exp"},
            )
            upsert_planned_cells(
                conn,
                experiment_id="exp",
                provider_source="internal_api",
                model_api_name=model,
                model_label=model,
                dataset="swebench-hard",
                instance_ids=instance_ids,
            )
            for case_id in instance_ids:
                record = _rollout(case_id)
                record["model"] = model
                upsert_rollout_record(
                    conn,
                    experiment_id="exp",
                    provider_source="internal_api",
                    model_api_name=model,
                    model_label=model,
                    dataset="swebench-hard",
                    record=record,
                    detail=_detail(case_id),
                )
        conn.commit()

    snapshot = build_dashboard_snapshot(DashboardRequest(db_path=db))

    assert snapshot["datasets"] == [
        {
            "dataset": "swebench-hard",
            "n_instances": 3,
            "n_eval_cells": 2,
            "n_trajectories": 6,
            "models": [f"model-{index}" for index in range(2)],
            "source_kinds": ["third_party_api"],
        }
    ]
    dist = snapshot["summary"]["distributions_by_dataset"]["swebench-hard"]
    assert dist["n_instances"] == 3
    assert dist["distributions"]["case_types"] == {"missing_bonus_map": 3}


def test_unified_dashboard_handles_empty_db(tmp_path):
    db = tmp_path / "empty.sqlite"
    snapshot = build_dashboard_snapshot(DashboardRequest(db_path=db))

    assert snapshot["model_metrics"] == []
    assert snapshot["summary"]["counts"]["n_records"] == 0
    assert not db.exists()


def test_unified_dashboard_loads_local_uni_agent_run_dir(tmp_path):
    bonus_dir = tmp_path / "bonus"
    bonus_dir.mkdir()
    (bonus_dir / "case-1.json").write_text(json.dumps(_bonus_map("case-1")), encoding="utf-8")

    run_root = tmp_path / "runs"
    run_dir = run_root / "run-case-1"
    run_dir.mkdir(parents=True)
    (run_dir / "run.log").write_text("Beginning environment startup\nSTEP 1\n", encoding="utf-8")
    (run_dir / "interaction_result.json").write_text(
        json.dumps({"messages": [], "trajectory": [], "execution_time": 1.0, "metrics": {}, "reward_score": 1.0}),
        encoding="utf-8",
    )
    with (run_dir / "rollout_cache.pkl").open("wb") as handle:
        pickle.dump(
            {
                "extra_fields": {
                    "instance_id": "case-1",
                    "data_source": "local",
                    "p2a_step_traces": [
                        {
                            "step_idx": 1,
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "execute_bash",
                                        "arguments": {"command": "sed -n '1,20p' /testbed/a.py"},
                                    }
                                }
                            ],
                        }
                    ],
                }
            },
            handle,
        )

    snapshot = build_dashboard_snapshot(DashboardRequest(log_dir=run_root, bonus_map_dir=bonus_dir))

    assert snapshot["runs"][0]["status"] == "completed"
    assert snapshot["experiments"][0]["source_kind"] == "local_inference"
    assert snapshot["details"][0]["instance_id"] == "case-1"
    assert snapshot["details"][0]["root_hit"] is True
    assert snapshot["details"][0]["step_inspection"][0]["tool_names"] == ["execute_bash"]
    assert snapshot["summary"]["counts"]["n_records"] == 1


def test_unified_static_dashboard_writes_html_snapshot_and_assets(tmp_path):
    snapshot = build_dashboard_snapshot(DashboardRequest())
    paths = write_static_dashboard(tmp_path / "dashboard", snapshot)

    html = paths["html"].read_text(encoding="utf-8")
    app = paths["app"].read_text(encoding="utf-8")
    assert "P2A unified dashboard" in html
    assert "Datasets and eval cells" in html
    assert "trace-inspector" in html
    assert "window.__P2A_DASHBOARD_SNAPSHOT__" in html
    assert 'href="styles.css?v=' in html
    assert 'src="app.js?v=' in html
    assert "selectedExperimentKey" in app
    assert "selectedEvalCellKey" in app
    assert "renderGraph" in app
    assert "step_inspection" in app
    assert "detail-toggle" in app
    assert paths["snapshot"].exists()
    assert paths["app"].exists()
    assert paths["css"].exists()


def test_static_dashboard_escapes_embedded_snapshot_script_end(tmp_path):
    out_dir = tmp_path / "dashboard"
    snapshot = {
        "schema_version": "p2a_unified_dashboard_v1",
        "details": [{"issue_description": "</script><script>alert(1)</script>"}],
        "summary": {"counts": {"n_records": 1}},
    }

    paths = write_static_dashboard(out_dir, snapshot)

    html = paths["html"].read_text(encoding="utf-8")
    assert html.count("</script>") == 2
    assert "\\u003c/script\\u003e\\u003cscript\\u003ealert(1)\\u003c/script\\u003e" in html
    assert "</script><script>alert(1)</script>" not in html


def test_snapshot_change_token_tracks_log_dir_child_updates(tmp_path):
    run_dir = tmp_path / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    run_log = run_dir / "run.log"
    run_log.write_text("queued", encoding="utf-8")
    request = DashboardRequest(log_dir=tmp_path / "runs")

    first = dashboard_server._snapshot_change_token(request)
    run_log.write_text("running with a longer status", encoding="utf-8")
    second = dashboard_server._snapshot_change_token(request)

    assert second != first


def test_snapshot_change_token_tracks_explicit_and_inferred_bonus_maps(tmp_path):
    bonus_dir = tmp_path / "bonus"
    bonus_dir.mkdir()
    bonus_path = bonus_dir / "case-1.json"
    bonus_path.write_text(json.dumps(_bonus_map("case-1")), encoding="utf-8")
    explicit_request = DashboardRequest(bonus_map_dir=bonus_dir)

    explicit_first = dashboard_server._snapshot_change_token(explicit_request)
    bonus = _bonus_map("case-1")
    bonus["call_graph_nodes"]["a.py::root"]["source"] = "def root():\n    return 20"
    bonus_path.write_text(json.dumps(bonus), encoding="utf-8")
    explicit_second = dashboard_server._snapshot_change_token(explicit_request)

    assert explicit_second != explicit_first

    artifact_root = tmp_path / "data"
    db = artifact_root / "evals" / "traces.sqlite"
    inferred_bonus_dir = artifact_root / "bonus_maps" / "swebench-hard"
    inferred_bonus_dir.mkdir(parents=True)
    inferred_bonus_path = inferred_bonus_dir / "case-2.json"
    inferred_bonus_path.write_text(json.dumps(_bonus_map("case-2")), encoding="utf-8")
    with ensure_db(db) as conn:
        upsert_experiment(conn, experiment_id="exp", provider_source="internal_api", dataset="swebench-hard", config_snapshot={})
        conn.commit()
    inferred_request = DashboardRequest(db_path=db, dataset="swebench-hard")

    inferred_first = dashboard_server._snapshot_change_token(inferred_request)
    inferred_bonus = _bonus_map("case-2")
    inferred_bonus["call_graph_nodes"]["a.py::root"]["source"] = "def root():\n    return 30"
    inferred_bonus_path.write_text(json.dumps(inferred_bonus), encoding="utf-8")
    inferred_second = dashboard_server._snapshot_change_token(inferred_request)

    assert inferred_second != inferred_first


def test_dashboard_serve_mode_does_not_prebuild_snapshot(monkeypatch, tmp_path):
    def fail_build(_request):
        raise AssertionError("serve mode must load snapshots lazily")

    served = {}

    def fake_serve(request, *, host, port):
        served["db_path"] = request.db_path
        served["host"] = host
        served["port"] = port

    monkeypatch.setattr(dashboard_server, "build_dashboard_snapshot", fail_build)
    monkeypatch.setattr(dashboard_server, "serve_dashboard", fake_serve)

    result = dashboard_server.main(["--db", str(tmp_path / "locked.sqlite")])

    assert result == 0
    assert served == {"db_path": tmp_path / "locked.sqlite", "host": "0.0.0.0", "port": 8766}


def test_dashboard_make_handler_does_not_migrate_on_startup(monkeypatch, tmp_path):
    db = tmp_path / "traces.sqlite"
    with ensure_db(db) as conn:
        conn.commit()

    def fail_migrate(_request):
        raise AssertionError("dashboard startup must not migrate build DB")

    monkeypatch.setattr(dashboard_server, "migrate_dashboard_databases", fail_migrate)

    handler_type = dashboard_server.make_handler(DashboardRequest(db_path=db))

    assert handler_type is not None


def test_dashboard_cli_migrate_build_db_is_explicit(monkeypatch, tmp_path, capsys):
    db = tmp_path / "traces.sqlite"
    calls = []

    def fake_migrate(request):
        calls.append(request.db_path)
        return {"legacy_build_rows": 3}

    monkeypatch.setattr(dashboard_server, "migrate_dashboard_databases", fake_migrate)

    result = dashboard_server.main(["--db", str(db), "--migrate-build-db"])

    assert result == 0
    assert calls == [db]
    assert '"legacy_build_rows": 3' in capsys.readouterr().out


def test_dashboard_cli_slim_raw_db_is_explicit(monkeypatch, tmp_path, capsys):
    db = tmp_path / "traces.sqlite"
    calls = []

    def fake_slim(request, *, vacuum=False):
        calls.append((request.db_path, vacuum))
        return {"legacy_metric_rows_slimmed": 4, "vacuum": vacuum}

    monkeypatch.setattr(dashboard_server, "slim_dashboard_raw_db", fake_slim)

    result = dashboard_server.main(["--db", str(db), "--slim-raw-db", "--vacuum"])

    assert result == 0
    assert calls == [(db, True)]
    assert '"legacy_metric_rows_slimmed": 4' in capsys.readouterr().out


def test_live_dashboard_root_does_not_embed_initial_snapshot(monkeypatch):
    def fail_build(_request):
        raise AssertionError("live root must not build or embed snapshots")

    monkeypatch.setattr(dashboard_server, "build_dashboard_snapshot", fail_build)
    handler_type = dashboard_server.make_handler(DashboardRequest())
    handler = object.__new__(handler_type)
    payloads = []
    handler.path = "/"
    handler._send_bytes = lambda payload, content_type, **_kwargs: payloads.append((payload, content_type))

    handler.do_GET()

    html = payloads[0][0].decode("utf-8")
    assert payloads[0][1] == "text/html; charset=utf-8"
    assert "window.__P2A_DASHBOARD_SNAPSHOT__" not in html
    assert 'src="app.js?' in html


def test_live_dashboard_cold_db_snapshot_is_read_only_deferred(monkeypatch, tmp_path):
    db = tmp_path / "traces.sqlite"
    with ensure_db(db) as conn:
        upsert_experiment(conn, experiment_id="exp", provider_source="internal_api", dataset="swebench-hard", config_snapshot={})
        conn.commit()
    calls = []

    def fake_build(request):
        calls.append(request.defer_db_scoring)
        return {"schema_version": "p2a_unified_dashboard_v1", "details": [], "deferred": request.defer_db_scoring}

    monkeypatch.setattr(dashboard_server, "build_dashboard_snapshot", fake_build)
    handler_type = dashboard_server.make_handler(DashboardRequest(db_path=db))
    handler = object.__new__(handler_type)

    payload = handler._build_or_cached_snapshot()

    assert payload["deferred"] is True
    assert calls == [True]


def test_dashboard_log_reader_rejects_paths_outside_run_dir(tmp_path):
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    (run_dir / "run.log").write_text("inside", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("outside", encoding="utf-8")

    ok = read_dashboard_log(DashboardRequest(log_dir=tmp_path), run_id="run-1", source="run.log")
    assert ok["text"] == "inside"

    with pytest.raises(FileNotFoundError):
        read_dashboard_log(DashboardRequest(log_dir=tmp_path), run_id="run-1", source="../secret.txt")
    with pytest.raises(FileNotFoundError):
        read_dashboard_log(DashboardRequest(log_dir=tmp_path), run_id="run-1", source="/etc/passwd")


def test_dashboard_response_write_ignores_client_disconnect():
    handler_type = dashboard_server.make_handler(DashboardRequest())
    handler = object.__new__(handler_type)
    calls = []

    class ClosedWfile:
        def write(self, _payload):
            raise BrokenPipeError("client closed")

    handler.wfile = ClosedWfile()
    handler.close_connection = False
    handler.send_response = lambda status: calls.append(("status", status))
    handler.send_header = lambda key, value: calls.append(("header", key, value))
    handler.end_headers = lambda: calls.append(("end",))

    handler._send_bytes(b"payload", "application/json")

    assert handler.close_connection is True
    assert calls[0][0] == "status"
    assert ("header", "Cache-Control", "no-store, max-age=0") in calls
    assert ("header", "Pragma", "no-cache") in calls
