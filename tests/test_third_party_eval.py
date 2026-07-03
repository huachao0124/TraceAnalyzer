import asyncio
import json
import sqlite3
from types import SimpleNamespace

import numpy as np

from p2a.core import BonusMapStore
from p2a.eval_fault_localization import score_record
from p2a.eval_cache import aggregate_model_metrics, default_dashboard_build_db_path, ensure_db
from p2a.dashboard_adapter import DashboardRequest, build_dashboard_snapshot
from p2a.third_party_eval import (
    IncrementalRolloutSink,
    _prompt,
    _select_scoped_rows,
    _select_rows,
    apply_cli_overrides,
    build_dump_record,
    build_step_traces,
    cache_rollouts,
    classify_error,
    format_report,
    load_config,
    parse_limit_arg,
    resolve_model_config,
    run_batch,
)


def _row():
    return {
        "prompt": [
            {"role": "system", "content": "You are a SWE agent."},
            {"role": "user", "content": "Fix the bug."},
        ],
        "data_source": "swebench-hard",
        "instance_id": "demo__abc123",
        "extra_info": {
            "data_source": "swebench-hard",
            "instance_id": "demo__abc123",
            "tools_kwargs": {
                "reward": {
                    "name": "swe_bench",
                    "metadata": {"instance_id": "demo__abc123"},
                }
            },
        },
    }


def _interaction_result():
    return {
        "trajectory": [
            SimpleNamespace(
                step_idx=1,
                response="inspect",
                thought="inspect target",
                tool_results=[],
                exit_reason="completed",
            )
        ],
        "messages": [
            {"role": "system", "content": "You are a SWE agent."},
            {"role": "user", "content": "Fix the bug."},
            {
                "role": "assistant",
                "content": "inspect",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "str_replace_editor",
                            "arguments": json.dumps(
                                {
                                    "command": "view",
                                    "path": "/testbed/pkg/demo.py",
                                    "view_range": [1, 5],
                                }
                            ),
                        },
                    }
                ],
            },
        ],
        "rollout_cache": {"metrics": {"generate_sequences": 0.1}},
        "execution_time": 1.2,
    }


def test_resolve_model_config_uses_environment(monkeypatch):
    monkeypatch.setenv("P2A_THIRD_PARTY_BASE_URL", "https://example.test")
    monkeypatch.setenv("P2A_THIRD_PARTY_API_KEY", "secret")
    monkeypatch.setenv("P2A_THIRD_PARTY_MODEL", "demo-model")

    config = load_config(None)
    model_config = resolve_model_config(config)

    assert model_config["base_url"] == "https://example.test"
    assert model_config["api_key"] == "secret"
    assert model_config["model_name"] == "demo-model"


def test_apply_cli_overrides_bounds_smoke_run():
    config = apply_cli_overrides(
        load_config(None),
        SimpleNamespace(
            base_url="https://override.test/v1",
            model_name="override-model",
            model_timeout=30,
            max_tokens=128,
            temperature=0.2,
            max_turns=1,
            action_timeout=15,
            tool_install_timeout=18,
            skip_tool_install=["str_replace_editor"],
            reward_eval_timeout=20,
            rollouts_per_instance=4,
            per_instance_parallelism=2,
        ),
    )

    assert config["model"]["base_url"] == "https://override.test/v1"
    assert config["model"]["model_name"] == "override-model"
    assert config["model"]["timeout"] == 30
    assert config["model"]["sampling_params"]["max_tokens"] == 128
    assert config["model"]["sampling_params"]["temperature"] == 0.2
    assert config["agent"]["interaction"]["max_turns"] == 1
    assert config["agent"]["interaction"]["action_timeout"] == 15
    assert config["agent"]["tool_install_timeout"] == 18
    assert config["agent"]["skip_tool_install_commands"] == ["str_replace_editor"]
    assert config["agent"]["reward_eval_timeout"] == 20
    assert config["experiment"]["rollouts_per_instance"] == 4
    assert config["experiment"]["per_instance_parallelism"] == 2


def test_prompt_accepts_parquet_numpy_array():
    prompt = [
        {"role": "system", "content": "You are a SWE agent."},
        {"role": "user", "content": "Fix the bug."},
    ]

    assert _prompt({"instance_id": "demo__abc123", "prompt": np.array(prompt, dtype=object)}) == prompt


def test_limit_arg_accepts_all_and_numeric_values():
    assert parse_limit_arg("all") is None
    assert parse_limit_arg("ALL") is None
    assert parse_limit_arg("0") == 0
    assert parse_limit_arg("3") == 3


def test_select_rows_treats_none_limit_as_unlimited():
    rows = [{"instance_id": "a"}, {"instance_id": "b"}, {"instance_id": "c"}]

    assert _select_rows(rows, limit=None, offset=1, instance_ids=None) == rows[1:]
    assert _select_rows(rows, limit=1, offset=1, instance_ids=None) == rows[1:2]


def test_select_scoped_rows_applies_bonus_map_filter_before_limit(tmp_path):
    bonus_dir = tmp_path / "bonus"
    bonus_dir.mkdir()
    (bonus_dir / "case-a.json").write_text(
        json.dumps(
            {
                "case_type": "standard",
                "selected_issue_anchor_nodes": ["pkg/root.py::root"],
                "root_cause_nodes": ["pkg/root.py::root"],
            }
        ),
        encoding="utf-8",
    )
    (bonus_dir / "case-b.json").write_text(
        json.dumps(
            {
                "case_type": "standard",
                "selected_issue_anchor_nodes": ["pkg/symptom.py::symptom"],
                "root_cause_nodes": ["pkg/root.py::root"],
                "reward_path_edges": [["pkg/symptom.py::symptom", "pkg/root.py::root"]],
                "call_graph_nodes": {
                    "pkg/symptom.py::symptom": {"normalized_distance": 1.0},
                    "pkg/root.py::root": {"normalized_distance": 0.0},
                },
            }
        ),
        encoding="utf-8",
    )
    rows, scope = _select_scoped_rows(
        [{"instance_id": "case-a"}, {"instance_id": "case-b"}],
        limit=1,
        offset=0,
        instance_ids=None,
        bonus_map_dir=bonus_dir,
        scope_filter_config={"case_type": "latent"},
    )

    assert rows == [{"instance_id": "case-b"}]
    assert scope["source_size"] == 2
    assert scope["selected_size_before_window"] == 1


def test_build_step_traces_preserves_structured_tool_calls():
    traces = build_step_traces(_interaction_result())

    assert traces == [
        {
            "step_idx": 1,
            "response_text": "inspect",
            "thought": "inspect target",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "str_replace_editor",
                        "arguments": {
                            "command": "view",
                            "path": "/testbed/pkg/demo.py",
                            "view_range": [1, 5],
                        },
                    },
                }
            ],
            "tool_results": [],
            "exit_reason": "completed",
        }
    ]


def test_build_step_traces_preserves_internal_api_reasoning_metadata():
    interaction = _interaction_result()
    interaction["messages"][2]["content"] = ""
    interaction["trajectory"][0].response = ""
    interaction["rollout_cache"]["internal_api_assistant_metadata"] = {
        "2": {
            "reasoning_content": "inspect before reading",
            "reasoning_blocks": [{"value": "inspect before reading", "signature": "sig"}],
            "text_blocks": [{"type": "text", "value": "I will open the file."}],
        }
    }

    traces = build_step_traces(interaction)
    record = build_dump_record(
        _row(),
        run_id="run-reasoning",
        model_name="demo-model",
        base_url="https://example.test",
        interaction_result=interaction,
        reward_score=False,
        reward_details={"resolved": False},
    )

    assert traces[0]["response_text"] == ""
    assert traces[0]["reasoning"] == "inspect before reading"
    assert traces[0]["reasoning_content"] == "inspect before reading"
    assert traces[0]["reasoning_blocks"] == [{"value": "inspect before reading", "signature": "sig"}]
    assert traces[0]["text_blocks"] == [{"type": "text", "value": "I will open the file."}]
    assert record["assistant_metadata"]["2"]["reasoning_content"] == "inspect before reading"


def test_classify_error_marks_arl_websocket_forbidden_as_system_error():
    assert classify_error("InvalidStatus: server rejected WebSocket connection: HTTP 403") == "arl_shell_forbidden"


def test_classify_error_marks_image_pull_failure_as_system_error():
    assert (
        classify_error(
            "ImagePullBackOff: ErrImagePull: pull access denied for "
            "pair-diag-cn-guangzhou.cr.volces.com/code/sweap-images:demo"
        )
        == "image_pull_failed"
    )


def test_classify_error_marks_api_resource_failures_as_system_error():
    assert classify_error("RuntimeError: insufficient balance, please recharge") == "api_quota_exhausted"
    assert classify_error("HTTP 429 Too Many Requests") == "api_quota_exhausted"
    assert classify_error("余额不足，请充值") == "api_quota_exhausted"


def test_dump_record_is_readable_by_fault_localization_scorer(tmp_path):
    bonus_dir = tmp_path / "bonus_maps"
    bonus_dir.mkdir()
    (bonus_dir / "demo__abc123.json").write_text(
        json.dumps(
            {
                "instance_id": "demo__abc123",
                "case_type": "direct",
                "traceable": True,
                "selected_issue_anchor_nodes": ["pkg/demo.py::target"],
                "symptom_nodes": [],
                "root_cause_nodes": ["pkg/demo.py::target"],
                "reward_path_edges": [],
                "call_graph_edges": [],
                "call_graph_nodes": {
                    "pkg/demo.py::target": {
                        "file_path": "pkg/demo.py",
                        "start_line": 1,
                        "end_line": 5,
                        "normalized_distance": 0.0,
                        "rewardable": True,
                        "node_role": "root_cause",
                        "source": "def target():\n    return 1",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    record = build_dump_record(
        _row(),
        run_id="run-1",
        model_name="demo-model",
        base_url="https://example.test",
        interaction_result=_interaction_result(),
        reward_score=False,
        reward_details={"resolved": False},
    )

    detail = score_record(
        record,
        index=0,
        bonus_maps=BonusMapStore(str(bonus_dir)),
        tracking_mode="view_and_bash",
        near_threshold=0.5,
        m_max=3.0,
    )

    assert record["schema_version"] == "p2a_third_party_rollout_v1"
    assert record["termination_reason"] == "completed"
    assert detail["has_step_traces"] is True
    assert detail["hit_call_graph"] is True
    assert detail["hit_ground_truth"] is True
    assert detail["chain_evaluable"] is True
    assert detail["chain_case_kind"] == "direct"
    assert detail["anchor_hit"] is True
    assert detail["root_hit"] is True
    assert detail["chain_hit"] is True
    assert detail["steps_anchor_to_root"] is None
    assert detail["anchor_before_root"] is None
    assert detail["chain_projection"]["anchors"] == ["pkg/demo.py::target"]
    assert detail["chain_projection"]["roots"] == ["pkg/demo.py::target"]


def test_format_report_contains_aggregate_and_instance_rows():
    report = format_report(
        {
            "counts": {"n_records": 1},
            "rates": {
                "bonus_map_coverage": 1.0,
                "call_graph_coverage": 1.0,
                "read_rate": 1.0,
                "graph_hit_rate_over_call_graphs": 1.0,
                "ground_truth_hit_rate_over_call_graphs": 1.0,
                "near_hit_rate_over_call_graphs": 1.0,
            },
            "averages": {"avg_min_distance_on_hits": 0.0},
        },
        [
            {
                "instance_id": "demo__abc123",
                "n_reads": 1,
                "hit_call_graph": True,
                "hit_ground_truth": True,
                "min_distance": 0.0,
                "first_ground_truth_step": 1,
            }
        ],
    )

    assert "Third-Party P2A Localization Baseline" in report
    assert "| demo__abc123 | 1 | yes | yes | 0.0 | 1 |" in report


def test_cache_rollouts_upserts_standard_third_party_artifacts(tmp_path):
    rollout_path = tmp_path / "rollouts.jsonl"
    details_path = tmp_path / "details.jsonl"
    db_path = tmp_path / "traces.sqlite"
    record = build_dump_record(
        _row(),
        run_id="run-cache",
        model_name="demo-model",
        base_url="https://example.test",
        interaction_result=_interaction_result(),
        reward_score=True,
        reward_details={"resolved": True},
    )
    record["wall_time"] = 1.2
    record["token_usage"] = {"input_tokens": 10, "output_tokens": 3, "cache_hit_tokens": 2}
    rollout_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    details_path.write_text(
        json.dumps(
            {
                "instance_id": "demo__abc123",
                "n_reads": 1,
                "hit_call_graph": True,
                "hit_ground_truth": True,
                "hit_near": True,
                "min_distance": 0.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    n_cached = cache_rollouts(
        db_path=db_path,
        experiment_id="single",
        provider_source_name="openai_compatible",
        model_api_name="demo-model",
        model_label="demo-model",
        dataset_name="swebench-hard",
        config_snapshot={"model": "demo-model"},
        rollouts_path=rollout_path,
        details_path=details_path,
    )

    assert n_cached == 1


def test_incremental_rollout_sink_writes_jsonl_and_cache_db(tmp_path):
    rollout_path = tmp_path / "rollouts.jsonl"
    db_path = tmp_path / "traces.sqlite"
    record = build_dump_record(
        _row(),
        run_id="run-stream",
        model_name="demo-model",
        base_url="https://example.test",
        interaction_result=_interaction_result(),
        reward_score=True,
        reward_details={"resolved": True},
    )
    sink = IncrementalRolloutSink(
        rollouts_path=rollout_path,
        db_path=db_path,
        experiment_id="stream",
        provider_source_name="openai_compatible",
        model_api_name="demo-model",
        model_label="Demo Model",
        dataset_name="swebench-hard",
        config_snapshot={"model": "demo-model"},
    )
    sink.prepare()

    asyncio.run(sink(record))

    assert sink.count == 1
    assert sink.n_cached == 1
    assert json.loads(rollout_path.read_text(encoding="utf-8"))["run_id"] == "run-stream"
    with ensure_db(db_path) as conn:
        cell = conn.execute("SELECT status, run_id FROM run_cells WHERE instance_id = ?", ("demo__abc123",)).fetchone()
        raw = conn.execute("SELECT rollout_json FROM raw_rollouts").fetchone()
    assert cell["status"] == "done"
    assert cell["run_id"] == "run-stream"
    assert json.loads(raw["rollout_json"])["run_id"] == "run-stream"


def test_incremental_rollout_sink_writes_dashboard_detail_cache(tmp_path):
    rollout_path = tmp_path / "rollouts.jsonl"
    db_path = tmp_path / "traces.sqlite"
    bonus_dir = tmp_path / "bonus_maps"
    bonus_dir.mkdir()
    (bonus_dir / "demo__abc123.json").write_text(
        json.dumps(
            {
                "instance_id": "demo__abc123",
                "case_type": "direct",
                "traceable": True,
                "selected_issue_anchor_nodes": ["pkg/demo.py::target"],
                "symptom_nodes": [],
                "root_cause_nodes": ["pkg/demo.py::target"],
                "reward_path_edges": [],
                "call_graph_edges": [],
                "call_graph_nodes": {
                    "pkg/demo.py::target": {
                        "file_path": "pkg/demo.py",
                        "start_line": 1,
                        "end_line": 5,
                        "normalized_distance": 0.0,
                        "rewardable": True,
                        "node_role": "root_cause",
                        "source": "def target():\n    return 1",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    record = build_dump_record(
        _row(),
        run_id="run-stream",
        model_name="demo-model",
        base_url="https://example.test",
        interaction_result=_interaction_result(),
        reward_score=True,
        reward_details={"resolved": True},
    )
    sink = IncrementalRolloutSink(
        rollouts_path=rollout_path,
        db_path=db_path,
        experiment_id="stream",
        provider_source_name="openai_compatible",
        model_api_name="demo-model",
        model_label="Demo Model",
        dataset_name="swebench-hard",
        config_snapshot={"model": "demo-model"},
        bonus_map_dir=bonus_dir,
        analysis_cfg={"tracking_mode": "view_and_bash", "near_threshold": 0.5, "m_max": 3.0},
    )
    sink.prepare()

    asyncio.run(sink(record))

    assert sink.n_cached == 1
    assert sink.n_detail_cached == 1
    assert sink.n_detail_cache_failed == 0
    with ensure_db(db_path) as conn:
        row = conn.execute("SELECT fingerprint, metrics_json FROM quantitative_metrics").fetchone()
    metrics = json.loads(row["metrics_json"])
    assert row["fingerprint"] is None
    assert "detail" not in metrics
    with sqlite3.connect(default_dashboard_build_db_path(db_path)) as build_conn:
        build_conn.row_factory = sqlite3.Row
        build_detail = build_conn.execute("SELECT fingerprint, detail_json, cache_metadata_json FROM dashboard_rollout_details").fetchone()
        build_metric = build_conn.execute("SELECT metrics_json FROM dashboard_model_metrics").fetchone()
    assert json.loads(build_detail["detail_json"])["root_hit"] is True
    assert json.loads(build_detail["cache_metadata_json"])["fingerprint"] == build_detail["fingerprint"]
    assert json.loads(build_metric["metrics_json"])["root_hit_rate"] == 1.0

    snapshot = build_dashboard_snapshot(
        DashboardRequest(
            db_path=db_path,
            experiment_id="stream",
            bonus_map_dir=bonus_dir,
            defer_db_scoring=True,
            include_db_raw_details=False,
        )
    )
    assert snapshot["details"] == []
    assert snapshot["eval_cells"][0]["cache_ready"] == 1
    assert snapshot["eval_cells"][0]["cache_pending"] == 0
    with ensure_db(db_path) as conn:
        cached_row = aggregate_model_metrics(
            conn,
            experiment_id="stream",
            include_detail_metrics=True,
            include_raw_trace_fallback=False,
        )[0]
    assert cached_row["root_hit_rate"] is None


def test_incremental_rollout_sink_keeps_raw_write_when_dashboard_cache_fails(monkeypatch, tmp_path):
    rollout_path = tmp_path / "rollouts.jsonl"
    db_path = tmp_path / "traces.sqlite"
    bonus_dir = tmp_path / "bonus_maps"
    bonus_dir.mkdir()
    record = build_dump_record(
        _row(),
        run_id="run-stream",
        model_name="demo-model",
        base_url="https://example.test",
        interaction_result=_interaction_result(),
        reward_score=True,
        reward_details={"resolved": True},
    )

    def fail_cache(*_args, **_kwargs):
        raise RuntimeError("cache failed")

    monkeypatch.setattr("p2a.third_party_eval.write_dashboard_detail_cache_for_record", fail_cache)
    sink = IncrementalRolloutSink(
        rollouts_path=rollout_path,
        db_path=db_path,
        experiment_id="stream",
        provider_source_name="openai_compatible",
        model_api_name="demo-model",
        model_label="Demo Model",
        dataset_name="swebench-hard",
        config_snapshot={"model": "demo-model"},
        bonus_map_dir=bonus_dir,
    )
    sink.prepare()

    asyncio.run(sink(record))

    assert sink.n_cached == 1
    assert sink.n_detail_cached == 0
    assert sink.n_detail_cache_failed == 1
    assert json.loads(rollout_path.read_text(encoding="utf-8"))["run_id"] == "run-stream"
    with ensure_db(db_path) as conn:
        cell = conn.execute("SELECT status, run_id FROM run_cells WHERE instance_id = ?", ("demo__abc123",)).fetchone()
        raw = conn.execute("SELECT rollout_json FROM raw_rollouts").fetchone()
    assert cell["status"] == "done"
    assert cell["run_id"] == "run-stream"
    assert json.loads(raw["rollout_json"])["run_id"] == "run-stream"


def test_incremental_rollout_sink_prepare_preserves_existing_jsonl(tmp_path):
    rollout_path = tmp_path / "rollouts.jsonl"
    existing = {"run_id": "already-done"}
    rollout_path.write_text(json.dumps(existing) + "\n", encoding="utf-8")
    sink = IncrementalRolloutSink(rollouts_path=rollout_path)

    sink.prepare()

    assert rollout_path.read_text(encoding="utf-8") == json.dumps(existing) + "\n"


def test_eval_batch_emits_records_as_each_rollout_finishes(monkeypatch):
    async def fake_run_one(row, *, rollout_index, **_kwargs):
        if row["instance_id"] == "slow":
            await asyncio.sleep(0.02)
        return {"instance_id": row["instance_id"], "rollout_index": rollout_index}

    emitted = []

    async def sink(record):
        emitted.append(record["instance_id"])

    monkeypatch.setattr("p2a.third_party_eval.run_one", fake_run_one)

    records = asyncio.run(
        run_batch(
            [{"instance_id": "slow"}, {"instance_id": "fast"}],
            model_cfg={"model_name": "dummy"},
            agent_cfg={},
            n_parallel=2,
            record_sink=sink,
        )
    )

    assert emitted == ["fast", "slow"]
    assert [record["instance_id"] for record in records] == ["slow", "fast"]
