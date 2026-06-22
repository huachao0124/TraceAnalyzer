import json
from types import SimpleNamespace

import numpy as np

from p2a.core import BonusMapStore
from p2a.eval_fault_localization import score_record
from p2a.third_party_eval import (
    _prompt,
    _select_rows,
    apply_cli_overrides,
    build_dump_record,
    build_step_traces,
    cache_rollouts,
    format_report,
    load_config,
    parse_limit_arg,
    resolve_model_config,
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


def test_dump_record_is_readable_by_fault_localization_scorer(tmp_path):
    bonus_dir = tmp_path / "bonus_maps"
    bonus_dir.mkdir()
    (bonus_dir / "demo__abc123.json").write_text(
        json.dumps(
            {
                "instance_id": "demo__abc123",
                "case_type": "direct",
                "traceable": True,
                "call_graph_nodes": {
                    "pkg/demo.py::target": {
                        "file_path": "pkg/demo.py",
                        "start_line": 1,
                        "end_line": 5,
                        "normalized_distance": 0.0,
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
                "first_ground_truth_step": 0,
            }
        ],
    )

    assert "Third-Party P2A Localization Baseline" in report
    assert "| demo__abc123 | 1 | yes | yes | 0.0 | 0 |" in report


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
