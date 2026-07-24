import json

import numpy as np

from p2a.dashboard import build_dashboard
from p2a.validation_metrics import compute_validation_p2a_metrics, validation_records_from_batch


class FakeBatch:
    def __init__(self, non_tensor_batch):
        self.non_tensor_batch = non_tensor_batch


def test_validation_records_infer_rollout_indexes_for_repeated_instances():
    batch = FakeBatch(
        {
            "uid": np.array(["u0", "u1", "u2", "u3"], dtype=object),
            "instance_id": np.array(["case-1", "case-1", "case-2", "case-2"], dtype=object),
            "data_source": np.array(["unit", "unit", "unit", "unit"], dtype=object),
        }
    )

    records = validation_records_from_batch(batch, output_texts=["", "", "", ""], scores=[0.0, 1.0, 0.0, 0.0])

    assert [(record["instance_id"], record["rollout_index"]) for record in records] == [
        ("case-1", 0),
        ("case-1", 1),
        ("case-2", 0),
        ("case-2", 1),
    ]
    assert records[1]["rollout_id"] == "case-1:1"
    assert records[1]["extra_fields"]["rollout_index"] == 1


def _schema_v5_bonus_map(instance_id="demo__path"):
    return {
        "instance_id": instance_id,
        "case_type": "standard",
        "traceable": True,
        "selected_issue_anchor_nodes": ["app/views.py::symptom"],
        "symptom_nodes": ["app/views.py::symptom"],
        "root_cause_nodes": ["app/root.py::patched_root"],
        "reward_path_edges": [
            ["app/views.py::symptom", "app/service.py::intermediate"],
            ["app/service.py::intermediate", "app/root.py::patched_root"],
        ],
        "call_graph_edges": [
            ["tests/test_issue.py::test_issue", "framework/request.py::dispatch"],
            ["framework/request.py::dispatch", "app/views.py::symptom"],
            ["app/views.py::symptom", "app/service.py::intermediate"],
            ["app/service.py::intermediate", "app/root.py::patched_root"],
        ],
        "call_graph_edge_metadata": [
            {
                "caller": "framework/request.py::dispatch",
                "callee": "app/views.py::symptom",
                "caller_role": "pre_symptom",
                "callee_role": "symptom",
                "role_transition": "pre_symptom->symptom",
                "reward_path_edge": False,
            }
        ],
        "call_graph_nodes": {
            "tests/test_issue.py::test_issue": {
                "file_path": "tests/test_issue.py",
                "start_line": 1,
                "end_line": 8,
                "normalized_distance": 1.0,
                "rewardable": False,
                "node_role": "test_harness",
                "source": "def test_issue():\n    pass",
            },
            "framework/request.py::dispatch": {
                "file_path": "framework/request.py",
                "start_line": 10,
                "end_line": 20,
                "normalized_distance": 1.0,
                "rewardable": False,
                "node_role": "pre_symptom",
                "source": "def dispatch():\n    pass",
            },
            "app/views.py::symptom": {
                "file_path": "app/views.py",
                "start_line": 20,
                "end_line": 30,
                "normalized_distance": 1.0,
                "rewardable": True,
                "node_role": "symptom",
                "source": "def symptom():\n    service()",
            },
            "app/service.py::intermediate": {
                "file_path": "app/service.py",
                "start_line": 30,
                "end_line": 40,
                "normalized_distance": 0.5,
                "rewardable": True,
                "node_role": "intermediate",
                "source": "def intermediate():\n    patched_root()",
            },
            "app/root.py::patched_root": {
                "file_path": "app/root.py",
                "start_line": 40,
                "end_line": 50,
                "normalized_distance": 0.0,
                "rewardable": True,
                "node_role": "root_cause",
                "source": "def patched_root():\n    return 1",
            },
        },
    }


def test_validation_records_read_instance_id_from_native_task_metadata():
    records = validation_records_from_batch(
        FakeBatch(
            {
                "extra_info": np.array(
                    [
                        {
                            "tools_kwargs": {
                                "task": {
                                    "name": "swe_bench",
                                    "metadata": {"instance_id": "demo__native-task"},
                                }
                            }
                        }
                    ],
                    dtype=object,
                )
            }
        ),
        output_texts=["done"],
    )

    assert records[0]["instance_id"] == "demo__native-task"
    assert records[0]["extra_fields"]["instance_id"] == "demo__native-task"


def test_validation_metrics_use_schema_v5_path_projection(tmp_path):
    bonus_dir = tmp_path / "bonus_maps"
    bonus_dir.mkdir()
    (bonus_dir / "demo__path.json").write_text(json.dumps(_schema_v5_bonus_map()), encoding="utf-8")

    batch = FakeBatch(
        {
            "uid": np.array(["uid-1"], dtype=object),
            "data_source": np.array(["unit"], dtype=object),
            "extra_fields": np.array(
                [
                    {
                        "instance_id": "demo__path",
                        "p2a_step_traces": [
                            {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "str_replace_editor",
                                            "arguments": {
                                                "command": "view",
                                                "path": "/testbed/tests/test_issue.py",
                                                "view_range": [1, 8],
                                            },
                                        }
                                    }
                                ]
                            },
                            {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "str_replace_editor",
                                            "arguments": {
                                                "command": "view",
                                                "path": "/testbed/app/views.py",
                                                "view_range": [20, 30],
                                            },
                                        }
                                    }
                                ]
                            },
                            {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "str_replace_editor",
                                            "arguments": {
                                                "command": "view",
                                                "path": "/testbed/app/root.py",
                                                "view_range": [40, 50],
                                            },
                                        }
                                    }
                                ]
                            },
                        ],
                    }
                ],
                dtype=object,
            ),
        }
    )

    records = validation_records_from_batch(batch, output_texts=[""], scores=[1.0])
    metrics, details = compute_validation_p2a_metrics(
        records,
        bonus_map_dir=str(bonus_dir),
        tracking_mode="view_and_bash",
        near_threshold=0.5,
        m_max=3.0,
    )

    detail = details[0]
    assert detail["path_evaluable"] is True
    assert detail["chain_evaluable"] is True
    assert detail["path_covered"] is True
    assert detail["chain_graph_covered"] is True
    assert detail["anchor_hit"] is True
    assert detail["root_hit"] is True
    assert detail["path_hit"] is True
    assert detail["chain_hit"] is True
    assert detail["path_node_recall"] == 2 / 3
    assert detail["path_read_precision"] == 2 / 3
    assert detail["chain_node_recall"] == 2 / 3
    assert detail["chain_read_precision"] == 2 / 3
    assert detail["first_anchor_step"] == 2
    assert detail["first_root_step"] == 3
    assert detail["steps_anchor_to_root"] == 1
    assert detail["anchor_before_root"] is True
    assert detail["path_pattern_flags"]["missed_anchor"] is False
    assert detail["chain_bad_patterns"]["missed_anchor"] is False
    assert detail["path_projection"]["path_edges"] == [
        {
            "caller": "app/views.py::symptom",
            "callee": "app/service.py::intermediate",
            "source": "app/views.py::symptom",
            "target": "app/service.py::intermediate",
            "edge_type": "path",
            "caller_role": "symptom",
            "callee_role": "intermediate",
            "role_transition": "symptom->intermediate",
        },
        {
            "caller": "app/service.py::intermediate",
            "callee": "app/root.py::patched_root",
            "source": "app/service.py::intermediate",
            "target": "app/root.py::patched_root",
            "edge_type": "path",
            "caller_role": "intermediate",
            "callee_role": "root_cause",
            "role_transition": "intermediate->root_cause",
        },
    ]
    context_by_key = {node["key"]: node for node in detail["path_projection"]["context_nodes"]}
    assert context_by_key["tests/test_issue.py::test_issue"]["hit"] is True
    assert context_by_key["tests/test_issue.py::test_issue"]["normalized_distance"] == 1.0
    assert "source_preview" not in context_by_key["tests/test_issue.py::test_issue"]
    assert context_by_key["framework/request.py::dispatch"]["node_role"] == "test_adapter"
    path_by_key = {node["key"]: node for node in detail["path_projection"]["path_nodes"]}
    assert path_by_key["app/views.py::symptom"]["normalized_distance"] == 1.0
    assert path_by_key["app/root.py::patched_root"]["normalized_distance"] == 0.0
    assert metrics["val-p2a/unit/path_coverage"] == 1.0
    assert metrics["val-p2a/unit/chain_graph_coverage"] == 1.0
    assert metrics["val-p2a/unit/anchor_hit_rate"] == 1.0
    assert metrics["val-p2a/unit/root_hit_rate"] == 1.0
    assert metrics["val-p2a/unit/path_hit_rate"] == 1.0
    assert metrics["val-p2a/unit/chain_hit_rate"] == 1.0
    assert metrics["val-p2a/unit/path_node_recall"] == 2 / 3
    assert metrics["val-p2a/unit/path_read_precision"] == 2 / 3
    assert metrics["val-p2a/unit/chain_node_recall"] == 2 / 3
    assert metrics["val-p2a/unit/chain_read_precision"] == 2 / 3
    assert metrics["val-p2a/unit/time_to_anchor"] == 2.0
    assert metrics["val-p2a/unit/time_to_root"] == 3.0
    assert metrics["val-p2a/unit/steps_anchor_to_root"] == 1.0


def test_path_evaluability_reports_missing_anchor_reason(tmp_path):
    bonus_dir = tmp_path / "bonus_maps"
    bonus_dir.mkdir()
    bonus_map = _schema_v5_bonus_map("demo__no_anchor")
    bonus_map["selected_issue_anchor_nodes"] = []
    bonus_map["reason_code"] = "standard"
    (bonus_dir / "demo__no_anchor.json").write_text(json.dumps(bonus_map), encoding="utf-8")

    batch = FakeBatch(
        {
            "uid": np.array(["uid-1"], dtype=object),
            "data_source": np.array(["unit"], dtype=object),
            "extra_fields": np.array([{"instance_id": "demo__no_anchor", "p2a_step_traces": []}], dtype=object),
        }
    )

    records = validation_records_from_batch(batch, output_texts=[""], scores=[0.0])
    _metrics, details = compute_validation_p2a_metrics(
        records,
        bonus_map_dir=str(bonus_dir),
        tracking_mode="view_and_bash",
        near_threshold=0.5,
        m_max=3.0,
    )

    assert details[0]["path_evaluable"] is False
    assert details[0]["chain_evaluable"] is False
    assert details[0]["not_path_evaluable_reason"] == "traceable_no_anchor"
    assert details[0]["not_chain_evaluable_reason"] == "traceable_no_anchor"


def test_path_pattern_rates_do_not_double_count_shared_legacy_flags(tmp_path):
    bonus_dir = tmp_path / "bonus_maps"
    bonus_dir.mkdir()
    (bonus_dir / "demo__path.json").write_text(json.dumps(_schema_v5_bonus_map()), encoding="utf-8")

    records = [
        {
            "instance_id": "demo__path",
            "data_source": "unit",
            "p2a_step_traces": [
                {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "str_replace_editor",
                                "arguments": {
                                    "command": "view",
                                    "path": "/testbed/app/root.py",
                                    "view_range": [40, 50],
                                },
                            }
                        }
                    ]
                }
            ],
        }
    ]

    metrics, details = compute_validation_p2a_metrics(
        records,
        bonus_map_dir=str(bonus_dir),
        tracking_mode="view_and_bash",
        near_threshold=0.5,
        m_max=3.0,
    )

    assert details[0]["path_evaluable"] is True
    assert details[0]["path_pattern_flags"]["missed_anchor"] is True
    assert details[0]["chain_bad_patterns"]["missed_anchor"] is True
    assert metrics["val-p2a/unit/missed_anchor_rate"] == 1.0


def test_path_metrics_fall_back_to_top_level_reads_when_step_traces_are_empty(tmp_path):
    bonus_dir = tmp_path / "bonus_maps"
    bonus_dir.mkdir()
    (bonus_dir / "demo__path.json").write_text(json.dumps(_schema_v5_bonus_map()), encoding="utf-8")
    records = [
        {
            "instance_id": "demo__path",
            "data_source": "unit",
            "p2a_step_traces": [{"step_idx": 0, "tool_calls": [], "response_text": "planning only"}],
            "response_text": "cat /testbed/app/views.py\ncat /testbed/app/root.py",
        }
    ]

    metrics, details = compute_validation_p2a_metrics(
        records,
        bonus_map_dir=str(bonus_dir),
        tracking_mode="view_and_bash",
        near_threshold=0.5,
        m_max=3.0,
    )

    detail = details[0]
    assert detail["hit_call_graph"] is True
    assert detail["path_evaluable"] is True
    assert detail["chain_evaluable"] is True
    assert detail["anchor_hit"] is True
    assert detail["root_hit"] is True
    assert detail["path_hit"] is True
    assert detail["chain_hit"] is True
    assert detail["path_node_recall"] == 2 / 3
    assert detail["path_read_precision"] == 1.0
    assert detail["chain_node_recall"] == 2 / 3
    assert detail["chain_read_precision"] == 1.0
    assert detail["first_anchor_step"] == 1
    assert detail["first_root_step"] == 1
    assert detail["steps_anchor_to_root"] == 0
    assert metrics["val-p2a/unit/path_hit_rate"] == 1.0
    assert metrics["val-p2a/unit/chain_hit_rate"] == 1.0
    assert metrics["val-p2a/unit/path_node_recall"] == 2 / 3
    assert metrics["val-p2a/unit/chain_node_recall"] == 2 / 3


def test_validation_records_from_extra_fields_and_metric_flattening(tmp_path):
    bonus_dir = tmp_path / "bonus_maps"
    bonus_dir.mkdir()
    (bonus_dir / "demo__abc123.json").write_text(
        json.dumps(
            {
                "instance_id": "demo__abc123",
                "case_type": "standard",
                "traceable": True,
                "call_graph_nodes": {
                    "pkg.demo:demo": {
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

    batch = FakeBatch(
        {
            "uid": np.array(["uid-1"], dtype=object),
            "data_source": np.array(["swebench"], dtype=object),
            "extra_fields": np.array(
                [
                    {
                        "instance_id": "demo__abc123",
                        "p2a_step_traces": [
                            {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "execute_bash",
                                            "arguments": {"command": "cat /testbed/pkg/demo.py"},
                                        }
                                    }
                                ]
                            }
                        ],
                    }
                ],
                dtype=object,
            ),
        }
    )

    records = validation_records_from_batch(batch, output_texts=[""], scores=[1.0])
    assert records[0]["instance_id"] == "demo__abc123"
    assert records[0]["data_source"] == "swebench"

    metrics, details = compute_validation_p2a_metrics(
        records,
        bonus_map_dir=str(bonus_dir),
        tracking_mode="view_and_bash",
        near_threshold=0.5,
        m_max=3.0,
    )

    assert details[0]["has_step_traces"] is True
    assert details[0]["hit_call_graph"] is True
    assert details[0]["hit_ground_truth"] is True
    assert metrics["val-p2a/swebench/bonus_map_coverage"] == 1.0
    assert metrics["val-p2a/swebench/call_graph_coverage"] == 1.0
    assert metrics["val-p2a/swebench/read_rate"] == 1.0
    assert metrics["val-p2a/swebench/graph_hit_rate_over_call_graphs"] == 1.0
    assert metrics["val-p2a/swebench/ground_truth_hit_rate_over_call_graphs"] == 1.0
    assert metrics["val-p2a/swebench/avg_min_distance_on_hits"] == 0.0
    assert metrics["val-p2a/swebench/block_achieve_rate"] == 1.0
    assert metrics["val-p2a/swebench/avg_block_efficiency_steps"] == 1.0
    assert details[0]["purpose_blocks"][0]["achieved"] is True


def test_validation_records_fall_back_to_extra_info_metadata(tmp_path):
    bonus_dir = tmp_path / "bonus_maps"
    bonus_dir.mkdir()
    (bonus_dir / "django__1234567890.json").write_text(
        json.dumps(
            {
                "instance_id": "django__1234567890",
                "case_type": "standard",
                "traceable": True,
                "call_graph_nodes": {
                    "django.core:target": {
                        "file_path": "django/core.py",
                        "start_line": 10,
                        "end_line": 20,
                        "normalized_distance": 0.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    batch = FakeBatch(
        {
            "uid": np.array(["rollout-1"], dtype=object),
            "extra_info": np.array(
                [
                    {
                        "data_source": "swebench-hard",
                        "tools_kwargs": {
                            "reward": {
                                "metadata": {
                                    "instance_id": "django__1234567890",
                                }
                            }
                        },
                    }
                ],
                dtype=object,
            ),
            "response_text": np.array(["cat /testbed/django/core.py"], dtype=object),
        }
    )

    records = validation_records_from_batch(batch, output_texts=[""], scores=[0.0])
    assert records[0]["instance_id"] == "django__1234567890"
    assert records[0]["data_source"] == "swebench-hard"

    metrics, details = compute_validation_p2a_metrics(
        records,
        bonus_map_dir=str(bonus_dir),
        tracking_mode="view_and_bash",
        near_threshold=0.5,
        m_max=3.0,
    )

    assert details[0]["hit_call_graph"] is True
    assert metrics["val-p2a/swebench-hard/bonus_map_coverage"] == 1.0
    assert metrics["val-p2a/swebench-hard/graph_hit_rate_over_call_graphs"] == 1.0


def test_validation_metrics_report_order_and_miracle_rates(tmp_path):
    bonus_dir = tmp_path / "bonus_maps"
    bonus_dir.mkdir()
    (bonus_dir / "demo__graph.json").write_text(
        json.dumps(
            {
                "instance_id": "demo__graph",
                "case_type": "standard",
                "traceable": True,
                "selected_issue_anchor_nodes": ["tests/test_demo.py::test_demo"],
                "root_cause_nodes": ["pkg/demo.py::target"],
                "reward_path_edges": [
                    ["tests/test_demo.py::test_demo", "pkg/mid.py::mid"],
                    ["pkg/mid.py::mid", "pkg/demo.py::target"],
                ],
                "call_graph_nodes": {
                    "tests/test_demo.py::test_demo": {
                        "file_path": "tests/test_demo.py",
                        "start_line": 1,
                        "end_line": 4,
                        "normalized_distance": 1.0,
                    },
                    "pkg/mid.py::mid": {
                        "file_path": "pkg/mid.py",
                        "start_line": 1,
                        "end_line": 5,
                        "normalized_distance": 0.5,
                    },
                    "pkg/demo.py::target": {
                        "file_path": "pkg/demo.py",
                        "start_line": 1,
                        "end_line": 5,
                        "normalized_distance": 0.0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    records = [
        {
            "instance_id": "demo__graph",
            "data_source": "unit",
            "p2a_step_traces": [
                {"tool_calls": [{"function": {"name": "str_replace_editor", "arguments": {"command": "view", "path": "/testbed/tests/test_demo.py", "view_range": [1, 4]}}}]},
                {"tool_calls": [{"function": {"name": "str_replace_editor", "arguments": {"command": "view", "path": "/testbed/pkg/mid.py", "view_range": [1, 5]}}}]},
                {"tool_calls": [{"function": {"name": "str_replace_editor", "arguments": {"command": "view", "path": "/testbed/pkg/demo.py", "view_range": [1, 5]}}}]},
            ],
        },
        {
            "instance_id": "demo__graph",
            "data_source": "unit",
            "p2a_step_traces": [
                {"tool_calls": [{"function": {"name": "str_replace_editor", "arguments": {"command": "view", "path": "/testbed/pkg/demo.py", "view_range": [1, 5]}}}]},
            ],
        },
    ]

    metrics, details = compute_validation_p2a_metrics(
        records,
        bonus_map_dir=str(bonus_dir),
        tracking_mode="view_and_bash",
        near_threshold=0.5,
        m_max=3.0,
    )

    assert details[0]["order_score"] == 1.0
    assert details[0]["miracle_step"] is False
    assert details[1]["miracle_step"] is True
    assert details[1]["miracle_severity"] == 1
    assert metrics["val-p2a/unit/avg_node_recall"] == (1.0 + (1 / 3)) / 2
    assert metrics["val-p2a/unit/miracle_rate_over_gt_hits"] == 0.5


def test_root_hit_without_prior_symptom_is_miracle_even_with_same_step_intermediate(tmp_path):
    bonus_dir = tmp_path / "bonus_maps"
    bonus_dir.mkdir()
    long_source = "\n".join([f"line {idx}" for idx in range(900)])
    (bonus_dir / "demo__missing_anchor.json").write_text(
        json.dumps(
            {
                "instance_id": "demo__missing_anchor",
                "case_type": "standard",
                "traceable": True,
                "selected_issue_anchor_nodes": ["tests/test_demo.py::test_demo"],
                "root_cause_nodes": ["pkg/demo.py::target"],
                "reward_path_edges": [
                    ["tests/test_demo.py::test_demo", "pkg/demo.py::mid"],
                    ["pkg/demo.py::mid", "pkg/demo.py::target"],
                ],
                "call_graph_nodes": {
                    "tests/test_demo.py::test_demo": {
                        "file_path": "tests/test_demo.py",
                        "start_line": 1,
                        "end_line": 4,
                        "normalized_distance": 1.0,
                        "node_role": "symptom",
                    },
                    "pkg/demo.py::mid": {
                        "file_path": "pkg/demo.py",
                        "start_line": 1,
                        "end_line": 20,
                        "normalized_distance": 0.5,
                        "node_role": "intermediate",
                    },
                    "pkg/demo.py::target": {
                        "file_path": "pkg/demo.py",
                        "start_line": 10,
                        "end_line": 30,
                        "normalized_distance": 0.0,
                        "node_role": "root_cause",
                        "source": long_source,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    records = [
        {
            "instance_id": "demo__missing_anchor",
            "data_source": "unit",
            "p2a_step_traces": [
                {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "str_replace_editor",
                                "arguments": {"command": "view", "path": "/testbed/pkg/demo.py", "view_range": [1, 30]},
                            }
                        }
                    ]
                },
            ],
        }
    ]

    metrics, details = compute_validation_p2a_metrics(
        records,
        bonus_map_dir=str(bonus_dir),
        tracking_mode="view_and_bash",
        near_threshold=0.5,
        m_max=3.0,
    )

    assert details[0]["anchor_hit"] is False
    assert details[0]["root_hit"] is True
    assert details[0]["miracle_step"] is True
    assert metrics["val-p2a/unit/miracle_rate_over_gt_hits"] == 1.0
    root_node = next(node for node in details[0]["graph_topology"]["nodes"] if node["key"] == "pkg/demo.py::target")
    assert root_node["source"] == long_source
    assert root_node["source_preview"].endswith("...")


def test_same_step_full_path_hit_is_not_miracle(tmp_path):
    bonus_dir = tmp_path / "bonus_maps"
    bonus_dir.mkdir()
    (bonus_dir / "demo__same_step_path.json").write_text(
        json.dumps(
            {
                "instance_id": "demo__same_step_path",
                "case_type": "standard",
                "traceable": True,
                "selected_issue_anchor_nodes": ["pkg/demo.py::symptom"],
                "root_cause_nodes": ["pkg/demo.py::target"],
                "reward_path_edges": [
                    ["pkg/demo.py::symptom", "pkg/demo.py::mid"],
                    ["pkg/demo.py::mid", "pkg/demo.py::target"],
                ],
                "call_graph_nodes": {
                    "pkg/demo.py::symptom": {
                        "file_path": "pkg/demo.py",
                        "start_line": 1,
                        "end_line": 10,
                        "normalized_distance": 1.0,
                        "node_role": "symptom",
                    },
                    "pkg/demo.py::mid": {
                        "file_path": "pkg/demo.py",
                        "start_line": 11,
                        "end_line": 20,
                        "normalized_distance": 0.5,
                        "node_role": "intermediate",
                    },
                    "pkg/demo.py::target": {
                        "file_path": "pkg/demo.py",
                        "start_line": 21,
                        "end_line": 30,
                        "normalized_distance": 0.0,
                        "node_role": "root_cause",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    records = [
        {
            "instance_id": "demo__same_step_path",
            "data_source": "unit",
            "p2a_step_traces": [
                {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "str_replace_editor",
                                "arguments": {"command": "view", "path": "/testbed/pkg/demo.py", "view_range": [1, 30]},
                            }
                        }
                    ]
                },
            ],
        }
    ]

    metrics, details = compute_validation_p2a_metrics(
        records,
        bonus_map_dir=str(bonus_dir),
        tracking_mode="view_and_bash",
        near_threshold=0.5,
        m_max=3.0,
    )

    assert details[0]["anchor_hit"] is True
    assert details[0]["root_hit"] is True
    assert details[0]["miracle_step"] is False
    assert details[0]["miracle_severity"] == 0
    assert details[0]["order_defined"] is False
    assert metrics["val-p2a/unit/miracle_rate_over_gt_hits"] == 0.0


def test_direct_symptom_root_node_is_not_order_or_miracle_candidate(tmp_path):
    bonus_dir = tmp_path / "bonus_maps"
    bonus_dir.mkdir()
    (bonus_dir / "demo__direct.json").write_text(
        json.dumps(
            {
                "instance_id": "demo__direct",
                "case_type": "direct",
                "traceable": True,
                "selected_issue_anchor_nodes": ["pkg/demo.py::target"],
                "root_cause_nodes": ["pkg/demo.py::target"],
                "reward_path_edges": [],
                "call_graph_nodes": {
                    "pkg/demo.py::target": {
                        "file_path": "pkg/demo.py",
                        "start_line": 1,
                        "end_line": 5,
                        "normalized_distance": 0.0,
                        "rewardable": True,
                        "node_role": "root_cause",
                    },
                    "pkg/mid.py::mid": {
                        "file_path": "pkg/mid.py",
                        "start_line": 1,
                        "end_line": 5,
                        "normalized_distance": 0.5,
                        "rewardable": True,
                        "node_role": "intermediate",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    records = [
        {
            "instance_id": "demo__direct",
            "data_source": "unit",
            "p2a_step_traces": [
                {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "str_replace_editor",
                                "arguments": {"command": "view", "path": "/testbed/pkg/demo.py", "view_range": [1, 5]},
                            }
                        }
                    ]
                },
            ],
        }
    ]

    metrics, details = compute_validation_p2a_metrics(
        records,
        bonus_map_dir=str(bonus_dir),
        tracking_mode="view_and_bash",
        near_threshold=0.5,
        m_max=3.0,
    )

    assert details[0]["path_evaluable"] is True
    assert details[0]["chain_evaluable"] is True
    assert details[0]["path_projection"]["anchors"] == ["pkg/demo.py::target"]
    assert details[0]["path_projection"]["roots"] == ["pkg/demo.py::target"]
    assert details[0]["order_defined"] is False
    assert details[0]["order_score"] is None
    assert details[0]["miracle_step"] is None
    assert metrics.get("val-p2a/unit/reverse_order_rate") is None
    assert metrics.get("val-p2a/unit/miracle_rate_over_gt_hits") is None


def test_validation_metrics_preserve_trace_step_indexes(tmp_path):
    bonus_dir = tmp_path / "bonus_maps"
    bonus_dir.mkdir()
    (bonus_dir / "demo__steps.json").write_text(
        json.dumps(
            {
                "instance_id": "demo__steps",
                "case_type": "standard",
                "traceable": True,
                "call_graph_nodes": {
                    "pkg/demo.py::target": {
                        "file_path": "pkg/demo.py",
                        "start_line": 1,
                        "end_line": 5,
                        "normalized_distance": 0.0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    records = [
        {
            "instance_id": "demo__steps",
            "data_source": "unit",
            "p2a_step_traces": [
                {"tool_calls": []},
                {"tool_calls": []},
                {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "str_replace_editor",
                                "arguments": {"command": "view", "path": "/testbed/pkg/demo.py", "view_range": [1, 5]},
                            }
                        }
                    ]
                },
            ],
        }
    ]

    _metrics, details = compute_validation_p2a_metrics(
        records,
        bonus_map_dir=str(bonus_dir),
        tracking_mode="view_and_bash",
        near_threshold=0.5,
        m_max=3.0,
    )

    assert details[0]["n_steps_with_reads"] == 1
    assert details[0]["first_hit_step"] == 3
    assert details[0]["graph_topology"]["nodes"][0]["first_step"] == 3


def test_validation_metrics_ignore_non_rewardable_nodes(tmp_path):
    bonus_dir = tmp_path / "bonus_maps"
    bonus_dir.mkdir()
    (bonus_dir / "demo__nonrewardable.json").write_text(
        json.dumps(
            {
                "instance_id": "demo__nonrewardable",
                "case_type": "standard",
                "traceable": True,
                "call_graph_nodes": {
                    "tests/test_demo.py::test_demo": {
                        "file_path": "tests/test_demo.py",
                        "start_line": 1,
                        "end_line": 20,
                        "normalized_distance": 1.0,
                        "rewardable": False,
                        "node_role": "test_harness",
                        "excluded_from_hop_max": True,
                        "exclusion_reason": "test_suite_or_harness:tests/**",
                    },
                    "pkg/demo.py::target": {
                        "file_path": "pkg/demo.py",
                        "start_line": 1,
                        "end_line": 5,
                        "normalized_distance": 0.0,
                        "rewardable": True,
                        "node_role": "program",
                        "excluded_from_hop_max": False,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    records = [
        {
            "instance_id": "demo__nonrewardable",
            "data_source": "unit",
            "p2a_step_traces": [
                {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "str_replace_editor",
                                "arguments": {
                                    "command": "view",
                                    "path": "/testbed/tests/test_demo.py",
                                    "view_range": [1, 20],
                                },
                            }
                        }
                    ]
                }
            ],
        }
    ]

    metrics, details = compute_validation_p2a_metrics(
        records,
        bonus_map_dir=str(bonus_dir),
        tracking_mode="view_and_bash",
        near_threshold=0.5,
        m_max=3.0,
    )

    assert details[0]["hit_call_graph"] is False
    assert details[0]["n_call_graph_nodes"] == 2
    assert details[0]["n_rewardable_call_graph_nodes"] == 1
    topology_nodes = {node["key"]: node for node in details[0]["graph_topology"]["nodes"]}
    assert topology_nodes["tests/test_demo.py::test_demo"]["rewardable"] is False
    assert topology_nodes["tests/test_demo.py::test_demo"]["hit"] is False
    assert metrics["val-p2a/unit/graph_hit_rate_over_call_graphs"] == 0.0


def test_dashboard_builds_static_artifacts(tmp_path):
    bonus_dir = tmp_path / "bonus_maps"
    bonus_dir.mkdir()
    (bonus_dir / "demo__abc123.json").write_text(
        json.dumps(
            {
                "instance_id": "demo__abc123",
                "case_type": "direct",
                "traceable": True,
                "selected_issue_anchor_nodes": ["pkg.demo:demo"],
                "symptom_nodes": [],
                "root_cause_nodes": ["pkg.demo:demo"],
                "reward_path_edges": [],
                "call_graph_nodes": {
                    "pkg.demo:demo": {
                        "file_path": "pkg/demo.py",
                        "start_line": 1,
                        "end_line": 5,
                        "normalized_distance": 0.0,
                        "rewardable": True,
                        "node_role": "root_cause",
                        "source": "def demo():\n    return 1",
                    }
                },
                "call_graph_edges": [["tests/test_demo.py::test_demo", "pkg.demo:demo"]],
            }
        ),
        encoding="utf-8",
    )
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    rollouts = rollout_dir / "part.jsonl"
    rollouts.write_text(
        json.dumps(
            {
                "instance_id": "demo__abc123",
                "global_step": 7,
                "p2a_step_traces": [
                    {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "execute_bash",
                                    "arguments": {"command": "cat /testbed/pkg/demo.py"},
                                }
                            }
                        ]
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    paths = build_dashboard(rollout_dir, bonus_dir, tmp_path / "dashboard")

    assert paths["details"].exists()
    assert paths["summary"].exists()
    html = paths["html"].read_text(encoding="utf-8")
    assert "P2A unified dashboard" in html
    assert "window.__P2A_DASHBOARD_SNAPSHOT__" in html
    assert "demo__abc123" in html
    assert "root_cause" in html
    assert "pkg.demo:demo" in html
    assert "def demo()" in html
