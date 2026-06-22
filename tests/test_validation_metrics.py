import json

import numpy as np

from p2a.dashboard import build_dashboard
from p2a.validation_metrics import compute_validation_p2a_metrics, validation_records_from_batch


class FakeBatch:
    def __init__(self, non_tensor_batch):
        self.non_tensor_batch = non_tensor_batch


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
    assert details[0]["first_hit_step"] == 2
    assert details[0]["graph_topology"]["nodes"][0]["first_step"] == 2


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
                "call_graph_nodes": {
                    "pkg.demo:demo": {
                        "file_path": "pkg/demo.py",
                        "start_line": 1,
                        "end_line": 5,
                        "normalized_distance": 0.0,
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
    assert "P2A trajectory dashboard" in html
    assert "demo__abc123" in html
    assert "Graph topology" in html
    assert "Trend panel" in html
    assert "Purpose blocks" in html
    assert "pkg.demo:demo" in html
    assert "def demo()" in html
