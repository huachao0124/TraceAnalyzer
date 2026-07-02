import json
import subprocess
import textwrap
from pathlib import Path


def test_dashboard_frontend_state_and_inspector_rendering(tmp_path):
    app_path = Path(__file__).resolve().parents[1] / "p2a" / "dashboard_static" / "app.js"
    index_path = app_path.with_name("index.html")
    css_path = app_path.with_name("styles.css")
    html = index_path.read_text(encoding="utf-8")
    css = css_path.read_text(encoding="utf-8")
    assert 'id="admin-login" class="admin-login"' in html
    assert 'id="admin-login-button" type="submit">Log in</button>' in html
    assert 'data-pattern-filter="error' not in html
    assert ".node-source-code" in css
    assert ".node-source-code {\n  flex: 1 1 auto;" in css
    assert "background: #101827;" in css[css.index(".node-source-code") : css.index(".node-source-code .code-view")]
    assert ".node-source-code .code-view" in css
    assert "max-height: none;" in css[css.index(".node-source-code .code-view") :]
    assert "grid-template-columns: repeat(2, minmax(0, 1fr));" in css
    assert ".graph-content.has-source" in css
    assert ".metric-group-filter" in css
    assert ".kpi-table th.metric-group-graph" in css
    assert ".kpi-table th.metric-group-path" in css
    assert ".trace-rollout-strip" in css
    assert ".trace-rollout-segment.is-resolved" in css
    assert ".trace-rollout-segment.is-unresolved" in css
    assert ".rollout-select-control" in css
    assert ".trace-global-rollout" in css
    assert ".trace-pattern-stats" in css
    assert 'id="trace-pattern-stats"' in html
    assert 'id="trace-global-rollout-slot"' in html
    assert 'class="trace-pattern-cycle"' in html
    assert ".trace-pattern-cycle.is-true .trace-pattern-mark::before" in css
    assert ".trace-pattern-cycle.is-false .trace-pattern-mark::before" in css
    assert ".trace-pattern-cycle.is-none .trace-pattern-mark" in css
    assert html.index('id="trace-pattern-stats"') < html.index('id="trace-global-rollout-slot"') < html.index('class="trace-pattern-filter"')
    snapshot = {
        "schema_version": "p2a_unified_dashboard_v1",
        "sources": [{"kind": "db", "path": "demo.sqlite"}],
        "summary": {
            "counts": {"n_records": 3, "n_not_chain_evaluable": 0},
            "rates": {"anchor_hit_rate": 1.0, "root_hit_rate": 1.0, "chain_node_recall": 1.0},
            "averages": {},
            "distributions": {},
            "distributions_by_dataset": {
                "swebench-hard": {
                    "dataset": "swebench-hard",
                    "n_instances": 2,
                    "distributions": {
                        "case_types": {"direct": 2},
                        "not_path_evaluable_reasons": {},
                        "not_chain_evaluable_reasons": {},
                        "availability": {"with_bonus_map": 2, "with_call_graph": 2, "path_evaluable": 2, "chain_evaluable": 2, "not_path_evaluable": 0, "not_chain_evaluable": 0},
                    },
                }
            },
            "by_case_type": {},
            "trends": [],
        },
        "datasets": [
            {
                "dataset": "swebench-hard",
                "n_instances": 2,
                "n_eval_cells": 2,
                "n_trajectories": 3,
                "models": ["model-a", "model-b"],
                "source_kinds": ["third_party_api"],
            }
        ],
        "eval_cells": [
            {
                "eval_cell_key": "third_party_api::exp-a::internal_api::swebench-hard::model-a::model-a",
                "experiment_key": "third_party_api::exp-a::internal_api::swebench-hard::model-a::model-a",
                "source_kind": "third_party_api",
                "experiment_id": "exp-a",
                "provider_source": "internal_api",
                "dataset": "swebench-hard",
                "model_api_name": "model-a",
                "model_label": "model-a",
                "target": 2,
                "done": 2,
                "cache_pending": 1,
                "trajectory_count": 2,
                "resolved_rate": 0.0,
                "root_hit_rate": 1.0,
                "path_node_recall": 1.0,
                "chain_node_recall": 1.0,
                "read_precision": 1.0,
            },
            {
                "eval_cell_key": "third_party_api::exp-b::internal_api::swebench-hard::model-b::model-b",
                "experiment_key": "third_party_api::exp-b::internal_api::swebench-hard::model-b::model-b",
                "source_kind": "third_party_api",
                "experiment_id": "exp-b",
                "provider_source": "internal_api",
                "dataset": "swebench-hard",
                "model_api_name": "model-b",
                "model_label": "model-b",
                "target": 1,
                "done": 1,
                "trajectory_count": 1,
            },
        ],
        "experiments": [],
        "model_metrics": [
            {
                "eval_cell_key": "third_party_api::exp-a::internal_api::swebench-hard::model-a::model-a",
                "experiment_key": "third_party_api::exp-a::internal_api::swebench-hard::model-a::model-a",
                "source_kind": "third_party_api",
                "experiment_id": "exp-a",
                "provider_source": "internal_api",
                "dataset": "swebench-hard",
                "model_api_name": "model-a",
                "model_label": "model-a",
                "target": 1,
                "done": 1,
                "avg_path_read_precision": 1.0,
                "avg_chain_read_precision": 1.0,
            },
            {
                "eval_cell_key": "third_party_api::exp-b::internal_api::swebench-hard::model-b::model-b",
                "experiment_key": "third_party_api::exp-b::internal_api::swebench-hard::model-b::model-b",
                "source_kind": "third_party_api",
                "experiment_id": "exp-b",
                "provider_source": "internal_api",
                "dataset": "swebench-hard",
                "model_api_name": "model-b",
                "model_label": "model-b",
                "target": 1,
                "done": 1,
                "avg_path_read_precision": 0.0,
                "avg_chain_read_precision": 0.0,
            },
        ],
        "path_metric_model_metrics": [
            {
                "eval_cell_key": "third_party_api::exp-a::internal_api::swebench-hard::model-a::model-a",
                "experiment_key": "third_party_api::exp-a::internal_api::swebench-hard::model-a::model-a",
                "source_kind": "third_party_api",
                "experiment_id": "exp-a",
                "provider_source": "internal_api",
                "dataset": "swebench-hard",
                "model_api_name": "model-a",
                "model_label": "model-a",
                "target": 1,
                "done": 1,
                "avg_path_read_precision": 1.0,
            }
        ],
        "dynamic_traceable_model_metrics": [
            {
                "eval_cell_key": "third_party_api::exp-a::internal_api::swebench-hard::model-a::model-a",
                "experiment_key": "third_party_api::exp-a::internal_api::swebench-hard::model-a::model-a",
                "source_kind": "third_party_api",
                "experiment_id": "exp-a",
                "provider_source": "internal_api",
                "dataset": "swebench-hard",
                "model_api_name": "model-a",
                "model_label": "model-a",
                "target": 1,
                "done": 1,
                "avg_path_read_precision": 1.0,
                "avg_chain_read_precision": 1.0,
            }
        ],
        "case_filter_model_metrics": {
            "others": [
                {
                    "eval_cell_key": "third_party_api::exp-b::internal_api::swebench-hard::model-b::model-b",
                    "experiment_key": "third_party_api::exp-b::internal_api::swebench-hard::model-b::model-b",
                    "source_kind": "third_party_api",
                    "experiment_id": "exp-b",
                    "provider_source": "internal_api",
                    "dataset": "swebench-hard",
                    "model_api_name": "model-b",
                    "model_label": "model-b",
                    "target": 99,
                    "done": 99,
                }
            ]
        },
        "path_metric_detail_count": 1,
        "dynamic_traceable_detail_count": 1,
        "runs": [],
        "details": [
            {
                "experiment_key": "third_party_api::exp-a::internal_api::swebench-hard::model-a::model-a",
                "eval_cell_key": "third_party_api::exp-a::internal_api::swebench-hard::model-a::model-a",
                "experiment_id": "exp-a",
                "source_kind": "third_party_api",
                "provider_source": "internal_api",
                "dataset": "swebench-hard",
                "model_label": "model-a",
                "instance_id": "case-a",
                "record_index": 0,
                "rollout_index": 0,
                "run_id": "run-a",
                "issue_description": "## Original issue body\n\n- with **details**\n\n```python\nprint('issue')\n```",
                "golden_patch": "diff --git a/pkg/root.py b/pkg/root.py\n--- a/pkg/root.py\n+++ b/pkg/root.py\n@@ -1 +1 @@\n-return bad\n+return good\n",
                "bonus_case_type": "direct",
                "resolved": True,
                "turns": 3.5,
                "tool_calls": 4.5,
                "wall_time": 5.5,
                "input_tokens": 1000,
                "output_tokens": 200,
                "reasoning_tokens": 50,
                "cache_hit_tokens": 500,
                "root_hit": True,
                "anchor_hit": True,
                "path_hit": True,
                "chain_hit": True,
                "order_score": -1.0,
                "order_defined": True,
                "miracle_step": True,
                "block_order_score": -1.0,
                "block_order_defined": True,
                "block_miracle_step": True,
                "path_evaluable": True,
                "chain_evaluable": True,
                "path_case_kind": "direct",
                "chain_case_kind": "direct",
                "path_node_recall": 1.0,
                "chain_node_recall": 1.0,
                "path_read_precision": 1.0,
                "chain_read_precision": 1.0,
                "edited_root_cause": True,
                "bad_patterns": {"has_loop": False, "error_spiral": False},
                "path_pattern_flags": {},
                "chain_bad_patterns": {},
                "path_projection": {
                    "anchors": ["pkg/symptom.py::symptom"],
                    "roots": ["pkg/root.py::root"],
                    "graph_context_nodes": [
                        {
                            "key": "tests/test_demo.py::test_demo",
                            "file_path": "tests/test_demo.py",
                            "start_line": 1,
                            "end_line": 10,
                            "normalized_distance": 1.0,
                            "node_role": "test_harness",
                            "hit": False,
                            "first_step": None,
                        }
                    ],
                    "path_nodes": [
                        {
                            "key": "pkg/symptom.py::symptom",
                            "file_path": "pkg/symptom.py",
                            "start_line": 1,
                            "end_line": 10,
                            "normalized_distance": 1.0,
                            "node_role": "symptom",
                            "hit": True,
                            "first_step": 0,
                        },
                        {
                            "key": "pkg/root.py::root",
                            "file_path": "pkg/root.py",
                            "start_line": 1,
                            "end_line": 10,
                            "normalized_distance": 0.0,
                            "node_role": "root_cause",
                            "hit": True,
                            "first_step": 1,
                            "source_preview": "def root(self, obj):\n    self.value = obj.good\n    return self.value",
                        },
                    ],
                    "path_edges": [],
                    "graph_context_edges": [
                        {"caller": "tests/test_demo.py::test_demo", "callee": "pkg/symptom.py::symptom"}
                    ],
                    "context_nodes": [
                        {
                            "key": "tests/test_demo.py::test_demo",
                            "file_path": "tests/test_demo.py",
                            "start_line": 1,
                            "end_line": 10,
                            "normalized_distance": 1.0,
                            "node_role": "test_harness",
                            "hit": False,
                            "first_step": None,
                        }
                    ],
                    "context_edges": [
                        {"caller": "tests/test_demo.py::test_demo", "callee": "pkg/symptom.py::symptom"}
                    ],
                },
                "chain_projection": {
                    "anchors": ["pkg/symptom.py::symptom"],
                    "roots": ["pkg/root.py::root"],
                    "context_nodes": [
                        {
                            "key": "tests/test_demo.py::test_demo",
                            "file_path": "tests/test_demo.py",
                            "start_line": 1,
                            "end_line": 10,
                            "normalized_distance": 1.0,
                            "node_role": "test_harness",
                            "hit": False,
                            "first_step": None,
                        }
                    ],
                    "chain_nodes": [
                        {
                            "key": "pkg/symptom.py::symptom",
                            "file_path": "pkg/symptom.py",
                            "start_line": 1,
                            "end_line": 10,
                            "normalized_distance": 1.0,
                            "node_role": "symptom",
                            "hit": True,
                            "first_step": 0,
                        },
                        {
                            "key": "pkg/root.py::root",
                            "file_path": "pkg/root.py",
                            "start_line": 1,
                            "end_line": 10,
                            "normalized_distance": 0.0,
                            "node_role": "root_cause",
                            "hit": True,
                            "first_step": 1,
                            "source_preview": "def root(self, obj):\n    self.value = obj.good\n    return self.value",
                        },
                    ],
                    "chain_edges": [],
                    "context_edges": [
                        {"caller": "tests/test_demo.py::test_demo", "callee": "pkg/symptom.py::symptom"}
                    ],
                },
                "graph_topology": {
                    "edges": [
                        {"source": "pkg/symptom.py::symptom", "target": "pkg/root.py::root"},
                        {"source": "tests/test_demo.py::test_demo", "target": "pkg/symptom.py::symptom"},
                    ]
                },
                "purpose_blocks": [
                    {
                        "block_index": 0,
                        "family": "read",
                        "target_path": "pkg/root.py",
                        "step_indices": [1, 2],
                        "trace_indices": [0, 1],
                        "achieved": True,
                        "wasted": False,
                        "loop": False,
                    }
                ],
                "step_inspection": [
                    {
                        "trace_index": 0,
                        "step_index": 0,
                        "tool_names": ["str_replace_editor"],
                        "tool_name": "str_replace_editor",
                        "action_family": "read",
                        "command": "view",
                        "path": "/testbed/pkg/symptom.py",
                        "view_range": [1, 10],
                        "thought": "inspect symptom",
                        "response_text": "view symptom",
                        "tool_args": [{"command": "view", "path": "/testbed/pkg/symptom.py", "view_range": [1, 10]}],
                        "tool_calls": [{"function": {"name": "str_replace_editor", "arguments": {"command": "view", "path": "/testbed/pkg/symptom.py", "view_range": [1, 10]}}}],
                        "tool_results": [{"observation": "symptom body"}],
                        "observation": "symptom body",
                        "execution_error": True,
                        "status": "error",
                        "recovered_reads": [{"file_path": "pkg/symptom.py", "start_line": 1, "end_line": 10}],
                        "scored": {
                            "trace_index": 0,
                            "step_index": 0,
                            "target_path": "pkg/symptom.py",
                            "n_reads": 1,
                            "reads": [{"file_path": "pkg/symptom.py", "start_line": 1, "end_line": 10}],
                            "hit_nodes": [{"key": "pkg/symptom.py::symptom", "node_role": "symptom"}],
                        },
                    },
                    {
                        "trace_index": 1,
                        "step_index": 1,
                        "tool_names": ["str_replace_editor"],
                        "tool_name": "str_replace_editor",
                        "action_family": "edit",
                        "command": "str_replace",
                        "path": "/testbed/pkg/root.py",
                        "old_str": "return bad",
                        "new_str": "return good",
                        "reasoning_text": "reason about root",
                        "chat_text": "view root",
                        "parsed_tool_calls": [
                            {
                                "name": "str_replace_editor",
                                "arguments": [
                                    {"key": "command", "value": "str_replace"},
                                    {"key": "path", "value": "/testbed/pkg/root.py"},
                                ],
                            }
                        ],
                        "thought": "inspect root",
                        "response_text": "view root",
                        "tool_args": [{"command": "str_replace", "path": "/testbed/pkg/root.py", "old_str": "return bad", "new_str": "return good"}],
                        "tool_calls": [{"function": {"name": "str_replace_editor", "arguments": {"command": "str_replace", "path": "/testbed/pkg/root.py"}}}],
                        "tool_results": [{"observation": "root body"}],
                        "observation": "root body",
                        "write_actions": [{"file_path": "pkg/root.py", "start_line": 1, "end_line": 999999, "command": "str_replace"}],
                        "edited_root_cause": True,
                        "recovered_reads": [{"file_path": "pkg/root.py", "start_line": 1, "end_line": 10}],
                        "scored": {
                            "trace_index": 1,
                            "step_index": 1,
                            "target_path": "pkg/root.py",
                            "n_reads": 1,
                            "reads": [{"file_path": "pkg/root.py", "start_line": 1, "end_line": 10}],
                            "hit_nodes": [{"key": "pkg/root.py::root", "node_role": "root_cause", "selected_issue_anchor": True}],
                            "write_hit_nodes": [{"key": "pkg/root.py::root", "node_role": "root_cause"}],
                        },
                    },
                ],
            },
            {
                "experiment_key": "third_party_api::exp-a::internal_api::swebench-hard::model-a::model-a",
                "eval_cell_key": "third_party_api::exp-a::internal_api::swebench-hard::model-a::model-a",
                "experiment_id": "exp-a",
                "source_kind": "third_party_api",
                "provider_source": "internal_api",
                "dataset": "swebench-hard",
                "model_label": "model-a",
                "instance_id": "case-a",
                "record_index": 2,
                "rollout_index": 1,
                "run_id": "run-a-1",
                "issue_description": "Second rollout issue body",
                "golden_patch": "diff --git a/pkg/root.py b/pkg/root.py\n+return good\n",
                "bonus_case_type": "direct",
                "resolved": False,
                "root_hit": False,
                "anchor_hit": True,
                "path_hit": True,
                "chain_hit": True,
                "path_evaluable": True,
                "chain_evaluable": True,
                "path_case_kind": "direct",
                "chain_case_kind": "direct",
                "path_node_recall": 0.5,
                "chain_node_recall": 0.5,
                "path_read_precision": 0.5,
                "chain_read_precision": 0.5,
                "bad_patterns": {"has_loop": False, "error_spiral": False},
                "path_pattern_flags": {},
                "chain_bad_patterns": {},
                "path_projection": {
                    "anchors": ["pkg/symptom.py::symptom"],
                    "roots": ["pkg/root.py::root"],
                    "path_nodes": [],
                    "path_edges": [],
                    "context_nodes": [],
                    "context_edges": [],
                },
                "chain_projection": {
                    "anchors": ["pkg/symptom.py::symptom"],
                    "roots": ["pkg/root.py::root"],
                    "chain_nodes": [],
                    "chain_edges": [],
                    "context_nodes": [],
                    "context_edges": [],
                },
                "purpose_blocks": [],
                "step_inspection": [
                    {
                        "trace_index": 0,
                        "step_index": 0,
                        "tool_names": ["str_replace_editor"],
                        "tool_name": "str_replace_editor",
                        "action_family": "read",
                        "command": "view",
                        "path": "/testbed/pkg/symptom.py",
                        "response_text": "second rollout",
                        "tool_args": [{"command": "view", "path": "/testbed/pkg/symptom.py"}],
                        "tool_calls": [{"function": {"name": "str_replace_editor", "arguments": {"command": "view", "path": "/testbed/pkg/symptom.py"}}}],
                        "tool_results": [{"observation": "second body"}],
                        "observation": "second body",
                        "recovered_reads": [{"file_path": "pkg/symptom.py", "start_line": 1, "end_line": 10}],
                        "scored": {
                            "trace_index": 0,
                            "step_index": 0,
                            "target_path": "pkg/symptom.py",
                            "n_reads": 1,
                            "reads": [{"file_path": "pkg/symptom.py", "start_line": 1, "end_line": 10}],
                            "hit_nodes": [{"key": "pkg/symptom.py::symptom", "node_role": "symptom"}],
                        },
                    }
                ],
            },
            {
                "experiment_key": "third_party_api::exp-b::internal_api::swebench-hard::model-b::model-b",
                "eval_cell_key": "third_party_api::exp-b::internal_api::swebench-hard::model-b::model-b",
                "experiment_id": "exp-b",
                "source_kind": "third_party_api",
                "provider_source": "internal_api",
                "dataset": "swebench-hard",
                "model_label": "model-b",
                "instance_id": "case-b",
                "record_index": 1,
                "bonus_case_type": "missing_bonus_map",
                "chain_evaluable": False,
                "step_inspection": [],
            },
        ],
    }
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    harness = tmp_path / "frontend_smoke.cjs"
    harness.write_text(
        textwrap.dedent(
            """
            const fs = require("fs");
            const vm = require("vm");
            const appPath = process.argv[2];
            const snapshot = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
            class Element {
              constructor(id) {
                this.id = id;
                this.innerHTML = "";
                this.textContent = "";
                this.dataset = {};
                this.scrollLeft = 0;
                this.scrollTop = 0;
                this.classList = { toggle() {}, add() {}, remove() {}, contains() { return false; } };
              }
              addEventListener() {}
            }
            const elements = new Map();
            const selectorElements = new Map();
            const document = {
              getElementById(id) {
                if (!elements.has(id)) elements.set(id, new Element(id));
                return elements.get(id);
              },
              querySelector(selector) {
                const value = selectorElements.get(selector);
                return Array.isArray(value) ? value[0] || null : value || null;
              },
              querySelectorAll(selector) {
                const value = selectorElements.get(selector);
                if (Array.isArray(value)) return value;
                return value ? [value] : [];
              },
            };
            const context = {
              window: { __P2A_DASHBOARD_SNAPSHOT__: snapshot },
              document,
              console,
              fetch: async () => { throw new Error("network disabled"); },
              setInterval: () => 1,
              clearInterval: () => {},
            };
            vm.createContext(context);
            vm.runInContext(fs.readFileSync(appPath, "utf8"), context);
            function run(expr) { return vm.runInContext(expr, context); }
            if (run("state.caseFilters.direct") !== true || run("state.caseFilters.latent") !== true || run("state.caseFilters.exposed") !== true || run("state.caseFilters.others") !== true) {
              throw new Error("default case filter should include all buckets");
            }
            if (run('BONUS_MAP_METRIC_CASE_TYPES.has("standard")') !== false) {
              throw new Error("dashboard should not expose legacy standard as a current case type");
            }
            const explicitLatentBucket = run(`detailCaseFilterBucket({
              bonus_case_type: "latent",
              path_evaluable: true,
              path_projection: {anchors: ["pkg/symptom.py::symptom"], roots: ["pkg/root.py::root"], path_edges: []}
            })`);
            if (explicitLatentBucket !== "latent") {
              throw new Error(`explicit latent detail should stay latent, got ${explicitLatentBucket}`);
            }
            const legacyStandardBucket = run(`detailCaseFilterBucket({
              bonus_case_type: "standard",
              path_evaluable: true,
              path_projection: {
                anchors: ["pkg/symptom.py::symptom"],
                roots: ["pkg/root.py::root"],
                path_edges: [{caller: "pkg/symptom.py::symptom", callee: "pkg/root.py::root"}]
              }
            })`);
            if (legacyStandardBucket !== "latent") {
              throw new Error(`legacy standard detail should canonicalize to latent, got ${legacyStandardBucket}`);
            }
            run("state.caseFilters.direct = true;");
            if (run("state.selectedDataset") !== "swebench-hard") {
              throw new Error("single dataset should be auto-selected");
            }
            if (run("state.selectedEvalCellKey") !== null || run("state.selectedExperimentKey") !== null) {
              throw new Error("multi-cell dataset must require explicit model/cell selection");
            }
            const escapedAttr = run(`esc('\\" onmouseover=\\"alert(1)')`);
            if (escapedAttr.includes('"') || !escapedAttr.includes("&quot;")) {
              throw new Error("esc should escape quotes for attribute contexts");
            }
            const expHtml = elements.get("experiment-table").innerHTML;
            if (!expHtml.includes("Datasets") || !expHtml.includes("Eval cells") || !expHtml.includes("exp-a") || !expHtml.includes("exp-b")) {
              throw new Error("dataset/eval-cell registry did not render");
            }
            if (!expHtml.includes("1 to rebuild") || !expHtml.includes("has-cache-pending")) {
              throw new Error("eval-cell registry should expose dashboard cache rebuild state");
            }
            run("state.admin.authenticated = true; renderExperiments(state.snapshot);");
            const adminExpHtml = elements.get("experiment-table").innerHTML;
            if (!adminExpHtml.includes("admin-delete-target") || !adminExpHtml.includes("admin-rebuild-select") || !adminExpHtml.includes("model_api_name")) {
              throw new Error("authenticated admin should see checkbox-based delete and rebuild controls");
            }
            run("renderAdminPanel(state.snapshot);");
            const adminPanelHtml = elements.get("admin-panel").innerHTML;
            if (!adminPanelHtml.includes("DB admin actions") || !adminPanelHtml.includes("Rebuild all") || !adminPanelHtml.includes("Rebuild selected") || !adminPanelHtml.includes("Preview delete")) {
              throw new Error("admin panel should align rebuild/delete actions without manual targets or typed confirmation");
            }
            run(`state.adminRebuildingTargets = [{dataset: "swebench-hard", provider_source: "internal_api", experiment_id: "exp-a", model_label: "model-a"}];
                 state.adminDeletingTargets = [{dataset: "swebench-hard", provider_source: "internal_api", experiment_id: "exp-b"}];
                 state.rebuildStatus = {active: true, phase: "warming", last_counts: {run_cells: 1}};
                 renderAdminPanel(state.snapshot);`);
            const activeAdminPanelHtml = elements.get("admin-panel").innerHTML;
            for (const needle of ["Rebuilding now:", "Deleting now:", "experiment=exp-a", "model=model-a", "experiment=exp-b", "Rebuilding (1 run cells)."]) {
              if (!activeAdminPanelHtml.includes(needle)) {
                throw new Error(`admin panel should show active admin operation target: ${needle}`);
              }
            }
            run("state.adminRebuildingTargets = []; state.adminDeletingTargets = []; state.rebuildStatus = null;");
            run("state.adminBusy = 'delete'; state.adminMessage = 'Deleting selected DB rows.'; renderAdminPanel(state.snapshot);");
            const busyAdminPanelHtml = elements.get("admin-panel").innerHTML;
            if (!busyAdminPanelHtml.includes("Deleting...") || !busyAdminPanelHtml.includes("Deleting selected DB rows.")) {
              throw new Error("admin delete panel should show immediate delete progress");
            }
            run("state.adminBusy = ''; state.adminMessage = '';");
            run("state.admin.authenticated = false; renderExperiments(state.snapshot);");
            const firstCell = snapshot.eval_cells[0].eval_cell_key;
            run(`state.selectedEvalCellKey = ${JSON.stringify("PLACEHOLDER")};
                 state.selectedExperimentKey = ${JSON.stringify("PLACEHOLDER")};
                 state.selectedTraceKey = ${JSON.stringify("TRACEKEY")};
                 state.selectedStepIndex = 1;
                 render();`.replaceAll("PLACEHOLDER", firstCell).replaceAll("TRACEKEY", `${firstCell}::case-a::idx-0`));
            const traceHtml = elements.get("trace-inspector").innerHTML;
            if (run("state.tracePanelOpen.graph") !== false || run("state.tracePanelOpen.steps") !== true || run("state.activeTracePanel") !== "steps") {
              throw new Error("trace workspace should default to graph collapsed and trace expanded");
            }
            for (const needle of ["trace-left", "trace-title-card", "trace-section", 'id="trace-graph-section"', 'id="trace-step-section"', "<summary>Graph</summary>", "<summary>Trace</summary>", "Instance overview", "Issue description", "markdown-body", "<strong>details</strong>", "language-python", "print", "Golden patch", "patch-view", "diff-meta", "diff-file", "diff-hunk", "diff-del", "diff-add", "diff --git", "<svg", "Show full Graph", "Edges", "Path edges", "Graph edges", "Trace edges", "case-a", "model-a · run-a", "1 context/harness nodes", "Showing edges: Path edges, Trace edges", "graph-trace-edge", "graph-edit-ring", "data-node-key"]) {
              if (!traceHtml.includes(needle)) throw new Error(`missing inspector fragment: ${needle}`);
            }
            if (traceHtml.includes('id="trace-graph-section" class="trace-section trace-graph-section" data-trace-panel="graph" open')) {
              throw new Error("graph section should be collapsed by default");
            }
            if (!traceHtml.includes('id="trace-step-section" class="trace-section trace-step-section" data-trace-panel="steps" open')) {
              throw new Error("trace section should be expanded by default");
            }
            for (const needle of ["trace-panel-tabs", ">Graph</button>", ">Trace</button>", "Node Source", "Click a graph node to inspect its source"]) {
              if (traceHtml.includes(needle)) throw new Error(`inspector should not render old panel/source fragment: ${needle}`);
            }
            for (const needle of ["Graph + Node Source", "Purpose Blocks + Step Detail", "Number is the first visited step", "Last edit landed on this node", "Faded node was not visited"]) {
              if (traceHtml.includes(needle)) throw new Error(`graph panel should not contain stale title or top legend fragment: ${needle}`);
            }
            const stepHtml = traceHtml;
            for (const needle of ["trace-middle", "trace-right", "step-thumb root-edit", "step-thumb symptom is-error", "data-step-tone=", "data-step-roles=", "Purpose blocks", "Visited Graph nodes", "pkg/root.py", "Reasoning", "Chat", "Tool calls", "str_replace_editor", "Action", "Observation", "Inline diff"]) {
              if (!stepHtml.includes(needle)) throw new Error(`missing step panel fragment: ${needle}`);
            }
            if (stepHtml.indexOf("Visited Graph nodes") > stepHtml.indexOf("Tool calls")) {
              throw new Error("step node hit summary should render above parsed tool calls");
            }
            for (const needle of ["Reward path", "Call graph", "Agent path", "Dependency-path edges", "Other call edges", "Agent-traversal edges", "dependency-path edges", "other call edges", "agent-traversal edges", "Dependency-path edge", "Other call edge", "Agent-traversal edge", "reward-path", "call-graph", "agent-path"]) {
              if (traceHtml.includes(needle)) throw new Error(`graph edge UI should not use stale wording: ${needle}`);
            }
            if (traceHtml.includes("trace-title-line")) {
              throw new Error("selected trace card should live in the graph header");
            }
            for (const needle of ["trace-row is-selected is-mixed", "trace-status-icons", "trace-icon-symptom", "trace-icon-root", "trace-icon-root-edit", "hit symptom", "hit root cause", "edited root cause"]) {
              if (!traceHtml.includes(needle)) throw new Error(`missing compact trace status: ${needle}`);
            }
            const rolloutToolbarHtml = elements.get("trace-global-rollout-slot").innerHTML;
            for (const needle of ["trace-global-rollout", 'id="trace-global-rollout-select"', "Rollout view", '<option value="0" selected>Rollout 1</option>']) {
              if (!rolloutToolbarHtml.includes(needle)) throw new Error(`missing toolbar rollout fragment: ${needle}`);
            }
            if (traceHtml.includes('id="trace-global-rollout-select"')) {
              throw new Error("global rollout selector should live in the trace toolbar, not the left trace pane");
            }
            for (const needle of ["trace-rollout-strip", "trace-rollout-segment is-resolved is-selected", "trace-rollout-segment is-unresolved", "1/2 success", "selected rollout 1", 'id="trace-rollout-select"', "Rollout 2 · failed"]) {
              if (!traceHtml.includes(needle)) throw new Error(`missing repeated-rollout trace fragment: ${needle}`);
            }
            if (!traceHtml.includes('<details class="detail-toggle " open><summary>Reasoning</summary>')) {
              throw new Error("reasoning toggle should be open by default");
            }
            const initialPatternStats = elements.get("trace-pattern-stats").innerHTML;
            for (const needle of ["Rollout</strong> 1", "Pass</strong> 100.0% / 100.0%", "Pass+tags</strong> 100.0% (1/1)", "Instances</strong> 1 / 1"]) {
              if (!initialPatternStats.includes(needle)) throw new Error(`missing initial pattern stats: ${needle} in ${initialPatternStats}`);
            }
            run("state.tracePatternFilters.hit_root_cause = 'false'; state.traceRolloutIndex = 0; renderTracePatternStats(state.snapshot);");
            const falsePatternStats = elements.get("trace-pattern-stats").innerHTML;
            for (const needle of ["Rollout</strong> 1", "Pass+tags</strong> 0.0% (0/1)", "Instances</strong> 0 / 1"]) {
              if (!falsePatternStats.includes(needle)) throw new Error(`missing false-pattern pass-share stats: ${needle} in ${falsePatternStats}`);
            }
            run("state.tracePatternFilters.hit_root_cause = true; state.traceRolloutIndex = 1; renderTracePatternStats(state.snapshot);");
            const filteredPatternStats = elements.get("trace-pattern-stats").innerHTML;
            for (const needle of ["Rollout</strong> 2", "Pass</strong> - / 0.0%", "Pass+tags</strong> - (0/0)", "Instances</strong> 0 / 1"]) {
              if (!filteredPatternStats.includes(needle)) throw new Error(`missing filtered pattern stats: ${needle} in ${filteredPatternStats}`);
            }
            run("state.tracePatternFilters.hit_root_cause = false; state.traceRolloutIndex = 0; renderTracePatternStats(state.snapshot);");
            if (run("rowKey(state.snapshot.details[0])") !== `${firstCell}::case-a::idx-0`) {
              throw new Error(`rollout 0 trace key should include rollout index: ${run("rowKey(state.snapshot.details[0])")}`);
            }
            if (run("rowKey(state.snapshot.details[1])") !== `${firstCell}::case-a::idx-1`) {
              throw new Error(`rollout 1 trace key should include rollout index: ${run("rowKey(state.snapshot.details[1])")}`);
            }
            if (run("groupedTraceDetails(state.snapshot).find((group) => group.key.endsWith('::case-a')).details.length") !== 2) {
              throw new Error("left trace list should group repeated rollouts under one instance row");
            }
            run("state.traceRolloutIndex = 1; state.selectedTraceKey = rowKey(state.snapshot.details[1]); renderTraceInspector(state.snapshot);");
            const rolloutTwoHtml = elements.get("trace-inspector").innerHTML;
            const rolloutTwoToolbarHtml = elements.get("trace-global-rollout-slot").innerHTML;
            if (!rolloutTwoToolbarHtml.includes('<option value="1" selected>Rollout 2</option>')) {
              throw new Error("toolbar rollout selector should switch to rollout 2");
            }
            for (const needle of ["model-a · run-a-1", "selected rollout 2", "Rollout 2 · failed", 'option value="' + firstCell + '::case-a::idx-1" selected']) {
              if (!rolloutTwoHtml.includes(needle)) throw new Error(`rollout selector did not switch detail: ${needle}`);
            }
            if (!rolloutTwoHtml.includes("trace-rollout-segment is-unresolved is-selected")) {
              throw new Error("global rollout switch should update left-row rollout thumbnail selection");
            }
            run("state.selectedTraceKey = rowKey(state.snapshot.details[0]); renderTraceInspector(state.snapshot);");
            if (run(`canonicalNodeRole("pre_symptom")`) !== "test_adapter") {
              throw new Error("legacy pre_symptom role should normalize to test_adapter");
            }
            if (run(`roleTone({node_role: "test_adapter"})`) !== "test-adapter") {
              throw new Error("test_adapter should use adapter graph tone");
            }
            if (run(`roleTone({node_role: "fix_adapter"})`) !== "fix-adapter") {
              throw new Error("fix_adapter should use adapter graph tone");
            }
            if (run(`nodeRoleLabel({node_role: "fix_adapter"})`) !== "fix-adapter") {
              throw new Error("fix_adapter should use public hyphenated label");
            }
            if (run(`roleTone({node_role: "symptom", selected_issue_anchor: true, patched_callable: true, patch_role: "fix_adapter"})`) !== "symptom-root-cause") {
              throw new Error("selected patched symptom should use dual graph tone");
            }
            if (run(`nodeRoleLabel({node_role: "symptom", selected_issue_anchor: true, patched_callable: true, patch_role: "fix_adapter"})`) !== "symptom + root cause") {
              throw new Error("selected patched symptom should use dual graph label");
            }
            const zeroBasedStepLabel = run(`displayStepLabel({step_index: 0}, {step_details: [{step_index: 0}]}, 0)`);
            if (zeroBasedStepLabel !== 1) {
              throw new Error(`zero-based stored step label should display as 1: ${zeroBasedStepLabel}`);
            }
            const oneBasedStepLabel = run(`displayStepLabel({step_index: 2}, {step_details: [{step_index: 1}, {step_index: 2}]}, 1)`);
            if (oneBasedStepLabel !== 2) {
              throw new Error(`one-based stored step label should stay 2: ${oneBasedStepLabel}`);
            }
            const fallbackStepLabel = run(`displayStepLabel({trace_index: 3}, {step_details: [{step_index: 0}]}, 3)`);
            if (fallbackStepLabel !== 4) {
              throw new Error(`missing step_index fallback should be one-based: ${fallbackStepLabel}`);
            }
            const blockLabel = run(`displayBlockIndex({block_index: 0}, {purpose_blocks: [{block_index: 0}]}, 0)`);
            if (blockLabel !== 1) {
              throw new Error(`zero-based block label should display as 1: ${blockLabel}`);
            }
            const derivedFirstSteps = run(`JSON.stringify([...displayFirstStepsByNode({step_details: [
              {trace_index: 0, step_index: 1, hit_nodes: []},
              {trace_index: 1, step_index: 2, hit_nodes: [{key: "pkg/symptom.py::symptom"}]}
            ]}).entries()])`);
            if (derivedFirstSteps !== JSON.stringify([["pkg/symptom.py::symptom", 2]])) {
              throw new Error(`Graph first-step labels should derive from step hits: ${derivedFirstSteps}`);
            }
            const traceEdgeSteps = run(`JSON.stringify(traceEdges({
              step_details: [
                {trace_index: 1, step_index: 2, hit_nodes: [{key: "A"}]},
                {trace_index: 98, step_index: 99, hit_nodes: [{key: "C"}]},
                {trace_index: 100, step_index: 101, hit_nodes: [{key: "A"}]},
                {trace_index: 101, step_index: 102, hit_nodes: [{key: "C"}]}
              ]
            }, {nodes: [{key: "A"}, {key: "C"}]}).map((edge) => [edge.source, edge.target, edge.first_step, edge.count]))`);
            if (traceEdgeSteps !== JSON.stringify([["A", "C", 99, 2], ["C", "A", 101, 1]])) {
              throw new Error(`Trace edge labels should use first target step, not flattened hop index: ${traceEdgeSteps}`);
            }
            const multiHitTraceEdges = run(`JSON.stringify(traceEdges({
              step_details: [
                {trace_index: 1, step_index: 2, hit_nodes: [{key: "A"}, {key: "B"}]},
                {trace_index: 98, step_index: 99, hit_nodes: [{key: "C"}]},
                {trace_index: 100, step_index: 101, hit_nodes: [{key: "A"}, {key: "B"}]},
                {trace_index: 101, step_index: 102, hit_nodes: [{key: "D"}, {key: "E"}]},
                {trace_index: 102, step_index: 103, hit_nodes: [{key: "D"}]}
              ]
            }, {nodes: [{key: "A"}, {key: "B"}, {key: "C"}, {key: "D"}, {key: "E"}]}).map((edge) => [edge.source, edge.target, edge.first_step, edge.count]))`);
            if (multiHitTraceEdges !== JSON.stringify([["A", "C", 99, 1], ["B", "C", 99, 1], ["C", "A", 101, 1], ["C", "B", 101, 1]])) {
              throw new Error(`Trace edges should connect across multi-hit steps only when one side is unambiguous: ${multiHitTraceEdges}`);
            }
            const blockedGraphRoute = run(`graphEdgeRouteOffset(
              {caller: "manager", callee: "update"},
              {x: 80, y: 55},
              {x: 540, y: 55},
              [{key: "manager"}, {key: "annotate"}, {key: "update"}],
              new Map([["manager", {x: 80, y: 55}], ["annotate", {x: 310, y: 55}], ["update", {x: 540, y: 55}]])
            )`);
            if (blockedGraphRoute <= 0) {
              throw new Error("long Graph edge should route around an intervening node");
            }
            const adjacentGraphRoute = run(`graphEdgeRouteOffset(
              {caller: "manager", callee: "annotate"},
              {x: 80, y: 55},
              {x: 310, y: 55},
              [{key: "manager"}, {key: "annotate"}, {key: "update"}],
              new Map([["manager", {x: 80, y: 55}], ["annotate", {x: 310, y: 55}], ["update", {x: 540, y: 55}]])
            )`);
            if (adjacentGraphRoute !== 0) {
              throw new Error("adjacent Graph edge should not be rerouted");
            }
            const routedGraphEdge = run(`graphEdgePath({x: 80, y: 55}, {x: 540, y: 55}, "context", ${blockedGraphRoute})`);
            const straightGraphEdge = run(`graphEdgePath({x: 80, y: 55}, {x: 540, y: 55}, "context", 0)`);
            if (routedGraphEdge === straightGraphEdge || !routedGraphEdge.includes("131.0")) {
              throw new Error(`routed Graph edge should bend away from covered node: ${routedGraphEdge}`);
            }
            const backwardTopEdge = run(`graphEdgePath({x: 540, y: 55}, {x: 80, y: 55}, "context", ${blockedGraphRoute})`);
            if (!backwardTopEdge.includes("131.0") || backwardTopEdge.includes("-21.0")) {
              throw new Error(`backward top-row Graph edge should route downward inside the SVG: ${backwardTopEdge}`);
            }
            const bottomGraphRoute = run(`graphEdgeRouteOffset(
              {caller: "manager", callee: "update"},
              {x: 80, y: 200},
              {x: 540, y: 200},
              [{key: "manager"}, {key: "annotate"}, {key: "update"}],
              new Map([["manager", {x: 80, y: 200}], ["annotate", {x: 310, y: 200}], ["update", {x: 540, y: 200}]])
            )`);
            if (bottomGraphRoute >= 0) {
              throw new Error("bottom-row Graph edge should route upward");
            }
            const backwardBottomEdge = run(`graphEdgePath({x: 540, y: 200}, {x: 80, y: 200}, "context", ${bottomGraphRoute})`);
            if (!backwardBottomEdge.includes("124.0") || backwardBottomEdge.includes("276.0")) {
              throw new Error(`backward bottom-row Graph edge should route upward inside the SVG: ${backwardBottomEdge}`);
            }
            if (run("combinedMiracleMarker({miracle_step: false, block_miracle_step: true})") !== false) {
              throw new Error("primary miracle marker should use step-level semantics");
            }
            if (run("combinedReverseMarker({order_score: 1, block_order_score: -1})") !== false) {
              throw new Error("primary reverse marker should use step-level semantics");
            }
            const blockOnlyPatternHtml = run(`traceStatusIcons({
              bonus_case_type: "standard",
              chain_case_kind: "standard",
              chain_evaluable: true,
              chain_projection: {anchors: ["a"], roots: ["b"], chain_edges: [{caller: "a", callee: "b"}]},
              purpose_blocks: [],
              anchor_hit: false,
              root_hit: false,
              miracle_step: false,
              block_miracle_step: true,
              order_score: 1,
              block_order_score: -1
            })`);
            if (blockOnlyPatternHtml.includes("trace-icon-miracle") || blockOnlyPatternHtml.includes("trace-icon-reverse")) {
              throw new Error("left trace icons should not promote block-level order diagnostics");
            }
            const separatedHitsTone = run(`stepTone({scored: {hit_nodes: [
              {key: "pkg/symptom.py::symptom", node_role: "symptom"},
              {key: "pkg/root.py::root", node_role: "root_cause"}
            ]}}, state.snapshot.details[0])`);
            if (separatedHitsTone === "symptom-root-cause") {
              throw new Error("separate symptom/root hits in one step should not use dual-node tone");
            }
            if (separatedHitsTone !== "multi-hit") {
              throw new Error("step that hits multiple roles should use split tone");
            }
            const splitRoles = run(`JSON.stringify(stepRoleSegments({scored: {hit_nodes: [
              {key: "pkg/symptom.py::symptom", node_role: "symptom"},
              {key: "framework/request.py::dispatch", node_role: "test_adapter"},
              {key: "pkg/mid.py::mid", node_role: "intermediate"},
              {key: "pkg/fix.py::adapter", node_role: "fix_adapter"},
              {key: "pkg/root.py::root", node_role: "root_cause"}
            ]}}, state.snapshot.details[0]))`);
            if (splitRoles !== JSON.stringify(["symptom", "test-adapter", "intermediate", "fix-adapter", "root"])) {
              throw new Error(`unexpected split roles: ${splitRoles}`);
            }
            const dualNodeTone = run(`stepTone({scored: {hit_nodes: [
              {key: "pkg/root.py::root", node_role: "root_cause", selected_issue_anchor: true}
            ]}}, state.snapshot.details[0])`);
            if (dualNodeTone !== "symptom-root-cause") {
              throw new Error("same callable with symptom and root-cause roles should use dual-node tone");
            }
            const patchedDualTone = run(`stepTone({scored: {hit_nodes: [
              {key: "pkg/fix.py::FixAdapter.wrap", node_role: "symptom", selected_issue_anchor: true, patched_callable: true, patch_role: "fix_adapter"}
            ]}}, state.snapshot.details[0])`);
            if (patchedDualTone !== "symptom-root-cause") {
              throw new Error("selected patched symptom should use dual step tone");
            }
            const dualSplitRoles = run(`JSON.stringify(stepRoleSegments({scored: {hit_nodes: [
              {key: "pkg/fix.py::FixAdapter.wrap", node_role: "symptom", selected_issue_anchor: true, patched_callable: true, patch_role: "fix_adapter"},
              {key: "framework/request.py::dispatch", node_role: "test_adapter"}
            ]}}, state.snapshot.details[0]))`);
            if (dualSplitRoles !== JSON.stringify(["symptom-root-cause", "test-adapter"])) {
              throw new Error(`unexpected dual split roles: ${dualSplitRoles}`);
            }
            const dualSplitThumb = run(`renderStepThumb({trace_index: 3, scored: {hit_nodes: [
              {key: "pkg/fix.py::FixAdapter.wrap", node_role: "symptom", selected_issue_anchor: true, patched_callable: true, patch_role: "fix_adapter"},
              {key: "framework/request.py::dispatch", node_role: "test_adapter"}
            ]}}, state.snapshot.details[0])`);
            for (const needle of ["step-thumb multi-hit", "data-step-roles=\\"symptom-root-cause,test-adapter\\"", "step-segment symptom-root-cause", "step-segment test-adapter"]) {
              if (!dualSplitThumb.includes(needle)) throw new Error(`missing dual split thumb fragment: ${needle}`);
            }
            const editTone = run(`stepTone({action_family: "edit", write_actions: [{file_path: "pkg/other.py"}]}, state.snapshot.details[0])`);
            if (editTone !== "edit") {
              throw new Error("non-root write step should use edit tone");
            }
            const execWriteTone = run(`stepTone({action_family: "exec", write_actions: [{file_path: "pkg/other.py"}]}, state.snapshot.details[0])`);
            if (execWriteTone !== "edit") {
              throw new Error("exec shell with parsed write semantics should use edit tone");
            }
            const execReadTone = run(`stepTone({action_family: "exec", scored: {hit_nodes: [{key: "pkg/root.py::root", node_role: "root_cause"}]}}, state.snapshot.details[0])`);
            if (execReadTone !== "root") {
              throw new Error("exec shell with parsed read hit should use map-hit tone");
            }
            const execOtherTone = run(`stepTone({action_family: "exec", scored: {n_reads: 0, hit_nodes: []}}, state.snapshot.details[0])`);
            if (execOtherTone !== "exec-other") {
              throw new Error("exec shell without parsed read/write semantics should use exec-other tone");
            }
            const nodeHitHtml = run(`renderStepNodeHits({scored: {hit_nodes: [
              {key: "pkg/symptom.py::Symptom.run", file_path: "pkg/symptom.py", start_line: 3, end_line: 8, node_role: "symptom"},
              {key: "framework/request.py::dispatch", file_path: "framework/request.py", start_line: 4, end_line: 4, node_role: "test_adapter"},
              {key: "pkg/mid.py::Middle.step", file_path: "pkg/mid.py", start_line: 9, end_line: 9, node_role: "intermediate"},
              {key: "pkg/fix.py::FixAdapter.wrap", file_path: "pkg/fix.py", start_line: 10, end_line: 12, node_role: "fix_adapter"},
              {key: "pkg/root.py::Root.fix", file_path: "pkg/root.py", start_line: 10, end_line: 20, node_role: "root_cause"}
            ]}}, state.snapshot.details[0])`);
            for (const needle of ["node-hit-group symptom", "node-hit-group test-adapter", "node-hit-group intermediate", "node-hit-group fix-adapter", "node-hit-group root", "pkg/symptom.py", "Symptom.run", "framework/request.py", "dispatch", "pkg/mid.py", "Middle.step", "pkg/fix.py", "FixAdapter.wrap", "pkg/root.py", "Root.fix", ":10-20"]) {
              if (!nodeHitHtml.includes(needle)) throw new Error(`missing step node hit fragment: ${needle}`);
            }
            if (!(nodeHitHtml.indexOf("node-hit-group symptom") < nodeHitHtml.indexOf("node-hit-group test-adapter") && nodeHitHtml.indexOf("node-hit-group test-adapter") < nodeHitHtml.indexOf("node-hit-group intermediate") && nodeHitHtml.indexOf("node-hit-group intermediate") < nodeHitHtml.indexOf("node-hit-group fix-adapter") && nodeHitHtml.indexOf("node-hit-group fix-adapter") < nodeHitHtml.indexOf("node-hit-group root"))) {
              throw new Error("node hit groups should render in Graph role order");
            }
            const dualNodeHitHtml = run(`renderStepNodeHits({scored: {hit_nodes: [
              {key: "pkg/fix.py::FixAdapter.wrap", file_path: "pkg/fix.py", start_line: 10, end_line: 12, node_role: "symptom", selected_issue_anchor: true, patched_callable: true, patch_role: "fix_adapter"}
            ]}}, state.snapshot.details[0])`);
            for (const needle of ["node-hit-group symptom-root-cause", "symptom + root cause", "pkg/fix.py", "FixAdapter.wrap"]) {
              if (!dualNodeHitHtml.includes(needle)) throw new Error(`missing dual node hit fragment: ${needle}`);
            }
            if (!stepHtml.includes('detail-toggle observation-toggle" open')) {
              throw new Error("observation toggle should default open");
            }
            const diffHtml = run(`renderDiff("old one\\\\nold two", "new one\\\\nnew two")`);
            if (!(diffHtml.indexOf("old one") < diffHtml.indexOf("old two") && diffHtml.indexOf("old two") < diffHtml.indexOf("new one") && diffHtml.indexOf("new one") < diffHtml.indexOf("new two"))) {
              throw new Error("inline diff should group delete lines before add lines for block replacements");
            }
            for (const needle of ['<span class="badge ok">symptom</span>', '<span class="badge ok">root cause</span>', '<span class="badge ok">chain</span>']) {
              if (traceHtml.includes(needle)) throw new Error(`left trace status should not use old badge: ${needle}`);
            }
            for (const needle of ['<span class="badge warn">miracle</span>', '<span class="badge warn">reverse</span>']) {
              if (traceHtml.includes(needle)) throw new Error(`edge-less graph should suppress badge: ${needle}`);
            }
            run("setTracePanelOpen('graph', true); renderTraceInspector(state.snapshot);");
            run("state.graphEdgeFilters.trace = false; renderTraceInspector(state.snapshot);");
            const noTraceHtml = elements.get("trace-inspector").innerHTML;
            if (noTraceHtml.includes("graph-trace-edge")) {
              throw new Error("Trace edge filter did not hide Trace arrows");
            }
            run("state.graphEdgeFilters.trace = true; renderTraceInspector(state.snapshot);");
            run("state.showGraphContext = false; state.graphEdgeFilters.graph = true; renderTraceInspector(state.snapshot);");
            const callGraphHtml = elements.get("trace-inspector").innerHTML;
            if (!callGraphHtml.includes("Core Path") || !callGraphHtml.includes("graph-edge context")) {
              throw new Error("graph filter should reveal in-scope Graph edges without full graph toggle");
            }
            for (const needle of ["tests/test_demo.py::test_demo", "test_harness"]) {
              if (callGraphHtml.includes(needle)) throw new Error(`call graph filter leaked full-graph node: ${needle}`);
            }
            run("state.graphEdgeFilters.graph = false; setGraphContext(true); renderTraceInspector(state.snapshot);");
            const fullGraphHtml = elements.get("trace-inspector").innerHTML;
            if (run("state.graphEdgeFilters.graph") !== true || !fullGraphHtml.includes("Showing edges: Path edges, Graph edges, Trace edges")) {
              throw new Error("full graph toggle should reveal Graph arrows");
            }
            for (const needle of ["Tool summary", "Recovered reads", "Matched bonus-map nodes"]) {
              if (stepHtml.includes(needle)) throw new Error(`stale inspector fragment: ${needle}`);
            }
            const legendHtml = elements.get("trace-legend").innerHTML;
            for (const needle of ["Trace Patterns", "Read Step Colors", "Write / Execute / Other Step Colors", "legend-icon", "Hit symptom: observed failure signal", "Hit root cause: expected cause or fix target", "Edited root cause: a write landed on a root-cause node", "Loop: repeated purpose block", "Reverse: traversal goes against dependency order", "Write action modified root cause", "Write action did not hit root cause", "Read step that hit multiple node roles", "Tool or command execution failed", "Exec or other tool without a parsed read hit", "exec / other", "Parsed read outside the Path", "Graph", "Nodes", "Edges", "Symbols", "test harness", "symptom", "intermediate", "Number is the first visited step", "test-adapter", "fix-adapter", "Last edit landed on this node", "Trace edge", "observed jump between adjacent Graph-hit steps", "Trace label 3x4 means first seen at step 3, repeated 4 times", "Faded node was not visited", "Path edge", "Graph edge", "Dependency direction", 'x1="0" y1="1" x2="1" y2="0"']) {
              if (!legendHtml.includes(needle)) throw new Error(`missing trajectory legend fragment: ${needle}`);
            }
            if (!fullGraphHtml.includes("Multi-hit read steps do not create internal edges") || !fullGraphHtml.includes("multi-hit to multi-hit transitions are omitted")) {
              throw new Error("graph note should state how multi-hit reads affect Trace edges");
            }
            if (!(legendHtml.indexOf("Symbols") < legendHtml.indexOf("Trace label 3x4 means first seen at step 3, repeated 4 times"))) {
              throw new Error("trace mxn label explanation should live in Symbols");
            }
            if (!fullGraphHtml.includes("mxn means first seen at step m and repeated n times")) {
              throw new Error("graph note should explain Trace edge mxn labels");
            }
            for (const needle of ["Path node", "Path hit"]) {
              if (legendHtml.includes(needle)) throw new Error(`graph legend should not use stale generic node label: ${needle}`);
            }
            if (legendHtml.includes("Trajectory labels and colors")) {
              throw new Error("trajectory legend should not render a title");
            }
            for (const needle of ["legend-step symptom-root-cause", "Read step that hit symptom + root cause"]) {
              if (!legendHtml.includes(needle)) throw new Error(`missing dual role legend fragment: ${needle}`);
            }
            const readGroupStart = legendHtml.indexOf("Read Step Colors");
            const otherGroupStart = legendHtml.indexOf("Write / Execute / Other Step Colors");
            const splitLegend = legendHtml.indexOf("Read step that hit multiple node roles");
            const dualLegend = legendHtml.indexOf("Read step that hit symptom + root cause");
            if (!(readGroupStart >= 0 && otherGroupStart > readGroupStart && splitLegend > otherGroupStart && dualLegend > splitLegend)) {
              throw new Error("split and S+RC legends should be read-labeled Other colors, with S+RC after split");
            }
            for (const needle of ["Icons", "legend-block"]) {
              if (legendHtml.includes(needle)) throw new Error(`trajectory legend should not include deleted block/loop legend: ${needle}`);
            }
            for (const needle of ['<span class="badge ok">symptom</span>', '<span class="badge ok">root cause</span>', '<span class="badge ok">chain</span>']) {
              if (legendHtml.includes(needle)) throw new Error(`trajectory legend should not use old badge sample: ${needle}`);
            }
            for (const needle of ["The number is the first trajectory step", "Selected issue/symptom", "Precomputed root-cause"]) {
              if (legendHtml.includes(needle)) throw new Error(`trajectory legend should not include long prose: ${needle}`);
            }
            const modelHtml = elements.get("model-table").innerHTML;
            for (const needle of ["KPI groups", "metric-group-checkbox", "metric-group-graph", "metric-group-path", "Metric definitions", "Filter totals", "Graph", "Outcome", "Path", "Pattern", "Purpose Blocks", "Efficiency", "Total instances", "Done traces", "Error traces", "ToDo traces", "Graph P.", "Graph R.", "Graph F1", "Path P.", "Path R.", "Path F1", "Symptom hit", "Root cause hit", "Evaluator-resolved pass rate", "Instances matching the current case filter"]) {
              if (!modelHtml.includes(needle)) throw new Error(`missing macro glossary fragment: ${needle}`);
            }
            for (const needle of ["Effect and Evidence", "Graph Hits", "Dependency Path", "Exploration Behavior", "Path-read hit ratio", "Trace P.", "Trace R.", "Trace F1", "Trace precision", "Trace recall", "Path node precision", "Path node recall", "Path read precision", "Path hit ratio", "Graph hit ratio", "Read recall", "Not defined: no canonical required-read set", "Scored read actions that hit useful"]) {
              if (modelHtml.includes(needle)) throw new Error(`metric glossary should use short prose: ${needle}`);
            }
            const allColumnHeaders = run("kpiColumns(false).map((column) => column.header).join('|')");
            if (!allColumnHeaders.includes("Graph P.") || !allColumnHeaders.includes("Path P.") || !allColumnHeaders.includes("Path F1")) {
              throw new Error("default KPI columns should include all metric groups");
            }
            if (allColumnHeaders.includes("Total instances") || allColumnHeaders.includes("Done traces") || allColumnHeaders.includes("Error traces") || allColumnHeaders.includes("ToDo traces")) {
              throw new Error(`filter totals should be hidden by default: ${allColumnHeaders}`);
            }
            if (!allColumnHeaders.startsWith("|Experiment|Kind|Model|")) {
              throw new Error(`fixed metric columns should be Experiment, Kind, Model: ${allColumnHeaders}`);
            }
            const defaultPassK = run("selectedRolloutK({rollouts_per_instance: 3})");
            const defaultPassValue = run("passAtValue({rollouts_per_instance: 3, pass_at: {'1': 0.25, '3': 0.75}, pass_at_n: 0.75})");
            if (defaultPassK !== 1 || defaultPassValue !== 0.25) {
              throw new Error(`Pass@K should default to k=1, got k=${defaultPassK} value=${defaultPassValue}`);
            }
            const expectedOrder = [
              "Graph P.", "Graph R.", "Graph F1",
              "Pass@K", "Avg@K", "Symptom hit", "Root cause hit", "First symptom", "First root cause",
              "Path P.", "Path R.", "Path F1",
              "Order score", "Reverse rate", "Miracle rate", "Loop trace", "Error spiral",
              "Blocks", "Achieved", "Wasted", "Loop blocks",
              "Turns"
            ].join("|");
            if (!allColumnHeaders.includes(expectedOrder)) {
              throw new Error(`KPI columns are not grouped by class: ${allColumnHeaders}`);
            }
            const pathP = run("kpiColumns(false).find((column) => column.header === 'Path P.').value({avg_path_node_precision: 0.25, avg_path_read_precision: 1.0, avg_path_node_recall: 0.5})");
            const pathF1 = run("kpiColumns(false).find((column) => column.header === 'Path F1').value({avg_path_node_precision: 0.25, avg_path_read_precision: 1.0, avg_path_node_recall: 0.5})");
            if (pathP !== "25.0%" || pathF1 !== "33.3%") {
              throw new Error(`Path P/F1 should use deduplicated node precision, got ${pathP}/${pathF1}`);
            }
            run("state.metricGroupFilters.graph = false;");
            const filteredColumnHeaders = run("kpiColumns(false).map((column) => column.header).join('|')");
            if (filteredColumnHeaders.includes("Graph P.") || filteredColumnHeaders.includes("Graph R.") || filteredColumnHeaders.includes("Graph F1")) {
              throw new Error("graph KPI group filter did not hide graph columns");
            }
            if (!filteredColumnHeaders.includes("Avg@K") || !filteredColumnHeaders.includes("Path P.")) {
              throw new Error("graph filter should not hide other KPI groups");
            }
            run("state.metricGroupFilters.graph = true;");
            run("state.selectedGraphNodeKey = 'pkg/root.py::root'; setTracePanelOpen('graph', true); renderTraceInspector(state.snapshot);");
            const sourceHtml = elements.get("trace-inspector").innerHTML;
            for (const needle of ["code-view language-python", "tok-keyword", "tok-symbol", "tok-lhs", "tok-dot", "tok-property", "def", "self", "value", "obj", "good", "pkg/root.py:1-10"]) {
              if (!sourceHtml.includes(needle)) throw new Error(`missing graph source fragment: ${needle}`);
            }
            const graphWrap = {scrollLeft: 320, scrollTop: 24};
            selectorElements.set("#trace-graph-pane .graph-wrap", graphWrap);
            const savedGraphScroll = run("captureInspectorScroll()");
            graphWrap.scrollLeft = 0;
            graphWrap.scrollTop = 0;
            run(`restoreInspectorScroll(${JSON.stringify(savedGraphScroll)})`);
            if (graphWrap.scrollLeft !== 320 || graphWrap.scrollTop !== 24) {
              throw new Error(`graph viewport scroll was not restored: ${graphWrap.scrollLeft}/${graphWrap.scrollTop}`);
            }
            run("state.selectedGraphNodeKey = null; renderTraceInspector(state.snapshot);");
            const collapsedSourceHtml = elements.get("trace-inspector").innerHTML;
            if (collapsedSourceHtml.includes("Node Source") || collapsedSourceHtml.includes("pkg/root.py:1-10")) {
              throw new Error("graph source panel should disappear after node selection is cleared");
            }
            run("state.caseFilters.direct = false; state.caseFilters.latent = false; state.caseFilters.exposed = false; state.caseFilters.others = true; render();");
            const othersModelHtml = elements.get("model-table").innerHTML;
            if (!othersModelHtml.includes("model-b") || othersModelHtml.includes("model-a")) {
              throw new Error("others case filter did not use backend filtered model rows");
            }
            if (othersModelHtml.includes("99")) {
              throw new Error("filter totals should remain hidden until explicitly enabled");
            }
            run("state.metricGroupFilters.filter_totals = true; render();");
            const othersTotalsHtml = elements.get("model-table").innerHTML;
            if (!othersTotalsHtml.includes("99")) {
              throw new Error("others case filter should use full-population backend metrics");
            }
            if (!othersTotalsHtml.includes("case types: others")) {
              throw new Error("case filter scope note missing");
            }
            run("state.caseFilters.direct = true; state.caseFilters.latent = true; state.caseFilters.exposed = true; state.caseFilters.others = false; render();");
            const filteredModelHtml = elements.get("model-table").innerHTML;
            if (!filteredModelHtml.includes("model-a") || filteredModelHtml.includes("model-b")) {
              throw new Error("direct/latent/exposed case filter did not use filtered model rows");
            }
            for (const needle of ["3.5", "4.5", "5.5", "1.0k"]) {
              if (!filteredModelHtml.includes(needle)) throw new Error(`filtered metrics missing efficiency fallback: ${needle}`);
            }
            const metricWrap = {dataset: {scrollKey: "model-kpi-table"}, scrollLeft: 640, scrollTop: 12};
            selectorElements.set(".table-wrap", [metricWrap]);
            run("render();");
            if (metricWrap.scrollLeft !== 640 || metricWrap.scrollTop !== 12) {
              throw new Error(`table scroll was not preserved across render: ${metricWrap.scrollLeft}/${metricWrap.scrollTop}`);
            }
            run("state.caseFilters.direct = true; state.caseFilters.latent = true; state.caseFilters.exposed = true; state.caseFilters.others = true; render();");
            const stepThumbs = (stepHtml.match(/step-thumb/g) || []).length;
            if (stepThumbs < 2) throw new Error("purpose block timeline did not render trace-indexed steps");
            const before = run("state.selectedTraceKey + '|' + state.selectedStepIndex");
            run("render();");
            const after = run("state.selectedTraceKey + '|' + state.selectedStepIndex");
            if (before !== after) throw new Error("refresh render did not preserve selected trace/step");
            """
        ),
        encoding="utf-8",
    )

    subprocess.run(["node", str(harness), str(app_path), str(snapshot_path)], check=True)


def test_dashboard_lazy_detail_loading_terminates_on_bad_tail_page(tmp_path):
    app_path = Path(__file__).resolve().parents[1] / "p2a" / "dashboard_static" / "app.js"
    snapshot = {
        "schema_version": "p2a_unified_dashboard_v1",
        "sources": [],
        "summary": {"counts": {"n_records": 2}, "rates": {}, "averages": {}, "distributions": {}, "trends": []},
        "datasets": [{"dataset": "ds", "n_instances": 1, "n_eval_cells": 1, "n_trajectories": 2}],
        "eval_cells": [
            {
                "eval_cell_key": "cell",
                "experiment_key": "cell",
                "source_kind": "third_party_api",
                "experiment_id": "exp",
                "provider_source": "internal_api",
                "dataset": "ds",
                "model_api_name": "model-api",
                "model_label": "model",
                "target": 1,
                "done_rollouts": 2,
                "errors": 0,
                "pending": 0,
                "detail_count": 1,
            }
        ],
        "model_metrics": [],
        "details": [
            {
                "cell_id": 101,
                "eval_cell_key": "cell",
                "experiment_key": "cell",
                "source_kind": "third_party_api",
                "experiment_id": "exp",
                "provider_source": "internal_api",
                "dataset": "ds",
                "model_api_name": "model-api",
                "model_label": "model",
                "instance_id": "case-a",
                "rollout_index": 0,
                "record_index": 0,
                "raw_available": True,
                "resolved": True,
                "step_inspection": [{"step_index": 0}],
            }
        ],
    }
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    harness = tmp_path / "frontend_lazy_details.cjs"
    harness.write_text(
        textwrap.dedent(
            """
            const fs = require("fs");
            const vm = require("vm");
            const appPath = process.argv[2];
            const baseSnapshot = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
            class Element {
              constructor(id) {
                this.id = id;
                this.innerHTML = "";
                this.textContent = "";
                this.value = "";
                this.checked = false;
                this.hidden = false;
                this.dataset = {};
                this.scrollLeft = 0;
                this.scrollTop = 0;
                this.classList = {toggle(){}, add(){}, remove(){}, contains(){ return false; }};
              }
              addEventListener() {}
            }
            const elements = new Map();
            const document = {
              getElementById(id) { if (!elements.has(id)) elements.set(id, new Element(id)); return elements.get(id); },
              querySelectorAll() { return []; },
              querySelector() { return null; },
            };
            let mode = "duplicate";
            let detailRequests = [];
            const duplicateDetail = {
              cell_id: 102,
              eval_cell_key: "cell",
              experiment_key: "cell",
              source_kind: "third_party_api",
              experiment_id: "exp",
              provider_source: "internal_api",
              dataset: "ds",
              model_api_name: "model-api",
              model_label: "model",
              instance_id: "case-a",
              rollout_index: 0,
              record_index: 1,
              raw_available: true,
              resolved: false,
              step_inspection: [{step_index: 0}],
            };
            const context = {
              window: {
                __P2A_DASHBOARD_SNAPSHOT__: JSON.parse(JSON.stringify(baseSnapshot)),
                location: {hash: ""},
                addEventListener() {},
                setTimeout(fn) { fn(); return 1; },
              },
              document,
              console,
              URLSearchParams,
              setInterval: () => 1,
              clearInterval: () => {},
              fetch: async (url) => {
                const text = String(url);
                if (text.startsWith("/api/details")) {
                  detailRequests.push(text);
                  return {
                    ok: true,
                    json: async () => ({
                      ok: true,
                      details: mode === "duplicate" ? [duplicateDetail] : [],
                      offset: 1,
                      limit: 5,
                    }),
                  };
                }
                return {ok: false, json: async () => ({})};
              },
            };
            vm.createContext(context);
            vm.runInContext(fs.readFileSync(appPath, "utf8"), context);
            function run(expr) { return vm.runInContext(expr, context); }
            async function runAsync(expr) { return await vm.runInContext(expr, context); }
            function resetSnapshot() {
              context.detailRequests = detailRequests = [];
              run(`state.snapshot = ${JSON.stringify(baseSnapshot)};
                   state.selectedDataset = "ds";
                   state.selectedEvalCellKey = "cell";
                   state.selectedExperimentKey = "cell";
                   state.selectedTraceKey = null;
                   state.detailLoadedCellKeys.clear();
                   state.detailLoadBusyKeys.clear();
                   state.detailLoadErrors = {};
                   state.detailLoadProgress = {};
                   state.metricLoadedDatasets.add("ds");`);
            }
            (async () => {
              resetSnapshot();
              run("state.detailLoadBusyKeys.add('cell'); state.detailLoadProgress.cell = {loaded: 1, total: 2}; renderTraceInspector(state.snapshot);");
              const partialHtml = run("document.getElementById('trace-inspector').innerHTML");
              if (!partialHtml.includes("trace-load-banner") || !partialHtml.includes("trace-left") || !partialHtml.includes("case-a")) {
                throw new Error(`busy detail loading should render already loaded traces: ${partialHtml}`);
              }
              resetSnapshot();
              mode = "duplicate";
              await runAsync("loadCellDetails('cell')");
              const keys = run("state.snapshot.details.map(rowKey)");
              if (keys.length !== 2 || !keys.includes("cell::case-a::cell-101") || !keys.includes("cell::case-a::cell-102")) {
                throw new Error(`DB cell_id should disambiguate duplicate rollout indexes: ${JSON.stringify(keys)}`);
              }
              if (run("loadedDetailCountForCell(state.snapshot, 'cell')") !== 2 || run("state.detailLoadedCellKeys.has('cell')") !== true) {
                throw new Error("duplicate rollout-index detail loading should reach the expected count");
              }
              const cellLocator = run("locatorForDetail(state.snapshot.details.find((item) => item.cell_id === 102), 'rollout')");
              if (!cellLocator.includes("cell_id=102") || cellLocator.includes("rollout_index=0")) {
                throw new Error(`DB detail locator should use cell_id: ${cellLocator}`);
              }
              resetSnapshot();
              mode = "empty";
              await runAsync("loadCellDetails('cell')");
              const placeholders = run("state.snapshot.details.filter((item) => item.dashboard_detail_load_error === true)");
              if (run("loadedDetailCountForCell(state.snapshot, 'cell')") !== 2 || placeholders.length !== 1 || run("state.detailLoadedCellKeys.has('cell')") !== true) {
                throw new Error(`empty tail page should create one terminal error placeholder: ${JSON.stringify(placeholders)}`);
              }
              if (!String(placeholders[0].error || "").includes("1/2")) {
                throw new Error(`placeholder should explain the incomplete load: ${placeholders[0].error}`);
              }
            })().catch((error) => {
              console.error(error);
              process.exitCode = 1;
            });
            """
        ),
        encoding="utf-8",
    )

    subprocess.run(["node", str(harness), str(app_path), str(snapshot_path)], check=True)


def test_dashboard_refresh_drops_selected_cell_details_but_preserves_others(tmp_path):
    app_path = Path(__file__).resolve().parents[1] / "p2a" / "dashboard_static" / "app.js"
    harness = tmp_path / "frontend_refresh_preserve.cjs"
    harness.write_text(
        textwrap.dedent(
            """
            const fs = require("fs");
            const vm = require("vm");
            const appPath = process.argv[2];
            class Element {
              constructor(id) {
                this.id = id;
                this.innerHTML = "";
                this.textContent = "";
                this.value = "";
                this.checked = false;
                this.hidden = false;
                this.dataset = {};
                this.scrollLeft = 0;
                this.scrollTop = 0;
                this.classList = {toggle(){}, add(){}, remove(){}, contains(){ return false; }};
              }
              addEventListener() {}
            }
            const elements = new Map();
            const document = {
              getElementById(id) { if (!elements.has(id)) elements.set(id, new Element(id)); return elements.get(id); },
              querySelectorAll() { return []; },
              querySelector() { return null; },
            };
            const baseRows = [
              {eval_cell_key: "cell-a", experiment_key: "cell-a", dataset: "ds", model_label: "model-a", experiment_id: "exp-a", provider_source: "internal_api", model_api_name: "model-a", done_rollouts: 1},
              {eval_cell_key: "cell-b", experiment_key: "cell-b", dataset: "ds", model_label: "model-b", experiment_id: "exp-b", provider_source: "internal_api", model_api_name: "model-b", done_rollouts: 1},
            ];
            const previousSnapshot = {
              datasets: [{dataset: "ds"}],
              eval_cells: baseRows,
              model_metrics: baseRows,
              details: [
                {eval_cell_key: "cell-a", experiment_key: "cell-a", dataset: "ds", instance_id: "case-a", rollout_index: 0, raw_available: true, step_inspection: [{step_index: 0}]},
                {eval_cell_key: "cell-b", experiment_key: "cell-b", dataset: "ds", instance_id: "case-b", rollout_index: 0, raw_available: true, step_inspection: [{step_index: 0}]},
              ],
            };
            const nextSnapshot = {datasets: [{dataset: "ds"}], eval_cells: baseRows, model_metrics: baseRows, details: []};
            const context = {
              window: {
                __P2A_DASHBOARD_SNAPSHOT__: JSON.parse(JSON.stringify(previousSnapshot)),
                location: {hash: ""},
                addEventListener() {},
                setTimeout(fn) { return 1; },
              },
              document,
              console,
              URLSearchParams,
              setInterval: () => 1,
              clearInterval: () => {},
              fetch: async () => ({ok: true, json: async () => ({ok: true, admin: false, admin_enabled: false})}),
            };
            vm.createContext(context);
            vm.runInContext(fs.readFileSync(appPath, "utf8"), context);
            function run(expr) { return vm.runInContext(expr, context); }
            context.previousSnapshot = previousSnapshot;
            context.nextSnapshot = JSON.parse(JSON.stringify(nextSnapshot));
            run("preserveLoadedDetails(nextSnapshot, previousSnapshot, {dropCellKeys: ['cell-a']})");
            const kept = context.nextSnapshot.details.map((detail) => detail.eval_cell_key + ":" + detail.instance_id);
            if (JSON.stringify(kept) !== JSON.stringify(["cell-b:case-b"])) {
              throw new Error(`manual refresh should reload selected cell while preserving others: ${JSON.stringify(kept)}`);
            }
            """
        ),
        encoding="utf-8",
    )

    subprocess.run(["node", str(harness), str(app_path)], check=True)


def test_dashboard_metrics_falls_back_to_eval_cells_when_endpoint_unavailable(tmp_path):
    app_path = Path(__file__).resolve().parents[1] / "p2a" / "dashboard_static" / "app.js"
    snapshot = {
        "schema_version": "p2a_unified_dashboard_v1",
        "sources": [],
        "summary": {"counts": {"n_records": 3}, "rates": {}, "averages": {}, "distributions": {}, "trends": []},
        "datasets": [{"dataset": "swebench-verified", "n_instances": 73, "n_eval_cells": 1, "n_trajectories": 219}],
        "eval_cells": [
            {
                "eval_cell_key": "cell",
                "experiment_key": "cell",
                "source_kind": "third_party_api",
                "experiment_id": "exp",
                "provider_source": "internal_api",
                "dataset": "swebench-verified",
                "model_api_name": "deepseek-v4",
                "model_label": "deepseek-v4",
                "target": 73,
                "target_rollouts": 219,
                "done_rollouts": 218,
                "errors": 1,
                "pending": 0,
                "cache_ready": 219,
                "cache_pending": 0,
            }
        ],
        "model_metrics": [],
        "case_filter_model_metrics": {},
        "details": [],
    }
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    harness = tmp_path / "frontend_metrics_501.cjs"
    harness.write_text(
        textwrap.dedent(
            """
            const fs = require("fs");
            const vm = require("vm");
            const appPath = process.argv[2];
            const snapshot = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
            class Element {
              constructor(id) {
                this.id = id;
                this.innerHTML = "";
                this.textContent = "";
                this.value = "";
                this.checked = false;
                this.hidden = false;
                this.dataset = {};
                this.scrollLeft = 0;
                this.scrollTop = 0;
                this.classList = {toggle(){}, add(){}, remove(){}, contains(){ return false; }};
              }
              addEventListener() {}
            }
            const elements = new Map();
            const document = {
              getElementById(id) { if (!elements.has(id)) elements.set(id, new Element(id)); return elements.get(id); },
              querySelectorAll() { return []; },
              querySelector() { return null; },
            };
            const context = {
              window: {
                __P2A_DASHBOARD_SNAPSHOT__: snapshot,
                location: {hash: ""},
                addEventListener() {},
                setTimeout(fn) { fn(); return 1; },
              },
              document,
              console,
              URLSearchParams,
              setInterval: () => 1,
              clearInterval: () => {},
              fetch: async (url) => {
                if (String(url).startsWith("/api/metrics")) {
                  return {ok: false, status: 501, json: async () => ({})};
                }
                return {ok: false, status: 404, json: async () => ({})};
              },
            };
            vm.createContext(context);
            vm.runInContext(fs.readFileSync(appPath, "utf8"), context);
            function run(expr) { return vm.runInContext(expr, context); }
            async function runAsync(expr) { return await vm.runInContext(expr, context); }
            (async () => {
              run(`state.snapshot = ${JSON.stringify(snapshot)};
                   state.selectedDataset = "swebench-verified";
                   state.caseFilters = {direct: true, latent: true, exposed: true, others: true};
                   state.metricLoadedDatasets.clear();
                   state.metricLoadBusyDatasets.clear();
                   state.metricLoadErrors = {};
                   state.metricLoadNotices = {};`);
              await runAsync("loadDatasetMetrics('swebench-verified')");
              run("renderModels(state.snapshot)");
              const html = elements.get("model-table").innerHTML;
              if (!html.includes("deepseek-v4") || !html.includes("Cached metrics endpoint is unavailable; showing snapshot metrics.")) {
                throw new Error(`Metrics panel should render eval-cell fallback rows after 501: ${html}`);
              }
              if (html.includes("Cached metrics failed to load") || run("state.metricLoadErrors['swebench-verified']") !== undefined) {
                throw new Error("HTTP 501 should be treated as unavailable lazy metrics, not a blocking metrics error");
              }
              const rows = run("activeModelMetrics(state.snapshot)");
              if (rows.length !== 1 || rows[0].target_rollouts !== 219 || rows[0].done_rollouts !== 218 || rows[0].errors !== 1) {
                throw new Error(`Eval-cell fallback metrics should preserve run counts: ${JSON.stringify(rows)}`);
              }
            })().catch((error) => {
              console.error(error);
              process.exitCode = 1;
            });
            """
        ),
        encoding="utf-8",
    )

    subprocess.run(["node", str(harness), str(app_path), str(snapshot_path)], check=True)


def test_dashboard_frontend_pattern_filters_and_permalinks(tmp_path):
    app_path = Path(__file__).resolve().parents[1] / "p2a" / "dashboard_static" / "app.js"
    snapshot = {
        "schema_version": "p2a_unified_dashboard_v1",
        "sources": [],
        "summary": {"counts": {"n_records": 3}, "trends": [], "distributions_by_dataset": {}},
        "datasets": [{"dataset": "ds", "n_instances": 2, "n_eval_cells": 1, "n_trajectories": 3}],
        "eval_cells": [
            {
                "eval_cell_key": "cell",
                "experiment_key": "cell",
                "source_kind": "third_party_api",
                "experiment_id": "exp",
                "provider_source": "internal_api",
                "dataset": "ds",
                "model_api_name": "model-api",
                "model_label": "model",
                "target": 2,
                "done": 2,
            }
        ],
        "model_metrics": [],
        "case_filter_model_metrics": {},
        "runs": [],
        "details": [
            {
                "eval_cell_key": "cell",
                "experiment_key": "cell",
                "experiment_id": "exp",
                "provider_source": "internal_api",
                "dataset": "ds",
                "model_api_name": "model-api",
                "model_label": "model",
                "instance_id": "case-a",
                "rollout_index": 0,
                "rollout_id": "stable-rollout",
                "record_index": 0,
                "bonus_case_type": "latent",
                "path_evaluable": True,
                "path_projection": {
                    "anchors": ["a.py::symptom"],
                    "roots": ["a.py::root"],
                    "path_edges": [{"caller": "a.py::symptom", "callee": "a.py::root"}],
                    "path_nodes": [
                        {"key": "a.py::symptom", "node_role": "symptom", "hit": True},
                        {"key": "a.py::root", "node_role": "root_cause", "hit": True},
                    ],
                    "context_nodes": [],
                },
                "order_score": -1,
                "order_defined": True,
                "miracle_step": True,
                "block_order_score": None,
                "block_miracle_step": None,
                "bad_patterns": {"has_loop": True, "error_spiral": False},
                "edited_root_cause": True,
                "step_inspection": [{"trace_index": 0, "step_index": 1, "tool_name": "read"}],
            },
            {
                "eval_cell_key": "cell",
                "experiment_key": "cell",
                "experiment_id": "exp",
                "provider_source": "internal_api",
                "dataset": "ds",
                "model_api_name": "model-api",
                "model_label": "model",
                "instance_id": "case-a",
                "rollout_index": 1,
                "record_index": 1,
                "bonus_case_type": "latent",
                "path_evaluable": True,
                "path_projection": {"anchors": [], "roots": [], "path_edges": [], "path_nodes": [], "context_nodes": []},
                "order_score": None,
                "order_defined": False,
                "miracle_step": None,
                "bad_patterns": {"has_loop": False, "error_spiral": False},
                "step_inspection": [],
            },
            {
                "eval_cell_key": "cell",
                "experiment_key": "cell",
                "experiment_id": "exp",
                "provider_source": "internal_api",
                "dataset": "ds",
                "model_api_name": "model-api",
                "model_label": "model",
                "instance_id": "case-b",
                "rollout_index": 0,
                "record_index": 2,
                "bonus_case_type": "latent",
                "path_evaluable": True,
                "path_projection": {"anchors": [], "roots": [], "path_edges": [], "path_nodes": [], "context_nodes": []},
                "order_score": -1,
                "order_defined": True,
                "miracle_step": None,
                "bad_patterns": {"has_loop": False, "error_spiral": False},
                "step_inspection": [],
            },
        ],
    }
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    harness = tmp_path / "frontend_pattern_permalink.js"
    harness.write_text(
        textwrap.dedent(
            """
            const fs = require("fs");
            const vm = require("vm");
            const appPath = process.argv[2];
            const snapshot = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
            class Element {
              constructor(id) { this.id = id; this.innerHTML = ""; this.textContent = ""; this.value = ""; this.checked = false; this.hidden = false; this.dataset = {}; this.classList = {toggle(){}, contains(){ return false; }}; }
              addEventListener() {}
            }
            const elements = new Map();
            const document = {
              getElementById(id) { if (!elements.has(id)) elements.set(id, new Element(id)); return elements.get(id); },
              querySelectorAll() { return []; },
              querySelector() { return null; },
            };
            const context = {
              window: {
                __P2A_DASHBOARD_SNAPSHOT__: snapshot,
                location: {origin: "http://dash", pathname: "/index.html", hash: ""},
                addEventListener() {},
              },
              document,
              console,
              URLSearchParams,
              fetch: async () => ({ok: false, json: async () => ({})}),
              setInterval: () => 1,
              clearInterval: () => {},
            };
            vm.createContext(context);
            vm.runInContext(fs.readFileSync(appPath, "utf8"), context);
            function run(expr) { return vm.runInContext(expr, context); }
            const fallbackSnapshot = {
              datasets: [{dataset: "ds"}],
              eval_cells: [{eval_cell_key: "cell", experiment_key: "cell", dataset: "ds", model_label: "model", experiment_id: "exp", provider_source: "internal_api"}],
              model_metrics: [],
              details: [
                {eval_cell_key: "cell", experiment_key: "cell", dataset: "ds", model_label: "model", experiment_id: "exp", provider_source: "internal_api", instance_id: "empty", record_index: 0, dashboard_cache_pending: true},
                {eval_cell_key: "cell", experiment_key: "cell", dataset: "ds", model_label: "model", experiment_id: "exp", provider_source: "internal_api", instance_id: "raw", record_index: 1, raw_available: true, step_inspection: [{step_index: 0}]},
              ],
            };
            run("state.caseFilters = {direct: true, latent: true, exposed: true, others: true}; state.selectedDataset = 'ds'; state.selectedEvalCellKey = 'cell'; state.selectedTraceKey = null;");
            context.fallbackSnapshot = fallbackSnapshot;
            run("ensureSelection(fallbackSnapshot);");
            if (run("state.selectedTraceKey") !== run("rowKey(fallbackSnapshot.details[1])") || run("rowKey(selectedDetail(fallbackSnapshot))") !== run("rowKey(fallbackSnapshot.details[1])")) {
              throw new Error("trace selection should prefer raw details over empty pending placeholders");
            }
            run("state.caseFilters = {direct: true, latent: true, exposed: true, others: true}; state.selectedDataset = 'ds'; state.selectedEvalCellKey = 'cell';");
            run("state.tracePatternFilters.miracle = true; state.tracePatternFilters.reverse = true;");
            const grouped = run("groupedTraceDetails(state.snapshot).map((group) => [group.key, group.details.map(rowKey)]);");
            if (grouped.length !== 1 || grouped[0][0] !== "cell::case-a" || grouped[0][1].length !== 1 || !grouped[0][1][0].endsWith("id-stable-rollout")) {
              throw new Error(`pattern filters should keep only rollout with all selected tags: ${JSON.stringify(grouped)}`);
            }
            if (run("tracePatternMatches(state.snapshot.details[1], 'miracle')") !== false) {
              throw new Error("undefined miracle marker should not match");
            }
            const cycleStates = run("[null, 'true', 'false', 'none'].map(nextTracePatternFilterValue)");
            if (JSON.stringify(cycleStates) !== JSON.stringify(["true", "false", "none", null])) {
              throw new Error(`pattern button should cycle neutral -> true -> false -> none -> neutral: ${JSON.stringify(cycleStates)}`);
            }
            if (run("tracePatternMatches({miracle_step: false}, 'miracle', 'false')") !== true) {
              throw new Error("false pattern filter should match explicit false marker");
            }
            if (run("tracePatternMatches({miracle_step: null}, 'miracle', 'none')") !== true) {
              throw new Error("none pattern filter should match unavailable marker");
            }
            run("state.tracePatternFilters = {miracle: 'none', reverse: null, loop: null, hit_symptom: null, hit_root_cause: null, edited_root_cause: null};");
            const noneGroups = run("groupedTraceDetails(state.snapshot).map((group) => [group.key, group.details.map(rowKey)]);");
            if (noneGroups.length !== 2 || noneGroups.some((group) => group[1].some((key) => key.endsWith("id-stable-rollout")))) {
              throw new Error(`none pattern filter should keep only unavailable miracle markers: ${JSON.stringify(noneGroups)}`);
            }
            const stepHash = run("state.selectedTraceKey = rowKey(state.snapshot.details[0]); state.selectedStepIndex = 4; locatorForDetail(state.snapshot.details[0], 'step');");
            if (!stepHash.includes("rollout_id=stable-rollout") || stepHash.includes("record_index") || !stepHash.includes("step_index=4")) {
              throw new Error(`step locator should use stable rollout id and step index: ${stepHash}`);
            }
            const instanceHash = run("locatorForDetail(state.snapshot.details[0], 'instance');");
            if (instanceHash.includes("rollout_id") || instanceHash.includes("rollout_index") || !instanceHash.includes("instance_id=case-a")) {
              throw new Error(`instance locator should stop at instance: ${instanceHash}`);
            }
            const experimentHash = run("locatorForDetail(state.snapshot.details[0], 'experiment');");
            if (experimentHash.includes("instance_id") || experimentHash.includes("rollout_id") || !experimentHash.includes("experiment_id=exp")) {
              throw new Error(`experiment locator should stop at experiment: ${experimentHash}`);
            }
            run("state.caseFilters = {direct: false, latent: true, exposed: false, others: false}; state.tracePatternFilters.loop = true;");
            const applied = run(`applyLocator(state.snapshot, parseLocator("http://dash/index.html${stepHash}"))`);
            if (applied !== true) throw new Error("valid locator did not apply");
            if (run("state.selectedEvalCellKey") !== "cell" || run("state.selectedTraceKey") !== "cell::case-a::id-stable-rollout" || run("state.selectedStepIndex") !== 4) {
              throw new Error("locator did not restore selected cell/trace/step");
            }
            if (run("Object.values(state.caseFilters).every(Boolean)") !== true || run("activeTracePatternFilters().length") !== 0 || run("state.traceQuery") !== "") {
              throw new Error("deep link should clear conflicting filters");
            }
            """
        ),
        encoding="utf-8",
    )

    subprocess.run(["node", str(harness), str(app_path), str(snapshot_path)], check=True)
