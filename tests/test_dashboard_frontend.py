import json
import subprocess
import textwrap
from pathlib import Path


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
