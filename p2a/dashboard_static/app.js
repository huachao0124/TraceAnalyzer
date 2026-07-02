const state = {
  snapshot: window.__P2A_DASHBOARD_SNAPSHOT__ || null,
  activeTab: "overview",
  selectedDataset: null,
  selectedEvalCellKey: null,
  selectedExperimentKey: null,
  selectedTraceKey: null,
  traceGlobalRolloutIndex: 0,
  selectedStepIndex: 0,
  selectedGraphNodeKey: null,
  activeTracePanel: "steps",
  tracePanelOpen: { graph: false, steps: true },
  traceQuery: "",
  refreshTimer: null,
  loadingSnapshot: false,
  queuedSnapshotOptions: null,
  rebuildStatus: null,
  rebuildStatusTimer: null,
  detailLoadBusyKeys: new Set(),
  detailLoadedCellKeys: new Set(),
  detailLoadErrors: {},
  detailLoadProgress: {},
  metricLoadedDatasets: new Set(),
  metricLoadBusyDatasets: new Set(),
  metricLoadErrors: {},
  metricLoadNotices: {},
  snapshotBusy: "",
  operationMessage: "",
  operationTone: "",
  caseFilters: { direct: true, latent: true, exposed: true, others: true },
  metricGroupFilters: {
    filter_totals: false,
    graph: true,
    outcome: true,
    path: true,
    exploration_behavior: true,
    purpose_blocks: true,
    efficiency_cost: true,
  },
  showGraphContext: false,
  graphEdgeFilters: { path: true, graph: false, trace: true },
  passAtK: 1,
  tracePatternFilters: {
    miracle: null,
    reverse: null,
    loop: null,
    hit_symptom: null,
    hit_root_cause: null,
    edited_root_cause: null,
  },
  permalinkNotice: "",
  permalinkMissing: false,
  pendingLocator: null,
  admin: { enabled: false, authenticated: false },
  adminDeleteKeys: new Set(),
  adminRebuildKeys: new Set(),
  adminDeletingTargets: [],
  adminRebuildingTargets: [],
  adminManualTarget: null,
  adminPreview: null,
  adminMessage: "",
  adminBusy: "",
};

const BONUS_MAP_METRIC_CASE_TYPES = new Set(["direct", "latent", "exposed"]);
const CASE_FILTER_BUCKETS = ["direct", "latent", "exposed", "others"];
const TRACE_PATTERN_FILTERS = ["miracle", "reverse", "loop", "hit_symptom", "hit_root_cause", "edited_root_cause"];
const TRACE_PATTERN_FILTER_VALUES = ["true", "false", "none"];

const MACRO_METRIC_GROUPS = [
  {
    key: "filter_totals",
    title: "Filter totals",
    items: [
      ["Total instances", "Instances matching the current case filter."],
      ["Done traces", "Completed traces matching the current case filter."],
      ["Error traces", "Traces in the current case filter that ended with an error."],
      ["ToDo traces", "Traces in the current case filter that have not completed."],
    ],
  },
  {
    key: "graph",
    title: "Graph",
    items: [
      ["Graph P.", "Parsed reads that hit a Graph node."],
      ["Graph R.", "Graph nodes hit by the agent Trace."],
      ["Graph F1", "Harmonic mean of Graph P. and Graph R."],
    ],
  },
  {
    key: "outcome",
    title: "Outcome",
    items: [
      ["First root cause", "Average first step that reached a root cause."],
      ["First symptom", "Average first step that reached the symptom."],
      ["Root cause hit", "Reached a precomputed cause or fix target."],
      ["Symptom hit", "Reached the observed failure signal."],
      ["Task success", "Evaluator-resolved pass rate."],
    ],
  },
  {
    key: "path",
    title: "Path",
    items: [
      ["Path P.", "Unique Trace-hit Graph nodes that are on the issue symptom-to-root-cause Path."],
      ["Path R.", "Unique Path nodes hit by the agent Trace."],
      ["Path F1", "Harmonic mean of Path P. and Path R."],
    ],
  },
  {
    key: "exploration_behavior",
    title: "Pattern",
    note: "Reverse and miracle rates use latent traces whose marker is defined, so their denominator can be smaller than the latent trace count.",
    items: [
      ["Order score", "Whether graph hits move from symptom toward root cause."],
      ["Reverse rate", "Filtered Trace share with reverse traversal marker."],
      ["Miracle rate", "Filtered Trace share with miracle marker."],
      ["Loop trace", "Traces with repeated exploration behavior."],
      ["Error spiral", "Long consecutive tool-error runs."],
    ],
  },
  {
    key: "purpose_blocks",
    title: "Purpose Blocks",
    items: [
      ["Blocks", "Average intention blocks per Trace."],
      ["Achieved", "Read blocks that hit useful graph nodes."],
      ["Wasted", "Read blocks with no map payoff."],
      ["Loop blocks", "Blocks repeating the same exploration intent."],
    ],
  },
  {
    key: "efficiency_cost",
    title: "Efficiency",
    items: [
      ["Turns/Tools/Wall", "Average turns, tool calls, and seconds."],
      ["In/Out/Reason", "Average provider token counts."],
      ["Cache hit/write", "Provider prompt-cache token ratios."],
    ],
  },
];

const TRACE_LEGEND_GROUPS = [
  {
    title: "Trace Patterns",
    items: [
      { sample: '<span class="legend-icon">🔎</span>', text: "Hit symptom: observed failure signal." },
      { sample: '<span class="legend-icon">🎯</span>', text: "Hit root cause: expected cause or fix target." },
      { sample: '<span class="legend-icon">✎</span>', text: "Edited root cause: a write landed on a root-cause node." },
      { sample: '<span class="legend-icon">🔁</span>', text: "Loop: repeated purpose block." },
      { sample: '<span class="legend-icon">✨</span>', text: "Miracle: cause hit before enough graph evidence." },
      { sample: '<span class="legend-icon">🌀</span>', text: "Reverse: traversal goes against dependency order." },
    ],
  },
  {
    title: "Read Step Colors",
    items: [
      { sample: '<span class="legend-step symptom"><span class="legend-step-num">3</span><span>symptom</span></span>', text: "This step hit symptom." },
      { sample: '<span class="legend-step test-adapter"><span class="legend-step-num">4</span><span>test-adapter</span></span>', text: "This step hit a bug-side adapter before the symptom." },
      { sample: '<span class="legend-step intermediate"><span class="legend-step-num">5</span><span>intermediate</span></span>', text: "This step hit an intermediate Graph node." },
      { sample: '<span class="legend-step fix-adapter"><span class="legend-step-num">6</span><span>fix-adapter</span></span>', text: "This step hit an upstream patched adapter." },
      { sample: '<span class="legend-step root"><span class="legend-step-num">7</span><span>root cause</span></span>', text: "This step hit root cause." },
      { sample: '<span class="legend-step offmap"><span class="legend-step-num">9</span><span>off Path</span></span>', text: "Parsed read outside the Path." },
    ],
  },
  {
    title: "Write / Execute / Other Step Colors",
    items: [
      { sample: '<span class="legend-step root-edit"><span class="legend-step-num">8</span><span>root edit</span></span>', text: "Write action modified root cause." },
      { sample: '<span class="legend-step edit"><span class="legend-step-num">4</span><span>edit</span></span>', text: "Write action did not hit root cause." },
      { sample: '<span class="legend-step neutral is-error"><span class="legend-step-num">6</span><span>failed</span></span>', text: "Tool or command execution failed." },
      { sample: '<span class="legend-step exec-other"><span class="legend-step-num">2</span><span>exec / other</span></span>', text: "Exec or other tool without a parsed read hit." },
      { sample: '<span class="legend-step multi-hit" style="--step-bg: linear-gradient(90deg, #dcfce7 0 20%, #ffedd5 20% 40%, #dbeafe 40% 60%, #fce7f3 60% 80%, #fee2e2 80% 100%);"><span class="legend-step-num">3</span><span>split</span></span>', text: "Read step that hit multiple node roles." },
      { sample: '<span class="legend-step symptom-root-cause"><span class="legend-step-num">3</span><span>S+RC</span></span>', text: "Read step that hit symptom + root cause." },
    ],
  },
];

const GRAPH_LEGEND_GROUPS = [
  {
    title: "Nodes",
    items: [
      { sample: '<svg class="legend-graph-sample" viewBox="0 0 124 38"><g class="graph-node test" transform="translate(20,19)"><circle r="14"></circle></g><text class="legend-graph-text" x="42" y="16">test harness</text><text class="legend-graph-sub" x="42" y="29">non-rewardable</text></svg>', text: "Test files, runners, and fixtures." },
      { sample: '<svg class="legend-graph-sample" viewBox="0 0 124 38"><g class="graph-node test-adapter" transform="translate(20,19)"><circle r="14"></circle></g><text class="legend-graph-text" x="42" y="16">test-adapter</text><text class="legend-graph-sub" x="42" y="29">before symptom</text></svg>', text: "Non-test frame before the symptom anchor." },
      { sample: '<svg class="legend-graph-sample" viewBox="0 0 124 38"><g class="graph-node symptom" transform="translate(20,19)"><circle r="14"></circle></g><text class="legend-graph-text" x="42" y="16">symptom</text><text class="legend-graph-sub" x="42" y="29">issue anchor</text></svg>', text: "Deepest Graph frame matched by the issue description." },
      { sample: '<svg class="legend-graph-sample" viewBox="0 0 124 38"><g class="graph-node path" transform="translate(20,19)"><circle r="14"></circle></g><text class="legend-graph-text" x="42" y="16">intermediate</text><text class="legend-graph-sub" x="42" y="29">rewardable</text></svg>', text: "Rewardable program frame between symptom and root cause." },
      { sample: '<svg class="legend-graph-sample" viewBox="0 0 124 38"><g class="graph-node fix-adapter" transform="translate(20,19)"><circle r="14"></circle></g><text class="legend-graph-text" x="42" y="16">fix-adapter</text><text class="legend-graph-sub" x="42" y="29">patched upstream</text></svg>', text: "Patched callable upstream of the terminal root cause." },
      { sample: '<svg class="legend-graph-sample" viewBox="0 0 124 38"><g class="graph-node root" transform="translate(20,19)"><circle r="14"></circle></g><text class="legend-graph-text" x="42" y="16">root cause</text><text class="legend-graph-sub" x="42" y="29">terminal patch</text></svg>', text: "Terminal patched callable or component." },
    ],
  },
  {
    title: "Edges",
    items: [
      { sample: '<svg class="legend-graph-sample" viewBox="0 0 124 38"><defs><marker id="legend-arrow-path" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#2563eb"></path></marker></defs><path class="graph-edge path" d="M12 20 C42 6, 74 6, 108 20" marker-end="url(#legend-arrow-path)"></path></svg>', text: "Path edge: fixed Graph edge on the symptom-to-root Path." },
      { sample: '<svg class="legend-graph-sample" viewBox="0 0 124 38"><defs><marker id="legend-arrow-context" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#667085"></path></marker></defs><path class="graph-edge context" d="M12 20 C42 32, 74 32, 108 20" marker-end="url(#legend-arrow-context)"></path></svg>', text: "Graph edge: fixed Graph edge outside the Path." },
      { sample: '<svg class="legend-graph-sample" viewBox="0 0 124 38"><defs><marker id="legend-arrow-trace" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#c2410c"></path></marker></defs><path class="graph-edge trace" d="M12 20 C42 6, 74 32, 108 20" marker-end="url(#legend-arrow-trace)"></path></svg>', text: "Trace edge: observed jump between adjacent Graph-hit steps when at least one side is single-hit." },
      { sample: '<svg class="legend-graph-sample" viewBox="0 0 124 38"><defs><marker id="legend-arrow-order" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#2563eb"></path></marker></defs><circle cx="16" cy="19" r="10" fill="#dcfce7" stroke="#15803d" stroke-width="2"></circle><path class="graph-edge path" d="M28 19 C48 19, 62 19, 82 19" marker-end="url(#legend-arrow-order)"></path><circle cx="98" cy="19" r="10" fill="#fee2e2" stroke="#b42318" stroke-width="2"></circle></svg>', text: "Dependency direction: symptom to root cause." },
    ],
  },
  {
    title: "Symbols",
    items: [
      { sample: '<svg class="legend-graph-sample" viewBox="0 0 124 38"><g class="graph-node symptom hit" transform="translate(20,19)"><circle r="14"></circle><text class="graph-step" y="4">7</text></g><text class="legend-graph-text" x="42" y="16">hit step</text><text class="legend-graph-sub" x="42" y="29">step 7</text></svg>', text: "Number is the first visited step." },
      { sample: '<svg class="legend-graph-sample" viewBox="0 0 124 38"><text class="graph-trace-label" x="18" y="23">3x4</text><text class="legend-graph-text" x="54" y="16">trace label</text><text class="legend-graph-sub" x="54" y="29">step x repeats</text></svg>', text: "Trace label 3x4 means first seen at step 3, repeated 4 times." },
      { sample: '<svg class="legend-graph-sample" viewBox="0 0 124 38"><g class="graph-node path miss" transform="translate(20,19)"><circle r="14"></circle></g><text class="legend-graph-text" x="42" y="16">not hit</text><text class="legend-graph-sub" x="42" y="29">faded</text></svg>', text: "Faded node was not visited." },
      { sample: '<svg class="legend-graph-sample" viewBox="0 0 124 38"><g class="graph-node path hit" transform="translate(20,19)"><circle r="14"></circle><text class="graph-step" y="4">4</text></g><text class="legend-graph-text" x="42" y="16">save x3</text><text class="legend-graph-sub" x="42" y="29">same span</text></svg>', text: "Multiple symbols share one source span." },
      { sample: '<svg class="legend-graph-sample" viewBox="0 0 124 38"><defs><linearGradient id="graph-symptom-root-cause-fill" x1="0" y1="1" x2="1" y2="0"><stop offset="50%" stop-color="#dcfce7"></stop><stop offset="50%" stop-color="#fee2e2"></stop></linearGradient></defs><g class="graph-node symptom-root-cause hit" transform="translate(20,19)"><circle r="14"></circle><text class="graph-step" y="4">2</text></g><text class="legend-graph-text" x="42" y="16">S+RC</text><text class="legend-graph-sub" x="42" y="29">same callable</text></svg>', text: "Same callable has both roles." },
      { sample: '<svg class="legend-graph-sample" viewBox="0 0 124 38"><g class="graph-node root hit edited" transform="translate(20,19)"><circle r="14"></circle><circle class="graph-edit-ring" r="18"></circle><text class="graph-step" y="4">8</text></g><text class="legend-graph-text" x="42" y="16">final edit</text><text class="legend-graph-sub" x="42" y="29">purple ring</text></svg>', text: "Last edit landed on this node." },
    ],
  },
];

function esc(value) {
  return String(value ?? "-")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(digits);
  return String(value);
}

function pct(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${(value * 100).toFixed(1)}%`;
}

function selectedRolloutK(row) {
  const n = rolloutN(row) || 1;
  return Math.max(1, Math.min(Number(state.passAtK || 1), n));
}

function avgAtValue(row, key) {
  const k = selectedRolloutK(row);
  return row?.avg_at?.[String(k)]?.[key] ?? row?.[key];
}

function avgAtStd(row, key) {
  const k = selectedRolloutK(row);
  return row?.avg_at_std?.[String(k)]?.[key] ?? row?.[`${key}_std`];
}

function withStd(row, key, formatter = fmt, digits = 3) {
  const value = avgAtValue(row, key);
  const shown = formatter === fmt ? formatter(value, digits) : formatter(value);
  const std = avgAtStd(row, key);
  if (selectedRolloutK(row) <= 1 || std === null || std === undefined || Number.isNaN(std)) return shown;
  const stdShown = formatter === fmt ? formatter(std, digits) : formatter(std);
  return `${shown} ± ${stdShown}`;
}

function passAtValue(row) {
  const n = rolloutN(row) || 1;
  const k = selectedRolloutK(row);
  return row?.pass_at?.[String(k)] ?? (k === n ? row?.pass_at_n : null);
}

function pathF1Value(row) {
  const pathF1Key = avgAtValue(row, "avg_path_node_f1") === null || avgAtValue(row, "avg_path_node_f1") === undefined
    ? "avg_chain_node_f1"
    : "avg_path_node_f1";
  if (avgAtValue(row, pathF1Key) !== null && avgAtValue(row, pathF1Key) !== undefined) {
    return withStd(row, pathF1Key, pct);
  }
  const precisionKey = avgAtValue(row, "avg_path_node_precision") === null || avgAtValue(row, "avg_path_node_precision") === undefined
    ? "avg_chain_node_precision"
    : "avg_path_node_precision";
  const recallKey = avgAtValue(row, "avg_path_node_recall") === null || avgAtValue(row, "avg_path_node_recall") === undefined
    ? "avg_chain_node_recall"
    : "avg_path_node_recall";
  return pct(f1(avgAtValue(row, precisionKey), avgAtValue(row, recallKey)));
}

function numeric(value) {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function avg(values) {
  const real = values.map(numeric).filter((value) => value !== null);
  return real.length ? real.reduce((sum, value) => sum + value, 0) / real.length : null;
}

function rate(values) {
  const real = values.filter((value) => value !== null && value !== undefined);
  return real.length ? real.filter(Boolean).length / real.length : null;
}

function sum(values) {
  return values.map(numeric).filter((value) => value !== null).reduce((total, value) => total + value, 0);
}

function f1(precision, recall) {
  const p = numeric(precision);
  const r = numeric(recall);
  if (p === null || r === null) return null;
  const denom = p + r;
  return denom ? 2 * p * r / denom : 0;
}

function token(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  if (value < 1000) return String(Math.round(value));
  if (value < 1000000) return `${(value / 1000).toFixed(1)}k`;
  return `${(value / 1000000).toFixed(1)}m`;
}

function badge(label, active = true, tone = "") {
  return `<span class="badge ${active ? tone : ""}">${esc(label)}</span>`;
}

function table(headers, rows) {
  if (!rows.length) return '<div class="empty">No rows.</div>';
  return `<table><thead><tr>${headers.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead><tbody>${rows.join("")}</tbody></table>`;
}

function metricGroupClass(groupKey) {
  return groupKey ? `metric-group-${String(groupKey).replaceAll("_", "-")}` : "";
}

function metricGroupLabel(groupKey) {
  return MACRO_METRIC_GROUPS.find((group) => group.key === groupKey)?.title || groupKey;
}

function metricGroupEnabled(groupKey) {
  return state.metricGroupFilters?.[groupKey] !== false;
}

function renderMetricGroupControls() {
  return `<fieldset class="metric-group-filter" aria-label="KPI groups">
    <legend>KPI groups</legend>
    ${MACRO_METRIC_GROUPS.map((group) => `<label class="${metricGroupClass(group.key)}">
      <input class="metric-group-checkbox" type="checkbox" data-metric-group="${esc(group.key)}" ${metricGroupEnabled(group.key) ? "checked" : ""}>
      ${esc(group.title)}
    </label>`).join("")}
  </fieldset>`;
}

function renderKpiTable(columns, rows) {
  if (!rows.length) return '<div class="empty">No rows.</div>';
  const header = columns.map((column) => {
    const groupClass = column.group ? metricGroupClass(column.group) : "";
    const title = column.group ? ` title="${esc(metricGroupLabel(column.group))}"` : "";
    return `<th class="${esc(groupClass)}"${title}>${esc(column.header)}</th>`;
  }).join("");
  const body = rows.map((row) => {
    const key = cellKey(row);
    const cells = columns.map((column) => {
      const groupClass = column.group ? metricGroupClass(column.group) : "";
      const value = column.value(row, key);
      return `<td class="${esc(groupClass)}">${column.html ? value : esc(value)}</td>`;
    }).join("");
    return `<tr class="${key === state.selectedEvalCellKey ? "is-selected" : ""}" data-eval-cell-key="${esc(key)}">${cells}</tr>`;
  }).join("");
  return `<table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;
}

function tableScrollKey(el, index) {
  if (el?.dataset?.scrollKey) return el.dataset.scrollKey;
  if (el?.id) return el.id;
  const parentId = typeof el?.closest === "function" ? el.closest("[id]")?.id : "";
  return parentId ? `${parentId}:${index}` : `table:${index}`;
}

function captureTableScroll() {
  const scroll = {};
  document.querySelectorAll(".table-wrap").forEach((el, index) => {
    scroll[tableScrollKey(el, index)] = { left: el.scrollLeft || 0, top: el.scrollTop || 0 };
  });
  return scroll;
}

function restoreTableScroll(scrollState) {
  if (!scrollState) return;
  document.querySelectorAll(".table-wrap").forEach((el, index) => {
    const saved = scrollState[tableScrollKey(el, index)];
    if (!saved) return;
    el.scrollLeft = saved.left || 0;
    el.scrollTop = saved.top || 0;
  });
}

function glossaryItem(item) {
  const term = Array.isArray(item) ? item[0] : item.term;
  const text = Array.isArray(item) ? item[1] : item.text;
  const sample = Array.isArray(item) ? "" : item.sample;
  return `<div class="glossary-item ${sample ? "has-sample" : ""}">
    ${sample ? `<div class="legend-sample">${sample}</div>` : ""}
    <div class="glossary-copy"><strong>${esc(term)}</strong><span>${esc(text)}</span></div>
  </div>`;
}

function glossary(title, items) {
  return `<section class="glossary" aria-label="${esc(title)}">
    <div class="glossary-title">${esc(title)}</div>
    <div class="glossary-grid">${items.map(glossaryItem).join("")}</div>
  </section>`;
}

function metricDefinitions(title, groups) {
  return `<section class="metric-defs" aria-label="${esc(title)}">
    <div class="glossary-title">${esc(title)}</div>
    <div class="metric-def-grid">${groups.map((group) => `<section class="metric-def-group">
      <h4>${esc(group.title)}</h4>
      ${group.note ? `<p class="metric-def-note">${esc(group.note)}</p>` : ""}
      <div class="metric-def-rows">${[...group.items].sort(([a], [b]) => a.localeCompare(b)).map(([term, text]) => `<div class="metric-def-row">
        <strong>${esc(term)}</strong>
        <span>${esc(text)}</span>
      </div>`).join("")}</div>
    </section>`).join("")}</div>
  </section>`;
}

function visualLegend(title, items) {
  const groups = items.length && items[0].items ? items : [{ title: "", items }];
  const titleHtml = title ? `<div class="glossary-title">${esc(title)}</div>` : "";
  return `<section class="glossary visual-legend" aria-label="${esc(title || "Legend")}">
    ${titleHtml}
    ${groups.map((group) => `<div class="visual-legend-group">
      ${group.title ? `<div class="visual-legend-group-title">${esc(group.title)}</div>` : ""}
      <div class="visual-legend-grid">${group.items.map((item) => `<div class="visual-legend-item">
        <div class="legend-sample">${item.sample || ""}</div>
        <div class="visual-legend-text">${esc(item.text || "")}</div>
      </div>`).join("")}</div>
    </div>`).join("")}
  </section>`;
}

function captureInspectorScroll() {
  const graphWrap = typeof document.querySelector === "function" ? document.querySelector("#trace-graph-pane .graph-wrap") : null;
  return {
    left: document.getElementById("trace-left-pane")?.scrollTop || 0,
    graph: document.getElementById("trace-graph-pane")?.scrollTop || 0,
    graphWrapLeft: graphWrap?.scrollLeft || 0,
    graphWrapTop: graphWrap?.scrollTop || 0,
    middle: document.getElementById("trace-middle-pane")?.scrollTop || 0,
    right: document.getElementById("trace-right-pane")?.scrollTop || 0,
  };
}

function restoreInspectorScroll(scrollState) {
  if (!scrollState) return;
  const apply = () => {
    const pairs = [
      ["trace-left-pane", scrollState.left],
      ["trace-graph-pane", scrollState.graph],
      ["trace-middle-pane", scrollState.middle],
      ["trace-right-pane", scrollState.right],
    ];
    pairs.forEach(([id, scrollTop]) => {
      const el = document.getElementById(id);
      if (el) el.scrollTop = scrollTop || 0;
    });
    const graphWrap = typeof document.querySelector === "function" ? document.querySelector("#trace-graph-pane .graph-wrap") : null;
    if (graphWrap) {
      graphWrap.scrollLeft = scrollState.graphWrapLeft || 0;
      graphWrap.scrollTop = scrollState.graphWrapTop || 0;
    }
  };
  if (typeof requestAnimationFrame === "function") requestAnimationFrame(apply);
  else apply();
}

function setGraphContext(show) {
  state.showGraphContext = Boolean(show);
  if (state.showGraphContext) state.graphEdgeFilters.graph = true;
}

function resetTracePanels() {
  state.activeTracePanel = "steps";
  state.tracePanelOpen = { graph: false, steps: true };
}

function tracePanelOpen(panel) {
  if (panel === "graph") return state.tracePanelOpen?.graph === true;
  if (panel === "steps") return state.tracePanelOpen?.steps !== false;
  return false;
}

function setTracePanelOpen(panel, open) {
  state.tracePanelOpen = { graph: false, steps: true, ...(state.tracePanelOpen || {}), [panel]: Boolean(open) };
  if (open) state.activeTracePanel = panel;
}

function rowKey(detail) {
  const cellId = detail?.cell_id;
  const rolloutId = detail?.rollout_id
    ? `id-${detail.rollout_id}`
    : (cellId !== null && cellId !== undefined && cellId !== "" ? `cell-${cellId}` : `idx-${rolloutIndex(detail)}`);
  return `${traceInstanceKey(detail)}::${rolloutId}`;
}

function traceInstanceId(detail) {
  return detail?.instance_id || `record-${detail?.record_index ?? 0}`;
}

function traceInstanceKey(detail) {
  return `${detailCellKey(detail) || "cell"}::${traceInstanceId(detail)}`;
}

function normalizeRolloutIndex(value) {
  const number = Number(value);
  return Number.isFinite(number) && number >= 0 ? Math.trunc(number) : 0;
}

function rolloutIndex(detail) {
  return normalizeRolloutIndex(detail?.rollout_index);
}

function traceGlobalRolloutIndex() {
  return normalizeRolloutIndex(state.traceGlobalRolloutIndex);
}

function setTraceGlobalRolloutIndex(value) {
  state.traceGlobalRolloutIndex = normalizeRolloutIndex(value);
}

function maxRolloutIndex(details) {
  return (details || []).reduce((maxValue, detail) => Math.max(maxValue, rolloutIndex(detail)), 0);
}

function clampTraceGlobalRolloutIndex(details) {
  const maxIndex = maxRolloutIndex(details);
  if (traceGlobalRolloutIndex() > maxIndex) setTraceGlobalRolloutIndex(maxIndex);
}

function compareTraceDetails(a, b) {
  const instanceCmp = String(traceInstanceId(a)).localeCompare(String(traceInstanceId(b)));
  if (instanceCmp) return instanceCmp;
  const rolloutCmp = rolloutIndex(a) - rolloutIndex(b);
  if (rolloutCmp) return rolloutCmp;
  return Number(a?.record_index ?? 0) - Number(b?.record_index ?? 0);
}

function detailsForInstance(details, instanceKey) {
  return details
    .filter((detail) => traceInstanceKey(detail) === instanceKey)
    .sort(compareTraceDetails);
}

function detailForRolloutIndex(details, index = traceGlobalRolloutIndex()) {
  const sorted = [...(details || [])].sort(compareTraceDetails);
  return sorted.find((detail) => rolloutIndex(detail) === index) || sorted[0] || null;
}

function detailHasRawTrace(detail) {
  return Boolean(
    detail?.raw_available
    || (detail?.messages || []).length
    || (detail?.trajectory || []).length
    || (detail?.step_details || []).length
    || (detail?.step_inspection || []).length
  );
}

function detailsForCell(snapshot, key) {
  return (snapshot?.details || []).filter((detail) => detailCellKey(detail) === key);
}

function rawDetailCountForCell(snapshot, key) {
  return detailsForCell(snapshot, key).filter(detailHasRawTrace).length;
}

function loadedDetailCountForCell(snapshot, key) {
  return detailsForCell(snapshot, key).length;
}

function currentCellKeys(snapshot) {
  return new Set(experimentRows(snapshot).map(cellKey).filter(Boolean));
}

function syncDetailLoadState(snapshot) {
  const keys = currentCellKeys(snapshot);
  [...state.detailLoadedCellKeys].forEach((key) => {
    if (!keys.has(key)) state.detailLoadedCellKeys.delete(key);
  });
  Object.keys(state.detailLoadErrors).forEach((key) => {
    if (!keys.has(key)) delete state.detailLoadErrors[key];
  });
  Object.keys(state.detailLoadProgress).forEach((key) => {
    if (!keys.has(key)) delete state.detailLoadProgress[key];
  });
}

function preserveLoadedDetails(nextSnapshot, previousSnapshot, options = {}) {
  if (!nextSnapshot || !previousSnapshot) {
    syncDetailLoadState(nextSnapshot);
    return;
  }
  const keys = currentCellKeys(nextSnapshot);
  const dropKeys = new Set(options.dropCellKeys || []);
  const rawByCell = new Map();
  (previousSnapshot.details || []).forEach((detail) => {
    const key = detailCellKey(detail);
    if (!keys.has(key) || dropKeys.has(key) || !detailHasRawTrace(detail)) return;
    if (!rawByCell.has(key)) rawByCell.set(key, []);
    rawByCell.get(key).push(detail);
  });
  rawByCell.forEach((details, key) => mergeCellDetails(nextSnapshot, key, details));
  syncDetailLoadState(nextSnapshot);
}

function datasetRows(snapshot) {
  if (snapshot?.datasets?.length) return snapshot.datasets;
  const names = new Set();
  (snapshot?.eval_cells || snapshot?.experiments || []).forEach((row) => names.add(row.dataset || "unknown-dataset"));
  (snapshot?.details || []).forEach((row) => names.add(row.dataset || row.data_source || "unknown-dataset"));
  return [...names].sort().map((dataset) => ({ dataset }));
}

function activeDatasetStats(snapshot) {
  if (allCaseFiltersEnabled()) return null;
  const key = activeCaseFilterKey();
  const stats = key ? snapshot?.case_filter_dataset_stats?.[key] : null;
  if (stats?.datasets?.length) return stats;
  return computedCaseFilterDatasetStats(snapshot);
}

function computedCaseFilterDatasetStats(snapshot) {
  const rows = new Map();
  const ensureRow = (dataset) => {
    const key = dataset || "unknown-dataset";
    if (!rows.has(key)) {
      rows.set(key, {
        dataset: key,
        instances: new Set(),
        cells: new Set(),
        traces: 0,
        models: new Set(),
        sourceKinds: new Set(),
      });
    }
    return rows.get(key);
  };
  datasetRows(snapshot).forEach((row) => ensureRow(row.dataset));
  caseFilteredDetails(snapshot).forEach((detail) => {
    const row = ensureRow(detail.dataset || detail.data_source);
    row.instances.add(traceInstanceId(detail));
    row.cells.add(detailCellKey(detail));
    row.traces += 1;
    if (detail.model_label) row.models.add(String(detail.model_label));
    if (detail.source_kind) row.sourceKinds.add(String(detail.source_kind));
  });
  const datasets = [...rows.values()].map((row) => ({
    dataset: row.dataset,
    n_instances: row.instances.size,
    n_eval_cells: row.cells.size,
    n_trajectories: row.traces,
    models: [...row.models].sort(),
    source_kinds: [...row.sourceKinds].sort(),
  })).sort((a, b) => String(a.dataset).localeCompare(String(b.dataset)));
  return {
    datasets,
    totals: {
      n_datasets: datasets.length,
      n_instances: sum(datasets.map((row) => row.n_instances || 0)),
      n_eval_cells: sum(datasets.map((row) => row.n_eval_cells || 0)),
      n_trajectories: sum(datasets.map((row) => row.n_trajectories || 0)),
    },
  };
}

function overviewDatasetRows(snapshot) {
  const stats = activeDatasetStats(snapshot);
  return stats?.datasets || datasetRows(snapshot);
}

function overviewDatasetTotals(snapshot) {
  const stats = activeDatasetStats(snapshot);
  if (stats?.totals) return stats.totals;
  const rows = datasetRows(snapshot);
  return {
    n_datasets: rows.length,
    n_instances: sum(rows.map((row) => row.n_instances || 0)),
    n_eval_cells: sum(rows.map((row) => row.n_eval_cells || 0)),
    n_trajectories: sum(rows.map((row) => row.n_trajectories || 0)),
  };
}

function experimentRows(snapshot) {
  return snapshot?.eval_cells || snapshot?.experiments || [];
}

function cellKey(row) {
  return row?.eval_cell_key || row?.experiment_key || "";
}

function detailCellKey(detail) {
  return detail?.eval_cell_key || detail?.experiment_key || "";
}

function pathValue(detail, currentKey, legacyKey, fallback = undefined) {
  if (!detail) return fallback;
  if (detail[currentKey] !== undefined) return detail[currentKey];
  if (detail[legacyKey] !== undefined) return detail[legacyKey];
  return fallback;
}

function pathProjection(detail) {
  const projection = pathValue(detail, "path_projection", "chain_projection", {});
  return projection && typeof projection === "object" ? projection : {};
}

function pathNodes(detail) {
  const projection = pathProjection(detail);
  const nodes = projection.path_nodes || projection.chain_nodes || [];
  return Array.isArray(nodes) ? nodes : [];
}

function graphContextNodes(detail) {
  const nodes = pathProjection(detail).context_nodes || [];
  return Array.isArray(nodes) ? nodes : [];
}

function pathEdges(detail) {
  const projection = pathProjection(detail);
  const edges = projection.path_edges || projection.chain_edges || [];
  return Array.isArray(edges) ? edges : [];
}

function canonicalCaseType(raw, detail) {
  const value = String(raw || "");
  if (BONUS_MAP_METRIC_CASE_TYPES.has(value)) return value;
  if (value !== "standard") return value;
  const projection = pathProjection(detail);
  const roots = new Set(projection.roots || []);
  const anchors = new Set(projection.anchors || []);
  const edges = pathEdges(detail);
  if (!roots.size || !anchors.size || !edges.length) return "exposed";
  return [...roots].some((root) => anchors.has(root)) ? "exposed" : "latent";
}

function detailCaseType(detail) {
  const raw = detail?.bonus_case_type || pathValue(detail, "path_case_kind", "chain_case_kind", "") || "";
  return canonicalCaseType(raw, detail);
}

function detailCaseFilterBucket(detail) {
  const caseType = detailCaseType(detail);
  if (pathValue(detail, "path_evaluable", "chain_evaluable") === true && CASE_FILTER_BUCKETS.includes(caseType) && caseType !== "others") return caseType;
  return "others";
}

function isPathMetricDetail(detail) {
  return pathValue(detail, "path_evaluable", "chain_evaluable") === true && BONUS_MAP_METRIC_CASE_TYPES.has(detailCaseType(detail));
}

function hasDualSymptomRoot(detail) {
  const projection = pathProjection(detail);
  const roots = new Set(projection.roots || []);
  return (projection.anchors || []).some((anchor) => roots.has(anchor));
}

function hasPathEdges(detail) {
  return pathEdges(detail).length > 0;
}

function isOrderMetricDetail(detail) {
  return isPathMetricDetail(detail) && detailCaseType(detail) === "latent" && hasPathEdges(detail);
}

function caseFilterEnabled(detail) {
  return state.caseFilters[detailCaseFilterBucket(detail)] !== false;
}

function activeCaseFilterLabels() {
  return Object.entries(state.caseFilters)
    .filter(([, enabled]) => enabled)
    .map(([name]) => name)
    .join(", ") || "none";
}

function allCaseFiltersEnabled() {
  return Object.values(state.caseFilters).every(Boolean);
}

function activeCaseFilterKey() {
  return CASE_FILTER_BUCKETS.filter((name) => state.caseFilters[name] !== false).join(",");
}

function combinedReverseMarker(item) {
  const order = numeric(item.order_score);
  return order === null ? null : order < 0;
}

function combinedMiracleMarker(item) {
  if (item.miracle_step === null || item.miracle_step === undefined) return null;
  return item.miracle_step === true;
}

function blockReverseMarker(item) {
  const blockOrder = numeric(item.block_order_score);
  return blockOrder === null ? null : blockOrder < 0;
}

function blockMiracleMarker(item) {
  if (item.block_miracle_step === null || item.block_miracle_step === undefined) return null;
  return item.block_miracle_step === true;
}

function activeDetails(snapshot) {
  return caseFilteredDetails(snapshot);
}

function resetLoadedCellDetails() {
  state.detailLoadBusyKeys.clear();
  state.detailLoadedCellKeys.clear();
  state.detailLoadErrors = {};
  state.detailLoadProgress = {};
}

function resetLoadedMetrics() {
  state.metricLoadedDatasets.clear();
  state.metricLoadBusyDatasets.clear();
  state.metricLoadErrors = {};
  state.metricLoadNotices = {};
}

function deferTask(fn) {
  if (typeof window !== "undefined" && typeof window.setTimeout === "function") {
    window.setTimeout(fn, 0);
  } else if (typeof setTimeout === "function") {
    setTimeout(fn, 0);
  } else {
    fn();
  }
}

function caseFilteredDetails(snapshot) {
  const details = snapshot?.details || [];
  return details.filter(caseFilterEnabled);
}

function pathNodePrecision(detail) {
  const path = pathNodes(detail);
  const context = graphContextNodes(detail);
  const hitPath = path.filter((node) => node?.hit).length;
  const hitContext = context.filter((node) => node?.hit).length;
  const denom = hitPath + hitContext;
  return denom ? hitPath / denom : null;
}

function pathNodeF1(detail) {
  return f1(pathNodePrecision(detail), pathValue(detail, "path_node_recall", "chain_node_recall"));
}

function metricsFromDetails(details, snapshot) {
  const cellLookup = new Map(experimentRows(snapshot).map((row) => [cellKey(row), row]));
  const groups = new Map();
  details.forEach((detail) => {
    const key = detailCellKey(detail);
    if (!key) return;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(detail);
  });
  return [...groups.entries()].map(([key, items]) => {
    const cell = cellLookup.get(key) || {};
    const first = items[0] || {};
    const bonusItems = items.filter(isPathMetricDetail);
    const orderMetricItems = items.filter(isOrderMetricDetail);
    const orderItems = orderMetricItems.filter((item) => item.order_defined === true);
    const blockOrderItems = orderMetricItems.filter((item) => item.block_order_defined === true);
    const scoredBlocks = sum(bonusItems.map((item) => item.n_scored_read_blocks));
    const totalBlocks = sum(bonusItems.map((item) => item.n_blocks));
    const cacheHit = sum(items.map((item) => item.cache_hit_tokens));
    const cacheWrite = sum(items.map((item) => item.cache_write_tokens));
    const inputTokens = sum(items.map((item) => item.input_tokens));
    return {
      eval_cell_key: key,
      experiment_key: key,
      source_kind: first.source_kind || cell.source_kind || "offline_artifact",
      experiment_id: first.experiment_id || cell.experiment_id || "adhoc",
      provider_source: first.provider_source || cell.provider_source || "unknown-provider",
      dataset: first.dataset || first.data_source || cell.dataset || "unknown-dataset",
      model_api_name: first.model_api_name || cell.model_api_name || first.model_label || cell.model_label || "unknown-model",
      model_label: first.model_label || cell.model_label || first.model_api_name || cell.model_api_name || "unknown-model",
      target: items.length,
      done: items.length,
      errors: items.filter((item) => item.error || item.system_error).length,
      pending: 0,
      resolved_rate: rate(items.map((item) => item.resolved)),
      reward_rate: avg(items.map((item) => item.reward)),
      avg_read_precision: avg(bonusItems.map((item) => item.hit_precision)),
      avg_node_recall: avg(bonusItems.map((item) => item.hit_recall)),
      avg_hit_f1: avg(bonusItems.map((item) => item.hit_f1)),
      anchor_hit_rate: rate(bonusItems.map((item) => item.anchor_hit)),
      root_hit_rate: rate(bonusItems.map((item) => item.root_hit)),
      avg_path_node_recall: avg(bonusItems.map((item) => pathValue(item, "path_node_recall", "chain_node_recall"))),
      avg_path_node_precision: avg(bonusItems.map(pathNodePrecision)),
      avg_path_node_f1: avg(bonusItems.map(pathNodeF1)),
      avg_path_read_precision: avg(bonusItems.map((item) => pathValue(item, "path_read_precision", "chain_read_precision"))),
      avg_chain_node_recall: avg(bonusItems.map((item) => pathValue(item, "path_node_recall", "chain_node_recall"))),
      avg_chain_node_precision: avg(bonusItems.map(pathNodePrecision)),
      avg_chain_node_f1: avg(bonusItems.map(pathNodeF1)),
      avg_chain_read_precision: avg(bonusItems.map((item) => pathValue(item, "path_read_precision", "chain_read_precision"))),
      avg_first_anchor_step: avg(bonusItems.map((item) => item.first_anchor_step)),
      avg_first_root_step: avg(bonusItems.map((item) => item.first_root_step)),
      avg_order_score: avg(orderItems.map((item) => item.order_score)),
      reverse_order_rate: rate(orderMetricItems.map(combinedReverseMarker)),
      miracle_rate: rate(orderMetricItems.map(combinedMiracleMarker)),
      avg_blocks_per_trace: totalBlocks && bonusItems.length ? totalBlocks / bonusItems.length : null,
      block_achieve_rate: scoredBlocks ? sum(bonusItems.map((item) => item.n_achieving_blocks)) / scoredBlocks : null,
      block_waste_rate: scoredBlocks ? sum(bonusItems.map((item) => item.n_wasted_blocks)) / scoredBlocks : null,
      block_loop_rate: totalBlocks ? sum(bonusItems.map((item) => item.n_loop_blocks)) / totalBlocks : null,
      block_reverse_order_rate: rate(orderMetricItems.map(blockReverseMarker)),
      block_miracle_rate: rate(orderMetricItems.map(blockMiracleMarker)),
      loop_trace_rate: rate(items.map((item) => (item.bad_patterns || {}).has_loop)),
      error_spiral_rate: rate(items.map((item) => (item.bad_patterns || {}).error_spiral)),
      avg_turns: avg(items.map((item) => item.turns)),
      avg_tool_calls: avg(items.map((item) => item.tool_calls)),
      avg_wall_time: avg(items.map((item) => item.wall_time)),
      avg_input_tokens: avg(items.map((item) => item.input_tokens)),
      avg_output_tokens: avg(items.map((item) => item.output_tokens)),
      avg_reasoning_tokens: avg(items.map((item) => item.reasoning_tokens)),
      cache_hit_rate: cacheHit && inputTokens + cacheHit ? cacheHit / (inputTokens + cacheHit) : null,
      cache_write_rate: cacheWrite && inputTokens + cacheWrite ? cacheWrite / (inputTokens + cacheWrite) : null,
    };
  }).sort((a, b) => String(a.model_label).localeCompare(String(b.model_label)));
}

function mergeMissingMetricFields(rows, fallbackRows) {
  if (!fallbackRows.length) return rows;
  const fallbackByKey = new Map(fallbackRows.map((row) => [cellKey(row), row]));
  return rows.map((row) => {
    const fallback = fallbackByKey.get(cellKey(row));
    if (!fallback) return row;
    const merged = { ...row };
    Object.entries(fallback).forEach(([key, value]) => {
      if ((merged[key] === null || merged[key] === undefined) && value !== null && value !== undefined) {
        merged[key] = value;
      }
    });
    return merged;
  });
}

function snapshotModelMetricRows(snapshot) {
  const rows = snapshot?.model_metrics || [];
  if (rows.length) return rows;
  return experimentRows(snapshot).map((row) => ({
    ...row,
    target: numeric(row.target) ?? 0,
    target_rollouts: runTotalCount(row),
    done: numeric(row.done) ?? runDoneCount(row),
    done_rollouts: runDoneCount(row),
    errors: runErrorCount(row),
    pending: runPendingCount(row),
    detail_cache_ready_rollouts: row.cache_ready,
    detail_cache_pending_rollouts: row.cache_pending,
  }));
}

function activeModelMetrics(snapshot) {
  const details = activeDetails(snapshot);
  const fallbackRows = details.length ? metricsFromDetails(details, snapshot) : [];
  if (state.caseFilters.direct && state.caseFilters.latent && state.caseFilters.exposed && !state.caseFilters.others) {
    const rows = snapshot?.path_metric_model_metrics || snapshot?.dynamic_traceable_model_metrics || [];
    if (rows.length) return mergeMissingMetricFields(rows, fallbackRows);
  }
  if (allCaseFiltersEnabled()) return mergeMissingMetricFields(snapshotModelMetricRows(snapshot), fallbackRows);
  const caseFilterRows = snapshot?.case_filter_model_metrics?.[activeCaseFilterKey()] || [];
  if (caseFilterRows.length) return mergeMissingMetricFields(caseFilterRows, fallbackRows);
  return fallbackRows;
}

function selectedDatasetRow(snapshot) {
  return overviewDatasetRows(snapshot).find((row) => row.dataset === state.selectedDataset) || null;
}

function selectedExperiment(snapshot) {
  return experimentRows(snapshot).find((row) => cellKey(row) === state.selectedEvalCellKey) || null;
}

function ensureSelection(snapshot) {
  const datasets = datasetRows(snapshot);
  if (!datasets.length) {
    state.selectedDataset = null;
    state.selectedEvalCellKey = null;
    state.selectedExperimentKey = null;
    state.selectedTraceKey = null;
    state.selectedGraphNodeKey = null;
    resetTracePanels();
    return;
  }
  if (!state.selectedDataset || !datasets.some((row) => row.dataset === state.selectedDataset)) {
    if (state.permalinkMissing) {
      state.selectedDataset = null;
      state.selectedEvalCellKey = null;
      state.selectedExperimentKey = null;
      state.selectedTraceKey = null;
      state.selectedStepIndex = 0;
      state.selectedGraphNodeKey = null;
      resetTracePanels();
      return;
    }
    state.selectedDataset = datasets.length === 1 ? datasets[0].dataset : null;
  }
  const cells = experimentRows(snapshot).filter((row) => !state.selectedDataset || row.dataset === state.selectedDataset);
  if (!state.selectedEvalCellKey && state.selectedExperimentKey) {
    state.selectedEvalCellKey = state.selectedExperimentKey;
  }
  if (!state.selectedEvalCellKey || !cells.some((row) => cellKey(row) === state.selectedEvalCellKey)) {
    state.selectedEvalCellKey = cells.length === 1 ? cellKey(cells[0]) : null;
  }
  state.selectedExperimentKey = state.selectedEvalCellKey;
  if (!state.selectedEvalCellKey) {
    state.selectedTraceKey = null;
    state.selectedStepIndex = 0;
    state.selectedGraphNodeKey = null;
    resetTracePanels();
    return;
  }
  const details = filteredDetails(snapshot);
  if (!details.length) {
    state.selectedTraceKey = null;
    state.selectedStepIndex = 0;
    state.selectedGraphNodeKey = null;
    resetTracePanels();
    return;
  }
  clampTraceGlobalRolloutIndex(details);
  if (!state.selectedTraceKey || !details.some((detail) => rowKey(detail) === state.selectedTraceKey)) {
    const preferred = details.find((detail) => detail.raw_available || (detail.step_details || []).length || (detail.step_inspection || []).length) || details[0];
    const selected = detailForRolloutIndex(detailsForInstance(details, traceInstanceKey(preferred)));
    state.selectedTraceKey = rowKey(selected || preferred);
    state.selectedStepIndex = 0;
    state.selectedGraphNodeKey = null;
    resetTracePanels();
  }
}

async function loadSnapshot(options = {}) {
  if (state.loadingSnapshot) {
    if (options.queueIfBusy) state.queuedSnapshotOptions = options;
    return;
  }
  const showOperation = options.silent !== true;
  const scrollState = captureInspectorScroll();
  const dropCellKeys = options.dropSelectedDetails && state.selectedEvalCellKey ? [state.selectedEvalCellKey] : [];
  let shouldRender = false;
  if (window.__P2A_DASHBOARD_SNAPSHOT__) {
    state.snapshot = window.__P2A_DASHBOARD_SNAPSHOT__;
    window.__P2A_DASHBOARD_SNAPSHOT__ = null;
    render({ scrollState });
    return;
  }
  state.loadingSnapshot = true;
  state.snapshotBusy = options.busy || "refresh";
  if (showOperation) {
    state.operationTone = "";
    state.operationMessage = options.startMessage || "Refreshing snapshot...";
    syncSnapshotControls();
    renderOperationStatus();
  }
  try {
    const response = await fetch("/api/snapshot", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const previousSnapshot = state.snapshot;
    state.snapshot = await response.json();
    preserveLoadedDetails(state.snapshot, previousSnapshot, { dropCellKeys });
    state.detailLoadErrors = {};
    resetLoadedMetrics();
    if (showOperation) {
      state.operationTone = "ok";
      state.operationMessage = options.successMessage || "Refresh finished.";
    }
    shouldRender = true;
  } catch (error) {
    if (showOperation) {
      state.operationTone = "bad";
      state.operationMessage = options.errorMessage || `Snapshot refresh failed: ${error.message || error}`;
    }
    if (!state.snapshot) {
      try {
        const response = await fetch("snapshot.json", { cache: "no-store" });
        if (response.ok) {
          const previousSnapshot = state.snapshot;
          state.snapshot = await response.json();
          preserveLoadedDetails(state.snapshot, previousSnapshot, { dropCellKeys });
          state.detailLoadErrors = {};
          resetLoadedMetrics();
          shouldRender = true;
        }
      } catch (_fallback) {
        state.snapshot = null;
        shouldRender = true;
      }
    }
  } finally {
    state.loadingSnapshot = false;
    state.snapshotBusy = "";
    if (showOperation) {
      syncSnapshotControls();
      renderOperationStatus();
    }
  }
  if (!state.snapshot) shouldRender = true;
  if (shouldRender) render({ scrollState });
  if (state.queuedSnapshotOptions) {
    const queuedOptions = state.queuedSnapshotOptions;
    state.queuedSnapshotOptions = null;
    window.setTimeout(() => loadSnapshot(queuedOptions), 0);
  }
}

function renderSources(snapshot) {
  const sources = snapshot?.sources || [];
  const text = sources.map((item) => `${item.kind}: ${item.path}`).join("  |  ");
  const status = snapshot?.snapshot_status?.stale ? `  |  stale: ${snapshot.snapshot_status.reason || "snapshot unavailable"}` : "";
  document.getElementById("source-line").textContent = (text || "No source loaded") + status;
}

function renderOperationStatus() {
  const el = document.getElementById("operation-status");
  if (!el) return;
  const isRebuildMessage = String(state.operationMessage || "").startsWith("Rebuild");
  const message = isRebuildMessage ? "" : state.operationMessage || "";
  el.hidden = !message;
  el.textContent = message;
  el.className = `operation-status ${state.operationTone || ""}`.trim();
}

function syncSnapshotControls() {
  const refresh = document.getElementById("refresh-button");
  if (refresh) {
    refresh.disabled = state.loadingSnapshot;
    refresh.textContent = state.loadingSnapshot ? "Refreshing..." : "Refresh";
  }
}

function rebuildStatusMessage() {
  const status = state.rebuildStatus;
  if (!status) return "";
  const counts = status.last_counts || {};
  const cells = counts.run_cells === undefined ? "" : ` (${counts.run_cells} run cells)`;
  if (status.active) {
    const phaseLabels = {
      queued: "Queued",
      waiting: "Waiting",
      clearing: "Clearing cache",
      warming: "Rebuilding",
    };
    const phase = phaseLabels[status.phase] || "Running";
    return `${phase}${cells}.`;
  }
  if (status.phase === "failed") return `Last rebuild failed: ${status.last_error || "unknown error"}`;
  return "";
}

function metricCard(label, value, formatter = fmt) {
  return `<section class="metric"><div class="label">${esc(label)}</div><div class="value">${esc(formatter(value))}</div></section>`;
}

function renderSelectedExperiment(snapshot) {
  const selected = selectedExperiment(snapshot);
  const dataset = state.selectedDataset || "None";
  const cell = selected
    ? `${selected.source_kind || "-"} / ${selected.experiment_id || "-"} / ${selected.model_label || "-"}`
    : "None";
  const label = `Dataset: ${dataset} | Eval cell: ${cell}`;
  const filters = [];
  if (selected?.selected_scope) filters.push(`run selection: ${scopeSummary(selected)}`);
  if (!allCaseFiltersEnabled()) filters.push(`case types: ${activeCaseFilterLabels()}`);
  const patternTags = activeTracePatternFilters();
  if (patternTags.length) filters.push(`patterns: ${patternTags.join("+")}`);
  document.getElementById("selected-experiment").textContent = label + (filters.length ? ` | Filter: ${filters.join("; ")}` : "");
  const notice = document.getElementById("permalink-notice");
  if (notice) {
    notice.hidden = !state.permalinkNotice;
    notice.textContent = state.permalinkNotice || "";
  }
}

function renderSummary(snapshot) {
  const counts = snapshot?.summary?.counts || {};
  const selectedDataset = selectedDatasetRow(snapshot);
  const totals = overviewDatasetTotals(snapshot);
  const scopedInstances = selectedDataset ? selectedDataset.n_instances || 0 : totals.n_instances || 0;
  const scopedTraces = selectedDataset ? selectedDataset.n_trajectories || 0 : totals.n_trajectories || 0;
  const scopedCells = selectedDataset ? selectedDataset.n_eval_cells || 0 : totals.n_eval_cells || 0;
  const cards = [
    ["Datasets", overviewDatasetRows(snapshot).length, fmt],
    ["Eval cells", scopedCells, fmt],
    ["Instances", scopedInstances, fmt],
    ["Traces", scopedTraces, fmt],
    ["Models", snapshotModelMetricRows(snapshot).length, fmt],
    ["Runs", (snapshot.runs || []).length, fmt],
    ["Raw records", snapshot.raw_record_count ?? 0, fmt],
    ["Loaded details", activeDetails(snapshot).length || counts.n_records || 0, fmt],
  ];
  document.getElementById("summary-grid").innerHTML = cards.map(([label, value, formatter]) => metricCard(label, value, formatter)).join("");
}

function runDoneCount(row) {
  const explicit = numeric(row.done_rollouts);
  if (explicit !== null) return explicit;
  const total = runTotalCount(row);
  const failures = runErrorCount(row);
  const pending = runPendingCount(row);
  if ((failures || pending) && total > 0) return Math.max(0, total - failures - pending);
  return numeric(row.done) ?? 0;
}

function runTotalCount(row) {
  return numeric(row.target_rollouts) ?? numeric(row.target) ?? numeric(row.detail_count) ?? 0;
}

function rolloutN(row) {
  const explicit = numeric(row.rollouts_per_instance);
  if (explicit !== null && explicit > 0) return explicit;
  const instances = numeric(row.target);
  const traces = numeric(row.target_rollouts) ?? numeric(row.detail_count);
  if (instances && traces !== null) return Math.max(1, Math.round(traces / instances));
  return null;
}

function runErrorCount(row) {
  return numeric(row.errors) ?? 0;
}

function runPendingCount(row) {
  return numeric(row.pending) ?? 0;
}

function cacheStatus(row) {
  const total = runDoneCount(row);
  const errors = runErrorCount(row);
  const hasReady = Object.prototype.hasOwnProperty.call(row, "cache_ready");
  const hasPending = Object.prototype.hasOwnProperty.call(row, "cache_pending");
  const ready = hasReady ? numeric(row.cache_ready) : null;
  const pending = hasPending ? numeric(row.cache_pending) : null;
  if (pending !== null && pending > 0) {
    if (ready !== null && total > 0) return `${ready}/${total} cached`;
    return `${pending} to rebuild`;
  }
  if (ready !== null && total > 0 && ready < total) return `${ready}/${total} cached`;
  if (ready !== null && total > 0) return `${ready}/${total} cached`;
  if (ready === null && pending === null && total > 0) return "unknown";
  if (total <= 0 && errors > 0) return "no completed traces";
  return "ready";
}

function rebuildQueuedMessage(result) {
  const jobs = numeric(result?.queued_jobs);
  if (jobs !== null && jobs > 1) return `Rebuild queued for ${jobs} targets.`;
  const cells = numeric(result?.counts?.run_cells);
  return cells === null ? "Rebuild queued." : `Rebuild queued for ${cells} run cells.`;
}

function applyQueuedRebuildStatus(result) {
  state.rebuildStatus = result?.rebuild_status || { active: true, phase: "queued", queued: 1, running: 0 };
  syncAdminRebuildTargetsFromStatus();
  scheduleRebuildStatusPoll();
}

function scopeSummary(row) {
  const scope = row?.selected_scope;
  if (!scope || typeof scope !== "object") return "-";
  const filter = scope.filter || {};
  const caseTypes = (filter.case_types || []).join("+") || "all";
  const sourceSize = scope.source_size ?? "-";
  const selectedBeforeWindow = scope.selected_size_before_window ?? scope.selected_size ?? "-";
  const selectedSize = scope.selected_size ?? row?.target ?? "-";
  const pattern = filter.pattern_computable === true ? " pattern" : "";
  return `${caseTypes}${pattern} · selected ${selectedBeforeWindow}/${sourceSize} · planned ${selectedSize}`;
}

function metricRowForCell(rows, row) {
  const key = cellKey(row);
  return (rows || []).find((item) => cellKey(item) === key) || null;
}

function evalCellFilterStats(snapshot, row) {
  const filterKey = activeCaseFilterKey();
  if (!filterKey) return { instances: 0, traces: 0 };
  const metricRows = allCaseFiltersEnabled()
    ? snapshotModelMetricRows(snapshot)
    : snapshot?.case_filter_model_metrics?.[filterKey] || [];
  const metric = metricRowForCell(metricRows, row);
  if (metric) {
    const instances = numeric(metric.target);
    const traces = numeric(metric.target_rollouts) ?? numeric(metric.target);
    return { instances: instances ?? 0, traces: traces ?? 0 };
  }
  if (allCaseFiltersEnabled()) {
    return { instances: numeric(row.target) ?? 0, traces: runTotalCount(row) };
  }
  const key = cellKey(row);
  const details = caseFilteredDetails(snapshot).filter((detail) => detailCellKey(detail) === key);
  const instances = new Set(details.map(traceInstanceId).filter(Boolean));
  return { instances: instances.size, traces: details.length };
}

function renderExperiments(snapshot) {
  const datasetRowsHtml = overviewDatasetRows(snapshot).map((row) => {
    const selected = row.dataset === state.selectedDataset;
    return `<tr class="clickable ${selected ? "is-selected" : ""}" data-dataset="${esc(row.dataset)}">
      <td><button class="select-dataset" type="button" data-dataset="${esc(row.dataset)}">${selected ? "Selected" : "Select"}</button></td>
      <td>${esc(row.dataset)}</td>
      <td>${esc(row.n_instances ?? "-")}</td>
      <td>${esc(row.n_eval_cells ?? "-")}</td>
      <td>${esc(row.n_trajectories ?? "-")}</td>
      <td>${esc((row.models || []).join(", ") || "-")}</td>
      <td>${esc((row.source_kinds || []).join(", ") || "-")}</td>
    </tr>`;
  });
  const cellRows = experimentRows(snapshot).filter((row) => !state.selectedDataset || row.dataset === state.selectedDataset);
  const rows = cellRows.map((row) => {
    const key = cellKey(row);
    const selected = key === state.selectedEvalCellKey;
    const deleteTarget = { experiment_id: row.experiment_id, provider_source: row.provider_source, dataset: row.dataset };
    const deleteKey = deleteTargetKey(deleteTarget);
    const rebuildTarget = {
      experiment_id: row.experiment_id,
      provider_source: row.provider_source,
      dataset: row.dataset,
      model_api_name: row.model_api_name,
      model_label: row.model_label,
    };
    const rebuildKey = deleteTargetKey(rebuildTarget);
    const cachePending = Number(row.cache_pending || 0) > 0;
    const adminCells = state.admin.authenticated
      ? `<td><input class="admin-delete-target" type="checkbox" data-delete-target="${esc(deleteKey)}" ${state.adminDeleteKeys.has(deleteKey) ? "checked" : ""}></td>
         <td><input class="admin-rebuild-select" type="checkbox" data-rebuild-target="${esc(rebuildKey)}" ${state.adminRebuildKeys.has(rebuildKey) ? "checked" : ""}></td>`
      : "";
    const filterStats = evalCellFilterStats(snapshot, row);
    return `<tr class="clickable ${selected ? "is-selected" : ""} ${cachePending ? "has-cache-pending" : ""}" data-eval-cell-key="${esc(key)}">
      ${adminCells}
      <td><button class="select-cell" type="button" data-eval-cell-key="${esc(key)}">${selected ? "Selected" : "Inspect"}</button></td>
      <td>${esc(row.source_kind)}</td>
      <td>${esc(row.experiment_id)}</td>
      <td>${esc(row.provider_source)}</td>
      <td>${esc(row.dataset)}</td>
      <td>${esc(row.model_label)}</td>
      <td>${esc(rolloutN(row) ?? "-")}</td>
      <td>${esc(filterStats.instances)}</td>
      <td>${esc(row.target ?? "-")}</td>
      <td>${esc(runDoneCount(row))}</td>
      <td>${runErrorCount(row) ? badge(String(runErrorCount(row)), true, "bad") : "0"}</td>
      <td>${runPendingCount(row) ? badge(String(runPendingCount(row)), true, "warn") : "0"}</td>
      <td class="cache-cell">${esc(cacheStatus(row))}</td>
    </tr>`;
  });
  document.getElementById("experiment-table").innerHTML = `
    <section class="subsection"><h3>Datasets</h3>${table(["", "Dataset", "Filtered instances", "Eval cells", "Filtered traces", "Models", "Sources"], datasetRowsHtml)}</section>
    <section class="subsection"><h3>Eval cells${state.selectedDataset ? ` in ${esc(state.selectedDataset)}` : ""}</h3>${table([...(state.admin.authenticated ? ["Delete", "Rebuild"] : []), "", "Kind", "Experiment", "Provider", "Dataset", "Model", "Rollout N", "Filtered instances", "Total instances", "Done traces", "Error traces", "ToDo traces", "Detail cache"], rows)}</section>`;
  document.querySelectorAll(".select-dataset, #experiment-table tr[data-dataset]").forEach((el) => {
    el.addEventListener("click", () => {
      const dataset = el.dataset.dataset;
      if (!dataset) return;
      state.permalinkMissing = false;
      state.permalinkNotice = "";
      state.selectedDataset = dataset;
      state.selectedEvalCellKey = null;
      state.selectedExperimentKey = null;
      state.selectedTraceKey = null;
      state.selectedStepIndex = 0;
      state.selectedGraphNodeKey = null;
      resetTracePanels();
      render();
    });
  });
  document.querySelectorAll(".select-cell, #experiment-table tr[data-eval-cell-key]").forEach((el) => {
    el.addEventListener("click", () => {
      if (el.classList.contains("admin-delete-target")) return;
      const key = el.dataset.evalCellKey;
      if (!key) return;
      state.permalinkMissing = false;
      state.permalinkNotice = "";
      state.selectedEvalCellKey = key;
      state.selectedExperimentKey = key;
      const cell = experimentRows(state.snapshot).find((row) => cellKey(row) === key);
      if (cell?.dataset) state.selectedDataset = cell.dataset;
      state.selectedTraceKey = null;
      state.selectedStepIndex = 0;
      state.selectedGraphNodeKey = null;
      resetTracePanels();
      setTab(el.classList.contains("select-cell") ? "traces" : state.activeTab);
      render();
    });
  });
  document.querySelectorAll(".admin-delete-target").forEach((input) => {
    input.addEventListener("click", (event) => event.stopPropagation());
    input.addEventListener("change", (event) => {
      const key = event.target.dataset.deleteTarget;
      if (!key) return;
      if (event.target.checked) state.adminDeleteKeys.add(key);
      else state.adminDeleteKeys.delete(key);
      state.adminPreview = null;
      renderAdminPanel(snapshot);
    });
  });
  document.querySelectorAll(".admin-rebuild-select").forEach((input) => {
    input.addEventListener("click", (event) => event.stopPropagation());
    input.addEventListener("change", (event) => {
      const key = event.target.dataset.rebuildTarget;
      if (!key) return;
      if (event.target.checked) state.adminRebuildKeys.add(key);
      else state.adminRebuildKeys.delete(key);
      state.adminPreview = null;
      renderAdminPanel(snapshot);
    });
  });
}

function scopedRows(rows) {
  if (!state.selectedEvalCellKey) return [];
  return (rows || []).filter((row) => cellKey(row) === state.selectedEvalCellKey);
}

function renderTrend(snapshot) {
  const dataset = state.selectedDataset;
  const trends = snapshot?.summary?.trends || [];
  const rows = trends
    .filter((row) => !dataset || row.data_source === dataset)
    .map((row) => {
      const rates = row.rates || {};
      return `<tr>
        <td>${esc(row.data_source)}</td>
        <td>${esc(row.run_step)}</td>
        <td>${esc(row.n_records)}</td>
        <td>${esc(pct(rates.bonus_map_coverage))}</td>
        <td>${esc(pct(rates.call_graph_coverage))}</td>
        <td>${esc(pct(rates.read_rate))}</td>
        <td>${esc(pct(rates.path_coverage ?? rates.chain_graph_coverage))}</td>
      </tr>`;
    });
  document.getElementById("trend-table").innerHTML = table(
    ["Data source", "Step", "N", "Bonus maps", "Graphs", "Read rate", "Path coverage"],
    rows
  );
}

function miniTable(title, mapping) {
  const rows = Object.entries(mapping || {}).map(([key, value]) => {
    const shown = value && typeof value === "object" ? (value.n ?? value.value ?? JSON.stringify(value)) : value;
    return `<tr><td>${esc(key)}</td><td>${esc(fmt(shown))}</td></tr>`;
  });
  return `<section class="mini-panel"><h3>${esc(title)}</h3>${table(["Key", "Value"], rows)}</section>`;
}

function renderDistributions(snapshot) {
  if (!state.selectedDataset) {
    document.getElementById("distribution-grid").innerHTML = '<div class="empty">Select a dataset before inspecting distributions.</div>';
    return;
  }
  const payload = snapshot?.summary?.distributions_by_dataset?.[state.selectedDataset] || snapshot?.summary?.by_dataset?.[state.selectedDataset] || {};
  const dist = payload.distributions || {};
  const population = payload.n_instances || payload.counts?.n_instances || 0;
  document.getElementById("distribution-grid").innerHTML = [
    miniTable(`Dataset population (${state.selectedDataset})`, { unique_instances: population }),
    miniTable("Case types", dist.case_types),
    miniTable("Not Path-evaluable", dist.not_path_evaluable_reasons || dist.not_chain_evaluable_reasons),
    miniTable("Graph availability", dist.availability),
  ].join("");
}

function kpiColumns(hasCacheWrite) {
  const columns = [
    {
      header: "",
      fixed: true,
      html: true,
      value: (_row, key) => `<button class="select-kpi-cell" type="button" data-eval-cell-key="${esc(key)}">${key === state.selectedEvalCellKey ? "Selected" : "Select"}</button>`,
    },
    { header: "Experiment", fixed: true, value: (row) => row.experiment_id },
    { header: "Kind", fixed: true, value: (row) => row.source_kind },
    { header: "Model", fixed: true, value: (row) => row.model_label },
    { header: "Total instances", group: "filter_totals", value: (row) => numeric(row.target) ?? 0 },
    { header: "Done traces", group: "filter_totals", value: (row) => runDoneCount(row) },
    { header: "Error traces", group: "filter_totals", value: (row) => runErrorCount(row) },
    { header: "ToDo traces", group: "filter_totals", value: (row) => runPendingCount(row) },
    { header: "Graph P.", group: "graph", value: (row) => withStd(row, "avg_read_precision", pct) },
    { header: "Graph R.", group: "graph", value: (row) => withStd(row, "avg_node_recall", pct) },
    { header: "Graph F1", group: "graph", value: (row) => withStd(row, "avg_hit_f1", pct) },
    { header: "Pass@K", group: "outcome", value: (row) => pct(passAtValue(row)) },
    { header: "Avg@K", group: "outcome", value: (row) => withStd(row, row.resolved_rate === null || row.resolved_rate === undefined ? "reward_rate" : "resolved_rate", pct) },
    { header: "Symptom hit", group: "outcome", value: (row) => withStd(row, "anchor_hit_rate", pct) },
    { header: "Root cause hit", group: "outcome", value: (row) => withStd(row, "root_hit_rate", pct) },
    { header: "First symptom", group: "outcome", value: (row) => withStd(row, "avg_first_anchor_step", fmt, 1) },
    { header: "First root cause", group: "outcome", value: (row) => withStd(row, "avg_first_root_step", fmt, 1) },
    { header: "Path P.", group: "path", value: (row) => withStd(row, row.avg_path_node_precision === null || row.avg_path_node_precision === undefined ? "avg_chain_node_precision" : "avg_path_node_precision", pct) },
    { header: "Path R.", group: "path", value: (row) => withStd(row, row.avg_path_node_recall === null || row.avg_path_node_recall === undefined ? "avg_chain_node_recall" : "avg_path_node_recall", pct) },
    { header: "Path F1", group: "path", value: (row) => pathF1Value(row) },
    { header: "Order score", group: "exploration_behavior", value: (row) => withStd(row, "avg_order_score") },
    { header: "Reverse rate", group: "exploration_behavior", value: (row) => withStd(row, "reverse_order_rate", pct) },
    { header: "Miracle rate", group: "exploration_behavior", value: (row) => withStd(row, "miracle_rate", pct) },
    { header: "Loop trace", group: "exploration_behavior", value: (row) => withStd(row, "loop_trace_rate", pct) },
    { header: "Error spiral", group: "exploration_behavior", value: (row) => withStd(row, "error_spiral_rate", pct) },
    { header: "Blocks", group: "purpose_blocks", value: (row) => withStd(row, "avg_blocks_per_trace", fmt, 1) },
    { header: "Achieved", group: "purpose_blocks", value: (row) => withStd(row, "block_achieve_rate", pct) },
    { header: "Wasted", group: "purpose_blocks", value: (row) => withStd(row, "block_waste_rate", pct) },
    { header: "Loop blocks", group: "purpose_blocks", value: (row) => withStd(row, "block_loop_rate", pct) },
    { header: "Turns", group: "efficiency_cost", value: (row) => withStd(row, "avg_turns", fmt, 1) },
    { header: "Tools", group: "efficiency_cost", value: (row) => withStd(row, "avg_tool_calls", fmt, 1) },
    { header: "Wall", group: "efficiency_cost", value: (row) => withStd(row, "avg_wall_time", fmt, 1) },
    { header: "In", group: "efficiency_cost", value: (row) => withStd(row, "avg_input_tokens", token) },
    { header: "Out", group: "efficiency_cost", value: (row) => withStd(row, "avg_output_tokens", token) },
    { header: "Reason", group: "efficiency_cost", value: (row) => withStd(row, "avg_reasoning_tokens", token) },
    { header: "Cache hit", group: "efficiency_cost", value: (row) => withStd(row, "cache_hit_rate", pct) },
  ];
  if (hasCacheWrite) {
    columns.push({ header: "Cache write", group: "efficiency_cost", value: (row) => withStd(row, "cache_write_rate", pct) });
  }
  return columns.filter((column) => column.fixed || metricGroupEnabled(column.group));
}

function maxRolloutN(rows) {
  return Math.max(1, ...rows.map((row) => rolloutN(row) || 1).filter(Number.isFinite));
}

function renderPassAtControl(rows) {
  const maxN = maxRolloutN(rows);
  if (!state.passAtK || state.passAtK > maxN) state.passAtK = 1;
  const options = Array.from({ length: maxN }, (_item, index) => index + 1)
    .map((k) => `<option value="${k}" ${Number(state.passAtK) === k ? "selected" : ""}>${k}</option>`)
    .join("");
  return `<label class="pass-at-control">Pass/Avg K <select id="pass-at-k">${options}</select></label>`;
}

function renderModels(snapshot) {
  if (!state.selectedDataset) {
    document.getElementById("model-table").innerHTML = '<div class="empty">Select a dataset in Overview before comparing Metrics.</div>';
    return;
  }
  const metricLoading = state.metricLoadBusyDatasets.has(state.selectedDataset);
  const metricError = state.metricLoadErrors[state.selectedDataset];
  const metricNotice = state.metricLoadNotices[state.selectedDataset];
  if (!state.metricLoadedDatasets.has(state.selectedDataset) && !metricLoading && !metricError) {
    deferTask(() => loadDatasetMetrics(state.selectedDataset));
  }
  const rows = activeModelMetrics(snapshot).filter((row) => row.dataset === state.selectedDataset);
  const hasCacheWrite = rows.some((row) => row.cache_write_rate !== null && row.cache_write_rate !== undefined);
  const columns = kpiColumns(hasCacheWrite);
  const scopeBits = [];
  scopeBits.push(`case types: ${activeCaseFilterLabels()}`);
  const scopeNote = `Metrics and Traces both use the current global filters (${scopeBits.join("; ")}).`;
  const loadNote = metricLoading
    ? '<div class="inline-status">Loading cached metrics...</div>'
    : (metricError
      ? `<div class="inline-status is-error">Cached metrics failed to load: ${esc(metricError)}</div>`
      : (metricNotice ? `<div class="inline-status">${esc(metricNotice)}</div>` : ""));
  document.getElementById("model-table").innerHTML = `
    <div class="panel-note">Metrics are scoped to dataset <strong>${esc(state.selectedDataset)}</strong>. ${esc(scopeNote)} Graph metrics score reads against the captured dependency Graph; Path metrics score the issue symptom-to-root-cause Path; Trace metrics describe the agent trajectory.</div>
    ${loadNote}
    ${renderPassAtControl(rows)}
    ${renderMetricGroupControls()}
    ${metricDefinitions("Metric definitions", MACRO_METRIC_GROUPS)}
    <div class="table-wrap kpi-table" data-scroll-key="model-kpi-table">${renderKpiTable(columns, rows)}</div>`;
  document.querySelectorAll(".metric-group-checkbox").forEach((input) => {
    input.addEventListener("change", (event) => {
      const group = event.target.dataset.metricGroup;
      if (!group) return;
      const tableScrollState = captureTableScroll();
      state.metricGroupFilters[group] = Boolean(event.target.checked);
      renderModels(state.snapshot);
      restoreTableScroll(tableScrollState);
    });
  });
  document.getElementById("pass-at-k")?.addEventListener("change", (event) => {
    const tableScrollState = captureTableScroll();
    state.passAtK = Number(event.target.value);
    renderModels(state.snapshot);
    restoreTableScroll(tableScrollState);
  });
  document.querySelectorAll(".select-kpi-cell, #model-table tr[data-eval-cell-key]").forEach((el) => {
    el.addEventListener("click", () => {
      const key = el.dataset.evalCellKey;
      if (!key) return;
      state.permalinkMissing = false;
      state.permalinkNotice = "";
      state.selectedEvalCellKey = key;
      state.selectedExperimentKey = key;
      const cell = experimentRows(state.snapshot).find((row) => cellKey(row) === key);
      if (cell?.dataset) state.selectedDataset = cell.dataset;
      state.selectedTraceKey = null;
      state.selectedStepIndex = 0;
      state.selectedGraphNodeKey = null;
      resetTracePanels();
      render();
    });
  });
}

function renderRuns(snapshot) {
  const selected = selectedExperiment(snapshot);
  if (!state.selectedDataset) {
    document.getElementById("run-list").innerHTML = '<div class="empty">Select a dataset before inspecting logs.</div>';
    return;
  }
  const allRuns = snapshot?.runs || [];
  const linked = allRuns.filter((run) => {
    const keys = run.eval_cell_keys || [];
    if (selected) return keys.includes(cellKey(selected));
    return (run.datasets || []).includes(state.selectedDataset);
  });
  const unlinked = allRuns.filter((run) => !(run.eval_cell_keys || []).length);
  const renderCards = (runs) => runs.map((run) => {
    const statusTone = run.status === "completed" ? "ok" : run.status === "running" || run.status === "verify" ? "warn" : "";
    const files = (run.files || []).slice(0, 8).map((name) => `<span class="badge">${esc(name)}</span>`).join("");
    const links = (run.eval_cell_keys || []).length
      ? `<div class="run-meta">Linked cells: ${esc((run.model_labels || []).join(", ") || run.eval_cell_keys.length)}</div>`
      : '<div class="run-meta">Unlinked logs: no eval-cell metadata in this artifact.</div>';
    const log = run.log_excerpt ? `<pre class="log">${esc(run.log_excerpt.slice(-6000))}</pre>` : '<div class="muted">No run.log tail.</div>';
    return `<article class="run-card">
      <div class="run-head"><div><div class="run-title">${esc(run.run_id)}</div><div class="run-meta">${esc(run.path)}</div></div>${badge(run.status || "unknown", true, statusTone)}</div>
      <div class="run-meta">Updated ${run.last_update ? new Date(run.last_update * 1000).toLocaleString() : "-"}</div>${links}
      <div>${files}</div>${log}
    </article>`;
  }).join("");
  document.getElementById("run-list").innerHTML = `
    <div class="panel-note">Logs map metrics and traces back to the artifact-producing execution: run id, artifact path, linked eval cell, files, and log tail. Use them to debug where a metric came from or to reproduce a run; day-to-day analysis belongs in Metrics and Traces.</div>
    <h3>${selected ? "Linked to selected eval cell" : `Linked to dataset ${esc(state.selectedDataset)}`}</h3>
    <div class="run-grid">${renderCards(linked) || '<div class="empty">No explicitly linked runs for this scope.</div>'}</div>
    <h3>Unlinked runs</h3>
    <div class="run-grid">${renderCards(unlinked) || '<div class="empty">No unlinked runs.</div>'}</div>`;
}

function traceBlob(detail) {
  return JSON.stringify({
    instance_id: detail.instance_id,
    files: detail.read_files,
    step_reads: (detail.step_inspection || []).map((step) => step.recovered_reads || step.target_path || step.path),
    bad: detail.bad_patterns,
    path_patterns: pathValue(detail, "path_pattern_flags", "chain_bad_patterns"),
    reason: pathValue(detail, "not_path_evaluable_reason", "not_chain_evaluable_reason"),
  }).toLowerCase();
}

function normalizeTracePatternFilterValue(value) {
  if (value === true || value === "true") return "true";
  if (value === "false") return "false";
  if (value === "none") return "none";
  return null;
}

function nextTracePatternFilterValue(value) {
  const current = normalizeTracePatternFilterValue(value);
  if (!current) return "true";
  if (current === "true") return "false";
  if (current === "false") return "none";
  return null;
}

function activeTracePatternFilterEntries() {
  return TRACE_PATTERN_FILTERS.map((tag) => ({
    tag,
    value: normalizeTracePatternFilterValue(state.tracePatternFilters?.[tag]),
  })).filter((item) => item.value);
}

function activeTracePatternFilters() {
  return activeTracePatternFilterEntries().map(({ tag, value }) => (value === "true" ? tag : `${tag}:${value}`));
}

function booleanPatternValue(value) {
  if (value === true) return true;
  if (value === false) return false;
  return null;
}

function pathProjectionAvailable(detail) {
  const projection = pathProjection(detail);
  return ["path_nodes", "chain_nodes", "context_nodes", "path_edges", "chain_edges"].some((key) => projection[key] !== undefined);
}

function pathNodeHasRoleHit(detail, roles) {
  const roleSet = new Set(roles);
  return [...pathNodes(detail), ...graphContextNodes(detail)].some((node) => {
    if (!node || node.hit !== true) return false;
    const role = canonicalNodeRole(node.node_role);
    if (roleSet.has(role)) return true;
    if (roles.includes("symptom") && node.selected_issue_anchor === true) return true;
    if (roles.includes("root_cause") && (node.root_cause === true || node.patched_callable === true)) return true;
    return false;
  });
}

function pathNodeRoleHitValue(detail, roles, directKey) {
  const direct = booleanPatternValue(detail?.[directKey]);
  if (direct !== null) return direct;
  if (!pathProjectionAvailable(detail)) return null;
  return pathNodeHasRoleHit(detail, roles);
}

function tracePatternValue(detail, tag) {
  if (tag === "miracle") {
    return combinedMiracleMarker(detail);
  }
  if (tag === "reverse") {
    return combinedReverseMarker(detail);
  }
  if (tag === "loop") {
    const badPattern = booleanPatternValue((detail.bad_patterns || {}).has_loop);
    if (badPattern !== null) return badPattern;
    if (Array.isArray(detail.purpose_blocks)) return detail.purpose_blocks.some((block) => block?.loop === true);
    return null;
  }
  if (tag === "hit_symptom") return pathNodeRoleHitValue(detail, ["symptom"], "anchor_hit");
  if (tag === "hit_root_cause") return pathNodeRoleHitValue(detail, ["root_cause"], "root_hit");
  if (tag === "edited_root_cause") {
    const direct = booleanPatternValue(detail?.edited_root_cause);
    if (direct !== null) return direct;
    if (Array.isArray(detail?.step_inspection)) return detail.step_inspection.some((step) => step?.edited_root_cause === true);
    return null;
  }
  return null;
}

function tracePatternMatches(detail, tag, value = "true") {
  const marker = tracePatternValue(detail, tag);
  const expected = normalizeTracePatternFilterValue(value) || "true";
  if (expected === "none") return marker === null;
  if (expected === "true") return marker === true;
  if (expected === "false") return marker === false;
  return false;
}

function tracePatternFilterEnabled(detail) {
  const active = activeTracePatternFilterEntries();
  return !active.length || active.every(({ tag, value }) => tracePatternMatches(detail, tag, value));
}

function filteredDetails(snapshot) {
  const query = state.traceQuery.trim().toLowerCase();
  if (!state.selectedEvalCellKey) return [];
  return activeDetails(snapshot)
    .filter((detail) => detailCellKey(detail) === state.selectedEvalCellKey)
    .filter(tracePatternFilterEnabled)
    .filter((detail) => !query || traceBlob(detail).includes(query));
}

function currentRolloutBatchDetails(snapshot) {
  if (!state.selectedEvalCellKey) return [];
  const index = traceGlobalRolloutIndex();
  return activeDetails(snapshot)
    .filter((detail) => detailCellKey(detail) === state.selectedEvalCellKey)
    .filter((detail) => rolloutIndex(detail) === index);
}

function tracePatternStats(snapshot) {
  const full = currentRolloutBatchDetails(snapshot);
  const selected = full.filter(tracePatternFilterEnabled);
  const fullInstances = new Set(full.map(traceInstanceId).filter(Boolean));
  const selectedInstances = new Set(selected.map(traceInstanceId).filter(Boolean));
  const fullPassedInstances = new Set(full.filter(traceResolved).map(traceInstanceId).filter(Boolean));
  const selectedPassedInstances = new Set(selected.filter(traceResolved).map(traceInstanceId).filter(Boolean));
  return {
    rollout: traceGlobalRolloutIndex() + 1,
    full,
    selected,
    fullPassRate: rate(full.map(traceResolved)),
    selectedPassRate: rate(selected.map(traceResolved)),
    fullInstances: fullInstances.size,
    selectedInstances: selectedInstances.size,
    fullPassedInstances: fullPassedInstances.size,
    selectedPassedInstances: selectedPassedInstances.size,
    selectedPassShareOfPassed: fullPassedInstances.size ? selectedPassedInstances.size / fullPassedInstances.size : null,
  };
}

function renderTracePatternStats(snapshot) {
  const el = document.getElementById("trace-pattern-stats");
  if (!el) return;
  if (!state.selectedEvalCellKey) {
    el.innerHTML = '<span><strong>Rollout</strong> -</span><span><strong>Pass</strong> - / -</span><span><strong>Pass+tags</strong> - (0/0)</span><span><strong>Instances</strong> 0 / 0</span>';
    return;
  }
  const stats = tracePatternStats(snapshot);
  el.innerHTML = `<span><strong>Rollout</strong> ${esc(stats.rollout)}</span><span><strong>Pass</strong> ${esc(pct(stats.selectedPassRate))} / ${esc(pct(stats.fullPassRate))}</span><span><strong>Pass+tags</strong> ${esc(pct(stats.selectedPassShareOfPassed))} (${esc(stats.selectedPassedInstances)}/${esc(stats.fullPassedInstances)})</span><span><strong>Instances</strong> ${esc(stats.selectedInstances)} / ${esc(stats.fullInstances)}</span>`;
}

function selectedDetail(snapshot) {
  const details = filteredDetails(snapshot);
  const selected = details.find((detail) => rowKey(detail) === state.selectedTraceKey);
  if (selected) return selected;
  const preferred = details.find((detail) => detail.raw_available || (detail.step_details || []).length || (detail.step_inspection || []).length) || details[0];
  return preferred ? detailForRolloutIndex(detailsForInstance(details, traceInstanceKey(preferred))) : null;
}

function locatorForDetail(detail, level = "step", options = {}) {
  if (!detail) return null;
  const params = new URLSearchParams();
  params.set("p2a", "1");
  if (options.tab) params.set("tab", options.tab);
  if (detail.experiment_id) params.set("experiment_id", detail.experiment_id);
  if (detail.provider_source) params.set("provider_source", detail.provider_source);
  if (detail.model_api_name) params.set("model_api_name", detail.model_api_name);
  if (detail.dataset || detail.data_source) params.set("dataset", detail.dataset || detail.data_source);
  if (level !== "experiment" && detail.instance_id) params.set("instance_id", detail.instance_id);
  if (level !== "experiment" && level !== "instance") {
    if (detail.rollout_id !== null && detail.rollout_id !== undefined && detail.rollout_id !== "") params.set("rollout_id", detail.rollout_id);
    else if (detail.cell_id !== null && detail.cell_id !== undefined && detail.cell_id !== "") params.set("cell_id", String(detail.cell_id));
    else params.set("rollout_index", String(rolloutIndex(detail)));
  }
  if (level === "step") params.set("step_index", String(state.selectedStepIndex || 0));
  if (state.selectedGraphNodeKey) params.set("graph_node", state.selectedGraphNodeKey);
  return `#${params.toString()}`;
}

function parseLocator(text) {
  const raw = String(text || "").trim();
  const hash = raw.includes("#") ? raw.slice(raw.indexOf("#") + 1) : raw.replace(/^#/, "");
  if (!hash) return null;
  const params = new URLSearchParams(hash);
  if (params.get("p2a") !== "1" && !params.get("experiment_id") && !params.get("instance_id")) return null;
  return Object.fromEntries(params.entries());
}

function clearReachabilityFilters() {
  CASE_FILTER_BUCKETS.forEach((key) => { state.caseFilters[key] = true; });
  TRACE_PATTERN_FILTERS.forEach((key) => { state.tracePatternFilters[key] = null; });
  state.traceQuery = "";
}

function applyLocator(snapshot, locator) {
  if (!locator) return false;
  clearReachabilityFilters();
  const cells = experimentRows(snapshot);
  const cell = cells.find((row) => (
    (!locator.experiment_id || row.experiment_id === locator.experiment_id)
    && (!locator.provider_source || row.provider_source === locator.provider_source)
    && (!locator.model_api_name || row.model_api_name === locator.model_api_name)
    && (!locator.dataset || row.dataset === locator.dataset)
  ));
  if (!cell) {
    state.permalinkNotice = "URL target was not found in the loaded experiments.";
    state.permalinkMissing = true;
    state.selectedDataset = null;
    state.selectedEvalCellKey = null;
    state.selectedExperimentKey = null;
    state.selectedTraceKey = null;
    return false;
  }
  state.permalinkMissing = false;
  state.selectedDataset = cell.dataset || locator.dataset || null;
  state.selectedEvalCellKey = cellKey(cell);
  state.selectedExperimentKey = state.selectedEvalCellKey;
  const needsTrace = Boolean(locator.instance_id || locator.rollout_id || locator.cell_id || locator.rollout_index !== undefined || locator.step_index !== undefined);
  const allDetails = activeDetails(snapshot).filter((detail) => detailCellKey(detail) === state.selectedEvalCellKey);
  const detail = needsTrace
    ? allDetails.find((item) => {
      if (locator.instance_id && String(item.instance_id) !== String(locator.instance_id)) return false;
      if (locator.rollout_id) return String(item.rollout_id || "") === String(locator.rollout_id);
      if (locator.cell_id) return String(item.cell_id || "") === String(locator.cell_id);
      if (locator.rollout_index !== undefined) return rolloutIndex(item) === Number(locator.rollout_index);
      return true;
    })
    : null;
  if (!detail && needsTrace) {
    const expected = expectedCellDetailCount(snapshot, state.selectedEvalCellKey);
    const loaded = loadedDetailCountForCell(snapshot, state.selectedEvalCellKey);
    const canStillLoad = state.detailLoadBusyKeys.has(state.selectedEvalCellKey)
      || (selectedCellCanHaveDetails(snapshot, state.selectedEvalCellKey) && loaded < expected);
    if (canStillLoad) {
      state.pendingLocator = locator;
      state.permalinkMissing = false;
      state.permalinkNotice = "Loading URL target trajectory details...";
      state.selectedTraceKey = null;
      if (locator.rollout_index !== undefined) setTraceGlobalRolloutIndex(locator.rollout_index);
      state.selectedStepIndex = Number(locator.step_index || 0);
      state.selectedGraphNodeKey = locator.graph_node || null;
      setTab(locator.tab === "traces" || needsTrace ? "traces" : "overview");
      return true;
    }
    state.permalinkNotice = "URL target was not found; it may have been deleted or not loaded in this snapshot.";
    state.permalinkMissing = true;
    state.selectedTraceKey = null;
    return false;
  }
  if (detail) {
    state.selectedTraceKey = rowKey(detail);
    setTraceGlobalRolloutIndex(rolloutIndex(detail));
  }
  state.selectedStepIndex = Number(locator.step_index || 0);
  state.selectedGraphNodeKey = locator.graph_node || null;
  state.permalinkNotice = "";
  setTab(locator.tab === "traces" && (detail || needsTrace) ? "traces" : "overview");
  return true;
}

function applyPendingLocator(snapshot) {
  if (!state.pendingLocator) return;
  const locator = state.pendingLocator;
  state.pendingLocator = null;
  applyLocator(snapshot, locator);
  syncFilterControls();
}

function currentDashboardUrl(level = "step") {
  const detail = selectedDetail(state.snapshot);
  const locator = locatorForDetail(detail, level, { tab: level === "experiment" ? "overview" : "traces" });
  if (!locator) return "";
  if (typeof window === "undefined" || !window.location) return locator;
  return `${window.location.origin || ""}${window.location.pathname || ""}${locator}`;
}

function syncFilterControls() {
  document.querySelectorAll(".case-filter-checkbox").forEach((input) => {
    const bucket = input.dataset.caseFilter;
    if (bucket) input.checked = state.caseFilters[bucket] !== false;
  });
  document.querySelectorAll(".trace-pattern-checkbox").forEach((input) => {
    const tag = input.dataset.patternFilter;
    if (tag) input.checked = state.tracePatternFilters[tag] === true;
  });
  document.querySelectorAll(".trace-pattern-cycle").forEach((button) => {
    const tag = button.dataset.patternFilter;
    const value = tag ? normalizeTracePatternFilterValue(state.tracePatternFilters[tag]) : null;
    const label = button.dataset.patternLabel || tag || "pattern";
    button.classList.toggle("is-true", value === "true");
    button.classList.toggle("is-false", value === "false");
    button.classList.toggle("is-none", value === "none");
    button.setAttribute("aria-pressed", value ? "true" : "false");
    button.setAttribute("data-pattern-state", value || "");
    const titleSuffix = value === "true" ? "true" : value === "false" ? "false" : value === "none" ? "unavailable" : "not filtered";
    button.setAttribute("title", `${label}: ${titleSuffix}`);
    button.setAttribute("aria-label", `${label}: ${titleSuffix}`);
  });
  const traceSearch = document.getElementById("trace-search");
  if (traceSearch) traceSearch.value = state.traceQuery || "";
}

async function writeClipboardText(text) {
  if (!text) return false;
  if (typeof navigator !== "undefined" && navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (_error) {
      // Fall back for plain HTTP dashboard sessions.
    }
  }
  if (typeof document === "undefined" || typeof document.createElement !== "function") return false;
  const parent = document.body || document.documentElement;
  if (!parent?.appendChild) return false;
  const input = document.createElement("textarea");
  input.value = text;
  input.setAttribute("readonly", "");
  input.style.position = "fixed";
  input.style.left = "-10000px";
  input.style.top = "-10000px";
  parent.appendChild(input);
  input.focus?.();
  input.select?.();
  input.setSelectionRange?.(0, text.length);
  let copied = false;
  try {
    copied = typeof document.execCommand === "function" && document.execCommand("copy");
  } catch (_error) {
    copied = false;
  }
  input.remove?.();
  if (input.parentNode?.removeChild) input.parentNode.removeChild(input);
  return copied;
}

async function copyDashboardLink(level) {
  const url = currentDashboardUrl(level);
  if (!url) return;
  state.permalinkNotice = `Copying ${level} URL...`;
  renderSelectedExperiment(state.snapshot);
  const copied = await writeClipboardText(url);
  state.permalinkNotice = copied ? `Copied ${level} URL.` : `Copy failed; URL: ${url}`;
  renderSelectedExperiment(state.snapshot);
}

function deleteTargetKey(target) {
  return JSON.stringify({
    experiment_id: target.experiment_id || "",
    provider_source: target.provider_source || "",
    dataset: target.dataset || "",
    model_api_name: target.model_api_name || "",
    model_label: target.model_label || "",
  });
}

function deleteTargetFromKey(key) {
  try {
    return JSON.parse(key);
  } catch (_error) {
    return {};
  }
}

function adminTargetLabel(target) {
  if (target?.scope === "all") return "all eval cells";
  const parts = [
    target?.dataset ? `dataset=${target.dataset}` : "",
    target?.provider_source ? `source=${target.provider_source}` : "",
    target?.experiment_id ? `experiment=${target.experiment_id}` : "",
    target?.model_label ? `model=${target.model_label}` : target?.model_api_name ? `model=${target.model_api_name}` : "",
  ].filter(Boolean);
  return parts.length ? parts.join(" / ") : "all eval cells";
}

function renderAdminTargetList(targets, emptyText) {
  const items = (targets || []).map((target) => `<code>${esc(adminTargetLabel(target))}</code>`);
  return items.join("") || `<span class="muted">${esc(emptyText)}</span>`;
}

function rebuildStatusTarget(status) {
  const scope = status?.last_scope;
  if (!scope || typeof scope !== "object") return null;
  const target = {
    experiment_id: String(scope.experiment_id || "").trim(),
    provider_source: String(scope.provider_source || "").trim(),
    dataset: String(scope.dataset || "").trim(),
    model_api_name: String(scope.model_api_name || "").trim(),
    model_label: String(scope.model_label || "").trim(),
  };
  return Object.values(target).some(Boolean) ? target : { scope: "all" };
}

function syncAdminRebuildTargetsFromStatus() {
  if (state.rebuildStatus?.active === true) {
    if (!state.adminRebuildingTargets.length) {
      const target = rebuildStatusTarget(state.rebuildStatus);
      state.adminRebuildingTargets = target ? [target] : [];
    }
    return;
  }
  state.adminRebuildingTargets = [];
}

function selectedDeleteTargets() {
  const targets = [...state.adminDeleteKeys].map(deleteTargetFromKey);
  if (state.adminManualTarget) targets.push(state.adminManualTarget);
  const seen = new Set();
  return targets.filter((target) => {
    const clean = {
      experiment_id: String(target.experiment_id || "").trim(),
      provider_source: String(target.provider_source || "").trim(),
      dataset: String(target.dataset || "").trim(),
    };
    if (!clean.experiment_id && !clean.provider_source && !clean.dataset) return false;
    const key = deleteTargetKey(clean);
    if (seen.has(key)) return false;
    seen.add(key);
    return clean;
  });
}

function selectedRebuildTargets() {
  const targets = [...state.adminRebuildKeys].map(deleteTargetFromKey);
  const seen = new Set();
  return targets.filter((target) => {
    const clean = {
      experiment_id: String(target.experiment_id || "").trim(),
      provider_source: String(target.provider_source || "").trim(),
      dataset: String(target.dataset || "").trim(),
      model_api_name: String(target.model_api_name || "").trim(),
      model_label: String(target.model_label || "").trim(),
    };
    if (!clean.experiment_id && !clean.provider_source && !clean.dataset && !clean.model_api_name && !clean.model_label) return false;
    const key = deleteTargetKey(clean);
    if (seen.has(key)) return false;
    seen.add(key);
    return clean;
  });
}

async function apiPost(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "same-origin",
    body: JSON.stringify(payload || {}),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || body.error || `HTTP ${response.status}`);
  return body;
}

function cellDetailsUrl(cell, { offset = 0, limit = 50 } = {}) {
  const params = new URLSearchParams();
  ["experiment_id", "provider_source", "dataset", "model_api_name", "model_label"].forEach((field) => {
    if (cell?.[field]) params.set(field, cell[field]);
  });
  params.set("offset", String(offset));
  params.set("limit", String(limit));
  return `/api/details?${params.toString()}`;
}

function datasetMetricsUrl(dataset) {
  const params = new URLSearchParams();
  if (dataset) params.set("dataset", dataset);
  return `/api/metrics?${params.toString()}`;
}

function mergeModelMetrics(snapshot, rows) {
  if (!snapshot || !Array.isArray(rows)) return;
  const merged = new Map((snapshot.model_metrics || []).map((row) => [cellKey(row), row]));
  rows.forEach((row) => {
    const key = cellKey(row);
    if (!key) return;
    merged.set(key, { ...(merged.get(key) || {}), ...row });
  });
  snapshot.model_metrics = [...merged.values()].sort((a, b) => (
    String(a.dataset || "").localeCompare(String(b.dataset || ""))
    || String(a.model_label || "").localeCompare(String(b.model_label || ""))
  ));
  snapshot.eval_cells = (snapshot.eval_cells || []).map((cell) => {
    const row = merged.get(cellKey(cell));
    if (!row) return cell;
    return {
      ...cell,
      cache_ready: row.detail_cache_ready_rollouts ?? cell.cache_ready,
      cache_pending: row.detail_cache_pending_rollouts ?? cell.cache_pending,
      resolved_rate: row.resolved_rate ?? cell.resolved_rate,
      root_hit_rate: row.root_hit_rate ?? row.ground_truth_hit_rate ?? cell.root_hit_rate,
      path_node_recall: row.avg_path_node_recall ?? row.avg_node_recall ?? cell.path_node_recall,
      chain_node_recall: row.avg_chain_node_recall ?? row.avg_node_recall ?? cell.chain_node_recall,
      read_precision: row.avg_path_read_precision ?? row.avg_read_precision ?? cell.read_precision,
    };
  });
}

function mergeCellDetails(snapshot, key, details) {
  if (!snapshot) return;
  const incoming = Array.isArray(details) ? details : [];
  const others = (snapshot.details || []).filter((detail) => detailCellKey(detail) !== key);
  const merged = new Map();
  (snapshot.details || []).forEach((detail) => {
    if (detailCellKey(detail) === key) merged.set(rowKey(detail), detail);
  });
  incoming.forEach((detail) => {
    const id = rowKey(detail);
    merged.set(id, { ...(merged.get(id) || {}), ...detail });
  });
  snapshot.details = [...others, ...merged.values()].sort(compareTraceDetails);
  snapshot.detail_count = (snapshot.details || []).length;
}

function missingDetailPlaceholder(cell, key, index, message) {
  return {
    eval_cell_key: key,
    experiment_key: key,
    source_kind: cell?.source_kind || "",
    experiment_id: cell?.experiment_id || "",
    provider_source: cell?.provider_source || "",
    dataset: cell?.dataset || "",
    data_source: cell?.dataset || "",
    model_api_name: cell?.model_api_name || "",
    model_label: cell?.model_label || "",
    instance_id: `missing-detail-${index + 1}`,
    rollout_id: `missing-detail-${index + 1}`,
    rollout_index: index,
    record_index: index,
    resolved: false,
    error: message,
    system_error: true,
    dashboard_detail_load_error: true,
    raw_available: false,
    issue_description: message,
    not_path_evaluable_reason: "detail_load_error",
    not_chain_evaluable_reason: "detail_load_error",
    step_details: [],
    step_inspection: [],
  };
}

function fillMissingDetails(snapshot, key, cell, total, message) {
  const loaded = loadedDetailCountForCell(snapshot, key);
  const missing = Math.max(0, Number(total || 0) - loaded);
  if (!missing) return;
  const placeholders = Array.from(
    { length: missing },
    (_item, offset) => missingDetailPlaceholder(cell, key, loaded + offset, message)
  );
  mergeCellDetails(snapshot, key, placeholders);
}

function selectedCellCanHaveDetails(snapshot, key) {
  const row = experimentRows(snapshot).find((item) => cellKey(item) === key);
  if (!row) return false;
  return runDoneCount(row) + Number(row.detail_count || 0) > 0;
}

function expectedCellDetailCount(snapshot, key) {
  const row = experimentRows(snapshot).find((item) => cellKey(item) === key);
  if (!row) return 0;
  return runDoneCount(row) || Number(row.detail_count || 0);
}

function renderDetailLoadProgress(key) {
  const progress = state.detailLoadProgress[key] || {};
  const loaded = Number(progress.loaded || 0);
  const total = Number(progress.total || 0);
  const max = total > 0 ? ` max="${esc(total)}" value="${esc(Math.min(loaded, total))}"` : "";
  const label = total > 0 ? `Loading trajectory details ${fmt(Math.min(loaded, total))}/${fmt(total)}...` : "Loading trajectory details...";
  return `<div class="trace-load-progress"><progress${max}></progress><span>${esc(label)}</span></div>`;
}

function cellNeedsDetailLoad(snapshot, key) {
  if (!key || state.detailLoadBusyKeys.has(key)) return false;
  if (state.detailLoadErrors[key]) return false;
  if (!selectedCellCanHaveDetails(snapshot, key)) return false;
  const expected = expectedCellDetailCount(snapshot, key);
  if (expected <= 0) return false;
  const loaded = loadedDetailCountForCell(snapshot, key);
  return loaded < expected;
}

async function loadCellDetails(key) {
  if (!state.snapshot || !key || state.detailLoadBusyKeys.has(key)) return;
  const cell = experimentRows(state.snapshot).find((row) => cellKey(row) === key);
  if (!cell) return;
  const total = expectedCellDetailCount(state.snapshot, key);
  const pageSize = 5;
  let loaded = loadedDetailCountForCell(state.snapshot, key);
  let offset = loaded;
  let exhausted = false;
  if (total > 0 && loaded >= total) {
    state.detailLoadedCellKeys.add(key);
    return;
  }
  state.detailLoadBusyKeys.add(key);
  state.detailLoadProgress[key] = { loaded, total };
  delete state.detailLoadErrors[key];
  renderTraceInspector(state.snapshot);
  try {
    while (loaded < Math.max(total, 1) && offset < Math.max(total, 1)) {
      const before = loaded;
      const requestOffset = offset;
      const response = await fetch(cellDetailsUrl(cell, { offset: requestOffset, limit: pageSize }), { cache: "no-store", credentials: "same-origin" });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(body.detail || body.error || `HTTP ${response.status}`);
      const details = Array.isArray(body.details) ? body.details : [];
      const responseOffset = Number(body.offset);
      mergeCellDetails(state.snapshot, key, details);
      loaded = loadedDetailCountForCell(state.snapshot, key);
      offset = Math.max(
        Number.isFinite(responseOffset) ? responseOffset + details.length : requestOffset + details.length,
        loaded
      );
      state.detailLoadProgress[key] = { loaded: total > 0 ? Math.min(loaded, total) : loaded, total };
      renderTraceInspector(state.snapshot);
      if (!details.length || details.length < pageSize) exhausted = true;
      if (exhausted || total <= 0 || offset <= requestOffset) break;
      if (loaded <= before && offset >= total) break;
    }
    if (total > 0 && loaded < total && (offset >= total || exhausted)) {
      fillMissingDetails(
        state.snapshot,
        key,
        cell,
        total,
        `Trajectory detail loading ended after ${loaded}/${total} traces; ${total - loaded} trace detail could not be loaded.`
      );
      loaded = loadedDetailCountForCell(state.snapshot, key);
    }
    if (total <= 0 || loaded >= total) state.detailLoadedCellKeys.add(key);
  } catch (error) {
    state.detailLoadErrors[key] = String(error.message || error);
  } finally {
    state.detailLoadBusyKeys.delete(key);
    delete state.detailLoadProgress[key];
    render();
  }
}

async function loadDatasetMetrics(dataset) {
  if (!state.snapshot || !dataset || state.metricLoadBusyDatasets.has(dataset) || state.metricLoadedDatasets.has(dataset)) return;
  state.metricLoadBusyDatasets.add(dataset);
  delete state.metricLoadErrors[dataset];
  delete state.metricLoadNotices[dataset];
  try {
    const response = await fetch(datasetMetricsUrl(dataset), { cache: "no-store", credentials: "same-origin" });
    const body = await response.json().catch(() => ({}));
    if (!response.ok && (response.status === 404 || response.status === 501)) {
      state.metricLoadNotices[dataset] = "Cached metrics endpoint is unavailable; showing snapshot metrics.";
      state.metricLoadedDatasets.add(dataset);
      return;
    }
    if (!response.ok) throw new Error(body.detail || body.error || `HTTP ${response.status}`);
    mergeModelMetrics(state.snapshot, body.model_metrics || []);
    state.metricLoadedDatasets.add(dataset);
  } catch (error) {
    state.metricLoadErrors[dataset] = String(error.message || error);
  } finally {
    state.metricLoadBusyDatasets.delete(dataset);
    renderModels(state.snapshot);
    renderExperiments(state.snapshot);
  }
}

async function loadRebuildStatus() {
  if (!state.admin.authenticated) {
    state.rebuildStatus = null;
    state.adminRebuildingTargets = [];
    stopRebuildStatusPoll();
    syncSnapshotControls();
    renderOperationStatus();
    renderAdminPanel(state.snapshot);
    return;
  }
  const wasActive = state.rebuildStatus?.active === true;
  try {
    const response = await fetch("/api/rebuild/status", { cache: "no-store", credentials: "same-origin" });
    if (!response.ok) return;
    const payload = await response.json();
    state.rebuildStatus = payload.status || null;
    syncAdminRebuildTargetsFromStatus();
    syncSnapshotControls();
    renderOperationStatus();
    renderAdminPanel(state.snapshot);
    const isActive = state.rebuildStatus?.active === true;
    if (isActive) scheduleRebuildStatusPoll();
    else {
      stopRebuildStatusPoll();
      if (wasActive) loadSnapshot({ silent: true, queueIfBusy: true });
    }
  } catch (_error) {
    // Keep the last visible status until the next successful poll.
  }
}

function scheduleRebuildStatusPoll() {
  if (state.rebuildStatusTimer) return;
  state.rebuildStatusTimer = window.setTimeout(() => {
    state.rebuildStatusTimer = null;
    loadRebuildStatus();
  }, 2000);
}

function stopRebuildStatusPoll() {
  if (!state.rebuildStatusTimer) return;
  window.clearTimeout(state.rebuildStatusTimer);
  state.rebuildStatusTimer = null;
}

async function loadAdminStatus() {
  try {
    const response = await fetch("/api/auth/status", { cache: "no-store", credentials: "same-origin" });
    if (!response.ok) return;
    const payload = await response.json();
    state.admin.enabled = payload.admin_enabled === true;
    state.admin.authenticated = payload.admin === true;
    syncAdminControls();
    renderAdminPanel(state.snapshot);
    loadRebuildStatus();
  } catch (_error) {
    state.admin.enabled = false;
    state.admin.authenticated = false;
    state.rebuildStatus = null;
    state.adminRebuildingTargets = [];
    syncAdminControls();
    renderOperationStatus();
  }
}

async function loginAdmin() {
  try {
    const password = document.getElementById("admin-password")?.value || "";
    await apiPost("/api/auth/login", { password });
    state.admin.enabled = true;
    state.admin.authenticated = true;
    state.adminMessage = "";
    syncAdminControls();
    loadRebuildStatus();
    render();
  } catch (error) {
    state.adminMessage = String(error.message || error);
    syncAdminControls();
    renderAdminPanel(state.snapshot);
  }
}

async function logoutAdmin() {
  try {
    await apiPost("/api/auth/logout", {});
  } catch (_error) {
    // Local session state is still cleared if the server already forgot the token.
  }
  state.admin.authenticated = false;
  state.adminDeleteKeys.clear();
  state.adminRebuildKeys.clear();
  state.adminDeletingTargets = [];
  state.adminRebuildingTargets = [];
  state.adminPreview = null;
  state.rebuildStatus = null;
  state.adminMessage = "";
  syncAdminControls();
  renderOperationStatus();
  render();
}

function syncAdminControls() {
  const form = document.getElementById("admin-login");
  const loginButton = document.getElementById("admin-login-button");
  const logoutButton = document.getElementById("admin-logout-button");
  const password = document.getElementById("admin-password");
  const status = document.getElementById("admin-status");
  if (!form) return;
  form.hidden = false;
  if (loginButton) {
    loginButton.hidden = state.admin.authenticated || !state.admin.enabled;
    loginButton.disabled = false;
    loginButton.textContent = "Log in";
  }
  if (logoutButton) logoutButton.hidden = !state.admin.authenticated;
  if (password) {
    password.hidden = state.admin.authenticated || !state.admin.enabled;
    password.disabled = false;
    password.placeholder = "Admin password";
  }
  if (status) {
    status.textContent = state.adminMessage || (state.admin.authenticated
      ? "Admin unlocked"
      : state.admin.enabled
        ? "Enter admin password"
        : "Admin not configured by server");
  }
}

function renderAdminPanel(snapshot) {
  const panel = document.getElementById("admin-panel");
  if (!panel) return;
  if (!state.admin.authenticated) {
    panel.hidden = true;
    panel.innerHTML = "";
    return;
  }
  panel.hidden = false;
  const deleteTargets = selectedDeleteTargets();
  const rebuildTargets = selectedRebuildTargets();
  const preview = state.adminPreview;
  const counts = preview?.counts || {};
  const busy = state.adminBusy;
  const rebuildMessage = rebuildStatusMessage();
  const deletingTargets = state.adminDeletingTargets || [];
  const rebuildingTargets = state.adminRebuildingTargets || [];
  const progressHtml = [
    rebuildingTargets.length
      ? `<div class="admin-targets"><span>Rebuilding now:</span> ${renderAdminTargetList(rebuildingTargets, "")}</div>`
      : "",
    deletingTargets.length
      ? `<div class="admin-targets"><span>Deleting now:</span> ${renderAdminTargetList(deletingTargets, "")}</div>`
      : "",
    rebuildMessage ? `<div class="rebuild-inline-status ${state.rebuildStatus?.phase === "failed" ? "bad" : ""}">${esc(rebuildMessage)}</div>` : "",
  ].join("");
  panel.innerHTML = `
    <h2>DB admin actions</h2>
    ${progressHtml ? `<div class="admin-progress">${progressHtml}</div>` : ""}
    <div class="admin-actions">
      <button id="admin-rebuild-all" type="button" ${!busy ? "" : "disabled"}>${busy === "rebuild-all" ? "Queueing..." : "Rebuild all"}</button>
      <button id="admin-rebuild-selected" type="button" ${rebuildTargets.length && !busy ? "" : "disabled"}>${busy === "rebuild" ? "Queueing..." : "Rebuild selected"}</button>
      <button id="admin-preview-delete" type="button" ${deleteTargets.length && !busy ? "" : "disabled"}>${busy === "preview-delete" ? "Previewing..." : "Preview delete"}</button>
      <button id="admin-confirm-delete" type="button" ${preview && !busy ? "" : "disabled"}>${busy === "delete" ? "Deleting..." : "Delete"}</button>
    </div>
    <div class="admin-message">${esc(state.adminMessage || "")}</div>
    ${preview ? `<div class="panel-note">Preview: ${esc(counts.run_cells || 0)} run cells, ${esc(counts.raw_rollouts || 0)} raw rollouts, ${esc(counts.quantitative_metrics || 0)} metrics, ${esc(counts.experiments || 0)} experiments.</div>` : ""}
  `;
  document.getElementById("admin-rebuild-all")?.addEventListener("click", async () => {
    let queued = false;
    try {
      state.adminBusy = "rebuild-all";
      state.adminRebuildingTargets = [{ scope: "all" }];
      state.adminMessage = "Rebuild queued; clearing all dashboard detail cache in the background.";
      renderAdminPanel(snapshot);
      const result = await apiPost("/api/rebuild", {});
      queued = true;
      applyQueuedRebuildStatus(result);
      state.adminPreview = null;
      state.adminMessage = rebuildQueuedMessage(result);
      await loadSnapshot({
        silent: true,
        queueIfBusy: true,
      });
    } catch (error) {
      if (!queued && state.rebuildStatus?.active !== true) state.adminRebuildingTargets = [];
      state.adminMessage = String(error.message || error);
    } finally {
      state.adminBusy = "";
      renderAdminPanel(state.snapshot || snapshot);
      renderExperiments(state.snapshot || snapshot);
      loadRebuildStatus();
    }
  });
  document.getElementById("admin-rebuild-selected")?.addEventListener("click", async () => {
    const targets = selectedRebuildTargets();
    let queued = false;
    try {
      state.adminBusy = "rebuild";
      state.adminRebuildingTargets = targets;
      state.adminMessage = "Rebuild queued; clearing selected cache in the background.";
      renderAdminPanel(snapshot);
      const result = await apiPost("/api/rebuild", { targets });
      queued = true;
      applyQueuedRebuildStatus(result);
      state.adminPreview = null;
      state.adminMessage = rebuildQueuedMessage(result);
      await loadSnapshot({
        silent: true,
        queueIfBusy: true,
      });
    } catch (error) {
      if (!queued && state.rebuildStatus?.active !== true) state.adminRebuildingTargets = [];
      state.adminMessage = String(error.message || error);
    } finally {
      state.adminBusy = "";
      renderAdminPanel(state.snapshot || snapshot);
      renderExperiments(state.snapshot || snapshot);
      loadRebuildStatus();
    }
  });
  document.getElementById("admin-preview-delete")?.addEventListener("click", async () => {
    const targets = selectedDeleteTargets();
    try {
      state.adminBusy = "preview-delete";
      state.adminMessage = "Previewing selected DB rows...";
      renderAdminPanel(snapshot);
      state.adminPreview = await apiPost("/api/delete/preview", { targets });
      state.adminMessage = "";
    } catch (error) {
      state.adminMessage = String(error.message || error);
    } finally {
      state.adminBusy = "";
    }
    renderAdminPanel(snapshot);
  });
  document.getElementById("admin-confirm-delete")?.addEventListener("click", async () => {
    const targets = selectedDeleteTargets();
    try {
      state.adminBusy = "delete";
      state.adminDeletingTargets = targets;
      state.adminMessage = "Deleting selected DB rows...";
      renderAdminPanel(snapshot);
      const result = await apiPost("/api/delete", { targets });
      state.adminMessage = `Deleted ${result.counts?.run_cells || 0} run cells.`;
      state.adminDeleteKeys.clear();
      state.adminRebuildKeys.clear();
      state.adminManualTarget = null;
      state.adminPreview = null;
      await loadSnapshot({
        silent: true,
      });
    } catch (error) {
      state.adminMessage = String(error.message || error);
      renderAdminPanel(snapshot);
    } finally {
      state.adminBusy = "";
      state.adminDeletingTargets = [];
      renderAdminPanel(state.snapshot || snapshot);
    }
  });
}

function canonicalNodeRole(role) {
  if (role === "pre_symptom") return "test_adapter";
  return role || "";
}

const NODE_ROLE_LABELS = {
  test_harness: "test harness",
  test_adapter: "test-adapter",
  symptom: "symptom",
  intermediate: "intermediate",
  fix_adapter: "fix-adapter",
  root_cause: "root cause",
};

function roleTone(node) {
  if (isSymptomRootCauseNode(node)) return "symptom-root-cause";
  const role = canonicalNodeRole(node?.node_role);
  if (role === "test_harness") return "test";
  if (role === "test_adapter") return "test-adapter";
  if (role === "root_cause") return "root";
  if (role === "symptom") return "symptom";
  if (role === "intermediate") return "path";
  if (role === "fix_adapter") return "fix-adapter";
  return "context";
}

function nodeDistance(node) {
  const distance = Number(node?.normalized_distance);
  return Number.isFinite(distance) ? distance : null;
}

function graphLayer(node) {
  const role = canonicalNodeRole(node?.node_role);
  const distance = nodeDistance(node);
  if (role === "test_harness") return 0;
  if (role === "test_adapter") return 1;
  if (role === "symptom") return 2;
  if (role === "root_cause") return 10;
  if (role === "intermediate" || role === "fix_adapter") {
    if (distance === null) return 6;
    return Math.max(3, Math.min(9, 3 + Math.round((1 - distance) * 6)));
  }
  if (distance === null) return 2;
  return Math.max(0, Math.min(10, Math.round((1 - distance) * 10)));
}

function compareGraphNodes(a, b) {
  const firstA = a.first_step === null || a.first_step === undefined ? Infinity : Number(a.first_step);
  const firstB = b.first_step === null || b.first_step === undefined ? Infinity : Number(b.first_step);
  if (firstA !== firstB) return firstA - firstB;
  const distanceA = nodeDistance(a) ?? Infinity;
  const distanceB = nodeDistance(b) ?? Infinity;
  if (distanceA !== distanceB) return distanceB - distanceA;
  return String(a.key || "").localeCompare(String(b.key || ""));
}

function detailRootKeys(detail) {
  return new Set(pathProjection(detail).roots || []);
}

function detailSymptomKeys(detail) {
  return new Set(pathProjection(detail).anchors || []);
}

function hitNodeIsRootCause(node, detail) {
  return canonicalNodeRole(node?.node_role) === "root_cause" || node?.root_cause === true || detailRootKeys(detail).has(node?.key);
}

function hitNodeIsSymptom(node, detail) {
  return canonicalNodeRole(node?.node_role) === "symptom" || node?.selected_issue_anchor === true || node?.anchor === true || detailSymptomKeys(detail).has(node?.key);
}

function nodeIsFaultSidePatch(node) {
  return node?.patched_callable === true || node?.patch_role === "root_cause" || node?.patch_role === "fix_adapter";
}

function hitNodeIsSymptomRootCause(node, detail) {
  return hitNodeIsSymptom(node, detail) && (hitNodeIsRootCause(node, detail) || nodeIsFaultSidePatch(node));
}

const STEP_ROLE_COLORS = {
  symptom: "#dcfce7",
  "test-adapter": "#ffedd5",
  intermediate: "#dbeafe",
  "fix-adapter": "#fce7f3",
  root: "#fee2e2",
};

function stepNodeSegment(node, detail) {
  if (hitNodeIsSymptomRootCause(node, detail)) return "symptom-root-cause";
  if (hitNodeIsSymptom(node, detail)) return "symptom";
  if (hitNodeIsRootCause(node, detail)) return "root";
  const role = canonicalNodeRole(node?.node_role);
  if (role === "test_adapter") return "test-adapter";
  if (role === "intermediate") return "intermediate";
  if (role === "fix_adapter") return "fix-adapter";
  return null;
}

function stepRoleSegments(step, detail) {
  const scored = step?.scored || step || {};
  const nodes = scored.hit_nodes || [];
  if (step?.edited_root_cause) return ["root-edit"];
  if (step?.action_family === "edit" || (step?.write_actions || []).length || (scored.writes || []).length) return ["edit"];
  const roles = [];
  const present = new Set(nodes.map((node) => stepNodeSegment(node, detail)).filter(Boolean));
  for (const role of ["symptom-root-cause", "symptom", "test-adapter", "intermediate", "fix-adapter", "root"]) {
    if (present.has(role)) roles.push(role);
  }
  if (roles.length) return roles;
  if ((scored.n_reads || 0) > 0 || (step?.recovered_reads || []).length || step?.action_family === "read") return ["offmap"];
  if (step?.action_family === "exec" || step?.action_family === "other" || scored.family === "exec" || scored.family === "other") return ["exec-other"];
  if ((detail?.bad_patterns || {}).error_spiral) return ["bad"];
  return ["neutral"];
}

function stepSegmentsStyle(segments) {
  const colorSegments = segments.filter((segment) => STEP_ROLE_COLORS[segment]);
  if (colorSegments.length <= 1) return "";
  const stops = colorSegments.map((segment, index) => {
    const start = (index / colorSegments.length) * 100;
    const end = ((index + 1) / colorSegments.length) * 100;
    return `${STEP_ROLE_COLORS[segment]} ${start.toFixed(2)}% ${end.toFixed(2)}%`;
  });
  return ` style="--step-bg: linear-gradient(90deg, ${stops.join(", ")});"`;
}

function stepSegmentsMarkup(segments) {
  const colorSegments = segments.filter((segment) => segment === "symptom-root-cause" || STEP_ROLE_COLORS[segment]);
  if (colorSegments.length <= 1) return "";
  return `<span class="step-segments" aria-hidden="true">${colorSegments.map((segment) => `<span class="step-segment ${esc(segment)}"></span>`).join("")}</span>`;
}

function nodeKeysFromSummaries(nodes) {
  return (nodes || [])
    .map((node) => node?.key)
    .filter(Boolean);
}

function rawStepLabel(step, fallback) {
  const value = step?.step_index;
  if (value !== null && value !== undefined && Number.isFinite(Number(value))) return Number(value);
  return fallback + 1;
}

function stepLabelOffset(steps) {
  const labels = (steps || [])
    .map((step, index) => rawStepLabel(step, index))
    .filter((value) => Number.isFinite(value));
  return labels.length && Math.min(...labels) === 0 ? 1 : 0;
}

function displayStepLabel(step, detail, fallback = 0) {
  const value = step?.step_index;
  if (value !== null && value !== undefined && Number.isFinite(Number(value))) {
    return Number(value) + stepLabelOffset(detailSteps(detail));
  }
  return fallback + 1;
}

function detailSteps(detail) {
  const steps = (detail?.step_inspection || []).length ? detail.step_inspection : detail?.step_details || [];
  return Array.isArray(steps) ? steps : [];
}

function stepHitNodesForFirstStep(step) {
  const scored = step?.scored || step || {};
  return scored.hit_nodes || [];
}

function displayFirstStepsByNode(detail) {
  const steps = detailSteps(detail);
  const offset = stepLabelOffset(steps);
  const first = new Map();
  steps.forEach((step, index) => {
    const value = step?.step_index;
    const label = value !== null && value !== undefined && Number.isFinite(Number(value))
      ? Number(value) + offset
      : index + 1;
    for (const node of stepHitNodesForFirstStep(step)) {
      const key = node?.key;
      if (!key) continue;
      if (!first.has(key) || label < first.get(key)) first.set(key, label);
    }
  });
  return first;
}

function displayBlockIndex(block, detail, fallback = 0) {
  const blocks = detail?.purpose_blocks || [];
  const values = blocks
    .map((item, index) => {
      const value = item?.block_index;
      return value !== null && value !== undefined && Number.isFinite(Number(value)) ? Number(value) : index + 1;
    })
    .filter((value) => Number.isFinite(value));
  const offset = values.length && Math.min(...values) === 0 ? 1 : 0;
  const raw = block?.block_index;
  if (raw !== null && raw !== undefined && Number.isFinite(Number(raw))) return Number(raw) + offset;
  return fallback + 1;
}

function finalEditNodeKeys(detail) {
  const steps = [...(detail?.step_details || [])]
    .map((step, index) => ({ ...step, __order: Number(step.trace_index ?? step.step_index ?? index) }))
    .sort((a, b) => a.__order - b.__order);
  for (let index = steps.length - 1; index >= 0; index -= 1) {
    const keys = nodeKeysFromSummaries(steps[index].write_hit_nodes);
    if (keys.length) return new Set(keys);
  }
  const inspection = [...(detail?.step_inspection || [])]
    .map((step, index) => ({ ...step, __order: Number(step.trace_index ?? step.step_index ?? index) }))
    .sort((a, b) => a.__order - b.__order);
  for (let index = inspection.length - 1; index >= 0; index -= 1) {
    const keys = nodeKeysFromSummaries(inspection[index].write_hit_nodes || inspection[index].scored?.write_hit_nodes);
    if (keys.length) return new Set(keys);
  }
  return new Set();
}

function stepTone(step, detail) {
  const segments = stepRoleSegments(step, detail);
  return segments.length > 1 ? "multi-hit" : segments[0];
}

function graphNodes(detail, { includeContext = state.showGraphContext } = {}) {
  const projection = pathProjection(detail);
  const currentPathNodes = pathNodes(detail);
  const contextNodes = graphContextNodes(detail);
  const anchors = new Set(projection.anchors || []);
  const roots = new Set(projection.roots || []);
  const displayFirstSteps = displayFirstStepsByNode(detail);
  const editedNodes = finalEditNodeKeys(detail);
  const topologyByKey = new Map((detail?.graph_topology?.nodes || []).map((node) => [node?.key, node]));
  const annotateNode = (node) => {
    const topologyNode = topologyByKey.get(node?.key) || {};
    const role = canonicalNodeRole(node?.node_role || topologyNode?.node_role);
    return ({
      ...topologyNode,
      ...node,
      node_role: role || node?.node_role || topologyNode?.node_role,
      source: node?.source || topologyNode?.source,
      source_preview: node?.source_preview || topologyNode?.source_preview,
      selected_issue_anchor: Boolean(node?.selected_issue_anchor || anchors.has(node?.key)),
      root_cause: Boolean(node?.root_cause || roots.has(node?.key) || role === "root_cause"),
      first_step: displayFirstSteps.has(node?.key) ? displayFirstSteps.get(node?.key) : node?.first_step ?? topologyNode?.first_step,
      final_edit: Boolean(editedNodes.has(node?.key)),
    });
  };
  const topologyNodes = detail?.graph_topology?.nodes || [];
  const nonTestTopologyNodes = topologyNodes.filter((node) => canonicalNodeRole(node?.node_role) !== "test_harness");
  const nonTestProjectedNodes = [...contextNodes, ...currentPathNodes]
    .filter((node) => canonicalNodeRole(node?.node_role) !== "test_harness");
  if (includeContext && topologyNodes.length) return topologyNodes.map((node) => annotateNode({ ...node, group: node.rewardable ? "path" : "context" }));
  if (!includeContext && nonTestTopologyNodes.length) {
    return nonTestTopologyNodes.map((node) => annotateNode({ ...node, group: node.rewardable ? "path" : "context" }));
  }
  const projected = includeContext ? [...contextNodes, ...currentPathNodes] : nonTestProjectedNodes;
  if (projected.length) return projected.map(annotateNode);
  return nonTestTopologyNodes.map((node) => annotateNode({ ...node, group: node.rewardable ? "path" : "context" }));
}

function normalizeGraphEdge(edge, edgeType = "path") {
  if (Array.isArray(edge)) {
    return { caller: edge[0], callee: edge[1], source: edge[0], target: edge[1], edge_type: edgeType };
  }
  const caller = edge?.caller || edge?.source;
  const callee = edge?.callee || edge?.target;
  return { ...edge, caller, callee, source: caller, target: callee, edge_type: edge?.edge_type || edgeType };
}

function graphEdges(detail, { includeContext = state.showGraphContext, includeGraph = state.graphEdgeFilters.graph === true } = {}) {
  const projection = pathProjection(detail);
  const currentPathEdges = pathEdges(detail);
  const contextEdges = projection.context_edges || [];
  const topologyEdges = detail?.graph_topology?.edges || [];
  if ((includeContext || includeGraph) && topologyEdges.length) {
    const pathEdgeKeys = new Set(currentPathEdges.map((edge) => `${edge.caller || edge.source}->${edge.callee || edge.target}`));
    return topologyEdges.map((edge) => {
      const normalized = normalizeGraphEdge(edge, "context");
      const key = `${normalized.caller}->${normalized.callee}`;
      return { ...normalized, edge_type: pathEdgeKeys.has(key) ? "path" : "context" };
    });
  }
  const edges = includeContext ? [...contextEdges, ...currentPathEdges] : currentPathEdges;
  const projected = [...graphContextNodes(detail), ...pathNodes(detail)];
  if (projected.length) return edges.map((edge) => normalizeGraphEdge(edge, edge.edge_type === "context" ? "context" : "path"));
  return topologyEdges.map((edge) => normalizeGraphEdge(edge, "path"));
}

function graphGroupKey(node) {
  const distance = nodeDistance(node);
  const filePath = node?.file_path || "";
  const start = node?.start_line ?? "";
  const end = node?.end_line ?? "";
  const role = canonicalNodeRole(node?.node_role) || node?.group || "";
  if (!filePath || start === "" || end === "") return node?.key || "";
  return `${filePath}:${start}:${end}:${role}:${distance ?? ""}`;
}

function compactMemberLabel(keys) {
  const names = keys.map((key) => String(key || "").split("::").slice(-1)[0]);
  const methodNames = new Set(names.map((name) => (name.includes(".") ? name.split(".").slice(-1)[0] : name)));
  if (methodNames.size === 1) return `${[...methodNames][0]} x${keys.length}`;
  return `${names[0]} +${keys.length - 1}`;
}

function aggregateGraph(nodes, edges) {
  const groups = new Map();
  nodes.forEach((node) => {
    const key = graphGroupKey(node);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(node);
  });
  const nodeKeyMap = new Map();
  const aggregateNodes = [];
  groups.forEach((members, groupKey) => {
    if (members.length === 1) {
      aggregateNodes.push(members[0]);
      nodeKeyMap.set(members[0].key, members[0].key);
      return;
    }
    const first = members[0];
    const memberKeys = members.map((member) => member.key);
    const hitSteps = members
      .map((member) => member.first_step)
      .filter((step) => step !== null && step !== undefined)
      .map((step) => Number(step))
      .filter((step) => Number.isFinite(step));
    const aggregateKey = `group::${groupKey}`;
    memberKeys.forEach((key) => nodeKeyMap.set(key, aggregateKey));
    aggregateNodes.push({
      ...first,
      key: aggregateKey,
      label: compactMemberLabel(memberKeys),
      member_keys: memberKeys,
      member_count: members.length,
      source: first.source || members.find((member) => member.source)?.source,
      source_preview: first.source_preview || members.find((member) => member.source_preview)?.source_preview,
      hit: members.some((member) => member.hit || member.first_step !== null && member.first_step !== undefined),
      first_step: hitSteps.length ? Math.min(...hitSteps) : null,
      selected_issue_anchor: members.some((member) => member.selected_issue_anchor),
      root_cause: members.some((member) => member.root_cause),
      patched_callable: members.some((member) => member.patched_callable),
      patch_role: members.find((member) => member.patch_role)?.patch_role,
    });
  });
  const aggregateEdgeByKey = new Map();
  edges.forEach((edge) => {
    const caller = edge.caller || edge.source;
    const callee = edge.callee || edge.target;
    const source = nodeKeyMap.get(caller);
    const target = nodeKeyMap.get(callee);
    if (!source || !target) return;
    if (source === target) return;
    const key = `${source}->${target}`;
    const aggregateEdge = { ...edge, caller: source, callee: target, source, target };
    const existing = aggregateEdgeByKey.get(key);
    if (!existing || existing.edge_type !== "path" && aggregateEdge.edge_type === "path") {
      aggregateEdgeByKey.set(key, aggregateEdge);
    }
  });
  return { nodes: aggregateNodes, edges: [...aggregateEdgeByKey.values()], nodeKeyMap };
}

function graphModel(detail, options = {}) {
  return aggregateGraph(graphNodes(detail, options), graphEdges(detail, options));
}

function graphEdgeBucket(edge) {
  if (edge.edge_type === "trace") return "trace";
  if (edge.edge_type === "agent") return "trace";
  if (edge.edge_type === "context") return "graph";
  return "path";
}

function graphEdgeVisible(edge) {
  return state.graphEdgeFilters[graphEdgeBucket(edge)] !== false;
}

function stepHitNodeKeys(step) {
  const scored = step?.scored || step || {};
  return (scored.hit_nodes || [])
    .map((node) => node?.key)
    .filter((key) => typeof key === "string" && key);
}

function visibleStepHitKeys(step, toVisibleKey) {
  const seen = new Set();
  const keys = [];
  stepHitNodeKeys(step).forEach((rawKey) => {
    const key = toVisibleKey(rawKey);
    if (!key || seen.has(key)) return;
    seen.add(key);
    keys.push(key);
  });
  return keys;
}

function firstHitNodesFromProjection(detail) {
  const nodes = [...graphContextNodes(detail), ...pathNodes(detail)];
  return nodes
    .filter((node) => node?.first_step !== null && node?.first_step !== undefined)
    .sort((a, b) => Number(a.first_step) - Number(b.first_step))
    .map((node) => ({ key: node.key, first_step: node.first_step }))
    .filter((node) => typeof node.key === "string" && node.key);
}

function traceEdges(detail, model) {
  const visible = new Set(model.nodes.map((node) => node.key));
  const toVisibleKey = (rawKey) => {
    const mapped = model.nodeKeyMap?.get(rawKey) || rawKey;
    return visible.has(mapped) ? mapped : null;
  };
  const rawSteps = (detail.step_inspection || []).length ? detail.step_inspection : detail.step_details || [];
  const steps = [...rawSteps]
    .sort((a, b) => Number(a.trace_index ?? a.step_index ?? 0) - Number(b.trace_index ?? b.step_index ?? 0));
  const edgeByKey = new Map();
  const addEdge = (source, target, firstStep) => {
    if (!source || !target || source === target) return;
    const key = `${source}->${target}`;
    const current = edgeByKey.get(key) || { source, target, caller: source, callee: target, edge_type: "trace", first_step: firstStep, count: 0 };
    current.count += 1;
    edgeByKey.set(key, current);
  };
  const addStepEdges = (sourceStep, targetStep) => {
    if (!sourceStep || !targetStep) return;
    const sourceMulti = sourceStep.keys.length > 1;
    const targetMulti = targetStep.keys.length > 1;
    if (sourceMulti && targetMulti) return;
    if (sourceMulti && targetStep.keys.every((target) => sourceStep.keys.includes(target))) return;
    sourceStep.keys.forEach((source) => {
      targetStep.keys.forEach((target) => addEdge(source, target, targetStep.step_label));
    });
  };
  let previous = null;
  let sawStepHit = false;
  steps.forEach((step, index) => {
    const keys = visibleStepHitKeys(step, toVisibleKey);
    if (!keys.length) return;
    sawStepHit = true;
    const traceIndex = Number(step?.trace_index ?? index);
    const stepLabel = displayStepLabel(step, detail, Number.isFinite(traceIndex) ? traceIndex : index);
    const current = { keys, step_label: stepLabel };
    addStepEdges(previous, current);
    previous = current;
  });
  if (!sawStepHit) {
    let fallbackPrevious = null;
    const byStep = new Map();
    firstHitNodesFromProjection(detail).forEach((node) => {
      const key = toVisibleKey(node.key);
      if (!key) return;
      const label = node.first_step;
      if (!byStep.has(label)) byStep.set(label, new Set());
      byStep.get(label).add(key);
    });
    [...byStep.entries()].sort((a, b) => Number(a[0]) - Number(b[0])).forEach(([stepLabel, keys]) => {
      const current = { keys: [...keys], step_label: stepLabel };
      addStepEdges(fallbackPrevious, current);
      fallbackPrevious = current;
    });
  }
  return [...edgeByKey.values()];
}

function isSymptomRootCauseNode(node) {
  return Boolean(node?.selected_issue_anchor && (node?.root_cause || nodeIsFaultSidePatch(node)));
}

function nodeRoleLabel(node) {
  if (isSymptomRootCauseNode(node)) return "symptom + root cause";
  const role = canonicalNodeRole(node?.node_role);
  return NODE_ROLE_LABELS[role] || role || node?.group || "-";
}

function graphEdgeMarker(edge) {
  if (edge.edge_type === "trace" || edge.edge_type === "agent") return "graph-arrow-trace";
  return edge.edge_type === "context" ? "graph-arrow-context" : "graph-arrow-path";
}

const GRAPH_NODE_RADIUS = 18;
const GRAPH_EDGE_NODE_CLEARANCE = GRAPH_NODE_RADIUS + 8;

function pointToSegmentDistance(point, a, b) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const lenSq = dx * dx + dy * dy;
  if (!lenSq) return Math.sqrt((point.x - a.x) ** 2 + (point.y - a.y) ** 2);
  const t = Math.max(0, Math.min(1, ((point.x - a.x) * dx + (point.y - a.y) * dy) / lenSq));
  const x = a.x + t * dx;
  const y = a.y + t * dy;
  return Math.sqrt((point.x - x) ** 2 + (point.y - y) ** 2);
}

function graphEdgeRouteOffset(edge, a, b, nodes, positions) {
  const caller = edge.caller || edge.source;
  const callee = edge.callee || edge.target;
  const obstacles = nodes.filter((node) => {
    if (!node?.key || node.key === caller || node.key === callee) return false;
    const pos = positions.get(node.key);
    if (!pos) return false;
    return pointToSegmentDistance(pos, a, b) <= GRAPH_EDGE_NODE_CLEARANCE;
  });
  if (!obstacles.length) return 0;
  const avgY = (a.y + b.y) / 2;
  const sign = avgY < 120 ? 1 : -1;
  return sign * (58 + Math.min(obstacles.length, 3) * 18);
}

function graphEdgePath(a, b, edgeType = "path", routeOffset = 0) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const len = Math.sqrt(dx * dx + dy * dy) || 1;
  const startPad = 22;
  const endPad = 24;
  const perpendicular = edgeType === "trace" ? 12 : 0;
  const px = -dy / len * perpendicular;
  const py = dx / len * perpendicular;
  let routeNormalX = -dy / len;
  let routeNormalY = dx / len;
  if (routeNormalY < 0 || Math.abs(routeNormalY) < 1e-9 && routeNormalX < 0) {
    routeNormalX *= -1;
    routeNormalY *= -1;
  }
  const routeX = routeNormalX * routeOffset;
  const routeY = routeNormalY * routeOffset;
  const sx = a.x + dx / len * startPad + px;
  const sy = a.y + dy / len * startPad + py;
  const tx = b.x - dx / len * endPad + px;
  const ty = b.y - dy / len * endPad + py;
  const direction = dx >= 0 ? 1 : -1;
  const bendFactor = edgeType === "trace" ? 0.58 : 0.45;
  const bend = Math.max(60, Math.abs(dx) * bendFactor);
  const sameLayerBend = Math.abs(dx) < 12 ? 90 : bend;
  const c1x = sx + direction * sameLayerBend + routeX;
  const c1y = sy + routeY;
  const c2x = tx - direction * sameLayerBend + routeX;
  const c2y = ty + routeY;
  return `M${sx.toFixed(1)} ${sy.toFixed(1)} C${c1x.toFixed(1)} ${c1y.toFixed(1)}, ${c2x.toFixed(1)} ${c2y.toFixed(1)}, ${tx.toFixed(1)} ${ty.toFixed(1)}`;
}

function graphEdgeLabelPosition(a, b) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 - 12 };
}

function detectCodeLanguage(filePath, source = "") {
  const path = String(filePath || "").toLowerCase();
  const ext = path.includes(".") ? path.split(".").pop() : "";
  const byExt = {
    py: "python",
    js: "javascript",
    jsx: "javascript",
    ts: "typescript",
    tsx: "typescript",
    json: "json",
    yaml: "yaml",
    yml: "yaml",
    sh: "shell",
    bash: "shell",
    sql: "sql",
    diff: "diff",
    patch: "diff",
    md: "markdown",
  };
  if (byExt[ext]) return byExt[ext];
  const text = String(source || "");
  if (/^\s*(def|class|import|from)\s+/m.test(text)) return "python";
  if (/^\s*(const|let|function|import|export)\s+/m.test(text)) return "javascript";
  if (/^\s*[{[]/.test(text)) return "json";
  if (/^\s*(select|with|insert|update)\s+/im.test(text)) return "sql";
  return "text";
}

function highlightCode(source, language) {
  const text = String(source || "");
  if (!text) return "";
  const span = (className, value) => `<span class="${className}">${esc(value)}</span>`;
  const highlightPath = (value, className) => {
    let out = "";
    let last = 0;
    String(value).replace(/[A-Za-z_$][\w$]*|\./g, (match, offset) => {
      out += esc(value.slice(last, offset));
      out += match === "." ? span("tok-dot", match) : span(className, match);
      last = offset + match.length;
      return match;
    });
    return out + esc(value.slice(last));
  };
  const emit = (regex, render) => {
    let out = "";
    let last = 0;
    text.replace(regex, (match, ...args) => {
      const offset = args[args.length - 2];
      out += esc(text.slice(last, offset));
      out += render(match);
      last = offset + match.length;
      return match;
    });
    return out + esc(text.slice(last));
  };
  if (language === "json") {
    return emit(/"(?:\\.|[^"\\])*"(?=\s*:)|"(?:\\.|[^"\\])*"|\b(?:true|false|null)\b|-?\b\d+(?:\.\d+)?\b/g, (match) => {
      if (/^".*"(?=\s*:)/.test(match)) return span("tok-key", match);
      if (/^"/.test(match)) return span("tok-string", match);
      if (/^-?\d/.test(match)) return span("tok-number", match);
      return span("tok-keyword", match);
    });
  }
  const commonKeywords = {
    python: "def|class|return|if|elif|else|for|while|try|except|finally|with|as|import|from|pass|raise|yield|async|await|lambda|True|False|None",
    javascript: "function|return|if|else|for|while|try|catch|finally|const|let|var|class|import|export|from|async|await|new|true|false|null|undefined",
    typescript: "function|return|if|else|for|while|try|catch|finally|const|let|var|class|interface|type|import|export|from|async|await|new|true|false|null|undefined",
    shell: "if|then|else|fi|for|do|done|case|esac|function|export|local|return|exit",
    sql: "select|from|where|join|left|right|inner|outer|on|group|by|order|insert|update|delete|create|table|with|as|and|or|null|is",
  };
  const keywords = commonKeywords[language] || "";
  if (!keywords) return esc(text);
  const string = String.raw`"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|` + "`" + String.raw`(?:\\.|[^` + "`" + String.raw`\\])*` + "`";
  const comment = language === "python" || language === "shell"
    ? String.raw`#[^\n]*`
    : language === "sql"
      ? String.raw`--[^\n]*|/\*[\s\S]*?\*/`
      : String.raw`//[^\n]*|/\*[\s\S]*?\*/`;
  const lhs = String.raw`\b[A-Za-z_$][\w$]*(?:\s*\.\s*[A-Za-z_$][\w$]*)*(?=\s*(?://=|<<=|>>=|[-+*/%&|^:]?=(?!=)))`;
  const member = String.raw`\.[A-Za-z_$][\w$]*`;
  const defName = language === "python"
    ? String.raw`\b(?:def|class)\s+[A-Za-z_]\w*`
    : language === "javascript" || language === "typescript"
      ? String.raw`\b(?:function|class)\s+[A-Za-z_$][\w$]*`
      : "";
  const operator = String.raw`==|!=|<=|>=|->|=>|//=|<<=|>>=|[-+*/%=<>:.,]`;
  const parts = [comment, string];
  if (defName) parts.push(defName);
  parts.push(lhs, member, `\\b(?:${keywords})\\b`, String.raw`-?\b\d+(?:\.\d+)?\b`, operator);
  const regex = new RegExp(parts.filter(Boolean).join("|"), "gim");
  const keywordOnly = new RegExp(`^(?:${keywords})$`, "i");
  return emit(regex, (match) => {
    if (/^(#|\/\/|--|\/\*)/.test(match)) return span("tok-comment", match);
    if (/^["'`]/.test(match)) return span("tok-string", match);
    if (/^(def|class|function)\s+/i.test(match)) {
      const pieces = match.match(/^(\S+)(\s+)(\S+)$/);
      return pieces ? `${span("tok-keyword", pieces[1])}${esc(pieces[2])}${span("tok-symbol", pieces[3])}` : span("tok-keyword", match);
    }
    if (/^-?\d/.test(match)) return span("tok-number", match);
    if (keywordOnly.test(match)) return span("tok-keyword", match);
    if (/^\.[A-Za-z_$]/.test(match)) return `${span("tok-dot", ".")}${span("tok-property", match.slice(1))}`;
    if (/[A-Za-z_$]/.test(match) && /(?:[A-Za-z_$][\w$]|\.)/.test(match) && !/[=<>:+\-*/%,]/.test(match)) {
      return highlightPath(match, "tok-lhs");
    }
    if (match === ".") return span("tok-dot", match);
    return span("tok-operator", match);
  });
}

function safeLinkHref(value) {
  const href = String(value || "").trim();
  return /^(https?:\/\/|#)/i.test(href) ? href : "";
}

function renderInlineMarkdown(text) {
  const placeholders = [];
  let escaped = esc(String(text || "")).replace(/`([^`]+)`/g, (_match, code) => {
    const token = `\u0000${placeholders.length}\u0000`;
    placeholders.push(`<code>${code}</code>`);
    return token;
  });
  escaped = escaped.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_match, label, href) => {
    const safeHref = safeLinkHref(href);
    return safeHref ? `<a href="${esc(safeHref)}" target="_blank" rel="noreferrer">${label}</a>` : label;
  });
  escaped = escaped
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>");
  placeholders.forEach((value, index) => {
    escaped = escaped.replaceAll(`\u0000${index}\u0000`, value);
  });
  return escaped;
}

function renderMarkdown(text) {
  const source = String(text || "").replace(/\r\n/g, "\n");
  if (!source.trim()) return '<div class="empty">No issue description was captured.</div>';
  const lines = source.split("\n");
  const out = [];
  let paragraph = [];
  let listItems = [];
  let inFence = false;
  let fenceLanguage = "";
  let fenceLines = [];
  const flushParagraph = () => {
    if (!paragraph.length) return;
    out.push(`<p>${renderInlineMarkdown(paragraph.join(" "))}</p>`);
    paragraph = [];
  };
  const flushList = () => {
    if (!listItems.length) return;
    out.push(`<ul>${listItems.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</ul>`);
    listItems = [];
  };
  const flushFence = () => {
    const language = fenceLanguage || detectCodeLanguage("", fenceLines.join("\n"));
    out.push(`<pre class="code-view language-${esc(language)}"><code>${highlightCode(fenceLines.join("\n"), language)}</code></pre>`);
    fenceLanguage = "";
    fenceLines = [];
  };
  lines.forEach((line) => {
    const fence = line.match(/^\s*```\s*([A-Za-z0-9_+-]*)\s*$/);
    if (fence) {
      if (inFence) {
        flushFence();
        inFence = false;
      } else {
        flushParagraph();
        flushList();
        inFence = true;
        fenceLanguage = fence[1] || "";
      }
      return;
    }
    if (inFence) {
      fenceLines.push(line);
      return;
    }
    if (!line.trim()) {
      flushParagraph();
      flushList();
      return;
    }
    const heading = line.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      flushParagraph();
      flushList();
      const level = Math.min(heading[1].length + 1, 5);
      out.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
      return;
    }
    const bullet = line.match(/^\s*[-*]\s+(.+)$/);
    if (bullet) {
      flushParagraph();
      listItems.push(bullet[1]);
      return;
    }
    const quote = line.match(/^\s*>\s?(.+)$/);
    if (quote) {
      flushParagraph();
      flushList();
      out.push(`<blockquote>${renderInlineMarkdown(quote[1])}</blockquote>`);
      return;
    }
    flushList();
    paragraph.push(line.trim());
  });
  if (inFence) flushFence();
  flushParagraph();
  flushList();
  return `<div class="markdown-body">${out.join("")}</div>`;
}

function patchLineClass(line) {
  if (/^(diff --git|index |new file mode |deleted file mode |similarity index |rename from |rename to )/.test(line)) return "diff-meta";
  if (/^@@/.test(line)) return "diff-hunk";
  if (/^\+\+\+|^---/.test(line)) return "diff-file";
  if (/^\+/.test(line)) return "diff-add";
  if (/^-/.test(line)) return "diff-del";
  return "diff-ctx";
}

function renderPatch(text) {
  const source = String(text || "").replace(/\r\n/g, "\n");
  if (!source.trim()) return '<div class="empty">No golden patch was captured.</div>';
  const rows = source.split("\n").map((line) => {
    const cls = patchLineClass(line);
    const sign = cls === "diff-add" ? "+" : cls === "diff-del" ? "-" : cls === "diff-hunk" ? "@" : " ";
    const body = cls === "diff-add" || cls === "diff-del" ? line.slice(1) : line;
    return `<div class="diff-line ${cls}"><span>${esc(sign)}</span><code>${esc(body)}</code></div>`;
  });
  return `<div class="diff-view patch-view">${rows.join("")}</div>`;
}

function nodeSourcePayload(node) {
  const full = node?.source;
  if (typeof full === "string" && full) return { text: full, truncated: false };
  const preview = node?.source_preview || "";
  return { text: preview, truncated: typeof preview === "string" && preview.trimEnd().endsWith("...") };
}

function renderGraphSourcePanel(model) {
  const selected = model.nodes.find((node) => node.key === state.selectedGraphNodeKey);
  if (!selected) {
    return "";
  }
  const sourcePayload = nodeSourcePayload(selected);
  const source = sourcePayload.text;
  const language = detectCodeLanguage(selected.file_path, source);
  const location = selected.file_path
    ? `${selected.file_path}${selected.start_line ? `:${selected.start_line}${selected.end_line ? `-${selected.end_line}` : ""}` : ""}`
    : "-";
  const members = selected.member_keys?.length
    ? `<div class="node-source-members">${selected.member_keys.map((key) => `<code>${esc(key)}</code>`).join("")}</div>`
    : "";
  return `<aside class="graph-source-panel">
    <h4>Node Source</h4>
    <div class="node-source-meta"><strong>${esc(selected.label || selected.key)}</strong><span>${esc(nodeRoleLabel(selected))}</span><span>${esc(location)}</span><span>${esc(language)}</span></div>
    ${sourcePayload.truncated ? '<div class="node-source-warning">Full source was unavailable from the Graph artifact; showing the stored preview.</div>' : ""}
    ${members}
    ${source ? `<div class="node-source-code"><pre class="code-view language-${esc(language)}"><code>${highlightCode(source, language)}</code></pre></div>` : '<div class="empty">No source was captured for this node.</div>'}
  </aside>`;
}

function renderRolloutSelector(detail, snapshot) {
  const rollouts = detailsForInstance(filteredDetails(snapshot), traceInstanceKey(detail));
  if (rollouts.length <= 1) {
    return `<div class="rollout-single">Rollout ${esc(rolloutIndex(detail) + 1)} · ${esc(traceRolloutOutcome(detail))}</div>`;
  }
  const successCount = rollouts.filter(traceResolved).length;
  const options = rollouts.map((item) => {
    const label = `Rollout ${rolloutIndex(item) + 1} · ${traceRolloutOutcome(item)}`;
    return `<option value="${esc(rowKey(item))}" ${rowKey(item) === rowKey(detail) ? "selected" : ""}>${esc(label)}</option>`;
  }).join("");
  return `<label class="rollout-select-control">
    <span>Rollout</span>
    <select id="trace-rollout-select">${options}</select>
    <strong>${esc(successCount)}/${esc(rollouts.length)} success</strong>
  </label>`;
}

function renderTraceTitleCard(detail, snapshot) {
  const issue = detail.issue_description || "";
  const patch = detail.golden_patch || "";
  return `<div class="trace-overview-stack">
    <div class="trace-title-card">
      <div>
        <strong>${esc(detail.instance_id || `record-${detail.record_index}`)}</strong>
        <div class="run-meta">${esc(detail.model_label || "-")} · ${esc(detail.run_id || "-")}</div>
      </div>
      <div class="trace-title-actions">
        <button class="copy-link" type="button" data-copy-link="experiment">Copy experiment URL</button>
        <button class="copy-link" type="button" data-copy-link="instance">Copy instance URL</button>
        <button class="copy-link" type="button" data-copy-link="rollout">Copy rollout URL</button>
        ${renderRolloutSelector(detail, snapshot)}
        ${traceStatusIcons(detail)}
      </div>
    </div>
    <details class="instance-overview-toggle">
      <summary>Instance overview</summary>
      <div class="instance-overview-body">
        <section class="instance-issue-section">
          <h4>Issue description</h4>
          ${renderMarkdown(issue)}
        </section>
        <section>
          <h4>Golden patch</h4>
          ${renderPatch(patch)}
        </section>
      </div>
    </details>
  </div>`;
}

function renderGraph(detail) {
  const projection = pathProjection(detail);
  const contextNodeCount = (projection.context_nodes || []).length;
  const contextEdgeCount = (projection.context_edges || []).length;
  const model = graphModel(detail);
  const nodes = model.nodes;
  if (!nodes.length) return '<div class="empty">No Graph available for this instance.</div>';
  const fixedEdges = model.edges;
  const edges = fixedEdges.filter(graphEdgeVisible);
  const layers = new Map();
  nodes.forEach((node) => {
    const layer = graphLayer(node);
    if (!layers.has(layer)) layers.set(layer, []);
    layers.get(layer).push(node);
  });
  const sortedLayers = [...layers.entries()].sort((a, b) => a[0] - b[0]);
  const maxLayerSize = Math.max(...sortedLayers.map((entry) => entry[1].length), 1);
  const rowGap = 82;
  const colGap = 230;
  const width = Math.max(980, 220 + sortedLayers.length * colGap);
  const height = Math.max(360, 110 + maxLayerSize * rowGap);
  const positions = new Map();
  sortedLayers.forEach(([_layer, layerNodes], layerIndex) => {
    const x = 80 + layerIndex * colGap;
    const sortedNodes = [...layerNodes].sort(compareGraphNodes);
    sortedNodes.forEach((node, index) => positions.set(node.key, { x, y: 55 + index * rowGap }));
  });
  const sortedEdges = [...edges].sort((a, b) => String(a.edge_type || "").localeCompare(String(b.edge_type || "")));
  const edgeSvg = sortedEdges.map((edge) => {
    const caller = edge.caller || edge.source;
    const callee = edge.callee || edge.target;
    const a = positions.get(caller);
    const b = positions.get(callee);
    if (!a || !b) return "";
    const edgeType = edge.edge_type === "context" ? "context" : "path";
    const routeOffset = graphEdgeRouteOffset(edge, a, b, nodes, positions);
    return `<path class="graph-edge ${esc(edgeType)}" d="${graphEdgePath(a, b, edgeType, routeOffset)}" marker-end="url(#${graphEdgeMarker(edge)})"><title>${esc(caller)} -> ${esc(callee)} (${esc(edgeType)})</title></path>`;
  }).join("");
  const allTraversalEdges = traceEdges(detail, model);
  const traversalEdges = state.graphEdgeFilters.trace ? allTraversalEdges : [];
  const traversalSvg = traversalEdges.map((edge) => {
    const a = positions.get(edge.source);
    const b = positions.get(edge.target);
    if (!a || !b) return "";
    const label = graphEdgeLabelPosition(a, b);
    const stepLabel = edge.count > 1 ? `${edge.first_step}x${edge.count}` : String(edge.first_step);
    return `<path class="graph-edge trace graph-trace-edge" d="${graphEdgePath(a, b, "trace")}" marker-end="url(#graph-arrow-trace)"><title>Trace edge ${esc(edge.source)} -> ${esc(edge.target)} first seen at step ${esc(edge.first_step)}${edge.count > 1 ? ` repeated ${esc(edge.count)} times` : ""}</title></path><text class="graph-trace-label" x="${label.x.toFixed(1)}" y="${label.y.toFixed(1)}">${esc(stepLabel)}</text>`;
  }).join("");
  const nodeSvg = nodes.map((node) => {
    const pos = positions.get(node.key);
    if (!pos) return "";
    const tone = roleTone(node);
    const hit = node.hit || node.first_step !== null && node.first_step !== undefined;
    const distance = nodeDistance(node);
    const roleLabel = nodeRoleLabel(node);
    const label = String(node.label || node.key || "").split("::").slice(-1)[0].slice(0, 30);
    const members = node.member_keys ? ` members: ${node.member_keys.slice(0, 8).join(", ")}${node.member_keys.length > 8 ? ", ..." : ""}` : "";
    const selected = node.key === state.selectedGraphNodeKey;
    return `<g class="graph-node ${tone} ${hit ? "hit" : "miss"} ${node.final_edit ? "edited" : ""} ${selected ? "is-selected" : ""}" transform="translate(${pos.x},${pos.y})" data-node-key="${esc(node.key)}" role="button" tabindex="0" aria-label="${esc(`Inspect ${node.label || node.key}`)}">
      <circle r="18"></circle>
      ${node.final_edit ? '<circle class="graph-edit-ring" r="22"></circle>' : ""}
      ${hit ? `<text class="graph-step" y="4">${esc(node.first_step)}</text>` : ""}
      <text class="graph-label" x="26" y="-2">${esc(label)}</text>
      <text class="graph-sub" x="26" y="13">${esc(roleLabel)}${node.member_count ? ` · x${esc(node.member_count)}` : ""}</text>
      ${isSymptomRootCauseNode(node) ? '<rect class="graph-dual-pill" x="-24" y="20" width="48" height="14" rx="4"></rect><text class="graph-dual-text" y="31">S+RC</text>' : ""}
      <title>${esc(node.member_count ? node.label : node.key)} role ${esc(roleLabel)} ${distance === null ? "" : `distance ${fmt(distance)}`} ${node.first_step !== null && node.first_step !== undefined ? `first step ${esc(node.first_step)}` : "not visited"}${esc(members)}</title>
    </g>`;
  }).join("");
  const hiddenFixedEdgeCount = fixedEdges.length - edges.length;
  const hiddenTraversalEdgeCount = allTraversalEdges.length - traversalEdges.length;
  const edgeFilterSummary = [
    state.graphEdgeFilters.path ? "Path edges" : null,
    state.graphEdgeFilters.graph ? "Graph edges" : null,
    state.graphEdgeFilters.trace ? "Trace edges" : null,
  ].filter(Boolean).join(", ") || "none";
  const scope = state.showGraphContext
    ? `Full Graph: ${nodes.length} visual nodes, ${edges.length}/${fixedEdges.length} fixed edges, ${traversalEdges.length}/${allTraversalEdges.length} Trace edges.`
    : `Core Path: ${nodes.length} visual nodes, ${edges.length}/${fixedEdges.length} fixed edges, ${traversalEdges.length}/${allTraversalEdges.length} Trace edges. ${contextNodeCount} context/harness nodes and ${contextEdgeCount} context edges are hidden.`;
  const hiddenNote = hiddenFixedEdgeCount || hiddenTraversalEdgeCount
    ? ` Hidden by arrow filter: ${hiddenFixedEdgeCount} fixed, ${hiddenTraversalEdgeCount} Trace.`
    : "";
  const note = edges.length || traversalEdges.length
    ? `${scope} Showing edges: ${edgeFilterSummary}. Blue edges are Path edges; gray edges are Graph edges outside the Path; orange dashed edges are Trace edges between adjacent Graph-hit steps when at least one side is single-hit. Multi-hit read steps do not create internal edges, and multi-hit to multi-hit transitions are omitted. Trace labels show first step; mxn means first seen at step m and repeated n times.${hiddenNote}`
    : `${scope} Showing edges: ${edgeFilterSummary}. No edges are visible under the current edge filter.`;
  const hasSelectedSource = model.nodes.some((node) => node.key === state.selectedGraphNodeKey);
  return `<section class="graph-panel" aria-label="Graph">
    <div class="graph-head">
      <div class="graph-controls">
        ${contextNodeCount || contextEdgeCount || (detail?.graph_topology?.nodes || []).length ? `<label class="graph-toggle"><input id="graph-context-toggle" type="checkbox" ${state.showGraphContext ? "checked" : ""}> Show full Graph</label>` : ""}
        <fieldset class="graph-edge-filter" aria-label="Graph edge filter">
          <legend>Edges</legend>
          <label><input class="graph-edge-filter-checkbox" type="checkbox" data-edge-filter="path" ${state.graphEdgeFilters.path ? "checked" : ""}> Path edges</label>
          <label><input class="graph-edge-filter-checkbox" type="checkbox" data-edge-filter="graph" ${state.graphEdgeFilters.graph ? "checked" : ""}> Graph edges</label>
          <label><input class="graph-edge-filter-checkbox" type="checkbox" data-edge-filter="trace" ${state.graphEdgeFilters.trace ? "checked" : ""}> Trace edges</label>
        </fieldset>
      </div>
    </div>
    <div class="graph-note">${esc(note)}</div>
    <div class="graph-content ${hasSelectedSource ? "has-source" : ""}">
      <div class="graph-wrap"><svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-label="Bonus map graph">
        <defs>
          <linearGradient id="graph-symptom-root-cause-fill" x1="0" y1="1" x2="1" y2="0">
            <stop offset="50%" stop-color="#dcfce7"></stop>
            <stop offset="50%" stop-color="#fee2e2"></stop>
          </linearGradient>
          <marker id="graph-arrow-path" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z"></path></marker>
          <marker id="graph-arrow-context" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z"></path></marker>
          <marker id="graph-arrow-trace" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z"></path></marker>
        </defs>
        ${edgeSvg}${traversalSvg}${nodeSvg}
      </svg></div>
      ${renderGraphSourcePanel(model)}
    </div>
  </section>`;
}

function traceResolved(detail) {
  if (detail?.resolved !== null && detail?.resolved !== undefined) return Boolean(detail.resolved);
  const reward = numeric(detail?.reward);
  return reward !== null ? reward > 0 : false;
}

function traceStatusItems(detail) {
  const canShowOrderMetrics = isOrderMetricDetail(detail);
  const hasLoopBlock = (detail.purpose_blocks || []).some((block) => block?.loop);
  const editedRootCause = detail.edited_root_cause === true || (detail.step_inspection || []).some((step) => step?.edited_root_cause);
  const orderScore = numeric(detail.order_score);
  return [
    { key: "symptom", label: "hit symptom", icon: "🔎", active: detail.anchor_hit },
    { key: "root", label: "hit root cause", icon: "🎯", active: detail.root_hit },
    { key: "root-edit", label: "edited root cause", icon: "✎", active: editedRootCause },
    { key: "loop", label: "loop", icon: "🔁", active: hasLoopBlock },
    { key: "miracle", label: "miracle", icon: "✨", active: canShowOrderMetrics && detail.miracle_step === true },
    { key: "reverse", label: "reverse", icon: "🌀", active: canShowOrderMetrics && orderScore !== null && orderScore < 0 },
  ];
}

function traceStatusIcons(detail) {
  const icons = traceStatusItems(detail).filter((item) => item.active);
  if (!icons.length) return '<span class="trace-status-empty">-</span>';
  const label = icons.map((item) => item.label).join(", ");
  return `<span class="trace-status-icons" aria-label="${esc(label)}">${icons.map((item) => `<span class="trace-status-icon trace-icon-${esc(item.key)}" title="${esc(item.label)}">${item.icon}</span>`).join("")}</span>`;
}

function traceSummary(detail) {
  const steps = (detail.step_inspection || detail.step_details || []).length;
  const blocks = (detail.purpose_blocks || []).length;
  const nPathNodes = pathValue(detail, "n_path_nodes", "n_chain_nodes", 0);
  const nHitPathNodes = pathValue(detail, "n_hit_path_nodes", "n_hit_chain_nodes", 0);
  const pathHits = nPathNodes ? ` · Path ${fmt(nHitPathNodes)}/${fmt(nPathNodes)}` : "";
  return `${fmt(steps)} steps · ${fmt(blocks)} blocks${pathHits}`;
}

function traceRolloutOutcome(detail) {
  return traceResolved(detail) ? "success" : "failed";
}

function selectedRolloutForGroup(details) {
  return detailForRolloutIndex(details);
}

function traceRolloutSegments(details) {
  const selectedRollout = selectedRolloutForGroup(details);
  const segments = details.map((detail) => {
    const resolved = traceResolved(detail);
    const selected = selectedRollout && rowKey(detail) === rowKey(selectedRollout);
    const label = `Rollout ${rolloutIndex(detail) + 1}: ${traceRolloutOutcome(detail)}`;
    return `<span class="trace-rollout-segment ${resolved ? "is-resolved" : "is-unresolved"} ${selected ? "is-selected" : ""}" title="${esc(label)}" aria-label="${esc(label)}"></span>`;
  }).join("");
  return `<span class="trace-rollout-strip" aria-hidden="true">${segments}</span>`;
}

function groupedTraceDetails(snapshot) {
  const groups = new Map();
  filteredDetails(snapshot).sort(compareTraceDetails).forEach((detail) => {
    const key = traceInstanceKey(detail);
    if (!groups.has(key)) groups.set(key, { key, details: [] });
    groups.get(key).details.push(detail);
  });
  return [...groups.values()];
}

function selectableRolloutDetails(snapshot) {
  if (!state.selectedEvalCellKey) return [];
  return activeDetails(snapshot).filter((detail) => detailCellKey(detail) === state.selectedEvalCellKey);
}

function renderGlobalRolloutSelector(snapshot) {
  const details = selectableRolloutDetails(snapshot);
  const maxIndex = details.reduce((maxValue, detail) => Math.max(maxValue, rolloutIndex(detail)), 0);
  if (maxIndex <= 0) return "";
  const current = Math.min(traceGlobalRolloutIndex(), maxIndex);
  const options = Array.from({ length: maxIndex + 1 }, (_, index) => {
    return `<option value="${esc(index)}" ${index === current ? "selected" : ""}>Rollout ${esc(index + 1)}</option>`;
  }).join("");
  return `<label class="trace-global-rollout">
    <span>Rollout view</span>
    <select id="trace-global-rollout-select">${options}</select>
  </label>`;
}

function handleTraceGlobalRolloutChange(event) {
  const scrollState = captureInspectorScroll();
  const current = selectedDetail(state.snapshot);
  const instanceKey = current ? traceInstanceKey(current) : null;
  setTraceGlobalRolloutIndex(event.target.value);
  const details = filteredDetails(state.snapshot);
  const nextDetail = instanceKey
    ? detailForRolloutIndex(detailsForInstance(details, instanceKey))
    : detailForRolloutIndex(details);
  if (nextDetail) state.selectedTraceKey = rowKey(nextDetail);
  state.selectedStepIndex = 0;
  state.selectedGraphNodeKey = null;
  resetTracePanels();
  renderTraceInspector(state.snapshot);
  restoreInspectorScroll({ ...scrollState, middle: 0, right: 0 });
}

function renderTraceGlobalRolloutControl(snapshot) {
  const el = document.getElementById("trace-global-rollout-slot");
  if (!el) return;
  el.innerHTML = renderGlobalRolloutSelector(snapshot);
  document.getElementById("trace-global-rollout-select")?.addEventListener("change", handleTraceGlobalRolloutChange);
}

function renderTraceList(snapshot) {
  const rows = groupedTraceDetails(snapshot).map((group) => {
    const selectedDetail = selectedRolloutForGroup(group.details);
    const selected = group.details.some((detail) => rowKey(detail) === state.selectedTraceKey);
    const resolvedCount = group.details.filter(traceResolved).length;
    const total = group.details.length;
    const id = traceInstanceId(selectedDetail);
    const statusClass = resolvedCount === total ? "is-resolved" : resolvedCount === 0 ? "is-unresolved" : "is-mixed";
    return `<button class="trace-row ${selected ? "is-selected" : ""} ${statusClass}" type="button" data-trace-key="${esc(rowKey(selectedDetail))}" data-instance-key="${esc(group.key)}" data-rollout-index="${esc(rolloutIndex(selectedDetail))}" aria-label="${esc(`${id} ${resolvedCount}/${total} successful rollouts`)}">
      ${traceRolloutSegments(group.details)}
      <span class="trace-id">${esc(id)}</span>
      <span class="trace-meta">${esc(`${resolvedCount}/${total} success · selected rollout ${rolloutIndex(selectedDetail) + 1} · ${traceSummary(selectedDetail)}`)}</span>
      ${traceStatusIcons(selectedDetail)}
    </button>`;
  });
  return `<div class="trace-list">${rows.join("") || '<div class="empty">No instances in the selected experiment.</div>'}</div>`;
}

function renderTraceLegend() {
  return `${visualLegend("", TRACE_LEGEND_GROUPS)}${visualLegend("Graph", GRAPH_LEGEND_GROUPS)}`;
}

function renderTraceWorkspacePanels(detail) {
  return `<details id="trace-graph-section" class="trace-section trace-graph-section" data-trace-panel="graph" ${tracePanelOpen("graph") ? "open" : ""}>
    <summary>Graph</summary>
    <section id="trace-graph-pane" class="trace-panel trace-graph-top">${renderGraph(detail)}</section>
  </details>
  <details id="trace-step-section" class="trace-section trace-step-section" data-trace-panel="steps" ${tracePanelOpen("steps") ? "open" : ""}>
    <summary>Trace</summary>
    <section class="trace-panel trace-step-panel">
      <section id="trace-middle-pane" class="trace-middle">
        <h3>Purpose blocks and step timeline</h3>
        ${renderTimeline(detail)}
      </section>
      <aside id="trace-right-pane" class="trace-right">${renderStepDetail(detail)}</aside>
    </section>
  </details>`;
}

function blockSteps(block) {
  return block.trace_indices || block.step_indices || [];
}

function stepByTraceIndex(detail) {
  const byIndex = new Map();
  (detail.step_inspection || []).forEach((step) => byIndex.set(Number(step.trace_index), step));
  (detail.step_details || []).forEach((scored, index) => {
    const traceIndex = Number(scored.trace_index ?? index);
    if (!byIndex.has(traceIndex)) byIndex.set(traceIndex, { trace_index: traceIndex, step_index: scored.step_index, scored });
  });
  return byIndex;
}

function renderTimeline(detail) {
  const byIndex = stepByTraceIndex(detail);
  const blocks = detail.purpose_blocks || [];
  if (!blocks.length && !byIndex.size) return '<div class="empty">No step-level trace was captured.</div>';
  if (!blocks.length) {
    return `<div class="timeline-grid">${[...byIndex.values()].map((step) => renderStepThumb(step, detail)).join("")}</div>`;
  }
  return blocks.map((block, blockIndex) => {
    const steps = blockSteps(block).map((idx) => byIndex.get(Number(idx))).filter(Boolean);
    return `<section class="purpose-block">
      <div class="block-head">
        <strong>Block ${esc(displayBlockIndex(block, detail, blockIndex))}</strong>
        <span>${esc(block.family || "-")} ${esc(block.target_path || "")}</span>
      </div>
      <div class="timeline-grid">${steps.map((step) => renderStepThumb(step, detail)).join("") || '<span class="muted">No captured step in this block.</span>'}</div>
    </section>`;
  }).join("");
}

function renderStepThumb(step, detail) {
  const traceIndex = Number(step.trace_index ?? step.step_index ?? 0);
  const selected = traceIndex === state.selectedStepIndex;
  const tone = stepTone(step, detail);
  const segments = stepRoleSegments(step, detail);
  const style = stepSegmentsStyle(segments);
  const segmentMarkup = stepSegmentsMarkup(segments);
  const failed = step.execution_error === true || step.status === "error";
  const tool = step.tool_name || (step.tool_names || [step.scored?.family || "step"]).join("+");
  const family = step.action_family && step.action_family !== "other" ? `${step.action_family}: ` : "";
  const target = step.target_path || step.path || step.scored?.target_path || "";
  return `<button type="button" class="step-thumb ${tone} ${failed ? "is-error" : ""} ${selected ? "is-selected" : ""}" data-step-index="${traceIndex}" data-step-tone="${esc(tone)}" data-step-roles="${esc(segments.join(","))}"${style}>
    ${segmentMarkup}
    <span class="step-num">${esc(displayStepLabel(step, detail, traceIndex))}</span>
    <span class="step-tool">${esc(`${family}${tool}`)}</span>
    <span class="step-target">${esc(target.split("/").slice(-2).join("/") || step.command || "no target")}</span>
  </button>`;
}

function jsonBlock(value) {
  const text = typeof value === "string" ? value : JSON.stringify(value ?? "", null, 2);
  return `<pre class="detail-pre">${esc(text || "-")}</pre>`;
}

function argValue(value) {
  if (value === null || value === undefined) return "-";
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}

function renderArgumentPairs(args) {
  const rows = (args || []).map((item) => `<tr><td>${esc(item.key)}</td><td><code>${esc(argValue(item.value))}</code></td></tr>`);
  return table(["Argument", "Value"], rows);
}

function displayToolCalls(step) {
  if ((step.parsed_tool_calls || []).length) return step.parsed_tool_calls;
  const names = step.tool_names || (step.tool_name ? [step.tool_name] : []);
  return names.map((name, index) => ({
    name,
    arguments: Object.entries((step.tool_args || [])[index] || {}).map(([key, value]) => ({ key, value })),
  }));
}

function renderDiff(oldText, newText) {
  if (oldText === undefined && newText === undefined) return "";
  const oldLines = String(oldText || "").split("\n");
  const newLines = String(newText || "").split("\n");
  const dp = Array.from({ length: oldLines.length + 1 }, () => Array(newLines.length + 1).fill(0));
  for (let i = oldLines.length - 1; i >= 0; i -= 1) {
    for (let j = newLines.length - 1; j >= 0; j -= 1) {
      dp[i][j] = oldLines[i] === newLines[j]
        ? dp[i + 1][j + 1] + 1
        : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }
  const rows = [];
  let i = 0;
  let j = 0;
  while (i < oldLines.length || j < newLines.length) {
    if (i < oldLines.length && j < newLines.length && oldLines[i] === newLines[j]) {
      rows.push(`<div class="diff-line diff-ctx"><span> </span><code>${esc(oldLines[i])}</code></div>`);
      i += 1;
      j += 1;
      continue;
    }
    if (i < oldLines.length && (j >= newLines.length || dp[i + 1][j] >= dp[i][j + 1])) {
      rows.push(`<div class="diff-line diff-del"><span>-</span><code>${esc(oldLines[i])}</code></div>`);
      i += 1;
      continue;
    }
    if (j < newLines.length) {
      rows.push(`<div class="diff-line diff-add"><span>+</span><code>${esc(newLines[j])}</code></div>`);
      j += 1;
    }
  }
  return `<section><h4>Inline diff</h4><div class="diff-view">${rows.join("") || '<div class="muted">No textual difference.</div>'}</div></section>`;
}

function renderToolCalls(step) {
  const calls = displayToolCalls(step);
  if (!calls.length) return '<div class="empty">No tool call captured.</div>';
  return calls.map((call, index) => `<article class="tool-call">
    <div class="tool-call-name">${esc(index + 1)}. ${esc(call.name || "unknown")}</div>
    ${renderArgumentPairs(call.arguments || [])}
  </article>`).join("");
}

function stepHitNodes(step) {
  const nodes = [
    ...((step?.scored || {}).hit_nodes || []),
    ...(step?.hit_nodes || []),
  ];
  const seen = new Set();
  return nodes.filter((node) => {
    const key = node?.key || `${node?.file_path || ""}:${node?.start_line || ""}:${node?.end_line || ""}:${node?.node_role || ""}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function nodePath(node) {
  if (node?.file_path) return node.file_path;
  const key = String(node?.key || "");
  return key.includes("::") ? key.split("::")[0] : key || "-";
}

function nodeCallable(node) {
  if (node?.callable_name) return node.callable_name;
  if (node?.qualified_name) return node.qualified_name;
  if (node?.symbol) return node.symbol;
  if (node?.name) return node.name;
  const key = String(node?.key || "");
  return key.includes("::") ? key.split("::").slice(1).join("::") : "-";
}

function nodeLineRange(node) {
  if (node?.start_line === undefined || node?.start_line === null) return "";
  if (node?.end_line === undefined || node?.end_line === null || node.end_line === node.start_line) return `:${node.start_line}`;
  return `:${node.start_line}-${node.end_line}`;
}

function groupedStepHitNodes(step, detail) {
  const groups = [
    { key: "symptom-root-cause", label: "symptom + root cause", nodes: [] },
    { key: "symptom", label: "symptom", nodes: [] },
    { key: "test-adapter", label: "test-adapter", nodes: [] },
    { key: "intermediate", label: "intermediate", nodes: [] },
    { key: "fix-adapter", label: "fix-adapter", nodes: [] },
    { key: "root", label: "root cause", nodes: [] },
  ];
  const byKey = new Map(groups.map((group) => [group.key, group]));
  for (const node of stepHitNodes(step)) {
    const segment = stepNodeSegment(node, detail);
    if (segment && byKey.has(segment)) byKey.get(segment).nodes.push(node);
  }
  return groups;
}

function renderStepNodeHits(step, detail) {
  const groups = groupedStepHitNodes(step, detail).filter((group) => group.nodes.length);
  if (!groups.length) {
    return `<section class="step-node-hits"><h4>Visited Graph nodes</h4><div class="empty">No Graph node hit in this step.</div></section>`;
  }
  const groupHtml = groups.map((group) => `<div class="node-hit-group ${esc(group.key)}">
    <div class="node-hit-group-title"><span class="node-hit-swatch"></span><span>${esc(group.label)}</span></div>
    <ul class="node-hit-list">
      ${group.nodes.map((node) => `<li>
        <code class="node-hit-path">${esc(nodePath(node))}</code>
        <span class="node-hit-callable">${esc(nodeCallable(node))}</span>
        ${nodeLineRange(node) ? `<span class="node-hit-lines">${esc(nodeLineRange(node))}</span>` : ""}
      </li>`).join("")}
    </ul>
  </div>`).join("");
  return `<section class="step-node-hits"><h4>Visited Graph nodes</h4><div class="node-hit-groups">${groupHtml}</div></section>`;
}

function renderToggle(title, value, className = "", open = false) {
  return `<details class="detail-toggle ${esc(className)}" ${open ? "open" : ""}><summary>${esc(title)}</summary>${jsonBlock(value || "(empty)")}</details>`;
}

function hasInlineDiff(step) {
  const isWrite = step.action_family === "edit" || ["str_replace", "create", "insert"].includes(String(step.command || ""));
  return isWrite && (step.old_str !== undefined || step.new_str !== undefined);
}

function renderStepDetail(detail) {
  const byIndex = stepByTraceIndex(detail);
  const step = byIndex.get(Number(state.selectedStepIndex)) || [...byIndex.values()][0];
  if (!step) {
    return `<section class="step-detail"><h3>Trajectory detail</h3><div class="empty">Raw step content was not captured for this artifact.</div></section>`;
  }
  return `<section class="step-detail">
    <div class="step-detail-head"><h3>Step ${esc(displayStepLabel(step, detail, Number(step.trace_index ?? step.step_index ?? 0)))}</h3><button class="copy-link" type="button" data-copy-link="step">Copy step URL</button></div>
    <div class="detail-badges">
      ${step.execution_error || step.status === "error" ? badge("execution error", true, "bad") : ""}
      ${step.parse_error ? badge("parse error", true, "bad") : ""}
      ${step.exit_reason ? badge(step.exit_reason) : ""}
    </div>
    <div class="detail-grid">
      ${renderStepNodeHits(step, detail)}
      <section><h4>Tool calls</h4>${renderToolCalls(step)}</section>
      ${renderToggle("Reasoning", step.reasoning_text || "(empty)", "", true)}
      ${renderToggle("Chat", step.chat_text || step.response_text || "(empty)")}
      ${hasInlineDiff(step) ? renderDiff(step.old_str, step.new_str) : ""}
      ${renderToggle("Action", step.raw_action || step.tool_calls || [])}
      ${renderToggle("Observation", step.observation || step.tool_results || [], "observation-toggle", true)}
    </div>
  </section>`;
}

function renderTraceInspector(snapshot) {
  renderTracePatternStats(snapshot);
  renderTraceGlobalRolloutControl(snapshot);
  if (!state.selectedEvalCellKey) {
    document.getElementById("trace-inspector").innerHTML = '<div class="empty">Select a dataset and eval cell/model before inspecting trajectories.</div>';
    return;
  }
  const detailsLoading = state.detailLoadBusyKeys.has(state.selectedEvalCellKey);
  if (detailsLoading && loadedDetailCountForCell(snapshot, state.selectedEvalCellKey) <= 0) {
    document.getElementById("trace-inspector").innerHTML = renderDetailLoadProgress(state.selectedEvalCellKey);
    return;
  }
  if (cellNeedsDetailLoad(snapshot, state.selectedEvalCellKey)) {
    state.detailLoadProgress[state.selectedEvalCellKey] = {
      loaded: 0,
      total: expectedCellDetailCount(snapshot, state.selectedEvalCellKey),
    };
    deferTask(() => loadCellDetails(state.selectedEvalCellKey));
    document.getElementById("trace-inspector").innerHTML = renderDetailLoadProgress(state.selectedEvalCellKey);
    return;
  }
  const detailError = state.detailLoadErrors[state.selectedEvalCellKey];
  if (detailError) {
    document.getElementById("trace-inspector").innerHTML = `<div class="empty">Trajectory details failed to load: ${esc(detailError)}</div>`;
    return;
  }
  const detail = selectedDetail(snapshot);
  if (!detail) {
    document.getElementById("trace-inspector").innerHTML = '<div class="empty">No trajectory details are available for this experiment.</div>';
    return;
  }
  state.selectedTraceKey = rowKey(detail);
  const selectedSteps = stepByTraceIndex(detail);
  if (!selectedSteps.has(Number(state.selectedStepIndex))) {
    state.selectedStepIndex = Number([...selectedSteps.keys()][0] ?? 0);
  }
  const detailLoadBanner = detailsLoading ? `<div class="trace-load-banner">${renderDetailLoadProgress(state.selectedEvalCellKey)}</div>` : "";
  document.getElementById("trace-inspector").innerHTML = `
    ${detailLoadBanner}
    <aside id="trace-left-pane" class="trace-left">${renderTraceList(snapshot)}</aside>
    <section class="trace-workspace">
      <div class="trace-overview">${renderTraceTitleCard(detail, snapshot)}</div>
      ${renderTraceWorkspacePanels(detail)}
    </section>`;
  document.querySelectorAll(".trace-section[data-trace-panel]").forEach((section) => {
    section.addEventListener("toggle", () => {
      setTracePanelOpen(section.dataset.tracePanel, section.open);
    });
  });
  document.querySelectorAll(".trace-row").forEach((button) => {
    button.addEventListener("click", () => {
      const leftScroll = document.getElementById("trace-left-pane")?.scrollTop || 0;
      state.selectedTraceKey = button.dataset.traceKey;
      state.selectedStepIndex = 0;
      state.selectedGraphNodeKey = null;
      resetTracePanels();
      renderTraceInspector(state.snapshot);
      restoreInspectorScroll({ left: leftScroll, middle: 0, right: 0 });
    });
  });
  document.getElementById("trace-rollout-select")?.addEventListener("change", (event) => {
    const scrollState = captureInspectorScroll();
    state.selectedTraceKey = event.target.value;
    state.selectedStepIndex = 0;
    state.selectedGraphNodeKey = null;
    resetTracePanels();
    renderTraceInspector(state.snapshot);
    restoreInspectorScroll({ ...scrollState, middle: 0, right: 0 });
  });
  document.querySelectorAll(".step-thumb").forEach((button) => {
    button.addEventListener("click", () => {
      const scrollState = captureInspectorScroll();
      state.selectedStepIndex = Number(button.dataset.stepIndex || 0);
      renderTraceInspector(state.snapshot);
      restoreInspectorScroll({ ...scrollState, right: 0 });
    });
  });
  document.querySelectorAll(".copy-link").forEach((button) => {
    button.addEventListener("click", () => copyDashboardLink(button.dataset.copyLink || "step"));
  });
  const graphContextToggle = document.getElementById("graph-context-toggle");
  if (graphContextToggle) {
    graphContextToggle.addEventListener("change", (event) => {
      const scrollState = captureInspectorScroll();
      setGraphContext(event.target.checked);
      renderTraceInspector(state.snapshot);
      restoreInspectorScroll(scrollState);
    });
  }
  document.querySelectorAll(".graph-edge-filter-checkbox").forEach((input) => {
    input.addEventListener("change", (event) => {
      const bucket = event.target.dataset.edgeFilter;
      if (!bucket) return;
      const scrollState = captureInspectorScroll();
      state.graphEdgeFilters[bucket] = Boolean(event.target.checked);
      renderTraceInspector(state.snapshot);
      restoreInspectorScroll(scrollState);
    });
  });
  document.querySelectorAll(".graph-node[data-node-key]").forEach((node) => {
    const selectNode = () => {
      const scrollState = captureInspectorScroll();
      state.selectedGraphNodeKey = state.selectedGraphNodeKey === node.dataset.nodeKey ? null : node.dataset.nodeKey;
      renderTraceInspector(state.snapshot);
      restoreInspectorScroll(scrollState);
    };
    node.addEventListener("click", selectNode);
    node.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectNode();
      }
    });
  });
}

function render(options = {}) {
  const snapshot = state.snapshot;
  if (!snapshot) {
    document.getElementById("summary-grid").innerHTML = '<div class="empty">Dashboard data is not available.</div>';
    return;
  }
  const tableScrollState = options.tableScrollState || captureTableScroll();
  applyPendingLocator(snapshot);
  ensureSelection(snapshot);
  syncFilterControls();
  renderSources(snapshot);
  renderOperationStatus();
  renderSelectedExperiment(snapshot);
  renderSummary(snapshot);
  renderAdminPanel(snapshot);
  renderExperiments(snapshot);
  renderTrend(snapshot);
  renderDistributions(snapshot);
  renderModels(snapshot);
  renderRuns(snapshot);
  document.getElementById("trace-legend").innerHTML = renderTraceLegend();
  renderTraceInspector(snapshot);
  renderTracePatternStats(snapshot);
  restoreTableScroll(tableScrollState);
  restoreInspectorScroll(options.scrollState);
}

function setTab(tabName) {
  state.activeTab = tabName;
  document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("is-active", tab.dataset.tab === tabName));
  document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.toggle("is-active", panel.id === tabName));
}

function configureEvents() {
  document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => setTab(tab.dataset.tab)));
  document.getElementById("refresh-button").addEventListener("click", () => loadSnapshot({ dropSelectedDetails: true }));
  document.querySelectorAll(".case-filter-checkbox").forEach((input) => {
    input.addEventListener("change", (event) => {
      const bucket = event.target.dataset.caseFilter;
      if (!bucket) return;
      state.caseFilters[bucket] = Boolean(event.target.checked);
      state.selectedTraceKey = null;
      state.selectedStepIndex = 0;
      state.selectedGraphNodeKey = null;
      resetTracePanels();
      render();
    });
  });
  document.getElementById("clear-experiment").addEventListener("click", () => {
    state.selectedDataset = null;
    state.selectedEvalCellKey = null;
    state.selectedExperimentKey = null;
    state.selectedTraceKey = null;
    state.selectedStepIndex = 0;
    state.selectedGraphNodeKey = null;
    resetTracePanels();
    render();
  });
  document.getElementById("permalink-jump")?.addEventListener("submit", (event) => {
    event.preventDefault();
    const locator = parseLocator(document.getElementById("permalink-input")?.value || "");
    if (!locator) {
      state.permalinkNotice = "URL format was not recognized.";
      renderSelectedExperiment(state.snapshot);
      return;
    }
    state.pendingLocator = locator;
    render();
  });
  document.getElementById("admin-login")?.addEventListener("submit", (event) => {
    event.preventDefault();
    loginAdmin();
  });
  document.getElementById("admin-logout-button")?.addEventListener("click", (event) => {
    event.preventDefault();
    logoutAdmin();
  });
  document.getElementById("trace-search").addEventListener("input", (event) => {
    state.traceQuery = event.target.value;
    state.selectedTraceKey = null;
    state.selectedStepIndex = 0;
    state.selectedGraphNodeKey = null;
    resetTracePanels();
    renderTraceInspector(state.snapshot);
  });
  document.querySelectorAll(".trace-pattern-checkbox").forEach((input) => {
    input.addEventListener("change", (event) => {
      const tag = event.target.dataset.patternFilter;
      if (!tag || !TRACE_PATTERN_FILTERS.includes(tag)) return;
      state.tracePatternFilters[tag] = Boolean(event.target.checked);
      state.selectedTraceKey = null;
      state.selectedStepIndex = 0;
      state.selectedGraphNodeKey = null;
      resetTracePanels();
      render();
    });
  });
  document.querySelectorAll(".trace-pattern-cycle").forEach((button) => {
    button.addEventListener("click", (event) => {
      const tag = event.currentTarget.dataset.patternFilter;
      if (!tag || !TRACE_PATTERN_FILTERS.includes(tag)) return;
      const value = nextTracePatternFilterValue(state.tracePatternFilters[tag]);
      state.tracePatternFilters[tag] = TRACE_PATTERN_FILTER_VALUES.includes(value) ? value : null;
      state.selectedTraceKey = null;
      state.selectedStepIndex = 0;
      state.selectedGraphNodeKey = null;
      resetTracePanels();
      render();
    });
  });
  document.getElementById("auto-refresh").addEventListener("change", (event) => {
    if (event.target.checked) startAutoRefresh();
    else stopAutoRefresh();
  });
  if (typeof window !== "undefined") {
    window.addEventListener?.("hashchange", () => {
      state.pendingLocator = parseLocator(window.location?.hash || "");
      render();
    });
  }
}

function startAutoRefresh() {
  stopAutoRefresh();
  state.refreshTimer = setInterval(() => {
    loadSnapshot({ silent: true, queueIfBusy: true });
    loadRebuildStatus();
  }, 3000);
  loadRebuildStatus();
}

function stopAutoRefresh() {
  if (state.refreshTimer) clearInterval(state.refreshTimer);
  state.refreshTimer = null;
}

configureEvents();
if (typeof window !== "undefined" && window.location) state.pendingLocator = parseLocator(window.location.hash || "");
loadAdminStatus();
loadSnapshot();
if (document.getElementById("auto-refresh").checked) startAutoRefresh();
