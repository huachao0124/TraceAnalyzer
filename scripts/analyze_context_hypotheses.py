"""Observation-backed, length-normalized test of two trajectory hypotheses.

Independent analysis (does not touch the canonical dashboard/reward metrics):

  H-graph:  the more a trajectory *clearly* explores unique on-graph context
            (higher new-node discovery / on-graph focus), the more likely it
            passes.
  H-lic:    the higher a trajectory's *unlicensed* reference ratio (names not
            grounded in the issue text or any prior observation), the more
            likely it fails.

The canonical graph KPIs credit a node whenever a read *command* covers its
line span, so a bare ``cat file.py`` / ``grep`` counts as reading the whole
file (end_line defaults to 999999). That over-estimates node exposure
(~71% of hits) and pins hit/recall near the ceiling. Here a node is exposed
only when its source lines *actually appear in the returned tool output*
(observation-backed), so the ratios are meaningful.

All metrics are normalized (rates), and every correlation is reported both raw
and partialled on trace length, because trace length is the dominant confound
(failing traces are longer, so any absolute count trends with failure).

For SWE-Bench Pro cheap-model analysis, the least-confounded read is the
mixed-instance slice: keep only instances where comparable rollouts include
both pass and fail outcomes, then compare normalized trace KPIs within the same
instance. This gives weak directional support for the hypotheses, especially
for unlicensed references, but not a strong causal estimate.

Read-only: queries the raw eval DB and bonus-map JSON; writes nothing back.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from statistics import mean

from p2a.core import is_rewardable_graph_node
from p2a.eval_fault_localization import (
    _initial_context_text,
    _observation_graph_metrics,
    _step_observation_pairs,
)
from p2a.provenance import license_references

CLEAN_MODELS = (
    "deepseek-v4-flash-passthrough",
    "doubao-seed-2-0-lite-passthrough",
    "step-3.7-flash-passthrough",
)

_ON_PATH_HEADER = re.compile(r"(?:cat -n`? on|running `cat -n` on)\s+(\S+?):")
_CATN_LINE = re.compile(r"^\s*(\d+)\t")
_GREP_LINE = re.compile(r"^(?P<file>[^\s:][^:\n]*?):(?P<line>\d+):")
_GREP_LINE_NOFILE = re.compile(r"^(?P<line>\d+)[:-]")


def _norm_path(path: str) -> str:
    path = (path or "").strip()
    for prefix in ("/app/", "/testbed/", "./"):
        if path.startswith(prefix):
            path = path[len(prefix) :]
    return path.lstrip("/")


def _paths_match(obs_path: str, node_path: str) -> bool:
    a, b = _norm_path(obs_path), _norm_path(node_path)
    return bool(a) and bool(b) and (a == b or a.endswith("/" + b) or b.endswith("/" + a))


def _action_file(action: str) -> str | None:
    m = re.search(r"--path\s+(\S+)", action or "")
    if m:
        return m.group(1)
    m = re.search(
        r"\b(?:cat|sed|grep|head|tail|nl|less|view)\b[^\n]*?\s(/\S+\.\w+|\S+\.py)",
        action or "",
    )
    return m.group(1) if m else None


def _shown_lines_from_observation(action: str, obs: str) -> dict[str, set[int]]:
    """Map file -> set of line numbers that literally appear in the tool output."""
    shown: dict[str, set[int]] = {}
    if not isinstance(obs, str) or not obs:
        return shown
    header_file = None
    hm = _ON_PATH_HEADER.search(obs)
    if hm:
        header_file = hm.group(1)
    cmd_file = _action_file(action)
    default_file = header_file or cmd_file
    for line in obs.splitlines():
        gm = _GREP_LINE.match(line)
        if gm:
            shown.setdefault(gm.group("file"), set()).add(int(gm.group("line")))
            continue
        cm = _CATN_LINE.match(line)
        if cm and default_file:
            shown.setdefault(default_file, set()).add(int(cm.group(1)))
            continue
        nm = _GREP_LINE_NOFILE.match(line)
        if nm and default_file and ":" in line[: line.index(nm.group("line")) + len(nm.group("line")) + 1]:
            shown.setdefault(default_file, set()).add(int(nm.group("line")))
    return shown


def _step_observations(step: dict) -> list[tuple[str, str]]:
    out = []
    for tr in step.get("tool_results") or []:
        if not isinstance(tr, dict):
            continue
        obs = tr.get("observation")
        if isinstance(obs, str) and obs.strip():
            out.append((str(tr.get("action") or ""), obs))
    return out


def _rewardable_nodes(bonus_map: dict) -> dict[str, dict]:
    nodes = bonus_map.get("call_graph_nodes") or {}
    return {k: v for k, v in nodes.items() if is_rewardable_graph_node(v)}


def _obs_exposed_nodes(step: dict, rnodes: dict[str, dict]) -> set[str]:
    """Rewardable nodes whose source lines actually appear in this step's output."""
    exposed: set[str] = set()
    for action, obs in _step_observations(step):
        shown = _shown_lines_from_observation(action, obs)
        if not shown:
            continue
        for key, node in rnodes.items():
            npath, ns, ne = (
                node.get("file_path", ""),
                node.get("start_line", 0),
                node.get("end_line", 0),
            )
            for obs_path, lines in shown.items():
                if _paths_match(obs_path, npath) and any(ns <= ln <= ne for ln in lines):
                    exposed.add(key)
                    break
    return exposed


def _rate(a: int, b: int) -> float:
    return a / b if b else 0.0


def compute_trace_metrics(steps: list[dict], record: dict, bonus_map: dict) -> dict:
    read_steps = sum(1 for s in steps if _step_observation_pairs(s))
    graph = _observation_graph_metrics(steps, bonus_map, read_steps)
    lic = license_references(steps, _initial_context_text(record))
    n_ref = lic.get("n_entity_references") or 0
    unlic = lic.get("n_unlicensed_references") or 0
    length = len(steps)
    return {
        "length": length,
        "read_steps": read_steps,
        # H-graph normalized proxies
        "obs_recall": graph.get("obs_recall") or 0.0,
        "new_graph_step_rate": graph.get("obs_new_graph_step_rate") or 0.0,
        "on_graph_focus": graph.get("obs_graph_focus") or 0.0,
        "unique_node_coverage_per_read": graph.get("obs_unique_node_coverage_per_read") or 0.0,
        "unique_node_coverage_per_turn": graph.get("obs_unique_node_coverage_per_turn") or 0.0,
        "repeat_node_exposure_per_read_node": graph.get("obs_repeat_node_exposure_per_read_node") or 0.0,
        "repeat_node_exposure_per_turn_node": graph.get("obs_repeat_node_exposure_per_turn_node") or 0.0,
        "obs_ongraph_steps": graph.get("obs_graph_step_count") or 0,
        # H-lic normalized proxies
        "unlicensed_reference_rate": (unlic / n_ref) if n_ref else 0.0,
        "unlicensed_step_rate": lic.get("unlicensed_step_rate") or 0.0,
    }


def pb_corr(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / n)
    sy = math.sqrt(sum((y - my) ** 2 for y in ys) / n)
    if sx == 0 or sy == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n * sx * sy)


def _residuals(x: list[float], z: list[float]) -> list[float]:
    n = len(x)
    mz, mx = mean(z), mean(x)
    vz = sum((v - mz) ** 2 for v in z)
    b = sum((z[i] - mz) * (x[i] - mx) for i in range(n)) / vz if vz else 0.0
    a = mx - b * mz
    return [x[i] - (a + b * z[i]) for i in range(n)]


def partial_corr(x: list[float], y: list[float], z: list[float]) -> float:
    return pb_corr(_residuals(x, z), _residuals(y, z))


def load_bonus_maps(path: str) -> dict[str, dict]:
    import glob

    bm = {}
    for f in glob.glob(f"{path}/*.json"):
        if "failure" in f:
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if d.get("instance_id") and d.get("call_graph_nodes"):
            bm[d["instance_id"]] = d
    return bm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/evals/traces.sqlite")
    ap.add_argument("--experiment", default="official-3-nooracle-full")
    ap.add_argument("--bonus-maps", default="data/bonus_maps/swebench-pro")
    ap.add_argument("--models", default=",".join(CLEAN_MODELS))
    ap.add_argument("--all-models", action="store_true")
    args = ap.parse_args()

    bm = load_bonus_maps(args.bonus_maps)
    keep = None if args.all_models else set(args.models.split(","))
    c = sqlite3.connect(args.db)
    c.row_factory = sqlite3.Row
    rows = c.execute(
        """select rc.instance_id iid, rc.model_api_name model, rr.resolved,
                  rr.p2a_step_traces_json steps, rr.messages_json msgs, rr.issue_description issue
           from run_cells rc join raw_rollouts rr on rr.cell_id=rc.id
           where rc.experiment_id=? and rc.status='done' and rr.p2a_step_traces_json is not null""",
        (args.experiment,),
    ).fetchall()

    recs = []
    for r in rows:
        if keep is not None and r["model"] not in keep:
            continue
        m = bm.get(r["iid"])
        if not m:
            continue
        try:
            steps = json.loads(r["steps"])
        except Exception:
            continue
        if not isinstance(steps, list) or not steps:
            continue
        record = {
            "p2a_step_traces": steps,
            "messages": json.loads(r["msgs"]) if r["msgs"] else [],
            "issue_description": r["issue"] or "",
        }
        met = compute_trace_metrics(steps, record, m)
        met["passed"] = 1.0 if r["resolved"] else 0.0
        met["model"] = r["model"]
        recs.append(met)

    H_GRAPH = [
        "obs_recall",
        "new_graph_step_rate",
        "on_graph_focus",
        "unique_node_coverage_per_read",
        "unique_node_coverage_per_turn",
        "repeat_node_exposure_per_read_node",
        "repeat_node_exposure_per_turn_node",
    ]
    H_LIC = ["unlicensed_reference_rate", "unlicensed_step_rate"]
    scope = "ALL 5 models" if args.all_models else "CLEAN 3 (DeepSeek/Doubao/Step)"
    y = [r["passed"] for r in recs]
    z = [float(r["length"]) for r in recs]
    npass = int(sum(y))
    print(f"\n### {scope} — {len(recs)} traces ({npass} pass / {len(recs) - npass} fail)")
    print(
        "length: pass %.1f / fail %.1f (r=%+.3f)  [confound]"
        % (
            mean(r["length"] for r in recs if r["passed"]),
            mean(r["length"] for r in recs if not r["passed"]),
            pb_corr(z, y),
        )
    )
    hdr = "%-40s %8s %8s %8s %10s" % ("metric", "pass", "fail", "r", "r|len")
    for group, name in (
        (H_GRAPH, "H-graph (higher -> more pass?)"),
        (H_LIC, "H-lic (higher -> more fail?)"),
    ):
        print(f"\n-- {name} --\n{hdr}")
        for k in group:
            xs = [r[k] for r in recs]
            pv = mean(r[k] for r in recs if r["passed"])
            fv = mean(r[k] for r in recs if not r["passed"])
            print("%-40s %8.3f %8.3f %+8.3f %+10.3f" % (k, pv, fv, pb_corr(xs, y), partial_corr(xs, y, z)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
