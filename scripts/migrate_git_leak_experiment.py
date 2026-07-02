#!/usr/bin/env python3
"""Re-tag git-leak-contaminated rollouts into a separate ``git-leak`` experiment.

Context: the SWE-bench-Verified ARL sandboxes shipped the full future git history,
so some rollouts looked up the fix commit (``git show <sha>``) and copied the gold
patch. Those rollouts are enumerated by the leak detector in
``data/analysis/2026-07-02_cheap-model-hacking/git_leak_confirmed.json``. Keeping
them under the normal ``official-1`` experiment inflates the model totals; moving
them to a dedicated ``git-leak`` experiment lets the dashboard show them as a
separate cohort and lets clean re-runs repopulate ``official-1`` without collision.

The migration only re-tags the authoritative ``experiment_id`` columns:
  raw DB   : run_cells.experiment_id (+ an ``experiments`` catalog row)
  build DB : dashboard_rollout_details.experiment_id
             dashboard_model_metrics rows for the raw DB are DELETED so the next
             dashboard admin rebuild recomputes the per-model rollups for both the
             cleaned ``official-1`` cohort and the new ``git-leak`` cohort (the
             per-rollout detail cache is preserved and simply re-scoped).

Raw ``rollout_json`` blobs keep their historical ``experiment_id`` (they are an
immutable record, not a scoping key). Idempotent: re-running finds 0 official-1
leaked cells the second time.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

LEAK_EXPERIMENT_ID = "git-leak"
DEFAULT_SOURCE_EXPERIMENT = "official-1"
DEFAULT_LEAK_LIST = "data/analysis/2026-07-02_cheap-model-hacking/git_leak_confirmed.json"
DEFAULT_RAW_DB = "data/evals/traces.sqlite"
DEFAULT_BUILD_DB = "data/evals/traces.dashboard.sqlite"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _backup(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dst = path.with_suffix(path.suffix + f".pre-git-leak.{stamp}.bak")
    shutil.copy2(path, dst)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leak-list", default=DEFAULT_LEAK_LIST)
    ap.add_argument("--raw-db", default=DEFAULT_RAW_DB)
    ap.add_argument("--build-db", default=DEFAULT_BUILD_DB)
    ap.add_argument("--source-experiment", default=DEFAULT_SOURCE_EXPERIMENT)
    ap.add_argument("--target-experiment", default=LEAK_EXPERIMENT_ID)
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    raw_path = Path(args.raw_db).resolve()
    build_path = Path(args.build_db).resolve()
    leak = json.loads(Path(args.leak_list).read_text())
    cell_ids = sorted({int(x["cell_id"]) for x in leak})
    print(f"leak list: {len(leak)} rows -> {len(cell_ids)} distinct cell_ids")

    raw = sqlite3.connect(raw_path)
    raw.row_factory = sqlite3.Row
    qmarks = ",".join("?" * len(cell_ids))

    scope = raw.execute(
        f"SELECT experiment_id, provider_source, dataset, COUNT(*) n "
        f"FROM run_cells WHERE id IN ({qmarks}) GROUP BY 1,2,3", cell_ids,
    ).fetchall()
    print("current scope of leaked cells:")
    for s in scope:
        print(f"  ({s['experiment_id']}, {s['provider_source']}, {s['dataset']}) x {s['n']}")

    movable = raw.execute(
        f"SELECT id FROM run_cells WHERE id IN ({qmarks}) AND experiment_id = ?",
        [*cell_ids, args.source_experiment],
    ).fetchall()
    move_ids = [r["id"] for r in movable]
    print(f"cells under '{args.source_experiment}' to move: {len(move_ids)}")
    if not move_ids:
        print("nothing to move (already migrated?). exiting.")
        return 0

    # experiment catalog rows to clone (one per provider/dataset present)
    combos = raw.execute(
        f"SELECT DISTINCT provider_source, dataset FROM run_cells "
        f"WHERE id IN ({','.join('?'*len(move_ids))})", move_ids,
    ).fetchall()

    if args.dry_run:
        print("[dry-run] would create experiment rows:",
              [(args.target_experiment, c["provider_source"], c["dataset"]) for c in combos])
        print(f"[dry-run] would UPDATE run_cells.experiment_id -> '{args.target_experiment}' for {len(move_ids)} cells")
        return 0

    if not args.no_backup:
        print("backup raw DB   ->", _backup(raw_path))
        if build_path.exists():
            print("backup build DB ->", _backup(build_path))

    now = _now()
    for c in combos:
        snap = raw.execute(
            "SELECT config_snapshot FROM experiments WHERE experiment_id=? AND provider_source=? AND dataset=?",
            (args.source_experiment, c["provider_source"], c["dataset"]),
        ).fetchone()
        base = json.loads(snap["config_snapshot"]) if snap and snap["config_snapshot"] else {}
        base["_git_leak_cohort"] = {
            "note": "rollouts re-tagged out of the clean experiment; observations contained "
                    "gold-patch content sourced from the sandbox's future git history",
            "source_experiment": args.source_experiment,
            "leak_list": args.leak_list,
            "migrated_at": now,
        }
        raw.execute(
            "INSERT INTO experiments(experiment_id, provider_source, dataset, config_snapshot, created_at, updated_at) "
            "VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(experiment_id, provider_source, dataset) DO UPDATE SET "
            "config_snapshot=excluded.config_snapshot, updated_at=excluded.updated_at",
            (args.target_experiment, c["provider_source"], c["dataset"], json.dumps(base), now, now),
        )
    raw.execute(
        f"UPDATE run_cells SET experiment_id=?, updated_at=? WHERE id IN ({','.join('?'*len(move_ids))})",
        [args.target_experiment, now, *move_ids],
    )
    raw.commit()
    moved = raw.execute(
        "SELECT COUNT(*) FROM run_cells WHERE experiment_id=?", (args.target_experiment,)
    ).fetchone()[0]
    print(f"raw DB: run_cells now under '{args.target_experiment}': {moved}")
    raw.close()

    # build DB
    if build_path.exists():
        b = sqlite3.connect(build_path)
        raw_identity = str(raw_path)
        det = b.execute(
            f"UPDATE dashboard_rollout_details SET experiment_id=?, updated_at=? "
            f"WHERE raw_db_path=? AND raw_cell_id IN ({','.join('?'*len(move_ids))})",
            [args.target_experiment, now, raw_identity, *move_ids],
        )
        met = b.execute(
            "DELETE FROM dashboard_model_metrics WHERE raw_db_path=?", (raw_identity,)
        )
        b.commit()
        print(f"build DB: dashboard_rollout_details re-tagged: {det.rowcount}")
        print(f"build DB: dashboard_model_metrics rows deleted (rebuild on next admin rebuild): {met.rowcount}")
        b.close()
    else:
        print("build DB not found; skipped.")

    print("\nDONE. Next: run a dashboard admin rebuild to repopulate per-model rollups "
          "for both 'official-1' (now leak-free) and 'git-leak'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
