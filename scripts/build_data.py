"""Self-contained data builder for P2A on Uni-Agent — ONE entry, no old code.

Every "build data" job lives here as a subcommand:
  r2e            HF -> R2E training parquet (full + skip-filtered .train.parquet)
  swebench-verified HF -> SWE-bench-Verified full eval set
  swebench-hard     HF -> SWE-bench-Verified HARD subset (validation)
  swebench-pro      HF -> SWE-Bench-Pro Python eval subset (Phase 1)
  skip-list      regenerate config/bad_instances.json from gate results (maintenance)

Each sources from HuggingFace and reuses Uni-Agent's schema/prompt constants by
import (never copied). No dependency on the retired src-backup fork.

Usage (from src/, HF reachable):
  PYTHONPATH=.:uni-agent:uni-agent/verl:uni-agent/examples/data_preprocess \
    uv run python scripts/build_data.py r2e          --out <path>/r2e_gym_subset_p2a.parquet
    uv run python scripts/build_data.py swebench-verified --out <path>/swe_bench_verified.parquet
    uv run python scripts/build_data.py swebench-hard     --out <path>/swe_bench_verified_hard.parquet
    uv run python scripts/build_data.py swebench-pro      --out <path>/swe_bench_pro.parquet
    uv run python scripts/build_data.py skip-list     --gate <gate.jsonl> [--gate ...]
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

import pandas as pd

# data_preprocess example modules require a known DEPLOYMENT value at import; we use
# only their prompt constants. This is not the runtime backend. The parquet rows
# below carry pair-diag image refs, and agent_config_arl.yaml supplies ARL at launch.
os.environ.setdefault("DEPLOYMENT", "vefaas")

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))  # src/ on path


def _ensure_uni_agent_data_preprocess_path() -> None:
    path = str(SRC_ROOT / "uni-agent" / "examples" / "data_preprocess")
    if path not in sys.path:
        sys.path.insert(0, path)

MIRROR = "pair-diag-cn-guangzhou.cr.volces.com/code"
HARD_DIFFICULTIES = {"1-4 hours", ">4 hours"}
CONFIG = SRC_ROOT / "config" / "bad_instances.json"

from p2a.datasets import (  # noqa: E402
    R2E_DATA_SOURCE,
    SWEBENCH_HARD_DATA_SOURCE,
    SWEBENCH_PRO_DATA_SOURCE,
    SWEBENCH_VERIFIED_DATA_SOURCE,
    last_nonempty_line,
    parse_string_list,
    selector_files,
)
from p2a.sandbox_git import append_sanitize  # noqa: E402


# ── r2e ───────────────────────────────────────────────────────────────────────
def cmd_r2e(args) -> int:
    from p2a.hf_assets import load_shared_dataset
    from p2a.skip_cases import load_skip_ids
    from r2egym.commit_models.diff_classes import ParsedCommit
    # Import Uni-Agent's prompt/setup constants only.  The row source below is
    # the canonical R2E dataset, not dyyyyyyyy/r2e-gym-subset-filtered.
    from r2e_gym_subset_filtered import POST_SETUP_CMD, SYSTEM_PROMPT, USER_PROMPT

    def ident(repo, pc_json, commit):
        pc = ParsedCommit(**json.loads(pc_json))
        fixed = pc.new_commit_hash or commit
        buggy = pc.old_commit_hash or f"{fixed}^"
        return f"{repo}__{fixed[:10]}", fixed, buggy

    print("Loading R2E-Gym/R2E-Gym-Subset ...", flush=True)
    rows, rel_hit, rel_miss = [], 0, 0
    for ex in load_shared_dataset("R2E-Gym/R2E-Gym-Subset", split="train"):
        repo, pc_json = ex["repo_name"], ex["parsed_commit_content"]
        iid, fixed, buggy = ident(repo, pc_json, ex["commit_hash"])
        md = {
            "repo": repo, "instance_id": iid, "commit_hash": buggy, "old_commit_hash": buggy,
            "new_commit_hash": fixed, "patch": ParsedCommit(**json.loads(pc_json)).get_patch(),
            "expected_output_json": ex["expected_output_json"],
        }
        relevant_files = ex.get("relevant_files")
        rel = json.dumps(list(relevant_files)) if relevant_files is not None else None
        rel_hit += rel is not None
        rel_miss += rel is None
        data_source = R2E_DATA_SOURCE
        tools_kwargs = {
            "env": {
                "deployment": {"image": f"{MIRROR}/{repo}_final:{fixed}"},
                # Image boots at the FIXED commit; restore the buggy state, then replace
                # .git with a neutral baseline so the model cannot recover the gold patch
                # from history (issue #29). Precompute strips this block and keeps history.
                "post_setup_cmd": append_sanitize(POST_SETUP_CMD, "/testbed", buggy_ref=buggy),
            },
            "reward": {"name": "r2e_gym", "metadata": md},
        }
        rows.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(problem_statement=ex["problem_statement"])},
            ],
            "data_source": data_source,
            "instance_id": iid,
            "agent_name": "swe_agent",
            "reward_model": {"ground_truth": md},
            "extra_info": {"data_source": data_source, "instance_id": iid, "tools_kwargs": tools_kwargs},
            "parsed_commit_content": pc_json,   # flat top-level (avoids nested-chunk read error)
            "relevant_files": rel,              # JSON string; normalize_task decodes it
        })
    if rel_miss:
        print(f"WARNING: {rel_miss} rows had no relevant_files (static layer widens)", file=sys.stderr)
    df = pd.DataFrame(rows)
    df.to_parquet(args.out, index=False)
    print(f"wrote {len(df)} rows (relevant_files {rel_hit} hit / {rel_miss} miss) -> {args.out}", flush=True)

    if not args.no_skip_filter:
        skip = load_skip_ids()
        ids = df["extra_info"].map(lambda ei: ei["tools_kwargs"]["reward"]["metadata"]["instance_id"])
        kept = df[~ids.isin(skip)]
        train_out = args.train_out or f"{args.out[:-len('.parquet')]}.train.parquet"
        kept.to_parquet(train_out, index=False)
        print(f"wrote {len(kept)} rows (skip-list excluded {len(df) - len(kept)}/{len(skip)}) -> {train_out}", flush=True)
    return 0


# ── swebench-verified / swebench-hard ─────────────────────────────────────────
def cmd_swebench(args) -> int:
    from p2a.hf_assets import load_shared_dataset
    # swe_bench_verified's modal branch is import-safe on Python <3.11; we use only its prompts.
    os.environ["DEPLOYMENT"] = "modal"
    _ensure_uni_agent_data_preprocess_path()
    from swe_bench_verified import SYSTEM_PROMPT, USER_PROMPT

    def reset(base: str) -> str:
        return " && ".join(["cd /testbed", "git restore .", "git reset --hard",
                            f"git checkout {base}", "git clean -fdq"])

    want = set(args.difficulties) if args.difficulties is not None else None
    # R2E-Gym/SWE-Bench-Verified carries the eval fields but no difficulty; take it from princeton.
    difficulty = {ex["instance_id"]: ex.get("difficulty")
                  for ex in load_shared_dataset("princeton-nlp/SWE-bench_Verified", split="test")}

    data_source = SWEBENCH_VERIFIED_DATA_SOURCE if want is None else SWEBENCH_HARD_DATA_SOURCE
    rows, total = [], 0
    for ex in load_shared_dataset("R2E-Gym/SWE-Bench-Verified", split="test"):
        total += 1
        iid, d = ex["instance_id"], difficulty.get(ex["instance_id"])
        if want is not None and d not in want:
            continue
        metadata = {**ex, "difficulty": d}
        tools_kwargs = {
            "env": {
                "deployment": {"image": f"{MIRROR}/swebench-verified:sweb.eval.x86_64.{iid}"},
                # Base checkout (reset) restores intended state; the sanitize tail then
                # replaces .git with a neutral baseline so the model cannot read the fix
                # from history (issue #29). Precompute strips the tail and keeps history.
                "post_setup_cmd": append_sanitize(reset(ex["base_commit"]), "/testbed"),
            },
            "reward": {"name": "swe_bench", "metadata": metadata},
        }
        rows.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(problem_statement=ex["problem_statement"])},
            ],
            "data_source": data_source,
            "instance_id": iid,
            "agent_name": "swe_agent",
            "reward_model": {"ground_truth": metadata},
            "extra_info": {"data_source": data_source, "instance_id": iid, "tools_kwargs": tools_kwargs},
        })
    pd.DataFrame(rows).to_parquet(args.out, index=False)
    print(f"swebench {'full' if want is None else sorted(want)}: {len(rows)}/{total} -> {args.out}", flush=True)
    return 0


# ── swebench-pro (Phase 1: Python subset only) ───────────────────────────────
def _swebench_pro_problem_text(ex: dict) -> str:
    sections = [("Issue description", ex.get("problem_statement") or "")]
    for title, key in (("Requirements", "requirements"), ("Interface", "interface")):
        value = str(ex.get(key) or "").strip()
        if value:
            sections.append((title, value))
    return "\n\n".join(f"{title}:\n{body.strip()}" for title, body in sections if str(body).strip())


def _swebench_pro_user_prompt(template: str, problem_statement: str, *, repo_path: str = "/app") -> str:
    return template.format(problem_statement=problem_statement).replace("/testbed", repo_path)


def _swebench_pro_restore_tests_cmd(before_repo_set_cmd: str) -> str:
    return last_nonempty_line(before_repo_set_cmd) or ""


def _swebench_pro_post_setup_cmd(before_repo_set_cmd: str, *, repo_path: str = "/app") -> str:
    lines = [line.strip() for line in str(before_repo_set_cmd or "").splitlines() if line.strip()]
    restore_tests_cmd = _swebench_pro_restore_tests_cmd(before_repo_set_cmd)
    if restore_tests_cmd:
        for idx in range(len(lines) - 1, -1, -1):
            if lines[idx] == restore_tests_cmd:
                del lines[idx]
                break
    quoted_repo = shlex.quote(repo_path)
    base = "\n".join(
        [
            "set -e",
            f"cd {quoted_repo}",
            f"git config --global --add safe.directory {quoted_repo} >/dev/null 2>&1 || true",
            *lines,
        ]
    )
    # Replace the original .git (image history / remote refs / reflogs) with a neutral
    # baseline after the repo is materialized at its intended state (issue #29). Supersedes
    # the earlier tag/reflog/gc pruning, which left the object database searchable.
    return append_sanitize(base, repo_path)


def _swebench_pro_missing_selected_files(selected_tests: list[str], expected_nodeids: list[str]) -> list[str]:
    selected_files = set(selector_files(selected_tests))
    if not selected_files:
        return []
    expected_files = set(selector_files(expected_nodeids))
    return sorted(expected_files - selected_files)


def _read_swebench_pro_scripts(scripts_dir: str | None, instance_id: str) -> dict[str, str]:
    if not scripts_dir:
        return {}
    base = Path(scripts_dir).expanduser() / instance_id
    run_script = base / "run_script.sh"
    parser_script = base / "parser.py"
    if not run_script.is_file() or not parser_script.is_file():
        return {}
    return {
        "run_tests": run_script.read_text(encoding="utf-8"),
        "swebench_pro_parser": parser_script.read_text(encoding="utf-8"),
        "swebench_pro_scripts_dir": str(Path(scripts_dir).expanduser()),
    }


def _require_swebench_pro_scripts_dir(scripts_dir: str | None) -> Path:
    if not scripts_dir:
        raise ValueError(
            "swebench-pro requires --scripts-dir <SWE-bench_Pro-os/run_scripts> "
            "or P2A_SWEBENCH_PRO_SCRIPTS_DIR"
        )
    path = Path(scripts_dir).expanduser()
    if not path.is_dir():
        raise ValueError(f"swebench-pro scripts dir does not exist or is not a directory: {path}")
    return path


def validate_swebench_pro_parquet(path: str | Path, *, allow_missing_scripts: bool = False) -> None:
    parquet_path = Path(path)
    df = pd.read_parquet(parquet_path)
    required = ("run_tests", "swebench_pro_parser")
    missing_columns = [column for column in required if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"swebench-pro parquet {parquet_path} is missing required columns: "
            f"{', '.join(missing_columns)}"
        )
    if df.empty:
        return
    script_columns = {"run_tests", "swebench_pro_parser"} if allow_missing_scripts else set()
    bad_columns = [
        column
        for column in required
        if column not in script_columns and df[column].fillna("").astype(str).str.strip().eq("").any()
    ]
    if bad_columns:
        bad_count = int(
            df[list(bad_columns)]
            .fillna("")
            .astype(str)
            .apply(lambda col: col.str.strip().eq(""))
            .any(axis=1)
            .sum()
        )
        raise ValueError(
            f"swebench-pro parquet {parquet_path} has {bad_count} row(s) with empty "
            f"{', '.join(bad_columns)}; rebuild with --scripts-dir "
            "<SWE-bench_Pro-os/run_scripts>"
        )
    if "test_patch" not in df.columns:
        raise ValueError(f"swebench-pro parquet {parquet_path} is missing required columns: test_patch")
    if df["test_patch"].fillna("").astype(str).str.strip().eq("").any():
        bad_count = int(df["test_patch"].fillna("").astype(str).str.strip().eq("").sum())
        raise ValueError(
            f"swebench-pro parquet {parquet_path} has {bad_count} row(s) with empty test_patch; "
            "rebuild with scripts/build_data.py swebench-pro"
        )


def cmd_swebench_pro(args) -> int:
    from env.images import mirror_image
    from p2a.hf_assets import load_shared_dataset

    os.environ["DEPLOYMENT"] = "modal"
    _ensure_uni_agent_data_preprocess_path()
    from swe_bench_verified import SYSTEM_PROMPT, USER_PROMPT

    language = (args.language or "python").strip().lower()
    if language != "python":
        raise ValueError("Phase 1 supports only --language python")

    scripts_dir = args.scripts_dir or os.getenv("P2A_SWEBENCH_PRO_SCRIPTS_DIR") or os.getenv("SWEBENCH_PRO_SCRIPTS_DIR")
    allow_missing_scripts = bool(getattr(args, "allow_missing_scripts", False))
    if not allow_missing_scripts:
        scripts_dir = str(_require_swebench_pro_scripts_dir(scripts_dir))
    output_columns = [
        "prompt",
        "data_source",
        "instance_id",
        "agent_name",
        "reward_model",
        "extra_info",
        "repo",
        "repo_language",
        "base_commit",
        "patch",
        "test_patch",
        "problem_statement",
        "requirements",
        "interface",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
        "selected_test_files_to_run",
        "dockerhub_tag",
        "docker_image",
        "run_tests",
        "swebench_pro_parser",
    ]
    rows, total, skipped_language, skipped_invalid_selected, scripts_hit, scripts_miss = [], 0, 0, 0, 0, 0
    missing_script_ids: list[str] = []
    for raw in load_shared_dataset("ScaleAI/SWE-bench_Pro", split="test"):
        total += 1
        ex = dict(raw)
        repo_language = str(ex.get("repo_language") or "").strip().lower()
        if repo_language != language:
            skipped_language += 1
            continue

        iid = str(ex["instance_id"])
        dockerhub_tag = str(ex["dockerhub_tag"])
        docker_image = f"jefzda/sweap-images:{dockerhub_tag}"
        f2p = parse_string_list(ex.get("fail_to_pass"))
        p2p = parse_string_list(ex.get("pass_to_pass"))
        selected_tests = parse_string_list(ex.get("selected_test_files_to_run"))
        missing_selected = _swebench_pro_missing_selected_files(selected_tests, [*f2p, *p2p])
        if missing_selected:
            skipped_invalid_selected += 1
            print(
                f"WARNING: skipping {iid}: selected_test_files_to_run does not cover "
                f"FAIL_TO_PASS/PASS_TO_PASS files {missing_selected}",
                file=sys.stderr,
            )
            continue
        script_fields = _read_swebench_pro_scripts(scripts_dir, iid)
        if script_fields:
            scripts_hit += 1
        else:
            scripts_miss += 1
            if not allow_missing_scripts:
                missing_script_ids.append(iid)
                continue

        metadata = {
            **ex,
            "data_source": SWEBENCH_PRO_DATA_SOURCE,
            "eval_only": True,
            "phase": "phase1-python",
            "repo_language": repo_language,
            "repo_path": "/app",
            "swebench_pro_repo_path": "/app",
            "FAIL_TO_PASS": json.dumps(f2p),
            "PASS_TO_PASS": json.dumps(p2p),
            "selected_test_files_to_run": json.dumps(selected_tests),
            "docker_image": docker_image,
            "image": mirror_image(docker_image),
            "swebench_pro_restore_tests_cmd": _swebench_pro_restore_tests_cmd(str(ex.get("before_repo_set_cmd") or "")),
            **script_fields,
        }
        tools_kwargs = {
            "env": {
                "deployment": {"image": metadata["image"]},
                "post_setup_cmd": _swebench_pro_post_setup_cmd(str(ex.get("before_repo_set_cmd") or "")),
            },
            "reward": {"name": "swe_bench_pro", "metadata": metadata},
        }
        row = {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _swebench_pro_user_prompt(USER_PROMPT, _swebench_pro_problem_text(ex))},
            ],
            "data_source": SWEBENCH_PRO_DATA_SOURCE,
            "instance_id": iid,
            "agent_name": "swe_agent",
            "reward_model": {"ground_truth": metadata},
            "extra_info": {"data_source": SWEBENCH_PRO_DATA_SOURCE, "instance_id": iid, "tools_kwargs": tools_kwargs},
            "repo": metadata["repo"],
            "repo_language": repo_language,
            "base_commit": metadata["base_commit"],
            "patch": metadata["patch"],
            "test_patch": metadata.get("test_patch") or "",
            "problem_statement": metadata["problem_statement"],
            "requirements": metadata.get("requirements") or "",
            "interface": metadata.get("interface") or "",
            "FAIL_TO_PASS": metadata["FAIL_TO_PASS"],
            "PASS_TO_PASS": metadata["PASS_TO_PASS"],
            "selected_test_files_to_run": metadata["selected_test_files_to_run"],
            "dockerhub_tag": dockerhub_tag,
            "docker_image": docker_image,
            "run_tests": metadata.get("run_tests"),
            "swebench_pro_parser": metadata.get("swebench_pro_parser"),
        }
        rows.append(row)

    if missing_script_ids:
        examples = ", ".join(missing_script_ids[:5])
        suffix = "" if len(missing_script_ids) <= 5 else f", ... (+{len(missing_script_ids) - 5} more)"
        raise ValueError(
            f"missing required SWE-Bench-Pro run scripts for {len(missing_script_ids)} "
            f"instance(s): {examples}{suffix}. Expected run_script.sh and parser.py under "
            f"{scripts_dir}/<instance_id>/"
        )

    pd.DataFrame(rows, columns=output_columns).to_parquet(args.out, index=False)
    validate_swebench_pro_parquet(args.out, allow_missing_scripts=allow_missing_scripts)
    print(
        f"swebench-pro phase1 language={language}: {len(rows)}/{total} "
        f"(skipped_language={skipped_language}, skipped_invalid_selected={skipped_invalid_selected}, "
        f"scripts {scripts_hit} hit / {scripts_miss} miss) -> {args.out}",
        flush=True,
    )
    if scripts_miss:
        print(
            "WARNING: missing official run scripts for some rows; third-party reward eval requires "
            "--scripts-dir <SWE-bench_Pro-os/run_scripts> or P2A_SWEBENCH_PRO_SCRIPTS_DIR.",
            file=sys.stderr,
        )
    return 0


def cmd_validate_swebench_pro(args) -> int:
    validate_swebench_pro_parquet(args.path)
    print(f"swebench-pro parquet valid: {args.path}")
    return 0


# ── skip-list (regenerate config/bad_instances.json from gate results) ─────────
def _fail_reason(rec: dict) -> str:
    if rec.get("error"):
        return "gate_error"
    b, f = rec.get("buggy_shim") or {}, rec.get("fixed") or {}
    if b.get("n_parsed") == 0 or f.get("n_parsed") == 0:
        return "collection_failed_zero_tests"
    if b.get("resolved") is True:
        return "buggy_did_not_fail_f2p"
    if f.get("resolved") is False:
        return "fixed_did_not_pass_p2p"
    return "f2p_p2p_mismatch"


def cmd_skip_list(args) -> int:
    existing = {e["id"]: e for e in json.loads(CONFIG.read_text()).get("skip", [])} if CONFIG.exists() else {}
    gate_pass, gate_fail = set(), {}
    for path in args.gate:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            iid = rec.get("instance_id")
            if not iid:
                continue
            if rec.get("gate_pass") is True:
                gate_pass.add(iid)
            else:
                entry = {"id": iid, "repo": iid.split("__")[0],
                         "reason": _fail_reason(rec), "source": args.source}
                if args.evidence:
                    entry["evidence"] = args.evidence
                gate_fail[iid] = entry
    coverage = len(gate_pass) + len(gate_fail)
    may_drop = args.drop_passed_report_seed and coverage >= args.expected_total
    merged: dict[str, dict] = {}
    for iid, e in existing.items():
        if e.get("source", "").startswith("report") and may_drop and iid in gate_pass:
            continue
        merged[iid] = e
    merged.update(gate_fail)
    skip = sorted(merged.values(), key=lambda e: (e["repo"], e["id"]))
    cfg = json.loads(CONFIG.read_text()) if CONFIG.exists() else {}
    cfg["skip"] = skip
    CONFIG.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"gate_pass={len(gate_pass)} gate_fail={len(gate_fail)} coverage={coverage} "
          f"drop_seed={may_drop} total_skip={len(skip)} -> {CONFIG}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Self-contained P2A data builder (subcommands).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("r2e", help="HF -> R2E training parquet (full + .train.parquet)")
    r.add_argument("--out", required=True)
    r.add_argument("--train-out", default=None)
    r.add_argument("--no-skip-filter", action="store_true")
    r.set_defaults(func=cmd_r2e)

    for name, default_diff, help_text in [
        ("swebench-verified", None, "HF -> SWE-bench-Verified full eval set"),
        ("swebench-hard", sorted(HARD_DIFFICULTIES), "HF -> SWE-bench-Verified HARD subset"),
    ]:
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("--out", required=True)
        sp.add_argument("--difficulties", nargs="*", default=default_diff)
        sp.set_defaults(func=cmd_swebench)

    pro = sub.add_parser("swebench-pro", help="HF -> SWE-Bench-Pro Python eval subset (Phase 1)")
    pro.add_argument("--out", required=True)
    pro.add_argument("--language", default="python", help="Phase-1 language gate; only python is supported")
    pro.add_argument(
        "--scripts-dir",
        default=os.getenv("P2A_SWEBENCH_PRO_SCRIPTS_DIR") or os.getenv("SWEBENCH_PRO_SCRIPTS_DIR"),
        help="Required path to SWE-bench_Pro-os/run_scripts; embeds run_script.sh/parser.py into rows",
    )
    pro.add_argument(
        "--allow-missing-scripts",
        action="store_true",
        help="Debug only: write rows even when official run scripts are unavailable",
    )
    pro.set_defaults(func=cmd_swebench_pro)

    validate_pro = sub.add_parser("validate-swebench-pro", help="validate executable SWE-Bench-Pro parquet fields")
    validate_pro.add_argument("--path", required=True)
    validate_pro.set_defaults(func=cmd_validate_swebench_pro)

    k = sub.add_parser("skip-list", help="regenerate config/bad_instances.json from gate results")
    k.add_argument("--gate", action="append", default=[], required=True)
    k.add_argument("--drop-passed-report-seed", action="store_true")
    k.add_argument("--expected-total", type=int, default=4503)
    k.add_argument("--source", default="uni_agent_gate")
    k.add_argument("--evidence", default=None)
    k.set_defaults(func=cmd_skip_list)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
