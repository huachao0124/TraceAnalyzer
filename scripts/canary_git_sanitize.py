"""Canary for the model-facing git-metadata sanitizer (issue #29).

Two modes:

  offline   Pure-string invariants — the sanitize tail is present in built rows, its
            commands are the expected replace-.git-with-baseline sequence, and the
            precompute strip restores the original history-preserving setup. Runs
            anywhere (no ARL, no parquet).

  arl       Boot ONE model-facing sandbox from a parquet row and assert on the live
            worktree: it starts at the intended buggy/base state, `.git` holds only the
            synthetic baseline, `git log --all --oneline` shows a single commit, and
            `git diff HEAD` still captures a model edit. Requires ARL env + a built
            parquet.

Usage (from src/):
  PYTHONPATH=.:uni-agent:uni-agent/verl uv run python scripts/canary_git_sanitize.py offline
  PYTHONPATH=.:uni-agent:uni-agent/verl uv run python scripts/canary_git_sanitize.py arl \
      --parquet ../../datasets/p2a/swe_bench_verified_hard.parquet --n 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from p2a.sandbox_git import (  # noqa: E402
    SANITIZE_BEGIN,
    SANITIZE_END,
    append_sanitize,
    git_sanitize_block,
    strip_sanitize_block,
)


def _check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def cmd_offline(_args) -> int:
    print("offline canary: sanitize command generation + precompute strip")
    ok = True

    setup = "cd /testbed && echo original-setup"
    r2e = append_sanitize(setup, "/testbed", buggy_ref="deadbeefdeadbeef")
    swe = append_sanitize("cd /testbed && git checkout BASECOMMIT", "/testbed")
    pro = append_sanitize("set -e\ncd /app", "/app")

    ok &= _check("R2E tail carries sentinels", SANITIZE_BEGIN in r2e and SANITIZE_END in r2e)
    ok &= _check("R2E restores buggy state before sanitize",
                 "checkout deadbeefdeadbeef" in r2e and r2e.index("checkout deadbeefdeadbeef") < r2e.index("rm -rf /testbed/.git"))
    ok &= _check("replaces .git with fresh init", "rm -rf /testbed/.git" in r2e and "git -C /testbed init -q" in r2e)
    ok &= _check("creates a single baseline commit", "commit -q -m baseline" in r2e)
    ok &= _check("SWE-bench base checkout precedes sanitize",
                 swe.index("checkout BASECOMMIT") < swe.index("rm -rf /testbed/.git"))
    ok &= _check("SWE-bench-Pro sanitizes /app", "rm -rf /app/.git" in pro and "git -C /app init -q" in pro)

    # Precompute must recover EXACTLY the original history-preserving setup.
    ok &= _check("strip restores original R2E setup", strip_sanitize_block(r2e) == setup,
                 repr(strip_sanitize_block(r2e)))
    ok &= _check("strip leaves no sentinel/rm -rf residue",
                 SANITIZE_BEGIN not in strip_sanitize_block(r2e) and "rm -rf" not in strip_sanitize_block(r2e))
    ok &= _check("strip is a no-op on un-sanitized setup", strip_sanitize_block(setup) == setup)

    # The joined startup script's exit status is the last command's; it must be 0-safe.
    ok &= _check("block ends on an exit-0 guard", git_sanitize_block("/testbed").rstrip().splitlines()[-2].endswith("|| true"))

    print("OFFLINE CANARY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _run(sandbox, command: str, timeout: int = 120) -> tuple[str, int]:
    import asyncio

    resp = asyncio.get_event_loop().run_until_complete(
        sandbox.exec_shell(command, timeout=timeout)
    )
    return ((resp.stdout or "") + (resp.stderr or "")), int(resp.exit_code or 0)


def cmd_arl(args) -> int:
    import pandas as pd

    from env.sandbox import build_sandbox_from_deployment
    from p2a.datasets import swebench_pro_repo_path
    from p2a.precompute.uni_agent_sandbox import build_agent_env_config

    rows = pd.read_parquet(args.parquet).to_dict("records")
    if not rows:
        print(f"no rows in {args.parquet}")
        return 1
    overall = True
    for row in rows[: args.n]:
        iid = row.get("instance_id") or "<unknown>"
        print(f"\narl canary: {iid}")
        env_dict = build_agent_env_config(row, instance_id=str(iid))
        if env_dict["deployment"].get("type") != "arl":
            print("  [SKIP] not an ARL row")
            continue
        repo = swebench_pro_repo_path(row.get("reward_model", {}).get("ground_truth", {})) if "swe_bench_pro" in str(row) else "/testbed"
        sandbox = build_sandbox_from_deployment(
            env_dict["deployment"],
            post_setup_cmd=env_dict.get("post_setup_cmd"),
        )
        import asyncio

        asyncio.get_event_loop().run_until_complete(sandbox.start())
        ok = True
        log_all, _ = _run(sandbox, f"cd {repo} && git log --all --oneline | wc -l")
        ok &= _check("git log --all shows one baseline commit", log_all.strip().splitlines()[-1].strip() == "1", log_all.strip())
        subject, _ = _run(sandbox, f"cd {repo} && git log -1 --pretty=%s")
        ok &= _check("HEAD is the baseline commit", subject.strip().splitlines()[-1].strip() == "baseline", subject.strip())
        # The sanitizer re-points at most ONE version tag (nearest ancestor release, kept
        # for setuptools-scm/versioneer installs) at the baseline commit; nothing else.
        refs, _ = _run(sandbox, f"cd {repo} && git for-each-ref --format='%(refname)' | grep -v -e '^refs/heads/' -e '^refs/tags/' | wc -l")
        ok &= _check("no remote/foreign refs remain", refs.strip().splitlines()[-1].strip() == "0", refs.strip())
        tags, _ = _run(sandbox, f"cd {repo} && git tag | wc -l")
        ok &= _check("at most one preserved version tag", int(tags.strip().splitlines()[-1].strip() or 0) <= 1, tags.strip())
        tag_shas, _ = _run(sandbox, f'cd {repo} && for t in $(git tag); do git rev-list -n1 "$t"; done | sort -u')
        head_sha, _ = _run(sandbox, f"cd {repo} && git rev-parse HEAD")
        tag_lines = [line.strip() for line in tag_shas.strip().splitlines() if line.strip()]
        ok &= _check("preserved tag points at baseline only",
                     not tag_lines or tag_lines == [head_sha.strip().splitlines()[-1].strip()], tag_shas.strip())
        _run(sandbox, f"cd {repo} && echo '# p2a-canary' >> README.md 2>/dev/null || echo '# p2a-canary' > p2a_canary.txt")
        diff, _ = _run(sandbox, f"cd {repo} && git diff HEAD --name-only | wc -l")
        ok &= _check("git diff HEAD captures a model edit", int(diff.strip().splitlines()[-1].strip() or 0) >= 1, diff.strip())
        asyncio.get_event_loop().run_until_complete(sandbox.stop())
        print(f"  {iid}:", "PASS" if ok else "FAIL")
        overall &= ok
    print("\nARL CANARY:", "PASS" if overall else "FAIL")
    return 0 if overall else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("offline", help="pure-string invariants (no ARL)").set_defaults(func=cmd_offline)
    p_arl = sub.add_parser("arl", help="boot a model-facing sandbox and assert on the live worktree")
    p_arl.add_argument("--parquet", required=True, help="built parquet (R2E train or SWE-bench)")
    p_arl.add_argument("--n", type=int, default=1, help="number of rows to canary")
    p_arl.set_defaults(func=cmd_arl)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
