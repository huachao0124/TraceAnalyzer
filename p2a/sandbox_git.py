"""Git metadata sanitizer for model-facing ARL rollout/eval sandboxes.

The pair-diag SWE-bench/R2E images keep the original ``/testbed/.git`` (or
``/app/.git``) after setup. That database still reaches the future/fixed commits,
remote/PR refs, and reflogs baked into the image, so a model with shell access can
recover the developer patch via ``git log --all`` / ``git show`` instead of solving
the task. See issue #29.

This module emits a two-phase setup tail for model-facing sandboxes:

1. restore the intended code state (R2E buggy checkout; SWE-bench base checkout is
   already done by the row's own setup);
2. replace ``.git`` with a single neutral baseline commit of the current worktree.

``git diff HEAD`` and ``git apply`` keep working against the synthetic baseline, so
patch extraction and reward/eval still function; ``git log --all`` reveals only the
baseline commit.

The emitted block is wrapped in sentinel comment lines so the precompute path — which
legitimately needs the real history for ``checkout_buggy_commit`` and callable diffs —
can strip it with :func:`strip_sanitize_block` and keep the original image history.
"""

from __future__ import annotations

import re
import shlex

SANITIZE_BEGIN = "# >>>P2A_GIT_SANITIZE>>>"
SANITIZE_END = "# <<<P2A_GIT_SANITIZE<<<"

# Heavy or moved paths that must never enter the baseline commit: the R2E image keeps
# a large ``/testbed/.venv`` and a moved ``/testbed/r2e_tests`` symlink, and byte-code
# is regenerated per run. Excluding them keeps ``git add -A`` fast; none is a reward
# target, so ``git diff HEAD`` / test-file reset are unaffected.
_BASELINE_EXCLUDES = (".venv/", ".venv", "r2e_tests/", "r2e_tests", "*.pyc", "__pycache__/")

_SANITIZE_RE = re.compile(
    re.escape(SANITIZE_BEGIN) + r".*?" + re.escape(SANITIZE_END) + r"\n?",
    re.DOTALL,
)


def git_sanitize_block(repo_path: str = "/testbed", *, buggy_ref: str | None = None) -> str:
    """Return the sentinel-wrapped setup tail that materializes intended state and
    replaces ``.git`` with a neutral baseline commit.

    ``buggy_ref`` (R2E) is checked out first, best-effort: an unreachable ref leaves
    the worktree untouched (never worse than today) rather than failing setup. The
    SWE-bench base checkout is emitted by the row's own reset command upstream of this
    block, so ``buggy_ref`` is only used for R2E.
    """
    repo = shlex.quote(repo_path)
    exclude_lines = "".join(f"{item}\n" for item in _BASELINE_EXCLUDES)
    lines = [SANITIZE_BEGIN]
    if buggy_ref:
        ref = shlex.quote(buggy_ref)
        # Plain checkout (never --force/reset --hard): mirror precompute so install-time
        # worktree fixups survive; non-fatal so a missing ref cannot abort the rollout.
        lines.append(
            f"git -C {repo} rev-parse --is-inside-work-tree >/dev/null 2>&1 "
            f"&& git -C {repo} checkout {ref} >/dev/null 2>&1 || true"
        )
    lines += [
        f"git config --global --add safe.directory {repo} >/dev/null 2>&1 || true",
        f"rm -rf {repo}/.git",
        f"git -C {repo} init -q",
        f"git -C {repo} config user.email p2a@example.invalid",
        f"git -C {repo} config user.name p2a-baseline",
        f"git -C {repo} config commit.gpgsign false",
        f"git -C {repo} config gc.auto 0",
        f"printf '%s' {shlex.quote(exclude_lines)} > {repo}/.git/info/exclude",
        f"git -C {repo} add -A",
        # Baseline must exist even if the worktree is empty, so reward's HEAD reset works.
        f"git -C {repo} commit -q -m baseline "
        f"|| git -C {repo} commit -q --allow-empty -m baseline",
        # Stable ref for callers that prefer a name over HEAD; keep exit status 0 so the
        # joined startup script (exit code = last command) does not fail the session.
        f"git -C {repo} branch -f p2a-baseline >/dev/null 2>&1 || true",
        SANITIZE_END,
    ]
    return "\n".join(lines)


def append_sanitize(post_setup_cmd: str, repo_path: str = "/testbed", *, buggy_ref: str | None = None) -> str:
    """Append the sanitize block to an existing post-setup command (newline-joined)."""
    block = git_sanitize_block(repo_path, buggy_ref=buggy_ref)
    base = (post_setup_cmd or "").rstrip("\n")
    return f"{base}\n{block}" if base else block


def strip_sanitize_block(post_setup_cmd: str | None) -> str:
    """Remove any sanitize block(s) from a post-setup command.

    Used by the precompute path, which needs the original image git history for
    ``checkout_buggy_commit`` and callable-diff extraction and must never sanitize.
    """
    if not post_setup_cmd:
        return post_setup_cmd or ""
    return _SANITIZE_RE.sub("", post_setup_cmd).rstrip("\n")
