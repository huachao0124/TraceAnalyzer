"""Uni-Agent sandbox adapter for dynamic P2A bonus-map construction.

This module starts Uni-Agent ``AgentEnv`` sandboxes and exposes the small
synchronous sandbox surface (``_run`` / ``_execute_raw`` / ``repo_path`` /
``alt_path``) that ``p2a.trace`` instrumentation helpers expect.
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import json
import os
import shlex
import uuid
from pathlib import Path
from typing import Any

from p2a.sandbox_git import strip_sanitize_block


_HEX_DIGITS = frozenset("0123456789abcdef")


def _hash_ref_matches_head(commit_ref: str, actual_head: str | None) -> bool:
    ref = commit_ref.strip().lower()
    head = (actual_head or "").strip().lower()
    if not (7 <= len(ref) <= 40) or not all(ch in _HEX_DIGITS for ch in ref):
        return False
    if len(head) < len(ref) or not all(ch in _HEX_DIGITS for ch in head):
        return False
    return head.startswith(ref)


R2E_POST_SETUP_CMD = """
export PIP_CACHE_DIR=~/.cache/pip
export PATH=/root/.venv/bin:/root/.local/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH

ln -s /testbed/.venv /root/.venv
ln -s /testbed/.venv/bin/python /root/.local/bin/python
ln -s /testbed/.venv/bin/python /root/.local/bin/python3
find "/testbed/.venv/bin" -type f -executable -exec ln -sf {} "/root/.local/bin/" \\;

find . -name '*.pyc' -delete
find . -name '__pycache__' -exec rm -rf {} +
find /r2e_tests -name '*.pyc' -delete
find /r2e_tests -name '__pycache__' -exec rm -rf {} +

mv /testbed/run_tests.sh /root/run_tests.sh
mv /testbed/r2e_tests /root/r2e_tests

mv /r2e_tests /root/r2e_tests
ln -s /root/r2e_tests /testbed/r2e_tests
""".strip()

# R2E-Gym containers run a plain venv (no conda); its DockerRuntime passes
# environment={"PATH": DOCKER_PATH} with the venv bin first. A one-shot ``execute`` is a
# fresh shell each call, so it does not
# inherit post_setup_cmd's PATH export; we must re-inject this every call or the sandbox's
# ``python``/pytest is not found (exit 127 → empty trace). Value copied from the pre-migration
# tracer (rllm swe.py DOCKER_PATH).
_R2E_DOCKER_PATH = "/root/.venv/bin:/root/.local/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

R2E_VEFAAS_IMAGE_TEMPLATE = "enterprise-public-cn-beijing.cr.volces.com/r2e-gym-subset/{instance_number}:latest"
VEFAAS_SWEREX_COMMAND = (
    "curl -fsSL https://vefaas-swe.tos-cn-beijing.ivolces.com/swe-rex/install_1.4.0.sh | bash -s -- {token}"
)


def _run_coro(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _decode_extra_info(task: dict[str, Any]) -> dict[str, Any]:
    extra = task.get("extra_info")
    if isinstance(extra, dict):
        return extra
    if isinstance(extra, str) and extra.strip():
        import json

        try:
            parsed = json.loads(extra)
        except (json.JSONDecodeError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def extract_tools_kwargs(task: dict[str, Any]) -> dict[str, Any]:
    tools_kwargs = task.get("tools_kwargs")
    if isinstance(tools_kwargs, dict):
        return tools_kwargs
    extra = _decode_extra_info(task)
    tools_kwargs = extra.get("tools_kwargs")
    return tools_kwargs if isinstance(tools_kwargs, dict) else {}


def extract_reward_metadata(task: dict[str, Any]) -> dict[str, Any]:
    tools_kwargs = extract_tools_kwargs(task)
    reward = tools_kwargs.get("reward")
    if isinstance(reward, dict) and isinstance(reward.get("metadata"), dict):
        return reward["metadata"]
    return {}


def _parsed_commit_content_dict(*sources: Any) -> dict[str, Any]:
    """Return parsed_commit_content as a dict from the first source that carries it."""
    for src in sources:
        if not isinstance(src, dict):
            continue
        pcc = src.get("parsed_commit_content")
        if isinstance(pcc, str):
            try:
                pcc = json.loads(pcc)
            except (ValueError, TypeError):
                pcc = None
        if isinstance(pcc, dict):
            return pcc
    return {}


def _old_file_contents_from_task(task: dict[str, Any]) -> dict[str, str]:
    """Return R2E old source contents keyed by repo-relative path."""
    metadata = extract_reward_metadata(task)
    pcc = _parsed_commit_content_dict(task, metadata)
    old_sources: dict[str, str] = {}
    for fd in pcc.get("file_diffs", []) or []:
        if not isinstance(fd, dict) or "old_file_content" not in fd:
            continue
        path = fd.get("header", {}).get("file", {}).get("path")
        old_content = fd.get("old_file_content")
        if isinstance(path, str) and path and isinstance(old_content, str):
            old_sources[path] = old_content
    return old_sources


def infer_commit_ref(task: dict[str, Any], instance_id: str | None = None) -> str | None:
    metadata = extract_reward_metadata(task)
    # R2E stores the buggy ref as parsed_commit_content.old_commit_hash (= commit_hash^).
    # It is NOT at the top level / reward metadata, so it MUST be read from parsed_commit_content;
    # otherwise we fall through to commit_hash (the FIXED commit) and run tests on fixed code
    # (the all_pass bug).
    pcc = _parsed_commit_content_dict(task, metadata)
    commit = (
        task.get("old_commit_hash")
        or task.get("base_commit")
        or metadata.get("old_commit_hash")
        or metadata.get("base_commit")
        or pcc.get("old_commit_hash")
        or pcc.get("base_commit")
    )
    if isinstance(commit, str) and commit:
        return commit
    # Last-resort fallback: the PARENT of the fixed commit (= buggy), never the fixed commit itself.
    fixed = task.get("commit_hash") or metadata.get("commit_hash")
    if isinstance(fixed, str) and fixed:
        return f"{fixed}^"

    iid = instance_id or task.get("instance_id") or metadata.get("instance_id")
    if isinstance(iid, str) and "__" in iid:
        suffix = iid.rsplit("__", 1)[-1]
        if suffix:
            # The instance suffix is the FIXED commit prefix; the buggy state is its parent.
            return f"{suffix}^"
    return None


def derive_r2e_vefaas_image(instance_id: str) -> str | None:
    if "__" not in instance_id:
        return None
    instance_number = instance_id.rsplit("__", 1)[-1].lower()
    if not instance_number:
        return None
    return R2E_VEFAAS_IMAGE_TEMPLATE.format(instance_number=instance_number)


def _extract_image(task: dict[str, Any], instance_id: str, *, deployment: str = "vefaas") -> str | None:
    tools_kwargs = extract_tools_kwargs(task)
    env_cfg = tools_kwargs.get("env") if isinstance(tools_kwargs.get("env"), dict) else {}
    deployment_cfg = env_cfg.get("deployment") if isinstance(env_cfg.get("deployment"), dict) else {}
    # An explicit per-sample image override always wins.
    explicit = deployment_cfg.get("image") or env_cfg.get("image")
    if isinstance(explicit, str) and explicit:
        return explicit
    # veFaaS only hosts the enterprise r2e-gym-subset registry. The parquet's
    # ``docker_image`` (e.g. ``namanjain12/*_final``) is NOT available on veFaaS and is built
    # with different/wrong baked package versions than the enterprise image (e.g. orange3 ships
    # numpy 1.24.4 instead of <1.18). Derive the enterprise image first for the veFaaS path;
    # other deployments keep docker_image.
    if deployment == "vefaas":
        enterprise = derive_r2e_vefaas_image(instance_id)
        if enterprise:
            return enterprise
    image = task.get("docker_image")
    if isinstance(image, str) and image:
        return image
    return derive_r2e_vefaas_image(instance_id)


def _extract_post_setup_cmd(task: dict[str, Any]) -> str:
    tools_kwargs = extract_tools_kwargs(task)
    env_cfg = tools_kwargs.get("env") if isinstance(tools_kwargs.get("env"), dict) else {}
    post_setup_cmd = env_cfg.get("post_setup_cmd")
    if isinstance(post_setup_cmd, str) and post_setup_cmd.strip():
        if _is_swebench_pro_task(task):
            return _strip_swebench_pro_reward_restore_from_setup(task, post_setup_cmd)
        return post_setup_cmd
    return R2E_POST_SETUP_CMD


def _strip_swebench_pro_reward_restore_from_setup(task: dict[str, Any], post_setup_cmd: str) -> str:
    restore_tests_cmd = _swebench_pro_restore_tests_cmd(task)
    if not restore_tests_cmd or restore_tests_cmd == "true":
        return post_setup_cmd
    lines = post_setup_cmd.splitlines()
    for idx in range(len(lines) - 1, -1, -1):
        if lines[idx].strip() == restore_tests_cmd:
            del lines[idx]
            break
    return "\n".join(lines)


def _default_env_variables() -> dict[str, str]:
    return {
        "PIP_PROGRESS_BAR": "off",
        "PIP_CACHE_DIR": "~/.cache/pip",
        "PAGER": "cat",
        "MANPAGER": "cat",
        "LESS": "-R",
        "TQDM_DISABLE": "1",
        "GIT_PAGER": "cat",
    }


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} is required; set it or source .secrets/ips.sh.")
    return value


def build_agent_env_config(task: dict[str, Any], *, instance_id: str, deployment: str | None = None) -> dict[str, Any]:
    """Build a Uni-Agent ``AgentEnvConfig`` dict from a dataset/sample row."""
    impl = (deployment or os.getenv("P2A_DEPLOYMENT") or os.getenv("DEPLOYMENT") or "vefaas").lower()
    if impl in ("arl", "nexus"):
        from env.images import select_image_for_sample

        image = select_image_for_sample(task, instance_id=instance_id)
    else:
        image = _extract_image(task, instance_id, deployment=impl)
    if not image:
        raise ValueError(f"Cannot infer sandbox image for {instance_id}")

    if impl == "vefaas":
        function_route = os.getenv("VEFAAS_FUNCTION_ROUTE")
        if function_route:
            function_route = function_route.rstrip("/")
        deployment_config = {
            "type": "vefaas",
            "image": image,
            "command": VEFAAS_SWEREX_COMMAND,
            "timeout": float(os.getenv("P2A_VEFAAS_TIMEOUT", "600")),
            "startup_timeout": float(os.getenv("P2A_VEFAAS_STARTUP_TIMEOUT", "180")),
            "function_id": os.getenv("VEFAAS_FUNCTION_ID"),
            "function_route": function_route,
        }
    elif impl == "modal":
        deployment_config = {
            "type": "modal",
            "image": image,
            "startup_timeout": float(os.getenv("P2A_MODAL_STARTUP_TIMEOUT", "600")),
            "runtime_timeout": float(os.getenv("P2A_MODAL_RUNTIME_TIMEOUT", "600")),
            "deployment_timeout": float(os.getenv("P2A_MODAL_DEPLOYMENT_TIMEOUT", "3600")),
        }
    elif impl == "local":
        deployment_config = {
            "type": "local",
            "image": image,
            "startup_timeout": float(os.getenv("P2A_LOCAL_STARTUP_TIMEOUT", "180")),
            "timeout": float(os.getenv("P2A_LOCAL_TIMEOUT", "600")),
            "container_runtime": os.getenv("P2A_LOCAL_CONTAINER_RUNTIME", "docker"),
        }
    elif impl == "arl":
        repo_path = _repo_path_for_task(task) if _is_swebench_pro_task(task) else "/testbed"
        deployment_config = {
            "type": "arl",
            "image": image,
            "gateway_url": _required_env("ARL_GATEWAY_URL"),
            "namespace": os.getenv("ARL_NAMESPACE", "default"),
            "experiment_id": os.getenv("ARL_EXPERIMENT_ID", "p2a-uniagent-arl-precompute"),
            "timeout": float(os.getenv("ARL_TIMEOUT", "600")),
            "idle_timeout_seconds": int(os.getenv("ARL_IDLE_TIMEOUT", "3600")),
            "max_lifetime_seconds": int(os.getenv("ARL_MAX_LIFETIME", "14400")),
            # Pool pods default to a 1Gi memory cgroup on the current gateway, which
            # OOM-kills any real test-suite run; request sane resources explicitly
            # (applied when the image pool is first created).
            "resources": {
                "requests": {"cpu": os.getenv("ARL_CPU_REQUEST", "1"), "memory": os.getenv("ARL_MEM_REQUEST", "1Gi")},
                "limits": {"cpu": os.getenv("ARL_CPU_LIMIT", "4"), "memory": os.getenv("ARL_MEM_LIMIT", "8Gi")},
            },
            "startup_timeout": float(os.getenv("ARL_STARTUP_TIMEOUT", os.getenv("ARL_SWEREX_STARTUP_TIMEOUT", "240"))),
            "session_cwd": repo_path,
        }
        max_replicas = os.getenv("ARL_MAX_REPLICAS")
        if max_replicas:
            deployment_config["max_replicas"] = int(max_replicas)
    elif impl == "nexus":
        deployment_config = {
            "type": "nexus",
            "image": image,
            "api_base_url": os.getenv("GONGFENG_API_BASE_URL", "http://hyrl-sandbox.prod.woa.com:8052"),
            "runtime_image": os.getenv("GONGFENG_RUNTIME_IMAGE", "mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:latest"),
            "timeout": float(os.getenv("ARL_TIMEOUT", "1200")),
            "startup_timeout": float(os.getenv("ARL_STARTUP_TIMEOUT", "600")),
        }
    else:
        raise ValueError(f"Unsupported P2A_DEPLOYMENT={impl!r}; expected vefaas/modal/local/arl/nexus")

    return {
        "deployment": deployment_config,
        "env_variables": _default_env_variables(),
        "post_setup_cmd": _extract_post_setup_cmd(task),
    }


class UniAgentSandboxAdapter:
    """Expose the minimal synchronous sandbox methods used by ``p2a.trace`` helpers."""

    repo_path = "/testbed"
    alt_path = "/root"

    def __init__(
        self,
        agent_env,
        *,
        default_timeout: int = 300,
        swebench_verified: bool = False,
        swebench_pro: bool = False,
        repo_path: str | None = None,
        startup_env_variables: dict[str, str] | None = None,
        post_setup_cmd: str | None = None,
    ):
        self.agent_env = agent_env
        self.default_timeout = default_timeout
        self.swebench_verified = swebench_verified
        self.swebench_pro = swebench_pro
        if repo_path:
            self.repo_path = repo_path
        self.startup_env_variables = startup_env_variables or {}
        self.post_setup_cmd = post_setup_cmd

    def start(self) -> None:
        self.agent_env.start()
        setup_parts = [
            f"export {key}={shlex.quote(str(value))}"
            for key, value in sorted(self.startup_env_variables.items())
        ]
        if self.post_setup_cmd:
            # Model-facing rows carry a git-sanitize tail (issue #29). Precompute must keep
            # the original image history for checkout_buggy_commit and callable-diff
            # extraction, so strip that tail here and never run it on the precompute path.
            setup_parts.append(strip_sanitize_block(self.post_setup_cmd))
        if setup_parts:
            stdout, stderr, exit_code = self._execute_raw(" && ".join(setup_parts), timeout=300)
            if exit_code != 0:
                raise RuntimeError(f"sandbox post-setup failed exit={exit_code}: {(stderr or stdout)[-1000:]}")

    def close(self) -> None:
        self.agent_env.close()

    def _execute_raw(self, command: str, timeout: int | float | None = None) -> tuple[str, str, int]:
        # One-shot ``execute`` (stateless ManagedSession.execute over HTTP, retry-wrapped
        # in ArlRuntime._session_execute). Tracing commands are self-contained
        # and all cross-step state lives on the sandbox filesystem. This mirrors
        # the pre-migration tracer and returns stdout/stderr as separate streams
        # the way ``_run`` callers expect.
        from swerex.runtime.abstract import Command

        # Inject the sandbox env every call (one-shot execute = fresh shell, no persistent state),
        # mirroring the pre-migration tracer: R2E-Gym → PATH with the venv first + cwd=/testbed;
        # SWE-bench → conda activate testbed. Without this the venv python is unresolved (exit 127).
        if self.swebench_verified and not self.swebench_pro:
            run_command = f"source /opt/miniconda3/bin/activate && conda activate testbed && {command}"
            env = None
        elif self.swebench_pro:
            run_command = command
            env = None
        else:
            run_command = command
            env = {"PATH": _R2E_DOCKER_PATH}
        cmd = Command(
            command=run_command,
            shell=True,
            check=False,
            timeout=float(timeout or self.default_timeout),
            cwd=self.repo_path,
            env=env,
        )
        result = _run_coro(self.agent_env.deployment.runtime.execute(cmd))
        return result.stdout or "", result.stderr or "", int(getattr(result, "exit_code", 0))

    def _run(self, command: str, timeout: int | float | None = None) -> tuple[str, str]:
        stdout, stderr, _ = self._execute_raw(command, timeout=timeout)
        return stdout, stderr

    def read_file(self, path: str | Path) -> str:
        return self.agent_env.read_file(path)

    def write_file(self, path: str | Path, content: str) -> None:
        self.agent_env.write_file(path, content)

    def _materialize_old_file_contents(self, task: dict[str, Any]) -> dict[str, Any]:
        old_sources = _old_file_contents_from_task(task)
        if not old_sources:
            return {
                "buggy_source_materialized": False,
                "buggy_materialize_source": None,
                "buggy_materialized_files": [],
            }

        for rel_path, content in sorted(old_sources.items()):
            abs_path = f"{self.repo_path}/{rel_path}"
            self._execute_raw(f"mkdir -p {shlex.quote(str(Path(abs_path).parent))}", timeout=30)
            self.write_file(abs_path, content)
        return {
            "buggy_source_materialized": True,
            "buggy_materialize_source": "parsed_commit_content.old_file_content",
            "buggy_materialized_files": sorted(old_sources),
        }

    def checkout_buggy_commit(self, task: dict[str, Any], *, instance_id: str) -> dict[str, Any]:
        """Restore the buggy commit even when the image starts at a fixed HEAD."""
        commit_ref = infer_commit_ref(task, instance_id)
        if not commit_ref:
            diag = {"buggy_checkout_ref": None, "buggy_checkout_exit": None, "buggy_checkout_skipped": True}
            materialize_diag = self._materialize_old_file_contents(task)
            diag.update(materialize_diag)
            if materialize_diag.get("buggy_source_materialized"):
                diag["sandbox_code_state"] = "old_sources_materialized"
            return diag

        quoted_ref = shlex.quote(commit_ref)
        expected, expected_err, expected_exit = self._execute_raw(
            f"cd {self.repo_path} && git rev-parse --verify {quoted_ref}^{{commit}}",
            timeout=30,
        )
        expected_commit = expected.strip().splitlines()[-1] if expected_exit == 0 and expected.strip() else None
        # PLAIN checkout only — never `--force` / `git reset --hard`. Those discard the image's
        # install-time worktree fixups (e.g. pandas setup.cfg without --strict-data-files, aiohttp
        # source-compat rewrites) and break test collection.
        cmd = (
            f"cd {self.repo_path} && "
            "git rev-parse --is-inside-work-tree >/dev/null 2>&1 && "
            f"git checkout {quoted_ref}"
        )
        output, stderr, exit_code = self._execute_raw(cmd, timeout=120)
        head, _, head_exit = self._execute_raw(f"cd {self.repo_path} && git rev-parse HEAD", timeout=30)
        actual_head = head.strip().splitlines()[-1] if head_exit == 0 and head.strip() else None
        diag = {
            "buggy_checkout_ref": commit_ref,
            "buggy_checkout_exit": exit_code,
            "buggy_checkout_expected_head": expected_commit,
            "buggy_checkout_expected_exit": expected_exit,
            "buggy_checkout_expected_stderr": expected_err.strip()[-500:] if expected_exit != 0 else "",
            "buggy_checkout_head": actual_head[:12] if actual_head else None,
            "buggy_checkout_head_full": actual_head,
            "buggy_checkout_stderr": stderr.strip()[-500:] if exit_code != 0 else "",
        }
        if exit_code == 0 and actual_head and (
            (expected_commit and actual_head == expected_commit)
            or _hash_ref_matches_head(commit_ref, actual_head)
        ):
            diag["buggy_checkout_verified"] = True
            diag["sandbox_code_state"] = "git_checkout_verified"
            diag["buggy_checkout_verification"] = (
                "expected_head" if expected_commit and actual_head == expected_commit else "commit_ref_head"
            )
            if self.swebench_pro:
                diag.update(self._restore_swebench_pro_tests(task))
            return diag

        diag["buggy_checkout_verified"] = False
        diag["buggy_checkout_stdout"] = output.strip()[-500:] if exit_code != 0 else ""
        materialize_diag = self._materialize_old_file_contents(task)
        diag.update(materialize_diag)
        if materialize_diag.get("buggy_source_materialized"):
            diag["sandbox_code_state"] = "old_sources_materialized"
            return diag

        diag["sandbox_code_state"] = "mismatch"
        diag["sandbox_code_state_mismatch"] = True
        return diag

    def _restore_swebench_pro_tests(self, task: dict[str, Any]) -> dict[str, Any]:
        cmd = _swebench_pro_restore_tests_cmd(task)
        if not cmd:
            return {"swebench_pro_restore_tests_cmd": None, "swebench_pro_restore_tests_exit": None}
        stdout, stderr, exit_code = self._execute_raw(f"cd {self.repo_path} && {cmd}", timeout=120)
        return {
            "swebench_pro_restore_tests_cmd": cmd,
            "swebench_pro_restore_tests_exit": exit_code,
            "swebench_pro_restore_tests_stdout": stdout.strip()[-500:] if exit_code != 0 else "",
            "swebench_pro_restore_tests_stderr": stderr.strip()[-500:] if exit_code != 0 else "",
        }


def create_uni_agent_sandbox(task: dict[str, Any], *, instance_id: str) -> UniAgentSandboxAdapter:
    from uni_agent.interaction import AgentEnv, AgentEnvConfig

    config = build_agent_env_config(task, instance_id=instance_id)
    swebench_pro = _is_swebench_pro_task(task)
    swebench_verified = False if swebench_pro else _is_swebench_verified_task(task)
    repo_path = _repo_path_for_task(task) if swebench_pro else None
    if config["deployment"].get("type") in ("arl", "nexus"):
        from env.deployment import make_env_config

        env_config = make_env_config(
            config["deployment"],
            env_variables=None,
            post_setup_cmd=None,
            tool_install_dir=config.get("tool_install_dir", "/usr/local/bin"),
        )
    else:
        env_config = AgentEnvConfig(**config)
    env = AgentEnv(run_id=f"p2a-bonus-{uuid.uuid4()}", env_config=env_config)
    if config["deployment"].get("type") == "arl":
        return UniAgentSandboxAdapter(
            env,
            swebench_verified=swebench_verified,
            swebench_pro=swebench_pro,
            repo_path=repo_path,
            startup_env_variables=config.get("env_variables"),
            post_setup_cmd=config.get("post_setup_cmd"),
        )
    return UniAgentSandboxAdapter(
        env,
        swebench_verified=swebench_verified,
        swebench_pro=swebench_pro,
        repo_path=repo_path,
    )


def _is_swebench_verified_task(task: dict[str, Any]) -> bool:
    tools_kwargs = extract_tools_kwargs(task)
    reward = tools_kwargs.get("reward") if isinstance(tools_kwargs.get("reward"), dict) else {}
    metadata = extract_reward_metadata(task)
    reward_name = reward.get("name")
    if reward_name == "swe_bench":
        return True
    if metadata.get("FAIL_TO_PASS") is not None or task.get("FAIL_TO_PASS") is not None:
        return True
    return False


def _is_swebench_pro_task(task: dict[str, Any]) -> bool:
    tools_kwargs = extract_tools_kwargs(task)
    reward = tools_kwargs.get("reward") if isinstance(tools_kwargs.get("reward"), dict) else {}
    metadata = extract_reward_metadata(task)
    if reward.get("name") == "swe_bench_pro":
        return True
    for source in (task, metadata):
        data_source = str(source.get("data_source") or "").strip().lower()
        if data_source in {"swebench-pro", "swe-bench-pro"}:
            return True
    return False


def _swebench_pro_restore_tests_cmd(task: dict[str, Any]) -> str | None:
    from p2a.datasets import last_nonempty_line

    for source in (task, extract_reward_metadata(task)):
        value = source.get("swebench_pro_restore_tests_cmd")
        if isinstance(value, str) and value.strip() and value.strip() != "true":
            return value.strip()
        line = last_nonempty_line(source.get("before_repo_set_cmd"))
        if line:
            return line
    return None


def _repo_path_for_task(task: dict[str, Any]) -> str:
    from p2a.datasets import swebench_pro_repo_path

    return swebench_pro_repo_path(task, extract_reward_metadata(task))


def _read_old_sources(env: UniAgentSandboxAdapter, files: set[str]) -> tuple[dict[str, str], set[str]]:
    from p2a.trace import _read_sandbox_file

    old_sources: dict[str, str] = {}
    existing_files: set[str] = set()
    for file_path in sorted(files):
        content, exit_code = _read_sandbox_file(env, f"{env.repo_path}/{file_path}")
        if exit_code == 0:
            old_sources[file_path] = content
            existing_files.add(file_path)
        else:
            old_sources[file_path] = ""
    return old_sources, existing_files


def _filter_patch_to_files(patch_text: str, files: set[str]) -> str:
    wanted = {path.strip("/") for path in files}
    chunks: list[list[str]] = []
    current: list[str] = []
    current_files: set[str] = set()

    def flush() -> None:
        if current and current_files & wanted:
            chunks.append(list(current))

    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            flush()
            current = [line]
            current_files = set()
            parts = line.split()
            for part in parts[2:4]:
                if part.startswith(("a/", "b/")):
                    current_files.add(part[2:])
            continue
        if current:
            if line.startswith("--- ") or line.startswith("+++ "):
                target = line[4:].strip()
                if target.startswith(("a/", "b/")):
                    current_files.add(target[2:])
            current.append(line)
    flush()
    filtered = "\n".join("\n".join(chunk) for chunk in chunks)
    return f"{filtered}\n" if filtered else ""


def _restore_sources_after_patch(
    env: UniAgentSandboxAdapter,
    *,
    files: set[str],
    old_sources: dict[str, str],
    existing_files: set[str],
) -> None:
    for file_path in sorted(files):
        abs_path = f"{env.repo_path}/{file_path}"
        if file_path in existing_files:
            env._execute_raw(f"mkdir -p {shlex.quote(str(Path(abs_path).parent))}", timeout=30)
            env.write_file(abs_path, old_sources.get(file_path, ""))
        else:
            env._execute_raw(f"rm -f {shlex.quote(abs_path)}", timeout=30)
        env._execute_raw(f"rm -f {shlex.quote(abs_path)}.rej {shlex.quote(abs_path)}.orig", timeout=30)


def _apply_patch_with_fallbacks(env: UniAgentSandboxAdapter, patch_path: str, cleanup) -> tuple[str, str]:
    commands = [
        f"git apply --whitespace=nowarn {shlex.quote(patch_path)}",
        f"git apply --3way --whitespace=nowarn {shlex.quote(patch_path)}",
        f"patch --batch --fuzz=5 -p1 -i {shlex.quote(patch_path)}",
    ]
    attempts: list[str] = []
    for command in commands:
        stdout, stderr, exit_code = env._execute_raw(f"cd {env.repo_path} && {command}", timeout=120)
        if exit_code == 0:
            return stdout, stderr
        detail = (stderr or stdout or "").strip()
        attempts.append(f"{command} exit={exit_code}: {detail[-1000:]}")
        cleanup()
    raise RuntimeError("failed to apply golden patch for source diff: " + " | ".join(attempts))


def _apply_patch_and_read_new_sources(
    env: UniAgentSandboxAdapter,
    *,
    patch_text: str,
    files: set[str],
    existing_files: set[str],
    old_sources: dict[str, str],
) -> dict[str, str]:
    from p2a.trace import _read_sandbox_file

    source_patch = _filter_patch_to_files(patch_text, files)
    if not source_patch.strip():
        return {}
    patch_b64 = base64.b64encode(source_patch.encode()).decode()
    env._run(f"printf '%s' '{patch_b64}' | base64 -d > /tmp/_p2a_golden_patch.diff")

    new_sources: dict[str, str] = {}
    def cleanup() -> None:
        _restore_sources_after_patch(
            env,
            files=files,
            old_sources=old_sources,
            existing_files=existing_files,
        )

    try:
        _apply_patch_with_fallbacks(env, "/tmp/_p2a_golden_patch.diff", cleanup)
        for file_path in sorted(files):
            content, read_exit = _read_sandbox_file(env, f"{env.repo_path}/{file_path}")
            if read_exit == 0:
                new_sources[file_path] = content
    finally:
        cleanup()
    return new_sources


def find_changed_callables_via_patch(env: UniAgentSandboxAdapter, task: dict[str, Any]) -> tuple[list[dict], list[dict], dict]:
    """Derive old/new callable changes from a Uni-Agent sample's patch field."""
    from p2a.trace import (
        _get_patched_py_files,
        _is_test_file,
        extract_callables_with_tolerant_fallback,
        extract_non_test_patch,
        find_modified_callables_from_sources,
    )

    patch_text = extract_non_test_patch(task)
    if not patch_text.strip():
        return [], [], {"callable_source": "patch", "patch_error": "missing_patch"}

    files = {file_path for file_path in _get_patched_py_files(patch_text) if not _is_test_file(file_path)}
    if not files:
        return [], [], {"callable_source": "patch", "patched_py_files": 0}

    old_sources, existing_files = _read_old_sources(env, files)
    new_sources = _apply_patch_and_read_new_sources(
        env,
        patch_text=patch_text,
        files=files,
        existing_files=existing_files,
        old_sources=old_sources,
    )

    modified: list[dict] = []
    newly_created: list[dict] = []
    for file_path in sorted(files):
        old_source = old_sources.get(file_path, "")
        new_source = new_sources.get(file_path, "")
        if not new_source:
            continue
        if old_source:
            modified.extend(find_modified_callables_from_sources(old_source, new_source, file_path))
            old_callables = extract_callables_with_tolerant_fallback(old_source, file_path)
        else:
            old_callables = {}
        new_callables = extract_callables_with_tolerant_fallback(new_source, file_path)
        for qname, info in new_callables.items():
            if qname not in old_callables:
                newly_created.append(info.to_dict())

    return modified, newly_created, {
        "callable_source": "sandbox_patch",
        "patched_py_files": len(files),
        "patched_py_files_with_old_source": len([p for p in files if old_sources.get(p)]),
        "patched_py_files_with_new_source": len([p for p in files if new_sources.get(p)]),
    }
