"""P2A-local Uni-Agent reward specs.

Importing this module registers reward specs without modifying the Uni-Agent
submodule.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import time
import uuid
from pathlib import Path
from typing import Any

from swerex.runtime.abstract import Command
from uni_agent.async_logging import get_logger
from uni_agent.interaction import AgentEnv
from uni_agent.reward.base import AbstractRewardSpec
from uni_agent.reward.registry import REWARD_SPEC_MODULES, REWARD_SPEC_REGISTRY, register_reward_spec
from uni_agent.utils import auto_await

from p2a.datasets import last_nonempty_line, parse_string_list, selector_files, swebench_pro_repo_path


def _script_from_metadata_or_dir(metadata: dict[str, Any], *, name: str, key: str) -> str:
    value = metadata.get(key)
    if isinstance(value, str) and value.strip():
        return value

    scripts_dir = (
        metadata.get("swebench_pro_scripts_dir")
        or os.getenv("P2A_SWEBENCH_PRO_SCRIPTS_DIR")
        or os.getenv("SWEBENCH_PRO_SCRIPTS_DIR")
    )
    instance_id = metadata.get("instance_id")
    if scripts_dir and instance_id:
        path = Path(str(scripts_dir)).expanduser() / str(instance_id) / name
        if path.is_file():
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"SWE-Bench-Pro {name} is required for {instance_id!r}; build the parquet "
        "with scripts/build_data.py swebench-pro --scripts-dir <SWE-bench_Pro-os/run_scripts> "
        "or set P2A_SWEBENCH_PRO_SCRIPTS_DIR."
    )


def _restore_tests_command(metadata: dict[str, Any]) -> str:
    value = metadata.get("swebench_pro_restore_tests_cmd")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return last_nonempty_line(metadata.get("before_repo_set_cmd")) or "true"


def _repo_path(metadata: dict[str, Any]) -> str:
    return swebench_pro_repo_path(metadata)


def _safe_json_loads(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _reward_test_args(metadata: dict[str, Any]) -> list[str]:
    selected_files = _reward_test_files(metadata)
    return [",".join(selected_files)] if selected_files else []


def _reward_test_files(metadata: dict[str, Any]) -> list[str]:
    f2p = parse_string_list(metadata.get("FAIL_TO_PASS") or metadata.get("fail_to_pass"))
    p2p = parse_string_list(metadata.get("PASS_TO_PASS") or metadata.get("pass_to_pass"))
    selected_files = selector_files(parse_string_list(metadata.get("selected_test_files_to_run")))
    if not selected_files:
        selected_files = selector_files([*f2p, *p2p])
    return selected_files


_TERMINAL_CONTROL_RE = re.compile(r"\x1b\][^\x07]*(?:\x07|\x1b\\)|\x1b\[[0-?]*[ -/]*[@-~]|\r")


def _strip_terminal_controls(text: str) -> str:
    return _TERMINAL_CONTROL_RE.sub("", text)


async def _run_env_command(env: AgentEnv, command: str, *, timeout: int | float | None = None, check: str = "ignore") -> str:
    runtime = getattr(getattr(env, "deployment", None), "runtime", None)
    execute = getattr(runtime, "execute", None)
    if callable(execute):
        response = await execute(Command(command=["bash", "-lc", command], timeout=timeout))
        output = _strip_terminal_controls((response.stdout or "") + (response.stderr or ""))
        if check == "raise" and int(response.exit_code or 0) != 0:
            raise RuntimeError(f"command failed with exit code {response.exit_code}: {output}")
        return output
    return _strip_terminal_controls(await env.communicate(command, timeout=timeout, check=check))


async def _read_env_file_text(env: AgentEnv, path: Path, *, tail_bytes: int | None = None) -> str:
    quoted_path = shlex.quote(path.as_posix())
    if tail_bytes is None:
        command = f"if [ -f {quoted_path} ]; then cat {quoted_path}; fi"
    else:
        command = f"if [ -f {quoted_path} ]; then tail -c {int(tail_bytes)} {quoted_path}; fi"
    try:
        return await _run_env_command(env, command, timeout=60, check="ignore")
    except Exception:
        try:
            return await env.read_file(path)
        except Exception:
            return ""


def _make_swebench_eval_script_list(instance, specs, env_name, repo_directory, test_patch):
    from swebench.harness.constants import END_TEST_OUTPUT, MAP_REPO_VERSION_TO_SPECS, START_TEST_OUTPUT
    from swebench.harness.test_spec.python import get_test_directives
    from swebench.harness.utils import get_modified_files

    heredoc_delimiter = "EOF_114329324912"
    base_commit = instance["base_commit"]
    test_files = get_modified_files(test_patch)
    if test_files:
        files = " ".join(test_files)
        # After git-sanitize (issue #29) the synthetic baseline HEAD holds the base state and
        # the original base_commit is unreachable; reset test files against HEAD, falling back
        # to base_commit for un-sanitized sandboxes.
        reset_tests_command = f"git checkout HEAD -- {files} 2>/dev/null || git checkout {base_commit} -- {files}"
    else:
        reset_tests_command = "echo 'skip reset'"
    apply_test_patch_command = f"git apply -v - <<'{heredoc_delimiter}'\n{test_patch}\n{heredoc_delimiter}"
    test_cmd = MAP_REPO_VERSION_TO_SPECS[instance["repo"]][instance["version"]]["test_cmd"]
    test_command = " ".join([test_cmd, *get_test_directives(instance)])

    eval_commands = [
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        f"cd {repo_directory}",
    ]
    if "eval_commands" in specs:
        eval_commands += specs["eval_commands"]
    eval_commands += [
        f"git config --global --add safe.directory {repo_directory}",
        f"cd {repo_directory}",
        "git status",
        "git show",
        # Diagnostic only (unparsed): diff against HEAD so it works against the synthetic
        # baseline after git-sanitize; base_commit is unreachable there.
        "git -c core.fileMode=false diff HEAD",
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
    ]
    if "install" in specs:
        eval_commands.append(specs["install"])
    eval_commands += [
        reset_tests_command,
        apply_test_patch_command,
        f": '{START_TEST_OUTPUT}'",
        test_command,
        f": '{END_TEST_OUTPUT}'",
        reset_tests_command,
    ]
    return eval_commands


class SWEBenchRewardSpec(AbstractRewardSpec):
    """SWE-Bench verifier that evaluates through the runtime execute interface."""

    def __init__(self, *, run_id: str, metadata: dict, env: AgentEnv, eval_timeout: int = 300):
        self.run_id = run_id
        self.metadata = metadata
        self.env = env
        self.logger = get_logger("reward_spec", run_id=run_id)
        self.eval_timeout = eval_timeout

    @auto_await
    async def apply_gold_patch(self) -> None:
        await self._apply_patch(str(self.metadata.get("patch") or ""))

    @auto_await
    async def compute_reward(self, **kwargs) -> tuple[bool, dict]:
        result = {
            "eval_completed": False,
            "eval_execution_time": None,
            "eval_report": None,
            "resolved": False,
        }
        try:
            eval_script = self._build_eval_script()
            eval_script_container = Path(f"/tmp/eval_script_{uuid.uuid4()}.sh")
            await self.env.write_file(eval_script_container, eval_script)

            execution_t0 = time.perf_counter()
            output = await _run_env_command(
                self.env,
                f"bash {shlex.quote(str(eval_script_container))} 2>&1",
                timeout=self.eval_timeout,
                check="ignore",
            )
            execution_time = time.perf_counter() - execution_t0
            result["eval_completed"] = True
            result["eval_execution_time"] = execution_time

            output = _strip_terminal_controls(output)
            eval_report = self._get_eval_report(output)
            result["eval_report"] = eval_report
            result["resolved"] = bool(eval_report["resolved"])
            self.logger.info(f"Eval report: {eval_report}")
        except Exception as exc:  # noqa: BLE001 - reward failures are captured per instance
            result["error"] = f"{type(exc).__name__}: {exc}"
            self.logger.error(f"Failed to evaluate SWE-Bench instance: {exc}")
        return result["resolved"], result

    def _build_eval_script(self) -> str:
        from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS

        instance = self.metadata
        repo = instance["repo"]
        version = instance.get("version")
        specs = MAP_REPO_VERSION_TO_SPECS[repo][version]
        env_name = "testbed"
        repo_directory = f"/{env_name}"
        eval_script_list = _make_swebench_eval_script_list(
            instance=instance,
            specs=specs,
            env_name=env_name,
            repo_directory=repo_directory,
            test_patch=instance["test_patch"],
        )
        return "\n".join(["#!/bin/bash", "set -uxo pipefail"] + eval_script_list) + "\n"

    @auto_await
    async def _get_interaction_env_patch(self) -> str:
        try:
            env_patch_file = Path(f"/tmp/patch_{uuid.uuid4()}.diff")
            await _run_env_command(
                self.env,
                f"cd /testbed && git add -A && git diff --no-color --cached > {shlex.quote(env_patch_file.as_posix())}",
                check="ignore",
            )
            return await self.env.read_file(env_patch_file)
        except Exception as exc:  # noqa: BLE001 - preserve upstream fallback semantics
            self.logger.error(f"Failed to get interaction environment patch: {exc}")
            return ""

    @auto_await
    async def _apply_patch(self, patch: str) -> None:
        if not patch.strip():
            self.logger.info("Empty patch, nothing to apply.")
            return
        patch_path = Path(f"/tmp/patch_{uuid.uuid4()}.diff")
        await self.env.write_file(patch_path, patch)
        commands = [
            f"cd /testbed && git apply --whitespace=fix {shlex.quote(patch_path.as_posix())}",
            f"cd /testbed && git apply --reject --whitespace=nowarn {shlex.quote(patch_path.as_posix())}",
            f"cd /testbed && patch --batch --fuzz=5 -p1 -i {shlex.quote(patch_path.as_posix())}",
        ]
        last_error: Exception | None = None
        for command in commands:
            try:
                await _run_env_command(self.env, command, check="raise")
                self.logger.info("Applied patch successfully!")
                return
            except RuntimeError as exc:
                last_error = exc
        raise RuntimeError("Failed to apply patch with any command") from last_error

    def _get_logs_eval(self, eval_output: str):
        from swebench.harness.constants import END_TEST_OUTPUT, START_TEST_OUTPUT
        from swebench.harness.log_parsers import MAP_REPO_TO_PARSER

        log_parser = MAP_REPO_TO_PARSER[self.metadata["repo"]]
        if START_TEST_OUTPUT in eval_output and END_TEST_OUTPUT in eval_output:
            test_content = eval_output.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]
            return log_parser(test_content, None), True
        return {}, False

    def _get_eval_report(self, eval_output: str):
        from swebench.harness.constants import FAIL_ONLY_REPOS, EvalType, ResolvedStatus
        from swebench.harness.grading import get_eval_tests_report, get_resolution_status

        eval_report = {
            "resolved": False,
            "found_eval_status": False,
            "test_status": None,
        }
        status_map, found = self._get_logs_eval(eval_output)
        eval_report["found_eval_status"] = found
        if not found:
            return eval_report

        eval_ref = {
            "instance_id": self.metadata["instance_id"],
            "FAIL_TO_PASS": parse_string_list(self.metadata.get("FAIL_TO_PASS")),
            "PASS_TO_PASS": parse_string_list(self.metadata.get("PASS_TO_PASS")),
        }
        repo = self.metadata["repo"]
        eval_type = EvalType.FAIL_ONLY if repo in FAIL_ONLY_REPOS else EvalType.PASS_AND_FAIL
        report = get_eval_tests_report(status_map, eval_ref, eval_type=eval_type)
        eval_report["test_status"] = report
        if get_resolution_status(report) == ResolvedStatus.FULL.value:
            eval_report["resolved"] = True
        return eval_report


REWARD_SPEC_REGISTRY["swe_bench"] = SWEBenchRewardSpec
REWARD_SPEC_MODULES["swe_bench"] = "p2a.reward_specs"


@register_reward_spec("swe_bench_pro")
class SWEBenchProRewardSpec(AbstractRewardSpec):
    """SWE-Bench-Pro verifier using the official per-instance run script/parser."""

    def __init__(self, *, run_id: str, metadata: dict, env: AgentEnv, eval_timeout: int = 1800):
        self.run_id = run_id
        self.metadata = metadata
        self.env = env
        self.eval_timeout = eval_timeout
        self.logger = get_logger("reward_spec", run_id=run_id)

    @auto_await
    async def apply_gold_patch(self) -> None:
        await self._apply_patch(str(self.metadata.get("patch") or ""))

    @auto_await
    async def compute_reward(self, **kwargs) -> tuple[bool, dict]:
        result = {
            "eval_completed": False,
            "eval_execution_time": None,
            "eval_report": None,
            "resolved": False,
        }
        try:
            run_script = _script_from_metadata_or_dir(self.metadata, name="run_script.sh", key="run_tests")
            parser_script = _script_from_metadata_or_dir(
                self.metadata,
                name="parser.py",
                key="swebench_pro_parser",
            )
            paths = await self._write_eval_files(run_script=run_script, parser_script=parser_script)
            eval_script = self._build_eval_script(paths)
            await self.env.write_file(paths["eval_script"], eval_script)

            t0 = time.perf_counter()
            await _run_env_command(
                self.env,
                f"bash {shlex.quote(str(paths['eval_script']))} 2>&1 | cat",
                timeout=self.eval_timeout,
                check="ignore",
            )
            result["eval_execution_time"] = time.perf_counter() - t0
            result["eval_completed"] = True

            output = _safe_json_loads(await _read_env_file_text(self.env, paths["output"]))
            stdout = await self._read_optional(paths["stdout"])
            stderr = await self._read_optional(paths["stderr"])
            eval_report = self._grade(output)
            eval_report["stdout_tail"] = stdout[-4000:]
            eval_report["stderr_tail"] = stderr[-4000:]
            result["eval_report"] = eval_report
            result["resolved"] = bool(eval_report["resolved"])
            self.logger.info(
                f"SWE-Bench-Pro eval: instance={self.metadata.get('instance_id')} "
                f"resolved={result['resolved']} tests={eval_report.get('n_tests')} "
                f"time={result['eval_execution_time']:.1f}s"
            )
        except Exception as exc:  # noqa: BLE001 - reward failures are captured per instance
            result["error"] = f"{type(exc).__name__}: {exc}"
            self.logger.error(f"Failed to evaluate SWE-Bench-Pro instance: {exc}")
        return result["resolved"], result

    async def _write_eval_files(self, *, run_script: str, parser_script: str) -> dict[str, Path]:
        base = Path(f"/tmp/p2a_swebench_pro_{uuid.uuid4().hex}")
        paths = {
            "run_script": base.with_name(base.name + "_run.sh"),
            "parser": base.with_name(base.name + "_parser.py"),
            "eval_script": base.with_name(base.name + "_eval.sh"),
            "stdout": base.with_name(base.name + "_stdout.log"),
            "stderr": base.with_name(base.name + "_stderr.log"),
            "output": base.with_name(base.name + "_output.json"),
        }
        await self.env.write_file(paths["run_script"], run_script if run_script.endswith("\n") else f"{run_script}\n")
        await self.env.write_file(paths["parser"], parser_script if parser_script.endswith("\n") else f"{parser_script}\n")
        return paths

    def _build_eval_script(self, paths: dict[str, Path]) -> str:
        selected = _reward_test_args(self.metadata)
        selected_arg = " ".join(shlex.quote(item) for item in selected)
        repo_path = _repo_path(self.metadata)
        quoted_repo = shlex.quote(repo_path)
        # After git-sanitize (issue #29), a dataset restore command that targets the original
        # base commit finds it unreachable. Fall back to the synthetic baseline HEAD, but only
        # for the test files; restoring "." would wipe the candidate patch before evaluation.
        fallback_files = _reward_test_files(self.metadata)
        fallback_restore = (
            f"git -C {quoted_repo} checkout HEAD -- "
            + " ".join(shlex.quote(item) for item in fallback_files)
            if fallback_files
            else "true"
        )
        restore_cmd = f"{{ {_restore_tests_command(self.metadata)}; }} || {fallback_restore}"
        output_path = shlex.quote(str(paths["output"]))
        stderr_path = shlex.quote(str(paths["stderr"]))
        return "\n".join(
            [
                "#!/bin/bash",
                "set -uo pipefail",
                f"cd {quoted_repo} || {{ printf '{{\"tests\":[],\"restore_error\":\"cd_failed\"}}\\n' > {output_path}; exit 0; }}",
                f"git config --global --add safe.directory {quoted_repo} >/dev/null 2>&1 || true",
                "restore_status=0",
                "(",
                "set -e",
                restore_cmd,
                f") >> {stderr_path} 2>&1 || restore_status=$?",
                'if [ "$restore_status" -ne 0 ]; then',
                f"  printf '{{\"tests\":[],\"restore_error\":\"restore_failed\",\"restore_status\":%s}}\\n' \"$restore_status\" > {output_path}",
                "  exit 0",
                "fi",
                f"chmod +x {shlex.quote(str(paths['run_script']))}",
                f"bash {shlex.quote(str(paths['run_script']))} {selected_arg} > {shlex.quote(str(paths['stdout']))} 2> {shlex.quote(str(paths['stderr']))}",
                "test_status=$?",
                "parser_python=\"$(command -v python3 || command -v python || true)\"",
                "if [ -z \"$parser_python\" ]; then",
                f"  printf '%s\\n' '{{\"tests\":[],\"parser_error\":\"python_not_found\"}}' > {output_path}",
                "else",
                f"  \"$parser_python\" {shlex.quote(str(paths['parser']))} {shlex.quote(str(paths['stdout']))} {shlex.quote(str(paths['stderr']))} {shlex.quote(str(paths['output']))} || "
                f"printf '%s\\n' '{{\"tests\":[],\"parser_error\":\"parser_failed\"}}' > {output_path}",
                "fi",
                "exit 0",
                "",
            ]
        )

    def _grade(self, output: dict[str, Any]) -> dict[str, Any]:
        tests = output.get("tests") if isinstance(output, dict) else None
        tests = tests if isinstance(tests, list) else []
        passed = {
            str(item.get("name"))
            for item in tests
            if isinstance(item, dict) and item.get("status") == "PASSED" and item.get("name") is not None
        }
        f2p = set(parse_string_list(self.metadata.get("FAIL_TO_PASS") or self.metadata.get("fail_to_pass")))
        p2p = set(parse_string_list(self.metadata.get("PASS_TO_PASS") or self.metadata.get("pass_to_pass")))
        missing = sorted((f2p | p2p) - passed)
        report = {
            "resolved": bool(f2p) and not missing,
            "found_eval_status": bool(tests),
            "n_tests": len(tests),
            "n_passed": len(passed),
            "FAIL_TO_PASS": {
                "success": sorted(f2p & passed),
                "failure": sorted(f2p - passed),
            },
            "PASS_TO_PASS": {
                "success": sorted(p2p & passed),
                "failure": sorted(p2p - passed),
            },
            "missing": missing,
        }
        for key in ("restore_error", "restore_status", "parser_error"):
            if key in output:
                report[key] = output[key]
        return report

    async def _read_optional(self, path: Path) -> str:
        return await _read_env_file_text(self.env, path, tail_bytes=4000)

    @auto_await
    async def _apply_patch(self, patch: str) -> None:
        if not patch.strip():
            return
        patch_path = Path(f"/tmp/p2a_swebench_pro_gold_{uuid.uuid4().hex}.diff")
        await self.env.write_file(patch_path, patch)
        repo_path = shlex.quote(_repo_path(self.metadata))
        patch_arg = shlex.quote(patch_path.as_posix())
        commands = [
            f"cd {repo_path} && git apply --whitespace=fix {patch_arg}",
            f"cd {repo_path} && git apply --reject --whitespace=nowarn {patch_arg}",
            f"cd {repo_path} && patch --batch --fuzz=5 -p1 -i {patch_arg}",
        ]
        last_error: Exception | None = None
        for command in commands:
            try:
                await _run_env_command(self.env, command, check="raise")
                return
            except RuntimeError as exc:
                last_error = exc
        raise RuntimeError("Failed to apply patch with any command") from last_error
