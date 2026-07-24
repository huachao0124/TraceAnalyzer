"""TraceAnalyzer task extensions for the Uni-Agent Agent Framework."""

from __future__ import annotations

import json
import logging
import re
import time

from pydantic import Field
from r2egym.repo_analysis.execution_log_parser import decolor_dict_keys, parse_log_fn

from uni_agent.tasks import Task, TaskConfig, TaskResult
from uni_agent.tasks.registry import register_task

logger = logging.getLogger(__name__)


class R2EGymTaskConfig(TaskConfig):
    name: str = "r2e_gym"
    eval_timeout: float = Field(default=600.0, gt=0)


class SWEBenchTaskConfig(TaskConfig):
    name: str = "swe_bench"
    eval_timeout: float = Field(default=600.0, gt=0)
    run_gold_patch: bool = False


class SWEBenchProTaskConfig(TaskConfig):
    name: str = "swe_bench_pro"
    eval_timeout: float = Field(default=1800.0, gt=0)


def _normalized_status_map(status: dict[str, object]) -> dict[str, object]:
    return {
        key.split(" - ")[0]: status[key]
        for key in sorted(status)
        if key.split(" - ")[0]
    }


async def compute_r2e_reward(
    metadata: dict,
    sandbox,
    *,
    eval_timeout: float,
) -> dict:
    result = {
        "eval_completed": False,
        "eval_execution_time": None,
        "eval_report": None,
        "resolved": False,
    }
    started = time.perf_counter()
    response = await sandbox.exec_shell(
        "bash /root/run_tests.sh 2>&1",
        timeout=eval_timeout,
        workdir="/testbed",
    )
    result["eval_execution_time"] = time.perf_counter() - started
    result["eval_completed"] = response.exit_code == 0

    output = re.sub(r"\x1b\[[0-9;]*m|\r", "", response.stdout)
    parsed = decolor_dict_keys(parse_log_fn(metadata["repo"])(output))
    expected = decolor_dict_keys(json.loads(metadata["expected_output_json"]))
    parsed_status = _normalized_status_map(parsed)
    expected_status = _normalized_status_map(expected)
    resolved = parsed_status == expected_status
    report = {
        "resolved": resolved,
        "found_eval_status": bool(parsed_status),
        "test_status": {
            "parsed_status": parsed_status,
            "expected_status": expected_status,
        },
    }
    result["eval_report"] = report
    result["resolved"] = resolved
    return result


@register_task("r2e_gym")
class R2EGymTask(Task):
    """R2E-Gym task using the new host-side ReAct agent and sandbox API."""

    name = "r2e_gym"
    config_model = R2EGymTaskConfig

    async def run(self) -> TaskResult:
        cfg: R2EGymTaskConfig = self.config  # type: ignore[assignment]
        instance_id = cfg.metadata.get("instance_id", "?")
        logger.info("starting r2e_gym task (instance_id=%s)", instance_id)
        async with self.build_sandbox() as sandbox:
            agent = self.build_agent()
            await agent.run(sandbox=sandbox, messages=cfg.prompt)
            result = await compute_r2e_reward(
                cfg.metadata,
                sandbox,
                eval_timeout=cfg.eval_timeout,
            )
        logger.info("r2e_gym task done: resolved=%s", result["resolved"])
        return TaskResult(
            reward=float(result["resolved"]),
            accuracy=float(result["resolved"]),
            info=result,
        )


@register_task("swe_bench")
class SWEBenchTask(Task):
    """SWE-bench task with the repository's evaluation timeout semantics."""

    name = "swe_bench"
    config_model = SWEBenchTaskConfig

    async def run(self) -> TaskResult:
        cfg: SWEBenchTaskConfig = self.config  # type: ignore[assignment]
        instance_id = cfg.metadata.get("instance_id", "?")
        logger.info("starting swe_bench task (instance_id=%s)", instance_id)
        async with self.build_sandbox() as sandbox:
            if cfg.run_gold_patch:
                await sandbox.write_file(
                    "/tmp/gold_patch.patch",
                    cfg.metadata["patch"],
                )
                await sandbox.exec(
                    [
                        "git",
                        "apply",
                        "--whitespace=fix",
                        "/tmp/gold_patch.patch",
                    ],
                    workdir="/testbed",
                )
            else:
                agent = self.build_agent()
                await agent.run(sandbox=sandbox, messages=cfg.prompt)

            from p2a.reward_specs import compute_swebench_reward

            result = await compute_swebench_reward(
                cfg.metadata,
                sandbox,
                eval_timeout=cfg.eval_timeout,
            )
        logger.info("swe_bench task done: resolved=%s", result["resolved"])
        return TaskResult(
            reward=float(result["resolved"]),
            accuracy=float(result["resolved"]),
            info=result,
        )


@register_task("swe_bench_pro")
class SWEBenchProTask(Task):
    """SWE-Bench-Pro task using its per-instance test script and parser."""

    name = "swe_bench_pro"
    config_model = SWEBenchProTaskConfig

    async def run(self) -> TaskResult:
        cfg: SWEBenchProTaskConfig = self.config  # type: ignore[assignment]
        instance_id = cfg.metadata.get("instance_id", "?")
        logger.info("starting swe_bench_pro task (instance_id=%s)", instance_id)
        async with self.build_sandbox() as sandbox:
            agent = self.build_agent()
            await agent.run(sandbox=sandbox, messages=cfg.prompt)

            from p2a.reward_specs import compute_swebench_pro_reward

            result = await compute_swebench_pro_reward(
                cfg.metadata,
                sandbox,
                eval_timeout=cfg.eval_timeout,
            )
        logger.info("swe_bench_pro task done: resolved=%s", result["resolved"])
        return TaskResult(
            reward=float(result["resolved"]),
            accuracy=float(result["resolved"]),
            info=result,
        )
