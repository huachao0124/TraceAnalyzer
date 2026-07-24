"""Task runner compatibility for TraceAnalyzer's existing parquet schema."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from uni_agent.framework.task_runner import run_task as _run_task

# Import-time registration keeps the pristine Uni-Agent submodule unaware of
# TraceAnalyzer-specific task and sandbox providers.
from . import sandbox as _sandbox  # noqa: F401
from . import tasks as _tasks  # noqa: F401


def legacy_tools_kwargs_to_task(
    tools_kwargs: Mapping[str, Any],
    *,
    raw_prompt: Any,
) -> dict[str, Any]:
    """Translate the pre-Agent-Framework row shape into a Task Config."""
    reward = tools_kwargs.get("reward")
    if not isinstance(reward, Mapping) or not reward.get("name"):
        raise ValueError("legacy tools_kwargs requires reward.name")

    env_config = tools_kwargs.get("env")
    env_config = env_config if isinstance(env_config, Mapping) else {}
    deployment = env_config.get("deployment")
    deployment = deployment if isinstance(deployment, Mapping) else {}

    sandbox_config: dict[str, Any] = {}
    provider = deployment.get("type")
    if provider:
        sandbox_config["provider"] = str(provider)
    image = deployment.get("image") or env_config.get("image")
    if image:
        sandbox_config["image"] = str(image)
    runtime_timeout = deployment.get("runtime_timeout", deployment.get("timeout"))
    if runtime_timeout is not None:
        sandbox_config["runtime_timeout"] = float(runtime_timeout)

    sandbox_kwargs = {
        key: value
        for key, value in deployment.items()
        if key not in {"type", "image", "runtime_timeout", "timeout"}
    }
    post_setup_cmd = env_config.get("post_setup_cmd")
    if post_setup_cmd:
        sandbox_kwargs["post_setup_cmd"] = post_setup_cmd
    if sandbox_kwargs:
        sandbox_config["sandbox_kwargs"] = sandbox_kwargs

    task = {
        "name": str(reward["name"]),
        "prompt": raw_prompt,
        "metadata": dict(reward.get("metadata") or {}),
    }
    if sandbox_config:
        task["sandbox"] = sandbox_config
    return task


def normalize_tools_kwargs(
    tools_kwargs: Mapping[str, Any] | None,
    *,
    raw_prompt: Any,
) -> dict[str, Any]:
    config = dict(tools_kwargs or {})
    task = config.get("task")
    if isinstance(task, Mapping):
        return {"task": dict(task)}
    return {
        "task": legacy_tools_kwargs_to_task(
            config,
            raw_prompt=raw_prompt,
        )
    }


async def run_task(
    *,
    tools_kwargs: dict[str, Any] | None = None,
    raw_prompt: Any = None,
    **kwargs: Any,
):
    """Run either a native new-framework row or an existing P2A row."""
    return await _run_task(
        tools_kwargs=normalize_tools_kwargs(
            tools_kwargs,
            raw_prompt=raw_prompt,
        ),
        raw_prompt=raw_prompt,
        **kwargs,
    )
