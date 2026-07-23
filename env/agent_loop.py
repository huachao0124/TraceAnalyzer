"""Uni-Agent loop adapter that supports ARL deployment without submodule edits."""

from __future__ import annotations

import copy
import os
from typing import Any

from uni_agent.agent_loop import UniAgentLoop
from uni_agent.interaction import AgentEnv

import p2a.reward_specs  # noqa: F401 - registers P2A-local Uni-Agent reward specs

from .deployment import make_env_config


class ArlUniAgentLoop(UniAgentLoop):
    """UniAgentLoop with one extra deployment path: ``env.deployment.type=arl``."""

    def _init_config(self, sampling_params: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        config_dict = super()._init_config(sampling_params, **kwargs)
        self._p2a_run_kwargs = dict(kwargs)
        self._p2a_run_config = config_dict
        return config_dict

    def _init_env(self, config_dict: dict[str, Any]) -> AgentEnv:
        deployment = config_dict.get("deployment")
        if not (isinstance(deployment, dict) and deployment.get("type") in ("arl", "nexus")):
            return super()._init_env(config_dict)

        env_config = make_env_config(
            deployment,
            env_variables=config_dict.get("env_variables"),
            post_setup_cmd=config_dict.get("post_setup_cmd"),
            tool_install_dir=config_dict.get("tool_install_dir", "/usr/local/bin"),
        )
        return AgentEnv(run_id=self.run_id, env_config=env_config)

    def _p2a_trace_enabled(self) -> bool:
        return os.getenv("UNI_AGENT_P2A_TRACE", "").lower() in {"1", "true", "yes", "on"}

    def _extract_p2a_instance_id(self) -> str | None:
        kwargs = getattr(self, "_p2a_run_kwargs", {}) or {}
        config_dict = getattr(self, "_p2a_run_config", {}) or {}

        direct = kwargs.get("instance_id") or kwargs.get("uid")
        if isinstance(direct, str) and direct:
            return direct

        tools_kwargs = kwargs.get("tools_kwargs") or {}
        extra_info = kwargs.get("extra_info") or {}
        for source in (tools_kwargs, extra_info, config_dict):
            reward = source.get("reward") if isinstance(source, dict) else None
            metadata = reward.get("metadata") if isinstance(reward, dict) else None
            instance_id = metadata.get("instance_id") if isinstance(metadata, dict) else None
            if isinstance(instance_id, str) and instance_id:
                return instance_id

        nested_tools = extra_info.get("tools_kwargs") if isinstance(extra_info, dict) else None
        if isinstance(nested_tools, dict):
            reward = nested_tools.get("reward")
            metadata = reward.get("metadata") if isinstance(reward, dict) else None
            instance_id = metadata.get("instance_id") if isinstance(metadata, dict) else None
            if isinstance(instance_id, str) and instance_id:
                return instance_id
        return None

    @staticmethod
    def _p2a_response_spans(response_mask: list[Any]) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        start: int | None = None
        for idx, value in enumerate(response_mask):
            is_generated = bool(int(value))
            if is_generated and start is None:
                start = idx
            elif not is_generated and start is not None:
                spans.append((start, idx))
                start = None
        if start is not None:
            spans.append((start, len(response_mask)))
        return spans

    @staticmethod
    def _serialize_p2a_tool_call(tool_call: Any) -> dict[str, Any]:
        if hasattr(tool_call, "model_dump"):
            return tool_call.model_dump(mode="json")
        if isinstance(tool_call, dict):
            return copy.deepcopy(tool_call)
        return {"raw": str(tool_call)}

    async def _parse_p2a_step(self, response: str, exit_reason: str) -> tuple[str, list[Any], str | None]:
        try:
            thought, tool_calls = await self.tools_manager.parse_action(model_output=response)
        except Exception as exc:  # noqa: BLE001 - parser failures are trace metadata, not rollout failures
            return "", [], str(exc)
        parse_error = None
        if not tool_calls and exit_reason == "format_error":
            parse_error = "No function call found in the response."
        return thought, list(tool_calls), parse_error

    async def _build_p2a_step_traces(self, interaction_result: dict[str, Any]) -> list[dict[str, Any]]:
        rollout_cache = interaction_result.get("rollout_cache")
        if not isinstance(rollout_cache, dict):
            return []

        spans = self._p2a_response_spans(rollout_cache.get("response_mask") or [])
        span_idx = 0
        traces: list[dict[str, Any]] = []
        for step in interaction_result.get("trajectory") or []:
            response = getattr(step, "response", "")
            if not response:
                continue
            if span_idx >= len(spans):
                break
            response_start, response_end = spans[span_idx]
            span_idx += 1

            thought, tool_calls, parse_error = await self._parse_p2a_step(
                response=response,
                exit_reason=getattr(step, "exit_reason", ""),
            )
            traces.append(
                {
                    "step_idx": int(getattr(step, "step_idx", len(traces) + 1)),
                    "response_start": response_start,
                    "response_end": response_end,
                    "thought": thought,
                    "tool_calls": [self._serialize_p2a_tool_call(tc) for tc in tool_calls],
                    "parse_error": parse_error,
                }
            )
        return traces

    async def _attach_p2a_extra_fields(self, interaction_result: dict[str, Any]) -> None:
        rollout_cache = interaction_result.get("rollout_cache")
        if not isinstance(rollout_cache, dict):
            return

        extra_fields = rollout_cache.setdefault("extra_fields", {})
        instance_id = self._extract_p2a_instance_id()
        if instance_id:
            extra_fields.setdefault("instance_id", instance_id)

        if not self._p2a_trace_enabled():
            return

        trajectory = interaction_result.get("trajectory") or []
        responses = [step.response for step in trajectory if getattr(step, "response", "")]
        extra_fields["response_text"] = "\n".join(responses)
        traces = await self._build_p2a_step_traces(interaction_result)
        if traces:
            extra_fields["p2a_step_traces"] = traces

    async def convert_to_agent_output(self, interaction_result: dict[str, Any]):
        await self._attach_p2a_extra_fields(interaction_result)
        return await super().convert_to_agent_output(interaction_result)
