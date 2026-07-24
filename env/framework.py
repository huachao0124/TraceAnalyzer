"""P2A metadata extension for Uni-Agent's Agent Framework rollout adapter."""

from __future__ import annotations

from typing import Any

from uni_agent.framework.framework import OpenAICompatibleAgentFramework


def _instance_id(sample_fields: dict[str, object]) -> str | None:
    extra_info = sample_fields.get("extra_info")
    if isinstance(extra_info, dict):
        direct = extra_info.get("instance_id")
        if isinstance(direct, str) and direct:
            return direct
    tools_kwargs = sample_fields.get("tools_kwargs")
    if not isinstance(tools_kwargs, dict) and isinstance(extra_info, dict):
        tools_kwargs = extra_info.get("tools_kwargs")
    if not isinstance(tools_kwargs, dict):
        return None
    task = tools_kwargs.get("task")
    if isinstance(task, dict):
        metadata = task.get("metadata")
    else:
        reward = tools_kwargs.get("reward")
        metadata = reward.get("metadata") if isinstance(reward, dict) else None
    value = metadata.get("instance_id") if isinstance(metadata, dict) else None
    return value if isinstance(value, str) and value else None


def _generated_spans(response_mask: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(response_mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(response_mask)))
    return spans


class P2AAgentFramework(OpenAICompatibleAgentFramework):
    """Attach stable instance and decoded response fields to TransferQueue rows."""

    def _decode(self, token_ids: list[int]) -> str:
        processor = self._processor
        tokenizer: Any = getattr(processor, "tokenizer", processor)
        decode = getattr(tokenizer, "decode", None)
        if not callable(decode):
            return ""
        return str(decode(token_ids, skip_special_tokens=False))

    def _trajectory_to_tq_field_and_tag(
        self,
        *,
        trajectory,
        sample_fields,
        session_index,
        global_steps,
        uid,
    ):
        extra_fields = dict(trajectory.extra_fields or {})
        instance_id = _instance_id(sample_fields)
        if instance_id:
            extra_fields.setdefault("instance_id", instance_id)

        spans = _generated_spans(list(trajectory.response_mask))
        step_traces = []
        responses = []
        for step_index, (start, end) in enumerate(spans, start=1):
            response_text = self._decode(list(trajectory.response_ids[start:end]))
            responses.append(response_text)
            step_traces.append(
                {
                    "step_idx": step_index,
                    "response_start": start,
                    "response_end": end,
                    "response_text": response_text,
                }
            )
        if responses:
            extra_fields.setdefault("response_text", "\n".join(responses))
            extra_fields.setdefault("p2a_step_traces", step_traces)

        trajectory.extra_fields = extra_fields
        return super()._trajectory_to_tq_field_and_tag(
            trajectory=trajectory,
            sample_fields=sample_fields,
            session_index=session_index,
            global_steps=global_steps,
            uid=uid,
        )
