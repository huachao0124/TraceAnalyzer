"""OpenAI-compatible chat client for standalone P2A evaluation."""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI


class OpenAICompatibleChatModel:
    """Expose the stable standalone-eval model protocol over AsyncOpenAI."""

    _TOP_LEVEL_FIELDS = frozenset(
        {
            "temperature",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "max_tokens",
            "max_completion_tokens",
            "stop",
            "n",
            "seed",
            "logprobs",
            "top_logprobs",
            "logit_bias",
            "user",
        }
    )

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model_name: str,
        sampling_params: dict[str, Any] | None = None,
        timeout: float = 300,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.sampling_params = dict(sampling_params or {})
        self.timeout = timeout
        self.tools_schemas: list[dict] | None = None
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=timeout,
        )

    def set_tools_schemas(self, tools_schemas: list[dict]) -> None:
        self.tools_schemas = tools_schemas

    async def prepare_rollout_cache(
        self,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {"metrics": {}, "token_usage": {}}

    async def append_messages_to_rollout_cache(
        self,
        new_messages: list[dict[str, Any]],
        rollout_cache: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        return rollout_cache

    @staticmethod
    def _api_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized = []
        for message in messages:
            item: dict[str, Any] = {"role": message["role"]}
            if message.get("content") is not None:
                item["content"] = message["content"]
            if message["role"] == "assistant" and message.get("tool_calls"):
                item["tool_calls"] = message["tool_calls"]
            if message["role"] == "tool":
                if message.get("tool_call_id") is not None:
                    item["tool_call_id"] = message["tool_call_id"]
                if message.get("name") is not None:
                    item["name"] = message["name"]
            normalized.append(item)
        return normalized

    async def query(
        self,
        messages: list[dict[str, Any]],
        rollout_cache: dict[str, Any] | None,
        **kwargs: Any,
    ) -> tuple[str, list[dict], dict[str, Any], dict[str, int]]:
        sampling_params = dict(
            kwargs.get("sampling_params", self.sampling_params) or {}
        )
        top_level = {
            key: value
            for key, value in sampling_params.items()
            if key in self._TOP_LEVEL_FIELDS
        }
        extra_body = {
            key: value
            for key, value in sampling_params.items()
            if key not in self._TOP_LEVEL_FIELDS
        }
        completion = await self.client.chat.completions.create(
            model=self.model_name,
            messages=self._api_messages(messages),
            tools=self.tools_schemas,
            extra_body=extra_body or None,
            **top_level,
        )
        response = completion.choices[0].message
        tool_calls = [
            {
                "id": call.id,
                "type": call.type,
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in (response.tool_calls or [])
        ]
        usage = completion.usage
        generation_info = {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
        }
        return (
            response.content or "",
            tool_calls,
            rollout_cache or {"metrics": {}, "token_usage": {}},
            generation_info,
        )

    async def aclose(self) -> None:
        close = getattr(self.client, "close", None)
        if close is not None:
            await close()
