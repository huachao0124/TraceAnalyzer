import asyncio

import openai
import pytest
import uni_agent.interaction as uni_agent_interaction

import p2a.internal_api_adapter as internal_api_adapter
from p2a.api_providers import (
    ProviderLoadError,
    check_provider_available,
    load_internal_adapter,
    make_chat_model,
    normalize_provider_config,
)


def _write_fake_api_module(path):
    path.write_text(
        """
class Response:
    def __init__(self, index):
        self.index = index

    status_code = 200
    headers = {"x-usage-prompt-tokens": "2"}

    def json(self):
        return {
            "code": 0,
            "request_id": f"resp_{self.index}",
            "account_id": "acct-1",
            "answer": [
                {
                    "type": "text",
                    "value": "inspect",
                },
                {
                    "type": "tool_calls",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "str_replace_editor",
                                "arguments": {"command": "view", "path": "/testbed/a.py"},
                            },
                        }
                    ],
                },
            ],
            "cost_info": {"completion_tokens": 3, "cost": 0.01},
        }


class RawUsageDetails:
    reasoning_tokens = 7


class RawUsage:
    input_tokens = 11
    output_tokens = 13
    total_tokens = 24
    output_tokens_details = RawUsageDetails()


class RawSummaryBlock:
    type = "summary_text"
    text = "reason about seed"


class RawReasoningItem:
    type = "reasoning"
    summary = [RawSummaryBlock()]


class RawTextBlock:
    type = "output_text"
    text = "inspect"


class RawMessageItem:
    type = "message"
    content = [RawTextBlock()]


class RawFunctionCall:
    type = "function_call"
    call_id = "call-raw"
    name = "str_replace_editor"
    arguments = '{"command":"view","path":"/testbed/a.py"}'


class RawResponse:
    id = "resp_raw"
    model = "doubao-seed-2-0-lite"
    usage = RawUsage()
    output = [RawReasoningItem(), RawMessageItem(), RawFunctionCall()]
    _x_usage = {
        "prompt_tokens": 11,
        "completion_tokens": 13,
        "total_tokens": 24,
        "cache_hit_tokens": 2,
        "cache_write_tokens": 0,
        "cost": 5,
    }
    _x_account_id = "acct-raw"


class Api:
    MODEL_CONFIGS = {
        "passthrough_models": [
            "deepseek-v4-flash-passthrough",
            "doubao-seed-2-0-lite-passthrough",
        ],
        "passthrough_chat_completions_models": {"deepseek-v4-flash-passthrough"},
        "passthrough_extra_body_map": {
            "deepseek-v4-flash-passthrough": {
                "max_completion_tokens": 384000,
                "temperature": 0.7,
                "thinking": {"max_completion_tokens": 128000},
            },
            "doubao-seed-2-0-lite-passthrough": {},
        },
    }

    def __init__(self, host, user_name, user_token):
        self.host = host
        self.user_name = user_name
        self.user_token = user_token
        self.calls = []
        self.raw_calls = []
        self.requests = []

    def set_retry_config(self, **kwargs):
        self.retry_config = kwargs

    def _normalize_model_name(self, model_name):
        return model_name

    def _make_request_with_retry(self, method, url, **kwargs):
        self.requests.append({"method": method, "url": url, **kwargs})
        return Response(len(self.calls))

    def call_data_eval(self, model_name, prompt, **kwargs):
        self.calls.append({
            "model_name": model_name,
            "prompt": prompt,
            **kwargs,
            "save_id_snapshot": dict(kwargs.get("save_id") or {}),
        })
        body = {
            "model": model_name,
            **self.MODEL_CONFIGS["passthrough_extra_body_map"][model_name],
        }
        return self._make_request_with_retry("POST", "http://provider.example", json=body)

    def _handle_openai_passthrough_request(self, model_name, prompt, history, tools=None, tools_mode=False, save_id=None):
        self.raw_calls.append({
            "model_name": model_name,
            "prompt": prompt,
            "history": history,
            "tools": tools,
            "tools_mode": tools_mode,
            "save_id_snapshot": dict(save_id or {}),
        })
        return RawResponse()


HOST = "http://internal.example"
user_name = "demo-user"
user_token = "demo-token"
""",
        encoding="utf-8",
    )


def test_internal_api_defaults_to_tracked_adapter():
    cfg = normalize_provider_config({"source": "internal_api"})

    assert "adapter" not in cfg
    assert "api_module" not in cfg
    assert load_internal_adapter(cfg).__name__ == "p2a.internal_api_adapter"


def test_openai_compatible_proxy_is_applied_outside_uni_agent(monkeypatch):
    class FakeUpstreamModel:
        def __init__(self, **data):
            self.__dict__.update(data)
            self.client = "upstream-default"

        def set_tools_schemas(self, _tools_schemas):
            pass

        async def prepare_rollout_cache(self, _messages):
            return {}

        async def append_messages_to_rollout_cache(self, _new_messages, rollout_cache):
            return rollout_cache

        async def query(self, _messages, rollout_cache, **_kwargs):
            return "", [], rollout_cache, {}

    class FakeHttpClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        uni_agent_interaction,
        "OpenAICompatibleChatModel",
        FakeUpstreamModel,
    )
    monkeypatch.setattr(openai, "DefaultAsyncHttpxClient", FakeHttpClient)
    monkeypatch.setattr(openai, "AsyncOpenAI", FakeAsyncOpenAI)

    model = make_chat_model(
        {
            "base_url": "https://example.test/v1",
            "api_key": "secret",
            "model_name": "demo-model",
            "timeout": 30,
            "proxy": "http://proxy.example:8080",
        },
        {"source": "openai_compatible"},
    )

    assert not hasattr(model.inner, "proxy")
    assert model.inner._p2a_http_client.kwargs == {
        "proxy": "http://proxy.example:8080",
        "timeout": 30,
    }
    assert model.inner.client.kwargs["http_client"] is model.inner._p2a_http_client


def test_internal_api_default_checks_private_api_module(tmp_path):
    cfg = {"source": "internal_api", "api_module": "missing_internal_api_eval.py"}

    with pytest.raises(
        ProviderLoadError,
        match="internal_api provider requires private API module",
    ):
        check_provider_available(cfg, repo_root=tmp_path)


def test_internal_api_uses_env_api_module_when_config_omits_path(tmp_path, monkeypatch):
    api_module = tmp_path / "internal_api_eval.py"
    _write_fake_api_module(api_module)
    monkeypatch.setenv("P2A_INTERNAL_API_MODULE", str(api_module))

    check_provider_available({"source": "internal_api"}, repo_root=tmp_path)
    model = make_chat_model(
        {"model_name": "deepseek-v4-flash-passthrough"},
        {"source": "internal_api"},
        repo_root=tmp_path,
    )

    assert model.inner.api.host == "http://internal.example"


def test_internal_api_custom_adapter_missing_fails_clearly(tmp_path):
    missing = tmp_path / "missing_adapter.py"

    with pytest.raises(ProviderLoadError, match="custom adapter was not found"):
        check_provider_available(
            {"source": "internal_api", "adapter": str(missing)},
            repo_root=tmp_path,
        )


def test_tracked_internal_api_adapter_queries_private_api_module(tmp_path, monkeypatch):
    api_module = tmp_path / "internal_api_eval.py"
    _write_fake_api_module(api_module)

    async def call_now(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(internal_api_adapter.asyncio, "to_thread", call_now)
    model = make_chat_model(
        {
            "model_name": "deepseek-v4-flash-passthrough",
            "sampling_params": {"max_tokens": 16, "temperature": 0.2, "top_p": 0.95},
        },
        {"source": "internal_api", "api_module": api_module.name},
        repo_root=tmp_path,
    )
    model.set_tools_schemas(
        [{"type": "function", "function": {"name": "str_replace_editor"}}]
    )

    async def _query():
        cache = await model.prepare_rollout_cache(
            [{"role": "user", "content": "Fix the bug."}]
        )
        return await model.query(
            [{"role": "user", "content": "Fix the bug."}],
            cache,
        )

    content, tool_calls, cache, info = asyncio.run(_query())

    assert content == "inspect"
    assert tool_calls[0]["function"]["name"] == "str_replace_editor"
    assert cache["internal_api_save_id"]["response_id"] == "resp_1"
    assert cache["internal_api_save_id"]["account_id"] == "acct-1"
    assert cache["token_usage"]["input_tokens"] == 2.0
    assert cache["token_usage"]["output_tokens"] == 3
    assert cache["token_usage"]["cost"] == 0.01
    call = model.inner.api.calls[0]
    assert call["save_id_snapshot"] == {}
    assert call["prompt"] == "Fix the bug."
    assert call["history"] == []
    assert call["tools"] == [
        {"type": "function", "function": {"name": "str_replace_editor"}}
    ]
    request_body = model.inner.api.requests[0]["json"]
    assert request_body["max_completion_tokens"] == 16
    assert request_body["thinking"]["max_completion_tokens"] == 16
    assert request_body["temperature"] == 0.2
    assert request_body["top_p"] == 0.95


def test_internal_api_passes_save_id_on_follow_up_requests(tmp_path, monkeypatch):
    api_module = tmp_path / "internal_api_eval.py"
    _write_fake_api_module(api_module)

    async def call_now(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(internal_api_adapter.asyncio, "to_thread", call_now)
    model = make_chat_model(
        {"model_name": "deepseek-v4-flash-passthrough"},
        {"source": "internal_api", "api_module": api_module.name},
        repo_root=tmp_path,
    )

    async def _queries():
        cache = await model.prepare_rollout_cache(
            [{"role": "user", "content": "Fix the bug."}]
        )
        await model.query([{"role": "user", "content": "Fix the bug."}], cache)
        await model.query(
            [
                {"role": "user", "content": "Fix the bug."},
                {"role": "assistant", "content": "inspect"},
                {"role": "user", "content": "Continue."},
            ],
            cache,
        )
        return cache

    asyncio.run(_queries())

    assert model.inner.api.calls[0]["save_id_snapshot"] == {}
    assert model.inner.api.calls[1]["save_id_snapshot"]["account_id"] == "acct-1"


def test_internal_api_responses_model_preserves_raw_reasoning(tmp_path, monkeypatch):
    api_module = tmp_path / "internal_api_eval.py"
    _write_fake_api_module(api_module)

    async def call_now(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(internal_api_adapter.asyncio, "to_thread", call_now)
    model = make_chat_model(
        {"model_name": "doubao-seed-2-0-lite-passthrough"},
        {"source": "internal_api", "api_module": api_module.name},
        repo_root=tmp_path,
    )
    model.set_tools_schemas(
        [{"type": "function", "function": {"name": "str_replace_editor"}}]
    )

    async def _query():
        cache = await model.prepare_rollout_cache(
            [{"role": "user", "content": "Fix the bug."}]
        )
        return await model.query(
            [{"role": "user", "content": "Fix the bug."}],
            cache,
        )

    content, tool_calls, cache, info = asyncio.run(_query())

    assert content == "inspect"
    assert tool_calls[0]["id"] == "call-raw"
    assert tool_calls[0]["function"]["name"] == "str_replace_editor"
    assert info["reasoning_content"] == "reason about seed"
    assert info["reasoning_tokens"] == 7
    assert info["cache_hit_tokens"] == 2
    assert cache["internal_api_save_id"]["response_id"] == "resp_raw"
    assert cache["internal_api_save_id"]["account_id"] == "acct-raw"
    assert cache["internal_api_assistant_metadata"]["1"]["reasoning_content"] == "reason about seed"
    assert model.inner.api.calls == []
    assert model.inner.api.raw_calls[0]["prompt"] == "Fix the bug."
