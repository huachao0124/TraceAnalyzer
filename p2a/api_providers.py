"""Provider loading for API-backed Uni-Agent evaluation.

The public path uses Uni-Agent's OpenAI-compatible chat model. The internal API
path uses a tracked adapter while keeping the private client and credentials in
an ignored module under ``.secrets/``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any
import uuid


OPENAI_COMPATIBLE = "openai_compatible"
INTERNAL_API = "internal_api"
SUPPORTED_PROVIDER_SOURCES = {OPENAI_COMPATIBLE, INTERNAL_API}
REQUIRED_MODEL_METHODS = (
    "set_tools_schemas",
    "prepare_rollout_cache",
    "append_messages_to_rollout_cache",
    "query",
)


class ProviderLoadError(RuntimeError):
    """Raised when a provider adapter cannot be loaded or is malformed."""


def normalize_provider_config(provider_cfg: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(provider_cfg or {})
    cfg.setdefault("source", OPENAI_COMPATIBLE)
    source = str(cfg["source"])
    if source not in SUPPORTED_PROVIDER_SOURCES:
        supported = ", ".join(sorted(SUPPORTED_PROVIDER_SOURCES))
        raise ValueError(
            f"Unsupported provider.source {source!r}; expected one of: {supported}"
        )
    cfg["source"] = source
    return cfg


def provider_source(provider_cfg: dict[str, Any] | None) -> str:
    return normalize_provider_config(provider_cfg)["source"]


def resolve_adapter_path(adapter: str | Path, *, repo_root: Path | None = None) -> Path:
    path = Path(adapter).expanduser()
    if path.is_absolute():
        return path
    return (repo_root or Path.cwd()) / path


def load_internal_adapter(
    provider_cfg: dict[str, Any], *, repo_root: Path | None = None
) -> ModuleType:
    cfg = normalize_provider_config(provider_cfg)
    if cfg["source"] != INTERNAL_API:
        raise ProviderLoadError(
            f"load_internal_adapter only supports provider.source={INTERNAL_API!r}"
        )
    adapter = cfg.get("adapter")
    if adapter is None:
        from p2a import internal_api_adapter

        return internal_api_adapter

    adapter_path = resolve_adapter_path(adapter, repo_root=repo_root)
    if not adapter_path.is_file():
        raise ProviderLoadError(
            "internal_api provider custom adapter was not found at "
            f"{adapter_path}. Omit provider.adapter to use the tracked adapter, or expose "
            "make_model(model_cfg, provider_cfg) from that file."
        )

    module_name = f"p2a_internal_api_adapter_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, adapter_path)
    if spec is None or spec.loader is None:
        raise ProviderLoadError(
            f"Could not import internal_api adapter from {adapter_path}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_provider_available(
    provider_cfg: dict[str, Any], *, repo_root: Path | None = None
) -> None:
    cfg = normalize_provider_config(provider_cfg)
    if cfg["source"] == INTERNAL_API:
        module = load_internal_adapter(cfg, repo_root=repo_root)
        checker = getattr(module, "check_available", None)
        if callable(checker):
            try:
                try:
                    checker(provider_cfg=cfg, repo_root=repo_root)
                except TypeError as keyword_error:
                    try:
                        checker(cfg)
                    except TypeError:
                        raise keyword_error
            except ProviderLoadError:
                raise
            except Exception as exc:
                raise ProviderLoadError(str(exc)) from exc


def _adapter_factory(module: ModuleType):
    for name in ("make_model", "create_model", "build_model"):
        factory = getattr(module, name, None)
        if callable(factory):
            return factory
    raise ProviderLoadError(
        "internal_api adapter must expose make_model(model_cfg, provider_cfg) "
        "or a compatible create_model/build_model factory"
    )


def _call_factory(
    factory: Any,
    model_cfg: dict[str, Any],
    provider_cfg: dict[str, Any],
    *,
    repo_root: Path | None = None,
) -> Any:
    try:
        return factory(
            model_cfg=model_cfg, provider_cfg=provider_cfg, repo_root=repo_root
        )
    except TypeError as keyword_error:
        try:
            return factory(model_cfg=model_cfg, provider_cfg=provider_cfg)
        except TypeError:
            try:
                return factory(model_cfg, provider_cfg)
            except TypeError:
                raise keyword_error


def _validate_model_protocol(model: Any, *, source: str) -> None:
    missing = [
        name
        for name in REQUIRED_MODEL_METHODS
        if not callable(getattr(model, name, None))
    ]
    if missing:
        raise ProviderLoadError(
            f"{source} adapter returned {type(model).__name__}, missing chat-model methods: {', '.join(missing)}"
        )


def _make_openai_compatible_model(model_cfg: dict[str, Any]) -> Any:
    """Build the upstream model and apply P2A-only HTTP transport settings."""
    from uni_agent.interaction import OpenAICompatibleChatModel

    cfg = dict(model_cfg)
    proxy = str(cfg.pop("proxy", "") or "").strip()
    model = OpenAICompatibleChatModel(**cfg)
    if not proxy:
        return model

    from openai import AsyncOpenAI, DefaultAsyncHttpxClient

    http_client = DefaultAsyncHttpxClient(proxy=proxy, timeout=model.timeout)
    model.client = AsyncOpenAI(
        api_key=model.api_key,
        base_url=model.base_url,
        timeout=model.timeout,
        http_client=http_client,
    )
    model._p2a_http_client = http_client
    return model


def _usage_number(payload: dict[str, Any], *keys: str) -> int | float:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int | float):
            return value
    return 0


def _accumulate_token_usage(
    rollout_cache: dict[str, Any] | None, generation_info: dict[str, Any]
) -> None:
    if rollout_cache is None:
        return
    usage = rollout_cache.setdefault("token_usage", {})
    input_tokens = _usage_number(generation_info, "input_tokens", "prompt_tokens")
    output_tokens = _usage_number(generation_info, "output_tokens", "completion_tokens")
    reasoning_tokens = _usage_number(generation_info, "reasoning_tokens")
    cache_hit_tokens = _usage_number(
        generation_info, "cache_hit_tokens", "cached_tokens"
    )
    cache_write_tokens = _usage_number(generation_info, "cache_write_tokens")

    usage["input_tokens"] = usage.get("input_tokens", 0) + input_tokens
    usage["output_tokens"] = usage.get("output_tokens", 0) + output_tokens
    usage["reasoning_tokens"] = usage.get("reasoning_tokens", 0) + reasoning_tokens
    usage["cache_hit_tokens"] = usage.get("cache_hit_tokens", 0) + cache_hit_tokens
    usage["cache_write_tokens"] = (
        usage.get("cache_write_tokens", 0) + cache_write_tokens
    )
    if isinstance(generation_info.get("cost"), int | float):
        usage["cost"] = usage.get("cost", 0.0) + generation_info["cost"]


class MeteredChatModel:
    """Small protocol wrapper that records token/cache/cost totals in rollout_cache."""

    def __init__(self, inner: Any):
        self.inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def set_tools_schemas(self, tools_schemas: list[dict]) -> None:
        self.inner.set_tools_schemas(tools_schemas)

    async def prepare_rollout_cache(
        self, messages: list[dict[str, str]]
    ) -> dict[str, Any]:
        cache = await self.inner.prepare_rollout_cache(messages)
        if cache is None:
            cache = {}
        cache.setdefault("metrics", {})
        cache.setdefault("token_usage", {})
        return cache

    async def append_messages_to_rollout_cache(
        self,
        new_messages: list[dict[str, Any]],
        rollout_cache: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        return await self.inner.append_messages_to_rollout_cache(
            new_messages, rollout_cache
        )

    async def query(
        self,
        messages: list[dict[str, str]],
        rollout_cache: dict[str, Any] | None,
        **kwargs: Any,
    ) -> tuple[str, list[dict], dict[str, Any], dict[str, Any]]:
        text, tool_calls, cache, generation_info = await self.inner.query(
            messages, rollout_cache, **kwargs
        )
        if not isinstance(generation_info, dict):
            generation_info = {}
        _accumulate_token_usage(cache, generation_info)
        return text, tool_calls, cache, generation_info


def make_chat_model(
    model_cfg: dict[str, Any],
    provider_cfg: dict[str, Any] | None = None,
    *,
    repo_root: Path | None = None,
) -> Any:
    cfg = normalize_provider_config(provider_cfg)
    source = cfg["source"]
    if source == OPENAI_COMPATIBLE:
        model = _make_openai_compatible_model(model_cfg)
    elif source == INTERNAL_API:
        module = load_internal_adapter(cfg, repo_root=repo_root)
        model = _call_factory(
            _adapter_factory(module), model_cfg, cfg, repo_root=repo_root
        )
    else:
        raise ValueError(f"Unsupported provider.source {source!r}")
    _validate_model_protocol(model, source=source)
    return MeteredChatModel(model)
