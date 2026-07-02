"""Native message shaping helpers for the internal API adapter.

The functions in this module are deliberately free of credentials and network
calls. The tracked adapter loads the private API client from ``.secrets/`` and
uses these helpers to keep Uni-Agent rollouts in each provider family's native
tool-call format.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any
import uuid


_PASSTHROUGH_ASSISTANT_FORMAT_KEYS = (
    "passthrough_models",
    "anthropic_passthrough_models",
    "aws_anthropic_passthrough_models",
    "aws_bedrock_passthrough_models",
    "ali_passthrough_models",
    "moonshot_passthrough_models",
    "yuewen_passthrough_models",
    "gemini_passthrough_models",
    "naci_passthrough_models",
    "google_passthrough_models",
    "zhipu_passthrough_models",
    "minimax_passthrough_models",
)

_MERGED_TOOL_RESULT_KEYS = (
    "anthropic_passthrough_models",
    "aws_anthropic_passthrough_models",
    "aws_bedrock_passthrough_models",
    "minimax_passthrough_models",
    "naci_passthrough_models",
    "google_passthrough_models",
)


def _model_configs(api_module: Any) -> dict[str, Any]:
    api_cls = getattr(api_module, "Api", None)
    configs = getattr(api_cls, "MODEL_CONFIGS", {}) if api_cls is not None else {}
    return configs if isinstance(configs, dict) else {}


def _contains_model(api_module: Any, model_name: str, keys: tuple[str, ...]) -> bool:
    configs = _model_configs(api_module)
    for key in keys:
        values = configs.get(key) or []
        if model_name in values:
            return True
    return False


def uses_passthrough_assistant_format(model_name: str, api_module: Any) -> bool:
    """Return whether assistant tool-call turns must use typed content blocks."""

    return _contains_model(api_module, model_name, _PASSTHROUGH_ASSISTANT_FORMAT_KEYS)


def requires_merged_tool_results(model_name: str, api_module: Any) -> bool:
    """Return whether parallel tool results must be one multi-part tool turn."""

    return _contains_model(api_module, model_name, _MERGED_TOOL_RESULT_KEYS)


def uses_openai_responses_api(model_name: str, api_module: Any) -> bool:
    """Return whether the model can use the internal Responses-API chain."""

    configs = _model_configs(api_module)
    if model_name not in (configs.get("passthrough_models") or []):
        return False
    return model_name not in (
        configs.get("passthrough_chat_completions_models") or set()
    )


def normalize_model_name(api: Any, model_name: str) -> str:
    normalizer = getattr(api, "_normalize_model_name", None)
    if callable(normalizer):
        return str(normalizer(model_name))
    return model_name


def _stringify_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(
                    str(
                        item.get("text")
                        or item.get("value")
                        or item.get("content")
                        or ""
                    )
                )
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def _json_arguments(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value if value is not None else {}, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps({"_value": str(value)}, ensure_ascii=False)


_TEXT_TOOL_ARGUMENT_KEYS = {
    "command",
    "path",
    "file",
    "old_str",
    "new_str",
    "insert_line",
    "content",
}
_JSON_TEXT_TOOL_ARGUMENT_KEYS = {"view_range"}


def _coerce_text_wrapped_tool_value(key: str, value: Any) -> Any:
    if not isinstance(value, dict) or "$text" not in value:
        return value
    text = value.get("$text")
    if not isinstance(text, str):
        return value
    if key in _JSON_TEXT_TOOL_ARGUMENT_KEYS:
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError, TypeError):
            return text
    if key in _TEXT_TOOL_ARGUMENT_KEYS:
        return text
    return value


def normalize_tool_arguments(name: Any, arguments: Any) -> Any:
    if isinstance(arguments, str):
        return arguments
    if not isinstance(arguments, dict):
        return arguments
    return {
        key: _coerce_text_wrapped_tool_value(str(key), value)
        for key, value in arguments.items()
    }


def _normalize_tool_arguments_for_json(name: Any, arguments: Any) -> Any:
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except (json.JSONDecodeError, ValueError, TypeError):
            return arguments
        return normalize_tool_arguments(name, parsed)
    return normalize_tool_arguments(name, arguments)


def _metadata_for_index(
    metadata_by_index: dict[Any, Any] | None, index: int
) -> dict[str, Any]:
    if not isinstance(metadata_by_index, dict):
        return {}
    for key in (index, str(index)):
        value = metadata_by_index.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _reasoning_blocks_from_metadata(metadata: dict[str, Any]) -> list[dict[str, str]]:
    blocks = metadata.get("reasoning_blocks")
    normalized = []
    if isinstance(blocks, list):
        for block in blocks:
            if not isinstance(block, dict):
                continue
            value = str(block.get("value") or "")
            if not value and not block.get("signature"):
                continue
            item = {"type": "reasoning", "value": value}
            if block.get("signature"):
                item["signature"] = str(block["signature"])
            normalized.append(item)
    return normalized


def _text_blocks_from_metadata(
    metadata: dict[str, Any], fallback: str
) -> list[dict[str, str]]:
    blocks = metadata.get("text_blocks")
    normalized = []
    if isinstance(blocks, list):
        for block in blocks:
            if not isinstance(block, dict):
                continue
            value = str(block.get("value") or "")
            if not value:
                continue
            item = {"type": "text", "value": value}
            if block.get("signature"):
                item["signature"] = str(block["signature"])
            normalized.append(item)
    if normalized:
        return normalized
    return [{"type": "text", "value": fallback}] if fallback else []


def to_prompt_history(
    messages: list[dict[str, Any]],
    *,
    model_name: str,
    api_module: Any,
    save_id: dict[str, Any] | None = None,
    assistant_metadata_by_index: dict[Any, Any] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Convert Uni-Agent messages into ``Api.call_data_eval`` prompt/history.

    ``call_data_eval`` takes only a trailing user turn as ``prompt``. A trailing
    tool turn is part of the structured history and must keep its
    ``tool_call_id`` so provider validators can match it to the previous
    assistant tool call.
    """

    use_typed_assistant = uses_passthrough_assistant_format(model_name, api_module)
    chain_seeded = uses_openai_responses_api(model_name, api_module) and bool(
        (save_id or {}).get("response_id")
    )
    merge_tools = requires_merged_tool_results(model_name, api_module) or chain_seeded

    prompt = ""
    source_messages = list(messages)
    if source_messages and source_messages[-1].get("role") == "user":
        prompt = _stringify_content(source_messages[-1].get("content"))
        source_messages = source_messages[:-1]

    history: list[dict[str, Any]] = []
    for message_index, message in enumerate(source_messages):
        role = message.get("role")
        if role in {"system", "user"}:
            history.append(
                {"role": role, "content": _stringify_content(message.get("content"))}
            )
            continue

        if role == "assistant":
            metadata = _metadata_for_index(assistant_metadata_by_index, message_index)
            content = _stringify_content(message.get("content"))
            tool_calls = message.get("tool_calls") or []
            normalized_tool_calls = []
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                fn = tool_call.get("function") or {}
                normalized_tool_calls.append(
                    {
                        "id": tool_call.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                        "type": tool_call.get("type", "function"),
                        "function": {
                            "name": fn.get("name", ""),
                            "arguments": _json_arguments(
                                _normalize_tool_arguments_for_json(
                                    fn.get("name", ""),
                                    fn.get("arguments", {}),
                                )
                            ),
                        },
                    }
                )
                if tool_call.get("signature"):
                    normalized_tool_calls[-1]["signature"] = str(tool_call["signature"])

            reasoning = (
                message.get("reasoning_content")
                or metadata.get("reasoning_content")
                or ""
            )
            reasoning_blocks = _reasoning_blocks_from_metadata(metadata)
            if not reasoning and reasoning_blocks:
                reasoning = "\n".join(
                    block["value"] for block in reasoning_blocks if block.get("value")
                )
            if normalized_tool_calls and use_typed_assistant:
                content_blocks: list[dict[str, Any]] = list(reasoning_blocks)
                if reasoning and not reasoning_blocks:
                    content_blocks.append(
                        {"type": "reasoning", "value": str(reasoning)}
                    )
                content_blocks.extend(_text_blocks_from_metadata(metadata, content))
                content_blocks.append(
                    {"type": "tool_calls", "tool_calls": normalized_tool_calls}
                )
                entry: dict[str, Any] = {
                    "role": "assistant",
                    "content": content_blocks,
                    "tool_calls": normalized_tool_calls,
                }
                if reasoning:
                    entry["reasoning_content"] = str(reasoning)
                history.append(entry)
            elif normalized_tool_calls:
                entry = {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": normalized_tool_calls,
                }
                if reasoning:
                    entry["reasoning_content"] = str(reasoning)
                history.append(entry)
            else:
                if use_typed_assistant and reasoning_blocks:
                    typed_content: list[dict[str, Any]] = list(reasoning_blocks)
                    typed_content.extend(_text_blocks_from_metadata(metadata, content))
                    entry = {"role": "assistant", "content": typed_content}
                else:
                    entry = {"role": "assistant", "content": content}
                if reasoning:
                    entry["reasoning_content"] = str(reasoning)
                history.append(entry)
            continue

        if role == "tool":
            tool_value = _stringify_content(message.get("content"))
            part = {
                "type": "text",
                "tool_call_id": message.get("tool_call_id", ""),
                "name": message.get("name", ""),
                "value": tool_value,
                "text": tool_value,
            }
            if (
                merge_tools
                and history
                and history[-1].get("role") == "tool"
                and isinstance(history[-1].get("content"), list)
            ):
                history[-1]["content"].append(part)
            else:
                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.get("tool_call_id", ""),
                        "content": [part],
                    }
                )
            continue

        history.append(
            {
                "role": role or "user",
                "content": _stringify_content(message.get("content")),
            }
        )

    return prompt, history


@dataclass(frozen=True)
class _NativeToolDialect:
    prefix: str
    block_re: re.Pattern[str]
    invoke_re: re.Pattern[str]
    param_re: re.Pattern[str]
    param_string_default: bool


_DSML_BAR = "｜"
_DSML_OPEN = f"<{_DSML_BAR}{_DSML_BAR}DSML{_DSML_BAR}{_DSML_BAR}"
_DSML_CLOSE = f"</{_DSML_BAR}{_DSML_BAR}DSML{_DSML_BAR}{_DSML_BAR}"

_DEEPSEEK_DIALECT = _NativeToolDialect(
    prefix=_DSML_OPEN,
    block_re=re.compile(
        re.escape(_DSML_OPEN)
        + r"tool_calls>([\s\S]*?)(?:"
        + re.escape(_DSML_CLOSE)
        + r"tool_calls>|\Z)",
        re.MULTILINE,
    ),
    invoke_re=re.compile(
        re.escape(_DSML_OPEN)
        + r'invoke name="([^"]+)">([\s\S]*?)(?:'
        + re.escape(_DSML_CLOSE)
        + r"invoke>|\Z)",
        re.MULTILINE,
    ),
    param_re=re.compile(
        re.escape(_DSML_OPEN)
        + r'parameter name="([^"]+)"(?:\s+string="(true|false)")?>([\s\S]*?)'
        + re.escape(_DSML_CLOSE)
        + r"parameter>",
        re.MULTILINE,
    ),
    param_string_default=True,
)

_MINIMAX_OPEN = "<minimax:tool_call>"
_MINIMAX_CLOSE = "</minimax:tool_call>"

_MINIMAX_DIALECT = _NativeToolDialect(
    prefix=_MINIMAX_OPEN,
    block_re=re.compile(
        re.escape(_MINIMAX_OPEN)
        + r"([\s\S]*?)(?:"
        + re.escape(_MINIMAX_CLOSE)
        + r"|\Z)",
        re.MULTILINE,
    ),
    invoke_re=re.compile(
        r'<invoke name="([^"]+)">([\s\S]*?)(?:</invoke>|\Z)', re.MULTILINE
    ),
    param_re=re.compile(
        r'<parameter name="([^"]+)">([\s\S]*?)</parameter>', re.MULTILINE
    ),
    param_string_default=False,
)


def _coerce_native_value(raw: str, is_string: bool) -> Any:
    raw = raw.strip()
    if is_string:
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def recover_native_tool_calls(text: str) -> tuple[str, list[dict[str, Any]], bool]:
    """Recover provider-native tool DSLs that leaked into assistant text."""

    for dialect in (_DEEPSEEK_DIALECT, _MINIMAX_DIALECT):
        if dialect.prefix not in text:
            continue
        recovered: list[dict[str, Any]] = []
        for block_match in dialect.block_re.finditer(text):
            block_body = block_match.group(1)
            for invoke_match in dialect.invoke_re.finditer(block_body):
                name = invoke_match.group(1).strip()
                invoke_body = invoke_match.group(2)
                if not name:
                    continue
                args: dict[str, Any] = {}
                for param_match in dialect.param_re.finditer(invoke_body):
                    groups = param_match.groups()
                    param_name = param_match.group(1).strip()
                    if len(groups) >= 3:
                        is_string = (
                            param_match.group(2)
                            or ("true" if dialect.param_string_default else "false")
                        ) == "true"
                        raw_value = param_match.group(3)
                    else:
                        is_string = dialect.param_string_default
                        raw_value = param_match.group(2)
                    args[param_name] = _coerce_native_value(raw_value, is_string)
                recovered.append(
                    {
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(
                                normalize_tool_arguments(name, args),
                                ensure_ascii=False,
                            ),
                        },
                    }
                )
        if recovered:
            return dialect.block_re.sub("", text).strip(), recovered, True
    return text, [], False


def _usage_number(payload: dict[str, Any], *keys: str) -> int | float:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int | float):
            return value
    return 0


def _parse_tool_calls(raw_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_calls, list):
        return []
    parsed = []
    for idx, raw in enumerate(raw_calls):
        if not isinstance(raw, dict):
            continue
        fn = raw.get("function") or {}
        parsed_call = {
            "id": raw.get("id") or f"call_{idx}_{uuid.uuid4().hex[:8]}",
            "type": raw.get("type", "function"),
            "function": {
                "name": fn.get("name", ""),
                "arguments": _json_arguments(
                    _normalize_tool_arguments_for_json(
                        fn.get("name", ""),
                        fn.get("arguments", {}),
                    )
                ),
            },
        }
        if raw.get("signature"):
            parsed_call["signature"] = str(raw["signature"])
        parsed.append(parsed_call)
    return parsed


def _text_blocks_for_content(
    content: str, original_blocks: list[dict[str, str]]
) -> list[dict[str, str]]:
    if not content:
        return []
    if len(original_blocks) == 1:
        block: dict[str, str] = {"value": content}
        if original_blocks[0].get("signature"):
            block["signature"] = original_blocks[0]["signature"]
        return [block]
    return [{"value": content}]


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _dump_response_obj(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            pass
    if isinstance(value, dict):
        return value
    return None


def _response_text_values(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    parts: list[str] = []
    for block in values:
        text = (
            block
            if isinstance(block, str)
            else _field(block, "text")
            or _field(block, "value")
            or _field(block, "content")
        )
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return parts


def _usage_detail_number(usage: Any, *keys: str) -> float | int | None:
    current = usage
    for key in keys:
        if current is None:
            return None
        current = _field(current, key)
    if isinstance(current, int | float) and not isinstance(current, bool):
        return current
    return None


def responses_api_response_to_payload(response: Any) -> dict[str, Any]:
    """Convert a raw OpenAI Responses-style object into internal API JSON.

    The private API shim converts Responses objects to a Chat Completions-shaped
    payload before the tracked adapter sees them. Some providers, including
    Doubao Seed, expose thinking text as top-level ``output`` reasoning items,
    so the tracked adapter must parse the raw object when it is available.
    """

    dumped = _dump_response_obj(response)
    output_items = _field(response, "output") or (dumped or {}).get("output") or []
    answer: list[dict[str, Any]] = []
    text_parts: list[str] = []
    text_blocks: list[dict[str, str]] = []
    reasoning_parts: list[str] = []
    reasoning_blocks: list[dict[str, str]] = []
    tool_calls: list[dict[str, Any]] = []

    for item in output_items:
        item_type = _field(item, "type")
        if item_type == "reasoning":
            values = _response_text_values(_field(item, "content"))
            if not values:
                values = _response_text_values(_field(item, "summary"))
            for value in values:
                reasoning_parts.append(value)
                reasoning_blocks.append({"value": value})
            continue

        if item_type == "message":
            values = _response_text_values(_field(item, "content"))
            for value in values:
                text_parts.append(value)
                text_blocks.append({"value": value})
            continue

        if item_type == "function_call":
            tool_calls.append(
                {
                    "id": str(_field(item, "call_id") or _field(item, "id") or ""),
                    "type": "function",
                    "function": {
                        "name": str(_field(item, "name") or ""),
                        "arguments": _json_arguments(
                            _normalize_tool_arguments_for_json(
                                _field(item, "name") or "",
                                _field(item, "arguments"),
                            )
                        ),
                    },
                }
            )

    if reasoning_parts:
        answer.append({"type": "reasoning", "value": "\n".join(reasoning_parts)})
    if text_parts:
        answer.append({"type": "text", "value": "".join(text_parts)})
    if tool_calls:
        answer.append({"type": "tool_calls", "tool_calls": tool_calls})

    usage = _field(response, "usage") or (dumped or {}).get("usage") or {}
    x_usage = _field(response, "_x_usage", {}) or {}
    reasoning_tokens = (
        _usage_detail_number(usage, "output_tokens_details", "reasoning_tokens")
        or _usage_detail_number(usage, "completion_tokens_details", "reasoning_tokens")
        or 0
    )
    input_tokens = x_usage.get("prompt_tokens") or _usage_detail_number(usage, "input_tokens") or 0
    output_tokens = (
        x_usage.get("completion_tokens")
        or _usage_detail_number(usage, "output_tokens")
        or _usage_detail_number(usage, "completion_tokens")
        or 0
    )
    total_tokens = x_usage.get("total_tokens") or _usage_detail_number(usage, "total_tokens") or 0
    cost_info = {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "cache_hit_tokens": x_usage.get("cache_hit_tokens", 0),
        "cache_write_tokens": x_usage.get("cache_write_tokens", 0),
        "total_tokens": total_tokens,
        "cost": x_usage.get("cost", 0),
        "reasoning_tokens": reasoning_tokens,
    }

    account_id = _field(response, "_x_account_id")
    raw_response = dumped if isinstance(dumped, dict) else {}
    if account_id and "account_id" not in raw_response:
        raw_response = dict(raw_response)
        raw_response["account_id"] = account_id

    payload = {
        "code": 0,
        "msg": "",
        "answer": answer,
        "cost_info": cost_info,
        "request_detail": {"response": raw_response},
    }
    response_id = _field(response, "id") or (dumped or {}).get("id")
    if response_id:
        payload["request_id"] = response_id
    model = _field(response, "model") or (dumped or {}).get("model")
    if model:
        payload["model"] = model
    if account_id:
        payload["account_id"] = account_id
    if text_blocks:
        payload["text_blocks"] = text_blocks
    if reasoning_blocks:
        payload["reasoning_blocks"] = reasoning_blocks
    return payload


def parse_internal_response(
    payload: dict[str, Any],
    headers: dict[str, Any] | None = None,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Return ``(content, tool_calls, generation_info)`` from internal API JSON."""

    headers = headers or {}
    text_parts: list[str] = []
    text_blocks: list[dict[str, str]] = []
    reasoning_parts: list[str] = []
    reasoning_blocks: list[dict[str, str]] = []
    tool_calls: list[dict[str, Any]] = []

    answer = payload.get("answer")
    if isinstance(answer, list):
        for item in answer:
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if kind == "text":
                value = str(item.get("value") or "")
                text_parts.append(value)
                if value or item.get("signature"):
                    block = {"value": value}
                    if item.get("signature"):
                        block["signature"] = str(item["signature"])
                    text_blocks.append(block)
            elif kind == "reasoning":
                value = str(item.get("value") or "")
                reasoning_parts.append(value)
                if value or item.get("signature"):
                    block = {"value": value}
                    if item.get("signature"):
                        block["signature"] = str(item["signature"])
                    reasoning_blocks.append(block)
            elif kind == "tool_calls":
                tool_calls.extend(_parse_tool_calls(item.get("tool_calls")))
    elif isinstance(payload.get("choices"), list):
        choices = payload.get("choices") or []
        message = (choices[0] if choices else {}).get("message") or {}
        text_parts.append(str(message.get("content") or ""))
        if message.get("reasoning_content"):
            reasoning_parts.append(str(message["reasoning_content"]))
            reasoning_blocks.append({"value": str(message["reasoning_content"])})
        tool_calls.extend(_parse_tool_calls(message.get("tool_calls")))

    content = "".join(text_parts)
    if not tool_calls and content:
        stripped, recovered, _ = recover_native_tool_calls(content)
        if recovered:
            content = stripped
            tool_calls = recovered
            text_blocks = _text_blocks_for_content(content, text_blocks)

    usage: dict[str, Any] = {}
    if isinstance(payload.get("usage"), dict):
        usage.update(payload["usage"])
    if isinstance(payload.get("cost_info"), dict):
        usage.update(payload["cost_info"])
    header_usage = {
        "prompt_tokens": headers.get("x-usage-prompt-tokens"),
        "completion_tokens": headers.get("x-usage-completion-tokens"),
        "cache_hit_tokens": headers.get("x-usage-cache-hit-tokens"),
        "cache_write_tokens": headers.get("x-usage-cache-write-tokens"),
        "cost": headers.get("x-usage-cost"),
    }
    for key, value in header_usage.items():
        if value is None:
            continue
        try:
            usage[key] = float(value)
        except (TypeError, ValueError):
            pass

    generation_info = {
        "prompt_tokens": _usage_number(usage, "prompt_tokens", "input_tokens"),
        "completion_tokens": _usage_number(usage, "completion_tokens", "output_tokens"),
        "input_tokens": _usage_number(usage, "prompt_tokens", "input_tokens"),
        "output_tokens": _usage_number(usage, "completion_tokens", "output_tokens"),
        "reasoning_tokens": _usage_number(usage, "reasoning_tokens"),
        "cache_hit_tokens": _usage_number(usage, "cache_hit_tokens", "cached_tokens"),
        "cache_write_tokens": _usage_number(usage, "cache_write_tokens"),
        "cost": _usage_number(usage, "cost"),
    }
    if reasoning_parts:
        generation_info["reasoning_content"] = "".join(reasoning_parts)
    if reasoning_blocks:
        generation_info["reasoning_blocks"] = reasoning_blocks
    if text_blocks:
        generation_info["text_blocks"] = text_blocks
    return content, tool_calls, generation_info


def update_save_id_from_response(
    save_id: dict[str, Any], payload: dict[str, Any]
) -> None:
    response_id = payload.get("request_id") or payload.get("id")
    if response_id:
        save_id["response_id"] = response_id

    request_detail = payload.get("request_detail") or {}
    inner_response = (
        request_detail.get("response") if isinstance(request_detail, dict) else None
    )
    account_id = None
    if isinstance(inner_response, dict):
        account_id = inner_response.get("account_id")
    account_id = account_id or payload.get("account_id")
    if account_id:
        save_id["account_id"] = account_id
