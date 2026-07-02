import json

from p2a.internal_api_native import (
    parse_internal_response,
    responses_api_response_to_payload,
    to_prompt_history,
)


class _FakeApi:
    MODEL_CONFIGS = {
        "passthrough_models": [
            "deepseek-v4-flash-passthrough",
            "doubao-seed-2-0-lite-passthrough",
        ],
        "passthrough_chat_completions_models": {"deepseek-v4-flash-passthrough"},
        "moonshot_passthrough_models": ["step-3.7-flash-passthrough"],
        "minimax_passthrough_models": ["minimax_m3-passthrough"],
        "naci_passthrough_models": [],
        "google_passthrough_models": [],
        "anthropic_passthrough_models": [],
        "aws_anthropic_passthrough_models": [],
    }


class _FakeApiModule:
    Api = _FakeApi


def _messages_with_tool_tail():
    return [
        {"role": "system", "content": "You are a SWE agent."},
        {"role": "user", "content": "Fix the bug."},
        {
            "role": "assistant",
            "content": "inspect",
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
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "str_replace_editor",
            "content": "file contents",
        },
    ]


def test_tool_tail_stays_structured_history_not_plain_prompt():
    prompt, history = to_prompt_history(
        _messages_with_tool_tail(),
        model_name="deepseek-v4-flash-passthrough",
        api_module=_FakeApiModule,
    )

    assert prompt == ""
    assert history[-1]["role"] == "tool"
    assert history[-1]["tool_call_id"] == "call-1"
    assert history[-1]["content"][0]["tool_call_id"] == "call-1"
    assert history[-1]["content"][0]["name"] == "str_replace_editor"
    assert history[-1]["content"][0]["value"] == "file contents"


def test_trailing_user_is_split_as_prompt():
    prompt, history = to_prompt_history(
        [
            {"role": "system", "content": "You are a SWE agent."},
            {"role": "user", "content": "Fix the bug."},
        ],
        model_name="deepseek-v4-flash-passthrough",
        api_module=_FakeApiModule,
    )

    assert prompt == "Fix the bug."
    assert history == [{"role": "system", "content": "You are a SWE agent."}]


def test_passthrough_assistant_uses_typed_tool_call_block():
    _prompt, history = to_prompt_history(
        _messages_with_tool_tail()[:-1],
        model_name="step-3.7-flash-passthrough",
        api_module=_FakeApiModule,
    )

    assistant = history[-1]
    assert assistant["role"] == "assistant"
    assert assistant["content"][-1]["type"] == "tool_calls"
    assert (
        assistant["content"][-1]["tool_calls"][0]["function"]["name"]
        == "str_replace_editor"
    )


def test_passthrough_assistant_replays_reasoning_and_signatures_from_metadata():
    messages = _messages_with_tool_tail()[:-1]
    messages[-1]["tool_calls"][0]["signature"] = "tool-sig"

    _prompt, history = to_prompt_history(
        messages,
        model_name="minimax_m3-passthrough",
        api_module=_FakeApiModule,
        assistant_metadata_by_index={
            2: {
                "reasoning_content": "thinking",
                "reasoning_blocks": [{"value": "thinking", "signature": "reason-sig"}],
                "text_blocks": [{"value": "inspect", "signature": "text-sig"}],
            }
        },
    )

    assistant = history[-1]
    assert assistant["content"][0] == {
        "type": "reasoning",
        "value": "thinking",
        "signature": "reason-sig",
    }
    assert assistant["content"][1] == {
        "type": "text",
        "value": "inspect",
        "signature": "text-sig",
    }
    assert assistant["content"][-1]["tool_calls"][0]["signature"] == "tool-sig"
    assert assistant["reasoning_content"] == "thinking"


def test_minimax_parallel_tool_results_are_merged():
    messages = _messages_with_tool_tail() + [
        {
            "role": "tool",
            "tool_call_id": "call-2",
            "name": "execute_bash",
            "content": "pytest output",
        }
    ]

    prompt, history = to_prompt_history(
        messages,
        model_name="minimax_m3-passthrough",
        api_module=_FakeApiModule,
    )

    assert prompt == ""
    assert history[-1]["role"] == "tool"
    assert [part["tool_call_id"] for part in history[-1]["content"]] == [
        "call-1",
        "call-2",
    ]


def test_openai_responses_chain_merges_parallel_tool_results_when_seeded():
    messages = _messages_with_tool_tail() + [
        {
            "role": "tool",
            "tool_call_id": "call-2",
            "name": "execute_bash",
            "content": "pytest output",
        }
    ]

    _prompt, unseeded = to_prompt_history(
        messages,
        model_name="doubao-seed-2-0-lite-passthrough",
        api_module=_FakeApiModule,
        save_id={},
    )
    _prompt, seeded = to_prompt_history(
        messages,
        model_name="doubao-seed-2-0-lite-passthrough",
        api_module=_FakeApiModule,
        save_id={"response_id": "resp_123"},
    )

    assert [item["role"] for item in unseeded[-2:]] == ["tool", "tool"]
    assert seeded[-1]["role"] == "tool"
    assert [part["tool_call_id"] for part in seeded[-1]["content"]] == [
        "call-1",
        "call-2",
    ]


def test_deepseek_dsml_content_recovers_structured_tool_call():
    payload = {
        "answer": [
            {
                "type": "text",
                "value": (
                    "I should inspect the file.\n"
                    "<｜｜DSML｜｜tool_calls>"
                    '<｜｜DSML｜｜invoke name="str_replace_editor">'
                    '<｜｜DSML｜｜parameter name="command" string="true">view'
                    "</｜｜DSML｜｜parameter>"
                    '<｜｜DSML｜｜parameter name="path" string="true">/testbed/a.py'
                    "</｜｜DSML｜｜parameter>"
                    '<｜｜DSML｜｜parameter name="view_range" string="false">[1, 20]'
                    "</｜｜DSML｜｜parameter>"
                    "</｜｜DSML｜｜invoke>"
                    "</｜｜DSML｜｜tool_calls>"
                ),
            }
        ],
        "cost_info": {"prompt_tokens": 10, "completion_tokens": 5},
    }

    content, tool_calls, info = parse_internal_response(payload)

    assert content == "I should inspect the file."
    assert tool_calls[0]["function"]["name"] == "str_replace_editor"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {
        "command": "view",
        "path": "/testbed/a.py",
        "view_range": [1, 20],
    }
    assert info["prompt_tokens"] == 10
    assert info["completion_tokens"] == 5
    assert info["text_blocks"] == [{"value": "I should inspect the file."}]


def test_recovered_dsml_is_not_replayed_as_assistant_text_metadata():
    dsml = (
        "<｜｜DSML｜｜tool_calls>"
        '<｜｜DSML｜｜invoke name="str_replace_editor">'
        '<｜｜DSML｜｜parameter name="command" string="true">view'
        "</｜｜DSML｜｜parameter>"
        '<｜｜DSML｜｜parameter name="path" string="true">/testbed/a.py'
        "</｜｜DSML｜｜parameter>"
        "</｜｜DSML｜｜invoke>"
        "</｜｜DSML｜｜tool_calls>"
    )
    content, tool_calls, info = parse_internal_response(
        {"answer": [{"type": "text", "value": f"I should inspect.\n{dsml}"}]}
    )

    _prompt, history = to_prompt_history(
        [
            {"role": "system", "content": "You are a SWE agent."},
            {"role": "user", "content": "Fix the bug."},
            {"role": "assistant", "content": content, "tool_calls": tool_calls},
        ],
        model_name="deepseek-v4-flash-passthrough",
        api_module=_FakeApiModule,
        assistant_metadata_by_index={2: info},
    )

    assistant = history[-1]
    replay_text = "".join(
        block.get("value", "")
        for block in assistant["content"]
        if block.get("type") == "text"
    )
    assert "DSML" not in replay_text
    assert "str_replace_editor" not in replay_text
    assert replay_text == "I should inspect."


def test_internal_response_preserves_reasoning_metadata_and_tool_signature():
    payload = {
        "answer": [
            {"type": "reasoning", "value": "chain", "signature": "reason-sig"},
            {"type": "text", "value": "inspect", "signature": "text-sig"},
            {
                "type": "tool_calls",
                "tool_calls": [
                    {
                        "id": "call-gemini",
                        "signature": "tool-sig",
                        "type": "function",
                        "function": {
                            "name": "execute_bash",
                            "arguments": {"command": "pytest -q"},
                        },
                    }
                ],
            },
        ],
    }

    content, tool_calls, info = parse_internal_response(payload)

    assert content == "inspect"
    assert tool_calls[0]["signature"] == "tool-sig"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {
        "command": "pytest -q"
    }
    assert info["reasoning_content"] == "chain"
    assert info["reasoning_blocks"] == [{"value": "chain", "signature": "reason-sig"}]
    assert info["text_blocks"] == [{"value": "inspect", "signature": "text-sig"}]


def test_internal_response_unwraps_minimax_text_wrapped_command():
    payload = {
        "answer": [
            {
                "type": "tool_calls",
                "tool_calls": [
                    {
                        "id": "call-minimax",
                        "type": "function",
                        "function": {
                            "name": "execute_bash",
                            "arguments": {
                                "command": {
                                    "$text": "cat /testbed/a.py",
                                },
                            },
                        },
                    },
                    {
                        "id": "call-minimax-json",
                        "type": "function",
                        "function": {
                            "name": "execute_bash",
                            "arguments": json.dumps(
                                {"command": {"$text": "sed -n '1,20p' /testbed/b.py"}}
                            ),
                        },
                    },
                ],
            }
        ],
    }

    _content, tool_calls, _info = parse_internal_response(payload)

    assert json.loads(tool_calls[0]["function"]["arguments"]) == {
        "command": "cat /testbed/a.py"
    }
    assert json.loads(tool_calls[1]["function"]["arguments"]) == {
        "command": "sed -n '1,20p' /testbed/b.py"
    }


def test_prompt_history_unwraps_text_wrapped_tool_arguments():
    messages = _messages_with_tool_tail()[:-1]
    messages[-1]["tool_calls"][0]["function"] = {
        "name": "execute_bash",
        "arguments": {"command": {"$text": "pytest -q"}},
    }

    _prompt, history = to_prompt_history(
        messages,
        model_name="minimax_m3-passthrough",
        api_module=_FakeApiModule,
    )

    tool_call = history[-1]["content"][-1]["tool_calls"][0]
    assert json.loads(tool_call["function"]["arguments"]) == {"command": "pytest -q"}


def test_responses_api_raw_output_preserves_reasoning_summary():
    payload = responses_api_response_to_payload(
        {
            "id": "resp_seed",
            "model": "doubao-seed-2-0-lite",
            "output": [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "check arithmetic"}],
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "TRUE"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "execute_bash",
                    "arguments": {"command": "pytest -q"},
                },
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 12,
                "total_tokens": 22,
                "output_tokens_details": {"reasoning_tokens": 7},
            },
        }
    )

    content, tool_calls, info = parse_internal_response(payload)

    assert payload["answer"][0] == {"type": "reasoning", "value": "check arithmetic"}
    assert content == "TRUE"
    assert tool_calls[0]["id"] == "call-1"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"command": "pytest -q"}
    assert info["reasoning_content"] == "check arithmetic"
    assert info["reasoning_tokens"] == 7


def test_minimax_native_content_recovers_structured_tool_call():
    payload = {
        "answer": [
            {
                "type": "text",
                "value": (
                    "<minimax:tool_call>"
                    '<invoke name="execute_bash">'
                    '<parameter name="command">pytest -q</parameter>'
                    "</invoke>"
                    "</minimax:tool_call>"
                ),
            }
        ]
    }

    content, tool_calls, info = parse_internal_response(payload)

    assert content == ""
    assert tool_calls[0]["function"]["name"] == "execute_bash"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {
        "command": "pytest -q"
    }
    assert "text_blocks" not in info
