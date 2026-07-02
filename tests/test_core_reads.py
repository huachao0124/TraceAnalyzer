"""Pin the file-view command parsing in p2a.core.

Both the raw-text path (`_parse_bash_read_commands`) and the structured tool-call
path (`parse_read_actions_from_tool_calls` -> `_parse_bash_read_commands_from_str`)
share one extractor (`_extract_reads_from_command`); these tests lock the extracted
ranges, the dedup, and the cross-path consistency.
"""

from p2a.core import (
    _parse_bash_read_commands,
    _parse_bash_read_commands_from_str,
    match_reads_to_callgraph,
    normalize_action,
    parse_read_actions_from_tool_calls,
    parse_write_actions,
    parse_write_actions_from_tool_calls,
    segment_purpose_blocks,
    writes_from_step_trace,
)


def test_extracts_cat_head_tail_sed_ranges():
    reads = _parse_bash_read_commands_from_str(
        "cat /testbed/foo.py && head -20 /testbed/bar.py && "
        "tail -5 /testbed/baz.py && sed -n '10,20p' /testbed/qux.py"
    )
    by_file = {r["file_path"]: (r["start_line"], r["end_line"]) for r in reads}
    assert by_file["foo.py"] == (1, 999999)   # cat → whole file
    assert by_file["bar.py"] == (1, 20)       # head -20
    assert by_file["baz.py"] == (1, 999999)   # tail length unknown → full range
    assert by_file["qux.py"] == (10, 20)      # sed -n 'start,endp'


def test_grep_n_yields_single_read_not_duplicate():
    # `grep -n` matches both the grep-n and the generic grep pattern; the shared
    # extractor must dedup it to one read.
    reads = _parse_bash_read_commands_from_str("grep -n needle /testbed/qux.py")
    assert reads == [{"file_path": "qux.py", "start_line": 1, "end_line": 999999}]


def test_text_and_str_paths_agree():
    cmd = "cat /testbed/a.py && grep -n x /testbed/b.py && sed -n '3,7p' /testbed/c.py"
    assert _parse_bash_read_commands(cmd) == _parse_bash_read_commands_from_str(cmd)


def test_structured_execute_bash_tool_call():
    tool_calls = [
        {"function": {"name": "execute_bash", "arguments": {"command": "cat /testbed/foo.py"}}},
        {"function": {"name": "str_replace_editor",
                      "arguments": {"command": "view", "path": "/testbed/bar.py", "view_range": [5, 9]}}},
    ]
    reads = parse_read_actions_from_tool_calls(tool_calls)
    assert {"file_path": "foo.py", "start_line": 1, "end_line": 999999} in reads
    assert {"file_path": "bar.py", "start_line": 5, "end_line": 9} in reads


def test_structured_execute_bash_unwraps_text_command_and_ignores_unknown_shapes():
    tool_calls = [
        {"function": {"name": "execute_bash", "arguments": {"command": {"cmd": "cat /testbed/foo.py"}}}},
        {"function": {"name": "execute_bash", "arguments": {"command": {"$text": "cat /testbed/bar.py"}}}},
        {"function": {"name": "execute_bash", "arguments": ["cat /testbed/bar.py"]}},
    ]

    assert parse_read_actions_from_tool_calls(tool_calls) == [
        {"file_path": "bar.py", "start_line": 1, "end_line": 999999}
    ]
    assert _parse_bash_read_commands_from_str({"cmd": "cat /testbed/foo.py"}) == []


def test_app_paths_normalize_like_repo_relative_paths():
    tool_calls = [
        {"function": {"name": "execute_bash", "arguments": {"command": "cat /app/pkg/root.py"}}},
        {"function": {"name": "str_replace_editor",
                      "arguments": {"command": "view", "path": "/app/pkg/helper.py", "view_range": [2, 4]}}},
        {"function": {"name": "str_replace_editor",
                      "arguments": {"command": "str_replace", "path": "/app/pkg/root.py"}}},
    ]

    reads = parse_read_actions_from_tool_calls(tool_calls)
    writes = parse_write_actions_from_tool_calls(tool_calls)

    assert {"file_path": "pkg/root.py", "start_line": 1, "end_line": 999999} in reads
    assert {"file_path": "pkg/helper.py", "start_line": 2, "end_line": 4} in reads
    assert {"file_path": "pkg/root.py", "start_line": 1, "end_line": 999999, "command": "str_replace"} in writes


def test_segments_purpose_blocks_by_family_and_target():
    traces = [
        {"step_idx": 1, "tool_calls": [{"function": {"name": "str_replace_editor", "arguments": {"command": "view", "path": "/testbed/a.py"}}}]},
        {"step_idx": 2, "tool_calls": [{"function": {"name": "str_replace_editor", "arguments": {"command": "view", "path": "/testbed/a.py"}}}]},
        {"step_idx": 3, "tool_calls": [{"function": {"name": "str_replace_editor", "arguments": {"command": "view", "path": "/testbed/b.py"}}}]},
        {"step_idx": 4, "tool_calls": [{"function": {"name": "str_replace_editor", "arguments": {"command": "str_replace", "path": "/testbed/b.py"}}}]},
    ]

    assert normalize_action(traces[0]) == {"family": "read", "target_path": "a.py"}
    assert segment_purpose_blocks(traces) == [
        {"family": "read", "target_path": "a.py", "step_indices": [1, 2], "trace_indices": [0, 1]},
        {"family": "read", "target_path": "b.py", "step_indices": [3], "trace_indices": [2]},
        {"family": "edit", "target_path": "b.py", "step_indices": [4], "trace_indices": [3]},
    ]


def test_extracts_file_editor_and_bash_writes():
    tool_calls = [
        {"function": {"name": "str_replace_editor", "arguments": {"command": "str_replace", "path": "/testbed/pkg/root.py"}}},
        {"function": {"name": "execute_bash", "arguments": {"command": "echo x > /testbed/pkg/a.py && sed -i '3,7s/x/y/' /testbed/pkg/b.py && cat patch | tee /testbed/pkg/c.py"}}},
    ]

    writes = parse_write_actions_from_tool_calls(tool_calls)
    assert {"file_path": "pkg/root.py", "start_line": 1, "end_line": 999999, "command": "str_replace"} in writes
    assert {"file_path": "pkg/a.py", "start_line": 1, "end_line": 999999, "command": "redirect"} in writes
    assert {"file_path": "pkg/b.py", "start_line": 3, "end_line": 7, "command": "sed -i"} in writes
    assert {"file_path": "pkg/c.py", "start_line": 1, "end_line": 999999, "command": "tee"} in writes


def test_execute_bash_write_normalizes_as_edit():
    trace = {"tool_calls": [{"function": {"name": "execute_bash", "arguments": {"command": "python -c \"open('/testbed/pkg/root.py', 'w').write('x')\""}}}]}

    assert writes_from_step_trace(trace) == [
        {"file_path": "pkg/root.py", "start_line": 1, "end_line": 999999, "command": "python open"}
    ]
    assert normalize_action(trace) == {"family": "edit", "target_path": "pkg/root.py"}


def test_execute_bash_common_sed_i_write_normalizes_as_edit():
    trace = {"tool_calls": [{"function": {"name": "execute_bash", "arguments": {"command": "sed -i 's/foo/bar/' /testbed/pkg/root.py"}}}]}

    assert writes_from_step_trace(trace) == [
        {"file_path": "pkg/root.py", "start_line": 1, "end_line": 999999, "command": "sed -i"}
    ]
    assert normalize_action(trace) == {"family": "edit", "target_path": "pkg/root.py"}


def test_raw_xml_and_text_write_actions_are_recovered():
    text = """
    <function=file_editor>
    <parameter=command>str_replace</parameter>
    <parameter=path>/testbed/pkg/root.py</parameter>
    </function>
    str_replace_editor(command="create", path="/testbed/pkg/new.py")
    """

    writes = parse_write_actions(text)
    assert {"file_path": "pkg/root.py", "start_line": 1, "end_line": 999999, "command": "str_replace"} in writes
    assert {"file_path": "pkg/new.py", "start_line": 1, "end_line": 999999, "command": "create"} in writes
    assert writes_from_step_trace({"response_text": text}) == writes


def test_match_reads_ignores_non_rewardable_call_graph_nodes():
    reads = [{"file_path": "tests/test_demo.py", "start_line": 1, "end_line": 20}]
    bonus_map = {
        "call_graph_nodes": {
            "tests/test_demo.py::test_demo": {
                "file_path": "tests/test_demo.py",
                "start_line": 1,
                "end_line": 20,
                "normalized_distance": 1.0,
                "rewardable": False,
                "node_role": "test_harness",
            }
        }
    }

    assert match_reads_to_callgraph(reads, bonus_map) == -1.0

    bonus_map["call_graph_nodes"]["tests/test_demo.py::fixture_target"] = {
        "file_path": "tests/test_demo.py",
        "start_line": 5,
        "end_line": 8,
        "normalized_distance": 0.25,
        "rewardable": True,
        "node_role": "program",
    }

    assert match_reads_to_callgraph(reads, bonus_map) == 0.25
