"""Pin the licensed / unlicensed reference scorer in p2a.provenance.

The scorer marks the first reference to each entity in a model action (path or
symbol read/search target) as licensed when its literal appears in the initial
context or a strictly earlier step's observations. These tests lock the
first-occurrence dedupe, the step-ordering license boundary, basename path
licensing, the self-created file exemption, stopword filtering, and
grep-pattern symbol extraction.
"""

from p2a.provenance import (
    extract_entity_references,
    license_references,
    symbols_from_pattern,
)


def _bash_call(command: str) -> dict:
    return {"function": {"name": "execute_bash", "arguments": {"command": command}}}


def _view_call(path: str) -> dict:
    return {"function": {"name": "str_replace_editor", "arguments": {"command": "view", "path": path}}}


def _step(tool_calls: list[dict], observation: str = "") -> dict:
    return {"tool_calls": tool_calls, "tool_results": [{"observation": observation}]}


def _unlicensed_entities(summary: dict) -> set[tuple[str, str]]:
    return {(ref["kind"], ref["entity"]) for ref in summary["unlicensed_references"]}


def test_first_occurrence_dedupe():
    steps = [
        _step([_view_call("/testbed/pkg/mod.py")]),
        _step([_bash_call("cat /testbed/pkg/mod.py")]),
        _step([_bash_call("grep -n helper_symbol_name /testbed/pkg/mod.py && grep helper_symbol_name /testbed/pkg/mod.py")]),
    ]
    summary = license_references(steps, "")
    assert summary["license_evaluable"] is True
    assert summary["n_path_references"] == 1
    assert summary["n_symbol_references"] == 1
    assert summary["n_entity_references"] == 2


def test_same_step_observation_does_not_license():
    steps = [
        _step([_bash_call("cat /testbed/pkg/mod.py")], observation="pkg/mod.py contents here"),
    ]
    summary = license_references(steps, "issue text without the file name")
    assert summary["n_unlicensed_references"] == 1
    assert ("path", "pkg/mod.py") in _unlicensed_entities(summary)


def test_strictly_earlier_observation_licenses():
    steps = [
        _step([_bash_call("ls /testbed")], observation="pkg/mod.py\npkg/other.py"),
        _step([_bash_call("cat /testbed/pkg/mod.py")]),
    ]
    summary = license_references(steps, "I have uploaded a repository in /testbed.")
    assert summary["n_unlicensed_references"] == 0
    assert summary["unlicensed_reference_rate"] == 0.0


def test_basename_licenses_full_path():
    steps = [_step([_view_call("/testbed/django/db/models/expressions.py")])]
    summary = license_references(steps, "The bug is somewhere in expressions.py.")
    assert summary["n_path_references"] == 1
    assert summary["n_unlicensed_references"] == 0


def test_self_created_file_is_exempt():
    steps = [
        _step(
            [
                {
                    "function": {
                        "name": "str_replace_editor",
                        "arguments": {"command": "create", "path": "/testbed/repro.py", "file_text": "print(1)"},
                    }
                }
            ]
        ),
        _step([_bash_call("cat /testbed/repro.py")]),
        _step([_bash_call("cat output.log")]),
    ]
    summary = license_references(steps, "")
    # repro.py is model-created and never counted; output.log still counts.
    assert summary["n_path_references"] == 1
    assert ("path", "output.log") in _unlicensed_entities(summary)


def test_bash_redirect_created_file_is_exempt():
    steps = [
        _step([_bash_call("cat > /testbed/scratch_probe.py <<'EOF'\nimport pkg\nEOF")]),
        _step([_bash_call("cat /testbed/scratch_probe.py")]),
    ]
    summary = license_references(steps, "")
    assert summary["n_entity_references"] == 0


def test_stopwords_are_not_counted():
    tool_calls = [_bash_call('grep -rn "class error import CustomThingHere" /testbed/pkg/mod.py')]
    refs = extract_entity_references(tool_calls)
    symbols = [ref["entity"] for ref in refs if ref["kind"] == "symbol"]
    assert symbols == ["CustomThingHere"]


def test_grep_pattern_symbol_extraction_strips_regex_and_splits_alternation():
    symbols = symbols_from_pattern(r"class CombinedExpression\|def resolve_expression\b.*[Tt]emporal?")
    assert "CombinedExpression" in symbols
    assert "resolve_expression" in symbols
    assert "class" not in symbols
    assert "def" not in symbols


def test_grep_targets_and_pattern_symbols_are_both_referenced():
    refs = extract_entity_references([_bash_call('grep -rn "mixed types" django/db/models/expressions.py')])
    kinds = {(ref["kind"], ref["entity"]) for ref in refs}
    assert ("path", "django/db/models/expressions.py") in kinds
    assert ("symbol", "mixed") in kinds
    for ref in refs:
        assert ref["action"]


def test_unlicensed_reference_record_shape():
    steps = [_step([_bash_call("cat /testbed/pkg/" + "a" * 150 + ".py")])]
    summary = license_references(steps, "")
    assert len(summary["unlicensed_references"]) == 1
    ref = summary["unlicensed_references"][0]
    assert set(ref) == {"entity", "kind", "step", "action"}
    assert ref["step"] == 1
    assert len(ref["action"]) <= 120


def test_no_step_traces_is_not_evaluable():
    summary = license_references([], "issue")
    assert summary["license_evaluable"] is False
    assert summary["unlicensed_reference_rate"] is None
