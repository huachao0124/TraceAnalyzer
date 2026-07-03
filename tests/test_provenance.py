"""Pin the licensed / unlicensed reference scorer in p2a.provenance.

The scorer marks the first reference to each entity in a model action (path or
symbol read/search target) as licensed when its literal appears in the initial
context or a strictly earlier step's observations. These tests lock the
first-occurrence dedupe, the step-ordering license boundary, basename path
licensing, the self-created file exemption, stopword filtering, grep-pattern
symbol extraction, the text-format tool-call fallback, prompt-borne licensing
through the eval/validation record paths, and the summarize/val-metric
aggregation.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from p2a.core import BonusMapStore
from p2a.eval_fault_localization import _initial_context_text, score_record, summarize
from p2a.provenance import (
    extract_entity_references,
    extract_entity_references_from_text,
    license_references,
    symbols_from_pattern,
)
from p2a.validation_metrics import (
    P2A_VALIDATION_METRICS,
    flatten_validation_metrics,
    validation_records_from_batch,
)


def _score(record: dict, bonus_map_dir: str = "/nonexistent-bonus-map-dir") -> dict:
    return score_record(
        record,
        index=0,
        bonus_maps=BonusMapStore(bonus_map_dir),
        tracking_mode="view_and_bash",
        near_threshold=0.25,
        m_max=3.0,
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


def test_text_format_tool_calls_are_extracted():
    text = (
        "<function=file_editor><parameter=command>view</parameter>"
        "<parameter=path>/testbed/pkg/viewed_mod.py</parameter></function>\n"
        '<function=execute_bash><parameter=command>grep -rn "SpecialThing" pkg/legacy_mod.py</parameter></function>'
    )
    refs = extract_entity_references_from_text(text)
    kinds = {(ref["kind"], ref["entity"]) for ref in refs}
    assert ("path", "pkg/viewed_mod.py") in kinds
    assert ("path", "pkg/legacy_mod.py") in kinds
    assert ("symbol", "SpecialThing") in kinds

    steps = [{"response_text": text, "tool_results": [{"observation": ""}]}]
    summary = license_references(steps, "")
    assert summary["n_path_references"] == 2
    assert summary["n_symbol_references"] == 1


def test_structured_tool_calls_win_over_text_channel():
    step = {
        "tool_calls": [_bash_call("cat pkg/structured_mod.py")],
        "response_text": "<function=execute_bash><parameter=command>cat pkg/text_mod.py</parameter></function>",
        "tool_results": [{"observation": ""}],
    }
    summary = license_references([step], "")
    assert summary["n_path_references"] == 1
    assert {(ref["kind"], ref["entity"]) for ref in summary["unlicensed_references"]} == {("path", "pkg/structured_mod.py")}


def test_prompt_borne_path_is_licensed_from_raw_prompt():
    record = {
        "instance_id": "repo__0123abcd",
        "raw_prompt": [
            {"role": "system", "content": "You are a software engineering agent."},
            {"role": "user", "content": "Fix the crash reported in pkg/prompt_named.py."},
        ],
        "p2a_step_traces": [_step([_bash_call("cat /testbed/pkg/prompt_named.py")])],
    }
    assert "prompt_named.py" in _initial_context_text(record)
    detail = _score(record)
    assert detail["license_evaluable"] is True
    assert detail["n_path_references"] == 1
    assert detail["n_unlicensed_references"] == 0


def test_validation_batch_records_carry_prompt_for_licensing():
    step = _step([_bash_call("cat /testbed/pkg/prompt_named.py")])
    batch = SimpleNamespace(
        non_tensor_batch={
            "uid": ["u0"],
            "instance_id": ["repo__0123abcd"],
            "data_source": ["r2e"],
            "raw_prompt": [
                [
                    {"role": "system", "content": "You are a software engineering agent."},
                    {"role": "user", "content": "Fix the crash reported in pkg/prompt_named.py."},
                ]
            ],
            "p2a_step_traces": [json.dumps([step])],
        }
    )
    records = validation_records_from_batch(batch, output_texts=["done"])
    assert records[0]["raw_prompt"][0]["role"] == "system"
    assert "prompt_named.py" in _initial_context_text(records[0])
    detail = _score(records[0])
    assert detail["n_path_references"] == 1
    assert detail["n_unlicensed_references"] == 0


def test_summarize_and_val_metrics_aggregate_unlicensed_references(tmp_path):
    records = [
        {
            "instance_id": "repo__aaaa1111",
            "issue_description": "There is a bug in pkg/known.py.",
            "p2a_step_traces": [_step([_bash_call("cat pkg/known.py")])],
        },
        {
            "instance_id": "repo__bbbb2222",
            "issue_description": "Something unrelated fails.",
            "p2a_step_traces": [_step([_bash_call("cat pkg/unknown.py")])],
        },
    ]
    details = [_score(record, str(tmp_path)) for record in records]
    summary = summarize(
        details,
        source=Path("test"),
        bonus_map_dir=tmp_path,
        tracking_mode="view_and_bash",
        near_threshold=0.25,
        m_max=3.0,
    )
    assert summary["counts"]["n_license_evaluable"] == 2
    assert summary["rates"]["unlicensed_reference_rate"] == 0.5
    assert summary["rates"]["unlicensed_trace_rate"] == 0.5
    assert summary["averages"]["avg_entity_references"] == 1.0
    assert summary["averages"]["avg_unlicensed_references"] == 0.5

    assert "unlicensed_reference_rate" in P2A_VALIDATION_METRICS
    assert "unlicensed_trace_rate" in P2A_VALIDATION_METRICS
    for detail in details:
        detail.setdefault("data_source", "unknown")
    metrics = flatten_validation_metrics(
        details,
        bonus_map_dir=str(tmp_path),
        tracking_mode="view_and_bash",
        near_threshold=0.25,
        m_max=3.0,
    )
    assert metrics["val-p2a/unknown/unlicensed_reference_rate"] == 0.5
    assert metrics["val-p2a/unknown/unlicensed_trace_rate"] == 0.5


def test_unlicensed_reference_rate_is_pooled_over_references(tmp_path):
    # 100 licensed references in one trace + 1 unlicensed reference in another:
    # the aggregate rate pools the references (1/101), it does not average the
    # per-trace rates (which would be 0.5).
    licensed_names = " ".join(f"pkg/mod{i}.py" for i in range(100))
    records = [
        {
            "instance_id": "repo__cccc3333",
            "issue_description": f"Affected files: {licensed_names}",
            "p2a_step_traces": [_step([_bash_call(" && ".join(f"cat pkg/mod{i}.py" for i in range(100)))])],
        },
        {
            "instance_id": "repo__dddd4444",
            "issue_description": "Something unrelated fails.",
            "p2a_step_traces": [_step([_bash_call("cat pkg/lonely_unlicensed.py")])],
        },
    ]
    details = [_score(record, str(tmp_path)) for record in records]
    assert details[0]["n_entity_references"] == 100
    assert details[0]["n_unlicensed_references"] == 0
    assert details[1]["n_entity_references"] == 1
    assert details[1]["n_unlicensed_references"] == 1

    summary = summarize(
        details,
        source=Path("test"),
        bonus_map_dir=tmp_path,
        tracking_mode="view_and_bash",
        near_threshold=0.25,
        m_max=3.0,
    )
    assert summary["rates"]["unlicensed_reference_rate"] == pytest.approx(1 / 101)
    assert summary["rates"]["unlicensed_trace_rate"] == 0.5

    for detail in details:
        detail.setdefault("data_source", "unknown")
    metrics = flatten_validation_metrics(
        details,
        bonus_map_dir=str(tmp_path),
        tracking_mode="view_and_bash",
        near_threshold=0.25,
        m_max=3.0,
    )
    assert metrics["val-p2a/unknown/unlicensed_reference_rate"] == pytest.approx(1 / 101)
    assert metrics["val-p2a/unknown/unlicensed_trace_rate"] == 0.5
