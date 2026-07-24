import asyncio
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from p2a.datasets import assert_training_data_sources_allowed, canonical_dataset, parse_string_list, selector_files


ROOT = Path(__file__).resolve().parents[1]


def _load_build_data_module():
    spec = importlib.util.spec_from_file_location("build_data", ROOT / "scripts" / "build_data.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample(repo_language="python"):
    return {
        "instance_id": "instance_qutebrowser__qutebrowser-f91ace96223cac8161c16dd061907e138fe85111-vabc",
        "repo": "qutebrowser/qutebrowser",
        "repo_language": repo_language,
        "dockerhub_tag": "qutebrowser.qutebrowser-qutebrowser__qutebrowser-f91ace96223cac8161c16dd061907e138fe85111-vabc",
        "base_commit": "ebfe9b7aa0c4ba9d451f993e08955004aaec4345",
        "before_repo_set_cmd": "\n".join(
            [
                "git reset --hard ebfe9b7aa0c4ba9d451f993e08955004aaec4345",
                "git clean -fd",
                "git checkout ebfe9b7aa0c4ba9d451f993e08955004aaec4345",
                "git checkout f91ace -- tests/unit/utils/test_qtlog.py",
            ]
        ),
        "fail_to_pass": "['tests/unit/utils/test_qtlog.py::TestHideQtWarning::test_unfiltered']",
        "pass_to_pass": '["tests/unit/utils/test_log.py::test_stub"]',
        "selected_test_files_to_run": '["tests/unit/utils/test_qtlog.py", "tests/unit/utils/test_log.py"]',
        "patch": "diff --git a/qutebrowser/utils/qtlog.py b/qutebrowser/utils/qtlog.py\n",
        "test_patch": "diff --git a/tests/unit/utils/test_qtlog.py b/tests/unit/utils/test_qtlog.py\n",
        "problem_statement": "Hide Qt warnings by prefix.",
        "requirements": "Keep unmatched warnings visible.",
        "interface": "Type: Function\nName: hide_qt_warning",
        "issue_categories": "",
        "issue_specificity": "",
    }


def _write_pro_scripts(scripts_dir: Path, instance_id: str | None = None) -> None:
    instance_dir = scripts_dir / (instance_id or _sample()["instance_id"])
    instance_dir.mkdir(parents=True)
    (instance_dir / "run_script.sh").write_text("pytest \"$@\"\n", encoding="utf-8")
    (instance_dir / "parser.py").write_text("print('parse')\n", encoding="utf-8")


def test_parse_string_list_accepts_python_repr_and_json():
    assert parse_string_list("['a', 'b']") == ["a", "b"]
    assert parse_string_list('["a", "b"]') == ["a", "b"]
    assert parse_string_list(["a", "b"]) == ["a", "b"]


def test_selector_files_strips_nodeids_and_deduplicates():
    assert selector_files(["tests/a.py::test_param[1,1]", "tests/a.py::test_other", " tests/b.py::test "]) == [
        "tests/a.py",
        "tests/b.py",
    ]


def test_canonical_dataset_accepts_swebench_pro_alias():
    assert canonical_dataset("swe-bench-pro") == "swebench-pro"
    assert canonical_dataset("pro") == "swebench-pro"


def test_dashboard_registers_swebench_pro_parquet():
    from p2a.dashboard_adapter import DATASET_PARQUET_FILENAMES

    assert DATASET_PARQUET_FILENAMES["swebench-pro"] == ("swe_bench_pro.parquet",)


def test_training_guard_rejects_swebench_pro_parquet(tmp_path):
    path = tmp_path / "swe_bench_pro.parquet"
    pd.DataFrame([{"data_source": "swebench-pro"}]).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="Eval-only dataset cannot be used"):
        assert_training_data_sources_allowed(path)


def test_training_guard_scans_all_rows(tmp_path):
    path = tmp_path / "mixed_training.parquet"
    rows = [{"data_source": "r2e-gym-subset"} for _ in range(25)]
    rows.append({"data_source": "swebench-pro"})
    pd.DataFrame(rows).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="swebench-pro"):
        assert_training_data_sources_allowed(path)


def test_training_guard_scans_nested_eval_only_markers(tmp_path):
    path = tmp_path / "nested_eval_marker.parquet"
    pd.DataFrame(
        [
            {
                "data_source": "r2e-gym-subset",
                "extra_info": {
                    "tools_kwargs": {
                        "reward": {
                            "metadata": {"data_source": "swebench-pro"},
                        }
                    }
                },
            }
        ]
    ).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="swebench-pro"):
        assert_training_data_sources_allowed(path)


def test_cmd_swebench_pro_builds_python_subset_with_scripts(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "path", [path for path in sys.path if "uni-agent/examples/data_preprocess" not in path])
    monkeypatch.delitem(sys.modules, "swe_bench_verified", raising=False)
    build_data = _load_build_data_module()

    scripts_dir = tmp_path / "run_scripts"
    _write_pro_scripts(scripts_dir)
    out = tmp_path / "swe_bench_pro.parquet"

    monkeypatch.setattr("p2a.hf_assets.load_shared_dataset", lambda *_args, **_kwargs: [_sample(), _sample("go")])
    rc = build_data.cmd_swebench_pro(SimpleNamespace(out=str(out), language="python", scripts_dir=str(scripts_dir)))

    assert rc == 0
    rows = pd.read_parquet(out).to_dict(orient="records")
    assert len(rows) == 1
    row = rows[0]
    assert row["data_source"] == "swebench-pro"
    assert row["repo_language"] == "python"
    assert json.loads(row["FAIL_TO_PASS"]) == ["tests/unit/utils/test_qtlog.py::TestHideQtWarning::test_unfiltered"]
    assert json.loads(row["PASS_TO_PASS"]) == ["tests/unit/utils/test_log.py::test_stub"]
    assert row["run_tests"] == "pytest \"$@\"\n"
    assert row["swebench_pro_parser"] == "print('parse')\n"
    assert "Interface:" in row["prompt"][1]["content"]
    assert "<uploaded_files>\n/app\n</uploaded_files>" in row["prompt"][1]["content"]
    assert "repository in the /app directory" in row["prompt"][1]["content"]
    assert "/app/reproduce_issue.py" in row["prompt"][1]["content"]
    assert "/testbed" not in row["prompt"][1]["content"]
    tools = row["extra_info"]["tools_kwargs"]
    assert tools["reward"]["name"] == "swe_bench_pro"
    assert tools["env"]["deployment"]["image"].startswith("pair-diag-cn-guangzhou.cr.volces.com/code/sweap-images:")
    assert tools["env"]["post_setup_cmd"].splitlines()[:2] == ["set -e", "cd /app"]
    assert tools["reward"]["metadata"]["swebench_pro_repo_path"] == "/app"
    assert tools["reward"]["metadata"]["swebench_pro_restore_tests_cmd"] == "git checkout f91ace -- tests/unit/utils/test_qtlog.py"


def test_cmd_swebench_pro_skips_selected_tests_that_do_not_cover_p2p(monkeypatch, tmp_path):
    build_data = _load_build_data_module()
    sample = _sample()
    sample["selected_test_files_to_run"] = '["tests/unit/utils/test_qtlog.py"]'
    scripts_dir = tmp_path / "run_scripts"
    scripts_dir.mkdir()
    out = tmp_path / "out.parquet"

    monkeypatch.setattr("p2a.hf_assets.load_shared_dataset", lambda *_args, **_kwargs: [sample])

    rc = build_data.cmd_swebench_pro(SimpleNamespace(out=str(out), language="python", scripts_dir=str(scripts_dir)))

    assert rc == 0
    assert pd.read_parquet(out).empty


def test_cmd_swebench_pro_keeps_empty_selected_tests_for_runtime_fallback(monkeypatch, tmp_path):
    build_data = _load_build_data_module()
    sample = _sample()
    sample["selected_test_files_to_run"] = "[]"
    scripts_dir = tmp_path / "run_scripts"
    _write_pro_scripts(scripts_dir)
    out = tmp_path / "out.parquet"

    monkeypatch.setattr("p2a.hf_assets.load_shared_dataset", lambda *_args, **_kwargs: [sample])

    rc = build_data.cmd_swebench_pro(SimpleNamespace(out=str(out), language="python", scripts_dir=str(scripts_dir)))

    assert rc == 0
    rows = pd.read_parquet(out).to_dict(orient="records")
    assert len(rows) == 1
    assert json.loads(rows[0]["selected_test_files_to_run"]) == []
    assert rows[0]["run_tests"] == "pytest \"$@\"\n"


def test_cmd_swebench_pro_requires_scripts_dir(monkeypatch, tmp_path):
    build_data = _load_build_data_module()
    out = tmp_path / "out.parquet"

    monkeypatch.setattr("p2a.hf_assets.load_shared_dataset", lambda *_args, **_kwargs: [_sample()])

    with pytest.raises(ValueError, match="requires --scripts-dir"):
        build_data.cmd_swebench_pro(SimpleNamespace(out=str(out), language="python", scripts_dir=None))
    assert not out.exists()


def test_cmd_swebench_pro_rejects_missing_instance_scripts(monkeypatch, tmp_path):
    build_data = _load_build_data_module()
    scripts_dir = tmp_path / "run_scripts"
    scripts_dir.mkdir()
    out = tmp_path / "out.parquet"

    monkeypatch.setattr("p2a.hf_assets.load_shared_dataset", lambda *_args, **_kwargs: [_sample()])

    with pytest.raises(ValueError, match="missing required SWE-Bench-Pro run scripts"):
        build_data.cmd_swebench_pro(SimpleNamespace(out=str(out), language="python", scripts_dir=str(scripts_dir)))
    assert not out.exists()


def test_cmd_swebench_pro_allows_missing_scripts_when_debug_flag_is_set(monkeypatch, tmp_path):
    build_data = _load_build_data_module()
    scripts_dir = tmp_path / "run_scripts"
    scripts_dir.mkdir()
    out = tmp_path / "out.parquet"

    monkeypatch.setattr("p2a.hf_assets.load_shared_dataset", lambda *_args, **_kwargs: [_sample()])

    rc = build_data.cmd_swebench_pro(
        SimpleNamespace(
            out=str(out),
            language="python",
            scripts_dir=str(scripts_dir),
            allow_missing_scripts=True,
        )
    )

    assert rc == 0
    row = pd.read_parquet(out).to_dict(orient="records")[0]
    assert pd.isna(row["run_tests"]) or row["run_tests"] == ""
    assert pd.isna(row["swebench_pro_parser"]) or row["swebench_pro_parser"] == ""


def test_validate_swebench_pro_parquet_rejects_empty_scripts(tmp_path):
    build_data = _load_build_data_module()
    path = tmp_path / "bad.parquet"
    pd.DataFrame(
        [
            {
                "data_source": "swebench-pro",
                "instance_id": "demo",
                "run_tests": "",
                "swebench_pro_parser": None,
            }
        ]
    ).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="empty run_tests, swebench_pro_parser"):
        build_data.validate_swebench_pro_parquet(path)


def test_swebench_reward_runs_eval_through_sandbox(monkeypatch):
    from p2a.reward_specs import SWEBenchRewardSpec

    calls = {"execute": [], "write_file": []}

    class FakeSandbox:
        async def exec_shell(self, command, **_kwargs):
            calls["execute"].append(command)
            return SimpleNamespace(stdout="clean eval output", stderr="", exit_code=0)

        async def write_file(self, path, content):
            calls["write_file"].append((path, content))

    spec = SWEBenchRewardSpec(
        run_id="run-1",
        metadata={},
        sandbox=FakeSandbox(),
    )
    assert isinstance(spec, SWEBenchRewardSpec)
    monkeypatch.setattr(spec, "_build_eval_script", lambda: "echo eval\n")
    monkeypatch.setattr(spec, "_get_eval_report", lambda output: {"resolved": True, "captured": output})

    reward, details = asyncio.run(spec.compute_reward())

    assert reward is True
    assert details["resolved"] is True
    assert details["eval_completed"] is True
    assert calls["write_file"]
    assert len(calls["execute"]) == 1
    assert "bash /tmp/eval_script_" in calls["execute"][0]


def test_swebench_pro_reward_grades_f2p_and_p2p():
    from p2a.reward_specs import SWEBenchProRewardSpec

    spec = object.__new__(SWEBenchProRewardSpec)
    spec.metadata = {
        "FAIL_TO_PASS": json.dumps(["test_a"]),
        "PASS_TO_PASS": json.dumps(["test_b"]),
    }

    report = spec._grade(
        {
            "tests": [
                {"name": "test_a", "status": "PASSED"},
                {"name": "test_b", "status": "FAILED"},
            ]
        }
    )

    assert report["resolved"] is False
    assert report["FAIL_TO_PASS"]["success"] == ["test_a"]
    assert report["PASS_TO_PASS"]["failure"] == ["test_b"]


def test_swebench_pro_reward_eval_script_uses_app_repo_path():
    from p2a.reward_specs import SWEBenchProRewardSpec

    spec = object.__new__(SWEBenchProRewardSpec)
    spec.metadata = {
        "selected_test_files_to_run": json.dumps(["tests/test_demo.py::test_demo"]),
        "swebench_pro_repo_path": "/app",
    }

    script = spec._build_eval_script(
        {
            "run_script": Path("/tmp/run.sh"),
            "parser": Path("/tmp/parser.py"),
            "stdout": Path("/tmp/stdout.log"),
            "stderr": Path("/tmp/stderr.log"),
            "output": Path("/tmp/output.json"),
        }
    )

    assert "cd /app" in script
    assert "safe.directory /app" in script
    assert 'parser_python="$(command -v python3 || command -v python || true)"' in script
    assert "python_not_found" in script
    assert "parser_failed" in script
    assert "/testbed" not in script


def test_swebench_pro_reward_eval_script_runs_selected_files_as_one_comma_arg():
    from p2a.reward_specs import SWEBenchProRewardSpec

    spec = object.__new__(SWEBenchProRewardSpec)
    spec.metadata = {
        "FAIL_TO_PASS": json.dumps(["tests/test_demo.py::test_valid[1,1-expected0]"]),
        "PASS_TO_PASS": json.dumps(["tests/test_other.py::test_pass"]),
        "selected_test_files_to_run": json.dumps(["tests/test_demo.py", "tests/test_other.py"]),
        "swebench_pro_repo_path": "/app",
    }

    script = spec._build_eval_script(
        {
            "run_script": Path("/tmp/run.sh"),
            "parser": Path("/tmp/parser.py"),
            "stdout": Path("/tmp/stdout.log"),
            "stderr": Path("/tmp/stderr.log"),
            "output": Path("/tmp/output.json"),
        }
    )

    assert "bash /tmp/run.sh tests/test_demo.py,tests/test_other.py >" in script
    assert "test_valid[1,1-expected0]" not in script


def test_swebench_pro_reward_restore_fallback_preserves_candidate_patch():
    from p2a.reward_specs import SWEBenchProRewardSpec

    spec = object.__new__(SWEBenchProRewardSpec)
    spec.metadata = {
        "selected_test_files_to_run": json.dumps(["tests/test_demo.py"]),
        "swebench_pro_repo_path": "/app",
        "swebench_pro_restore_tests_cmd": "git checkout abc123 -- tests/test_demo.py",
    }

    script = spec._build_eval_script(
        {
            "run_script": Path("/tmp/run.sh"),
            "parser": Path("/tmp/parser.py"),
            "stdout": Path("/tmp/stdout.log"),
            "stderr": Path("/tmp/stderr.log"),
            "output": Path("/tmp/output.json"),
        }
    )

    assert "git -C /app checkout HEAD -- tests/test_demo.py" in script
    assert "git -C /app checkout HEAD -- ." not in script


def test_swebench_pro_reward_eval_script_falls_back_to_nodeid_files():
    from p2a.reward_specs import SWEBenchProRewardSpec

    spec = object.__new__(SWEBenchProRewardSpec)
    spec.metadata = {
        "FAIL_TO_PASS": json.dumps(["tests/test_demo.py::test_valid[1,1-expected0]"]),
        "PASS_TO_PASS": json.dumps(["tests/test_other.py::test_pass"]),
        "swebench_pro_repo_path": "/app",
    }

    script = spec._build_eval_script(
        {
            "run_script": Path("/tmp/run.sh"),
            "parser": Path("/tmp/parser.py"),
            "stdout": Path("/tmp/stdout.log"),
            "stderr": Path("/tmp/stderr.log"),
            "output": Path("/tmp/output.json"),
        }
    )

    assert "bash /tmp/run.sh tests/test_demo.py,tests/test_other.py >" in script
    assert "test_valid[1,1-expected0]" not in script




def test_swebench_pro_sandbox_execute_does_not_activate_verified_conda():
    from p2a.precompute.uni_agent_sandbox import UniAgentSandboxAdapter

    calls = []

    class FakeSandbox:
        async def exec_shell(self, command, **_kwargs):
            calls.append(command)
            return SimpleNamespace(stdout="", stderr="", exit_code=0)

    adapter = UniAgentSandboxAdapter(
        FakeSandbox(),
        swebench_pro=True,
        repo_path="/app",
    )

    adapter._execute_raw("echo ok")

    assert adapter.repo_path == "/app"
    assert adapter.swebench_verified is False
    assert calls == ["echo ok"]


def test_swebench_pro_create_sandbox_does_not_mark_verified(monkeypatch):
    from p2a.precompute import uni_agent_sandbox

    created = {}

    class FakeSandbox:
        pass

    fake_sandbox = FakeSandbox()

    def fake_builder(deployment):
        created["deployment"] = deployment
        return fake_sandbox

    monkeypatch.setattr(
        "env.sandbox.build_sandbox_from_deployment",
        fake_builder,
    )

    task = {
        "data_source": "swebench-pro",
        "FAIL_TO_PASS": json.dumps(["tests/test_demo.py::test_fail"]),
        "extra_info": {
            "tools_kwargs": {
                "env": {"deployment": {"image": "registry.local/swebench-pro:latest"}},
                "reward": {
                    "name": "swe_bench_pro",
                    "metadata": {"swebench_pro_repo_path": "/app"},
                }
            }
        },
    }

    adapter = uni_agent_sandbox.create_uni_agent_sandbox(task, instance_id="demo")

    assert adapter.swebench_pro is True
    assert adapter.swebench_verified is False
    assert adapter.repo_path == "/app"
    assert adapter.sandbox is fake_sandbox


def test_swebench_pro_arl_env_config_uses_app_session_cwd(monkeypatch):
    from p2a.precompute.uni_agent_sandbox import build_agent_env_config

    monkeypatch.setenv("ARL_GATEWAY_URL", "http://gateway")
    monkeypatch.delenv("ARL_MAX_REPLICAS", raising=False)
    task = {
        "data_source": "swebench-pro",
        "dockerhub_tag": "demo/pro:latest",
        "extra_info": {
            "tools_kwargs": {
                "env": {"deployment": {"image": "registry.local/swebench-pro:latest"}},
                "reward": {
                    "name": "swe_bench_pro",
                    "metadata": {"swebench_pro_repo_path": "/app"},
                },
            }
        },
    }

    config = build_agent_env_config(task, instance_id="demo", deployment="arl")

    assert config["deployment"]["session_cwd"] == "/app"


def test_swebench_pro_precompute_runner_uses_official_script_with_selected_files():
    from p2a.precompute.precompute_bonus_maps import _prepare_swebench_pro_test_script

    writes = {}
    commands = []

    class FakeEnv:
        repo_path = "/app"

        def write_file(self, path, content):
            writes[str(path)] = content

        def _run(self, command, timeout=None):
            commands.append(command)
            return "", ""

    task = {
        "run_tests": "pytest \"$@\"\n",
        "FAIL_TO_PASS": json.dumps(["tests/test_demo.py::test_valid[1,1-expected0]"]),
        "PASS_TO_PASS": json.dumps(["tests/test_other.py::test_pass"]),
        "selected_test_files_to_run": json.dumps(["tests/test_demo.py", "tests/test_other.py"]),
    }

    diag = _prepare_swebench_pro_test_script(FakeEnv(), task, "/tmp/p2a_runner.sh")
    wrapper = writes["/tmp/p2a_runner.sh"]

    assert diag["swebench_pro_run_tests_source"] == "metadata"
    assert "cd /app" in wrapper
    assert "/opt/miniconda3/envs/testbed" not in wrapper
    assert "tests/test_demo.py,tests/test_other.py" in wrapper
    assert "test_valid[1,1-expected0]" not in wrapper
    assert writes["/tmp/p2a_swebench_pro_official_run.sh"] == "pytest \"$@\"\n"
    assert any("chmod +x /tmp/p2a_runner.sh /tmp/p2a_swebench_pro_official_run.sh" in command for command in commands)


def test_swebench_pro_precompute_runner_falls_back_to_f2p_files():
    from p2a.precompute.precompute_bonus_maps import _prepare_swebench_pro_test_script

    writes = {}
    commands = []

    class FakeEnv:
        repo_path = "/app"

        def write_file(self, path, text):
            writes[path] = text

        def _run(self, command, timeout=None):
            commands.append(command)
            return "", ""

    task = {
        "run_tests": "pytest \"$@\"\n",
        "FAIL_TO_PASS": json.dumps(
            [
                "tests/test_demo.py::test_valid[1,1-expected0]",
                "tests/test_second.py::test_fail",
                "tests/test_demo.py::test_other",
            ]
        ),
        "PASS_TO_PASS": json.dumps(["tests/test_p2p.py::test_pass"]),
        "selected_test_files_to_run": "[]",
    }

    diag = _prepare_swebench_pro_test_script(FakeEnv(), task, "/tmp/p2a_runner.sh")
    wrapper = writes["/tmp/p2a_runner.sh"]

    assert diag["swebench_pro_selected_files"] == ["tests/test_demo.py", "tests/test_second.py"]
    assert "tests/test_demo.py,tests/test_second.py" in wrapper
    assert "tests/test_p2p.py" not in wrapper
    assert any("chmod +x /tmp/p2a_runner.sh /tmp/p2a_swebench_pro_official_run.sh" in command for command in commands)


def test_swebench_pro_precompute_short_circuits_empty_f2p():
    from p2a.precompute import precompute_bonus_maps as bonus_maps

    modified = [
        {
            "name": "demo",
            "qualified_name": "demo",
            "file_path": "pkg/demo.py",
            "start_line": 10,
            "end_line": 12,
        }
    ]

    class FakeEnv:
        swebench_verified = False
        swebench_pro = True
        repo_path = "/app"
        alt_path = "/root"

        def start(self):
            pass

        def close(self):
            pass

        def checkout_buggy_commit(self, task, *, instance_id):
            return {"buggy_checkout_ref": "abc123^", "buggy_checkout_exit": 0}

        def _run(self, command, timeout=None):
            raise AssertionError("empty F2P Pro rows must not run tests")

    with (
        patch.object(bonus_maps, "find_modified_callables_from_task", return_value=modified),
        patch.object(bonus_maps, "find_newly_created_callables", return_value=[]),
        patch("p2a.precompute.uni_agent_sandbox.create_uni_agent_sandbox", return_value=FakeEnv()),
    ):
        result = bonus_maps.compute_dynamic_bonus_map(
            {
                "instance_id": "demo",
                "data_source": "swebench-pro",
                "patch": "diff --git a/pkg/demo.py b/pkg/demo.py\n",
                "FAIL_TO_PASS": json.dumps([]),
            }
        )

    assert result["case_type"] == "no_f2p"
    assert result["reason_code"] == "missing_fail_to_pass"


def test_swebench_pro_precompute_missing_scripts_is_precompute_failure():
    from p2a.precompute import precompute_bonus_maps as bonus_maps

    modified = [
        {
            "name": "demo",
            "qualified_name": "demo",
            "file_path": "pkg/demo.py",
            "start_line": 10,
            "end_line": 12,
        }
    ]

    class FakeEnv:
        swebench_verified = False
        swebench_pro = True
        repo_path = "/app"
        alt_path = "/root"

        def start(self):
            pass

        def close(self):
            pass

        def checkout_buggy_commit(self, task, *, instance_id):
            return {"buggy_checkout_ref": "abc123^", "buggy_checkout_exit": 0}

        def _run(self, command, timeout=None):
            raise AssertionError("missing Pro scripts must not run commands")

    with (
        patch.object(bonus_maps, "find_modified_callables_from_task", return_value=modified),
        patch.object(bonus_maps, "find_newly_created_callables", return_value=[]),
        patch("p2a.precompute.uni_agent_sandbox.create_uni_agent_sandbox", return_value=FakeEnv()),
    ):
        result = bonus_maps.compute_dynamic_bonus_map(
            {
                "instance_id": "demo",
                "data_source": "swebench-pro",
                "patch": "diff --git a/pkg/demo.py b/pkg/demo.py\n",
                "FAIL_TO_PASS": json.dumps(["tests/test_demo.py::test_bug"]),
            }
        )

    assert result["case_type"] == "precompute_failed"
    assert result["reason_code"] == "missing_swebench_pro_scripts"
    assert result["precompute_failure"] is True
    assert result["failure_kind"] == "missing_swebench_pro_scripts"
    assert result["missing_swebench_pro_fields"] == ["run_tests", "swebench_pro_parser"]


def test_swebench_pro_task_detection_does_not_match_generic_python_docker_rows():
    from p2a.precompute.uni_agent_sandbox import _is_swebench_pro_task

    assert _is_swebench_pro_task(
        {
            "data_source": "r2e-gym-subset",
            "repo_language": "python",
            "dockerhub_tag": "example-tag",
            "before_repo_set_cmd": "git checkout abc123",
        }
    ) is False
    assert _is_swebench_pro_task(
        {
            "extra_info": {
                "tools_kwargs": {
                    "reward": {
                        "metadata": {
                            "data_source": "swebench-pro",
                            "docker_image": "jefzda/sweap-images:example-tag",
                            "repo_path": "/app",
                        }
                    }
                }
            }
        }
    ) is True
