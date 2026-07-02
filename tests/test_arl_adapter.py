"""Non-live regression tests for the ARL Uni-Agent adapter."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import uuid
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from env.deployment import ArlDeployment, ArlDeploymentConfig, make_env_config
from env.images import select_r2e_image


ARL_IMAGE_ENV_KEYS = (
    "P2A_ARL_IMAGE_OVERRIDES_JSON",
)


@contextmanager
def patched_env(updates: dict[str, str] | None = None, *, remove: tuple[str, ...] = ()):
    keys = set(updates or {}) | set(remove)
    old = {key: os.environ.get(key) for key in keys}
    try:
        for key in remove:
            os.environ.pop(key, None)
        if updates:
            os.environ.update(updates)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class ArlAdapterTests(unittest.TestCase):
    def test_agent_config_pins_qwen3_coder_tool_parser(self) -> None:
        text = Path("env/agent_config_arl.yaml").read_text(encoding="utf-8")
        self.assertIn("_target_: env.agent_loop.ArlUniAgentLoop", text)
        self.assertIn("tool_parser: qwen3_coder", text)

    def test_make_env_config_maps_arl_deployment_without_uni_agent_union(self) -> None:
        env_config = make_env_config(
            {
                "type": "arl",
                "image": "registry.local/r2e:latest",
                "gateway_url": "http://gateway",
                "namespace": "p2a",
                "profile": "gpu",
                "api_key": "secret",
                "experiment_id": "exp-1",
                "startup_timeout": 12,
                "max_replicas": 2,
                "unknown_future_field": "ignored",
            },
            env_variables={"PIP_CACHE_DIR": "~/.cache/pip"},
            post_setup_cmd="git checkout abc123",
            tool_install_dir="/tools",
        )

        self.assertIsInstance(env_config.deployment, ArlDeploymentConfig)
        self.assertEqual(env_config.deployment.type, "arl")
        self.assertEqual(env_config.deployment.image, "registry.local/r2e:latest")
        self.assertEqual(env_config.deployment.gateway_url, "http://gateway")
        self.assertEqual(env_config.deployment.namespace, "p2a")
        self.assertEqual(env_config.deployment.profile, "gpu")
        self.assertEqual(env_config.deployment.api_key, "secret")
        self.assertEqual(env_config.deployment.experiment_id, "exp-1")
        self.assertEqual(env_config.deployment.startup_timeout, 12)
        self.assertEqual(env_config.deployment.max_replicas, 2)
        self.assertEqual(env_config.deployment.session_cwd, "/testbed")
        self.assertEqual(env_config.deployment.startup_env_variables, {"PIP_CACHE_DIR": "~/.cache/pip"})
        self.assertEqual(env_config.deployment.shell_post_setup_cmd, "git checkout abc123")
        self.assertIsNone(env_config.env_variables)
        self.assertIsNone(env_config.post_setup_cmd)
        self.assertEqual(str(env_config.tool_install_dir), "/tools")

    def test_deployment_config_builds_direct_arl_deployment(self) -> None:
        config = ArlDeploymentConfig(image="registry.local/r2e:latest", gateway_url="http://gateway")
        deployment = config.get_deployment(run_id="run-1")

        self.assertIsInstance(deployment, ArlDeployment)
        self.assertEqual(deployment.run_id, "run-1")
        with self.assertRaisesRegex(Exception, "runtime not started"):
            _ = deployment.runtime

    def test_arl_deployment_filters_managed_session_kwargs_by_sdk_signature(self) -> None:
        from env.deployment import _supported_kwargs, _validate_managed_session_signature

        def old_sdk(image, experiment_id, namespace, gateway_url, timeout, workspace_dir, max_replicas):
            return None

        def new_sdk(image, experiment_id, gateway_url, timeout, resources, workspace_dir, profile, api_key):
            return None

        kwargs = {
            "image": "img",
            "experiment_id": "exp",
            "namespace": "arl",
            "profile": "default",
            "gateway_url": "http://gateway",
            "timeout": 1,
            "resources": {"cpu": 1},
            "workspace_dir": "/workspace",
            "max_replicas": 1,
            "api_key": "secret",
        }

        self.assertEqual(
            set(_supported_kwargs(old_sdk, kwargs)),
            {"image", "experiment_id", "namespace", "gateway_url", "timeout", "workspace_dir", "max_replicas"},
        )
        self.assertEqual(
            set(_supported_kwargs(new_sdk, kwargs)),
            {"image", "experiment_id", "gateway_url", "timeout", "resources", "workspace_dir", "profile", "api_key"},
        )
        _validate_managed_session_signature(new_sdk, {**kwargs, "namespace": "p2a", "max_replicas": None})
        with self.assertRaisesRegex(RuntimeError, "max_replicas"):
            _validate_managed_session_signature(new_sdk, {**kwargs, "namespace": "default", "max_replicas": 2})

    def test_arl_deployment_attaches_managed_session_without_pool_ref(self) -> None:
        from env.deployment import _attach_managed_session_payload, _missing_pool_ref_payload

        class MissingPoolRefError(Exception):
            def errors(self):
                return [
                    {
                        "type": "missing",
                        "loc": ("poolRef",),
                        "input": {
                            "id": "gw-1",
                            "sandboxName": "gw-1",
                            "namespace": "arl",
                            "podIP": "172.31.0.10",
                            "podName": "managed-pod",
                            "managed": True,
                            "experimentId": "exp-1",
                        },
                    }
                ]

        session = SimpleNamespace(namespace="arl")
        payload = _missing_pool_ref_payload(MissingPoolRefError())
        self.assertIsNotNone(payload)
        info = _attach_managed_session_payload(session, payload)

        self.assertEqual(session._session_id, "gw-1")
        self.assertEqual(session.pool_ref, "")
        self.assertIs(session._session_info, info)
        self.assertEqual(info.id, "gw-1")
        self.assertEqual(info.pool_ref, "")
        self.assertEqual(info.pod_name, "managed-pod")

    def test_arl_runtime_uses_managed_session_private_id(self) -> None:
        from env.runtime import ArlRuntime

        class FakeClient:
            base_url = "http://gateway"

        class FakeSession:
            _client = FakeClient()
            _session_id = "arl-session-1"

        runtime = ArlRuntime(FakeSession(), run_id="run-1")
        self.assertEqual(runtime._arl_session_id, "arl-session-1")

    def test_arl_runtime_strips_terminal_control_sequences(self) -> None:
        from env.runtime import _strip_terminal_controls

        self.assertEqual(_strip_terminal_controls("\x1b[?2004hhello\r\n"), "hello\n")

    def test_arl_runtime_execute_backed_sessions_replay_only_repeatable_startup(self) -> None:
        from env.runtime import ArlRuntime

        class FakeClient:
            base_url = "http://gateway"

        class FakeSession:
            _client = FakeClient()
            _session_id = "arl-session-1"

        runtime = ArlRuntime(
            FakeSession(),
            run_id="run-1",
            startup_commands=["export PAGER=cat"],
            one_time_startup_commands=["mv /testbed/run_tests.sh /tmp/run_tests.sh"],
        )
        scripts = []
        commands = []
        runtime._write_bytes_sync = lambda data, _path: scripts.append(data.decode("utf-8"))
        runtime._exec_sync = lambda command, _timeout=None: commands.append(command) or SimpleNamespace(
            stdout="",
            stderr="",
            exit_code=0,
        )

        runtime._ensure_session_sync("default", None)
        runtime._initialized_sessions.remove("default")
        runtime._ensure_session_sync("default", None)

        self.assertIn("export PAGER=cat", scripts[0])
        self.assertIn("mv /testbed/run_tests.sh /tmp/run_tests.sh", scripts[0])
        self.assertIn("export PAGER=cat", scripts[1])
        self.assertNotIn("mv /testbed/run_tests.sh /tmp/run_tests.sh", scripts[1])
        self.assertEqual(len(commands), 2)

    def test_arl_runtime_execute_backed_sessions_preserve_cwd_and_exported_env(self) -> None:
        from swerex.runtime.abstract import BashAction

        from env.runtime import ArlRuntime

        class FakeClient:
            base_url = "http://gateway"

        class FakeSession:
            _client = FakeClient()
            _session_id = "arl-session-1"

            def execute(self, steps):
                results = []
                for step in steps:
                    command = list(step["command"])
                    if command[:2] == ["bash", "-lc"]:
                        command[1] = "-c"
                    completed = subprocess.run(
                        command,
                        text=True,
                        capture_output=True,
                        timeout=step.get("timeout") or 60,
                        check=False,
                    )
                    results.append(
                        SimpleNamespace(
                            output=SimpleNamespace(
                                exit_code=completed.returncode,
                                stdout=completed.stdout,
                                stderr=completed.stderr,
                            )
                        )
                    )
                return SimpleNamespace(results=results)

        runtime = ArlRuntime(FakeSession(), run_id=f"test-{uuid.uuid4().hex}", session_cwd="/tmp")

        async def _run():
            try:
                first = await runtime.run_in_session(
                    BashAction(command="printf '%s' \"$PWD\"", timeout=10),
                )
                second = await runtime.run_in_session(
                    BashAction(command="cd / && export P2A_ARL_TEST_VALUE=kept", timeout=10),
                )
                third = await runtime.run_in_session(
                    BashAction(command="printf '%s %s' \"$PWD\" \"$P2A_ARL_TEST_VALUE\"", timeout=10),
                )
                return first, second, third
            finally:
                await runtime.close()

        first, second, third = asyncio.run(_run())

        self.assertEqual(first.output, "/tmp")
        self.assertEqual(first.exit_code, 0)
        self.assertEqual(first.failure_reason, "")
        self.assertEqual(second.exit_code, 0)
        self.assertEqual(second.output, "")
        self.assertEqual(second.failure_reason, "")
        self.assertEqual(third.exit_code, 0)
        self.assertEqual(third.failure_reason, "")
        self.assertEqual(third.output, "/ kept")

    def test_arl_runtime_failed_startup_is_not_cached(self) -> None:
        from env.runtime import ArlRuntime

        class FakeClient:
            base_url = "http://gateway"

        class FakeSession:
            _client = FakeClient()
            _session_id = "arl-session-1"

        runtime = ArlRuntime(
            FakeSession(),
            run_id="run-1",
            one_time_startup_commands=["false"],
        )
        runtime._write_bytes_sync = lambda _data, _path: None
        runtime._exec_sync = lambda _command, _timeout=None: SimpleNamespace(
            stdout="failed setup",
            stderr="",
            exit_code=1,
        )

        with self.assertRaisesRegex(RuntimeError, "session startup failed"):
            runtime._ensure_session_sync("default", None)

        self.assertNotIn("default", runtime._initialized_sessions)
        self.assertNotIn("default", runtime._completed_one_time_startup_sessions)

    def test_arl_runtime_retries_gateway_execute_refusal(self) -> None:
        from env.runtime import ArlRuntime

        class FakeClient:
            base_url = "http://gateway"

        class FakeSession:
            _client = FakeClient()
            _session_id = "arl-session-1"

            def __init__(self) -> None:
                self.calls = 0

            def execute(self, steps):
                self.calls += 1
                if self.calls == 1:
                    output = SimpleNamespace(
                        exit_code=1,
                        stdout="",
                        stderr=(
                            "gRPC Execute failed: rpc error: code = Unavailable "
                            "desc = connection error: transport: Error while dialing: "
                            "connect: connection refused"
                        ),
                    )
                else:
                    output = SimpleNamespace(exit_code=0, stdout="ok\n", stderr="")
                return SimpleNamespace(results=[SimpleNamespace(output=output)])

        session = FakeSession()
        runtime = ArlRuntime(session, run_id="run-1")
        with patch("env.runtime.time.sleep", return_value=None):
            output = runtime._exec_sync("echo ok")

        self.assertEqual(session.calls, 2)
        self.assertEqual(output.exit_code, 0)
        self.assertEqual(output.stdout, "ok\n")

    def test_arl_runtime_does_not_retry_user_command_failures(self) -> None:
        from env.runtime import ArlRuntime

        class FakeClient:
            base_url = "http://gateway"

        class FakeSession:
            _client = FakeClient()
            _session_id = "arl-session-1"

            def __init__(self) -> None:
                self.calls = 0

            def execute(self, steps):
                self.calls += 1
                output = SimpleNamespace(exit_code=1, stdout="", stderr="pytest failed")
                return SimpleNamespace(results=[SimpleNamespace(output=output)])

        session = FakeSession()
        runtime = ArlRuntime(session, run_id="run-1")
        output = runtime._exec_sync("pytest")

        self.assertEqual(session.calls, 1)
        self.assertEqual(output.exit_code, 1)
        self.assertEqual(output.stderr, "pytest failed")

    def test_deployment_waits_until_runtime_is_executable(self) -> None:
        deployment = ArlDeployment.from_config(
            ArlDeploymentConfig(image="registry.local/r2e:latest", gateway_url="http://gateway"),
            run_id="run-1",
        )
        calls = {"count": 0}

        class FakeRuntime:
            async def is_alive(self, *, timeout=None):
                calls["count"] += 1
                return SimpleNamespace(is_alive=calls["count"] >= 3, message="not ready")

        deployment._runtime = FakeRuntime()

        async def no_sleep(_delay):
            return None

        with patch("env.deployment.asyncio.sleep", side_effect=no_sleep):
            asyncio.run(deployment._wait_until_runtime_ready(10))

        self.assertEqual(calls["count"], 3)

    def test_deployment_config_accepts_required_bash_session(self) -> None:
        config = ArlDeploymentConfig.from_mapping(
            {
                "type": "arl",
                "image": "registry.local/r2e:latest",
                "gateway_url": "http://gateway",
                "require_bash_session": True,
            }
        )

        self.assertTrue(config.require_bash_session)

    def test_deployment_config_uses_startup_timeout_for_required_bash_session(self) -> None:
        required = ArlDeploymentConfig(
            image="registry.local/r2e:latest",
            gateway_url="http://gateway",
            require_bash_session=True,
            startup_timeout=240,
        )
        optional = ArlDeploymentConfig(image="registry.local/r2e:latest", gateway_url="http://gateway")

        self.assertEqual(required.bash_session_preflight_timeout(10), 240)
        self.assertEqual(optional.bash_session_preflight_timeout(10), 10)

    def test_deployment_config_accepts_legacy_required_session_alias(self) -> None:
        config = ArlDeploymentConfig.from_mapping(
            {
                "type": "arl",
                "image": "registry.local/r2e:latest",
                "gateway_url": "http://gateway",
                "require_interactive_shell": True,
            }
        )

        self.assertTrue(config.require_bash_session)

    def test_uni_agent_sandbox_adapter_runs_post_setup_via_execute(self) -> None:
        from p2a.precompute.uni_agent_sandbox import UniAgentSandboxAdapter

        calls = {"start": 0, "commands": []}

        class FakeRuntime:
            async def execute(self, command):
                calls["commands"].append(command)
                return SimpleNamespace(stdout="", stderr="", exit_code=0)

        fake_env = SimpleNamespace(
            start=lambda: calls.__setitem__("start", calls["start"] + 1),
            deployment=SimpleNamespace(runtime=FakeRuntime()),
        )

        adapter = UniAgentSandboxAdapter(
            fake_env,
            startup_env_variables={"PAGER": "cat"},
            post_setup_cmd="echo ready",
        )
        adapter.start()

        self.assertEqual(calls["start"], 1)
        self.assertEqual(len(calls["commands"]), 1)
        command = calls["commands"][0].command
        self.assertIn("export PAGER=cat", command)
        self.assertIn("echo ready", command)

    def test_agent_loop_maps_arl_env_without_pydantic_deployment_union(self) -> None:
        import env as env_package

        created = {}

        def fake_agent_env(run_id, env_config):
            created["run_id"] = run_id
            created["env_config"] = env_config
            return SimpleNamespace(run_id=run_id, env_config=env_config)

        fake_agent_loop = ModuleType("uni_agent.agent_loop")

        class FakeUniAgentLoop:
            def _init_env(self, config_dict):
                return SimpleNamespace(super_config=config_dict)

        fake_agent_loop.UniAgentLoop = FakeUniAgentLoop
        fake_interaction = ModuleType("uni_agent.interaction")
        fake_interaction.AgentEnv = fake_agent_env

        config_dict = {
            "deployment": {
                "type": "arl",
                "image": "registry.local/r2e:latest",
                "gateway_url": "http://gateway",
            },
            "env_variables": {"PIP_CACHE_DIR": "~/.cache/pip"},
            "post_setup_cmd": "git checkout abc123",
            "tool_install_dir": "/tools",
        }

        try:
            sys.modules.pop("env.agent_loop", None)
            if hasattr(env_package, "agent_loop"):
                delattr(env_package, "agent_loop")
            with patch.dict(
                sys.modules,
                {
                    "uni_agent.agent_loop": fake_agent_loop,
                    "uni_agent.interaction": fake_interaction,
                },
            ):
                from env import agent_loop as agent_loop_module

                loop = object.__new__(agent_loop_module.ArlUniAgentLoop)
                loop.run_id = "run-1"
                agent_env = loop._init_env(config_dict)
        finally:
            sys.modules.pop("env.agent_loop", None)
            if hasattr(env_package, "agent_loop"):
                delattr(env_package, "agent_loop")

        self.assertIs(agent_env.env_config, created["env_config"])
        self.assertEqual(created["run_id"], "run-1")
        self.assertEqual(agent_env.env_config.deployment.type, "arl")
        self.assertEqual(agent_env.env_config.deployment.image, "registry.local/r2e:latest")
        self.assertEqual(agent_env.env_config.deployment.gateway_url, "http://gateway")
        self.assertEqual(
            agent_env.env_config.deployment.startup_env_variables,
            {"PIP_CACHE_DIR": "~/.cache/pip"},
        )
        self.assertEqual(agent_env.env_config.deployment.shell_post_setup_cmd, "git checkout abc123")
        self.assertIsNone(agent_env.env_config.env_variables)
        self.assertIsNone(agent_env.env_config.post_setup_cmd)

    def test_agent_loop_attaches_p2a_traces_from_rollout_cache_without_uni_agent_patch(self) -> None:
        import env as env_package

        fake_agent_loop = ModuleType("uni_agent.agent_loop")

        class FakeUniAgentLoop:
            pass

        fake_agent_loop.UniAgentLoop = FakeUniAgentLoop
        fake_interaction = ModuleType("uni_agent.interaction")
        fake_interaction.AgentEnv = lambda run_id, env_config: SimpleNamespace(run_id=run_id, env_config=env_config)

        class FakeToolsManager:
            async def parse_action(self, model_output):
                if "no-tool" in model_output:
                    return model_output, []
                return "inspect file", [
                    {
                        "function": {
                            "name": "str_replace_editor",
                            "arguments": {
                                "command": "view",
                                "path": "/testbed/pkg/demo.py",
                                "view_range": [1, 5],
                            },
                        }
                    }
                ]

        try:
            sys.modules.pop("env.agent_loop", None)
            if hasattr(env_package, "agent_loop"):
                delattr(env_package, "agent_loop")
            with patch.dict(
                sys.modules,
                {
                    "uni_agent.agent_loop": fake_agent_loop,
                    "uni_agent.interaction": fake_interaction,
                },
            ):
                from env import agent_loop as agent_loop_module

                loop = object.__new__(agent_loop_module.ArlUniAgentLoop)
                loop.tools_manager = FakeToolsManager()
                loop._p2a_run_kwargs = {
                    "tools_kwargs": {
                        "reward": {
                            "metadata": {
                                "instance_id": "demo__abc123",
                            }
                        }
                    }
                }
                loop._p2a_run_config = {}
                interaction_result = {
                    "trajectory": [
                        SimpleNamespace(step_idx=1, response="tool response", exit_reason="completed"),
                        SimpleNamespace(step_idx=2, response="no-tool response", exit_reason="format_error"),
                    ],
                    "rollout_cache": {
                        "response_mask": [1, 1, 0, 0, 1, 1, 1],
                        "extra_fields": {},
                    },
                }
                with patched_env({"UNI_AGENT_P2A_TRACE": "1"}):
                    asyncio.run(loop._attach_p2a_extra_fields(interaction_result))
        finally:
            sys.modules.pop("env.agent_loop", None)
            if hasattr(env_package, "agent_loop"):
                delattr(env_package, "agent_loop")

        extra_fields = interaction_result["rollout_cache"]["extra_fields"]
        self.assertEqual(extra_fields["instance_id"], "demo__abc123")
        self.assertEqual(extra_fields["response_text"], "tool response\nno-tool response")
        self.assertEqual(
            extra_fields["p2a_step_traces"],
            [
                {
                    "step_idx": 1,
                    "response_start": 0,
                    "response_end": 2,
                    "thought": "inspect file",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "str_replace_editor",
                                "arguments": {
                                    "command": "view",
                                    "path": "/testbed/pkg/demo.py",
                                    "view_range": [1, 5],
                                },
                            }
                        }
                    ],
                    "parse_error": None,
                },
                {
                    "step_idx": 2,
                    "response_start": 4,
                    "response_end": 7,
                    "thought": "no-tool response",
                    "tool_calls": [],
                    "parse_error": "No function call found in the response.",
                },
            ],
        )

    def test_image_override_and_pair_diag_routing_are_explicit(self) -> None:
        # 1. exact per-instance override wins.
        with patched_env(
            {"P2A_ARL_IMAGE_OVERRIDES_JSON": '{"demo__abc123": "registry.local/custom:tag"}'},
        ):
            self.assertEqual(
                select_r2e_image(instance_id="demo__abc123", docker_image="namanjain12/demo_final"),
                "registry.local/custom:tag",
            )

        # 2. a raw namanjain12 R2E ref maps to its pair-diag mirror.
        with patched_env(remove=ARL_IMAGE_ENV_KEYS):
            self.assertEqual(
                select_r2e_image(
                    instance_id="orange3__abcdef1234",
                    docker_image="namanjain12/orange3_final:abcdef1234",
                ),
                "pair-diag-cn-guangzhou.cr.volces.com/code/orange3_final:abcdef1234",
            )

        # 3. an image already on the pair-diag mirror passes through untouched.
        with patched_env(remove=ARL_IMAGE_ENV_KEYS):
            ref = "pair-diag-cn-guangzhou.cr.volces.com/code/django_final:abcdef1234"
            self.assertEqual(
                select_r2e_image(instance_id="django__abcdef1234", docker_image=ref),
                ref,
            )

    def test_p2a_precompute_arl_config_mapping_uses_overrides(self) -> None:
        from p2a.precompute.uni_agent_sandbox import build_agent_env_config

        task = {
            "docker_image": "namanjain12/demo_final",
            "extra_info": {
                "tools_kwargs": {
                    "reward": {
                        "metadata": {
                            "repo": "demo",
                            "instance_id": "demo__abc123",
                        }
                    }
                }
            },
        }
        env_vars = {
            "ARL_GATEWAY_URL": "http://gateway",
            "ARL_NAMESPACE": "p2a",
            "ARL_EXPERIMENT_ID": "exp-precompute",
            "ARL_TIMEOUT": "12",
            "ARL_STARTUP_TIMEOUT": "34",
            "ARL_MAX_REPLICAS": "3",
            "P2A_ARL_IMAGE_OVERRIDES_JSON": '{"demo__abc123": "registry.local/custom:tag"}',
        }
        with patched_env(env_vars, remove=("ARL_SWEREX_ENDPOINT_HOST", "ARL_SWEREX_COMMAND")):
            config = build_agent_env_config(task, instance_id="demo__abc123", deployment="arl")

        deployment = config["deployment"]
        self.assertEqual(deployment["type"], "arl")
        self.assertEqual(deployment["image"], "registry.local/custom:tag")
        self.assertEqual(deployment["gateway_url"], "http://gateway")
        self.assertEqual(deployment["namespace"], "p2a")
        self.assertEqual(deployment["experiment_id"], "exp-precompute")
        self.assertEqual(deployment["timeout"], 12.0)
        self.assertEqual(deployment["startup_timeout"], 34.0)
        self.assertEqual(deployment["max_replicas"], 3)
        self.assertEqual(deployment["session_cwd"], "/testbed")
        self.assertNotIn("endpoint_host", deployment)
        self.assertNotIn("command", deployment)
        self.assertIn("PIP_CACHE_DIR", config["env_variables"])
        self.assertIn("ln -s /testbed/.venv", config["post_setup_cmd"])

    def test_r2e_buggy_checkout_materializes_old_sources_on_head_mismatch(self) -> None:
        from p2a.precompute.uni_agent_sandbox import UniAgentSandboxAdapter

        class FakeSandbox(UniAgentSandboxAdapter):
            repo_path = "/testbed"

            def __init__(self):
                self.writes = {}

            def _execute_raw(self, command: str, timeout: int | float | None = None):
                if "git rev-parse --verify" in command:
                    return "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n", "", 0
                if "git checkout" in command:
                    return "", "", 0
                if "git rev-parse HEAD" in command:
                    return "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n", "", 0
                return "", "", 0

            def write_file(self, path: str | Path, content: str) -> None:
                self.writes[str(path)] = content

        task = {
            "base_commit": "abc123",
            "parsed_commit_content": {
                "file_diffs": [
                    {
                        "header": {"file": {"path": "pkg/demo.py"}},
                        "old_file_content": "def demo():\n    return 'old'\n",
                        "new_file_content": "def demo():\n    return 'new'\n",
                    }
                ]
            },
        }

        sandbox = FakeSandbox()
        diag = sandbox.checkout_buggy_commit(task, instance_id="demo__abc123")

        self.assertFalse(diag["buggy_checkout_verified"])
        self.assertEqual(diag["sandbox_code_state"], "old_sources_materialized")
        self.assertTrue(diag["buggy_source_materialized"])
        self.assertEqual(diag["buggy_materialized_files"], ["pkg/demo.py"])
        self.assertEqual(sandbox.writes["/testbed/pkg/demo.py"], "def demo():\n    return 'old'\n")
        self.assertNotIn("sandbox_code_state_mismatch", diag)

    def test_r2e_buggy_checkout_accepts_hash_ref_when_expected_stdout_is_empty(self) -> None:
        from p2a.precompute.uni_agent_sandbox import UniAgentSandboxAdapter

        commit = "30379ea6e225e37833a764ac2da7b7fadf5fe374"

        class FakeSandbox(UniAgentSandboxAdapter):
            repo_path = "/testbed"
            swebench_pro = False

            def __init__(self):
                pass

            def _execute_raw(self, command: str, timeout: int | float | None = None):
                if "git rev-parse --verify" in command:
                    return "", "", 0
                if "git checkout" in command:
                    return "", "", 0
                if "git rev-parse HEAD" in command:
                    return f"{commit}\n", "", 0
                return "", "", 0

            def write_file(self, path: str | Path, content: str) -> None:
                raise AssertionError("no old sources should be written")

        diag = FakeSandbox().checkout_buggy_commit({"base_commit": commit}, instance_id="sympy__sympy-13372")

        self.assertTrue(diag["buggy_checkout_verified"])
        self.assertEqual(diag["sandbox_code_state"], "git_checkout_verified")
        self.assertEqual(diag["buggy_checkout_verification"], "commit_ref_head")
        self.assertNotIn("sandbox_code_state_mismatch", diag)

    def test_r2e_buggy_checkout_reports_code_state_mismatch_without_fallback(self) -> None:
        from p2a.precompute.uni_agent_sandbox import UniAgentSandboxAdapter

        class FakeSandbox(UniAgentSandboxAdapter):
            repo_path = "/testbed"

            def __init__(self):
                pass

            def _execute_raw(self, command: str, timeout: int | float | None = None):
                if "git rev-parse --verify" in command:
                    return "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n", "", 0
                if "git checkout" in command:
                    return "", "pathspec not found", 1
                if "git rev-parse HEAD" in command:
                    return "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n", "", 0
                return "", "", 0

            def write_file(self, path: str | Path, content: str) -> None:
                raise AssertionError("no old sources should be written")

        diag = FakeSandbox().checkout_buggy_commit({"base_commit": "abc123"}, instance_id="demo__abc123")

        self.assertFalse(diag["buggy_checkout_verified"])
        self.assertFalse(diag["buggy_source_materialized"])
        self.assertTrue(diag["sandbox_code_state_mismatch"])
        self.assertEqual(diag["sandbox_code_state"], "mismatch")
        self.assertIn("pathspec not found", diag["buggy_checkout_stderr"])

    def test_swebench_prepare_uses_metadata_and_skips_scikit_editable_install(self) -> None:
        from p2a.precompute.precompute_bonus_maps import _prepare_swebench_test_script

        class FakeEnv:
            def write_file(self, path: str, content: str) -> None:
                Path(path).write_text(content, encoding="utf-8")

            def _run(self, command: str, timeout: int | float | None = None) -> tuple[str, str]:
                if "python - <<'PY'" not in command:
                    return "", ""
                result = subprocess.run(command, shell=True, text=True, capture_output=True, check=False)
                return result.stdout, result.stderr

        with tempfile.TemporaryDirectory() as tmp:
            script = str(Path(tmp) / "run_tests.sh")
            diag = _prepare_swebench_test_script(
                FakeEnv(),
                {
                    "repo": "scikit-learn/scikit-learn",
                    "run_tests": "\n".join(
                        [
                            "#!/bin/bash",
                            "python -m pip install -v --no-use-pep517 --no-build-isolation -e .",
                            "pytest sklearn/tests",
                        ]
                    ),
                },
                script,
            )

            text = Path(script).read_text(encoding="utf-8")

        self.assertEqual(diag["swebench_run_tests_source"], "metadata")
        self.assertIn("changed=True", diag["swebench_test_script_patch_stdout"])
        self.assertIn("skipped_editable=1", diag["swebench_test_script_patch_stdout"])
        self.assertIn("skip editable install; sklearn=", text)
        self.assertNotIn("pip install -v --no-use-pep517 --no-build-isolation -e .", text)

    def test_swebench_prepare_adds_no_build_isolation_for_regular_editable_install(self) -> None:
        from p2a.precompute.precompute_bonus_maps import _prepare_swebench_test_script

        class FakeEnv:
            def write_file(self, path: str, content: str) -> None:
                Path(path).write_text(content, encoding="utf-8")

            def _run(self, command: str, timeout: int | float | None = None) -> tuple[str, str]:
                if "python - <<'PY'" not in command:
                    return "", ""
                result = subprocess.run(command, shell=True, text=True, capture_output=True, check=False)
                return result.stdout, result.stderr

        with tempfile.TemporaryDirectory() as tmp:
            script = str(Path(tmp) / "run_tests.sh")
            diag = _prepare_swebench_test_script(
                FakeEnv(),
                {
                    "repo": "django/django",
                    "run_tests": "\n".join(
                        [
                            "#!/bin/bash",
                            "python -m pip install -e .",
                            "pytest tests",
                        ]
                    ),
                },
                script,
            )

            text = Path(script).read_text(encoding="utf-8")

        self.assertIn("changed=True", diag["swebench_test_script_patch_stdout"])
        self.assertIn("skipped_editable=0", diag["swebench_test_script_patch_stdout"])
        self.assertIn("python -m pip install --no-build-isolation -e .", text)

    def test_swebench_prepare_replaces_sphinx_tox_current_env(self) -> None:
        from p2a.precompute.precompute_bonus_maps import _prepare_swebench_test_script

        class FakeEnv:
            def write_file(self, path: str, content: str) -> None:
                Path(path).write_text(content, encoding="utf-8")

            def _run(self, command: str, timeout: int | float | None = None) -> tuple[str, str]:
                if "python - <<'PY'" not in command:
                    return "", ""
                result = subprocess.run(command, shell=True, text=True, capture_output=True, check=False)
                return result.stdout, result.stderr

        with tempfile.TemporaryDirectory() as tmp:
            script = str(Path(tmp) / "run_tests.sh")
            diag = _prepare_swebench_test_script(
                FakeEnv(),
                {
                    "repo": "sphinx-doc/sphinx",
                    "run_tests": "\n".join(
                        [
                            "#!/bin/bash",
                            "python -m pip install -e .[test]",
                            "tox --current-env -epy39 -v -- tests/test_domain_cpp.py",
                        ]
                    ),
                },
                script,
            )

            text = Path(script).read_text(encoding="utf-8")

        self.assertIn("replaced_tox=1", diag["swebench_test_script_patch_stdout"])
        self.assertIn("python -m pytest -rA --color=no -vv tests/test_domain_cpp.py", text)
        self.assertNotIn("tox --current-env", text)

    def test_swebench_unittest_display_names_keep_django_module_selection(self) -> None:
        from p2a.precompute.precompute_bonus_maps import _normalize_test_func_name, _prepare_swebench_test_script

        class FakeEnv:
            def write_file(self, path: str, content: str) -> None:
                Path(path).write_text(content, encoding="utf-8")

            def _run(self, command: str, timeout: int | float | None = None) -> tuple[str, str]:
                if "python - <<'PY'" not in command:
                    return "", ""
                result = subprocess.run(command, shell=True, text=True, capture_output=True, check=False)
                return result.stdout, result.stderr

        with tempfile.TemporaryDirectory() as tmp:
            script = str(Path(tmp) / "run_tests.sh")
            diag = _prepare_swebench_test_script(
                FakeEnv(),
                {
                    "repo": "django/django",
                    "FAIL_TO_PASS": (
                        '["test_cull_delete_when_store_empty (cache.tests.DBCacheTests)", '
                        '"test_zero_values (template_tests.filter_tests.test_floatformat.FunctionTests.test_zero_values)"]'
                    ),
                    "run_tests": "\n".join(
                        [
                            "#!/bin/bash",
                            "python -m pip install -e .",
                            "git checkout abc123 tests/runtests.py tests/deprecation/test_middleware_mixin.py",
                            "./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 cache.tests",
                        ]
                    ),
                },
                script,
            )

            text = Path(script).read_text(encoding="utf-8")

        self.assertEqual(
            _normalize_test_func_name("test_cull_delete_when_store_empty (cache.tests.DBCacheTests)"),
            "test_cull_delete_when_store_empty",
        )
        self.assertIn("cache.tests.DBCacheTests.test_cull_delete_when_store_empty", diag["swebench_f2p_django_labels"])
        self.assertIn(
            "template_tests.filter_tests.test_floatformat.FunctionTests.test_zero_values",
            diag["swebench_f2p_django_labels"],
        )
        self.assertNotIn(
            "template_tests.filter_tests.test_floatformat.FunctionTests.test_zero_values.test_zero_values",
            diag["swebench_f2p_django_labels"],
        )
        self.assertIn("targeted_django=0", diag["swebench_test_script_patch_stdout"])
        self.assertIn("git checkout abc123 tests/runtests.py tests/deprecation/test_middleware_mixin.py", text)
        self.assertIn("./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 cache.tests", text)

    def test_swebench_sympy_runner_keeps_file_level_selection(self) -> None:
        from p2a.precompute.precompute_bonus_maps import _prepare_swebench_test_script

        class FakeEnv:
            def write_file(self, path: str, content: str) -> None:
                Path(path).write_text(content, encoding="utf-8")

            def _run(self, command: str, timeout: int | float | None = None) -> tuple[str, str]:
                if "python - <<'PY'" not in command:
                    return "", ""
                result = subprocess.run(command, shell=True, text=True, capture_output=True, check=False)
                return result.stdout, result.stderr

        with tempfile.TemporaryDirectory() as tmp:
            script = str(Path(tmp) / "run_tests.sh")
            diag = _prepare_swebench_test_script(
                FakeEnv(),
                {
                    "repo": "sympy/sympy",
                    "FAIL_TO_PASS": '["test_MatrixElement_printing", "test_MatrixSymbol_printing"]',
                    "run_tests": "\n".join(
                        [
                            "#!/bin/bash",
                            "python -m pip install -e .",
                            "bin/test -C --verbose sympy/printing/tests/test_latex.py",
                        ]
                    ),
                },
                script,
            )

            text = Path(script).read_text(encoding="utf-8")

        self.assertIn("targeted_sympy=0", diag["swebench_test_script_patch_stdout"])
        self.assertIn("sympy_file_level_selection=1", diag["swebench_test_script_patch_stdout"])
        self.assertIn("sympy_f2p_file_coverage_complete=0", diag["swebench_test_script_patch_stdout"])
        self.assertIn("bin/test -C --verbose sympy/printing/tests/test_latex.py", text)
        self.assertNotIn(" -k ", text)

    def test_swebench_sympy_file_level_selection_covers_all_f2p(self) -> None:
        from p2a.precompute.precompute_bonus_maps import _prepare_swebench_test_script

        class FakeEnv:
            def write_file(self, path: str, content: str) -> None:
                Path(path).write_text(content, encoding="utf-8")

            def _run(self, command: str, timeout: int | float | None = None) -> tuple[str, str]:
                if "python - <<'PY'" not in command:
                    return "", ""
                result = subprocess.run(command, shell=True, text=True, capture_output=True, check=False)
                return result.stdout, result.stderr

        with tempfile.TemporaryDirectory() as tmp:
            script = str(Path(tmp) / "run_tests.sh")
            diag = _prepare_swebench_test_script(
                FakeEnv(),
                {
                    "repo": "sympy/sympy",
                    "FAIL_TO_PASS": '["test_evalf_bugs"]',
                    "test_patch": "\n".join(
                        [
                            "diff --git a/sympy/core/tests/test_evalf.py b/sympy/core/tests/test_evalf.py",
                            "--- a/sympy/core/tests/test_evalf.py",
                            "+++ b/sympy/core/tests/test_evalf.py",
                            "@@ -220,10 +220,11 @@ def test_evalf_helpers():",
                            " def test_evalf_bugs():",
                            "     assert True",
                            "+    assert 1 == 1",
                        ]
                    ),
                    "run_tests": "\n".join(
                        [
                            "#!/bin/bash",
                            "python -m pip install -e .",
                            "PYTHONWARNINGS='ignore::UserWarning' bin/test -C --verbose sympy/core/tests/test_evalf.py",
                        ]
                    ),
                },
                script,
            )

        stdout = diag["swebench_test_script_patch_stdout"]
        self.assertIn("sympy_file_level_selection=1", stdout)
        self.assertIn("sympy_f2p_file_coverage_complete=1", stdout)
        self.assertIn('"test_evalf_bugs": ["sympy/core/tests/test_evalf.py"]', stdout)
        self.assertIn("sympy_f2p_uncovered_nodeids=[]", stdout)

    def test_swebench_sympy_file_level_selection_rejects_partial_f2p_coverage(self) -> None:
        from p2a.precompute.precompute_bonus_maps import _prepare_swebench_test_script

        class FakeEnv:
            def write_file(self, path: str, content: str) -> None:
                Path(path).write_text(content, encoding="utf-8")

            def _run(self, command: str, timeout: int | float | None = None) -> tuple[str, str]:
                if "python - <<'PY'" not in command:
                    return "", ""
                result = subprocess.run(command, shell=True, text=True, capture_output=True, check=False)
                return result.stdout, result.stderr

        with tempfile.TemporaryDirectory() as tmp:
            script = str(Path(tmp) / "run_tests.sh")
            diag = _prepare_swebench_test_script(
                FakeEnv(),
                {
                    "repo": "sympy/sympy",
                    "FAIL_TO_PASS": '["test_a", "test_b"]',
                    "test_patch": "\n".join(
                        [
                            "diff --git a/sympy/core/tests/test_a.py b/sympy/core/tests/test_a.py",
                            "--- a/sympy/core/tests/test_a.py",
                            "+++ b/sympy/core/tests/test_a.py",
                            "@@ -1,3 +1,4 @@ def test_a():",
                            "+    assert True",
                            "diff --git a/sympy/core/tests/test_b.py b/sympy/core/tests/test_b.py",
                            "--- a/sympy/core/tests/test_b.py",
                            "+++ b/sympy/core/tests/test_b.py",
                            "@@ -1,3 +1,4 @@ def test_b():",
                            "+    assert True",
                        ]
                    ),
                    "run_tests": "\n".join(
                        [
                            "#!/bin/bash",
                            "python -m pip install -e .",
                            "bin/test -C --verbose sympy/core/tests/test_a.py",
                        ]
                    ),
                },
                script,
            )

        stdout = diag["swebench_test_script_patch_stdout"]
        self.assertIn("sympy_file_level_selection=1", stdout)
        self.assertIn("sympy_f2p_file_coverage_complete=0", stdout)
        self.assertIn('"test_a": ["sympy/core/tests/test_a.py"]', stdout)
        self.assertIn('"test_b": []', stdout)
        self.assertIn('sympy_f2p_uncovered_nodeids=["test_b"]', stdout)

    def test_swebench_sympy_file_level_selection_covers_helper_patch_file(self) -> None:
        from p2a.precompute.precompute_bonus_maps import _prepare_swebench_test_script

        class FakeEnv:
            def write_file(self, path: str, content: str) -> None:
                Path(path).write_text(content, encoding="utf-8")

            def _run(self, command: str, timeout: int | float | None = None) -> tuple[str, str]:
                if "python - <<'PY'" not in command:
                    return "", ""
                result = subprocess.run(command, shell=True, text=True, capture_output=True, check=False)
                return result.stdout, result.stderr

        with tempfile.TemporaryDirectory() as tmp:
            script = str(Path(tmp) / "run_tests.sh")
            diag = _prepare_swebench_test_script(
                FakeEnv(),
                {
                    "repo": "sympy/sympy",
                    "FAIL_TO_PASS": '["test_sparse_matrix"]',
                    "test_patch": "\n".join(
                        [
                            "diff --git a/sympy/matrices/tests/test_sparse.py b/sympy/matrices/tests/test_sparse.py",
                            "--- a/sympy/matrices/tests/test_sparse.py",
                            "+++ b/sympy/matrices/tests/test_sparse.py",
                            "@@ -26,6 +26,12 @@ def sparse_zeros(n):",
                            "+    sparse_matrices = [SparseMatrix.zeros(0, n) for n in range(4)]",
                            "+    assert SparseMatrix.hstack(*sparse_matrices) == Matrix(0, 6, [])",
                        ]
                    ),
                    "run_tests": "\n".join(
                        [
                            "#!/bin/bash",
                            "python -m pip install -e .",
                            "bin/test -C --verbose sympy/matrices/tests/test_sparse.py",
                        ]
                    ),
                },
                script,
            )

        stdout = diag["swebench_test_script_patch_stdout"]
        self.assertIn("sympy_file_level_selection=1", stdout)
        self.assertIn("sympy_f2p_file_coverage_complete=1", stdout)
        self.assertIn('sympy_helper_fallback_files=["sympy/matrices/tests/test_sparse.py"]', stdout)
        self.assertIn('"test_sparse_matrix": ["sympy/matrices/tests/test_sparse.py"]', stdout)
        self.assertIn('sympy_f2p_helper_file_fallback={"test_sparse_matrix": ["sympy/matrices/tests/test_sparse.py"]}', stdout)
        self.assertIn("sympy_f2p_uncovered_nodeids=[]", stdout)

    def test_swebench_sympy_file_mapping_ignores_non_definition_mentions(self) -> None:
        from p2a.precompute.precompute_bonus_maps import _prepare_swebench_test_script

        class FakeEnv:
            def write_file(self, path: str, content: str) -> None:
                Path(path).write_text(content, encoding="utf-8")

            def _run(self, command: str, timeout: int | float | None = None) -> tuple[str, str]:
                if "python - <<'PY'" not in command:
                    return "", ""
                result = subprocess.run(command, shell=True, text=True, capture_output=True, check=False)
                return result.stdout, result.stderr

        with tempfile.TemporaryDirectory() as tmp:
            script = str(Path(tmp) / "run_tests.sh")
            diag = _prepare_swebench_test_script(
                FakeEnv(),
                {
                    "repo": "sympy/sympy",
                    "FAIL_TO_PASS": '["test_real_f2p"]',
                    "test_patch": "\n".join(
                        [
                            "diff --git a/sympy/core/tests/test_wrong.py b/sympy/core/tests/test_wrong.py",
                            "--- a/sympy/core/tests/test_wrong.py",
                            "+++ b/sympy/core/tests/test_wrong.py",
                            "@@ -1,5 +1,8 @@ def test_other():",
                            " def test_other():",
                            "+    # mention test_real_f2p without defining it",
                            "+    name = 'test_real_f2p'",
                            "+    helper(test_real_f2p)",
                        ]
                    ),
                    "run_tests": "\n".join(
                        [
                            "#!/bin/bash",
                            "python -m pip install -e .",
                            "bin/test -C --verbose sympy/core/tests/test_wrong.py",
                        ]
                    ),
                },
                script,
            )

        stdout = diag["swebench_test_script_patch_stdout"]
        self.assertIn("sympy_file_level_selection=1", stdout)
        self.assertIn("sympy_f2p_file_coverage_complete=0", stdout)
        self.assertIn("sympy_helper_fallback_files=[]", stdout)
        self.assertIn('"test_real_f2p": []', stdout)
        self.assertIn('sympy_f2p_uncovered_nodeids=["test_real_f2p"]', stdout)

    def test_swebench_zero_tests_output_is_detected(self) -> None:
        from p2a.precompute.precompute_bonus_maps import (
            _swebench_output_has_zero_tests,
            _swebench_output_known_clean,
        )

        self.assertTrue(_swebench_output_has_zero_tests("tests finished: 0 passed, in 0.00 seconds"))
        self.assertTrue(_swebench_output_has_zero_tests("no tests ran in 0.01s"))
        self.assertFalse(_swebench_output_has_zero_tests("Ran 1 test in 0.007s\n\nOK"))
        self.assertTrue(_swebench_output_known_clean("tests finished: 1 passed, in 0.10 seconds"))
        self.assertTrue(_swebench_output_known_clean("1 passed, 2 warnings in 0.10s"))
        self.assertFalse(
            _swebench_output_known_clean(
                "Traceback (most recent call last):\nImportError: broken\n1 passed in 0.10s"
            )
        )
        self.assertFalse(_swebench_output_known_clean("ERROR collecting sympy/core/tests/test_evalf.py"))
        self.assertFalse(_swebench_output_known_clean("tests finished: 16 passed, 1 failed, in 0.10 seconds"))
        self.assertFalse(_swebench_output_known_clean("tests finished: 16 passed, 1 error, in 0.10 seconds"))

    def test_trace_instrumentation_caches_tracer_after_future_imports(self) -> None:
        from p2a.trace import instrument_source

        source = (
            '"""demo module"""\n'
            "from __future__ import annotations\n"
            "\n"
            "def modify_sys_path():\n"
            "    return 1\n"
        )
        instrumented = instrument_source(
            source,
            [
                {
                    "name": "modify_sys_path",
                    "qualified_name": "modify_sys_path",
                    "file_path": "pylint/__init__.py",
                    "start_line": 4,
                    "end_line": 5,
                }
            ],
        )

        compile(instrumented, "<instrumented>", "exec")
        self.assertLess(
            instrumented.index("from __future__ import annotations"),
            instrumented.index("import _swe_fault_tracer as _p2a_ft"),
        )
        self.assertLess(
            instrumented.index("import _swe_fault_tracer as _p2a_ft"),
            instrumented.index("def modify_sys_path"),
        )
        self.assertIn('globals().get("_p2a_ft") or __import__("_swe_fault_tracer")', instrumented)

    def test_swebench_f2p_failure_detection_uses_test_name_boundaries(self) -> None:
        from p2a.precompute.precompute_bonus_maps import (
            _swebench_f2p_collection_observation,
            _swebench_output_has_f2p_failure,
        )

        self.assertFalse(
            _swebench_output_has_f2p_failure(
                "tests/test_demo.py::test_foo_bar FAILED",
                ["tests/test_demo.py::test_foo"],
            )
        )
        self.assertTrue(
            _swebench_output_has_f2p_failure(
                "tests/test_demo.py::test_foo FAILED",
                ["tests/test_demo.py::test_foo"],
            )
        )
        self.assertTrue(
            _swebench_output_has_f2p_failure(
                "FAIL: test_edit_only (model_formsets.tests.ModelFormsetTest)",
                ["test_edit_only (model_formsets.tests.ModelFormsetTest)"],
            )
        )
        self.assertEqual(
            _swebench_f2p_collection_observation(
                "\n".join(
                    [
                        "+def test_foo():",
                        "tests/test_demo.py::test_foo_bar PASSED",
                        "tests/test_demo.py::test_foo PASSED",
                    ]
                ),
                ["tests/test_demo.py::test_foo", "tests/test_demo.py::test_missing"],
            ),
            {
                "observed": ["tests/test_demo.py::test_foo"],
                "missing": ["tests/test_demo.py::test_missing"],
            },
        )

    def test_swebench_description_f2p_maps_to_unittest_method(self) -> None:
        from p2a.precompute.precompute_bonus_maps import _get_f2p_test_funcs

        raw_output = "\n".join(
            [
                "ERROR: test_loading_namespace_package (migrations.test_loader.LoaderTests)",
                "Migration directories without an __init__.py file are loaded.",
                "Traceback (most recent call last):",
                "FAILED (errors=1)",
            ]
        )
        funcs = _get_f2p_test_funcs(
            {"FAIL_TO_PASS": json.dumps(["Migration directories without an __init__.py file are loaded."])},
            raw_output,
            swebench_verified=True,
        )

        self.assertEqual(funcs, {"test_loading_namespace_package"})

    def test_r2e_f2p_intersects_buggy_failures_with_fixed_passes(self) -> None:
        from p2a.precompute.precompute_bonus_maps import (
            _filter_traces_to_f2p,
            _get_f2p_test_funcs,
            _test_func_names_from_traces,
        )

        raw_output = "\n".join(
            [
                "FAILED r2e_tests/test_1.py::TestImportDialog::test_dialog - AssertionError",
                "FAILED r2e_tests/test_1.py::TestUtils::test_open_compressed - AssertionError",
            ]
        )
        task = {
            "extra_info": {
                "tools_kwargs": {
                    "reward": {
                        "metadata": {
                            "expected_output_json": json.dumps(
                                {
                                    "TestUtils.test_open_compressed": "PASSED",
                                    "TestImportDialog.test_dialog": "FAILED",
                                }
                            )
                        }
                    }
                }
            }
        }

        funcs = _get_f2p_test_funcs(task, raw_output, swebench_verified=False)

        self.assertEqual(funcs, {"test_open_compressed"})
        raw_gt_traces = [
            [
                {"file_path": "r2e_tests/test_1.py", "func_name": "test_dialog"},
                {"file_path": "Orange/widgets/data/owcsvimport.py", "func_name": "_open"},
            ],
            [
                {"file_path": "r2e_tests/test_1.py", "func_name": "test_open_compressed"},
                {"file_path": "Orange/widgets/data/owcsvimport.py", "func_name": "_open"},
            ],
        ]
        f2p_traces = _filter_traces_to_f2p(raw_gt_traces, funcs)
        self.assertEqual(_test_func_names_from_traces(f2p_traces), ["test_open_compressed"])

    def test_swebench_exit_override_classifies_signature_mismatch(self) -> None:
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
            swebench_verified = True
            repo_path = "/testbed"
            alt_path = "/root"

            def start(self) -> None:
                pass

            def close(self) -> None:
                pass

            def checkout_buggy_commit(self, task, *, instance_id):
                return {"buggy_checkout_ref": "abc123^", "buggy_checkout_exit": 0}

            def _run(self, command: str, timeout: int | float | None = None):
                return "", ""

            def _execute_raw(self, command: str, timeout: int | float | None = None):
                if "wc -l" in command:
                    return "0\n", "", 0
                return "", "", 0

            def write_file(self, path: str, content: str) -> None:
                pass

        task = {
            "instance_id": "django__django-1",
            "repo": "django/django",
            "patch": "diff --git a/pkg/demo.py b/pkg/demo.py\n",
            "FAIL_TO_PASS": json.dumps(["tests/test_demo.py::test_foo"]),
            "extra_info": {
                "tools_kwargs": {
                    "reward": {"name": "swe_bench", "metadata": {}},
                }
            },
        }
        raw_output = "\n".join(
            [
                "tests/test_demo.py::test_foo FAILED",
                "TypeError: demo() got an unexpected keyword argument 'value'",
            ]
        )

        with (
            patch.object(bonus_maps, "find_modified_callables_from_task", return_value=modified),
            patch.object(bonus_maps, "find_newly_created_callables", return_value=[]),
            patch.object(
                bonus_maps,
                "_prepare_swebench_test_script",
                return_value={"swebench_test_script_patch_stdout": "targeted_pytest=1\n"},
            ),
            patch.object(
                bonus_maps,
                "_run_tests_with_file_capture",
                return_value=(
                    raw_output,
                    "",
                    0,
                    {
                        "stdout_read_exit": 0,
                        "stderr_read_exit": 0,
                        "exit_read_exit": 0,
                        "wrapper_exit": 0,
                        "exit_parse_failed": False,
                        "all_three_read_failed": False,
                        "trusted_test_exit": True,
                    },
                ),
            ),
            patch.object(
                bonus_maps,
                "_detect_import_targets",
                return_value=[{"module": "pkg.demo", "matches_repo_path": True}],
            ),
            patch(
                "p2a.precompute.uni_agent_sandbox.create_uni_agent_sandbox",
                return_value=FakeEnv(),
            ),
            patch("p2a.trace.instrument_sandbox", return_value=modified),
            patch("p2a.trace.parse_fault_traces_from_file", return_value=[]),
        ):
            result = bonus_maps.compute_dynamic_bonus_map(task)

        self.assertEqual(result["case_type"], "signature_mismatch")
        self.assertEqual(result["reason_code"], "signature_mismatch_before_entry")
        self.assertEqual(result["test_exit"], 1)
        self.assertTrue(result["test_output_capture_detail"]["test_exit_overridden_from_output"])

    def test_instrumentation_empty_is_instrumentation_failed(self) -> None:
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
            repo_path = "/testbed"
            alt_path = "/root"

            def start(self) -> None:
                pass

            def close(self) -> None:
                pass

            def checkout_buggy_commit(self, task, *, instance_id):
                return {"buggy_checkout_ref": "abc123^", "buggy_checkout_exit": 0}

            def _run(self, command: str, timeout: int | float | None = None):
                return "", ""

        task = {
            "instance_id": "demo__abc123",
            "repo": "demo",
            "patch": "diff --git a/pkg/demo.py b/pkg/demo.py\n",
        }

        with (
            patch.object(bonus_maps, "find_modified_callables_from_task", return_value=modified),
            patch.object(bonus_maps, "find_newly_created_callables", return_value=[]),
            patch(
                "p2a.precompute.uni_agent_sandbox.create_uni_agent_sandbox",
                return_value=FakeEnv(),
            ),
            patch("p2a.trace.instrument_sandbox", return_value=[]),
        ):
            result = bonus_maps.compute_dynamic_bonus_map(task)

        self.assertEqual(result["case_type"], "instrumentation_failed")
        self.assertEqual(result["reason_code"], "instrumentation_empty")
        self.assertTrue(result["error"])

    def test_all_pass_reason_is_single_early_decision(self) -> None:
        from p2a.precompute.precompute_bonus_maps import _all_pass_reason_code

        self.assertEqual(_all_pass_reason_code(0, {"all_three_read_failed": False}), "buggy_version_passes")
        self.assertIsNone(_all_pass_reason_code(1, {"all_three_read_failed": False}))
        self.assertIsNone(_all_pass_reason_code(0, {"all_three_read_failed": True}))
        self.assertIsNone(
            _all_pass_reason_code(
                0,
                {"all_three_read_failed": False},
                swebench_f2p_collection_missing=True,
            )
        )
        self.assertEqual(
            _all_pass_reason_code(
                0,
                {"all_three_read_failed": False},
                swebench_f2p_collection_missing=True,
                allow_missing_f2p_collection=True,
            ),
            "buggy_version_passes",
        )

    def test_all_pass_short_circuits_before_trace_parse(self) -> None:
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
            swebench_verified = True
            repo_path = "/testbed"
            alt_path = "/root"

            def start(self) -> None:
                pass

            def close(self) -> None:
                pass

            def checkout_buggy_commit(self, task, *, instance_id):
                return {"buggy_checkout_ref": "abc123^", "buggy_checkout_exit": 0}

            def _run(self, command: str, timeout: int | float | None = None):
                return "", ""

            def _execute_raw(self, command: str, timeout: int | float | None = None):
                raise AssertionError(f"trace parsing should be skipped for all_pass: {command}")

            def write_file(self, path: str, content: str) -> None:
                pass

        task = {
            "instance_id": "sympy__sympy-13372",
            "repo": "sympy/sympy",
            "patch": "diff --git a/pkg/demo.py b/pkg/demo.py\n",
            "FAIL_TO_PASS": json.dumps(["test_foo"]),
            "extra_info": {
                "tools_kwargs": {
                    "reward": {"name": "swe_bench", "metadata": {}},
                }
            },
        }

        with (
            patch.object(bonus_maps, "find_modified_callables_from_task", return_value=modified),
            patch.object(bonus_maps, "find_newly_created_callables", return_value=[]),
            patch.object(
                bonus_maps,
                "_prepare_swebench_test_script",
                return_value={
                    "swebench_test_script_patch_stdout": (
                        "targeted_sympy=0\n"
                        "sympy_file_level_selection=1\n"
                        "sympy_f2p_file_coverage_complete=1\n"
                    )
                },
            ),
            patch.object(
                bonus_maps,
                "_run_tests_with_file_capture",
                return_value=(
                    "1 passed, 2 warnings in 0.10s\n",
                    "",
                    0,
                    {
                        "stdout_read_exit": 0,
                        "stderr_read_exit": 0,
                        "exit_read_exit": 0,
                        "wrapper_exit": 0,
                        "exit_parse_failed": False,
                        "all_three_read_failed": False,
                        "trusted_test_exit": True,
                    },
                ),
            ),
            patch(
                "p2a.precompute.uni_agent_sandbox.create_uni_agent_sandbox",
                return_value=FakeEnv(),
            ),
            patch("p2a.trace.instrument_sandbox", return_value=modified),
            patch(
                "p2a.trace.parse_fault_traces_from_file",
                side_effect=AssertionError("trace parser should not run for all_pass"),
            ),
        ):
            result = bonus_maps.compute_dynamic_bonus_map(task)

        self.assertEqual(result["case_type"], "all_pass")
        self.assertEqual(result["reason_code"], "buggy_version_passes")
        self.assertTrue(result["trace_parse_skipped"])
        self.assertEqual(result["trace_parse_skip_reason"], "all_pass")
        self.assertIsNone(result["trace_file_line_count"])
        self.assertEqual(result["parsed_trace_count"], 0)
        self.assertTrue(result["swebench_f2p_collection_missing_allowed"])
        self.assertTrue(result["swebench_sympy_f2p_file_coverage_complete"])
        self.assertEqual(result["swebench_f2p_observed_nodeids"], [])
        self.assertEqual(result["swebench_f2p_missing_nodeids"], ["test_foo"])

    def test_sympy_masked_collection_error_is_not_all_pass(self) -> None:
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
            swebench_verified = True
            repo_path = "/testbed"
            alt_path = "/root"

            def start(self) -> None:
                pass

            def close(self) -> None:
                pass

            def checkout_buggy_commit(self, task, *, instance_id):
                return {"buggy_checkout_ref": "abc123^", "buggy_checkout_exit": 0}

            def _run(self, command: str, timeout: int | float | None = None):
                return "", ""

            def _execute_raw(self, command: str, timeout: int | float | None = None):
                if "wc -l" in command:
                    return "0\n", "", 0
                return "", "", 0

            def write_file(self, path: str, content: str) -> None:
                pass

        task = {
            "instance_id": "sympy__masked-error",
            "repo": "sympy/sympy",
            "patch": "diff --git a/pkg/demo.py b/pkg/demo.py\n",
            "FAIL_TO_PASS": json.dumps(["test_foo"]),
            "extra_info": {
                "tools_kwargs": {
                    "reward": {"name": "swe_bench", "metadata": {}},
                }
            },
        }

        with (
            patch.object(bonus_maps, "find_modified_callables_from_task", return_value=modified),
            patch.object(bonus_maps, "find_newly_created_callables", return_value=[]),
            patch.object(
                bonus_maps,
                "_prepare_swebench_test_script",
                return_value={
                    "swebench_test_script_patch_stdout": (
                        "targeted_sympy=0\n"
                        "sympy_file_level_selection=1\n"
                        "sympy_f2p_file_coverage_complete=1\n"
                    )
                },
            ),
            patch.object(
                bonus_maps,
                "_run_tests_with_file_capture",
                return_value=(
                    "ERROR collecting sympy/core/tests/test_evalf.py\n"
                    "Traceback (most recent call last):\n"
                    "ImportError: broken import\n",
                    "",
                    0,
                    {
                        "stdout_read_exit": 0,
                        "stderr_read_exit": 0,
                        "exit_read_exit": 0,
                        "wrapper_exit": 0,
                        "exit_parse_failed": False,
                        "all_three_read_failed": False,
                        "trusted_test_exit": True,
                    },
                ),
            ),
            patch(
                "p2a.precompute.uni_agent_sandbox.create_uni_agent_sandbox",
                return_value=FakeEnv(),
            ),
            patch("p2a.trace.instrument_sandbox", return_value=modified),
            patch("p2a.trace.parse_fault_traces_from_file", return_value=[]),
        ):
            result = bonus_maps.compute_dynamic_bonus_map(task)

        self.assertEqual(result["case_type"], "no_trace")
        self.assertEqual(result["reason_code"], "f2p_collection_missing")
        self.assertFalse(result["swebench_output_known_clean"])
        self.assertFalse(result["swebench_f2p_collection_missing_allowed"])
        self.assertTrue(result["swebench_sympy_f2p_file_coverage_complete"])

    def test_swebench_missing_f2p_collection_is_not_all_pass(self) -> None:
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
            swebench_verified = True
            repo_path = "/testbed"
            alt_path = "/root"

            def start(self) -> None:
                pass

            def close(self) -> None:
                pass

            def checkout_buggy_commit(self, task, *, instance_id):
                return {"buggy_checkout_ref": "abc123^", "buggy_checkout_exit": 0}

            def _run(self, command: str, timeout: int | float | None = None):
                return "", ""

            def _execute_raw(self, command: str, timeout: int | float | None = None):
                if "wc -l" in command:
                    return "0\n", "", 0
                return "", "", 0

            def write_file(self, path: str, content: str) -> None:
                pass

        task = {
            "instance_id": "pytest__pytest-1",
            "repo": "pytest-dev/pytest",
            "patch": "diff --git a/pkg/demo.py b/pkg/demo.py\n",
            "FAIL_TO_PASS": json.dumps(["tests/test_demo.py::test_foo"]),
            "extra_info": {
                "tools_kwargs": {
                    "reward": {"name": "swe_bench", "metadata": {}},
                }
            },
        }

        with (
            patch.object(bonus_maps, "find_modified_callables_from_task", return_value=modified),
            patch.object(bonus_maps, "find_newly_created_callables", return_value=[]),
            patch.object(
                bonus_maps,
                "_prepare_swebench_test_script",
                return_value={"swebench_test_script_patch_stdout": "targeted_pytest=1\n"},
            ),
            patch.object(
                bonus_maps,
                "_run_tests_with_file_capture",
                return_value=(
                    "tests/test_demo.py::test_unrelated PASSED\n",
                    "",
                    0,
                    {
                        "stdout_read_exit": 0,
                        "stderr_read_exit": 0,
                        "exit_read_exit": 0,
                        "wrapper_exit": 0,
                        "exit_parse_failed": False,
                        "all_three_read_failed": False,
                        "trusted_test_exit": True,
                    },
                ),
            ),
            patch(
                "p2a.precompute.uni_agent_sandbox.create_uni_agent_sandbox",
                return_value=FakeEnv(),
            ),
            patch("p2a.trace.instrument_sandbox", return_value=modified),
            patch("p2a.trace.parse_fault_traces_from_file", return_value=[]),
        ):
            result = bonus_maps.compute_dynamic_bonus_map(task)

        self.assertEqual(result["case_type"], "no_trace")
        self.assertEqual(result["reason_code"], "f2p_collection_missing")
        self.assertEqual(result["swebench_f2p_missing_nodeids"], ["tests/test_demo.py::test_foo"])

    def test_trace_parse_cap_is_disabled_by_default(self) -> None:
        from p2a.precompute import precompute_bonus_maps as bonus_maps

        modified = [
            {
                "name": "demo",
                "qualified_name": "demo",
                "file_path": "pkg/demo.py",
                "start_line": 10,
                "end_line": 12,
                "instr_start_line": 10,
                "instr_end_line": 12,
            }
        ]
        trace = [
            {
                "file_path": "tests/test_demo.py",
                "line_no": 1,
                "func_name": "test_foo",
                "qualified_name": "test_foo",
                "is_patched": False,
            },
            {
                "file_path": "pkg/demo.py",
                "line_no": 10,
                "func_name": "demo",
                "qualified_name": "demo",
                "is_patched": True,
            },
        ]

        class FakeEnv:
            swebench_verified = True
            repo_path = "/testbed"
            alt_path = "/root"

            def __init__(self) -> None:
                self.commands: list[str] = []

            def start(self) -> None:
                pass

            def close(self) -> None:
                pass

            def checkout_buggy_commit(self, task, *, instance_id):
                return {"buggy_checkout_ref": "abc123^", "buggy_checkout_exit": 0}

            def _run(self, command: str, timeout: int | float | None = None):
                self.commands.append(command)
                return "", ""

            def _execute_raw(self, command: str, timeout: int | float | None = None):
                self.commands.append(command)
                if "wc -l" in command:
                    return "1500\n", "", 0
                return "", "", 0

            def write_file(self, path: str, content: str) -> None:
                pass

        task = {
            "instance_id": "pytest__pytest-1",
            "repo": "pytest-dev/pytest",
            "patch": "diff --git a/pkg/demo.py b/pkg/demo.py\n",
            "FAIL_TO_PASS": json.dumps(["tests/test_demo.py::test_foo"]),
            "extra_info": {
                "tools_kwargs": {
                    "reward": {"name": "swe_bench", "metadata": {}},
                }
            },
        }
        env = FakeEnv()
        parse_calls: list[str] = []

        def fake_parse(*args, **kwargs):
            parse_calls.append(kwargs["trace_file_path"])
            return [trace]

        with (
            patched_env(remove=("P2A_TRACE_PARSE_MAX_LINES", "P2A_TRACE_MAX_EVENTS")),
            patch.object(bonus_maps, "find_modified_callables_from_task", return_value=modified),
            patch.object(bonus_maps, "find_newly_created_callables", return_value=[]),
            patch.object(
                bonus_maps,
                "_prepare_swebench_test_script",
                return_value={"swebench_test_script_patch_stdout": ""},
            ),
            patch.object(
                bonus_maps,
                "_run_tests_with_file_capture",
                return_value=(
                    "tests/test_demo.py::test_foo FAILED\n",
                    "",
                    1,
                    {
                        "stdout_read_exit": 0,
                        "stderr_read_exit": 0,
                        "exit_read_exit": 0,
                        "wrapper_exit": 1,
                        "exit_parse_failed": False,
                        "all_three_read_failed": False,
                        "trusted_test_exit": True,
                    },
                ),
            ),
            patch(
                "p2a.precompute.uni_agent_sandbox.create_uni_agent_sandbox",
                return_value=env,
            ),
            patch("p2a.trace.instrument_sandbox", return_value=modified),
            patch("p2a.trace.parse_fault_traces_from_file", side_effect=fake_parse),
            patch(
                "p2a.trace.build_call_graph_from_traces",
                return_value={
                    "call_graph_nodes": {
                        "tests/test_demo.py::test_foo": {
                            "file_path": "tests/test_demo.py",
                            "normalized_distance": 0,
                        }
                    },
                    "call_graph_edges": [],
                    "hop_max": 0,
                },
            ),
        ):
            result = bonus_maps.compute_dynamic_bonus_map(task)

        self.assertEqual(result["case_type"], "direct")
        self.assertIsNone(result["trace_parse_line_cap"])
        self.assertFalse(result["trace_parse_line_cap_reached"])
        self.assertEqual(result["trace_event_cap"], 10000)
        self.assertEqual(parse_calls, [bonus_maps.TRACE_PARSE_PATH, bonus_maps.TRACE_PARSE_PATH])
        self.assertTrue(any("sed -n '1,1000p'" in command for command in env.commands))
        self.assertTrue(any("sed -n '1001,1500p'" in command for command in env.commands))
        self.assertFalse(any("head -n 500" in command for command in env.commands))

    def test_explicit_trace_parse_cap_sets_metadata(self) -> None:
        from p2a.precompute import precompute_bonus_maps as bonus_maps

        modified = [
            {
                "name": "demo",
                "qualified_name": "demo",
                "file_path": "pkg/demo.py",
                "start_line": 10,
                "end_line": 12,
                "instr_start_line": 10,
                "instr_end_line": 12,
            }
        ]
        trace = [
            {
                "file_path": "tests/test_demo.py",
                "line_no": 1,
                "func_name": "test_foo",
                "qualified_name": "test_foo",
                "is_patched": False,
            },
            {
                "file_path": "pkg/demo.py",
                "line_no": 10,
                "func_name": "demo",
                "qualified_name": "demo",
                "is_patched": True,
            },
        ]

        class FakeEnv:
            swebench_verified = True
            repo_path = "/testbed"
            alt_path = "/root"

            def __init__(self) -> None:
                self.commands: list[str] = []

            def start(self) -> None:
                pass

            def close(self) -> None:
                pass

            def checkout_buggy_commit(self, task, *, instance_id):
                return {"buggy_checkout_ref": "abc123^", "buggy_checkout_exit": 0}

            def _run(self, command: str, timeout: int | float | None = None):
                self.commands.append(command)
                return "", ""

            def _execute_raw(self, command: str, timeout: int | float | None = None):
                self.commands.append(command)
                if "wc -l" in command:
                    return "1500\n", "", 0
                return "", "", 0

            def write_file(self, path: str, content: str) -> None:
                pass

        task = {
            "instance_id": "pytest__pytest-1",
            "repo": "pytest-dev/pytest",
            "patch": "diff --git a/pkg/demo.py b/pkg/demo.py\n",
            "FAIL_TO_PASS": json.dumps(["tests/test_demo.py::test_foo"]),
            "extra_info": {
                "tools_kwargs": {
                    "reward": {"name": "swe_bench", "metadata": {}},
                }
            },
        }
        env = FakeEnv()

        with (
            patched_env({"P2A_TRACE_PARSE_MAX_LINES": "500"}, remove=("P2A_TRACE_MAX_EVENTS",)),
            patch.object(bonus_maps, "find_modified_callables_from_task", return_value=modified),
            patch.object(bonus_maps, "find_newly_created_callables", return_value=[]),
            patch.object(
                bonus_maps,
                "_prepare_swebench_test_script",
                return_value={"swebench_test_script_patch_stdout": ""},
            ),
            patch.object(
                bonus_maps,
                "_run_tests_with_file_capture",
                return_value=(
                    "tests/test_demo.py::test_foo FAILED\n",
                    "",
                    1,
                    {
                        "stdout_read_exit": 0,
                        "stderr_read_exit": 0,
                        "exit_read_exit": 0,
                        "wrapper_exit": 1,
                        "exit_parse_failed": False,
                        "all_three_read_failed": False,
                        "trusted_test_exit": True,
                    },
                ),
            ),
            patch(
                "p2a.precompute.uni_agent_sandbox.create_uni_agent_sandbox",
                return_value=env,
            ),
            patch("p2a.trace.instrument_sandbox", return_value=modified),
            patch("p2a.trace.parse_fault_traces_from_file", return_value=[trace]),
            patch(
                "p2a.trace.build_call_graph_from_traces",
                return_value={
                    "call_graph_nodes": {
                        "tests/test_demo.py::test_foo": {
                            "file_path": "tests/test_demo.py",
                            "normalized_distance": 0,
                        }
                    },
                    "call_graph_edges": [],
                    "hop_max": 0,
                },
            ),
        ):
            result = bonus_maps.compute_dynamic_bonus_map(task)

        self.assertEqual(result["case_type"], "direct")
        self.assertEqual(result["trace_parse_line_cap"], 500)
        self.assertTrue(result["trace_parse_line_cap_reached"])
        self.assertTrue(any("sed -n '1,500p'" in command for command in env.commands))
        self.assertFalse(any("sed -n '501," in command for command in env.commands))

    def test_event_cap_reached_no_f2p_is_inconclusive(self) -> None:
        from p2a.precompute import precompute_bonus_maps as bonus_maps

        modified = [
            {
                "name": "demo",
                "qualified_name": "demo",
                "file_path": "pkg/demo.py",
                "start_line": 10,
                "end_line": 12,
                "instr_start_line": 10,
                "instr_end_line": 12,
            }
        ]
        trace = [
            {
                "file_path": "tests/test_demo.py",
                "line_no": 1,
                "func_name": "test_unrelated",
                "qualified_name": "test_unrelated",
                "is_patched": False,
            },
            {
                "file_path": "pkg/demo.py",
                "line_no": 10,
                "func_name": "demo",
                "qualified_name": "demo",
                "is_patched": True,
            },
        ]

        class FakeEnv:
            swebench_verified = True
            repo_path = "/testbed"
            alt_path = "/root"

            def start(self) -> None:
                pass

            def close(self) -> None:
                pass

            def checkout_buggy_commit(self, task, *, instance_id):
                return {"buggy_checkout_ref": "abc123^", "buggy_checkout_exit": 0}

            def _run(self, command: str, timeout: int | float | None = None):
                return "", ""

            def _execute_raw(self, command: str, timeout: int | float | None = None):
                if "wc -l" in command:
                    return "2\n", "", 0
                return "", "", 0

            def write_file(self, path: str, content: str) -> None:
                pass

        task = {
            "instance_id": "pytest__pytest-1",
            "repo": "pytest-dev/pytest",
            "patch": "diff --git a/pkg/demo.py b/pkg/demo.py\n",
            "FAIL_TO_PASS": json.dumps(["tests/test_demo.py::test_foo"]),
            "extra_info": {
                "tools_kwargs": {
                    "reward": {"name": "swe_bench", "metadata": {}},
                }
            },
        }

        with (
            patched_env({"P2A_TRACE_MAX_EVENTS": "2"}, remove=("P2A_TRACE_PARSE_MAX_LINES",)),
            patch.object(bonus_maps, "find_modified_callables_from_task", return_value=modified),
            patch.object(bonus_maps, "find_newly_created_callables", return_value=[]),
            patch.object(
                bonus_maps,
                "_prepare_swebench_test_script",
                return_value={"swebench_test_script_patch_stdout": ""},
            ),
            patch.object(
                bonus_maps,
                "_run_tests_with_file_capture",
                return_value=(
                    "tests/test_demo.py::test_foo FAILED\n",
                    "",
                    1,
                    {
                        "stdout_read_exit": 0,
                        "stderr_read_exit": 0,
                        "exit_read_exit": 0,
                        "wrapper_exit": 1,
                        "exit_parse_failed": False,
                        "all_three_read_failed": False,
                        "trusted_test_exit": True,
                    },
                ),
            ),
            patch(
                "p2a.precompute.uni_agent_sandbox.create_uni_agent_sandbox",
                return_value=FakeEnv(),
            ),
            patch("p2a.trace.instrument_sandbox", return_value=modified),
            patch("p2a.trace.parse_fault_traces_from_file", return_value=[trace]),
        ):
            result = bonus_maps.compute_dynamic_bonus_map(task)

        self.assertEqual(result["case_type"], "trace_cap_inconclusive")
        self.assertEqual(result["reason_code"], "trace_event_cap_no_f2p_inconclusive")
        self.assertTrue(result["trace_event_cap_reached"])
        self.assertFalse(result["trace_parse_line_cap_reached"])
        self.assertTrue(result["trace_cap_inconclusive"])
        self.assertEqual(result["would_be_case_type"], "no_f2p")
        self.assertEqual(result["would_be_reason_code"], "f2p_filter_dropped")
        self.assertEqual(result["f2p_trace_count"], 0)

    def test_generated_tracer_default_event_cap_is_10000(self) -> None:
        from p2a.trace import generate_tracer_module

        tracer_source = generate_tracer_module("/testbed")

        self.assertIn('_MAX_EVENTS = _env_int("P2A_TRACE_MAX_EVENTS", 10000)', tracer_source)


if __name__ == "__main__":
    unittest.main()
