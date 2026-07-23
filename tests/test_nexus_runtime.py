from __future__ import annotations

import asyncio
from types import SimpleNamespace

from swerex.runtime.abstract import BashAction, CloseBashSessionRequest

from env.deployment import NexusDeploymentConfig, make_env_config
from env.runtime import NexusRuntime


class FakeNexusRuntime:
    def __init__(self) -> None:
        self.start_calls: list[str] = []
        self.destroyed: list[str] = []
        self.terminated: list[str] = []
        self.fail_busy = False
        self.timeout_once = False

    async def create_terminal_session(self, *, session_id: str):
        return SimpleNamespace(session_id=session_id)

    async def start_command_in_terminal_session(self, *, session_id: str, command: str):
        self.start_calls.append(command)
        if self.timeout_once and command == "echo recover":
            self.timeout_once = False
            raise asyncio.TimeoutError
        if self.fail_busy and command != "BASH_ARGV0=bash; bind 'set enable-bracketed-paste off' 2>/dev/null; true":
            raise RuntimeError("command is already running")
        return SimpleNamespace(command_id=f"command-{len(self.start_calls)}")

    async def query_terminal_command_status(self, *, session_id: str, command_id: str):
        return SimpleNamespace(
            end_time=1,
            stdout="ok\n",
            stderr="",
            output="ok\n",
            exit_code=0,
        )

    async def send_keys_to_terminal_session(self, *, session_id: str, keys: list[str]):
        return None

    async def terminate_terminal_session_processes(self, *, session_id: str):
        self.terminated.append(session_id)

    async def destroy_terminal_session(self, *, session_id: str):
        self.destroyed.append(session_id)


def test_nexus_config_and_environment_are_explicit():
    config = NexusDeploymentConfig.from_mapping(
        {
            "type": "nexus",
            "purpose": "test",
            "image": "repo/image:tag",
        }
    )
    assert config.purpose == "test"
    assert config.resource_spec == {"cpu": 16, "memory": 16}

    env_config = make_env_config(
        {"type": "nexus", "purpose": "test", "image": "repo/image:tag"},
        env_variables={"PAGER": "cat"},
    )
    assert env_config.deployment.environment == {"PAGER": "cat"}
    assert env_config.env_variables == {"PAGER": "cat"}


def test_nexus_runtime_runs_commands_and_cleans_up_sessions():
    fake = FakeNexusRuntime()
    runtime = NexusRuntime(fake, run_id="run-1")

    async def run():
        observation = await runtime.run_in_session(BashAction(command="echo ok", timeout=1))
        await runtime.close_session(CloseBashSessionRequest(session="default"))
        return observation

    observation = asyncio.run(run())

    assert observation.output == "ok\n"
    assert observation.exit_code == 0
    assert fake.destroyed == ["default-run-1"]


def test_nexus_runtime_bounds_busy_session_retries(monkeypatch):
    fake = FakeNexusRuntime()
    fake.fail_busy = True
    runtime = NexusRuntime(fake, run_id="run-2")
    monkeypatch.setattr("env.runtime.asyncio.sleep", _no_sleep)

    async def run():
        return await runtime.run_in_session(BashAction(command="echo blocked", timeout=1))

    observation = asyncio.run(run())

    assert observation.failure_reason == "max_retries"
    assert len(fake.terminated) == runtime._MAX_RETRY_DEPTH
    assert len(fake.start_calls) == runtime._MAX_RETRY_DEPTH + 2


def test_nexus_runtime_recovers_from_start_timeout(monkeypatch):
    fake = FakeNexusRuntime()
    fake.timeout_once = True
    runtime = NexusRuntime(fake, run_id="run-3")
    monkeypatch.setattr("env.runtime.asyncio.sleep", _no_sleep)

    async def run():
        return await runtime.run_in_session(BashAction(command="echo recover", timeout=1))

    observation = asyncio.run(run())

    assert observation.output == "ok\n"
    assert observation.exit_code == 0
    assert fake.terminated == ["default-run-3"]


async def _no_sleep(*_args, **_kwargs):
    return None
