"""Uni-Agent Agent Framework sandbox providers backed by ARL and Nexus."""

from __future__ import annotations

from typing import Any

from swerex.runtime.abstract import Command, ReadFileRequest, WriteFileRequest

from uni_agent.sandbox import ExecResult, Sandbox, SandboxConfig
from uni_agent.sandbox.registry import register_sandbox

from .deployment import (
    ArlDeployment,
    ArlDeploymentConfig,
    NexusDeployment,
    NexusDeploymentConfig,
)


def build_sandbox_from_deployment(
    deployment: dict[str, Any],
    *,
    post_setup_cmd: str | None = None,
) -> Sandbox:
    """Build any supported provider from the legacy deployment mapping."""
    config = dict(deployment)
    provider = str(config.pop("type"))
    if provider in {"arl", "nexus"}:
        # Importing this module has already registered both local providers.
        pass
    elif provider == "local":
        provider = "docker"

    image = str(config.pop("image"))
    runtime_timeout = float(
        config.pop(
            "runtime_timeout",
            config.pop("timeout", 3600),
        )
    )
    for obsolete in ("command", "function_id", "function_route", "deployment_timeout"):
        config.pop(obsolete, None)
    if post_setup_cmd:
        config["post_setup_cmd"] = post_setup_cmd
    from uni_agent.sandbox import build_sandbox

    return build_sandbox(
        SandboxConfig(
            provider=provider,
            image=image,
            runtime_timeout=runtime_timeout,
            sandbox_kwargs=config,
        )
    )


class _DeploymentSandbox(Sandbox):
    """Adapt the existing SWE-ReX deployment bridge to the new Sandbox API."""

    def __init__(
        self,
        deployment: ArlDeployment | NexusDeployment,
        *,
        post_setup_cmd: str | None = None,
    ) -> None:
        self.deployment = deployment
        self.post_setup_cmd = post_setup_cmd

    async def start(self) -> None:
        await self.deployment.start()
        if self.post_setup_cmd:
            result = await self.exec_shell(self.post_setup_cmd, timeout=600)
            if result.exit_code != 0:
                raise RuntimeError(
                    "sandbox post-setup failed "
                    f"(exit={result.exit_code}): {(result.stderr or result.stdout)[-1000:]}"
                )

    async def stop(self) -> None:
        await self.deployment.stop()

    async def is_alive(self) -> bool:
        try:
            response = await self.deployment.is_alive(timeout=10)
        except Exception:
            return False
        return bool(response.is_alive)

    async def _exec(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        response = await self.deployment.runtime.execute(
            Command(
                command=list(argv),
                shell=False,
                check=False,
                timeout=timeout,
                cwd=workdir,
                env=env,
            )
        )
        return ExecResult(
            exit_code=int(response.exit_code or 0),
            stdout=response.stdout or "",
            stderr=response.stderr or "",
        )

    async def read_file(self, path: str) -> bytes:
        response = await self.deployment.runtime.read_file(
            ReadFileRequest(path=path, encoding="utf-8", errors="surrogateescape")
        )
        return response.content.encode("utf-8", errors="surrogateescape")

    async def write_file(self, path: str, content: bytes | str) -> None:
        text = (
            content.decode("utf-8", errors="surrogateescape")
            if isinstance(content, bytes)
            else content
        )
        await self.deployment.runtime.write_file(
            WriteFileRequest(path=path, content=text)
        )


def _sandbox_kwargs(config: SandboxConfig) -> dict[str, Any]:
    return dict(config.sandbox_kwargs or {})


@register_sandbox("arl")
class ArlSandbox(_DeploymentSandbox):
    """Run an Agent Framework task in an ARL managed sandbox."""

    @classmethod
    def from_config(cls, config: SandboxConfig) -> ArlSandbox:
        kwargs = _sandbox_kwargs(config)
        post_setup_cmd = kwargs.pop("post_setup_cmd", None)
        kwargs.setdefault("timeout", config.runtime_timeout)
        deployment_config = ArlDeploymentConfig.from_mapping(
            {"image": config.image, **kwargs}
        )
        return cls(
            ArlDeployment.from_config(deployment_config),
            post_setup_cmd=post_setup_cmd,
        )


@register_sandbox("nexus")
class NexusSandbox(_DeploymentSandbox):
    """Run an Agent Framework task through the Nexus sandbox service."""

    @classmethod
    def from_config(cls, config: SandboxConfig) -> NexusSandbox:
        kwargs = _sandbox_kwargs(config)
        post_setup_cmd = kwargs.pop("post_setup_cmd", None)
        kwargs.setdefault("timeout", config.runtime_timeout)
        deployment_config = NexusDeploymentConfig.from_mapping(
            {"image": config.image, **kwargs}
        )
        return cls(
            NexusDeployment.from_config(deployment_config),
            post_setup_cmd=post_setup_cmd,
        )
