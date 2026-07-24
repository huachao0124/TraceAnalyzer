"""ARL-native and Nexus runtime adapters implementing swe-rex's ``AbstractRuntime``.

The local Agent Framework sandbox providers use this compatibility interface
to back Uni-Agent's ``Sandbox`` API with the ARL SDK or Nexus, without running
a swe-rex server inside the sandbox:

  * bash-session methods -> execute-backed session-state wrapper over
    ``ManagedSession.execute``. ``cd`` and exported env persist through small
    state files, while every command still uses the same one-shot API as
    precompute.
  * ``read_file``/``write_file``/``upload`` -> chunked base64 over one-shot
    ``execute`` (the SDK has no generic file-transfer API in 0.3.1; swap this
    layer if a stable upload/download lands upstream).

The SDK calls are synchronous (httpx / websockets.sync); every async method
offloads them to a worker thread so we satisfy the async ``AbstractRuntime``
contract without blocking the event loop.

This adapter intentionally does not use ARL's human-oriented interactive shell
client for model/tool execution.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import re
import shlex
import tarfile
import threading
import time
import uuid
import warnings
from typing import Any

from swerex.runtime.abstract import (
    AbstractRuntime,
    BashAction,
    BashInterruptAction,
    BashObservation,
    CloseBashSessionRequest,
    CloseBashSessionResponse,
    CloseResponse,
    Command,
    CommandResponse,
    CreateBashSessionRequest,
    CreateBashSessionResponse,
    IsAliveResponse,
    ReadFileRequest,
    ReadFileResponse,
    UploadRequest,
    UploadResponse,
    WriteFileRequest,
    WriteFileResponse,
)

# base64 chunk size per execute step: keep a single shell command well under any
# arg-length limit. base64 expands ~4/3, so 60k chars ≈ 45 KiB of raw bytes/step.
_B64_CHUNK = 60_000
_DEFAULT_CMD_TIMEOUT = 60.0
_NEXUS_START_COMMAND_TIMEOUT = 10.0
_READ_POLL = 0.5

# Transient-error retry. At full-corpus (~4.5k instance) scale the ARL gateway
# occasionally drops a connection mid-call. Stateless execute calls are safe to
# retry, so we treat a bounded set of network exceptions as transient rather
# than failing the instance.
_RETRY_ATTEMPTS = 3
_RETRY_BASE_DELAY = 1.5  # seconds; exponential backoff: 1.5, 3.0, ...
_TERMINAL_CONTROL_RE = re.compile(
    r"\x1b\][^\x07]*(?:\x07|\x1b\\)|\x1b\[[0-?]*[ -/]*[@-~]|\r"
)


def _strip_terminal_controls(text: str) -> str:
    """Remove PTY/readline control sequences before they reach the agent trace."""
    return _TERMINAL_CONTROL_RE.sub("", text)


def _is_transient_step_failure(output: Any) -> bool:
    """Return true when ARL failed before the user's command could run."""
    if int(getattr(output, "exit_code", 0) or 0) == 0:
        return False
    text = "\n".join(
        str(getattr(output, attr, "") or "")
        for attr in ("stderr", "stdout")
    )
    if "gRPC Execute failed" not in text:
        return False
    return (
        "transport: Error while dialing" in text
        or "connection refused" in text
        or "rpc error: code = Unavailable" in text
    )


def _transient_exc_types() -> tuple[type[BaseException], ...]:
    """Best-effort tuple of transient network errors worth retrying.

    Built defensively so this module imports even when a backend lib is absent.
    """
    types: list[type[BaseException]] = [ConnectionError, TimeoutError, OSError]
    try:
        import httpx

        types += [
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
            httpx.PoolTimeout,
        ]
    except Exception:  # noqa: BLE001 - httpx optional at import time
        pass
    try:
        import websockets.exceptions as _wse

        types += [_wse.ConnectionClosed, _wse.ConnectionClosedError, _wse.ConnectionClosedOK]
        for name in ("InvalidStatus", "InvalidStatusCode", "InvalidHandshake"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                exc_type = getattr(_wse, name, None)
            if isinstance(exc_type, type):
                types.append(exc_type)
    except Exception:  # noqa: BLE001 - websockets optional at import time
        pass
    return tuple(dict.fromkeys(types))


_TRANSIENT = _transient_exc_types()


def _extract_gateway_url(session: Any, explicit: str | None) -> str:
    if explicit:
        return explicit
    client = getattr(session, "_client", None)
    for attr in ("_base_url", "base_url"):
        url = getattr(client, attr, None)
        if isinstance(url, str) and url:
            return url
    raise ValueError(
        "Cannot determine ARL gateway_url from session; pass gateway_url= explicitly."
    )


class ArlRuntime(AbstractRuntime):
    """swe-rex ``AbstractRuntime`` backed by an ARL ``ManagedSession``."""

    def __init__(
        self,
        session: Any,
        *,
        run_id: str,
        logger: Any | None = None,
        gateway_url: str | None = None,
        api_key: str | None = None,
        startup_commands: list[str] | None = None,
        one_time_startup_commands: list[str] | None = None,
        session_cwd: str | None = "/testbed",
    ) -> None:
        self._session = session
        self.run_id = run_id
        self.logger = logger or logging.getLogger(f"arl-runtime.{run_id}")
        self._gateway_url = _extract_gateway_url(session, gateway_url)
        self._api_key = api_key
        self._session_cwd = session_cwd.strip() if isinstance(session_cwd, str) and session_cwd.strip() else None
        self._startup_commands = [command for command in (startup_commands or []) if command.strip()]
        self._one_time_startup_commands = [
            command for command in (one_time_startup_commands or []) if command.strip()
        ]
        self._completed_one_time_startup_sessions: set[str] = set()
        self._session_startup_sources: dict[str, list[str]] = {}
        self._initialized_sessions: set[str] = set()
        safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id)
        self._state_root = f"/tmp/arl_runtime_sessions/{safe_run_id}"
        self._closed = False

    # ── async plumbing ────────────────────────────────────────────────────
    async def _blocking(self, func, *args):
        result: dict[str, Any] = {}

        def _target() -> None:
            try:
                result["value"] = func(*args)
            except BaseException as exc:  # noqa: BLE001 - transfer sync failure to async caller
                result["error"] = exc

        thread = threading.Thread(target=_target, name="arl-runtime", daemon=True)
        thread.start()
        while thread.is_alive():
            await asyncio.sleep(0.05)
        if "error" in result:
            raise result["error"]
        return result.get("value")

    @property
    def _arl_session_id(self) -> str:
        sid = getattr(self._session, "session_id", None) or getattr(self._session, "_session_id", None)
        if not sid:
            raise RuntimeError("ARL ManagedSession has no session_id (not created?).")
        return sid

    # ── transient-error retry ─────────────────────────────────────────────
    def _retry_sync(self, what: str, func, *args):
        """Run a synchronous SDK call, retrying transient connection drops."""
        last: BaseException | None = None
        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            try:
                return func(*args)
            except _TRANSIENT as exc:
                last = exc
                if attempt >= _RETRY_ATTEMPTS:
                    break
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                self.logger.warning(
                    "ARL %s transient error (attempt %d/%d), retry in %.1fs: %r",
                    what, attempt, _RETRY_ATTEMPTS, delay, exc,
                )
                time.sleep(delay)
        raise last  # type: ignore[misc]

    def _session_execute(self, steps: list[dict[str, Any]]) -> Any:
        """ManagedSession.execute with transient-drop retry (stateless → safe)."""
        return self._retry_sync("execute", self._session.execute, steps)

    def _execute_steps_sync(self, what: str, steps: list[dict[str, Any]]) -> Any:
        """Execute ARL steps, retrying gateway failures reported as step output."""
        last_resp: Any | None = None
        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            resp = self._session_execute(steps)
            last_resp = resp
            transient = next(
                (
                    result.output
                    for result in getattr(resp, "results", []) or []
                    if _is_transient_step_failure(getattr(result, "output", None))
                ),
                None,
            )
            if transient is None:
                return resp
            if attempt >= _RETRY_ATTEMPTS:
                break
            delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
            message = (getattr(transient, "stderr", "") or getattr(transient, "stdout", "") or "").strip()
            self.logger.warning(
                "ARL %s transient step failure (attempt %d/%d), retry in %.1fs: %s",
                what, attempt, _RETRY_ATTEMPTS, delay, message[-300:],
            )
            time.sleep(delay)
        return last_resp

    # ── execute-backed bash session emulation ─────────────────────────────
    def _session_state_dir(self, name: str) -> str:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        return f"{self._state_root}/{safe_name}"

    def _session_wrapper(self, name: str, command: str) -> str:
        state_dir = shlex.quote(self._session_state_dir(name))
        initial_cwd = shlex.quote(self._session_cwd or "")
        return "\n".join(
            [
                "#!/usr/bin/env bash",
                f"__arl_state_dir={state_dir}",
                f"__arl_initial_cwd={initial_cwd}",
                'mkdir -p "$__arl_state_dir"',
                'if [ -f "$__arl_state_dir/env.sh" ]; then source "$__arl_state_dir/env.sh"; fi',
                'if [ -f "$__arl_state_dir/cwd" ]; then cd "$(cat "$__arl_state_dir/cwd")" 2>/dev/null || true; fi',
                'if [ ! -f "$__arl_state_dir/cwd" ] && [ -n "$__arl_initial_cwd" ]; then cd "$__arl_initial_cwd" || exit $?; fi',
                "__arl_dump_state() {",
                "  __arl_status=$?",
                "  trap - EXIT",
                "  set +e",
                '  pwd > "$__arl_state_dir/cwd.tmp" && mv "$__arl_state_dir/cwd.tmp" "$__arl_state_dir/cwd"',
                '  export -p > "$__arl_state_dir/env.sh.tmp" && mv "$__arl_state_dir/env.sh.tmp" "$__arl_state_dir/env.sh"',
                "  exit $__arl_status",
                "}",
                "trap __arl_dump_state EXIT",
                "__arl_user_command() {",
                command,
                "}",
                "__arl_user_command",
                "exit $?",
                "",
            ]
        )

    def _run_session_command_sync(self, name: str, command: str, timeout: float) -> tuple[str, int, str]:
        script_path = f"{self._session_state_dir(name)}/cmd_{uuid.uuid4().hex}.sh"
        self._write_bytes_sync(self._session_wrapper(name, command).encode("utf-8"), script_path)
        output = self._exec_sync(f"bash {shlex.quote(script_path)} 2>&1", timeout)
        text = _strip_terminal_controls((output.stdout or "") + (output.stderr or ""))
        return text, int(output.exit_code or 0), ""

    def _ensure_session_sync(self, name: str, startup_source: list[str] | None = None) -> str:
        if startup_source is not None:
            self._session_startup_sources[name] = list(startup_source)
        if name in self._initialized_sessions:
            return ""

        commands: list[str] = [
            f"source {shlex.quote(src)} 2>/dev/null || true"
            for src in self._session_startup_sources.get(name, [])
        ]
        commands.extend(self._startup_commands)
        if name not in self._completed_one_time_startup_sessions:
            commands.extend(self._one_time_startup_commands)
        command = "\n".join(commands) if commands else "true"
        output, exit_code, failure = self._run_session_command_sync(name, command, _DEFAULT_CMD_TIMEOUT)
        if exit_code != 0 or failure:
            raise RuntimeError(f"ARL execute-backed session startup failed ({exit_code=} {failure}): {output[-1000:]}")
        self._initialized_sessions.add(name)
        self._completed_one_time_startup_sessions.add(name)
        return output

    # ── one-shot execute helpers (stateless ManagedSession.execute) ────────
    def _exec_sync(self, shell_cmd: str, timeout: float | None = None) -> Any:
        step: dict[str, Any] = {"name": "arl-exec", "command": ["bash", "-lc", shell_cmd]}
        if timeout:
            step["timeout"] = int(timeout)
        resp = self._execute_steps_sync("execute", [step])
        if not resp.results:
            raise RuntimeError("ARL execute returned no results")
        return resp.results[0].output

    def _write_bytes_sync(self, data: bytes, remote_path: str) -> None:
        b64 = base64.b64encode(data).decode("ascii")
        quoted = shlex.quote(remote_path)
        parent = os.path.dirname(remote_path)
        steps: list[dict[str, Any]] = []
        if parent:
            steps.append({"name": "mkdir", "command": ["bash", "-lc", f"mkdir -p {shlex.quote(parent)}"]})
        # First chunk truncates (`>`), the rest append (`>>`). The `or [0]`
        # guarantees one iteration for empty content -> an empty file is written.
        first = True
        for i in range(0, len(b64), _B64_CHUNK) or [0]:
            chunk = b64[i : i + _B64_CHUNK] if b64 else ""
            redirect = ">" if first else ">>"
            steps.append({
                "name": f"write-{i}",
                "command": ["bash", "-lc", f"printf %s {shlex.quote(chunk)} | base64 -d {redirect} {quoted}"],
            })
            first = False
        resp = self._execute_steps_sync("write_file", steps)
        bad = [r for r in resp.results if r.output.exit_code != 0]
        if bad:
            raise RuntimeError(f"write_file failed: {bad[-1].output.stderr}")

    # ── AbstractRuntime: 9 methods ─────────────────────────────────────────
    async def create_session(self, request: CreateBashSessionRequest) -> CreateBashSessionResponse:
        name = request.session
        output = await self._blocking(self._ensure_session_sync, name, list(request.startup_source or []))
        return CreateBashSessionResponse(output=output, session_type="bash")

    async def run_in_session(self, action: BashAction | BashInterruptAction) -> BashObservation:
        name = getattr(action, "session", "default")
        if getattr(action, "action_type", "command") == "interrupt":
            return BashObservation(output="", exit_code=0, session_type="bash")

        timeout = float(getattr(action, "timeout", None) or _DEFAULT_CMD_TIMEOUT)
        await self._blocking(self._ensure_session_sync, name, None)
        output, exit_code, failure = await self._blocking(self._run_session_command_sync, name, action.command, timeout)
        return BashObservation(
            output=output,
            exit_code=exit_code,
            failure_reason=failure,
            session_type="bash",
        )

    async def close_session(self, request: CloseBashSessionRequest) -> CloseBashSessionResponse:
        self._initialized_sessions.discard(request.session)
        await self._blocking(self._exec_sync, f"rm -rf {shlex.quote(self._session_state_dir(request.session))}")
        return CloseBashSessionResponse(session_type="bash")

    async def execute(self, command: Command) -> CommandResponse:
        cmd = command.command
        cmd_str = cmd if isinstance(cmd, str) else " ".join(shlex.quote(c) for c in cmd)
        prefix = ""
        if getattr(command, "cwd", ""):
            prefix += f"cd {shlex.quote(command.cwd)} && "
        for k, v in (getattr(command, "env", None) or {}).items():
            prefix += f"export {k}={shlex.quote(str(v))}; "
        output = await self._blocking(self._exec_sync, prefix + cmd_str, getattr(command, "timeout", None))
        return CommandResponse(stdout=output.stdout, stderr=output.stderr, exit_code=output.exit_code)

    async def read_file(self, request: ReadFileRequest) -> ReadFileResponse:
        output = await self._blocking(self._exec_sync, f"base64 {shlex.quote(request.path)}")
        if output.exit_code != 0:
            raise FileNotFoundError(f"read_file {request.path!r} failed: {output.stderr}")
        raw = base64.b64decode("".join(output.stdout.split()))
        content = raw.decode(request.encoding or "utf-8", request.errors or "strict")
        return ReadFileResponse(content=content)

    async def write_file(self, request: WriteFileRequest) -> WriteFileResponse:
        data = request.content.encode("utf-8") if isinstance(request.content, str) else request.content
        await self._blocking(self._write_bytes_sync, data, request.path)
        return WriteFileResponse()

    async def upload(self, request: UploadRequest) -> UploadResponse:
        src, dst = request.source_path, request.target_path

        def _upload_sync() -> None:
            if os.path.isdir(src):
                buf = io.BytesIO()
                with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                    tar.add(src, arcname=".")
                staging = f"/tmp/arl_upload_{uuid.uuid4().hex}.tgz"
                self._write_bytes_sync(buf.getvalue(), staging)
                out = self._exec_sync(
                    f"mkdir -p {shlex.quote(dst)} && tar xzf {shlex.quote(staging)} -C {shlex.quote(dst)} "
                    f"&& rm -f {shlex.quote(staging)}"
                )
                if out.exit_code != 0:
                    raise RuntimeError(f"upload(dir) failed: {out.stderr}")
            else:
                with open(src, "rb") as fh:
                    self._write_bytes_sync(fh.read(), dst)

        await self._blocking(_upload_sync)
        return UploadResponse()

    async def is_alive(self, *, timeout: float | None = None) -> IsAliveResponse:
        try:
            output = await self._blocking(self._exec_sync, "echo ok", timeout)
            ok = output.exit_code == 0 and "ok" in output.stdout
            return IsAliveResponse(is_alive=ok, message="" if ok else output.stderr)
        except Exception as exc:  # noqa: BLE001 - report liveness failures, don't raise
            return IsAliveResponse(is_alive=False, message=str(exc))

    async def close(self) -> CloseResponse:
        if self._closed:
            return CloseResponse()
        self._closed = True
        self._initialized_sessions.clear()
        try:
            await self._blocking(self._exec_sync, f"rm -rf {shlex.quote(self._state_root)}")
        except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
            self.logger.debug(f"session-state cleanup failed: {exc}")
        return CloseResponse()


class NexusRuntime(AbstractRuntime):
    """swe-rex ``AbstractRuntime`` backed by a Nexus ``RemoteRuntime`` (HttpRuntime).

    Uses Nexus terminal sessions (``create_terminal_session`` /
    ``run_command_in_terminal_session``) for a **persistent** shell — ``cd``,
    ``export``, aliases survive across ``run_in_session`` calls, matching the
    behavior of ARL's WebSocket PTY shell.

    Falls back to stateless ``run_command`` for one-shot ``execute`` calls.
    """

    def __init__(
        self,
        nexus_runtime: Any,
        *,
        run_id: str,
        logger: Any | None = None,
    ) -> None:
        self._nexus = nexus_runtime
        self.run_id = run_id
        self.logger = logger or logging.getLogger(f"nexus-runtime.{run_id}")
        self._terminal_sessions: dict[str, str] = {}
        self._closed = False

    async def create_session(self, request: CreateBashSessionRequest) -> CreateBashSessionResponse:
        name = request.session
        if name not in self._terminal_sessions:
            unique_sid = f"{name}-{self.run_id}"
            resp = await self._nexus.create_terminal_session(session_id=unique_sid)
            sid = getattr(resp, "session_id", None) or unique_sid
            self._terminal_sessions[name] = sid
            self.logger.info(f"Created terminal session: {name} -> {sid}")
            try:
                await asyncio.wait_for(
                    self._nexus.start_command_in_terminal_session(
                        session_id=sid,
                        command="BASH_ARGV0=bash; bind 'set enable-bracketed-paste off' 2>/dev/null; true",
                    ),
                    timeout=_NEXUS_START_COMMAND_TIMEOUT,
                )
            except asyncio.TimeoutError:
                self.logger.warning("Nexus terminal session init timed out for %s", name)
                try:
                    await asyncio.wait_for(
                        self._nexus.terminate_terminal_session_processes(session_id=sid),
                        timeout=_NEXUS_START_COMMAND_TIMEOUT,
                    )
                except Exception:
                    pass
            except Exception:
                pass
        return CreateBashSessionResponse(output="", session_type="bash")

    _MAX_RETRY_DEPTH = 2

    async def run_in_session(
        self,
        action: BashAction | BashInterruptAction,
        *,
        _retry_depth: int = 0,
    ) -> BashObservation:
        """Execute a command via Nexus async terminal API.

        Uses ``start_command_in_terminal_session`` (non-blocking) +
        ``query_terminal_command_status`` (poll until done). Output comes from
        nexus_bash's fd-redirect hooks (clean stdout/stderr, no prompt/echo).

        On timeout or stuck commands (trailing backslash, unmatched quotes),
        sends Ctrl+C via ``send_keys`` to recover the session.
        """
        name = getattr(action, "session", "default")
        if getattr(action, "action_type", "command") == "interrupt":
            sid = self._terminal_sessions.get(name)
            if sid:
                try:
                    await self._nexus.send_keys_to_terminal_session(
                        session_id=sid, keys=["C-c", "C-c"],
                    )
                except Exception:
                    pass
            return BashObservation(output="", exit_code=0, session_type="bash")

        if name not in self._terminal_sessions:
            await self.create_session(CreateBashSessionRequest(session=name))

        sid = self._terminal_sessions[name]
        timeout = float(getattr(action, "timeout", None) or _DEFAULT_CMD_TIMEOUT)

        try:
            # The gateway can wait on a stale shell hook before returning. Bound
            # the SDK call so one wedged session cannot block the whole rollout.
            try:
                info = await asyncio.wait_for(
                    self._nexus.start_command_in_terminal_session(
                        session_id=sid,
                        command=action.command,
                    ),
                    timeout=_NEXUS_START_COMMAND_TIMEOUT,
                )
            except asyncio.TimeoutError:
                self.logger.warning(
                    "Nexus start_command timed out on session %s; terminating children",
                    name,
                )
                try:
                    await asyncio.wait_for(
                        self._nexus.terminate_terminal_session_processes(session_id=sid),
                        timeout=_NEXUS_START_COMMAND_TIMEOUT,
                    )
                except Exception:
                    pass
                await asyncio.sleep(2)
                try:
                    info = await asyncio.wait_for(
                        self._nexus.start_command_in_terminal_session(
                            session_id=sid,
                            command=action.command,
                        ),
                        timeout=_NEXUS_START_COMMAND_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    return BashObservation(
                        output="Nexus start_command timed out after recovery",
                        exit_code=-1,
                        failure_reason="timeout",
                        session_type="bash",
                    )
            command_id = info.command_id

            # Poll until command completes (end_time is set by nexus_bash postexec hook)
            deadline = time.monotonic() + timeout
            while True:
                if time.monotonic() > deadline:
                    # Timeout — interrupt and wait for nexus to mark command done
                    try:
                        await self._nexus.send_keys_to_terminal_session(
                            session_id=sid, keys=["C-c", "C-c"],
                        )
                        await asyncio.sleep(1)
                        await self._nexus.terminate_terminal_session_processes(
                            session_id=sid,
                        )
                    except Exception:
                        pass
                    # Poll until nexus marks the command as finished
                    output = ""
                    for _ in range(30):
                        try:
                            final = await self._nexus.query_terminal_command_status(
                                session_id=sid, command_id=command_id,
                            )
                            output = final.output or final.stdout or ""
                            if final.end_time is not None:
                                break
                        except Exception:
                            break
                        await asyncio.sleep(1)
                    return BashObservation(
                        output=output,
                        exit_code=-1,
                        failure_reason="timeout",
                        session_type="bash",
                    )

                await asyncio.sleep(_READ_POLL)
                status = await self._nexus.query_terminal_command_status(
                    session_id=sid, command_id=command_id,
                )
                if status.end_time is not None:
                    stdout = status.stdout or ""
                    stderr = status.stderr or ""
                    output = status.output or (stdout + stderr) or ""
                    exit_code = status.exit_code if status.exit_code is not None else -1
                    return BashObservation(
                        output=output,
                        exit_code=exit_code,
                        failure_reason="",
                        session_type="bash",
                    )

        except Exception as exc:
            exc_str = str(exc)
            if _retry_depth >= self._MAX_RETRY_DEPTH:
                self.logger.error(
                    "Nexus session retry depth exceeded for %s",
                    name,
                )
                return BashObservation(
                    output=exc_str,
                    exit_code=1,
                    failure_reason="max_retries",
                    session_type="bash",
                )
            if "Failed to send command" in exc_str:
                self.logger.warning("Shell dead, recreating session %s and retrying", name)
                try:
                    await self._nexus.destroy_terminal_session(session_id=sid)
                except Exception:
                    pass
                self._terminal_sessions.pop(name, None)
                try:
                    await self.create_session(CreateBashSessionRequest(session=name))
                    return await self.run_in_session(action, _retry_depth=_retry_depth + 1)
                except Exception:
                    pass
            elif "command is already running" in exc_str.lower():
                self.logger.warning("Nexus session %s is still busy; terminating children", name)
                try:
                    await self._nexus.terminate_terminal_session_processes(session_id=sid)
                except Exception:
                    pass
                await asyncio.sleep(2)
                try:
                    return await self.run_in_session(action, _retry_depth=_retry_depth + 1)
                except Exception:
                    pass
            elif "Command failed to start" in exc_str:
                try:
                    await self._nexus.send_keys_to_terminal_session(
                        session_id=sid, keys=["C-c", "C-c"],
                    )
                except Exception:
                    pass
            return BashObservation(
                output=exc_str,
                exit_code=1,
                failure_reason="",
                session_type="bash",
            )

    async def close_session(self, request: CloseBashSessionRequest) -> CloseBashSessionResponse:
        name = request.session
        sid = self._terminal_sessions.pop(name, None)
        if sid:
            try:
                await self._nexus.destroy_terminal_session(session_id=sid)
            except Exception as exc:
                self.logger.warning(f"Failed to destroy terminal session {sid}: {exc}")
        return CloseBashSessionResponse(session_type="bash")

    async def execute(self, command: Command) -> CommandResponse:
        cmd = command.command
        cmd_str = cmd if isinstance(cmd, str) else " ".join(shlex.quote(c) for c in cmd)
        prefix = ""
        if getattr(command, "cwd", ""):
            prefix += f"cd {shlex.quote(command.cwd)} && "
        for k, v in (getattr(command, "env", None) or {}).items():
            prefix += f"export {k}={shlex.quote(str(v))}; "
        timeout = float(getattr(command, "timeout", None) or _DEFAULT_CMD_TIMEOUT)
        resp = await self._nexus.run_command(
            command=prefix + cmd_str,
            timeout=timeout,
        )
        return CommandResponse(
            stdout=resp.stdout or "",
            stderr=resp.stderr or "",
            exit_code=resp.return_code if resp.return_code is not None else 0,
        )

    async def read_file(self, request: ReadFileRequest) -> ReadFileResponse:
        resp = await self._nexus.run_command(
            command=f"base64 {shlex.quote(request.path)}",
            timeout=60,
        )
        if (resp.return_code or 0) != 0:
            raise FileNotFoundError(f"read_file {request.path!r} failed: {resp.stderr}")
        raw = base64.b64decode("".join((resp.stdout or "").split()))
        content = raw.decode(request.encoding or "utf-8", request.errors or "strict")
        return ReadFileResponse(content=content)

    async def write_file(self, request: WriteFileRequest) -> WriteFileResponse:
        data = request.content.encode("utf-8") if isinstance(request.content, str) else request.content
        b64 = base64.b64encode(data).decode("ascii")
        parent = os.path.dirname(request.path)
        quoted = shlex.quote(request.path)
        if parent:
            await self._nexus.run_command(
                command=f"mkdir -p {shlex.quote(parent)}",
                timeout=30,
            )
        # Write in chunks to stay under shell arg limits.
        first = True
        for i in range(0, max(len(b64), 1), _B64_CHUNK):
            chunk = b64[i : i + _B64_CHUNK] if b64 else ""
            redirect = ">" if first else ">>"
            resp = await self._nexus.run_command(
                command=f"printf %s {shlex.quote(chunk)} | base64 -d {redirect} {quoted}",
                timeout=60,
            )
            if (resp.return_code or 0) != 0:
                raise RuntimeError(f"write_file failed: {resp.stderr}")
            first = False
        return WriteFileResponse()

    async def upload(self, request: UploadRequest) -> UploadResponse:
        src, dst = request.source_path, request.target_path

        if os.path.isdir(src):
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                tar.add(src, arcname=".")
            staging = f"/tmp/nexus_upload_{uuid.uuid4().hex}.tgz"
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            # Upload tarball in chunks.
            first = True
            quoted_staging = shlex.quote(staging)
            for i in range(0, max(len(b64), 1), _B64_CHUNK):
                chunk = b64[i : i + _B64_CHUNK] if b64 else ""
                redirect = ">" if first else ">>"
                await self._nexus.run_command(
                    command=f"printf %s {shlex.quote(chunk)} | base64 -d {redirect} {quoted_staging}",
                    timeout=60,
                )
                first = False
            resp = await self._nexus.run_command(
                command=(
                    f"mkdir -p {shlex.quote(dst)} && "
                    f"tar xzf {quoted_staging} -C {shlex.quote(dst)} && "
                    f"rm -f {quoted_staging}"
                ),
                timeout=120,
            )
            if (resp.return_code or 0) != 0:
                raise RuntimeError(f"upload(dir) failed: {resp.stderr}")
        else:
            with open(src, "rb") as fh:
                content = fh.read()
            write_req = WriteFileRequest(path=dst, content=content)
            await self.write_file(write_req)

        return UploadResponse()

    async def is_alive(self, *, timeout: float | None = None) -> IsAliveResponse:
        try:
            resp = await self._nexus.run_command(
                command="echo ok",
                timeout=timeout or 30,
            )
            ok = (resp.return_code or 0) == 0 and "ok" in (resp.stdout or "")
            return IsAliveResponse(is_alive=ok, message="" if ok else (resp.stderr or ""))
        except Exception as exc:  # noqa: BLE001 - report liveness failures, don't raise
            return IsAliveResponse(is_alive=False, message=str(exc))

    async def close(self) -> CloseResponse:
        if self._closed:
            return CloseResponse()
        self._closed = True
        for sid in list(self._terminal_sessions.values()):
            try:
                await self._nexus.destroy_terminal_session(session_id=sid)
            except Exception as exc:
                self.logger.debug("Failed to destroy Nexus terminal session %s: %r", sid, exc)
        self._terminal_sessions.clear()
        return CloseResponse()
