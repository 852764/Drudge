"""Cancellable terminal tool with host-side risk classification."""

from __future__ import annotations

import asyncio
import codecs
import json
import locale
import os
import signal
import subprocess
import sys
from pathlib import Path

from .context import ToolContext
from .output_capture import OutputCapture
from .registry import registry
from .result import ToolResult
from .risk import RiskLevel, ToolRisk


_CRITICAL_MARKERS = (
    "rm -rf /",
    "format ",
    "mkfs.",
    "dd if=",
    "shutdown",
    "restart-computer",
    "remove-item -recurse",
    "del /s",
    "rmdir /s",
    ":(){ :|:& };:",
)

_HIGH_RISK_MARKERS = (
    "git push",
    "pip install",
    "npm install",
    "pnpm install",
    "yarn add",
    "cargo install",
    "curl ",
    "wget ",
    "invoke-webrequest",
    "remove-item",
    " del ",
    " rmdir ",
    "taskkill",
    "stop-process",
    "sc.exe ",
    "reg add",
)


def _terminal_risk(args: dict, context: ToolContext) -> ToolRisk:
    command = str(args.get("command", ""))
    lowered = f" {command.lower()} "
    if any(marker in lowered for marker in _CRITICAL_MARKERS):
        return ToolRisk(RiskLevel.CRITICAL, "Potentially destructive system command", command)
    if any(marker in lowered for marker in _HIGH_RISK_MARKERS):
        return ToolRisk(RiskLevel.HIGH, "Command may alter external or system state", command)
    return ToolRisk(RiskLevel.MEDIUM, "Execute a local shell command", command)


async def _terminate_process_tree(process: asyncio.subprocess.Process) -> None:
    if os.name == "nt":
        # This PID is the isolated owner, not the shell. Killing it closes its
        # Job Object handle and terminates descendants even if the shell exited.
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        await asyncio.wait_for(process.wait(), timeout=3)
        return
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=2)
        except asyncio.TimeoutError:
            pass
        # The shell can exit before its descendants close inherited pipes.
        # Finish terminating the process group even if its leader has exited.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    await asyncio.wait_for(process.wait(), timeout=3)


async def terminal_handler(
    command: str,
    timeout: int = 180,
    workdir: str | None = None,
    context: ToolContext | None = None,
) -> str | ToolResult:
    """Execute a shell command and terminate its process tree on timeout/cancel."""
    if context is None:
        return ToolResult.failure("ToolContext is required", blocked=True)
    allowed, reason = context.terminal_allowed(command)
    if not allowed:
        return ToolResult.failure(reason or "Terminal command blocked", blocked=True)
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        return ToolResult.failure("timeout must be a positive integer")

    cwd = context.resolve_path(workdir or ".")
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        shell_cmd = ["bash", "-c", command]
        creationflags = 0

    encoding = locale.getpreferredencoding(False) if os.name == "nt" else "utf-8"
    captures: dict[str, OutputCapture] = {}
    try:
        for name in ("stdout", "stderr"):
            captures[name] = OutputCapture(persist=context.save_tool_output is not None, max_bytes=context.max_output_bytes)
        try:
            if os.name == "nt":
                process = await asyncio.create_subprocess_exec(
                    sys.executable, "-I", "-u", str(Path(__file__).with_name("_windows_job.py")), command, cwd=str(cwd),
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, creationflags=creationflags,
                )
            else:
                process = await asyncio.create_subprocess_exec(
                    *shell_cmd, cwd=str(cwd), stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE, creationflags=creationflags, start_new_session=True,
                )
        except OSError as exc:
            return ToolResult.failure(f"Command launch failed: {type(exc).__name__}: {exc}", exit_code=-1)

        async def drain(stream, capture):
            decoder = codecs.getincrementaldecoder(encoding)(errors="replace")
            while True:
                data = await stream.read(16_384)
                if not data:
                    break
                capture.write(decoder.decode(data))
            capture.write(decoder.decode(b"", final=True))

        readers = [asyncio.create_task(drain(process.stdout, captures["stdout"])), asyncio.create_task(drain(process.stderr, captures["stderr"]))]

        async def monitor():
            await asyncio.gather(process.wait(), *readers)

        monitor_task = asyncio.create_task(monitor())

        cleanup_warnings = []

        async def stop():
            try:
                await _terminate_process_tree(process)
                await asyncio.wait_for(asyncio.shield(monitor_task), timeout=2)
            except Exception as exc:
                cleanup_warnings.append(f"Process cleanup incomplete (pid={process.pid}): {type(exc).__name__}: {exc}")
            finally:
                for task in [monitor_task, *readers]:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(monitor_task, *readers, return_exceptions=True)

        async def stop_until_done() -> bool:
            # Retain and await the cleanup task: a second Ctrl-C must not let it
            # outlive the captures or the event loop. Preserve cancellation.
            cleanup = asyncio.create_task(stop())
            cancelled = False
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    cancelled = True
            cleanup.result()
            return cancelled

        status = "completed"
        error = None
        try:
            await asyncio.wait_for(asyncio.shield(monitor_task), timeout=timeout)
        except asyncio.TimeoutError:
            status = "timed_out"
            error = f"Command timed out after {timeout}s"
            if await stop_until_done():
                status = "cancelled"
        except asyncio.CancelledError:
            status = "cancelled"
            await stop_until_done()
        except Exception as exc:
            status = "capture_error"
            error = f"Command output capture failed: {type(exc).__name__}: {exc}"
            if await stop_until_done():
                status = "cancelled"

        refs = {}
        warnings = list(cleanup_warnings)
        for name, capture in captures.items():
            ref = capture.finish(context.save_tool_output, tool_name=f"terminal.{name}", status=status)
            if ref:
                refs[name] = ref
            if capture.warning:
                warnings.append(capture.warning)
        if status == "cancelled":
            raise asyncio.CancelledError("; ".join(cleanup_warnings) or "Command cancelled")
        out = captures["stdout"].preview()
        err = captures["stderr"].preview()
        output = out + ("\n[STDERR]\n" + err if err else "")
        exit_code = process.returncode if process.returncode is not None else -1
        if error is None and exit_code != 0:
            error = f"Command exited with code {exit_code}"
        return json.dumps({
            "ok": error is None, "content": output or "(no output)", "error": error,
            "metadata": {
                "exit_code": exit_code, "timed_out": status == "timed_out", "encoding": encoding,
                "output_refs": refs,
                "stdout_chars": captures["stdout"].source_chars, "stderr_chars": captures["stderr"].source_chars,
                "preview_truncated": any(capture.truncated for capture in captures.values()),
                "warnings": warnings,
            },
            # Keep legacy fields for existing clients.
            "output": output or "(no output)", "exit_code": exit_code,
        }, ensure_ascii=False)
    finally:
        for capture in captures.values():
            capture.close()


def terminal_check() -> bool:
    return True


registry.register(
    name="terminal",
    description="Execute a cancellable shell command and return stdout, stderr, and exit code. "
    "Use for scripts, package managers, git, and builds; use file tools for reading/searching.",
    parameters={
        "command": {"type": str, "description": "The shell command to execute"},
        "timeout": {"type": int, "description": "Maximum runtime in seconds (default: 180)"},
        "workdir": {"type": str, "description": "Optional working directory"},
    },
    handler=terminal_handler,
    toolset="terminal",
    check_fn=terminal_check,
    required=["command"],
    risk_fn=_terminal_risk,
)
