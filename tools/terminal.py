"""Cancellable terminal tool with host-side risk classification."""

from __future__ import annotations

import asyncio
import json
import locale
import os
import signal
import subprocess

from .context import ToolContext
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
    if process.returncode is not None:
        return
    if os.name == "nt":
        try:
            killer = await asyncio.create_subprocess_exec(
                "taskkill",
                "/PID",
                str(process.pid),
                "/T",
                "/F",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(killer.wait(), timeout=3)
            if killer.returncode != 0 and process.returncode is None:
                process.kill()
        except (FileNotFoundError, asyncio.TimeoutError):
            if process.returncode is None:
                process.kill()
        try:
            await asyncio.wait_for(process.communicate(), timeout=3)
        except asyncio.TimeoutError:
            if process.returncode is None:
                process.kill()
        return
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=2)
            return
        except asyncio.TimeoutError:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
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

    cwd = context.resolve_path(workdir or ".")
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        shell_cmd = ["bash", "-c", command]
        creationflags = 0

    try:
        if os.name == "nt":
            process = await asyncio.create_subprocess_shell(
                command,
                executable=os.environ.get("COMSPEC", "cmd.exe"),
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                creationflags=creationflags,
            )
        else:
            process = await asyncio.create_subprocess_exec(
                *shell_cmd,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                creationflags=creationflags,
                start_new_session=True,
            )
    except FileNotFoundError:
        return ToolResult.failure(
            "Shell not found. On Windows, ensure cmd.exe is available.",
            exit_code=-1,
        )

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        await asyncio.shield(_terminate_process_tree(process))
        return ToolResult.failure(f"Command timed out after {timeout}s", exit_code=-1)
    except asyncio.CancelledError:
        await asyncio.shield(_terminate_process_tree(process))
        raise

    encoding = locale.getpreferredencoding(False) if os.name == "nt" else "utf-8"
    out = stdout.decode(encoding, errors="replace")
    err = stderr.decode(encoding, errors="replace")
    output = out
    if err:
        output += "\n[STDERR]\n" + err
    return json.dumps({
        "output": output.strip() or "(no output)",
        "exit_code": process.returncode,
    }, ensure_ascii=False)


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
