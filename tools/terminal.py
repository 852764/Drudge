"""Terminal 工具 — 执行 shell 命令"""

import json
import subprocess
import os
from pathlib import Path
from .registry import registry


def terminal_handler(
    command: str,
    timeout: int = 180,
    workdir: str | None = None,
    workspace: str | None = None,
    allow_outside_workspace: bool = False,
    allow_terminal: bool = True,
) -> str:
    """执行 shell 命令"""
    # 安全检查：危险命令警告
    if not allow_terminal:
        return json.dumps({"error": "Terminal tool is disabled by config", "blocked": True})

    dangerous_patterns = ["rm -rf /", "mkfs.", "dd if=", ":(){ :|:& };:"]
    for pattern in dangerous_patterns:
        if pattern in command:
            return json.dumps({
                "error": f"Dangerous command detected: pattern '{pattern}' found. Command blocked.",
                "blocked": True,
            })

    try:
        cwd = Path(workdir or workspace or os.getcwd()).expanduser().resolve()
        if workspace and not allow_outside_workspace:
            workspace_root = Path(workspace).expanduser().resolve()
            if cwd != workspace_root and workspace_root not in cwd.parents:
                return json.dumps({
                    "error": f"Workdir outside workspace is blocked: {cwd}",
                    "blocked": True,
                })

        if os.name == "nt":
            shell_cmd = ["cmd.exe", "/c", command]
        else:
            shell_cmd = ["bash", "-c", command]

        result = subprocess.run(
            shell_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd),
            shell=False,
        )

        output = result.stdout
        if result.stderr:
            output += "\n[STDERR]\n" + result.stderr

        return json.dumps({
            "output": output.strip() or "(no output)",
            "exit_code": result.returncode,
        }, ensure_ascii=False)

    except subprocess.TimeoutExpired:
        return json.dumps({
            "error": f"Command timed out after {timeout}s",
            "exit_code": -1,
        })
    except FileNotFoundError:
        return json.dumps({
            "error": "Shell not found. On Windows, ensure cmd.exe is available.",
            "exit_code": -1,
        })
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "exit_code": -1,
        })


def terminal_check() -> bool:
    """Terminal 工具始终可用"""
    return True


registry.register(
    name="terminal",
    description="Execute a shell command. Returns stdout, stderr, and exit code. "
    "Use for running scripts, package managers, git, builds, and file system operations. "
    "Do NOT use for reading files (use read_file) or searching (use search_files). "
    "On Windows, commands run through cmd.exe /c.",
    parameters={
        "command": {
            "type": str,
            "description": "The shell command to execute",
        },
        "timeout": {
            "type": int,
            "description": "Maximum time in seconds to wait for the command (default: 180)",
        },
        "workdir": {
            "type": str,
            "description": "Optional working directory for the command",
        },
    },
    handler=terminal_handler,
    toolset="terminal",
    check_fn=terminal_check,
    required=["command"],
)
