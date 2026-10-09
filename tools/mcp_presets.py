"""Host-owned MCP presets; no downloads, process launch, or credential lookup."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


# Reviewed against JSReverser-MCP 2.0.4, commit 65e2e3cb70c1.
# Use the shared DevTools context. Upstream collector-based helpers initialize a
# second browser (default port 9222) before consulting their bound page resolver.
# DOM/form inspection is available through evaluate_script on the selected page.
BROWSER_TOOLS = (
    "list_pages", "new_page", "select_page", "navigate_page", "click_element",
    "console_message", "network_request",
    "list_scripts", "get_script_source", "find_in_script", "search_in_sources",
    "breakpoint", "set_breakpoint_on_text", "xhr_breakpoint", "get_request_initiator",
    "get_paused_info", "pause", "resume", "step_into", "step_over", "step_out",
    "evaluate_script", "evaluate_on_callframe", "list_frames", "select_frame",
    "take_screenshot",
)

# Deliberately excludes API keys, proxy credentials and arbitrary host variables.
BROWSER_ENV = (
    "PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "TMPDIR",
    "HOME", "USERPROFILE", "LOCALAPPDATA", "APPDATA", "PROGRAMFILES",
    "PROGRAMFILES(X86)", "PROGRAMW6432", "LANG", "LC_ALL", "DISPLAY",
    "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR", "XAUTHORITY",
)


def expand_mcp_preset(config: dict[str, Any], workspace: str | Path) -> dict[str, Any]:
    """Expand only explicitly selected presets; plain MCP settings are unchanged."""
    preset = config.get("preset")
    if preset is None:
        return dict(config)
    if preset != "jsreverser":
        raise ValueError(f"Unknown MCP preset: {preset}")
    server_path = config.get("server_path")
    if not isinstance(server_path, str) or not server_path.strip():
        raise ValueError("jsreverser preset requires server_path to build/src/index.js")
    root = Path(workspace).expanduser().resolve()
    entry = Path(server_path).expanduser()
    if not entry.is_absolute():
        entry = root / entry
    profile = config.get("profile", "browser")
    if profile not in ("browser", "full"):
        raise ValueError("jsreverser profile must be browser or full")
    headless = config.get("headless", True)
    if not isinstance(headless, bool):
        raise ValueError("jsreverser headless must be a boolean")
    browser_url = config.get("browser_url")
    executable = config.get("executable_path")
    if browser_url and executable:
        raise ValueError("Choose browser_url or executable_path, not both")
    args = [str(entry.resolve()), "--toolProfile", "full"]
    if browser_url:
        url = urlsplit(str(browser_url))
        # Attaching a browser exposes full DevTools access; keep the preset local.
        if (url.scheme not in ("http", "https") or url.hostname not in (
            "127.0.0.1", "localhost", "::1",
        ) or url.username or url.password or url.query or url.fragment
                or url.path not in ("", "/")):
            raise ValueError("browser_url must be a loopback Chrome debugging URL")
        try:
            if url.port == 0:
                raise ValueError
        except ValueError as exc:
            raise ValueError("browser_url has an invalid port") from exc
        args += ["--browserUrl", str(browser_url)]
    else:
        args += ["--isolated", f"--headless={'true' if headless else 'false'}"]
        if executable:
            path = Path(str(executable)).expanduser()
            args += ["--executablePath", str((root / path).resolve())]
    expanded: dict[str, Any] = {
        "command": "node",
        "args": args,
        "cwd": str(root / ".drudge" / "browser"),
        "create_cwd": True,
        "timeout": 90,
        "max_message_bytes": 8 * 1024 * 1024,
        "risk": "medium",
        "requires_network": True,
        "inherit_env": False,
        "env_passthrough": list(BROWSER_ENV),
        "env": {"JSREVERSER_ARTIFACTS_DIR": str(root / ".drudge" / "browser" / "artifacts")},
        "include_resources": False,
        "include_prompts": False,
    }
    if profile == "browser":
        expanded["allowed_tools"] = list(BROWSER_TOOLS)
    # These overrides are host configuration, never model tool arguments.
    expanded.update(config)
    return expanded
