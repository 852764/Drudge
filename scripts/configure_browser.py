"""Register an already-built JSReverser-MCP in a local Drudge YAML config."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from tools.file_io import atomic_write
from tools.mcp_presets import expand_mcp_preset
from tools.provider import MCPServerProvider


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-path", required=True, help="Built JSReverser-MCP build/src/index.js")
    parser.add_argument("--config", default="config.local.yaml", help="Local YAML to merge into")
    parser.add_argument("--profile", choices=("browser", "full"), default="browser")
    connection = parser.add_mutually_exclusive_group()
    connection.add_argument("--browser-url", help="Attach to loopback Chrome DevTools, e.g. http://127.0.0.1:9222")
    connection.add_argument("--executable-path", help="Custom Chrome/Chromium/Edge executable")
    parser.add_argument("--headed", action="store_true", help="Show a separate browser window instead of headless")
    parser.add_argument("--enable-network", action="store_true", help="Explicitly enable Drudge network tools")
    parser.add_argument("--replace", action="store_true", help="Replace the existing browser server entry")
    parser.add_argument("--probe", action="store_true", help="Check MCP handshake and tool discovery before saving; does not open a page")
    return parser.parse_args(argv)


async def probe_server(server: dict, workspace: Path) -> int:
    provider = MCPServerProvider("browser", server, workspace)
    try:
        await provider.start()
        count = len(provider.tool_names())
        if not count:
            raise ValueError("Browser MCP returned no tools")
        if not provider.owns("mcp__browser__navigate_page"):
            raise ValueError("Browser MCP is missing navigate_page; check the upstream version")
        return count
    finally:
        await provider.close()


def configure(args: argparse.Namespace) -> Path:
    path = Path(args.config).expanduser().absolute()
    # Reject credential paths and symlink aliases before any file content is read.
    if path.suffix.lower() not in (".yaml", ".yml") or path.resolve().suffix.lower() not in (".yaml", ".yml") or path.is_symlink():
        raise ValueError("--config must be a regular YAML file, not a symlink")
    original = path.read_bytes() if path.exists() else None
    try:
        config = yaml.safe_load(original.decode("utf-8-sig")) if original else {}
    except (UnicodeError, yaml.YAMLError) as exc:
        raise ValueError("Invalid UTF-8 YAML configuration") from exc
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    for key in ("security", "mcp_servers", "tool_selection"):
        if key not in config:
            config[key] = {}
        if not isinstance(config[key], dict):
            raise ValueError(f"Configuration {key} must be a mapping")
    entry = Path(args.server_path).expanduser().resolve()
    if not entry.is_file() or entry.suffix != ".js":
        raise ValueError("--server-path must point to a built .js entry; run npm ci and npm run build first")
    existing = config["mcp_servers"].get("browser")
    if existing is not None and not args.replace:
        raise ValueError("A browser server is already configured; review it and use --replace to update")
    workspace = Path(config["security"].get("workspace_root") or Path.cwd()).expanduser().resolve()
    server = {
        "preset": "jsreverser", "enabled": True, "server_path": str(entry),
        "profile": args.profile, "headless": not args.headed,
    }
    if args.browser_url:
        server["browser_url"] = args.browser_url
    if args.executable_path:
        executable = Path(args.executable_path).expanduser().resolve()
        if not executable.is_file():
            raise ValueError("--executable-path does not exist")
        server["executable_path"] = str(executable)
    expand_mcp_preset(server, workspace)  # Validate before updating any configuration.
    config["mcp_servers"]["browser"] = server
    if args.enable_network:
        config["security"]["allow_network"] = True
    always = config["tool_selection"].setdefault("always_include", [])
    if not isinstance(always, list) or any(not isinstance(name, str) for name in always):
        raise ValueError("tool_selection.always_include must be a list of tool names")
    for name in ("mcp__browser__list_pages", "mcp__browser__navigate_page"):
        if name not in always:
            always.append(name)
    if args.probe:
        count = asyncio.run(probe_server(server, workspace))
        print(f"MCP handshake passed: {count} tools (browser page not opened)")
    atomic_write(path, yaml.safe_dump(config, allow_unicode=True, sort_keys=False).encode("utf-8"), expected=original)
    return path


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        path = configure(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Browser setup failed: {exc}", file=sys.stderr)
        return 1
    print(f"Browser MCP configured in {path}")
    print("Launch Drudge with this config; use /mcp and /tools to inspect availability.")
    print("Browser calls require security.allow_network=true and follow the host approval policy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
