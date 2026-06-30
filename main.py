"""Drudge Lite — CLI 入口"""

import sys
import asyncio
import argparse
import os
import subprocess
from pathlib import Path

from agent import Agent
from agent.llm import create_client
from config import ConfigManager, get_config


VERSION = "0.1.0"


def _validate_runtime_config(config) -> None:
    if config.get("model", "auth_mode") == "codex_oauth":
        from agent.codex_auth import auth_status

        if not auth_status().get("authenticated"):
            print(
                "Codex OAuth is not configured. Run: drudge auth login",
                file=sys.stderr,
            )
            sys.exit(1)
        return
    headers = config.get("model", "headers", default={}) or {}
    auth_header_names = {"authorization", "proxy-authorization", "x-api-key"}
    has_auth_header = any(str(name).lower() in auth_header_names for name in headers)
    if (
        not config.get("model", "api_key")
        and not config.get("model", "allow_unauthenticated", default=False)
        and not has_auth_header
    ):
        env_name = config.get("model", "api_key_env", default="OPENAI_API_KEY or DRUDGE_API_KEY")
        print(
            f"Missing API key. Set {env_name}, or provide model.api_key in your config file.",
            file=sys.stderr,
        )
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="drudge",
        description="A lightweight terminal AI agent",
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        help="Single query, non-interactive mode",
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        help="Path to config file",
    )
    parser.add_argument(
        "--codex-config",
        nargs="?",
        const=str(ConfigManager.default_codex_config_path()),
        metavar="PATH",
        help="Load model/provider settings from Codex config.toml (default: ~/.codex/config.toml)",
    )
    parser.add_argument(
        "--codex-oauth",
        action="store_true",
        help="Use experimental direct ChatGPT Codex OAuth provider",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        help="Model name override",
    )
    parser.add_argument(
        "-t", "--toolsets",
        type=str,
        help="Comma-separated toolsets (default: terminal,file,web)",
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Disable tool schemas for this run",
    )
    parser.add_argument(
        "--version", "-V",
        action="store_true",
        help="Show version",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--models",
        action="store_true",
        help="List models from the configured provider",
    )
    subparsers = parser.add_subparsers(dest="command")
    auth_parser = subparsers.add_parser("auth", help="Manage Drudge Codex OAuth credentials")
    auth_parser.add_argument("action", choices=("login", "status", "logout"))
    auth_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Print the login URL without opening a browser",
    )
    subparsers.add_parser("doctor", help="Check Drudge configuration, auth, tools, and workspace state")
    return parser.parse_args()


async def run_query(
    query: str,
    config_path: str | None = None,
    toolsets: list[str] | None = None,
    model: str | None = None,
    no_tools: bool = False,
    codex_config_path: str | None = None,
    codex_oauth: bool = False,
) -> None:
    """执行单次查询"""
    config = get_config(config_path, codex_config_path)

    if toolsets:
        config.override("toolsets", value=toolsets)
    if no_tools:
        config.override("toolsets", value=[])
    if model:
        config.override("model", "name", value=model)
    if codex_oauth:
        config.enable_codex_oauth()
    _validate_runtime_config(config)
    agent = Agent(config)

    print(f"Drudge v{VERSION}")
    print(f"Model: {config.get('model', 'name')}")
    if config.codex_config_path:
        print(f"Codex config: {config.codex_config_path}")
    print(f"Toolsets: {', '.join(config.get_toolsets())}")
    print("-" * 60)

    try:
        response = await agent.run(query)
        print("\n" + response)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

    usage = agent.get_token_usage()
    if config.get("display", "show_cost"):
        print(f"\n--- Tokens: {usage['total_tokens']} | Turns: {usage['turns']} ---")
    if agent.session_id:
        print(f"Session: {agent.session_id}")


async def show_models(
    config_path: str | None = None,
    model: str | None = None,
    codex_config_path: str | None = None,
    codex_oauth: bool = False,
) -> None:
    config = get_config(config_path, codex_config_path)
    if codex_oauth:
        raise RuntimeError("Model listing is not supported by the Codex OAuth provider")
    if model:
        config.override("model", "name", value=model)
    _validate_runtime_config(config)
    client = create_client(config.get_model_config())
    print(f"Base URL: {config.get('model', 'base_url')}")
    print(f"Model API: {config.get('model', 'api', default='auto')}")
    try:
        models = await client.list_models()
    except Exception as e:
        print(f"Failed to list models: {e}", file=sys.stderr)
        sys.exit(1)
    if not models:
        print("No models returned by provider.")
        return
    for item in models:
        print(item)


def run_interactive(
    config_path: str | None = None,
    model: str | None = None,
    no_tools: bool = False,
    codex_config_path: str | None = None,
    codex_oauth: bool = False,
) -> None:
    """交互式对话模式"""
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.styles import Style
    except ImportError:
        print("prompt_toolkit not installed. Install with: pip install prompt-toolkit")
        _run_simple_interactive(
            config_path,
            model,
            no_tools,
            codex_config_path,
            codex_oauth,
        )
        return

    config = get_config(config_path, codex_config_path)
    if model:
        config.override("model", "name", value=model)
    if no_tools:
        config.override("toolsets", value=[])
    if codex_oauth:
        config.enable_codex_oauth()
    _validate_runtime_config(config)
    agent = Agent(config)

    style = Style.from_dict({
        "prompt": "ansicyan bold",
    })

    print(f"Drudge v{VERSION}")
    print(f"Model: {config.get('model', 'name')} | Toolsets: {', '.join(config.get_toolsets())}")
    if config.codex_config_path:
        print(f"Codex config: {config.codex_config_path}")
    print("Type /quit to exit, /help for commands, /tools to list tools")
    print("-" * 60)

    session = PromptSession(style=style)

    while True:
        try:
            user_input = session.prompt([("class:prompt", "\n> ")])

            if not user_input.strip():
                continue

            # 处理 slash 命令
            if user_input.startswith("/"):
                _handle_command(user_input, config, agent)
                continue

            response = asyncio.run(agent.run(user_input))
            print(response)

            usage = agent.get_token_usage()
            if config.get("display", "show_cost"):
                print(f"Tokens: {usage['total_tokens']} | Turns: {usage['turns']}")
            if agent.session_id:
                print(f"Session: {agent.session_id}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break


def _run_simple_interactive(
    config_path: str | None = None,
    model: str | None = None,
    no_tools: bool = False,
    codex_config_path: str | None = None,
    codex_oauth: bool = False,
) -> None:
    """简单交互模式（不依赖 prompt_toolkit）"""
    config = get_config(config_path, codex_config_path)
    if model:
        config.override("model", "name", value=model)
    if no_tools:
        config.override("toolsets", value=[])
    if codex_oauth:
        config.enable_codex_oauth()
    _validate_runtime_config(config)
    agent = Agent(config)

    print(f"Drudge v{VERSION}")
    print(f"Model: {config.get('model', 'name')}")
    if config.codex_config_path:
        print(f"Codex config: {config.codex_config_path}")
    print("Type /quit to exit")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n> ")

            if not user_input.strip():
                continue

            if user_input.startswith("/"):
                _handle_command(user_input, config, agent)
                continue

            response = asyncio.run(agent.run(user_input))
            print(response)

            usage = agent.get_token_usage()
            print(f"\nTokens: {usage['total_tokens']} | Turns: {usage['turns']}")
            if agent.session_id:
                print(f"Session: {agent.session_id}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break


def _handle_command(cmd: str, config, agent: Agent | None = None) -> None:
    """处理 slash 命令"""
    parts = cmd.strip().split()
    command = parts[0].lower()

    if command in ("/quit", "/exit", "/q"):
        print("Goodbye!")
        sys.exit(0)
    elif command == "/help":
        print("Commands:")
        print("  /quit, /exit, /q    Exit Drudge")
        print("  /help               Show this help")
        print("  /tools              List available tools")
        print("  /config             Show current config")
        print("  /models             List provider models")
        print("  /sessions           List saved sessions")
        print("  /history [id]       Show saved messages")
        print("  /clear              Clear screen")
    elif command == "/tools":
        from tools import registry
        toolsets = config.get_toolsets()
        names = registry.list_tools(toolsets)
        print(f"Available tools ({', '.join(toolsets)}):")
        for name in sorted(names):
            print(f"  - {name}")
    elif command == "/config":
        import yaml
        print(yaml.dump(config.as_safe_dict(), default_flow_style=False, allow_unicode=True))
    elif command == "/models":
        asyncio.run(show_models_from_config(config))
    elif command == "/sessions":
        _show_sessions(config)
    elif command == "/history":
        session_id = parts[1] if len(parts) > 1 else getattr(agent, "session_id", None)
        if not session_id:
            print("No active session yet. Run a prompt first or pass /history <session_id>.")
        else:
            _show_history(config, session_id)
    elif command == "/clear":
        import os
        os.system("cls" if os.name == "nt" else "clear")
    else:
        print(f"Unknown command: {command}. Type /help for available commands.")


def _get_store(config):
    from agent.storage import ConversationStore

    storage_config = config.get_storage_config()
    if not storage_config.get("enabled", True):
        print("Conversation storage is disabled.")
        return None
    return ConversationStore(storage_config.get("path", "~/.drudge/drudge.db"))


async def show_models_from_config(config) -> None:
    _validate_runtime_config(config)
    client = create_client(config.get_model_config())
    print(f"Base URL: {config.get('model', 'base_url')}")
    print(f"Model API: {config.get('model', 'api', default='auto')}")
    try:
        models = await client.list_models()
    except Exception as e:
        print(f"Failed to list models: {e}", file=sys.stderr)
        return
    if not models:
        print("No models returned by provider.")
        return
    for item in models:
        print(item)


def _show_sessions(config) -> None:
    store = _get_store(config)
    if not store:
        return
    sessions = store.list_sessions()
    if not sessions:
        print("No saved sessions yet.")
        return
    for item in sessions:
        print(f"{item['id']}  {item['updated_at']}  {item['model']}  {item['title']}")


def _show_history(config, session_id: str) -> None:
    store = _get_store(config)
    if not store:
        return
    messages = store.get_messages(session_id)
    if not messages:
        print(f"No messages found for session: {session_id}")
        return
    for message in messages:
        content = (message.get("content") or "").replace("\n", " ")
        if len(content) > 300:
            content = content[:300] + "..."
        print(f"[{message['created_at']}] {message['role']}: {content}")


def _handle_auth_action(action: str, no_browser: bool = False) -> None:
    from agent.codex_auth import auth_status, login_device_code, logout

    if action == "login":
        credentials = login_device_code(open_browser=not no_browser)
        print("Codex OAuth login successful.")
        print(f"Account ID present: {bool(credentials.get('account_id'))}")
        return
    if action == "status":
        status = auth_status()
        print(f"Authenticated: {status.get('authenticated', False)}")
        if status.get("authenticated"):
            print(f"Refresh token: {status.get('has_refresh_token', False)}")
            print(f"Account ID present: {status.get('account_id_present', False)}")
            print(f"Store: {status.get('path')}")
        return
    if action == "logout":
        print("Codex OAuth credentials removed." if logout() else "No Codex OAuth credentials found.")


def run_doctor(
    config_path: str | None = None,
    model: str | None = None,
    no_tools: bool = False,
    codex_config_path: str | None = None,
    codex_oauth: bool = False,
) -> None:
    config = get_config(config_path, codex_config_path)
    if model:
        config.override("model", "name", value=model)
    if no_tools:
        config.override("toolsets", value=[])
    if codex_oauth:
        config.enable_codex_oauth()

    print(f"Drudge v{VERSION} doctor")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"CWD: {os.getcwd()}")
    print(f"Model: {config.get('model', 'name')}")
    print(f"Provider: {config.get('model', 'provider', default='openai-compatible')}")
    print(f"Base URL: {config.get('model', 'base_url')}")
    print(f"Model API: {config.get('model', 'api', default='auto')}")
    print(f"Toolsets: {', '.join(config.get_toolsets()) or '(none)'}")

    security = config.get_security_config()
    workspace = Path(security.get("workspace_root") or os.getcwd()).expanduser().resolve()
    print(f"Workspace: {workspace}")
    print(f"Workspace exists: {workspace.exists()}")
    print(f"Approval mode: {security.get('approval_mode', 'auto')}")
    print(f"Terminal allowed: {security.get('allow_terminal', True)}")
    print(f"Network allowed: {security.get('allow_network', True)}")

    storage = config.get_storage_config()
    print(f"Storage enabled: {storage.get('enabled', True)}")
    if storage.get("enabled", True):
        print(f"Storage path: {Path(storage.get('path', '~/.drudge/drudge.db')).expanduser()}")

    if config.get("model", "auth_mode") == "codex_oauth":
        from agent.codex_auth import auth_status

        status = auth_status()
        print(f"Codex OAuth authenticated: {status.get('authenticated', False)}")
        print(f"Codex OAuth account id present: {status.get('account_id_present', False)}")
        print(f"Codex OAuth store: {status.get('path', '(default)')}")
    else:
        headers = config.get("model", "headers", default={}) or {}
        has_api_key = bool(config.get("model", "api_key"))
        has_auth_header = any(str(name).lower() in {"authorization", "x-api-key"} for name in headers)
        print(f"API credential present: {has_api_key or has_auth_header}")

    from tools import registry

    print("Tools:")
    for name in sorted(registry.list_tools(config.get_toolsets())):
        print(f"  - {name}")

    git_summary = _git_status_summary(workspace)
    if git_summary:
        print("Git:")
        for line in git_summary:
            print(f"  {line}")


def _git_status_summary(workspace: Path) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(workspace), "-c", f"safe.directory={workspace.as_posix()}", "status", "--short"],
            capture_output=True,
            text=True,
            timeout=5,
            shell=False,
        )
    except Exception as exc:
        return [f"unavailable: {exc}"]
    if result.returncode != 0:
        return ["not a git repository or git status failed"]
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return ["clean"] if not lines else [f"dirty files: {len(lines)}"] + lines[:10]


def main():
    args = parse_args()

    if args.command == "auth":
        _handle_auth_action(args.action, args.no_browser)
        return
    if args.command == "doctor":
        run_doctor(
            args.config,
            args.model,
            args.no_tools,
            args.codex_config,
            args.codex_oauth,
        )
        return

    if args.version:
        print(f"Drudge v{VERSION}")
        return

    # 工具集覆盖
    if args.models:
        asyncio.run(show_models(
            args.config,
            args.model,
            args.codex_config,
            args.codex_oauth,
        ))
        return

    toolsets = None
    if args.toolsets:
        toolsets = [t.strip() for t in args.toolsets.split(",")]

    if args.query:
        # 单次查询
        asyncio.run(run_query(
            args.query,
            config_path=args.config,
            codex_config_path=args.codex_config,
            toolsets=toolsets,
            model=args.model,
            no_tools=args.no_tools,
            codex_oauth=args.codex_oauth,
        ))
    else:
        # 交互模式
        run_interactive(
            args.config,
            args.model,
            args.no_tools,
            args.codex_config,
            args.codex_oauth,
        )


if __name__ == "__main__":
    main()
