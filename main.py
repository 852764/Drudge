"""Hermes Lite — CLI 入口"""

import sys
import asyncio
import argparse

from agent import Agent
from config import get_config


VERSION = "0.1.0"


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
        "--version", "-V",
        action="store_true",
        help="Show version",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    return parser.parse_args()


async def run_query(
    query: str,
    config_path: str | None = None,
    toolsets: list[str] | None = None,
    model: str | None = None,
) -> None:
    """执行单次查询"""
    config = get_config(config_path)
    agent = Agent(config)

    if toolsets:
        config._config["toolsets"] = toolsets
    if model:
        config._config["model"]["name"] = model

    print(f"Drudge v{VERSION}")
    print(f"Model: {config.get('model', 'name')}")
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


def run_interactive(config_path: str | None = None, model: str | None = None) -> None:
    """交互式对话模式"""
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.styles import Style
    except ImportError:
        print("prompt_toolkit not installed. Install with: pip install prompt-toolkit")
        _run_simple_interactive(config_path, model)
        return

    config = get_config(config_path)
    if model:
        config._config["model"]["name"] = model
    agent = Agent(config)

    style = Style.from_dict({
        "prompt": "ansicyan bold",
    })

    print(f"Drudge v{VERSION}")
    print(f"Model: {config.get('model', 'name')} | Toolsets: {', '.join(config.get_toolsets())}")
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
                _handle_command(user_input, config)
                continue

            response = asyncio.run(agent.run(user_input))
            print(response)

            usage = agent.get_token_usage()
            if config.get("display", "show_cost"):
                print(f"Tokens: {usage['total_tokens']} | Turns: {usage['turns']}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break


def _run_simple_interactive(config_path: str | None = None, model: str | None = None) -> None:
    """简单交互模式（不依赖 prompt_toolkit）"""
    config = get_config(config_path)
    if model:
        config._config["model"]["name"] = model
    agent = Agent(config)

    print(f"Drudge v{VERSION}")
    print(f"Model: {config.get('model', 'name')}")
    print("Type /quit to exit")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n> ")

            if not user_input.strip():
                continue

            if user_input.startswith("/"):
                _handle_command(user_input, config)
                continue

            response = asyncio.run(agent.run(user_input))
            print(response)

            usage = agent.get_token_usage()
            print(f"\nTokens: {usage['total_tokens']} | Turns: {usage['turns']}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break


def _handle_command(cmd: str, config) -> None:
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
        print(yaml.dump(config._config, default_flow_style=False, allow_unicode=True))
    elif command == "/clear":
        import os
        os.system("cls" if os.name == "nt" else "clear")
    else:
        print(f"Unknown command: {command}. Type /help for available commands.")


def main():
    args = parse_args()

    if args.version:
        print(f"Drudge v{VERSION}")
        return

    # 工具集覆盖
    toolsets = None
    if args.toolsets:
        toolsets = [t.strip() for t in args.toolsets.split(",")]

    if args.query:
        # 单次查询
        asyncio.run(run_query(args.query, args.config, toolsets, args.model))
    else:
        # 交互模式
        run_interactive(args.config, args.model)


if __name__ == "__main__":
    main()
