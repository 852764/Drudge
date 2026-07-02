"""Drudge Lite — CLI 入口"""

import sys
import asyncio
import argparse
import json
import os
import subprocess
from pathlib import Path

from agent import Agent, AgentRuntime
from agent.llm import create_client
from config import ConfigManager, get_config
from tools import ApprovalDecision, ApprovalRequest


VERSION = "0.1.0"


class ConsoleApproval:
    """Interactive approval callback for approval_mode=on_request."""

    async def __call__(self, request: ApprovalRequest) -> ApprovalDecision:
        if not sys.stdin.isatty():
            print(
                f"\n[approval denied] {request.tool_name}: no interactive terminal",
                file=sys.stderr,
            )
            return ApprovalDecision.DENY
        print(f"\n[approval] risk={request.risk.level.value} tool={request.tool_name}")
        print(f"Action: {request.risk.action}")
        answer = await asyncio.to_thread(
            input,
            "Allow? [y] once / [a] this tool+risk for session / [N] deny: ",
        )
        normalized = answer.strip().lower()
        if normalized in ("y", "yes"):
            return ApprovalDecision.ALLOW_ONCE
        if normalized in ("a", "always"):
            return ApprovalDecision.ALLOW_SESSION
        return ApprovalDecision.DENY


class _StreamPrinter:
    def __init__(self) -> None:
        self.seen = False
        self.parts: list[str] = []

    def __call__(self, delta: str) -> None:
        self.seen = True
        self.parts.append(delta)
        print(delta, end="", flush=True)


async def _run_and_print(agent: Agent, query: str) -> str:
    printer = _StreamPrinter()
    response = await agent.run(query, stream_callback=printer)
    if printer.seen:
        print()
        streamed = "".join(printer.parts).rstrip()
        if response.strip() and not streamed.endswith(response.rstrip()):
            print(response)
    else:
        print(response)
    return response


async def _run_runtime_and_print(runtime: AgentRuntime, query: str) -> str:
    printer = _StreamPrinter()
    response = await runtime.run_turn(query, stream_callback=printer)
    if printer.seen:
        print()
        streamed = "".join(printer.parts).rstrip()
        if response.strip() and not streamed.endswith(response.rstrip()):
            print(response)
    else:
        print(response)
    return response


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
        "--approval-mode",
        choices=("auto", "on_request", "never"),
        help="Tool approval policy override",
    )
    parser.add_argument(
        "--resume",
        metavar="SESSION_ID",
        help="Resume a saved SQLite conversation session",
    )
    parser.add_argument(
        "--skill",
        action="append",
        default=[],
        metavar="NAME",
        help="Activate a local skill (repeatable)",
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
    status_parser = subparsers.add_parser("status", help="Show session status and Codex limits")
    status_parser.add_argument(
        "--json",
        action="store_true",
        dest="status_json",
        help="Print machine-readable status JSON",
    )
    return parser.parse_args()


def _configure_agent_extensions(
    agent: Agent,
    resume_id: str | None,
    skill_names: list[str] | None,
) -> None:
    if resume_id:
        session = agent.resume_session(resume_id)
        print(
            f"Resumed session {session['id']}: {session['message_count']} messages, "
            f"{session['repaired_tool_calls']} repaired tool calls"
        )
    for name in skill_names or []:
        skill = agent.activate_skill(name)
        print(f"Activated skill: {skill.name} ({skill.description})")


async def run_query(
    query: str,
    config_path: str | None = None,
    toolsets: list[str] | None = None,
    model: str | None = None,
    no_tools: bool = False,
    codex_config_path: str | None = None,
    codex_oauth: bool = False,
    approval_mode: str | None = None,
    resume_id: str | None = None,
    skill_names: list[str] | None = None,
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
    if approval_mode:
        config.override("security", "approval_mode", value=approval_mode)
    _validate_runtime_config(config)
    agent = Agent(config, approval_callback=ConsoleApproval())

    print(f"Drudge v{VERSION}")
    print(f"Model: {config.get('model', 'name')}")
    if config.codex_config_path:
        print(f"Codex config: {config.codex_config_path}")
    print(f"Toolsets: {', '.join(config.get_toolsets())}")
    print("-" * 60)

    try:
        _configure_agent_extensions(agent, resume_id, skill_names)
        print()
        await _run_and_print(agent, query)
    except asyncio.CancelledError:
        print("\nCancelled.", file=sys.stderr)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

    usage = agent.get_token_usage()
    if config.get("display", "show_cost"):
        print(
            f"\n--- Tokens: {usage['total_tokens']} | Utility: {usage['utility_tokens']} "
            f"| Turns: {usage['turns']} ---"
        )
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


async def show_status(config, agent: Agent, *, as_json: bool = False) -> dict:
    import json

    result = {"local": agent.get_status(), "account_usage": None, "account_usage_error": None}
    if config.get("model", "auth_mode") == "codex_oauth":
        from agent.codex_usage import CodexUsageClient

        try:
            result["account_usage"] = await CodexUsageClient().fetch()
        except Exception as exc:
            result["account_usage_error"] = str(exc)

    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    local = result["local"]
    print("Drudge status")
    print(f"Session: {local['session_id'] or '(new)'}")
    print(f"Run status: {local['run_status']}")
    print(f"Runtime: {'started' if local['runtime_started'] else 'scoped/not started'}")
    print(f"Model: {local['model']} ({local['provider']}, {local['model_api']})")
    utility_suffix = "configured" if local["utility_model_configured"] else "primary reused"
    print(f"Utility model: {local['utility_model']} ({utility_suffix})")
    print(f"Turns: {local['turns']} | Messages: {local['message_count']}")
    print(
        f"Tokens this process: {local['tokens_this_process']} "
        f"(utility: {local['utility_tokens_this_process']})"
    )
    if local["context_limit"]:
        used = local["context_used_percent"] or 0.0
        print(
            f"Context: ~{local['estimated_context_tokens']}/{local['context_limit']} "
            f"tokens ({max(0.0, 100.0 - used):.1f}% left)"
        )
    print(f"Workspace: {local['workspace']}")
    print(f"Approval mode: {local['approval_mode']}")
    print(f"Active skills: {', '.join(local['active_skills']) or '(none)'}")
    print(f"Open tasks: {local['open_tasks']}")
    print(f"MCP servers: {', '.join(local['mcp_servers']) or '(none)'}")
    selection = local.get("last_tool_selection")
    if selection:
        print(
            f"Tool selection: {selection['mode']} "
            f"({len(selection['selected'])}/{selection['catalog_tools']} tools, "
            f"~{selection['schema_tokens']} schema tokens)"
        )
    if result["account_usage"]:
        from agent.codex_usage import format_codex_usage

        print("\nCodex account usage")
        for line in format_codex_usage(result["account_usage"]):
            print(line)
    elif result["account_usage_error"]:
        print(f"\nCodex account usage unavailable: {result['account_usage_error']}")
    else:
        print("\nAccount limits: unavailable for the current non-Codex provider")
    return result


async def run_status(
    config_path: str | None = None,
    model: str | None = None,
    no_tools: bool = False,
    codex_config_path: str | None = None,
    codex_oauth: bool = False,
    approval_mode: str | None = None,
    resume_id: str | None = None,
    skill_names: list[str] | None = None,
    as_json: bool = False,
) -> None:
    config = get_config(config_path, codex_config_path)
    if model:
        config.override("model", "name", value=model)
    if no_tools:
        config.override("toolsets", value=[])
    if codex_oauth:
        config.enable_codex_oauth()
    if approval_mode:
        config.override("security", "approval_mode", value=approval_mode)
    if config.get("model", "auth_mode") == "codex_oauth":
        _validate_runtime_config(config)
    agent = Agent(config, approval_callback=ConsoleApproval())
    if resume_id:
        agent.resume_session(resume_id)
    for name in skill_names or []:
        agent.activate_skill(name)
    await show_status(config, agent, as_json=as_json)


def run_interactive(
    config_path: str | None = None,
    model: str | None = None,
    no_tools: bool = False,
    codex_config_path: str | None = None,
    codex_oauth: bool = False,
    approval_mode: str | None = None,
    resume_id: str | None = None,
    skill_names: list[str] | None = None,
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
            approval_mode,
            resume_id,
            skill_names,
        )
        return

    config = get_config(config_path, codex_config_path)
    if model:
        config.override("model", "name", value=model)
    if no_tools:
        config.override("toolsets", value=[])
    if codex_oauth:
        config.enable_codex_oauth()
    if approval_mode:
        config.override("security", "approval_mode", value=approval_mode)
    _validate_runtime_config(config)
    agent = Agent(config, approval_callback=ConsoleApproval())
    try:
        _configure_agent_extensions(agent, resume_id, skill_names)
    except (KeyError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return

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

    asyncio.run(_interactive_loop(config, agent, session=session))


def _run_simple_interactive(
    config_path: str | None = None,
    model: str | None = None,
    no_tools: bool = False,
    codex_config_path: str | None = None,
    codex_oauth: bool = False,
    approval_mode: str | None = None,
    resume_id: str | None = None,
    skill_names: list[str] | None = None,
) -> None:
    """简单交互模式（不依赖 prompt_toolkit）"""
    config = get_config(config_path, codex_config_path)
    if model:
        config.override("model", "name", value=model)
    if no_tools:
        config.override("toolsets", value=[])
    if codex_oauth:
        config.enable_codex_oauth()
    if approval_mode:
        config.override("security", "approval_mode", value=approval_mode)
    _validate_runtime_config(config)
    agent = Agent(config, approval_callback=ConsoleApproval())
    try:
        _configure_agent_extensions(agent, resume_id, skill_names)
    except (KeyError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return

    print(f"Drudge v{VERSION}")
    print(f"Model: {config.get('model', 'name')}")
    if config.codex_config_path:
        print(f"Codex config: {config.codex_config_path}")
    print("Type /quit to exit")
    print("-" * 60)

    asyncio.run(_interactive_loop(config, agent, session=None))


async def _interactive_loop(config, agent: Agent, *, session=None) -> None:
    runtime = AgentRuntime(agent)
    async with runtime:
        while True:
            try:
                if session is not None:
                    user_input = await session.prompt_async([("class:prompt", "\n> ")])
                else:
                    user_input = await asyncio.to_thread(input, "\n> ")
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

            if not user_input.strip():
                continue
            if user_input.startswith("/"):
                if await _handle_command(user_input, config, agent):
                    break
                continue

            try:
                await _run_runtime_and_print(runtime, user_input)
            except KeyboardInterrupt:
                runtime.cancel()
                print("\nCancelled. Ready for the next prompt.")
                continue

            usage = agent.get_token_usage()
            if config.get("display", "show_cost"):
                print(
                    f"Tokens: {usage['total_tokens']} | Utility: {usage['utility_tokens']} "
                    f"| Turns: {usage['turns']}"
                )
            if agent.session_id:
                print(f"Session: {agent.session_id}")


async def _handle_command(cmd: str, config, agent: Agent | None = None) -> bool:
    """处理 slash 命令"""
    parts = cmd.strip().split()
    command = parts[0].lower()

    if command in ("/quit", "/exit", "/q"):
        print("Goodbye!")
        return True
    elif command == "/help":
        print("Commands:")
        print("  /quit, /exit, /q    Exit Drudge")
        print("  /help               Show this help")
        print("  /tools              List available tools")
        print("  /mcp                Inspect configured MCP stdio servers")
        print("  /config             Show current config")
        print("  /models             List provider models")
        print("  /sessions           List saved sessions")
        print("  /history [id]       Show saved messages")
        print("  /runs               List recent runs")
        print("  /trace [run_id]     Show a persisted run trace")
        print("  /tasks [all]        List persistent session tasks")
        print("  /task add <title>   Create a persistent task")
        print("  /task start|done|cancel|reopen <id>")
        print("  /status             Show session, context, and account limits")
        print("  /compact            Compact older conversation context")
        print("  /resume <id>        Resume a saved session")
        print("  /new                Start a new session")
        print("  /skills             List discovered skills")
        print("  /skill <name>       Activate a skill")
        print("  /skill off <name>   Deactivate a skill")
        print("  /skill show <name>  Show skill metadata")
        print("  /skill clear        Deactivate all skills")
        print("  /clear              Clear screen")
    elif command == "/tools":
        names = await agent.list_available_tools() if agent else []
        print("Available tools:")
        for name in sorted(names):
            print(f"  - {name}")
    elif command == "/mcp":
        if agent is None:
            print("Agent is unavailable.")
        else:
            _show_mcp(await agent.inspect_mcp())
    elif command == "/config":
        import yaml
        print(yaml.dump(config.as_safe_dict(), default_flow_style=False, allow_unicode=True))
    elif command == "/models":
        await show_models_from_config(config)
    elif command == "/sessions":
        _show_sessions(config)
    elif command == "/history":
        session_id = parts[1] if len(parts) > 1 else getattr(agent, "session_id", None)
        if not session_id:
            print("No active session yet. Run a prompt first or pass /history <session_id>.")
        else:
            _show_history(config, session_id)
    elif command == "/runs":
        _show_runs(agent)
    elif command == "/trace":
        _show_trace(agent, parts[1] if len(parts) > 1 else None)
    elif command == "/tasks":
        _show_tasks(agent, include_closed=len(parts) > 1 and parts[1].lower() == "all")
    elif command == "/task":
        _handle_task_command(parts[1:], agent)
    elif command in ("/status", "/usage"):
        if agent is None:
            print("Agent is unavailable.")
        else:
            await show_status(config, agent)
    elif command == "/compact":
        if agent is None:
            print("Agent is unavailable.")
        else:
            result = await agent.compact_context()
            print(
                f"Context compacted: {result['before_messages']} -> {result['after_messages']} messages, "
                f"~{result['before_tokens']} -> ~{result['after_tokens']} tokens "
                f"(mode={result['mode']}, model={result.get('summary_model') or 'deterministic'}, "
                f"summary_tokens={result['summary_tokens']})"
            )
            if result.get("fallback_reason"):
                print(f"LLM summary failed; used deterministic fallback: {result['fallback_reason']}")
    elif command == "/resume":
        if agent is None:
            print("Agent is unavailable.")
        elif len(parts) < 2:
            print("Usage: /resume <session_id>")
        else:
            try:
                session = agent.resume_session(parts[1])
                print(
                    f"Resumed {session['id']}: {session['title']} "
                    f"({session['message_count']} messages, "
                    f"{session['repaired_tool_calls']} repaired tool calls)"
                )
                if session["active_skills"]:
                    print(f"Active skills: {', '.join(session['active_skills'])}")
            except (KeyError, RuntimeError) as exc:
                print(str(exc))
    elif command == "/new":
        if agent is not None:
            agent.new_session()
        print("Started a new session. Active skills were kept.")
    elif command == "/skills":
        _show_skills(agent)
    elif command == "/skill":
        _handle_skill_command(parts[1:], agent)
    elif command == "/clear":
        import os
        os.system("cls" if os.name == "nt" else "clear")
    else:
        print(f"Unknown command: {command}. Type /help for available commands.")
    return False


def _show_skills(agent: Agent | None) -> None:
    if agent is None:
        print("Agent is unavailable.")
        return
    skills = agent.list_skills()
    if not skills:
        print("No skills found under .drudge/skills/*/SKILL.md")
        return
    for skill in skills:
        marker = "*" if skill["active"] else " "
        print(f"[{marker}] {skill['name']}: {skill['description']}")


def _show_mcp(status: dict) -> None:
    providers = status.get("providers") or []
    if not providers:
        print("No MCP servers configured.")
        return
    for item in providers:
        state = "connected" if item.get("connected") else "unavailable"
        print(f"{item['name']}: {state}")
        if item.get("error"):
            print(f"  error: {item['error']}")
        for tool in item.get("tools") or []:
            print(f"  - {tool}")


def _show_runs(agent: Agent | None) -> None:
    if agent is None:
        print("Agent is unavailable.")
        return
    try:
        runs = agent.list_runs()
    except RuntimeError as exc:
        print(str(exc))
        return
    if not runs:
        print("No persisted runs for this session.")
        return
    for run in runs:
        print(
            f"{run['id']}  {run['status']:<10}  {run['model']}  "
            f"{run['started_at']}  {run['prompt'][:60]}"
        )


def _show_trace(agent: Agent | None, run_id: str | None) -> None:
    if agent is None:
        print("Agent is unavailable.")
        return
    try:
        trace = agent.get_trace(run_id)
    except RuntimeError as exc:
        print(str(exc))
        return
    if trace is None:
        print("No persisted trace found.")
        return
    print(
        f"Run {trace['id']} | {trace['status']} | model={trace['model']} | "
        f"started={trace['started_at']}"
    )
    if trace.get("error"):
        print(f"Error: {trace['error']}")
    print("Model calls:")
    for call in trace.get("model_calls") or []:
        print(
            f"  turn={call['turn']} purpose={call['purpose']} model={call['model']} "
            f"status={call['status']} tokens={call['total_tokens']} latency={call['latency_ms']}ms"
        )
    print("Events:")
    for event in trace.get("events") or []:
        detail = json.dumps(event["detail"], ensure_ascii=False)
        print(f"  turn={event['turn']} {event['kind']}: {detail[:500]}")


def _show_tasks(agent: Agent | None, *, include_closed: bool = False) -> None:
    if agent is None:
        print("Agent is unavailable.")
        return
    try:
        tasks = agent.list_tasks(include_closed=include_closed)
    except RuntimeError as exc:
        print(str(exc))
        return
    if not tasks:
        print("No tasks for the active session.")
        return
    for task in tasks:
        print(f"#{task['id']} [{task['status']}] {task['title']}")
        if task.get("details"):
            print(f"  {task['details']}")


def _handle_task_command(arguments: list[str], agent: Agent | None) -> None:
    if agent is None:
        print("Agent is unavailable.")
        return
    if not arguments:
        print("Usage: /task add <title> | start|done|cancel|reopen <id>")
        return
    action = arguments[0].lower()
    try:
        if action == "add":
            title = " ".join(arguments[1:]).strip()
            task = agent.create_task(title)
            print(f"Created task #{task['id']}: {task['title']}")
            return
        status_map = {
            "start": "in_progress",
            "done": "completed",
            "cancel": "cancelled",
            "reopen": "pending",
        }
        if action not in status_map or len(arguments) != 2:
            print("Usage: /task add <title> | start|done|cancel|reopen <id>")
            return
        task = agent.update_task(int(arguments[1]), status_map[action])
        print(f"Task #{task['id']} -> {task['status']}: {task['title']}")
    except (KeyError, RuntimeError, ValueError) as exc:
        print(str(exc))


def _handle_skill_command(arguments: list[str], agent: Agent | None) -> None:
    if agent is None:
        print("Agent is unavailable.")
        return
    if not arguments:
        print("Usage: /skill <name> | off <name> | show <name> | clear")
        return
    action = arguments[0].lower()
    try:
        if action == "clear":
            agent.clear_skills()
            print("All skills deactivated.")
        elif action == "off":
            if len(arguments) < 2:
                print("Usage: /skill off <name>")
            elif agent.deactivate_skill(arguments[1]):
                print(f"Deactivated skill: {arguments[1]}")
            else:
                print(f"Skill is not active: {arguments[1]}")
        elif action == "show":
            if len(arguments) < 2:
                print("Usage: /skill show <name>")
            else:
                skill = agent.get_skill(arguments[1])
                print(f"Name: {skill.name}")
                print(f"Description: {skill.description}")
                print(f"Path: {skill.path}")
        else:
            skill = agent.activate_skill(arguments[0])
            print(f"Activated skill: {skill.name} ({skill.description})")
    except KeyError as exc:
        print(str(exc))


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
    approval_mode: str | None = None,
) -> None:
    config = get_config(config_path, codex_config_path)
    if model:
        config.override("model", "name", value=model)
    if no_tools:
        config.override("toolsets", value=[])
    if codex_oauth:
        config.enable_codex_oauth()
    if approval_mode:
        config.override("security", "approval_mode", value=approval_mode)

    print(f"Drudge v{VERSION} doctor")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"CWD: {os.getcwd()}")
    print(f"Model: {config.get('model', 'name')}")
    if config.has_utility_model():
        print(f"Utility model: {config.get_utility_model_config().get('name')}")
    print(f"Provider: {config.get('model', 'provider', default='openai-compatible')}")
    print(f"Base URL: {config.get('model', 'base_url')}")
    print(f"Model API: {config.get('model', 'api', default='auto')}")
    print(f"Toolsets: {', '.join(config.get_toolsets()) or '(none)'}")
    mcp_servers = config.get("mcp_servers", default={}) or {}
    print(f"MCP servers: {', '.join(sorted(mcp_servers)) or '(none)'}")
    selection = config.get("tool_selection", default={}) or {}
    print(
        f"Tool selection: enabled={selection.get('enabled', True)}, "
        f"min_tools={selection.get('min_tools', 16)}, "
        f"min_schema_tokens={selection.get('min_schema_tokens', 3000)}"
    )

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

    from agent.project_instructions import load_project_instructions
    from agent.skills import SkillManager

    instruction_files = load_project_instructions(
        workspace,
        cwd=Path.cwd(),
        filename=str(config.get("agent", "instructions_filename", default="AGENTS.md")),
        max_chars=int(config.get("agent", "instructions_max_chars", default=64_000)),
    ) if config.get("agent", "instructions_enabled", default=True) else []
    print(f"AGENTS.md files: {len(instruction_files)}")
    for item in instruction_files:
        print(f"  - {item.path}")
    skills = SkillManager(
        workspace,
        max_chars=int(config.get("agent", "skill_max_chars", default=32_000)),
    ).discover()
    print(f"Skills discovered: {len(skills)}")
    for skill in skills.values():
        print(f"  - {skill.name}: {skill.description}")

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
            args.approval_mode,
        )
        return
    if args.command == "status":
        asyncio.run(run_status(
            args.config,
            args.model,
            args.no_tools,
            args.codex_config,
            args.codex_oauth,
            args.approval_mode,
            args.resume,
            args.skill,
            getattr(args, "status_json", False),
        ))
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
        try:
            asyncio.run(run_query(
                args.query,
                config_path=args.config,
                codex_config_path=args.codex_config,
                toolsets=toolsets,
                model=args.model,
                no_tools=args.no_tools,
                codex_oauth=args.codex_oauth,
                approval_mode=args.approval_mode,
                resume_id=args.resume,
                skill_names=args.skill,
            ))
        except KeyboardInterrupt:
            print("\nCancelled.", file=sys.stderr)
    else:
        # 交互模式
        run_interactive(
            args.config,
            args.model,
            args.no_tools,
            args.codex_config,
            args.codex_oauth,
            args.approval_mode,
            args.resume,
            args.skill,
        )


if __name__ == "__main__":
    main()
