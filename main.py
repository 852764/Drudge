"""Drudge Lite — CLI 入口"""

import sys
import asyncio
import argparse
import json
import os
import subprocess
from pathlib import Path

from agent import Agent, AgentRuntime
from agent.cli_renderer import CliRenderer
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


def _make_renderer(config) -> CliRenderer:
    return CliRenderer(
        pretty=bool(config.get("display", "pretty_cli", default=True)),
    )


def _attach_renderer(agent: Agent, renderer: CliRenderer) -> None:
    agent.tool_log_callback = renderer.print_tool_event


def _render_run_status(agent: Agent) -> str:
    activity = agent.get_activity_label()
    if activity:
        return activity
    state = agent.get_run_state()
    status = getattr(state.status, "value", str(state.status))
    detail = state.events[-1].detail if state.events else {}
    if status == "executing_tools":
        tool = detail.get("tool")
        return f"Running tool: {tool}" if tool else "Running tools"
    if status == "waiting_for_approval":
        tool = detail.get("tool")
        return f"Waiting approval: {tool}" if tool else "Waiting approval"
    if status == "waiting_for_model":
        return "Thinking"
    if status == "completed":
        return "Completed"
    if status == "failed":
        return "Failed"
    if status == "cancelled":
        return "Cancelled"
    return "Thinking"


async def _run_and_print(agent: Agent, query: str, renderer: CliRenderer) -> str:
    printer = renderer.make_stream_printer("Assistant")
    ticker = renderer.make_activity_ticker("Thinking", state_getter=lambda: _render_run_status(agent))
    task = asyncio.create_task(ticker.run())
    response = ""
    try:
        response = await agent.run(query, stream_callback=printer)
    finally:
        ticker.stop()
        await asyncio.gather(task, return_exceptions=True)
    printer.finish(response)
    return response


async def _run_runtime_and_print(runtime: AgentRuntime, query: str, renderer: CliRenderer) -> str:
    printer = renderer.make_stream_printer("Assistant")
    ticker = renderer.make_activity_ticker("Thinking", state_getter=lambda: _render_run_status(runtime.agent))
    task = asyncio.create_task(ticker.run())
    response = ""
    try:
        response = await runtime.run_turn(query, stream_callback=printer)
    finally:
        ticker.stop()
        await asyncio.gather(task, return_exceptions=True)
    printer.finish(response)
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
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check Drudge configuration, auth, tools, and workspace state",
    )
    doctor_parser.add_argument(
        "--probe-model",
        nargs="?",
        const="",
        metavar="MODEL",
        help="Probe Chat, Responses, tools, and streaming support for a model",
    )
    doctor_parser.add_argument(
        "--probe-json",
        action="store_true",
        help="Print provider probe results as JSON",
    )
    doctor_parser.add_argument(
        "--no-probe-streaming",
        action="store_true",
        help="Skip streaming checks during provider probing",
    )
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
    renderer: CliRenderer,
) -> None:
    if resume_id:
        session = agent.resume_session(resume_id)
        renderer.print_note(
            f"Resumed session {session['id']}: {session['message_count']} messages, "
            f"{session['repaired_tool_calls']} repaired tool calls",
            level="success",
        )
    for name in skill_names or []:
        skill = agent.activate_skill(name)
        renderer.print_note(f"Activated skill: {skill.name} ({skill.description})", level="success")


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
    renderer = _make_renderer(config)

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
    _attach_renderer(agent, renderer)

    renderer.print_banner(
        version=VERSION,
        model=config.get("model", "name"),
        toolsets=config.get_toolsets(),
        codex_config_path=config.codex_config_path,
    )

    try:
        _configure_agent_extensions(agent, resume_id, skill_names, renderer)
        await _run_and_print(agent, query, renderer)
    except asyncio.CancelledError:
        renderer.print_note("Cancelled.", level="warning", error=True)
    except Exception as e:
        renderer.print_note(f"Error: {e}", level="error", error=True)
        sys.exit(1)

    usage = agent.get_token_usage()
    if config.get("display", "show_cost"):
        renderer.print_usage(usage, session_id=agent.session_id)


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


async def show_status(config, agent: Agent, *, as_json: bool = False, renderer: CliRenderer | None = None) -> dict:
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

    (renderer or _make_renderer(config)).print_status(result)
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
    renderer = _make_renderer(config)
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
    _attach_renderer(agent, renderer)
    if resume_id:
        agent.resume_session(resume_id)
    for name in skill_names or []:
        agent.activate_skill(name)
    await show_status(config, agent, as_json=as_json, renderer=renderer)


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
    renderer = _make_renderer(config)
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
    _attach_renderer(agent, renderer)
    try:
        _configure_agent_extensions(agent, resume_id, skill_names, renderer)
    except (KeyError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return

    style = Style.from_dict({
        "prompt": "ansicyan bold",
    })

    renderer.print_banner(
        version=VERSION,
        model=config.get("model", "name"),
        toolsets=config.get_toolsets(),
        codex_config_path=config.codex_config_path,
        subtitle="Type /quit to exit, /help for commands, /tools to list tools",
    )

    session = PromptSession(style=style)

    asyncio.run(_interactive_loop(config, agent, renderer=renderer, session=session))


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
    renderer = _make_renderer(config)
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
    _attach_renderer(agent, renderer)
    try:
        _configure_agent_extensions(agent, resume_id, skill_names, renderer)
    except (KeyError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return

    renderer.print_banner(
        version=VERSION,
        model=config.get("model", "name"),
        toolsets=config.get_toolsets(),
        codex_config_path=config.codex_config_path,
        subtitle="Type /quit to exit",
    )

    asyncio.run(_interactive_loop(config, agent, renderer=renderer, session=None))


async def _interactive_loop(config, agent: Agent, *, renderer: CliRenderer, session=None) -> None:
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
                if await _handle_command(user_input, config, agent, renderer=renderer):
                    break
                continue

            try:
                await _run_runtime_and_print(runtime, user_input, renderer)
            except KeyboardInterrupt:
                runtime.cancel()
                renderer.print_note("Cancelled. Ready for the next prompt.", level="warning")
                continue

            usage = agent.get_token_usage()
            if config.get("display", "show_cost"):
                renderer.print_usage(usage, session_id=agent.session_id)


async def _handle_command(cmd: str, config, agent: Agent | None = None, *, renderer: CliRenderer | None = None) -> bool:
    """处理 slash 命令"""
    renderer = renderer or _make_renderer(config)
    parts = cmd.strip().split()
    command = parts[0].lower()

    if command in ("/quit", "/exit", "/q"):
        renderer.print_note("Goodbye!", level="info")
        return True
    elif command == "/help":
        renderer.print_help()
    elif command == "/tools":
        names = await agent.list_available_tools() if agent else []
        renderer.print_list("Available Tools", sorted(names))
    elif command == "/mcp":
        if agent is None:
            renderer.print_note("Agent is unavailable.", level="warning")
        else:
            _show_mcp(await agent.inspect_mcp(), renderer=renderer)
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
        _show_runs(agent, renderer=renderer)
    elif command == "/trace":
        _show_trace(agent, parts[1] if len(parts) > 1 else None)
    elif command == "/tasks":
        _show_tasks(agent, include_closed=len(parts) > 1 and parts[1].lower() == "all", renderer=renderer)
    elif command == "/task":
        _handle_task_command(parts[1:], agent)
    elif command in ("/status", "/usage"):
        if agent is None:
            renderer.print_note("Agent is unavailable.", level="warning")
        else:
            await show_status(config, agent, renderer=renderer)
    elif command == "/compact":
        if agent is None:
            renderer.print_note("Agent is unavailable.", level="warning")
        else:
            result = await agent.compact_context()
            renderer.print_note(
                f"Context compacted: {result['before_messages']} -> {result['after_messages']} messages, "
                f"~{result['before_tokens']} -> ~{result['after_tokens']} tokens "
                f"(mode={result['mode']}, model={result.get('summary_model') or 'deterministic'}, "
                f"summary_tokens={result['summary_tokens']})",
                level="success",
            )
            if result.get("fallback_reason"):
                renderer.print_note(
                    f"LLM summary failed; used deterministic fallback: {result['fallback_reason']}",
                    level="warning",
                )
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
        _show_skills(agent, renderer=renderer)
    elif command == "/skill":
        await _handle_skill_command(parts[1:], agent, renderer=renderer)
    elif command == "/memory":
        await _handle_memory_command(parts[1:], agent, renderer=renderer)
    elif command == "/changes":
        _show_file_revisions(agent, renderer=renderer)
    elif command == "/undo":
        _handle_undo_command(agent, renderer=renderer)
    elif command == "/clear":
        import os
        os.system("cls" if os.name == "nt" else "clear")
    else:
        renderer.print_note(f"Unknown command: {command}. Type /help for available commands.", level="warning")
    return False


def _show_skills(agent: Agent | None, *, renderer: CliRenderer | None = None) -> None:
    renderer = renderer or CliRenderer(pretty=False)
    if agent is None:
        renderer.print_note("Agent is unavailable.", level="warning")
        return
    skills = agent.list_skills()
    if not skills:
        renderer.print_note("No skills found under .drudge/skills/*/SKILL.md", level="info")
        return
    renderer.print_list(
        "Skills",
        [f"[{'*' if skill['active'] else ' '}] {skill['name']}: {skill['description']}" for skill in skills],
    )


def _show_mcp(status: dict, *, renderer: CliRenderer | None = None) -> None:
    renderer = renderer or CliRenderer(pretty=False)
    providers = status.get("providers") or []
    if not providers:
        renderer.print_note("No MCP servers configured.", level="info")
        return
    items: list[str] = []
    for item in providers:
        state = "connected" if item.get("connected") else "unavailable"
        items.append(f"{item['name']}: {state}")
        capabilities = item.get("capabilities") or {}
        if capabilities:
            enabled = [name for name, active in capabilities.items() if active]
            items.append(f"capabilities: {', '.join(enabled) or '(none)'}")
        if item.get("resource_count"):
            items.append(f"resources: {item['resource_count']}")
        if item.get("prompt_count"):
            items.append(f"prompts: {item['prompt_count']}")
        if item.get("error"):
            items.append(f"error: {item['error']}")
        for tool in item.get("tools") or []:
            items.append(f"- {tool}")
    renderer.print_list("MCP Servers", items)


def _show_runs(agent: Agent | None, *, renderer: CliRenderer | None = None) -> None:
    renderer = renderer or CliRenderer(pretty=False)
    if agent is None:
        renderer.print_note("Agent is unavailable.", level="warning")
        return
    try:
        runs = agent.list_runs()
    except RuntimeError as exc:
        renderer.print_note(str(exc), level="warning")
        return
    if not runs:
        renderer.print_note("No persisted runs for this session.", level="info")
        return
    renderer.print_list(
        "Runs",
        [
            f"{run['id']}  {run['status']:<10}  {run['model']}  {run['started_at']}  {run['prompt'][:60]}"
            for run in runs
        ],
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


def _show_tasks(agent: Agent | None, *, include_closed: bool = False, renderer: CliRenderer | None = None) -> None:
    renderer = renderer or CliRenderer(pretty=False)
    if agent is None:
        renderer.print_note("Agent is unavailable.", level="warning")
        return
    try:
        tasks = agent.list_tasks(include_closed=include_closed)
    except RuntimeError as exc:
        renderer.print_note(str(exc), level="warning")
        return
    if not tasks:
        renderer.print_note("No tasks for the active session.", level="info")
        return
    renderer.print_list(
        "Tasks",
        [
            f"#{task['id']} [{task['status']}] {task['title']}" + (f" | {task['details']}" if task.get("details") else "")
            for task in tasks
        ],
    )


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


async def _handle_skill_command(
    arguments: list[str],
    agent: Agent | None,
    *,
    renderer: CliRenderer | None = None,
) -> None:
    renderer = renderer or CliRenderer(pretty=False)
    if agent is None:
        renderer.print_note("Agent is unavailable.", level="warning")
        return
    if not arguments:
        renderer.print_note("Usage: /skill <name> | off <name> | show <name> | run <name> [phase] | clear", level="warning")
        return
    action = arguments[0].lower()
    try:
        if action == "clear":
            agent.clear_skills()
            renderer.print_note("All skills deactivated.", level="success")
        elif action == "off":
            if len(arguments) < 2:
                renderer.print_note("Usage: /skill off <name>", level="warning")
            elif agent.deactivate_skill(arguments[1]):
                renderer.print_note(f"Deactivated skill: {arguments[1]}", level="success")
            else:
                renderer.print_note(f"Skill is not active: {arguments[1]}", level="warning")
        elif action == "show":
            if len(arguments) < 2:
                renderer.print_note("Usage: /skill show <name>", level="warning")
            else:
                skill = agent.get_skill(arguments[1])
                lines = [
                    f"Description: {skill.description}",
                    f"Path: {skill.path}",
                    f"Workflow phases: {', '.join(skill.scripts) or '(none)'}",
                    f"References: {', '.join(name for name, _ in skill.references) or '(none)'}",
                ]
                renderer.print_panel(f"Skill {skill.name}", lines)
        elif action == "run":
            if len(arguments) < 2:
                renderer.print_note("Usage: /skill run <name> [phase]", level="warning")
            else:
                phase = arguments[2] if len(arguments) > 2 else "run"
                results = await agent.run_skill_phase(arguments[1], phase)
                renderer.print_panel(
                    f"Skill {arguments[1]}:{phase}",
                    [f"{item['command']} => {'ok' if item.get('ok') else 'error'}" for item in results],
                )
        else:
            skill = agent.activate_skill(arguments[0])
            renderer.print_note(f"Activated skill: {skill.name} ({skill.description})", level="success")
    except (KeyError, RuntimeError, ValueError) as exc:
        renderer.print_note(str(exc), level="warning")


async def _handle_memory_command(
    arguments: list[str],
    agent: Agent | None,
    *,
    renderer: CliRenderer | None = None,
) -> None:
    renderer = renderer or CliRenderer(pretty=False)
    if agent is None:
        renderer.print_note("Agent is unavailable.", level="warning")
        return
    if not arguments or arguments[0].lower() == "list":
        scope = arguments[1].lower() if len(arguments) > 1 else None
        memories = agent.list_memories(scope=scope)
        if not memories:
            renderer.print_note("No persistent memories found.", level="info")
            return
        renderer.print_list(
            "Memories",
            [
                f"#{item['id']} [{'PIN' if item['pinned'] else item['scope']}] {item['title'] or item['content'][:60]}"
                for item in memories
            ],
        )
        return
    action = arguments[0].lower()
    try:
        if action == "add":
            if len(arguments) < 3:
                renderer.print_note("Usage: /memory add <project|user> <content>", level="warning")
                return
            scope = arguments[1].lower()
            content = " ".join(arguments[2:]).strip()
            memory = agent.create_memory(content, scope=scope)
            renderer.print_note(f"Saved memory #{memory['id']} ({memory['scope']}).", level="success")
        elif action in {"pin", "unpin"}:
            if len(arguments) != 2:
                renderer.print_note(f"Usage: /memory {action} <id>", level="warning")
                return
            memory = agent.update_memory(int(arguments[1]), pinned=action == "pin")
            renderer.print_note(f"Memory #{memory['id']} pinned={memory['pinned']}", level="success")
        elif action == "rm":
            if len(arguments) != 2:
                renderer.print_note("Usage: /memory rm <id>", level="warning")
                return
            deleted = agent.delete_memory(int(arguments[1]))
            renderer.print_note("Memory deleted." if deleted else "Memory not found.", level="success" if deleted else "warning")
        else:
            renderer.print_note("Usage: /memory list [scope] | add <project|user> <content> | pin <id> | unpin <id> | rm <id>", level="warning")
    except (KeyError, RuntimeError, ValueError) as exc:
        renderer.print_note(str(exc), level="warning")


def _show_file_revisions(agent: Agent | None, *, renderer: CliRenderer | None = None) -> None:
    renderer = renderer or CliRenderer(pretty=False)
    if agent is None:
        renderer.print_note("Agent is unavailable.", level="warning")
        return
    try:
        revisions = agent.list_file_revisions()
    except RuntimeError as exc:
        renderer.print_note(str(exc), level="warning")
        return
    if not revisions:
        renderer.print_note("No reversible file changes for the active session.", level="info")
        return
    renderer.print_list(
        "File Changes",
        [f"#{item['id']} {item['operation']} {item['path']}" for item in revisions],
    )


def _handle_undo_command(agent: Agent | None, *, renderer: CliRenderer | None = None) -> None:
    renderer = renderer or CliRenderer(pretty=False)
    if agent is None:
        renderer.print_note("Agent is unavailable.", level="warning")
        return
    try:
        revision = agent.undo_last_file_change()
        renderer.print_note(f"Reverted change #{revision['id']} -> {revision['path']}", level="success")
    except RuntimeError as exc:
        renderer.print_note(str(exc), level="warning")


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
    probe_model: str | None = None,
    probe_json: bool = False,
    probe_streaming: bool = True,
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

    if probe_model is not None and probe_json:
        report = _run_provider_probe(config, probe_model, probe_streaming)
        if report is None:
            print(json.dumps({"error": "Provider probe is not available for Codex OAuth."}))
        else:
            print(report.to_json())
        return

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

    if probe_model is not None:
        report = _run_provider_probe(config, probe_model, probe_streaming)
        if report is None:
            print("Provider probe is not available for Codex OAuth.")
            return
        from agent.provider_probe import format_probe_report

        print(format_probe_report(report))


def _run_provider_probe(config, probe_model: str, probe_streaming: bool):
    if config.get("model", "provider") == "openai-codex":
        return None
    from agent.provider_probe import probe_provider

    target_model = probe_model or str(config.get("model", "name"))
    return asyncio.run(probe_provider(
        config.get_model_config(),
        model=target_model,
        include_streaming=probe_streaming,
    ))


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
            getattr(args, "probe_model", None),
            getattr(args, "probe_json", False),
            not getattr(args, "no_probe_streaming", False),
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
