from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
import textwrap
from dataclasses import dataclass
from time import monotonic
from typing import Any, Callable, TextIO


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
_LIST_RE = re.compile(r"^\s{0,3}(?:[-*+]\s+|\d+\.\s+)")


@dataclass(slots=True)
class _Theme:
    border: str = "36"
    title: str = "96"
    text: str = "97"
    dim: str = "90"
    success: str = "92"
    warning: str = "93"
    error: str = "91"
    accent: str = "95"
    heading: str = "96"
    bullet: str = "92"
    code: str = "93"


class CliRenderer:
    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        err_stream: TextIO | None = None,
        pretty: bool = True,
        color: bool | None = None,
        width: int | None = None,
        is_tty: bool | None = None,
    ) -> None:
        self.stream = stream or sys.stdout
        self.err_stream = err_stream or sys.stderr
        tty = bool(is_tty) if is_tty is not None else bool(getattr(self.stream, "isatty", lambda: False)())
        self.pretty = bool(pretty and tty)
        self.color = bool(self.pretty if color is None else color)
        if self.color:
            self.color = self._enable_color_support()
        terminal_width = width or shutil.get_terminal_size((100, 20)).columns
        self.width = max(60, min(terminal_width, 120))
        self.theme = _Theme()
        self._status_width = 0

    def _enable_color_support(self) -> bool:
        if not self.pretty or os.getenv("NO_COLOR"):
            return False
        if os.name != "nt":
            return True
        try:
            from colorama import just_fix_windows_console

            just_fix_windows_console()
            return True
        except Exception:
            return bool(
                os.getenv("WT_SESSION")
                or os.getenv("ANSICON")
                or os.getenv("TERM_PROGRAM")
                or str(os.getenv("ConEmuANSI", "")).upper() == "ON"
            )

    def make_stream_printer(self, title: str = "Assistant") -> "StreamPrinter":
        return StreamPrinter(self, title=title)

    def make_activity_ticker(
        self,
        label: str = "Thinking",
        *,
        state_getter: Callable[[], str] | None = None,
    ) -> "ActivityTicker":
        return ActivityTicker(self, label=label, state_getter=state_getter)

    def print_banner(
        self,
        *,
        version: str,
        model: str,
        toolsets: list[str] | None = None,
        codex_config_path: Any = None,
        subtitle: str | None = None,
    ) -> None:
        lines = [f"Model: {model}"]
        if toolsets is not None:
            lines.append(f"Toolsets: {', '.join(toolsets) or '(none)'}")
        if codex_config_path:
            lines.append(f"Codex config: {codex_config_path}")
        if subtitle:
            lines.append(subtitle)
        self.print_panel(f"Drudge v{version}", lines, accent=self.theme.accent)

    def print_panel(self, title: str, content: str | list[str], *, accent: str | None = None) -> None:
        lines = content.splitlines() if isinstance(content, str) else [str(item) for item in content]
        self._writeln(self._render_panel(title, lines, accent=accent))

    def print_key_values(self, title: str, pairs: list[tuple[str, Any]]) -> None:
        if not pairs:
            self.print_panel(title, ["(none)"])
            return
        key_width = max(len(str(key)) for key, _ in pairs)
        lines = [f"{str(key).ljust(key_width)} : {value}" for key, value in pairs]
        self.print_panel(title, lines)

    def print_list(self, title: str, items: list[str]) -> None:
        lines = [f"- {item}" for item in items] if items else ["(none)"]
        self.print_panel(title, lines)

    def print_note(self, text: str, *, level: str = "info", error: bool = False) -> None:
        stream = self.err_stream if error else self.stream
        prefix = {
            "info": "INFO",
            "success": "OK",
            "warning": "WARN",
            "error": "ERR",
        }.get(level, "INFO")
        color_name = {
            "info": self.theme.title,
            "success": self.theme.success,
            "warning": self.theme.warning,
            "error": self.theme.error,
        }.get(level, self.theme.title)
        line = f"[{prefix}] {text}"
        self._writeln(self._style(line, color_name, bold=True) if self.color else line, stream=stream)

    def print_usage(self, usage: dict[str, Any], *, session_id: str | None = None) -> None:
        parts = [
            f"Tokens {usage['total_tokens']}",
            f"Utility {usage['utility_tokens']}",
            f"Turns {usage['turns']}",
        ]
        if session_id:
            parts.append(f"Session {session_id}")
        self.print_panel("Run Summary", [" | ".join(parts)], accent=self.theme.success)

    def print_assistant_message(self, text: str, *, title: str = "Assistant") -> None:
        self.print_panel(title, text.splitlines() or [""], accent=self.theme.title)

    def print_tool_event(self, name: str, args: dict[str, Any] | None, result: str | None) -> None:
        lines = [f"tool: {name}"]
        if args:
            lines.append("args:")
            lines.extend(self._serialize_block(args, max_chars=320))
        if result:
            lines.append("result:")
            lines.extend(self._serialize_block(result, max_chars=420))
        self._writeln(
            self._render_panel(
                "Tool Log",
                lines,
                accent=self.theme.accent,
                panel_width=self.width,
            )
        )

    def print_status(self, result: dict[str, Any]) -> None:
        local = result["local"]
        self.print_key_values(
            "Runtime",
            [
                ("Session", local["session_id"] or "(new)"),
                ("Run status", local["run_status"]),
                ("Runtime", "started" if local["runtime_started"] else "scoped/not started"),
                ("Model", f"{local['model']} ({local['provider']}, {local['model_api']})"),
                (
                    "Utility model",
                    f"{local['utility_model']} ({'configured' if local['utility_model_configured'] else 'primary reused'})",
                ),
                ("Turns", local["turns"]),
                ("Messages", local["message_count"]),
                (
                    "Tokens",
                    f"{local['tokens_this_process']} (utility: {local['utility_tokens_this_process']})",
                ),
                ("Workspace", local["workspace"]),
                ("Approval", local["approval_mode"]),
                ("Skills", ", ".join(local["active_skills"]) or "(none)"),
                ("Open tasks", local["open_tasks"]),
                ("Project memories", local.get("project_memory_count", 0)),
                ("User memories", local.get("user_memory_count", 0)),
                ("Reversible edits", local.get("file_revisions", 0)),
                ("MCP", ", ".join(local["mcp_servers"]) or "(none)"),
            ],
        )
        if local["context_limit"]:
            used = local["context_used_percent"] or 0.0
            self.print_panel(
                "Context",
                [
                    f"~{local['estimated_context_tokens']}/{local['context_limit']} tokens",
                    f"{max(0.0, 100.0 - used):.1f}% left",
                ],
                accent=self.theme.heading,
            )
        selection = local.get("last_tool_selection")
        if selection:
            self.print_panel(
                "Tool Selection",
                [
                    f"Mode: {selection['mode']}",
                    f"Tools: {len(selection['selected'])}/{selection['catalog_tools']}",
                    f"Schema tokens: ~{selection['schema_tokens']}",
                ],
            )
        if result.get("account_usage"):
            from agent.codex_usage import format_codex_usage

            self.print_panel("Codex Account Usage", format_codex_usage(result["account_usage"]))
        elif result.get("account_usage_error"):
            self.print_panel(
                "Codex Account Usage",
                [f"Unavailable: {result['account_usage_error']}"],
                accent=self.theme.warning,
            )
        else:
            self.print_panel(
                "Codex Account Usage",
                ["Unavailable for the current non-Codex provider"],
                accent=self.theme.dim,
            )

    def print_help(self) -> None:
        self.print_panel(
            "Commands",
            [
                "/quit, /exit, /q    Exit Drudge",
                "/help               Show this help",
                "/tools              List available tools",
                "/mcp                Inspect configured MCP stdio servers",
                "/config             Show current config",
                "/models             List provider models",
                "/sessions           List saved sessions",
                "/history [id]       Show saved messages",
                "/runs               List recent runs",
                "/trace [run_id]     Show a persisted run trace",
                "/tasks [all]        List persistent session tasks",
                "/task add <title>   Create a persistent task",
                "/task start|done|cancel|reopen <id>",
                "/memory [...]       Manage durable project/user memories",
                "/changes            List reversible file changes",
                "/undo               Revert the latest file change",
                "/status             Show session, context, and account limits",
                "/compact            Compact older conversation context",
                "/resume <id>        Resume a saved session",
                "/new                Start a new session",
                "/skills             List discovered skills",
                "/skill <name>       Activate a skill",
                "/skill off <name>   Deactivate a skill",
                "/skill show <name>  Show skill metadata",
                "/skill run <name>   Execute a skill workflow phase",
                "/skill clear        Deactivate all skills",
                "/clear              Clear screen",
            ],
        )

    def _render_panel(
        self,
        title: str,
        lines: list[str],
        *,
        accent: str | None = None,
        panel_width: int | None = None,
        indent: int = 0,
    ) -> str:
        color = accent or self.theme.border
        body_width = max(24, (panel_width or self.width) - 4)
        wrapped: list[str] = []
        for line in lines or [""]:
            wrapped.extend(self._wrap_line(line, body_width))
        content_width = min(
            body_width,
            max([len(self._strip_ansi(title)) + 2] + [self._visible_len(line) for line in wrapped] + [24]),
        )
        top_fill = max(0, content_width - len(self._strip_ansi(title)) - 1)
        top = "+- " + title + " " + ("-" * top_fill) + "+"
        bottom = "+" + ("-" * (content_width + 2)) + "+"
        prefix = " " * max(0, indent)
        rows = [prefix + self._style(top, color, bold=True)]
        in_code_block = False
        for line in wrapped or [""]:
            styled = self._style_content(line, in_code_block=in_code_block)
            rows.append(prefix + "| " + self._pad_ansi(styled, content_width) + " |")
            if line.strip().startswith(("```", "~~~")):
                in_code_block = not in_code_block
        rows.append(prefix + self._style(bottom, color, bold=True))
        return "\n".join(rows)

    def _style_content(self, line: str, *, in_code_block: bool = False) -> str:
        if not self.color:
            return line
        stripped = line.strip()
        if not stripped:
            return line
        if stripped.startswith(("```", "~~~")):
            return self._style(line, self.theme.code)
        if in_code_block:
            return self._highlight_code_line(line)
        if _HEADING_RE.match(line):
            return self._style(line, self.theme.heading, bold=True)
        if _LIST_RE.match(line):
            return self._style(line, self.theme.bullet)
        if stripped.startswith(("Error:", "ERR", "[ERR]")):
            return self._style(line, self.theme.error, bold=True)
        return self._style(line, self.theme.text)

    def _highlight_code_line(self, line: str) -> str:
        highlighted = line
        for token in (
            "def", "class", "return", "if", "elif", "else", "for", "while", "try", "except",
            "finally", "import", "from", "as", "with", "await", "async", "True", "False",
            "None", "raise", "yield", "lambda", "pass", "break", "continue",
        ):
            highlighted = highlighted.replace(token, self._style(token, self.theme.accent, bold=True))
        for token in ("true", "false", "null"):
            highlighted = highlighted.replace(token, self._style(token, self.theme.accent, bold=True))
        if line.lstrip().startswith(("$", ">")):
            parts = highlighted.split(maxsplit=1)
            if parts:
                head = parts[0]
                tail = parts[1] if len(parts) > 1 else ""
                highlighted = self._style(head, self.theme.success, bold=True)
                if tail:
                    highlighted = highlighted + " " + tail
        return highlighted

    def _wrap_line(self, line: str, width: int) -> list[str]:
        if width <= 0:
            return [line]
        if not line:
            return [""]
        if line.strip().startswith(("```", "~~~")):
            return [line]
        wrapped = textwrap.wrap(
            line,
            width=width,
            replace_whitespace=False,
            drop_whitespace=False,
            break_long_words=True,
            break_on_hyphens=False,
        )
        return wrapped or [""]

    @staticmethod
    def _serialize_block(value: Any, *, max_chars: int) -> list[str]:
        if isinstance(value, str):
            text = value
        else:
            text = json.dumps(value, ensure_ascii=False, indent=2)
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        if "\n" in text or text.strip().startswith(("{", "[")):
            return ["```json", *text.splitlines(), "```"]
        return [text]

    def _style(self, text: str, color: str, *, bold: bool = False) -> str:
        if not self.color:
            return text
        prefix = [color]
        if bold:
            prefix.append("1")
        return f"\x1b[{';'.join(prefix)}m{text}\x1b[0m"

    @staticmethod
    def _strip_ansi(text: str) -> str:
        return _ANSI_RE.sub("", text)

    def _visible_len(self, text: str) -> int:
        return len(self._strip_ansi(text))

    def _pad_ansi(self, text: str, width: int) -> str:
        padding = max(0, width - self._visible_len(text))
        return text + (" " * padding)

    def show_status_line(self, text: str) -> None:
        if not self.pretty:
            return
        visible = self._visible_len(text)
        padding = max(0, self._status_width - visible)
        self.stream.write("\r" + text + (" " * padding))
        self.stream.flush()
        self._status_width = visible

    def clear_status_line(self) -> None:
        if not self.pretty or self._status_width <= 0:
            return
        self.stream.write("\r" + (" " * self._status_width) + "\r")
        self.stream.flush()
        self._status_width = 0

    def _writeln(self, text: str = "", *, stream: TextIO | None = None) -> None:
        self.clear_status_line()
        handle = stream or self.stream
        handle.write(text + "\n")
        handle.flush()


class StreamPrinter:
    def __init__(self, renderer: CliRenderer, *, title: str = "Assistant") -> None:
        self.renderer = renderer
        self.title = title
        self.seen = False
        self.parts: list[str] = []
        self._buffer = ""
        self._closed = False

    def __call__(self, delta: str) -> None:
        if not delta:
            return
        self.seen = True
        self.parts.append(delta)
        if self.renderer.pretty:
            self._buffer += delta
        else:
            self.renderer.stream.write(delta)
            self.renderer.stream.flush()

    def finish(self, final_text: str) -> None:
        if self._closed:
            return
        self._closed = True
        if self.renderer.pretty:
            self._buffer = ""
            self.renderer.print_assistant_message(final_text, title=self.title)
        else:
            if self.seen:
                self.renderer._writeln()
                streamed = "".join(self.parts).rstrip()
                if final_text.strip() and not streamed.endswith(final_text.rstrip()):
                    self.renderer._writeln(final_text)
            else:
                self.renderer._writeln(final_text)


class ActivityTicker:
    def __init__(
        self,
        renderer: CliRenderer,
        *,
        label: str = "Thinking",
        state_getter: Callable[[], str] | None = None,
    ) -> None:
        self.renderer = renderer
        self.label = label
        self.state_getter = state_getter
        self._stopped = asyncio.Event()
        self._frames = ("-", "\\", "|", "/")
        self._frame_index = 0

    async def run(self) -> None:
        started = monotonic()
        while not self._stopped.is_set():
            elapsed = int(monotonic() - started)
            label = self.state_getter() if self.state_getter is not None else self.label
            frame = self._frames[self._frame_index % len(self._frames)]
            self._frame_index += 1
            line = self.renderer._style(f"[{frame} {label} {elapsed}s]", self.renderer.theme.dim, bold=True)
            self.renderer.show_status_line(line)
            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=0.12)
            except asyncio.TimeoutError:
                continue

    def stop(self) -> None:
        self._stopped.set()
        self.renderer.clear_status_line()

