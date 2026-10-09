"""Keyboard-only approval UI; decisions still belong to the host policy."""

from __future__ import annotations

import asyncio
import json
import textwrap
import unicodedata

from tools.risk import ApprovalDecision, ApprovalRequest


class ApprovalCancelled(asyncio.CancelledError):
    """The user cancelled this run from the approval menu."""


OPTIONS = (
    (ApprovalDecision.ALLOW_ONCE, "仅本次允许"),
    (ApprovalDecision.ALLOW_SESSION, "本会话允许此工具（同一风险等级）"),
    (ApprovalDecision.DENY, "拒绝本次操作"),
)


def _plain_text(value: str) -> str:
    """Display terminal controls and invisible format characters literally."""
    return "".join(
        char if char == "\n" or not unicodedata.category(char).startswith("C")
        else (f"\\u{ord(char):04x}" if ord(char) > 255 else f"\\x{ord(char):02x}")
        for char in value
    )


def _redact(value):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            sensitive = normalized in {
                "api_key", "token", "secret", "password", "authorization", "proxy_authorization", "cookie", "set_cookie",
            } or normalized.endswith(("_api_key", "_token", "_secret", "_password"))
            result[key] = "***" if sensitive else _redact(item)
        return result
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def request_details(request: ApprovalRequest) -> str:
    text = (
        f"操作: {request.risk.action}\n原因: {request.risk.reason}\n参数:\n"
        + json.dumps(_redact(request.arguments), ensure_ascii=False, indent=2, default=str)
    )
    # Hard wraps make PageUp/PageDown useful even for a one-line shell command.
    return "\n".join(
        piece for line in _plain_text(text).split("\n")
        for piece in (textwrap.wrap(line, width=88, replace_whitespace=False, drop_whitespace=False) or [""])
    )


def create_approval_application(request: ApprovalRequest, *, input=None, output=None, color: bool = True):
    # Keep --help / --version usable even when optional interactive imports fail.
    from prompt_toolkit.application import Application
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.layout import HSplit, Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.layout.dimension import Dimension
    from prompt_toolkit.styles import Style
    from prompt_toolkit.widgets import Frame, TextArea

    selected = 2  # A new request always starts on DENY, never the last selection.
    bindings = KeyBindings()
    details = TextArea(
        text=request_details(request), read_only=True, focusable=False,
        scrollbar=True, wrap_lines=True, height=Dimension(min=2, preferred=5, max=8),
    )

    def choices():
        fragments = []
        for index, (_, label) in enumerate(OPTIONS):
            style = "class:selected" if index == selected else ""
            fragments.append((style, f"{'>' if index == selected else ' '} {index + 1}. {label}\n"))
        return fragments

    @bindings.add("up")
    @bindings.add("s-tab")
    def move_up(event):
        nonlocal selected
        selected = (selected - 1) % len(OPTIONS)

    @bindings.add("down")
    @bindings.add("tab")
    def move_down(event):
        nonlocal selected
        selected = (selected + 1) % len(OPTIONS)

    @bindings.add("1")
    @bindings.add("2")
    @bindings.add("3")
    def select_number(event):
        nonlocal selected
        selected = int(event.data) - 1

    @bindings.add("enter")
    def confirm(event):
        event.app.exit(result=OPTIONS[selected][0])

    @bindings.add("escape", eager=True)
    @bindings.add("c-d")
    def deny(event):
        event.app.exit(result=ApprovalDecision.DENY)

    @bindings.add("c-c")
    @bindings.add(Keys.SIGINT)
    def cancel(event):
        event.app.exit(exception=ApprovalCancelled())

    @bindings.add("pageup")
    def page_up(event):
        details.buffer.cursor_up(count=5)

    @bindings.add("pagedown")
    def page_down(event):
        details.buffer.cursor_down(count=5)

    @bindings.add(Keys.BracketedPaste)
    @bindings.add(Keys.Any)
    def ignore_text(event):
        # Pasted text (including embedded newlines) is not an approval gesture.
        pass

    control = FormattedTextControl(
        choices, focusable=True, get_cursor_position=lambda: Point(x=0, y=selected),
    )
    body = HSplit([
        Window(FormattedTextControl([("class:title", "权限申请")]), height=1),
        Window(FormattedTextControl(_plain_text(
            f"工具: {request.tool_name}   风险: {request.risk.level.value}"
        )), wrap_lines=True, dont_extend_height=True),
        Frame(details, title="操作详情 · PgUp / PgDn 翻页"),
        Window(control, wrap_lines=True, dont_extend_height=True),
        Window(FormattedTextControl(
            "↑/↓ 或 Tab 选择 · Enter 确认 · Esc 拒绝 · Ctrl+C 取消任务\n"
            "会话允许仅限此工具及风险等级，后续参数可以不同；退出或新建会话后失效。"
        ), wrap_lines=True, dont_extend_height=True),
    ])
    app = Application(
        layout=Layout(body, focused_element=control), key_bindings=bindings,
        input=input, output=output, full_screen=False, mouse_support=False,
        style=Style.from_dict({
            "title": "bold ansicyan" if color else "bold",
            "selected": "bold reverse ansicyan" if color else "reverse",
        }),
    )
    app.ttimeoutlen = 0.05
    return app


async def prompt_approval(request: ApprovalRequest, *, input=None, output=None, color: bool = True) -> ApprovalDecision:
    app = create_approval_application(request, input=input, output=output, color=color)
    try:
        return await app.run_async()
    except EOFError:
        return ApprovalDecision.DENY
