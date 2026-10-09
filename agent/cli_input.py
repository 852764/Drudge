"""Slash-command completion for the prompt-toolkit interactive input only."""

from __future__ import annotations

import sys

from prompt_toolkit import PromptSession
from prompt_toolkit.buffer import CompletionState
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.filters import Condition, completion_is_selected, has_completions
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.output import ColorDepth, create_output
from prompt_toolkit.styles import Style

from agent.slash_commands import COMMAND_OPTIONS, SLASH_COMMANDS


class SlashCommandCompleter(Completer):
    def get_completions(self, document, complete_event):
        text = document.text
        # Don't replace text in the middle of a prompt or interpret a pasted
        # multiline block, a URL, or a slash within normal prose as a command.
        if (not text.startswith("/") or document.cursor_position != len(text)
                or any(char in text for char in "\r\n")):
            return
        words = text.split()
        if len(words) == 1 and not text[-1].isspace():
            prefix = text.lower()
            for command in SLASH_COMMANDS:
                names = (command.name,) + (command.aliases if prefix != "/" else ())
                for name in names:
                    if name.startswith(prefix) and name != prefix:
                        suffix = " " if command.takes_input else ""
                        # Keep menus compact; argument syntax lives in metadata.
                        description = f"{command.menu_description}  {command.arguments}".strip()
                        yield Completion(name + suffix, start_position=-len(text),
                                         display=name, display_meta=description)
            return

        prefix = "" if text[-1].isspace() else words.pop()
        options = COMMAND_OPTIONS.get(tuple(word.lower() for word in words), ())
        for option in options:
            if option.name.startswith(prefix.lower()) and option.name != prefix.lower():
                suffix = " " if option.takes_input else ""
                yield Completion(option.name + suffix, start_position=-len(prefix),
                                 display=option.name, display_meta=option.description)


def create_prompt_session(*, color: bool = True, input=None, output=None) -> PromptSession:
    if output is None and sys.platform == "win32":
        # Windows console capability comes from its native output, not TERM.
        # Passing the output explicitly prevents prompt-toolkit from silently
        # switching to its menu-less dumb prompt when TERM=dumb is inherited.
        output = create_output()
    completer = SlashCommandCompleter()
    bindings = KeyBindings()

    @Condition
    def can_complete():
        from prompt_toolkit.application.current import get_app

        buffer = get_app().current_buffer
        return bool(buffer.complete_state or next(
            completer.get_completions(buffer.document, CompleteEvent()), None,
        ))

    def ensure_completion_state(buffer):
        # Candidates are a small, local static list. Populate synchronously on
        # navigation so a quick "/, Down, Enter" cannot outrun the background
        # completion task and accidentally submit the original prefix.
        if buffer.complete_state is None:
            completions = list(completer.get_completions(buffer.document, CompleteEvent()))
            if completions:
                buffer.complete_state = CompletionState(buffer.document, completions)

    @bindings.add("down", filter=can_complete)
    def next_item(event):
        buffer = event.current_buffer
        ensure_completion_state(buffer)
        buffer.complete_next()

    @bindings.add("up", filter=can_complete)
    @bindings.add("s-tab", filter=can_complete)
    def previous_item(event):
        buffer = event.current_buffer
        ensure_completion_state(buffer)
        buffer.complete_previous()

    @bindings.add("tab", filter=can_complete)
    def fill_completion(event):
        buffer = event.current_buffer
        state = buffer.complete_state
        completion = (state.current_completion or state.completions[0]) if state else next(
            completer.get_completions(buffer.document, CompleteEvent()), None,
        )
        if completion is not None:
            buffer.apply_completion(completion)

    @bindings.add("enter", filter=has_completions & completion_is_selected)
    def confirm_selection(event):
        # Arrow keys preview a completion in the buffer. Confirmation ONLY fills
        # the input; a separate Enter submits to the unchanged host dispatcher.
        buffer = event.current_buffer
        completion = buffer.complete_state.current_completion
        buffer.apply_completion(completion)

    @bindings.add("escape", filter=has_completions)
    def dismiss(event):
        event.current_buffer.cancel_completion()

    def toolbar():
        if session.default_buffer.complete_state:
            return [("class:bottom-toolbar", " ↑/↓ 选择 · Tab 填入 · Enter 确认已选项 · Esc 关闭 ")]
        return [("class:bottom-toolbar", " / 命令菜单 · 已填入命令后 Enter 执行 ")]

    session = PromptSession(
        completer=completer,
        complete_while_typing=True,
        reserve_space_for_menu=8,
        key_bindings=bindings,
        bottom_toolbar=toolbar,
        style=Style.from_dict({
            "prompt": "ansicyan bold",
            "completion-menu.completion.current": "bg:ansicyan ansiblack" if color else "reverse bold",
            "completion-menu.meta.completion.current": "bg:ansicyan ansiblack" if color else "reverse bold",
        }),
        color_depth=None if color else ColorDepth.DEPTH_1_BIT,
        input=input,
        output=output,
    )
    return session
