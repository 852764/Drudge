from __future__ import annotations

import ast
import asyncio
import builtins
import inspect
import io
import unittest
from unittest.mock import AsyncMock, Mock, patch

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import ColorDepth, DummyOutput

from agent.cli_input import SlashCommandCompleter, create_prompt_session
from agent.cli_renderer import CliRenderer
from agent.slash_commands import SLASH_COMMANDS, command_help_lines
from config import ConfigManager
from main import _handle_command, run_interactive


class SlashCompletionTests(unittest.TestCase):
    def complete(self, text, cursor=None):
        return list(SlashCommandCompleter().get_completions(Document(text, cursor), CompleteEvent()))

    def test_slash_lists_canonical_commands_with_descriptions(self):
        choices = self.complete("/")
        self.assertEqual([choice.text.rstrip() for choice in choices], [item.name for item in SLASH_COMMANDS])
        self.assertTrue(all(choice.display_meta_text for choice in choices))
        self.assertTrue(all(choice.start_position == -1 for choice in choices))
        self.assertIn("恢复", next(choice.display_meta_text for choice in choices if choice.text == "/resume "))

    def test_prefix_filter_aliases_case_and_argument_hints(self):
        for text, expected in (("/RE", ["/resume "]), ("/ex", ["/exit"]),
                               ("/us", ["/usage"]), ("/no-such-command", [])):
            with self.subTest(text=text):
                self.assertEqual([item.text for item in self.complete(text)], expected)
        self.assertIn("<id>", self.complete("/re")[0].display_meta_text)

    def test_normal_text_paths_multiline_and_cursor_edits_are_untouched(self):
        for text in ("", "hello", "解释 /help", "https://example.test/", " /help", "/tmp/file",
                     "/help\n/quit", "/help\r/quit", "/resume session-id", "/task add 普通标题",
                     "/skill custom-name", "/memory add project 普通内容", "/undo --dry-run extra"):
            with self.subTest(text=text):
                self.assertEqual(self.complete(text), [])
        self.assertEqual(self.complete("/re text", 3), [])

    def test_static_subcommands_and_nested_options(self):
        for text, expected in (
            ("/task ", ["add ", "start ", "done ", "cancel ", "reopen "]),
            ("/task d", ["done "]), ("/TASK  D", ["done "]),
            ("/tasks a", ["all"]), ("/undo --", ["--dry-run"]),
            ("/skill ", ["show ", "off ", "run ", "clear"]),
            ("/memory add ", ["project ", "user "]),
            ("/memory list p", ["project"]),
        ):
            with self.subTest(text=text):
                self.assertEqual([item.text for item in self.complete(text)], expected)
        completion = self.complete("/memory add pr")[0]
        self.assertEqual(completion.start_position, -2)

    def test_complete_command_is_not_replaced_by_itself(self):
        for text in ("/help", "/resume", "/task done", "/undo --dry-run"):
            self.assertEqual(self.complete(text), [])

    def test_help_registry_covers_every_host_command_and_alias(self):
        names = [name for item in SLASH_COMMANDS for name in (item.name, *item.aliases)]
        self.assertEqual(len(names), len(set(names)))
        tree = ast.parse(inspect.getsource(_handle_command))
        handled = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == "command":
                handled.update(child.value for child in ast.walk(node)
                               if isinstance(child, ast.Constant) and isinstance(child.value, str))
        self.assertEqual(set(names), handled - {"/"})
        help_text = "\n".join(command_help_lines())
        for name in names:
            self.assertIn(name, help_text)


class PromptKeyboardTests(unittest.TestCase):
    async def wait_until(self, predicate):
        async def wait():
            while not predicate():
                await asyncio.sleep(0.01)
        await asyncio.wait_for(wait(), 3)

    async def exercise(self, action):
        with create_pipe_input() as pipe:
            session = create_prompt_session(input=pipe, output=DummyOutput())
            # Make standalone Escape deterministic without changing real user
            # terminal timings or truncating arrow-key sequences in production.
            session.app.ttimeoutlen = 0.03
            async def read_prompt():
                # Catch BaseException-derived keyboard interrupts in the child
                # task itself; otherwise asyncio stops the test event loop.
                try:
                    return await session.prompt_async("> ")
                except (KeyboardInterrupt, EOFError) as exc:
                    return exc

            task = asyncio.create_task(read_prompt())
            try:
                await self.wait_until(lambda: session.app.is_running)
                return await action(pipe, session, task)
            finally:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    def test_slash_automatically_opens_menu_without_submitting(self):
        async def action(pipe, session, task):
            pipe.send_text("/")
            await self.wait_until(lambda: session.default_buffer.complete_state is not None)
            state = session.default_buffer.complete_state
            self.assertIsNone(state.current_completion)
            self.assertEqual(len(state.completions), len(SLASH_COMMANDS))
            self.assertFalse(task.done())
        asyncio.run(self.exercise(action))

    def test_down_up_enter_fills_only_then_second_enter_submits(self):
        async def action(pipe, session, task):
            pipe.send_text("/")
            await self.wait_until(lambda: session.default_buffer.complete_state is not None)
            pipe.send_text("\x1b[B\x1b[B\x1b[A\r")
            await self.wait_until(lambda: session.default_buffer.text == "/help" and session.default_buffer.complete_state is None)
            self.assertFalse(task.done(), "Choosing a command must not execute it")
            pipe.send_text("\r")
            self.assertEqual(await asyncio.wait_for(task, 3), "/help")
        asyncio.run(self.exercise(action))

    def test_tab_completes_prefix_and_leaves_argument_placeholder_out(self):
        async def action(pipe, session, task):
            pipe.send_text("/re\t")
            await self.wait_until(lambda: session.default_buffer.text == "/resume ")
            self.assertFalse(task.done())
            pipe.send_text("fixture-id\r")
            self.assertEqual(await asyncio.wait_for(task, 3), "/resume fixture-id")
        asyncio.run(self.exercise(action))

    def test_fast_arrow_selection_before_async_menu_is_ready(self):
        async def action(pipe, session, task):
            pipe.send_text("/\x1b[B\r")
            await self.wait_until(lambda: session.default_buffer.text == "/help" or task.done())
            self.assertFalse(task.done())
            self.assertEqual(session.default_buffer.text, "/help")
            pipe.send_text("\r")
            self.assertEqual(await asyncio.wait_for(task, 3), "/help")
        asyncio.run(self.exercise(action))

    def test_shift_tab_selects_last_and_tab_fills_without_exiting(self):
        async def action(pipe, session, task):
            pipe.send_text("/")
            await self.wait_until(lambda: session.default_buffer.complete_state is not None)
            pipe.send_text("\x1b[Z\t")
            await self.wait_until(lambda: session.default_buffer.text == "/quit" and session.default_buffer.complete_state is None)
            self.assertFalse(task.done())
            pipe.send_text("\r")
            self.assertEqual(await asyncio.wait_for(task, 3), "/quit")
        asyncio.run(self.exercise(action))

    def test_escape_restores_typed_prefix_and_allows_plain_text(self):
        async def action(pipe, session, task):
            pipe.send_text("/re")
            await self.wait_until(lambda: session.default_buffer.complete_state is not None)
            pipe.send_text("\x1b[B")
            await self.wait_until(lambda: session.default_buffer.text == "/resume ")
            pipe.send_text("\x1b")
            await self.wait_until(lambda: session.default_buffer.complete_state is None)
            self.assertEqual(session.default_buffer.text, "/re")
            self.assertFalse(task.done())
            pipe.send_text("\x15普通问题 /help\r")
            self.assertEqual(await asyncio.wait_for(task, 3), "普通问题 /help")
        asyncio.run(self.exercise(action))

    def test_nested_subcommand_tab_completion(self):
        async def action(pipe, session, task):
            pipe.send_text("/memory a\tpr\t内容\r")
            self.assertEqual(await asyncio.wait_for(task, 3), "/memory add project 内容")
        asyncio.run(self.exercise(action))

    def test_enter_without_selection_preserves_manually_typed_command(self):
        async def action(pipe, session, task):
            pipe.send_text("/task")  # /tasks is suggested but not chosen.
            await self.wait_until(lambda: session.default_buffer.complete_state is not None)
            pipe.send_text("\r")
            self.assertEqual(await asyncio.wait_for(task, 3), "/task")
        asyncio.run(self.exercise(action))

    def test_bracketed_paste_does_not_execute_or_select_commands(self):
        async def action(pipe, session, task):
            text = "/help\n/quit"
            pipe.send_text("\x1b[200~" + text + "\x1b[201~")
            await self.wait_until(lambda: session.default_buffer.text == text)
            self.assertIsNone(session.default_buffer.complete_state)
            self.assertFalse(task.done())
            pipe.send_text("\x03")
            self.assertIsInstance(await asyncio.wait_for(task, 3), KeyboardInterrupt)
        asyncio.run(self.exercise(action))

    def test_plain_text_history_and_prompt_reuse(self):
        async def run():
            with create_pipe_input() as pipe:
                session = create_prompt_session(color=False, input=pipe, output=DummyOutput())
                self.assertEqual(session.color_depth, ColorDepth.DEPTH_1_BIT)
                for keys, expected in (("第一条消息\r", "第一条消息"),
                                       ("\x1b[A\r", "第一条消息"),
                                       ("/he\t\r", "/help"), ("继续\r", "继续")):
                    task = asyncio.create_task(session.prompt_async("> "))
                    try:
                        await self.wait_until(lambda: session.app.is_running)
                        if keys.startswith("\x1b[A"):
                            # History loading is asynchronous in prompt-toolkit.
                            # Wait for the previous input to reach the buffer,
                            # rather than racing it with a prequeued Up+Enter.
                            await self.wait_until(lambda: session.default_buffer.working_index > 0)
                        pipe.send_text(keys)
                        self.assertEqual(await asyncio.wait_for(task, 3), expected)
                    finally:
                        if not task.done():
                            task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
        asyncio.run(run())

    def test_control_c_and_eof_keep_existing_input_exit_behavior(self):
        async def run(keys, error):
            with create_pipe_input() as pipe:
                session = create_prompt_session(input=pipe, output=DummyOutput())
                pipe.send_text(keys)
                async def interrupted_prompt():
                    with self.assertRaises(error):
                        await session.prompt_async("> ")
                await asyncio.wait_for(interrupted_prompt(), 3)
                pipe.send_text("next\r")
                self.assertEqual(await asyncio.wait_for(session.prompt_async("> "), 3), "next")
        for keys, error in (("/\x03", KeyboardInterrupt), ("\x04", EOFError)):
            with self.subTest(keys=repr(keys)):
                asyncio.run(run(keys, error))


class InteractiveInputIntegrationTests(unittest.TestCase):
    def test_windows_dumb_environment_keeps_the_real_completion_layout(self):
        async def run():
            with create_pipe_input() as pipe:
                output = DummyOutput()
                with patch("agent.cli_input.sys.platform", "win32"), \
                     patch("agent.cli_input.create_output", return_value=output) as output_factory, \
                     patch.dict("os.environ", {"TERM": "dumb"}):
                    session = create_prompt_session(color=False, input=pipe)
                    output_factory.assert_called_once_with()
                    self.assertIs(session.output, output)
                    # The dumb prompt runs a different Application and ignores
                    # this pre-run callback on the full completion layout.
                    entered_full_layout = Mock()
                    session.app.pre_run_callables.append(entered_full_layout)
                    pipe.send_text("/he\t\r")
                    self.assertEqual(await session.prompt_async("> "), "/help")
                    entered_full_layout.assert_called_once_with()
        asyncio.run(run())

    def test_bare_slash_prints_help_without_model_or_tool_calls(self):
        agent = Mock()
        stream = io.StringIO()
        self.assertFalse(asyncio.run(_handle_command("/", ConfigManager(), agent,
                                                     renderer=CliRenderer(stream=stream, pretty=False))))
        self.assertIn("/resume <id>", stream.getvalue())
        self.assertEqual(agent.mock_calls, [])

    def test_noninteractive_or_dumb_terminal_uses_simple_input(self):
        for stdin_tty, stdout_tty, term, platform in ((False, True, "xterm", "win32"),
                                                     (True, False, "xterm", "linux"),
                                                     (True, True, "dumb", "linux")):
            with self.subTest(stdin=stdin_tty, stdout=stdout_tty, term=term):
                with patch("main.sys.stdin.isatty", return_value=stdin_tty), \
                     patch("main.sys.stdout.isatty", return_value=stdout_tty), \
                     patch("main.sys.platform", platform), patch.dict("os.environ", {"TERM": term}), \
                     patch("main._run_simple_interactive") as fallback, \
                     patch("agent.cli_input.create_prompt_session") as factory:
                    run_interactive(resume_id="fixture-id", no_tools=True)
                    factory.assert_not_called()
                    fallback.assert_called_once_with(None, None, True, None, False, None, "fixture-id", None)

    def test_windows_interactive_startup_connects_new_prompt_session(self):
        with patch("main.sys.stdin.isatty", return_value=True), \
             patch("main.sys.stdout.isatty", return_value=True), \
             patch("main.sys.platform", "win32"), patch.dict("os.environ", {"TERM": "dumb"}), \
             patch("main.get_config", return_value=ConfigManager()), \
             patch("main._validate_runtime_config"), patch("main.Agent") as agent, \
             patch("main._configure_agent_extensions"), patch("main._make_renderer") as renderer, \
             patch("main._interactive_loop", new_callable=AsyncMock) as loop, \
             patch("agent.cli_input.create_prompt_session") as factory:
            run_interactive()
            factory.assert_called_once_with(color=renderer.return_value.color)
            self.assertIs(loop.await_args.kwargs["session"], factory.return_value)
            self.assertIs(loop.await_args.args[1], agent.return_value)

    def test_missing_prompt_toolkit_retains_simple_mode(self):
        original_import = builtins.__import__

        def without_ui(name, *args, **kwargs):
            if name == "agent.cli_input":
                raise ImportError("fixture: UI dependency missing")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=without_ui), \
             patch("main._run_simple_interactive") as fallback, patch("main.sys.stdout", new_callable=io.StringIO):
            run_interactive()
        fallback.assert_called_once()
