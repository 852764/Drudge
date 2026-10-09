from __future__ import annotations

import asyncio
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from agent import Agent, RunStatus
from agent.approval_menu import (
    ApprovalCancelled, create_approval_application, prompt_approval, request_details,
)
from agent.cli_renderer import CliRenderer
from config import ConfigManager
from main import ConsoleApproval, _interactive_loop
from tests.fakes import FakeLLM, chat_response, function_call
from tools.context import ToolContext
from tools.risk import ApprovalDecision, ApprovalRequest, RiskLevel, ToolRisk


def request(tool="terminal", level=RiskLevel.MEDIUM):
    return ApprovalRequest(tool, {"command": "echo fixture"}, ToolRisk(level, "Run a local command", "echo fixture"))


class TTYBuffer(io.StringIO):
    def isatty(self):
        return True


class ApprovalMenuTests(unittest.TestCase):
    def choose(self, keys):
        async def exercise():
            with create_pipe_input() as pipe:
                pipe.send_text(keys)
                return await asyncio.wait_for(prompt_approval(request(), input=pipe, output=DummyOutput()), 3)
        return asyncio.run(exercise())

    def test_enter_defaults_to_deny(self):
        self.assertEqual(self.choose("\r"), ApprovalDecision.DENY)

    def test_arrow_keys_choose_all_three_decisions(self):
        for keys, expected in (
            ("\x1b[A\x1b[A\r", ApprovalDecision.ALLOW_ONCE),
            ("\x1b[A\r", ApprovalDecision.ALLOW_SESSION),
            ("\x1b[A\x1b[B\r", ApprovalDecision.DENY),
            ("\x1b[B\r", ApprovalDecision.ALLOW_ONCE),
        ):
            with self.subTest(keys=repr(keys)):
                self.assertEqual(self.choose(keys), expected)

    def test_tab_and_number_selection_still_require_enter(self):
        for keys, expected in (("\t\r", ApprovalDecision.ALLOW_ONCE),
                               ("\x1b[Z\r", ApprovalDecision.ALLOW_SESSION),
                               ("1\r", ApprovalDecision.ALLOW_ONCE),
                               ("2\r", ApprovalDecision.ALLOW_SESSION),
                               ("3\r", ApprovalDecision.DENY)):
            with self.subTest(keys=repr(keys)):
                self.assertEqual(self.choose(keys), expected)

        async def exercise():
            with create_pipe_input() as pipe:
                app = create_approval_application(request(), input=pipe, output=DummyOutput())
                ready = asyncio.Event()
                task = asyncio.create_task(app.run_async(pre_run=ready.set))
                try:
                    await ready.wait()
                    pipe.send_text("1")
                    await asyncio.sleep(0.05)
                    self.assertFalse(task.done())
                    pipe.send_text("\r")
                    self.assertEqual(await asyncio.wait_for(task, 3), ApprovalDecision.ALLOW_ONCE)
                finally:
                    if not task.done():
                        task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
        asyncio.run(exercise())

    def test_escape_and_control_d_deny_after_selecting_allow(self):
        for keys in ("1\x1b", "2\x04"):
            with self.subTest(keys=repr(keys)):
                self.assertEqual(self.choose(keys), ApprovalDecision.DENY)

    def test_yes_text_and_bracketed_paste_do_not_grant_permission(self):
        self.assertEqual(self.choose("yes\r"), ApprovalDecision.DENY)
        self.assertEqual(self.choose("\x1b[200~1\n\x1b[A\r\x1b[201~\r"), ApprovalDecision.DENY)

    def test_control_c_cancels_instead_of_returning_an_approval(self):
        with self.assertRaises(ApprovalCancelled):
            self.choose("1\x03")

    def test_eof_denies_and_new_request_resets_selection(self):
        async def exercise():
            with create_pipe_input() as pipe:
                pipe.send_text("1\r")
                self.assertEqual(await prompt_approval(request(), input=pipe, output=DummyOutput()), ApprovalDecision.ALLOW_ONCE)
                pipe.send_text("\r")
                self.assertEqual(await prompt_approval(request(), input=pipe, output=DummyOutput()), ApprovalDecision.DENY)
            with create_pipe_input() as pipe:
                pipe.close()
                self.assertEqual(await asyncio.wait_for(prompt_approval(request(), input=pipe, output=DummyOutput()), 3), ApprovalDecision.DENY)
        asyncio.run(exercise())

    def test_external_cancellation_releases_input_for_the_next_menu(self):
        async def exercise():
            with create_pipe_input() as pipe:
                app = create_approval_application(request(), input=pipe, output=DummyOutput())
                ready = asyncio.Event()
                task = asyncio.create_task(app.run_async(pre_run=ready.set))
                await ready.wait()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertFalse(app.is_running)
                pipe.send_text("\r")
                self.assertEqual(await asyncio.wait_for(prompt_approval(request(), input=pipe, output=DummyOutput()), 3), ApprovalDecision.DENY)
        asyncio.run(exercise())

    def test_long_details_paging_does_not_approve(self):
        from prompt_toolkit.layout.controls import BufferControl
        async def exercise():
            item = ApprovalRequest("fixture", {}, ToolRisk(RiskLevel.MEDIUM, "fixture", "\n".join(f"line {i}" for i in range(100))))
            with create_pipe_input() as pipe:
                app = create_approval_application(item, input=pipe, output=DummyOutput())
                detail = next(control.buffer for control in app.layout.find_all_controls() if isinstance(control, BufferControl))
                pipe.send_text("\x1b[6~\r")
                self.assertEqual(await asyncio.wait_for(app.run_async(), 3), ApprovalDecision.DENY)
                self.assertGreater(detail.document.cursor_position_row, 0)
        asyncio.run(exercise())

    def test_details_redact_secret_fields_escape_controls_and_preserve_original(self):
        item = ApprovalRequest("fixture", {"nested": {"api_key": "fake-private", "text": "中文\x1b[2J\u202e"}},
                               ToolRisk(RiskLevel.MEDIUM, "fixture", "echo\rhidden"))
        details = request_details(item)
        self.assertNotIn("fake-private", details)
        self.assertNotIn("\x1b", details)
        self.assertNotIn("\u202e", details)
        self.assertIn("\\u202e", details)
        self.assertNotIn("\r", details)
        self.assertIn("中文", details)
        self.assertEqual(item.arguments["nested"]["api_key"], "fake-private")


class ConsoleApprovalTests(unittest.TestCase):
    def setUp(self):
        self.output = TTYBuffer()
        self.renderer = CliRenderer(stream=self.output, pretty=True, color=False, is_tty=True)

    def test_noninteractive_and_dumb_terminals_do_not_open_menu(self):
        for stdin, stdout, term in ((io.StringIO(), self.output, "xterm"),
                                    (TTYBuffer(), io.StringIO(), "xterm"),
                                    (TTYBuffer(), self.output, "dumb")):
            with self.subTest(term=term), patch("sys.platform", "linux"), patch("sys.stdin", stdin), patch("sys.stdout", stdout), patch("sys.stderr", io.StringIO()), patch.dict(os.environ, {"TERM": term}), patch("main.prompt_approval", new_callable=AsyncMock) as menu:
                result = asyncio.run(ConsoleApproval(self.renderer)(request()))
                self.assertEqual(result, ApprovalDecision.DENY)
                menu.assert_not_called()

    def test_windows_console_does_not_require_unix_term_capabilities(self):
        with patch("sys.platform", "win32"), patch("sys.stdin", TTYBuffer()), patch("sys.stdout", self.output), patch.dict(os.environ, {"TERM": "dumb"}), patch("main.prompt_approval", new=AsyncMock(return_value=ApprovalDecision.DENY)) as menu:
            self.assertEqual(asyncio.run(ConsoleApproval(self.renderer)(request())), ApprovalDecision.DENY)
        menu.assert_awaited_once()

    def test_status_rendering_is_suspended_and_restored_on_all_exit_paths(self):
        for outcome in (ApprovalDecision.ALLOW_ONCE, OSError("fixture"), ApprovalCancelled()):
            async def menu(*args, **kwargs):
                before = self.output.getvalue()
                self.renderer.show_status_line("should not render over menu")
                self.assertEqual(self.output.getvalue(), before)
                if isinstance(outcome, BaseException):
                    raise outcome
                return outcome

            with self.subTest(outcome=type(outcome).__name__), patch("sys.stdin", TTYBuffer()), patch("sys.stdout", self.output), patch("sys.stderr", io.StringIO()), patch.dict(os.environ, {"TERM": "xterm"}), patch("main.prompt_approval", side_effect=menu):
                self.renderer.show_status_line("before")
                if isinstance(outcome, ApprovalCancelled):
                    with self.assertRaises(ApprovalCancelled):
                        asyncio.run(ConsoleApproval(self.renderer)(request()))
                else:
                    result = asyncio.run(ConsoleApproval(self.renderer)(request()))
                    expected = ApprovalDecision.DENY if isinstance(outcome, Exception) else outcome
                    self.assertEqual(result, expected)
                self.renderer.show_status_line("resumed")
                self.assertIn("resumed", self.output.getvalue())
                self.assertEqual(self.renderer._status_suspensions, 0)

    def test_ticker_waits_while_menu_owns_terminal(self):
        async def exercise():
            ticker = self.renderer.make_activity_ticker()
            task = asyncio.create_task(ticker.run())
            try:
                await asyncio.sleep(0.03)
                with self.renderer.suspend_status():
                    with self.renderer.suspend_status():
                        before = self.output.getvalue()
                        await asyncio.sleep(0.25)
                        self.assertEqual(before, self.output.getvalue())
                    self.assertEqual(self.renderer._status_suspensions, 1)
                await asyncio.sleep(0.15)
                self.assertGreater(len(self.output.getvalue()), len(before))
            finally:
                ticker.stop()
                await task
        asyncio.run(exercise())

    def test_session_permission_stays_scoped_to_tool_and_risk(self):
        config = ConfigManager()
        config.override("storage", "enabled", value=False)
        config.override("security", "approval_mode", value="on_request")
        callback = AsyncMock(return_value=ApprovalDecision.ALLOW_SESSION)
        agent = Agent(config, approval_callback=callback)
        agent.tool_context = ToolContext.from_config(config.get_security_config(), config.get_toolsets())
        async def exercise():
            with patch.object(agent.tool_provider, "assess_risk", return_value=ToolRisk(RiskLevel.MEDIUM, "fixture", "fixture")):
                self.assertTrue(await agent._approve_tool("fixture-a", {}))
                self.assertTrue(await agent._approve_tool("fixture-a", {"other": "arguments"}))
                self.assertEqual(callback.await_count, 1)
                await agent._approve_tool("fixture-b", {})
                self.assertEqual(callback.await_count, 2)
            with patch.object(agent.tool_provider, "assess_risk", return_value=ToolRisk(RiskLevel.HIGH, "fixture", "fixture")):
                await agent._approve_tool("fixture-a", {})
                self.assertEqual(callback.await_count, 3)
            with patch.object(agent.tool_provider, "assess_risk", return_value=ToolRisk(RiskLevel.CRITICAL, "fixture", "fixture")):
                self.assertFalse(await agent._approve_tool("fixture-a", {}))
                self.assertEqual(callback.await_count, 3)
        asyncio.run(exercise())

    def test_menu_cancel_returns_interactive_cli_to_prompt_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            config = ConfigManager()
            config.override("storage", "enabled", value=False)
            config.override("security", "workspace_root", value=directory)
            config.override("security", "approval_mode", value="on_request")
            config.override("agent", "repo_map_enabled", value=False)
            config.override("agent", "instructions_enabled", value=False)
            config.override("display", "show_cost", value=False)
            config.override("toolsets", value=["file"])
            agent = Agent(config, approval_callback=AsyncMock(side_effect=ApprovalCancelled()))
            agent.llm = FakeLLM([chat_response(finish_reason="tool_calls", tool_calls=[
                function_call("call-1", "write_file", '{"path":"should-not-exist","content":"fixture"}'),
            ])])
            session = AsyncMock()
            session.prompt_async.side_effect = ["write a file", "/quit"]
            asyncio.run(_interactive_loop(config, agent, renderer=self.renderer, session=session))
            self.assertEqual(session.prompt_async.await_count, 2)
            self.assertEqual(agent.run_state.status, RunStatus.CANCELLED)
            self.assertFalse(Path(directory, "should-not-exist").exists())
            self.assertIn("Ready for the next prompt", self.output.getvalue())


if __name__ == "__main__":
    unittest.main()
