from __future__ import annotations

import asyncio
import io
import sqlite3
import unittest
from unittest.mock import Mock

from agent.cli_renderer import CliRenderer
from config import ConfigManager
from main import _display_output_text, _handle_command


class OutputCliTests(unittest.TestCase):
    output_id = "a" * 32

    def render(self, command, agent):
        stream = io.StringIO()
        result = asyncio.run(_handle_command(
            command, ConfigManager(), agent,
            renderer=CliRenderer(stream=stream, pretty=False),
        ))
        self.assertFalse(result)
        return stream.getvalue()

    def page(self, **overrides):
        return {
            "id": self.output_id, "content": "page body", "offset": 0,
            "next_offset": 9, "char_count": 100, "complete": True, "eof": False,
            **overrides,
        }

    def test_outputs_lists_complete_and_partial_captures(self):
        agent = Mock()
        agent.list_tool_outputs.return_value = [
            {"id": self.output_id, "tool_name": "terminal.stdout", "size_bytes": 5000,
             "complete": True, "status": "completed"},
            {"id": "b" * 32, "tool_name": "terminal.stderr", "size_bytes": 200,
             "complete": False, "status": "timed_out"},
        ]
        output = self.render("/outputs", agent)
        agent.list_tool_outputs.assert_called_once_with()
        self.assertIn(self.output_id, output)
        self.assertIn("5000 B", output)
        self.assertIn("complete (completed)", output)
        self.assertIn("partial (timed_out)", output)
        agent.read_tool_output.assert_not_called()

    def test_output_default_and_explicit_pagination(self):
        for suffix, offset, limit in (("", 0, 1000), (" 25", 25, 1000), (" 25 100", 25, 100)):
            with self.subTest(suffix=suffix):
                agent = Mock()
                agent.read_tool_output.return_value = self.page(offset=offset, next_offset=offset + 9)
                output = self.render(f"/output {self.output_id}{suffix}", agent)
                agent.read_tool_output.assert_called_once_with(self.output_id, offset=offset, limit=limit)
                self.assertIn("page body", output)
                self.assertIn(f"Next: /output {self.output_id} {offset + 9} {limit}", output)

    def test_partial_eof_is_not_reported_as_complete_output(self):
        agent = Mock()
        agent.read_tool_output.return_value = self.page(complete=False, eof=True)
        output = self.render(f"/output {self.output_id}", agent)
        self.assertIn("Captured output is partial", output)
        self.assertIn("EOF only marks the end of stored text", output)
        self.assertNotIn("Next:", output)

    def test_control_characters_are_displayed_literally_without_mutating_page(self):
        content = "\x1b[2Jabc\x00\r\b\x9b\n\tZ"
        agent = Mock()
        page = self.page(content=content)
        agent.read_tool_output.return_value = page
        output = self.render(f"/output {self.output_id}", agent)
        self.assertNotIn("\x1b", output)
        self.assertNotIn("\x00", output)
        self.assertIn("\\x1b[2J", output)
        self.assertIn("\\x00", output)
        self.assertEqual(page["content"], content)
        self.assertEqual(_display_output_text("line\n\ttab"), "line\n\ttab")

    def test_invalid_arguments_do_not_access_output_storage(self):
        for command in (
            "/outputs extra", "/output", f"/output {self.output_id} 0 10 extra",
            f"/output {self.output_id} -1", f"/output {self.output_id} 0 1001",
            f"/output {self.output_id} 0 0", f"/output {self.output_id} invalid",
            f"/output {self.output_id} 0 invalid",
        ):
            with self.subTest(command=command):
                agent = Mock()
                self.assertTrue(self.render(command, agent))
                agent.read_tool_output.assert_not_called()
                agent.list_tool_outputs.assert_not_called()

    def test_storage_errors_do_not_exit_interactive_session(self):
        for error in (KeyError("output missing"), RuntimeError("storage disabled"), sqlite3.OperationalError("database busy")):
            for command, method in (("/outputs", "list_tool_outputs"), (f"/output {self.output_id}", "read_tool_output")):
                with self.subTest(error=type(error).__name__, command=command):
                    agent = Mock()
                    getattr(agent, method).side_effect = error
                    self.assertIn(str(error), self.render(command, agent))

    def test_empty_outputs_and_unavailable_agent_are_explicit(self):
        agent = Mock()
        agent.list_tool_outputs.return_value = []
        self.assertIn("No captured outputs", self.render("/outputs", agent))
        self.assertIn("Agent is unavailable", self.render("/outputs", None))
        self.assertIn("Agent is unavailable", self.render(f"/output {self.output_id}", None))

    def test_help_lists_output_commands(self):
        output = self.render("/help", Mock())
        self.assertIn("/outputs", output)
        self.assertIn("/output <id>", output)
