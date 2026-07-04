from __future__ import annotations

import asyncio
import io
import unittest

from agent.cli_renderer import CliRenderer


class _TTYBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


class CliRendererTests(unittest.TestCase):
    def test_banner_renders_model_and_toolsets(self):
        stream = _TTYBuffer()
        renderer = CliRenderer(stream=stream, pretty=True, color=False, width=72, is_tty=True)

        renderer.print_banner(
            version="0.1.0",
            model="gpt-test",
            toolsets=["file", "web"],
            codex_config_path="C:/demo/config.toml",
        )

        output = stream.getvalue()
        self.assertIn("Drudge v0.1.0", output)
        self.assertIn("Model: gpt-test", output)
        self.assertIn("Toolsets: file, web", output)
        self.assertIn("Codex config: C:/demo/config.toml", output)

    def test_stream_printer_pretty_mode_keeps_only_final_panel(self):
        stream = _TTYBuffer()
        renderer = CliRenderer(stream=stream, pretty=True, color=False, width=72, is_tty=True)
        printer = renderer.make_stream_printer("Assistant")

        printer("## Title\n")
        printer("- item")
        printer.finish("## Title\n\n- item")

        output = stream.getvalue()
        self.assertIn("+- Assistant", output)
        self.assertIn("- item", output)
        self.assertNotIn("AI>", output)

    def test_activity_ticker_prints_elapsed_status(self):
        stream = _TTYBuffer()
        renderer = CliRenderer(stream=stream, pretty=True, color=False, width=72, is_tty=True)
        ticker = renderer.make_activity_ticker("Thinking")

        async def run_ticker() -> None:
            task = asyncio.create_task(ticker.run())
            await asyncio.sleep(0.05)
            ticker.stop()
            await asyncio.gather(task, return_exceptions=True)

        asyncio.run(run_ticker())

        output = stream.getvalue()
        self.assertIn("Thinking 0s", output)

    def test_status_renders_named_sections(self):
        stream = _TTYBuffer()
        renderer = CliRenderer(stream=stream, pretty=True, color=False, width=72, is_tty=True)

        renderer.print_status(
            {
                "local": {
                    "session_id": "abc",
                    "run_status": "completed",
                    "runtime_started": True,
                    "model": "gpt-test",
                    "provider": "openai-compatible",
                    "model_api": "responses",
                    "utility_model": "cheap",
                    "utility_model_configured": True,
                    "turns": 3,
                    "message_count": 8,
                    "tokens_this_process": 123,
                    "utility_tokens_this_process": 12,
                    "workspace": "F:/Drudge",
                    "approval_mode": "auto",
                    "active_skills": ["review"],
                    "open_tasks": 2,
                    "mcp_servers": ["demo"],
                    "context_limit": 1000,
                    "estimated_context_tokens": 250,
                    "context_used_percent": 25.0,
                    "last_tool_selection": {
                        "mode": "llm",
                        "selected": ["read_file"],
                        "catalog_tools": 10,
                        "schema_tokens": 200,
                    },
                },
                "account_usage": None,
                "account_usage_error": None,
            }
        )

        output = stream.getvalue()
        self.assertIn("Runtime", output)
        self.assertIn("Context", output)
        self.assertIn("Tool Selection", output)
        self.assertIn("Codex Account Usage", output)

    def test_tool_event_renders_indented_sidebar_panel(self):
        stream = _TTYBuffer()
        renderer = CliRenderer(stream=stream, pretty=True, color=False, width=96, is_tty=True)

        renderer.print_tool_event("read_file", {"path": "demo.py"}, '{"ok": true, "content": "print(1)"}')

        output = stream.getvalue()
        self.assertIn("Tool Log", output)
        self.assertIn("tool: read_file", output)
        self.assertIn("args:", output)
        self.assertIn("result:", output)

    def test_code_block_uses_ansi_highlighting_when_color_enabled(self):
        stream = _TTYBuffer()
        renderer = CliRenderer(stream=stream, pretty=True, color=True, width=72, is_tty=True)

        renderer.print_assistant_message("```python\ndef answer(x):\n    return 42\n```")

        output = stream.getvalue()
        self.assertIn("\x1b[95;1mdef\x1b[0m", output)
        self.assertIn("\x1b[95;1mreturn\x1b[0m 42", output)


if __name__ == "__main__":
    unittest.main()
