from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from agent import Agent
from agent.tool_selector import parse_tool_selection, rank_tool_catalog
from config import ConfigManager
from tests.fakes import FakeLLM, chat_response, function_call


def selection_config(workspace: str, *, threshold: int = 1) -> ConfigManager:
    config = ConfigManager()
    config.override("storage", "enabled", value=False)
    config.override("display", "show_tool_calls", value=False)
    config.override("agent", "refusal_review_enabled", value=False)
    config.override("agent", "repo_map_enabled", value=False)
    config.override("security", "workspace_root", value=workspace)
    config.override("toolsets", value=["terminal", "file", "web"])
    config.override("utility_model", value={"name": "cheap-selector"})
    config.override("tool_selection", "min_tools", value=threshold)
    config.override("tool_selection", "min_schema_tokens", value=999999)
    config.override("tool_selection", "max_selected", value=3)
    return config


class ToolSelectionTests(unittest.TestCase):
    def test_parser_rejects_unknown_names_and_caps_selection(self):
        catalog = [
            {"name": "read_file"},
            {"name": "terminal"},
            {"name": "web_request"},
        ]

        selection = parse_tool_selection(
            '{"tools":["read_file","invented","terminal","web_request"],"reason":"files"}',
            catalog,
            max_selected=2,
        )

        self.assertEqual(selection.names, ["read_file", "terminal"])

    def test_deterministic_ranking_understands_categories(self):
        catalog = [
            {"name": "terminal", "description": "Run a command", "category": "terminal", "risk": "high"},
            {"name": "read_file", "description": "Read a file", "category": "file", "risk": "low"},
        ]

        selected = rank_tool_catalog("运行测试命令", catalog, limit=1)

        self.assertEqual(selected[0]["name"], "terminal")

    def test_auxiliary_model_selects_small_schema_set(self):
        with tempfile.TemporaryDirectory() as workspace:
            agent = Agent(selection_config(workspace))
            primary = FakeLLM([chat_response("done")])
            selector = FakeLLM([
                chat_response('{"tools":["read_file"],"reason":"inspect one file"}'),
            ])
            selector.model = "cheap-selector"
            agent.llm = primary
            agent.utility_llm = selector

            result = asyncio.run(agent.run("inspect the file"))

            self.assertEqual(result, "done")
            names = [tool["function"]["name"] for tool in primary.requests[0]["tools"]]
            self.assertEqual(names, ["read_file", "tool_search"])
            self.assertIsNone(selector.requests[0]["tools"])
            self.assertNotIn("inputSchema", selector.requests[0]["messages"][1]["content"])
            self.assertEqual(agent.get_token_usage()["total_tokens"], 20)
            self.assertEqual(agent.get_token_usage()["utility_tokens"], 10)
            self.assertEqual(agent.get_status()["last_tool_selection"]["mode"], "llm")

    def test_selector_failure_uses_deterministic_fallback(self):
        with tempfile.TemporaryDirectory() as workspace:
            agent = Agent(selection_config(workspace))
            primary = FakeLLM([chat_response("done")])
            agent.llm = primary
            agent.utility_llm = FakeLLM([RuntimeError("selector offline")])

            result = asyncio.run(agent.run("run the tests in terminal"))

            self.assertEqual(result, "done")
            names = [tool["function"]["name"] for tool in primary.requests[0]["tools"]]
            self.assertIn("terminal", names)
            selection = agent.get_status()["last_tool_selection"]
            self.assertEqual(selection["mode"], "fallback")
            self.assertIn("selector offline", selection["fallback_reason"])

    def test_tool_search_activates_tools_for_next_model_call(self):
        with tempfile.TemporaryDirectory() as workspace:
            agent = Agent(selection_config(workspace))
            primary = FakeLLM([
                chat_response(
                    finish_reason="tool_calls",
                    tool_calls=[function_call(
                        "search-1",
                        "tool_search",
                        '{"query":"run terminal command","limit":1}',
                    )],
                ),
                chat_response("found it"),
            ])
            selector = FakeLLM([
                chat_response('{"tools":["read_file"],"reason":"start with inspection"}'),
            ])
            selector.model = "cheap-selector"
            agent.llm = primary
            agent.utility_llm = selector

            result = asyncio.run(agent.run("inspect then execute if needed"))

            self.assertEqual(result, "found it")
            first_names = [
                tool["function"]["name"] for tool in primary.requests[0]["tools"]
            ]
            second_names = [
                tool["function"]["name"] for tool in primary.requests[1]["tools"]
            ]
            self.assertNotIn("terminal", first_names)
            self.assertIn("terminal", second_names)
            self.assertIn("tool_search", second_names)
            self.assertIn("activated", primary.requests[1]["messages"][-1]["content"])

    def test_below_threshold_sends_all_without_tool_search_or_selector_call(self):
        with tempfile.TemporaryDirectory() as workspace:
            agent = Agent(selection_config(workspace, threshold=100))
            primary = FakeLLM([chat_response("done")])
            selector = FakeLLM([])
            agent.llm = primary
            agent.utility_llm = selector

            asyncio.run(agent.run("simple request"))

            names = [tool["function"]["name"] for tool in primary.requests[0]["tools"]]
            self.assertIn("terminal", names)
            self.assertNotIn("tool_search", names)
            self.assertEqual(selector.requests, [])
            self.assertEqual(agent.get_status()["last_tool_selection"]["mode"], "all")

    def test_selection_decision_and_model_cost_are_persisted_in_trace(self):
        with tempfile.TemporaryDirectory() as workspace:
            config = selection_config(workspace)
            config.override("storage", "enabled", value=True)
            config.override("storage", "path", value=str(Path(workspace, "drudge.db")))
            agent = Agent(config)
            agent.llm = FakeLLM([chat_response("done")])
            selector = FakeLLM([
                chat_response('{"tools":["read_file"],"reason":"inspect"}'),
            ])
            selector.model = "cheap-selector"
            agent.utility_llm = selector

            asyncio.run(agent.run("inspect source"))

            trace = agent.get_trace()
            self.assertTrue(
                any(event["kind"] == "tool_selection" for event in trace["events"])
            )
            selector_calls = [
                call
                for call in trace["model_calls"]
                if call["purpose"] == "tool_selection"
            ]
            self.assertEqual(len(selector_calls), 1)
            self.assertEqual(selector_calls[0]["model"], "cheap-selector")
            self.assertEqual(selector_calls[0]["total_tokens"], 10)


if __name__ == "__main__":
    unittest.main()
