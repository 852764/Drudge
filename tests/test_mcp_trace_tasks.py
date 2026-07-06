from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

from agent import Agent, AgentRuntime
from config import ConfigManager
from tests.fakes import FakeLLM, chat_response, function_call
from tools import ToolContext, create_tool_provider, registry


def mcp_config(workspace: str, *, storage: bool = False) -> ConfigManager:
    config = ConfigManager()
    config.override("storage", "enabled", value=storage)
    if storage:
        config.override("storage", "path", value=str(Path(workspace, "drudge.db")))
    config.override("display", "show_tool_calls", value=False)
    config.override("agent", "refusal_review_enabled", value=False)
    config.override("agent", "repo_map_enabled", value=False)
    config.override("security", "workspace_root", value=workspace)
    config.override("security", "approval_mode", value="auto")
    config.override("toolsets", value=["file"])
    config.override("mcp_servers", value={
        "demo": {
            "command": sys.executable,
            "args": [str(Path(__file__).parent / "fixtures" / "fake_mcp_server.py")],
            "risk": "medium",
            "timeout": 5,
        },
    })
    return config


class MCPTraceTaskTests(unittest.TestCase):
    def test_mcp_stdio_discovers_calls_and_closes(self):
        with tempfile.TemporaryDirectory() as workspace:
            provider = create_tool_provider(
                registry,
                [],
                mcp_config(workspace).get("mcp_servers"),
                workspace,
            )
            context = ToolContext.from_config(
                {"workspace_root": workspace, "approval_mode": "auto"},
                [],
            )

            async def exercise():
                await provider.start()
                tool_names = provider.tool_names()
                self.assertIn("mcp__demo__echo", provider.tool_names())
                result = await provider.call(
                    "mcp__demo__echo",
                    {"text": "hello"},
                    context,
                    approved=True,
                )
                status = provider.status()
                await provider.close()
                return result, status, tool_names

            result, status, tool_names = asyncio.run(exercise())

            payload = json.loads(result)
            self.assertTrue(payload["ok"])
            self.assertIn("echo:hello", payload["content"])
            self.assertIn("mcp__demo__list_resources", tool_names)
            self.assertIn("mcp__demo__get_prompt", tool_names)
            self.assertEqual(status["errors"], {})
            self.assertTrue(status["providers"][0]["connected"])

    def test_mcp_resources_prompts_and_auto_restart(self):
        with tempfile.TemporaryDirectory() as workspace:
            provider = create_tool_provider(
                registry,
                [],
                mcp_config(workspace).get("mcp_servers"),
                workspace,
            )
            context = ToolContext.from_config(
                {"workspace_root": workspace, "approval_mode": "auto"},
                [],
            )

            async def exercise():
                await provider.start()
                try:
                    resources = json.loads(await provider.call("mcp__demo__list_resources", {}, context, approved=True))
                    resource = json.loads(await provider.call("mcp__demo__read_resource", {"uri": "memory://readme"}, context, approved=True))
                    prompt = json.loads(await provider.call("mcp__demo__get_prompt", {"name": "summarize"}, context, approved=True))
                    process = provider.providers[1].process
                    process.kill()
                    await process.wait()
                    restarted = json.loads(await provider.call("mcp__demo__echo", {"text": "again"}, context, approved=True))
                    return resources, resource, prompt, restarted
                finally:
                    await provider.close()

            resources, resource, prompt, restarted = asyncio.run(exercise())

            self.assertTrue(resources["ok"])
            self.assertIn("memory://readme", resources["content"])
            self.assertIn("resource-body", resource["content"])
            self.assertIn("prompt:summarize", prompt["content"])
            self.assertIn("echo:again", restarted["content"])

    def test_mcp_medium_risk_requires_on_request_approval(self):
        with tempfile.TemporaryDirectory() as workspace:
            provider = create_tool_provider(
                registry,
                [],
                mcp_config(workspace).get("mcp_servers"),
                workspace,
            )
            context = ToolContext.from_config(
                {"workspace_root": workspace, "approval_mode": "on_request"},
                [],
            )

            async def exercise():
                await provider.start()
                result = await provider.call(
                    "mcp__demo__echo",
                    {"text": "blocked"},
                    context,
                    approved=False,
                )
                await provider.close()
                return result

            payload = json.loads(asyncio.run(exercise()))

            self.assertFalse(payload["ok"])
            self.assertTrue(payload["blocked"])
            self.assertTrue(payload["metadata"]["approval_required"])

    def test_mcp_tool_goes_through_agent_loop(self):
        with tempfile.TemporaryDirectory() as workspace:
            fake = FakeLLM([
                chat_response(
                    finish_reason="tool_calls",
                    tool_calls=[function_call(
                        "call-1",
                        "mcp__demo__echo",
                        '{"text":"from-agent"}',
                    )],
                ),
                chat_response("MCP complete"),
            ])
            agent = Agent(mcp_config(workspace))
            agent.llm = fake

            result = asyncio.run(agent.run("use MCP echo"))

            self.assertEqual(result, "MCP complete")
            tool_message = fake.requests[1]["messages"][-1]
            self.assertEqual(tool_message["role"], "tool")
            self.assertIn("echo:from-agent", tool_message["content"])

    def test_agent_runtime_reuses_one_mcp_process_across_turns(self):
        with tempfile.TemporaryDirectory() as workspace:
            fake = FakeLLM([
                chat_response(
                    finish_reason="tool_calls",
                    tool_calls=[function_call("call-1", "mcp__demo__echo", '{"text":"one"}')],
                ),
                chat_response("first complete"),
                chat_response(
                    finish_reason="tool_calls",
                    tool_calls=[function_call("call-2", "mcp__demo__echo", '{"text":"two"}')],
                ),
                chat_response("second complete"),
            ])
            agent = Agent(mcp_config(workspace))
            agent.llm = fake
            runtime = AgentRuntime(agent)

            async def exercise():
                async with runtime:
                    first = await runtime.run_turn("first")
                    self.assertTrue(agent.started)
                    second = await runtime.run_turn("second")
                    self.assertTrue(agent.started)
                    return first, second

            first, second = asyncio.run(exercise())

            self.assertEqual((first, second), ("first complete", "second complete"))
            first_tool = json.loads(fake.requests[1]["messages"][-1]["content"])
            second_tool = json.loads(fake.requests[3]["messages"][-1]["content"])
            first_pid = first_tool["content"].split(";pid:")[-1]
            second_pid = second_tool["content"].split(";pid:")[-1]
            self.assertEqual(first_pid, second_pid)
            self.assertFalse(agent.started)
            self.assertFalse(runtime.started)
            with self.assertRaisesRegex(RuntimeError, "closed"):
                asyncio.run(runtime.run_turn("too late"))

    def test_trace_and_tasks_survive_resume(self):
        with tempfile.TemporaryDirectory() as workspace:
            config = mcp_config(workspace, storage=True)
            config.override("mcp_servers", value={})
            fake = FakeLLM([
                chat_response(
                    finish_reason="tool_calls",
                    tool_calls=[function_call(
                        "task-call",
                        "task_create",
                        '{"title":"Implement parser","details":"Keep it resumable"}',
                    )],
                ),
                chat_response("Task recorded"),
            ])
            agent = Agent(config)
            agent.llm = fake

            result = asyncio.run(agent.run("plan the work"))
            session_id = agent.session_id

            self.assertEqual(result, "Task recorded")
            tasks = agent.list_tasks()
            self.assertEqual(tasks[0]["title"], "Implement parser")
            runs = agent.list_runs()
            self.assertEqual(runs[0]["status"], "completed")
            trace = agent.get_trace(runs[0]["id"])
            self.assertEqual(len(trace["model_calls"]), 2)
            self.assertTrue(any(event["kind"] == "tool_call" for event in trace["events"]))
            self.assertTrue(any(event["kind"] == "task_created" for event in trace["events"]))

            resumed = Agent(config)
            resumed.resume_session(session_id)
            self.assertEqual(resumed.list_tasks()[0]["title"], "Implement parser")
            updated = resumed.update_task(tasks[0]["id"], "completed")
            self.assertEqual(updated["status"], "completed")
            self.assertEqual(resumed.list_tasks(), [])
            self.assertEqual(resumed.list_tasks(include_closed=True)[0]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
