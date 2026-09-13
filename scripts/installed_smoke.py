"""Run with an installed wheel's venv Python -I, outside runtime source directories."""

from __future__ import annotations

import asyncio
import json
from importlib.metadata import version
from pathlib import Path
import sys
import tempfile

import main
import agent
import tools
from agent.llm import LLMClient
from config import ConfigManager


class OfflineClient(LLMClient):
    def __init__(self, api):
        super().__init__(base_url="https://example.invalid", api_key="offline", model="offline", api_type=api)
        self.api = api
        self.calls = 0

    async def _post_json(self, url, body):
        self.calls += 1
        if self.calls == 1:
            name, arguments = "update_plan", {"plan": [{"step": "Read fixture", "status": "in_progress"}]}
        elif self.calls == 2:
            name, arguments = "read_file", {"path": "sample.txt"}
        else:
            content = json.dumps(body)
            assert "installed fixture" in content
            if self.api == "chat":
                return {"choices": [{"message": {"role": "assistant", "content": "verified"}, "finish_reason": "stop"}]}
            return {"status": "completed", "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "verified"}]}]}
        arguments = json.dumps(arguments)
        if self.api == "chat":
            return {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [{"id": f"call-{self.calls}", "type": "function", "function": {"name": name, "arguments": arguments}}]}, "finish_reason": "tool_calls"}]}
        return {"status": "completed", "output": [{"type": "function_call", "call_id": f"call-{self.calls}", "name": name, "arguments": arguments}]}


async def exercise(root):
    config = ConfigManager()
    for section, key, value in (
        ("storage", "enabled", True), ("storage", "path", str(root / "sessions.db")),
        ("security", "workspace_root", str(root)), ("security", "approval_mode", "on_request"),
        ("agent", "refusal_review_enabled", False), ("agent", "repo_map_enabled", False),
        ("agent", "instructions_enabled", False), ("display", "show_tool_calls", False),
    ):
        config.override(section, key, value=value)
    config.override("toolsets", value=["file", "terminal"])
    (root / "sample.txt").write_text("installed fixture", encoding="utf-8")
    for api in ("chat", "responses"):
        runtime = agent.Agent(config)
        runtime.llm = OfflineClient(api)
        assert await runtime.run("read fixture with a plan") == "verified"
        restored = agent.Agent(config)
        restored.resume_session(runtime.session_id)
        assert restored.get_plan() == [{"step": "Read fixture", "status": "in_progress"}]
    script = root / "command.py"
    script.write_text("print('installed terminal')", encoding="utf-8")
    context = tools.ToolContext(root, frozenset({"terminal"}), approval_mode="on_request")
    result = json.loads(await tools.registry.dispatch_async(
        "terminal", {"command": f'"{sys.executable}" "{script}"', "timeout": 10}, context=context, approved=True,
    ))
    assert result["ok"] and "installed terminal" in result["content"], result


if __name__ == "__main__":
    for module in (main, agent, tools):
        assert Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), module.__file__
    assert version("drudge") == main.VERSION
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        asyncio.run(exercise(Path(directory).resolve()))
    print("Installed wheel: imports, CLI version, both APIs, persistent plans and terminal passed (offline).")
