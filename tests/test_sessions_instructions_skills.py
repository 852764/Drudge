from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from agent import Agent
from agent.llm import LLMClient
from agent.project_instructions import load_project_instructions
from agent.storage import ConversationStore
from config import ConfigManager
from tests.fakes import FakeLLM, chat_response, function_call


def configured(workspace: str, db_path: str) -> ConfigManager:
    config = ConfigManager()
    config.override("storage", "enabled", value=True)
    config.override("storage", "path", value=db_path)
    config.override("display", "show_tool_calls", value=False)
    config.override("agent", "refusal_review_enabled", value=False)
    config.override("security", "workspace_root", value=workspace)
    config.override("toolsets", value=["file"])
    return config


class SessionInstructionSkillTests(unittest.TestCase):
    def test_resumed_provider_items_are_not_sent_to_chat_api(self):
        converted = LLMClient._messages_to_chat_input([{
            "role": "assistant",
            "content": "done",
            "provider_items": [{"type": "reasoning", "encrypted_content": "secret"}],
        }])

        self.assertEqual(converted, [{"role": "assistant", "content": "done"}])

    def test_sqlite_session_resume_continues_same_conversation(self):
        with tempfile.TemporaryDirectory() as workspace:
            db_path = str(Path(workspace, "drudge.db"))
            config = configured(workspace, db_path)
            first = Agent(config)
            first.llm = FakeLLM([chat_response("first answer")])
            asyncio.run(first.run("first question"))
            session_id = first.session_id

            resumed = Agent(config)
            resumed.llm = FakeLLM([chat_response("second answer")])
            info = resumed.resume_session(session_id)
            result = asyncio.run(resumed.run("second question"))

            self.assertEqual(result, "second answer")
            self.assertEqual(resumed.session_id, session_id)
            self.assertEqual(info["message_count"], 3)
            request = resumed.llm.requests[0]["messages"]
            contents = [message.get("content") for message in request]
            self.assertIn("first question", contents)
            self.assertIn("first answer", contents)
            self.assertIn("second question", contents)
            self.assertEqual(len(resumed.store.list_sessions()), 1)

    def test_resume_repairs_interrupted_tool_transaction(self):
        with tempfile.TemporaryDirectory() as workspace:
            db_path = str(Path(workspace, "drudge.db"))
            store = ConversationStore(db_path)
            session_id = store.create_session("interrupted", "fake")
            store.append_message(session_id, "system", "system")
            store.append_message(session_id, "user", "read")
            call = function_call("call-1", "read_file", '{"path":"sample.txt"}')
            store.append_message(
                session_id,
                "assistant",
                "",
                metadata={"tool_calls": [call]},
            )

            agent = Agent(configured(workspace, db_path))
            info = agent.resume_session(session_id)

            self.assertEqual(info["repaired_tool_calls"], 1)
            repaired = agent.get_messages()[-1]
            self.assertEqual(repaired["role"], "tool")
            self.assertEqual(repaired["tool_call_id"], "call-1")
            payload = json.loads(repaired["content"])
            self.assertTrue(payload["metadata"]["interrupted"])

            second = Agent(configured(workspace, db_path))
            second_info = second.resume_session(session_id)
            self.assertEqual(second_info["repaired_tool_calls"], 0)

    def test_old_sqlite_schema_is_migrated(self):
        with tempfile.TemporaryDirectory() as workspace:
            db_path = Path(workspace, "legacy.db")
            connection = sqlite3.connect(db_path)
            connection.execute(
                """
                CREATE TABLE sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    model TEXT NOT NULL,
                    cwd TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.commit()
            connection.close()

            store = ConversationStore(str(db_path))
            session_id = store.create_session("migrated", "fake", metadata={"active_skills": []})

            self.assertEqual(store.get_session(session_id)["metadata"], {"active_skills": []})

    def test_agents_md_loads_root_to_leaf(self):
        with tempfile.TemporaryDirectory() as workspace:
            root = Path(workspace)
            nested = root / "src" / "feature"
            nested.mkdir(parents=True)
            (root / "AGENTS.md").write_text("root rule", encoding="utf-8")
            (root / "src" / "AGENTS.md").write_text("src rule", encoding="utf-8")

            loaded = load_project_instructions(root, cwd=nested)

            self.assertEqual([item.content for item in loaded], ["root rule", "src rule"])
            self.assertEqual(loaded[1].scope, root / "src")

    def test_agents_md_is_injected_into_system_prompt(self):
        with tempfile.TemporaryDirectory() as workspace:
            Path(workspace, "AGENTS.md").write_text("Always run the local verifier.", encoding="utf-8")
            config = configured(workspace, str(Path(workspace, "drudge.db")))
            config.override("storage", "enabled", value=False)
            agent = Agent(config)
            agent.llm = FakeLLM([chat_response("done")])

            asyncio.run(agent.run("work"))

            system = agent.llm.requests[0]["messages"][0]["content"]
            self.assertIn("PROJECT INSTRUCTIONS", system)
            self.assertIn("Always run the local verifier.", system)

    def test_skill_discovery_activation_and_resume(self):
        with tempfile.TemporaryDirectory() as workspace:
            skill_dir = Path(workspace, ".drudge", "skills", "review")
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                """---
name: review
description: Review source changes
---
Check correctness and run focused tests.
""",
                encoding="utf-8",
            )
            db_path = str(Path(workspace, "drudge.db"))
            config = configured(workspace, db_path)
            agent = Agent(config)
            skill = agent.activate_skill("review")
            agent.llm = FakeLLM([chat_response("reviewed")])
            asyncio.run(agent.run("review this"))

            self.assertEqual(skill.description, "Review source changes")
            system = agent.llm.requests[0]["messages"][0]["content"]
            self.assertIn("LOADED SKILLS", system)
            self.assertIn("Check correctness and run focused tests.", system)

            resumed = Agent(config)
            info = resumed.resume_session(agent.session_id)
            self.assertEqual(info["active_skills"], ["review"])
            self.assertIn("review", resumed.active_skill_names)


if __name__ == "__main__":
    unittest.main()
