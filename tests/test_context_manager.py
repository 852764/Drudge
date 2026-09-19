from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.context_manager import (
    build_context_summary_messages,
    build_repo_map,
    compact_messages,
    fit_compaction_summary,
    partition_messages_for_compaction,
    summarize_messages,
)
from agent.llm import LLMClient


class ContextManagerTests(unittest.TestCase):
    def test_repo_map_excludes_private_dirs(self):
        with tempfile.TemporaryDirectory() as workspace:
            Path(workspace, "agent.py").write_text("print('ok')", encoding="utf-8")
            hidden = Path(workspace, ".drudge")
            hidden.mkdir()
            Path(hidden, "auth.json").write_text("secret", encoding="utf-8")

            repo_map = build_repo_map(workspace)

            self.assertIn("agent.py", repo_map)
            self.assertNotIn("auth.json", repo_map)

    def test_compact_messages_keeps_system_and_recent(self):
        messages = [{"role": "system", "content": "sys"}]
        for index in range(12):
            messages.append({"role": "user", "content": f"question {index}"})

        compacted = compact_messages(messages, keep_recent=4)

        self.assertEqual(compacted[0]["role"], "system")
        self.assertIn("Previous conversation summary", compacted[1]["content"])
        self.assertEqual(len(compacted), 6)
        self.assertEqual(compacted[-1]["content"], "question 11")

    def test_summarize_messages_mentions_tool_errors(self):
        summary = summarize_messages([
            {"role": "user", "content": "please inspect"},
            {"role": "tool", "content": '{"error":"failed"}'},
        ])

        self.assertIn("User:", summary)
        self.assertIn("Tool error", summary)

    def test_compaction_keeps_complete_tool_transaction(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "result"},
            {"role": "user", "content": "latest"},
        ]

        compacted = compact_messages(messages, keep_recent=2)

        roles = [message["role"] for message in compacted]
        self.assertEqual(roles[-3:], ["assistant", "tool", "user"])
        self.assertEqual(compacted[-2]["tool_call_id"], "call-1")

    def test_summary_request_keeps_tool_semantics_but_omits_provider_state(self):
        request = build_context_summary_messages([
            {
                "role": "assistant",
                "content": "checking",
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"a.txt"}'},
                }],
                "provider_items": [{"type": "reasoning", "encrypted_content": "secret"}],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "file contents"},
        ])

        transcript = request[1]["content"]
        self.assertIn("read_file", transcript)
        self.assertIn("tool_call_id", transcript)
        self.assertNotIn("provider_items", transcript)
        self.assertNotIn("encrypted_content", transcript)

    def test_fallback_keeps_original_requirement_and_latest_correction(self):
        messages = [{"role": "user", "content": "initial constraint"}] + [
            {"role": "assistant", "content": f"step {index}"} for index in range(30)
        ] + [{"role": "user", "content": "latest correction"}]
        summary = summarize_messages(messages, max_items=4)
        self.assertIn("initial constraint", summary)
        self.assertIn("latest correction", summary)
        self.assertIn("step 29", summary)
        self.assertNotIn("step 0", summary)
        self.assertEqual(len(summary.splitlines()), 4)

    def test_fallback_distinguishes_success_envelope_from_tool_error(self):
        summary = summarize_messages([
            {"role": "tool", "content": '{"ok":true,"error":null,"content":"success"}'},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_file"}}]},
        ])
        self.assertIn("Tool result:", summary)
        self.assertNotIn("Tool error:", summary)
        self.assertIn("read_file", summary)

    def test_compaction_rejects_non_positive_recent_window(self):
        for count in (0, -1):
            with self.assertRaisesRegex(ValueError, "at least 1"):
                partition_messages_for_compaction([{"role": "user", "content": "latest"}], keep_recent=count)

    def test_fit_compaction_summary_preserves_latest_tool_transaction(self):
        system = [{"role": "system", "content": "instructions"}]
        summary = "historical detail " * 500
        recent = [
            {"role": "user", "content": "latest request"},
            {"role": "assistant", "content": "checking", "tool_calls": [{
                "id": "call-1", "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }]},
            {"role": "tool", "tool_call_id": "call-1", "content": "latest file result"},
        ]
        protected = LLMClient.estimate_tokens(system + recent)

        fitted, truncated = fit_compaction_summary(
            system,
            summary,
            recent,
            max_tokens=protected + 80,
            estimate_tokens=LLMClient.estimate_tokens,
        )

        self.assertTrue(truncated)
        self.assertLessEqual(LLMClient.estimate_tokens(fitted), protected + 80)
        self.assertEqual([item["role"] for item in fitted[-3:]], ["user", "assistant", "tool"])
        self.assertEqual(fitted[-1]["tool_call_id"], "call-1")
        self.assertLess(len(fitted[1]["content"]), len(summary))


if __name__ == "__main__":
    unittest.main()
