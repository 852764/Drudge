from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock, patch

from agent import Agent
from agent.llm import LLMClient
from tests.fakes import chat_response, function_call
from tests.test_tool_outputs import OutputFixture
from tools import registry
from tools.file_ops import read_files_handler


class BatchReadTests(OutputFixture):
    def call(self, **args):
        return json.loads(registry.dispatch("read_files", args, context=self.agent.tool_context))

    def test_schema_includes_array_shape_and_bounds(self):
        schema = next(item["function"]["parameters"] for item in registry.get_schemas(["file"]) if item["function"]["name"] == "read_files")
        self.assertEqual(schema["required"], ["paths"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["paths"]["items"], {"type": "string"})
        self.assertEqual(schema["properties"]["paths"]["maxItems"], 8)
        self.assertEqual(schema["properties"]["limit"]["maximum"], 500)

    def test_independent_files_keep_order_hashes_and_line_metadata(self):
        sources = {"a.txt": b"first\r\nsecond\r\nlast", "b.txt": "中文\nnext\n".encode("utf-8")}
        for name, data in sources.items():
            (self.root / name).write_bytes(data)
        result = self.call(paths=list(sources), offset=2, limit=1)
        self.assertTrue(result["ok"])
        self.assertEqual(result["metadata"], {"files_read": 2, "files_failed": 0})
        items = json.loads(result["content"])
        self.assertEqual([item["path"] for item in items], list(sources))
        for item in items:
            self.assertTrue(item["ok"])
            self.assertEqual(item["sha256"], hashlib.sha256(sources[item["path"]]).hexdigest())
            self.assertEqual(item["offset"], 2)
            self.assertEqual(item["shown_lines"], 1)
        self.assertEqual(items[0]["content"], "2|second")

    def test_partial_failure_retains_successful_files(self):
        (self.root / "exists.txt").write_text("readable", encoding="utf-8")
        result = self.call(paths=["missing.txt", "exists.txt"])
        self.assertFalse(result["ok"])
        self.assertEqual(result["metadata"], {"files_read": 1, "files_failed": 1})
        missing, present = json.loads(result["content"])
        self.assertFalse(missing["ok"])
        self.assertIn("not found", missing["error"].lower())
        self.assertTrue(present["ok"])
        self.assertIn("readable", present["content"])

    def test_invalid_full_input_is_rejected_before_any_io(self):
        invalid = [
            {"paths": []}, {"paths": ["a"] * 9}, {"paths": "a"},
            {"paths": ["good.txt", 1]}, {"paths": [" "]}, {"paths": ["a" * 4097]},
            {"paths": ["a"], "offset": True}, {"paths": ["a"], "offset": 0},
            {"paths": ["a"], "limit": True}, {"paths": ["a"], "limit": 501},
        ]
        with patch("tools.file_ops.read_file_handler") as read:
            for args in invalid:
                with self.subTest(args=str(args)[:100]):
                    self.assertFalse(read_files_handler(context=self.agent.tool_context, **args).ok)
        read.assert_not_called()

    def test_each_path_obeys_host_policy_without_opening_credentials(self):
        (self.root / "public.txt").write_text("public", encoding="utf-8")
        reads = []
        original = Path.read_bytes
        def observe(path):
            reads.append(path)
            return original(path)
        with patch.object(Path, "read_bytes", observe):
            result = self.call(paths=[".drudge/auth.json", ".codex/auth.json", "../outside.txt", "public.txt"])
        items = json.loads(result["content"])
        self.assertEqual(reads, [self.root / "public.txt"])
        self.assertFalse(result["ok"])
        self.assertTrue(all(item["blocked"] for item in items[:3]))
        self.assertTrue(items[-1]["ok"])

    def test_model_context_override_is_rejected_and_read_only_mode_still_reads(self):
        (self.root / "public.txt").write_text("public", encoding="utf-8")
        with patch("tools.file_ops.read_file_handler") as read:
            result = self.call(paths=["public.txt"], context={"allow_outside_workspace": True})
            self.assertTrue(result["blocked"])
            read.assert_not_called()
        result = json.loads(registry.dispatch("read_files", {"paths": ["public.txt"]}, context=replace(self.agent.tool_context, approval_mode="never")))
        self.assertTrue(result["ok"])

    def test_long_item_is_bounded_and_receipt_survives_session_resume(self):
        source = "HEAD-" + "x" * 30000 + "-TAIL"
        (self.root / "large.txt").write_text(source, encoding="utf-8")
        result = self.call(paths=["large.txt"])
        item = json.loads(result["content"])[0]
        self.assertLessEqual(len(json.dumps({key: value for key, value in item.items() if key != "path"}, ensure_ascii=False)), 6000)
        self.assertTrue(item["metadata"]["truncated"])
        self.assertEqual(item["metadata"]["sha256"], hashlib.sha256(source.encode()).hexdigest())
        receipt = item["metadata"]["output_ref"]
        resumed = Agent(self.config)
        resumed.resume_session(self.session_id)
        page = resumed.read_tool_output(receipt["id"], receipt["char_count"] - 1000)
        self.assertIn("-TAIL", page["content"])
        self.assertTrue(page["complete"])

    def test_output_storage_failure_does_not_turn_a_read_into_operation_failure(self):
        (self.root / "large.txt").write_text("x" * 30000, encoding="utf-8")
        context = replace(self.agent.tool_context, save_tool_output=Mock(side_effect=OSError("fixture disk full")))
        result = read_files_handler(["large.txt"], context=context).to_dict()
        self.assertTrue(result["ok"])
        item = json.loads(result["content"])[0]
        self.assertIn("persistence failed", item["metadata"]["output_warning"])

    def test_both_api_loops_receive_batch_results_and_keep_schema(self):
        for name in ("a.txt", "b.txt"):
            (self.root / name).write_text(name + "-sentinel", encoding="utf-8")
        case = self
        class WireClient(LLMClient):
            def __init__(self, api):
                super().__init__("https://example.invalid", "offline", "offline", api_type=api)
                self.api = api
                self.count = 0

            async def _post_json(self, url, body):
                self.count += 1
                if self.count == 1:
                    tools = [item["function"] for item in body["tools"]] if self.api == "chat" else body["tools"]
                    params = next(tool["parameters"] for tool in tools if tool["name"] == "read_files")
                    case.assertEqual(params["properties"]["paths"]["items"], {"type": "string"})
                    args = json.dumps({"paths": ["a.txt", "b.txt"]})
                    if self.api == "chat":
                        return chat_response(finish_reason="tool_calls", tool_calls=[function_call("batch", "read_files", args)])
                    return {"status": "completed", "output": [{"type": "function_call", "call_id": "batch", "name": "read_files", "arguments": args}]}
                results = [item["content"] for item in body["messages"] if item["role"] == "tool"] if self.api == "chat" else [item["output"] for item in body["input"] if item.get("type") == "function_call_output"]
                payload = json.loads(results[-1])
                case.assertTrue(payload["ok"])
                case.assertEqual(len(json.loads(payload["content"])), 2)
                if self.api == "chat":
                    return chat_response("batch verified")
                return {"status": "completed", "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "batch verified"}]}]}
        for api in ("chat", "responses"):
            with self.subTest(api=api):
                agent = Agent(self.config)
                agent.llm = WireClient(api)
                self.assertEqual(asyncio.run(agent.run("read two files")), "batch verified")
                self.assertEqual(agent.llm.count, 2)
