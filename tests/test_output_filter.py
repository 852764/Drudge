from __future__ import annotations

import unittest

from agent.output_filter import (
    DegenerateReasoningError,
    ReasoningTagFilter,
    filter_reasoning_text,
    sanitize_provider_items,
)


class OutputFilterTests(unittest.TestCase):
    def test_stream_filter_handles_tags_split_across_chunks(self):
        parser = ReasoningTagFilter()
        output = []
        for chunk in ("<thi", "nk>secret", " reasoning</th", "ink>Final", " answer"):
            output.append(parser.feed(chunk))
        output.append(parser.finish())

        self.assertEqual("".join(output), "Final answer")
        self.assertTrue(parser.saw_tag)
        self.assertGreater(parser.reasoning_chars, 0)

    def test_unclosed_think_block_is_removed(self):
        result = filter_reasoning_text("<think>never finished")

        self.assertEqual(result.text, "")
        self.assertTrue(result.saw_tag)
        self.assertTrue(result.unclosed_tag)

    def test_stray_closing_tag_is_removed(self):
        result = filter_reasoning_text("</think>Final answer")

        self.assertEqual(result.text, "Final answer")
        self.assertTrue(result.saw_tag)

    def test_repeated_nested_think_tags_are_stopped(self):
        parser = ReasoningTagFilter(max_depth=2)

        with self.assertRaises(DegenerateReasoningError):
            parser.feed("<think><think><think>")

    def test_provider_items_keep_encrypted_reasoning_but_clean_message(self):
        items = [
            {"type": "reasoning", "encrypted_content": "keep-me"},
            {
                "type": "message",
                "content": [{
                    "type": "output_text",
                    "text": "<think>secret</think>visible",
                }],
            },
        ]

        cleaned = sanitize_provider_items(items)

        self.assertEqual(cleaned[0]["encrypted_content"], "keep-me")
        self.assertEqual(cleaned[1]["content"][0]["text"], "visible")


if __name__ == "__main__":
    unittest.main()
