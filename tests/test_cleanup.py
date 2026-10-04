"""The deterministic cleanup spec (src/api/cleanup.py), as tables.

Each row is one rule's contract. Also checks the properties the spec promises:
idempotent text cleanup, strictly valid JSON out, string contents protected, and
no guessing (unterminated strings and mismatched brackets stay broken).
"""

from __future__ import annotations

import asyncio
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from src.api import cleanup, formatter, openai_routes
from src.api.openai_schemas import ChatCompletionRequest, ChatMessage, ToolDefinition

TOOLS = [ToolDefinition(type="function", function={"name": "read_file", "parameters": {"type": "object"}})]

TEXT_CASES = [
    # (rule, input, expected)
    ("strip_invisible", "he​ll﻿o", "hello"),
    ("normalize_newlines", "a\r\nb\rc", "a\nb\nc"),
    ("strip_speaker_label", "ChatGPT said:\nhello", "hello"),
    ("strip_speaker_label", "You said:\nhi", "hi"),
    ("strip_thinking_label", "Thought for 6s\n\nParis is sunny.", "Paris is sunny."),
    ("strip_thinking_label", "Thought for 1m 12s\nDone", "Done"),
    ("strip_thinking_label", "thought for 2 minutes\n\nok", "ok"),
    ("both labels, either order of arrival", "ChatGPT said:\nThought for 3s\n\nhello", "hello"),
    ("final trim", "  \n spaced \n\n", "spaced"),
    # Things that must NOT change:
    ("prose that starts like the label", "Thought for 6 hours was wasted on this.", "Thought for 6 hours was wasted on this."),
    ("a quote of the label mid-text", "He wrote:\nThought for 6s\nand left.", "He wrote:\nThought for 6s\nand left."),
    ("blank lines inside code", "```\nline1\n\n\nline2\n```", "```\nline1\n\n\nline2\n```"),
    ("speaker label mid-sentence", "I said: hi", "I said: hi"),
]

JSON_CASES = [
    # (name, input, expected parsed object)
    ("fence with language label", '```json\n{"a": 1}\n```', {"a": 1}),
    ("unclosed fence", '```json\n{"a": 1}', {"a": 1}),
    ("prose before and after", 'Sure! {"a": 1} hope that helps {not json}', {"a": 1}),
    ("trailing commas", '{"a": [1, 2,], "b": {"c": 3,},}', {"a": [1, 2], "b": {"c": 3}}),
    ("comments, URL in a string kept", '{"a": 1, // note\n "b": "http://x.y/*z*/"}', {"a": 1, "b": "http://x.y/*z*/"}),
    ("python literals, string contents kept", '{"a": True, "b": None, "c": "True story"}', {"a": True, "b": None, "c": "True story"}),
    ("python dict with single quotes", "{'a': 'x', 'b': [1, 2], 'c': False}", {"a": "x", "b": [1, 2], "c": False}),
    ("typographic quotes as delimiters", "{“a”: “b”}", {"a": "b"}),
    ("truncated: missing closers", '{"tool_calls":[{"name":"read_file","arguments":{"path":"a.py"}}',
     {"tool_calls": [{"name": "read_file", "arguments": {"path": "a.py"}}]}),
    ("truncated after a comma", '{"a": [1, 2,', {"a": [1, 2]}),
    ("raw newline inside a string", '{"command": "echo a\necho b"}', {"command": "echo a\necho b"}),
    ("non-breaking space", '{"a": 1}', {"a": 1}),
    ("already valid is untouched", '{"a": "x, ]", "b": [1, 2]}', {"a": "x, ]", "b": [1, 2]}),
]

UNREPAIRABLE = [
    ("unterminated string is never guessed", '{"a": "oops'),
    ("mismatched brackets are never guessed", '{"a": [1, 2}'),
    ("no object at all", "just prose"),
    ("a list is not an object", "[1, 2]"),
    ("empty", ""),
]


class TextTierTest(unittest.TestCase):
    def test_table(self) -> None:
        for rule, text, expected in TEXT_CASES:
            with self.subTest(rule=rule, text=text):
                self.assertEqual(cleanup.clean_text(text), expected)

    def test_idempotent(self) -> None:
        for _, text, _ in TEXT_CASES:
            once = cleanup.clean_text(text)
            self.assertEqual(cleanup.clean_text(once), once, text)

    def test_none_is_empty(self) -> None:
        self.assertEqual(cleanup.clean_text(None), "")

    def test_a_reply_that_is_only_a_scraped_label_becomes_empty(self) -> None:
        self.assertEqual(cleanup.clean_text("Thought for 6s\n\n"), "")
        self.assertEqual(cleanup.clean_text("ChatGPT said:\n"), "")


class JsonTierTest(unittest.TestCase):
    def test_table_yields_strictly_valid_json_equal_to_the_intended_object(self) -> None:
        for name, text, expected in JSON_CASES:
            with self.subTest(case=name):
                repaired = cleanup.repair_json(text)
                self.assertIsNotNone(repaired, name)
                self.assertEqual(json.loads(repaired), expected)

    def test_unrepairable_stays_unrepaired(self) -> None:
        for name, text in UNREPAIRABLE:
            with self.subTest(case=name):
                self.assertIsNone(cleanup.repair_json(text))

    def test_every_rule_is_named_and_explained(self) -> None:
        names = [r.name for r in cleanup.TEXT_RULES + cleanup.JSON_RULES]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(r.why for r in cleanup.TEXT_RULES + cleanup.JSON_RULES))


class WiredIntoTheRequestPathTest(unittest.TestCase):
    def test_tool_call_with_trailing_commas_in_a_fence_parses(self) -> None:
        reply = '```json\n{"tool_calls":[{"name":"read_file","arguments":{"path":"a.py",}},]}\n```'
        calls = openai_routes._parse_tool_calls(reply, TOOLS)
        self.assertEqual([(c.function.name, json.loads(c.function.arguments)) for c in calls], [("read_file", {"path": "a.py"})])

    def test_truncated_tool_call_parses(self) -> None:
        reply = '{"tool_calls":[{"name":"read_file","arguments":{"path":"a.py"}}'
        self.assertEqual(openai_routes._parse_tool_calls(reply, TOOLS)[0].function.name, "read_file")

    def test_the_spec_cannot_introduce_an_unoffered_tool(self) -> None:
        self.assertIsNone(openai_routes._parse_tool_calls('{"tool_calls":[{"name":"rm_rf","arguments":{},}]}', TOOLS))

    def test_answer_with_a_trailing_comma_unwraps(self) -> None:
        self.assertEqual(openai_routes._unwrap_answer('{"answer": "hi",}'), "hi")


class FakeClient:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.page = SimpleNamespace(url="https://chatgpt.com/?temporary-chat=true")

    async def new_chat(self) -> None:
        self.page.url = "https://chatgpt.com/?temporary-chat=true"

    async def send_message(self, text, image_paths=None, file_paths=None):
        return SimpleNamespace(message=self.reply)


def complete(reply: str):
    openai_routes.set_openai_client(FakeClient(reply))
    request = ChatCompletionRequest(model="m", tools=TOOLS, messages=[ChatMessage(role="user", content="Open a.py")])
    return asyncio.run(openai_routes._run_completion(request)).choices[0]


class EndToEndTest(unittest.TestCase):
    def setUp(self) -> None:
        saved = (openai_routes._client, openai_routes._thread_message_count,
                 openai_routes._last_response_time, openai_routes._lock)
        openai_routes._thread_message_count, openai_routes._last_response_time, openai_routes._lock = 0, 0.0, None
        self.addCleanup(lambda: (setattr(openai_routes, "_client", saved[0]),
                                 setattr(openai_routes, "_thread_message_count", saved[1]),
                                 setattr(openai_routes, "_last_response_time", saved[2]),
                                 setattr(openai_routes, "_lock", saved[3])))

    def test_scraped_label_plus_sloppy_json_becomes_a_clean_tool_call(self) -> None:
        reply = 'Thought for 6s\n\n{"tool_calls":[{"name":"read_file","arguments":{"path":"a.py"},}]}'
        choice = complete(reply)
        self.assertEqual(choice.finish_reason, "tool_calls")
        self.assertEqual(json.loads(choice.message.tool_calls[0].function.arguments), {"path": "a.py"})

    def test_prose_answer_loses_the_scraped_label_but_nothing_else(self) -> None:
        self.assertEqual(complete("Thought for 3s\n\nThe file is empty.").message.content, "The file is empty.")

    def test_a_reply_that_was_only_a_label_is_an_error_not_an_empty_answer(self) -> None:
        with self.assertRaises(HTTPException) as caught:
            complete("Thought for 3s")
        self.assertEqual(caught.exception.status_code, 502)

    def test_the_model_formatter_only_sees_what_the_spec_could_not_fix(self) -> None:
        fixable = '{"tool_calls":[{"name":"read_file","arguments":{"path":"a.py"}}'
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test"}), \
             patch.object(formatter, "_request", side_effect=AssertionError("model called for a fixable reply")):
            self.assertEqual(complete(fixable).finish_reason, "tool_calls")


if __name__ == "__main__":
    unittest.main()
