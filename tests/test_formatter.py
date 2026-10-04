"""Optional OpenRouter formatter: repairs broken tool-call JSON, off without a key.

No network: the HTTP call is replaced. Checks what is (and is not) sent, that failures
degrade to today's behavior, and the full path through a chat completion.
"""

from __future__ import annotations

import asyncio
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.api import formatter, openai_routes
from src.api.openai_schemas import ChatCompletionRequest, ChatMessage, ToolDefinition

TOOLS = [ToolDefinition(type="function", function={"name": "read_file", "parameters": {"type": "object"}})]
# An unterminated string: the deterministic cleanup spec refuses to guess at it, so only the
# model formatter gets to try. (Missing closing brackets alone are fixed by the spec first.)
BROKEN = '{"tool_calls":[{"name":"read_file","arguments":{"path":"a.py}}]}'
REPAIRED = '{"tool_calls":[{"name":"read_file","arguments":{"path":"a.py"}}]}'


def with_key(key: str = "sk-test"):
    return patch.dict(os.environ, {"OPENROUTER_API_KEY": key})


class FormatterUnitTest(unittest.TestCase):
    def test_disabled_without_a_key_and_never_touches_the_network(self) -> None:
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}), \
             patch.object(formatter, "_request", side_effect=AssertionError("network used")):
            self.assertFalse(formatter.enabled())
            self.assertIsNone(asyncio.run(formatter.repair_tool_call(BROKEN, TOOLS)))

    def test_sends_only_the_broken_reply_and_tool_schemas_to_the_free_router(self) -> None:
        sent = {}

        def fake(body):
            sent.update(body)
            return REPAIRED

        with with_key(), patch.dict(os.environ, {"CATGPT_FORMATTER_MODEL": ""}), patch.object(formatter, "_request", fake):
            os.environ.pop("CATGPT_FORMATTER_MODEL")
            self.assertEqual(asyncio.run(formatter.repair_tool_call(BROKEN, TOOLS)), REPAIRED)
        self.assertEqual(sent["model"], "openrouter/free")
        self.assertEqual([m["role"] for m in sent["messages"]], ["system", "user"])
        user = sent["messages"][1]["content"]
        self.assertIn(BROKEN, user)
        self.assertIn("read_file", user)

    def test_unavailable_service_degrades_to_none(self) -> None:
        with with_key(), patch.object(formatter, "_request", side_effect=OSError("429")):
            self.assertIsNone(asyncio.run(formatter.repair_tool_call(BROKEN, TOOLS)))


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
    request = ChatCompletionRequest(
        model="m", tools=TOOLS, messages=[ChatMessage(role="user", content="Open a.py")]
    )
    return asyncio.run(openai_routes._run_completion(request)).choices[0]


class FormatterIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        saved = (openai_routes._client, openai_routes._thread_message_count,
                 openai_routes._last_response_time, openai_routes._lock)
        openai_routes._thread_message_count, openai_routes._last_response_time, openai_routes._lock = 0, 0.0, None
        self.addCleanup(lambda: (setattr(openai_routes, "_client", saved[0]),
                                 setattr(openai_routes, "_thread_message_count", saved[1]),
                                 setattr(openai_routes, "_last_response_time", saved[2]),
                                 setattr(openai_routes, "_lock", saved[3])))

    def test_broken_tool_call_is_repaired_into_a_real_one(self) -> None:
        with with_key(), patch.object(formatter, "_request", return_value=REPAIRED):
            choice = complete(BROKEN)
        self.assertEqual(choice.finish_reason, "tool_calls")
        call = choice.message.tool_calls[0].function
        self.assertEqual((call.name, json.loads(call.arguments)), ("read_file", {"path": "a.py"}))

    def test_repair_cannot_introduce_a_tool_the_request_did_not_offer(self) -> None:
        evil = '{"tool_calls":[{"name":"rm_rf","arguments":{}}]}'
        with with_key(), patch.object(formatter, "_request", return_value=evil):
            choice = complete(BROKEN)
        self.assertEqual(choice.finish_reason, "stop")
        self.assertIsNone(choice.message.tool_calls)

    def test_plain_prose_is_never_sent_to_openrouter(self) -> None:
        with with_key(), patch.object(formatter, "_request", side_effect=AssertionError("prose was sent out")):
            choice = complete("The file is empty.")
        self.assertEqual(choice.message.content, "The file is empty.")

    def test_without_a_key_behaviour_is_unchanged(self) -> None:
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}):
            choice = complete(BROKEN)
        self.assertEqual(choice.finish_reason, "stop")
        self.assertEqual(choice.message.content, BROKEN)


if __name__ == "__main__":
    unittest.main()
