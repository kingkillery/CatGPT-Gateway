"""Thread reuse through the real request path, with fake tabs standing in for ChatGPT.

A follow-up that extends a tab's conversation goes back to that tab and sends only the new
messages (no new chat, no reload); anything else gets an empty chat with the full history.
"""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from src.api import openai_routes
from src.api.openai_schemas import ChatCompletionRequest, ChatMessage, ResponsesRequest, ToolDefinition

ROOT = "https://chatgpt.com/?temporary-chat=true"
TOOLS = [ToolDefinition(type="function", function={"name": "read_file", "parameters": {"type": "object"}})]
CALL = '{"tool_calls":[{"name":"read_file","arguments":{"path":"a.py"}}]}'


class FakeTab:
    """Records what is sent and how often the chat is reset; the URL becomes /c/<name> after a message."""

    def __init__(self, name: str, replies: list[str]) -> None:
        self.name, self.replies = name, list(replies)
        self.sent: list[str] = []
        self.new_chats = 0
        self.fail_next = False
        self.page = SimpleNamespace(url=ROOT)

    async def new_chat(self) -> None:
        self.new_chats += 1
        self.page.url = ROOT

    async def send_message(self, text, image_paths=None, file_paths=None):
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("tab died")
        self.sent.append(text)
        self.page.url = f"https://chatgpt.com/c/{self.name}?temporary-chat=true"
        return SimpleNamespace(message=self.replies.pop(0))


def msg(role, content=None, **kw) -> ChatMessage:
    return ChatMessage(role=role, content=content, **kw)


def chat(*messages, tools=None) -> ChatCompletionRequest:
    return ChatCompletionRequest(model="m", messages=list(messages), tools=tools)


def run(request):
    return asyncio.run(openai_routes._run_completion(request)).choices[0]


class ThreadReuseTest(unittest.TestCase):
    def setUp(self) -> None:
        keys = ("_client", "_pool", "_last_response_time", "_lock", "_MIN_MESSAGE_GAP")
        saved = {k: getattr(openai_routes, k) for k in keys}
        self.addCleanup(lambda: [setattr(openai_routes, k, v) for k, v in saved.items()])
        openai_routes._last_response_time, openai_routes._lock, openai_routes._MIN_MESSAGE_GAP = 0.0, None, 0.0

    def tabs(self, *replies: list[str]) -> list[FakeTab]:
        tabs = [FakeTab(f"t{i}", r) for i, r in enumerate(replies)]
        openai_routes.set_openai_client(tabs[0])
        for t in tabs[1:]:
            openai_routes.add_openai_tab(t)
        return tabs

    def test_follow_up_continues_the_chat_and_sends_only_what_is_new(self) -> None:
        (a,) = self.tabs(["Paris.", "Madrid."])
        first = msg("user", "What is the capital of France?")
        run(chat(first))
        reply = run(chat(first, msg("assistant", "Paris."), msg("user", "And Spain?")))
        self.assertEqual(reply.message.content, "Madrid.")
        self.assertEqual(a.sent[1], "And Spain?")  # bare new message, no transcript
        self.assertEqual(a.new_chats, 0)  # no reset, so no reload

    def test_tool_loop_stays_in_one_chat_and_names_the_tool_it_answers(self) -> None:
        (a,) = self.tabs([CALL, '{"answer":"It prints hi."}'])
        user = msg("user", "Open a.py")
        called = run(chat(user, tools=TOOLS)).message
        self.assertEqual(called.tool_calls[0].function.name, "read_file")
        history = [user, msg("assistant", None, tool_calls=called.tool_calls),
                   msg("tool", "print('hi')", tool_call_id=called.tool_calls[0].id)]
        done = run(chat(*history, tools=TOOLS))
        self.assertEqual(done.message.content, "It prints hi.")
        self.assertEqual(a.new_chats, 0)
        self.assertIn(f"[Tool result for read_file ({called.tool_calls[0].id})]: print('hi')", a.sent[1])
        self.assertNotIn("Available functions", a.sent[1])  # the tool block is already in the thread

    def test_interleaved_conversations_each_keep_their_own_tab(self) -> None:
        a, b = self.tabs(["A1", "A2"], ["B1", "B2"])
        ua, ub = msg("user", "alpha question"), msg("user", "beta question")
        run(chat(ua)), run(chat(ub))
        self.assertEqual((len(a.sent), len(b.sent)), (1, 1))
        run(chat(ua, msg("assistant", "A1"), msg("user", "alpha again")))
        run(chat(ub, msg("assistant", "B1"), msg("user", "beta again")))
        self.assertEqual((a.sent[1], b.sent[1]), ("alpha again", "beta again"))
        self.assertEqual((a.new_chats, b.new_chats), (0, 0))

    def test_edited_history_never_shares_a_chat_with_the_original(self) -> None:
        a, b = self.tabs(["Paris.", "Answer"], ["Fresh"])
        run(chat(msg("user", "What is the capital of France?")))
        run(chat(msg("user", "What is the capital of Germany?"), msg("assistant", "Paris."), msg("user", "And Spain?")))
        self.assertEqual(len(a.sent), 1)  # the original thread was not touched
        self.assertIn("capital of Germany", b.sent[0])  # the other tab got the whole history
        self.assertIn("Assistant: Paris.", b.sent[0])

    def test_failed_continuation_retries_in_a_clean_chat_with_the_full_history(self) -> None:
        (a,) = self.tabs(["Paris.", "Madrid."])
        first = msg("user", "What is the capital of France?")
        run(chat(first))
        a.fail_next = True
        reply = run(chat(first, msg("assistant", "Paris."), msg("user", "And Spain?")))
        self.assertEqual(reply.message.content, "Madrid.")
        self.assertEqual(a.new_chats, 1)  # emptied before the retry
        self.assertIn("capital of France", a.sent[1])  # full history, not just the follow-up
        self.assertIn("And Spain?", a.sent[1])

    def test_a_correction_turn_makes_the_thread_unreusable(self) -> None:
        (a,) = self.tabs(["I cannot produce a fake tool call here.", "Paris.", "Madrid."])
        first = msg("user", "What is the capital of France?")
        run(chat(first, tools=TOOLS))  # refusal -> correction turn: the thread now holds extra turns
        run(chat(first, msg("assistant", "Paris."), msg("user", "And Spain?"), tools=TOOLS))
        self.assertEqual(a.new_chats, 1)
        self.assertIn("capital of France", a.sent[-1])

    def test_responses_api_continues_the_thread_too(self) -> None:
        (a,) = self.tabs(["Paris.", "Madrid."])
        asyncio.run(openai_routes.create_response(
            ResponsesRequest(model="m", input="What is the capital of France?")))
        out = asyncio.run(openai_routes.create_response(ResponsesRequest(model="m", input=[
            {"role": "user", "content": "What is the capital of France?"},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Paris."}]},
            {"role": "user", "content": "And Spain?"},
        ])))
        self.assertEqual(a.sent[1], "And Spain?")
        self.assertEqual(a.new_chats, 0)
        self.assertIn("Madrid.", str(out))


if __name__ == "__main__":
    unittest.main()
