"""Regression guards for request isolation and empty replies, through the real request path.

A fake browser client stands in for ChatGPT. Covers:
- an empty reply (how ChatGPT's 429 rate limiting appears) is a 502, not an empty 200;
- every request starts in a fresh chat, including after a request that failed.
"""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from fastapi import HTTPException

from src.api import openai_routes
from src.api.openai_schemas import ChatCompletionRequest, ChatMessage


class FakeClient:
    """Records new_chat() calls; the page URL becomes /c/<id> once a message is sent."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.new_chats = 0
        self.page = SimpleNamespace(url="https://chatgpt.com/?temporary-chat=true")

    async def new_chat(self) -> None:
        self.new_chats += 1
        self.page.url = "https://chatgpt.com/?temporary-chat=true"

    async def send_message(self, text, image_paths=None, file_paths=None):
        self.page.url = "https://chatgpt.com/c/abc123?temporary-chat=true"
        return SimpleNamespace(message=self.replies.pop(0))


def ask(client: FakeClient, text: str = "hi"):
    request = ChatCompletionRequest(model="m", messages=[ChatMessage(role="user", content=text)])
    return asyncio.run(openai_routes._run_completion(request))


class RequestIsolationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = (openai_routes._client, openai_routes._thread_message_count,
                       openai_routes._last_response_time, openai_routes._lock)
        openai_routes._thread_message_count = 0
        openai_routes._last_response_time = 0.0
        openai_routes._lock = None  # a fresh lock per asyncio.run() loop
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        (openai_routes._client, openai_routes._thread_message_count,
         openai_routes._last_response_time, openai_routes._lock) = self._saved

    def use(self, client: FakeClient) -> FakeClient:
        openai_routes.set_openai_client(client)
        return client

    def test_empty_reply_is_a_retryable_error_not_an_empty_answer(self) -> None:
        client = self.use(FakeClient([""]))
        with self.assertRaises(HTTPException) as caught:
            ask(client)
        self.assertEqual(caught.exception.status_code, 502)
        self.assertIn("rate-limiting", caught.exception.detail)

    def test_each_request_after_the_first_starts_a_new_chat(self) -> None:
        client = self.use(FakeClient(["one", "two", "three"]))
        for _ in range(3):
            ask(client)
        # Already on a fresh chat for the first request; a new one for each after.
        self.assertEqual(client.new_chats, 2)

    def test_chat_that_held_a_failed_request_is_not_reused(self) -> None:
        client = self.use(FakeClient(["", "ok"]))
        with self.assertRaises(HTTPException):
            ask(client)  # fails: the success counter never increments
        self.assertEqual(ask(client).choices[0].message.content, "ok")
        self.assertEqual(client.new_chats, 1)  # the failed chat was not carried into the retry


if __name__ == "__main__":
    unittest.main()
