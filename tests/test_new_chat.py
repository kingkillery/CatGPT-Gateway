"""Regression guard: new_chat() must not reuse a conversation that already has messages.

It used to count `conversation-turn-N` sections to decide a temporary chat was
"fresh". That count is always 0 in the current ChatGPT layout, so after the first
message every request kept using the same chat and later requests saw earlier ones.
"""

from __future__ import annotations

import asyncio
import unittest

from src.chatgpt.client import ChatGPTClient


class FakePage:
    def __init__(self, url: str, assistant_replies: int = 0) -> None:
        self.url = url
        self.assistant_replies = assistant_replies

    async def evaluate(self, *_args, **_kwargs):
        return self.assistant_replies  # what count_assistant_messages() reads


def is_fresh(url: str, replies: int = 0) -> bool:
    client = ChatGPTClient.__new__(ChatGPTClient)
    client._page = FakePage(url, replies)
    return asyncio.run(client._is_fresh_temporary_chat())


class FreshTemporaryChatTest(unittest.TestCase):
    def test_empty_temporary_chat_is_fresh(self) -> None:
        self.assertTrue(is_fresh("https://chatgpt.com/?temporary-chat=true"))

    def test_started_conversation_is_not_fresh_even_if_turns_cannot_be_counted(self) -> None:
        # Reply count reads 0 (layout not recognised) but the URL has a conversation id.
        self.assertFalse(is_fresh("https://chatgpt.com/c/6ac189a0-7894?temporary-chat=true", replies=0))

    def test_chat_with_replies_is_not_fresh(self) -> None:
        self.assertFalse(is_fresh("https://chatgpt.com/?temporary-chat=true", replies=2))

    def test_saved_chat_or_other_site_is_not_fresh(self) -> None:
        self.assertFalse(is_fresh("https://chatgpt.com/"))
        self.assertFalse(is_fresh("https://example.com/?temporary-chat=true"))


if __name__ == "__main__":
    unittest.main()
