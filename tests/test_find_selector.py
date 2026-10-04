"""Regression guard: a stale selector must not eat the wait time of a working one.

The send path used to wait 10 s per selector in order. The retired
#prompt-textarea selectors burned 20 s, leaving the real composer 10 s, but
ChatGPT takes ~35 s to render it after a page load, so every send failed.
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from src.chatgpt import client as client_module
from src.chatgpt.client import ChatGPTClient
from src.selectors import Selectors


class FakeElement:
    async def is_visible(self) -> bool:
        return True


class FakePage:
    """The composer matches only one selector, and only after `appears_at` polls."""

    def __init__(self, match_selector: str, appears_at: int) -> None:
        self.match_selector = match_selector
        self.appears_at = appears_at
        self.polls = 0

    async def query_selector(self, selector: str):
        if selector == Selectors.CHAT_INPUT[0]:  # first selector = one poll round
            self.polls += 1
        return FakeElement() if selector == self.match_selector and self.polls > self.appears_at else None


def find(page: FakePage, timeout_ms: int) -> str | None:
    chat = ChatGPTClient.__new__(ChatGPTClient)
    chat._page = page

    async def instant_sleep(_seconds: float) -> None:
        return None

    clock = iter(x * 0.25 for x in range(100_000))  # one fake quarter-second per poll round
    with patch.object(client_module.asyncio, "sleep", instant_sleep), \
         patch.object(client_module.time, "monotonic", lambda: next(clock)):
        return asyncio.run(chat._find_selector(Selectors.CHAT_INPUT, "chat input", timeout_ms=timeout_ms))


class FindSelectorTest(unittest.TestCase):
    def test_current_composer_is_first_choice(self) -> None:
        self.assertIn("ProseMirror", Selectors.CHAT_INPUT[0])

    def test_finds_slow_composer_with_stale_selectors_listed_first(self) -> None:
        page = FakePage("div[contenteditable='true']", appears_at=100)  # ~25 fake seconds in
        self.assertEqual(find(page, timeout_ms=60_000), "div[contenteditable='true']")

    def test_gives_up_at_the_shared_deadline(self) -> None:
        page = FakePage("never-matches", appears_at=0)
        self.assertIsNone(find(page, timeout_ms=5_000))
        self.assertLess(page.polls, 40)


if __name__ == "__main__":
    unittest.main()
