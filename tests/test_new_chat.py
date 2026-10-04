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


class SoftPage:
    """Mimics the app: a pushState route change swaps the conversation for an empty temporary chat."""

    def __init__(self, url: str, replies: int, draft: str = "", app_responds: bool = True) -> None:
        self.url, self.replies, self.draft, self.app_responds = url, replies, draft, app_responds
        self.pushed = 0

    async def evaluate(self, script, *_args):
        if "pushState" in script:
            self.pushed += 1
            if self.app_responds:
                self.url, self.replies = "https://chatgpt.com/?temporary-chat=true", 0
            return None
        if "sels" in script:  # composer-empty probe
            return self.draft.strip() == ""
        return self.replies  # count_assistant_messages()


def soft_reset(page: SoftPage) -> bool:
    from unittest.mock import patch
    from src.chatgpt import client as client_module

    c = ChatGPTClient.__new__(ChatGPTClient)
    c._page = page
    ticks = iter(x * 0.25 for x in range(10_000))

    async def instant(_s):
        return None

    with patch.object(client_module.asyncio, "sleep", instant), \
         patch.object(client_module.time, "monotonic", lambda: next(ticks)):
        return asyncio.run(c._soft_new_temporary_chat())


CONVERSATION = "https://chatgpt.com/c/6ac189a0-7894?temporary-chat=true"


class SoftNewChatTest(unittest.TestCase):
    def test_route_change_reaches_an_empty_temporary_chat_without_reloading(self) -> None:
        page = SoftPage(CONVERSATION, replies=3)
        self.assertTrue(soft_reset(page))
        self.assertEqual(page.pushed, 1)
        self.assertEqual(page.url, "https://chatgpt.com/?temporary-chat=true")

    def test_falls_back_to_a_reload_when_the_app_ignores_the_route_change(self) -> None:
        self.assertFalse(soft_reset(SoftPage(CONVERSATION, replies=3, app_responds=False)))

    def test_leftover_draft_text_is_not_a_fresh_chat(self) -> None:
        self.assertFalse(soft_reset(SoftPage(CONVERSATION, replies=1, draft="half-typed message")))

    def test_never_runs_outside_chatgpt(self) -> None:
        page = SoftPage("https://example.com/", replies=0)
        self.assertFalse(soft_reset(page))
        self.assertEqual(page.pushed, 0)


class RootPageTest(unittest.TestCase):
    def test_plain_root_page_skips_the_in_page_switch_and_its_wasted_wait(self) -> None:
        # From "/" the router sees the same path, so pushState changes nothing; reload instead.
        page = SoftPage("https://chatgpt.com/", replies=0)
        self.assertFalse(soft_reset(page))
        self.assertEqual(page.pushed, 0)
