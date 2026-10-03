"""Regression guard: the login check waits for a slow-rendering ChatGPT page.

ChatGPT reloads itself after load and can take 30+ seconds to show the composer,
so a check that gives up after a few seconds treats a signed-in session as signed out.
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from src.browser import manager
from src.browser.manager import BrowserManager


class FakeElement:
    async def is_visible(self) -> bool:
        return True


class FakePage:
    """query_selector returns nothing until `appears_at` polls have happened."""

    def __init__(self, match_selector: str, appears_at: int) -> None:
        self.match_selector = match_selector
        self.appears_at = appears_at
        self.polls = 0

    async def query_selector(self, selector: str):
        if selector == "#prompt-textarea":  # first selector = one poll
            self.polls += 1
        return FakeElement() if selector == self.match_selector and self.polls > self.appears_at else None


def run_check(page: FakePage, seconds: int = 60) -> bool:
    mgr = BrowserManager.__new__(BrowserManager)
    mgr._page = page

    async def instant_sleep(_seconds: float) -> None:
        return None

    clock = iter(range(0, 10_000))  # one fake second per poll
    with patch.object(manager.asyncio, "sleep", instant_sleep), \
         patch.object(manager.time, "monotonic", lambda: next(clock)), \
         patch.object(manager, "_LOGIN_CHECK_SECONDS", seconds):
        return asyncio.run(mgr.is_logged_in())


class LoginCheckTest(unittest.TestCase):
    def test_waits_for_slow_composer_without_id(self) -> None:
        # The composer is now a bare contenteditable div and shows up ~35 polls in.
        page = FakePage("div[contenteditable='true']", appears_at=35)
        self.assertTrue(run_check(page))
        self.assertGreater(page.polls, 35)

    def test_login_button_means_signed_out(self) -> None:
        self.assertFalse(run_check(FakePage("button[data-testid='login-button']", appears_at=2)))

    def test_nothing_ever_renders_is_uncertain_not_a_hang(self) -> None:
        page = FakePage("never-matches", appears_at=0)
        self.assertFalse(run_check(page, seconds=10))
        self.assertLess(page.polls, 20)


if __name__ == "__main__":
    unittest.main()
