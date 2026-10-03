"""Regression guard: without a terminal, sign-in is polled instead of blocking on input()."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from src.browser import auto_login


class FakeBrowser:
    def __init__(self, logged_in_after: int) -> None:
        self.calls = 0
        self.logged_in_after = logged_in_after

    async def is_logged_in(self) -> bool:
        self.calls += 1
        return self.calls > self.logged_in_after


class NonInteractiveLoginTest(unittest.TestCase):
    def _run(self, browser: FakeBrowser) -> bool:
        async def instant_sleep(_seconds: float) -> None:
            return None

        with patch.object(auto_login.sys.stdin, "isatty", return_value=False), \
             patch.object(auto_login.asyncio, "sleep", instant_sleep), \
             patch("builtins.input", side_effect=AssertionError("input() must not be called")):
            return asyncio.run(auto_login.ensure_logged_in(browser))

    def test_continues_once_user_signs_in(self) -> None:
        browser = FakeBrowser(logged_in_after=3)
        self.assertTrue(self._run(browser))
        self.assertGreater(browser.calls, 3)

    def test_gives_up_instead_of_hanging(self) -> None:
        with patch.object(auto_login, "_LOGIN_WAIT_SECONDS", 0):
            self.assertFalse(self._run(FakeBrowser(logged_in_after=10**9)))


if __name__ == "__main__":
    unittest.main()
