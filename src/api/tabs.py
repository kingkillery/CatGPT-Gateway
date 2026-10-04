"""One ChatGPT conversation thread per browser tab, reused while a request continues it.

Every OpenAI request carries the whole message history. Instead of starting a fresh chat
and re-sending all of it each time, a tab remembers which messages its thread already
holds (as fingerprints). A later request whose history starts with exactly those messages
is a continuation: it goes back to that tab and only the new messages are sent. Anything
else gets an empty tab, so unrelated conversations never share context.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit


@dataclass
class Tab:
    client: Any
    sent: list[str] | None = None  # fingerprints of the messages in this tab's thread; None = unknown/empty
    url: str = ""  # page URL after the last reply (/c/<id> once a conversation has started)
    turns: int = 0  # requests served from the current thread
    intensity: str | None = None
    last_used: float = 0.0


def _same_thread(remembered: str, current: str) -> bool:
    """Same conversation page; the query string may differ."""
    return bool(remembered) and urlsplit(remembered).path == urlsplit(current or "").path


class TabPool:
    def __init__(self, max_turns: int = 12, idle_seconds: float = 1800.0) -> None:
        self.tabs: list[Tab] = []
        self.max_turns = max_turns
        self.idle_seconds = idle_seconds
        self.last_miss = ""  # why the last pick() could not continue a thread

    def add(self, client: Any) -> Tab:
        tab = Tab(client)
        self.tabs.append(tab)
        return tab

    def index(self, tab: Tab) -> int:
        return self.tabs.index(tab)

    def _blocker(self, tab: Tab, fps: list[str], intensity: str | None, now: float) -> str | None:
        """Why this tab cannot continue the conversation, or None when it can."""
        n = len(tab.sent or [])
        if n == 0:
            return "holds no thread"
        if len(fps) <= n or fps[:n] != tab.sent:
            return "history differs"
        if tab.turns >= self.max_turns:
            return "turn limit reached"
        if now - tab.last_used > self.idle_seconds:
            return "idle too long"
        if tab.intensity != intensity:
            return "model intensity changed"
        try:
            current = tab.client.page.url
        except Exception:
            current = ""
        if not _same_thread(tab.url, current):
            return "page left the conversation"
        return None

    def pick(self, fps: list[str], intensity: str | None = None, now: float | None = None) -> tuple[Tab, int]:
        """Choose a tab for a request: (tab, start). start > 0 means the tab's thread already
        holds fps[:start], so only fps[start:] needs sending; 0 means an empty chat is required."""
        now = time.monotonic() if now is None else now
        best, best_n, reasons = None, 0, []
        for i, tab in enumerate(self.tabs):
            why = self._blocker(tab, fps, intensity, now)
            if why is None:
                if len(tab.sent) > best_n:
                    best, best_n = tab, len(tab.sent)
            elif why != "holds no thread":
                reasons.append(f"tab {i}: {why}")
        if best is not None:
            self.last_miss = ""
            return best, best_n
        self.last_miss = "; ".join(reasons) or "no tab holds a thread"
        # Prefer a tab that was never used, then the least recently used one.
        return min(self.tabs, key=lambda t: (t.turns > 0, t.last_used)), 0

    def lease(self, tab: Tab, now: float | None = None) -> None:
        """Mark the tab's thread unknown while a request runs; remember() restores it on success.
        A failure in between leaves the tab unmatchable instead of wrongly reusable."""
        tab.sent = None
        tab.last_used = time.monotonic() if now is None else now

    def remember(
        self, tab: Tab, fps: list[str], intensity: str | None = None, continued: bool = False,
        now: float | None = None,
    ) -> None:
        """Record that the tab's thread now holds exactly fps (including our reply)."""
        tab.sent = list(fps)
        tab.url = tab.client.page.url or ""
        tab.intensity = intensity
        tab.turns = tab.turns + 1 if continued else 1
        tab.last_used = time.monotonic() if now is None else now
