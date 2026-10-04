"""Tab pool: continue a tab's thread only when the request provably extends it."""

from __future__ import annotations

import unittest

from src.api.tabs import TabPool


class FakePage:
    def __init__(self, url: str = "https://chatgpt.com/?temporary-chat=true") -> None:
        self.url = url


class FakeClient:
    def __init__(self) -> None:
        self.page = FakePage()


def pool(n: int = 2, **kw) -> TabPool:
    p = TabPool(**kw)
    for _ in range(n):
        p.add(FakeClient())
    return p


def converse(p: TabPool, tab, fps, url="https://chatgpt.com/c/abc?temporary-chat=true", intensity=None, now=100.0):
    """What a successful request does: lease, then remember the thread incl. the reply."""
    p.lease(tab, now)
    tab.client.page.url = url
    p.remember(tab, fps, intensity, now=now)


class TabPoolTest(unittest.TestCase):
    def test_new_conversation_prefers_an_unused_tab(self) -> None:
        p = pool(3)
        a, start = p.pick(["s", "u1"], now=1)
        self.assertEqual(start, 0)
        converse(p, a, ["s", "u1", "a1"])
        b, start = p.pick(["x", "y"], now=2)
        self.assertEqual(start, 0)
        self.assertIsNot(b, a)  # the used tab keeps its thread while spare tabs exist

    def test_follow_up_returns_to_the_same_tab_and_sends_only_new_messages(self) -> None:
        p = pool()
        a, _ = p.pick(["s", "u1"], now=1)
        converse(p, a, ["s", "u1", "a1"])
        again, start = p.pick(["s", "u1", "a1", "u2"], now=2)
        self.assertIs(again, a)
        self.assertEqual(start, 3)  # system, user and our reply are already in the thread

    def test_two_interleaved_conversations_keep_their_own_tabs(self) -> None:
        p = pool()
        a, _ = p.pick(["s", "u1"], now=1)
        converse(p, a, ["s", "u1", "a1"], url="https://chatgpt.com/c/aaa?temporary-chat=true")
        b, _ = p.pick(["s", "v1"], now=2)
        converse(p, b, ["s", "v1", "b1"], url="https://chatgpt.com/c/bbb?temporary-chat=true", now=2)
        self.assertIsNot(a, b)
        ta, sa = p.pick(["s", "u1", "a1", "u2"], now=3)
        tb, sb = p.pick(["s", "v1", "b1", "v2"], now=3)
        self.assertEqual((ta, sa, tb, sb), (a, 3, b, 3))

    def test_edited_or_unrelated_history_never_shares_context(self) -> None:
        p = pool(1)
        a, _ = p.pick(["s", "u1"], now=1)
        converse(p, a, ["s", "u1", "a1"])
        for fps in (["s", "u1-edited", "a1", "u2"], ["other-system", "u1", "a1", "u2"], ["s", "u1", "a1"], ["s"]):
            tab, start = p.pick(fps, now=2)
            self.assertEqual(start, 0, fps)
        self.assertIn("history differs", p.last_miss)

    def test_guards_turn_limit_idle_intensity_and_navigation(self) -> None:
        cases = {
            "turn limit reached": dict(pool_kw=dict(max_turns=1)),
            "idle too long": dict(now=100.0 + 5000),
            "model intensity changed": dict(intensity="pro"),
            "page left the conversation": dict(url="https://chatgpt.com/?temporary-chat=true"),
        }
        for reason, c in cases.items():
            p = pool(1, **c.get("pool_kw", {}))
            a, _ = p.pick(["s", "u1"], now=1)
            converse(p, a, ["s", "u1", "a1"])
            if "url" in c:
                a.client.page.url = c["url"]
            _, start = p.pick(["s", "u1", "a1", "u2"], c.get("intensity"), now=c.get("now", 101.0))
            self.assertEqual(start, 0, reason)
            self.assertIn(reason, p.last_miss)

    def test_failed_request_leaves_the_tab_unmatchable(self) -> None:
        p = pool(1)
        a, _ = p.pick(["s", "u1"], now=1)
        converse(p, a, ["s", "u1", "a1"])
        p.lease(a, now=2)  # next request starts, then fails before remember()
        _, start = p.pick(["s", "u1", "a1", "u2"], now=3)
        self.assertEqual(start, 0)

    def test_when_all_tabs_hold_threads_the_least_recently_used_is_recycled(self) -> None:
        p = pool(2)
        a, _ = p.pick(["a"], now=1)
        converse(p, a, ["a", "ra"], url="https://chatgpt.com/c/1?temporary-chat=true", now=10)
        b, _ = p.pick(["b"], now=2)
        converse(p, b, ["b", "rb"], url="https://chatgpt.com/c/2?temporary-chat=true", now=20)
        c, start = p.pick(["c"], now=30)
        self.assertEqual(start, 0)
        self.assertIs(c, a)

    def test_turn_counter_counts_continuations(self) -> None:
        p = pool(1)
        a, _ = p.pick(["s", "u1"], now=1)
        converse(p, a, ["s", "u1", "a1"])
        self.assertEqual(a.turns, 1)
        p.lease(a, 2)
        p.remember(a, ["s", "u1", "a1", "u2", "a2"], continued=True, now=2)
        self.assertEqual(a.turns, 2)


if __name__ == "__main__":
    unittest.main()
