"""Turn detection against both ChatGPT layouts, run in a real browser.

The old layout has <section data-testid="conversation-turn-N" data-turn="assistant">.
The current layout has no turn sections or role attributes: a finished reply owns a
"Copy" button in its .turn-action-controls row, the user's own is "Copy message", and
code blocks carry their own "Copy" button outside that row. Skipped when no browser
is installed (it runs inside the gateway image).
"""

from __future__ import annotations

import asyncio
import unittest

from src.chatgpt import detector

CURRENT_ONE_REPLY = """
<div data-content-search-turn-key="fallback-turn-0"><div class="group flex flex-col">
  <div class="flex flex-col gap-3">
    <div class="group/user-message"><h5>You said:</h5>Say PONG
      <div class="turn-action-controls"><button aria-label="Copy message"></button></div></div>
    <h6>ChatGPT said:</h6><div class="markdown">PONG<pre>code<button aria-label="Copy">c</button></pre></div>
  </div>
  <div class="mt-1.5 flex turn-action-controls"><button aria-label="Copy"></button></div>
</div></div>"""

CURRENT_STREAMING_SECOND = """
<div data-content-search-turn-key="fallback-turn-1"><div class="group flex flex-col">
  <div class="flex flex-col gap-3">
    <div class="group/user-message"><h5>You said:</h5>Again
      <div class="turn-action-controls"><button aria-label="Copy message"></button></div></div>
    <h6>ChatGPT said:</h6><div class="markdown">thinking...</div>
  </div>
</div></div>"""

LEGACY = """
<section data-testid="conversation-turn-1" data-turn="user">hi</section>
<section data-testid="conversation-turn-2" data-turn="assistant" data-turn-id="abc">
  <div data-message-author-role="assistant">Hello there</div>
  <button aria-label="Copy"></button>
</section>"""


async def run_in_browser(html: str, coro_factory):
    from patchright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        try:
            page = await browser.new_page()
            await page.set_content(html)
            return await coro_factory(page)
        finally:
            await browser.close()


def browser_run(html: str, coro_factory):
    try:
        return asyncio.run(run_in_browser(html, coro_factory))
    except Exception as e:  # no browser binary on this machine
        if "Executable doesn't exist" in str(e) or "playwright install" in str(e).lower():
            raise unittest.SkipTest("no browser installed")
        raise


class TurnDetectionTest(unittest.TestCase):
    def test_current_layout_reply_is_found_and_text_excludes_user_message(self) -> None:
        async def check(page):
            snap = await detector._latest_assistant_turn_snapshot(page)
            return (snap, await detector.count_assistant_messages(page),
                    await detector._count_copy_buttons(page), await detector._extract_via_dom(page))

        snap, turns, copies, text = browser_run(CURRENT_ONE_REPLY, check)
        self.assertTrue(snap["found"] and snap["hasCopyButton"])
        self.assertEqual((turns, copies), (1, 1))  # the code block's Copy button is not a reply
        self.assertEqual(snap["signature"], "0:fallback-turn-0")
        self.assertTrue(text.startswith("PONG"), text)
        self.assertNotIn("Say PONG", text)

    def test_in_progress_reply_is_not_mistaken_for_a_new_finished_one(self) -> None:
        async def check(page):
            return await detector._latest_assistant_turn_snapshot(page)

        # Second exchange is still streaming (no Copy yet): latest finished reply is still turn 0.
        snap = browser_run(CURRENT_ONE_REPLY + CURRENT_STREAMING_SECOND, check)
        self.assertEqual(snap["signature"], "0:fallback-turn-0")

    def test_empty_chat_has_no_assistant_turn(self) -> None:
        async def check(page):
            return await detector._latest_assistant_turn_snapshot(page), await detector.count_assistant_messages(page)

        snap, turns = browser_run("<div>New chat</div>", check)
        self.assertFalse(snap["found"])
        self.assertEqual(turns, 0)

    def test_legacy_layout_still_works(self) -> None:
        async def check(page):
            return await detector._latest_assistant_turn_snapshot(page), await detector._extract_via_dom(page)

        snap, text = browser_run(LEGACY, check)
        self.assertEqual(snap["signature"], "1:abc")
        self.assertTrue(snap["hasCopyButton"])
        self.assertIn("Hello there", text)


if __name__ == "__main__":
    unittest.main()
