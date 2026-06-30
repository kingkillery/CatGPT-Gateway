#!/usr/bin/env python3
"""
Deterministic micro-benchmark for the *controllable* (non-LLM) overhead of a
ChatGPT send_message round-trip.

It drives the REAL `ChatGPTClient.send_message` against a fake page on a virtual
clock, so the only thing that "takes time" is whatever the gateway code itself
chooses to wait for. The LLM's generation is modelled as a fixed virtual
duration (GEN_MS); everything beyond that is overhead we control.

    controllable_overhead = total_virtual_wall_time - GEN_MS

GEN_MS is held identical across runs, so the delta between two runs is exactly
the latency we added or removed. Run it before and after a change to compare.

    python scripts/bench_overhead.py
"""

from __future__ import annotations

import asyncio
import random
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── Stub patchright (only used for type hints in the hot path) ──────────────
_patchright = types.ModuleType("patchright")
_async_api = types.ModuleType("patchright.async_api")


class _Stub:  # placeholder for Page / BrowserContext / etc.
    ...


for _name in ("Page", "BrowserContext", "Playwright", "Request", "Response", "Frame"):
    setattr(_async_api, _name, _Stub)
_async_api.async_playwright = _Stub
_impl = types.ModuleType("patchright._impl")
_errors = types.ModuleType("patchright._impl._errors")


class TargetClosedError(Exception):
    ...


_errors.TargetClosedError = TargetClosedError
_patchright.async_api = _async_api
sys.modules["patchright"] = _patchright
sys.modules["patchright.async_api"] = _async_api
sys.modules["patchright._impl"] = _impl
sys.modules["patchright._impl._errors"] = _errors

# Stub playwright_stealth (pulled in by src.browser.__init__ -> manager -> stealth)
_pw_stealth = types.ModuleType("playwright_stealth")


class Stealth:  # noqa: D401 - stub
    def __init__(self, *a, **k) -> None:
        self.script_payload = ""


_pw_stealth.Stealth = Stealth
sys.modules["playwright_stealth"] = _pw_stealth


# ── Virtual clock ───────────────────────────────────────────────────────────
class Clock:
    def __init__(self) -> None:
        self.now_ms: float = 0.0

    def advance(self, ms: float) -> None:
        if ms > 0:
            self.now_ms += ms


# ── Per-operation synthetic costs (ms) — model real CDP / eval latency ──────
EVAL_BASE = 4.0          # a trivial page.evaluate round-trip
INNERTEXT_PER_CHAR = 0.01  # innerText forces layout; scales with response size
IMAGE_SCAN = 15.0        # DALL-E detection evaluate (DOM walk over the turn)
GRANT_PERMS = 30.0       # context.grant_permissions CDP call
WAIT_SELECTOR = 6.0      # wait_for_selector hit
CLICK = 3.0
HOVER = 3.0
MOUSE_MOVE = 3.0
INSERT_TEXT = 5.0

# ── Simulated DOM timeline (ms, relative to submit) ─────────────────────────
GEN_MS = 5000.0          # how long the model "generates" (non-controllable)
AUTO_SUBMIT_LATENCY = 250.0   # execCommand insert -> frontend auto-submits
TURN_RENDER_LATENCY = 150.0   # submit -> new assistant turn bubble appears
RESPONSE_LEN = 4000      # chars in the assistant reply
RESPONSE_TEXT = "x" * RESPONSE_LEN
PREV_TEXT_LEN = 400


class Stats:
    def __init__(self) -> None:
        self.sleep_ms = 0.0
        self.evals = 0
        self.innertext_evals = 0
        self.image_scans = 0
        self.grant_calls = 0
        self.send_clicks = 0


class FakeElement:
    def __init__(self, page: "FakePage", selector: str) -> None:
        self._page = page
        self._sel = selector

    async def click(self) -> None:
        self._page.clock.advance(CLICK)
        self._page._maybe_submit_via_click(self._sel)

    async def hover(self) -> None:
        self._page.clock.advance(HOVER)


class FakeLocator:
    def __init__(self, page: "FakePage", selector: str) -> None:
        self._page = page
        self._sel = selector

    @property
    def first(self) -> FakeElement:
        return FakeElement(self._page, self._sel)


class FakeKeyboard:
    def __init__(self, page: "FakePage") -> None:
        self._page = page

    async def insert_text(self, text: str) -> None:
        self._page.clock.advance(INSERT_TEXT)

    async def press(self, key: str) -> None:
        self._page.clock.advance(CLICK)
        if key.lower() == "enter":
            self._page._maybe_submit_via_click("button[data-testid=\"send-button\"]")


class FakeMouse:
    def __init__(self, page: "FakePage") -> None:
        self._page = page

    async def move(self, x: int, y: int, steps: int = 1) -> None:
        self._page.clock.advance(MOUSE_MOVE)


class FakeContext:
    def __init__(self, page: "FakePage") -> None:
        self._page = page

    async def grant_permissions(self, perms, **kwargs) -> None:
        self._page.clock.advance(GRANT_PERMS)
        self._page.stats.grant_calls += 1


# Known send-button selectors (clicking them submits).
SEND_SELECTORS = (
    'button[data-testid="send-button"]',
    "#composer-submit-button",
    "button[aria-label='Send prompt']",
    'button[aria-label="Send prompt"]',
)


class FakePage:
    """Drives the real send_message code against a simulated ChatGPT DOM."""

    def __init__(self, clock: Clock, stats: Stats, *, auto_submit: bool, gen_ms: float) -> None:
        self.clock = clock
        self.stats = stats
        self.auto_submit = auto_submit
        self.gen_ms = gen_ms
        self.keyboard = FakeKeyboard(self)
        self.mouse = FakeMouse(self)
        self.context = FakeContext(self)
        self.viewport_size = {"width": 1280, "height": 720}

        self.text_inserted_at: float | None = None
        self.explicit_submit_at: float | None = None
        self.clipboard = ""

    # ── submission model ────────────────────────────────────────────────
    def _maybe_submit_via_click(self, selector: str) -> None:
        tokens = ("send-button", "composer-submit", "Send prompt", "prompt-textarea ~ button")
        if any(t in selector for t in tokens):
            self.stats.send_clicks += 1
            if self.explicit_submit_at is None and not self._submitted(self.clock.now_ms):
                self.explicit_submit_at = self.clock.now_ms

    def _submit_at(self) -> float | None:
        if self.auto_submit and self.text_inserted_at is not None:
            return self.text_inserted_at + AUTO_SUBMIT_LATENCY
        return self.explicit_submit_at

    def _submitted(self, now: float) -> bool:
        s = self._submit_at()
        return s is not None and now >= s

    def _new_turn_present(self, now: float) -> bool:
        s = self._submit_at()
        return s is not None and now >= s + TURN_RENDER_LATENCY

    def _copy_ready(self, now: float) -> bool:
        s = self._submit_at()
        return s is not None and now >= s + self.gen_ms

    def _signature(self, now: float) -> str:
        return "2:assistant-new" if self._new_turn_present(now) else "0:assistant-prev"

    def _input_text(self, now: float) -> str:
        if self.text_inserted_at is None:
            return ""
        if self._submitted(now):
            return ""  # cleared on submit
        return RESPONSE_TEXT[:10] if self.clock.now_ms >= self.text_inserted_at else ""

    # ── event wiring / misc page surface ────────────────────────────────
    def on(self, *_args, **_kwargs) -> None:
        return None

    @property
    def url(self) -> str:
        return "https://chatgpt.com/?temporary-chat=true"

    async def wait_for_selector(self, selector, timeout=None, state=None):
        self.clock.advance(WAIT_SELECTOR)
        return FakeElement(self, selector)

    async def wait_for_load_state(self, *_a, **_k) -> None:
        self.clock.advance(EVAL_BASE)

    async def goto(self, *_a, **_k) -> None:
        self.clock.advance(EVAL_BASE)

    async def screenshot(self, *_a, **_k) -> None:
        return None

    async def set_input_files(self, *_a, **_k) -> None:
        self.clock.advance(EVAL_BASE)

    async def query_selector_all(self, *_a, **_k):
        self.clock.advance(EVAL_BASE)
        return []

    def locator(self, selector: str) -> FakeLocator:
        return FakeLocator(self, selector)

    # ── the heart: evaluate dispatch ────────────────────────────────────
    async def evaluate(self, js: str, arg=None):
        self.stats.evals += 1
        now = self.clock.now_ms

        # page error checks -> no error
        if "DNS_PROBE_FINISHED_NXDOMAIN" in js or "Something went wrong" in js:
            self.clock.advance(EVAL_BASE)
            return None

        # dismiss overlays
        if "info.dismissed" in js or "Continue generating" in js:
            self.clock.advance(EVAL_BASE)
            return {"dismissed": [], "found": []}

        # _dump_all_turns (debug heartbeat dump) -> list of turn dicts
        if "turns.map(" in js or "childTags" in js:
            self.clock.advance(EVAL_BASE)
            return []

        # human_type: clear input
        if "execCommand('selectAll'" in js or 'execCommand("selectAll"' in js:
            self.clock.advance(EVAL_BASE)
            return None

        # human_type: execCommand insert  -> marks text inserted (auto-submit arms)
        if "execCommand('insertText'" in js or 'execCommand("insertText"' in js:
            self.clock.advance(EVAL_BASE)
            if self.text_inserted_at is None:
                self.text_inserted_at = self.clock.now_ms
            return "ok"

        # human_type: paste-event fallback (not reached when execCommand ok)
        if "ClipboardEvent('paste'" in js:
            self.clock.advance(EVAL_BASE)
            if self.text_inserted_at is None:
                self.text_inserted_at = self.clock.now_ms
            return "ok"

        # clipboard
        if "clipboard.readText" in js:
            self.clock.advance(EVAL_BASE)
            return self.clipboard
        if "clipboard.writeText" in js:
            self.clock.advance(EVAL_BASE)
            self.clipboard = ""
            return None

        # copy-button click extraction
        if "btn.click()" in js:
            self.clock.advance(EVAL_BASE)
            if self._copy_ready(now) and self._new_turn_present(now):
                self.clipboard = RESPONSE_TEXT
                return {"clicked": True, "reason": "ok", "signature": self._signature(now)}
            return {"clicked": False, "reason": "no-copy-button", "signature": self._signature(now)}

        # send button state probe
        if "ariaDisabled" in js or ("send-button" in js and "disabled" in js):
            self.clock.advance(EVAL_BASE)
            return {"selector": SEND_SELECTORS[0], "disabled": False, "ariaDisabled": None,
                    "visible": True, "classes": ""}

        # copy-button count
        if "copy-turn-action-button" in js and "total" in js:
            self.clock.advance(EVAL_BASE)
            return 2 if self._copy_ready(now) else 1

        # assistant message count
        if "total++" in js or ("hasAssistantRole" in js and "total" in js):
            self.clock.advance(EVAL_BASE)
            return 2 if self._new_turn_present(now) else 1

        # latest-turn snapshot (per-poll workhorse) — MUST precede image branch
        if "hasCopyButton" in js:
            need_text = True
            if arg is False:
                need_text = False
            elif isinstance(arg, dict) and arg.get("needText") is False:
                need_text = False
            new_turn = self._new_turn_present(now)
            text = ""
            if need_text:
                text = (RESPONSE_TEXT if new_turn else "y" * PREV_TEXT_LEN)
                self.clock.advance(EVAL_BASE + len(text) * INNERTEXT_PER_CHAR)
                self.stats.innertext_evals += 1
            else:
                self.clock.advance(EVAL_BASE)
            return {
                "found": True,
                "index": 2 if new_turn else 0,
                "signature": self._signature(now),
                "hasCopyButton": self._copy_ready(now) and new_turn,
                "hasImage": False,
                "text": text,
            }

        # image detection (after snapshot: snapshot's hasImage selector also
        # mentions "Generated image", so it must not be routed here)
        if "Generated image" in js or 'div[id^="image-"]' in js:
            self.clock.advance(IMAGE_SCAN)
            self.stats.image_scans += 1
            return []

        # optimized auto-submit probe (single combined-signal evaluate)
        if "inputEmpty" in js or "submitProbe" in js:
            self.clock.advance(EVAL_BASE)
            return {"newAssistant": self._new_turn_present(now),
                    "stopVisible": self._new_turn_present(now) and not self._copy_ready(now),
                    "inputEmpty": self._input_text(now) == "",
                    "submitProbe": True}

        # stop-button visibility
        if "Stop streaming" in js or "Stop generating" in js or "stop-button" in js:
            self.clock.advance(EVAL_BASE)
            return self._new_turn_present(now) and not self._copy_ready(now)

        # image-turn text / DOM extract fallbacks
        if "ChatGPT said" in js:
            self.clock.advance(EVAL_BASE)
            return ""
        if "turn.innerText" in js and "previousSignature" in js:
            self.clock.advance(EVAL_BASE + RESPONSE_LEN * INNERTEXT_PER_CHAR)
            return RESPONSE_TEXT

        # new-chat turn count
        if "conversation-turn-\"]').length" in js or "conversation-turn-\\\"]').length" in js:
            self.clock.advance(EVAL_BASE)
            return 0

        # unmatched -> log so we can fix the harness
        print(f"  [unmatched evaluate] {js.strip()[:90]!r}", file=sys.stderr)
        self.clock.advance(EVAL_BASE)
        return None


async def _run_once(auto_submit: bool, gen_ms: float):
    from src.chatgpt.client import ChatGPTClient

    clock = Clock()
    stats = Stats()
    page = FakePage(clock, stats, auto_submit=auto_submit, gen_ms=gen_ms)

    # Patch asyncio.sleep -> advance virtual clock; random -> deterministic mid.
    real_sleep = asyncio.sleep

    async def fake_sleep(delay, *a, **k):
        ms = float(delay) * 1000.0
        clock.advance(ms)
        stats.sleep_ms += ms
        await real_sleep(0)

    orig_sleep = asyncio.sleep
    orig_uniform = random.uniform
    orig_randint = random.randint
    asyncio.sleep = fake_sleep
    random.uniform = lambda a, b: (a + b) / 2.0
    random.randint = lambda a, b: int((a + b) / 2)
    try:
        client = ChatGPTClient(page)
        resp = await client.send_message("Benchmark prompt — please reply.")
    finally:
        asyncio.sleep = orig_sleep
        random.uniform = orig_uniform
        random.randint = orig_randint

    total = clock.now_ms
    overhead = total - gen_ms
    return {
        "auto_submit": auto_submit,
        "total_ms": total,
        "overhead_ms": overhead,
        "sleep_ms": stats.sleep_ms,
        "evals": stats.evals,
        "innertext_evals": stats.innertext_evals,
        "image_scans": stats.image_scans,
        "grant_calls": stats.grant_calls,
        "resp_chars": len(resp.message),
    }


# Controllable-overhead captured from the PRE-optimization revision (run on the
# parent commit: `git stash` the src/ changes, `python scripts/bench_overhead.py`).
# Kept here so a normal run proves the >=25% target without needing the old code.
BASELINE = {
    "auto-submit": {"overhead_ms": 1483.0, "innertext_evals": 18, "image_scans": 1},
    "manual-send": {"overhead_ms": 4427.0, "innertext_evals": 19, "image_scans": 1},
}
TARGET_REDUCTION = 25.0


def main() -> int:
    print(f"GEN_MS (non-controllable LLM time) = {GEN_MS:.0f}ms, response = {RESPONSE_LEN} chars\n")
    worst = 100.0
    for auto in (True, False):
        r = asyncio.run(_run_once(auto, GEN_MS))
        label = "auto-submit" if auto else "manual-send"
        base = BASELINE[label]["overhead_ms"]
        opt = r["overhead_ms"]
        reduction = (base - opt) / base * 100.0 if base else 0.0
        worst = min(worst, reduction)
        print(f"[{label}]")
        print(f"  controllable overhead : baseline {base:7.0f} ms  ->  now {opt:7.0f} ms")
        print(f"  reduction             : {reduction:6.1f} %   (target >= {TARGET_REDUCTION:.0f}%)")
        print(f"  total wall            : {r['total_ms']:7.0f} ms  (incl. {GEN_MS:.0f}ms model gen)")
        print(f"  innerText evals       : {BASELINE[label]['innertext_evals']:4d}  ->  {r['innertext_evals']}")
        print(f"  image DOM scans       : {BASELINE[label]['image_scans']:4d}  ->  {r['image_scans']}")
        print()
    verdict = "PASS" if worst >= TARGET_REDUCTION else "FAIL"
    print(f"Worst-case reduction across scenarios: {worst:.1f}%  ->  {verdict}")
    return 0 if worst >= TARGET_REDUCTION else 1


if __name__ == "__main__":
    raise SystemExit(main())
