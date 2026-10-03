"""
Auto-login helper — detects missing login and prompts user to sign in.

Used by both the FastAPI server and the TUI to automatically trigger
first-time login when no existing session is found, instead of crashing.
"""

from __future__ import annotations

import asyncio
import sys
import time

from src.browser.manager import BrowserManager
from src.config import Config
from src.log import setup_logging

log = setup_logging("auto_login")

_LOGIN_WAIT_SECONDS = 15 * 60  # non-interactive sign-in window before giving up


async def ensure_logged_in(browser: BrowserManager) -> bool:
    """
    Check if the user is logged in. If not, guide them through login.

    This replaces the need to manually run `scripts/first_login.py`.
    Opens the browser to ChatGPT, waits for the user to sign in,
    and verifies the login before returning.

    Returns True if logged in (or successfully logged in now).
    Raises RuntimeError if login fails after the user presses Enter.
    """
    if await browser.is_logged_in():
        log.info("Already logged in")
        return True

    log.info("Not logged in — starting interactive login flow")

    provider_name = "Claude" if Config.PROVIDER == "claude" else "ChatGPT"
    target_url = Config.provider_url()

    print("\n" + "=" * 60)
    print(f"  🔐 {provider_name} Login Required — First-Time Setup")
    print("=" * 60)
    print(f"\n  Browser data dir: {Config.BROWSER_DATA_DIR}")
    print(f"  Target: {target_url}")
    print(f"\n  A Chrome window is open. Please:")
    print(f"  1. Sign in to {provider_name} with your account")
    print("  2. Complete any CAPTCHA / verification checks")
    print("  3. Wait until you see the chat interface")
    print("  4. Come back here and press Enter")
    print("\n" + "=" * 60 + "\n")

    # Wait for user to sign in
    if sys.stdin.isatty():
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: input("  Press ENTER after you've signed in successfully > ")
        )
    else:
        # Docker/supervisor: there is no terminal, so input() would block startup
        # forever and the API would never listen. Poll the page instead; the user
        # signs in through noVNC (port 6080) and the server continues by itself.
        print("  No terminal attached: waiting for sign-in in the browser window...\n", flush=True)
        deadline = time.monotonic() + _LOGIN_WAIT_SECONDS
        while time.monotonic() < deadline:
            if await browser.is_logged_in():
                print("\n  ✅ Login verified! Session saved.\n", flush=True)
                log.info("Non-interactive login completed successfully")
                return True
            await asyncio.sleep(10)

    # Give the page a moment to settle
    await asyncio.sleep(2)

    # Verify login
    if await browser.is_logged_in():
        print("\n  ✅ Login verified! Session saved.")
        print("  You won't need to sign in again.\n")
        log.info("Interactive login completed successfully")
        return True
    else:
        print("\n  ⚠️  Could not verify login.")
        print("  The session may still be saved — trying to continue...\n")
        log.warning("Login verification uncertain after interactive login")
        # Don't crash — let the caller decide what to do
        return False
