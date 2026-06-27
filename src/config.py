"""
Centralized configuration — loads from .env with sensible defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv

_CODE_ROOT = Path(__file__).resolve().parent.parent
_CWD = Path.cwd()

# Prefer the invocation directory as project root when running from
# a checkout (e.g. `nix run .#proxy` from repo root). Fall back to the
# code location (used for packaged/store execution).
if (_CWD / "src").exists() and (_CWD / "scripts").exists():
    _PROJECT_ROOT = _CWD
else:
    _PROJECT_ROOT = _CODE_ROOT

# Load .env from current working directory first, then from the
# resolved project root.
load_dotenv(_CWD / ".env")
load_dotenv(_PROJECT_ROOT / ".env")


def _optional_int(value: str | None) -> int | None:
    if value is None or value.strip() == "":
        return None
    return int(value)


class Config:
    """All project settings in one place."""

    # Paths
    PROJECT_ROOT: Path = _PROJECT_ROOT
    BROWSER_DATA_DIR: Path = _PROJECT_ROOT / os.getenv("BROWSER_DATA_DIR", "browser_data")
    LOG_DIR: Path = _PROJECT_ROOT / os.getenv("LOG_DIR", "logs")
    IMAGES_DIR: Path = _PROJECT_ROOT / os.getenv("IMAGES_DIR", "downloads/images")

    # Browser
    HEADLESS: bool = os.getenv("HEADLESS", "false").lower() == "true"
    SLOW_MO: int = int(os.getenv("SLOW_MO", "25"))
    CHATGPT_URL: str = os.getenv("CHATGPT_URL", "https://chatgpt.com")
    CLAUDE_URL: str = os.getenv("CLAUDE_URL", "https://claude.ai")


    # Browser navigation safety. Defaults cover current providers; extra
    # UI-agent hosts can be added as comma-separated hostnames/URLs.
    CHAT_AGENT_ALLOWED_HOSTS: str = os.getenv("CHAT_AGENT_ALLOWED_HOSTS", "")

    @staticmethod
    def _hostname_from_url_or_host(value: str) -> str:
        stripped = value.strip()
        if not stripped:
            return ""
        parsed = urlparse(stripped if "://" in stripped else f"https://{stripped}")
        return (parsed.hostname or "").lower()

    @classmethod
    def allowed_chat_hosts(cls) -> set[str]:
        """Return HTTPS hosts the browser session may navigate to."""
        defaults = (cls.CHATGPT_URL, cls.CLAUDE_URL, "chatgpt.com", "chat.openai.com", "claude.ai")
        configured = tuple(cls.CHAT_AGENT_ALLOWED_HOSTS.split(","))
        return {
            host
            for host in (cls._hostname_from_url_or_host(item) for item in (*defaults, *configured))
            if host
        }
    # Provider selection: "chatgpt" or "claude"
    PROVIDER: str = os.getenv("PROVIDER", "chatgpt").lower()
    DEFAULT_MODEL: str = os.getenv("CATGPT_MODEL", "gpt-5.5-pro")

    @classmethod
    def provider_url(cls) -> str:
        """Return the target URL for the active provider."""
        if cls.PROVIDER == "claude":
            return cls.CLAUDE_URL
        return cls.CHATGPT_URL

    # Timeouts (ms)
    RESPONSE_TIMEOUT: int = int(os.getenv("RESPONSE_TIMEOUT", "2100000"))
    SELECTOR_TIMEOUT: int = int(os.getenv("SELECTOR_TIMEOUT", "10000"))
    # Max time to wait for an attachment to finish uploading before sending (ms).
    UPLOAD_TIMEOUT: int = int(os.getenv("UPLOAD_TIMEOUT", "180000"))

    # Human simulation (ms)
    TYPING_SPEED_MIN: int = int(os.getenv("TYPING_SPEED_MIN", "50"))
    TYPING_SPEED_MAX: int = int(os.getenv("TYPING_SPEED_MAX", "150"))
    THINKING_PAUSE_MIN: int = int(os.getenv("THINKING_PAUSE_MIN", "500"))
    THINKING_PAUSE_MAX: int = int(os.getenv("THINKING_PAUSE_MAX", "1500"))
    # Completion poll interval — how often to check if response is ready (ms)
    POLL_INTERVAL_MS: int = int(os.getenv("POLL_INTERVAL_MS", "300"))
    # Deep liveness probe interval for slow pro models that can take many
    # minutes to respond — verifies generation is still progressing (ms).
    LIVENESS_INTERVAL_MS: int = int(os.getenv("LIVENESS_INTERVAL_MS", "180000"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "DEBUG")
    VERBOSE: bool = os.getenv("VERBOSE", "true").lower() == "true"

    # API (Phase 3)
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    RATE_LIMIT_SECONDS: int = int(os.getenv("RATE_LIMIT_SECONDS", "5"))
    API_TOKEN: str = os.getenv("API_TOKEN", "")  # Bearer token for API auth (empty = no auth)

    # VNC
    VNC_PASSWORD: str = os.getenv("VNC_PASSWORD", "catgpt")

    # Viewport base (will be jittered ±20px), bounded to the visible display.
    VIEWPORT_WIDTH: int = int(os.getenv("VIEWPORT_WIDTH", "1280"))
    VIEWPORT_HEIGHT: int = int(os.getenv("VIEWPORT_HEIGHT", "720"))
    DISPLAY_WIDTH: int | None = _optional_int(os.getenv("DISPLAY_WIDTH"))
    DISPLAY_HEIGHT: int | None = _optional_int(os.getenv("DISPLAY_HEIGHT"))
    MIN_VIEWPORT_WIDTH: int = int(os.getenv("MIN_VIEWPORT_WIDTH", "960"))
    MIN_VIEWPORT_HEIGHT: int = int(os.getenv("MIN_VIEWPORT_HEIGHT", "480"))
    BROWSER_CHROME_HEIGHT: int = int(os.getenv("BROWSER_CHROME_HEIGHT", "148"))

    @classmethod
    def fit_viewport_to_display(cls, width: int, height: int) -> tuple[int, int]:
        """Keep the browser viewport inside the visible headed display."""
        bounded_width = width
        bounded_height = height
        if cls.DISPLAY_WIDTH is not None:
            bounded_width = min(bounded_width, max(cls.MIN_VIEWPORT_WIDTH, cls.DISPLAY_WIDTH))
        if cls.DISPLAY_HEIGHT is not None:
            max_height = max(cls.MIN_VIEWPORT_HEIGHT, cls.DISPLAY_HEIGHT - cls.BROWSER_CHROME_HEIGHT)
            bounded_height = min(bounded_height, max_height)
        return bounded_width, bounded_height

    @classmethod
    def ensure_dirs(cls) -> None:
        """Create required directories if they don't exist."""
        cls.BROWSER_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
        cls.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
