from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ConfigViewportTest(unittest.TestCase):
    def test_viewport_height_is_clamped_to_visible_display(self) -> None:
        previous = {name: os.environ.get(name) for name in ("DISPLAY_WIDTH", "DISPLAY_HEIGHT", "BROWSER_CHROME_HEIGHT")}
        os.environ["DISPLAY_WIDTH"] = "1366"
        os.environ["DISPLAY_HEIGHT"] = "768"
        os.environ["BROWSER_CHROME_HEIGHT"] = "148"
        try:
            spec = importlib.util.spec_from_file_location("config_viewport_under_test", ROOT / "src" / "config.py")
            if spec is None or spec.loader is None:
                raise RuntimeError("Could not load config module")
            config = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = config
            spec.loader.exec_module(config)

            viewport = config.Config.fit_viewport_to_display(1280, 740)
        finally:
            sys.modules.pop("config_viewport_under_test", None)
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

        self.assertEqual(viewport, (1280, 620))


    def test_allowed_chat_hosts_include_custom_ui_agents(self) -> None:
        previous = {
            name: os.environ.get(name)
            for name in ("CHATGPT_URL", "CLAUDE_URL", "CHAT_AGENT_ALLOWED_HOSTS")
        }
        os.environ["CHATGPT_URL"] = "https://chatgpt.com/?temporary-chat=true"
        os.environ["CLAUDE_URL"] = "https://claude.ai"
        os.environ["CHAT_AGENT_ALLOWED_HOSTS"] = "https://agent.example.test/app, gemini.google.com"
        try:
            spec = importlib.util.spec_from_file_location("config_hosts_under_test", ROOT / "src" / "config.py")
            if spec is None or spec.loader is None:
                raise RuntimeError("Could not load config module")
            config = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = config
            spec.loader.exec_module(config)

            hosts = config.Config.allowed_chat_hosts()
        finally:
            sys.modules.pop("config_hosts_under_test", None)
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

        self.assertIn("chatgpt.com", hosts)
        self.assertIn("claude.ai", hosts)
        self.assertIn("agent.example.test", hosts)
        self.assertIn("gemini.google.com", hosts)

if __name__ == "__main__":
    unittest.main()
