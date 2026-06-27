from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_MISSING = object()
_STUBBED_MODULES = (
    "patchright",
    "patchright.async_api",
    "src",
    "src.chatgpt",
    "src.chatgpt.model_selector_scripts",
)
_previous_modules = {name: sys.modules.get(name, _MISSING) for name in _STUBBED_MODULES}


def _restore_modules() -> None:
    for name, module in _previous_modules.items():
        if module is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module



patchright = types.ModuleType("patchright")
async_api = types.ModuleType("patchright.async_api")
async_api.Page = type("Page", (), {})
patchright.async_api = async_api
sys.modules["patchright"] = patchright
sys.modules["patchright.async_api"] = async_api
src_pkg = types.ModuleType("src")
src_pkg.__path__ = [str(ROOT / "src")]
chatgpt_pkg = types.ModuleType("src.chatgpt")
chatgpt_pkg.__path__ = [str(ROOT / "src" / "chatgpt")]
sys.modules["src"] = src_pkg
sys.modules["src.chatgpt"] = chatgpt_pkg

spec = importlib.util.spec_from_file_location(
    "model_selector_under_test",
    ROOT / "src" / "chatgpt" / "model_selector.py",
)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load model_selector module")
model_selector = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = model_selector
spec.loader.exec_module(model_selector)
_restore_modules()

ModelOption = model_selector.ModelOption
_score_option = model_selector._score_option


class ModelSelectorTest(unittest.TestCase):
    def test_gpt_55_pro_alias_matches_visible_gpt_55_option(self) -> None:
        option = ModelOption(label="GPT-5.5", source="menuitem")

        score = _score_option(option, "gpt-5.5-pro", None)

        self.assertGreaterEqual(score, 18)

    def test_gpt_55_pro_alias_still_matches_when_intensity_is_present(self) -> None:
        option = ModelOption(label="GPT-5.5", source="menuitem")

        score = _score_option(option, "gpt-5.5-pro deep", "deep")

        self.assertGreaterEqual(score, 18)


if __name__ == "__main__":
    unittest.main()
