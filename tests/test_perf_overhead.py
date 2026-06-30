"""
Regression guard for the controllable (non-LLM) overhead of a ChatGPT
send_message round-trip.

Drives the real client/detector code on a virtual clock via the deterministic
harness in ``scripts/bench_overhead.py`` and asserts the per-request overhead we
control stays well below the pre-optimization baseline. A change that reintroduces
a fixed multi-second poll, a redundant per-poll innerText, or an always-on image
scan will trip these thresholds.
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


class ControllableOverheadTest(unittest.TestCase):
    def _bench(self):
        import bench_overhead as b  # imported lazily so collection stays clean
        return b

    def test_reduction_at_least_25_percent_vs_baseline(self) -> None:
        b = self._bench()
        for auto, label in ((True, "auto-submit"), (False, "manual-send")):
            r = asyncio.run(b._run_once(auto, b.GEN_MS))
            base = b.BASELINE[label]["overhead_ms"]
            reduction = (base - r["overhead_ms"]) / base * 100.0
            self.assertGreaterEqual(
                reduction,
                b.TARGET_REDUCTION,
                f"{label}: controllable overhead reduction only {reduction:.1f}% "
                f"({base:.0f}ms -> {r['overhead_ms']:.0f}ms)",
            )

    def test_absolute_overhead_ceilings(self) -> None:
        # Hard ceilings (deterministic harness) so a regression is caught even
        # if the baseline constants are ever edited.
        b = self._bench()
        auto = asyncio.run(b._run_once(True, b.GEN_MS))
        manual = asyncio.run(b._run_once(False, b.GEN_MS))
        self.assertLess(auto["overhead_ms"], 1100, "auto-submit overhead regressed")
        self.assertLess(manual["overhead_ms"], 2700, "manual-send overhead regressed")

    def test_per_poll_innertext_and_image_scan_minimized(self) -> None:
        b = self._bench()
        r = asyncio.run(b._run_once(True, b.GEN_MS))
        # Baseline computed innerText on every 300ms poll (18 times for a 5s
        # generation); now only diagnostics fetch text.
        self.assertLessEqual(r["innertext_evals"], 2, "per-poll innerText regressed")
        # A copy-button completion is always a text turn, so no image DOM scan.
        self.assertEqual(r["image_scans"], 0, "image scan should be skipped on text turns")


if __name__ == "__main__":
    unittest.main()
