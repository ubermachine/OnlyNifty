"""Stops must never sit inside single-bar noise.

On 2026-08-19 the day's highest-rated signal (SHORT_ORDER_FLOW, 100% confluence, HIGH
conviction) placed a 15-point spot stop derived from a 1.1-point "shelf" (24081.1-24082.2)
while 5m bars that session were ranging 20-27 points. A stop tighter than one bar's typical
range converts a directional read into a coin flip that still pays full round-trip friction.

A stop CAP (2x ATR) already existed; the FLOOR did not. This covers the floor.
"""

import pandas as pd
import pytest

from src.config import STOP_FLOOR_ATR_MULT
from src.strategy_rules import StrategyEngine, Signal, SignalType


def _engine_with_atr(atr):
    eng = StrategyEngine()
    eng._last_atr14 = atr
    return eng


def _apply(eng, signal):
    """Runs only the floor stage by calling the wrapper with a stubbed core."""
    eng._evaluate_bar_core = lambda *a, **k: signal
    df = pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]})
    return eng.evaluate_bar(df, current_idx=0)


class TestStopFloor:
    def test_tight_short_stop_is_widened(self):
        eng = _engine_with_atr(25.0)
        sig = Signal(SignalType.SHORT_ORDER_FLOW, entry_price=24072.65, sl_price=24087.65,
                     target_1=24042.65, target_2=24010.15)
        eng._last_atr14 = 25.0
        out = _apply(eng, sig)
        expected = 24072.65 + STOP_FLOOR_ATR_MULT * 25.0
        assert out.sl_price == pytest.approx(expected, abs=0.01)
        assert out.sl_price > 24087.65, "short stop must move further away from entry"
        assert "STOP FLOOR" in out.reason
        assert out.details["stop_floor_applied"]["original_pts"] == pytest.approx(15.0, abs=0.01)

    def test_tight_long_stop_is_widened_downward(self):
        eng = _engine_with_atr(25.0)
        sig = Signal(SignalType.LONG, entry_price=24000.0, sl_price=23990.0,
                     target_1=24050.0, target_2=24080.0)
        eng._last_atr14 = 25.0
        out = _apply(eng, sig)
        expected = 24000.0 - STOP_FLOOR_ATR_MULT * 25.0
        assert out.sl_price == pytest.approx(expected, abs=0.01)
        assert out.sl_price < 23990.0, "long stop must move further below entry"

    def test_already_wide_stop_untouched(self):
        """Trade 1 that day used 50pts on a 25pt ATR (2.0x) — must not be altered."""
        eng = _engine_with_atr(25.0)
        sig = Signal(SignalType.SHORT_ORDER_FLOW, entry_price=24095.45, sl_price=24145.45,
                     target_1=24065.45, target_2=24032.95)
        eng._last_atr14 = 25.0
        out = _apply(eng, sig)
        assert out.sl_price == pytest.approx(24145.45, abs=0.01)
        assert "STOP FLOOR" not in out.reason

    def test_floor_never_tightens(self):
        eng = _engine_with_atr(10.0)  # small ATR -> small floor
        sig = Signal(SignalType.LONG, entry_price=24000.0, sl_price=23900.0,
                     target_1=24100.0, target_2=24200.0)
        eng._last_atr14 = 10.0
        out = _apply(eng, sig)
        assert out.sl_price == pytest.approx(23900.0, abs=0.01), "wide stop must be preserved"

    def test_wait_signal_untouched(self):
        eng = _engine_with_atr(25.0)
        sig = Signal(SignalType.WAIT, entry_price=24000.0, sl_price=0.0,
                     target_1=0.0, target_2=0.0)
        eng._last_atr14 = 25.0
        out = _apply(eng, sig)
        assert out.sl_price == 0.0
        assert "STOP FLOOR" not in out.reason
