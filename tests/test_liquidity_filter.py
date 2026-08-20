"""Untraded strikes carry a theoretical settlement price, not a fill. Filtering them is
not optional — over 1,234 trading days a 1SD/2SD condor measured +115.97/trade (t=9.32)
on unfiltered bhavcopy and +7.12/trade (t=0.60) once every leg was required to have traded.
94% of that "edge" was buying protection nobody was selling."""

import pandas as pd
import pytest

from src.nse_bhavcopy import tradeable


def frame():
    return pd.DataFrame([
        {"strike": 24000, "close": 120.0, "volume": 5000, "oi": 90000},   # liquid ATM
        {"strike": 26000, "close": 13.95, "volume": 0, "oi": 0},          # phantom far wing
        {"strike": 22000, "close": 11.20, "volume": 0, "oi": 4500},       # priced, never traded today
        {"strike": 24500, "close": 40.0, "volume": 120, "oi": 700},       # thin but real
    ])


class TestTradeableFilter:
    def test_drops_zero_volume_phantoms(self):
        out = tradeable(frame())
        assert 26000 not in out.strike.values
        assert 22000 not in out.strike.values, "priced but untraded is still not tradeable"
        assert 24000 in out.strike.values

    def test_thresholds_screen_thin_strikes(self):
        out = tradeable(frame(), min_volume=1000, min_oi=5000)
        assert list(out.strike.values) == [24000]

    def test_permissive_default_keeps_real_trades(self):
        assert len(tradeable(frame())) == 2

    def test_empty_input_is_safe(self):
        assert tradeable(pd.DataFrame()).empty

    def test_missing_columns_do_not_crash(self):
        df = pd.DataFrame([{"strike": 24000, "close": 100.0}])
        assert len(tradeable(df)) == 1
