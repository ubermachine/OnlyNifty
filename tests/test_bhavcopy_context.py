"""Real positioning context from bhavcopy — the honest input for historical verdict tests.

Every prior historical test fed the strategy a SYNTHETIC chain (confluence 27-57) while the
live system scores 78-100, so we were measuring a configuration that does not exist in
production. Bhavcopy carries genuine per-strike OI, which reconstructs the gates the strategy
actually uses. Open interest is inherently end-of-day in Indian markets, so daily granularity
is closer to what was knowable at the time than any interpolated intraday figure.
"""

import pandas as pd
import pytest

from src.bhavcopy_context import build_context_from_bhavcopy


def chain(spot=24000, n=12, vol=500, oi=10000):
    rows = []
    for j in range(-n, n + 1):
        k = spot + j * 50
        for t in ("CE", "PE"):
            rows.append({"symbol": "NIFTY", "expiry": "2026-08-25", "strike": float(k),
                         "option_type": t, "close": max(5.0, 100 - abs(j) * 5),
                         "oi": oi + (1000 if (t == "CE" and j == 4) or (t == "PE" and j == -4) else 0),
                         "chg_oi": 10, "volume": vol, "underlying": spot})
    return pd.DataFrame(rows)


class TestContextConstruction:
    def test_builds_gates_the_strategy_needs(self):
        c = build_context_from_bhavcopy(chain(), "2026-08-19", 24000.0)
        assert c is not None
        assert c["source"] == "NSE_BHAVCOPY"
        assert c["gex_chart"]["walls_verified"] is True
        assert "dir_flow" in c and "chain_df" in c
        assert not c["chain_df"].empty

    def test_walls_bracket_spot(self):
        c = build_context_from_bhavcopy(chain(), "2026-08-19", 24000.0)
        g = c["gex_chart"]
        assert g["put_wall_strike"] < 24000.0 < g["call_wall_strike"]

    def test_walls_track_oi_concentration(self):
        c = build_context_from_bhavcopy(chain(), "2026-08-19", 24000.0)
        g = c["gex_chart"]
        assert g["call_wall_strike"] == pytest.approx(24200.0)  # CE OI peak at +4 strikes
        assert g["put_wall_strike"] == pytest.approx(23800.0)   # PE OI peak at -4 strikes


class TestRefusesToFabricate:
    def test_untraded_strikes_excluded(self):
        """Untraded contracts carry a settlement price, not a quote — inventing positioning
        from them manufactures signal (see nse_bhavcopy.tradeable)."""
        assert build_context_from_bhavcopy(chain(vol=0, oi=0), "2026-08-19", 24000.0) is None

    def test_thin_day_returns_none_not_a_guess(self):
        assert build_context_from_bhavcopy(chain(n=1), "2026-08-19", 24000.0) is None

    def test_empty_input_returns_none(self):
        assert build_context_from_bhavcopy(pd.DataFrame(), "2026-08-19", 24000.0) is None

    def test_invalid_spot_returns_none(self):
        assert build_context_from_bhavcopy(chain(), "2026-08-19", 0.0) is None
