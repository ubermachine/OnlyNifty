"""Unit tests for OnlyNifty v3.6 SVI Volatility Smile, Initial Balance Day Types, and Sectoral Breadth Momentum (SBM)."""

import pytest
import numpy as np
import pandas as pd

from src.options_engine import compute_svi_volatility_skew, generate_svi_smile_curve
from src.indicators import compute_initial_balance_and_day_type
from src.data_engine import DataEngine


def test_svi_volatility_skew_and_smile_curve():
    spot = 24350.0
    iv_atm = compute_svi_volatility_skew(spot, 24350.0, base_iv=0.12, is_call=True)
    iv_otm_call = compute_svi_volatility_skew(spot, 24650.0, base_iv=0.12, is_call=True)
    iv_otm_put = compute_svi_volatility_skew(spot, 24050.0, base_iv=0.12, is_call=False)
    
    assert iv_atm > 0.05
    # OTM Puts have structural skew premium
    assert iv_otm_put >= iv_atm
    
    # Full smile curve table
    df_smile = generate_svi_smile_curve(spot, base_iv=0.12)
    assert not df_smile.empty
    assert "strike" in df_smile.columns
    assert "call_iv_pct" in df_smile.columns
    assert "put_iv_pct" in df_smile.columns
    assert "skew_spread_bps" in df_smile.columns


def test_initial_balance_day_type_classification():
    # 1. Accumulating IB (< 12 bars)
    dates_short = pd.date_range("2026-08-14 09:15", periods=6, freq="5min")
    df_short = pd.DataFrame({
        "open": [24300] * 6, "high": [24330] * 6, "low": [24290] * 6, "close": [24310] * 6, "volume": [100000] * 6
    }, index=dates_short)
    ib_short = compute_initial_balance_and_day_type(df_short)
    assert ib_short["ib_established"] is False
    assert "ACCUMULATING" in ib_short["day_type"]

    # 2. Bullish Trend Day (Unilateral upward expansion >= 1.5x)
    dates_trend = pd.date_range("2026-08-14 09:15", periods=30, freq="5min")
    # IB range: 24300 to 24350 (50 pts range)
    highs = [24350] * 12 + list(np.linspace(24360, 24500, 18)) # expands by 150 pts (3.0x IB range)
    lows = [24300] * 12 + [24320] * 18
    opens = [24310] * 30
    closes = highs
    volumes = [150000] * 30
    df_trend = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}, index=dates_trend)

    ib_trend = compute_initial_balance_and_day_type(df_trend)
    assert ib_trend["ib_established"] is True
    assert "BULLISH_TREND_DAY" in ib_trend["day_type"]
    assert ib_trend["strategy_mode"] == "HOLD_RUNNERS_FOR_T3_MOONSHOT"


def test_sectoral_breadth_momentum_pulse():
    engine = DataEngine(use_cache=False)
    pulse = engine.fetch_sectoral_pulse()
    
    assert "sbm_score" in pulse
    assert "alignment" in pulse
    assert "conviction" in pulse
    assert "sectors" in pulse
    assert len(pulse["sectors"]) == 5
