"""Unit tests for OnlyNifty v3.7 Multi-Sigma VWAP Dispersion, OU Mean-Reversion Half-Life, Footprint Delta Divergences, and Iron Condor Suite."""

import pytest
import numpy as np
import pandas as pd

from src.indicators import compute_vwap_multi_dispersion_and_half_life, detect_footprint_delta_divergences
from src.options_engine import construct_delta_neutral_iron_condor


def test_vwap_multi_dispersion_and_half_life():
    dates = pd.date_range("2026-08-14 09:15", periods=30, freq="5min")
    closes = np.linspace(24300, 24450, 30)
    df = pd.DataFrame({
        "open": closes - 5, "high": closes + 10, "low": closes - 10, "close": closes, "volume": [100000] * 30
    }, index=dates)

    disp = compute_vwap_multi_dispersion_and_half_life(df)
    
    assert "vwap" in disp
    assert "sigma_1_up" in disp and "sigma_1_down" in disp
    assert "sigma_2_up" in disp and "sigma_2_down" in disp
    assert "sigma_3_up" in disp and "sigma_3_down" in disp
    assert disp["sigma_3_up"] > disp["sigma_2_up"] > disp["sigma_1_up"] > disp["vwap"]
    assert "z_score_vwap" in disp
    assert "half_life_bars" in disp
    assert 2.0 <= disp["half_life_bars"] <= 60.0


def test_footprint_delta_divergences():
    dates = pd.date_range("2026-08-14 09:15", periods=20, freq="5min")
    
    # Simulate Bullish Delta Divergence: Lower Low on price, Higher Low on delta
    # Bar 0-9: Low at 24350 with heavy negative delta
    # Bar 10-19: Low at 24335 with positive/absorbed delta
    opens = [24380] * 10 + [24350] * 10
    highs = [24390] * 10 + [24360] * 10
    lows = [24350] * 10 + [24335] * 10 # lower low
    closes = [24355] * 10 + [24358] * 10 # higher closes
    volumes = [100000] * 20

    df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}, index=dates)
    div_res = detect_footprint_delta_divergences(df, lookback=10)
    
    assert "divergence_detected" in div_res
    assert "type" in div_res


def test_delta_neutral_iron_condor_construction():
    spot = 24395.85
    condor = construct_delta_neutral_iron_condor(
        spot=spot,
        wing_width=150,
        short_offset=100,
        t_days=3.5,
        iv=0.12
    )

    assert condor["status"] == "STRUCTURED"
    assert condor["strategy"] == "DELTA_NEUTRAL_IRON_CONDOR"
    assert "legs" in condor
    assert len(condor["legs"]) == 4
    assert condor["total_net_credit_pts"] > 0.0
    assert condor["max_loss_pts"] > 0.0
    assert condor["lower_breakeven"] < spot < condor["upper_breakeven"]
    assert condor["probability_of_profit_pct"] >= 50.0

