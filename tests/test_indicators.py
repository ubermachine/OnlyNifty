import numpy as np
import pandas as pd
import pytest
from src.indicators import (
    compute_ema,
    compute_envelopes,
    compute_vwap,
    compute_cpr,
    compute_fibonacci_levels,
    compute_vf_trade_table,
    compute_volume_profile
)

def test_ema_and_envelopes():
    series = pd.Series(np.linspace(100, 200, 300))
    ema200 = compute_ema(series, 200)
    assert len(ema200) == 300
    upper, lower = compute_envelopes(ema200, pct=0.015)
    assert (upper > ema200).dropna().all()
    assert (lower < ema200).dropna().all()

def test_vwap():
    dates = pd.date_range("2026-08-13 09:15", periods=50, freq="5min")
    df = pd.DataFrame({
        "high": [105.0]*50, "low": [95.0]*50, "close": [100.0]*50, "volume": [1000]*50
    }, index=dates)
    vwap, upper_sd, lower_sd = compute_vwap(df)
    assert len(vwap) == 50
    assert np.isclose(vwap.iloc[-1], 100.0)

def test_cpr():
    daily_df = pd.DataFrame({
        "high": [24600.0], "low": [24400.0], "close": [24500.0]
    })
    cpr = compute_cpr(daily_df)
    assert "pivot" in cpr
    assert "bc" in cpr
    assert "tc" in cpr
    assert "is_narrow" in cpr
    assert np.isclose(cpr["pivot"], 24500.0)

def test_fibonacci_levels():
    fib_up = compute_fibonacci_levels(high=24600.0, low=24400.0, is_uptrend=True)
    assert fib_up["fib_500"] == 24500.0
    assert fib_up["fib_618"] == 24476.4
    assert fib_up["sl_level"] < 24476.4

def test_vf_trade_table():
    vf = compute_vf_trade_table(open_price=24500.0, atr=50.0)
    for i in range(1, 7):
        assert f"T{i}_Long" in vf
        assert f"T{i}_Short" in vf
        assert vf[f"T{i}_Long"] > 24500.0
        assert vf[f"T{i}_Short"] < 24500.0

def test_volume_profile():
    df = pd.DataFrame({
        "high": [24520, 24550, 24530, 24540, 24510],
        "low": [24480, 24500, 24490, 24500, 24470],
        "close": [24510, 24520, 24510, 24530, 24490],
        "volume": [10000, 50000, 20000, 15000, 10000]
    }, index=pd.date_range("2026-08-13 09:15", periods=5, freq="5min"))
    vp = compute_volume_profile(df, n_bins=10)
    assert "poc" in vp
    assert "vah" in vp
    assert "val" in vp
    assert vp["poc"] >= df["low"].min()
    assert vp["poc"] <= df["high"].max()
