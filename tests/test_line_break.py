"""Tests for 6-Line Break and Monotonic EMA (15/20/50) Trend Engine."""
import numpy as np
import pandas as pd
import pytest

from src.indicators import compute_line_break, compute_line_break_trend
from src.config import LINE_BREAK_COUNT, LINE_BREAK_EMA_PERIODS


def test_line_break_prefix_stability():
    """
    CRITICAL INVARIANT: Prefix-Stability (No Look-Ahead / No Repainting).
    For any row k, computing on df[:k] must yield identical results for rows 0..k-1
    as computing on the full dataset.
    """
    np.random.seed(42)
    n = 100
    prices = 24000.0 + np.cumsum(np.random.normal(0, 15, n))
    dates = pd.date_range("2026-08-17 09:15", periods=n, freq="5min")
    df = pd.DataFrame({"close": prices, "open": prices, "high": prices + 5, "low": prices - 5, "volume": 1000}, index=dates)

    full_res = compute_line_break(df, lines=6)

    # Test prefix stability across multiple cutoffs
    for cutoff in [15, 30, 50, 75, 99]:
        prefix_df = df.iloc[:cutoff]
        prefix_res = compute_line_break(prefix_df, lines=6)

        # Assert all columns match identically on the prefix slice
        for col in ["lb_direction", "lb_high", "lb_low", "lb_reversal_up", "lb_reversal_dn", "lb_blocks_count"]:
            np.testing.assert_array_equal(
                prefix_res[col].values,
                full_res[col].iloc[:cutoff].values,
                err_msg=f"Prefix-stability violation in column {col} at cutoff {cutoff}"
            )


def test_line_break_continuation_and_reversal():
    """
    Verifies that continuation happens on simple higher high / lower low,
    while reversal requires breaking the extreme of the last 6 blocks.
    """
    # 7 consecutive higher closes to form 7 UP blocks
    closes = [24000, 24020, 24040, 24060, 24080, 24100, 24120, 24140]
    dates = pd.date_range("2026-08-17 09:15", periods=len(closes), freq="5min")
    df = pd.DataFrame({"close": closes}, index=dates)

    res = compute_line_break(df, lines=6)
    assert res["lb_direction"].iloc[-1] == 1
    assert res["lb_blocks_count"].iloc[-1] == 7
    # 6-block lowest low should be the low of the 2nd block (24020) since 7 blocks exist
    assert res["lb_reversal_dn"].iloc[-1] == 24020

    # A minor pullback to 24100 (above 24020) should NOT trigger a reversal
    closes_pullback = closes + [24100]
    df_pb = pd.DataFrame({"close": closes_pullback}, index=pd.date_range("2026-08-17 09:15", periods=len(closes_pullback), freq="5min"))
    res_pb = compute_line_break(df_pb, lines=6)
    assert res_pb["lb_direction"].iloc[-1] == 1
    assert res_pb["lb_flipped"].iloc[-1] == False
    assert res_pb["lb_blocks_count"].iloc[-1] == 7  # No new block created

    # A breakdown to 24000 (below 24020) MUST trigger a reversal to DOWN (-1)
    closes_rev = closes + [24000]
    df_rev = pd.DataFrame({"close": closes_rev}, index=pd.date_range("2026-08-17 09:15", periods=len(closes_rev), freq="5min"))
    res_rev = compute_line_break(df_rev, lines=6)
    assert res_rev["lb_direction"].iloc[-1] == -1
    assert res_rev["lb_flipped"].iloc[-1] == True
    assert res_rev["lb_blocks_count"].iloc[-1] == 8


def test_line_break_monotonic_ema_trend_synthesis():
    """
    Verifies that compute_line_break_trend emits BULLISH only when line break is UP
    and EMA stack is monotonically bullish (close >= 15 >= 20 >= 50).
    """
    n = 60
    # Strong upward trend
    prices = 24000.0 + np.linspace(0, 300, n)
    dates = pd.date_range("2026-08-17 09:15", periods=n, freq="5min")
    df = pd.DataFrame({"close": prices, "open": prices, "high": prices + 5, "low": prices - 5, "volume": 1000}, index=dates)

    res = compute_line_break_trend(df, lines=6, ema_periods=(15, 20, 50))
    assert res["lb_bias"].iloc[-1] == "BULLISH"
    assert res["ema_stack_bullish"].iloc[-1] == True
    assert res["lb_direction"].iloc[-1] == 1

    # Strong downward trend
    prices_down = 24500.0 - np.linspace(0, 300, n)
    df_down = pd.DataFrame({"close": prices_down, "open": prices_down, "high": prices_down + 5, "low": prices_down - 5, "volume": 1000}, index=dates)
    res_down = compute_line_break_trend(df_down, lines=6, ema_periods=(15, 20, 50))
    assert res_down["lb_bias"].iloc[-1] == "BEARISH"
    assert res_down["ema_stack_bearish"].iloc[-1] == True
    assert res_down["lb_direction"].iloc[-1] == -1
