"""Unit tests for OnlyNifty v3.5 Global Macro Engine, Iceberg Orders, Liquidity Sweeps, and Volatility Ratio Scaling."""

import pytest
import numpy as np
import pandas as pd

from src.macro_engine import GlobalMacroEngine
from src.indicators import detect_iceberg_orders_and_liquidity_sweeps
from src.strategy_rules import StrategyEngine, SignalType


def test_global_macro_engine_sentiment_score():
    macro = GlobalMacroEngine()
    snap = macro.fetch_global_macro_snapshot(current_spot=24395.85)
    
    assert "macro_sentiment_score" in snap
    assert -1.0 <= snap["macro_sentiment_score"] <= 1.0
    assert "macro_bias" in snap
    assert "components" in snap
    assert len(snap["components"]) == 5
    assert "gift_basis_pts" in snap


def test_iceberg_order_detection():
    # Construct 25 bars with normal volume, then 1 bar with 3x volume on tiny range
    dates = pd.date_range("2026-08-14 09:15", periods=25, freq="5min")
    opens = [24400.0] * 24 + [24402.0]
    highs = [24415.0] * 24 + [24405.0] # 3 pt candle range (very narrow)
    lows = [24390.0] * 24 + [24402.0]
    closes = [24405.0] * 24 + [24404.0]
    volumes = [100000] * 24 + [350000] # 3.5x volume surge

    df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}, index=dates)
    micro = detect_iceberg_orders_and_liquidity_sweeps(df)

    assert micro["iceberg_detected"] is True
    assert micro["iceberg_event"] is not None
    assert micro["iceberg_event"]["volume_multiple"] >= 2.5
    assert "BUY_ICEBERG" in micro["iceberg_side"]


def test_liquidity_sweep_ssl_bullish_trap():
    dates = pd.date_range("2026-08-14 09:15", periods=25, freq="5min")
    # Establish a swing low at 24350
    opens = [24380.0] * 20 + [24370.0, 24365.0, 24360.0, 24355.0, 24354.0]
    highs = [24390.0] * 20 + [24375.0, 24370.0, 24365.0, 24360.0, 24362.0]
    lows = [24360.0] * 20 + [24355.0, 24352.0, 24350.0, 24350.0, 24335.0] # Probes below 24350 to 24335
    closes = [24375.0] * 20 + [24365.0, 24360.0, 24355.0, 24352.0, 24358.0] # Rebounds and closes at 24358 (above 24350 with 75% lower wick)
    volumes = [120000] * 25

    df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}, index=dates)
    micro = detect_iceberg_orders_and_liquidity_sweeps(df, lookback_swing=15)

    assert micro["liquidity_sweep_detected"] is True
    assert micro["sweep_event"] is not None
    assert micro["sweep_event"]["type"] == "SSL_SWEEP"
    assert micro["sweep_event"]["side"] == "LONG"
    assert "SSL Trap" in micro["sweep_side"]


def test_liquidity_sweep_bsl_bearish_trap():
    dates = pd.date_range("2026-08-14 09:15", periods=25, freq="5min")
    # Establish a swing high at 24450
    opens = [24420.0] * 20 + [24430.0, 24435.0, 24440.0, 24445.0, 24446.0]
    highs = [24440.0] * 20 + [24445.0, 24448.0, 24450.0, 24450.0, 24468.0] # Probes above 24450 to 24468
    lows = [24410.0] * 20 + [24425.0, 24430.0, 24435.0, 24440.0, 24438.0]
    closes = [24430.0] * 20 + [24435.0, 24440.0, 24445.0, 24448.0, 24442.0] # Rebounds and closes at 24442 (below 24450 with 75% upper wick)
    volumes = [120000] * 25

    df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}, index=dates)
    micro = detect_iceberg_orders_and_liquidity_sweeps(df, lookback_swing=15)

    assert micro["liquidity_sweep_detected"] is True
    assert micro["sweep_event"] is not None
    assert micro["sweep_event"]["type"] == "BSL_SWEEP"
    assert micro["sweep_event"]["side"] == "SHORT"
    assert "BSL Trap" in micro["sweep_side"]
