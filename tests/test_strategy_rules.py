import pandas as pd
import pytest
from src.strategy_rules import StrategyEngine, SignalType

def test_freak_candle_suppression():
    engine = StrategyEngine()
    dates = pd.date_range("2026-08-13 09:15", periods=3, freq="5min")
    df = pd.DataFrame({
        "open": [24500.0, 24510.0, 24520.0],
        "high": [24550.0, 24560.0, 24570.0],
        "low": [24480.0, 24490.0, 24500.0],
        "close": [24520.0, 24540.0, 24550.0],
        "volume": [10000, 12000, 15000]
    }, index=dates)
    signal = engine.evaluate_bar(df, current_idx=1)
    assert signal.signal_type == SignalType.WAIT
    assert "Opening 15-min range" in signal.reason

def test_3pm_breakout_strategy():
    engine = StrategyEngine()
    dates = pd.date_range("2026-08-13 14:55", periods=3, freq="5min")
    # index 1 is 15:00, index 2 is 15:05 breaking 15:00 high
    df = pd.DataFrame({
        "open": [24500.0, 24510.0, 24560.0],
        "high": [24520.0, 24550.0, 24600.0],
        "low": [24490.0, 24500.0, 24540.0],
        "close": [24510.0, 24530.0, 24590.0],
        "volume": [10000, 20000, 50000]
    }, index=dates)
    signal = engine.evaluate_bar(df, current_idx=2)
    assert signal.signal_type == SignalType.LONG_3PM
    assert signal.entry_price == 24590.0
    assert signal.sl_price == 24500.0  # Low of 15:00 candle

def test_ma_stretch_nuance_filter():
    engine = StrategyEngine()
    # Create 50 bars where price suddenly rockets far above 21 EMA
    dates = pd.date_range("2026-08-13 09:30", periods=50, freq="5min")
    prices = [24500.0 + i * 2 for i in range(49)] + [25500.0] # extreme stretch on last bar
    df = pd.DataFrame({
        "open": prices,
        "high": [p + 10 for p in prices],
        "low": [p - 10 for p in prices],
        "close": prices,
        "volume": [10000]*50
    }, index=dates)
    signal = engine.evaluate_bar(df, current_idx=49)
    # Should be rejected or gated with MA stretch warning
    assert signal.signal_type == SignalType.WAIT
    assert "overextended" in signal.reason or "pullback" in signal.reason
