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
    n = 20
    dates = pd.date_range("2026-08-13 13:30", periods=n, freq="5min")
    opens = [24500.0 + i * 2 for i in range(n)]
    closes = [24500.0 + i * 2 + (2.0 if i % 2 == 0 else -1.0) for i in range(n)]
    highs = [max(o, c) + 5.0 for o, c in zip(opens, closes)]
    lows = [min(o, c) - 5.0 for o, c in zip(opens, closes)]
    volumes = [10000] * (n - 2) + [20000, 50000]
    
    # index 18 is 15:00, index 19 is 15:05 breaking 15:00 high
    lows[18] = 24510.0
    highs[18] = 24540.0
    closes[18] = 24535.0
    
    opens[19] = 24535.0
    closes[19] = 24555.0  # Breaks above 15:00 high (24540)
    highs[19] = 24560.0
    lows[19] = 24530.0
    
    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes
    }, index=dates)
    
    opt_ctx = {
        "chain_df": pd.DataFrame({"strike": [24550, 24550], "type": ["CE", "PE"], "iv": [0.13, 0.13]}),
        "gex_chart": {"call_wall_strike": 25000.0, "put_wall_strike": 24000.0, "net_dealer_regime": "DEALER_LONG_GAMMA"},
        "dir_flow": {"directional_vector": 0.25}
    }
    signal = engine.evaluate_bar(df, current_idx=19, options_context=opt_ctx)
    assert signal.signal_type == SignalType.LONG_3PM
    assert signal.entry_price == 24555.0
    assert signal.sl_price == 24510.0  # Low of 15:00 candle (45 pts, within 15-60 bounds)
    assert signal.target_1 > signal.entry_price



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
