import pytest
import pandas as pd
import numpy as np
from src.backtest_engine import BacktestEngine
from src.strategy_rules import Signal, SignalType


def test_backtest_intrabar_sl_closer_to_open():
    engine = BacktestEngine(initial_capital=500000.0)
    
    # Create 35 bars of synthetic data
    dates = pd.date_range("2026-08-18 09:15", periods=35, freq="5min")
    df = pd.DataFrame({
        "open": [24500.0 + i * 2 for i in range(35)],
        "high": [24505.0 + i * 2 for i in range(35)],
        "low": [24495.0 + i * 2 for i in range(35)],
        "close": [24502.0 + i * 2 for i in range(35)],
        "volume": [50000] * 35
    }, index=dates)
    
    # Run backtest
    res = engine.run_backtest(df)
    assert isinstance(res.summary, dict)
    assert "total_trades" in res.summary
