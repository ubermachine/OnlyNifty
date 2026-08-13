import pytest
import pandas as pd
from src.data_engine import DataEngine
from src.indicators import compute_ema, compute_vwap, compute_cpr, compute_volume_profile
from src.strategy_rules import StrategyEngine
from src.options_engine import generate_option_trade_ticket, select_institutional_strike
from src.backtest_engine import BacktestEngine

def test_full_pipeline_smoke():
    engine = DataEngine(use_cache=False)
    df = engine.generate_synthetic_nifty(bars=60)
    
    # Check indicator pipeline
    df["ema200"] = compute_ema(df["close"], 200)
    df["vwap"], _, _ = compute_vwap(df)
    cpr = compute_cpr(df)
    vp = compute_volume_profile(df)
    assert "pivot" in cpr
    assert "poc" in vp
    
    # Check strategy & ticket pipeline
    strategy = StrategyEngine()
    signal = strategy.evaluate_bar(df)
    ticket = generate_option_trade_ticket(df.iloc[-1]["close"], signal)
    assert ticket is not None
    
    # Check backtest pipeline
    bt = BacktestEngine(initial_capital=500000.0)
    res = bt.run_backtest(df)
    assert res.summary["initial_capital"] == 500000.0
