import pytest
from src.data_engine import DataEngine
from src.backtest_engine import BacktestEngine

def test_backtest_execution():
    engine = DataEngine(use_cache=False)
    df = engine.generate_synthetic_nifty(bars=120, interval_mins=5)
    bt = BacktestEngine(initial_capital=500000.0)
    results = bt.run_backtest(df)
    
    assert results is not None
    assert "total_trades" in results.summary
    assert "win_rate" in results.summary
    assert "pnl_rupees" in results.summary
    assert "return_pct" in results.summary
    assert isinstance(results.trade_log, list)
    assert len(results.equity_curve) >= 1
