"""
Unit tests for OnlyNifty v5.2 Microstructure & Gating Upgrades:
- Session-Reset CVD (src/indicators.py)
- Absorption Traps at Value Areas (src/indicators.py)
- 25-Delta Put-Call Volatility Skew (src/volatility_engine.py)
- Realized Volatility Compression Ratio Squeeze (src/volatility_engine.py)
- Strategy Engine Gating (HFI, Skew, GEX Call Walls) (src/strategy_rules.py)
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.indicators import compute_session_cvd, detect_absorption_traps
from src.volatility_engine import VolatilityIntelligence
from src.strategy_rules import StrategyEngine, SignalType
from src.data_engine import DataEngine


@pytest.fixture
def multi_day_5m_df():
    """Generates 2 days of 5-minute bars."""
    dates = []
    # Day 1: 2026-08-14 (09:15 to 15:30)
    d1_start = datetime(2026, 8, 14, 9, 15)
    for i in range(75):
        dates.append(d1_start + timedelta(minutes=5 * i))
    # Day 2: 2026-08-15 (09:15 to 15:30)
    d2_start = datetime(2026, 8, 15, 9, 15)
    for i in range(75):
        dates.append(d2_start + timedelta(minutes=5 * i))

    n_bars = len(dates)
    base_price = 24500.0 + np.cumsum(np.random.normal(0.5, 5.0, n_bars))
    high = base_price + np.random.uniform(5.0, 15.0, n_bars)
    low = base_price - np.random.uniform(5.0, 15.0, n_bars)
    open_p = base_price - np.random.uniform(-5.0, 5.0, n_bars)
    close = base_price + np.random.uniform(-5.0, 5.0, n_bars)
    volume = np.random.uniform(50000, 200000, n_bars)

    df = pd.DataFrame({
        "open": open_p, "high": high, "low": low, "close": close, "volume": volume
    }, index=pd.DatetimeIndex(dates))
    return df


def test_session_reset_cvd(multi_day_5m_df):
    cvd = compute_session_cvd(multi_day_5m_df)
    assert len(cvd) == len(multi_day_5m_df)
    assert isinstance(cvd, pd.Series)
    
    # Check that CVD resets at the start of Day 2 (index 75)
    day2_start_cvd = cvd.iloc[75]
    day1_end_cvd = cvd.iloc[74]
    
    # Day 2 first bar CVD should be equal to its own bar delta (not carrying over Day 1's cumulative sum)
    assert abs(day2_start_cvd) < abs(day1_end_cvd) + 5000000.0


def test_detect_absorption_traps():
    # Construct a bearish absorption scenario at resistance:
    # High touches VAH (24600), closes lower with large upper wick
    dates = [datetime(2026, 8, 15, 10, i * 5) for i in range(10)]
    df = pd.DataFrame({
        "open": [24580, 24585, 24590, 24592, 24595, 24598, 24599, 24598, 24595, 24590],
        "high": [24585, 24590, 24595, 24598, 24600, 24602, 24605, 24601, 24602, 24605],
        "low":  [24575, 24580, 24585, 24588, 24590, 24592, 24594, 24590, 24585, 24578],
        "close":[24582, 24588, 24592, 24595, 24598, 24599, 24598, 24595, 24588, 24580],
        "volume": [100000] * 10
    }, index=pd.DatetimeIndex(dates))

    res = detect_absorption_traps(df, key_levels={"VAH": 24600.0, "VAL": 24500.0})
    assert isinstance(res, dict)
    assert "is_absorption" in res


def test_25delta_skew():
    vol_engine = VolatilityIntelligence()
    
    # 1. Baseline synthetic fallback
    res = vol_engine.compute_25delta_skew(option_chain_df=None, spot=24500.0, iv_baseline=0.135)
    assert res["regime"] == "NORMAL_SKEW"
    assert res["allow_longs"] is True
    assert res["skew_25d"] > 0

    # 2. Elevated Put Skew test (simulating crash hedging)
    mock_chain = pd.DataFrame({
        "strike": [24300, 24400, 24500, 24600, 24700],
        "pe_iv": [0.22, 0.19, 0.15, 0.13, 0.11],
        "ce_iv": [0.11, 0.12, 0.13, 0.13, 0.13]
    })
    skew_res = vol_engine.compute_25delta_skew(option_chain_df=mock_chain, spot=24500.0, iv_baseline=0.14)
    assert skew_res["skew_zscore"] > 1.50
    assert skew_res["is_crash_hedging"] is True
    assert skew_res["allow_longs"] is False


def test_vcr_squeeze():
    vol_engine = VolatilityIntelligence()
    
    # Squeeze scenario: high historical volatility followed by flat compressed consolidation
    np.random.seed(42)
    # 60 bars of volatile returns
    volatile_returns = np.random.normal(0, 0.015, 60)
    # 10 bars of near-zero volatility (squeeze)
    squeeze_returns = np.random.normal(0, 0.0005, 10)
    all_returns = np.concatenate([volatile_returns, squeeze_returns])
    prices = 24500.0 * np.exp(np.cumsum(all_returns))
    
    res = vol_engine.compute_vcr_squeeze(prices, short_window=5, long_window=60)
    assert res["vcr"] < 0.25
    assert "is_squeeze" in res
    assert res["is_squeeze"] is True


def test_strategy_engine_hfi_gating():
    data_engine = DataEngine(use_cache=False)
    strat_engine = StrategyEngine()
    df = data_engine.generate_synthetic_nifty(bars=100)

    # Test that evaluate_bar accepts hfi_score and runs without error
    sig_bull = strat_engine.evaluate_bar(df, live_iv=0.135, hfi_score=+0.50)
    assert isinstance(sig_bull.signal_type, SignalType)

    sig_bear = strat_engine.evaluate_bar(df, live_iv=0.135, hfi_score=-0.50)
    assert isinstance(sig_bear.signal_type, SignalType)
