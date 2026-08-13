"""Unit tests for OnlyNifty v3.4 Markov Regime Switching, Kalman Filter, GEX Profile, HFI, and VaR."""

import pytest
import numpy as np
import pandas as pd

from src.regime_switching import KalmanFilterTrendEstimator, MarkovRegimeSwitcher
from src.data_engine import DataEngine
from src.options_engine import compute_full_chain_gex_profile, construct_ratio_spread
from src.performance_analytics import calculate_value_at_risk_and_cvar


def test_kalman_filter_trend_estimator():
    prices = pd.Series([24300.0 + i * 5.0 + np.random.normal(0, 1) for i in range(50)])
    kf = KalmanFilterTrendEstimator(process_noise_std=0.8, measurement_noise_std=3.5)
    
    df_kf = kf.filter_series(prices)
    assert not df_kf.empty
    assert "kalman_price" in df_kf.columns
    assert "kalman_velocity" in df_kf.columns
    assert "kalman_vel_zscore" in df_kf.columns
    
    # In an upward trend, estimated velocity should be positive
    avg_velocity = df_kf["kalman_velocity"].tail(10).mean()
    assert avg_velocity > 0.0


def test_markov_regime_switcher():
    switcher = MarkovRegimeSwitcher()
    
    # 1. Trending synthetic series
    dates = pd.date_range("2026-08-14 09:15", periods=40, freq="5min")
    trending_prices = np.linspace(24300, 24600, 40)
    df_trend = pd.DataFrame({"close": trending_prices}, index=dates)
    
    res_trend = switcher.infer_regimes(df_trend)
    assert res_trend["active_regime"] in ["LOW_VOL_TRENDING", "HIGH_VOL_EXPANSION"]
    assert res_trend["kelly_multiplier"] >= 0.75
    assert "state_probabilities" in res_trend


def test_heavyweight_flow_index():
    engine = DataEngine(use_cache=False)
    hfi = engine.fetch_heavyweight_flow_index()
    
    assert "hfi_score" in hfi
    assert "breadth_bias" in hfi
    assert "constituents" in hfi
    assert len(hfi["constituents"]) == 5
    assert hfi["total_top5_weight_pct"] == 41.2


def test_full_chain_gex_profile_and_walls():
    strikes = [24200, 24250, 24300, 24350, 24400, 24450, 24500]
    ce_oi = [100000, 250000, 500000, 1200000, 800000, 400000, 200000] # Max CE at 24350
    pe_oi = [300000, 600000, 1400000, 500000, 300000, 100000, 50000]  # Max PE at 24300
    
    df_chain = pd.DataFrame({
        "strike": strikes,
        "ce_oi": ce_oi,
        "pe_oi": pe_oi
    })

    gex = compute_full_chain_gex_profile(df_chain, spot=24330.0, iv=0.12, t_days=3.0)
    
    assert gex["call_wall"] == 24350.0
    assert gex["put_wall"] == 24300.0
    assert "zero_gex_strike" in gex
    assert "total_net_gex_cr" in gex
    assert "gex_df" in gex
    assert len(gex["gex_df"]) == len(strikes)


def test_ratio_spread_construction():
    spread_call = construct_ratio_spread(spot=24350.0, is_call=True, iv=0.14, t_days=3.0)
    assert spread_call["spread_name"] == "1:2 CE Ratio Spread"
    assert spread_call["long_leg"]["qty_multiplier"] == 1
    assert spread_call["short_leg"]["qty_multiplier"] == 2
    assert spread_call["max_profit_pts"] > 0.0
    assert spread_call["breakeven_point"] > spread_call["max_profit_strike"]

    spread_put = construct_ratio_spread(spot=24350.0, is_call=False, iv=0.14, t_days=3.0)
    assert spread_put["spread_name"] == "1:2 PE Ratio Spread"
    assert spread_put["breakeven_point"] < spread_put["max_profit_strike"]


def test_value_at_risk_and_cvar():
    returns = [0.01, 0.015, -0.008, -0.012, 0.02, -0.005, 0.014, -0.018, 0.005, -0.002]
    var_cvar = calculate_value_at_risk_and_cvar(returns, initial_capital=500000.0)
    
    assert var_cvar["var_95_pct"] > 0.0
    assert var_cvar["cvar_95_pct"] >= var_cvar["var_95_pct"]
    assert var_cvar["var_99_pct"] >= var_cvar["var_95_pct"]
    assert var_cvar["var_95_rupees"] > 0.0
