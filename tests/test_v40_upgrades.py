"""
Unit tests for OnlyNifty v4.0 Quantitative, Microstructure, Execution, and Risk Upgrades.
Covers:
- Phase 2: Detrended Fluctuation Analysis (DFA), Walk-Forward Optimizer, Multi-Asset Kalman Cointegration
- Phase 3: VPIN Toxicity, Volume-Synchronized Gamma Impact
- Phase 4: OrderManager Limit Chase & Cancel, Child Order Slicing
- Phase 5: Volatility-Adjusted Golden Vault, Regime-Scaled Dynamic Kelly Criterion
"""

import pytest
import numpy as np
import pandas as pd

from src.indicators import compute_dfa_alpha, compute_vpin_toxicity, compute_volume_synchronized_gamma_tracker
from src.regime_switching import MultiAssetKalmanCointegrator
from src.optimizer import WalkForwardOptimizer
from src.execution import OrderManager, OrderState, slice_institutional_order
from src.options_engine import evaluate_golden_vault_lock, calculate_dynamic_kelly


def test_dfa_alpha_computation():
    np.random.seed(42)
    # Trending series with cumulative positive drift
    trend_prices = pd.Series(100.0 + np.cumsum(np.random.normal(0.5, 0.2, 120)))
    dfa_trend = compute_dfa_alpha(trend_prices, min_lag=5, max_lag=40)
    assert "dfa_alpha" in dfa_trend
    assert 0.05 <= dfa_trend["dfa_alpha"] <= 0.95

    # Random walk
    rw_prices = pd.Series(100.0 + np.cumsum(np.random.normal(0.0, 1.0, 120)))
    dfa_rw = compute_dfa_alpha(rw_prices, min_lag=5, max_lag=40)
    assert 0.05 <= dfa_rw["dfa_alpha"] <= 0.95


def test_vpin_toxicity_bvc():
    np.random.seed(42)
    bars = 50
    df = pd.DataFrame({
        "close": 24000.0 + np.cumsum(np.random.normal(0, 10, bars)),
        "volume": np.random.uniform(5000, 20000, bars)
    })
    vpin_res = compute_vpin_toxicity(df, bucket_volume=15000)
    assert "vpin" in vpin_res
    assert 0.0 <= vpin_res["vpin"] <= 1.0
    assert "toxicity_level" in vpin_res
    assert "action_advice" in vpin_res


def test_volume_synchronized_gamma_tracker():
    strikes = [24200, 24250, 24300, 24350, 24400]
    call_vols = [10000.0, 25000.0, 80000.0, 15000.0, 5000.0]
    put_vols = [5000.0, 12000.0, 40000.0, 30000.0, 8000.0]
    call_gammas = [0.0004, 0.0007, 0.0011, 0.0008, 0.0005]
    put_gammas = [0.0005, 0.0008, 0.0010, 0.0007, 0.0004]

    gamma_tracker = compute_volume_synchronized_gamma_tracker(
        strikes=strikes,
        call_volumes=call_vols,
        put_volumes=put_vols,
        call_gammas=call_gammas,
        put_gammas=put_gammas,
        current_spot=24300.0
    )
    assert gamma_tracker["gamma_magnet_strike"] == 24300
    assert gamma_tracker["total_call_gamma_impact"] > 0
    assert gamma_tracker["pin_conviction"] is not None


def test_multi_asset_kalman_cointegrator():
    np.random.seed(42)
    nifty = pd.Series(24000.0 + np.cumsum(np.random.normal(0, 10, 40)))
    banknifty = nifty * 2.10 + np.random.normal(0, 20, 40)

    cointegrator = MultiAssetKalmanCointegrator()
    res = cointegrator.evaluate_spread_divergence(nifty, banknifty)

    assert "hedge_ratio_beta" in res
    assert 0.35 <= res["hedge_ratio_beta"] <= 0.65
    assert "spread_zscore" in res
    assert "is_divergent" in res


def test_order_manager_chase_and_cancel():
    om = OrderManager(tick_size=0.05, max_slippage_pts=3.0, passive_timeout_ms=500, aggressive_timeout_ms=1000)

    # 1. Passive Fill
    fill_pass = om.simulate_chase_and_cancel_execution("NIFTY 24300 CE", "BUY", initial_best_ask=150.0, simulated_market_drift_ticks=0, fill_latency_ms=250)
    assert fill_pass["final_state"] == OrderState.FILLED.value
    assert fill_pass["fill_price"] == 150.0

    # 2. Aggressive Fill (+1 tick)
    fill_agg = om.simulate_chase_and_cancel_execution("NIFTY 24300 CE", "BUY", initial_best_ask=150.0, simulated_market_drift_ticks=1, fill_latency_ms=750)
    assert fill_agg["final_state"] == OrderState.FILLED.value
    assert fill_agg["fill_price"] == 150.05

    # 3. Abort on extreme slippage
    fill_abort = om.simulate_chase_and_cancel_execution("NIFTY 24300 CE", "BUY", initial_best_ask=150.0, simulated_market_drift_ticks=100, fill_latency_ms=1500)
    assert fill_abort["final_state"] == OrderState.ABORTED.value
    assert fill_abort["is_successful"] is False


def test_child_order_slicer():
    # Large 20 lot order sliced into 4 VWAP child slices
    slices = slice_institutional_order(total_lots=20, lot_size=65, slice_count=4, interval_seconds=30, algo="VWAP")
    assert len(slices) == 4
    total_lots_allocated = sum(s["lots"] for s in slices)
    assert total_lots_allocated == 20
    assert slices[0]["scheduled_offset_sec"] == 0
    assert slices[-1]["scheduled_offset_sec"] == 90


def test_volatility_adjusted_golden_vault():
    # Vault locked with high volatility: runners permitted
    res_vol = evaluate_golden_vault_lock(
        initial_capital=100_000.0,
        current_intraday_pnl=1200.0, # Below 75% of 2000 peak (1500)
        peak_intraday_pnl=2000.0,
        profit_trigger_pct=0.015,
        lock_pct=0.75,
        realized_volatility=0.22, # > 1.30 * 0.12 (elevated)
        baseline_volatility=0.12
    )
    assert res_vol["is_vault_triggered"] is True
    assert res_vol["allow_runners"] is True


def test_regime_scaled_dynamic_kelly():
    # Trend Day: 1.5x Multiplier
    k_trend = calculate_dynamic_kelly(win_rate=0.70, payoff_ratio=2.0, day_type="BULLISH_TREND_DAY")
    assert k_trend["day_type_multiplier"] == 1.5
    assert k_trend["dynamic_risk_pct"] > 0.005

    # Neutral Day: 0.3x Multiplier
    k_chop = calculate_dynamic_kelly(win_rate=0.70, payoff_ratio=2.0, day_type="NEUTRAL_DAY")
    assert k_chop["day_type_multiplier"] == 0.3
    assert k_chop["dynamic_risk_pct"] < k_trend["dynamic_risk_pct"]


def test_walk_forward_optimizer():
    np.random.seed(42)
    bars = 60
    df = pd.DataFrame({
        "open": 24000.0 + np.cumsum(np.random.normal(0, 5, bars)),
        "high": 24010.0 + np.cumsum(np.random.normal(0, 5, bars)),
        "low": 23990.0 + np.cumsum(np.random.normal(0, 5, bars)),
        "close": 24000.0 + np.cumsum(np.random.normal(0, 5, bars)),
        "volume": np.random.uniform(10000, 50000, bars)
    })
    optimizer = WalkForwardOptimizer(initial_capital=100_000.0, output_file="data/test_optimized_params.json")
    params = optimizer.optimize(df, n_trials=3)
    assert "vakc_lambda" in params
    assert "dfa_trend_threshold" in params
    assert "vpin_threshold" in params
