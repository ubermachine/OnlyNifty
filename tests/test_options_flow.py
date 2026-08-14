"""Unit test suite for src/options_flow.py."""

import pytest
import pandas as pd
import numpy as np

from src.options_flow import (
    compute_atm_straddle_metrics,
    compute_cumulative_oi_delta_and_traps,
    compute_pcr_momentum_derivative,
    compute_vanna_charm_drift_vector,
    compute_short_term_directional_vector
)


def test_compute_atm_straddle_metrics_theoretical():
    spot = 24810.0
    res = compute_atm_straddle_metrics(spot, option_chain_df=None, live_iv=12.0)
    
    assert res["atm_strike"] == 24800
    assert res["straddle_premium"] > 50.0
    assert res["upper_breakeven"] == round(res["atm_strike"] + res["straddle_premium"], 2)
    assert res["lower_breakeven"] == round(res["atm_strike"] - res["straddle_premium"], 2)
    assert res["range_width_pts"] == round(2 * res["straddle_premium"], 2)
    assert res["vol_state"] in ["VOL_EXPANSION", "THETA_DECAY"]
    assert 0.0 <= res["spot_range_pct"] <= 100.0


def test_compute_cumulative_oi_delta_and_traps_classification():
    spot = 24800.0
    
    # Create synthetic option chain with Put writing dominance (Bullish)
    chain_data = []
    for k in range(24600, 25050, 50):
        chain_data.append({
            "strikePrice": k,
            "CE_oi": 150000 if k == 24900 else 50000,
            "PE_oi": 180000 if k == 24700 else 60000,
            "CE_changeinOpenInterest": 10000 if k > 24800 else -5000,
            "PE_changeinOpenInterest": 50000 if k <= 24800 else 10000,
            "CE_ltp": 120.0,
            "PE_ltp": 110.0
        })
    df_chain = pd.DataFrame(chain_data)
    
    res = compute_cumulative_oi_delta_and_traps(df_chain, spot=spot)
    assert res["net_oi_delta"] > 0
    assert res["net_oi_pulse_score"] > 0
    assert res["call_wall"] >= 24800
    assert res["put_wall"] <= 24800
    assert len(res["strike_diagnostics"]) > 0


def test_compute_short_covering_trap_detection():
    spot = 24850.0
    # Simulate heavy Call unwinding without Put additions (Short covering trap)
    chain_data = []
    for k in range(24600, 25050, 50):
        chain_data.append({
            "strikePrice": k,
            "CE_oi": 80000,
            "PE_oi": 90000,
            "CE_changeinOpenInterest": -60000,
            "PE_changeinOpenInterest": 500,
            "CE_ltp": 150.0,
            "PE_ltp": 80.0
        })
    df_chain = pd.DataFrame(chain_data)
    res = compute_cumulative_oi_delta_and_traps(df_chain, spot=spot)
    
    assert res["trap_flag"] is True
    assert res["active_quadrant"] == "SHORT_COVERING_TRAP"
    assert "SHORT COVERING" in res["trap_warning"]


def test_compute_pcr_momentum_derivative():
    res_bull = compute_pcr_momentum_derivative(current_pcr=1.20, prev_pcr=1.00, delta_t_mins=15.0)
    assert res_bull["pcr_momentum_score"] > 0
    assert res_bull["status"] == "BULLISH_PCR_EXPANSION"

    res_bear = compute_pcr_momentum_derivative(current_pcr=0.80, prev_pcr=1.05, delta_t_mins=15.0)
    assert res_bear["pcr_momentum_score"] < 0
    assert res_bear["status"] == "BEARISH_PCR_COLLAPSE"


def test_compute_vanna_charm_drift_vector():
    res = compute_vanna_charm_drift_vector(spot=24800.0, strike=24800.0, iv=12.0, d_iv_dt=-0.02)
    assert "vanna" in res
    assert "charm_daily" in res
    assert -1.0 <= res["drift_score"] <= 1.0


def test_compute_short_term_directional_vector_synthesis():
    spot = 24800.0
    df = pd.DataFrame({
        "open": [24750.0, 24780.0, 24800.0],
        "high": [24760.0, 24790.0, 24810.0],
        "low": [24740.0, 24770.0, 24790.0],
        "close": [24755.0, 24785.0, 24800.0],
        "volume": [10000, 12000, 15000]
    })
    
    res = compute_short_term_directional_vector(
        spot=spot,
        df=df,
        live_iv=12.0,
        hfi_score=0.40
    )
    
    assert -1.0 <= res["directional_vector"] <= 1.0
    assert 0.0 <= res["conviction_pct"] <= 100.0
    assert res["bias"] in [
        "STRONG_BULLISH_LONG", "MILD_BULLISH_ACCUMULATION",
        "NEUTRAL_RANGEBOUND_CHOP", "MILD_BEARISH_DISTRIBUTION",
        "STRONG_BEARISH_SHORT"
    ]
    assert "suggested_action" in res
    assert "target_price" in res
    assert "stop_price" in res
    assert "component_scores" in res
