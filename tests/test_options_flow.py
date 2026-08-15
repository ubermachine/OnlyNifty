"""Unit test suite for src/options_flow.py."""

import pytest
import pandas as pd
import numpy as np

from src.options_flow import (
    compute_atm_straddle_metrics,
    compute_cumulative_oi_delta_and_traps,
    compute_pcr_momentum_derivative,
    compute_vanna_charm_drift_vector,
    compute_short_term_directional_vector,
    compute_oi_change_heatmap,
    compute_strike_level_gex_chart_data,
    compute_oi_based_range_forecast
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


def test_compute_oi_change_heatmap_call_writing_bias():
    spot = 24800.0
    # Simulate heavy call writing at 24900 and 25000
    chain_data = []
    for k in range(24400, 25250, 50):
        chain_data.append({
            "strikePrice": k,
            "CE_oi": 250000 if k == 25000 else 60000,
            "PE_oi": 200000 if k == 24600 else 50000,
            "CE_changeinOpenInterest": 80000 if k == 24900 else (60000 if k == 25000 else 5000),
            "PE_changeinOpenInterest": 3000 if k <= 24800 else -2000,
            "CE_volume": 120000,
            "PE_volume": 80000,
            "CE_ltp": 95.0,
            "PE_ltp": 85.0
        })
    df_chain = pd.DataFrame(chain_data)

    res = compute_oi_change_heatmap(df_chain, spot=spot, range_pts=400, highlight_threshold_mult=2.0)

    assert "heatmap_rows" in res
    assert len(res["heatmap_rows"]) > 0
    assert "hot_ce_strikes" in res
    assert 24900 in res["hot_ce_strikes"] or 25000 in res["hot_ce_strikes"]
    assert res["total_ce_writing"] > res["total_pe_writing"]
    assert res["writing_bias"] == "CALL_WRITING_HEAVY_RESISTANCE"
    assert res["resistance"] == 25000
    assert res["support"] == 24600
    assert "range_forecast" in res

    # Check structure of rows
    row = res["heatmap_rows"][0]
    assert "strike" in row
    assert "ce_oi" in row
    assert "pe_oi" in row
    assert "ce_change_oi" in row
    assert "pe_change_oi" in row
    assert "ce_volume" in row
    assert "pe_volume" in row
    assert "net_strike_oi_delta" in row
    assert "is_hot_ce" in row
    assert "is_hot_pe" in row
    assert -1.0 <= row["color_intensity"] <= 1.0


def test_compute_oi_change_heatmap_put_writing_bias_and_fallback():
    spot = 24800.0
    # Test with None fallback
    res_fb = compute_oi_change_heatmap(None, spot=spot, range_pts=500)
    assert len(res_fb["heatmap_rows"]) > 0
    assert res_fb["writing_bias"] in ["CALL_WRITING_HEAVY_RESISTANCE", "PUT_WRITING_HEAVY_SUPPORT", "BALANCED_RANGE"]
    assert "support" in res_fb
    assert "resistance" in res_fb

    # Test Put writing dominance
    chain_data = []
    for k in range(24500, 25150, 50):
        chain_data.append({
            "strike": k,
            "ce_oi": 50000,
            "pe_oi": 150000 if k == 24700 else 50000,
            "ce_change_oi": 1000,
            "pe_change_oi": 95000 if k == 24700 else 10000,
            "ce_volume": 40000,
            "pe_volume": 90000
        })
    df_chain = pd.DataFrame(chain_data)
    res_put = compute_oi_change_heatmap(df_chain, spot=spot, range_pts=300)
    assert res_put["writing_bias"] == "PUT_WRITING_HEAVY_SUPPORT"
    assert 24700 in res_put["hot_pe_strikes"]


def test_compute_strike_level_gex_chart_data():
    spot = 24800.0
    chain_data = []
    for k in range(24400, 25250, 50):
        chain_data.append({
            "strike": k,
            "ce_oi": 250000 if k == 25000 else 40000,
            "pe_oi": 300000 if k == 24600 else 45000,
            "ce_change_oi": 10000,
            "pe_change_oi": 15000
        })
    df_chain = pd.DataFrame(chain_data)

    gex_data = compute_strike_level_gex_chart_data(df_chain, spot=spot, iv=12.0, t_days=4.0)

    assert "strikes" in gex_data
    assert "net_gex_per_strike" in gex_data
    assert "call_gex_per_strike" in gex_data
    assert "put_gex_per_strike" in gex_data
    assert "call_wall_strike" in gex_data
    assert "put_wall_strike" in gex_data
    assert "zero_gex_strike" in gex_data
    assert gex_data["net_dealer_regime"] in ["DEALER_LONG_GAMMA", "DEALER_SHORT_GAMMA"]
    assert len(gex_data["strikes"]) == len(df_chain)
    assert gex_data["call_wall_strike"] == 25000.0
    assert gex_data["put_wall_strike"] == 24600.0

    # Fallback test with None
    fb_gex = compute_strike_level_gex_chart_data(None, spot=spot)
    assert len(fb_gex["strikes"]) > 0
    assert "zero_gex_strike" in fb_gex


def test_compute_oi_based_range_forecast_locations():
    spot_support = 24620.0
    chain_data = []
    for k in range(24400, 25250, 50):
        chain_data.append({
            "strike": k,
            "ce_oi": 300000 if k == 25000 else 30000,
            "pe_oi": 350000 if k == 24600 else 35000
        })
    df_chain = pd.DataFrame(chain_data)

    # 1. Near support (put wall: 24600, call wall: 25000, spot: 24620 -> (24620-24600)/400 = 5% -> NEAR_SUPPORT)
    res_sup = compute_oi_based_range_forecast(df_chain, spot=spot_support, max_pain=24800.0)
    assert res_sup["put_wall"] == 24600.0
    assert res_sup["call_wall"] == 25000.0
    assert res_sup["max_pain"] == 24800.0
    assert res_sup["spot_position_pct"] <= 35.0
    assert res_sup["location_bias"] == "NEAR_SUPPORT_ACCUMULATION"

    # 2. Near resistance (spot: 24980 -> (24980-24600)/400 = 95% -> NEAR_RESISTANCE)
    res_res = compute_oi_based_range_forecast(df_chain, spot=24980.0, max_pain=24800.0)
    assert res_res["spot_position_pct"] >= 65.0
    assert res_res["location_bias"] == "NEAR_RESISTANCE_DISTRIBUTION"

    # 3. Mid range (spot: 24800 -> 50% -> MID_RANGE)
    res_mid = compute_oi_based_range_forecast(df_chain, spot=24800.0, max_pain=24800.0)
    assert 35.0 < res_mid["spot_position_pct"] < 65.0
    assert res_mid["location_bias"] == "MID_RANGE_CONSOLIDATION"

    # 4. Fallback test
    res_fb = compute_oi_based_range_forecast(None, spot=24800.0)
    assert res_fb["put_wall"] < res_fb["call_wall"]
    assert 0.0 <= res_fb["spot_position_pct"] <= 100.0

