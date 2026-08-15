"""
Unit tests for OI Change Heatmap, Strike GEX & Range Forecast (v5.1).
"""
import pytest
import numpy as np
import pandas as pd
from src.data_engine import DataEngine
from src.options_flow import (
    compute_oi_change_heatmap,
    compute_strike_level_gex_chart_data,
    compute_oi_based_range_forecast
)

@pytest.fixture
def sample_option_chain():
    de = DataEngine(use_cache=False)
    oc = de.generate_synthetic_option_chain(spot=24500.0)
    return oc["dataframe"]

def test_oi_change_heatmap_structure(sample_option_chain):
    res = compute_oi_change_heatmap(sample_option_chain, spot=24500.0, range_pts=400.0)
    assert "heatmap_rows" in res
    assert len(res["heatmap_rows"]) > 0
    assert "writing_bias" in res
    assert res["writing_bias"] in ["CALL_WRITING_HEAVY_RESISTANCE", "PUT_WRITING_HEAVY_SUPPORT", "BALANCED_RANGE"]
    assert "hot_ce_strikes" in res
    assert "hot_pe_strikes" in res
    
    first_row = res["heatmap_rows"][0]
    assert "strike" in first_row
    assert "ce_change_oi" in first_row
    assert "pe_change_oi" in first_row
    assert "color_intensity" in first_row

def test_strike_level_gex_chart_data(sample_option_chain):
    res = compute_strike_level_gex_chart_data(sample_option_chain, spot=24500.0, iv=0.13, t_days=3.5)
    assert "strikes" in res
    assert len(res["strikes"]) > 0
    assert "net_gex_per_strike" in res
    assert "call_wall_strike" in res
    assert "put_wall_strike" in res
    assert "zero_gex_strike" in res
    assert res["net_dealer_regime"] in ["DEALER_LONG_GAMMA", "DEALER_SHORT_GAMMA"]

def test_oi_based_range_forecast(sample_option_chain):
    res = compute_oi_based_range_forecast(sample_option_chain, spot=24500.0, max_pain=24500.0)
    assert "put_wall" in res
    assert "call_wall" in res
    assert res["put_wall"] <= res["call_wall"]
    assert 0.0 <= res["spot_position_pct"] <= 100.0
    assert res["location_bias"] in ["NEAR_SUPPORT_ACCUMULATION", "NEAR_RESISTANCE_DISTRIBUTION", "MID_RANGE_CONSOLIDATION"]
    assert "expected_corridor" in res
