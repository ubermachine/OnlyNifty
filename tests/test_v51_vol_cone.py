"""
Unit tests for Volatility Cone & Realized Vol Term Structure (v5.1).
"""
import pytest
import numpy as np
import pandas as pd
from src.volatility_engine import VolatilityIntelligence

def test_volatility_cone_generation():
    # Synthetic random walk
    np.random.seed(42)
    returns = np.random.normal(0, 0.002, 300)
    prices = 24000.0 * np.exp(np.cumsum(returns))
    
    res = VolatilityIntelligence.compute_volatility_cone(prices, windows=[5, 10, 20, 60])
    assert "cone_data" in res
    assert "cone_dataframe" in res
    assert len(res["cone_dataframe"]) == 4
    
    # Check monotonic quantiles for 20-period window
    w20 = res["cone_data"][20]
    assert w20["p10"] <= w20["p25"] <= w20["p50"] <= w20["p75"] <= w20["p90"]

def test_rv_term_structure_classification():
    # Normal trend series
    prices = pd.Series([24000.0 + i*5.0 for i in range(100)])
    rv_res = VolatilityIntelligence.compute_rv_term_structure(prices)
    assert "rv_5" in rv_res
    assert "rv_20" in rv_res
    assert "classification" in rv_res
    assert rv_res["classification"] in ["INVERTED_EXPANDING", "NORMAL_COMPRESSING", "FLAT"]
    assert "compression_ratio" in rv_res

def test_iv_term_structure_contango_backwardation():
    contango_pairs = [("Near Expiry", 0.12), ("Monthly Expiry", 0.15)]
    res_c = VolatilityIntelligence.compute_iv_term_structure(contango_pairs)
    assert res_c["term_structure"] == "CONTANGO"
    assert res_c["iv_spread"] > 0
    
    backwardation_pairs = [("Near Expiry", 0.18), ("Monthly Expiry", 0.13)]
    res_b = VolatilityIntelligence.compute_iv_term_structure(backwardation_pairs)
    assert res_b["term_structure"] == "BACKWARDATION"
    assert res_b["iv_spread"] < 0
