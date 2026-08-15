"""Unit tests for upgraded VolatilityIntelligence methods in src/volatility_engine.py."""

import pytest
import numpy as np
import pandas as pd
from src.volatility_engine import VolatilityIntelligence


def test_volatility_cone_computation():
    np.random.seed(42)
    # Generate 100 periods of prices
    returns = np.random.normal(0.0002, 0.01, 100)
    prices = pd.Series(100.0 * np.exp(np.cumsum(returns)))

    res = VolatilityIntelligence.compute_volatility_cone(
        close_prices=prices,
        windows=[5, 10, 20, 60],
        percentiles=[10, 25, 50, 75, 90]
    )

    assert "windows" in res
    assert "percentiles" in res
    assert "cone_data" in res
    assert "cone_dataframe" in res
    assert len(res["cone_data"]) == 4
    for w in [5, 10, 20, 60]:
        assert w in res["cone_data"]
        w_data = res["cone_data"][w]
        assert "p10" in w_data
        assert "p50" in w_data
        assert "p90" in w_data
        assert "current_rv" in w_data
        assert w_data["min"] <= w_data["p50"] <= w_data["max"]


def test_volatility_cone_empty_or_small():
    prices = pd.Series([100.0, 101.0])
    res = VolatilityIntelligence.compute_volatility_cone(prices)
    assert "cone_data" in res
    assert len(res["cone_data"]) == 4


def test_rv_term_structure_classification():
    # Expanding regime (short term vol > long term vol)
    # Create high vol recent 10 bars and low vol older 50 bars
    np.random.seed(42)
    low_vol = np.random.normal(0, 0.002, 50)
    high_vol = np.random.normal(0, 0.03, 15)
    all_ret = np.concatenate([low_vol, high_vol])
    prices = pd.Series(100.0 * np.exp(np.cumsum(all_ret)))

    res = VolatilityIntelligence.compute_rv_term_structure(prices)
    assert "rv_5" in res
    assert "rv_20" in res
    assert "rv_60" in res
    assert res["rv_5"] > res["rv_20"]
    assert res["classification"] == "INVERTED_EXPANDING"

    # Compressing regime (short term vol < long term vol)
    high_vol_old = np.random.normal(0, 0.03, 50)
    low_vol_recent = np.random.normal(0, 0.001, 15)
    all_ret_comp = np.concatenate([high_vol_old, low_vol_recent])
    prices_comp = pd.Series(100.0 * np.exp(np.cumsum(all_ret_comp)))

    res_comp = VolatilityIntelligence.compute_rv_term_structure(prices_comp)
    assert res_comp["rv_5"] < res_comp["rv_20"]
    assert res_comp["classification"] == "NORMAL_COMPRESSING"
    assert bool(res_comp["compression_breakout_signal"]) is True


def test_iv_term_structure():
    # Contango: Near IV < Far IV
    contango_pairs = [("Near Weekly", 0.13), ("Far Monthly", 0.16)]
    res_contango = VolatilityIntelligence.compute_iv_term_structure(contango_pairs)
    assert res_contango["term_structure"] == "CONTANGO"
    assert res_contango["iv_spread"] > 0
    assert res_contango["near_iv"] == 0.13
    assert res_contango["far_iv"] == 0.16

    # Backwardation: Near IV > Far IV
    backwardation_pairs = [("Near Weekly", 0.18), ("Far Monthly", 0.14)]
    res_back = VolatilityIntelligence.compute_iv_term_structure(backwardation_pairs)
    assert res_back["term_structure"] == "BACKWARDATION"
    assert res_back["iv_spread"] < 0

    # Flat
    flat_pairs = [("Near Weekly", 0.140), ("Far Monthly", 0.141)]
    res_flat = VolatilityIntelligence.compute_iv_term_structure(flat_pairs)
    assert res_flat["term_structure"] == "FLAT"

    # Dict format
    res_dict = VolatilityIntelligence.compute_iv_term_structure({"near": 0.12, "far": 0.15})
    assert res_dict["term_structure"] == "CONTANGO"
