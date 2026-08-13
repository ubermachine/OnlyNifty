"""Unit tests for JustNifty v3.0 Tier-1 Institutional Modules."""

import pytest
import pandas as pd
import numpy as np
from src.indicators import (
    compute_hurst_exponent, compute_vakc_envelopes,
    compute_order_flow_imbalance, compute_dealer_gex
)
from src.options_engine import black_scholes_greeks, calculate_tca_friction
from src.data_engine import DataEngine

def test_hurst_exponent():
    np.random.seed(42)
    # Strongly trending series
    trending_series = pd.Series(np.linspace(24000, 25000, 100) + np.random.normal(0, 5, 100))
    h_trend = compute_hurst_exponent(trending_series)
    assert "hurst" in h_trend
    assert 0.0 < h_trend["hurst"] < 1.0
    
    # Mean-reverting sine wave series
    sine_series = pd.Series(24500 + 50 * np.sin(np.linspace(0, 20, 100)) + np.random.normal(0, 2, 100))
    h_sine = compute_hurst_exponent(sine_series)
    assert "regime" in h_sine

def test_vakc_envelopes():
    engine = DataEngine()
    df = engine.generate_synthetic_nifty(bars=100)
    upper, lower = compute_vakc_envelopes(df)
    assert len(upper) == len(df)
    assert len(lower) == len(df)
    assert (upper > lower).all()

def test_order_flow_imbalance():
    engine = DataEngine()
    df = engine.generate_synthetic_nifty(bars=50)
    ofi = compute_order_flow_imbalance(df)
    assert "ofi" in ofi
    assert "cvd" in ofi
    assert "buyer_defense" in ofi

def test_dealer_gex():
    gex = compute_dealer_gex(24500.0, 15000000.0, 12000000.0)
    assert "net_gex_crores" in gex
    assert "is_positive_gamma" in gex
    assert "gamma_flip_strike" in gex

def test_second_order_greeks_and_tca():
    g = black_scholes_greeks(24500.0, 24500.0, t_days=4.0, is_call=True)
    assert "vanna" in g
    assert "charm" in g
    assert "volga" in g
    
    tca = calculate_tca_friction(entry_prem=142.50, exit_prem=188.00, total_qty=150, lots=6)
    assert tca["total_friction"] > 0
    assert tca["stt"] > 0
    assert tca["brokerage"] == 40.0  # ₹20 buy + ₹20 sell
