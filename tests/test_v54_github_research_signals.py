"""
Unit tests for Institutional v5.4 Upgrades (SVI Calibration, Peter Jäckel IV Solver, Black-76, DWV, and ML Institutional Flow Score).
"""

import math
import pytest
import numpy as np
import pandas as pd

from src.volatility_engine import VolatilityIntelligence
from src.options_engine import black_76_greeks, black_scholes_greeks
from src.options_positioning import (
    compute_delta_weighted_volume,
    compute_zero_gamma_level,
    compute_options_desk_state
)
from src.institutional_flow import (
    compute_institutional_flow_score,
    InstitutionalFlowAggregator
)
from src.strategy_rules import StrategyEngine, Signal, SignalType
from src.data_engine import DataEngine


def test_peter_jaeckel_implied_volatility():
    spot = 24500.0
    strike = 24500.0
    t_years = 4.0 / 365.0
    true_iv = 0.15
    r = 0.065

    # Generate a theoretical Black-Scholes call price with true_iv
    bs_res = black_scholes_greeks(spot, strike, t_days=4.0, r=r, sigma=true_iv, is_call=True)
    market_price = bs_res["price"]

    # Invert using Peter Jäckel solver
    solved_iv = VolatilityIntelligence.calculate_jaeckel_implied_volatility(
        price=market_price,
        spot=spot,
        strike=strike,
        t_years=t_years,
        r=r,
        q=0.0,
        is_call=True
    )

    assert abs(solved_iv - true_iv) < 0.005, f"Expected ~{true_iv}, got {solved_iv}"


def test_svi_surface_fitting_and_interpolation():
    spot = 24500.0
    t_years = 7.0 / 365.0
    strikes = [24000.0, 24200.0, 24400.0, 24500.0, 24600.0, 24800.0, 25000.0]
    # Realistic equity put skew
    ivs = [0.18, 0.165, 0.155, 0.145, 0.140, 0.138, 0.142]

    fitted = VolatilityIntelligence.fit_svi_surface(
        strikes=strikes,
        ivs=ivs,
        spot=spot,
        t_years=t_years
    )

    assert "a" in fitted and "b" in fitted and "rho" in fitted
    assert fitted["b"] > 0
    assert abs(fitted["rho"]) < 1.0

    # Test continuous interpolation at unobserved strike 24450
    interp_iv = VolatilityIntelligence.compute_svi_interpolated_iv(
        strike=24450.0,
        spot=spot,
        t_years=t_years,
        svi_params=fitted
    )

    assert 0.10 <= interp_iv <= 0.25
    assert isinstance(interp_iv, float)


def test_black_76_greeks():
    futures_price = 24550.0
    strike = 24500.0
    res_call = black_76_greeks(futures_price, strike, t_days=3.0, sigma=0.14, is_call=True)
    res_put = black_76_greeks(futures_price, strike, t_days=3.0, sigma=0.14, is_call=False)

    assert res_call["price"] > 0
    assert res_put["price"] > 0
    assert 0.0 < res_call["delta"] <= 1.0
    assert -1.0 <= res_put["delta"] < 0.0
    assert res_call["gamma"] > 0
    assert res_call["vega"] > 0
    assert res_call["theta"] < 0  # theta decay is negative


def test_institutional_flow_score_aggregation():
    # Bullish scenario
    bull_flow = compute_institutional_flow_score(
        pcr_zscore=1.5,
        dwv_score=0.8,
        fii_ls_ratio=1.8,
        vwap_dispersion_pct=0.4,
        hfi_score=0.6
    )
    assert bull_flow["flow_score"] >= 70.0
    assert bull_flow["bias"] == "BULLISH"
    assert bull_flow["can_long"] is True
    assert bull_flow["can_short"] is False

    # Bearish scenario
    bear_flow = compute_institutional_flow_score(
        pcr_zscore=-1.8,
        dwv_score=-0.7,
        fii_ls_ratio=0.5,
        vwap_dispersion_pct=-0.5,
        hfi_score=-0.7
    )
    assert bear_flow["flow_score"] <= 30.0
    assert bear_flow["bias"] == "BEARISH"
    assert bear_flow["can_long"] is False
    assert bear_flow["can_short"] is True


def test_strategy_engine_smart_money_flow_veto():
    engine = StrategyEngine()
    
    # Fake long candidate
    skew_info = {"is_crash_hedging": False, "skew_zscore": 0.0}
    vpin_info = {"vpin": 0.30}
    gex_info = {"is_positive_gamma": True, "call_wall_strike": 25000.0, "put_wall_strike": 24000.0}
    htf_regime = {"htf_aligned_long": True, "htf_aligned_short": False}
    
    # Bearish institutional flow (Flow Score = 20)
    options_context = {"flow_score": 20.0}

    passed, reason, audit = engine._apply_universal_gates(
        candidate_direction="LONG",
        close=24500.0,
        skew_info=skew_info,
        vpin_info=vpin_info,
        hfi_score=0.0,
        gex_info=gex_info,
        htf_regime=htf_regime,
        options_context=options_context
    )

    assert passed is False
    assert audit["veto_gate"] == "INSTITUTIONAL_FLOW_BEARISH_VETO"
    assert "Smart Money flow is net bearish" in reason
