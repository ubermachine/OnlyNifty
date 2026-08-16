"""Exhaustive mathematical and PDE edge-case verification test suite for Option Pricing Engine."""

import pytest
import math
from src.options_engine import (
    black_scholes_greeks,
    calculate_position_size,
    convert_to_free_vertical_spread,
    generate_option_trade_ticket,
    calculate_tca_friction,
    compute_volatility_surface
)
from src.strategy_rules import Signal, SignalType

def test_put_call_parity():
    """Verify European Put-Call Parity: C - P = S*exp(-q*T) - K*exp(-r*T)"""
    spot = 24500.0
    strike = 24500.0
    t_days = 5.0
    r = 0.065
    sigma = 0.12
    
    call = black_scholes_greeks(spot, strike, t_days=t_days, r=r, sigma=sigma, is_call=True)
    t_years = t_days / 365.0
    # Both legs must be priced off the SAME sigma at the same strike. This test used to
    # pass `sigma - 0.025` to the put to pre-compensate for a PUT_SKEW_PREMIUM the pricer
    # added back only to puts — i.e. the test encoded the parity bug as a workaround.
    # Skew is now applied by moneyness (same for C and P at a strike), so parity holds
    # directly and the tolerance can be tight.
    put = black_scholes_greeks(spot, strike, t_days=t_days, r=r, sigma=sigma, is_call=False)

    c_price = call["price"]
    p_price = put["price"]
    theoretical_diff = spot - strike * math.exp(-r * t_years)
    assert abs((c_price - p_price) - theoretical_diff) < 0.05
    # Second-order greeks are type-independent under parity.
    assert call["gamma"] == pytest.approx(put["gamma"], rel=1e-9)
    assert call["vega"] == pytest.approx(put["vega"], rel=1e-9)

def test_zero_dte_expiry_limits():
    """Verify stability under extreme 0DTE final minutes (T -> 0)."""
    spot = 24500.0
    
    # 0DTE ITM Call: Spot 24500, Strike 24400, T = 15 mins (0.01 days)
    itm_call = black_scholes_greeks(spot, 24400.0, t_days=0.01, is_call=True)
    assert itm_call["delta"] >= 0.95
    assert itm_call["price"] >= 99.0
    
    # 0DTE OTM Call: Spot 24500, Strike 24600, T = 15 mins
    otm_call = black_scholes_greeks(spot, 24600.0, t_days=0.01, is_call=True)
    assert otm_call["delta"] <= 0.05
    assert otm_call["price"] <= 1.0

def test_deep_moneyness_asymptotics():
    """Verify asymptotic limits for Deep ITM / Deep OTM strikes."""
    spot = 24500.0
    
    # Deep OTM Call (Strike 28000)
    deep_otm = black_scholes_greeks(spot, 28000.0, t_days=4.0, is_call=True)
    assert deep_otm["delta"] == 0.0
    assert deep_otm["gamma"] == 0.0
    assert deep_otm["price"] == 0.20  # Minimum tick floor
    
    # Deep ITM Call (Strike 20000)
    deep_itm = black_scholes_greeks(spot, 20000.0, t_days=4.0, is_call=True)
    assert deep_itm["delta"] >= 0.999
    assert deep_itm["price"] > 4400.0

def test_extreme_volatility_states():
    """Verify stability under extreme low (IV=8%) vs extreme high (IV=45%) volatility."""
    spot = 24500.0
    strike = 24500.0
    
    low_iv = black_scholes_greeks(spot, strike, sigma=0.08, is_call=True)
    high_iv = black_scholes_greeks(spot, strike, sigma=0.45, is_call=True)
    
    assert low_iv["price"] < high_iv["price"]
    assert low_iv["gamma"] > high_iv["gamma"]
    assert high_iv["vega"] > low_iv["vega"]

def test_convex_gamma_protection_taylor_series():
    """Verify that Gamma convexity strictly reduces option loss compared to linear delta."""
    spot = 24500.0
    sig_long = Signal(
        signal_type=SignalType.LONG,
        entry_price=24500.0,
        sl_price=24440.0,
        target_1=24580.0,
        target_2=24650.0,
        target_3_moonshot=24720.0,
        pyramid_trigger=24580.0,
        reason="Convexity Verification",
        htf_aligned=True,
        fib_retracement=0.55,
        details={}
    )
    ticket = generate_option_trade_ticket(spot, sig_long)
    
    entry_prem = ticket["entry_premium"]
    sl_prem = ticket["sl_premium"]
    opt_risk = entry_prem - sl_prem
    
    delta = ticket["delta"]
    gamma = ticket["gamma"]
    spot_risk = 60.0
    linear_risk = spot_risk * delta
    
    # Modeled risk before theta must be strictly less than linear risk due to +0.5 * Gamma * (dS)^2
    assert (opt_risk - abs(ticket["theta_decay_daily"]) * 0.15) < linear_risk

def test_tca_backward_compatibility_alias():
    """Ensure calculate_tca_friction is callable without errors."""
    res = calculate_tca_friction(142.50, 188.00, 150, 6)
    assert res["total_friction"] > 0
    assert res["brokerage"] == 40.0
