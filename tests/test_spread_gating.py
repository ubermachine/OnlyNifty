import pytest
import pandas as pd
from src.strategy_rules import Signal, SignalType
from src.options_engine import generate_option_trade_ticket, select_institutional_strike


def test_spread_gating_normal_spread_passes():
    spot = 24500.0
    sig = Signal(
        signal_type=SignalType.LONG,
        entry_price=24500.0,
        sl_price=24465.0,
        target_1=24545.0,
        target_2=24590.0
    )
    
    strike_info = select_institutional_strike(spot, is_call=True)
    k = strike_info["strike"]
    
    # Create option chain with tight spread (₹0.50 spread on ₹150 option with ₹25 stop -> 2% ratio)
    chain_df = pd.DataFrame([{
        "strike": k,
        "ce_ltp": 155.0,
        "ce_bid": 154.75,
        "ce_ask": 155.25,
        "ce_spread": 0.50,
        "ce_symbol": f"NIFTY{k}CE",
        "ce_iv": 13.5,
        "pe_ltp": 80.0,
        "pe_bid": 79.5,
        "pe_ask": 80.5,
        "pe_spread": 1.0,
        "pe_symbol": f"NIFTY{k}PE",
        "pe_iv": 14.0
    }])
    
    tkt = generate_option_trade_ticket(spot, sig, option_chain_df=chain_df)
    assert tkt["status"] == "READY"
    assert tkt["pricing_source"] == "MARKET_QUOTE"
    assert tkt["entry_premium"] == 155.25
    assert tkt["market_spread"] == 0.50


def test_spread_gating_wide_spread_vetoes():
    spot = 24500.0
    sig = Signal(
        signal_type=SignalType.LONG,
        entry_price=24500.0,
        sl_price=24465.0,
        target_1=24545.0,
        target_2=24590.0
    )
    
    strike_info = select_institutional_strike(spot, is_call=True)
    k = strike_info["strike"]
    
    # Create option chain with wide spread (₹8.00 spread on ₹20 stop distance -> 40% ratio > 15%)
    chain_df = pd.DataFrame([{
        "strike": k,
        "ce_ltp": 155.0,
        "ce_bid": 151.0,
        "ce_ask": 159.0,
        "ce_spread": 8.0,
        "ce_symbol": f"NIFTY{k}CE",
        "ce_iv": 13.5,
        "pe_ltp": 80.0,
        "pe_bid": 75.0,
        "pe_ask": 85.0,
        "pe_spread": 10.0,
        "pe_symbol": f"NIFTY{k}PE",
        "pe_iv": 14.0
    }])
    
    tkt = generate_option_trade_ticket(spot, sig, option_chain_df=chain_df, max_spread_risk_ratio=0.15)
    assert tkt["status"] == "VETOED"
    assert tkt["gate"] == "GATE_VETO_SPREAD_TOO_WIDE"
    assert tkt["spread_risk_ratio"] > 0.15


def test_spread_gating_one_sided_quote_fails_closed():
    """Verify missing or zero bid/ask quotes are vetoed fail-closed."""
    spot = 24500.0
    sig = Signal(
        signal_type=SignalType.LONG,
        entry_price=24500.0,
        sl_price=24465.0,
        target_1=24545.0,
        target_2=24590.0
    )
    strike_info = select_institutional_strike(spot, is_call=True)
    k = strike_info["strike"]

    # 1. Zero Bid quote
    chain_df = pd.DataFrame([{
        "strike": k,
        "ce_ltp": 155.0,
        "ce_bid": 0.0,
        "ce_ask": 155.0,
        "ce_spread": 0.0,
        "ce_symbol": f"NIFTY{k}CE",
        "ce_iv": 13.5
    }])
    tkt = generate_option_trade_ticket(spot, sig, option_chain_df=chain_df)
    assert tkt["status"] == "VETOED"
    assert tkt["gate"] == "GATE_VETO_ILLIQUID_MARKET_QUOTE"


def test_cheap_option_relative_stop_geometry():
    """Verify cheap options (e.g. ₹4.90 expiry options) have relative stops and non-inverted geometry."""
    spot = 24154.90
    strike_info = select_institutional_strike(spot, is_call=True)
    k = strike_info["strike"]

    for stop_pts in [15.0, 35.0, 120.0]:
        sig = Signal(
            signal_type=SignalType.LONG,
            entry_price=spot,
            sl_price=spot - stop_pts,
            target_1=spot + 45.0,
            target_2=spot + 90.0
        )
        
        # Real quote: Ask 4.90, Bid 4.85, Spread 0.05
        chain_df = pd.DataFrame([{
            "strike": k,
            "ce_ltp": 4.90,
            "ce_bid": 4.85,
            "ce_ask": 4.90,
            "ce_spread": 0.05,
            "ce_symbol": f"NIFTY{k}CE",
            "ce_iv": 13.5
        }])

        tkt = generate_option_trade_ticket(spot, sig, option_chain_df=chain_df)
        assert tkt["status"] == "READY"
        assert tkt["pricing_source"] == "MARKET_QUOTE"
        assert tkt["entry_premium"] == 4.90
        # Invariant: sl_premium must be strictly below entry_premium
        assert 0.05 <= tkt["sl_premium"] < 4.90
        # Invariant: stop_dist must be positive
        stop_dist = tkt["entry_premium"] - tkt["sl_premium"]
        assert stop_dist > 0.0
        # Targets must be above entry
        assert tkt["target1_premium"] > tkt["entry_premium"]
        assert tkt["target2_premium"] > tkt["target1_premium"]


def test_cheap_option_wide_spread_veto():
    """Verify cheap options with wide spread (e.g. ₹0.50 spread on ₹4.90 option) get vetoed."""
    spot = 24154.90
    strike_info = select_institutional_strike(spot, is_call=True)
    k = strike_info["strike"]

    sig = Signal(
        signal_type=SignalType.LONG,
        entry_price=spot,
        sl_price=spot - 35.0,
        target_1=spot + 45.0,
        target_2=spot + 90.0
    )
    
    # Wide spread: Ask 4.90, Bid 4.10, Spread 0.80 (> 15% of stop distance)
    chain_df = pd.DataFrame([{
        "strike": k,
        "ce_ltp": 4.90,
        "ce_bid": 4.10,
        "ce_ask": 4.90,
        "ce_spread": 0.80,
        "ce_symbol": f"NIFTY{k}CE",
        "ce_iv": 13.5
    }])

    tkt = generate_option_trade_ticket(spot, sig, option_chain_df=chain_df, max_spread_risk_ratio=0.15)
    assert tkt["status"] == "VETOED"
    assert tkt["gate"] == "GATE_VETO_SPREAD_TOO_WIDE"


def test_sub_minimum_premium_veto():
    """Verify sub-₹2.00 deep OTM options fail closed."""
    spot = 24500.0
    strike_info = select_institutional_strike(spot, is_call=True)
    k = strike_info["strike"]

    sig = Signal(
        signal_type=SignalType.LONG,
        entry_price=spot,
        sl_price=spot - 35.0,
        target_1=spot + 45.0,
        target_2=spot + 90.0
    )
    
    chain_df = pd.DataFrame([{
        "strike": k,
        "ce_ltp": 1.20,
        "ce_bid": 1.15,
        "ce_ask": 1.20,
        "ce_spread": 0.05,
        "ce_symbol": f"NIFTY{k}CE",
        "ce_iv": 13.5
    }])

    tkt = generate_option_trade_ticket(spot, sig, option_chain_df=chain_df)
    assert tkt["status"] == "VETOED"
    assert tkt["gate"] == "GATE_VETO_PREMIUM_TOO_LOW"
