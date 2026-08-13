"""Unit tests for OnlyNifty v3.2 Quantitative & Option Chain Enhancements."""

import pytest
import pandas as pd
import numpy as np
from src.options_engine import (
    calculate_pcr_and_max_pain,
    generate_option_trade_ticket,
    convert_to_free_vertical_spread
)
from src.indicators import (
    compute_pre_open_gap_filter,
    detect_volume_profile_triggers,
    compute_volume_profile,
    compute_order_flow_imbalance,
    compute_vakc_envelopes
)
from src.strategy_rules import Signal, SignalType
from src.data_engine import DataEngine

def test_pcr_and_max_pain_calculation():
    engine = DataEngine(use_cache=False)
    oc_data = engine.generate_synthetic_option_chain(spot=24500.0)
    df = oc_data["dataframe"]
    
    res = calculate_pcr_and_max_pain(df)
    assert "max_pain_strike" in res
    assert 24000.0 <= res["max_pain_strike"] <= 25000.0
    assert "pcr_oi" in res
    assert res["pcr_oi"] > 0.0
    assert "pcr_sentiment" in res
    assert not res["pain_distribution"].empty
    assert res["strikes_analyzed"] > 0

def test_pre_open_gap_filter_large_gap_up():
    engine = DataEngine(use_cache=False)
    df = engine.generate_synthetic_nifty(bars=20, interval_mins=5, start_price=24650.0)
    
    # Gap Up from 24500 to 24650 (+0.61%)
    gap_info = compute_pre_open_gap_filter(
        df_5m=df,
        prev_close=24500.0,
        pre_open_data={"iep": 24650.0}
    )
    assert gap_info["regime"] == "LARGE_GAP_UP"
    assert gap_info["is_large_gap"]
    assert gap_info["gap_pct"] > 0.50
    assert gap_info["gap_golden_pocket"] is not None
    assert "gf_support_500" in gap_info["gap_golden_pocket"]

def test_pre_open_gap_filter_normal_range():
    engine = DataEngine(use_cache=False)
    df = engine.generate_synthetic_nifty(bars=20, interval_mins=5, start_price=24520.0)
    
    # Flat opening (24500 -> 24520, +0.08%)
    gap_info = compute_pre_open_gap_filter(
        df_5m=df,
        prev_close=24500.0,
        pre_open_data={"iep": 24520.0}
    )
    assert gap_info["regime"] == "NORMAL"
    assert not gap_info["is_large_gap"]
    assert gap_info["gap_golden_pocket"] is None

def test_detect_volume_profile_triggers_and_vakc_elasticity():
    engine = DataEngine(use_cache=False)
    df = engine.generate_synthetic_nifty(bars=50, interval_mins=5)
    
    upper, lower = compute_vakc_envelopes(df, iv=0.18)
    assert len(upper) == len(df)
    assert (upper > lower).all()
    
    vp = compute_volume_profile(df, n_bins=24)
    ofi = compute_order_flow_imbalance(df)
    triggers = detect_volume_profile_triggers(df, vp, ofi)
    assert "trigger" in triggers
    assert "side" in triggers
    assert "confidence" in triggers

def test_free_vertical_spread_ticket_fields():
    sig = Signal(
        signal_type=SignalType.LONG,
        entry_price=24500.0,
        sl_price=24450.0,
        target_1=24580.0,
        target_2=24660.0,
        reason="Test Signal",
        htf_aligned=True,
        fib_retracement=0.55
    )
    ticket = generate_option_trade_ticket(spot=24500.0, signal=sig, capital=500000.0)
    assert "free_spread_t1" in ticket
    fs = ticket["free_spread_t1"]
    assert fs["status"] == "T1_FREE_SPREAD_AVAILABLE"
    assert fs["strike_k2_short"] == ticket["strike"] + 100
    assert fs["credit_received"] > 0.0
    assert "max_profit_pts" in fs
    assert "breakeven_spot" in fs
