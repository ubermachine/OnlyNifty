import pytest
import pandas as pd
import numpy as np
from src.strategy_rules import StrategyEngine, SignalType
from src.config import SIGNAL_MIN_CONFLUENCE, VPIN_TOXICITY_THRESHOLD


def test_vpin_toxicity_gate_blocks_all_trades():
    engine = StrategyEngine()
    # Create valid dataframe
    dates = pd.date_range("2026-08-13 10:00", periods=50, freq="5min")
    prices = [24500.0 + i * 5 for i in range(50)]
    df = pd.DataFrame({
        "open": prices,
        "high": [p + 10 for p in prices],
        "low": [p - 10 for p in prices],
        "close": prices,
        "volume": [20000] * 50
    }, index=dates)

    # Test universal gate method directly
    passed, reason, audit = engine._apply_universal_gates(
        candidate_direction="LONG",
        close=24750.0,
        skew_info={"skew_zscore": 0.5, "is_crash_hedging": False},
        vpin_info={"vpin": 0.85},  # Highly toxic
        hfi_score=0.10,
        gex_info={"is_positive_gamma": True, "call_wall_strike": 25500.0, "put_wall_strike": 24000.0},
        htf_regime={"htf_aligned_long": True, "htf_aligned_short": False}
    )
    assert not passed
    assert "VPIN Flow Toxicity Veto" in reason
    assert audit["veto_gate"] == "VPIN_TOXICITY"


def test_skew_crash_gate_blocks_long_trades():
    engine = StrategyEngine()
    passed, reason, audit = engine._apply_universal_gates(
        candidate_direction="LONG",
        close=24750.0,
        skew_info={"skew_zscore": 2.2, "is_crash_hedging": True},  # Crash hedge spike
        vpin_info={"vpin": 0.25},
        hfi_score=0.10,
        gex_info={"is_positive_gamma": True, "call_wall_strike": 25500.0, "put_wall_strike": 24000.0},
        htf_regime={"htf_aligned_long": True, "htf_aligned_short": False}
    )
    assert not passed
    assert "25-Delta Put Skew Crash Gate" in reason
    assert audit["veto_gate"] == "SKEW_CRASH_HEDGING"


def test_hfi_divergence_blocks_trades():
    engine = StrategyEngine()
    # Bearish HFI blocks LONG
    passed_long, reason_long, audit_long = engine._apply_universal_gates(
        candidate_direction="LONG",
        close=24750.0,
        skew_info={"skew_zscore": 0.0, "is_crash_hedging": False},
        vpin_info={"vpin": 0.25},
        hfi_score=-0.45,  # Strong negative heavyweight divergence
        gex_info={"is_positive_gamma": False},
        htf_regime={"htf_aligned_long": True, "htf_aligned_short": False}
    )
    assert not passed_long
    assert "Heavyweight Flow Veto" in reason_long

    # Bullish HFI blocks SHORT
    passed_short, reason_short, audit_short = engine._apply_universal_gates(
        candidate_direction="SHORT",
        close=24750.0,
        skew_info={"skew_zscore": 0.0, "is_crash_hedging": False},
        vpin_info={"vpin": 0.25},
        hfi_score=0.45,  # Strong positive heavyweight divergence
        gex_info={"is_positive_gamma": False},
        htf_regime={"htf_aligned_long": False, "htf_aligned_short": True}
    )
    assert not passed_short
    assert "Heavyweight Flow Veto" in reason_short


def test_gex_wall_pin_blocks_approaching_trades():
    engine = StrategyEngine()
    # Spot near Call Wall (24800) in positive gamma
    passed_long, reason_long, _ = engine._apply_universal_gates(
        candidate_direction="LONG",
        close=24795.0,  # Within 15pts of 24800
        skew_info={"skew_zscore": 0.0, "is_crash_hedging": False},
        vpin_info={"vpin": 0.25},
        hfi_score=0.0,
        gex_info={"is_positive_gamma": True, "call_wall_strike": 24800.0, "put_wall_strike": 24200.0},
        htf_regime={"htf_aligned_long": True, "htf_aligned_short": False}
    )
    assert not passed_long
    assert "Dealer GEX Pin Veto" in reason_long


def test_htf_alignment_veto():
    engine = StrategyEngine()
    # Long without HTF bullish alignment
    passed, reason, _ = engine._apply_universal_gates(
        candidate_direction="LONG",
        close=24500.0,
        skew_info={"skew_zscore": 0.0, "is_crash_hedging": False},
        vpin_info={"vpin": 0.25},
        hfi_score=0.0,
        gex_info={"is_positive_gamma": False},
        htf_regime={"htf_aligned_long": False, "htf_aligned_short": False, "tf_15m": {"bias": "Bearish"}, "tf_1h": {"bias": "Bearish"}}
    )
    assert not passed
    assert "HTF Confluence Veto" in reason
