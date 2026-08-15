import pytest
import pandas as pd
import numpy as np
from src.decision_engine import DecisionEngine, SetupCandidate, DecisionContext
from src.strategy_rules import SignalType
from src.risk_state import SessionRiskState


def test_decision_engine_single_candidate_selection():
    engine = DecisionEngine()

    dates = pd.date_range("2026-08-13 10:00", periods=50, freq="5min")
    prices = [24500.0 + i * 2 for i in range(50)]
    df = pd.DataFrame({
        "open": prices,
        "high": [p + 5 for p in prices],
        "low": [p - 5 for p in prices],
        "close": prices,
        "volume": [10000] * 50
    }, index=dates)

    cand1 = SetupCandidate(
        setup_id="IB_BREAKOUT_LONG",
        signal_type=SignalType.LONG,
        direction="LONG",
        entry_price=24600.0,
        sl_price=24560.0,
        target_1=24650.0,
        target_2=24700.0,
        target_3_moonshot=24750.0,
        pyramid_trigger=24625.0,
        reason="IB Breakout Long candidate."
    )
    cand2 = SetupCandidate(
        setup_id="ABSORPTION_LONG",
        signal_type=SignalType.LONG_ORDER_FLOW,
        direction="LONG",
        entry_price=24600.0,
        sl_price=24570.0,
        target_1=24640.0,
        target_2=24680.0,
        target_3_moonshot=24720.0,
        pyramid_trigger=24620.0,
        reason="Absorption Long candidate."
    )

    ctx = DecisionContext(
        markov_regime="LOW_VOL_TRENDING",
        htf_aligned_long=True,
        htf_aligned_short=False,
        skew_z=0.2,
        vpin=0.20,
        hfi_score=0.10
    )

    sig = engine.decide([cand1, cand2], ctx, df)
    # Must return at most ONE actionable trade
    assert sig.signal_type in [SignalType.LONG, SignalType.LONG_ORDER_FLOW, SignalType.WAIT]


def test_decision_engine_universal_gate_block():
    engine = DecisionEngine()
    dates = pd.date_range("2026-08-13 10:00", periods=30, freq="5min")
    df = pd.DataFrame({
        "open": [24500.0] * 30,
        "high": [24510.0] * 30,
        "low": [24490.0] * 30,
        "close": [24500.0] * 30,
        "volume": [10000] * 30
    }, index=dates)

    cand = SetupCandidate(
        setup_id="IB_BREAKOUT_LONG",
        signal_type=SignalType.LONG,
        direction="LONG",
        entry_price=24500.0,
        sl_price=24470.0,
        target_1=24550.0,
        target_2=24600.0,
        target_3_moonshot=24650.0,
        pyramid_trigger=24520.0,
        reason="IB Breakout Long."
    )

    # Toxic VPIN should block
    ctx = DecisionContext(
        markov_regime="LOW_VOL_TRENDING",
        htf_aligned_long=True,
        htf_aligned_short=False,
        vpin=0.85
    )
    sig = engine.decide([cand], ctx, df)
    assert sig.signal_type == SignalType.WAIT
    assert "No candidate passed universal gates" in sig.reason
