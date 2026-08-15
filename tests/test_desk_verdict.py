"""Comprehensive unit tests for Desk Verdict Decision Engine (src/desk_verdict.py)."""

import pytest
import pandas as pd
import numpy as np
from src.strategy_rules import Signal, SignalType
from src.options_positioning import OptionsDeskState
from src.desk_verdict import build_desk_verdict, DeskVerdict
from src.risk_state import SessionRiskState

@pytest.fixture
def base_desk_state():
    return OptionsDeskState(
        trend_bias="BULLISH",
        trend_conviction_pct=75.0,
        d_vector=0.25,
        pcr_level=1.05,
        pcr_zscore=0.2,
        pcr_momentum_score=0.02,
        put_wall=24400.0,
        call_wall=24600.0,
        max_pain=24500.0,
        max_pain_drift_pts=0.0,
        expected_move_pts=85.0,
        actual_range_pts=60.0,
        move_ratio=0.71,
        gamma_regime="DEALER_LONG_GAMMA",
        is_positive_gamma=True,
        zero_gex_strike=24500.0,
        writing_bias="PUT_WRITING_HEAVY_SUPPORT",
        itm_otm_shift=0.35,
        agreement_count=3,
        data_quality="VERIFIED"
    )

@pytest.fixture
def dummy_vol_report():
    return {
        "composite_vol_regime": "BUY_VOL",
        "iv_rv_spread": {"spread_pct": -1.2, "iv": 0.13, "rv": 0.14},
        "iv_percentile": {"iv_percentile": 35.0},
        "intraday_quality": {"sizing_multiplier": 1.0}
    }

@pytest.fixture
def dummy_regime_state():
    return {"active_regime": "LOW_VOL_TRENDING"}

@pytest.fixture
def dummy_htf_data():
    return {"htf_aligned_long": True, "htf_aligned_short": False}

def test_build_desk_verdict_confirmed_long(base_desk_state, dummy_vol_report, dummy_regime_state, dummy_htf_data):
    signal = Signal(
        signal_type=SignalType.LONG,
        entry_price=24500.0,
        sl_price=24460.0,
        target_1=24550.0,
        target_2=24600.0,
        target_3_moonshot=24680.0,
        pyramid_trigger=24540.0,
        reason="Golden Pocket Retracement Confirmed",
        htf_aligned=True,
        details={"confluence_score": 82.0, "confluence_grade": "A Standard"}
    )
    ticket = {
        "status": "READY",
        "symbol": "NIFTY 24500 CE",
        "entry_premium": 140.0,
        "sl_premium": 110.0,
        "target1_premium": 180.0,
        "target2_premium": 225.0,
        "target3_moonshot_premium": 280.0,
        "lots": 6,
        "total_qty": 150,
        "delta": 0.55,
        "gamma": 0.0008,
        "theta_decay_daily": -14.0,
        "vanna": 0.04,
        "tca_friction": {"total_friction": 180.0}
    }
    
    verdict = build_desk_verdict(
        signal=signal,
        ticket=ticket,
        desk_state=base_desk_state,
        vol_report=dummy_vol_report,
        regime_state=dummy_regime_state,
        htf_data=dummy_htf_data,
        current_spot=24500.0
    )
    
    assert isinstance(verdict, DeskVerdict)
    assert verdict.action == "BUY_CE"
    assert verdict.trend_bias == "BULLISH"
    assert verdict.option_pick is not None
    assert verdict.option_pick["symbol"] == "NIFTY 24500 CE"
    assert verdict.data_quality == "VERIFIED"

def test_build_desk_verdict_wait_state(base_desk_state, dummy_vol_report, dummy_regime_state, dummy_htf_data):
    signal = Signal(
        signal_type=SignalType.WAIT,
        entry_price=24500.0,
        sl_price=0.0,
        target_1=0.0,
        target_2=0.0,
        target_3_moonshot=0.0,
        pyramid_trigger=0.0,
        reason="Market consolidating, awaiting confluence.",
        htf_aligned=True
    )
    ticket = {"status": "WAIT"}
    
    verdict = build_desk_verdict(
        signal=signal,
        ticket=ticket,
        desk_state=base_desk_state,
        vol_report=dummy_vol_report,
        regime_state=dummy_regime_state,
        htf_data=dummy_htf_data,
        current_spot=24500.0
    )
    
    assert verdict.action == "WAIT"
    assert verdict.option_pick is None
    assert "consolidating" in verdict.reason.lower() or "awaiting" in verdict.reason.lower()

def test_build_desk_verdict_risk_halt(base_desk_state, dummy_vol_report, dummy_regime_state, dummy_htf_data):
    signal = Signal(
        signal_type=SignalType.WAIT,
        entry_price=24500.0,
        sl_price=0.0,
        target_1=0.0,
        target_2=0.0,
        target_3_moonshot=0.0,
        pyramid_trigger=0.0,
        reason="Session Risk Circuit Breaker: Daily Loss Limit Hit",
        htf_aligned=False
    )
    ticket = {"status": "WAIT"}
    
    # Session state halted
    session_state = SessionRiskState(locked=True, lock_reason="Daily Loss Limit Hit")
    
    verdict = build_desk_verdict(
        signal=signal,
        ticket=ticket,
        desk_state=base_desk_state,
        vol_report=dummy_vol_report,
        regime_state=dummy_regime_state,
        htf_data=dummy_htf_data,
        session_state=session_state,
        current_spot=24500.0
    )
    
    assert verdict.action == "WAIT"
    assert "Daily Loss Limit" in verdict.reason
