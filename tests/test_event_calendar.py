import pytest
import pandas as pd
from datetime import datetime
from src.event_calendar import is_trading_holiday, get_event_risk_status, check_event_risk_gate
from src.decision_engine import DecisionEngine, DecisionContext
from src.strategy_rules import StrategyEngine, SignalType


def test_nse_holiday_identification():
    assert is_trading_holiday("2026-01-26") is True  # Republic Day
    assert is_trading_holiday("2026-04-03") is True  # Good Friday
    assert is_trading_holiday("2026-08-15") is True  # Independence Day
    assert is_trading_holiday("2026-08-18") is False # Open trading session
    assert is_trading_holiday("2026-08-19") is False # Regular trading day

    # Check gate veto on an actual holiday (e.g. Independence Day)
    passed_hol, reason_hol, audit_hol = check_event_risk_gate("2026-08-15 10:30")
    assert passed_hol is False
    assert "Trading Holiday" in reason_hol
    assert audit_hol["sizing_cap"] == 0.0


def test_event_risk_blackout_window():
    # RBI MPC Announcement window: 2026-08-07 between 09:45 and 11:00 IST
    status_blackout = get_event_risk_status("2026-08-07 10:15")
    assert status_blackout["is_blackout"] is True
    assert status_blackout["risk_level"] == "CRITICAL"
    assert status_blackout["sizing_cap"] == 0.0

    # Same day after announcement: 14:00 IST
    status_after = get_event_risk_status("2026-08-07 14:00")
    assert status_after["is_blackout"] is False
    assert status_after["risk_level"] == "ELEVATED"
    assert status_after["sizing_cap"] == 0.5

    # Normal non-event day
    status_normal = get_event_risk_status("2026-08-19 10:15")
    assert status_normal["is_blackout"] is False
    assert status_normal["risk_level"] == "NORMAL"
    assert status_normal["sizing_cap"] == 1.0


def test_decision_engine_event_blackout_veto():
    engine = DecisionEngine()
    ctx = DecisionContext(
        markov_regime="LOW_VOL_TRENDING",
        htf_aligned_long=True,
        htf_aligned_short=True,
        bar_timestamp="2026-08-07 10:15"  # Active RBI MPC blackout
    )
    passed, reason, audit = engine.check_universal_gates("LONG", 24500.0, ctx)
    assert not passed
    assert audit["veto_gate"] == "EVENT_RISK_BLACKOUT"
    assert "RBI MPC Policy Decision" in reason


def test_strategy_engine_event_blackout_veto():
    engine = StrategyEngine()
    
    # Create 35 bars on an RBI MPC morning
    dates = pd.date_range("2026-08-07 09:15", periods=35, freq="5min")
    df = pd.DataFrame({
        "open": [24500.0 + i * 2 for i in range(35)],
        "high": [24505.0 + i * 2 for i in range(35)],
        "low": [24495.0 + i * 2 for i in range(35)],
        "close": [24502.0 + i * 2 for i in range(35)],
        "volume": [50000] * 35
    }, index=dates)
    
    # Bar index 12 corresponds to 10:15 AM IST (inside blackout window)
    sig = engine.evaluate_bar(df, current_idx=12)
    assert sig.signal_type == SignalType.WAIT
    assert "Event Risk Gate Veto" in sig.reason
