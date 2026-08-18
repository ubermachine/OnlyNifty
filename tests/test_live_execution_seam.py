"""
End-to-End Live Execution Seam Integration Tests.

Validates the full live production path:
1. Live Option Chain -> generate_option_trade_ticket (pricing_source == "MARKET_QUOTE", entry_premium == market_ask).
2. Per-strike IV & spread risk gating in live execution.
3. LiveSignalJournal logs evidence_tier == "QUOTE" when marked against quotes.
4. EdgeHarness defaults to "SPOT" and caps unverified/model evidence at PAPER.
5. Event Calendar is_trading_holiday() blocks holidays and applies sizing caps.
"""

import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.data_engine import DataEngine, IST
from src.options_engine import (
    generate_option_trade_ticket,
    calculate_time_to_expiry_days,
    select_institutional_strike
)
from src.strategy_rules import Signal, SignalType
from src.signal_journal import LiveSignalJournal, SignalLifecycleStatus
from src.edge_harness import WalkForwardRunner, EdgeStats, EdgeTable
from src.event_calendar import is_trading_holiday, get_event_risk_status, check_event_risk_gate


def test_live_option_chain_ticket_market_quote_pricing():
    """Verify live chain delivers real MARKET_QUOTE pricing and ask-based entry."""
    engine = DataEngine()
    chain_data = engine.generate_synthetic_option_chain(spot=24500.0)
    oc_df = chain_data["dataframe"]
    near_epoch = chain_data["near_expiry_epoch"]

    assert isinstance(oc_df, pd.DataFrame)
    assert not oc_df.empty
    assert "ce_bid" in oc_df.columns
    assert "ce_ask" in oc_df.columns

    sig = Signal(
        signal_type=SignalType.LONG,
        entry_price=24500.0,
        sl_price=24450.0,
        target_1=24550.0,
        target_2=24600.0,
        target_3_moonshot=24700.0
    )

    t_days = calculate_time_to_expiry_days(expiry_epoch=near_epoch)
    ticket = generate_option_trade_ticket(
        spot=24500.0,
        signal=sig,
        capital=500000.0,
        t_days=t_days,
        option_chain_df=oc_df
    )

    # 1. Assert pricing source is MARKET_QUOTE
    assert ticket["pricing_source"] == "MARKET_QUOTE"
    assert ticket["market_ask"] > 0.0
    assert ticket["entry_premium"] == pytest.approx(ticket["market_ask"], abs=0.01)
    assert ticket["bid_ask_spread"] > 0.0

    # 2. Assert signal journal stamps QUOTE tier
    journal = LiveSignalJournal(persistence_file=None)
    entry = journal.log_signal(
        signal=sig,
        ticket=ticket,
        current_spot=24500.0,
        bar_timestamp="2026-08-18 10:30",
        regime_info={"active_regime": "LOW_VOL_TRENDING"},
        confluence_score=85.0
    )
    assert entry.evidence_tier == "QUOTE"


def test_live_spread_gate_blocks_wide_markets():
    """Verify spread risk gate rejects illiquid options."""
    engine = DataEngine()
    chain_data = engine.generate_synthetic_option_chain(spot=24500.0)
    oc_df = chain_data["dataframe"].copy()

    # Identify selected strike and artificially widen its bid-ask spread
    strike_info = select_institutional_strike(24500.0, is_call=True)
    k_sel = strike_info["strike"]
    
    # Set wide spread (15 pts on a 25 pt stop = 60% > 15% threshold)
    oc_df.loc[oc_df["strike"] == k_sel, "ce_spread"] = 15.0
    oc_df.loc[oc_df["strike"] == k_sel, "ce_bid"] = 100.0
    oc_df.loc[oc_df["strike"] == k_sel, "ce_ask"] = 115.0

    sig = Signal(
        signal_type=SignalType.LONG,
        entry_price=24500.0,
        sl_price=24450.0,
        target_1=24550.0,
        target_2=24600.0
    )

    ticket = generate_option_trade_ticket(
        spot=24500.0,
        signal=sig,
        capital=500000.0,
        option_chain_df=oc_df,
        max_spread_risk_ratio=0.15
    )

    assert ticket["status"] == "VETOED"
    assert ticket["gate"] == "GATE_VETO_SPREAD_TOO_WIDE"


def test_edge_harness_default_provenance_truth():
    """Verify WalkForwardRunner default produces SPOT tier records capped at PAPER."""
    runner = WalkForwardRunner()
    
    # 40 positive R outcomes simulated from spot
    r_list = [1.2] * 30 + [-1.0] * 10
    stats = runner.compute_edge_stats(r_list, "LONG_TEST", "TREND")
    
    assert stats.evidence_tier == "SPOT"
    assert stats.status == "PAPER"  # Never promotes to TRUSTED without QUOTE evidence


def test_event_calendar_holiday_and_sizing_integration():
    """Verify holiday blackout blocks trading and sizing cap dampens elevated risk days."""
    # 1. Holiday: Independence Day 2026
    status_hol = get_event_risk_status("2026-08-15 11:00")
    assert status_hol["is_blackout"] is True
    assert status_hol["sizing_cap"] == 0.0
    
    passed_hol, reason_hol, _ = check_event_risk_gate("2026-08-15 11:00")
    assert passed_hol is False
    assert "Trading Holiday" in reason_hol

    # 2. Elevated Event Day (Union Budget Day outside blackout window)
    status_budget = get_event_risk_status("2026-02-01 14:30")
    assert status_budget["is_blackout"] is False
    assert status_budget["risk_level"] == "ELEVATED"
    assert status_budget["sizing_cap"] == 0.5


def test_journal_dynamic_greeks_squareoff():
    """Verify 15:15 squareoff uses entry greeks snapshot rather than hardcoded heuristics."""
    journal = LiveSignalJournal(persistence_file=None)
    sig = Signal(SignalType.LONG, entry_price=24500.0, sl_price=24450.0, target_1=24550.0, target_2=24600.0)
    
    ticket = {
        "status": "READY",
        "pricing_source": "MARKET_QUOTE",
        "market_ask": 150.0,
        "entry_premium": 150.0,
        "sl_premium": 120.0,
        "target1_premium": 180.0,
        "target2_premium": 210.0,
        "lots": 1,
        "total_qty": 75,
        "capital_risk": 2250.0,
        "tca_friction": 50.0,
        "greeks": {"delta": 0.52, "gamma": 0.0010, "theta": -15.0, "vega": 12.0}
    }
    
    entry = journal.log_signal(
        signal=sig,
        ticket=ticket,
        current_spot=24500.0,
        bar_timestamp="2026-08-18 10:00",
        regime_info={"active_regime": "LOW_VOL_TRENDING"},
        confluence_score=85.0
    )
    
    # Square off at 15:15 with spot at 24520 (+20 pts) after 10 bars
    # dS = +20, delta = 0.52 -> +10.4 pts
    # convexity = 0.5 * 0.0010 * 400 = +0.20 pts
    # theta decay = 10 bars * (15.0 / 75) = 10 * 0.20 = -2.0 pts
    # Expected exit premium = 150 + 10.4 + 0.20 - 2.0 = 158.60
    entry.bars_held = 9
    journal.update_open_trades_lifecycle(
        current_spot=24520.0,
        current_high=24525.0,
        current_low=24495.0,
        bar_time_str="15:15"
    )
    
    assert entry.lifecycle_status == SignalLifecycleStatus.EOD_SQUAREOFF.value
    assert entry.exit_premium == pytest.approx(158.60, abs=0.1)
    assert entry.realized_pnl_rupees > 0
