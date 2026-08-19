"""Tests for OnlyNifty v4.0 LiveSignalJournal Engine."""

import os
import tempfile
import pytest
import pandas as pd
from src.strategy_rules import Signal, SignalType
from src.signal_journal import LiveSignalJournal


@pytest.fixture
def temp_journal():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name
    journal = LiveSignalJournal(persistence_file=tmp_path, allow_cloud_restore=False)
    yield journal
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


def test_log_signal_and_deduplication(temp_journal):
    sig_wait = Signal(SignalType.WAIT, 24350.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Wait", True, 0.0, {})
    ticket_empty = {"status": "WAIT"}
    
    # First WAIT is logged
    res1 = temp_journal.log_signal(sig_wait, ticket_empty, 24350.0, bar_timestamp="2026-08-14 09:15:00")
    assert res1 is not None
    assert res1["signal_type"] == "WAIT"
    assert res1["is_actionable"] is False
    
    # Second identical WAIT on same bar is filtered out (deduplication)
    res2 = temp_journal.log_signal(sig_wait, ticket_empty, 24350.0, bar_timestamp="2026-08-14 09:15:00")
    assert res2 is None
    assert len(temp_journal.entries) == 1
    
    # Actionable Long Signal
    sig_long = Signal(
        SignalType.LONG, 24350.0, 24300.0, 24420.0, 24500.0, 24600.0, 24370.0,
        "VAKC Pullback", True, 0.50, {}
    )
    ticket_long = {
        "status": "READY",
        "symbol": "NIFTY 24350 CE",
        "strike": 24350,
        "option_type": "CE",
        "entry_premium": 142.50,
        "sl_premium": 112.50,
        "target1_premium": 182.00,
        "target2_premium": 228.00,
        "target3_moonshot_premium": 285.00,
        "lots": 6,
        "total_qty": 150,
        "actual_risk_rupees": 4500.0,
        "delta": 0.55,
        "gamma": 0.00078,
        "theta_decay_daily": 14.20,
        "vanna": 0.042
    }
    
    res_long = temp_journal.log_signal(sig_long, ticket_long, 24350.0, bar_timestamp="2026-08-14 09:20:00")
    assert res_long is not None
    assert res_long["signal_type"] == "LONG"
    assert res_long["is_actionable"] is True
    assert res_long["symbol"] == "NIFTY 24350 CE"
    assert res_long["lots"] == 6
    assert len(temp_journal.entries) == 2


def test_trade_lifecycle_progression(temp_journal):
    sig_long = Signal(
        SignalType.LONG, 24350.0, 24300.0, 24420.0, 24500.0, 24600.0, 24370.0,
        "VAKC Pullback", True, 0.50, {}
    )
    ticket_long = {
        "status": "READY",
        "symbol": "NIFTY 24350 CE",
        "strike": 24350,
        "option_type": "CE",
        "entry_premium": 142.50,
        "sl_premium": 112.50,
        "target1_premium": 182.00,
        "target2_premium": 228.00,
        "target3_moonshot_premium": 285.00,
        "lots": 6,
        "total_qty": 150,
        "actual_risk_rupees": 4500.0,
        "delta": 0.55,
        "gamma": 0.00078,
        "theta_decay_daily": 14.20,
        "vanna": 0.042
    }
    
    temp_journal.log_signal(sig_long, ticket_long, 24350.0, bar_timestamp="2026-08-14 09:20:00")
    
    # Bar 1: Price goes up to Target 1 (24420)
    up1 = temp_journal.update_open_trades_lifecycle(current_spot=24425.0, current_high=24430.0, current_low=24340.0)
    assert up1 == 1
    assert temp_journal.entries[-1]["status"] == "T1_REACHED"
    
    # Bar 2: Price hits Target 2 (24500)
    up2 = temp_journal.update_open_trades_lifecycle(current_spot=24505.0, current_high=24510.0, current_low=24410.0)
    assert up2 == 1
    assert temp_journal.entries[-1]["status"] == "T2_REACHED"


def test_journal_summary_and_export(temp_journal):
    sig_short = Signal(
        SignalType.SHORT_ORDER_FLOW, 24400.0, 24450.0, 24330.0, 24250.0, 24150.0, 24380.0,
        "Order Flow", True, 0.50, {}
    )
    ticket_short = {
        "status": "READY",
        "symbol": "NIFTY 24400 PE",
        "strike": 24400,
        "option_type": "PE",
        "entry_premium": 138.00,
        "sl_premium": 108.00,
        "target1_premium": 178.00,
        "target2_premium": 222.00,
        "target3_moonshot_premium": 280.00,
        "lots": 4,
        "total_qty": 100,
        "actual_risk_rupees": 3000.0,
        "delta": -0.55,
        "gamma": 0.00078,
        "theta_decay_daily": 14.20,
        "vanna": 0.042
    }
    
    temp_journal.log_signal(sig_short, ticket_short, 24400.0, bar_timestamp="2026-08-14 10:00:00")
    
    summary = temp_journal.compute_daily_journal_summary()
    assert summary["total_signals"] >= 1
    assert summary["actionable_trades"] == 1
    assert summary["short_trades"] == 1
    
    df = temp_journal.get_journal_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    
    csv_bytes = temp_journal.export_csv_bytes()
    assert len(csv_bytes) > 50


def test_drain_lifecycle_events(temp_journal):
    sig_long = Signal(
        SignalType.GAMMA_BREAKOUT_LONG, 24350.0, 24300.0, 24420.0, 24500.0, 24600.0, 24380.0,
        "Gamma Breakout", True, 0.50, {"confluence_score": 85.0, "confluence_grade": "A+ Institutional"}
    )
    ticket_long = {
        "status": "READY",
        "symbol": "NIFTY 24350 CE",
        "strike": 24350,
        "option_type": "CE",
        "entry_premium": 145.00,
        "sl_premium": 112.50,
        "target1_premium": 182.00,
        "target2_premium": 228.00,
        "target3_moonshot_premium": 285.00,
        "lots": 2,
        "total_qty": 50,
        "actual_risk_rupees": 2500.0,
    }
    temp_journal.log_signal(sig_long, ticket_long, 24350.0, bar_timestamp="2026-08-14 09:20:00")
    
    # Check T1 trigger event
    temp_journal.update_open_trades_lifecycle(current_spot=24425.0, current_high=24430.0, current_low=24340.0)
    events = temp_journal.drain_lifecycle_events()
    assert len(events) == 1
    entry, evt_status, spot_val, prem_val = events[0]
    assert evt_status == "T1_REACHED"
    assert spot_val == 24430.0
    assert prem_val == 182.00

    # Draining again yields empty list
    assert len(temp_journal.drain_lifecycle_events()) == 0


def test_confluence_preservation_in_log_signal(temp_journal):
    sig = Signal(
        SignalType.LONG, 24500.0, 24450.0, 24580.0, 24650.0, 24750.0, 24520.0,
        "Confluence Test", True, 0.50, {"confluence_score": 78.5, "confluence_grade": "A Standard"}
    )
    ticket = {
        "status": "READY",
        "symbol": "NIFTY 24500 CE",
        "strike": 24500,
        "option_type": "CE",
        "entry_premium": 150.00,
        "sl_premium": 120.00,
        "target1_premium": 190.00,
        "target2_premium": 230.00,
        "target3_moonshot_premium": 290.00,
        "lots": 2,
        "total_qty": 50,
        "actual_risk_rupees": 2000.0,
    }
    dummy_df = pd.DataFrame({
        "open": [24480, 24490, 24500],
        "high": [24490, 24510, 24520],
        "low": [24470, 24480, 24490],
        "close": [24485, 24500, 24515],
        "volume": [1000, 1200, 1500]
    })
    entry = temp_journal.log_signal(sig, ticket, 24500.0, df_context=dummy_df)
    assert entry is not None
    assert entry.confluence_score == 78.5
    assert entry.confluence_grade == "A Standard"


def test_t2_lifecycle_progression_to_t3(temp_journal):
    sig_long = Signal(
        SignalType.LONG, 24500.0, 24450.0, 24550.0, 24600.0, 24700.0, 24525.0,
        "Breakout Long", True, 0.50, {}
    )
    ticket_long = {
        "status": "READY",
        "symbol": "NIFTY 24500 CE",
        "strike": 24500,
        "option_type": "CE",
        "entry_premium": 150.00,
        "sl_premium": 120.00,
        "target1_premium": 185.00,
        "target2_premium": 225.00,
        "target3_moonshot_premium": 295.00,
        "lots": 2,
        "total_qty": 50,
        "actual_risk_rupees": 2500.0,
    }
    entry = temp_journal.log_signal(sig_long, ticket_long, 24500.0, bar_timestamp="2026-08-14 09:20:00")
    assert entry.is_active() is True
    
    # 1. Reach T1
    temp_journal.update_open_trades_lifecycle(current_spot=24555.0, current_high=24560.0, current_low=24490.0)
    assert entry.lifecycle_status == "T1_REACHED"
    assert entry.is_active() is True
    
    # 2. Reach T2
    temp_journal.update_open_trades_lifecycle(current_spot=24605.0, current_high=24610.0, current_low=24540.0)
    assert entry.lifecycle_status == "T2_REACHED"
    assert entry.is_active() is True  # Must stay active to allow T3 or trailed SL!
    
    # 3. Reach T3 Moonshot
    temp_journal.update_open_trades_lifecycle(current_spot=24705.0, current_high=24710.0, current_low=24590.0)
    assert entry.lifecycle_status == "T3_MOONSHOT"
    assert entry.is_active() is False
    assert entry.realized_r_multiple > 0


def test_wait_confluence_isolation(temp_journal):
    sig_wait = Signal(
        SignalType.WAIT, 24500.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        "Awaiting Confluence Floor", False, 0.0, {"confluence_score": 38.0, "confluence_grade": "C Weak / Vetoed"}
    )
    ticket_wait = {"status": "WAIT"}
    entry_wait = temp_journal.log_signal(sig_wait, ticket_wait, 24500.0)
    assert entry_wait is not None
    assert entry_wait.confluence_score == 0.0
    assert entry_wait.confluence_grade == "Consolidation"
    
    # Now log an actionable trade
    sig_long = Signal(
        SignalType.LONG, 24500.0, 24450.0, 24550.0, 24600.0, 24700.0, 24525.0,
        "A+ Setup", True, 0.50, {"confluence_score": 88.0, "confluence_grade": "A+ Institutional"}
    )
    ticket_long = {
        "status": "READY",
        "symbol": "NIFTY 24500 CE",
        "strike": 24500,
        "option_type": "CE",
        "entry_premium": 150.00,
        "sl_premium": 120.00,
        "target1_premium": 185.00,
        "target2_premium": 225.00,
        "target3_moonshot_premium": 295.00,
        "lots": 2,
        "total_qty": 50,
        "actual_risk_rupees": 2500.0,
    }
    entry_long = temp_journal.log_signal(sig_long, ticket_long, 24500.0)
    assert entry_long.confluence_score == 88.0
    
    summary = temp_journal.compute_daily_journal_summary()
    assert summary["avg_confluence_score"] == 88.0  # Actionable trades only!


def test_daily_journal_summary_date_scoping(temp_journal):
    """Verifies that compute_daily_journal_summary accurately scopes to target dates."""
    sig_1 = Signal(SignalType.LONG, 24500.0, 24450.0, 24550.0, 24600.0, 24700.0, 24525.0, "Day 1", True, 0.50)
    ticket_1 = {"status": "READY", "symbol": "NIFTY 24500 CE", "strike": 24500, "option_type": "CE", "entry_premium": 150.0, "sl_premium": 120.0, "target1_premium": 185.0}
    
    # Day 1 entry
    e1 = temp_journal.log_signal(sig_1, ticket_1, 24500.0, bar_timestamp="2026-08-14 10:00:00")
    e1.lifecycle_status = "T3_MOONSHOT"
    e1.realized_r_multiple = 1.0
    e1.realized_pnl_rupees = 2625.0
    
    # Day 2 entry
    sig_2 = Signal(SignalType.SHORT, 24600.0, 24650.0, 24550.0, 24500.0, 24400.0, 24575.0, "Day 2", True, 0.50)
    ticket_2 = {"status": "READY", "symbol": "NIFTY 24600 PE", "strike": 24600, "option_type": "PE", "entry_premium": 160.0, "sl_premium": 130.0, "target1_premium": 195.0}
    e2 = temp_journal.log_signal(sig_2, ticket_2, 24600.0, bar_timestamp="2026-08-15 11:00:00")
    e2.lifecycle_status = "STOPPED_OUT"
    e2.realized_r_multiple = -1.0
    e2.realized_pnl_rupees = -2250.0
    
    # Scoped to Day 1
    sum_day1 = temp_journal.compute_daily_journal_summary(target_date="2026-08-14")
    assert sum_day1["total_signals"] == 1
    assert sum_day1["win_rate_pct"] == 100.0
    assert sum_day1["total_realized_pnl"] == 2625.0
    assert sum_day1["session_date"] == "2026-08-14"
    
    # Scoped to Day 2
    sum_day2 = temp_journal.compute_daily_journal_summary(target_date="2026-08-15")
    assert sum_day2["total_signals"] == 1
    assert sum_day2["win_rate_pct"] == 0.0
    assert sum_day2["total_realized_pnl"] == -2250.0
    assert sum_day2["session_date"] == "2026-08-15"
    
    # Scoped to All Dates
    sum_all = temp_journal.compute_daily_journal_summary(scope="all")
    assert sum_all["total_signals"] == 2
    assert sum_all["win_rate_pct"] == 50.0
    assert sum_all["total_realized_pnl"] == 375.0


def test_log_signal_wait_reason_normalization(temp_journal):
    """Verifies that score fluctuations on the same veto gate do not generate duplicate WAIT rows."""
    sig_wait1 = Signal(SignalType.WAIT, 24500.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Confluence Veto: Score 54.2 < 55.0", False, 0.0)
    sig_wait2 = Signal(SignalType.WAIT, 24500.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Confluence Veto: Score 54.8 < 55.0", False, 0.0)
    ticket_wait = {"status": "WAIT"}
    
    e1 = temp_journal.log_signal(sig_wait1, ticket_wait, 24500.0)
    assert e1 is not None
    
    # Second wait with slightly different score in reason should be normalized and suppressed
    e2 = temp_journal.log_signal(sig_wait2, ticket_wait, 24500.0)
    assert e2 is None
    assert len(temp_journal.entries) == 1
    
    # But a distinct veto gate (e.g. Data Sufficiency Gate) should be logged!
    sig_wait3 = Signal(SignalType.WAIT, 24500.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Data Sufficiency Gate: 3 core inputs unavailable", False, 0.0)
    e3 = temp_journal.log_signal(sig_wait3, ticket_wait, 24500.0)
    assert e3 is not None
    assert len(temp_journal.entries) == 2

