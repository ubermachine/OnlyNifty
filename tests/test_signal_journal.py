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
    journal = LiveSignalJournal(persistence_file=tmp_path)
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
