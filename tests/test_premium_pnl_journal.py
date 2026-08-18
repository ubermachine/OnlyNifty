import pytest
import os
import json
from src.config import LOT_SIZE
from src.signal_journal import LiveSignalJournal, SignalEntry, SignalLifecycleStatus
from src.strategy_rules import Signal, SignalType


def test_signal_entry_evidence_tier():
    journal = LiveSignalJournal(persistence_file=None)
    sig = Signal(SignalType.LONG, entry_price=24500.0, sl_price=24465.0, target_1=24545.0, target_2=24590.0)
    
    # Quote priced ticket
    ticket_quote = {
        "status": "READY",
        "symbol": "NIFTY2681824500CE",
        "strike": 24500,
        "option_type": "CE",
        "entry_premium": 150.0,
        "sl_premium": 125.0,
        "target1_premium": 185.0,
        "target2_premium": 215.0,
        "target3_moonshot_premium": 260.0,
        "lots": 1,
        "total_qty": LOT_SIZE,
        "delta": 0.55,
        "capital_risk_rupees": 25.0 * LOT_SIZE,
        "tca_friction_est": 45.0,
        "pricing_source": "MARKET_QUOTE"
    }
    
    entry = journal.log_signal(sig, ticket_quote, current_spot=24500.0)
    assert entry is not None
    assert entry.evidence_tier == "QUOTE"
    assert entry.entry_premium == 150.0


def test_premium_pnl_sl_hit():
    journal = LiveSignalJournal(persistence_file=None)
    sig = Signal(SignalType.LONG, entry_price=24500.0, sl_price=24465.0, target_1=24545.0, target_2=24590.0)
    ticket = {
        "status": "READY",
        "symbol": "NIFTY2681824500CE",
        "strike": 24500,
        "option_type": "CE",
        "entry_premium": 150.0,
        "sl_premium": 125.0,
        "target1_premium": 185.0,
        "target2_premium": 215.0,
        "target3_moonshot_premium": 260.0,
        "lots": 1,
        "total_qty": LOT_SIZE,
        "delta": 0.55,
        "capital_risk_rupees": 25.0 * LOT_SIZE,
        "tca_friction_est": 45.0,
        "pricing_source": "MARKET_QUOTE"
    }
    entry = journal.log_signal(sig, ticket, current_spot=24500.0)
    
    # Next bar drops below spot SL (24465)
    journal.update_open_trades_lifecycle(current_spot=24460.0, current_high=24490.0, current_low=24455.0, current_open=24490.0)
    assert entry.lifecycle_status == SignalLifecycleStatus.STOPPED_OUT.value
    # Premium loss: (125 - 150) * LOT_SIZE = -25 * 75 = -1875.0 Rs
    assert entry.realized_pnl_rupees == round(-25.0 * LOT_SIZE, 2)


def test_premium_pnl_t1_hit():
    journal = LiveSignalJournal(persistence_file=None)
    sig = Signal(SignalType.LONG, entry_price=24500.0, sl_price=24465.0, target_1=24545.0, target_2=24590.0)
    ticket = {
        "status": "READY",
        "symbol": "NIFTY2681824500CE",
        "strike": 24500,
        "option_type": "CE",
        "entry_premium": 150.0,
        "sl_premium": 125.0,
        "target1_premium": 185.0,
        "target2_premium": 215.0,
        "target3_moonshot_premium": 260.0,
        "lots": 1,
        "total_qty": LOT_SIZE,
        "delta": 0.55,
        "capital_risk_rupees": 25.0 * LOT_SIZE,
        "tca_friction_est": 40.0,
        "pricing_source": "MARKET_QUOTE"
    }
    entry = journal.log_signal(sig, ticket, current_spot=24500.0)
    
    # Next bar rallies to T1 (24545)
    journal.update_open_trades_lifecycle(current_spot=24550.0, current_high=24555.0, current_low=24500.0, current_open=24505.0)
    assert entry.lifecycle_status == SignalLifecycleStatus.T1_REACHED.value
    # 50% booked at 185: 0.5 * (185 - 150) * 75 = +1312.5 Rs
    assert entry.realized_pnl_rupees == round(0.5 * 35.0 * LOT_SIZE, 2)
