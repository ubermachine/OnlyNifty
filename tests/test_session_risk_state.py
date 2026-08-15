import pytest
import os
import tempfile
from src.risk_state import SessionRiskState


def test_session_risk_state_trade_limit():
    state = SessionRiskState(max_trades_per_day=3)
    assert state.can_take_new_trade()[0] is True
    
    state.record_trade_entry(10)
    state.record_trade_exit(realized_pnl=500.0, r_multiple=1.0, is_loss=False)
    assert state.trades_today == 1
    assert state.can_take_new_trade(current_bar_idx=25)[0] is True

    state.record_trade_entry(25)
    state.record_trade_exit(realized_pnl=400.0, r_multiple=0.8, is_loss=False)
    assert state.trades_today == 2

    state.record_trade_entry(40)
    state.record_trade_exit(realized_pnl=600.0, r_multiple=1.2, is_loss=False)
    assert state.trades_today == 3

    # 4th trade must be refused
    can_trade, reason = state.can_take_new_trade(current_bar_idx=55)
    assert can_trade is False
    assert "session budget exceeded" in reason.lower()


def test_session_risk_state_two_strike_loss_circuit_breaker():
    state = SessionRiskState(max_consecutive_losses=2, account_capital=500000.0, daily_loss_limit_pct=0.05)
    state.record_trade_entry(5)
    state.record_trade_exit(realized_pnl=-2000.0, r_multiple=-1.0, is_loss=True)
    assert state.consecutive_losses == 1
    assert state.locked is False

    state.record_trade_entry(20)
    state.record_trade_exit(realized_pnl=-2000.0, r_multiple=-1.0, is_loss=True)
    assert state.consecutive_losses == 2
    assert state.locked is True
    assert "2-strike loss" in state.lock_reason.lower()

    can_trade, reason = state.can_take_new_trade(current_bar_idx=35)
    assert can_trade is False
    assert "session locked" in reason.lower()


def test_session_risk_state_daily_loss_limit():
    state = SessionRiskState(daily_loss_limit_pct=0.015, account_capital=500000.0)
    # 1.5% of 500,000 is 7,500
    state.record_trade_entry(5)
    state.record_trade_exit(realized_pnl=-8000.0, r_multiple=-1.6, is_loss=True)
    assert state.locked is True
    assert "daily loss limit breached" in state.lock_reason.lower()


def test_session_risk_state_cooldown_bars():
    state = SessionRiskState(cooldown_bars=12)
    state.record_trade_entry(10)
    state.record_trade_exit(realized_pnl=1000.0, r_multiple=1.0, is_loss=False)

    # Within cooldown (e.g. 5 bars later)
    can_trade, reason = state.can_take_new_trade(current_bar_idx=15)
    assert can_trade is False
    assert "cooldown active" in reason.lower()

    # After cooldown (13 bars later)
    can_trade, _ = state.can_take_new_trade(current_bar_idx=23)
    assert can_trade is True


def test_session_risk_state_disk_persistence():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name

    try:
        state = SessionRiskState(persistence_file=tmp_path)
        state.record_trade_entry(10)
        state.record_trade_exit(realized_pnl=1250.0, r_multiple=1.5, is_loss=False)
        state.save_to_disk(tmp_path)

        # Reload
        loaded = SessionRiskState.load_from_disk(tmp_path)
        assert loaded.trades_today == 1
        assert loaded.realized_pnl_today == 1250.0
        assert loaded.last_entry_bar_idx == 10
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
