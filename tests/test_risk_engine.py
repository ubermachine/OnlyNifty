"""Unit tests for OnlyNifty v3.2 Quantitative Risk Engine & Monte Carlo Simulator."""

import pytest
import numpy as np
from src.options_engine import (
    run_monte_carlo_simulation,
    evaluate_golden_vault_lock,
    calculate_position_size
)

def test_evaluate_golden_vault_lock_unlocked():
    # Below +1.5% trigger threshold (e.g. ₹5,000 profit on ₹5,00,000 capital)
    res = evaluate_golden_vault_lock(
        initial_capital=500000.0,
        current_intraday_pnl=5000.0,
        peak_intraday_pnl=5000.0,
        profit_trigger_pct=0.015,
        lock_pct=0.75
    )
    assert res["status"] == "UNLOCKED"
    assert not res["is_vault_triggered"]
    assert not res["is_session_halted"]
    assert res["locked_profit_floor"] == 0.0

def test_evaluate_golden_vault_lock_active_and_cushion():
    # Peak PnL ₹10,000 (>= +1.5% = ₹7,500). Current PnL = ₹9,000
    # 75% of ₹10,000 = ₹7,500 locked floor. Remaining cushion = ₹1,500
    res = evaluate_golden_vault_lock(
        initial_capital=500000.0,
        current_intraday_pnl=9000.0,
        peak_intraday_pnl=10000.0,
        profit_trigger_pct=0.015,
        lock_pct=0.75
    )
    assert res["status"] == "VAULT_ACTIVE"
    assert res["is_vault_triggered"]
    assert not res["is_session_halted"]
    assert res["locked_profit_floor"] == 7500.0
    assert res["risk_cushion"] == 1500.0

def test_evaluate_golden_vault_lock_session_halt():
    # Peak PnL ₹10,000. Current PnL drops to ₹7,500 (Locked floor hit)
    res = evaluate_golden_vault_lock(
        initial_capital=500000.0,
        current_intraday_pnl=7500.0,
        peak_intraday_pnl=10000.0,
        profit_trigger_pct=0.015,
        lock_pct=0.75
    )
    assert res["status"] == "LOCKED_GOLDEN_VAULT"
    assert res["is_session_halted"]
    assert res["risk_cushion"] == 0.0

def test_calculate_position_size_with_golden_vault_constraint():
    # Capital ₹5,00,000, 1% Risk = ₹5,000
    # Vault locked floor = ₹7,500, Current PnL = ₹8,500 -> Cushion = ₹1,000
    # Sizing must be constrained by ₹1,000 cushion instead of standard ₹5,000
    sizing = calculate_position_size(
        capital=500000.0,
        risk_pct=0.01,
        entry_prem=150.0,
        sl_prem=110.0,
        lot_size=25,
        current_intraday_pnl=8500.0,
        peak_intraday_pnl=10000.0,
        enforce_golden_vault=True
    )
    assert sizing["vault_constrained"]
    assert sizing["max_risk_rupees"] == 1000.0
    assert sizing["lots"] == 1  # 1 lot * 40 risk * 25 = 1000

def test_calculate_position_size_golden_vault_halt():
    # When locked floor is breached, lots must be 0
    sizing = calculate_position_size(
        capital=500000.0,
        risk_pct=0.01,
        entry_prem=150.0,
        sl_prem=110.0,
        lot_size=25,
        current_intraday_pnl=7000.0,
        peak_intraday_pnl=10000.0,
        enforce_golden_vault=True
    )
    assert sizing["lots"] == 0
    assert sizing["max_risk_rupees"] == 0.0

def test_monte_carlo_ruin_simulation():
    # 1,000 paths across 100 trades
    mc = run_monte_carlo_simulation(
        initial_capital=500000.0,
        base_risk_pct=0.01,
        win_rate=0.58,
        win_payoff_r=2.10,
        num_simulations=1000,
        num_trades=100,
        ruin_threshold_pct=0.50,
        random_seed=42
    )
    assert mc["num_simulations"] == 1000
    assert mc["num_trades"] == 100
    assert mc["prob_of_ruin_pct"] < 0.01
    assert mc["is_ruin_safe"]
    assert mc["var_95_pct"] > 0.0
    assert mc["cvar_95_pct"] >= mc["var_95_pct"]
    assert mc["var_99_pct"] >= mc["var_95_pct"]
    assert mc["mean_final_equity"] > 500000.0
    assert len(mc["percentile_50"]) == 101
