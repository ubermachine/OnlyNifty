"""Unit tests for OnlyNifty v3.3 Quantitative Performance & Risk Analytics Suite."""

import pytest
import numpy as np
from src.performance_analytics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_ulcer_index_and_martin_ratio,
    calculate_profit_factor,
    calculate_payoff_ratio,
    calculate_consecutive_streaks_distribution,
    compute_institutional_performance_suite,
    DynamicKellyRecoveryProtocol
)

def test_sharpe_and_sortino_normal():
    returns = [0.02, 0.015, -0.005, 0.03, -0.01, 0.025, 0.01, -0.004, 0.02, 0.018]
    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    assert sharpe > 0.0
    assert sortino >= sharpe  # Sortino is higher when upside volatility is large

def test_sharpe_zero_variance():
    assert calculate_sharpe_ratio([0.01, 0.01, 0.01]) == 0.0
    assert calculate_sharpe_ratio([]) == 0.0

def test_sortino_no_downside():
    returns = [0.02, 0.03, 0.01, 0.04]
    sortino = calculate_sortino_ratio(returns)
    assert sortino == 999.0

def test_calmar_and_ulcer_index():
    equity = [500000.0, 510000.0, 495000.0, 490000.0, 520000.0, 530000.0]
    calmar = calculate_calmar_ratio(equity, trading_days=10)
    ulcer = calculate_ulcer_index_and_martin_ratio(equity, cagr_pct=calmar["cagr_pct"])
    
    assert calmar["calmar_ratio"] > 0.0
    assert calmar["max_drawdown_pct"] > 0.0
    assert ulcer["ulcer_index"] > 0.0
    assert ulcer["martin_ratio"] > 0.0

def test_profit_factor_and_payoff():
    pnls = [5000.0, 3000.0, -2000.0, 4000.0, -1500.0]
    pf = calculate_profit_factor(pnls)
    payoff = calculate_payoff_ratio(pnls)
    
    assert pf["profit_factor"] == round(12000.0 / 3500.0, 3)
    assert pf["gross_profit"] == 12000.0
    assert pf["gross_loss"] == 3500.0
    assert payoff["avg_win"] == 4000.0
    assert payoff["avg_loss"] == 1750.0
    assert payoff["payoff_ratio"] == round(4000.0 / 1750.0, 3)

def test_consecutive_streaks_distribution():
    # W, W, L, W, L, L, L, W, W
    pnls = [100.0, 200.0, -50.0, 150.0, -30.0, -40.0, -60.0, 300.0, 100.0]
    streaks = calculate_consecutive_streaks_distribution(pnls)
    
    assert streaks["max_consecutive_wins"] == 2
    assert streaks["max_consecutive_losses"] == 3
    assert streaks["current_streak"] == 2
    assert streaks["win_streak_distribution"] == {1: 1, 2: 2}
    assert streaks["loss_streak_distribution"] == {1: 1, 3: 1}

def test_dynamic_kelly_recovery_protocol_progression():
    protocol = DynamicKellyRecoveryProtocol(initial_capital=500000.0, base_risk_pct=0.01)
    
    # 1. Start at ATH
    s0 = protocol.evaluate_capital_state(500000.0)
    assert s0.recovery_stage == "ALL_TIME_HIGH"
    assert s0.dampener == 1.0
    
    # 2. Suffer a 5% drawdown (500k -> 475k)
    s1 = protocol.evaluate_capital_state(475000.0, recent_trade_pnl=-25000.0)
    assert s1.recovery_stage == "DRAWDOWN_EXPANSION"
    assert s1.current_drawdown_pct == 5.0
    assert s1.dampener < 1.0
    
    # 3. Recover 50% of the trough with 1 winning trade
    s2 = protocol.evaluate_capital_state(487500.0, recent_trade_pnl=12500.0)
    assert s2.recovery_stage == "PROGRESSIVE_RECOVERY"
    assert s2.recovery_progress_ratio == 0.5
    assert s2.consecutive_wins_since_trough == 1
    
    # 4. Recover further with 2nd consecutive win (unlocks confirmation gate)
    s3 = protocol.evaluate_capital_state(495000.0, recent_trade_pnl=7500.0)
    assert s3.consecutive_wins_since_trough == 2
    assert s3.dampener > s2.dampener
    
    # 5. Breaching 10% MDD triggers circuit breaker
    s_halt = protocol.evaluate_capital_state(440000.0, recent_trade_pnl=-55000.0)
    assert s_halt.is_halted
    assert s_halt.dampener == 0.0
    assert s_halt.recovery_stage == "CIRCUIT_BREAKER_HALT"
