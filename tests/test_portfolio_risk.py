"""Unit tests for PortfolioRiskManager in src/portfolio_risk.py."""

import pytest
import numpy as np
import pandas as pd
from src.portfolio_risk import PortfolioRiskManager
from src.signal_journal import SignalEntry, SignalLifecycleStatus


def test_portfolio_risk_manager_empty():
    manager = PortfolioRiskManager()
    greeks = manager.compute_portfolio_greeks([], spot=24500.0)
    assert greeks["net_delta"] == 0.0
    assert greeks["net_notional_delta_rupees"] == 0.0
    assert greeks["active_positions_count"] == 0

    grid = manager.compute_scenario_pnl_grid([], spot=24500.0)
    assert grid["max_profit_rupees"] == 0.0
    assert grid["max_loss_rupees"] == 0.0
    assert grid["grid_dataframe"].empty

    var = manager.compute_var_stress_test([], spot=24500.0)
    assert var["var_95_rupees"] == 0.0
    assert var["var_99_rupees"] == 0.0
    assert var["max_stress_loss_rupees"] == 0.0


def test_portfolio_greeks_aggregation():
    manager = PortfolioRiskManager()
    
    # 2 Active long CE positions
    trades = [
        {
            "strike": 24500,
            "option_type": "CE",
            "direction": "LONG",
            "entry_premium": 140.0,
            "qty": 150, # 6 lots
            "is_active": True
        },
        {
            "strike": 24600,
            "option_type": "CE",
            "direction": "LONG",
            "entry_premium": 90.0,
            "qty": 75,  # 3 lots
            "is_active": True
        }
    ]
    
    greeks = manager.compute_portfolio_greeks(trades, spot=24500.0, iv=0.14, t_days=4.0)
    assert greeks["active_positions_count"] == 2
    assert greeks["total_contracts"] == 225
    assert greeks["net_delta"] > 0
    assert greeks["net_notional_delta_rupees"] > 0
    assert greeks["net_gamma"] > 0
    assert greeks["net_theta_daily_rupees"] < 0 # Long options suffer negative theta
    assert greeks["net_vega_rupees"] > 0        # Long options have positive vega
    assert greeks["directional_bias"] == "BULLISH_DELTA"


def test_portfolio_greeks_delta_hedged_straddle():
    manager = PortfolioRiskManager()
    
    # ATM Straddle: 1 Long CE + 1 Long PE
    trades = [
        {
            "strike": 24500,
            "option_type": "CE",
            "direction": "LONG",
            "entry_premium": 140.0,
            "qty": 100,
            "is_active": True
        },
        {
            "strike": 24500,
            "option_type": "PE",
            "direction": "LONG",
            "entry_premium": 140.0,
            "qty": 100,
            "is_active": True
        }
    ]
    
    greeks = manager.compute_portfolio_greeks(trades, spot=24500.0, iv=0.14, t_days=4.0)
    # Net delta should be close to 0
    assert abs(greeks["net_delta"]) < 15.0
    assert greeks["net_gamma"] > 0
    assert greeks["net_vega_rupees"] > 0


def test_scenario_pnl_grid():
    manager = PortfolioRiskManager()
    trades = [
        {
            "strike": 24500,
            "option_type": "CE",
            "direction": "LONG",
            "entry_premium": 140.0,
            "qty": 150,
            "is_active": True
        }
    ]

    grid_res = manager.compute_scenario_pnl_grid(
        trades,
        spot=24500.0,
        iv=0.14,
        t_days=4.0,
        spot_range=200,
        spot_step=25,
        iv_shocks=[-0.03, 0.0, 0.03],
        time_steps_days=[0.0, 1.0, 2.0, 4.0]
    )

    assert "scenario_grid" in grid_res
    assert len(grid_res["scenario_grid"]) > 0
    assert grid_res["max_profit_rupees"] > 0
    assert grid_res["max_loss_rupees"] < 0
    assert not grid_res["grid_dataframe"].empty
    assert len(grid_res["pnl_curve_now"]) == 17 # -200 to +200 in steps of 25 = 17 points
    assert len(grid_res["pnl_curve_expiry"]) == 17


def test_var_stress_test():
    manager = PortfolioRiskManager(default_capital=500000.0)
    trades = [
        {
            "strike": 24500,
            "option_type": "CE",
            "direction": "LONG",
            "entry_premium": 140.0,
            "qty": 150,
            "is_active": True
        }
    ]

    var_res = manager.compute_var_stress_test(trades, spot=24500.0, iv=0.14, t_days=4.0, portfolio_capital=500000.0)
    assert var_res["var_95_rupees"] > 0.0
    assert var_res["var_99_rupees"] >= var_res["var_95_rupees"]
    assert "stress_scenarios" in var_res
    assert len(var_res["stress_scenarios"]) >= 5
    assert var_res["max_stress_loss_rupees"] > 0.0
    assert var_res["margin_adequacy_assessment"] in ["ADEQUATE", "ELEVATED_RISK", "CRITICAL_MARGIN_CALL"]
