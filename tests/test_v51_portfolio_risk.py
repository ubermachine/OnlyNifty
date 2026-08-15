"""
Unit tests for PortfolioRiskManager & What-If Scenario Matrix (v5.1).
"""
import pytest
import numpy as np
import pandas as pd
from src.portfolio_risk import PortfolioRiskManager

@pytest.fixture
def risk_manager():
    return PortfolioRiskManager(lot_size=25)

@pytest.fixture
def sample_active_signals():
    return [
        {
            "selected_strike": 24500,
            "option_type": "CE",
            "lots_suggested": 2,
            "direction": "LONG",
            "entry_premium": 140.0,
            "sl_premium": 110.0
        },
        {
            "selected_strike": 24400,
            "option_type": "PE",
            "lots_suggested": 1,
            "direction": "LONG",
            "entry_premium": 95.0,
            "sl_premium": 75.0
        }
    ]

def test_portfolio_greeks_aggregation(risk_manager, sample_active_signals):
    greeks = risk_manager.compute_portfolio_greeks(
        active_signals=sample_active_signals,
        spot=24500.0,
        iv=0.13,
        t_days=3.5
    )
    assert "net_delta" in greeks
    assert "net_gamma" in greeks
    assert "net_theta_daily_rupees" in greeks
    assert "net_vega_rupees" in greeks
    assert greeks["active_position_count"] == 2
    assert "directional_bias" in greeks

def test_scenario_pnl_grid(risk_manager, sample_active_signals):
    scenario = risk_manager.compute_scenario_pnl_grid(
        active_signals=sample_active_signals,
        spot=24500.0,
        iv=0.13,
        t_days=3.5,
        spot_range=200,
        spot_step=25
    )
    assert "scenario_dataframe" in scenario
    assert len(scenario["scenario_dataframe"]) > 0
    df_scen = scenario["scenario_dataframe"]
    assert "spot_shift" in df_scen.columns
    assert "pnl_t0" in df_scen.columns
    assert "pnl_expiry" in df_scen.columns
    assert "breakeven_levels" in scenario

def test_var_stress_test(risk_manager, sample_active_signals):
    stress = risk_manager.compute_var_stress_test(
        active_signals=sample_active_signals,
        spot=24500.0,
        iv=0.13,
        t_days=3.5,
        portfolio_capital=500000.0
    )
    assert "parametric_var_95_rupees" in stress
    assert "stress_scenarios" in stress
    assert "flash_crash_pnl_rupees" in stress["stress_scenarios"]
    assert "black_swan_pnl_rupees" in stress["stress_scenarios"]
    assert stress["margin_adequacy_status"] in ["ADEQUATE", "ELEVATED_RISK", "CRITICAL_MARGIN_CALL"]
