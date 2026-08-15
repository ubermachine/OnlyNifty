"""
Unit tests for OnlyNifty v5.1 Institutional Flow Engine and Data Engine Upgrades.
"""

import pytest
import numpy as np
import pandas as pd

from src.data_engine import DataEngine
from src.institutional_flow import (
    InstitutionalFlowEngine,
    fetch_live_participant_oi,
    compute_fii_flow_trend,
    compute_rollover_analysis,
    generate_institutional_flow_report
)


def test_fetch_live_participant_oi_structure():
    engine = DataEngine(use_cache=False)
    data = fetch_live_participant_oi(engine)
    
    assert "fii" in data
    assert "dii" in data
    assert "pro" in data
    assert "client" in data
    assert "fii_ls_ratio" in data
    assert "fii_options_pcr" in data
    assert "dii_net_bias" in data
    assert "client_net_bias" in data
    assert "institutional_consensus_bias" in data
    assert "consensus_score" in data
    
    valid_biases = {
        "STRONG_BULLISH_INSTITUTIONAL",
        "MILD_BULLISH",
        "NEUTRAL",
        "MILD_BEARISH",
        "STRONG_BEARISH_INSTITUTIONAL"
    }
    assert data["institutional_consensus_bias"] in valid_biases
    assert data["fii_ls_ratio"] > 0
    assert data["fii_options_pcr"] > 0


def test_fii_flow_trend_accumulation():
    # 5 consecutive rising days
    series = [1.20, 1.35, 1.50, 1.65, 1.80]
    res = compute_fii_flow_trend(series)
    assert res["trend_classification"] == "ACCUMULATION"
    assert res["is_accumulation"] is True
    assert res["is_distribution"] is False
    assert res["is_capitulation"] is False
    assert res["momentum_score"] > 0


def test_fii_flow_trend_distribution():
    # 5 consecutive falling days
    series = [2.10, 1.90, 1.70, 1.50, 1.30]
    res = compute_fii_flow_trend(series)
    assert res["trend_classification"] == "DISTRIBUTION"
    assert res["is_distribution"] is True
    assert res["is_accumulation"] is False
    assert res["is_capitulation"] is False
    assert res["momentum_score"] < 0


def test_fii_flow_trend_capitulation_bullish():
    # Single day spike > 0.30
    series = [1.20, 1.22, 1.25, 1.24, 1.65] # latest change = +0.41
    res = compute_fii_flow_trend(series)
    assert res["trend_classification"] == "CAPITULATION"
    assert res["is_capitulation"] is True
    assert "BULLISH" in res["pattern"] or "SHORT_SQUEEZE" in res["pattern"]


def test_fii_flow_trend_capitulation_bearish():
    # Single day drop > 0.30
    series = [1.80, 1.82, 1.85, 1.83, 1.45] # latest change = -0.38
    res = compute_fii_flow_trend(series)
    assert res["trend_classification"] == "CAPITULATION"
    assert res["is_capitulation"] is True
    assert "BEARISH" in res["pattern"] or "LIQUIDATION" in res["pattern"]


def test_fii_flow_trend_neutral():
    # Range bound series
    series = [1.50, 1.52, 1.49, 1.51, 1.50]
    res = compute_fii_flow_trend(series)
    assert res["trend_classification"] == "NEUTRAL"
    assert res["is_accumulation"] is False
    assert res["is_distribution"] is False
    assert res["is_capitulation"] is False


def test_rollover_analysis_bullish_premium():
    res = compute_rollover_analysis(
        current_month_oi=4000000,
        next_month_oi=6000000,
        prev_day_current_oi=4800000,
        prev_day_next_oi=5200000,
        current_basis_pts=20.0,
        next_basis_pts=45.0
    )
    assert res["rollover_pct"] == 60.0
    assert res["roll_cost"] == 25.0
    assert res["roll_spread_pts"] == 25.0
    assert res["classification"] == "BULLISH_PREMIUM_ROLL"


def test_rollover_analysis_bearish_discount():
    res = compute_rollover_analysis(
        current_month_oi=4000000,
        next_month_oi=6000000,
        prev_day_current_oi=4800000,
        prev_day_next_oi=5200000,
        current_basis_pts=45.0,
        next_basis_pts=20.0
    )
    assert res["rollover_pct"] == 60.0
    assert res["roll_cost"] == -25.0
    assert res["roll_spread_pts"] == -25.0
    assert res["classification"] == "BEARISH_DISCOUNT_ROLL"


def test_rollover_analysis_neutral():
    res = compute_rollover_analysis(
        current_month_oi=5000000,
        next_month_oi=5000000,
        prev_day_current_oi=5000000,
        prev_day_next_oi=5000000,
        current_basis_pts=30.0,
        next_basis_pts=30.0
    )
    assert res["rollover_pct"] == 50.0
    assert res["roll_cost"] == 0.0
    assert res["classification"] == "NEUTRAL"


def test_generate_institutional_flow_report():
    engine = DataEngine(use_cache=False)
    report = generate_institutional_flow_report(engine)
    
    assert "timestamp" in report
    assert "composite_flow_score" in report
    assert "institutional_consensus_bias" in report
    assert "participant_oi" in report
    assert "fii_trend" in report
    assert "rollover_analysis" in report
    assert "multi_expiry" in report
    assert "recommendation" in report
    
    valid_biases = {
        "STRONG_BULLISH_INSTITUTIONAL",
        "MILD_BULLISH",
        "NEUTRAL",
        "MILD_BEARISH",
        "STRONG_BEARISH_INSTITUTIONAL"
    }
    assert report["institutional_consensus_bias"] in valid_biases
    assert -1.0 <= report["composite_flow_score"] <= 1.0


def test_data_engine_multi_expiry_chain():
    engine = DataEngine(use_cache=False)
    multi_chain = engine.fetch_multi_expiry_option_chain()
    
    assert "underlying_value" in multi_chain
    assert "expiry_dates" in multi_chain
    assert len(multi_chain["expiry_dates"]) >= 3
    assert "near_chain" in multi_chain
    assert "next_chain" in multi_chain
    assert "monthly_chain" in multi_chain
    assert "dataframe" in multi_chain
    
    assert not multi_chain["near_chain"].empty
    assert not multi_chain["next_chain"].empty
    assert not multi_chain["monthly_chain"].empty
    assert "strike" in multi_chain["near_chain"].columns
    assert "ce_oi" in multi_chain["near_chain"].columns
    assert "pe_oi" in multi_chain["near_chain"].columns


def test_institutional_flow_engine_class():
    engine = InstitutionalFlowEngine()
    
    p_data = engine.fetch_live_participant_oi()
    assert "fii_ls_ratio" in p_data
    assert len(engine.fii_ls_history) >= 1
    
    t_data = engine.compute_fii_flow_trend(p_data)
    assert "trend_classification" in t_data
    
    r_data = engine.compute_rollover_analysis(
        current_month_oi=4500000,
        next_month_oi=6500000,
        prev_day_current_oi=5000000,
        prev_day_next_oi=5800000,
        current_basis_pts=15.0,
        next_basis_pts=35.0
    )
    assert r_data["classification"] == "BULLISH_PREMIUM_ROLL"
    
    full_report = engine.generate_institutional_flow_report()
    assert "institutional_consensus_bias" in full_report
