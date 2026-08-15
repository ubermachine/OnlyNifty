"""
Unit tests for Institutional Flow Intelligence Engine (v5.1).
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

def test_fetch_live_participant_oi():
    de = DataEngine(use_cache=False)
    snap = fetch_live_participant_oi(de)
    assert "fii_ls_ratio" in snap
    assert "fii_options_pcr" in snap
    assert "institutional_consensus_bias" in snap
    assert snap["institutional_consensus_bias"] in [
        "STRONG_BULLISH_INSTITUTIONAL",
        "MILD_BULLISH",
        "NEUTRAL",
        "MILD_BEARISH",
        "STRONG_BEARISH_INSTITUTIONAL"
    ]
    assert snap["fii_ls_ratio"] >= 0.0

def test_fii_flow_trend_accumulation():
    # Pass historical series with 3+ rising days
    history = [1.50, 1.65, 1.80, 2.05, 2.25]
    trend_res = compute_fii_flow_trend(current_snapshot=2.25, historical_series=history)
    assert trend_res["is_accumulation"] is True
    assert trend_res["trend_classification"] == "ACCUMULATION"
    assert trend_res["net_5d_change"] > 0.0

def test_fii_flow_trend_distribution_and_capitulation():
    # Sharp drop from 1.60 -> 1.20 (>0.30 drop in 1 day)
    history = [1.80, 1.75, 1.70, 1.60, 1.20]
    trend_res = compute_fii_flow_trend(current_snapshot=1.20, historical_series=history)
    assert trend_res["is_capitulation"] is True
    assert trend_res["trend_classification"] == "CAPITULATION"

def test_rollover_analysis_premium():
    roll_res = compute_rollover_analysis(
        current_month_oi=1000000,
        next_month_oi=850000,
        prev_day_current_oi=1200000,
        prev_day_next_oi=680000,
        current_basis_pts=15.0,
        next_basis_pts=45.0
    )
    assert roll_res["rollover_pct"] > 0.0
    assert roll_res["roll_spread_pts"] == 30.0
    assert roll_res["classification"] == "BULLISH_PREMIUM_ROLL"

def test_master_institutional_flow_report():
    de = DataEngine(use_cache=False)
    engine = InstitutionalFlowEngine()
    rep = engine.generate_institutional_flow_report(de)
    assert "composite_flow_score" in rep
    assert -1.0 <= rep["composite_flow_score"] <= 1.0
    assert "current_snapshot" in rep
    assert "flow_trend" in rep
    assert "flow_summary" in rep
