"""
Unit tests for SignalPerformanceAnalyzer & Trade Attribution (v5.1).
"""
import pytest
import pandas as pd
import numpy as np
from src.signal_journal import SignalPerformanceAnalyzer, SignalEntry, SignalLifecycleStatus

def make_dummy_signal_entry(sig_id, sig_type, r_mult, pnl, conf, time_str, regime="TRENDING_EXPANSION"):
    return SignalEntry(
        signal_id=sig_id,
        timestamp_ist=time_str,
        timestamp_utc_ms=1723700000000,
        bar_timestamp=time_str,
        spot_price=24500.0,
        signal_type=sig_type,
        direction="LONG" if "LONG" in sig_type else "SHORT",
        trigger_reason="Test setup",
        selected_strike=24500,
        option_type="CE",
        symbol="NIFTY 24500 CE",
        entry_premium=140.0,
        sl_spot=24470.0,
        sl_premium=110.0,
        sl_points_spot=30.0,
        sl_risk_premium_pts=30.0,
        target_1_spot=24560.0,
        target_1_premium=190.0,
        target_2_spot=24620.0,
        target_2_premium=235.0,
        target_3_spot=24700.0,
        target_3_premium=300.0,
        r_multiple_t1=1.67,
        r_multiple_t2=3.17,
        confluence_score=conf,
        confluence_grade="A+ Institutional" if conf >= 80 else "A Standard",
        regime_summary="Bullish",
        kalman_velocity=1.2,
        kalman_zscore=1.5,
        markov_regime=regime,
        htf_alignment="1H: BULL",
        is_0dte=False,
        lots_suggested=2,
        total_qty=50,
        capital_risk_rupees=1500.0,
        tca_friction_est=120.0,
        lifecycle_status=SignalLifecycleStatus.T3_MOONSHOT.value if r_mult > 0 else SignalLifecycleStatus.STOPPED_OUT.value,
        realized_r_multiple=r_mult,
        realized_pnl_rupees=pnl
    )

def test_signal_performance_analyzer_aggregations():
    entries = [
        make_dummy_signal_entry("SIG-1", "TRENDING_EXPANSION_LONG", 1.8, 2700.0, 85.0, "2026-08-15 09:45:00 IST", "TRENDING_EXPANSION"),
        make_dummy_signal_entry("SIG-2", "TRENDING_EXPANSION_LONG", -1.0, -1500.0, 72.0, "2026-08-15 10:15:00 IST", "TRENDING_EXPANSION"),
        make_dummy_signal_entry("SIG-3", "MEAN_REVERSION_SHORT", 1.2, 1800.0, 88.0, "2026-08-15 13:30:00 IST", "MEAN_REVERTING"),
        make_dummy_signal_entry("SIG-4", "MEAN_REVERSION_SHORT", 1.5, 2250.0, 92.0, "2026-08-15 14:45:00 IST", "MEAN_REVERTING"),
    ]
    analyzer = SignalPerformanceAnalyzer(entries)
    
    df_type = analyzer.win_rate_by_signal_type()
    assert len(df_type) == 2
    assert "win_rate_pct" in df_type.columns
    
    df_time = analyzer.win_rate_by_time_bucket(bucket_minutes=30)
    assert len(df_time) > 0
    
    df_reg = analyzer.win_rate_by_regime()
    assert len(df_reg) == 2

def test_confluence_correlation_and_tilt():
    entries = [
        make_dummy_signal_entry("SIG-1", "LONG", 2.0, 3000.0, 90.0, "09:35"),
        make_dummy_signal_entry("SIG-2", "LONG", 1.5, 2250.0, 82.0, "10:05"),
        make_dummy_signal_entry("SIG-3", "SHORT", -1.0, -1500.0, 60.0, "11:00"),
        make_dummy_signal_entry("SIG-4", "SHORT", -1.0, -1500.0, 55.0, "11:10"),
        make_dummy_signal_entry("SIG-5", "SHORT", -1.0, -1500.0, 52.0, "11:18"),
    ]
    analyzer = SignalPerformanceAnalyzer(entries)
    report = analyzer.generate_performance_report()
    
    assert "confluence_correlation" in report
    assert report["confluence_correlation"]["pearson_r"] > 0.5  # Higher confluence -> higher R
    
    # Check streak and tilt
    streak_res = report["streak_and_tilt"]
    assert streak_res["consecutive_losses"] == 3
    assert streak_res["tilt_warning_level"] in ["ELEVATED", "WARNING"]
