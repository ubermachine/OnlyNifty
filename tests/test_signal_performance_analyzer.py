"""Unit tests for SignalPerformanceAnalyzer in src/signal_journal.py."""

import pytest
import pandas as pd
import numpy as np
from src.strategy_rules import SignalType
from src.signal_journal import SignalEntry, SignalLifecycleStatus, SignalPerformanceAnalyzer


def _make_dummy_entry(
    signal_id: str = "SIG-1",
    signal_type: str = "EMA_BREAKOUT_LONG",
    direction: str = "LONG",
    r_multiple: float = 2.0,
    pnl: float = 5000.0,
    confluence: float = 85.0,
    regime: str = "Trend",
    timestamp_ist: str = "2026-08-15 09:35:00 IST",
    status: str = SignalLifecycleStatus.T2_REACHED.value,
    utc_ms: int = 1700000000000
) -> SignalEntry:
    return SignalEntry(
        signal_id=signal_id,
        timestamp_ist=timestamp_ist,
        timestamp_utc_ms=utc_ms,
        bar_timestamp="2026-08-15 09:35",
        spot_price=24500.0,
        signal_type=signal_type,
        direction=direction,
        trigger_reason="Test Trigger",
        selected_strike=24500,
        option_type="CE" if direction == "LONG" else "PE",
        symbol=f"NIFTY 24500 {'CE' if direction == 'LONG' else 'PE'}",
        entry_premium=140.0,
        sl_spot=24450.0,
        sl_premium=110.0,
        sl_points_spot=50.0,
        sl_risk_premium_pts=30.0,
        target_1_spot=24570.0,
        target_1_premium=180.0,
        target_2_spot=24650.0,
        target_2_premium=230.0,
        target_3_spot=24750.0,
        target_3_premium=290.0,
        r_multiple_t1=1.4,
        r_multiple_t2=3.0,
        confluence_score=confluence,
        confluence_grade="A+ Institutional",
        regime_summary="Trend",
        kalman_velocity=1.5,
        kalman_zscore=1.2,
        markov_regime=regime,
        htf_alignment="Bullish",
        is_0dte=False,
        lots_suggested=6,
        total_qty=150,
        capital_risk_rupees=4500.0,
        tca_friction_est=180.0,
        lifecycle_status=status,
        realized_r_multiple=r_multiple,
        realized_pnl_rupees=pnl
    )


def test_empty_analyzer():
    analyzer = SignalPerformanceAnalyzer([])
    assert analyzer.closed_entries == []
    
    df_sig = analyzer.win_rate_by_signal_type()
    assert isinstance(df_sig, pd.DataFrame)
    assert df_sig.empty
    assert "win_rate_pct" in df_sig.columns

    df_time = analyzer.win_rate_by_time_bucket(30)
    assert isinstance(df_time, pd.DataFrame)
    assert df_time.empty

    df_regime = analyzer.win_rate_by_regime()
    assert isinstance(df_regime, pd.DataFrame)
    assert df_regime.empty

    conf = analyzer.confluence_vs_outcome_correlation()
    assert conf["pearson_r"] == 0.0
    assert conf["p_value"] == 1.0
    assert len(conf["buckets"]) == 5

    streak = analyzer.streak_and_tilt_analysis()
    assert streak["tilt_detected"] is False
    assert streak["current_streak_count"] == 0

    rep = analyzer.generate_performance_report()
    assert rep["summary"]["total_closed_trades"] == 0
    assert rep["summary"]["win_rate_pct"] == 0.0


def test_win_rate_by_signal_type():
    entries = [
        _make_dummy_entry("S1", "EMA_BREAKOUT_LONG", "LONG", 2.0, 5000.0, 85.0),
        _make_dummy_entry("S2", "EMA_BREAKOUT_LONG", "LONG", -1.0, -2500.0, 75.0),
        _make_dummy_entry("S3", "PULLBACK_SHORT", "SHORT", 1.5, 3500.0, 80.0),
        _make_dummy_entry("S4", "WAIT", "WAIT", 0.0, 0.0, 0.0, status="AWAITING_SETUP"), # should be filtered
        _make_dummy_entry("S5", "ACTIVE_SIG", "LONG", 0.0, 0.0, 80.0, status=SignalLifecycleStatus.ACTIVE.value), # active filtered
    ]
    analyzer = SignalPerformanceAnalyzer(entries)
    assert len(analyzer.closed_entries) == 3

    df = analyzer.win_rate_by_signal_type()
    assert len(df) == 2
    ema_row = df[df["signal_type"] == "EMA_BREAKOUT_LONG"].iloc[0]
    assert ema_row["total_trades"] == 2
    assert ema_row["winning_trades"] == 1
    assert ema_row["losing_trades"] == 1
    assert ema_row["win_rate_pct"] == 50.0
    assert ema_row["avg_r_multiple"] == 0.5
    assert ema_row["total_pnl_rupees"] == 2500.0


def test_win_rate_by_time_bucket():
    entries = [
        _make_dummy_entry("S1", "SIG1", "LONG", 2.0, 5000.0, 85.0, timestamp_ist="2026-08-15 09:35:00 IST"),
        _make_dummy_entry("S2", "SIG2", "LONG", -1.0, -2500.0, 75.0, timestamp_ist="2026-08-15 09:50:00 IST"),
        _make_dummy_entry("S3", "SIG3", "SHORT", 3.0, 7500.0, 90.0, timestamp_ist="2026-08-15 14:40:00 IST"),
    ]
    analyzer = SignalPerformanceAnalyzer(entries)
    df = analyzer.win_rate_by_time_bucket(bucket_minutes=30)
    assert len(df) == 2
    assert any("09:30 - 10:00" in b for b in df["time_bucket"])
    assert any("14:30 - 15:00" in b for b in df["time_bucket"])


def test_win_rate_by_regime():
    entries = [
        _make_dummy_entry("S1", "SIG1", "LONG", 2.0, 5000.0, 85.0, regime="TRENDING_EXPANSION"),
        _make_dummy_entry("S2", "SIG2", "LONG", 1.5, 3500.0, 80.0, regime="TRENDING_EXPANSION"),
        _make_dummy_entry("S3", "SIG3", "SHORT", -1.0, -2500.0, 60.0, regime="MEAN_REVERTING"),
    ]
    analyzer = SignalPerformanceAnalyzer(entries)
    df = analyzer.win_rate_by_regime()
    assert len(df) == 2
    trend_row = df[df["regime"] == "TRENDING_EXPANSION"].iloc[0]
    assert trend_row["win_rate_pct"] == 100.0
    assert trend_row["total_trades"] == 2


def test_confluence_vs_outcome_correlation():
    entries = [
        _make_dummy_entry("S1", "SIG1", "LONG", -1.0, -2500.0, 45.0), # 0-50
        _make_dummy_entry("S2", "SIG2", "LONG", 0.5, 1000.0, 55.0),   # 50-65
        _make_dummy_entry("S3", "SIG3", "LONG", 1.5, 3000.0, 70.0),   # 65-75
        _make_dummy_entry("S4", "SIG4", "LONG", 2.5, 5000.0, 80.0),   # 75-85
        _make_dummy_entry("S5", "SIG5", "LONG", 4.0, 8000.0, 92.0),   # 85-100
    ]
    analyzer = SignalPerformanceAnalyzer(entries)
    res = analyzer.confluence_vs_outcome_correlation()
    assert res["pearson_r"] > 0.80
    assert "buckets" in res
    assert len(res["buckets"]) == 5
    assert res["bucket_dict"]["85-100"]["count"] == 1
    assert res["bucket_dict"]["85-100"]["win_rate_pct"] == 100.0
    assert res["bucket_dict"]["0-50"]["win_rate_pct"] == 0.0


def test_streak_and_tilt_analysis():
    base_ms = 1700000000000
    # 3 consecutive losses in rapid succession (e.g. 5 mins apart)
    entries = [
        _make_dummy_entry("S1", "SIG1", "LONG", 2.0, 5000.0, 85.0, utc_ms=base_ms),
        _make_dummy_entry("S2", "SIG2", "LONG", 1.5, 3500.0, 80.0, utc_ms=base_ms + 1800000), # +30m
        _make_dummy_entry("S3", "SIG3", "LONG", -1.0, -2500.0, 60.0, utc_ms=base_ms + 3600000), # +30m
        _make_dummy_entry("S4", "SIG4", "LONG", -1.0, -2500.0, 60.0, utc_ms=base_ms + 3900000), # +5m
        _make_dummy_entry("S5", "SIG5", "LONG", -1.0, -2500.0, 60.0, utc_ms=base_ms + 4200000), # +5m
    ]
    analyzer = SignalPerformanceAnalyzer(entries)
    streak = analyzer.streak_and_tilt_analysis()
    assert streak["current_streak_type"] == "LOSS"
    assert streak["current_streak_count"] == 3
    assert streak["max_win_streak"] == 2
    assert streak["max_loss_streak"] == 3
    assert streak["consecutive_losses"] == 3
    assert streak["tilt_detected"] is True
    assert streak["tilt_warning_level"] in ["ELEVATED", "CRITICAL"]


def test_generate_performance_report():
    entries = [
        _make_dummy_entry("S1", "SIG1", "LONG", 2.0, 5000.0, 85.0),
        _make_dummy_entry("S2", "SIG2", "LONG", -1.0, -2500.0, 70.0),
    ]
    analyzer = SignalPerformanceAnalyzer(entries)
    report = analyzer.generate_performance_report()
    assert "summary" in report
    assert report["summary"]["total_closed_trades"] == 2
    assert report["summary"]["win_rate_pct"] == 50.0
    assert "by_signal_type" in report
    assert "by_time_bucket" in report
    assert "by_regime" in report
    assert "confluence_correlation" in report
    assert "streak_and_tilt" in report
