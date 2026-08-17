import pytest
import os
import tempfile
import pandas as pd
from src.strategy_rules import StrategyEngine, Signal, SignalType
from src.signal_journal import LiveSignalJournal, SignalEntry, SignalLifecycleStatus, SignalPerformanceAnalyzer


def test_signal_entry_fields():
    entry = SignalEntry(
        signal_id="SIG-TEST-001",
        timestamp_ist="2026-08-15 10:00:00 IST",
        timestamp_utc_ms=1723700000000,
        bar_timestamp="2026-08-15 10:00",
        spot_price=24500.0,
        signal_type="LONG",
        direction="LONG",
        trigger_reason="Test Long",
        selected_strike=24500,
        option_type="CE",
        symbol="NIFTY 24500 CE",
        entry_premium=150.0,
        sl_spot=24460.0,
        sl_premium=120.0,
        sl_points_spot=40.0,
        sl_risk_premium_pts=30.0,
        target_1_spot=24560.0,
        target_1_premium=195.0,
        target_2_spot=24620.0,
        target_2_premium=240.0,
        target_3_spot=24680.0,
        target_3_premium=290.0,
        r_multiple_t1=1.5,
        r_multiple_t2=3.0,
        confluence_score=85.0,
        confluence_grade="A+ Institutional",
        regime_summary="Trending",
        kalman_velocity=1.5,
        kalman_zscore=1.2,
        markov_regime="LOW_VOL_TRENDING",
        htf_alignment="Bullish",
        is_0dte=False,
        lots_suggested=4,
        total_qty=100,
        capital_risk_rupees=5000.0,
        tca_friction_est=180.0,
        is_seed=False,
        setup_id="IB_BREAKOUT_LONG"
    )
    assert entry.is_seed is False
    assert entry.setup_id == "IB_BREAKOUT_LONG"
    assert entry.lifecycle_status == SignalLifecycleStatus.TRIGGERED.value


def test_journal_partial_profit_at_t1_and_breakeven_trail():
    journal = LiveSignalJournal(persistence_file=None)
    sig = Signal(SignalType.LONG, 24500.0, 24460.0, 24560.0, 24620.0, 24680.0)
    ticket = {"status": "READY", "strike": 24500, "entry_premium": 150.0, "sl_premium": 120.0, "target1_premium": 195.0, "target2_premium": 240.0, "actual_risk_rupees": 5000.0}
    entry = journal.log_signal(sig, ticket, 24500.0, bar_timestamp="2026-08-15 10:00")
    assert entry is not None

    # Price hits T1 (24560)
    journal.update_open_trades_lifecycle(current_spot=24565.0, current_high=24570.0, current_low=24490.0)
    assert entry.lifecycle_status == SignalLifecycleStatus.T1_REACHED.value
    assert entry.sl_spot == 24500.0  # Trailed to breakeven
    assert entry.realized_pnl_rupees > 0

    # Price reverses and hits breakeven stop (24500)
    journal.update_open_trades_lifecycle(current_spot=24490.0, current_high=24510.0, current_low=24490.0)
    assert entry.lifecycle_status == SignalLifecycleStatus.STOPPED_OUT.value
    assert "Breakeven SL hit on remaining 50%" in entry.notes


def test_analyzer_filters_seeded_entries():
    journal = LiveSignalJournal(persistence_file=None)
    
    # 1 seed entry
    entry_seed = SignalEntry(
        signal_id="SIG-SEED-001",
        timestamp_ist="2026-08-15 09:30:00 IST",
        timestamp_utc_ms=1723700000000,
        bar_timestamp="2026-08-15 09:30",
        spot_price=24500.0,
        signal_type="LONG",
        direction="LONG",
        trigger_reason="Seed Long",
        selected_strike=24500,
        option_type="CE",
        symbol="NIFTY 24500 CE",
        entry_premium=150.0,
        sl_spot=24460.0,
        sl_premium=120.0,
        sl_points_spot=40.0,
        sl_risk_premium_pts=30.0,
        target_1_spot=24560.0,
        target_1_premium=195.0,
        target_2_spot=24620.0,
        target_2_premium=240.0,
        target_3_spot=24680.0,
        target_3_premium=290.0,
        r_multiple_t1=1.5,
        r_multiple_t2=3.0,
        confluence_score=85.0,
        confluence_grade="A+ Institutional",
        regime_summary="Trending",
        kalman_velocity=1.5,
        kalman_zscore=1.2,
        markov_regime="LOW_VOL_TRENDING",
        htf_alignment="Bullish",
        is_0dte=False,
        lots_suggested=4,
        total_qty=100,
        capital_risk_rupees=5000.0,
        tca_friction_est=180.0,
        lifecycle_status=SignalLifecycleStatus.STOPPED_OUT.value,
        realized_r_multiple=-1.0,
        realized_pnl_rupees=-5000.0,
        is_seed=True
    )
    journal.entries.append(entry_seed)

    # 1 live trade entry
    entry_live = SignalEntry(
        signal_id="SIG-LIVE-001",
        timestamp_ist="2026-08-15 10:30:00 IST",
        timestamp_utc_ms=1723700000000,
        bar_timestamp="2026-08-15 10:30",
        spot_price=24550.0,
        signal_type="LONG",
        direction="LONG",
        trigger_reason="Live Long",
        selected_strike=24550,
        option_type="CE",
        symbol="NIFTY 24550 CE",
        entry_premium=150.0,
        sl_spot=24510.0,
        sl_premium=120.0,
        sl_points_spot=40.0,
        sl_risk_premium_pts=30.0,
        target_1_spot=24610.0,
        target_1_premium=195.0,
        target_2_spot=24670.0,
        target_2_premium=240.0,
        target_3_spot=24730.0,
        target_3_premium=290.0,
        r_multiple_t1=1.5,
        r_multiple_t2=3.0,
        confluence_score=88.0,
        confluence_grade="A+ Institutional",
        regime_summary="Trending",
        kalman_velocity=1.8,
        kalman_zscore=1.4,
        markov_regime="LOW_VOL_TRENDING",
        htf_alignment="Bullish",
        is_0dte=False,
        lots_suggested=4,
        total_qty=100,
        capital_risk_rupees=5000.0,
        tca_friction_est=180.0,
        lifecycle_status=SignalLifecycleStatus.T3_MOONSHOT.value,
        realized_r_multiple=2.25,
        realized_pnl_rupees=11250.0,
        is_seed=False
    )
    journal.entries.append(entry_live)

    # Without seeds: only 1 live trade (win) -> 100% win rate
    analyzer_no_seeds = SignalPerformanceAnalyzer(journal.entries, include_seeds=False)
    assert len(analyzer_no_seeds.closed_entries) == 1
    df_res = analyzer_no_seeds.win_rate_by_signal_type()
    assert len(df_res) == 1
    assert df_res.iloc[0]["win_rate_pct"] == 100.0

    # With seeds: 2 trades (1 win, 1 loss) -> 50% win rate
    analyzer_with_seeds = SignalPerformanceAnalyzer(journal.entries, include_seeds=True)
    assert len(analyzer_with_seeds.closed_entries) == 2
