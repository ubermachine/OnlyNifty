import pytest
import numpy as np
import pandas as pd

from src.options_engine import compute_dynamic_trailing_option_sl, calculate_adaptive_tca_friction
from src.risk_state import SessionRiskState
from src.desk_verdict import build_desk_verdict, compute_conviction
from src.options_positioning import OptionsDeskState
from src.signal_journal import SignalEntry, SignalLifecycleStatus
from src.edge_harness import WalkForwardRunner, EdgeStats
from src.indicators import compute_hurst_exponent, compute_order_flow_imbalance, compute_cpr
from src.volatility_engine import VolatilityIntelligence
from src.decision_engine import DecisionEngine, DecisionContext
from src.strategy_rules import Signal, SignalType
from src.fyers_auth import _today_ist_str


def _create_sample_signal_entry(sig_id: str, r_multiple: float, pnl_rupees: float, status: str = SignalLifecycleStatus.STOPPED_OUT.value) -> SignalEntry:
    return SignalEntry(
        signal_id=sig_id,
        timestamp_ist="2026-08-16 10:00:00 IST",
        timestamp_utc_ms=1723788000000,
        bar_timestamp="2026-08-16 10:00",
        spot_price=24500.0,
        signal_type="LONG",
        direction="LONG",
        trigger_reason="Test Signal",
        selected_strike=24500,
        option_type="CE",
        symbol="NIFTY 24500 CE",
        entry_premium=100.0,
        sl_spot=24460.0,
        sl_premium=70.0,
        sl_points_spot=40.0,
        sl_risk_premium_pts=30.0,
        target_1_spot=24550.0,
        target_1_premium=130.0,
        target_2_spot=24600.0,
        target_2_premium=160.0,
        target_3_spot=24650.0,
        target_3_premium=190.0,
        r_multiple_t1=1.25,
        r_multiple_t2=2.50,
        confluence_score=85.0,
        confluence_grade="A+ Institutional",
        regime_summary="TREND_EXPANSION",
        kalman_velocity=1.5,
        kalman_zscore=1.8,
        markov_regime="LOW_VOL_TRENDING",
        htf_alignment="BULLISH_ALIGNED",
        is_0dte=False,
        lots_suggested=1,
        total_qty=25,
        capital_risk_rupees=750.0,
        tca_friction_est=50.0,
        lifecycle_status=status,
        realized_r_multiple=r_multiple,
        realized_pnl_rupees=pnl_rupees
    )


def test_taylor_convexity_positive_on_downside():
    """Verify that gamma convexity is strictly positive and protects option floor on downside spot moves."""
    # For a Call: Spot moves down from 24500 to 24450 (spot_diff = -50)
    sl_prem = compute_dynamic_trailing_option_sl(
        entry_prem=150.0,
        spot_entry=24500.0,
        current_trailing_spot_sl=24450.0,
        delta=0.50,
        gamma=0.0008,
        theta_daily=0.0,
        elapsed_bars_5m=0,
        is_call=True
    )
    # spot_diff = -50, delta * spot_diff = -25.0
    # convexity = 0.5 * 0.0008 * (-50)^2 = +1.0
    # Expected: 150.0 - 25.0 + 1.0 = 126.0
    assert sl_prem == pytest.approx(126.0, abs=0.01)


def test_tca_friction_single_lot():
    """Verify that 1-lot trade applies 2-order fee and does not perform 0-lot partial splits."""
    tca_1lot = calculate_adaptive_tca_friction(
        entry_prem=100.0,
        t1_prem=130.0,
        final_exit_prem=160.0,
        total_qty=25,
        lots=1,
        part_booked=True
    )
    assert tca_1lot["brokerage"] == 40.0  # 2 orders * ₹20


def test_scratch_trade_loss_streak_protection():
    """Verify that breakeven scratch trades (R = 0.0) do not increment consecutive losses."""
    state = SessionRiskState(account_capital=500000.0)
    
    e1 = _create_sample_signal_entry("SIG-1", r_multiple=0.0, pnl_rupees=0.0)
    e2 = _create_sample_signal_entry("SIG-2", r_multiple=0.0, pnl_rupees=0.0)
    
    state.sync_from_journal([e1, e2], current_bar_idx=10, persist=False)
    assert state.consecutive_losses == 0
    assert not state.locked


def test_desk_verdict_clamped_target_repricing():
    """Verify that spot targets clamped to dealer walls re-price option target premiums on ticket."""
    sig = Signal(
        signal_type=SignalType.LONG,
        entry_price=24480.0,
        sl_price=24440.0,
        target_1=24550.0,
        target_2=24620.0,
        reason="IB Breakout Long",
        details={"confluence_score": 85.0, "setup_id": "LONG_IB_BREAKOUT"}
    )
    ticket = {
        "status": "READY", "symbol": "NIFTY 24500 CE", "strike": 24500, "option_type": "CE",
        "delta": 0.50, "gamma": 0.0008, "entry_premium": 100.0, "sl_premium": 70.0,
        "target1_premium": 135.0, "target2_premium": 170.0, "target3_premium": 210.0,
        "lots": 2, "total_qty": 50, "tca_friction": 80.0, "r_multiple_t1": 1.5, "r_multiple_t2": 3.0
    }
    desk_state = OptionsDeskState(
        trend_bias="BULLISH",
        trend_conviction_pct=80.0,
        d_vector=0.65,
        pcr_level=1.20,
        pcr_zscore=1.5,
        pcr_momentum_score=0.8,
        put_wall=24300.0,
        call_wall=24550.0,
        max_pain=24500.0,
        max_pain_drift_pts=50.0,
        expected_move_pts=85.0,
        actual_range_pts=60.0,
        move_ratio=0.70,
        gamma_regime="DEALER_LONG_GAMMA",
        is_positive_gamma=True,
        zero_gex_strike=24400.0,
        writing_bias="PUT_WRITING_HEAVY_SUPPORT",
        itm_otm_shift=0.40,
        agreement_count=4,
        data_quality="VERIFIED"
    )
    htf_data = {"htf_aligned_long": True, "bias": "BULLISH", "trend_strength": 0.8}
    vol_report = {"regime": "LOW_VOL_TRENDING", "is_squeeze": False, "is_crash_hedging": False}
    edge_stats = EdgeStats(
        setup_id="LONG_IB_BREAKOUT",
        regime="LOW_VOL_TRENDING",
        n=40,
        win_rate=60.0,
        mean_r=1.2,
        ev=0.45,
        ci_low=0.10,
        ci_high=0.80,
        status="TRUSTED"
    )
    
    verdict = build_desk_verdict(
        signal=sig,
        ticket=ticket,
        desk_state=desk_state,
        htf_data=htf_data,
        vol_report=vol_report,
        current_spot=24480.0,
        edge_stats=edge_stats
    )
    assert verdict.action == "BUY_CE"
    assert verdict.option_pick is not None
    # Spot target 24550 clamped by call wall 24550 -> recomputed premium
    assert verdict.option_pick["target1_premium"] > 0


def test_conviction_unmeasured_vs_paper_cap():
    """Verify UNMEASURED is capped at 50.0 and PAPER at 65.0."""
    votes = {"structure": 1, "flow": 1, "positioning": 1, "macro": 1}
    score_unm, tier_unm, _, _ = compute_conviction(action="BUY_CE", votes=votes, confluence_score=95.0, edge_status="UNMEASURED")
    assert score_unm <= 50.0
    
    score_paper, tier_paper, _, _ = compute_conviction(action="BUY_CE", votes=votes, confluence_score=95.0, edge_status="PAPER")
    assert score_paper <= 65.0
    assert score_paper > score_unm


def test_t1_reached_lifecycle_active():
    """Verify T1_REACHED is considered active for trailing to T2 and T3."""
    entry = _create_sample_signal_entry("SIG-T1", r_multiple=0.75, pnl_rupees=750.0, status=SignalLifecycleStatus.T1_REACHED.value)
    assert entry.is_active()


def test_edge_harness_t2_and_t3_simulation():
    """Verify WalkForwardRunner accurately simulates multi-target lifecycles."""
    dates = pd.date_range("2026-08-16 09:15", periods=10, freq="5min")
    # Spot moves from 24500 -> 24550 (T1) -> 24600 (T2)
    highs = [24520, 24560, 24610, 24600, 24590, 24580, 24570, 24560, 24550, 24540]
    lows = [24490, 24510, 24550, 24550, 24540, 24530, 24520, 24510, 24500, 24490]
    df_window = pd.DataFrame({"high": highs, "low": lows, "close": highs, "volume": 1000}, index=dates)
    
    sig = Signal(
        signal_type=SignalType.LONG,
        entry_price=24500.0,
        sl_price=24460.0,
        target_1=24550.0,
        target_2=24600.0,
        target_3_moonshot=24650.0
    )
    runner = WalkForwardRunner()
    r_outcome = runner.simulate_trade_outcome(
        sig=sig,
        future_window=df_window
    )
    assert r_outcome is not None
    # 0.5 * (50/40) + 0.5 * (100/40) = 0.5*1.25 + 0.5*2.50 = 1.88
    assert r_outcome == pytest.approx(1.88, abs=0.01)

    # Direct T3 Moonshot jump (single surge bar hitting 24660)
    df_t3 = pd.DataFrame({"high": [24660], "low": [24510], "close": [24660], "volume": [1000]}, index=dates[:1])
    r_t3 = runner.simulate_trade_outcome(sig=sig, future_window=df_t3)
    # (24650 - 24500) / 40 = 3.75
    assert r_t3 == pytest.approx(3.75, abs=0.01)


def test_hurst_flat_series_protection():
    """Verify compute_hurst_exponent does not crash on flat price series."""
    flat_series = pd.Series([24500.0] * 50)
    res = compute_hurst_exponent(flat_series)
    assert "hurst" in res
    assert 0.10 <= res["hurst"] <= 0.90


def test_order_flow_subpenny_range_clip():
    """Verify OFI handles infinitesimal sub-penny price fluctuations without float explosion."""
    df_micro = pd.DataFrame({
        "open": [24500.0, 24500.000000001],
        "high": [24500.000000002, 24500.000000002],
        "low": [24500.0, 24500.0],
        "close": [24500.000000001, 24500.000000002],
        "volume": [100.0, 100.0]
    })
    res = compute_order_flow_imbalance(df_micro)
    assert not np.isnan(res["ofi_zscore"])
    assert not np.isinf(res["ofi_zscore"])


def test_cpr_previous_day_anchor():
    """Verify compute_cpr uses previous day's completed bar (T-1) when multiple days exist."""
    idx = pd.to_datetime(["2026-08-14", "2026-08-15"])
    df_daily = pd.DataFrame({
        "open": [24400.0, 24500.0],
        "high": [24480.0, 24600.0],
        "low": [24380.0, 24480.0],
        "close": [24450.0, 24550.0]
    }, index=idx)
    
    res = compute_cpr(df_daily)
    # Expected pivot from 2026-08-14 (T-1): (24480 + 24380 + 24450) / 3.0 = 24436.67
    assert res["pivot"] == pytest.approx(24436.67, abs=0.05)


def test_skew_percentage_iv_normalization():
    """Verify compute_25delta_skew converts percentage IVs (e.g. 15.0) to decimal fractions."""
    oc_pct = pd.DataFrame({
        "strike": [24300, 24500, 24700],
        "pe_iv": [15.5, 15.0, 14.5],
        "ce_iv": [14.0, 14.5, 15.0]
    })
    skew_res = VolatilityIntelligence.compute_25delta_skew(oc_pct, spot=24500.0, iv_baseline=0.15)
    assert skew_res["skew_25d"] < 0.10
    assert abs(skew_res["skew_zscore"]) < 10.0  # Not exploding to 80+


def test_decision_engine_gate0_data_sufficiency():
    """Verify DecisionEngine rejects trades when required institutional inputs are absent."""
    engine = DecisionEngine()
    ctx_empty = DecisionContext(
        markov_regime="LOW_VOL_TRENDING",
        htf_aligned_long=True,
        htf_aligned_short=False,
        skew_z=0.0,
        gex_walls={},
        options_context=None
    )
    passed, msg, audit = engine.check_universal_gates("LONG", 24500.0, ctx_empty)
    assert not passed
    assert "Data Sufficiency Gate" in msg


def test_fyers_today_ist_format():
    """Verify _today_ist_str returns valid ISO date string in Asia/Kolkata timezone."""
    ist_str = _today_ist_str()
    assert len(ist_str) == 10
    assert ist_str.count("-") == 2
