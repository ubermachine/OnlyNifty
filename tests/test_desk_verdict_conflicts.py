"""Tests for Desk Verdict conflict detection and desk-only fade/breakout branches.

The base test_desk_verdict.py covers the happy path (confirmed long, plain WAIT,
session halt). These cover the conflict-detection paths that downgrade an otherwise
valid chart signal to WAIT, and the desk-only branches that act when the chart is WAIT.
"""

import pytest

from src.strategy_rules import Signal, SignalType
from src.options_positioning import OptionsDeskState
from src.desk_verdict import build_desk_verdict


def make_desk_state(**overrides):
    base = dict(
        trend_bias="BULLISH",
        trend_conviction_pct=75.0,
        d_vector=0.25,
        pcr_level=1.05,
        pcr_zscore=0.2,
        pcr_momentum_score=0.02,
        put_wall=24400.0,
        call_wall=24600.0,
        max_pain=24500.0,
        max_pain_drift_pts=0.0,
        expected_move_pts=85.0,
        actual_range_pts=60.0,
        move_ratio=0.71,
        gamma_regime="DEALER_LONG_GAMMA",
        is_positive_gamma=True,
        zero_gex_strike=24500.0,
        writing_bias="PUT_WRITING_HEAVY_SUPPORT",
        itm_otm_shift=0.35,
        agreement_count=3,
        data_quality="VERIFIED",
    )
    base.update(overrides)
    return OptionsDeskState(**base)


def make_long_signal():
    return Signal(
        signal_type=SignalType.LONG,
        entry_price=24500.0,
        sl_price=24460.0,
        target_1=24550.0,
        target_2=24600.0,
        target_3_moonshot=24680.0,
        pyramid_trigger=24540.0,
        reason="Golden Pocket Retracement Confirmed",
        htf_aligned=True,
        details={"confluence_score": 82.0, "confluence_grade": "A Standard"},
    )


def make_wait_signal():
    return Signal(
        signal_type=SignalType.WAIT,
        entry_price=24500.0,
        sl_price=0.0,
        target_1=0.0,
        target_2=0.0,
        reason="Market in consolidation.",
        htf_aligned=False,
    )


READY_TICKET = {
    "status": "READY",
    "symbol": "NIFTY 24500 CE",
    "entry_premium": 140.0,
    "sl_premium": 110.0,
    "target1_premium": 180.0,
    "target2_premium": 225.0,
    "lots": 6,
    "total_qty": 150,
    "tca_friction": {"total_friction": 180.0},
}


class TestConflictDetection:
    def test_bearish_positioning_vetoes_chart_long(self):
        verdict = build_desk_verdict(
            signal=make_long_signal(),
            ticket=READY_TICKET,
            desk_state=make_desk_state(d_vector=-0.75, trend_bias="BEARISH"),
            current_spot=24500.0,
        )
        assert verdict.action == "WAIT"
        assert verdict.option_pick is None
        assert any("POSITIONING_OPPOSES_CHART" in c for c in verdict.conflicts)

    def test_long_at_call_wall_in_positive_gamma_is_blocked(self):
        # Spot pinned right at the call wall in +gamma -> dealers defend it.
        verdict = build_desk_verdict(
            signal=make_long_signal(),
            ticket=READY_TICKET,
            desk_state=make_desk_state(),
            current_spot=24600.0,
        )
        assert verdict.action == "WAIT"
        assert any("GEX_CALL_WALL_BLOCK" in c for c in verdict.conflicts)

    def test_institutional_flow_veto_blocks_long(self):
        verdict = build_desk_verdict(
            signal=make_long_signal(),
            ticket=READY_TICKET,
            desk_state=make_desk_state(),
            current_spot=24500.0,
            options_context={"flow_score": 12.0},
        )
        assert verdict.action == "WAIT"
        assert any("INSTITUTIONAL_FLOW_VETO" in c for c in verdict.conflicts)

    def test_term_structure_crisis_blocks_long(self):
        verdict = build_desk_verdict(
            signal=make_long_signal(),
            ticket=READY_TICKET,
            desk_state=make_desk_state(),
            vol_report={"term_structure_regime": {"is_crisis": True, "slope": -0.05}},
            current_spot=24500.0,
        )
        assert verdict.action == "WAIT"
        assert any("TERM_STRUCTURE_CRISIS" in c for c in verdict.conflicts)

    def test_clean_long_survives_with_no_conflicts(self):
        verdict = build_desk_verdict(
            signal=make_long_signal(),
            ticket=READY_TICKET,
            desk_state=make_desk_state(),
            current_spot=24500.0,
            options_context={"flow_score": 55.0},
        )
        assert verdict.action == "BUY_CE"
        assert verdict.conflicts == []
        assert verdict.option_pick is not None


class TestDeskOnlyBranches:
    def test_put_wall_fade_fires_when_chart_is_wait(self):
        # Spot defending the put wall in +gamma with supportive flow. Corroborating
        # structure/flow/macro supplied as a live setup would carry them, so this
        # exercises the fade branch rather than incidentally testing the conviction floor.
        verdict = build_desk_verdict(
            signal=make_wait_signal(),
            ticket={"status": "WAIT"},
            desk_state=make_desk_state(d_vector=0.35),
            current_spot=24405.0,
            htf_data={"htf_aligned_long": True},
            options_context={
                "flow_score": 68.0,
                "futures_basis": {"data_quality": "VERIFIED", "bias_score": 0.5,
                                  "basis_pts": 60.0, "annualised_basis_pct": 11.0},
            },
        )
        assert verdict.action == "BUY_CE"
        assert "Put Wall" in verdict.action_label

    def test_call_wall_fade_fires_when_chart_is_wait(self):
        verdict = build_desk_verdict(
            signal=make_wait_signal(),
            ticket={"status": "WAIT"},
            desk_state=make_desk_state(d_vector=-0.35, trend_bias="BEARISH"),
            current_spot=24595.0,
            htf_data={"htf_aligned_short": True},
            options_context={"flow_score": 30.0, "futures_basis": {"data_quality": "VERIFIED", "bias_score": -0.5, "basis_pts": -20.0, "annualised_basis_pct": 1.0}},
        )
        assert verdict.action == "BUY_PE"
        assert "Call Wall" in verdict.action_label

    def test_negative_gamma_breakout_fires_above_call_wall(self):
        verdict = build_desk_verdict(
            signal=make_wait_signal(),
            ticket={"status": "WAIT"},
            desk_state=make_desk_state(
                d_vector=0.65,
                is_positive_gamma=False,
                gamma_regime="DEALER_SHORT_GAMMA",
            ),
            current_spot=24650.0,
            htf_data={"htf_aligned_long": True},
            options_context={"flow_score": 72.0, "futures_basis": {"data_quality": "VERIFIED", "bias_score": 0.6, "basis_pts": 70.0, "annualised_basis_pct": 12.0}},
        )
        assert verdict.action == "BUY_CE"
        assert "Breakout" in verdict.action_label

    def test_term_structure_crisis_suppresses_desk_only_long_fade(self):
        # A crisis tape must not produce a desk-invented long.
        verdict = build_desk_verdict(
            signal=make_wait_signal(),
            ticket={"status": "WAIT"},
            desk_state=make_desk_state(d_vector=0.35),
            vol_report={"term_structure_regime": {"is_crisis": True, "slope": -0.05}},
            current_spot=24405.0,
        )
        assert verdict.action == "WAIT"

    def test_unverified_data_suppresses_desk_only_branches(self):
        verdict = build_desk_verdict(
            signal=make_wait_signal(),
            ticket={"status": "WAIT"},
            desk_state=make_desk_state(d_vector=0.35, data_quality="POSITIONING_UNVERIFIED"),
            current_spot=24405.0,
        )
        assert verdict.action == "WAIT"


class TestConvictionFloorIsAVeto:
    """Conviction used to be computed AFTER the action and used only to prefix the
    label, so a LOW-conviction setup still emitted a full ticket with entry, stops and
    lot count — the system grading its own trade poorly and taking it anyway."""

    def test_thin_evidence_is_refused(self):
        # Desk-only fade backed by positioning alone: structure, flow and macro have no
        # data at all. One family out of four is not a trade.
        verdict = build_desk_verdict(
            signal=make_wait_signal(),
            ticket={"status": "WAIT"},
            desk_state=make_desk_state(d_vector=0.35, trend_bias="BULLISH"),
            current_spot=24405.0,
        )
        assert verdict.action == "WAIT"
        assert any("CONVICTION_FLOOR" in c for c in verdict.conflicts)
        assert verdict.option_pick is None

    def test_corroborated_setup_survives(self):
        verdict = build_desk_verdict(
            signal=make_wait_signal(),
            ticket={"status": "WAIT"},
            desk_state=make_desk_state(d_vector=0.35, trend_bias="BULLISH"),
            current_spot=24405.0,
            htf_data={"htf_aligned_long": True},
            options_context={
                "flow_score": 68.0,
                "futures_basis": {"data_quality": "VERIFIED", "bias_score": 0.5,
                                  "basis_pts": 60.0, "annualised_basis_pct": 11.0},
            },
        )
        assert verdict.action == "BUY_CE"
        assert not any("CONVICTION_FLOOR" in c for c in verdict.conflicts)


class TestEvidenceOpposesTrade:
    """The four families were synthesised into a net directional score that the action
    table then ignored entirely."""

    def test_net_bearish_evidence_blocks_a_desk_long(self):
        verdict = build_desk_verdict(
            signal=make_wait_signal(),
            ticket={"status": "WAIT"},
            desk_state=make_desk_state(d_vector=0.35, trend_bias="BULLISH"),
            current_spot=24405.0,
            htf_data={"htf_aligned_short": True},
            options_context={
                "flow_score": 12.0,
                "futures_basis": {"data_quality": "VERIFIED", "bias_score": -0.9,
                                  "basis_pts": -50.0, "annualised_basis_pct": -4.0},
                "macro_report": {"macro_sentiment_score": -0.9},
            },
        )
        assert verdict.action == "WAIT"
        assert any("EVIDENCE_OPPOSES_TRADE" in c for c in verdict.conflicts)
