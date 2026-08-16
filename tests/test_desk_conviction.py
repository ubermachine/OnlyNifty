"""Tests for desk verdict conviction synthesis and evidence-family voting.

The goal these protect: direction alone is not a trade. A signal must also carry a
defensible measure of HOW HARD to bet, built from evidence families that can
genuinely disagree with each other.
"""

import pytest

from src.strategy_rules import Signal, SignalType
from src.options_positioning import OptionsDeskState
from src.desk_verdict import (
    build_desk_verdict,
    compute_evidence_families,
    compute_conviction,
)


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
        dealer_drift_score=0.10,
        dwv_momentum_score=0.30,
        gamma_flip_distance_pts=120.0,
    )
    base.update(overrides)
    return OptionsDeskState(**base)


def long_signal(score=82.0):
    return Signal(
        signal_type=SignalType.LONG,
        entry_price=24500.0,
        sl_price=24460.0,
        target_1=24550.0,
        target_2=24600.0,
        target_3_moonshot=24680.0,
        reason="Golden Pocket Confirmed",
        details={"confluence_score": score, "confluence_grade": "A Standard"},
    )


BULL_HTF = {"htf_aligned_long": True, "htf_aligned_short": False}
BEAR_HTF = {"htf_aligned_long": False, "htf_aligned_short": True}


class TestEvidenceFamilies:
    def test_all_four_families_are_scored(self):
        votes, why, _ = compute_evidence_families(
            desk_state=make_desk_state(), htf_data=BULL_HTF,
            regime_state={"active_regime": "LOW_VOL_TRENDING"},
        )
        assert set(votes) == {"structure", "flow", "positioning", "macro"}
        assert set(why) == {"structure", "flow", "positioning", "macro"}

    def test_bullish_stack_votes_bullish(self):
        votes, _, direction = compute_evidence_families(
            desk_state=make_desk_state(), htf_data=BULL_HTF,
            options_context={"flow_score": 78.0},
        )
        assert votes["structure"] == 1
        assert votes["positioning"] == 1
        assert votes["flow"] == 1
        assert direction > 0

    def test_bearish_stack_votes_bearish(self):
        votes, _, direction = compute_evidence_families(
            desk_state=make_desk_state(
                d_vector=-0.6, itm_otm_shift=-0.4,
                writing_bias="CALL_WRITING_HEAVY_RESISTANCE",
                dwv_momentum_score=-0.4, dealer_drift_score=-0.2,
            ),
            htf_data=BEAR_HTF,
            options_context={"flow_score": 18.0},
        )
        assert votes["structure"] == -1
        assert votes["positioning"] == -1
        assert votes["flow"] == -1
        assert direction < 0

    def test_families_can_disagree(self):
        # Bullish chart structure against bearish options positioning.
        votes, _, _ = compute_evidence_families(
            desk_state=make_desk_state(
                d_vector=-0.7, itm_otm_shift=-0.5,
                writing_bias="CALL_WRITING_HEAVY_RESISTANCE",
            ),
            htf_data=BULL_HTF,
        )
        assert votes["structure"] == 1
        assert votes["positioning"] == -1

    def test_previously_discarded_fields_move_the_vote(self):
        # writing_bias / itm_otm_shift / dealer_drift were computed but never read.
        neutral, _, _ = compute_evidence_families(
            desk_state=make_desk_state(
                d_vector=0.0, itm_otm_shift=0.0, writing_bias="BALANCED_RANGE",
                pcr_momentum_score=0.0, dealer_drift_score=0.0,
            )
        )
        loaded, _, _ = compute_evidence_families(
            desk_state=make_desk_state(
                d_vector=0.0, itm_otm_shift=0.6,
                writing_bias="PUT_WRITING_HEAVY_SUPPORT",
                pcr_momentum_score=0.0, dealer_drift_score=0.0,
            )
        )
        assert neutral["positioning"] == 0
        assert loaded["positioning"] == 1

    def test_term_structure_crisis_makes_macro_bearish(self):
        votes, _, _ = compute_evidence_families(
            desk_state=make_desk_state(),
            vol_report={"term_structure_regime": {"is_crisis": True}},
        )
        assert votes["macro"] == -1


class TestConvictionTiers:
    def test_full_agreement_is_high_or_extreme(self):
        votes = {"structure": 1, "flow": 1, "positioning": 1, "macro": 1}
        score, tier, agree, _ = compute_conviction("BUY_CE", votes, 85.0)
        assert agree == 4
        assert tier in {"HIGH", "EXTREME"}
        assert score >= 65

    def test_opposed_families_reduce_conviction(self):
        agreed = {"structure": 1, "flow": 1, "positioning": 1, "macro": 1}
        split = {"structure": 1, "flow": -1, "positioning": -1, "macro": 0}
        high, _, _, _ = compute_conviction("BUY_CE", agreed, 80.0)
        low, _, _, _ = compute_conviction("BUY_CE", split, 80.0)
        assert low < high

    def test_wait_has_no_conviction(self):
        votes = {"structure": 1, "flow": 1, "positioning": 1, "macro": 1}
        score, tier, agree, _ = compute_conviction("WAIT", votes, 90.0)
        assert score == 0.0
        assert tier == "LOW"
        assert agree == 0

    def test_quarantined_setup_is_capped_low(self):
        votes = {"structure": 1, "flow": 1, "positioning": 1, "macro": 1}
        score, tier, _, notes = compute_conviction(
            "BUY_CE", votes, 95.0, edge_status="QUARANTINED"
        )
        assert score <= 20.0
        assert tier == "LOW"
        assert any("QUARANTINED" in n for n in notes)

    def test_unverified_positioning_caps_conviction(self):
        votes = {"structure": 1, "flow": 1, "positioning": 1, "macro": 1}
        score, _, _, notes = compute_conviction(
            "BUY_CE", votes, 95.0, data_quality="POSITIONING_UNVERIFIED"
        )
        assert score <= 55.0
        assert any("UNVERIFIED" in n for n in notes)

    def test_range_exhaustion_penalises_late_breakouts(self):
        # edge_status=TRUSTED so the UNMEASURED/PAPER 65-cap does not clip both sides
        # to the same value and hide the dampener under test.
        votes = {"structure": 1, "flow": 1, "positioning": 1, "macro": 1}
        fresh, _, _, _ = compute_conviction(
            "BUY_CE", votes, 80.0, desk_state=make_desk_state(move_ratio=0.5),
            edge_status="TRUSTED", is_breakout=True
        )
        late, _, _, notes = compute_conviction(
            "BUY_CE", votes, 80.0, desk_state=make_desk_state(move_ratio=1.8),
            edge_status="TRUSTED", is_breakout=True
        )
        assert late < fresh
        assert any("exhausted" in n for n in notes)

    def test_sitting_on_gamma_flip_reduces_conviction(self):
        votes = {"structure": 1, "flow": 1, "positioning": 1, "macro": 1}
        stable, _, _, _ = compute_conviction(
            "BUY_CE", votes, 80.0, edge_status="TRUSTED",
            desk_state=make_desk_state(gamma_flip_distance_pts=200.0)
        )
        unstable, _, _, notes = compute_conviction(
            "BUY_CE", votes, 80.0, edge_status="TRUSTED",
            desk_state=make_desk_state(gamma_flip_distance_pts=5.0)
        )
        assert unstable < stable
        assert any("gamma flip" in n for n in notes)

    def test_unmeasured_never_outranks_paper(self):
        # Less evidence must not score higher: UNMEASURED was uncapped while PAPER
        # capped at 65, and with no edge_table.json on disk the uncapped branch was
        # the one always taken.
        votes = {"structure": 1, "flow": 1, "positioning": 1, "macro": 1}
        unmeasured, _, _, _ = compute_conviction("BUY_CE", votes, 95.0, edge_status="UNMEASURED")
        paper, _, _, _ = compute_conviction("BUY_CE", votes, 95.0, edge_status="PAPER")
        trusted, _, _, _ = compute_conviction("BUY_CE", votes, 95.0, edge_status="TRUSTED")
        assert unmeasured <= paper
        assert trusted >= paper


class TestVerdictIntegration:
    def test_verdict_exposes_conviction_fields(self):
        v = build_desk_verdict(
            signal=long_signal(),
            ticket={"status": "READY", "symbol": "NIFTY 24500 CE", "entry_premium": 140.0},
            desk_state=make_desk_state(),
            htf_data=BULL_HTF,
            regime_state={"active_regime": "LOW_VOL_TRENDING"},
            options_context={"flow_score": 72.0},
            current_spot=24500.0,
        )
        assert v.action == "BUY_CE"
        assert v.conviction_tier in {"LOW", "MODERATE", "HIGH", "EXTREME"}
        assert 0.0 <= v.conviction_score <= 100.0
        assert v.family_agreement >= 3
        assert v.conviction_tier in v.action_label

    def test_htf_data_now_influences_the_verdict(self):
        # htf_data used to be an accepted-but-ignored parameter.
        common = dict(
            signal=long_signal(),
            ticket={"status": "READY", "symbol": "NIFTY 24500 CE", "entry_premium": 140.0},
            desk_state=make_desk_state(),
            current_spot=24500.0,
        )
        aligned = build_desk_verdict(htf_data=BULL_HTF, **common)
        against = build_desk_verdict(htf_data=BEAR_HTF, **common)
        assert aligned.family_votes["structure"] == 1
        assert against.family_votes["structure"] == -1
        assert aligned.conviction_score > against.conviction_score

    def test_desk_only_trade_gets_a_real_confluence_score(self):
        # Desk-invented fades previously inherited a hardcoded 50.0 placeholder.
        wait_sig = Signal(
            signal_type=SignalType.WAIT, entry_price=24405.0, sl_price=0.0,
            target_1=0.0, target_2=0.0, reason="Market in consolidation.",
        )
        v = build_desk_verdict(
            signal=wait_sig,
            ticket={"status": "WAIT"},
            desk_state=make_desk_state(d_vector=0.35),
            htf_data=BULL_HTF,
            current_spot=24405.0,
        )
        assert v.action == "BUY_CE"
        assert v.confluence_score != 50.0
        assert v.confluence_grade in {
            "A+ Institutional", "A Standard", "B Moderate", "C Weak / Vetoed",
        }

    def test_wait_verdict_reports_zero_conviction(self):
        wait_sig = Signal(
            signal_type=SignalType.WAIT, entry_price=24500.0, sl_price=0.0,
            target_1=0.0, target_2=0.0, reason="Market in consolidation.",
        )
        v = build_desk_verdict(
            signal=wait_sig, ticket={"status": "WAIT"},
            desk_state=make_desk_state(d_vector=0.0, pcr_zscore=0.0),
            current_spot=24500.0,
        )
        assert v.action == "WAIT"
        assert v.conviction_score == 0.0

    def test_evidence_pills_surface_previously_unused_fields(self):
        v = build_desk_verdict(
            signal=long_signal(),
            ticket={"status": "READY", "symbol": "NIFTY 24500 CE", "entry_premium": 140.0},
            desk_state=make_desk_state(),
            htf_data=BULL_HTF,
            current_spot=24500.0,
        )
        positioning = v.evidence["positioning"]
        assert "OI shift" in positioning
        assert "Writing" in positioning or "Put Writing" in positioning
        assert "Vanna/Charm" in positioning
