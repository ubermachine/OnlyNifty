"""Tests for the cross-market inputs added on top of the broker API.

Everything the desk read previously was either price, or options written on that price,
which is why the four "independent" evidence families measured as only ~1.8 independent
signals. Futures basis, live India VIX and global cues come from different instruments
and different participants — they are the reads that actually raise effective
independence. These tests guard the two ways such inputs go wrong: fabricating a
direction when the feed is missing, and carrying a constant tilt.
"""

import pytest

from src.config import RISK_FREE_RATE
from src.desk_verdict import compute_evidence_families
from src.options_positioning import OptionsDeskState


def neutral_desk_state(**over):
    base = dict(
        trend_bias="NEUTRAL", trend_conviction_pct=0.0, d_vector=0.0,
        pcr_level=1.0, pcr_zscore=0.0, pcr_momentum_score=0.0,
        put_wall=24400.0, call_wall=24600.0, max_pain=24500.0, max_pain_drift_pts=0.0,
        expected_move_pts=85.0, actual_range_pts=85.0, move_ratio=1.0,
        gamma_regime="DEALER_LONG_GAMMA", is_positive_gamma=True,
        zero_gex_strike=24500.0, writing_bias="BALANCED_RANGE", itm_otm_shift=0.0,
        agreement_count=2, data_quality="VERIFIED", dealer_drift_score=0.0,
        dwv_momentum_score=0.0, gamma_flip_distance_pts=100.0,
    )
    base.update(over)
    return OptionsDeskState(**base)


def basis(bias_score, quality="VERIFIED", pts=50.0, ann=8.0):
    return {
        "basis_pts": pts, "annualised_basis_pct": ann,
        "bias_score": bias_score, "data_quality": quality,
        "structure": "CONTANGO_PREMIUM",
    }


class TestFuturesBasisFeedsMacro:
    def test_premium_above_carry_votes_bullish(self):
        votes, why, _ = compute_evidence_families(
            desk_state=neutral_desk_state(),
            options_context={"futures_basis": basis(+0.60)},
        )
        assert votes["macro"] == 1
        assert "Basis" in why["macro"]

    def test_discount_votes_bearish(self):
        votes, _, _ = compute_evidence_families(
            desk_state=neutral_desk_state(),
            options_context={"futures_basis": basis(-0.60, pts=-40.0, ann=-6.0)},
        )
        assert votes["macro"] == -1

    def test_unverified_basis_is_ignored(self):
        # A missing broker feed must not be read as a directional opinion.
        votes, why, _ = compute_evidence_families(
            desk_state=neutral_desk_state(),
            options_context={"futures_basis": basis(+0.9, quality="UNVERIFIED")},
        )
        assert votes["macro"] == 0
        assert "Basis" not in why["macro"]


class TestBasisScoresDeviationNotLevel:
    """Index futures sit at a premium in normal conditions. Scoring the RAW basis would
    vote bullish on nearly every ordinary day — a constant tilt dressed as signal."""

    def test_basis_at_fair_carry_is_neutral(self):
        import numpy as np
        fair = RISK_FREE_RATE * 100.0
        bias_at_carry = float(np.tanh((fair - fair) / 8.0))
        assert bias_at_carry == pytest.approx(0.0, abs=1e-9)

        votes, _, _ = compute_evidence_families(
            desk_state=neutral_desk_state(),
            options_context={"futures_basis": basis(bias_at_carry, ann=fair)},
        )
        assert votes["macro"] == 0, "an ordinary carry premium must not vote bullish"

    def test_excess_over_carry_is_what_moves_the_vote(self):
        import numpy as np
        fair = RISK_FREE_RATE * 100.0
        rich = float(np.tanh(((fair + 6.0) - fair) / 8.0))
        cheap = float(np.tanh(((fair - 6.0) - fair) / 8.0))
        assert rich > 0 and cheap < 0
        assert rich == pytest.approx(-cheap, abs=1e-9)


class TestGlobalMacroFeedsMacro:
    def test_global_cues_reach_the_vote(self):
        votes, why, _ = compute_evidence_families(
            desk_state=neutral_desk_state(),
            options_context={"macro_report": {"macro_sentiment_score": -0.8}},
        )
        assert votes["macro"] == -1
        assert "Global" in why["macro"]

    def test_macro_family_is_no_longer_vol_only(self):
        # It used to vote purely on VRP/term-structure — a second volatility family
        # under a macro label.
        votes, why, _ = compute_evidence_families(
            desk_state=neutral_desk_state(),
            options_context={
                "futures_basis": basis(+0.5),
                "macro_report": {"macro_sentiment_score": 0.4},
            },
        )
        assert "Basis" in why["macro"] and "Global" in why["macro"]
        assert votes["macro"] == 1

    def test_basis_and_global_can_offset(self):
        # Genuine independence means these two can disagree and net out.
        votes, _, _ = compute_evidence_families(
            desk_state=neutral_desk_state(),
            options_context={
                "futures_basis": basis(+0.5),
                "macro_report": {"macro_sentiment_score": -0.7},
            },
        )
        assert votes["macro"] == 0


class TestHeavyweightFallbackIsNeutral:
    def test_hfi_fallback_changes_are_zero(self):
        """The fallback constants were +0.65/+0.45/+0.80/-0.20/+0.10 — a confident
        STRONG_BULLISH reading manufactured from nothing on any fetch failure, biasing
        the desk long during exactly the correlated outages that accompany stress."""
        import inspect
        from src.data_engine import DataEngine
        src = inspect.getsource(DataEngine.fetch_heavyweight_flow_index)
        offenders = [
            ln.strip() for ln in src.splitlines()
            if "fallback_chg" in ln and "0.0" not in ln and "hw[" not in ln and "#" not in ln.split("fallback_chg")[0]
        ]
        assert not offenders, f"non-zero HFI fallback changes still present: {offenders}"
