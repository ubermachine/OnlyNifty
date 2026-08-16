"""Unit tests for Regime-Conditional Evidence Weights (v5.3).

Verifies that evidence family weights dynamically adapt to the active market regime:
- LOW_VOL_TRENDING emphasizes structure (0.40) and flow (0.30).
- MEAN_REVERTING_CHOP emphasizes options positioning (0.45).
- HIGH_VOL_EXPANSION emphasizes macro (0.35) and flow (0.35).
- All weight sets strictly sum to 1.00.
"""

import pytest
from src.config import REGIME_EVIDENCE_WEIGHTS
from src.desk_verdict import compute_evidence_families
from src.options_positioning import OptionsDeskState


def make_desk_state(**kwargs) -> OptionsDeskState:
    defaults = {
        "trend_bias": "NEUTRAL",
        "trend_conviction_pct": 50.0,
        "d_vector": 0.0,
        "pcr_level": 1.0,
        "pcr_zscore": 0.0,
        "pcr_momentum_score": 0.0,
        "put_wall": 24300.0,
        "call_wall": 24700.0,
        "max_pain": 24500.0,
        "max_pain_drift_pts": 0.0,
        "expected_move_pts": 120.0,
        "actual_range_pts": 80.0,
        "move_ratio": 0.67,
        "gamma_regime": "DEALER_LONG_GAMMA",
        "is_positive_gamma": True,
        "zero_gex_strike": 24500.0,
        "writing_bias": "BALANCED_RANGE",
        "itm_otm_shift": 0.0,
        "agreement_count": 2,
        "data_quality": "VERIFIED",
        "dealer_drift_score": 0.0,
        "dwv_momentum_score": 0.0,
        "gamma_flip_distance_pts": 0.0,
    }
    defaults.update(kwargs)
    return OptionsDeskState(**defaults)


class TestRegimeConditionalWeights:
    def test_weights_sum_to_one(self):
        for regime, weights in REGIME_EVIDENCE_WEIGHTS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-6, f"Weights for {regime} sum to {total}, expected 1.0"

    def test_trending_regime_weights_structure_heavily(self):
        # In trending regime, structure vote +1 should have 0.40 weight
        htf_data = {"htf_aligned_long": True, "htf_aligned_short": False}
        regime_data = {"active_regime": "LOW_VOL_TRENDING"}
        desk = make_desk_state(d_vector=0.0)
        
        votes, _, direction = compute_evidence_families(
            desk_state=desk,
            htf_data=htf_data,
            regime_state=regime_data
        )
        assert votes["structure"] == 1
        # With only structure voting +1 and trending weight = 0.40, direction must equal 0.40
        assert direction == 0.40

    def test_chop_regime_weights_positioning_heavily(self):
        # In chop regime, positioning vote +1 should have 0.45 weight
        regime_data = {"active_regime": "MEAN_REVERTING_CHOP"}
        desk = make_desk_state(d_vector=0.45)
        
        votes, _, direction = compute_evidence_families(
            desk_state=desk,
            regime_state=regime_data
        )
        assert votes["positioning"] == 1
        # With only positioning voting +1 and chop weight = 0.45, direction must equal 0.45
        assert direction == 0.45

    def test_high_vol_expansion_weights_macro_and_flow(self):
        # In high-vol expansion, macro vote -1 should have 0.35 weight
        regime_data = {"active_regime": "HIGH_VOL_EXPANSION"}
        desk = make_desk_state(d_vector=0.0)
        vol_report = {"term_structure_regime": {"is_crisis": True}}
        
        votes, _, direction = compute_evidence_families(
            desk_state=desk,
            regime_state=regime_data,
            vol_report=vol_report
        )
        assert votes["macro"] == -1
        assert direction == -0.35
