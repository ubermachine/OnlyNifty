"""Unit tests verifying that Options Positioning is orthogonalized:
- Positioning collapses collinear OI numerators into PC1 (D_vector) + Gamma regime.
- Agreement count uses 4 genuinely independent pillars (D-vector, PCR Z-Score, Gamma structure, HFI).
- No multi-counting of identical OI delta measures (r=0.88-0.96).
"""

import pytest
import pandas as pd
import numpy as np

from src.desk_verdict import compute_evidence_families
from src.options_positioning import OptionsDeskState, compute_options_desk_state
from src.data_engine import DataEngine


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


class TestOrthogonalPositioning:
    def test_d_vector_dominates_pc1(self):
        """Primary directional flow is captured by D_vector without redundant multiplier inflation."""
        bull_desk = make_desk_state(d_vector=0.45, itm_otm_shift=0.0, writing_bias="BALANCED_RANGE")
        votes, why, _ = compute_evidence_families(desk_state=bull_desk)
        assert votes["positioning"] == 1
        assert "D +0.45" in why["positioning"]
        assert "Dealer +Γ" in why["positioning"]

    def test_gamma_regime_is_orthogonal_to_direction(self):
        """Dealer gamma regime provides structural context independent of directional sign."""
        pos_gamma = make_desk_state(d_vector=0.0, gamma_regime="DEALER_LONG_GAMMA", is_positive_gamma=True)
        neg_gamma = make_desk_state(d_vector=0.0, gamma_regime="DEALER_SHORT_GAMMA", is_positive_gamma=False)
        
        _, why_pos, _ = compute_evidence_families(desk_state=pos_gamma)
        _, why_neg, _ = compute_evidence_families(desk_state=neg_gamma)
        
        assert "Dealer +Γ (Mean Reversion / Pin)" in why_pos["positioning"]
        assert "Dealer -Γ (Breakout Expansion)" in why_neg["positioning"]

    def test_orthogonal_agreement_count(self):
        """Agreement count synthesizes 4 distinct pillars: D-vector, PCR Z-Score, Gamma, and Institutional Flow."""
        # Bullish alignment across all 4 orthogonal dimensions
        spot = 24550.0
        chain = pd.DataFrame({
            "strike": [24500, 24550, 24600],
            "type": ["CE", "CE", "CE"],
            "open_interest": [100000, 150000, 200000],
            "change_in_oi": [10000, 20000, 30000],
            "ltp": [150.0, 110.0, 80.0],
            "iv": [0.13, 0.13, 0.13]
        })
        state = compute_options_desk_state(
            option_chain_df=chain,
            spot=spot,
            live_iv=0.135,
            hfi_score=0.25,
            dir_flow_res={"directional_vector": 0.60, "short_term_bias": "STRONG_BULLISH_LONG", "conviction_percentage": 60.0, "sub_scores": {}},
            pcr_analytics={"pcr_oi": 1.25, "max_pain_strike": 24500.0},
            persist_history=False
        )
        assert state.trend_bias == "BULLISH"
        assert state.agreement_count >= 3
