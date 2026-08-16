"""Gates must fail CLOSED when the desk is blind.

Almost every universal gate fails OPEN on missing input: an absent VPIN reads 0.0 and
passes, an absent skew has is_crash_hedging=False and passes, an absent chain leaves
walls unverified so the pin check is skipped entirely. The net effect was that the exact
moment the desk was blindest — a data outage — was when the fewest vetoes could fire.

GATE_FAIL_TO_WAIT existed in config and was imported by strategy_rules but never read.
"""

import pandas as pd
import pytest

from src.config import GATE_FAIL_TO_WAIT, GATE_MIN_MISSING_TO_BLOCK
from src.strategy_rules import StrategyEngine


REAL_SKEW = {"skew_zscore": 0.4, "is_crash_hedging": False, "data_quality": "VERIFIED"}
REAL_GEX = {"is_positive_gamma": True, "call_wall_strike": 25500.0,
            "put_wall_strike": 24000.0, "walls_verified": True}
REAL_CTX = {"dir_flow": {"directional_vector": 0.1}}
ALIGNED = {"htf_aligned_long": True, "htf_aligned_short": False}


def gates(engine, **over):
    kwargs = dict(
        candidate_direction="LONG", close=24750.0,
        skew_info=dict(REAL_SKEW), vpin_info={"vpin": 0.2}, hfi_score=0.05,
        gex_info=dict(REAL_GEX), htf_regime=dict(ALIGNED), options_context=dict(REAL_CTX),
    )
    kwargs.update(over)
    return engine._apply_universal_gates(**kwargs)


class TestDataSufficiencyGate:
    def test_config_flag_is_on(self):
        assert GATE_FAIL_TO_WAIT is True

    def test_full_data_passes(self):
        passed, reason, audit = gates(StrategyEngine())
        assert passed, reason
        assert audit["missing_inputs"] == []

    def test_total_blindness_blocks(self):
        """No chain at all: skew synthetic, walls unverified, no positioning flow."""
        passed, reason, audit = gates(
            StrategyEngine(),
            skew_info={"skew_zscore": 0.0, "is_crash_hedging": False, "data_quality": "SYNTHETIC"},
            gex_info={"is_positive_gamma": True, "call_wall_strike": 25500.0,
                      "put_wall_strike": 24000.0, "walls_verified": False},
            options_context={},
        )
        assert not passed
        assert audit["veto_gate"] == "INSUFFICIENT_GATE_DATA"
        assert len(audit["missing_inputs"]) >= GATE_MIN_MISSING_TO_BLOCK
        assert "standing aside" in reason

    def test_partial_degradation_still_trades(self):
        """Thin data reduces size downstream rather than blocking — the desk should stay
        useful when data is merely incomplete, and stand aside only when truly blind."""
        passed, _, audit = gates(
            StrategyEngine(),
            gex_info={"is_positive_gamma": True, "call_wall_strike": 25500.0,
                      "put_wall_strike": 24000.0, "walls_verified": False},
        )
        assert passed
        assert "dealer_walls" in audit["missing_inputs"]

    def test_missing_inputs_are_always_audited(self):
        # Even when the gate passes, what was unavailable must be recorded so the
        # journal shows the basis on which the trade was taken.
        _, _, audit = gates(
            StrategyEngine(),
            options_context={},
        )
        assert "missing_inputs" in audit
        assert "positioning_flow" in audit["missing_inputs"]

    @pytest.mark.parametrize("direction", ["LONG", "SHORT"])
    def test_blindness_blocks_both_directions(self, direction):
        passed, _, audit = gates(
            StrategyEngine(),
            candidate_direction=direction,
            skew_info={"skew_zscore": 0.0, "is_crash_hedging": False, "data_quality": "SYNTHETIC"},
            gex_info={"is_positive_gamma": True, "call_wall_strike": 25500.0,
                      "put_wall_strike": 24000.0, "walls_verified": False},
            options_context={},
            htf_regime={"htf_aligned_long": True, "htf_aligned_short": True},
        )
        assert not passed
        assert audit["veto_gate"] == "INSUFFICIENT_GATE_DATA"

    def test_sufficiency_runs_before_direction_specific_gates(self):
        """A blind desk must report blindness, not a downstream veto that happened to
        evaluate on absent data."""
        passed, _, audit = gates(
            StrategyEngine(),
            hfi_score=-0.9,   # would otherwise trip the HFI veto
            skew_info={"skew_zscore": 0.0, "is_crash_hedging": False, "data_quality": "SYNTHETIC"},
            gex_info={"is_positive_gamma": True, "call_wall_strike": 25500.0,
                      "put_wall_strike": 24000.0, "walls_verified": False},
            options_context={},
        )
        assert not passed
        assert audit["veto_gate"] == "INSUFFICIENT_GATE_DATA"
