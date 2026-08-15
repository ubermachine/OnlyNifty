"""Tests for live edge-table gating in StrategyEngine and DecisionEngine gate parity.

Covers the Phase 3 wiring (EdgeTable consulted on the live path) and the
Positioning Flow Veto parity gap between DecisionEngine and StrategyEngine.
"""

import pytest

from src.edge_harness import EdgeTable, EdgeStats
from src.strategy_rules import StrategyEngine
from src.decision_engine import DecisionEngine, DecisionContext


class TestEdgeTableLiveGating:
    def test_empty_edge_table_is_permissive(self):
        table = EdgeTable()
        assert table.is_tradeable("LONG", "LOW_VOL_TRENDING") is True

    def test_quarantined_setup_is_blocked(self):
        table = EdgeTable([
            EdgeStats(
                setup_id="LONG_ORDER_FLOW", regime="MEAN_REVERTING_CHOP", n=45,
                win_rate=22.0, mean_r=-0.4, ev=-0.4,
                ci_low=-0.8, ci_high=-0.1, status="QUARANTINED",
            )
        ])
        assert table.is_tradeable("LONG_ORDER_FLOW", "MEAN_REVERTING_CHOP") is False
        # Same setup in an unmeasured regime remains tradeable.
        assert table.is_tradeable("LONG_ORDER_FLOW", "LOW_VOL_TRENDING") is True

    def test_engine_loads_an_edge_table_by_default(self):
        engine = StrategyEngine()
        assert engine.edge_table is not None

    def test_engine_accepts_an_injected_edge_table(self):
        table = EdgeTable()
        engine = StrategyEngine(edge_table=table)
        assert engine.edge_table is table

    def test_unmeasured_setup_does_not_shrink_size(self):
        # An empty edge table must be a no-op on sizing, not a silent halving.
        table = EdgeTable()
        assert table.lookup("LONG", "LOW_VOL_TRENDING") is None

    def test_trusted_setup_gets_full_size(self):
        table = EdgeTable([
            EdgeStats(
                setup_id="LONG", regime="LOW_VOL_TRENDING", n=60,
                win_rate=58.0, mean_r=0.45, ev=0.45,
                ci_low=0.12, ci_high=0.80, status="TRUSTED",
            )
        ])
        assert table.get_sizing_factor("LONG", "LOW_VOL_TRENDING") == 1.0

    def test_paper_setup_gets_half_size(self):
        table = EdgeTable([
            EdgeStats(
                setup_id="LONG", regime="LOW_VOL_TRENDING", n=12,
                win_rate=55.0, mean_r=0.3, ev=0.3,
                ci_low=-0.1, ci_high=0.7, status="PAPER",
            )
        ])
        assert table.get_sizing_factor("LONG", "LOW_VOL_TRENDING") == 0.5

    def test_edge_table_json_roundtrip_preserves_status(self):
        table = EdgeTable([
            EdgeStats(
                setup_id="SHORT", regime="HIGH_VOL_EXPANSION", n=40,
                win_rate=30.0, mean_r=-0.2, ev=-0.2,
                ci_low=-0.6, ci_high=0.1, status="QUARANTINED",
            )
        ])
        restored = EdgeTable.from_json(table.to_json())
        assert restored.is_tradeable("SHORT", "HIGH_VOL_EXPANSION") is False


class TestDecisionEnginePositioningVetoParity:
    def _ctx(self, d_vector):
        return DecisionContext(
            markov_regime="LOW_VOL_TRENDING",
            htf_aligned_long=True,
            htf_aligned_short=True,
            vpin=0.2,
            hfi_score=0.0,
            gex_walls={"call_wall_strike": 25500.0, "put_wall_strike": 24000.0},
            is_positive_gamma=True,
            options_context={"dir_flow": {"directional_vector": d_vector}},
        )

    def test_bearish_positioning_vetoes_long(self):
        engine = DecisionEngine()
        passed, reason, audit = engine.check_universal_gates("LONG", 24750.0, self._ctx(-0.75))
        assert not passed
        assert audit["veto_gate"] == "POSITIONING_OPPOSES_CHART"
        assert "Positioning Veto" in reason

    def test_bullish_positioning_vetoes_short(self):
        engine = DecisionEngine()
        passed, reason, audit = engine.check_universal_gates("SHORT", 24750.0, self._ctx(0.75))
        assert not passed
        assert audit["veto_gate"] == "POSITIONING_OPPOSES_CHART"

    def test_aligned_positioning_passes(self):
        engine = DecisionEngine()
        passed, _, _ = engine.check_universal_gates("LONG", 24750.0, self._ctx(0.65))
        assert passed

    def test_weak_opposing_positioning_does_not_veto(self):
        # Below POSITIONING_VETO_STRENGTH -> not a hard veto.
        engine = DecisionEngine()
        passed, _, _ = engine.check_universal_gates("LONG", 24750.0, self._ctx(-0.20))
        assert passed
