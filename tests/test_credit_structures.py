"""Premium-selling branch: fires only where it should, and refuses where it must.

Motivating incident (2026-08-19): the vol engine printed SELL_VOL with VRP +3.8%..+6.7% on
all 8 bars while the desk could only BUY_CE / BUY_PE / WAIT — structurally unable to act on
the one edge it measured all day.
"""

import pytest

from src.credit_structures import (
    evaluate_credit_opportunity,
    is_hard_veto,
    MIN_VRP_PCT,
    MIN_DTE_FOR_CREDIT,
)


def _vol(vrp_pct, regime="SELL_VOL"):
    return {"iv_rv_spread": {"spread_pct": vrp_pct, "vol_regime": regime},
            "composite_vol_regime": regime}


def _call(**over):
    kw = dict(
        spot=24100.0, vol_report=_vol(6.7), regime_state={"active_regime": "LOW_VOL_TRENDING"},
        t_days=3.5, iv=0.13, capital=500000.0, directional_action="WAIT",
        directional_reason="Market in consolidation / No confluence across core indicators.",
        risk_pct=0.01, lot_size=75, expected_move_pts=114.0,
    )
    kw.update(over)
    return evaluate_credit_opportunity(**kw)


class TestHardVetoNeverConverted:
    @pytest.mark.parametrize("reason", [
        "Event Risk Gate Veto: Active blackout window for 'RBI MPC Policy Decision' (CRITICAL).",
        "SESSION_RISK_LOCKED: Session Locked: 2-Strike Loss Circuit Breaker (2 consecutive losses).",
        "Data Sufficiency Gate: 3 core inputs unavailable (25d_skew, dealer_walls, positioning_flow).",
        "Opening 15-min range (Freak Candle isolation). True opening range is establishing.",
        "Session Risk Circuit Breaker: Concurrency Limit: 1 active position already open.",
    ])
    def test_hard_veto_blocks_credit(self, reason):
        assert is_hard_veto(reason) is True
        res = _call(directional_reason=reason)
        assert res["eligible"] is False
        assert "hard risk veto" in res["reason"].lower()

    def test_soft_consolidation_is_not_a_hard_veto(self):
        assert is_hard_veto("Market in consolidation / No confluence across core indicators.") is False


class TestFallbackDiscipline:
    def test_stands_down_when_directional_is_live(self):
        for action in ("BUY_CE", "BUY_PE"):
            res = _call(directional_action=action)
            assert res["eligible"] is False
            assert "stands down" in res["reason"].lower()


class TestVolGating:
    def test_requires_sell_vol_regime(self):
        res = _call(vol_report=_vol(8.0, regime="BUY_VOL"))
        assert res["eligible"] is False
        assert "does not favour selling" in res["reason"]

    def test_requires_meaningful_vrp(self):
        res = _call(vol_report=_vol(MIN_VRP_PCT - 0.5))
        assert res["eligible"] is False
        assert "too thin" in res["reason"]

    def test_refuses_near_expiry_short_gamma(self):
        res = _call(t_days=MIN_DTE_FOR_CREDIT - 0.5)
        assert res["eligible"] is False
        assert "short gamma" in res["reason"]

    def test_refuses_expansionary_regime(self):
        res = _call(regime_state={"active_regime": "HIGH_VOL_EXPANSION"})
        assert res["eligible"] is False
        assert "expansionary" in res["reason"]


class TestSizingAndOutput:
    def test_fires_on_the_real_2026_08_19_conditions(self):
        """VRP +6.7%, SELL_VOL, no directional edge, 3.5 DTE -> should be actionable."""
        res = _call()
        assert res["eligible"] is True, res["reason"]
        assert res["lots"] >= 1
        assert res["max_loss_rupees"] > 0
        assert res["net_credit_rupees"] > 0
        assert res["structure"]["strategy"] == "DELTA_NEUTRAL_IRON_CONDOR"

    def test_sizing_respects_defined_max_loss_budget(self):
        res = _call(capital=500000.0, risk_pct=0.01)
        assert res["eligible"] is True
        # Never risk more than the budget on the structure's DEFINED max loss.
        assert res["max_loss_rupees"] <= res["risk_budget_rupees"] * 1.0000001

    def test_declines_when_budget_cannot_cover_one_lot(self):
        res = _call(capital=10000.0, risk_pct=0.01)  # Rs.100 budget
        assert res["eligible"] is False
        assert "cannot size responsibly" in res["reason"]

    def test_never_promoted_off_directional_stats(self):
        res = _call()
        assert res["evidence_tier"] == "UNMEASURED"

    def test_all_legs_are_defined_risk(self):
        """Both short legs must be protected by a long wing — never naked."""
        legs = _call()["structure"]["legs"]
        assert legs["long_put"]["side"] == "BUY" and legs["short_put"]["side"] == "SELL"
        assert legs["long_call"]["side"] == "BUY" and legs["short_call"]["side"] == "SELL"
        assert legs["long_put"]["strike"] < legs["short_put"]["strike"]
        assert legs["long_call"]["strike"] > legs["short_call"]["strike"]
