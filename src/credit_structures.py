"""Premium-selling branch — lets the desk express the vol edge it already measures.

THE GAP THIS CLOSES
On 2026-08-19 the volatility engine printed SELL_VOL with a positive variance risk premium
(+3.8% to +6.7%) on all 8 bars of the session — implied vol richer than realized, all day.
The desk's entire action space, however, is BUY_CE / BUY_PE / WAIT: every one of the 17
SignalTypes is a directional option *purchase*. The system was therefore structurally
incapable of harvesting the one edge it detects continuously, and instead bought premium
into a rich-vol tape, paying theta and crossing spreads on deep-ITM contracts.

The defined-risk structures to express it already existed (construct_delta_neutral_iron_condor,
construct_ratio_spread, construct_jade_lizard) — but sat in a collapsed expander inside the
backtest tab, wired to nothing.

DESIGN
This is a strict FALLBACK, never a competitor:
  * It is only consulted when the directional engine has already declined (action == WAIT).
  * It refuses to fire on a HARD veto (event blackout, session risk lock, data
    insufficiency, opening-range isolation) — those mean "take no risk", not "take
    different risk".
  * It is defined-risk only (iron condor). Naked short premium is never constructed here.
  * It refuses near expiry, where short gamma is the classic blowup.
  * It sizes off the structure's DEFINED MAX LOSS, not off a directional stop.

Its outcomes are deliberately NOT folded into the directional edge table: an option-credit
R is not commensurable with a directional R, and mixing them would corrupt both.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.options_engine import construct_delta_neutral_iron_condor

# --- Gating thresholds -------------------------------------------------------------
# Variance risk premium must be genuinely rich, not merely positive: crossing four legs'
# bid/ask is expensive, so a marginal edge is consumed by friction before it is earned.
MIN_VRP_PCT: float = 3.0
# Short gamma near expiry is the classic premium-seller blowup — a condor that is
# comfortable on Monday is a liability on expiry morning.
MIN_DTE_FOR_CREDIT: float = 1.0
# Structure must offer a genuinely wide profit corridor relative to the expected move,
# otherwise a single ordinary session walks straight through a short strike.
MIN_CORRIDOR_TO_EXPECTED_MOVE: float = 1.30
MIN_POP_PCT: float = 60.0

# A WAIT carrying any of these is a hard risk veto — the correct response is no exposure
# in any form, not a different structure.
HARD_VETO_MARKERS = (
    "Event Risk Gate",
    "Session Risk Circuit Breaker",
    "SESSION_RISK_LOCKED",
    "Data Sufficiency Gate",
    "Opening 15-min range",
    "Freak",
    "Concurrency Limit",
    "blackout",
    "Circuit Breaker",
)


def is_hard_veto(reason: str) -> bool:
    """True when a WAIT means 'take no risk at all' rather than 'no directional edge'."""
    if not reason:
        return False
    return any(marker.lower() in reason.lower() for marker in HARD_VETO_MARKERS)


def evaluate_credit_opportunity(
    spot: float,
    vol_report: Optional[Dict[str, Any]],
    regime_state: Optional[Dict[str, Any]],
    t_days: float,
    iv: float,
    capital: float,
    directional_action: str,
    directional_reason: str = "",
    risk_pct: float = 0.01,
    lot_size: int = 75,
    expected_move_pts: float = 0.0,
) -> Dict[str, Any]:
    """Returns a defined-risk credit recommendation, or a declined verdict explaining why.

    Always returns a dict with "eligible" (bool) and "reason" (str) so the caller can
    surface *why* the vol edge was or wasn't actionable.
    """
    declined = lambda why: {"eligible": False, "reason": why, "structure": None}

    # 1. Fallback discipline: never compete with a live directional call.
    if str(directional_action).upper() not in ("WAIT", "", "NONE"):
        return declined("Directional setup active — credit branch stands down.")

    # 2. Never convert a hard risk veto into a different kind of risk.
    if is_hard_veto(directional_reason):
        return declined("Hard risk veto in force — no exposure in any structure.")

    # 3. Volatility must actually be rich.
    ivrv = (vol_report or {}).get("iv_rv_spread") or {}
    vrp_pct = float(ivrv.get("spread_pct", 0.0) or 0.0)
    vol_regime = str(ivrv.get("vol_regime", "") or (vol_report or {}).get("composite_vol_regime", ""))
    if "SELL_VOL" not in vol_regime.upper():
        return declined(f"Vol regime {vol_regime or 'UNKNOWN'} does not favour selling premium.")
    if vrp_pct < MIN_VRP_PCT:
        return declined(f"VRP +{vrp_pct:.1f}% below the +{MIN_VRP_PCT:.1f}% floor — edge too thin for 4-leg friction.")

    # 4. Short gamma into expiry is the classic blowup — refuse it.
    if t_days < MIN_DTE_FOR_CREDIT:
        return declined(f"{t_days:.2f} DTE below the {MIN_DTE_FOR_CREDIT:.1f}d floor — short gamma too hot near expiry.")

    # 5. Prefer a genuinely range-bound tape; a trending one runs through a short strike.
    regime = str((regime_state or {}).get("active_regime", "") or "")
    if regime.upper() in ("HIGH_VOL_EXPANSION",):
        return declined(f"Regime {regime} is expansionary — a short-premium corridor is the wrong side of it.")

    # 6. Build the defined-risk structure.
    structure = construct_delta_neutral_iron_condor(
        spot=float(spot), wing_width=150, short_offset=100, t_days=float(t_days), iv=float(iv)
    )
    max_loss_pts = float(structure.get("max_loss_pts", 0.0) or 0.0)
    if max_loss_pts <= 0:
        return declined("Structure returned a non-positive max loss — refusing to size it.")

    corridor = float(structure.get("profit_range_pts", 0.0) or 0.0)
    pop = float(structure.get("probability_of_profit_pct", 0.0) or 0.0)

    # 7. The corridor must clear the expected move with margin, and clear the POP floor.
    if expected_move_pts > 0 and corridor < MIN_CORRIDOR_TO_EXPECTED_MOVE * expected_move_pts:
        return declined(
            f"Corridor {corridor:.0f}pts too narrow vs expected move {expected_move_pts:.0f}pts "
            f"(needs {MIN_CORRIDOR_TO_EXPECTED_MOVE:.2f}x)."
        )
    if pop < MIN_POP_PCT:
        return declined(f"Probability of profit {pop:.0f}% below the {MIN_POP_PCT:.0f}% floor.")

    # 8. Size off DEFINED MAX LOSS — the whole point of a defined-risk structure.
    risk_budget = float(capital) * float(risk_pct)
    risk_per_lot = max_loss_pts * float(lot_size)
    lots = int(risk_budget // risk_per_lot) if risk_per_lot > 0 else 0
    if lots <= 0:
        return declined(
            f"Risk budget Rs.{risk_budget:,.0f} below one lot's defined max loss "
            f"Rs.{risk_per_lot:,.0f} — cannot size responsibly."
        )

    total_credit = float(structure.get("total_net_credit_pts", 0.0)) * lots * lot_size
    total_max_loss = max_loss_pts * lots * lot_size

    return {
        "eligible": True,
        "reason": (
            f"Vol rich (VRP +{vrp_pct:.1f}%, {vol_regime}) with no directional edge and "
            f"{t_days:.1f} DTE — harvest premium via defined-risk condor."
        ),
        "structure": structure,
        "lots": lots,
        "total_qty": lots * lot_size,
        "net_credit_rupees": round(total_credit, 2),
        "max_loss_rupees": round(total_max_loss, 2),
        "risk_budget_rupees": round(risk_budget, 2),
        "vrp_pct": round(vrp_pct, 2),
        "vol_regime": vol_regime,
        "corridor_pts": corridor,
        "probability_of_profit_pct": pop,
        "evidence_tier": "UNMEASURED",  # never promoted off directional stats
    }
