"""Measured signal conviction — grounds signal quality in realized premium edge.

Retail systems grade a signal by how many indicators agree (a heuristic confluence count).
This layer instead grades a setup by its *realized* option-premium expectancy, measured by
the perishable harvester at evidence_tier="QUOTE", falling back to weaker tiers only when no
quote evidence exists yet — and never letting a weak tier masquerade as strong.

Evidence precedence:  QUOTE  >  MODEL  >  SPOT  >  (unmeasured)

A setup is graded A+/A/B/C and given a conviction multiplier in [0, 1.25] from realized EV
and the one-sided lower confidence bound, so sizing scales with *demonstrated* edge, not the
number of green lights on the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from src.edge_harness import EdgeTable, EdgeStats

_TIER_RANK = {"QUOTE": 3, "MODEL": 2, "SPOT": 1, "": 0}


@dataclass
class SignalConviction:
    setup_id: str
    regime: str
    grade: str                 # "A+", "A", "B", "C", "UNMEASURED"
    conviction: float          # sizing multiplier, [0.0, 1.25]
    evidence_tier: str         # tier of the record that decided the grade
    realized_ev: float         # realized option R (0.0 when unmeasured)
    n: int
    rationale: str


class SignalEdgeWeighter:
    """Merges every available edge source, QUOTE-first, into one measured conviction."""

    def __init__(self, *tables: EdgeTable):
        # Highest-tier record wins per (setup, regime); ties broken by larger sample.
        self.best: Dict[Tuple[str, str], EdgeStats] = {}
        for tbl in tables:
            if tbl is None:
                continue
            for key, rec in tbl.records.items():
                cur = self.best.get(key)
                if cur is None or self._supersedes(rec, cur):
                    self.best[key] = rec

    @staticmethod
    def _supersedes(new: EdgeStats, cur: EdgeStats) -> bool:
        nr = _TIER_RANK.get(getattr(new, "evidence_tier", ""), 0)
        cr = _TIER_RANK.get(getattr(cur, "evidence_tier", ""), 0)
        if nr != cr:
            return nr > cr
        return new.n > cur.n

    def conviction_for(self, setup_id: str, regime: str) -> SignalConviction:
        rec = self.best.get((setup_id, regime))
        if rec is None:
            return SignalConviction(
                setup_id, regime, "UNMEASURED", 0.5, "", 0.0, 0,
                "No edge evidence yet — capped at half size (PAPER exploration).",
            )

        tier = getattr(rec, "evidence_tier", "")
        ev, ci_low, n = rec.ev, rec.ci_low, rec.n

        # Negative realized expectancy on an adequate sample -> no conviction.
        if rec.status == "QUARANTINED" or (n >= 20 and ev <= 0.0):
            return SignalConviction(
                setup_id, regime, "C", 0.0, tier, ev, n,
                f"Quarantined: realized EV {ev:+.2f}R over {n} trades ({tier}).",
            )

        # Only genuine QUOTE evidence can earn the top grades; weaker tiers are capped.
        quote = tier == "QUOTE"
        if quote and n >= 30 and ci_low > 0.0 and ev >= 0.30:
            grade, mult = "A+", min(1.25, 1.0 + ev * 0.25)
        elif quote and n >= 20 and ev > 0.0:
            grade, mult = "A", 1.0
        elif ev > 0.0 and n >= 10:
            grade, mult = "B", 0.75
        else:
            grade, mult = "B", 0.6 if ev > 0 else 0.4

        # Non-quote evidence never sizes above 0.75, however good it looks — it is unproven
        # in the instrument actually traded.
        if not quote:
            mult = min(mult, 0.75)
            if grade in ("A+", "A"):
                grade = "B"

        return SignalConviction(
            setup_id, regime, grade, round(mult, 3), tier, ev, n,
            f"{grade}: realized EV {ev:+.2f}R (CI_low {ci_low:+.2f}) over {n} {tier} trades.",
        )

    def grade_label(self, setup_id: str, regime: str) -> str:
        c = self.conviction_for(setup_id, regime)
        return f"{c.grade} · {c.realized_ev:+.2f}R ×{c.n} [{c.evidence_tier or 'none'}]"
