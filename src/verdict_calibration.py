"""Calibrates the desk verdict's asserted score against realized outcomes.

WHY
The confluence score is a hand-weighted heuristic. Every weight in it was asserted, none was
ever fitted to outcomes. Measured over an out-of-sample walk it carries no information:

    all entries (overlapping)     n=282  r(score, R) = -0.1238  t=-2.09  "significant"
    non-overlapping (>=12 bars)   n= 50  r(score, R) = -0.0047  t=-0.03  flat

The overlapping run looked anti-predictive; that was shared forward windows, the same
artifact that inflated the condor result. Independent sampling gives r = -0.005 — zero.
Bucketed: <50 -> -0.093R, 50-55 -> -0.077R, 55+ -> -0.101R. Perfectly flat.

Caveat that shapes this module: that walk never scored above 57, while live signals reach
78-100. The high-conviction band has never been measured at all. So the correct behaviour is
not "discard the score" — it is "report what the score has actually been worth, and say
UNCALIBRATED where nothing is known."

WHAT THIS DOES
  1. calibrate(score) -> the realized win rate / mean R for that score band, with an explicit
     CALIBRATED / SPARSE / UNCALIBRATED status. A band with no outcomes returns UNCALIBRATED
     rather than an implied endorsement.
  2. family_attribution() -> per-pillar (structure / flow / positioning / macro) hit rates,
     so the four families can eventually be weighted by what they are worth instead of
     equally by assertion.

Nothing here fabricates. Absent evidence is reported as absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# A band needs this many resolved trades before its number means anything.
CALIBRATED_MIN_N = 30
SPARSE_MIN_N = 10

# Sanity bound on a realized R multiple. A 3-tier structure with a breakeven trail cannot
# produce |R| anywhere near this; values beyond it are corrupt, not extreme.
# This is not hypothetical: the historical journal contains rows at -256.55R and -88.46R,
# produced by the zero-lot bug where realized_r_multiple was raw rupees divided by a
# capital_risk_rupees of 0.0 (floored to 1.0). Ingesting those would poison the exact
# numbers this module exists to make honest, so they are rejected and counted.
R_SANITY_BOUND = 10.0

# Score bands. The top band is deliberately isolated: it is where live signals live and
# where we currently have no measurements at all.
BANDS: Sequence[Tuple[float, float, str]] = (
    (0.0, 45.0, "<45"),
    (45.0, 55.0, "45-55"),
    (55.0, 70.0, "55-70"),
    (70.0, 85.0, "70-85"),
    (85.0, 100.1, "85+"),
)

FAMILIES = ("structure", "flow", "positioning", "macro")


@dataclass
class Calibration:
    band: str
    n: int
    win_rate: float
    mean_r: float
    status: str          # "CALIBRATED" | "SPARSE" | "UNCALIBRATED"
    note: str

    @property
    def is_actionable(self) -> bool:
        return self.status == "CALIBRATED"


def band_for(score: float) -> str:
    for lo, hi, label in BANDS:
        if lo <= float(score) < hi:
            return label
    return BANDS[-1][2]


class VerdictCalibrator:
    """Maps an asserted confluence score to what that score has historically been worth."""

    def __init__(self, outcomes: Optional[Iterable[Tuple[float, float]]] = None):
        """outcomes: iterable of (confluence_score, realized_R) for CLOSED trades only."""
        self._by_band: Dict[str, List[float]] = {label: [] for _, _, label in BANDS}
        self.rejected_corrupt: int = 0
        for score, r in (outcomes or []):
            try:
                rv = float(r)
            except Exception:
                continue
            if not np.isfinite(rv) or abs(rv) > R_SANITY_BOUND:
                self.rejected_corrupt += 1   # see R_SANITY_BOUND: corrupt, not extreme
                continue
            try:
                self._by_band[band_for(float(score))].append(rv)
            except Exception:
                continue

    @classmethod
    def from_entries(cls, entries: Iterable[Any]) -> "VerdictCalibrator":
        """Builds from journal entries; counts only closed, actionable trades."""
        pairs: List[Tuple[float, float]] = []
        for e in entries:
            get = (lambda k, d=None: e.get(k, d)) if isinstance(e, dict) else (lambda k, d=None: getattr(e, k, d))
            if str(get("direction", "WAIT")) not in ("LONG", "SHORT"):
                continue
            status = str(get("lifecycle_status", ""))
            if status in ("TRIGGERED", "ACTIVE", "T1_REACHED", "T2_REACHED", "AWAITING_SETUP"):
                continue  # still open — outcome not yet known
            score = get("confluence_score", None)
            r = get("realized_r_multiple", None)
            if score is None or r is None:
                continue
            pairs.append((float(score), float(r)))
        return cls(pairs)

    def calibrate(self, score: float) -> Calibration:
        label = band_for(score)
        rs = self._by_band.get(label, [])
        n = len(rs)
        if n == 0:
            return Calibration(label, 0, 0.0, 0.0, "UNCALIBRATED",
                               f"No resolved trades in the {label} band — this score has never been measured.")
        arr = np.array(rs, dtype=float)
        win = float((arr > 0).mean() * 100.0)
        mean_r = float(arr.mean())
        if n >= CALIBRATED_MIN_N:
            status, note = "CALIBRATED", f"{mean_r:+.2f}R realized over {n} trades in {label}."
        elif n >= SPARSE_MIN_N:
            status, note = "SPARSE", f"{mean_r:+.2f}R over only {n} trades in {label} — indicative, not established."
        else:
            status, note = "UNCALIBRATED", f"Only {n} trades in {label} — far too few to mean anything."
        return Calibration(label, n, round(win, 1), round(mean_r, 3), status, note)

    def table(self) -> List[Calibration]:
        return [self.calibrate((lo + hi) / 2.0) for lo, hi, _ in BANDS]

    def coverage_summary(self) -> str:
        parts = [f"{c.band}:n={c.n}" for c in self.table()]
        s = " | ".join(parts)
        if self.rejected_corrupt:
            s += f"  [{self.rejected_corrupt} corrupt rows rejected]"
        return s


def family_attribution(entries: Iterable[Any]) -> Dict[str, Dict[str, Any]]:
    """Per-pillar hit rate: when a family voted WITH the trade's direction, did it pay?

    The four families are currently weighted equally by assertion. This measures what each
    is actually worth, so weighting can eventually be earned rather than assumed.
    """
    agg: Dict[str, Dict[str, List[float]]] = {f: {"with": [], "against": []} for f in FAMILIES}
    for e in entries:
        get = (lambda k, d=None: e.get(k, d)) if isinstance(e, dict) else (lambda k, d=None: getattr(e, k, d))
        direction = str(get("direction", "WAIT"))
        if direction not in ("LONG", "SHORT"):
            continue
        status = str(get("lifecycle_status", ""))
        if status in ("TRIGGERED", "ACTIVE", "T1_REACHED", "T2_REACHED", "AWAITING_SETUP"):
            continue
        r = get("realized_r_multiple", None)
        votes = get("family_votes", None) or {}
        if r is None or not isinstance(votes, dict) or not votes:
            continue
        sign = 1 if direction == "LONG" else -1
        for fam in FAMILIES:
            if fam not in votes:
                continue
            try:
                v = float(votes[fam])
            except Exception:
                continue
            if v == 0:
                continue
            bucket = "with" if (v > 0) == (sign > 0) else "against"
            agg[fam][bucket].append(float(r))

    out: Dict[str, Dict[str, Any]] = {}
    for fam, d in agg.items():
        w, a = np.array(d["with"] or [], float), np.array(d["against"] or [], float)
        out[fam] = {
            "n_with": len(w),
            "n_against": len(a),
            "mean_r_when_with": round(float(w.mean()), 3) if len(w) else None,
            "mean_r_when_against": round(float(a.mean()), 3) if len(a) else None,
            "edge": (round(float(w.mean() - a.mean()), 3) if len(w) and len(a) else None),
            "status": "CALIBRATED" if min(len(w), len(a)) >= CALIBRATED_MIN_N
                      else ("SPARSE" if min(len(w), len(a)) >= SPARSE_MIN_N else "UNCALIBRATED"),
        }
    return out
