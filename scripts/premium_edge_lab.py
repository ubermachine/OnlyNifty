"""Premium-structure edge lab — tests short-premium structures on REAL archived option prices.

Why this exists
---------------
Edge claims in this repo have historically been made in spot points, on model prices, or on
samples far too small to support them. This script tests a structure against actual archived
option premium (data/premium_archive), reports results GROSS and NET of bid/ask friction,
and — critically — reports the smallest edge the current sample could even detect.

Findings on first run (2026-08-19, 2 expiries, real premium):
  * Centered 4-leg iron condor, intraday: 88% win, +1.59 pts/day GROSS — but 8 bid/ask
    crossings wipe it out. Negative at >=2 pts total friction.
  * Decomposed: the entire gross edge sat on the PUT vertical; the CALL vertical lost money.
  * Put vertical alone at n=8 looked strong (62% win, +2.11 pts/day)...
  * ...and collapsed to 41% win / +0.00 pts/day when the sample was expanded to n=17.
    The n=8 result was sample-size noise.

Power: measured daily P&L std is ~3.6 pts. At n=17 the smallest detectable edge is ~2.4
pts/day, while the effects being chased are 1-2 pts/day — i.e. structurally invisible at
this sample size. Roughly 25 days are needed for a 2 pt/day edge, ~100 for 1 pt/day.

Treat every result here as inconclusive until n clears the power bar printed at the end.

Usage:
    python scripts/premium_edge_lab.py
    python scripts/premium_edge_lab.py --structure put_spread --width 100
    python scripts/premium_edge_lab.py --entry 09:20 --exit 15:20 --friction 1.0
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from src.premium_archive import PremiumArchive
from src.data_engine import DataEngine

STRUCTURES = ("condor", "put_spread", "call_spread")


def _build(arch, cov):
    cache = {}

    def ser(expiry, strike, opt):
        key = (expiry, strike, opt)
        if key not in cache:
            s = arch.load_series(expiry, strike, opt)
            cache[key] = s[~s.index.duplicated(keep="last")] if not s.empty else s
        return cache[key]

    def px(expiry, strike, opt, day, hm):
        s = ser(expiry, strike, opt)
        if s.empty:
            return None
        d = s[(s.index.strftime("%Y-%m-%d") == day) & (s.index.strftime("%H:%M") <= hm)]
        return float(d["close"].iloc[-1]) if len(d) else None

    return px


def run(structure: str, width: int, offset: int, entry: str, exit_: str,
        min_dte: int, max_dte: int) -> pd.DataFrame:
    arch = PremiumArchive()
    cov = arch.coverage()
    if cov.empty:
        raise SystemExit("Premium archive is empty — run src/premium_archive.py first.")
    px = _build(arch, cov)

    spot = DataEngine(use_cache=False).fetch_yfinance_nifty(
        interval="5m", period="60d", max_cache_age_seconds=0
    )
    days = sorted(set(spot.index.strftime("%Y-%m-%d")))
    rows = []

    for expiry in sorted(set(cov.expiry)):
        strikes = sorted(set(int(s) for s in cov[cov.expiry == expiry].strike))
        exp_ts = pd.Timestamp(expiry)
        for day in days:
            dte = (exp_ts - pd.Timestamp(day)).days
            if not (min_dte <= dte <= max_dte):
                continue
            d = spot[spot.index.strftime("%Y-%m-%d") == day]
            d9 = d[d.index.strftime("%H:%M") <= entry]
            if len(d9) == 0:
                continue
            sp = float(d9["close"].iloc[-1])
            atm = int(round(sp / 50) * 50)

            if structure == "put_spread":
                legs = [(atm - offset, "PE", +1), (atm - offset - width, "PE", -1)]
            elif structure == "call_spread":
                legs = [(atm + offset, "CE", +1), (atm + offset + width, "CE", -1)]
            else:
                legs = [(atm - offset, "PE", +1), (atm + offset, "CE", +1),
                        (atm - offset - width, "PE", -1), (atm + offset + width, "CE", -1)]

            if not all(k in strikes for k, _, _ in legs):
                continue
            o = [px(expiry, k, t, day, entry) for k, t, _ in legs]
            c = [px(expiry, k, t, day, exit_) for k, t, _ in legs]
            if any(v is None for v in o + c):
                continue
            sg = [s for _, _, s in legs]
            credit = sum(s * v for s, v in zip(sg, o))
            closing = sum(s * v for s, v in zip(sg, c))
            max_loss = width - credit
            if credit <= 0 or max_loss <= 0:
                continue
            pnl = credit - closing
            rows.append(dict(expiry=expiry, day=day, dte=dte, atm=atm,
                             credit=round(credit, 2), pnl=round(pnl, 2),
                             max_loss=round(max_loss, 2), R=pnl / max_loss))
    return pd.DataFrame(rows)


def report(df: pd.DataFrame, structure: str, frictions):
    if df.empty:
        print("No observations — archive lacks the needed strikes for this structure.")
        return
    P, R = df.pnl.values, df.R.values
    legs = 2 if structure != "condor" else 4
    print(f"\n{structure.upper()} | n={len(df)} observations | {legs} legs = {legs*2} bid/ask crossings")
    print("-" * 78)
    print(f"GROSS: win {100*(P>0).mean():.0f}%  total {P.sum():+.1f} pts  "
          f"mean {P.mean():+.2f} pts/day  meanR {R.mean():+.3f}  std {P.std():.2f}")
    for f in frictions:
        n_ = P - f
        se = n_.std() / np.sqrt(len(n_)) if len(n_) > 1 and n_.std() > 0 else float("inf")
        t = n_.mean() / se if se not in (0, float("inf")) else 0.0
        print(f"  net of {f:>4.1f} pts: mean {n_.mean():+.2f}/day  win {100*(n_>0).mean():>3.0f}%  "
              f"total {n_.sum():+7.1f}  t={t:+.2f}")

    sd = P.std()
    print(f"\nPOWER (measured std {sd:.2f} pts/day):")
    for mu in (0.5, 1.0, 2.0, 3.0):
        n_need = (2.8 * sd / mu) ** 2
        print(f"  detect {mu:>4.1f} pts/day edge -> need ~{n_need:>5.0f} days (~{n_need/21:.1f} months)")
    detectable = 2.8 * sd / np.sqrt(len(df))
    print(f"  smallest edge detectable at n={len(df)}: {detectable:.2f} pts/day")
    if detectable > abs(P.mean()):
        print("  => VERDICT: sample too small to distinguish this result from zero. Inconclusive.")


def main() -> int:
    p = argparse.ArgumentParser(description="Short-premium edge lab on real archived option prices")
    p.add_argument("--structure", default="condor", choices=STRUCTURES)
    p.add_argument("--width", type=int, default=100, help="wing width in points")
    p.add_argument("--offset", type=int, default=100, help="short strike distance from ATM")
    p.add_argument("--entry", default="09:20")
    p.add_argument("--exit", dest="exit_", default="15:20")
    p.add_argument("--min-dte", type=int, default=1)
    p.add_argument("--max-dte", type=int, default=21)
    p.add_argument("--friction", type=float, nargs="*", default=[0.5, 1.0, 2.0])
    a = p.parse_args()

    print("=" * 78)
    print("PREMIUM STRUCTURE EDGE LAB — real archived option premium")
    print("=" * 78)
    df = run(a.structure, a.width, a.offset, a.entry, a.exit_, a.min_dte, a.max_dte)
    report(df, a.structure, a.friction)
    if not df.empty:
        print("\nBy DTE bucket (gross):")
        df["bucket"] = pd.cut(df.dte, [0, 3, 7, 14, 21], labels=["1-3d", "4-7d", "8-14d", "15-21d"])
        print(df.groupby("bucket", observed=True).agg(
            n=("pnl", "size"), win=("pnl", lambda x: f"{100*(x>0).mean():.0f}%"),
            mean_pts=("pnl", "mean"), meanR=("R", "mean")).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
