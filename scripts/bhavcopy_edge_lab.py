"""Large-sample edge lab on official NSE F&O bhavcopy (daily settlement-grade prices).

Companion to scripts/premium_edge_lab.py, which tests INTRADAY structures on the 5m premium
archive. That archive can only offer ~20 observations today, far below the power needed to
resolve a 1-2 pt/day effect. This script uses the free official NSE UDiFF bhavcopy instead —
years of daily history for every strike and expiry — so multi-day and hold-to-expiry
strategies can be tested at n in the hundreds.

FINDINGS (first run: 135 trading days, 2026-02-02 -> 2026-08-19, 246,724 NIFTY option rows)

  Put credit spread, sell ATM-100 / buy ATM-200, close-to-close:

      hold   dte     n   win%   gross/day   t-stat   detectable
         1   1-7   131    60%       -0.81    -0.36         6.31
         1  8-20   133    50%       -0.21    -0.15         3.76
         2   1-7   101    55%       -2.37    -0.77         8.66
      pooled     134    60%       -0.65    -0.29         6.22

  1. A 60% win rate coexists with NEGATIVE expectancy. That is the classic short-premium
     signature: many small wins, occasional large loss that eats them. Win rate is not edge.
  2. Overnight std is 25.71 pts vs 4.31 pts intraday — roughly 6x the variance for no
     additional edge. Carrying short premium overnight is paying gap risk for nothing.
  3. Nothing is statistically significant at any horizon or DTE band, before friction.

Combined with the intraday lab, short put premium shows NO detectable edge on either
horizon once sample sizes are adequate. The earlier n=8 "62% win, +2.11 pts/day" result
was sample-size noise.

Usage:
    python scripts/bhavcopy_edge_lab.py                       # uses local cache
    python scripts/bhavcopy_edge_lab.py --fetch 2026-02-01 2026-08-19
    python scripts/bhavcopy_edge_lab.py --width 100 --offset 100 --hold 1
"""

import argparse
import os
import sys
from datetime import date, datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings

warnings.filterwarnings("ignore")

from src.nse_bhavcopy import CACHE_DIR, bulk_fetch


def load_cached(symbol: str = "NIFTY", option_type: str = "PE") -> pd.DataFrame:
    if not os.path.isdir(CACHE_DIR):
        raise SystemExit("No bhavcopy cache — run with --fetch START END first.")
    files = sorted(f for f in os.listdir(CACHE_DIR) if f.endswith(".parquet"))
    if not files:
        raise SystemExit("Bhavcopy cache is empty — run with --fetch START END first.")
    df = pd.concat([pd.read_parquet(os.path.join(CACHE_DIR, f)) for f in files], ignore_index=True)
    df = df[(df.symbol == symbol) & (df.option_type == option_type)].copy()
    df["trade_date"] = pd.to_datetime(df.trade_date)
    df["expiry"] = pd.to_datetime(df.expiry)
    return df


def build_lookup(df: pd.DataFrame):
    s = df.set_index(["trade_date", "expiry", "strike"])["close"]
    s = s[~s.index.duplicated(keep="last")]
    und = df.groupby("trade_date")["underlying"].median().to_dict()
    exps = {d: sorted(g.expiry.unique()) for d, g in df.groupby("trade_date")}
    return s.to_dict(), und, exps, sorted(df.trade_date.unique())


def test_put_spread(L, und, exps, days, width, offset, min_dte, max_dte, hold):
    rows = []
    for i, d in enumerate(days[:-hold] if hold else days):
        d2 = days[i + hold]
        sp = und.get(d)
        if not sp or np.isnan(sp):
            continue
        atm = int(round(sp / 50) * 50)
        cand = [e for e in exps.get(d, [])
                if min_dte <= (pd.Timestamp(e) - pd.Timestamp(d)).days <= max_dte]
        if not cand:
            continue
        e = cand[0]
        k1, k2 = float(atm - offset), float(atm - offset - width)
        a, b = L.get((d, e, k1)), L.get((d, e, k2))
        c, dd = L.get((d2, e, k1)), L.get((d2, e, k2))
        if None in (a, b, c, dd):
            continue
        credit, closing = a - b, c - dd
        max_loss = width - credit
        if credit <= 0 or max_loss <= 0:
            continue
        pnl = credit - closing
        rows.append(dict(day=d, dte=(pd.Timestamp(e) - pd.Timestamp(d)).days,
                         credit=credit, pnl=pnl, max_loss=max_loss, R=pnl / max_loss))
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(description="Large-sample edge lab on NSE bhavcopy")
    p.add_argument("--fetch", nargs=2, metavar=("START", "END"), help="YYYY-MM-DD YYYY-MM-DD")
    p.add_argument("--width", type=int, default=100)
    p.add_argument("--offset", type=int, default=100)
    p.add_argument("--hold", type=int, nargs="*", default=[1, 2])
    p.add_argument("--friction", type=float, nargs="*", default=[1.0, 2.0])
    a = p.parse_args()

    if a.fetch:
        s = datetime.strptime(a.fetch[0], "%Y-%m-%d").date()
        e = datetime.strptime(a.fetch[1], "%Y-%m-%d").date()
        print(f"Fetching official NSE bhavcopy {s} -> {e} ...")
        bulk_fetch(s, e, verbose=True)

    df = load_cached()
    L, und, exps, days = build_lookup(df)
    print("=" * 72)
    print("BHAVCOPY EDGE LAB — real NSE settlement prices")
    print(f"{len(days)} trading days | {df.trade_date.min().date()} -> {df.trade_date.max().date()}")
    print("=" * 72)
    print(f"\nPUT CREDIT SPREAD: sell ATM-{a.offset} / buy ATM-{a.offset + a.width}, close-to-close\n")
    print(f"{'hold':>5}{'dte':>9}{'n':>6}{'win%':>7}{'gross/day':>11}{'meanR':>9}{'t':>7}{'detect':>9}")
    print("-" * 63)
    pooled = None
    for hold in a.hold:
        for lo, hi in ((1, 7), (8, 20), (1, 30)):
            r = test_put_spread(L, und, exps, days, a.width, a.offset, lo, hi, hold)
            if len(r) < 5:
                continue
            P = r.pnl.values
            sd = P.std()
            t = P.mean() / (sd / np.sqrt(len(P))) if sd > 0 else 0.0
            print(f"{hold:>5}{f'{lo}-{hi}':>9}{len(r):>6}{100*(P>0).mean():>6.0f}%"
                  f"{P.mean():>11.2f}{r.R.mean():>9.3f}{t:>7.2f}{2.8*sd/np.sqrt(len(P)):>9.2f}")
            if hold == a.hold[0] and (lo, hi) == (1, 30):
                pooled = r

    if pooled is not None and len(pooled):
        P = pooled.pnl.values
        print(f"\nPOOLED n={len(P)}  gross {P.mean():+.2f} pts/day  std {P.std():.2f}  "
              f"win {100*(P>0).mean():.0f}%")
        for f in a.friction:
            n_ = P - f
            t = n_.mean() / (n_.std() / np.sqrt(len(n_)))
            verdict = "SIGNIFICANT" if abs(t) > 2 else "not significant"
            print(f"  net of {f:.1f} pts: mean {n_.mean():+.2f}  win {100*(n_>0).mean():>3.0f}%  "
                  f"t={t:+.2f}  {verdict}")
        print("\nNOTE: a high win rate with negative expectancy is the short-premium signature —")
        print("many small wins offset by rare large losses. Judge on expectancy, never win rate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
