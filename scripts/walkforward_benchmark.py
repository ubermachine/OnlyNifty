"""Walk-forward out-of-sample edge benchmark.

Measures each (setup_id, regime) pair's real OOS edge and writes data/edge_table.json,
which StrategyEngine loads at startup to quarantine negative-EV setups.

Run this OFFLINE, never inside the Streamlit app: a full run takes ~15 minutes and the
app's 60s autorefresh would tear it down mid-flight. The app only ever reads the
resulting JSON.

Usage (PowerShell):
    python scripts/walkforward_benchmark.py
    python scripts/walkforward_benchmark.py --period 60d --train-days 30 --test-days 5
    python scripts/walkforward_benchmark.py --dry-run     # report only, don't write
"""

import argparse
import os
import sys
import time
import warnings
from typing import Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")

from src.data_engine import DataEngine
from src.edge_harness import EdgeTable, WalkForwardRunner
from src.strategy_rules import StrategyEngine
from src.config import QUARANTINE_MIN_SAMPLES

BARS_PER_DAY = 75
DEFAULT_OUT = os.path.join("data", "edge_table.json")


def load_history(period: str, interval: str) -> pd.DataFrame:
    engine = DataEngine()
    df = engine.fetch_yfinance_nifty(interval=interval, period=period, max_cache_age_seconds=0)
    if df is None or df.empty:
        raise SystemExit(f"No data returned for period={period} interval={interval}.")
    return df


def estimate_windows(total_bars: int, train_days: int, test_days: int, purge: int, embargo: int):
    train_bars = train_days * BARS_PER_DAY
    test_bars = test_days * BARS_PER_DAY
    windows, evals, start = 0, 0, train_bars
    while start + test_bars <= total_bars:
        span = (start + test_bars - embargo) - (start + purge)
        if span > 0:
            windows += 1
            evals += span
        start += test_bars
    return windows, evals


def main() -> int:
    p = argparse.ArgumentParser(description="Walk-forward OOS edge benchmark")
    p.add_argument("--period", default="60d", help="yfinance lookback (5m intraday caps at 60d)")
    p.add_argument("--interval", default="5m")
    p.add_argument("--train-days", type=int, default=30)
    p.add_argument("--test-days", type=int, default=5)
    p.add_argument("--purge-bars", type=int, default=60)
    p.add_argument("--embargo-bars", type=int, default=12)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--dry-run", action="store_true", help="print results without writing the table")
    args = p.parse_args()

    print("=" * 78)
    print("WALK-FORWARD OUT-OF-SAMPLE EDGE BENCHMARK")
    print("=" * 78)

    print(f"\nFetching {args.period} of {args.interval} history...")
    df = load_history(args.period, args.interval)
    print(f"  {len(df)} bars | {df.index.min()} -> {df.index.max()}")

    windows, evals = estimate_windows(
        len(df), args.train_days, args.test_days, args.purge_bars, args.embargo_bars
    )
    if windows == 0:
        print(
            f"\nNot enough history: need > {args.train_days * BARS_PER_DAY + args.test_days * BARS_PER_DAY} "
            f"bars for even one window, have {len(df)}."
        )
        return 1

    print(
        f"\nPlan: {windows} window(s), ~{evals} bar evaluations "
        f"({args.train_days}d train / {args.test_days}d test, "
        f"purge {args.purge_bars}, embargo {args.embargo_bars})"
    )
    print(f"  Estimated runtime: ~{evals * 0.6 / 60:.0f} min\n")

    runner = WalkForwardRunner(strategy_engine=StrategyEngine(edge_table=EdgeTable()))
    t0 = time.time()
    table = runner.run(
        df,
        train_days=args.train_days,
        test_days=args.test_days,
        purge_bars=args.purge_bars,
        embargo_bars=args.embargo_bars,
    )
    elapsed = time.time() - t0

    records = sorted(table.records.values(), key=lambda r: (r.status, -r.n))
    print(f"Completed in {elapsed / 60:.1f} min. {len(records)} (setup, regime) bucket(s) measured.\n")

    if not records:
        print("No trades fired across the OOS windows — nothing to measure.")
        print("The gates may be too strict for this sample, or history is too short.")
        return 0

    print(f"{'SETUP':<26} {'REGIME':<22} {'N':>4} {'WIN%':>6} {'EV(R)':>7} {'CI_LOW':>7}  STATUS")
    print("-" * 90)
    for r in records:
        print(
            f"{r.setup_id:<26} {r.regime:<22} {r.n:>4} {r.win_rate:>6.1f} "
            f"{r.ev:>7.2f} {r.ci_low:>7.2f}  {r.status}"
        )

    total = sum(r.n for r in records)
    trusted = [r for r in records if r.status == "TRUSTED"]
    quarantined = [r for r in records if r.status == "QUARANTINED"]
    paper = [r for r in records if r.status == "PAPER"]

    print("-" * 90)
    print(f"\n{total} OOS trades | TRUSTED {len(trusted)} | PAPER {len(paper)} | QUARANTINED {len(quarantined)}")

    if quarantined:
        print("\nQuarantined (blocked live until they prove positive OOS edge):")
        for r in quarantined:
            print(f"  - {r.setup_id} in {r.regime}: EV {r.ev:+.2f}R over {r.n} trades")

    if paper and not trusted:
        print(
            f"\nNOTE: every bucket is PAPER (n < {QUARANTINE_MIN_SAMPLES}). These numbers are a "
            f"plumbing check, not a verdict — treat them as inconclusive until you have "
            f"deeper history."
        )

    if args.dry_run:
        print("\n--dry-run: table NOT written.")
        return 0

    table.save_to_disk(args.out)
    print(f"\nWrote {args.out} — StrategyEngine picks this up on next start.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
