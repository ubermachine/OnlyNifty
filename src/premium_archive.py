"""Perishable real-premium archive for NIFTY option contracts.

WHY THIS MODULE EXISTS
----------------------
Fyers serves 5-minute OHLCV for *live* option contracts only. Once a weekly
contract expires its symbol leaves the instrument master and history requests
fail outright:

    {'code': -300, 'message': 'Invalid symbol provided'}

Verified empirically on 2026-08-18:
  * live expiry 18-08-2026, strike 24150 CE -> 539 bars across 8 sessions,
    real premium range Rs.4.85 - Rs.440.65
  * expired 11-08-2026 and 28-07-2026 contracts -> code -300, no data

Consequence: **real option premium evidence cannot be backfilled.** Every week
that passes without harvesting is evidence permanently lost, and no later effort
can recover it. Spot-denominated R is not a substitute -- it omits theta, vega
and the spread, which is precisely where an option buyer's money goes.

This module harvests the live contract's own history while it is still
retrievable and persists it locally, so premium-denominated edge measurement
becomes possible going forward and *accumulates* rather than expiring. It is the
prerequisite for any QUOTE-tier record in `src/edge_harness.py`.

Run it at least once per expiry cycle (daily is better):

    python -m src.premium_archive

Storage layout
--------------
    data/premium_archive/index.json                  # contract catalog
    data/premium_archive/<EXPIRY>/<STRIKE><CE|PE>_<RES>.parquet
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pytz

IST = pytz.timezone("Asia/Kolkata")

ARCHIVE_ROOT = os.path.join("data", "premium_archive")
INDEX_FILE = "index.json"

# Fyers/NSE weekly option symbol month codes. Jan-Sep are digits, Oct-Dec are
# letters. Validated against the live chain symbol NSE:NIFTY2681824150CE
# (expiry 18-08-2026, strike 24150, CE).
MONTH_CODE: Dict[int, str] = {
    1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6",
    7: "7", 8: "8", 9: "9", 10: "O", 11: "N", 12: "D",
}
MONTH_ABBR: Dict[int, str] = {
    1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
    7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC",
}


def parse_expiry_label(label: str) -> Optional[datetime]:
    """Parses broker expiry labels such as '18-08-2026' or '18-Aug-2026'."""
    if not label:
        return None
    for fmt in ("%d-%m-%Y", "%d-%b-%Y", "%Y-%m-%d", "%d-%B-%Y"):
        try:
            return datetime.strptime(str(label).strip(), fmt)
        except ValueError:
            continue
    try:
        return pd.to_datetime(label).to_pydatetime()
    except Exception:
        return None


def build_weekly_symbol(expiry: Any, strike: float, option_type: str) -> str:
    """Constructs a Fyers weekly option symbol: NSE:NIFTY<YY><M><DD><STRIKE><CE|PE>."""
    dt = parse_expiry_label(expiry) if not hasattr(expiry, "year") else expiry
    if dt is None:
        raise ValueError(f"Unparseable expiry: {expiry!r}")
    yy = str(dt.year)[2:]
    return f"NSE:NIFTY{yy}{MONTH_CODE[dt.month]}{dt.day:02d}{int(strike)}{option_type.upper()}"


def build_monthly_symbol(expiry: Any, strike: float, option_type: str) -> str:
    """Constructs a Fyers monthly option symbol: NSE:NIFTY<YY><MMM><STRIKE><CE|PE>."""
    dt = parse_expiry_label(expiry) if not hasattr(expiry, "year") else expiry
    if dt is None:
        raise ValueError(f"Unparseable expiry: {expiry!r}")
    yy = str(dt.year)[2:]
    return f"NSE:NIFTY{yy}{MONTH_ABBR[dt.month]}{int(strike)}{option_type.upper()}"


def expiry_key(expiry: Any) -> str:
    """Normalizes an expiry label into a filesystem-safe key (YYYY-MM-DD)."""
    dt = parse_expiry_label(expiry) if not hasattr(expiry, "year") else expiry
    if dt is None:
        return str(expiry).replace("/", "-").replace(":", "-")
    return dt.strftime("%Y-%m-%d")


class PremiumArchive:
    """Incremental, idempotent store of real option premium candles."""

    def __init__(self, root: str = ARCHIVE_ROOT):
        self.root = root
        os.makedirs(self.root, exist_ok=True)
        self._index_path = os.path.join(self.root, INDEX_FILE)
        self.index: Dict[str, Dict[str, Any]] = self._load_index()

    # ------------------------------------------------------------------ index

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        if os.path.exists(self._index_path):
            try:
                with open(self._index_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_index(self) -> None:
        tmp = f"{self._index_path}.tmp.{os.getpid()}"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.index, f, indent=2, sort_keys=True)
            os.replace(tmp, self._index_path)
        except Exception:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    # ------------------------------------------------------------------ paths

    def contract_key(self, expiry: Any, strike: float, option_type: str, resolution: str = "5") -> str:
        return f"{expiry_key(expiry)}|{int(strike)}{option_type.upper()}|{resolution}"

    def contract_path(self, expiry: Any, strike: float, option_type: str, resolution: str = "5") -> str:
        d = os.path.join(self.root, expiry_key(expiry))
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{int(strike)}{option_type.upper()}_{resolution}.parquet")

    # ------------------------------------------------------------------- write

    def upsert(
        self,
        expiry: Any,
        strike: float,
        option_type: str,
        df: pd.DataFrame,
        symbol: str = "",
        resolution: str = "5",
    ) -> int:
        """Merges new candles into the contract's store. Returns rows added."""
        if df is None or df.empty:
            return 0
        path = self.contract_path(expiry, strike, option_type, resolution)
        incoming = df.copy()
        if not isinstance(incoming.index, pd.DatetimeIndex):
            return 0
        if incoming.index.tz is None:
            incoming.index = incoming.index.tz_localize(IST)
        else:
            incoming.index = incoming.index.tz_convert(IST)

        before = 0
        if os.path.exists(path):
            try:
                existing = pd.read_parquet(path)
                before = len(existing)
                incoming = pd.concat([existing, incoming])
            except Exception:
                pass

        incoming = incoming[~incoming.index.duplicated(keep="last")].sort_index()
        incoming.to_parquet(path)

        added = len(incoming) - before
        self.index[self.contract_key(expiry, strike, option_type, resolution)] = {
            "symbol": symbol,
            "expiry": expiry_key(expiry),
            "strike": int(strike),
            "option_type": option_type.upper(),
            "resolution": resolution,
            "rows": int(len(incoming)),
            "first_bar": str(incoming.index[0]),
            "last_bar": str(incoming.index[-1]),
            "path": path,
            "updated_utc": datetime.utcnow().isoformat(timespec="seconds"),
        }
        self._save_index()
        return max(added, 0)

    # -------------------------------------------------------------- harvesting

    def harvest_from_chain(
        self,
        chain: Dict[str, Any],
        resolution: str = "5",
        days_back: int = 8,
        strike_window: int = 10,
        which: Tuple[str, ...] = ("near_chain",),
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Harvests real premium candles for strikes around ATM using the chain's
        own authoritative symbols (avoids all symbol-construction risk).

        `chain` is the dict returned by `fyers_client.get_multi_expiry_chain()`.
        """
        from src import fyers_client as fc

        spot = float(chain.get("underlying_value") or 0.0)
        summary: Dict[str, Any] = {"contracts": 0, "rows_added": 0, "failures": [], "expiries": []}

        to_date = datetime.now(IST).date()
        from_date = to_date - timedelta(days=max(days_back, 1))

        for which_key in which:
            cdf = chain.get(which_key)
            if cdf is None or not isinstance(cdf, pd.DataFrame) or cdf.empty:
                continue
            expiry_label = (
                chain.get("near_expiry") if which_key == "near_chain"
                else chain.get("next_expiry") if which_key == "next_chain"
                else chain.get("monthly_expiry")
            )
            summary["expiries"].append(expiry_label)

            if spot > 0 and "strike" in cdf.columns:
                order = (cdf["strike"] - spot).abs().argsort()
                cdf = cdf.iloc[order[: max(strike_window * 2, 2)]]

            for _, row in cdf.iterrows():
                strike = float(row.get("strike", 0.0))
                if strike <= 0:
                    continue
                for opt in ("CE", "PE"):
                    sym = str(row.get(f"{opt.lower()}_symbol", "") or "")
                    if not sym:
                        try:
                            sym = build_weekly_symbol(expiry_label, strike, opt)
                        except Exception:
                            continue
                    try:
                        hist = fc.get_history(
                            sym, resolution, from_date.isoformat(), to_date.isoformat()
                        )
                    except Exception as ex:
                        summary["failures"].append({"symbol": sym, "error": str(ex)[:160]})
                        continue
                    added = self.upsert(
                        expiry_label, strike, opt, hist, symbol=sym, resolution=resolution
                    )
                    summary["contracts"] += 1
                    summary["rows_added"] += added
                    if verbose and added:
                        print(f"  archived {sym:32s} +{added:5d} rows (total {len(hist)})")

        return summary

    # ----------------------------------------------------------------- reading

    def load_series(
        self,
        expiry: Any,
        strike: float,
        option_type: str,
        resolution: str = "5",
    ) -> pd.DataFrame:
        """Loads all archived candles for one contract. Empty frame if absent."""
        path = self.contract_path(expiry, strike, option_type, resolution)
        if not os.path.exists(path):
            return pd.DataFrame()
        try:
            df = pd.read_parquet(path)
            if isinstance(df.index, pd.DatetimeIndex):
                if df.index.tz is None:
                    df.index = df.index.tz_localize(IST)
                else:
                    df.index = df.index.tz_convert(IST)
            return df
        except Exception:
            return pd.DataFrame()

    def load_window(
        self,
        expiry: Any,
        strike: float,
        option_type: str,
        start_ts: Any,
        bars: int = 12,
        resolution: str = "5",
    ) -> pd.DataFrame:
        """Returns up to `bars` archived candles strictly AFTER `start_ts`.

        This is the forward window an outcome replay consumes; returning only
        strictly-later bars keeps the replay free of the entry bar itself.
        """
        df = self.load_series(expiry, strike, option_type, resolution)
        if df.empty:
            return df
        ts = pd.Timestamp(start_ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize(IST)
        else:
            ts = ts.tz_convert(IST)
        fwd = df[df.index > ts]
        return fwd.iloc[:bars] if bars and bars > 0 else fwd

    def has_coverage(
        self,
        expiry: Any,
        strike: float,
        option_type: str,
        start_ts: Any,
        bars: int = 12,
        resolution: str = "5",
    ) -> bool:
        """True when the archive can supply a usable forward window."""
        w = self.load_window(expiry, strike, option_type, start_ts, bars, resolution)
        return len(w) >= max(min(bars, 3), 1)

    def coverage(self) -> pd.DataFrame:
        """Tabular view of everything archived, for provenance reporting."""
        if not self.index:
            return pd.DataFrame(
                columns=["expiry", "strike", "option_type", "resolution", "rows", "first_bar", "last_bar"]
            )
        rows = [
            {
                "expiry": v.get("expiry"),
                "strike": v.get("strike"),
                "option_type": v.get("option_type"),
                "resolution": v.get("resolution"),
                "rows": v.get("rows"),
                "first_bar": v.get("first_bar"),
                "last_bar": v.get("last_bar"),
                "symbol": v.get("symbol"),
            }
            for v in self.index.values()
        ]
        return pd.DataFrame(rows).sort_values(["expiry", "strike", "option_type"]).reset_index(drop=True)


def harvest_now(
    resolution: str = "5",
    days_back: int = 8,
    strike_window: int = 10,
    include_next: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Convenience entry point: fetch the live chain and archive around ATM."""
    from src import fyers_client as fc

    chain = fc.get_multi_expiry_chain()
    which: Tuple[str, ...] = ("near_chain", "next_chain") if include_next else ("near_chain",)
    arch = PremiumArchive()
    if verbose:
        print(f"Harvesting premium archive | spot={chain.get('underlying_value')} "
              f"| near_expiry={chain.get('near_expiry')} | window=+/-{strike_window} strikes")
    result = arch.harvest_from_chain(
        chain, resolution=resolution, days_back=days_back,
        strike_window=strike_window, which=which, verbose=verbose,
    )
    if verbose:
        print(f"\nDone: {result['contracts']} contracts touched, "
              f"{result['rows_added']} new rows, {len(result['failures'])} failures")
        cov = arch.coverage()
        if not cov.empty:
            print(f"Archive now holds {len(cov)} contracts / {int(cov['rows'].sum())} candles")
    return result


if __name__ == "__main__":
    harvest_now()
