"""Official NSE F&O bhavcopy (UDiFF) — free, open historical option data.

WHY THIS MODULE EXISTS
The repo's existing DataEngine.download_fno_bhavcopy() calls jugaad_data.bhavcopy_fo_save(),
which is broken against current NSE infrastructure (the old /content/historical/DERIVATIVES
path is gone; the call dies with BadZipFile). On failure it silently returned a HARDCODED
synthetic frame — every strike carrying identical OHLC 150/200/100/145, identical volume
15000, identical OI 850000 — with no provenance flag. Research built on that would be
fabricated while looking entirely real.

This module fetches the CURRENT official source:
    https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_<YYYYMMDD>_F_0000.csv.zip
verified working: 1.15 MB/day, 35,433 rows, 1,726 NIFTY option rows, strikes 12000-34500
across 8 expiries, with real per-strike OHLC / settlement / OI / volume.

It NEVER synthesises. A holiday, a missing file or a network failure returns an EMPTY frame
so the caller can tell "no data" from "data" — the distinction the old fallback destroyed.

Granularity is daily (settlement-grade). Intraday work still needs the 5m premium archive
(src/premium_archive.py); this is the source for multi-day and hold-to-expiry studies, where
it supplies years of history instead of the ~20 observations the 5m archive can offer today.
"""

from __future__ import annotations

import io
import os
import zipfile
from datetime import date, timedelta
from typing import List, Optional

import pandas as pd
import requests

BASE_URL = "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{ds}_F_0000.csv.zip"
CACHE_DIR = os.path.join("data", "bhavcopy")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

# Normalised subset kept on disk: NIFTY options only (1,726 of 35,433 rows/day).
COLUMNS = {
    "TradDt": "trade_date",
    "XpryDt": "expiry",
    "StrkPric": "strike",
    "OptnTp": "option_type",
    "OpnPric": "open",
    "HghPric": "high",
    "LwPric": "low",
    "ClsPric": "close",
    "SttlmPric": "settle",
    "UndrlygPric": "underlying",
    "OpnIntrst": "oi",
    "ChngInOpnIntrst": "chg_oi",
    "TtlTradgVol": "volume",
}


def _cache_path(d: date) -> str:
    return os.path.join(CACHE_DIR, f"{d.isoformat()}.parquet")


def fetch_fo_bhavcopy(
    trade_date: date,
    symbol: str = "NIFTY",
    use_cache: bool = True,
    timeout: int = 25,
) -> pd.DataFrame:
    """Returns one day's option rows for `symbol`. EMPTY frame if unavailable — never synthetic."""
    path = _cache_path(trade_date)
    if use_cache and os.path.exists(path):
        try:
            df = pd.read_parquet(path)
            return df[df.symbol == symbol] if symbol and "symbol" in df.columns else df
        except Exception:
            pass

    url = BASE_URL.format(ds=trade_date.strftime("%Y%m%d"))
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code != 200 or len(r.content) < 1000:
            return pd.DataFrame()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        raw = pd.read_csv(z.open(z.namelist()[0]))
    except Exception:
        return pd.DataFrame()  # holiday / missing / network — caller must see emptiness

    if "TckrSymb" not in raw.columns or "OptnTp" not in raw.columns:
        return pd.DataFrame()

    opts = raw[raw["OptnTp"].isin(["CE", "PE"])].copy()
    keep = {k: v for k, v in COLUMNS.items() if k in opts.columns}
    out = opts[list(keep) + ["TckrSymb"]].rename(columns={**keep, "TckrSymb": "symbol"})
    for c in ("strike", "open", "high", "low", "close", "settle", "underlying", "oi", "chg_oi", "volume"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out["trade_date"] = trade_date.isoformat()

    if use_cache:
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            out.to_parquet(path, index=False)
        except Exception:
            pass

    return out[out.symbol == symbol] if symbol else out


def bulk_fetch(
    start: date,
    end: date,
    symbol: str = "NIFTY",
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetches every weekday in [start, end]. Missing days (holidays) are skipped silently."""
    frames: List[pd.DataFrame] = []
    d, got, missed = start, 0, 0
    while d <= end:
        if d.weekday() < 5:  # NSE trades Mon-Fri
            df = fetch_fo_bhavcopy(d, symbol=symbol)
            if df.empty:
                missed += 1
            else:
                frames.append(df)
                got += 1
            if verbose and (got + missed) % 25 == 0:
                print(f"  {d}: {got} days fetched, {missed} unavailable")
        d += timedelta(days=1)
    if verbose:
        print(f"  done: {got} trading days, {missed} unavailable (holidays/weekends excluded)")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def coverage() -> pd.DataFrame:
    """What is cached locally."""
    if not os.path.isdir(CACHE_DIR):
        return pd.DataFrame(columns=["trade_date", "rows"])
    rows = []
    for f in sorted(os.listdir(CACHE_DIR)):
        if f.endswith(".parquet"):
            p = os.path.join(CACHE_DIR, f)
            try:
                rows.append({"trade_date": f[:-8], "rows": len(pd.read_parquet(p))})
            except Exception:
                continue
    return pd.DataFrame(rows)
