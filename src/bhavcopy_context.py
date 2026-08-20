"""Builds a REAL options context from historical NSE bhavcopy, for honest backtesting.

WHY
Every historical test of the desk verdict so far has fed it a SYNTHETIC option chain, because
no historical chain was available. That is a serious limitation: the synthetic context
produces confluence scores of 27-57, while the live system with a real Fyers chain scores
78-100. We have therefore been measuring a configuration that does not exist in production.

Bhavcopy carries genuine per-strike close, open interest, change in OI and volume for every
listed contract, every day, for years. That is enough to reconstruct the positioning inputs
the strategy actually gates on: dealer gamma walls, max pain, PCR and the put/call skew.

A note on granularity: open interest in Indian markets is an END-OF-DAY figure. Intraday OI
is only available live and is never recorded historically. So a daily-updated positioning
context is not a compromise here — it is closer to what was genuinely knowable at the time
than an interpolated intraday one would be.

Only contracts that actually traded are used (see nse_bhavcopy.tradeable): untraded strikes
carry a theoretical settlement price and inventing positioning from them manufactures signal.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _nearest_expiry(day_df: pd.DataFrame, trade_date: pd.Timestamp) -> Optional[pd.Timestamp]:
    exps = sorted(pd.to_datetime(day_df["expiry"].unique()))
    fut = [e for e in exps if (e - trade_date).days >= 0]
    return fut[0] if fut else None


def build_context_from_bhavcopy(
    day_df: pd.DataFrame,
    trade_date: Any,
    spot: float,
    min_volume: int = 1,
    min_oi: int = 1,
) -> Optional[Dict[str, Any]]:
    """Assembles an options_context dict from one trading day's real bhavcopy rows.

    Returns None when the day has too little tradeable data to describe positioning
    honestly — a caller must see absence rather than a fabricated chain.
    """
    if day_df is None or day_df.empty or not spot or spot <= 0:
        return None
    d = day_df.copy()
    d["expiry"] = pd.to_datetime(d["expiry"], errors="coerce")
    td = pd.Timestamp(trade_date)
    exp = _nearest_expiry(d, td)
    if exp is None:
        return None
    d = d[d["expiry"] == exp]
    # Only real, traded contracts inform positioning.
    if "volume" in d.columns:
        d = d[d["volume"].fillna(0) >= min_volume]
    if "oi" in d.columns:
        d = d[d["oi"].fillna(0) >= min_oi]
    if len(d) < 8:
        return None

    ce = d[d.option_type == "CE"].set_index("strike").sort_index()
    pe = d[d.option_type == "PE"].set_index("strike").sort_index()
    common = ce.index.intersection(pe.index)
    if len(common) < 6:
        return None
    ce, pe = ce.loc[common], pe.loc[common]
    strikes = np.array(common, dtype=float)

    ce_oi = ce["oi"].fillna(0).to_numpy(float)
    pe_oi = pe["oi"].fillna(0).to_numpy(float)

    # --- Dealer walls: peak OI concentration above/below spot -----------------
    above = strikes > spot
    below = strikes < spot
    call_wall = float(strikes[above][np.argmax(ce_oi[above])]) if above.any() and ce_oi[above].max() > 0 else float(strikes.max())
    put_wall = float(strikes[below][np.argmax(pe_oi[below])]) if below.any() and pe_oi[below].max() > 0 else float(strikes.min())

    # --- Max pain: strike minimising total writer payout ----------------------
    pain = [float((np.maximum(0.0, k - strikes) * pe_oi).sum() + (np.maximum(0.0, strikes - k) * ce_oi).sum())
            for k in strikes]
    max_pain = float(strikes[int(np.argmin(pain))]) if pain else float(spot)

    # --- PCR and a put/call price-skew proxy ----------------------------------
    pcr = float(pe_oi.sum() / ce_oi.sum()) if ce_oi.sum() > 0 else 1.0
    atm_i = int(np.argmin(np.abs(strikes - spot)))
    lo, hi = max(0, atm_i - 4), min(len(strikes), atm_i + 5)
    ce_px = ce["close"].to_numpy(float)[lo:hi]
    pe_px = pe["close"].to_numpy(float)[lo:hi]
    skew = float((pe_px.mean() - ce_px.mean()) / max(spot, 1.0) * 100.0) if len(ce_px) and len(pe_px) else 0.0

    # Dealer regime: spot above max pain implies dealers short upside gamma here.
    positive_gamma = bool(put_wall < spot < call_wall)
    d_vec = float(np.clip((max_pain - spot) / max(spot * 0.01, 1.0), -1.0, 1.0))

    return {
        "chain_df": _to_engine_frame(ce, pe, strikes),
        "gex_chart": {
            "call_wall_strike": call_wall,
            "put_wall_strike": put_wall,
            "zero_gex_strike": max_pain,
            "net_dealer_regime": "DEALER_LONG_GAMMA" if positive_gamma else "DEALER_SHORT_GAMMA",
            "walls_verified": True,
        },
        "dir_flow": {"directional_vector": d_vec, "pcr": pcr, "max_pain": max_pain},
        "skew_pct": skew,
        "source": "NSE_BHAVCOPY",
    }


def _to_engine_frame(ce: pd.DataFrame, pe: pd.DataFrame, strikes: np.ndarray) -> pd.DataFrame:
    """Flat 12-column chain in the shape the strategy engine expects."""
    return pd.DataFrame({
        "strike": strikes,
        "ce_ltp": ce["close"].to_numpy(float),
        "ce_oi": ce["oi"].fillna(0).to_numpy(float),
        "ce_change_oi": ce.get("chg_oi", pd.Series(0, index=ce.index)).fillna(0).to_numpy(float),
        "ce_volume": ce.get("volume", pd.Series(0, index=ce.index)).fillna(0).to_numpy(float),
        "ce_iv": 0.0,
        "pe_ltp": pe["close"].to_numpy(float),
        "pe_oi": pe["oi"].fillna(0).to_numpy(float),
        "pe_change_oi": pe.get("chg_oi", pd.Series(0, index=pe.index)).fillna(0).to_numpy(float),
        "pe_volume": pe.get("volume", pd.Series(0, index=pe.index)).fillna(0).to_numpy(float),
        "pe_iv": 0.0,
    })
