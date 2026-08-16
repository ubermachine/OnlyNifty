"""Fyers Data API v3 client: quotes, historical candles, and option chains.

Authenticates via src/fyers_auth.py. Fyers' option-chain endpoint returns real strikes/OI/LTP
but no IV field, so per-strike IV is solved from LTP via Black-Scholes bisection (r=RISK_FREE_RATE,
q=0) — independent of src/options_engine.py's black_scholes_greeks, which bakes in a synthetic
put-skew premium meant to compensate for flat placeholder IV; real market LTP already embeds skew.
"""

import math
from datetime import datetime
from typing import Any, Dict

import pandas as pd
import pytz
import requests
from scipy.stats import norm

from src.config import RISK_FREE_RATE
from src.fyers_auth import _load_config, get_access_token

IST = pytz.timezone("Asia/Kolkata")
DATA_BASE = "https://api-t1.fyers.in/data"


def _headers() -> Dict[str, str]:
    cfg = _load_config()
    return {"Authorization": f"{cfg['client_id']}:{get_access_token()}"}


def get_quote(symbol: str) -> Dict[str, Any]:
    resp = requests.get(f"{DATA_BASE}/quotes", params={"symbols": symbol}, headers=_headers(), timeout=10)
    data = resp.json()
    if data.get("s") != "ok" or not data.get("d"):
        raise RuntimeError(f"Fyers quote fetch failed for {symbol}: {data}")
    return data["d"][0]["v"]


def get_history(symbol: str, resolution: str, range_from: str, range_to: str) -> pd.DataFrame:
    """Returns real OHLCV candles (IST-indexed) — resolution: '1'/'5'/'15'/... minutes or 'D'."""
    params = {
        "symbol": symbol,
        "resolution": resolution,
        "date_format": "1",
        "range_from": range_from,
        "range_to": range_to,
        "cont_flag": "1",
    }
    resp = requests.get(f"{DATA_BASE}/history", params=params, headers=_headers(), timeout=10)
    data = resp.json()
    if data.get("s") != "ok":
        raise RuntimeError(f"Fyers history fetch failed for {symbol}: {data}")
    candles = data.get("candles", [])
    if not candles:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(IST)
    df = df.drop(columns=["ts"])
    return df


def _implied_vol(price: float, spot: float, strike: float, t_years: float, is_call: bool) -> float:
    """Solves Black-Scholes implied vol (%) from a market price via bisection."""
    intrinsic = max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)
    if price <= intrinsic + 0.05 or t_years <= 0:
        return 0.0

    def bs_price(sigma: float) -> float:
        sqrt_t = math.sqrt(t_years)
        d1 = (math.log(spot / strike) + (RISK_FREE_RATE + 0.5 * sigma ** 2) * t_years) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        disc = math.exp(-RISK_FREE_RATE * t_years)
        if is_call:
            return spot * norm.cdf(d1) - strike * disc * norm.cdf(d2)
        return strike * disc * norm.cdf(-d2) - spot * norm.cdf(-d1)

    lo, hi = 0.01, 3.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if bs_price(mid) > price:
            hi = mid
        else:
            lo = mid
    return round((lo + hi) / 2.0 * 100.0, 2)


def get_option_chain_raw(symbol: str, strikecount: int = 20, timestamp: str = "") -> Dict[str, Any]:
    params = {"symbol": symbol, "strikecount": str(strikecount), "timestamp": timestamp}
    resp = requests.get(f"{DATA_BASE}/options-chain-v3", params=params, headers=_headers(), timeout=10)
    data = resp.json()
    if data.get("s") != "ok":
        raise RuntimeError(f"Fyers option chain fetch failed for {symbol}: {data}")
    return data["data"]


def _rows_from_chain(raw: Dict[str, Any], spot: float, expiry_label: str, expiry_epoch: int) -> pd.DataFrame:
    """Pairs CE/PE legs per strike from a raw Fyers chain into the app's flat 12-column schema."""
    now_epoch = datetime.now(IST).timestamp()
    t_years = max((expiry_epoch - now_epoch) / (365.0 * 86400.0), 1e-6)

    by_strike: Dict[float, Dict[str, Any]] = {}
    for leg in raw.get("optionsChain", []):
        strike = leg.get("strike_price")
        if strike is None or strike < 0:
            continue
        by_strike.setdefault(strike, {})[leg.get("option_type")] = leg

    rows = []
    for strike, legs in sorted(by_strike.items()):
        ce, pe = legs.get("CE", {}), legs.get("PE", {})
        ce_ltp = float(ce.get("ltp", 0.0))
        pe_ltp = float(pe.get("ltp", 0.0))
        rows.append({
            "strike": strike,
            "expiry": expiry_label,
            "ce_ltp": ce_ltp,
            "ce_oi": int(ce.get("oi", 0)),
            "ce_change_oi": int(ce.get("oich", 0)),
            "ce_iv": _implied_vol(ce_ltp, spot, strike, t_years, True) if ce_ltp > 0 else 0.0,
            "ce_volume": int(ce.get("volume", 0)),
            "pe_ltp": pe_ltp,
            "pe_oi": int(pe.get("oi", 0)),
            "pe_change_oi": int(pe.get("oich", 0)),
            "pe_iv": _implied_vol(pe_ltp, spot, strike, t_years, False) if pe_ltp > 0 else 0.0,
            "pe_volume": int(pe.get("volume", 0)),
        })
    return pd.DataFrame(rows)


def get_multi_expiry_chain(symbol: str = "NSE:NIFTY50-INDEX", strikecount: int = 20) -> Dict[str, Any]:
    """Fetches genuine (not extrapolated) near-week, next-week, and monthly option chains."""
    base = get_option_chain_raw(symbol, strikecount=strikecount, timestamp="")
    underlying_leg = next((leg for leg in base.get("optionsChain", []) if leg.get("strike_price", 0) < 0), None)
    spot = float(underlying_leg["ltp"]) if underlying_leg else 0.0
    expiries = base.get("expiryData", [])
    if not expiries or spot <= 0:
        raise RuntimeError("Fyers option chain returned no usable expiry/spot data.")

    near = expiries[0]
    next_ = expiries[1] if len(expiries) > 1 else expiries[0]
    monthly = next((e for e in expiries if e.get("expiry_flag") == "M"), None) or expiries[min(3, len(expiries) - 1)]

    def fetch(exp: Dict[str, Any]) -> pd.DataFrame:
        raw = base if exp is near else get_option_chain_raw(symbol, strikecount=strikecount, timestamp=exp["expiry"])
        return _rows_from_chain(raw, spot, exp["date"], int(exp["expiry"]))

    near_chain, next_chain, monthly_chain = fetch(near), fetch(next_), fetch(monthly)
    combined = pd.concat([near_chain, next_chain, monthly_chain], ignore_index=True)

    return {
        "underlying_value": spot,
        "expiry_dates": [near["date"], next_["date"], monthly["date"]],
        "near_expiry": near["date"],
        "next_expiry": next_["date"],
        "monthly_expiry": monthly["date"],
        "near_chain": near_chain,
        "next_chain": next_chain,
        "monthly_chain": monthly_chain,
        "expiries": {near["date"]: near_chain, next_["date"]: next_chain, monthly["date"]: monthly_chain},
        "dataframe": combined,
        "source": "Fyers (Live)",
    }


def get_vix() -> float:
    raw = get_option_chain_raw("NSE:NIFTY50-INDEX", strikecount=1, timestamp="")
    return float(raw.get("indiavixData", {}).get("ltp", 0.0))
