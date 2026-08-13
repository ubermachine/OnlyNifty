"""Quantitative technical indicators for the JustNifty v2.0 trading methodology."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Computes Exponential Moving Average with span = period."""
    return series.ewm(span=period, adjust=False).mean()

def compute_envelopes(ema_series: pd.Series, pct: float = 0.015) -> Tuple[pd.Series, pd.Series]:
    """Computes upper and lower percentage envelope bands around a moving average."""
    upper = ema_series * (1.0 + pct)
    lower = ema_series * (1.0 - pct)
    return upper, lower

def compute_vwap(df: pd.DataFrame, anchor_session: bool = True) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Computes Session Volume Weighted Average Price (VWAP) and ±2 standard deviation bands with zero-volume resilience."""
    df = df.copy()
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    
    # Clean volume: If volume is 0 or all zeros (common for index feeds like ^NSEI), use price range as activity proxy
    vol = df["volume"].copy().astype(float)
    if vol.sum() == 0 or (vol == 0).all():
        vol = (df["high"] - df["low"]).clip(lower=1.0)
    else:
        vol = vol.replace(0, 1.0)
        
    tp_vol = typical_price * vol
    
    if anchor_session:
        # Vectorized intraday session VWAP by date group
        dates = pd.Series(df.index.date, index=df.index)
        cum_vol = vol.groupby(dates).cumsum()
        cum_tp_vol = tp_vol.groupby(dates).cumsum()
        vwap = (cum_tp_vol / cum_vol.clip(lower=1.0)).fillna(typical_price)
        
        # Standard deviation bands
        squared_diff = ((typical_price - vwap) ** 2) * vol
        cum_sq_diff = squared_diff.groupby(dates).cumsum()
        std_dev = np.sqrt(np.maximum(cum_sq_diff / cum_vol.clip(lower=1.0), 0)).fillna(0)
    else:
        cum_vol = vol.cumsum().clip(lower=1.0)
        cum_tp_vol = tp_vol.cumsum()
        vwap = (cum_tp_vol / cum_vol).fillna(typical_price)
        std_dev = np.sqrt((((typical_price - vwap) ** 2) * vol).cumsum() / cum_vol).fillna(0)

    upper_sd = vwap + (2.0 * std_dev)
    lower_sd = vwap - (2.0 * std_dev)
    return vwap, upper_sd, lower_sd

def compute_cpr(daily_df: pd.DataFrame) -> Dict[str, Any]:
    """Computes Central Pivot Range (Pivot, Bottom Central, Top Central) and classifies day regime."""
    if daily_df.empty:
        return {
            "pivot": 0.0, "bc": 0.0, "tc": 0.0,
            "cpr_top": 0.0, "cpr_bottom": 0.0,
            "width_pct": 0.0, "is_narrow": False,
            "regime": "UNKNOWN"
        }
    
    last = daily_df.iloc[-1]
    high = float(last["high"])
    low = float(last["low"])
    close = float(last["close"])
    
    pivot = (high + low + close) / 3.0
    bc = (high + low) / 2.0
    tc = (pivot - bc) + pivot
    
    width_pct = (abs(tc - bc) / pivot) * 100.0 if pivot > 0 else 0.0
    is_narrow = width_pct < 0.20
    
    return {
        "pivot": round(pivot, 2),
        "bc": round(bc, 2),
        "tc": round(tc, 2),
        "cpr_top": round(max(bc, tc), 2),
        "cpr_bottom": round(min(bc, tc), 2),
        "width_pct": round(width_pct, 4),
        "is_narrow": is_narrow,
        "regime": "TRENDING DAY (Narrow CPR - Full Conviction)" if is_narrow else "RANGE-BOUND / CHOP (Wide CPR - Theta Decay Risk)"
    }

def compute_fibonacci_levels(high: float, low: float, is_uptrend: bool) -> Dict[str, float]:
    """Calculates Golden Pocket (50.0% - 61.8%) retracements and invalidation Stop-Loss levels."""
    diff = max(high - low, 1.0)
    if is_uptrend:
        # Bullish impulse: Retracement downwards from High
        return {
            "swing_high": round(high, 2),
            "swing_low": round(low, 2),
            "fib_236": round(high - 0.236 * diff, 2),
            "fib_382": round(high - 0.382 * diff, 2),
            "fib_500": round(high - 0.500 * diff, 2),
            "fib_618": round(high - 0.618 * diff, 2),
            "fib_786": round(high - 0.786 * diff, 2),
            "sl_level": round(high - 0.786 * diff - 5.0, 2)
        }
    else:
        # Bearish impulse: Retracement upwards from Low
        return {
            "swing_high": round(high, 2),
            "swing_low": round(low, 2),
            "fib_236": round(low + 0.236 * diff, 2),
            "fib_382": round(low + 0.382 * diff, 2),
            "fib_500": round(low + 0.500 * diff, 2),
            "fib_618": round(low + 0.618 * diff, 2),
            "fib_786": round(low + 0.786 * diff, 2),
            "sl_level": round(low + 0.382 * diff + 5.0, 2)
        }

def compute_vf_trade_table(open_price: float, atr: float = 60.0) -> Dict[str, float]:
    """Calculates the JustNifty VF Trade Table (Targets T1 through T6 for Long and Short)."""
    step = max(atr * 0.45, 25.0)
    table = {}
    for i in range(1, 7):
        table[f"T{i}_Long"] = round(open_price + (i * step), 2)
        table[f"T{i}_Short"] = round(open_price - (i * step), 2)
    return table

def compute_volume_profile(df: pd.DataFrame, n_bins: int = 24) -> Dict[str, Any]:
    """Computes Volume Profile, Point of Control (POC), Value Area High (VAH), and Value Area Low (VAL)."""
    if df.empty or len(df) < 3:
        return {"poc": 0.0, "vah": 0.0, "val": 0.0, "bins": [], "volumes": []}
    
    price_min = float(df["low"].min())
    price_max = float(df["high"].max())
    if np.isclose(price_min, price_max):
        return {"poc": price_min, "vah": price_max, "val": price_min, "bins": [price_min, price_max], "volumes": [int(df['volume'].sum())]}
        
    bins = np.linspace(price_min, price_max, n_bins)
    bin_volumes = np.zeros(n_bins - 1)
    
    for _, row in df.iterrows():
        mid = (row["high"] + row["low"]) / 2.0
        idx = np.clip(np.digitize(mid, bins) - 1, 0, n_bins - 2)
        bin_volumes[idx] += row["volume"]
            
    poc_idx = int(np.argmax(bin_volumes))
    poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2.0
    
    total_vol = bin_volumes.sum()
    target_vol = total_vol * 0.70
    
    low_idx, high_idx = poc_idx, poc_idx
    curr_vol = bin_volumes[poc_idx]
    
    while curr_vol < target_vol and (low_idx > 0 or high_idx < len(bin_volumes) - 1):
        v_below = bin_volumes[low_idx - 1] if low_idx > 0 else 0
        v_above = bin_volumes[high_idx + 1] if high_idx < len(bin_volumes) - 1 else 0
        if v_above >= v_below and high_idx < len(bin_volumes) - 1:
            high_idx += 1
            curr_vol += v_above
        elif low_idx > 0:
            low_idx -= 1
            curr_vol += v_below
        else:
            break
            
    val = bins[low_idx]
    vah = bins[high_idx + 1]
    
    return {
        "poc": round(float(poc_price), 2),
        "vah": round(float(vah), 2),
        "val": round(float(val), 2),
        "bins": [round(float(b), 2) for b in bins],
        "volumes": [int(v) for v in bin_volumes]
    }
