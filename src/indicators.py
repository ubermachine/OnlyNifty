"""JustNifty v3.0 Tier-1 Quantitative Indicators & Adaptive Stochastic Models."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, List
from src.config import (
    EMA_FAST, EMA_MID, EMA_SLOW, VAKC_LAMBDA, VAKC_ATR_SPAN,
    HURST_TRENDING_MIN, HURST_MEAN_REV_MAX, DEFAULT_IV
)

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Computes Exponential Moving Average with span = period."""
    return series.ewm(span=period, adjust=False).mean()

def compute_envelopes(ema_series: pd.Series, pct: float = 0.015) -> Tuple[pd.Series, pd.Series]:
    """Computes upper and lower percentage envelope bands around a moving average."""
    upper = ema_series * (1.0 + pct)
    lower = ema_series * (1.0 - pct)
    return upper, lower

def compute_hurst_exponent(series: pd.Series, min_lag: int = 4, max_lag: int = 35) -> Dict[str, Any]:
    """
    Computes the Fractional Hurst Exponent (H) via Rescaled Range (R/S) analysis.
    H > 0.55 -> Persistent / Trending Regime
    H < 0.45 -> Anti-Persistent / Mean-Reverting Regime
    0.45 <= H <= 0.55 -> Random Walk (No alpha)
    """
    if len(series) < max_lag + 10:
        return {"hurst": 0.50, "regime": "RANDOM_WALK (Accumulating Data)", "is_trending": False}
        
    prices = series.values[-100:] if len(series) >= 100 else series.values
    lags = np.unique(np.linspace(min_lag, min(max_lag, len(prices) // 2), 8, dtype=int))
    rs_values = []
    
    for lag in lags:
        sub_returns = np.diff(prices[:len(prices) - (len(prices) % lag)])
        chunks = np.array_split(sub_returns, len(sub_returns) // lag)
        
        chunk_rs = []
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            mean = np.mean(chunk)
            cum_dev = np.cumsum(chunk - mean)
            r = np.max(cum_dev) - np.min(cum_dev)
            s = np.std(chunk, ddof=1)
            if s > 1e-8:
                chunk_rs.append(r / s)
                
        if chunk_rs:
            rs_values.append(np.mean(chunk_rs))
            
    if len(rs_values) >= 3:
        valid_lags = lags[:len(rs_values)]
        poly = np.polyfit(np.log(valid_lags), np.log(rs_values), 1)
        h = float(np.clip(poly[0], 0.05, 0.95))
    else:
        h = 0.50
        
    if h > HURST_TRENDING_MIN:
        regime = "PERSISTENT TRENDING (H > 0.55 - Golden Pocket Active)"
        is_trending = True
    elif h < HURST_MEAN_REV_MAX:
        regime = "ANTI-PERSISTENT MEAN-REVERTING (H < 0.45 - Fade Extremes)"
        is_trending = False
    else:
        regime = "RANDOM WALK (0.45 <= H <= 0.55 - High Noise)"
        is_trending = False
        
    return {
        "hurst": round(h, 4),
        "regime": regime,
        "is_trending": is_trending
    }

def compute_vakc_envelopes(
    df: pd.DataFrame,
    ema_span: int = EMA_SLOW,
    atr_span: int = VAKC_ATR_SPAN,
    k: float = VAKC_LAMBDA,
    iv: float = DEFAULT_IV
) -> Tuple[pd.Series, pd.Series]:
    """
    Computes Volatility-Adaptive Keltner Channels (VAKC):
    Upper = EMA200 + k * ATR * sqrt(IV / IV_baseline)
    Lower = EMA200 - k * ATR * sqrt(IV / IV_baseline)
    """
    df = df.copy()
    ema200 = compute_ema(df["close"], ema_span)
    
    # Calculate Average True Range (ATR)
    high_low = df["high"] - df["low"]
    high_close_prev = (df["high"] - df["close"].shift(1)).abs()
    low_close_prev = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_span, min_periods=1).mean()
    
    # Volatility expansion scaling
    vol_scalar = np.sqrt(max(iv, 0.08) / 0.12)
    band_width = (k * atr * vol_scalar).fillna(ema200 * 0.015)
    
    upper_vakc = ema200 + band_width
    lower_vakc = ema200 - band_width
    return upper_vakc, lower_vakc

def compute_vwap(df: pd.DataFrame, anchor_session: bool = True) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Computes Session Anchored VWAP and ±2σ dispersion bands with zero-volume resilience."""
    df = df.copy()
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    
    vol = df["volume"].copy().astype(float)
    if vol.sum() == 0 or (vol == 0).all():
        vol = (df["high"] - df["low"]).clip(lower=1.0)
    else:
        vol = vol.replace(0, 1.0)
        
    tp_vol = typical_price * vol
    
    if anchor_session:
        dates = pd.Series(df.index.date, index=df.index)
        cum_vol = vol.groupby(dates).cumsum()
        cum_tp_vol = tp_vol.groupby(dates).cumsum()
        vwap = (cum_tp_vol / cum_vol.clip(lower=1.0)).fillna(typical_price)
        
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

def compute_order_flow_imbalance(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes Cumulative Volume Delta (CVD) and Order Flow Imbalance (OFI) proxy.
    Positive OFI indicates aggressive buyer market orders absorbing liquidity.
    """
    if df.empty or len(df) < 2:
        return {"ofi": 0.0, "cvd": 0.0, "buyer_defense": True, "cvd_series": pd.Series()}
        
    vol = df["volume"].copy().astype(float)
    if vol.sum() == 0 or (vol == 0).all():
        vol = (df["high"] - df["low"]).clip(lower=1.0)
        
    # Proxy delta: Signed volume based on candle body position relative to high/low range
    candle_range = (df["high"] - df["low"]).replace(0, 1.0)
    delta_weight = (df["close"] - df["open"]) / candle_range
    bar_delta = vol * delta_weight
    
    cvd = bar_delta.cumsum()
    recent_ofi = float(bar_delta.tail(5).sum())
    
    return {
        "ofi": round(recent_ofi, 2),
        "cvd": round(float(cvd.iloc[-1]), 2),
        "buyer_defense": recent_ofi >= 0,
        "cvd_series": cvd
    }

def compute_cpr(daily_df: pd.DataFrame) -> Dict[str, Any]:
    """Computes Central Pivot Range (Pivot, Bottom Central, Top Central) and classifies day regime."""
    if daily_df.empty:
        return {
            "pivot": 0.0, "bc": 0.0, "tc": 0.0,
            "cpr_top": 0.0, "cpr_bottom": 0.0,
            "width_pct": 0.0, "is_narrow": False,
            "regime": "UNKNOWN"
        }
    
    # Resample to true Daily OHLC if intraday dataset is passed
    if len(daily_df) > 10 and hasattr(daily_df.index, "date"):
        daily_resampled = daily_df.resample("D").agg({
            "open": "first", "high": "max", "low": "min", "close": "last"
        }).dropna()
        last = daily_resampled.iloc[-1] if not daily_resampled.empty else daily_df.iloc[-1]
    else:
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
        "regime": "TRENDING REGIME (Narrow CPR < 0.20%)" if is_narrow else "RANGE-BOUND CHOP (Wide CPR >= 0.20%)"
    }

def compute_fibonacci_levels(high: float, low: float, is_uptrend: bool) -> Dict[str, float]:
    """Calculates Golden Pocket (50.0% - 61.8%) retracements and invalidation Stop-Loss levels."""
    diff = max(high - low, 1.0)
    if is_uptrend:
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
        return {
            "swing_high": round(high, 2),
            "swing_low": round(low, 2),
            "fib_236": round(low + 0.236 * diff, 2),
            "fib_382": round(low + 0.382 * diff, 2),
            "fib_500": round(low + 0.500 * diff, 2),
            "fib_618": round(low + 0.618 * diff, 2),
            "fib_786": round(low + 0.786 * diff, 2),
            "sl_level": round(low + 0.786 * diff + 5.0, 2)
        }

def compute_vf_trade_table(open_price: float, atr: float = 60.0) -> Dict[str, float]:
    """Calculates the JustNifty VF Trade Table (Targets T1 through T6 for Long and Short)."""
    step = max(atr * 0.45, 25.0)
    table = {}
    for i in range(1, 7):
        table[f"T{i}_Long"] = round(open_price + (i * step), 2)
        table[f"T{i}_Short"] = round(open_price - (i * step), 2)
    return table

def compute_volume_profile(df: pd.DataFrame, n_bins: int = 36) -> Dict[str, Any]:
    """Computes Dual-Bracket 70% Volume Profile, POC, VAH, and VAL."""
    if df.empty or len(df) < 3:
        return {"poc": 0.0, "vah": 0.0, "val": 0.0, "bins": [], "volumes": []}
    
    price_min = float(df["low"].min())
    price_max = float(df["high"].max())
    if np.isclose(price_min, price_max):
        return {"poc": price_min, "vah": price_max, "val": price_min, "bins": [price_min, price_max], "volumes": [int(df['volume'].sum())]}
        
    bins = np.linspace(price_min, price_max, n_bins)
    bin_volumes = np.zeros(n_bins - 1)
    
    for _, row in df.iterrows():
        # Distribute volume across candle range
        c_low, c_high = float(row["low"]), float(row["high"])
        vol = float(row["volume"]) if row["volume"] > 0 else (c_high - c_low)
        
        idx_low = np.clip(np.digitize(c_low, bins) - 1, 0, n_bins - 2)
        idx_high = np.clip(np.digitize(c_high, bins) - 1, 0, n_bins - 2)
        
        span = max(idx_high - idx_low + 1, 1)
        for b_i in range(idx_low, idx_high + 1):
            bin_volumes[b_i] += vol / span
            
    poc_idx = int(np.argmax(bin_volumes))
    poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2.0
    
    total_vol = bin_volumes.sum()
    target_vol = total_vol * 0.70
    
    low_idx, high_idx = poc_idx, poc_idx
    curr_vol = bin_volumes[poc_idx]
    
    # Dual-bracket expansion for Value Area
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

def compute_dealer_gex(spot: float, call_oi: float = 14500000.0, put_oi: float = 12800000.0) -> Dict[str, Any]:
    """
    Computes Net Dealer Gamma Exposure (GEX) and Gamma Flip Level.
    Positive GEX: Market makers Long Gamma -> Mean-Reversion Mode
    Negative GEX: Market makers Short Gamma -> Momentum Breakout Mode
    """
    # Standard Nifty GEX approximation in ₹ Crores per 1% spot move
    net_oi_diff = (call_oi - put_oi) / 100000.0
    gex_crores = net_oi_diff * (spot / 24000.0) * 4.5
    
    is_positive_gex = gex_crores >= 0
    flip_strike = int(round(spot / 50.0) * 50) + (50 if not is_positive_gex else -50)
    
    return {
        "net_gex_crores": round(gex_crores, 2),
        "is_positive_gamma": is_positive_gex,
        "gamma_regime": "POSITIVE GAMMA (MM Long Gamma -> Mean-Reversion S/R Holds)" if is_positive_gex else "NEGATIVE GAMMA (MM Short Gamma -> High Velocity Breakouts)",
        "gamma_flip_strike": flip_strike
    }
