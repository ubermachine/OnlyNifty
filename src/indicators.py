"""JustNifty v3.0 Tier-1 Quantitative Indicators, Stochastic Models & Order Flow Mechanics."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, List
from src.config import (
    EMA_FAST, EMA_MID, EMA_SLOW, VAKC_LAMBDA, VAKC_ATR_SPAN,
    HURST_TRENDING_MIN, HURST_MEAN_REV_MAX, DEFAULT_IV, OFI_ZSCORE_MIN
)

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Computes Exponential Moving Average with span = period."""
    return series.ewm(span=period, adjust=False).mean()

def compute_envelopes(ema_series: pd.Series, pct: float = 0.015) -> Tuple[pd.Series, pd.Series]:
    """Computes upper and lower percentage envelope bands around a moving average."""
    upper = ema_series * (1.0 + pct)
    lower = ema_series * (1.0 - pct)
    return upper, lower

def compute_hurst_exponent(series: pd.Series, min_lag: int = 5, max_lag: int = 30) -> Dict[str, Any]:
    """
    Computes the Fractional Hurst Exponent (H) via Log-Returns Rescaled Range (R/S) Analysis
    with finite-sample Anis-Lloyd / Peters Bias Correction.
    H > 0.52 -> Persistent / Trending Regime (Golden Pocket Active)
    H < 0.45 -> Anti-Persistent / Mean-Reverting Regime
    0.45 <= H <= 0.52 -> Random Walk / Noise
    """
    if len(series) < max_lag + 10:
        return {"hurst": 0.50, "regime": "RANDOM_WALK (Accumulating Data)", "is_trending": False}
        
    prices = series.values[-100:] if len(series) >= 100 else series.values
    # Compute continuously compounded log-returns for weak-sense stationarity
    log_returns = np.diff(np.log(np.maximum(prices, 1.0)))
    
    if len(log_returns) < max_lag:
        return {"hurst": 0.50, "regime": "RANDOM_WALK (Accumulating Data)", "is_trending": False}

    lags = np.unique(np.linspace(min_lag, min(max_lag, len(log_returns) // 2), 6, dtype=int))
    rs_values = []
    
    for lag in lags:
        n_chunks = len(log_returns) // lag
        if n_chunks < 1:
            continue
            
        chunk_rs = []
        for i in range(n_chunks):
            chunk = log_returns[i * lag : (i + 1) * lag]
            if len(chunk) < 2:
                continue
            mean = np.mean(chunk)
            cum_dev = np.cumsum(chunk - mean)
            r = np.max(cum_dev) - np.min(cum_dev)
            s = np.std(chunk, ddof=1)
            if s > 1e-8:
                chunk_rs.append(r / s)
                
        if chunk_rs:
            # Anis-Lloyd analytical expected R/S for standard white noise
            expected_rs = np.sqrt((lag - 0.5) / (np.pi * 0.5)) if lag > 2 else 1.0
            raw_rs = np.mean(chunk_rs)
            # Subtract finite-sample bias
            rs_values.append(raw_rs / max(expected_rs * 0.85, 0.1))
            
    if len(rs_values) >= 3:
        valid_lags = lags[:len(rs_values)]
        poly = np.polyfit(np.log(valid_lags), np.log(rs_values), 1)
        # Shift regression slope back to H scale
        h = float(np.clip(poly[0] + 0.50, 0.10, 0.90))
    else:
        h = 0.50
        
    if h >= HURST_TRENDING_MIN:
        regime = f"PERSISTENT TRENDING (H={h:.2f} >= 0.52 - Golden Pocket Active)"
        is_trending = True
    elif h < HURST_MEAN_REV_MAX:
        regime = f"ANTI-PERSISTENT MEAN-REVERTING (H={h:.2f} < 0.45 - Fade Extremes)"
        is_trending = False
    else:
        regime = f"RANDOM WALK (H={h:.2f} - High Noise Filtered)"
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
    Computes Volatility-Adaptive Keltner Channels (VAKC) using Wilder's RMA for ATR
    and a smooth C1 non-linear volatility elasticity function:
    xi(IV) = (IV / 0.12)^0.45 * (1 + 0.12 * tanh((IV - 0.18) / 0.06))
    Upper = EMA200 + k * ATR * xi(IV)
    Lower = EMA200 - k * ATR * xi(IV)
    """
    df = df.copy()
    ema200 = compute_ema(df["close"], ema_span)
    
    # Calculate True Range
    high_low = df["high"] - df["low"]
    high_close_prev = (df["high"] - df["close"].shift(1)).abs()
    low_close_prev = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    
    # Wilder's RMA (Exponential Running Moving Average for ATR)
    atr = tr.ewm(alpha=1.0/atr_span, adjust=False).mean()
    
    # Smooth Non-Linear Volatility Elasticity Multiplier
    clamped_iv = max(iv if iv is not None else DEFAULT_IV, 0.06)
    vol_ratio = clamped_iv / 0.12
    vol_scalar = (vol_ratio ** 0.45) * (1.0 + 0.12 * np.tanh((clamped_iv - 0.18) / 0.06))
    
    band_width = (k * atr * vol_scalar).fillna(ema200 * 0.015)
    
    upper_vakc = ema200 + band_width
    lower_vakc = ema200 - band_width
    return upper_vakc, lower_vakc


def compute_vwap(df: pd.DataFrame, anchor_session: bool = True) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Computes Session Anchored VWAP and Exact Online 2nd-Moment Variance (±2σ dispersion bands)."""
    df = df.copy()
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    
    vol = df["volume"].copy().astype(float)
    if vol.sum() == 0 or (vol == 0).all():
        vol = (df["high"] - df["low"]).clip(lower=1.0)
    else:
        vol = vol.replace(0, 1.0)
        
    tp_vol = typical_price * vol
    tp_sq_vol = (typical_price ** 2) * vol
    
    if anchor_session:
        dates = pd.Series(df.index.date, index=df.index) if hasattr(df.index, "date") else pd.Series(0, index=df.index)
        cum_vol = vol.groupby(dates).cumsum().clip(lower=1.0)
        cum_tp_vol = tp_vol.groupby(dates).cumsum()
        cum_tp_sq_vol = tp_sq_vol.groupby(dates).cumsum()
        
        vwap = (cum_tp_vol / cum_vol).fillna(typical_price)
        # Exact online 2nd-moment standard deviation: sqrt(E[X^2] - (E[X])^2)
        variance = (cum_tp_sq_vol / cum_vol) - (vwap ** 2)
        std_dev = np.sqrt(np.maximum(variance, 0.0)).fillna(0.0)
    else:
        cum_vol = vol.cumsum().clip(lower=1.0)
        cum_tp_vol = tp_vol.cumsum()
        cum_tp_sq_vol = tp_sq_vol.cumsum()
        vwap = (cum_tp_vol / cum_vol).fillna(typical_price)
        variance = (cum_tp_sq_vol / cum_vol) - (vwap ** 2)
        std_dev = np.sqrt(np.maximum(variance, 0.0)).fillna(0.0)

    upper_sd = vwap + (2.0 * std_dev)
    lower_sd = vwap - (2.0 * std_dev)
    return vwap, upper_sd, lower_sd

def compute_order_flow_imbalance(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes Composite Price-Wick Weighted Bar Delta & Rolling 20-Bar OFI Z-Score.
    Detects institutional absorption at Session AVWAP.
    """
    if df.empty or len(df) < 2:
        return {"ofi": 0.0, "ofi_zscore": 0.0, "cvd": 0.0, "buyer_defense": True, "cvd_series": pd.Series()}
        
    vol = df["volume"].copy().astype(float)
    if vol.sum() == 0 or (vol == 0).all():
        vol = (df["high"] - df["low"]).clip(lower=1.0)
        
    c_range = (df["high"] - df["low"]).replace(0, 1.0)
    # Composite Delta Weight: 60% Body Displacement + 40% Close Location Relative to Midpoint
    body_weight = (df["close"] - df["open"]) / c_range
    close_loc_weight = (2.0 * df["close"] - df["high"] - df["low"]) / c_range
    composite_weight = (0.60 * body_weight) + (0.40 * close_loc_weight)
    
    bar_delta = vol * composite_weight
    cvd = bar_delta.cumsum()
    
    # Rolling 20-Bar OFI Z-Score
    rolling_mean = bar_delta.rolling(window=20, min_periods=3).mean()
    rolling_std = bar_delta.rolling(window=20, min_periods=3).std().replace(0, 1.0)
    ofi_zscore_series = ((bar_delta - rolling_mean) / rolling_std).fillna(0.0)
    
    recent_z = float(ofi_zscore_series.iloc[-1])
    recent_ofi = float(bar_delta.tail(5).sum())
    
    return {
        "ofi": round(recent_ofi, 2),
        "ofi_zscore": round(recent_z, 3),
        "cvd": round(float(cvd.iloc[-1]), 2),
        "buyer_defense": recent_z >= -0.20,
        "seller_defense": recent_z <= 0.20,
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

def compute_pre_open_gap_filter(
    df_5m: pd.DataFrame,
    prev_close: Optional[float] = None,
    pre_open_data: Optional[Dict[str, Any]] = None,
    gap_threshold_pct: float = 0.0050,
    half_life_mins: float = 45.0
) -> Dict[str, Any]:
    """
    Evaluates 09:08 - 09:30 AM Pre-Open & Opening Gap dynamics and adapts Fibonacci/AVWAP rules.
    """
    if df_5m.empty:
        return {
            "regime": "NORMAL", "gap_pct": 0.0, "is_large_gap": False,
            "anchor_high": 0.0, "anchor_low": 0.0, "gap_golden_pocket": None,
            "slope_tolerance_mult": 1.0, "elapsed_mins": 0.0
        }
        
    first_bar = df_5m.iloc[0]
    session_open = float(pre_open_data.get("iep", first_bar["open"])) if pre_open_data else float(first_bar["open"])
    
    if prev_close is None or prev_close <= 0:
        prev_close = session_open
        
    gap_pts = session_open - prev_close
    gap_pct = (gap_pts / prev_close) if prev_close > 0 else 0.0
    
    # 09:15-09:30 Opening Range
    opening_bars = df_5m.head(3)
    opening_high = float(opening_bars["high"].max()) if not opening_bars.empty else session_open
    opening_low = float(opening_bars["low"].min()) if not opening_bars.empty else session_open
    
    # Time decay since 09:15
    elapsed_mins = float(min(len(df_5m) * 5.0, 360.0))
    decay_factor = np.exp(-elapsed_mins / max(half_life_mins, 1.0))
    slope_tolerance_mult = float(1.0 + (abs(gap_pct) / gap_threshold_pct) * decay_factor)
    
    if gap_pct >= gap_threshold_pct:
        regime = "LARGE_GAP_UP"
        is_large = True
        anchor_low = min(session_open, opening_low)
        anchor_high = float(df_5m["high"].max())
        gap_gp = {
            "gf_support_500": round(prev_close + 0.500 * gap_pts, 2),
            "gf_support_618": round(prev_close + 0.618 * gap_pts, 2),
            "gap_fill_target": round(prev_close, 2)
        }
    elif gap_pct <= -gap_threshold_pct:
        regime = "LARGE_GAP_DOWN"
        is_large = True
        anchor_high = max(session_open, opening_high)
        anchor_low = float(df_5m["low"].min())
        gap_gp = {
            "gf_resist_500": round(prev_close - 0.500 * abs(gap_pts), 2),
            "gf_resist_618": round(prev_close - 0.618 * abs(gap_pts), 2),
            "gap_fill_target": round(prev_close, 2)
        }
    else:
        regime = "NORMAL"
        is_large = False
        anchor_high = float(df_5m["high"].max())
        anchor_low = float(df_5m["low"].min())
        gap_gp = None
        
    return {
        "regime": regime,
        "gap_pct": round(gap_pct * 100.0, 3),
        "gap_pts": round(gap_pts, 2),
        "is_large_gap": is_large,
        "anchor_high": round(anchor_high, 2),
        "anchor_low": round(anchor_low, 2),
        "gap_golden_pocket": gap_gp,
        "slope_tolerance_mult": round(slope_tolerance_mult, 3),
        "elapsed_mins": round(elapsed_mins, 1)
    }

def detect_volume_profile_triggers(
    df: pd.DataFrame,
    vp_data: Dict[str, Any],
    ofi_data: Dict[str, Any],
    atr_14: float = 35.0
) -> Dict[str, Any]:
    """
    Evaluates Auction Market Theory (AMT) rejection & acceptance triggers:
    1. VAH Rejection (Bearish Fade): Probe > VAH, close back inside, upper wick >= 35%, OFI Z <= 0.10.
    2. VAL Rejection (Bullish Defense): Probe < VAL, close back inside, lower wick >= 35%, OFI Z >= -0.10.
    3. Value Area Breakout (Initiative Expansion): 2 consecutive closes outside VA + OFI confirmation.
    """
    if df.empty or vp_data.get("poc", 0) == 0:
        return {"trigger": "NONE", "side": "NEUTRAL", "confidence": 0.0, "reason": "Insufficient VP data"}
        
    last_bar = df.iloc[-1]
    c_open, c_high, c_low, c_close = float(last_bar["open"]), float(last_bar["high"]), float(last_bar["low"]), float(last_bar["close"])
    c_range = max(c_high - c_low, 1.0)
    
    vah = vp_data.get("vah", c_close)
    val = vp_data.get("val", c_close)
    poc = vp_data.get("poc", c_close)
    ofi_z = ofi_data.get("ofi_zscore", 0.0)
    
    upper_wick_ratio = (c_high - max(c_open, c_close)) / c_range
    lower_wick_ratio = (min(c_open, c_close) - c_low) / c_range
    
    # 1. VAH Rejection (Bearish Fade)
    if c_high >= vah and c_close < vah and upper_wick_ratio >= 0.35 and ofi_z <= 0.10:
        return {
            "trigger": "VAH_REJECTION",
            "side": "SHORT",
            "confidence": 0.85,
            "entry": c_close,
            "sl": round(c_high + 5.0, 2),
            "target_1": poc,
            "target_2": val,
            "reason": f"VAH Rejection confirmed: Probe above VAH ({vah:.2f}) rejected with {upper_wick_ratio*100:.1f}% upper wick & OFI Z={ofi_z:.2f}."
        }
        
    # 2. VAL Rejection (Bullish Defense / Spring)
    if c_low <= val and c_close > val and lower_wick_ratio >= 0.35 and ofi_z >= -0.10:
        return {
            "trigger": "VAL_REJECTION",
            "side": "LONG",
            "confidence": 0.85,
            "entry": c_close,
            "sl": round(c_low - 5.0, 2),
            "target_1": poc,
            "target_2": vah,
            "reason": f"VAL Rejection confirmed: Probe below VAL ({val:.2f}) defended with {lower_wick_ratio*100:.1f}% lower wick & OFI Z={ofi_z:.2f}."
        }
        
    # 3. Value Area Breakout (Initiative Expansion)
    if len(df) >= 2:
        prev_bar = df.iloc[-2]
        if float(prev_bar["close"]) > vah and c_close > vah and ofi_z >= OFI_ZSCORE_MIN:
            return {
                "trigger": "VA_EXPANSION_BULLISH",
                "side": "LONG",
                "confidence": 0.80,
                "entry": c_close,
                "sl": round(vah - 5.0, 2),
                "target_1": round(c_close + 1.5 * atr_14, 2),
                "target_2": round(c_close + 3.0 * atr_14, 2),
                "reason": f"Value Area Bullish Breakout: 2 consecutive closes above VAH ({vah:.2f}) with aggressive OFI Z={ofi_z:.2f}."
            }
        elif float(prev_bar["close"]) < val and c_close < val and ofi_z <= -OFI_ZSCORE_MIN:
            return {
                "trigger": "VA_EXPANSION_BEARISH",
                "side": "SHORT",
                "confidence": 0.80,
                "entry": c_close,
                "sl": round(val + 5.0, 2),
                "target_1": round(c_close - 1.5 * atr_14, 2),
                "target_2": round(c_close - 3.0 * atr_14, 2),
                "reason": f"Value Area Bearish Breakdown: 2 consecutive closes below VAL ({val:.2f}) with aggressive OFI Z={ofi_z:.2f}."
            }
            
    return {"trigger": "IN_VALUE", "side": "NEUTRAL", "confidence": 0.50, "reason": "Auction trading within Value Area equilibrium."}

def compute_dealer_gex(spot: float, call_oi: float = 14500000.0, put_oi: float = 12800000.0) -> Dict[str, Any]:
    """Computes Net Dealer Gamma Exposure (GEX) in ₹ Crores and Gamma Flip Level."""
    net_oi_diff = (call_oi - put_oi) / 100000.0
    gex_crores = net_oi_diff * (spot / 24000.0) * 4.5
    is_positive_gex = gex_crores >= 0
    flip_strike = int(round(spot / 50.0) * 50) + (50 if not is_positive_gex else -50)
    
    return {
        "net_gex_crores": round(gex_crores, 2),
        "is_positive_gamma": is_positive_gex,
        "gamma_regime": "POSITIVE GAMMA (MM Long Gamma -> S/R Pinning)" if is_positive_gex else "NEGATIVE GAMMA (MM Short Gamma -> High Velocity Breakouts)",
        "gamma_flip_strike": flip_strike
    }


