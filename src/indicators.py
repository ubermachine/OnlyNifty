import math
from typing import Dict, Tuple, Any, List, Optional
import numpy as np
import pandas as pd
from src.config import (
    EMA_FAST, EMA_MID, EMA_SLOW, VAKC_LAMBDA, VAKC_ATR_SPAN,
    HURST_TRENDING_MIN, HURST_MEAN_REV_MAX, DEFAULT_IV, OFI_ZSCORE_MIN,
    VPIN_TOXICITY_THRESHOLD, LINE_BREAK_COUNT, LINE_BREAK_EMA_PERIODS
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
    with finite-sample Anis-Lloyd / Peters Bias Correction (Vectorized NumPy).
    H > 0.52 -> Persistent / Trending Regime (Golden Pocket Active)
    H < 0.45 -> Anti-Persistent / Mean-Reverting Regime
    0.45 <= H <= 0.52 -> Random Walk / Noise
    """
    if len(series) < max_lag + 10:
        return {"hurst": 0.50, "regime": "RANDOM_WALK (Accumulating Data)", "is_trending": False}
        
    prices = series.values[-100:] if len(series) >= 100 else series.values
    # Compute continuously compounded log-returns for weak-sense stationarity
    log_returns = np.diff(np.log(np.maximum(prices, 1.0)))
    n_ret = len(log_returns)
    
    if n_ret < max_lag:
        return {"hurst": 0.50, "regime": "RANDOM_WALK (Accumulating Data)", "is_trending": False}

    lags = np.unique(np.linspace(min_lag, min(max_lag, n_ret // 2), 16, dtype=int))
    rs_values = []
    valid_lags = []
    
    for lag in lags:
        n_chunks = n_ret // lag
        if n_chunks < 1:
            continue
            
        # Fast 2D vectorized chunk operations
        chunks = log_returns[:n_chunks * lag].reshape(n_chunks, lag)
        means = chunks.mean(axis=1, keepdims=True)
        cum_dev = np.cumsum(chunks - means, axis=1)
        r = np.ptp(cum_dev, axis=1)
        s = np.std(chunks, axis=1, ddof=1)
        
        valid = s > 1e-8
        if np.any(valid):
            raw_rs = float(np.mean(r[valid] / s[valid]))
            expected_rs = np.sqrt((lag - 0.5) / (np.pi * 0.5)) if lag > 2 else 1.0
    if len(rs_values) >= 3:
        safe_rs = np.maximum(np.array(rs_values, dtype=float), 1e-6)
        poly = np.polyfit(np.log(valid_lags), np.log(safe_rs), 1)
        h = float(np.clip(poly[0] + 0.50, 0.10, 0.90))
        if np.isnan(h) or np.isinf(h):
            h = 0.50
    else:
        h = 0.50
        
    if h >= HURST_TRENDING_MIN:
        regime = f"PERSISTENT TRENDING (H={h:.2f} >= 0.52 - Golden Pocket Active)"
        is_trending = True
    elif h < HURST_MEAN_REV_MAX:
        regime = f"ANTI-PERSISTENT MEAN-REVERTING (H={h:.2f} < 0.45 - Fade Extremes)"
        is_trending = False
    else:
        regime = f"RANDOM WALK NOISE (H={h:.2f} in [0.45, 0.52] - Range Bound)"
        is_trending = False
        
    return {
        "hurst": round(h, 4),
        "regime": regime,
        "is_trending": is_trending,
        "r_squared_proxy": round(float(np.corrcoef(np.log(valid_lags), np.log(rs_values))[0, 1] ** 2) if len(rs_values) >= 3 else 0.80, 2)
    }

def compute_vakc_envelopes(
    df: pd.DataFrame,
    ema_span: int = EMA_SLOW,
    atr_span: int = VAKC_ATR_SPAN,
    k: float = VAKC_LAMBDA,
    iv: Union[float, pd.Series, None] = DEFAULT_IV
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
    if isinstance(iv, pd.Series):
        clamped_iv = iv.clip(lower=0.06)
    elif "iv" in df.columns and (iv is None or isinstance(iv, str)):
        clamped_iv = df["iv"].clip(lower=0.06)
    else:
        clamped_iv = max(iv if iv is not None else DEFAULT_IV, 0.06)
        
    vol_ratio = clamped_iv / 0.12
    if isinstance(vol_ratio, pd.Series):
        vol_scalar = (vol_ratio ** 0.45) * (1.0 + 0.12 * np.tanh((clamped_iv - 0.18) / 0.06))
    else:
        vol_scalar = (vol_ratio ** 0.45) * (1.0 + 0.12 * np.tanh((clamped_iv - 0.18) / 0.06))
    
    band_width = (k * atr * vol_scalar).fillna(ema200 * 0.015)
    
    upper_vakc = ema200 + band_width
    lower_vakc = ema200 - band_width
    return upper_vakc, lower_vakc


def compute_vwap(df: pd.DataFrame, anchor_session: bool = True) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Computes Session Anchored VWAP and Exact Numerically Stable Online Variance (±2σ dispersion bands)."""
    if df.empty:
        s = pd.Series(dtype=float)
        return s, s, s
        
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    vol_raw = df["volume"].to_numpy(dtype=float) if "volume" in df.columns else np.ones(len(df))
    
    typical_price = (high + low + close) / 3.0
    vol = np.where(vol_raw > 0, vol_raw, np.maximum(high - low, 1.0))
    
    vwap_arr = np.zeros(len(df), dtype=float)
    variance_arr = np.zeros(len(df), dtype=float)
    
    if anchor_session and isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
        day_nums = df.index.dayofyear.to_numpy()
        session_breaks = np.concatenate(([0], np.where(np.diff(day_nums) != 0)[0] + 1))
        
        for i in range(len(session_breaks)):
            start_i = session_breaks[i]
            end_i = session_breaks[i + 1] if i + 1 < len(session_breaks) else len(df)
            
            p0 = typical_price[start_i]
            delta = typical_price[start_i:end_i] - p0
            w = vol[start_i:end_i]
            
            cum_w = np.cumsum(w)
            cum_w_safe = np.maximum(cum_w, 1.0)
            cum_delta_w = np.cumsum(delta * w)
            cum_delta_sq_w = np.cumsum((delta ** 2) * w)
            
            delta_bar = cum_delta_w / cum_w_safe
            vwap_arr[start_i:end_i] = p0 + delta_bar
            variance_arr[start_i:end_i] = np.maximum((cum_delta_sq_w / cum_w_safe) - (delta_bar ** 2), 0.0)
    else:
        p0 = typical_price[0]
        delta = typical_price - p0
        w = vol
        cum_w = np.cumsum(w)
        cum_w_safe = np.maximum(cum_w, 1.0)
        cum_delta_w = np.cumsum(delta * w)
        cum_delta_sq_w = np.cumsum((delta ** 2) * w)
        
        delta_bar = cum_delta_w / cum_w_safe
        vwap_arr = p0 + delta_bar
        variance_arr = np.maximum((cum_delta_sq_w / cum_w_safe) - (delta_bar ** 2), 0.0)
        
    std_dev_arr = np.sqrt(variance_arr)
    
    vwap = pd.Series(vwap_arr, index=df.index)
    upper_sd = pd.Series(vwap_arr + (2.0 * std_dev_arr), index=df.index)
    lower_sd = pd.Series(vwap_arr - (2.0 * std_dev_arr), index=df.index)
    return vwap, upper_sd, lower_sd


def compute_vwap_multi_dispersion_and_half_life(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes Full Multi-Sigma VWAP Dispersion Envelopes (±1σ, ±2σ, ±3σ),
    Real-Time VWAP Z-Score (Z_vwap), and Vectorized Ornstein-Uhlenbeck (OU) Half-Life (τ_1/2).
    """
    if df.empty or len(df) < 5:
        return {
            "vwap": 0.0, "std_dev": 1.0, "sigma_1_up": 0.0, "sigma_1_down": 0.0,
            "sigma_2_up": 0.0, "sigma_2_down": 0.0, "sigma_3_up": 0.0, "sigma_3_down": 0.0,
            "z_score_vwap": 0.0, "half_life_bars": 12.0, "half_life_mins": 60.0,
            "mean_reverting_urgency": "NORMAL"
        }

    vwap_series, upper_2sd, lower_2sd = compute_vwap(df, anchor_session=True)
    std_series = (upper_2sd - vwap_series) / 2.0
    
    curr_close = float(df["close"].iloc[-1])
    curr_vwap = float(vwap_series.iloc[-1])
    curr_std = max(float(std_series.iloc[-1]), 1.0)

    z_score = (curr_close - curr_vwap) / curr_std
    
    # 1σ, 2σ, 3σ bands
    s1_up = round(curr_vwap + 1.0 * curr_std, 2)
    s1_down = round(curr_vwap - 1.0 * curr_std, 2)
    s2_up = round(curr_vwap + 2.0 * curr_std, 2)
    s2_down = round(curr_vwap - 2.0 * curr_std, 2)
    s3_up = round(curr_vwap + 3.0 * curr_std, 2)
    s3_down = round(curr_vwap - 3.0 * curr_std, 2)

    # Fast Analytical OU Mean-Reversion Half-Life Estimation
    price_diff = (df["close"] - vwap_series).to_numpy(dtype=float)
    n_pts = len(price_diff)
    if n_pts >= 15:
        x = price_diff[:-1]
        y = np.diff(price_diff)
        n = len(x)
        x_c = x - np.mean(x)
        var_x = np.dot(x_c, x_c) / n
        if var_x > 1e-6:
            cov_xy = np.dot(x_c, y - np.mean(y)) / n
            beta = cov_xy / var_x
            if beta < 0:
                theta = -beta
                half_life_bars = round(math.log(2.0) / max(theta, 0.01), 1)
            else:
                half_life_bars = 45.0
        else:
            half_life_bars = 15.0
    else:
        half_life_bars = 12.0

    half_life_bars = float(np.clip(half_life_bars, 2.0, 60.0))
    half_life_mins = round(half_life_bars * 5.0, 1)

    if abs(z_score) >= 2.5 and half_life_bars <= 8.0:
        urgency = "HIGH_CONVICTION_MEAN_REVERSION_FADE"
    elif abs(z_score) <= 1.0:
        urgency = "FAIR_VALUE_CONSOLIDATION"
    else:
        urgency = "TREND_EXTENSION_ACTIVE"

    return {
        "vwap": round(curr_vwap, 2),
        "std_dev": round(curr_std, 2),
        "sigma_1_up": s1_up,
        "sigma_1_down": s1_down,
        "sigma_2_up": s2_up,
        "sigma_2_down": s2_down,
        "sigma_3_up": s3_up,
        "sigma_3_down": s3_down,
        "z_score_vwap": round(z_score, 2),
        "half_life_bars": half_life_bars,
        "half_life_mins": half_life_mins,
        "mean_reverting_urgency": urgency
    }


def detect_footprint_delta_divergences(
    df: pd.DataFrame,
    lookback: int = 15
) -> Dict[str, Any]:
    """
    Detects Institutional Footprint Delta Divergences:
    - Regular Bullish Divergence: Price Lower Low vs CVD Higher Low (Buyer Passive Absorption)
    - Regular Bearish Divergence: Price Higher High vs CVD Lower High (Seller Passive Absorption)
    """
    if df.empty or len(df) < lookback + 5:
        return {"divergence_detected": False, "type": "NONE", "bias": "NEUTRAL", "thesis": ""}

    sub = df.tail(lookback + 5)
    vol = sub["volume"].copy().astype(float).clip(lower=1.0)
    c_range = (sub["high"] - sub["low"]).clip(lower=1e-4)
    body_weight = (sub["close"] - sub["open"]) / c_range
    close_loc_weight = (2.0 * sub["close"] - sub["high"] - sub["low"]) / c_range
    bar_deltas = vol * ((0.60 * body_weight) + (0.40 * close_loc_weight))
    cvd = bar_deltas.cumsum()

    prices = sub["close"].values
    cvd_vals = cvd.values

    # Halfway split
    mid = len(prices) // 2
    p1_min, p1_max = np.min(prices[:mid]), np.max(prices[:mid])
    p2_min, p2_max = np.min(prices[mid:]), np.max(prices[mid:])

    cvd1_min, cvd1_max = np.min(cvd_vals[:mid]), np.max(cvd_vals[:mid])
    cvd2_min, cvd2_max = np.min(cvd_vals[mid:]), np.max(cvd_vals[mid:])

    # Bullish Divergence: Lower Low on price, Higher Low on CVD
    if (p2_min < p1_min - 3.0) and (cvd2_min > cvd1_min + 5000):
        return {
            "divergence_detected": True,
            "type": "REGULAR_BULLISH_DELTA_DIVERGENCE",
            "bias": "BULLISH_ABSORPTION_REVERSAL",
            "thesis": f"Spot made Lower Low ({p2_min:.1f} < {p1_min:.1f}) but Cumulative Volume Delta made Higher Low. Institutions passively absorbing selling."
        }

    # Bearish Divergence: Higher High on price, Lower High on CVD
    if (p2_max > p1_max + 3.0) and (cvd2_max < cvd1_max - 5000):
        return {
            "divergence_detected": True,
            "type": "REGULAR_BEARISH_DELTA_DIVERGENCE",
            "bias": "BEARISH_DISTRIBUTION_REVERSAL",
            "thesis": f"Spot made Higher High ({p2_max:.1f} > {p1_max:.1f}) but Cumulative Volume Delta made Lower High. Institutions distributing into retail FOMO."
        }

    return {"divergence_detected": False, "type": "NONE", "bias": "NEUTRAL", "thesis": "No Delta Divergence detected."}


def compute_order_flow_imbalance(df: pd.DataFrame, decay_lambda: float = 0.15) -> Dict[str, Any]:
    """
    Computes Composite Price-Wick Weighted Bar Delta & Rolling 20-Bar OFI Z-Score,
    augmented with a Hawkes Process Exponential Decay Kernel (e^(-lambda * dt))
    to model high-frequency order flow clustering and predictive liquidity shocks.
    """
    if df.empty or len(df) < 2:
        return {
            "ofi": 0.0, "ofi_zscore": 0.0, "cvd": 0.0,
            "buyer_defense": True, "seller_defense": True,
            "hawkes_ofi": 0.0, "is_hawkes_surge": False,
            "cvd_series": pd.Series()
        }
        
    vol = df["volume"].copy().astype(float)
    if vol.sum() == 0 or (vol == 0).all():
        vol = (df["high"] - df["low"]).clip(lower=1.0)
        
    c_range = (df["high"] - df["low"]).clip(lower=1e-4)
    # 3-Tier Composite Microstructure Delta:
    # 1. 50% Body Directional Displacement
    # 2. 30% Close Location relative to mid-point
    # 3. 20% Wick Absorption (Buyer Lower-Shadow Defense vs Seller Upper-Shadow Distribution)
    body_weight = (df["close"] - df["open"]) / c_range
    close_loc_weight = (2.0 * df["close"] - df["high"] - df["low"]) / c_range
    lower_wick = (df[["open", "close"]].min(axis=1) - df["low"]) / c_range
    upper_wick = (df["high"] - df[["open", "close"]].max(axis=1)) / c_range
    wick_bias = (lower_wick - upper_wick)

    composite_weight = (0.50 * body_weight) + (0.30 * close_loc_weight) + (0.20 * wick_bias)
    composite_weight = composite_weight.clip(-1.0, 1.0)
    
    bar_delta = vol * composite_weight
    cvd = bar_delta.cumsum()
    
    # Rolling 20-Bar OFI Z-Score
    rolling_mean = bar_delta.rolling(window=20, min_periods=3).mean()
    rolling_std = bar_delta.rolling(window=20, min_periods=3).std().clip(lower=1e-4)
    ofi_zscore_series = ((bar_delta - rolling_mean) / rolling_std).fillna(0.0)
    
    recent_z = float(ofi_zscore_series.iloc[-1])
    recent_ofi = float(bar_delta.tail(5).sum())

    # Hawkes Process Exponential Decay Kernel: w_k = exp(-lambda * (N - 1 - k))
    k_len = min(len(bar_delta), 15)
    decay_weights = np.exp(-decay_lambda * np.arange(k_len)[::-1])
    hawkes_ofi = float(np.sum(bar_delta.iloc[-k_len:].values * decay_weights))
    is_hawkes_surge = abs(recent_z) >= 1.50 or abs(hawkes_ofi) >= 5000.0
    
    return {
        "ofi": round(recent_ofi, 2),
        "ofi_zscore": round(recent_z, 3),
        "cvd": round(float(cvd.iloc[-1]), 2),
        "buyer_defense": recent_z >= -0.20,
        "seller_defense": recent_z <= 0.20,
        "hawkes_ofi": round(hawkes_ofi, 2),
        "is_hawkes_surge": is_hawkes_surge,
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
    
    if len(daily_df) > 1 and hasattr(daily_df.index, "date"):
        try:
            daily_resampled = daily_df.resample("D").agg({
                "open": "first", "high": "max", "low": "min", "close": "last"
            }).dropna()
            if len(daily_resampled) >= 2:
                last = daily_resampled.iloc[-2]  # Prior completed session
            elif not daily_resampled.empty:
                last = daily_resampled.iloc[-1]
            else:
                last = daily_df.iloc[-1]
        except Exception:
            last = daily_df.iloc[-2] if len(daily_df) >= 2 else daily_df.iloc[-1]
    elif len(daily_df) >= 2:
        last = daily_df.iloc[-2]
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
    """Computes Dual-Bracket 70% Volume Profile, POC, VAH, and VAL (Vectorized NumPy)."""
    if df.empty or len(df) < 3:
        return {"poc": 0.0, "vah": 0.0, "val": 0.0, "bins": [], "volumes": []}
    
    lows = df["low"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    vols = df["volume"].to_numpy(dtype=float)
    
    price_min = float(np.min(lows))
    price_max = float(np.max(highs))
    if np.isclose(price_min, price_max):
        return {"poc": price_min, "vah": price_max, "val": price_min, "bins": [price_min, price_max], "volumes": [int(np.sum(vols))]}
        
    bins = np.linspace(price_min, price_max, n_bins)
    bin_volumes = np.zeros(n_bins - 1, dtype=float)
    
    # Vectorized / fast array iteration without pandas overhead
    safe_vols = np.where(vols > 0, vols, highs - lows)
    idx_lows = np.clip(np.digitize(lows, bins) - 1, 0, n_bins - 2)
    idx_highs = np.clip(np.digitize(highs, bins) - 1, 0, n_bins - 2)
    spans = np.maximum(idx_highs - idx_lows + 1, 1)
    
    for i in range(len(lows)):
        v_share = safe_vols[i] / spans[i]
        bin_volumes[idx_lows[i]:idx_highs[i] + 1] += v_share
            
    poc_idx = int(np.argmax(bin_volumes))
    poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2.0
    
    total_vol = bin_volumes.sum()
    target_vol = total_vol * 0.70
    
    low_idx, high_idx = poc_idx, poc_idx
    curr_vol = bin_volumes[poc_idx]
    
    while curr_vol < target_vol and (low_idx > 0 or high_idx < len(bin_volumes) - 1):
        v_below = bin_volumes[low_idx - 1] if low_idx > 0 else 0.0
        v_above = bin_volumes[high_idx + 1] if high_idx < len(bin_volumes) - 1 else 0.0
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
    """
    Coarse Net Dealer GEX proxy in ₹ Crores, for use ONLY when no option chain is available.

    WARNING: the default call_oi/put_oi are placeholder constants with call_oi > put_oi, so
    calling this WITHOUT real OI always yields is_positive_gamma=True — a silent, permanent
    "dealers are long gamma / expect pinning" verdict. Callers must pass live OI, and must
    treat walls_verified=False as untrusted. Prefer compute_strike_level_gex_chart_data,
    which derives gamma per strike from the real chain.
    """
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


# =====================================================================
# ONLYNIFTY v3.3: MULTI-TIMEFRAME CONFLUENCE & FOOTPRINT IMBALANCES
# =====================================================================

def _resample_ohlcv_if_needed(df_source: pd.DataFrame, freq: str, bar_multiplier: int) -> pd.DataFrame:
    """Safely and swiftly resamples OHLCV DataFrame aligning to standard market clock."""
    if df_source.empty:
        return df_source
    if isinstance(df_source.index, pd.DatetimeIndex) and len(df_source) > 1:
        try:
            resampled = df_source.resample(freq, origin="start_day").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum"
            }).dropna()
            if not resampled.empty and len(resampled) >= 2:
                return resampled
        except Exception:
            pass

    n = len(df_source)
    k = max(int(bar_multiplier), 1)
    n_full = (n // k) * k
    if n_full < k:
        return df_source.copy()
        
    o_arr = df_source["open"].to_numpy(dtype=float)
    h_arr = df_source["high"].to_numpy(dtype=float)
    l_arr = df_source["low"].to_numpy(dtype=float)
    c_arr = df_source["close"].to_numpy(dtype=float)
    v_arr = df_source["volume"].to_numpy(dtype=float)
    
    opens = o_arr[:n_full].reshape(-1, k)[:, 0]
    highs = np.max(h_arr[:n_full].reshape(-1, k), axis=1)
    lows = np.min(l_arr[:n_full].reshape(-1, k), axis=1)
    closes = c_arr[:n_full].reshape(-1, k)[:, -1]
    volumes = np.sum(v_arr[:n_full].reshape(-1, k), axis=1)
    
    if n > n_full:
        opens = np.append(opens, o_arr[n_full])
        highs = np.append(highs, np.max(h_arr[n_full:]))
        lows = np.append(lows, np.min(l_arr[n_full:]))
        closes = np.append(closes, c_arr[-1])
        volumes = np.append(volumes, np.sum(v_arr[n_full:]))
        
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes
    })


def compute_line_break(
    df: pd.DataFrame,
    lines: int = LINE_BREAK_COUNT,
    price_col: str = "close"
) -> pd.DataFrame:
    """
    Computes causal, strictly non-repainting N-Line Break blocks.
    
    Mathematical & Algorithmic Specification:
    - State is a sequential list of formed blocks: (low_b, high_b, dir_b).
    - Blocks are price events, not time bars; most bars produce no new block.
    - Seeding: hold the first close as reference c0; the first block forms on the first close != c0.
    - Continuation:
        If current direction is UP (+1) and close > high_b:
            forms new UP block (high_b, close, +1).
        If current direction is DOWN (-1) and close < low_b:
            forms new DOWN block (close, low_b, -1).
    - Asymmetric Reversal:
        Reversal requires close to breach the extreme of the last N blocks:
            rev_up = max(high of blocks in window of last N blocks)
            rev_dn = min(low of blocks in window of last N blocks)
        If current direction is UP (+1) and close < rev_dn:
            forms new DOWN reversal block (close, rev_dn, -1).
        If current direction is DOWN (-1) and close > rev_up:
            forms new UP reversal block (rev_up, close, +1).
    - Causal / Prefix-Stable Invariant:
        Row i depends strictly on close[0..i]. State at row i-1 is identical whether
        computed on df[:i] or full df.
    """
    if df.empty or price_col not in df.columns:
        return pd.DataFrame(index=df.index, columns=[
            "lb_direction", "lb_high", "lb_low", "lb_reversal_up",
            "lb_reversal_dn", "lb_blocks_count", "lb_flipped"
        ])

    closes = df[price_col].values.astype(float)
    n = len(closes)

    lb_dir = np.zeros(n, dtype=int)
    lb_high = np.zeros(n, dtype=float)
    lb_low = np.zeros(n, dtype=float)
    lb_rev_up = np.zeros(n, dtype=float)
    lb_rev_dn = np.zeros(n, dtype=float)
    lb_count = np.zeros(n, dtype=int)
    lb_flip = np.zeros(n, dtype=bool)

    blocks: List[Tuple[float, float, int]] = []
    c0 = closes[0]

    for i in range(n):
        c = closes[i]
        flipped = False

        if not blocks:
            if c > c0:
                blocks.append((c0, c, 1))
            elif c < c0:
                blocks.append((c, c0, -1))
            else:
                lb_dir[i] = 0
                lb_high[i] = c0
                lb_low[i] = c0
                lb_rev_up[i] = c0
                lb_rev_dn[i] = c0
                lb_count[i] = 0
                lb_flip[i] = False
                continue

        low_b, high_b, dir_b = blocks[-1]
        w_blocks = blocks[-lines:]
        rev_up = max(b[1] for b in w_blocks)
        rev_dn = min(b[0] for b in w_blocks)

        if dir_b == 1:
            if c > high_b:
                blocks.append((high_b, c, 1))
                low_b, high_b, dir_b = blocks[-1]
            elif c < rev_dn:
                blocks.append((c, rev_dn, -1))
                low_b, high_b, dir_b = blocks[-1]
                flipped = True
        elif dir_b == -1:
            if c < low_b:
                blocks.append((c, low_b, -1))
                low_b, high_b, dir_b = blocks[-1]
            elif c > rev_up:
                blocks.append((rev_up, c, 1))
                low_b, high_b, dir_b = blocks[-1]
                flipped = True

        w_blocks = blocks[-lines:]
        curr_rev_up = max(b[1] for b in w_blocks)
        curr_rev_dn = min(b[0] for b in w_blocks)

        lb_dir[i] = dir_b
        lb_high[i] = high_b
        lb_low[i] = low_b
        lb_rev_up[i] = curr_rev_up
        lb_rev_dn[i] = curr_rev_dn
        lb_count[i] = len(blocks)
        lb_flip[i] = flipped

    return pd.DataFrame({
        "lb_direction": lb_dir,
        "lb_high": lb_high,
        "lb_low": lb_low,
        "lb_reversal_up": lb_rev_up,
        "lb_reversal_dn": lb_rev_dn,
        "lb_blocks_count": lb_count,
        "lb_flipped": lb_flip
    }, index=df.index)


def compute_line_break_trend(
    df: pd.DataFrame,
    lines: int = LINE_BREAK_COUNT,
    ema_periods: Tuple[int, int, int] = LINE_BREAK_EMA_PERIODS,
    price_col: str = "close"
) -> pd.DataFrame:
    """
    Synthesizes 6-Line Break direction with monotonic EMA(15, 20, 50) stack.
    
    Rules:
    - Bullish Stack: close >= ema15 >= ema20 >= ema50
    - Bearish Stack: close <= ema15 <= ema20 <= ema50
    - Combined Bias:
        'BULLISH' if lb_direction == +1 and Bullish Stack
        'BEARISH' if lb_direction == -1 and Bearish Stack
        'NEUTRAL' on any disagreement (chop suppression)
    """
    if df.empty or price_col not in df.columns:
        return pd.DataFrame(index=df.index, columns=[
            "lb_bias", "lb_direction", "ema_stack_bullish", "ema_stack_bearish"
        ])

    lb_df = compute_line_break(df, lines=lines, price_col=price_col)
    
    p1, p2, p3 = ema_periods
    ema1 = compute_ema(df[price_col], min(p1, max(len(df), 2)))
    ema2 = compute_ema(df[price_col], min(p2, max(len(df), 2)))
    ema3 = compute_ema(df[price_col], min(p3, max(len(df), 2)))
    c = df[price_col]

    stack_bull = (c >= ema1) & (ema1 >= ema2) & (ema2 >= ema3)
    stack_bear = (c <= ema1) & (ema1 <= ema2) & (ema2 <= ema3)

    dirs = lb_df["lb_direction"].values
    s_bull = stack_bull.values
    s_bear = stack_bear.values

    biases = np.where((dirs == 1) & s_bull, "BULLISH", np.where((dirs == -1) & s_bear, "BEARISH", "NEUTRAL"))

    res = lb_df.copy()
    res["ema_stack_bullish"] = stack_bull
    res["ema_stack_bearish"] = stack_bear
    res["lb_bias"] = biases
    return res


def _evaluate_single_tf_regime(df_tf: pd.DataFrame, tf_name: str) -> Dict[str, Any]:
    """Evaluates EMA200, EMA55, EMA21, AVWAP, and Hurst Exponent for a single timeframe."""
    if df_tf.empty or len(df_tf) < 1:
        return {
            "tf": tf_name,
            "bias": "NEUTRAL",
            "close": 0.0,
            "ema200": 0.0,
            "ema55": 0.0,
            "ema21": 0.0,
            "vwap": 0.0,
            "hurst": 0.50,
            "is_trending": False,
            "ema_slope": 0.0,
            "summary": f"{tf_name}: Insufficient data"
        }
        
    close = float(df_tf["close"].iloc[-1])
    n = len(df_tf)
    
    ema200_span = min(EMA_SLOW, max(n, 5))
    ema55_span = min(EMA_MID, max(n, 5))
    ema21_span = min(EMA_FAST, max(n, 3))
    
    ema200_s = compute_ema(df_tf["close"], ema200_span)
    ema55_s = compute_ema(df_tf["close"], ema55_span)
    ema21_s = compute_ema(df_tf["close"], ema21_span)
    
    ema200 = float(ema200_s.iloc[-1])
    ema55 = float(ema55_s.iloc[-1])
    ema21 = float(ema21_s.iloc[-1])
    
    if n >= 2:
        ema_slope = float((ema21_s.iloc[-1] - ema21_s.iloc[-2]) / max(ema21_s.iloc[-2], 1.0))
    else:
        ema_slope = 0.0
        
    vwap_s, _, _ = compute_vwap(df_tf, anchor_session=True)
    vwap = float(vwap_s.iloc[-1])
    
    hurst_info = compute_hurst_exponent(df_tf["close"])
    hurst_val = hurst_info.get("hurst", 0.50)
    is_trending = hurst_info.get("is_trending", False)
    
    is_above_200 = close >= ema200
    is_above_vwap = close >= vwap
    is_bullish_ma_stack = ema21 >= ema55
    
    if is_above_200 and is_above_vwap and (is_bullish_ma_stack or ema_slope >= 0):
        bias = "BULLISH"
    elif is_above_200 or (is_above_vwap and ema_slope >= 0):
        bias = "NEUTRAL_BULLISH"
    elif (not is_above_200) and (not is_above_vwap) and ((not is_bullish_ma_stack) or ema_slope <= 0):
        bias = "BEARISH"
    elif (not is_above_200) or ((not is_above_vwap) and ema_slope <= 0):
        bias = "NEUTRAL_BEARISH"
    else:
        bias = "NEUTRAL"

    df_lb = df_tf.iloc[-80:] if len(df_tf) > 80 else df_tf
    lb_trend_df = compute_line_break_trend(df_lb)
    lb_bias = "NEUTRAL"
    lb_direction = 0
    if not lb_trend_df.empty:
        lb_bias = str(lb_trend_df["lb_bias"].iloc[-1])
        lb_direction = int(lb_trend_df["lb_direction"].iloc[-1])
        
    return {
        "tf": tf_name,
        "bias": bias,
        "close": round(close, 2),
        "ema200": round(ema200, 2),
        "ema55": round(ema55, 2),
        "ema21": round(ema21, 2),
        "vwap": round(vwap, 2),
        "hurst": round(hurst_val, 4),
        "is_trending": is_trending,
        "ema_slope": round(ema_slope * 1000.0, 4),
        "lb_bias": lb_bias,
        "lb_direction": lb_direction,
        "summary": f"{tf_name}: {bias} | C={close:.2f}, EMA200={ema200:.2f}, AVWAP={vwap:.2f}, H={hurst_val:.2f}, LB={lb_bias}"
    }


def compute_multi_timeframe_regime(
    df_5m: pd.DataFrame,
    df_15m: Optional[pd.DataFrame] = None,
    df_1h: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Formulates OnlyNifty v3.3 Multi-Timeframe Alignment Engine (1H + 15m + 5m Confluence).
    Computes higher timeframe 1H/15m trend bias (EMA200, AVWAP, and Hurst exponent H).
    Enforces HTF alignment:
    - 5m Longs allowed ONLY if 15m and 1H trends are Bullish or Neutral-Bullish.
    - 5m Shorts allowed ONLY if 15m and 1H trends are Bearish or Neutral-Bearish.
    """
    if df_5m.empty:
        return {
            "htf_aligned_long": False,
            "htf_aligned_short": False,
            "confluence_regime": "INSUFFICIENT_DATA",
            "tf_1h": {},
            "tf_15m": {},
            "tf_5m": {},
            "alignment_score": 0.0,
            "reason": "Empty 5m DataFrame provided."
        }
        
    if df_15m is None or df_15m.empty:
        df_15m = _resample_ohlcv_if_needed(df_5m, freq="15min", bar_multiplier=3)
    if df_1h is None or df_1h.empty:
        df_1h = _resample_ohlcv_if_needed(df_5m, freq="1h", bar_multiplier=12)
        
    tf_1h_regime = _evaluate_single_tf_regime(df_1h, "1H")
    tf_15m_regime = _evaluate_single_tf_regime(df_15m, "15m")
    tf_5m_regime = _evaluate_single_tf_regime(df_5m, "5m")
    
    bias_1h = tf_1h_regime["bias"]
    bias_15m = tf_15m_regime["bias"]
    bias_5m = tf_5m_regime["bias"]
    
    bullish_biases = {"BULLISH", "NEUTRAL_BULLISH"}
    bearish_biases = {"BEARISH", "NEUTRAL_BEARISH"}
    
    htf_aligned_long = (bias_1h in bullish_biases) and (bias_15m in bullish_biases)
    htf_aligned_short = (bias_1h in bearish_biases) and (bias_15m in bearish_biases)
    
    score = 0.0
    if bias_1h == "BULLISH": score += 0.40
    elif bias_1h == "NEUTRAL_BULLISH": score += 0.20
    elif bias_1h == "BEARISH": score -= 0.40
    elif bias_1h == "NEUTRAL_BEARISH": score -= 0.20
    
    if bias_15m == "BULLISH": score += 0.35
    elif bias_15m == "NEUTRAL_BULLISH": score += 0.175
    elif bias_15m == "BEARISH": score -= 0.35
    elif bias_15m == "NEUTRAL_BEARISH": score -= 0.175
    
    if bias_5m == "BULLISH": score += 0.25
    elif bias_5m == "NEUTRAL_BULLISH": score += 0.125
    elif bias_5m == "BEARISH": score -= 0.25
    elif bias_5m == "NEUTRAL_BEARISH": score -= 0.125
    
    if htf_aligned_long and bias_5m in bullish_biases:
        confluence_regime = "FULL_BULLISH_CONFLUENCE (1H + 15m + 5m)"
    elif htf_aligned_short and bias_5m in bearish_biases:
        confluence_regime = "FULL_BEARISH_CONFLUENCE (1H + 15m + 5m)"
    elif htf_aligned_long:
        confluence_regime = "HTF_BULLISH_ALIGNMENT (1H + 15m Bullish | 5m Pullback)"
    elif htf_aligned_short:
        confluence_regime = "HTF_BEARISH_ALIGNMENT (1H + 15m Bearish | 5m Pullback)"
    else:
        confluence_regime = "HTF_MIXED_OR_CHOP (Conflicting 1H vs 15m Regimes)"
        
    reason = (
        f"Multi-Timeframe Alignment: 1H={bias_1h}, 15m={bias_15m}, 5m={bias_5m}. "
        f"Long Permitted={htf_aligned_long}, Short Permitted={htf_aligned_short}."
    )
    
    return {
        "htf_aligned_long": htf_aligned_long,
        "htf_aligned_short": htf_aligned_short,
        "confluence_regime": confluence_regime,
        "alignment_score": round(score, 3),
        "tf_1h": tf_1h_regime,
        "tf_15m": tf_15m_regime,
        "tf_5m": tf_5m_regime,
        "reason": reason
    }


def detect_stacked_order_flow_imbalances(
    df: pd.DataFrame,
    key_levels: Optional[Dict[str, float]] = None,
    imbalance_ratio_threshold: float = 2.5,
    min_stacked_bars: int = 3,
    lookback: int = 15
) -> Dict[str, Any]:
    """
    Formulates OnlyNifty v3.3 Stacked Footprint Order Flow Imbalance & Absorption Detector.
    1. Reconstructs bar-by-bar aggressive buy vs sell volume via microstructure body/wick decomposition.
    2. Identifies consecutive stacked aggressive buy/sell deltas (Stacked Order Flow Imbalances).
    3. Detects footprint absorption wicks at Key Levels (CPR, VAH, VAL, AVWAP, +/-2SD).
    """
    if df.empty or len(df) < 2:
        return {
            "has_stacked_buy": False,
            "has_stacked_sell": False,
            "stacked_buy_count": 0,
            "stacked_sell_count": 0,
            "stacked_support_zone": None,
            "stacked_resistance_zone": None,
            "absorption_event": None,
            "order_flow_bias": "NEUTRAL",
            "recent_delta": 0.0,
            "summary": "Insufficient data for Order Flow Footprint analysis."
        }
        
    sub = df.tail(lookback).copy()
    vol = sub["volume"].copy().astype(float)
    if vol.sum() == 0 or (vol == 0).all():
        vol = (sub["high"] - sub["low"]).clip(lower=1.0) * 1000.0
    else:
        vol = vol.replace(0, 1.0)
        
    c_range = (sub["high"] - sub["low"]).replace(0, 1.0)
    body = sub["close"] - sub["open"]
    close_pos = (sub["close"] - sub["low"]) / c_range
    
    buy_frac = np.clip(0.50 + 0.35 * (body / c_range) + 0.15 * (2.0 * close_pos - 1.0), 0.05, 0.95)
    buy_vol = vol * buy_frac
    sell_vol = vol * (1.0 - buy_frac)
    bar_deltas = buy_vol - sell_vol
    
    buy_imbalance_ratio = buy_vol / np.maximum(sell_vol, 1.0)
    sell_imbalance_ratio = sell_vol / np.maximum(buy_vol, 1.0)
    
    n_bars = len(sub)
    buy_stacked_count = 0
    sell_stacked_count = 0
    
    for i in range(n_bars - 1, -1, -1):
        if buy_imbalance_ratio.iloc[i] >= imbalance_ratio_threshold or (bar_deltas.iloc[i] > 0 and buy_frac.iloc[i] >= 0.70):
            buy_stacked_count += 1
        else:
            break
            
    for i in range(n_bars - 1, -1, -1):
        if sell_imbalance_ratio.iloc[i] >= imbalance_ratio_threshold or (bar_deltas.iloc[i] < 0 and buy_frac.iloc[i] <= 0.30):
            sell_stacked_count += 1
        else:
            break
            
    has_stacked_buy = buy_stacked_count >= min_stacked_bars or (buy_stacked_count >= 2 and buy_imbalance_ratio.iloc[-1] >= 3.0)
    has_stacked_sell = sell_stacked_count >= min_stacked_bars or (sell_stacked_count >= 2 and sell_imbalance_ratio.iloc[-1] >= 3.0)
    
    stacked_support_zone = None
    if has_stacked_buy:
        recent_buy_bars = sub.iloc[-buy_stacked_count:]
        stacked_support_zone = (
            round(float(recent_buy_bars["low"].min()), 2),
            round(float(recent_buy_bars["open"].mean()), 2)
        )
        
    stacked_resistance_zone = None
    if has_stacked_sell:
        recent_sell_bars = sub.iloc[-sell_stacked_count:]
        stacked_resistance_zone = (
            round(float(recent_sell_bars["open"].mean()), 2),
            round(float(recent_sell_bars["high"].max()), 2)
        )
        
    absorption_event = None
    if key_levels:
        last_bar = sub.iloc[-1]
        c_open, c_high, c_low, c_close = float(last_bar["open"]), float(last_bar["high"]), float(last_bar["low"]), float(last_bar["close"])
        bar_range = max(c_high - c_low, 1.0)
        upper_wick_ratio = (c_high - max(c_open, c_close)) / bar_range
        lower_wick_ratio = (min(c_open, c_close) - c_low) / bar_range
        last_delta = float(bar_deltas.iloc[-1])
        
        for lvl_name, lvl_price in key_levels.items():
            if lvl_price <= 0:
                continue
                
            if c_low <= (lvl_price + 3.0) and c_close >= (lvl_price - 2.0) and lower_wick_ratio >= 0.35:
                absorption_event = {
                    "type": "BUYER_ABSORPTION",
                    "side": "LONG",
                    "level_name": lvl_name,
                    "level_price": round(lvl_price, 2),
                    "wick_ratio": round(lower_wick_ratio * 100.0, 1),
                    "delta": round(last_delta, 1),
                    "suggested_sl": round(c_low - 5.0, 2),
                    "reason": f"Buyer Absorption at {lvl_name} ({lvl_price:.2f}): {lower_wick_ratio*100.0:.1f}% lower wick absorption."
                }
                break
                
            if c_high >= (lvl_price - 3.0) and c_close <= (lvl_price + 2.0) and upper_wick_ratio >= 0.35:
                absorption_event = {
                    "type": "SELLER_ABSORPTION",
                    "side": "SHORT",
                    "level_name": lvl_name,
                    "level_price": round(lvl_price, 2),
                    "wick_ratio": round(upper_wick_ratio * 100.0, 1),
                    "delta": round(last_delta, 1),
                    "suggested_sl": round(c_high + 5.0, 2),
                    "reason": f"Seller Absorption at {lvl_name} ({lvl_price:.2f}): {upper_wick_ratio*100.0:.1f}% upper wick absorption."
                }
                break

    if absorption_event and absorption_event["type"] == "BUYER_ABSORPTION":
        order_flow_bias = "BULLISH_ABSORPTION"
    elif absorption_event and absorption_event["type"] == "SELLER_ABSORPTION":
        order_flow_bias = "BEARISH_ABSORPTION"
    elif has_stacked_buy:
        order_flow_bias = "AGGRESSIVE_BUYING_IMBALANCE"
    elif has_stacked_sell:
        order_flow_bias = "AGGRESSIVE_SELLING_IMBALANCE"
    else:
        order_flow_bias = "BALANCED_OR_NEUTRAL"

    summary_parts = []
    if has_stacked_buy:
        summary_parts.append(f"{buy_stacked_count} Stacked Buy Bars (Shelf: {stacked_support_zone})")
    if has_stacked_sell:
        summary_parts.append(f"{sell_stacked_count} Stacked Sell Bars (Shelf: {stacked_resistance_zone})")
    if absorption_event:
        summary_parts.append(absorption_event["reason"])
    if not summary_parts:
        summary_parts.append(f"Delta Balanced ({bar_deltas.iloc[-1]:.0f})")
        
    return {
        "has_stacked_buy": has_stacked_buy,
        "has_stacked_sell": has_stacked_sell,
        "stacked_buy_count": buy_stacked_count,
        "stacked_sell_count": sell_stacked_count,
        "stacked_support_zone": stacked_support_zone,
        "stacked_resistance_zone": stacked_resistance_zone,
        "absorption_event": absorption_event,
        "order_flow_bias": order_flow_bias,
        "recent_delta": round(float(bar_deltas.iloc[-1]), 2),
        "summary": " | ".join(summary_parts)
    }


def detect_iceberg_orders_and_liquidity_sweeps(
    df: pd.DataFrame,
    lookback_swing: int = 20,
    volume_surge_mult: float = 1.8,
    narrow_range_mult: float = 0.50
) -> Dict[str, Any]:
    """
    Microstructure Order Book Liquidity Engine:
    
    1. Algorithmic Iceberg Detection:
       Identifies hidden block orders where Volume / Average Volume >= 1.8x but Candle Range <= 0.50x ATR.
       Indicates heavy passive limit orders soaking up market velocity.
       
    2. Liquidity Sweep (Stop Hunt / False Breakout) Detection:
       Identifies price probing above prior 20-bar Swing High or below Swing Low, followed by aggressive
       intra-bar rejection and closing back inside the range with >= 40% wick.
    """
    if df.empty or len(df) < lookback_swing + 5:
        return {
            "iceberg_detected": False,
            "liquidity_sweep_detected": False,
            "iceberg_side": "NONE",
            "sweep_side": "NONE",
            "sweep_event": None,
            "iceberg_event": None,
            "microstructure_status": "NORMAL_LIQUIDITY"
        }

    sub = df.tail(lookback_swing + 5)
    bar = sub.iloc[-1]
    prev_bars = sub.iloc[:-1]

    c_open = float(bar["open"])
    c_high = float(bar["high"])
    c_low = float(bar["low"])
    c_close = float(bar["close"])
    c_vol = float(bar["volume"])
    
    candle_range = max(c_high - c_low, 1.0)
    atr = float((sub["high"] - sub["low"]).mean())
    avg_vol = float(prev_bars["volume"].mean()) if not prev_bars.empty else c_vol
    
    vol_ratio = c_vol / max(avg_vol, 1.0)
    range_ratio = candle_range / max(atr, 1.0)
    
    # 1. Iceberg Logic
    is_iceberg = (vol_ratio >= volume_surge_mult) and (range_ratio <= narrow_range_mult)
    iceberg_side = "NONE"
    iceberg_event = None

    if is_iceberg:
        if c_close >= c_open:
            iceberg_side = "BUY_ICEBERG_ACCUMULATION"
            thesis = f"Hidden Buyer Iceberg: {vol_ratio:.1f}x Vol absorbed within tight {candle_range:.1f} pt range."
        else:
            iceberg_side = "SELL_ICEBERG_DISTRIBUTION"
            thesis = f"Hidden Seller Iceberg: {vol_ratio:.1f}x Vol absorbed within tight {candle_range:.1f} pt range."
            
        iceberg_event = {
            "side": iceberg_side,
            "volume_multiple": round(vol_ratio, 2),
            "candle_range_pts": round(candle_range, 1),
            "range_atr_ratio": round(range_ratio, 2),
            "price_level": round((c_high + c_low) / 2.0, 2),
            "thesis": thesis
        }

    # 2. Liquidity Sweep Logic
    swing_high = float(prev_bars["high"].tail(lookback_swing).max())
    swing_low = float(prev_bars["low"].tail(lookback_swing).min())
    
    upper_wick = c_high - max(c_open, c_close)
    lower_wick = min(c_open, c_close) - c_low
    upper_wick_ratio = upper_wick / candle_range
    lower_wick_ratio = lower_wick / candle_range
    
    sweep_side = "NONE"
    sweep_event = None
    
    # Bearish Sweep: Pierced Swing High, closed back below with >= 40% upper wick
    if c_high > (swing_high + 1.5) and c_close < swing_high and upper_wick_ratio >= 0.40:
        sweep_side = "BEARISH_BUY_SIDE_LIQUIDITY_SWEEP (BSL Trap)"
        sweep_event = {
            "type": "BSL_SWEEP",
            "side": "SHORT",
            "swept_swing_high": round(swing_high, 2),
            "probe_high": round(c_high, 2),
            "wick_ratio_pct": round(upper_wick_ratio * 100.0, 1),
            "suggested_sl": round(c_high + 4.0, 2),
            "thesis": f"Buy-Side Liquidity Purged at {c_high:.1f} (+{c_high - swing_high:.1f} pts above swing high). Trapped breakout buyers."
        }
    # Bullish Sweep: Pierced Swing Low, closed back above with >= 40% lower wick
    elif c_low < (swing_low - 1.5) and c_close > swing_low and lower_wick_ratio >= 0.40:
        sweep_side = "BULLISH_SELL_SIDE_LIQUIDITY_SWEEP (SSL Trap)"
        sweep_event = {
            "type": "SSL_SWEEP",
            "side": "LONG",
            "swept_swing_low": round(swing_low, 2),
            "probe_low": round(c_low, 2),
            "wick_ratio_pct": round(lower_wick_ratio * 100.0, 1),
            "suggested_sl": round(c_low - 4.0, 2),
            "thesis": f"Sell-Side Liquidity Purged at {c_low:.1f} (-{swing_low - c_low:.1f} pts below swing low). Trapped breakdown sellers."
        }

    status_str = "NORMAL_LIQUIDITY"
    if sweep_event:
        status_str = sweep_side
    elif iceberg_event:
        status_str = iceberg_side

    return {
        "iceberg_detected": is_iceberg,
        "liquidity_sweep_detected": sweep_event is not None,
        "iceberg_side": iceberg_side,
        "sweep_side": sweep_side,
        "sweep_event": sweep_event,
        "iceberg_event": iceberg_event,
        "microstructure_status": status_str
    }


def compute_initial_balance_and_day_type(
    df_5m: pd.DataFrame,
    ib_bars: int = 12
) -> Dict[str, Any]:
    """
    Market Profile Initial Balance (IB: 09:15 - 10:15 IST) & Day Type Classification:
    
    Identifies:
    1. Initial Balance High, Low, and Range.
    2. Unilateral / Bilateral Range Expansion multiples (1.5x, 2.0x, 3.0x).
    3. Day Type:
       - TREND_DAY: Unilateral expansion >= 1.5x IB Range (Hold runners for T3).
       - NORMAL_VARIATION_DAY: Unilateral expansion between 0.5x and 1.5x.
       - NEUTRAL_DAY (Chop): Bilateral expansion on both sides (mean-reversion bracket).
       - NORMAL_DAY: Price contained strictly within Initial Balance.
    """
    if df_5m.empty:
        return {
            "ib_established": False, "ib_high": 0.0, "ib_low": 0.0, "ib_range": 0.0,
            "day_type": "ACCUMULATING_IB", "expansion_multiple": 0.0, "strategy_mode": "WAIT_FOR_IB"
        }

    ib_df = df_5m.iloc[:ib_bars] if len(df_5m) >= ib_bars else df_5m
    ib_high = float(ib_df["high"].max())
    ib_low = float(ib_df["low"].min())
    ib_range = max(ib_high - ib_low, 5.0)
    
    if len(df_5m) < ib_bars:
        return {
            "ib_established": False,
            "ib_high": round(ib_high, 2),
            "ib_low": round(ib_low, 2),
            "ib_range": round(ib_range, 2),
            "day_type": "ACCUMULATING_INITIAL_BALANCE (09:15-10:15 IST)",
            "expansion_multiple": 0.0,
            "strategy_mode": "STANDARD_5M_PULLBACKS"
        }

    curr_high = float(df_5m["high"].max())
    curr_low = float(df_5m["low"].min())
    
    exp_up = max(curr_high - ib_high, 0.0) / ib_range
    exp_down = max(ib_low - curr_low, 0.0) / ib_range
    total_exp = exp_up + exp_down

    if exp_up >= 1.0 and exp_down < 0.25:
        day_type = "BULLISH_TREND_DAY (Unilateral Upward Expansion)"
        strategy_mode = "HOLD_RUNNERS_FOR_T3_MOONSHOT"
    elif exp_down >= 1.0 and exp_up < 0.25:
        day_type = "BEARISH_TREND_DAY (Unilateral Downward Expansion)"
        strategy_mode = "HOLD_RUNNERS_FOR_T3_MOONSHOT"
    elif exp_up >= 0.40 and exp_down >= 0.40:
        day_type = "NEUTRAL_DAY (Bilateral Expansion / Whipsaw Bracket)"
        strategy_mode = "SCALP_AT_T1_AVOID_RUNNERS"
    elif exp_up >= 0.40 or exp_down >= 0.40:
        side_str = "Upward" if exp_up > exp_down else "Downward"
        day_type = f"NORMAL_VARIATION_DAY ({side_str} 1.5x IB Extension)"
        strategy_mode = "STANDARD_3TIER_LADDER"
    else:
        day_type = "NORMAL_DAY (Contained Within Initial Balance)"
        strategy_mode = "MEAN_REVERSION_AT_IB_EXTREMES"

    return {
        "ib_established": True,
        "ib_high": round(ib_high, 2),
        "ib_low": round(ib_low, 2),
        "ib_range": round(ib_range, 2),
        "day_type": day_type,
        "expansion_up_mult": round(exp_up, 2),
        "expansion_down_mult": round(exp_down, 2),
        "total_expansion_mult": round(total_exp, 2),
        "strategy_mode": strategy_mode
    }


def compute_dfa_alpha(series_or_returns: pd.Series, min_lag: int = 5, max_lag: int = 60) -> Dict[str, Any]:
    """
    Computes the Detrended Fluctuation Analysis (DFA) scaling exponent (Alpha).
    Isolates true long-term memory while detrending local polynomial trends and U-shaped intraday seasonality.
    Alpha ~ 0.5 (Random Walk / Efficient Market)
    Alpha > 0.52 (Persistent / Strong Trend Memory)
    Alpha < 0.48 (Anti-Persistent / Mean Reversion)
    """
    if len(series_or_returns) < min_lag * 3:
        return {"dfa_alpha": 0.50, "regime": "RANDOM_WALK (Insufficient Data)", "is_trending": False}
    
    vals = np.asarray(series_or_returns.values, dtype=float)
    if np.all(vals > 100.0):
        returns = np.diff(np.log(np.maximum(vals, 1.0)))
    else:
        returns = vals
        
    returns = returns[~np.isnan(returns)]
    if len(returns) < min_lag * 3:
        return {"dfa_alpha": 0.50, "regime": "RANDOM_WALK (Insufficient Data)", "is_trending": False}
        
    y = np.cumsum(returns - np.mean(returns))
    N = len(y)
    max_lag_adj = min(max_lag, N // 2)
    if max_lag_adj <= min_lag:
        return {"dfa_alpha": 0.50, "regime": "RANDOM_WALK (Insufficient Lags)", "is_trending": False}
        
    lags = np.unique(np.linspace(min_lag, max_lag_adj, 8, dtype=int))
    fluctuations = []
    valid_lags = []
    
    for lag in lags:
        n_boxes = int(np.floor(N / lag))
        if n_boxes == 0:
            continue
        
        y_boxes = y[:n_boxes * lag].reshape(n_boxes, lag)
        x = np.arange(lag)
        
        rms = 0.0
        for box in y_boxes:
            p = np.polyfit(x, box, 1)
            trend = np.polyval(p, x)
            rms += np.mean((box - trend) ** 2)
            
        fluct = np.sqrt(rms / n_boxes)
        if fluct > 1e-12:
            fluctuations.append(fluct)
            valid_lags.append(lag)
            
    if len(valid_lags) < 3:
        return {"dfa_alpha": 0.50, "regime": "RANDOM_WALK (Sparse Fits)", "is_trending": False}
        
    log_lags = np.log(valid_lags)
    log_flucts = np.log(fluctuations)
    alpha, _ = np.polyfit(log_lags, log_flucts, 1)
    alpha = float(np.clip(alpha, 0.05, 0.95))
    
    if alpha >= 0.55:
        regime = "STRONG_TREND (DFA Alpha Persistent)"
        is_trending = True
    elif alpha >= 0.52:
        regime = "MILD_TREND (DFA Alpha Persistent)"
        is_trending = True
    elif alpha <= 0.45:
        regime = "MEAN_REVERSION (DFA Alpha Anti-Persistent)"
        is_trending = False
    else:
        regime = "RANDOM_WALK (DFA Alpha Noise)"
        is_trending = False
        
    return {
        "dfa_alpha": round(alpha, 4),
        "regime": regime,
        "is_trending": is_trending
    }


def compute_vpin_toxicity(df: pd.DataFrame, bucket_volume: int = 10000) -> Dict[str, Any]:
    """
    Computes VPIN (Volume-Synchronized Probability of Informed Trading) using Bulk Volume Classification (BVC).
    Detects toxic institutional flow without requiring tick-level order books.
    VPIN >= 0.75 -> High Toxic Flow (Institutions aggressively front-running/dumping)
    """
    from scipy.stats import norm

    if df.empty or len(df) < 5 or "close" not in df.columns or "volume" not in df.columns:
        return {
            "vpin": 0.0,
            "is_toxic": False,
            "toxicity_level": "LOW_TOXICITY (Standard Market)",
            "action_advice": "STANDARD_RISK_SIZING",
            "bucket_count": 0
        }
        
    df_calc = df.copy()
    ret = df_calc["close"].pct_change().fillna(0.0)
    std_ret = ret.rolling(20, min_periods=3).std().fillna(0.001)
    std_ret = np.maximum(std_ret, 1e-5)
    
    # BVC: Standardized return mapped through standard normal CDF
    buy_ratio = norm.cdf(ret / std_ret)
    vol = np.asarray(df_calc["volume"].values, dtype=float)
    buy_vol = vol * buy_ratio
    sell_vol = vol * (1.0 - buy_ratio)
    
    total_vol = float(np.sum(vol))
    if total_vol <= 0:
        return {
            "vpin": 0.0,
            "is_toxic": False,
            "toxicity_level": "LOW_TOXICITY (Zero Traded Volume)",
            "action_advice": "STANDARD_RISK_SIZING",
            "bucket_count": 0
        }
        
    eff_bucket_vol = max(min(bucket_volume, int(total_vol / 10)), 500)
    
    cumulative_vol = 0.0
    vpin_buckets = []
    bucket_buy, bucket_sell = 0.0, 0.0
    
    for b_buy, b_sell, b_vol in zip(buy_vol, sell_vol, vol):
        cumulative_vol += b_vol
        bucket_buy += b_buy
        bucket_sell += b_sell
        
        if cumulative_vol >= eff_bucket_vol:
            imbalance = abs(bucket_buy - bucket_sell)
            vpin_buckets.append(imbalance / cumulative_vol)
            cumulative_vol, bucket_buy, bucket_sell = 0.0, 0.0, 0.0
            
    vpin_val = float(np.mean(vpin_buckets)) if vpin_buckets else 0.0
    vpin_val = min(max(round(vpin_val, 4), 0.0), 1.0)
    
    is_toxic = vpin_val >= VPIN_TOXICITY_THRESHOLD
    if vpin_val >= 0.85:
        toxicity_level = "CRITICAL_TOXICITY (Severe Informed Dumping / Front-Running)"
        advice = "WIDEN_SL_1.5X_HALVE_POSITION"
    elif vpin_val >= VPIN_TOXICITY_THRESHOLD:
        toxicity_level = "HIGH_TOXICITY (Informed Order Flow Prevalent)"
        advice = "WIDEN_SL_1.5X_HALVE_POSITION"
    elif vpin_val >= 0.50:
        toxicity_level = "MODERATE_FLOW (Standard Two-Sided Liquidity)"
        advice = "STANDARD_EXECUTION"
    else:
        toxicity_level = "LOW_TOXICITY (Retail Balancing / Clean Liquidity)"
        advice = "STANDARD_EXECUTION"
        
    return {
        "vpin": vpin_val,
        "is_toxic": is_toxic,
        "toxicity_level": toxicity_level,
        "action_advice": advice,
        "bucket_count": len(vpin_buckets)
    }


def compute_volume_synchronized_gamma_tracker(
    strikes: List[int],
    call_volumes: List[float],
    put_volumes: List[float],
    call_gammas: List[float],
    put_gammas: List[float],
    current_spot: float,
    multiplier: float = 65.0
) -> Dict[str, Any]:
    """
    Computes Volume-Synchronized Gamma Impact (Trade Gamma Flow):
    Gamma Impact = Volume * Option Gamma * Spot^2 * 0.01 * Multiplier
    Accumulates intraday call vs put traded gamma to pinpoint real-time Gamma Pins and Magnet levels.
    """
    if not strikes:
        return {"gamma_magnet_strike": int(round(current_spot / 50.0) * 50), "net_traded_gamma": 0.0, "pin_conviction": "NEUTRAL", "strike_impacts": []}
        
    strike_impacts = []
    total_call_impact = 0.0
    total_put_impact = 0.0
    
    for k, cv, pv, cg, pg in zip(strikes, call_volumes, put_volumes, call_gammas, put_gammas):
        c_impact = cv * cg * multiplier * (current_spot ** 2) * 0.01
        p_impact = pv * pg * multiplier * (current_spot ** 2) * 0.01
        net_impact = c_impact - p_impact
        total_call_impact += c_impact
        total_put_impact += p_impact
        
        strike_impacts.append({
            "strike": k,
            "call_impact": round(c_impact, 2),
            "put_impact": round(p_impact, 2),
            "net_impact": round(net_impact, 2),
            "total_gamma_volume": round(c_impact + p_impact, 2)
        })
        
    max_impact_item = max(strike_impacts, key=lambda x: x["total_gamma_volume"]) if strike_impacts else {"strike": int(round(current_spot / 50.0) * 50)}
    magnet_strike = max_impact_item["strike"]
    net_total = total_call_impact - total_put_impact
    
    if total_call_impact > 1.4 * total_put_impact:
        conviction = "CALL_GAMMA_MAGNET (Upward Resistance / Pin)"
    elif total_put_impact > 1.4 * total_call_impact:
        conviction = "PUT_GAMMA_MAGNET (Downward Support / Pin)"
    else:
        conviction = "BALANCED_GAMMA_PIN"
        
    return {
        "gamma_magnet_strike": magnet_strike,
        "total_call_gamma_impact": round(total_call_impact, 2),
        "total_put_gamma_impact": round(total_put_impact, 2),
        "net_traded_gamma": round(net_total, 2),
        "pin_conviction": conviction,
        "strike_impacts": strike_impacts
    }


def compute_session_cvd(df: pd.DataFrame) -> pd.Series:
    """
    Computes Session-Reset Cumulative Volume Delta (CVD) using 3-Tier Intrabar Decomposition:
    W_t = 0.50 * Phi_body + 0.30 * Phi_CLV + 0.20 * Phi_wick.
    Automatically resets CVD to 0.0 at 09:15 IST at the start of each trading day.
    """
    if df.empty or len(df) < 2:
        return pd.Series([0.0] * len(df), index=df.index if not df.empty else None)

    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    close = df["close"].astype(float).values
    open_p = df["open"].astype(float).values
    volume = df["volume"].astype(float).values if "volume" in df.columns else np.ones(len(df))

    ranges = np.maximum(high - low, 1e-4)

    # 1. Body Ratio
    phi_body = (close - open_p) / ranges

    # 2. Close Location Value (CLV)
    phi_clv = ((close - low) - (high - close)) / ranges

    # 3. Wick Imbalance
    upper_wick = high - np.maximum(close, open_p)
    lower_wick = np.minimum(close, open_p) - low
    wick_diff = (lower_wick - upper_wick) / ranges
    phi_wick = np.clip(wick_diff, -1.0, 1.0)

    # 3-Tier Composite Delta Weight
    w_t = 0.50 * phi_body + 0.30 * phi_clv + 0.20 * phi_wick
    bar_deltas = w_t * volume

    # Session Reset Logic (09:15 IST or Date Change)
    cvd = np.zeros(len(df))
    current_cum = 0.0

    dates = df.index.date if hasattr(df.index, "date") else None
    times = df.index.strftime("%H:%M") if hasattr(df.index, "strftime") else None

    for i in range(len(df)):
        is_new_session = False
        if i > 0:
            if dates is not None and dates[i] != dates[i - 1]:
                is_new_session = True
            elif times is not None and times[i] == "09:15":
                is_new_session = True

        if is_new_session:
            current_cum = 0.0

        current_cum += bar_deltas[i]
        cvd[i] = current_cum

    return pd.Series(cvd, index=df.index, name="session_cvd")


def detect_absorption_traps(df: pd.DataFrame, key_levels: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Detects Passive Institutional Limit Order Absorption at Value Area / CPR Extremes.
    Identifies when aggressive buyers/sellers are absorbed by opposing iceberg orders.
    """
    if len(df) < 6:
        return {"is_absorption": False, "type": "NONE", "confidence": 0.0, "reason": "Insufficient data"}

    sub_df = df.tail(6)
    close = float(sub_df["close"].iloc[-1])
    open_p = float(sub_df["open"].iloc[-1])
    high = float(sub_df["high"].iloc[-1])
    low = float(sub_df["low"].iloc[-1])
    
    cvd_series = compute_session_cvd(df)
    recent_cvd = cvd_series.tail(6).values
    cvd_delta = recent_cvd[-1] - recent_cvd[0]

    vah = key_levels.get("VAH", 999999.0) if key_levels else 999999.0
    val = key_levels.get("VAL", 0.0) if key_levels else 0.0
    cpr_tc = key_levels.get("CPR_TC", 999999.0) if key_levels else 999999.0
    cpr_bc = key_levels.get("CPR_BC", 0.0) if key_levels else 0.0

    # 1. Bearish Absorption at Resistance (Buyers absorbed at VAH / TC)
    tested_res = high >= vah - 5.0 or high >= cpr_tc - 5.0
    if tested_res and cvd_delta > 0 and close < open_p and (high - max(close, open_p)) >= 0.40 * (high - low):
        return {
            "is_absorption": True,
            "type": "BEARISH_ABSORPTION",
            "confidence": 0.88,
            "reason": "Aggressive buyers absorbed at Resistance (VAH/TC). Large upper wick with positive CVD divergence.",
            "suggested_sl": round(high + 5.0, 2)
        }

    # 2. Bullish Absorption at Support (Sellers absorbed at VAL / BC)
    tested_sup = low <= val + 5.0 or low <= cpr_bc + 5.0
    if tested_sup and cvd_delta < 0 and close > open_p and (min(close, open_p) - low) >= 0.40 * (high - low):
        return {
            "is_absorption": True,
            "type": "BULLISH_ABSORPTION",
            "confidence": 0.88,
            "reason": "Aggressive sellers absorbed at Support (VAL/BC). Large lower wick with negative CVD divergence.",
            "suggested_sl": round(low - 5.0, 2)
        }

    return {"is_absorption": False, "type": "NONE", "confidence": 0.0, "reason": "No absorption trap detected"}






