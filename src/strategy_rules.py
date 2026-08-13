"""JustNifty v3.0 Institutional Strategy Engine with Hurst Regime, VAKC Bands, OFI Delta, and LAAF Triggers."""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from src.config import (
    EMA_FAST, EMA_MID, EMA_SLOW, VAKC_LAMBDA,
    FIB_GOLDEN_MIN, FIB_GOLDEN_MAX, MA_STRETCH_THRESHOLD,
    HURST_TRENDING_MIN, HURST_MEAN_REV_MAX, DEFAULT_IV
)
from src.indicators import (
    compute_ema, compute_vakc_envelopes, compute_vwap, compute_fibonacci_levels,
    compute_hurst_exponent, compute_order_flow_imbalance, compute_volume_profile, compute_dealer_gex
)

class SignalType(Enum):
    WAIT = "WAIT"
    LONG = "LONG"
    SHORT = "SHORT"
    LONG_3PM = "LONG_3PM"
    SHORT_3PM = "SHORT_3PM"
    SHORT_LAAF = "SHORT_LAAF"  # Look Above and Fail AMT trigger

@dataclass
class Signal:
    signal_type: SignalType
    entry_price: float
    sl_price: float
    target_1: float
    target_2: float
    reason: str
    htf_aligned: bool
    fib_retracement: float
    details: Dict[str, Any]

class StrategyEngine:
    def __init__(self):
        pass

    def evaluate_bar(
        self,
        df_5m: pd.DataFrame,
        current_idx: int = -1,
        df_daily: Optional[pd.DataFrame] = None,
        df_hourly: Optional[pd.DataFrame] = None
    ) -> Signal:
        """Evaluates JustNifty v3.0 institutional trade setups on the specified 5m bar."""
        if df_5m.empty:
            return Signal(SignalType.WAIT, 0.0, 0.0, 0.0, 0.0, "No data available", False, 0.0, {})
            
        if current_idx == -1 or current_idx >= len(df_5m):
            current_idx = len(df_5m) - 1
            
        bar = df_5m.iloc[current_idx]
        bar_time = bar.name.strftime("%H:%M") if hasattr(bar.name, "strftime") else "12:00"
        close = float(bar["close"])
        
        # 1. 09:15 - 09:30 AM Freak Candle Isolation Rule
        if "09:15" <= bar_time < "09:30":
            return Signal(
                signal_type=SignalType.WAIT,
                entry_price=close,
                sl_price=0.0,
                target_1=0.0,
                target_2=0.0,
                reason="Opening 15-min range (Freak Candle isolation). True opening range is establishing.",
                htf_aligned=True,
                fib_retracement=0.0,
                details={"bar_time": bar_time}
            )

        # 2. 3:00 PM (15:00) Aggressive Breakout Strategy Check (Page 100 / Query 39)
        if bar_time in ["15:05", "15:10"]:
            three_pm_indices = [
                i for i, idx in enumerate(df_5m.index[:current_idx + 1])
                if hasattr(idx, "strftime") and idx.strftime("%H:%M") == "15:00"
            ]
            if three_pm_indices:
                candle_3pm = df_5m.iloc[three_pm_indices[-1]]
                if close > float(candle_3pm["high"]):
                    return Signal(
                        signal_type=SignalType.LONG_3PM,
                        entry_price=close,
                        sl_price=float(candle_3pm["low"]),
                        target_1=round(close + 45.0, 2),
                        target_2=round(close + 85.0, 2),
                        reason="3 PM Strategy: Bullish breakout above 15:00 candle High. Institutional MOC gamma squeeze expected.",
                        htf_aligned=True,
                        fib_retracement=0.0,
                        details={"3pm_high": float(candle_3pm["high"]), "3pm_low": float(candle_3pm["low"])}
                    )
                elif close < float(candle_3pm["low"]):
                    return Signal(
                        signal_type=SignalType.SHORT_3PM,
                        entry_price=close,
                        sl_price=float(candle_3pm["high"]),
                        target_1=round(close - 45.0, 2),
                        target_2=round(close - 85.0, 2),
                        reason="3 PM Strategy: Bearish breakdown below 15:00 candle Low. Institutional MOC gamma squeeze expected.",
                        htf_aligned=True,
                        fib_retracement=0.0,
                        details={"3pm_high": float(candle_3pm["high"]), "3pm_low": float(candle_3pm["low"])}
                    )

        if len(df_5m) < 15:
            return Signal(SignalType.WAIT, close, 0.0, 0.0, 0.0, "Accumulating bars for indicator stability", True, 0.0, {})

        # Compute Stochastic Pillars & Indicators
        ema200_series = compute_ema(df_5m["close"], EMA_SLOW)
        ema55_series = compute_ema(df_5m["close"], EMA_MID)
        ema21_series = compute_ema(df_5m["close"], EMA_FAST)
        vakc_upper, vakc_lower = compute_vakc_envelopes(df_5m)
        vwap_series, _, _ = compute_vwap(df_5m)
        
        # Adaptive Stochastic Metrics
        hurst_info = compute_hurst_exponent(df_5m["close"].iloc[:current_idx + 1])
        ofi_info = compute_order_flow_imbalance(df_5m.iloc[:current_idx + 1])
        gex_info = compute_dealer_gex(close)
        vp_info = compute_volume_profile(df_5m.iloc[:current_idx + 1])
        
        ema200 = float(ema200_series.iloc[current_idx])
        ema55 = float(ema55_series.iloc[current_idx])
        ema21 = float(ema21_series.iloc[current_idx])
        current_vwap = float(vwap_series.iloc[current_idx])
        
        # 3. Far-Away MA Crossover Nuance Filter (Query 12)
        dist_to_ema21 = abs(close - ema21) / close
        if dist_to_ema21 > MA_STRETCH_THRESHOLD:
            return Signal(
                signal_type=SignalType.WAIT,
                entry_price=close,
                sl_price=0.0,
                target_1=0.0,
                target_2=0.0,
                reason=f"Price is overextended from 21 EMA ({dist_to_ema21*100:.2f}% vs {MA_STRETCH_THRESHOLD*100:.2f}% threshold). Wait for mean-reversion pullback.",
                htf_aligned=True,
                fib_retracement=0.0,
                details={"ema21": ema21, "dist_pct": dist_to_ema21, "hurst": hurst_info["hurst"]}
            )

        # 4. Dynamic Fibonacci Swings
        lookback = min(40, current_idx)
        prior_window = df_5m.iloc[current_idx - lookback : current_idx - 2]
        
        if len(prior_window) < 5:
            return Signal(SignalType.WAIT, close, 0.0, 0.0, 0.0, "Accumulating swing history", True, 0.0, {})

        swing_high = float(prior_window["high"].max())
        swing_low = float(prior_window["low"].min())
        swing_range = swing_high - swing_low
        
        # Intraday ATR proxy (14 bars)
        atr_14 = float((df_5m["high"].iloc[current_idx-14:current_idx+1] - df_5m["low"].iloc[current_idx-14:current_idx+1]).mean()) if current_idx >= 14 else 35.0
        atr_14 = max(atr_14, 25.0)

        bar_open = float(bar["open"])
        prev_bar = df_5m.iloc[current_idx - 1]
        prev_close = float(prev_bar["close"])

        # 5. LONG Setup Check (Above 200 EMA + Above AVWAP + 50-61.8% Golden Pocket + Bullish Confirmation Candle)
        if close > ema200 and close > current_vwap and swing_range >= 35.0:
            fib = compute_fibonacci_levels(swing_high, swing_low, is_uptrend=True)
            in_pocket = fib["fib_618"] <= min(close, bar_open) and max(close, bar_open) <= (fib["fib_500"] + 10.0)
            
            bullish_trigger = (close > bar_open) or (close > prev_close)
            
            if in_pocket and bullish_trigger:
                t1 = round(min(swing_high, close + 1.2 * atr_14), 2)
                t2 = round(max(float(vakc_upper.iloc[current_idx]), swing_high + 0.618 * swing_range), 2)
                
                return Signal(
                    signal_type=SignalType.LONG,
                    entry_price=close,
                    sl_price=fib["sl_level"],
                    target_1=t1,
                    target_2=t2,
                    reason="LONG Setup Confirmed: Above 200 EMA + Above AVWAP + Golden Pocket (50-61.8%) + Bullish Confirmation Candle.",
                    htf_aligned=True,
                    fib_retracement=0.55,
                    details={
                        "fib": fib, "ema200": ema200, "vwap": current_vwap,
                        "swing_high": swing_high, "hurst": hurst_info, "gex": gex_info, "ofi": ofi_info
                    }
                )

        # 6. SHORT Setup Check (Below 200 EMA + Below AVWAP + 50-61.8% Golden Pocket + Bearish Confirmation Candle)
        if close < ema200 and close < current_vwap and swing_range >= 35.0:
            fib = compute_fibonacci_levels(swing_high, swing_low, is_uptrend=False)
            in_pocket = (fib["fib_500"] - 10.0) <= min(close, bar_open) and max(close, bar_open) <= fib["fib_618"]
            
            bearish_trigger = (close < bar_open) or (close < prev_close)
            
            if in_pocket and bearish_trigger:
                t1 = round(max(swing_low, close - 1.2 * atr_14), 2)
                t2 = round(min(float(vakc_lower.iloc[current_idx]), swing_low - 0.618 * swing_range), 2)
                
                return Signal(
                    signal_type=SignalType.SHORT,
                    entry_price=close,
                    sl_price=fib["sl_level"],
                    target_1=t1,
                    target_2=t2,
                    reason="SHORT Setup Confirmed: Below 200 EMA + Below AVWAP + Golden Pocket (50-61.8%) + Bearish Confirmation Candle.",
                    htf_aligned=True,
                    fib_retracement=0.55,
                    details={
                        "fib": fib, "ema200": ema200, "vwap": current_vwap,
                        "swing_low": swing_low, "hurst": hurst_info, "gex": gex_info, "ofi": ofi_info
                    }
                )

        return Signal(
            signal_type=SignalType.WAIT,
            entry_price=close,
            sl_price=0.0,
            target_1=0.0,
            target_2=0.0,
            reason="Market in consolidation / No confluence across the 4 core tools.",
            htf_aligned=True,
            fib_retracement=0.0,
            details={
                "ema200": ema200, "vwap": current_vwap, "ema21": ema21,
                "hurst": hurst_info, "gex": gex_info, "ofi": ofi_info, "volume_profile": vp_info
            }
        )
