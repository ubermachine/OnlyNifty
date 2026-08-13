"""JustNifty v2.0 Rule Engine with institutional microstructure filters."""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from src.config import (
    EMA_FAST, EMA_MID, EMA_SLOW, ENVELOPE_PCT,
    FIB_GOLDEN_MIN, FIB_GOLDEN_MAX, MA_STRETCH_THRESHOLD
)
from src.indicators import (
    compute_ema, compute_envelopes, compute_vwap, compute_fibonacci_levels
)

class SignalType(Enum):
    WAIT = "WAIT"
    LONG = "LONG"
    SHORT = "SHORT"
    LONG_3PM = "LONG_3PM"
    SHORT_3PM = "SHORT_3PM"

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
        """Evaluates JustNifty v2.0 trade setups on the specified 5m bar."""
        if df_5m.empty:
            return Signal(SignalType.WAIT, 0.0, 0.0, 0.0, 0.0, "No data available", False, 0.0, {})
            
        if current_idx == -1 or current_idx >= len(df_5m):
            current_idx = len(df_5m) - 1
            
        bar = df_5m.iloc[current_idx]
        bar_time = bar.name.strftime("%H:%M")
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
                if idx.strftime("%H:%M") == "15:00"
            ]
            if three_pm_indices:
                candle_3pm = df_5m.iloc[three_pm_indices[-1]]
                if close > float(candle_3pm["high"]):
                    return Signal(
                        signal_type=SignalType.LONG_3PM,
                        entry_price=close,
                        sl_price=float(candle_3pm["low"]),
                        target_1=round(close + 80.0, 2),
                        target_2=round(close + 160.0, 2),
                        reason="3 PM Strategy: Bullish breakout above 15:00 candle High. Fast momentum move expected.",
                        htf_aligned=True,
                        fib_retracement=0.0,
                        details={"3pm_high": float(candle_3pm["high"]), "3pm_low": float(candle_3pm["low"])}
                    )
                elif close < float(candle_3pm["low"]):
                    return Signal(
                        signal_type=SignalType.SHORT_3PM,
                        entry_price=close,
                        sl_price=float(candle_3pm["high"]),
                        target_1=round(close - 80.0, 2),
                        target_2=round(close - 160.0, 2),
                        reason="3 PM Strategy: Bearish breakdown below 15:00 candle Low. Fast momentum move expected.",
                        htf_aligned=True,
                        fib_retracement=0.0,
                        details={"3pm_high": float(candle_3pm["high"]), "3pm_low": float(candle_3pm["low"])}
                    )

        # Calculate indicators if sufficient data
        if len(df_5m) < 15:
            return Signal(SignalType.WAIT, close, 0.0, 0.0, 0.0, "Accumulating bars for indicator stability", True, 0.0, {})

        ema200_series = compute_ema(df_5m["close"], EMA_SLOW)
        ema55_series = compute_ema(df_5m["close"], EMA_MID)
        ema21_series = compute_ema(df_5m["close"], EMA_FAST)
        env_upper, env_lower = compute_envelopes(ema200_series, ENVELOPE_PCT)
        vwap_series, _, _ = compute_vwap(df_5m)
        
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
                reason=f"Price is overextended from 21 EMA ({dist_to_ema21*100:.2f}% vs {MA_STRETCH_THRESHOLD*100:.2f}% threshold). Wait for pullback before entry.",
                htf_aligned=True,
                fib_retracement=0.0,
                details={"ema21": ema21, "dist_pct": dist_to_ema21}
            )

        # 4. Dynamic Fibonacci Swings
        lookback = min(35, current_idx)
        window = df_5m.iloc[current_idx - lookback : current_idx + 1]
        swing_high = float(window["high"].max())
        swing_low = float(window["low"].min())
        
        # 5. LONG Setup Check (Above 200 EMA + Above AVWAP + 50-61.8% Golden Pocket)
        if close > ema200 and close > current_vwap:
            fib = compute_fibonacci_levels(swing_high, swing_low, is_uptrend=True)
            if fib["fib_618"] <= close <= fib["fib_500"]:
                return Signal(
                    signal_type=SignalType.LONG,
                    entry_price=close,
                    sl_price=fib["sl_level"],
                    target_1=round(float(env_upper.iloc[current_idx]), 2),
                    target_2=round(swing_high + (swing_high - fib["fib_500"]), 2),
                    reason="LONG Setup Confirmed: Above 200 EMA + Above AVWAP + 50.0% to 61.8% Golden Pocket Retracement.",
                    htf_aligned=True,
                    fib_retracement=0.55,
                    details={"fib": fib, "ema200": ema200, "vwap": current_vwap}
                )

        # 6. SHORT Setup Check (Below 200 EMA + Below AVWAP + 50-61.8% Golden Pocket)
        if close < ema200 and close < current_vwap:
            fib = compute_fibonacci_levels(swing_high, swing_low, is_uptrend=False)
            if fib["fib_500"] <= close <= fib["fib_618"]:
                return Signal(
                    signal_type=SignalType.SHORT,
                    entry_price=close,
                    sl_price=fib["sl_level"],
                    target_1=round(float(env_lower.iloc[current_idx]), 2),
                    target_2=round(swing_low - (fib["fib_500"] - swing_low), 2),
                    reason="SHORT Setup Confirmed: Below 200 EMA + Below AVWAP + 50.0% to 61.8% Golden Pocket Retracement.",
                    htf_aligned=True,
                    fib_retracement=0.55,
                    details={"fib": fib, "ema200": ema200, "vwap": current_vwap}
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
            details={"ema200": ema200, "vwap": current_vwap, "ema21": ema21}
        )
