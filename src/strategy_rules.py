"""JustNifty v3.1 Institutional Strategy Engine with 3-Tier Targets, AVWAP Corridor Gating, and Pyramiding Triggers."""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from src.config import (
    EMA_FAST, EMA_MID, EMA_SLOW, VAKC_LAMBDA,
    FIB_GOLDEN_MIN, FIB_GOLDEN_MAX, MA_STRETCH_THRESHOLD,
    HURST_TRENDING_MIN, HURST_MEAN_REV_MAX, DEFAULT_IV, OFI_ZSCORE_MIN
)
from src.indicators import (
    compute_ema, compute_vakc_envelopes, compute_vwap, compute_fibonacci_levels,
    compute_hurst_exponent, compute_order_flow_imbalance, compute_volume_profile,
    compute_dealer_gex, compute_pre_open_gap_filter, detect_volume_profile_triggers
)

class SignalType(Enum):
    WAIT = "WAIT"
    LONG = "LONG"
    SHORT = "SHORT"
    LONG_3PM = "LONG_3PM"
    SHORT_3PM = "SHORT_3PM"
    SHORT_LAAF = "SHORT_LAAF"

@dataclass
class Signal:
    signal_type: SignalType
    entry_price: float
    sl_price: float
    target_1: float
    target_2: float
    target_3_moonshot: float = 0.0
    pyramid_trigger: float = 0.0
    reason: str = ""
    htf_aligned: bool = True
    fib_retracement: float = 0.0
    details: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}
        if self.target_3_moonshot == 0.0 and self.target_2 > 0:
            diff = abs(self.target_2 - self.entry_price)
            self.target_3_moonshot = round(self.target_2 + (diff * 0.618), 2)
        if self.pyramid_trigger == 0.0 and self.entry_price > 0:
            self.pyramid_trigger = round(self.entry_price + 25.0, 2)

class StrategyEngine:
    def __init__(self):
        self.session_losses: int = 0
        self.last_session_date: Optional[Any] = None

    def evaluate_bar(
        self,
        df_5m: pd.DataFrame,
        current_idx: int = -1,
        df_daily: Optional[pd.DataFrame] = None,
        df_hourly: Optional[pd.DataFrame] = None,
        live_iv: float = DEFAULT_IV,
        live_vix: Optional[float] = None,
        pre_open_gap: Optional[Dict[str, Any]] = None,
        prev_close: Optional[float] = None
    ) -> Signal:
        """
        Evaluates JustNifty v3.2 institutional setups integrating:
        1. Non-linear Live VIX VAKC Scaling
        2. Pre-Open Market Gap Filter
        3. 70% Value Area & POC AMT Triggers
        """
        if df_5m.empty:
            return Signal(SignalType.WAIT, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "No data available", False, 0.0, {})
            
        if current_idx == -1 or current_idx >= len(df_5m):
            current_idx = len(df_5m) - 1
            
        bar = df_5m.iloc[current_idx]
        bar_time = bar.name.strftime("%H:%M") if hasattr(bar.name, "strftime") else "12:00"
        close = float(bar["close"])
        bar_open = float(bar["open"])
        
        # 1. 09:15 - 09:30 AM Freak Candle Isolation Rule
        if "09:15" <= bar_time < "09:30":
            return Signal(
                signal_type=SignalType.WAIT,
                entry_price=close,
                sl_price=0.0,
                target_1=0.0,
                target_2=0.0,
                target_3_moonshot=0.0,
                pyramid_trigger=0.0,
                reason="Opening 15-min range (Freak Candle isolation). True opening range is establishing.",
                htf_aligned=True,
                fib_retracement=0.0,
                details={"bar_time": bar_time}
            )

        # 2. 3:00 PM (15:00) Aggressive Breakout Strategy Check
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
                        target_1=round(close + 40.0, 2),
                        target_2=round(close + 75.0, 2),
                        target_3_moonshot=round(close + 120.0, 2),
                        pyramid_trigger=round(float(candle_3pm["high"]) + 10.0, 2),
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
                        target_1=round(close - 40.0, 2),
                        target_2=round(close - 75.0, 2),
                        target_3_moonshot=round(close - 120.0, 2),
                        pyramid_trigger=round(float(candle_3pm["low"]) - 10.0, 2),
                        reason="3 PM Strategy: Bearish breakdown below 15:00 candle Low. Institutional MOC gamma squeeze expected.",
                        htf_aligned=True,
                        fib_retracement=0.0,
                        details={"3pm_high": float(candle_3pm["high"]), "3pm_low": float(candle_3pm["low"])}
                    )

        if len(df_5m) < 15:
            return Signal(SignalType.WAIT, close, 0.0, 0.0, 0.0, 0.0, 0.0, "Accumulating bars for indicator stability", True, 0.0, {})

        # Compute Stochastic Indicators & Dynamic VAKC
        sub_df = df_5m.iloc[:current_idx + 1]
        ema200_series = compute_ema(df_5m["close"], EMA_SLOW)
        ema55_series = compute_ema(df_5m["close"], EMA_MID)
        ema21_series = compute_ema(df_5m["close"], EMA_FAST)
        vakc_upper, vakc_lower = compute_vakc_envelopes(df_5m, iv=live_iv)
        vwap_series, vwap_up_2sd, vwap_low_2sd = compute_vwap(df_5m)
        
        hurst_info = compute_hurst_exponent(sub_df["close"])
        ofi_info = compute_order_flow_imbalance(sub_df)
        gex_info = compute_dealer_gex(close)
        vp_info = compute_volume_profile(sub_df)
        gap_info = compute_pre_open_gap_filter(sub_df, prev_close=prev_close, pre_open_data=pre_open_gap)
        
        ema200 = float(ema200_series.iloc[current_idx])
        ema55 = float(ema55_series.iloc[current_idx])
        ema21 = float(ema21_series.iloc[current_idx])
        current_vwap = float(vwap_series.iloc[current_idx])
        upper_2sd = float(vwap_up_2sd.iloc[current_idx])
        lower_2sd = float(vwap_low_2sd.iloc[current_idx])
        upper_vakc_val = float(vakc_upper.iloc[current_idx])
        lower_vakc_val = float(vakc_lower.iloc[current_idx])
        
        # ATR 14 proxy
        atr_14 = float((df_5m["high"].iloc[max(0, current_idx-14):current_idx+1] - df_5m["low"].iloc[max(0, current_idx-14):current_idx+1]).mean())
        atr_14 = max(atr_14, 25.0)

        # 3. Auction Market Theory (AMT) Value Area Trigger Check
        amt_trigger = detect_volume_profile_triggers(sub_df, vp_info, ofi_info, atr_14=atr_14)
        if amt_trigger["trigger"] in ["VAH_REJECTION", "VAL_REJECTION"] and amt_trigger["confidence"] >= 0.85:
            if amt_trigger["side"] == "LONG" and close > ema55:
                return Signal(
                    signal_type=SignalType.LONG,
                    entry_price=close,
                    sl_price=amt_trigger["sl"],
                    target_1=amt_trigger["target_1"],
                    target_2=amt_trigger["target_2"],
                    target_3_moonshot=round(max(upper_vakc_val, upper_2sd), 2),
                    pyramid_trigger=round(vp_info.get("poc", close) + 5.0, 2),
                    reason=f"AMT Setup: {amt_trigger['reason']}",
                    htf_aligned=True,
                    fib_retracement=0.50,
                    details={"amt": amt_trigger, "vp": vp_info, "hurst": hurst_info, "ofi": ofi_info}
                )
            elif amt_trigger["side"] == "SHORT" and close < ema55:
                return Signal(
                    signal_type=SignalType.SHORT,
                    entry_price=close,
                    sl_price=amt_trigger["sl"],
                    target_1=amt_trigger["target_1"],
                    target_2=amt_trigger["target_2"],
                    target_3_moonshot=round(min(lower_vakc_val, lower_2sd), 2),
                    pyramid_trigger=round(vp_info.get("poc", close) - 5.0, 2),
                    reason=f"AMT Setup: {amt_trigger['reason']}",
                    htf_aligned=True,
                    fib_retracement=0.50,
                    details={"amt": amt_trigger, "vp": vp_info, "hurst": hurst_info, "ofi": ofi_info}
                )

        # 4. Far-Away MA Crossover Filter (with Gap-Decay Tolerance)
        dist_to_ema21 = abs(close - ema21) / close
        effective_stretch_threshold = MA_STRETCH_THRESHOLD * gap_info["slope_tolerance_mult"]
        if dist_to_ema21 > effective_stretch_threshold:
            return Signal(
                signal_type=SignalType.WAIT,
                entry_price=close,
                sl_price=0.0,
                target_1=0.0,
                target_2=0.0,
                target_3_moonshot=0.0,
                pyramid_trigger=0.0,
                reason=f"Price is overextended from 21 EMA ({dist_to_ema21*100:.2f}% vs {effective_stretch_threshold*100:.2f}% threshold). Wait for pullback.",
                htf_aligned=True,
                fib_retracement=0.0,
                details={"ema21": ema21, "dist_pct": dist_to_ema21, "hurst": hurst_info["hurst"]}
            )

        # 5. Dynamic Fibonacci Swings & Pre-Open Gap Anchoring
        if gap_info["is_large_gap"]:
            swing_high = gap_info["anchor_high"]
            swing_low = gap_info["anchor_low"]
        else:
            lookback = min(40, current_idx)
            prior_window = df_5m.iloc[current_idx - lookback : current_idx - 2]
            if len(prior_window) < 5:
                return Signal(SignalType.WAIT, close, 0.0, 0.0, 0.0, 0.0, 0.0, "Accumulating swing history", True, 0.0, {})
            swing_high = float(prior_window["high"].max())
            swing_low = float(prior_window["low"].min())
            
        swing_range = swing_high - swing_low
        prev_bar = df_5m.iloc[current_idx - 1]
        prev_close_val = float(prev_bar["close"])

        # 6. LONG Setup (3-Tier Asymmetric Target Calculation)
        long_avwap_cond = close > (current_vwap - 0.35 * (upper_2sd - current_vwap) / 2.0)
        if close > ema200 and long_avwap_cond and swing_range >= 35.0 and ofi_info["buyer_defense"]:
            fib = compute_fibonacci_levels(swing_high, swing_low, is_uptrend=True)
            in_pocket = fib["fib_618"] <= min(close, bar_open) and max(close, bar_open) <= (fib["fib_500"] + 10.0)
            bullish_trigger = (close > bar_open) or (close > prev_close_val)
            
            if in_pocket and bullish_trigger:
                t1 = round(close + 1.2 * atr_14, 2)
                t2 = round(close + 2.5 * atr_14, 2)
                t3_moonshot = round(max(upper_vakc_val, upper_2sd, close + 3.8 * atr_14), 2)
                pyramid_trigger_lvl = round(swing_high + 2.0, 2)
                
                return Signal(
                    signal_type=SignalType.LONG,
                    entry_price=close,
                    sl_price=fib["sl_level"],
                    target_1=t1,
                    target_2=t2,
                    target_3_moonshot=t3_moonshot,
                    pyramid_trigger=pyramid_trigger_lvl,
                    reason=f"LONG Setup Confirmed: Above 200 EMA + Above AVWAP ({gap_info['regime']}) + Golden Pocket + OFI Defense.",
                    htf_aligned=True,
                    fib_retracement=0.55,
                    details={
                        "fib": fib, "ema200": ema200, "vwap": current_vwap, "ema21": ema21,
                        "swing_high": swing_high, "hurst": hurst_info, "gex": gex_info, "ofi": ofi_info,
                        "atr_14": atr_14, "gap_info": gap_info
                    }
                )

        # 7. SHORT Setup (3-Tier Asymmetric Target Calculation)
        short_avwap_cond = close < (current_vwap + 0.35 * (current_vwap - lower_2sd) / 2.0)
        if close < ema200 and short_avwap_cond and swing_range >= 35.0 and ofi_info["seller_defense"]:
            fib = compute_fibonacci_levels(swing_high, swing_low, is_uptrend=False)
            in_pocket = (fib["fib_500"] - 10.0) <= min(close, bar_open) and max(close, bar_open) <= fib["fib_618"]
            bearish_trigger = (close < bar_open) or (close < prev_close_val)
            
            if in_pocket and bearish_trigger:
                t1 = round(close - 1.2 * atr_14, 2)
                t2 = round(close - 2.5 * atr_14, 2)
                t3_moonshot = round(min(lower_vakc_val, lower_2sd, close - 3.8 * atr_14), 2)
                pyramid_trigger_lvl = round(swing_low - 2.0, 2)
                
                return Signal(
                    signal_type=SignalType.SHORT,
                    entry_price=close,
                    sl_price=fib["sl_level"],
                    target_1=t1,
                    target_2=t2,
                    target_3_moonshot=t3_moonshot,
                    pyramid_trigger=pyramid_trigger_lvl,
                    reason=f"SHORT Setup Confirmed: Below 200 EMA + Below AVWAP ({gap_info['regime']}) + Golden Pocket + OFI Defense.",
                    htf_aligned=True,
                    fib_retracement=0.55,
                    details={
                        "fib": fib, "ema200": ema200, "vwap": current_vwap, "ema21": ema21,
                        "swing_low": swing_low, "hurst": hurst_info, "gex": gex_info, "ofi": ofi_info,
                        "atr_14": atr_14, "gap_info": gap_info
                    }
                )

        return Signal(
            signal_type=SignalType.WAIT,
            entry_price=close,
            sl_price=0.0,
            target_1=0.0,
            target_2=0.0,
            target_3_moonshot=0.0,
            pyramid_trigger=0.0,
            reason="Market in consolidation / No confluence across core indicators.",
            htf_aligned=True,
            fib_retracement=0.0,
            details={"ema200": ema200, "vwap": current_vwap, "ema21": ema21, "hurst": hurst_info, "gex": gex_info, "ofi": ofi_info, "gap_info": gap_info}
        )

