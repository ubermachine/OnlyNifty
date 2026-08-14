"""JustNifty v3.3 Institutional Strategy Engine with Multi-Timeframe Alignment & Stacked Order Flow Gating."""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
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
    compute_dealer_gex, compute_pre_open_gap_filter, detect_volume_profile_triggers,
    compute_cpr, compute_multi_timeframe_regime, detect_stacked_order_flow_imbalances,
    detect_iceberg_orders_and_liquidity_sweeps
)
from src.regime_switching import KalmanFilterTrendEstimator, MarkovRegimeSwitcher
from src.macro_engine import GlobalMacroEngine



class SignalType(Enum):
    WAIT = "WAIT"
    LONG = "LONG"
    SHORT = "SHORT"
    LONG_3PM = "LONG_3PM"
    SHORT_3PM = "SHORT_3PM"
    SHORT_LAAF = "SHORT_LAAF"
    LONG_ORDER_FLOW = "LONG_ORDER_FLOW"
    SHORT_ORDER_FLOW = "SHORT_ORDER_FLOW"

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

    @property
    def stop_loss(self) -> float:
        return self.sl_price

class StrategyEngine:
    """Vectorized and streaming bar-by-bar JustNifty v3.5 institutional strategy rules evaluator."""
    def __init__(self):
        self.kalman_filter = KalmanFilterTrendEstimator()
        self.markov_switcher = MarkovRegimeSwitcher()
        self.macro_engine = GlobalMacroEngine()
        self.session_losses: int = 0
        self.last_session_date: Optional[Any] = None


    def evaluate_bar(
        self,
        df_5m: pd.DataFrame,
        current_idx: int = -1,
        df_daily: Optional[pd.DataFrame] = None,
        df_hourly: Optional[pd.DataFrame] = None,
        df_15m: Optional[pd.DataFrame] = None,
        df_1h: Optional[pd.DataFrame] = None,
        live_iv: float = DEFAULT_IV,
        live_vix: Optional[float] = None,
        pre_open_gap: Optional[Dict[str, Any]] = None,
        prev_close: Optional[float] = None
    ) -> Signal:
        """
        Evaluates OnlyNifty v3.3 Institutional Multi-Timeframe Alignment and Stacked Footprint Setups:
        1. Multi-Timeframe (1H + 15m + 5m) Confluence Gating:
           - 5m Longs strictly gated by 15m and 1H Bullish / Neutral-Bullish trends.
           - 5m Shorts strictly gated by 15m and 1H Bearish / Neutral-Bearish trends.
        2. Stacked Order Flow Imbalances & Footprint Absorption at Key Levels (CPR, VAH, VAL, AVWAP).
        3. Auction Market Theory 70% Value Area triggers & Non-Linear VAKC Elasticity.
        4. Fibonacci Golden Pocket (50% - 61.8%) entries with dynamic Pre-Open gap anchoring.
        """
        if df_5m.empty:
            return Signal(SignalType.WAIT, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "No data available", False, 0.0, {})
            
        if current_idx == -1 or current_idx >= len(df_5m):
            current_idx = len(df_5m) - 1
            
        sub_df = df_5m.iloc[:current_idx + 1]
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

        if len(sub_df) < 15:
            return Signal(SignalType.WAIT, close, 0.0, 0.0, 0.0, 0.0, 0.0, "Accumulating bars for indicator stability", True, 0.0, {})

        # Compute Stochastic Indicators & Dynamic VAKC
        ema200_series = compute_ema(sub_df["close"], EMA_SLOW)
        ema55_series = compute_ema(sub_df["close"], EMA_MID)
        ema21_series = compute_ema(sub_df["close"], EMA_FAST)
        vakc_upper, vakc_lower = compute_vakc_envelopes(sub_df, iv=live_iv)
        vwap_series, vwap_up_2sd, vwap_low_2sd = compute_vwap(sub_df)
        
        hurst_info = compute_hurst_exponent(sub_df["close"])
        ofi_info = compute_order_flow_imbalance(sub_df)
        gex_info = compute_dealer_gex(close)
        vp_info = compute_volume_profile(sub_df)
        gap_info = compute_pre_open_gap_filter(sub_df, prev_close=prev_close, pre_open_data=pre_open_gap)
        cpr_info = compute_cpr(df_daily if df_daily is not None else sub_df)
        
        # Latent Kalman Spot Velocity & Markov Regime Model
        kalman_price, kalman_vel, kalman_z = self.kalman_filter.update(close)
        markov_info = self.markov_switcher.infer_regimes(sub_df)
        
        ema200 = float(ema200_series.iloc[-1])
        ema55 = float(ema55_series.iloc[-1])
        ema21 = float(ema21_series.iloc[-1])

        current_vwap = float(vwap_series.iloc[-1])
        upper_2sd = float(vwap_up_2sd.iloc[-1])
        lower_2sd = float(vwap_low_2sd.iloc[-1])
        upper_vakc_val = float(vakc_upper.iloc[-1])
        lower_vakc_val = float(vakc_lower.iloc[-1])
        
        # ATR 14 proxy
        atr_14 = float((sub_df["high"].tail(14) - sub_df["low"].tail(14)).mean())
        atr_14 = max(atr_14, 25.0)

        # 3. Multi-Timeframe Alignment Engine (1H + 15m + 5m)
        effective_1h = df_1h if df_1h is not None else df_hourly
        htf_regime = compute_multi_timeframe_regime(sub_df, df_15m=df_15m, df_1h=effective_1h)
        htf_aligned_long = htf_regime["htf_aligned_long"]
        htf_aligned_short = htf_regime["htf_aligned_short"]

        # 4. Key Levels & Stacked Footprint Order Flow Imbalance Detector
        key_levels = {
            "CPR_PIVOT": cpr_info.get("pivot", 0.0),
            "CPR_TC": cpr_info.get("tc", 0.0),
            "CPR_BC": cpr_info.get("bc", 0.0),
            "VAH": vp_info.get("vah", 0.0),
            "VAL": vp_info.get("val", 0.0),
            "POC": vp_info.get("poc", 0.0),
            "AVWAP": current_vwap,
            "AVWAP_UPPER_2SD": upper_2sd,
            "AVWAP_LOWER_2SD": lower_2sd
        }
        order_flow = detect_stacked_order_flow_imbalances(sub_df, key_levels=key_levels)
        microstructure = detect_iceberg_orders_and_liquidity_sweeps(sub_df)
        
        # Realized Volatility / Implied Volatility Ratio
        log_rets = np.diff(np.log(np.maximum(sub_df["close"].tail(30).values, 1.0)))
        realized_vol = float(np.std(log_rets, ddof=1) * np.sqrt(252 * 75)) if len(log_rets) > 5 else live_iv
        vol_ratio = round(realized_vol / max(live_iv, 0.05), 2)

        # 4.1 Liquidity Sweep Trap Strategy (SSL / BSL Purges)
        if microstructure["liquidity_sweep_detected"] and microstructure["sweep_event"]:
            sw = microstructure["sweep_event"]
            if sw["side"] == "LONG" and htf_aligned_long:
                return Signal(
                    signal_type=SignalType.LONG_ORDER_FLOW,
                    entry_price=close,
                    sl_price=sw["suggested_sl"],
                    target_1=round(close + 1.2 * atr_14, 2),
                    target_2=round(close + 2.5 * atr_14, 2),
                    target_3_moonshot=round(close + 4.0 * atr_14, 2),
                    pyramid_trigger=round(sw["swept_swing_low"] + 15.0, 2),
                    reason=f"Bullish SSL Liquidity Sweep Trap: {sw['thesis']} | HTF Aligned.",
                    htf_aligned=True,
                    fib_retracement=0.50,
                    details={"sweep": sw, "microstructure": microstructure, "order_flow": order_flow, "vol_ratio": vol_ratio}
                )
            elif sw["side"] == "SHORT" and htf_aligned_short:
                return Signal(
                    signal_type=SignalType.SHORT_ORDER_FLOW,
                    entry_price=close,
                    sl_price=sw["suggested_sl"],
                    target_1=round(close - 1.2 * atr_14, 2),
                    target_2=round(close - 2.5 * atr_14, 2),
                    target_3_moonshot=round(close - 4.0 * atr_14, 2),
                    pyramid_trigger=round(sw["swept_swing_high"] - 15.0, 2),
                    reason=f"Bearish BSL Liquidity Sweep Trap: {sw['thesis']} | HTF Aligned.",
                    htf_aligned=True,
                    fib_retracement=0.50,
                    details={"sweep": sw, "microstructure": microstructure, "order_flow": order_flow, "vol_ratio": vol_ratio}
                )


        # 5. Auction Market Theory (AMT) Value Area Trigger Check (HTF Gated)
        amt_trigger = detect_volume_profile_triggers(sub_df, vp_info, ofi_info, atr_14=atr_14)
        if amt_trigger["trigger"] in ["VAH_REJECTION", "VAL_REJECTION"] and amt_trigger["confidence"] >= 0.85:
            if amt_trigger["side"] == "LONG" and close > ema55:
                if not htf_aligned_long:
                    return Signal(
                        signal_type=SignalType.WAIT,
                        entry_price=close,
                        sl_price=0.0,
                        target_1=0.0,
                        target_2=0.0,
                        reason=f"HTF Confluence Veto: AMT Long rejected. 15m ({htf_regime['tf_15m']['bias']}) or 1H ({htf_regime['tf_1h']['bias']}) not Bullish.",
                        htf_aligned=False,
                        details={"htf_regime": htf_regime, "amt": amt_trigger, "order_flow": order_flow}
                    )
                return Signal(
                    signal_type=SignalType.LONG,
                    entry_price=close,
                    sl_price=amt_trigger["sl"],
                    target_1=amt_trigger["target_1"],
                    target_2=amt_trigger["target_2"],
                    target_3_moonshot=round(max(upper_vakc_val, upper_2sd), 2),
                    pyramid_trigger=round(vp_info.get("poc", close) + 5.0, 2),
                    reason=f"AMT Setup Confirmed: {amt_trigger['reason']} | HTF Aligned ({htf_regime['confluence_regime']}).",
                    htf_aligned=True,
                    fib_retracement=0.50,
                    details={"amt": amt_trigger, "vp": vp_info, "hurst": hurst_info, "ofi": ofi_info, "htf_regime": htf_regime, "order_flow": order_flow}
                )
            elif amt_trigger["side"] == "SHORT" and close < ema55:
                if not htf_aligned_short:
                    return Signal(
                        signal_type=SignalType.WAIT,
                        entry_price=close,
                        sl_price=0.0,
                        target_1=0.0,
                        target_2=0.0,
                        reason=f"HTF Confluence Veto: AMT Short rejected. 15m ({htf_regime['tf_15m']['bias']}) or 1H ({htf_regime['tf_1h']['bias']}) not Bearish.",
                        htf_aligned=False,
                        details={"htf_regime": htf_regime, "amt": amt_trigger, "order_flow": order_flow}
                    )
                return Signal(
                    signal_type=SignalType.SHORT,
                    entry_price=close,
                    sl_price=amt_trigger["sl"],
                    target_1=amt_trigger["target_1"],
                    target_2=amt_trigger["target_2"],
                    target_3_moonshot=round(min(lower_vakc_val, lower_2sd), 2),
                    pyramid_trigger=round(vp_info.get("poc", close) - 5.0, 2),
                    reason=f"AMT Setup Confirmed: {amt_trigger['reason']} | HTF Aligned ({htf_regime['confluence_regime']}).",
                    htf_aligned=True,
                    fib_retracement=0.50,
                    details={"amt": amt_trigger, "vp": vp_info, "hurst": hurst_info, "ofi": ofi_info, "htf_regime": htf_regime, "order_flow": order_flow}
                )

        # 6. Stacked Order Flow Absorption Setup Check
        if order_flow["absorption_event"] is not None:
            abs_event = order_flow["absorption_event"]
            if abs_event["type"] == "BUYER_ABSORPTION" and htf_aligned_long and close > ema55:
                return Signal(
                    signal_type=SignalType.LONG_ORDER_FLOW,
                    entry_price=close,
                    sl_price=abs_event["suggested_sl"],
                    target_1=round(close + 1.2 * atr_14, 2),
                    target_2=round(close + 2.5 * atr_14, 2),
                    target_3_moonshot=round(max(upper_vakc_val, upper_2sd), 2),
                    pyramid_trigger=round(close + 15.0, 2),
                    reason=f"Order Flow Buyer Absorption: {abs_event['reason']} | HTF Aligned.",
                    htf_aligned=True,
                    fib_retracement=0.50,
                    details={"order_flow": order_flow, "htf_regime": htf_regime, "abs_event": abs_event}
                )
            elif abs_event["type"] == "SELLER_ABSORPTION" and htf_aligned_short and close < ema55:
                return Signal(
                    signal_type=SignalType.SHORT_ORDER_FLOW,
                    entry_price=close,
                    sl_price=abs_event["suggested_sl"],
                    target_1=round(close - 1.2 * atr_14, 2),
                    target_2=round(close - 2.5 * atr_14, 2),
                    target_3_moonshot=round(min(lower_vakc_val, lower_2sd), 2),
                    pyramid_trigger=round(close - 15.0, 2),
                    reason=f"Order Flow Seller Absorption: {abs_event['reason']} | HTF Aligned.",
                    htf_aligned=True,
                    fib_retracement=0.50,
                    details={"order_flow": order_flow, "htf_regime": htf_regime, "abs_event": abs_event}
                )

        # 7. Far-Away MA Crossover Filter (with Gap-Decay Tolerance)
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

        # 8. Dynamic Fibonacci Swings & Pre-Open Gap Anchoring
        if gap_info["is_large_gap"]:
            swing_high = gap_info["anchor_high"]
            swing_low = gap_info["anchor_low"]
        else:
            lookback = min(40, current_idx)
            prior_window = sub_df.iloc[max(0, current_idx - lookback) : max(0, current_idx - 2)]
            if len(prior_window) < 5:
                return Signal(SignalType.WAIT, close, 0.0, 0.0, 0.0, 0.0, 0.0, "Accumulating swing history", True, 0.0, {})
            swing_high = float(prior_window["high"].max())
            swing_low = float(prior_window["low"].min())
            
        swing_range = swing_high - swing_low
        prev_bar = sub_df.iloc[current_idx - 1]
        prev_close_val = float(prev_bar["close"])

        # 9. LONG Setup (3-Tier Asymmetric Target Calculation with HTF Gating)
        long_avwap_cond = close > (current_vwap - 0.35 * (upper_2sd - current_vwap) / 2.0)
        if close > ema200 and long_avwap_cond and swing_range >= 35.0 and ofi_info["buyer_defense"]:
            if not htf_aligned_long:
                return Signal(
                    signal_type=SignalType.WAIT,
                    entry_price=close,
                    sl_price=0.0,
                    target_1=0.0,
                    target_2=0.0,
                    reason=f"HTF Confluence Veto: Golden Pocket Long rejected. 15m ({htf_regime['tf_15m']['bias']}) or 1H ({htf_regime['tf_1h']['bias']}) not Bullish.",
                    htf_aligned=False,
                    details={"htf_regime": htf_regime, "order_flow": order_flow}
                )
                
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
                    reason=f"LONG Setup Confirmed: Above 200 EMA + Above AVWAP ({gap_info['regime']}) + Golden Pocket + OFI Defense | HTF Aligned.",
                    htf_aligned=True,
                    fib_retracement=0.55,
                    details={
                        "fib": fib, "ema200": ema200, "vwap": current_vwap, "ema21": ema21,
                        "swing_high": swing_high, "hurst": hurst_info, "gex": gex_info, "ofi": ofi_info,
                        "atr_14": atr_14, "gap_info": gap_info, "htf_regime": htf_regime, "order_flow": order_flow
                    }
                )

        # 10. SHORT Setup (3-Tier Asymmetric Target Calculation with HTF Gating)
        short_avwap_cond = close < (current_vwap + 0.35 * (current_vwap - lower_2sd) / 2.0)
        if close < ema200 and short_avwap_cond and swing_range >= 35.0 and ofi_info["seller_defense"]:
            if not htf_aligned_short:
                return Signal(
                    signal_type=SignalType.WAIT,
                    entry_price=close,
                    sl_price=0.0,
                    target_1=0.0,
                    target_2=0.0,
                    reason=f"HTF Confluence Veto: Golden Pocket Short rejected. 15m ({htf_regime['tf_15m']['bias']}) or 1H ({htf_regime['tf_1h']['bias']}) not Bearish.",
                    htf_aligned=False,
                    details={"htf_regime": htf_regime, "order_flow": order_flow}
                )
                
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
                    reason=f"SHORT Setup Confirmed: Below 200 EMA + Below AVWAP ({gap_info['regime']}) + Golden Pocket + OFI Defense | HTF Aligned.",
                    htf_aligned=True,
                    fib_retracement=0.55,
                    details={
                        "fib": fib, "ema200": ema200, "vwap": current_vwap, "ema21": ema21,
                        "swing_low": swing_low, "hurst": hurst_info, "gex": gex_info, "ofi": ofi_info,
                        "atr_14": atr_14, "gap_info": gap_info, "htf_regime": htf_regime, "order_flow": order_flow
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
            details={
                "ema200": ema200, "vwap": current_vwap, "ema21": ema21,
                "hurst": hurst_info, "gex": gex_info, "ofi": ofi_info,
                "gap_info": gap_info, "htf_regime": htf_regime, "order_flow": order_flow,
                "kalman_price": kalman_price, "kalman_velocity": kalman_vel, "kalman_vel_zscore": kalman_z,
                "markov_regime": markov_info
            }
        )



