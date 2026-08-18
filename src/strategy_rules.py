"""JustNifty v3.3 Institutional Strategy Engine with Multi-Timeframe Alignment & Stacked Order Flow Gating."""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
from src.config import (
    EMA_FAST, EMA_MID, EMA_SLOW, VAKC_LAMBDA,
    FIB_GOLDEN_MIN, FIB_GOLDEN_MAX, MA_STRETCH_THRESHOLD,
    HURST_TRENDING_MIN, HURST_MEAN_REV_MAX, DEFAULT_IV, OFI_ZSCORE_MIN,
    SKEW_ZSCORE_THRESHOLD, GEX_WALL_BUFFER_PTS, VCR_SQUEEZE_THRESHOLD,
    SIGNAL_MIN_CONFLUENCE, VPIN_TOXICITY_THRESHOLD, STOP_MIN_ATR_FRACTION,
    STOP_MAX_ATR_MULTIPLE, STOP_MAX_POINTS, STOP_NOISE_BAND_MULT, GATE_FAIL_TO_WAIT, GATE_MIN_MISSING_TO_BLOCK
)
from src.indicators import (
    compute_ema, compute_vakc_envelopes, compute_vwap, compute_fibonacci_levels,
    compute_hurst_exponent, compute_order_flow_imbalance, compute_volume_profile,
    compute_dealer_gex, compute_pre_open_gap_filter, detect_volume_profile_triggers,
    compute_cpr, compute_multi_timeframe_regime, detect_stacked_order_flow_imbalances,
    detect_iceberg_orders_and_liquidity_sweeps, compute_dfa_alpha, compute_vpin_toxicity,
    compute_volume_synchronized_gamma_tracker,
    compute_initial_balance_and_day_type,
    compute_session_cvd, detect_absorption_traps
)
from src.regime_switching import KalmanFilterTrendEstimator, MarkovRegimeSwitcher, MultiAssetKalmanCointegrator
from src.macro_engine import GlobalMacroEngine
from src.volatility_engine import VolatilityIntelligence
from src.risk_state import SessionRiskState
from src.config import (
    POSITIONING_VETO_STRENGTH,
    WALL_BUFFER_PTS,
    PCR_Z_CONTRARIAN_THRESHOLD,
    POSITIONING_UNVERIFIED_SIZE_CAP,
    TERM_STRUCTURE_BACKWARDATION_THRESHOLD, TERM_STRUCTURE_CRISIS_SIZE_MULT,
    IV_RANK_SPREAD_THRESHOLD, IV_RANK_CONVEXITY_THRESHOLD,
    GEX_WALL_BUFFER_ATR_MULT, VAL_BUFFER_ATR_MULT,
    GAMMA_SQUEEZE_TARGET_MULT, EXPIRY_PIN_MIN_DISTANCE_PTS
)



class SignalType(Enum):
    WAIT = "WAIT"
    LONG = "LONG"
    SHORT = "SHORT"
    LONG_3PM = "LONG_3PM"
    SHORT_3PM = "SHORT_3PM"
    SHORT_LAAF = "SHORT_LAAF"
    LONG_ORDER_FLOW = "LONG_ORDER_FLOW"
    SHORT_ORDER_FLOW = "SHORT_ORDER_FLOW"
    RANGE_FADE_LONG = "RANGE_FADE_LONG"
    RANGE_FADE_SHORT = "RANGE_FADE_SHORT"
    GAMMA_BREAKOUT_LONG = "GAMMA_BREAKOUT_LONG"
    GAMMA_BREAKOUT_SHORT = "GAMMA_BREAKOUT_SHORT"
    EXPIRY_PIN_LONG = "EXPIRY_PIN_LONG"
    EXPIRY_PIN_SHORT = "EXPIRY_PIN_SHORT"
    GAMMA_SQUEEZE_LONG = "GAMMA_SQUEEZE_LONG"
    GAMMA_SQUEEZE_SHORT = "GAMMA_SQUEEZE_SHORT"
    INTRADAY_0DTE_STRADDLE = "INTRADAY_0DTE_STRADDLE"

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
    htf_aligned: bool = False
    fib_retracement: float = 0.0
    details: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}
        sig_str = self.signal_type.value if hasattr(self.signal_type, "value") else str(self.signal_type)
        is_short = "SHORT" in sig_str
        is_long = "LONG" in sig_str
        is_trade = is_long or is_short
        if self.target_3_moonshot == 0.0 and self.target_2 > 0 and self.entry_price > 0:
            diff = abs(self.target_2 - self.entry_price)
            if is_short:
                self.target_3_moonshot = round(min(self.target_2 - (diff * 0.618), self.target_2), 2)
            else:
                self.target_3_moonshot = round(max(self.target_2 + (diff * 0.618), self.target_2), 2)
        elif is_trade and self.target_2 > 0 and self.entry_price > 0:
            # Guard against an explicitly-passed but inverted moonshot target (T3 must
            # extend beyond T2 in the trade direction) instead of silently shipping a
            # nonsensical target ladder to the journal/UI.
            diff = abs(self.target_2 - self.entry_price)
            if is_short and self.target_3_moonshot > self.target_2:
                self.target_3_moonshot = round(self.target_2 - max(diff * 0.618, 1.0), 2)
            elif is_long and self.target_3_moonshot < self.target_2:
                self.target_3_moonshot = round(self.target_2 + max(diff * 0.618, 1.0), 2)
        if self.pyramid_trigger == 0.0 and self.entry_price > 0:
            if is_short:
                self.pyramid_trigger = round(self.entry_price - 25.0, 2)
            else:
                self.pyramid_trigger = round(self.entry_price + 25.0, 2)
        if is_trade and self.entry_price > 0 and self.sl_price == self.entry_price:
            # SL == entry is not a stop; nudge it a minimal safety distance away rather
            # than shipping a trade ticket with zero effective risk-defined stop.
            nudge = max(abs(self.entry_price) * 0.001, 5.0)
            self.sl_price = round(self.entry_price - nudge if is_long else self.entry_price + nudge, 2)

    @property
    def stop_loss(self) -> float:
        return self.sl_price

    @stop_loss.setter
    def stop_loss(self, value: float):
        self.sl_price = float(value)

    @property
    def target1(self) -> float:
        return self.target_1

    @target1.setter
    def target1(self, value: float):
        self.target_1 = float(value)

    @property
    def target2(self) -> float:
        return self.target_2

    @target2.setter
    def target2(self, value: float):
        self.target_2 = float(value)

    @property
    def target3(self) -> float:
        return self.target_3_moonshot

    @target3.setter
    def target3(self, value: float):
        self.target_3_moonshot = float(value)

# Vetoes that block EVERY candidate this bar rather than just the one being tested.
# Everything else (direction-specific skew/HFI/HTF/wall vetoes, edge quarantine, the
# confluence floor) rejects a single candidate and the ladder keeps looking.
_ABSOLUTE_VETOES = {"VPIN_TOXICITY", "SESSION_RISK_LIMIT"}


class StrategyEngine:
    """Vectorized and streaming bar-by-bar JustNifty v3.5 institutional strategy rules evaluator."""
    def __init__(self, edge_table: Optional[Any] = None):
        self.kalman_filter = KalmanFilterTrendEstimator()
        self.markov_switcher = MarkovRegimeSwitcher()
        self.macro_engine = GlobalMacroEngine()
        self.cointegrator = MultiAssetKalmanCointegrator()
        self.vol_intelligence = VolatilityIntelligence()
        self.session_losses: int = 0
        self.last_session_date: Optional[Any] = None
        # Walk-forward OOS edge table. Absent/empty table is permissive by design
        # (EdgeTable.is_tradeable returns True for unmeasured setups), so trading
        # behaviour is unchanged until a walk-forward run actually populates it.
        if edge_table is None:
            from src.edge_harness import EdgeTable
            edge_table = EdgeTable.load_from_disk()
        self.edge_table = edge_table



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
        prev_close: Optional[float] = None,
        hfi_score: float = 0.0,
        option_chain_df: Optional[pd.DataFrame] = None,
        options_context: Optional[Dict[str, Any]] = None
    ) -> Signal:
        self._last_vol_report = None
        self._last_lunch_lull = False
        self._last_markov_info = None
        self._last_options_context = options_context

        signal = self._evaluate_bar_core(
            df_5m, current_idx, df_daily, df_hourly, df_15m, df_1h,
            live_iv, live_vix, pre_open_gap, prev_close,
            hfi_score=hfi_score, option_chain_df=option_chain_df,
            options_context=options_context
        )

        if signal.details is None:
            signal.details = {}

        if getattr(self, '_last_vol_report', None) is not None:
            signal.details['vol_report'] = self._last_vol_report

        # Stamp the active regime onto EVERY signal, not just the fall-through WAIT.
        # The walk-forward harness buckets edge stats by (setup_id, regime) and the live
        # edge gate looks trades up by the same key — if trade signals carry no regime the
        # two disagree, every lookup misses, and the quarantine gate silently never fires.
        if "markov_regime" not in signal.details:
            signal.details["markov_regime"] = self._last_markov_info or {"active_regime": "UNKNOWN"}

        if options_context is not None:
            signal.details['options_context'] = options_context
            
        if getattr(self, '_last_lunch_lull', False) and signal.signal_type != SignalType.WAIT:
            signal.reason += " [LUNCH LULL: Reduced confidence, halved sizing recommended]"
            
        return signal

    def _apply_universal_gates(
        self,
        candidate_direction: str,
        close: float,
        skew_info: Dict[str, Any],
        vpin_info: Dict[str, Any],
        hfi_score: float,
        gex_info: Dict[str, Any],
        htf_regime: Dict[str, Any],
        session_risk_state: Optional[SessionRiskState] = None,
        current_bar_idx: int = 0,
        options_context: Optional[Dict[str, Any]] = None,
        candidate_signal_type: Optional[str] = None
    ) -> Tuple[bool, str, Dict[str, Any]]:
        audit = {
            "vpin": vpin_info.get("vpin", 0.0),
            "skew_zscore": skew_info.get("skew_zscore", 0.0),
            "hfi_score": hfi_score,
            "gex_regime": gex_info.get("is_positive_gamma", True),
            "htf_aligned_long": htf_regime.get("htf_aligned_long", False),
            "htf_aligned_short": htf_regime.get("htf_aligned_short", False),
            "signal_type": candidate_signal_type or "",
            "passed": True,
            "veto_gate": None
        }

        # 0. DATA SUFFICIENCY (GATE_FAIL_TO_WAIT)
        #
        # Almost every gate below fails OPEN when its input is missing: a absent VPIN
        # reads 0.0 and passes, an absent skew has is_crash_hedging=False and passes, an
        # absent chain leaves walls unverified and the pin check is skipped. So the exact
        # moment the desk is blindest — a data outage — is when the fewest vetoes can
        # fire. GATE_FAIL_TO_WAIT existed in config and was imported here but never read.
        #
        # A single missing input degrades to reduced size (handled downstream by
        # POSITIONING_UNVERIFIED_SIZE_CAP). Losing the option chain takes out skew, walls
        # and positioning together — at that point this is no longer an options desk and
        # it should stand aside rather than trade on what remains.
        missing_inputs: List[str] = []
        if str(skew_info.get("data_quality", "")).upper() == "SYNTHETIC":
            missing_inputs.append("25d_skew")
        if not gex_info.get("walls_verified", False):
            missing_inputs.append("dealer_walls")
        if not (options_context and options_context.get("dir_flow")):
            missing_inputs.append("positioning_flow")
        audit["missing_inputs"] = missing_inputs

        if GATE_FAIL_TO_WAIT and len(missing_inputs) >= GATE_MIN_MISSING_TO_BLOCK:
            audit["passed"] = False
            audit["veto_gate"] = "INSUFFICIENT_GATE_DATA"
            return False, (
                f"Data Sufficiency Gate: {len(missing_inputs)} core inputs unavailable "
                f"({', '.join(missing_inputs)}). Gates cannot be evaluated — standing aside."
            ), audit

        # 1. VPIN Flow Toxicity Veto
        vpin_val = float(vpin_info.get("vpin", 0.0))
        if vpin_val > VPIN_TOXICITY_THRESHOLD:
            audit["passed"] = False
            audit["veto_gate"] = "VPIN_TOXICITY"
            return False, f"VPIN Flow Toxicity Veto: Toxic order flow detected (VPIN={vpin_val:.2f} > {VPIN_TOXICITY_THRESHOLD}). Entries blocked.", audit

        # 2. 25-Delta Put Skew Crash Veto
        if candidate_direction == "LONG" and skew_info.get("is_crash_hedging", False):
            audit["passed"] = False
            audit["veto_gate"] = "SKEW_CRASH_HEDGING"
            return False, f"25-Delta Put Skew Crash Gate: Long blocked. Put Skew spiked (Z={skew_info.get('skew_zscore', 0):+.2f}) indicating crash hedging.", audit

        # 3. Heavyweight Flow Index (HFI) Alignment Veto
        if candidate_direction == "LONG" and hfi_score < -0.20:
            audit["passed"] = False
            audit["veto_gate"] = "HFI_BEARISH_DIVERGENCE"
            return False, f"Heavyweight Flow Veto: Long blocked. Top-5 Heavyweights are net negative (HFI={hfi_score:+.2f}).", audit
        elif candidate_direction == "SHORT" and hfi_score > 0.20:
            audit["passed"] = False
            audit["veto_gate"] = "HFI_BULLISH_DIVERGENCE"
            return False, f"Heavyweight Flow Veto: Short blocked. Top-5 Heavyweights are net positive (HFI={hfi_score:+.2f}).", audit

        # 4. Dealer GEX Wall Pinning Defense
        if gex_info.get("is_positive_gamma", False):
            call_wall = float(gex_info.get("call_wall_strike", 999999.0))
            put_wall = float(gex_info.get("put_wall_strike", 0.0))
            if candidate_direction == "LONG" and (call_wall - GEX_WALL_BUFFER_PTS <= close <= call_wall + GEX_WALL_BUFFER_PTS):
                audit["passed"] = False
                audit["veto_gate"] = "GEX_CALL_WALL_PIN"
                return False, f"Dealer GEX Pin Veto: Long blocked. Spot at Call Wall ({call_wall:.0f}) in Positive Gamma regime.", audit
            elif candidate_direction == "SHORT" and (put_wall - GEX_WALL_BUFFER_PTS <= close <= put_wall + GEX_WALL_BUFFER_PTS):
                audit["passed"] = False
                audit["veto_gate"] = "GEX_PUT_WALL_PIN"
                return False, f"Dealer GEX Pin Veto: Short blocked. Spot at Put Wall ({put_wall:.0f}) in Positive Gamma regime.", audit

        # 5. Higher Timeframe Alignment Veto
        if candidate_direction == "LONG" and not htf_regime.get("htf_aligned_long", False):
            audit["passed"] = False
            audit["veto_gate"] = "HTF_NOT_ALIGNED_LONG"
            return False, f"HTF Confluence Veto: Long not supported by 15m ({htf_regime.get('tf_15m', {}).get('bias')}) / 1H ({htf_regime.get('tf_1h', {}).get('bias')}).", audit
        elif candidate_direction == "SHORT" and not htf_regime.get("htf_aligned_short", False):
            audit["passed"] = False
            audit["veto_gate"] = "HTF_NOT_ALIGNED_SHORT"
            return False, f"HTF Confluence Veto: Short not supported by 15m ({htf_regime.get('tf_15m', {}).get('bias')}) / 1H ({htf_regime.get('tf_1h', {}).get('bias')}).", audit

        # 6. Positioning Flow Direction Veto
        if options_context and options_context.get("dir_flow"):
            d_vec = float(options_context["dir_flow"].get("directional_vector", 0.0))
            if candidate_direction == "LONG" and d_vec <= -POSITIONING_VETO_STRENGTH:
                audit["passed"] = False
                audit["veto_gate"] = "POSITIONING_OPPOSES_CHART"
                return False, f"Positioning Veto: Long blocked. Bearish options flow (D={d_vec:+.2f}).", audit
            elif candidate_direction == "SHORT" and d_vec >= POSITIONING_VETO_STRENGTH:
                audit["passed"] = False
                audit["veto_gate"] = "POSITIONING_OPPOSES_CHART"
                return False, f"Positioning Veto: Short blocked. Bullish options flow (D={d_vec:+.2f}).", audit

        # 7. Session Risk Rails
        if session_risk_state:
            can_trade, reason = session_risk_state.can_take_new_trade(current_bar_idx)
            if not can_trade:
                audit["passed"] = False
                audit["veto_gate"] = "SESSION_RISK_LIMIT"
                return False, f"Session Risk Gate: {reason}", audit

        # 8. Term Structure Inversion Crisis Gate (IMP-4)
        if options_context and options_context.get("vol_report"):
            vol_r = options_context["vol_report"]
            ts_regime = vol_r.get("term_structure_regime", {})
            if isinstance(ts_regime, dict) and ts_regime.get("is_crisis", False):
                if candidate_direction == "LONG":
                    audit["passed"] = False
                    audit["veto_gate"] = "TERM_STRUCTURE_CRISIS"
                    return False, f"Term Structure Crisis Gate: Long blocked. IV backwardation detected (slope={ts_regime.get('slope', 0):.4f}). Risk-off mode.", audit

        # 9. Gamma Flip Level Regime Gate (IMP-3)
        if not gex_info.get("is_positive_gamma", True):
            # In -Gamma regime, block range fade strategies (dealers will amplify the move)
            sig_name = str(audit.get("signal_type", ""))
            if "RANGE_FADE" in sig_name:
                audit["passed"] = False
                audit["veto_gate"] = "GAMMA_REGIME_BLOCKS_FADE"
                return False, f"Gamma Regime Gate: Range Fade blocked in -Γ expansion regime. Use momentum strategies instead.", audit

        # 10. Smart Money Institutional Flow Score Veto
        if options_context:
            inst_flow = options_context.get("inst_flow") or options_context.get("flow_score_data")
            flow_val = None
            if isinstance(inst_flow, dict):
                flow_val = inst_flow.get("flow_score")
            elif "flow_score" in options_context:
                flow_val = options_context["flow_score"]

            if flow_val is not None:
                audit["flow_score"] = float(flow_val)
                if candidate_direction == "LONG" and float(flow_val) < 30.0:
                    audit["passed"] = False
                    audit["veto_gate"] = "INSTITUTIONAL_FLOW_BEARISH_VETO"
                    return False, f"Institutional Flow Veto: Long blocked. Smart Money flow is net bearish (Flow Score={float(flow_val):.1f} < 30).", audit
                elif candidate_direction == "SHORT" and float(flow_val) > 70.0:
                    audit["passed"] = False
                    audit["veto_gate"] = "INSTITUTIONAL_FLOW_BULLISH_VETO"
                    return False, f"Institutional Flow Veto: Short blocked. Smart Money flow is net bullish (Flow Score={float(flow_val):.1f} > 70).", audit

        # 11. Variance Risk Premium (VRP) Backwardation Veto
        if options_context:
            vrp_data = options_context.get("vrp_data") or options_context.get("vrp_info")
            vrp_val = None
            if isinstance(vrp_data, dict):
                vrp_val = vrp_data.get("vrp")
            elif "vrp" in options_context:
                vrp_val = options_context["vrp"]

            if vrp_val is not None:
                audit["vrp"] = float(vrp_val)
                sig_name = str(audit.get("signal_type", ""))
                if ("RANGE_FADE" in sig_name or "STRADDLE" in sig_name) and float(vrp_val) < -0.02:
                    audit["passed"] = False
                    audit["veto_gate"] = "VRP_BACKWARDATION_VETO"
                    return False, f"Variance Risk Premium Gate: Short premium blocked. Negative VRP (IV in backwardation relative to Realized Vol, VRP={float(vrp_val):+.3f}).", audit

        return True, "PASSED", audit

    def _finalize_candidate(
        self,
        candidate_sig: Signal,
        sub_df: pd.DataFrame,
        htf_regime: Dict[str, Any],
        kalman_vel: float,
        kalman_z: float,
        markov_info: Dict[str, Any],
        ofi_info: Dict[str, Any],
        gex_info: Dict[str, Any],
        vp_info: Dict[str, Any],
        atr_14: float,
        gate_audit: Dict[str, Any],
        options_context: Optional[Dict[str, Any]] = None
    ) -> Signal:
        sig_str = candidate_sig.signal_type.value if hasattr(candidate_sig.signal_type, "value") else str(candidate_sig.signal_type)
        is_long = "LONG" in sig_str
        is_short = "SHORT" in sig_str

        # 1. Stop Loss & Target Hygiene
        if is_long or is_short:
            sl_dist = abs(candidate_sig.entry_price - candidate_sig.sl_price)
            min_sl = max(STOP_MIN_ATR_FRACTION * atr_14, 15.0)
            max_sl = min(STOP_MAX_POINTS, STOP_MAX_ATR_MULTIPLE * atr_14)
            max_sl = max(max_sl, min_sl)  # Never invert the band
            
            if sl_dist < min_sl:
                candidate_sig.sl_price = round(candidate_sig.entry_price - min_sl if is_long else candidate_sig.entry_price + min_sl, 2)
            elif sl_dist > max_sl:
                candidate_sig.sl_price = round(candidate_sig.entry_price - max_sl if is_long else candidate_sig.entry_price + max_sl, 2)

            # Ensure proper target ladder
            if is_long:
                if candidate_sig.target_1 <= candidate_sig.entry_price:
                    candidate_sig.target_1 = round(candidate_sig.entry_price + 1.2 * atr_14, 2)
                if candidate_sig.target_2 <= candidate_sig.target_1:
                    candidate_sig.target_2 = round(candidate_sig.target_1 + 1.3 * atr_14, 2)
                if candidate_sig.target_3_moonshot <= candidate_sig.target_2:
                    candidate_sig.target_3_moonshot = round(candidate_sig.target_2 + 1.5 * atr_14, 2)
            elif is_short:
                if candidate_sig.target_1 >= candidate_sig.entry_price:
                    candidate_sig.target_1 = round(candidate_sig.entry_price - 1.2 * atr_14, 2)
                if candidate_sig.target_2 >= candidate_sig.target_1:
                    candidate_sig.target_2 = round(candidate_sig.target_1 - 1.3 * atr_14, 2)
                if candidate_sig.target_3_moonshot >= candidate_sig.target_2:
                    candidate_sig.target_3_moonshot = round(candidate_sig.target_2 - 1.5 * atr_14, 2)

        # 2. Pre-Decision Confluence Score
        from src.signal_journal import calculate_confluence_score
        score, grade = calculate_confluence_score(
            candidate_sig,
            sub_df,
            htf_data=htf_regime,
            kalman_vel=kalman_vel,
            kalman_z=kalman_z,
            regime_state=markov_info,
            ofi_data=ofi_info,
            gex_data=gex_info,
            vol_profile=vp_info,
            options_context=options_context
        )

        if candidate_sig.details is None:
            candidate_sig.details = {}
        candidate_sig.details["confluence_score"] = score
        candidate_sig.details["confluence_grade"] = grade
        candidate_sig.details["gate_audit"] = gate_audit

        # 3. Confluence Score Veto (Floor check)
        if score < SIGNAL_MIN_CONFLUENCE and (is_long or is_short):
            return Signal(
                signal_type=SignalType.WAIT,
                entry_price=candidate_sig.entry_price,
                sl_price=0.0,
                target_1=0.0,
                target_2=0.0,
                target_3_moonshot=0.0,
                pyramid_trigger=0.0,
                reason=f"Confluence Veto: Score {score:.1f} < {SIGNAL_MIN_CONFLUENCE:.1f} ({grade}). Setup rejected for insufficient confluence.",
                htf_aligned=candidate_sig.htf_aligned,
                fib_retracement=candidate_sig.fib_retracement,
                details=candidate_sig.details
            )

        return candidate_sig

    def _evaluate_bar_core(
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
        prev_close: Optional[float] = None,
        hfi_score: float = 0.0,
        option_chain_df: Optional[pd.DataFrame] = None,
        options_context: Optional[Dict[str, Any]] = None
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

        # Event & Holiday Blackout Gate
        from src.event_calendar import check_event_risk_gate
        bar_ts_full = bar.name.strftime("%Y-%m-%d %H:%M") if hasattr(bar.name, "strftime") else str(bar.name)
        ev_passed, ev_reason, ev_audit = check_event_risk_gate(bar_ts_full)
        if not ev_passed:
            return Signal(
                signal_type=SignalType.WAIT,
                entry_price=close,
                sl_price=0.0,
                target_1=0.0,
                target_2=0.0,
                target_3_moonshot=0.0,
                pyramid_trigger=0.0,
                reason=ev_reason,
                htf_aligned=False,
                fib_retracement=0.0,
                details={"event_blackout": ev_audit}
            )

        if len(sub_df) < 15:
            return Signal(SignalType.WAIT, close, 0.0, 0.0, 0.0, 0.0, 0.0, "Accumulating bars for indicator stability", True, 0.0, {})

        # Lunch Lull Filter
        lunch_lull_active = False
        if "11:30" <= bar_time <= "13:00":
            intraday_quality = self.vol_intelligence.compute_intraday_quality_score(bar_time)
            lunch_lull_active = True
        self._last_lunch_lull = lunch_lull_active

        # Compute Stochastic Indicators & Dynamic VAKC
        ema200_series = compute_ema(sub_df["close"], EMA_SLOW)
        ema55_series = compute_ema(sub_df["close"], EMA_MID)
        ema21_series = compute_ema(sub_df["close"], EMA_FAST)
        vakc_upper, vakc_lower = compute_vakc_envelopes(sub_df, iv=live_iv)
        vwap_series, vwap_up_2sd, vwap_low_2sd = compute_vwap(sub_df)

        sub_df = sub_df.copy()
        sub_df["ema200"] = ema200_series
        sub_df["ema55"] = ema55_series
        sub_df["ema21"] = ema21_series
        sub_df["vwap"] = vwap_series
        
        hurst_info = compute_hurst_exponent(sub_df["close"])
        ofi_info = compute_order_flow_imbalance(sub_df)

        if options_context and options_context.get("gex_chart"):
            g = options_context["gex_chart"]
            gex_info = {
                "call_wall_strike": float(g.get("call_wall_strike", 999999.0)),
                "put_wall_strike": float(g.get("put_wall_strike", 0.0)),
                "zero_gex_strike": float(g.get("zero_gex_strike", close)),
                "is_positive_gamma": str(g.get("net_dealer_regime", "")).startswith("POSITIVE") or str(g.get("net_dealer_regime", "")).startswith("DEALER_LONG"),
                "walls_verified": True
            }
        else:
            gex_info = compute_dealer_gex(close)
            gex_info["walls_verified"] = False

        vp_info = compute_volume_profile(sub_df)
        gap_info = compute_pre_open_gap_filter(sub_df, prev_close=prev_close, pre_open_data=pre_open_gap)
        cpr_info = compute_cpr(df_daily if df_daily is not None else sub_df)
        
        # Latent Kalman Spot Velocity, Markov Regime Model & Cointegration
        kalman_price, kalman_vel, kalman_z = self.kalman_filter.update(close)
        markov_info = self.markov_switcher.infer_regimes(sub_df)
        self._last_markov_info = markov_info
        dfa_info = compute_dfa_alpha(sub_df["close"])
        vpin_info = compute_vpin_toxicity(sub_df)
        cointegration = self.cointegrator.evaluate_spread_divergence(sub_df["close"])

        # Vol Intelligence: IV-RV Spread & Regime
        vol_report = self.vol_intelligence.generate_vol_intelligence_report(
            close_prices=sub_df['close'],
            current_iv=live_iv,
            bar_time=bar_time
        )
        vol_regime = vol_report['composite_vol_regime']
        intraday_quality = vol_report['intraday_quality']
        self._last_vol_report = vol_report

        # 25-Delta Put-Call Skew & VCR Squeeze (v5.2)
        chain_for_skew = options_context.get("chain_df") if options_context else option_chain_df
        skew_info = self.vol_intelligence.compute_25delta_skew(chain_for_skew, spot=close, iv_baseline=live_iv)
        skew_info["data_quality"] = "VERIFIED" if (chain_for_skew is not None and not chain_for_skew.empty) else "SYNTHETIC"
        vcr_info = self.vol_intelligence.compute_vcr_squeeze(sub_df['close'])
        
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
        prev_close_val = float(sub_df.iloc[-2]["close"]) if len(sub_df) >= 2 else close

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
        absorption_trap = detect_absorption_traps(sub_df, key_levels=key_levels)
        
        # Realized Volatility / Implied Volatility Ratio
        log_rets = np.diff(np.log(np.maximum(sub_df["close"].tail(30).values, 1.0)))
        realized_vol = float(np.std(log_rets, ddof=1) * np.sqrt(252 * 75)) if len(log_rets) > 5 else live_iv
        vol_ratio = round(realized_vol / max(live_iv, 0.05), 2)

        # Candidate rejections are collected so the ladder can keep evaluating instead of
        # ending the bar on the first setup that fails. Only an ABSOLUTE veto (toxic
        # flow, session lock) stops everything.
        _rejections: List[str] = []
        _rejected: List[Dict[str, Any]] = []
        _absolute_veto: List[str] = []

        # Define internal candidate gate check and pre-decision finalizer
        def _check_and_return(direction: str, candidate_sig: Signal) -> Signal:
            setup_id = candidate_sig.signal_type.value if hasattr(candidate_sig.signal_type, "value") else str(candidate_sig.signal_type)
            active_regime = markov_info.get("active_regime", "UNKNOWN")
            edge_stat_early = self.edge_table.lookup(setup_id, active_regime) if self.edge_table else None
            edge_status_early = getattr(edge_stat_early, "status", "UNMEASURED")

            gate_ok, gate_msg, audit = self._apply_universal_gates(
                direction, close, skew_info, vpin_info, hfi_score, gex_info, htf_regime,
                current_bar_idx=current_idx, options_context=options_context,
                candidate_signal_type=setup_id
            )

            if not gate_ok:
                # Direction- and candidate-specific vetoes reject THIS candidate only;
                # the ladder must keep looking. Absolute vetoes (toxic flow, session
                # lock) genuinely block every trade this bar and stop it.
                _rejections.append(gate_msg)
                _rejected.append({
                    "signal_type": setup_id,
                    "direction": direction,
                    "entry": candidate_sig.entry_price,
                    "sl": candidate_sig.sl_price,
                    "t1": candidate_sig.target_1,
                    "t2": candidate_sig.target_2,
                    "t3": getattr(candidate_sig, "target_3_moonshot", 0.0),
                    "veto_gate": audit.get("veto_gate", "GATE_BLOCKED"),
                    "reason": gate_msg,
                    "confluence": (candidate_sig.details or {}).get("confluence_score", 0.0),
                    "edge_status": edge_status_early,
                })
                if audit.get("veto_gate") in _ABSOLUTE_VETOES:
                    _absolute_veto.append(gate_msg)
                return None

            # Walk-forward OOS edge gate: quarantined (measured negative-EV) setups
            # are blocked in the regimes where they lost money.
            #
            # This must NOT terminate the ladder. Returning WAIT here meant quarantining
            # a loser converted it into a no-trade rather than into the next-best trade —
            # so a QUARANTINED setup firing early still pre-empted a better one later in
            # the ladder, forfeiting its edge instead of capturing it.
            if self.edge_table is not None and not self.edge_table.is_tradeable(setup_id, active_regime):
                audit["passed"] = False
                audit["veto_gate"] = "EDGE_TABLE_QUARANTINED"
                _rej_msg = f"Edge Table Veto: '{setup_id}' is QUARANTINED in {active_regime} regime (measured negative out-of-sample EV)."
                _rejections.append(_rej_msg)
                _rejected.append({
                    "signal_type": setup_id,
                    "direction": direction,
                    "entry": candidate_sig.entry_price,
                    "sl": candidate_sig.sl_price,
                    "t1": candidate_sig.target_1,
                    "t2": candidate_sig.target_2,
                    "t3": getattr(candidate_sig, "target_3_moonshot", 0.0),
                    "veto_gate": "EDGE_TABLE_QUARANTINED",
                    "reason": _rej_msg,
                    "confluence": (candidate_sig.details or {}).get("confluence_score", 0.0),
                    "edge_status": "QUARANTINED",
                })
                return None

            candidate_sig.htf_aligned = htf_aligned_long if direction == "LONG" else htf_aligned_short

            # Sizing factors: VCR squeeze & Unverified Positioning cap
            if candidate_sig.details is None: candidate_sig.details = {}
            if option_chain_df is not None and not option_chain_df.empty:
                candidate_sig.details["option_chain_available"] = True
            if vcr_info.get("vcr_ratio", 1.0) < VCR_SQUEEZE_THRESHOLD:
                candidate_sig.details["size_factor"] = 0.5
            if self.edge_table is not None:
                edge_stats = self.edge_table.lookup(setup_id, active_regime)
                # Only let the edge table scale size once it has actually measured this
                # setup. An unmeasured setup keeps its existing size rather than being
                # silently halved, so an empty edge table is a no-op on live sizing.
                if edge_stats is not None:
                    edge_size = self.edge_table.get_sizing_factor(setup_id, active_regime)
                    candidate_sig.details["size_factor"] = min(candidate_sig.details.get("size_factor", 1.0), edge_size)
                candidate_sig.details["edge_status"] = getattr(edge_stats, "status", "UNMEASURED")
            if skew_info.get("data_quality") == "SYNTHETIC" or not gex_info.get("walls_verified", False):
                candidate_sig.details["size_factor"] = min(candidate_sig.details.get("size_factor", 1.0), POSITIONING_UNVERIFIED_SIZE_CAP)
                audit["data_quality"] = "POSITIONING_UNVERIFIED"

            _final = self._finalize_candidate(
                candidate_sig, sub_df, htf_regime, kalman_vel, kalman_z,
                markov_info, ofi_info, gex_info, vp_info, atr_14, audit,
                options_context=options_context
            )
            # A confluence-floor rejection is candidate-specific: try the next setup
            # rather than ending the bar on the first weak one.
            if _final.signal_type == SignalType.WAIT:
                _rejections.append(_final.reason)
                _rejected.append({
                    "signal_type": setup_id,
                    "direction": direction,
                    "entry": candidate_sig.entry_price,
                    "sl": candidate_sig.sl_price,
                    "t1": candidate_sig.target_1,
                    "t2": candidate_sig.target_2,
                    "t3": getattr(candidate_sig, "target_3_moonshot", 0.0),
                    "veto_gate": "CONFLUENCE_FLOOR",
                    "reason": _final.reason,
                    "confluence": (_final.details or {}).get("confluence_score", 0.0),
                    "edge_status": candidate_sig.details.get("edge_status", "UNMEASURED"),
                })
                return None
            return _final

        # 4.05 Passive Institutional Limit Absorption Trap Strategy (v5.2)
        if absorption_trap["is_absorption"]:
            if absorption_trap["type"] == "BULLISH_ABSORPTION" and (close > bar_open or ofi_info["buyer_defense"]):
                _c = _check_and_return("LONG", Signal(
                    signal_type=SignalType.LONG_ORDER_FLOW,
                    entry_price=close,
                    sl_price=absorption_trap["suggested_sl"],
                    target_1=round(close + 1.2 * atr_14, 2),
                    target_2=round(close + 2.5 * atr_14, 2),
                    target_3_moonshot=round(max(upper_vakc_val, upper_2sd), 2),
                    pyramid_trigger=round(close + 15.0, 2),
                    reason=f"Institutional Support Absorption: {absorption_trap['reason']}",
                    htf_aligned=htf_aligned_long,
                    fib_retracement=0.50,
                    details={"absorption": absorption_trap, "skew": skew_info, "vcr": vcr_info, "order_flow": order_flow}
                ))
                if _c is not None:
                    return _c
            elif absorption_trap["type"] == "BEARISH_ABSORPTION" and (close < bar_open or ofi_info["seller_defense"]):
                _c = _check_and_return("SHORT", Signal(
                    signal_type=SignalType.SHORT_ORDER_FLOW,
                    entry_price=close,
                    sl_price=absorption_trap["suggested_sl"],
                    target_1=round(close - 1.2 * atr_14, 2),
                    target_2=round(close - 2.5 * atr_14, 2),
                    target_3_moonshot=round(min(lower_vakc_val, lower_2sd), 2),
                    pyramid_trigger=round(close - 15.0, 2),
                    reason=f"Institutional Resistance Absorption: {absorption_trap['reason']}",
                    htf_aligned=htf_aligned_short,
                    fib_retracement=0.50,
                    details={"absorption": absorption_trap, "skew": skew_info, "vcr": vcr_info, "order_flow": order_flow}
                ))
                if _c is not None:
                    return _c

        # 4.1 Liquidity Sweep Trap Strategy (SSL / BSL Purges)
        if microstructure["liquidity_sweep_detected"] and microstructure["sweep_event"]:
            sw = microstructure["sweep_event"]
            if sw["side"] == "LONG" and (close > bar_open or close > prev_close_val or ofi_info["buyer_defense"] or order_flow["recent_delta"] >= 0):
                _c = _check_and_return("LONG", Signal(
                    signal_type=SignalType.LONG_ORDER_FLOW,
                    entry_price=close,
                    sl_price=sw["suggested_sl"],
                    target_1=round(close + 1.2 * atr_14, 2),
                    target_2=round(close + 2.5 * atr_14, 2),
                    target_3_moonshot=round(close + 4.0 * atr_14, 2),
                    pyramid_trigger=round(sw["swept_swing_low"] + 15.0, 2),
                    reason=f"Bullish SSL Liquidity Sweep Trap: {sw['thesis']} | Institutional Absorption Reversal.",
                    htf_aligned=htf_aligned_long,
                    fib_retracement=0.50,
                    details={"sweep": sw, "microstructure": microstructure, "order_flow": order_flow, "vol_ratio": vol_ratio}
                ))
                if _c is not None:
                    return _c
            elif sw["side"] == "SHORT" and (close < bar_open or close < prev_close_val or ofi_info["seller_defense"] or order_flow["recent_delta"] <= 0):
                _c = _check_and_return("SHORT", Signal(
                    signal_type=SignalType.SHORT_ORDER_FLOW,
                    entry_price=close,
                    sl_price=sw["suggested_sl"],
                    target_1=round(close - 1.2 * atr_14, 2),
                    target_2=round(close - 2.5 * atr_14, 2),
                    target_3_moonshot=round(close - 4.0 * atr_14, 2),
                    pyramid_trigger=round(sw["swept_swing_high"] - 15.0, 2),
                    reason=f"Bearish BSL Liquidity Sweep Trap: {sw['thesis']} | Institutional Distribution Reversal.",
                    htf_aligned=htf_aligned_short,
                    fib_retracement=0.50,
                    details={"sweep": sw, "microstructure": microstructure, "order_flow": order_flow, "vol_ratio": vol_ratio}
                ))
                if _c is not None:
                    return _c

        # 4.2 Mean-Reversion Strategy (Active in MEAN_REVERTING_CHOP regime)
        if markov_info['active_regime'] == 'MEAN_REVERTING_CHOP':
            # LONG: price at or near VAL with buyer defense confirmation
            if close <= vp_info.get('val', 0) + VAL_BUFFER_ATR_MULT * atr_14 and ofi_info['buyer_defense'] and (close > bar_open or close > lower_2sd):
                _c = _check_and_return("LONG", Signal(
                    signal_type=SignalType.LONG,
                    entry_price=close,
                    sl_price=round(close - 0.8 * atr_14, 2),
                    target_1=round(vp_info.get('poc', close + 20), 2),
                    target_2=round(vp_info.get('vah', close + 40), 2),
                    reason="Mean-Reversion Long: Price at VAL in Choppy Regime. OFI confirms buyer defense. Quick scalp to POC.",
                    htf_aligned=htf_aligned_long,
                    details={}
                ))
                if _c is not None:
                    return _c
            # SHORT: price at or near VAH with seller defense confirmation
            if close >= vp_info.get('vah', 99999) - VAL_BUFFER_ATR_MULT * atr_14 and ofi_info['seller_defense'] and (close < bar_open or close < upper_2sd):
                _c = _check_and_return("SHORT", Signal(
                    signal_type=SignalType.SHORT,
                    entry_price=close,
                    sl_price=round(close + 0.8 * atr_14, 2),
                    target_1=round(vp_info.get('poc', close - 20), 2),
                    target_2=round(vp_info.get('val', close - 40), 2),
                    reason="Mean-Reversion Short: Price at VAH in Choppy Regime. OFI confirms seller defense. Quick scalp to POC.",
                    htf_aligned=htf_aligned_short,
                    details={}
                ))
                if _c is not None:
                    return _c

        # 4.25 3:00 PM (15:00) Hardened Breakout Strategy (v5.3: Volume + GEX + Auto-Squareoff protection)
        if bar_time in ["15:05", "15:10"]:
            is_expiry_day = options_context.get("is_expiry_day", False) if options_context else False
            if not is_expiry_day:
                three_pm_indices = [
                    i for i, idx in enumerate(df_5m.index[:current_idx + 1])
                    if hasattr(idx, "strftime") and idx.strftime("%H:%M") == "15:00"
                ]
                if three_pm_indices:
                    candle_3pm = df_5m.iloc[three_pm_indices[-1]]
                    avg_vol_3pm = float(sub_df['volume'].mean()) if 'volume' in sub_df.columns else 0.0
                    curr_vol_3pm = float(bar.get('volume', 0))
                    volume_confirmed = curr_vol_3pm > 1.5 * avg_vol_3pm if avg_vol_3pm > 0 else True
                    
                    if volume_confirmed and close > float(candle_3pm["high"]):
                        _c = _check_and_return("LONG", Signal(
                            signal_type=SignalType.LONG_3PM,
                            entry_price=close,
                            sl_price=float(candle_3pm["low"]),
                            target_1=round(close + 1.2 * atr_14, 2),
                            target_2=round(close + 2.5 * atr_14, 2),
                            target_3_moonshot=round(close + 4.0 * atr_14, 2),
                            pyramid_trigger=round(float(candle_3pm["high"]) + 10.0, 2),
                            reason="3 PM Strategy (Hardened): Bullish breakout above 15:00 candle High. Volume confirmed. Exit before 15:15.",
                            htf_aligned=htf_aligned_long,
                            fib_retracement=0.0,
                            details={"3pm_high": float(candle_3pm["high"]), "3pm_low": float(candle_3pm["low"]), "volume_ratio": round(curr_vol_3pm / max(avg_vol_3pm, 1), 2)}
                        ))
                        if _c is not None:
                            return _c
                    elif volume_confirmed and close < float(candle_3pm["low"]):
                        _c = _check_and_return("SHORT", Signal(
                            signal_type=SignalType.SHORT_3PM,
                            entry_price=close,
                            sl_price=float(candle_3pm["high"]),
                            target_1=round(close - 1.2 * atr_14, 2),
                            target_2=round(close - 2.5 * atr_14, 2),
                            target_3_moonshot=round(close - 4.0 * atr_14, 2),
                            pyramid_trigger=round(float(candle_3pm["low"]) - 10.0, 2),
                            reason="3 PM Strategy (Hardened): Bearish breakdown below 15:00 candle Low. Volume confirmed. Exit before 15:15.",
                            htf_aligned=htf_aligned_short,
                            fib_retracement=0.0,
                            details={"3pm_high": float(candle_3pm["high"]), "3pm_low": float(candle_3pm["low"]), "volume_ratio": round(curr_vol_3pm / max(avg_vol_3pm, 1), 2)}
                        ))
                        if _c is not None:
                            return _c

        # 4.3 IB Breakout Strategy (Active in trending / high vol expansion regimes after 10:15 IST)
        if markov_info['active_regime'] in ['LOW_VOL_TRENDING', 'HIGH_VOL_EXPANSION'] and bar_time >= '10:15':
            ib_state = compute_initial_balance_and_day_type(sub_df)
            avg_vol = float(sub_df['volume'].tail(10).mean()) if 'volume' in sub_df.columns else 0.0
            curr_vol = float(bar.get('volume', 0))
            
            # LONG
            if close > ib_state.get('ib_high', 99999) and htf_aligned_long and curr_vol > avg_vol:
                _c = _check_and_return("LONG", Signal(
                    signal_type=SignalType.LONG,
                    entry_price=close,
                    sl_price=ib_state.get('ib_low', close - atr_14),
                    target_1=round(close + 1.5 * atr_14, 2),
                    target_2=round(close + 3.0 * atr_14, 2),
                    target_3_moonshot=round(close + 5.0 * atr_14, 2),
                    pyramid_trigger=round(close + 1.0 * atr_14, 2),
                    reason="IB Breakout Long: Price cleared Initial Balance High in Trending Regime. HTF aligned. Volume confirmed.",
                    htf_aligned=htf_aligned_long,
                    details={'ib_state': ib_state}
                ))
                if _c is not None:
                    return _c
            # SHORT
            if close < ib_state.get('ib_low', 0) and htf_aligned_short and curr_vol > avg_vol:
                _c = _check_and_return("SHORT", Signal(
                    signal_type=SignalType.SHORT,
                    entry_price=close,
                    sl_price=ib_state.get('ib_high', close + atr_14),
                    target_1=round(close - 1.5 * atr_14, 2),
                    target_2=round(close - 3.0 * atr_14, 2),
                    target_3_moonshot=round(close - 5.0 * atr_14, 2),
                    pyramid_trigger=round(close - 1.0 * atr_14, 2),
                    reason="IB Breakdown Short: Price broke Initial Balance Low in Trending Regime. HTF aligned. Volume confirmed.",
                    htf_aligned=htf_aligned_short,
                    details={'ib_state': ib_state}
                ))
                if _c is not None:
                    return _c

        # 5. Auction Market Theory (AMT) Value Area Trigger Check
        amt_trigger = detect_volume_profile_triggers(sub_df, vp_info, ofi_info, atr_14=atr_14)
        if amt_trigger["trigger"] in ["VAH_REJECTION", "VAL_REJECTION"] and amt_trigger["confidence"] >= 0.85:
            if amt_trigger["side"] == "LONG" and (close >= vp_info.get("val", 0.0) or close > ema55):
                _c = _check_and_return("LONG", Signal(
                    signal_type=SignalType.LONG,
                    entry_price=close,
                    sl_price=amt_trigger["sl"],
                    target_1=amt_trigger["target_1"],
                    target_2=amt_trigger["target_2"],
                    target_3_moonshot=round(max(upper_vakc_val, upper_2sd), 2),
                    pyramid_trigger=round(vp_info.get("poc", close) + 5.0, 2),
                    reason=f"AMT Setup Confirmed: {amt_trigger['reason']} | Value Area Defense ({htf_regime['confluence_regime']}).",
                    htf_aligned=htf_aligned_long,
                    fib_retracement=0.50,
                    details={"amt": amt_trigger, "vp": vp_info, "hurst": hurst_info, "ofi": ofi_info, "htf_regime": htf_regime, "order_flow": order_flow}
                ))
                if _c is not None:
                    return _c
            elif amt_trigger["side"] == "SHORT" and (close <= vp_info.get("vah", 999999.0) or close < ema55):
                _c = _check_and_return("SHORT", Signal(
                    signal_type=SignalType.SHORT,
                    entry_price=close,
                    sl_price=amt_trigger["sl"],
                    target_1=amt_trigger["target_1"],
                    target_2=amt_trigger["target_2"],
                    target_3_moonshot=round(min(lower_vakc_val, lower_2sd), 2),
                    pyramid_trigger=round(vp_info.get("poc", close) - 5.0, 2),
                    reason=f"AMT Setup Confirmed: {amt_trigger['reason']} | Value Area Defense ({htf_regime['confluence_regime']}).",
                    htf_aligned=htf_aligned_short,
                    fib_retracement=0.50,
                    details={"amt": amt_trigger, "vp": vp_info, "hurst": hurst_info, "ofi": ofi_info, "htf_regime": htf_regime, "order_flow": order_flow}
                ))
                if _c is not None:
                    return _c

        # 5.1 Institutional Support Reclaim & Mean-Reversion Spring (V-Reversal Setup)
        lowest_recent = float(sub_df["low"].tail(6).min())
        val_lvl = float(vp_info.get("val", 0.0))
        if (lowest_recent <= lower_2sd or lowest_recent <= lower_vakc_val or (val_lvl > 0 and lowest_recent <= val_lvl + 5.0)) and \
           close > current_vwap and close > ema21 and ofi_info["buyer_defense"] and (close > bar_open or close > prev_close_val):
            sl_lvl = round(lowest_recent - 5.0, 2)
            if abs(close - sl_lvl) <= 45.0:
                _c = _check_and_return("LONG", Signal(
                    signal_type=SignalType.LONG,
                    entry_price=close,
                    sl_price=sl_lvl,
                    target_1=round(close + 1.2 * atr_14, 2),
                    target_2=round(close + 2.5 * atr_14, 2),
                    target_3_moonshot=round(max(upper_vakc_val, upper_2sd), 2),
                    pyramid_trigger=round(current_vwap + 15.0, 2),
                    reason="Institutional Spring Reclaim: Sub-AVWAP probe rejected + 21 EMA / AVWAP Reclaimed with Buyer Delta Absorption.",
                    htf_aligned=htf_aligned_long,
                    fib_retracement=0.50,
                    details={"spring_low": lowest_recent, "vwap": current_vwap, "ema21": ema21, "ofi": ofi_info, "order_flow": order_flow}
                ))
                if _c is not None:
                    return _c

        highest_recent = float(sub_df["high"].tail(6).max())
        vah_lvl = float(vp_info.get("vah", 999999.0))
        if (highest_recent >= upper_2sd or highest_recent >= upper_vakc_val or (vah_lvl < 999999.0 and highest_recent >= vah_lvl - 5.0)) and \
           close < current_vwap and close < ema21 and ofi_info["seller_defense"] and (close < bar_open or close < prev_close_val):
            sl_lvl = round(highest_recent + 5.0, 2)
            if abs(sl_lvl - close) <= 45.0:
                _c = _check_and_return("SHORT", Signal(
                    signal_type=SignalType.SHORT,
                    entry_price=close,
                    sl_price=sl_lvl,
                    target_1=round(close - 1.2 * atr_14, 2),
                    target_2=round(close - 2.5 * atr_14, 2),
                    target_3_moonshot=round(min(lower_vakc_val, lower_2sd), 2),
                    pyramid_trigger=round(current_vwap - 15.0, 2),
                    reason="Institutional Distribution Thrust: Above-AVWAP probe rejected + 21 EMA / AVWAP Breakdown with Seller Delta Distribution.",
                    htf_aligned=htf_aligned_short,
                    fib_retracement=0.50,
                    details={"thrust_high": highest_recent, "vwap": current_vwap, "ema21": ema21, "ofi": ofi_info, "order_flow": order_flow}
                ))
                if _c is not None:
                    return _c

        # 6. Stacked Order Flow Absorption & Footprint Imbalance Setups
        if order_flow["absorption_event"] is not None:
            abs_event = order_flow["absorption_event"]
            if abs_event["type"] == "BUYER_ABSORPTION" and (close > bar_open or ofi_info["buyer_defense"] or order_flow["recent_delta"] >= 0):
                _c = _check_and_return("LONG", Signal(
                    signal_type=SignalType.LONG_ORDER_FLOW,
                    entry_price=close,
                    sl_price=abs_event["suggested_sl"],
                    target_1=round(close + 1.2 * atr_14, 2),
                    target_2=round(close + 2.5 * atr_14, 2),
                    target_3_moonshot=round(max(upper_vakc_val, upper_2sd), 2),
                    pyramid_trigger=round(close + 15.0, 2),
                    reason=f"Order Flow Buyer Absorption: {abs_event['reason']} | Key Level Defense.",
                    htf_aligned=htf_aligned_long,
                    fib_retracement=0.50,
                    details={"order_flow": order_flow, "htf_regime": htf_regime, "abs_event": abs_event}
                ))
                if _c is not None:
                    return _c
            elif abs_event["type"] == "SELLER_ABSORPTION" and (close < bar_open or ofi_info["seller_defense"] or order_flow["recent_delta"] <= 0):
                _c = _check_and_return("SHORT", Signal(
                    signal_type=SignalType.SHORT_ORDER_FLOW,
                    entry_price=close,
                    sl_price=abs_event["suggested_sl"],
                    target_1=round(close - 1.2 * atr_14, 2),
                    target_2=round(close - 2.5 * atr_14, 2),
                    target_3_moonshot=round(min(lower_vakc_val, lower_2sd), 2),
                    pyramid_trigger=round(close - 15.0, 2),
                    reason=f"Order Flow Seller Absorption: {abs_event['reason']} | Key Level Defense.",
                    htf_aligned=htf_aligned_short,
                    fib_retracement=0.50,
                    details={"order_flow": order_flow, "htf_regime": htf_regime, "abs_event": abs_event}
                ))
                if _c is not None:
                    return _c

        # 6.1 Stacked Aggressive Delta Imbalance (3+ Bars Cumulative Delta Momentum)
        if order_flow.get("has_stacked_buy") and (close > ema21 or ofi_info["buyer_defense"]):
            _c = _check_and_return("LONG", Signal(
                signal_type=SignalType.LONG_ORDER_FLOW,
                entry_price=close,
                sl_price=round(order_flow["stacked_support_zone"][0] - 5.0, 2) if order_flow["stacked_support_zone"] else round(close - 1.0 * atr_14, 2),
                target_1=round(close + 1.2 * atr_14, 2),
                target_2=round(close + 2.5 * atr_14, 2),
                target_3_moonshot=round(max(upper_vakc_val, upper_2sd), 2),
                pyramid_trigger=round(close + 15.0, 2),
                reason=f"Stacked Buying Imbalance: {order_flow['stacked_buy_count']} consecutive aggressive buy bars | Shelf Support @ {order_flow.get('stacked_support_zone')}.",
                htf_aligned=htf_aligned_long,
                fib_retracement=0.50,
                details={"order_flow": order_flow, "htf_regime": htf_regime}
            ))
            if _c is not None:
                return _c
        elif order_flow.get("has_stacked_sell") and (close < ema21 or ofi_info["seller_defense"]):
            _c = _check_and_return("SHORT", Signal(
                signal_type=SignalType.SHORT_ORDER_FLOW,
                entry_price=close,
                sl_price=round(order_flow["stacked_resistance_zone"][1] + 5.0, 2) if order_flow["stacked_resistance_zone"] else round(close + 1.0 * atr_14, 2),
                target_1=round(close - 1.2 * atr_14, 2),
                target_2=round(close - 2.5 * atr_14, 2),
                target_3_moonshot=round(min(lower_vakc_val, lower_2sd), 2),
                pyramid_trigger=round(close - 15.0, 2),
                reason=f"Stacked Selling Imbalance: {order_flow['stacked_sell_count']} consecutive aggressive sell bars | Shelf Resistance @ {order_flow.get('stacked_resistance_zone')}.",
                htf_aligned=htf_aligned_short,
                fib_retracement=0.50,
                details={"order_flow": order_flow, "htf_regime": htf_regime}
            ))
            if _c is not None:
                return _c

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
        is_trending_flow = hurst_info.get("hurst", 0.50) >= HURST_TRENDING_MIN or hurst_info.get("is_trending", True)
        if close > ema200 and long_avwap_cond and swing_range >= 35.0 and ofi_info["buyer_defense"] and is_trending_flow:
            fib = compute_fibonacci_levels(swing_high, swing_low, is_uptrend=True)
            in_pocket = fib["fib_618"] <= min(close, bar_open) and max(close, bar_open) <= (fib["fib_500"] + 10.0)
            bullish_trigger = (close > bar_open) or (close > prev_close_val)
            
            if in_pocket and bullish_trigger:
                t1 = round(close + 1.2 * atr_14, 2)
                t2 = round(close + 2.5 * atr_14, 2)
                t3_moonshot = round(max(upper_vakc_val, upper_2sd, close + 3.8 * atr_14), 2)
                pyramid_trigger_lvl = round(swing_high + 2.0, 2)
                
                _c = _check_and_return("LONG", Signal(
                    signal_type=SignalType.LONG,
                    entry_price=close,
                    sl_price=fib["sl_level"],
                    target_1=t1,
                    target_2=t2,
                    target_3_moonshot=t3_moonshot,
                    pyramid_trigger=pyramid_trigger_lvl,
                    reason=f"LONG Setup Confirmed: Above 200 EMA + Above AVWAP ({gap_info['regime']}) + Golden Pocket + OFI Defense | Persistent Trend (H={hurst_info.get('hurst', 0.52):.2f}) | HTF Aligned.",
                    htf_aligned=htf_aligned_long,
                    fib_retracement=0.55,
                    details={
                        "fib": fib, "ema200": ema200, "vwap": current_vwap, "ema21": ema21,
                        "swing_high": swing_high, "hurst": hurst_info, "gex": gex_info, "ofi": ofi_info,
                        "atr_14": atr_14, "gap_info": gap_info, "htf_regime": htf_regime, "order_flow": order_flow,
                        "skew": skew_info, "vcr": vcr_info, "hfi_score": hfi_score
                    }
                ))
                
                if _c is not None:
                
                    return _c

        # 10. SHORT Setup (3-Tier Asymmetric Target Calculation with HTF Gating)
        short_avwap_cond = close < (current_vwap + 0.35 * (current_vwap - lower_2sd) / 2.0)
        if close < ema200 and short_avwap_cond and swing_range >= 35.0 and ofi_info["seller_defense"] and is_trending_flow:
            fib = compute_fibonacci_levels(swing_high, swing_low, is_uptrend=False)
            in_pocket = (fib["fib_500"] - 10.0) <= min(close, bar_open) and max(close, bar_open) <= fib["fib_618"]
            bearish_trigger = (close < bar_open) or (close < prev_close_val)
            
            if in_pocket and bearish_trigger:
                t1 = round(close - 1.2 * atr_14, 2)
                t2 = round(close - 2.5 * atr_14, 2)
                t3_moonshot = round(min(lower_vakc_val, lower_2sd, close - 3.8 * atr_14), 2)
                pyramid_trigger_lvl = round(swing_low - 2.0, 2)
                
                _c = _check_and_return("SHORT", Signal(
                    signal_type=SignalType.SHORT,
                    entry_price=close,
                    sl_price=fib["sl_level"],
                    target_1=t1,
                    target_2=t2,
                    target_3_moonshot=t3_moonshot,
                    pyramid_trigger=pyramid_trigger_lvl,
                    reason=f"SHORT Setup Confirmed: Below 200 EMA + Below AVWAP ({gap_info['regime']}) + Golden Pocket + OFI Defense | HTF Aligned.",
                    htf_aligned=htf_aligned_short,
                    fib_retracement=0.55,
                    details={
                        "fib": fib, "ema200": ema200, "vwap": current_vwap, "ema21": ema21,
                        "swing_low": swing_low, "hurst": hurst_info, "gex": gex_info, "ofi": ofi_info,
                        "atr_14": atr_14, "gap_info": gap_info, "htf_regime": htf_regime, "order_flow": order_flow,
                        "skew": skew_info, "vcr": vcr_info, "hfi_score": hfi_score
                    }
                ))
                
                if _c is not None:
                
                    return _c

        # 11. Options Desk Range Fade Strategy (+Γ Wall Pinning / Mean-Reversion)
        if options_context and gex_info.get("is_positive_gamma", False):
            put_w = float(gex_info.get("put_wall_strike", 0.0))
            call_w = float(gex_info.get("call_wall_strike", 999999.0))
            max_p = float(options_context.get("pcr", {}).get("max_pain_strike", round(close / 50.0) * 50.0))
            d_v = float(options_context.get("dir_flow", {}).get("directional_vector", 0.0))
            pcr_z = float(options_context.get("pcr_zscore", 0.0))

            # Long fade only when defending above put wall (not broken below)
            if put_w - 5.0 <= close <= put_w + WALL_BUFFER_PTS and (d_v >= 0.2 or pcr_z >= PCR_Z_CONTRARIAN_THRESHOLD or ofi_info.get("buyer_defense")):
                _c = _check_and_return("LONG", Signal(
                    signal_type=SignalType.RANGE_FADE_LONG,
                    entry_price=close,
                    sl_price=round(put_w - WALL_BUFFER_PTS, 2),
                    target_1=max_p,
                    target_2=call_w,
                    target_3_moonshot=round(call_w + 1.0 * atr_14, 2),
                    reason=f"Options Desk Range Fade Long: Spot ({close:.1f}) defending Put Wall ({put_w:.0f}) in +Γ regime.",
                    htf_aligned=htf_aligned_long,
                    details={"put_wall": put_w, "call_wall": call_w, "max_pain": max_p, "gex": gex_info, "ofi": ofi_info}
                ))
                if _c is not None:
                    return _c
            # Short fade only when defending below call wall (not broken above)
            elif call_w - WALL_BUFFER_PTS <= close <= call_w + 5.0 and (d_v <= -0.2 or pcr_z <= -PCR_Z_CONTRARIAN_THRESHOLD or ofi_info.get("seller_defense")):
                _c = _check_and_return("SHORT", Signal(
                    signal_type=SignalType.RANGE_FADE_SHORT,
                    entry_price=close,
                    sl_price=round(call_w + WALL_BUFFER_PTS, 2),
                    target_1=max_p,
                    target_2=put_w,
                    target_3_moonshot=round(put_w - 1.0 * atr_14, 2),
                    reason=f"Options Desk Range Fade Short: Spot ({close:.1f}) testing Call Wall ({call_w:.0f}) in +Γ regime.",
                    htf_aligned=htf_aligned_short,
                    details={"put_wall": put_w, "call_wall": call_w, "max_pain": max_p, "gex": gex_info, "ofi": ofi_info}
                ))
                if _c is not None:
                    return _c

        # 12. Options Desk Gamma Breakout Strategy (-Γ Expansion / Wall Clearance)
        if options_context and not gex_info.get("is_positive_gamma", True):
            call_w = float(gex_info.get("call_wall_strike", 999999.0))
            put_w = float(gex_info.get("put_wall_strike", 0.0))
            d_v = float(options_context.get("dir_flow", {}).get("directional_vector", 0.0))

            if d_v >= 0.45 and close >= call_w and ofi_info.get("buyer_defense"):
                _c = _check_and_return("LONG", Signal(
                    signal_type=SignalType.GAMMA_BREAKOUT_LONG,
                    entry_price=close,
                    sl_price=round(close - 1.0 * atr_14, 2),
                    target_1=round(close + 1.5 * atr_14, 2),
                    target_2=round(close + 3.0 * atr_14, 2),
                    target_3_moonshot=round(close + 5.0 * atr_14, 2),
                    reason=f"Gamma Breakout Long: Price ({close:.1f}) cleared Call Wall ({call_w:.0f}) in -Γ expansion regime.",
                    htf_aligned=htf_aligned_long,
                    details={"call_wall": call_w, "gex": gex_info, "d_vector": d_v}
                ))
                if _c is not None:
                    return _c
            elif d_v <= -0.45 and close <= put_w and ofi_info.get("seller_defense"):
                _c = _check_and_return("SHORT", Signal(
                    signal_type=SignalType.GAMMA_BREAKOUT_SHORT,
                    entry_price=close,
                    sl_price=round(close + 1.0 * atr_14, 2),
                    target_1=round(close - 1.5 * atr_14, 2),
                    target_2=round(close - 3.0 * atr_14, 2),
                    target_3_moonshot=round(close - 5.0 * atr_14, 2),
                    reason=f"Gamma Breakdown Short: Price ({close:.1f}) breached Put Wall ({put_w:.0f}) in -Γ expansion regime.",
                    htf_aligned=htf_aligned_short,
                    details={"put_wall": put_w, "gex": gex_info, "d_vector": d_v}
                ))
                if _c is not None:
                    return _c

        # 13. Novel: Expiry Pin Trade (IMP-11) — 3-Vector Pin Trade
        if options_context:
            is_expiry_day = options_context.get("is_expiry_day", False)
            if is_expiry_day and bar_time >= "13:00" and gex_info.get("is_positive_gamma", False):
                put_w = float(gex_info.get("put_wall_strike", 0.0))
                call_w = float(gex_info.get("call_wall_strike", 999999.0))
                max_p = float(options_context.get("pcr", {}).get("max_pain_strike", round(close / 50.0) * 50.0))
                mp_distance = close - max_p
                
                if put_w < close < call_w and abs(mp_distance) > EXPIRY_PIN_MIN_DISTANCE_PTS:
                    pin_direction = "LONG" if mp_distance < 0 else "SHORT"
                    if pin_direction == "LONG":
                        _c = _check_and_return("LONG", Signal(
                            signal_type=SignalType.EXPIRY_PIN_LONG,
                            entry_price=close,
                            sl_price=round(put_w - 15.0, 2),
                            target_1=max_p,
                            target_2=round(max_p + 0.5 * atr_14, 2),
                            target_3_moonshot=call_w,
                            reason=f"Expiry Pin Long: GEX+ corridor, spot below Max Pain ({max_p:.0f}) by {abs(mp_distance):.0f} pts. Charm-driven drift expected.",
                            htf_aligned=htf_aligned_long,
                            details={"max_pain": max_p, "mp_distance": mp_distance, "gamma_regime": "POSITIVE"}
                        ))
                        if _c is not None:
                            return _c
                    else:
                        _c = _check_and_return("SHORT", Signal(
                            signal_type=SignalType.EXPIRY_PIN_SHORT,
                            entry_price=close,
                            sl_price=round(call_w + 15.0, 2),
                            target_1=max_p,
                            target_2=round(max_p - 0.5 * atr_14, 2),
                            target_3_moonshot=put_w,
                            reason=f"Expiry Pin Short: GEX+ corridor, spot above Max Pain ({max_p:.0f}) by {abs(mp_distance):.0f} pts. Charm-driven drift expected.",
                            htf_aligned=htf_aligned_short,
                            details={"max_pain": max_p, "mp_distance": mp_distance, "gamma_regime": "POSITIVE"}
                        ))
                        if _c is not None:
                            return _c

        # 14. Novel: Gamma Squeeze Breakout (IMP-12) — Dealer hedging cascade
        if options_context and not gex_info.get("is_positive_gamma", True):
            d_v = float(options_context.get("dir_flow", {}).get("directional_vector", 0.0))
            call_w = float(gex_info.get("call_wall_strike", 999999.0))
            put_w = float(gex_info.get("put_wall_strike", 0.0))
            zero_gex = float(gex_info.get("zero_gex_strike", close))
            avg_vol_gs = float(sub_df['volume'].tail(20).mean()) if 'volume' in sub_df.columns else 0.0
            curr_vol_gs = float(bar.get('volume', 0))
            volume_surge = curr_vol_gs > 1.5 * avg_vol_gs if avg_vol_gs > 0 else False
            
            # Bullish Gamma Squeeze: spot crossed above zero-GEX in -Gamma with volume
            if d_v >= 0.3 and close > zero_gex and volume_surge and close > ema21:
                _c = _check_and_return("LONG", Signal(
                    signal_type=SignalType.GAMMA_SQUEEZE_LONG,
                    entry_price=close,
                    sl_price=round(zero_gex - 0.5 * atr_14, 2),
                    target_1=round(close + GAMMA_SQUEEZE_TARGET_MULT * 1.5 * atr_14, 2),
                    target_2=round(close + GAMMA_SQUEEZE_TARGET_MULT * 3.0 * atr_14, 2),
                    target_3_moonshot=round(close + GAMMA_SQUEEZE_TARGET_MULT * 5.0 * atr_14, 2),
                    reason=f"Gamma Squeeze Long: -Γ regime, spot ({close:.1f}) above Zero-GEX ({zero_gex:.0f}). Dealer hedging cascade active. Volume surge confirmed.",
                    htf_aligned=htf_aligned_long,
                    details={"zero_gex": zero_gex, "d_vector": d_v, "volume_surge": volume_surge, "gamma_regime": "NEGATIVE"}
                ))
                if _c is not None:
                    return _c
            # Bearish Gamma Squeeze: spot crossed below zero-GEX in -Gamma with volume
            elif d_v <= -0.3 and close < zero_gex and volume_surge and close < ema21:
                _c = _check_and_return("SHORT", Signal(
                    signal_type=SignalType.GAMMA_SQUEEZE_SHORT,
                    entry_price=close,
                    sl_price=round(zero_gex + 0.5 * atr_14, 2),
                    target_1=round(close - GAMMA_SQUEEZE_TARGET_MULT * 1.5 * atr_14, 2),
                    target_2=round(close - GAMMA_SQUEEZE_TARGET_MULT * 3.0 * atr_14, 2),
                    target_3_moonshot=round(close - GAMMA_SQUEEZE_TARGET_MULT * 5.0 * atr_14, 2),
                    reason=f"Gamma Squeeze Short: -Γ regime, spot ({close:.1f}) below Zero-GEX ({zero_gex:.0f}). Dealer hedging cascade active. Volume surge confirmed.",
                    htf_aligned=htf_aligned_short,
                    details={"zero_gex": zero_gex, "d_vector": d_v, "volume_surge": volume_surge, "gamma_regime": "NEGATIVE"}
                ))
                if _c is not None:
                    return _c

        # Report WHY nothing fired. Previously the first rejected candidate ended the bar
        # and its message became the verdict; now every candidate is evaluated, so the
        # fall-through names the actual blocking reasons instead of a generic
        # "no confluence" when setups were in fact tested and rejected.
        if _absolute_veto:
            _wait_reason = _absolute_veto[0]
        elif _rejections:
            _wait_reason = f"{len(_rejections)} setup(s) evaluated and rejected: " + " | ".join(_rejections[:2])
        else:
            _wait_reason = "Market in consolidation / No confluence across core indicators."

        return Signal(
            signal_type=SignalType.WAIT,
            entry_price=close,
            sl_price=0.0,
            target_1=0.0,
            target_2=0.0,
            target_3_moonshot=0.0,
            pyramid_trigger=0.0,
            reason=_wait_reason,
            htf_aligned=True,
            fib_retracement=0.0,
            details={
                "ema200": ema200, "vwap": current_vwap, "ema21": ema21,
                "hurst": hurst_info, "gex": gex_info, "ofi": ofi_info,
                "gap_info": gap_info, "htf_regime": htf_regime, "order_flow": order_flow,
                "kalman_price": kalman_price, "kalman_velocity": kalman_vel, "kalman_vel_zscore": kalman_z,
                "markov_regime": markov_info,
                "rejected_candidates": sorted(_rejected, key=lambda c: c.get("confluence") or 0.0, reverse=True)
            }
        )



