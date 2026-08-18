"""OnlyNifty v5.2 Decision Engine.

Separates setup detection from trade gating, ranking, and risk decisions.
Enforces the universal rule: The default answer of this system is NO-TRADE (WAIT).
Only setups that pass all universal gates, edge table trust levels, and the confluence floor
are granted execution trade tickets.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from src.strategy_rules import Signal, SignalType
from src.risk_state import SessionRiskState
from src.config import (
    SIGNAL_MIN_CONFLUENCE, VPIN_TOXICITY_THRESHOLD, SKEW_ZSCORE_THRESHOLD,
    GEX_WALL_BUFFER_PTS, STOP_MIN_ATR_FRACTION, STOP_MAX_POINTS,
    LUNCH_LULL_SIZE_FACTOR, GATE_FAIL_TO_WAIT, GATE_MIN_MISSING_TO_BLOCK,
    POSITIONING_VETO_STRENGTH
)


@dataclass
class SetupCandidate:
    """Represents a candidate trading setup detected by pattern/microstructure rules."""
    setup_id: str                 # e.g. "IB_BREAKOUT_LONG", "CHOP_FADE_SHORT", "ABSORPTION_LONG"
    signal_type: SignalType
    direction: str                # "LONG" | "SHORT"
    entry_price: float
    sl_price: float
    target_1: float
    target_2: float
    target_3_moonshot: float
    pyramid_trigger: float
    reason: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionContext:
    """Comprehensive real-time market context required for institutional gating & decisioning."""
    markov_regime: str
    htf_aligned_long: bool
    htf_aligned_short: bool
    skew_z: float = 0.0
    is_crash_hedging: bool = False
    vpin: float = 0.0
    hfi_score: float = 0.0
    gex_walls: Dict[str, Any] = field(default_factory=dict)
    is_positive_gamma: bool = True
    vcr: float = 0.20
    lunch_lull: bool = False
    session_state: Optional[SessionRiskState] = None
    current_bar_idx: int = 0
    live_iv: float = 0.135
    htf_regime: Dict[str, Any] = field(default_factory=dict)
    options_context: Optional[Dict[str, Any]] = None
    vol_report: Optional[Dict[str, Any]] = None
    bar_timestamp: Optional[str] = None


class DecisionEngine:
    """
    Central Institutional Decision Engine.
    
    Evaluates candidate setups against hard market microstructure gates, session risk limits,
    statistical edge tables, and pre-decision confluence floors.
    """

    def __init__(self, edge_table: Optional[Any] = None):
        self.edge_table = edge_table

    def check_universal_gates(
        self,
        direction: str,
        close: float,
        ctx: DecisionContext,
        candidate_signal_type: Optional[str] = None
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Evaluates universal market-wide and microstructure gates."""
        audit = {
            "vpin": ctx.vpin,
            "skew_zscore": ctx.skew_z,
            "hfi_score": ctx.hfi_score,
            "is_positive_gamma": ctx.is_positive_gamma,
            "htf_aligned_long": ctx.htf_aligned_long,
            "htf_aligned_short": ctx.htf_aligned_short,
            "signal_type": candidate_signal_type or "",
            "passed": True,
            "veto_gate": None
        }

        # 0. EVENT & HOLIDAY BLACKOUT GATE
        if ctx.bar_timestamp:
            from src.event_calendar import check_event_risk_gate
            ev_passed, ev_reason, ev_audit = check_event_risk_gate(ctx.bar_timestamp)
            if not ev_passed:
                audit["passed"] = False
                audit["veto_gate"] = "EVENT_RISK_BLACKOUT"
                audit["event_info"] = ev_audit
                return False, ev_reason, audit

        # 0. DATA SUFFICIENCY (GATE_FAIL_TO_WAIT)
        missing_inputs: List[str] = []
        if ctx.skew_z == 0.0 and not ctx.is_crash_hedging:
            if not ctx.options_context or not ctx.options_context.get("chain_df") is not None:
                missing_inputs.append("25d_skew")
        if not ctx.gex_walls or not ctx.gex_walls.get("walls_verified", False):
            missing_inputs.append("dealer_walls")
        if not (ctx.options_context and ctx.options_context.get("dir_flow")):
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
        if ctx.vpin > VPIN_TOXICITY_THRESHOLD:
            audit["passed"] = False
            audit["veto_gate"] = "VPIN_TOXICITY"
            return False, f"VPIN Toxicity Veto: Toxic order flow detected (VPIN={ctx.vpin:.2f} > {VPIN_TOXICITY_THRESHOLD}).", audit

        # 2. 25-Delta Put Skew Crash Veto
        if direction == "LONG" and (ctx.is_crash_hedging or ctx.skew_z > SKEW_ZSCORE_THRESHOLD):
            audit["passed"] = False
            audit["veto_gate"] = "SKEW_CRASH_HEDGING"
            return False, f"25-Delta Put Skew Crash Gate: Long blocked. Put Skew spiked (Z={ctx.skew_z:+.2f}).", audit

        # 3. Heavyweight Flow Index (HFI) Alignment Veto
        if direction == "LONG" and ctx.hfi_score < -0.20:
            audit["passed"] = False
            audit["veto_gate"] = "HFI_BEARISH_DIVERGENCE"
            return False, f"Heavyweight Flow Veto: Long blocked. Top-5 Heavyweights are net negative (HFI={ctx.hfi_score:+.2f}).", audit
        elif direction == "SHORT" and ctx.hfi_score > 0.20:
            audit["passed"] = False
            audit["veto_gate"] = "HFI_BULLISH_DIVERGENCE"
            return False, f"Heavyweight Flow Veto: Short blocked. Top-5 Heavyweights are net positive (HFI={ctx.hfi_score:+.2f}).", audit

        # 4. Dealer GEX Wall Pinning Defense
        if ctx.is_positive_gamma:
            call_wall = float(ctx.gex_walls.get("call_wall_strike", 999999.0))
            put_wall = float(ctx.gex_walls.get("put_wall_strike", 0.0))
            if direction == "LONG" and close >= call_wall - GEX_WALL_BUFFER_PTS:
                audit["passed"] = False
                audit["veto_gate"] = "GEX_CALL_WALL_PIN"
                return False, f"Dealer GEX Pin Veto: Long blocked. Spot at Call Wall ({call_wall:.0f}) in +Γ regime.", audit
            elif direction == "SHORT" and close <= put_wall + GEX_WALL_BUFFER_PTS:
                audit["passed"] = False
                audit["veto_gate"] = "GEX_PUT_WALL_PIN"
                return False, f"Dealer GEX Pin Veto: Short blocked. Spot at Put Wall ({put_wall:.0f}) in +Γ regime.", audit

        # 5. Higher Timeframe Alignment Veto
        if direction == "LONG" and not ctx.htf_aligned_long:
            audit["passed"] = False
            audit["veto_gate"] = "HTF_NOT_ALIGNED_LONG"
            return False, "HTF Confluence Veto: Long not supported by 15m/1H timeframes.", audit
        elif direction == "SHORT" and not ctx.htf_aligned_short:
            audit["passed"] = False
            audit["veto_gate"] = "HTF_NOT_ALIGNED_SHORT"
            return False, "HTF Confluence Veto: Short not supported by 15m/1H timeframes.", audit

        # 6. Positioning Flow Direction Veto
        # Parity with StrategyEngine._apply_universal_gates / desk_verdict conflict detection:
        # options positioning diametrically opposing the chart setup is a hard veto.
        if ctx.options_context and ctx.options_context.get("dir_flow"):
            d_vec = float(ctx.options_context["dir_flow"].get("directional_vector", 0.0))
            audit["d_vector"] = d_vec
            if direction == "LONG" and d_vec <= -POSITIONING_VETO_STRENGTH:
                audit["passed"] = False
                audit["veto_gate"] = "POSITIONING_OPPOSES_CHART"
                return False, f"Positioning Veto: Long blocked. Bearish options flow (D={d_vec:+.2f}).", audit
            elif direction == "SHORT" and d_vec >= POSITIONING_VETO_STRENGTH:
                audit["passed"] = False
                audit["veto_gate"] = "POSITIONING_OPPOSES_CHART"
                return False, f"Positioning Veto: Short blocked. Bullish options flow (D={d_vec:+.2f}).", audit

        # 7. Session Risk Rails
        if ctx.session_state:
            can_trade, reason = ctx.session_state.can_take_new_trade(ctx.current_bar_idx)
            if not can_trade:
                audit["passed"] = False
                audit["veto_gate"] = "SESSION_RISK_LIMIT"
                return False, f"Session Risk Gate: {reason}", audit

        # 8. Term Structure Inversion Crisis Gate (IMP-4)
        vol_rep = ctx.vol_report or (ctx.options_context.get("vol_report") if ctx.options_context else None)
        if vol_rep:
            ts_regime = vol_rep.get("term_structure_regime", {})
            if isinstance(ts_regime, dict) and ts_regime.get("is_crisis", False):
                if direction == "LONG":
                    audit["passed"] = False
                    audit["veto_gate"] = "TERM_STRUCTURE_CRISIS"
                    return False, f"Term Structure Crisis Gate: Long blocked. IV backwardation detected. Risk-off mode.", audit

        # 9. Gamma Flip Level Regime Gate (IMP-3)
        if not ctx.is_positive_gamma:
            sig_name = str(candidate_signal_type or "")
            if "RANGE_FADE" in sig_name:
                audit["passed"] = False
                audit["veto_gate"] = "GAMMA_REGIME_BLOCKS_FADE"
                return False, "Gamma Regime Gate: Range Fade blocked in -Γ expansion regime.", audit

        # 10. Smart Money Institutional Flow Score Veto
        if ctx.options_context:
            inst_flow = ctx.options_context.get("inst_flow") or ctx.options_context.get("flow_score_data")
            flow_val = None
            if isinstance(inst_flow, dict):
                flow_val = inst_flow.get("flow_score")
            elif "flow_score" in ctx.options_context:
                flow_val = ctx.options_context["flow_score"]

            if flow_val is not None:
                audit["flow_score"] = float(flow_val)
                if direction == "LONG" and float(flow_val) < 30.0:
                    audit["passed"] = False
                    audit["veto_gate"] = "INSTITUTIONAL_FLOW_BEARISH_VETO"
                    return False, f"Institutional Flow Veto: Long blocked. Smart Money flow is net bearish (Flow Score={float(flow_val):.1f} < 30).", audit
                elif direction == "SHORT" and float(flow_val) > 70.0:
                    audit["passed"] = False
                    audit["veto_gate"] = "INSTITUTIONAL_FLOW_BULLISH_VETO"
                    return False, f"Institutional Flow Veto: Short blocked. Smart Money flow is net bullish (Flow Score={float(flow_val):.1f} > 70).", audit

        # 11. Variance Risk Premium (VRP) Backwardation Veto
        if ctx.options_context:
            vrp_data = ctx.options_context.get("vrp_data") or ctx.options_context.get("vrp_info")
            vrp_val = None
            if isinstance(vrp_data, dict):
                vrp_val = vrp_data.get("vrp")
            elif "vrp" in ctx.options_context:
                vrp_val = ctx.options_context["vrp"]

            if vrp_val is not None:
                audit["vrp"] = float(vrp_val)
                sig_name = str(candidate_signal_type or "")
                if ("RANGE_FADE" in sig_name or "STRADDLE" in sig_name) and float(vrp_val) < -0.02:
                    audit["passed"] = False
                    audit["veto_gate"] = "VRP_BACKWARDATION_VETO"
                    return False, f"Variance Risk Premium Gate: Short premium blocked. Negative VRP (VRP={float(vrp_val):+.3f}).", audit

        return True, "PASSED", audit

    def decide(
        self,
        candidates: List[SetupCandidate],
        ctx: DecisionContext,
        sub_df: pd.DataFrame,
        atr_14: float = 25.0,
        kalman_vel: float = 0.0,
        kalman_z: float = 0.0,
        ofi_info: Optional[Dict[str, Any]] = None,
        gex_info: Optional[Dict[str, Any]] = None,
        vp_info: Optional[Dict[str, Any]] = None,
        option_chain_df: Optional[pd.DataFrame] = None
    ) -> Signal:
        """
        Ranks candidates, applies universal gates, statistical edge tables,
        and confluence scoring, returning at most ONE trade Signal or WAIT.
        """
        if sub_df.empty:
            return Signal(SignalType.WAIT, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "No data available", False, 0.0, {})

        close = float(sub_df.iloc[-1]["close"])

        if not candidates:
            return Signal(
                signal_type=SignalType.WAIT,
                entry_price=close,
                sl_price=0.0,
                target_1=0.0,
                target_2=0.0,
                reason="Market in consolidation / No setup patterns active.",
                htf_aligned=False,
                details={"markov_regime": ctx.markov_regime}
            )

        from src.signal_journal import calculate_confluence_score

        scored_candidates = []

        for cand in candidates:
            # 1. Universal Gating
            passed, gate_reason, audit = self.check_universal_gates(
                cand.direction, close, ctx,
                candidate_signal_type=cand.signal_type.value if hasattr(cand.signal_type, "value") else str(cand.signal_type)
            )
            if not passed:
                continue

            # 2. Statistical Edge Table Check (Phase 3)
            if self.edge_table is not None:
                if not self.edge_table.is_tradeable(cand.setup_id, ctx.markov_regime):
                    audit["passed"] = False
                    audit["veto_gate"] = "EDGE_TABLE_QUARANTINED"
                    continue

            # 3. Stop Loss & Target Hygiene
            sl_dist = abs(cand.entry_price - cand.sl_price)
            min_sl = max(STOP_MIN_ATR_FRACTION * atr_14, 15.0)
            max_sl = STOP_MAX_POINTS
            
            clean_sl = cand.sl_price
            if sl_dist < min_sl:
                clean_sl = round(cand.entry_price - min_sl if cand.direction == "LONG" else cand.entry_price + min_sl, 2)
            elif sl_dist > max_sl:
                clean_sl = round(cand.entry_price - max_sl if cand.direction == "LONG" else cand.entry_price + max_sl, 2)

            clean_t1 = cand.target_1
            clean_t2 = cand.target_2
            clean_t3 = cand.target_3_moonshot

            min_spacing = max(0.8 * atr_14, 15.0)
            if cand.direction == "LONG":
                if clean_t1 <= cand.entry_price or (clean_t1 - cand.entry_price) < min_spacing:
                    clean_t1 = round(cand.entry_price + 1.2 * atr_14, 2)
                if clean_t2 <= clean_t1 or (clean_t2 - clean_t1) < min_spacing:
                    clean_t2 = round(clean_t1 + 1.3 * atr_14, 2)
                if clean_t3 <= clean_t2 or (clean_t3 - clean_t2) < min_spacing:
                    clean_t3 = round(clean_t2 + 1.5 * atr_14, 2)
            else:
                if clean_t1 >= cand.entry_price or (cand.entry_price - clean_t1) < min_spacing:
                    clean_t1 = round(cand.entry_price - 1.2 * atr_14, 2)
                if clean_t2 >= clean_t1 or (clean_t1 - clean_t2) < min_spacing:
                    clean_t2 = round(clean_t1 - 1.3 * atr_14, 2)
                if clean_t3 >= clean_t2 or (clean_t2 - clean_t3) < min_spacing:
                    clean_t3 = round(clean_t2 - 1.5 * atr_14, 2)

            cand_details = {**cand.details, "setup_id": cand.setup_id, "evidence": cand.evidence}
            if option_chain_df is not None and not option_chain_df.empty:
                cand_details["option_chain_available"] = True

            cand_signal = Signal(
                signal_type=cand.signal_type,
                entry_price=cand.entry_price,
                sl_price=clean_sl,
                target_1=clean_t1,
                target_2=clean_t2,
                target_3_moonshot=clean_t3,
                pyramid_trigger=cand.pyramid_trigger,
                reason=cand.reason,
                htf_aligned=ctx.htf_aligned_long if cand.direction == "LONG" else ctx.htf_aligned_short,
                details=cand_details
            )

            # 4. Pre-Decision Confluence Score with full HTF context
            htf_dict = ctx.htf_regime if ctx.htf_regime else {
                "tf_15m": {"bias": "Bullish" if ctx.htf_aligned_long else "Bearish"},
                "tf_1h": {"bias": "Bullish" if ctx.htf_aligned_long else "Bearish"},
                "confluence_regime": ctx.markov_regime
            }
            score, grade = calculate_confluence_score(
                cand_signal,
                sub_df,
                htf_data=htf_dict,
                kalman_vel=kalman_vel,
                kalman_z=kalman_z,
                regime_state={"active_regime": ctx.markov_regime},
                ofi_data=ofi_info,
                gex_data=gex_info,
                vol_profile=vp_info
            )

            cand_signal.details["confluence_score"] = score
            cand_signal.details["confluence_grade"] = grade
            cand_signal.details["gate_audit"] = audit

            # 5. Confluence Floor Check
            if score >= SIGNAL_MIN_CONFLUENCE:
                scored_candidates.append((score, cand_signal))

        if not scored_candidates:
            return Signal(
                signal_type=SignalType.WAIT,
                entry_price=close,
                sl_price=0.0,
                target_1=0.0,
                target_2=0.0,
                reason=f"No candidate passed universal gates and confluence floor (≥ {SIGNAL_MIN_CONFLUENCE:.1f}).",
                htf_aligned=False,
                details={"markov_regime": ctx.markov_regime, "evaluated_candidates_count": len(candidates)}
            )

        # Rank candidates by confluence score descending and pick the highest-conviction setup
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        best_signal = scored_candidates[0][1]

        if ctx.lunch_lull:
            best_signal.reason += " [LUNCH LULL: Halved sizing recommended]"

        return best_signal
