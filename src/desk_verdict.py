"""OnlyNifty v5.2 Main-Page Desk Verdict.

Pure functional decision engine that synthesizes chart setups, order flow,
options desk positioning, and session risk rails into ONE authoritative decision:
WHAT TO DO (BUY_CE / BUY_PE / WAIT), TREND, RANGE, and CONCRETE OPTION TICKET.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from src.strategy_rules import Signal, SignalType
from src.options_positioning import OptionsDeskState, clamp_targets_to_corridor
from src.risk_state import SessionRiskState
from src.config import (
    POSITIONING_VETO_STRENGTH,
    WALL_BUFFER_PTS,
    PCR_Z_CONTRARIAN_THRESHOLD,
    POSITIONING_UNVERIFIED_SIZE_CAP,
    MIN_CONVICTION_TO_TRADE,
    EVIDENCE_OPPOSITION_THRESHOLD,
    LOT_SIZE
)


@dataclass
class DeskVerdict:
    """The unified, single source of truth verdict rendered on the main cockpit."""
    action: str                          # "BUY_CE" | "BUY_PE" | "WAIT"
    action_label: str                    # Human headline
    reason: str                          # Reason including any named conflicts/vetoes
    trend_bias: str                      # "BULLISH" | "BEARISH" | "NEUTRAL"
    trend_conviction_pct: float          # 0-100
    range_corridor: Tuple[float, float]  # (support_put_wall, resistance_call_wall)
    max_pain: float
    expected_move_pts: float
    spot_position_pct: float             # 0-100% within corridor
    option_pick: Optional[Dict[str, Any]] # concrete ticket details or None
    evidence: Dict[str, str]             # family -> one-line verdict
    gate_audit: List[Dict[str, Any]]     # [{gate, value, passed, note}]
    conflicts: List[str]                 # named disagreements
    confluence_score: float
    confluence_grade: str
    data_quality: str                    # "VERIFIED" | "POSITIONING_UNVERIFIED"

    # --- Conviction synthesis (v5.3) ---
    conviction_score: float = 0.0        # 0-100 composite across independent evidence families
    conviction_tier: str = "LOW"         # "EXTREME" | "HIGH" | "MODERATE" | "LOW"
    family_votes: Dict[str, int] = field(default_factory=dict)   # family -> -1 | 0 | +1
    family_agreement: int = 0            # families agreeing with the action (0..4)
    directional_score: float = 0.0       # net composite direction [-1.0, +1.0]
    conviction_notes: List[str] = field(default_factory=list)    # what raised/lowered conviction
    edge_status: str = "UNMEASURED"      # walk-forward OOS status of the firing setup

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _desk_only_candidate_direction(
    spot: float,
    desk_state: Optional[OptionsDeskState],
    vol_report: Optional[Dict[str, Any]],
    put_wall: float,
    call_wall: float,
    d_vector: float,
    data_quality: str
) -> Optional[str]:
    """
    What the desk WOULD trade on its own when the chart says WAIT.

    Split out as a pure pre-check so conflict detection can run against the intended
    direction. Previously the desk-only branches lived after conflict detection and
    keyed it off the chart signal, so on a WAIT every conflict rule was unreachable
    and the desk could invent a trade nothing had vetted. Mirrors the branch
    conditions exactly; returns None when the desk would stand aside.
    """
    if desk_state is None or data_quality != "VERIFIED":
        return None

    is_crisis = False
    if vol_report and isinstance(vol_report, dict):
        ts = vol_report.get("term_structure_regime")
        if isinstance(ts, dict):
            is_crisis = bool(ts.get("is_crisis", False))

    if desk_state.is_positive_gamma:
        if (not is_crisis and spot <= put_wall + WALL_BUFFER_PTS
                and (d_vector >= 0.2 or desk_state.pcr_zscore >= PCR_Z_CONTRARIAN_THRESHOLD)):
            return "LONG"
        if (spot >= call_wall - WALL_BUFFER_PTS
                and (d_vector <= -0.2 or desk_state.pcr_zscore <= -PCR_Z_CONTRARIAN_THRESHOLD)):
            return "SHORT"
    else:
        if abs(d_vector) >= 0.5:
            if not is_crisis and d_vector >= 0.5 and spot >= call_wall:
                return "LONG"
            if d_vector <= -0.5 and spot <= put_wall:
                return "SHORT"
    return None


def compute_evidence_families(
    desk_state: Optional[OptionsDeskState] = None,
    htf_data: Optional[Dict[str, Any]] = None,
    regime_state: Optional[Dict[str, Any]] = None,
    vol_report: Optional[Dict[str, Any]] = None,
    options_context: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, int], Dict[str, str], float]:
    """
    Scores four INDEPENDENT evidence families, each voting -1 (bearish) / 0 / +1 (bullish).

    Independence is the point: EMA/AVWAP/VAKC/fib are all functions of the same price
    series and count as ONE family (structure), not four confirmations. Real conviction
    requires families that can genuinely disagree — structure, order flow, options
    positioning, and volatility/macro.

    Returns (votes, rationale_by_family, net_directional_score).
    """
    votes: Dict[str, int] = {"structure": 0, "flow": 0, "positioning": 0, "macro": 0}
    why: Dict[str, str] = {}

    # --- 1. STRUCTURE: higher-timeframe alignment + active regime -------------
    struct_bits: List[str] = []
    if htf_data:
        if htf_data.get("htf_aligned_long"):
            votes["structure"] = 1
            struct_bits.append("15m+1H aligned bullish")
        elif htf_data.get("htf_aligned_short"):
            votes["structure"] = -1
            struct_bits.append("15m+1H aligned bearish")
        else:
            tf15 = (htf_data.get("tf_15m") or {}).get("bias", "N/A")
            tf1h = (htf_data.get("tf_1h") or {}).get("bias", "N/A")
            struct_bits.append(f"HTF split (15m {tf15} / 1H {tf1h})")
    else:
        struct_bits.append("HTF unavailable")

    if regime_state and regime_state.get("active_regime"):
        struct_bits.append(str(regime_state["active_regime"]))
    why["structure"] = " | ".join(struct_bits)

    # --- 2. FLOW: delta-weighted volume + smart-money institutional score -----
    flow_bits: List[str] = []
    flow_sum = 0.0
    if desk_state is not None and desk_state.dwv_momentum_score != 0.0:
        flow_sum += float(desk_state.dwv_momentum_score)
        flow_bits.append(f"DWV {desk_state.dwv_momentum_score:+.2f}")

    flow_score_val = None
    if options_context:
        if "flow_score" in options_context:
            flow_score_val = options_context["flow_score"]
        elif isinstance(options_context.get("inst_flow"), dict):
            flow_score_val = options_context["inst_flow"].get("flow_score")
    if flow_score_val is not None:
        # Smart-money score is 0-100 centred at 50 -> map to [-1, +1]
        flow_sum += (float(flow_score_val) - 50.0) / 50.0
        flow_bits.append(f"SmartFlow {float(flow_score_val):.0f}")

    if flow_sum > 0.15:
        votes["flow"] = 1
    elif flow_sum < -0.15:
        votes["flow"] = -1
    why["flow"] = " | ".join(flow_bits) if flow_bits else "no flow data"

    # --- 3. POSITIONING: Orthogonalised into PC1 (Directional Flow) + Gamma Regime ---
    # Collinearity fix (v5.3): d_vector, itm_otm_shift, writing_bias, and pcr_mom
    # previously shared identical OI numerators (r=0.88-0.96) and were multi-counted.
    # We collapse them into:
    # 1. Primary Directional Flow (PC1) via D_intraday (or fallback to OI shift/writing if D=0)
    # 2. Orthogonal Gamma Regime & Wall Boundary Convexity (Dealer Gamma pin vs breakout)
    pos_bits: List[str] = []
    pos_sum = 0.0
    if desk_state is not None:
        # 1. Directional PC1 Flow
        d_val = float(desk_state.d_vector)
        if abs(d_val) > 0.05:
            pos_sum += d_val
            pos_bits.append(f"D {d_val:+.2f}")
        else:
            # Fallback if D-vector is unpopulated/neutral: use primary OI delta proxy
            if desk_state.itm_otm_shift != 0.0:
                pos_sum += float(desk_state.itm_otm_shift)
                pos_bits.append(f"OI shift {desk_state.itm_otm_shift:+.2f}")
            elif desk_state.writing_bias and desk_state.writing_bias != "BALANCED_RANGE":
                wb = str(desk_state.writing_bias)
                if "PUT_WRITING" in wb:
                    pos_sum += 0.4
                    pos_bits.append("put-writing support")
                elif "CALL_WRITING" in wb:
                    pos_sum -= 0.4
                    pos_bits.append("call-writing resistance")

        # 2. Orthogonal Gamma Regime & Wall Convexity
        if desk_state.is_positive_gamma:
            pos_bits.append("Dealer +Γ (Mean Reversion / Pin)")
        else:
            pos_bits.append("Dealer -Γ (Breakout Expansion)")

    if pos_sum > 0.15:
        votes["positioning"] = 1
    elif pos_sum < -0.15:
        votes["positioning"] = -1
    why["positioning"] = " | ".join(pos_bits) if pos_bits else "no positioning data"

    # --- 4. MACRO: cross-market reads that are NOT derived from the option chain ---
    # This family previously voted only on VRP and term structure — i.e. it was a second
    # volatility family wearing a macro label, which is part of why the four families
    # measured as only ~1.8 independent signals. Futures basis, global cues and the VIX
    # regime come from different instruments and different participants, so they carry
    # information the price/option complex does not.
    macro_bits: List[str] = []
    macro_sum = 0.0

    basis = (options_context or {}).get("futures_basis")
    if isinstance(basis, dict) and basis.get("data_quality") == "VERIFIED":
        # Deviation from fair-value carry: premium above carry = long demand.
        macro_sum += float(basis.get("bias_score", 0.0))
        macro_bits.append(
            f"Basis {basis.get('basis_pts', 0.0):+.0f}pts ({basis.get('annualised_basis_pct', 0.0):+.1f}% ann)"
        )

    macro_ctx = (options_context or {}).get("macro_report")
    if isinstance(macro_ctx, dict):
        ms = macro_ctx.get("macro_sentiment_score", macro_ctx.get("sentiment_score"))
        if ms is not None:
            # Global cues (USDINR, Brent, US10Y) — normalised to [-1, +1] by the engine.
            macro_sum += float(np.clip(float(ms), -1.0, 1.0)) * 0.7
            macro_bits.append(f"Global {float(ms):+.2f}")

    if vol_report:
        ts = vol_report.get("term_structure_regime")
        if isinstance(ts, dict) and ts.get("is_crisis"):
            macro_sum -= 1.0
            macro_bits.append("IV backwardation (crisis)")
        composite = vol_report.get("composite_vol_regime")
        if composite:
            macro_bits.append(str(composite))

    vrp_val = None
    if options_context and "vrp" in options_context:
        vrp_val = options_context["vrp"]
    elif vol_report and isinstance(vol_report.get("vrp_data"), dict):
        vrp_val = vol_report["vrp_data"].get("vrp")
    if vrp_val is not None:
        # Positive VRP (IV richer than realized) is the normal, risk-on state — but a
        # fixed +/-0.5 on the SIGN alone always cleared the +/-0.15 deadband, so MACRO
        # could never abstain: the whole family collapsed to sign(VRP). Scale by
        # magnitude so an ordinary VRP reads as neutral and only a pronounced one votes.
        macro_sum += float(np.clip(float(vrp_val) / 0.04, -1.0, 1.0)) * 0.5
        macro_bits.append(f"VRP {float(vrp_val):+.1%}")

    if macro_sum > 0.15:
        votes["macro"] = 1
    elif macro_sum < -0.15:
        votes["macro"] = -1
    why["macro"] = " | ".join(macro_bits) if macro_bits else "no macro data"

    weights = {"structure": 0.30, "flow": 0.25, "positioning": 0.30, "macro": 0.15}
    directional = sum(votes[k] * weights[k] for k in votes)
    return votes, why, round(float(np.clip(directional, -1.0, 1.0)), 3)


def compute_conviction(
    action: str,
    votes: Dict[str, int],
    confluence_score: float,
    desk_state: Optional[OptionsDeskState] = None,
    data_quality: str = "VERIFIED",
    edge_status: str = "UNMEASURED",
    is_breakout: bool = False
) -> Tuple[float, str, int, List[str]]:
    """
    Blends independent-family agreement with the confluence score, then applies
    real-world dampeners. Returns (score_0_100, tier, agreement_count, notes).
    """
    notes: List[str] = []

    if action == "BUY_CE":
        wanted = 1
    elif action == "BUY_PE":
        wanted = -1
    else:
        return 0.0, "LOW", 0, ["No directional action"]

    agreement = sum(1 for v in votes.values() if v == wanted)
    opposed = sum(1 for v in votes.values() if v == -wanted)

    # Agreement across independent families is the backbone of conviction.
    agreement_pts = (agreement / 4.0) * 100.0
    score = 0.6 * agreement_pts + 0.4 * float(confluence_score)

    if agreement >= 3:
        notes.append(f"{agreement}/4 evidence families agree")
    if opposed >= 2:
        score -= 15.0
        notes.append(f"{opposed} families actively oppose")
    elif opposed == 1:
        score -= 7.0
        notes.append("1 family opposes")

    if desk_state is not None:
        # Range exhaustion: if the day has already travelled well past its implied
        # move, a fresh breakout is late and mean-reversion is the better prior.
        if is_breakout and desk_state.move_ratio > 1.3:
            score -= 12.0
            notes.append(f"range exhausted ({desk_state.move_ratio:.2f}x expected move)")
        elif is_breakout and desk_state.move_ratio < 0.6:
            score += 5.0
            notes.append(f"expansion room ({desk_state.move_ratio:.2f}x expected move)")

        # Sitting on the gamma flip means the dealer regime itself is unstable.
        if abs(desk_state.gamma_flip_distance_pts) < 25.0:
            score -= 10.0
            notes.append(f"at gamma flip ({desk_state.gamma_flip_distance_pts:+.0f} pts) — regime unstable")

        # The desk's own 4-pillar positioning vote.
        if desk_state.agreement_count >= 3:
            score += 8.0
            notes.append(f"positioning pillars {desk_state.agreement_count}/4")
        elif desk_state.agreement_count <= 1:
            score -= 5.0
            notes.append(f"positioning pillars only {desk_state.agreement_count}/4")

    if data_quality != "VERIFIED":
        score = min(score, 55.0)
        notes.append("positioning UNVERIFIED — conviction capped")

    if edge_status == "QUARANTINED":
        score = min(score, 20.0)
        notes.append("setup QUARANTINED by walk-forward edge table")
    elif edge_status in ("PAPER", "UNMEASURED"):
        # UNMEASURED must not outrank PAPER. Leaving it uncapped meant a setup with NO
        # out-of-sample evidence scored HIGHER than one with some — and since
        # data/edge_table.json is absent in practice, the uncapped branch was the one
        # always taken. Unmeasured is the weaker claim and is capped accordingly.
        score = min(score, 65.0)
        notes.append(
            "setup still PAPER (insufficient OOS samples)" if edge_status == "PAPER"
            else "setup UNMEASURED (no walk-forward edge data)"
        )
    elif edge_status == "TRUSTED":
        notes.append("setup TRUSTED by walk-forward edge table")

    score = float(np.clip(score, 0.0, 100.0))

    if score >= 80.0:
        tier = "EXTREME"
    elif score >= 65.0:
        tier = "HIGH"
    elif score >= 45.0:
        tier = "MODERATE"
    else:
        tier = "LOW"

    return round(score, 1), tier, agreement, notes


def build_desk_verdict(
    signal: Signal,
    ticket: Optional[Dict[str, Any]] = None,
    desk_state: Optional[OptionsDeskState] = None,
    vol_report: Optional[Dict[str, Any]] = None,
    regime_state: Optional[Dict[str, Any]] = None,
    htf_data: Optional[Dict[str, Any]] = None,
    edge_stats: Optional[Any] = None,
    session_state: Optional[SessionRiskState] = None,
    current_spot: float = 0.0,
    options_context: Optional[Dict[str, Any]] = None
) -> DeskVerdict:
    """
    Synthesizes all quantitative components into ONE cohesive desk verdict.
    """
    conflicts: List[str] = []
    gate_audit: List[Dict[str, Any]] = []

    spot = current_spot if current_spot > 0 else (signal.entry_price if signal and signal.entry_price > 0 else 24500.0)

    # 1. Baseline Trend, Range & Corridor
    if desk_state is not None:
        put_wall = desk_state.put_wall
        call_wall = desk_state.call_wall
        max_pain = desk_state.max_pain
        expected_move = desk_state.expected_move_pts
        d_vector = desk_state.d_vector
        conviction = desk_state.trend_conviction_pct
        trend_bias = desk_state.trend_bias
        data_quality = desk_state.data_quality
    else:
        put_wall = round(spot - 150.0, -2)
        call_wall = round(spot + 150.0, -2)
        max_pain = round(spot / 50.0) * 50.0
        expected_move = 85.0
        d_vector = 0.0
        conviction = 50.0
        trend_bias = "NEUTRAL"
        data_quality = "POSITIONING_UNVERIFIED"

    # Spot position % inside corridor
    corridor_width = max(call_wall - put_wall, 10.0)
    spot_pos_pct = round(max(0.0, min(100.0, ((spot - put_wall) / corridor_width) * 100.0)), 1)
    range_corridor = (put_wall, call_wall)

    # 2. Gate Audit Extraction from Signal
    if signal and signal.details and "gate_audit" in signal.details:
        sig_audit = signal.details["gate_audit"]
        if isinstance(sig_audit, dict):
            for k, v in sig_audit.items():
                if k != "passed" and k != "veto_gate":
                    gate_audit.append({"gate": k, "value": v, "passed": sig_audit.get("passed", True)})
        elif isinstance(sig_audit, list):
            gate_audit.extend(sig_audit)

    # 3. Conflict Detection
    sig_type_str = signal.signal_type.value if signal and hasattr(signal.signal_type, "value") else str(signal.signal_type if signal else "WAIT")
    is_chart_long = "LONG" in sig_type_str
    is_chart_short = "SHORT" in sig_type_str

    # Resolve the INTENDED direction before testing anything against it. These flags
    # previously came only from the chart signal, so on a WAIT they were both False —
    # every conflict rule below became structurally unreachable, and the desk-only
    # branches further down could invent a trade that no conflict had ever been
    # evaluated against (including a direction the ladder had just vetoed).
    desk_only_dir = None
    if not (is_chart_long or is_chart_short):
        desk_only_dir = _desk_only_candidate_direction(
            spot=spot, desk_state=desk_state, vol_report=vol_report,
            put_wall=put_wall, call_wall=call_wall, d_vector=d_vector,
            data_quality=data_quality
        )
    intended_long = is_chart_long or desk_only_dir == "LONG"
    intended_short = is_chart_short or desk_only_dir == "SHORT"

    if desk_state is not None:
        if intended_long and d_vector <= -POSITIONING_VETO_STRENGTH:
            conflicts.append(f"POSITIONING_OPPOSES_CHART: Long vs Bearish Options Flow (D={d_vector:+.2f})")
        elif intended_short and d_vector >= POSITIONING_VETO_STRENGTH:
            conflicts.append(f"POSITIONING_OPPOSES_CHART: Short vs Bullish Options Flow (D={d_vector:+.2f})")

        # Wall fading into positive gamma
        if intended_long and desk_state.is_positive_gamma and spot >= call_wall - WALL_BUFFER_PTS:
            conflicts.append(f"GEX_CALL_WALL_BLOCK: Long near Call Wall ({call_wall:.0f}) in positive gamma (+Γ).")
        elif intended_short and desk_state.is_positive_gamma and spot <= put_wall + WALL_BUFFER_PTS:
            conflicts.append(f"GEX_PUT_WALL_BLOCK: Short near Put Wall ({put_wall:.0f}) in positive gamma (+Γ).")

    # Smart Money Institutional Flow Conflicts
    flow_score_val = None
    if options_context and "flow_score" in options_context:
        flow_score_val = options_context["flow_score"]
    elif options_context and "inst_flow" in options_context and isinstance(options_context["inst_flow"], dict):
        flow_score_val = options_context["inst_flow"].get("flow_score")

    if flow_score_val is not None:
        if intended_long and flow_score_val < 30.0:
            conflicts.append(f"INSTITUTIONAL_FLOW_VETO: Smart Money Flow Score ({flow_score_val:.1f}/100) indicates institutional selling.")
        elif intended_short and flow_score_val > 70.0:
            conflicts.append(f"INSTITUTIONAL_FLOW_VETO: Smart Money Flow Score ({flow_score_val:.1f}/100) indicates institutional accumulation.")

    # Term Structure & VRP Crisis Conflict
    if vol_report and vol_report.get("term_structure_regime"):
        ts_regime = vol_report["term_structure_regime"]
        if isinstance(ts_regime, dict) and ts_regime.get("is_crisis", False):
            if intended_long:
                conflicts.append(f"TERM_STRUCTURE_CRISIS: IV backwardation detected. Long trades carry extreme risk.")

    vrp_val = None
    if options_context and "vrp" in options_context:
        vrp_val = options_context["vrp"]
    elif vol_report and "vrp_data" in vol_report:
        vrp_val = vol_report["vrp_data"].get("vrp", 0.0)

    # 4. Session Risk Limits
    if session_state is not None:
        can_trade, risk_reason = session_state.can_take_new_trade()
        if not can_trade:
            conflicts.append(f"SESSION_RISK_LOCKED: {risk_reason}")

    # 4b. Net evidence opposes the intended trade.
    # The four families are synthesised into a net directional score and, until now,
    # that score was reported on the verdict and never consulted when picking a side.
    # If the independent evidence nets meaningfully AGAINST the direction, that is a
    # disagreement worth naming — the same standing as a positioning or flow conflict.
    _pre_votes, _, _pre_direction = compute_evidence_families(
        desk_state=desk_state, htf_data=htf_data, regime_state=regime_state,
        vol_report=vol_report, options_context=options_context
    )
    if intended_long and _pre_direction <= -EVIDENCE_OPPOSITION_THRESHOLD:
        conflicts.append(
            f"EVIDENCE_OPPOSES_TRADE: Long vs net bearish evidence "
            f"({_pre_direction:+.2f} across {sum(1 for v in _pre_votes.values() if v < 0)} families)."
        )
    elif intended_short and _pre_direction >= EVIDENCE_OPPOSITION_THRESHOLD:
        conflicts.append(
            f"EVIDENCE_OPPOSES_TRADE: Short vs net bullish evidence "
            f"({_pre_direction:+.2f} across {sum(1 for v in _pre_votes.values() if v > 0)} families)."
        )

    # 5. Build Evidence Map
    zg_str = f" | Flip: {desk_state.zero_gex_strike:.0f}" if desk_state and desk_state.zero_gex_strike > 0 else ""
    dwv_str = f" | DWV: {desk_state.dwv_momentum_score:+.2f}" if desk_state and desk_state.dwv_momentum_score != 0.0 else ""
    flow_str = f" | SmartFlow: {flow_score_val:.0f}" if flow_score_val is not None else ""
    rough_str = " | ⚡ Rough-Vol" if regime_state and regime_state.get("is_rough_volatility", False) else ""
    
    vrp_str = ""
    if vrp_val is not None:
        vrp_str = f" | VRP: {vrp_val:+.1%}"
    elif vol_report and "iv_rv_spread" in vol_report:
        v_spread = vol_report["iv_rv_spread"].get("spread", 0.0)
        vrp_str = f" | VRP: {v_spread:+.1%}"

    disp_str = ""
    if options_context and "disp_data" in options_context and isinstance(options_context["disp_data"], dict):
        disp_z = options_context["disp_data"].get("spread_zscore", 0.0)
        disp_str = f" | Disp: {disp_z:+.1f}σ"

    # Independent evidence families (structure / flow / positioning / macro).
    # This is what turns "the options model likes it" into a real conviction measure.
    family_votes, family_why, directional_score = compute_evidence_families(
        desk_state=desk_state,
        htf_data=htf_data,
        regime_state=regime_state,
        vol_report=vol_report,
        options_context=options_context
    )

    arrow = {1: "↑", -1: "↓", 0: "→"}
    oi_str = f" | OI shift: {desk_state.itm_otm_shift:+.2f}" if desk_state and desk_state.itm_otm_shift != 0.0 else ""
    wb_str = f" | {desk_state.writing_bias.replace('_', ' ').title()}" if desk_state and desk_state.writing_bias else ""
    drift_str = f" | Vanna/Charm: {desk_state.dealer_drift_score:+.2f}" if desk_state and desk_state.dealer_drift_score != 0.0 else ""
    range_str = f" | Range: {desk_state.move_ratio:.2f}x exp" if desk_state and desk_state.move_ratio > 0 else ""
    flip_str = f" | Flip: {desk_state.zero_gex_strike:.0f} ({desk_state.gamma_flip_distance_pts:+.0f}pts)" if desk_state and desk_state.zero_gex_strike > 0 else zg_str

    evidence = {
        "structure": f"{arrow[family_votes['structure']]} {family_why['structure']}{rough_str}",
        "flow": f"{arrow[family_votes['flow']]} D-Vector: {d_vector:+.2f} ({conviction:.0f}%){dwv_str}{flow_str}",
        "positioning": f"{arrow[family_votes['positioning']]} Walls: [{put_wall:.0f} - {call_wall:.0f}] | Pain: {max_pain:.0f}{flip_str}{oi_str}{wb_str}{drift_str}",
        "macro": f"{arrow[family_votes['macro']]} Vol: {vol_report.get('composite_vol_regime', 'BALANCED') if vol_report else 'BALANCED'}{vrp_str}{disp_str}{range_str}" +
                 (f" | TS: {vol_report.get('term_structure_regime', {}).get('regime', 'N/A')}" if vol_report and vol_report.get('term_structure_regime') else "")
    }

    # 6. Confluence Score & Grade
    confluence_score = signal.details.get("confluence_score", 50.0) if (signal and signal.details) else 50.0
    confluence_grade = signal.details.get("confluence_grade", "Standard") if (signal and signal.details) else "Standard"

    # Walk-forward OOS status of the firing setup (stamped by the live edge gate).
    edge_status = "UNMEASURED"
    if edge_stats is not None:
        edge_status = getattr(edge_stats, "status", str(edge_stats))
    elif signal and signal.details:
        edge_status = signal.details.get("edge_status", "UNMEASURED")

    # 7. Action Decision Table
    action = "WAIT"
    action_label = "AWAITING CONFLUENCE"
    reason = signal.reason if signal else "Market in consolidation."
    option_pick: Optional[Dict[str, Any]] = None
    is_breakout_action = "BREAKOUT" in sig_type_str or "GAMMA_SQUEEZE" in sig_type_str

    if conflicts:
        action = "WAIT"
        action_label = "NO-TRADE (WAIT) — CONFLICT DETECTED"
        reason = " | ".join(conflicts)
    elif signal and signal.signal_type != SignalType.WAIT:
        if is_chart_long:
            action = "BUY_CE"
            t1, t2, _ = clamp_targets_to_corridor(spot, signal.target_1, signal.target_2, "LONG", put_wall, call_wall)
            action_label = f"BUY CE @ ₹{ticket.get('entry_premium', 0):.1f} | Target {t1:.0f} (T1) / {call_wall:.0f} (Wall)" if ticket else "BUY CE (CONFIRMED)"
        elif is_chart_short:
            action = "BUY_PE"
            t1, t2, _ = clamp_targets_to_corridor(spot, signal.target_1, signal.target_2, "SHORT", put_wall, call_wall)
            action_label = f"BUY PE @ ₹{ticket.get('entry_premium', 0):.1f} | Target {t1:.0f} (T1) / {put_wall:.0f} (Wall)" if ticket else "BUY PE (CONFIRMED)"

        if ticket and ticket.get("status") == "READY":
            option_pick = {
                "symbol": ticket.get("symbol", f"NIFTY {ticket.get('strike', 24500)} {ticket.get('option_type', 'CE')}"),
                "strike": ticket.get("strike", 24500),
                "option_type": ticket.get("option_type", "CE"),
                "entry_premium": ticket.get("entry_premium", 0.0),
                "sl_premium": ticket.get("sl_premium", 0.0),
                "target1_premium": ticket.get("target1_premium", 0.0),
                "target2_premium": ticket.get("target2_premium", 0.0),
                "target3_premium": ticket.get("target3_premium", 0.0),
                "lots": ticket.get("lots", 1),
                "total_qty": ticket.get("total_qty", LOT_SIZE),
                "tca_friction": ticket.get("tca_friction", 0.0),
                "r_t1": ticket.get("r_multiple_t1", 1.5),
                "r_t2": ticket.get("r_multiple_t2", 3.0)
            }
            option_pick["structure_recommendation"] = ticket.get("structure_recommendation", "NAKED_LONG")
            option_pick["iv_rank_label"] = ticket.get("iv_rank_label", "AVERAGE_IV")
    else:
        # Desk-Only Fades / Breakouts when chart is WAIT
        if desk_state is not None and data_quality == "VERIFIED" and not conflicts:
            is_crisis = vol_report.get("term_structure_regime", {}).get("is_crisis", False) if (vol_report and isinstance(vol_report, dict)) else False
            # Wall Range Fade Long
            if not is_crisis and spot <= put_wall + WALL_BUFFER_PTS and desk_state.is_positive_gamma and (d_vector >= 0.2 or desk_state.pcr_zscore >= PCR_Z_CONTRARIAN_THRESHOLD):
                action = "BUY_CE"
                action_label = f"BUY {round(spot/50)*50} CE | Put Wall Support Fade"
                reason = f"Desk Wall Fade: Spot ({spot:.1f}) defending Put Wall ({put_wall:.0f}) in +Γ regime."
            # Wall Range Fade Short
            elif spot >= call_wall - WALL_BUFFER_PTS and desk_state.is_positive_gamma and (d_vector <= -0.2 or desk_state.pcr_zscore <= -PCR_Z_CONTRARIAN_THRESHOLD):
                action = "BUY_PE"
                action_label = f"BUY {round(spot/50)*50} PE | Call Wall Resistance Fade"
                reason = f"Desk Wall Fade: Spot ({spot:.1f}) testing Call Wall ({call_wall:.0f}) in +Γ regime."
            # Negative Gamma Breakout
            elif not desk_state.is_positive_gamma and abs(d_vector) >= 0.5:
                if not is_crisis and d_vector >= 0.5 and spot >= call_wall:
                    action = "BUY_CE"
                    action_label = f"BUY {round(spot/50)*50} CE | -Γ Explosive Breakout"
                    reason = f"Gamma Breakout: Price cleared Call Wall ({call_wall:.0f}) in -Γ expansion regime."
                    is_breakout_action = True
                elif d_vector <= -0.5 and spot <= put_wall:
                    action = "BUY_PE"
                    action_label = f"BUY {round(spot/50)*50} PE | -Γ Explosive Breakdown"
                    reason = f"Gamma Breakdown: Price breached Put Wall ({put_wall:.0f}) in -Γ expansion regime."
                    is_breakout_action = True

        # A desk-invented trade carries no chart confluence score, so it previously
        # inherited the 50.0 placeholder — i.e. no quality measurement at all.
        # Derive a real one from the independent families and the positioning pillars.
        if action != "WAIT":
            wanted_vote = 1 if action == "BUY_CE" else -1
            fam_agree = sum(1 for v in family_votes.values() if v == wanted_vote)
            confluence_score = round(
                55.0 + 10.0 * fam_agree + 2.5 * (desk_state.agreement_count - 2), 1
            )
            confluence_score = float(np.clip(confluence_score, 0.0, 100.0))
            confluence_grade = (
                "A+ Institutional" if confluence_score >= 85
                else "A Standard" if confluence_score >= 70
                else "B Moderate" if confluence_score >= 55
                else "C Weak / Vetoed"
            )

    # 8. Conviction Synthesis — how hard to bet, not just which way.
    conviction_score, conviction_tier, family_agreement, conviction_notes = compute_conviction(
        action=action,
        votes=family_votes,
        confluence_score=confluence_score,
        desk_state=desk_state,
        data_quality=data_quality,
        edge_status=edge_status,
        is_breakout=is_breakout_action
    )

    # 8b. Conviction floor — an actual veto, not a caption.
    # Conviction was computed after the action was already decided and used only to
    # prefix the label, so a LOW-conviction setup still emitted a full trade ticket
    # with entry, stops and lot count. That repeats the original sin this whole branch
    # exists to undo: the system grading its own trade poorly and taking it anyway.
    if action != "WAIT" and conviction_score < MIN_CONVICTION_TO_TRADE:
        conflicts.append(
            f"CONVICTION_FLOOR: {conviction_score:.0f} < {MIN_CONVICTION_TO_TRADE:.0f} "
            f"({conviction_tier}, {family_agreement}/4 families agree)."
        )
        action = "WAIT"
        action_label = "NO-TRADE (WAIT) — BELOW CONVICTION FLOOR"
        reason = " | ".join(conflicts)
        option_pick = None

    if action != "WAIT":
        action_label = f"[{conviction_tier}] {action_label}"

    return DeskVerdict(
        action=action,
        action_label=action_label,
        reason=reason,
        trend_bias=trend_bias,
        trend_conviction_pct=conviction,
        range_corridor=range_corridor,
        max_pain=max_pain,
        expected_move_pts=expected_move,
        spot_position_pct=spot_pos_pct,
        option_pick=option_pick,
        evidence=evidence,
        gate_audit=gate_audit,
        conflicts=conflicts,
        confluence_score=confluence_score,
        confluence_grade=confluence_grade,
        data_quality=data_quality,
        conviction_score=conviction_score,
        conviction_tier=conviction_tier,
        family_votes=family_votes,
        family_agreement=family_agreement,
        directional_score=directional_score,
        conviction_notes=conviction_notes,
        edge_status=edge_status
    )
