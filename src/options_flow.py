"""
JustNifty v5.1 Institutional Options Flow & Strike-Level Heatmap Analytics Engine.

Provides real-time mathematical derivatives microstructure analytics:
1. Combined ATM Straddle & Expected Day Range Corridor (Upper / Lower Breakevens)
2. Cumulative OI Delta (COID) & 4-Quadrant Trap Detection (Long/Short Buildup vs Short-Covering Traps)
3. Dealer Vanna-Charm Mechanical Drift Vector (Continuous Delta-Hedging Flow)
4. PCR 15-Minute Momentum Velocity (dPCR/dt)
5. Composite Short-Term Directional Vector (D_intraday in [-1.0, +1.0]) with Conviction Scoring
6. Strike-Level Open Interest Change Heatmap & Writing Bias Classification (v5.1)
7. Strike-by-Strike Dealer Gamma Exposure (GEX) Distribution & Gamma Flip Analytics (v5.1)
8. Institutional OI-Based Range Forecast & Location Bias Classification (v5.1)
"""

import math
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.config import DEFAULT_IV, RISK_FREE_RATE, LOT_SIZE
from src.options_engine import black_scholes_greeks, calculate_pcr_and_max_pain


def compute_atm_straddle_metrics(
    spot: float,
    option_chain_df: Optional[pd.DataFrame] = None,
    live_iv: float = DEFAULT_IV,
    t_days: float = 4.0,
    r: float = RISK_FREE_RATE
) -> Dict[str, Any]:
    """
    Computes Combined ATM Straddle premium, expected range corridor, and theoretical VWAP.
    
    Formula:
    - ATM Strike: K_atm = round(spot / 50) * 50
    - Combined Straddle: P_straddle = C_atm + P_atm
    - Expected Upper Range: K_atm + P_straddle
    - Expected Lower Range: K_atm - P_straddle
    """
    spot = max(float(spot), 1.0)
    atm_strike = int(round(spot / 50.0) * 50)
    sigma_clean = (live_iv / 100.0) if live_iv > 1.0 else live_iv
    sigma_clean = max(sigma_clean, 0.01)
    
    call_prem = 0.0
    put_prem = 0.0
    call_vol = 0.0
    put_vol = 0.0
    
    found_in_chain = False
    if option_chain_df is not None and not option_chain_df.empty:
        strike_col = "strikePrice" if "strikePrice" in option_chain_df.columns else "strike"
        if strike_col in option_chain_df.columns:
            atm_rows = option_chain_df[option_chain_df[strike_col] == atm_strike]
            if not atm_rows.empty:
                row = atm_rows.iloc[0]
                call_prem = float(row.get("CE_ltp", row.get("ce_ltp", 0.0)) or 0.0)
                put_prem = float(row.get("PE_ltp", row.get("pe_ltp", 0.0)) or 0.0)
                call_vol = float(row.get("CE_volume", row.get("ce_volume", 0.0)) or 0.0)
                put_vol = float(row.get("PE_volume", row.get("pe_volume", 0.0)) or 0.0)
                if call_prem > 0 and put_prem > 0:
                    found_in_chain = True

    if not found_in_chain or call_prem <= 0 or put_prem <= 0:
        # BSM Theoretical fallback
        greeks_ce = black_scholes_greeks(spot, atm_strike, t_days=t_days, sigma=sigma_clean, is_call=True, r=r)
        greeks_pe = black_scholes_greeks(spot, atm_strike, t_days=t_days, sigma=sigma_clean, is_call=False, r=r)
        call_prem = greeks_ce["price"]
        put_prem = greeks_pe["price"]
        call_vol = 50000.0
        put_vol = 50000.0

    straddle_prem = round(float(call_prem + put_prem), 2)
    upper_breakeven = round(float(atm_strike + straddle_prem), 2)
    lower_breakeven = round(float(atm_strike - straddle_prem), 2)
    range_width_pts = round(float(upper_breakeven - lower_breakeven), 2)
    expected_move_pct = round((straddle_prem / spot) * 100.0, 2)
    
    theoretical_straddle_vwap = round(straddle_prem * 1.02, 2)
    vol_state = "VOL_EXPANSION" if straddle_prem >= theoretical_straddle_vwap else "THETA_DECAY"
    
    spot_range_pos = 0.0
    if range_width_pts > 0:
        spot_range_pos = round(((spot - lower_breakeven) / range_width_pts) * 100.0, 1)

    return {
        "atm_strike": atm_strike,
        "call_premium": round(float(call_prem), 2),
        "put_premium": round(float(put_prem), 2),
        "straddle_premium": straddle_prem,
        "upper_breakeven": upper_breakeven,
        "lower_breakeven": lower_breakeven,
        "range_width_pts": range_width_pts,
        "expected_move_pct": expected_move_pct,
        "straddle_vwap": theoretical_straddle_vwap,
        "vol_state": vol_state,
        "spot_range_pct": spot_range_pos
    }


def compute_cumulative_oi_delta_and_traps(
    option_chain_df: Optional[pd.DataFrame] = None,
    spot: float = 24800.0,
    prev_chain_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Computes Cumulative OI Delta (COID), Net Put-Call OI velocity, and 4-Quadrant Trap Classifications.
    """
    if option_chain_df is None or option_chain_df.empty:
        atm = int(round(spot / 50.0) * 50)
        return {
            "net_oi_delta": 450000,
            "net_oi_pulse_score": 0.45,
            "call_wall": atm + 150,
            "put_wall": atm - 150,
            "active_quadrant": "LONG_BUILDUP",
            "trap_flag": False,
            "trap_warning": "No Trap Detected (Active Accumulation)",
            "total_ce_oi_change": -20000,
            "total_pe_oi_change": 430000,
            "strike_diagnostics": []
        }
        
    df = option_chain_df.copy()
    strike_col = "strikePrice" if "strikePrice" in df.columns else "strike"
    
    if strike_col in df.columns:
        df["dist"] = (df[strike_col] - spot).abs()
        df = df.sort_values("dist").head(14).sort_values(strike_col)
    
    total_ce_oi_change = 0.0
    total_pe_oi_change = 0.0
    total_ce_oi = 0.0
    total_pe_oi = 0.0
    
    strike_diagnostics = []
    
    max_ce_oi = -1.0
    max_pe_oi = -1.0
    call_wall = spot + 100
    put_wall = spot - 100
    
    for _, row in df.iterrows():
        k = float(row.get(strike_col, 0.0))
        ce_oi = float(row.get("CE_oi", row.get("ce_oi", 0.0)) or 0.0)
        pe_oi = float(row.get("PE_oi", row.get("pe_oi", 0.0)) or 0.0)
        ce_chg = float(row.get("CE_changeinOpenInterest", row.get("ce_change_oi", 0.0)) or 0.0)
        pe_chg = float(row.get("PE_changeinOpenInterest", row.get("pe_change_oi", 0.0)) or 0.0)
        
        total_ce_oi += ce_oi
        total_pe_oi += pe_oi
        total_ce_oi_change += ce_chg
        total_pe_oi_change += pe_chg
        
        # Prioritize strikes near or above spot for Call wall
        if ce_oi > max_ce_oi and (k >= spot - 100):
            max_ce_oi = ce_oi
            call_wall = k
        elif max_ce_oi < 0:
            max_ce_oi = ce_oi
            call_wall = k
            
        # Prioritize strikes near or below spot for Put wall
        if pe_oi > max_pe_oi and (k <= spot + 100):
            max_pe_oi = pe_oi
            put_wall = k
        elif max_pe_oi < 0:
            max_pe_oi = pe_oi
            put_wall = k
            
        ce_quad = "CALL_WRITING_RESISTANCE" if ce_chg > 0 else ("CALL_UNWINDING_SHORT_COVER" if ce_chg < 0 else "NEUTRAL")
        pe_quad = "PUT_WRITING_SUPPORT" if pe_chg > 0 else ("PUT_UNWINDING_LONG_EXIT" if pe_chg < 0 else "NEUTRAL")
            
        strike_diagnostics.append({
            "strike": int(k),
            "ce_oi": int(ce_oi),
            "pe_oi": int(pe_oi),
            "ce_change_oi": int(ce_chg),
            "pe_change_oi": int(pe_chg),
            "net_strike_oi_delta": int(pe_chg - ce_chg),
            "ce_regime": ce_quad,
            "pe_regime": pe_quad
        })
        
    net_oi_delta = total_pe_oi_change - total_ce_oi_change
    total_abs_chg = abs(total_pe_oi_change) + abs(total_ce_oi_change) + 1.0
    net_oi_pulse_score = max(min(net_oi_delta / total_abs_chg, 1.0), -1.0)
    
    # Global Trap Classification
    trap_flag = False
    trap_warning = "Normal Institutional Positioning"
    active_quad = "NEUTRAL"
    
    if total_ce_oi_change < -25000 and total_pe_oi_change <= max(0.50 * abs(total_ce_oi_change), 50000):
        active_quad = "SHORT_COVERING_TRAP"
        trap_flag = True
        trap_warning = "⚠️ SHORT COVERING RALLY: Call unwinding without Put buildup. Do NOT chase breakout!"
    elif total_pe_oi_change < -25000 and total_ce_oi_change <= max(0.50 * abs(total_pe_oi_change), 50000):
        active_quad = "LONG_LIQUIDATION_TRAP"
        trap_flag = True
        trap_warning = "⚠️ LONG LIQUIDATION: Weak hands exiting without heavy Call writing. Watch for bounce!"
    elif net_oi_pulse_score > 0.20:
        active_quad = "LONG_BUILDUP"
        trap_warning = "🟢 FRESH LONG ACCUMULATION: Put writing outpaces Call writing."
    elif net_oi_pulse_score < -0.20:
        active_quad = "SHORT_BUILDUP"
        trap_warning = "🔴 FRESH SHORT DISTRIBUTION: Heavy Call writing overhead."
    else:
        active_quad = "NEUTRAL_CONSOLIDATION"
        trap_warning = "🟡 RANGEBOUND: Balanced Call and Put positioning."
        
    return {
        "net_oi_delta": int(net_oi_delta),
        "net_oi_pulse_score": round(float(net_oi_pulse_score), 3),
        "call_wall": int(call_wall),
        "put_wall": int(put_wall),
        "active_quadrant": active_quad,
        "trap_flag": trap_flag,
        "trap_warning": trap_warning,
        "total_ce_oi_change": int(total_ce_oi_change),
        "total_pe_oi_change": int(total_pe_oi_change),
        "strike_diagnostics": strike_diagnostics
    }


def compute_pcr_momentum_derivative(
    current_pcr: float,
    prev_pcr: float = 1.0,
    delta_t_mins: float = 15.0
) -> Dict[str, Any]:
    """
    Computes 15-minute PCR Momentum Slope d(PCR)/dt.
    """
    dt = max(float(delta_t_mins), 1.0)
    dpcr_dt = (float(current_pcr) - float(prev_pcr)) / dt
    
    pcr_score = float(np.tanh(35.0 * dpcr_dt))
    
    if pcr_score > 0.25:
        status = "BULLISH_PCR_EXPANSION"
    elif pcr_score < -0.25:
        status = "BEARISH_PCR_COLLAPSE"
    else:
        status = "STABLE_PCR"
        
    return {
        "current_pcr": round(float(current_pcr), 3),
        "prev_pcr": round(float(prev_pcr), 3),
        "dpcr_dt_per_min": round(dpcr_dt, 5),
        "pcr_momentum_score": round(pcr_score, 3),
        "status": status
    }


def compute_vanna_charm_drift_vector(
    spot: float,
    strike: float,
    iv: float = DEFAULT_IV,
    d_iv_dt: float = 0.0,
    t_days: float = 4.0,
    r: float = RISK_FREE_RATE
) -> Dict[str, Any]:
    """
    Computes Dealer Vanna-Charm Mechanical Drift Vector.
    """
    spot = max(float(spot), 1.0)
    strike = max(float(strike), 1.0)
    t_years = max(t_days / 365.0, 0.0001)
    sigma = (iv / 100.0) if iv > 1.0 else iv
    sigma = max(sigma, 0.01)
    
    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r + 0.5 * (sigma ** 2)) * t_years) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    
    pdf_d1 = norm.pdf(d1)
    vega = spot * pdf_d1 * sqrt_t * 0.01
    
    vanna = (vega / spot) * (1.0 - (d1 / (sigma * sqrt_t))) if (sigma * sqrt_t) > 0 else 0.0
    charm_annual = - pdf_d1 * ((2.0 * r * t_years - d2 * sigma * sqrt_t) / (2.0 * t_years * sigma * sqrt_t)) if (2.0 * t_years * sigma * sqrt_t) > 0 else 0.0
    charm_daily = charm_annual / 365.0
    
    vanna_drift = - float(d_iv_dt) * vanna * 25.0
    charm_drift = charm_daily * 15.0
    
    composite_drift = max(min(vanna_drift + charm_drift, 1.0), -1.0)
    
    if composite_drift > 0.30:
        regime = "VANNA_BULLISH_VACUUM (IV Crushing on Rally)"
    elif composite_drift < -0.30:
        regime = "GAMMA_BEARISH_CASCADE (IV Expanding on Drop)"
    else:
        regime = "BALANCED_DEALER_INVENTORY"
        
    return {
        "vanna": round(float(vanna), 6),
        "charm_daily": round(float(charm_daily), 6),
        "drift_score": round(float(composite_drift), 3),
        "regime": regime
    }


def compute_short_term_directional_vector(
    spot: float,
    df: pd.DataFrame,
    option_chain_df: Optional[pd.DataFrame] = None,
    prev_chain_df: Optional[pd.DataFrame] = None,
    live_iv: float = DEFAULT_IV,
    prev_pcr: float = 1.0,
    hfi_score: float = 0.0
) -> Dict[str, Any]:
    """
    Synthesizes the Unified 5-Pillar Short-Term Directional Vector (D_intraday in [-1.0, +1.0]).
    """
    spot = max(float(spot), 1.0)
    
    straddle_res = compute_atm_straddle_metrics(spot, option_chain_df, live_iv=live_iv)
    s_straddle = 0.0
    if len(df) >= 2:
        price_ret = (df["close"].iloc[-1] - df["close"].iloc[-2])
        if straddle_res["vol_state"] == "VOL_EXPANSION":
            s_straddle = 0.75 if price_ret >= 0 else -0.75
        else:
            s_straddle = -0.25 if spot > straddle_res["atm_strike"] else 0.25
            
    oi_res = compute_cumulative_oi_delta_and_traps(option_chain_df, spot=spot, prev_chain_df=prev_chain_df)
    s_doi = oi_res["net_oi_pulse_score"]
    
    curr_pcr = 1.0
    if option_chain_df is not None and not option_chain_df.empty:
        tot_pe = oi_res.get("total_pe_oi_change", 0) + 100000
        tot_ce = oi_res.get("total_ce_oi_change", 0) + 100000
        curr_pcr = tot_pe / max(tot_ce, 1.0)
    pcr_res = compute_pcr_momentum_derivative(curr_pcr, prev_pcr=prev_pcr, delta_t_mins=15.0)
    s_pcr = pcr_res["pcr_momentum_score"]
    
    d_iv_dt = -0.02 if (len(df) >= 3 and df["close"].iloc[-1] >= df["close"].iloc[-3]) else 0.02
    vc_res = compute_vanna_charm_drift_vector(spot, straddle_res["atm_strike"], iv=live_iv, d_iv_dt=d_iv_dt)
    s_vc = vc_res["drift_score"]
    
    s_hfi = max(min(float(hfi_score), 1.0), -1.0)
    
    d_intraday = (
        0.30 * s_doi +
        0.25 * s_vc +
        0.20 * s_pcr +
        0.15 * s_straddle +
        0.10 * s_hfi
    )
    d_intraday = max(min(round(d_intraday, 3), 1.0), -1.0)
    conviction_pct = round(abs(d_intraday) * 100.0, 1)
    
    if d_intraday >= 0.50:
        bias = "STRONG_BULLISH_LONG"
        emoji = "🚀"
        badge_color = "#05df72"
        action = "BUY_ATM_CALL (Delta 0.55-0.65) | Target: Call Wall / Upper Straddle Range"
        target_price = min(oi_res["call_wall"], straddle_res["upper_breakeven"])
        stop_price = round(spot - 35.0, 1)
    elif d_intraday >= 0.20:
        bias = "MILD_BULLISH_ACCUMULATION"
        emoji = "🟢"
        badge_color = "#34d399"
        action = "BUY_CALL_ON_PULLBACK (AVWAP / 21 EMA Support) | Target: T1"
        target_price = round(spot + 45.0, 1)
        stop_price = round(spot - 25.0, 1)
    elif d_intraday <= -0.50:
        bias = "STRONG_BEARISH_SHORT"
        emoji = "🩸"
        badge_color = "#ff3355"
        action = "BUY_ATM_PUT (Delta -0.50 to -0.60) | Target: Put Wall / Lower Straddle Range"
        target_price = max(oi_res["put_wall"], straddle_res["lower_breakeven"])
        stop_price = round(spot + 35.0, 1)
    elif d_intraday <= -0.20:
        bias = "MILD_BEARISH_DISTRIBUTION"
        emoji = "🔴"
        badge_color = "#f87171"
        action = "BUY_PUT_ON_BOUNCE (AVWAP / Resistance Rejection) | Target: T1"
        target_price = round(spot - 45.0, 1)
        stop_price = round(spot + 25.0, 1)
    else:
        bias = "NEUTRAL_RANGEBOUND_CHOP"
        emoji = "🟡"
        badge_color = "#fbb024"
        action = "RANGE_TRADE_OR_IRON_CONDOR | Fade Straddle Boundaries"
        target_price = straddle_res["upper_breakeven"]
        stop_price = straddle_res["lower_breakeven"]

    return {
        "directional_vector": d_intraday,
        "conviction_pct": conviction_pct,
        "bias": bias,
        "emoji": emoji,
        "badge_color": badge_color,
        "suggested_action": action,
        "target_price": target_price,
        "stop_price": stop_price,
        "straddle_metrics": straddle_res,
        "oi_metrics": oi_res,
        "pcr_metrics": pcr_res,
        "vanna_charm_metrics": vc_res,
        "component_scores": {
            "s_doi": round(s_doi, 3),
            "s_vc": round(s_vc, 3),
            "s_pcr": round(s_pcr, 3),
            "s_straddle": round(s_straddle, 3),
            "s_hfi": round(s_hfi, 3)
        }
    }


# =============================================================================
# v5.1 STRIKE-LEVEL & HEATMAP ANALYTICS FUNCTIONS
# =============================================================================

def _extract_normalized_chain(option_chain_df: Optional[pd.DataFrame], spot: float) -> pd.DataFrame:
    """
    Normalizes option chain DataFrame column names and extracts standard derivatives metrics.
    If DataFrame is empty or None, generates a sensible synthetic fallback around spot.
    """
    if option_chain_df is None or option_chain_df.empty:
        atm = int(round(spot / 50.0) * 50)
        strikes = [atm + i * 50 for i in range(-12, 13)]
        rows = []
        for k in strikes:
            dist = abs(k - spot)
            base_oi = max(int(1000000 * math.exp(-0.5 * (dist / 300.0) ** 2)), 25000)
            rows.append({
                "strike": float(k),
                "ce_oi": base_oi if k >= spot else int(base_oi * 0.7),
                "pe_oi": base_oi if k <= spot else int(base_oi * 0.7),
                "ce_change_oi": int(base_oi * 0.15) if k >= spot else int(base_oi * -0.05),
                "pe_change_oi": int(base_oi * 0.15) if k <= spot else int(base_oi * -0.05),
                "ce_volume": int(base_oi * 0.4),
                "pe_volume": int(base_oi * 0.4),
                "ce_ltp": max(spot - k + 30.0, 10.0) if spot > k else max(100.0 - dist * 0.3, 5.0),
                "pe_ltp": max(k - spot + 30.0, 10.0) if k > spot else max(100.0 - dist * 0.3, 5.0)
            })
        return pd.DataFrame(rows)

    df = option_chain_df.copy()
    col_map = {c: str(c).strip().lower().replace(" ", "_") for c in df.columns}
    df.rename(columns=col_map, inplace=True)

    strike_col = next((c for c in ["strike", "strikeprice", "strike_price"] if c in df.columns), None)
    if not strike_col:
        for c in df.columns:
            if pd.to_numeric(df[c], errors="coerce").notnull().all():
                strike_col = c
                break
    if not strike_col:
        strike_col = df.columns[0]

    ce_oi_col = next((c for c in ["ce_oi", "ce_open_interest", "call_oi", "openinterest_ce", "ce_openinterest"] if c in df.columns), None)
    pe_oi_col = next((c for c in ["pe_oi", "pe_open_interest", "put_oi", "openinterest_pe", "pe_openinterest"] if c in df.columns), None)
    
    ce_chg_col = next((c for c in ["ce_change_oi", "ce_changeinopeninterest", "ce_change_in_open_interest", "call_change_oi", "ce_chg_oi"] if c in df.columns), None)
    pe_chg_col = next((c for c in ["pe_change_oi", "pe_changeinopeninterest", "pe_change_in_open_interest", "put_change_oi", "pe_chg_oi"] if c in df.columns), None)
    
    ce_vol_col = next((c for c in ["ce_volume", "ce_total_volume", "ce_totaltradedvolume", "call_volume", "ce_vol"] if c in df.columns), None)
    pe_vol_col = next((c for c in ["pe_volume", "pe_total_volume", "pe_totaltradedvolume", "put_volume", "pe_vol"] if c in df.columns), None)

    ce_ltp_col = next((c for c in ["ce_ltp", "ce_lastprice", "ce_last_price", "call_ltp", "call_price"] if c in df.columns), None)
    pe_ltp_col = next((c for c in ["pe_ltp", "pe_lastprice", "pe_last_price", "put_ltp", "put_price"] if c in df.columns), None)

    clean_df = pd.DataFrame()
    clean_df["strike"] = pd.to_numeric(df[strike_col], errors="coerce")
    clean_df = clean_df.dropna(subset=["strike"]).sort_values("strike").reset_index(drop=True)

    clean_df["ce_oi"] = pd.to_numeric(df[ce_oi_col], errors="coerce").fillna(0) if ce_oi_col else 0
    clean_df["pe_oi"] = pd.to_numeric(df[pe_oi_col], errors="coerce").fillna(0) if pe_oi_col else 0
    
    clean_df["ce_change_oi"] = pd.to_numeric(df[ce_chg_col], errors="coerce").fillna(0) if ce_chg_col else 0
    clean_df["pe_change_oi"] = pd.to_numeric(df[pe_chg_col], errors="coerce").fillna(0) if pe_chg_col else 0

    clean_df["ce_volume"] = pd.to_numeric(df[ce_vol_col], errors="coerce").fillna(0) if ce_vol_col else 0
    clean_df["pe_volume"] = pd.to_numeric(df[pe_vol_col], errors="coerce").fillna(0) if pe_vol_col else 0

    clean_df["ce_ltp"] = pd.to_numeric(df[ce_ltp_col], errors="coerce").fillna(0.0) if ce_ltp_col else 0.0
    clean_df["pe_ltp"] = pd.to_numeric(df[pe_ltp_col], errors="coerce").fillna(0.0) if pe_ltp_col else 0.0

    return clean_df


def compute_oi_based_range_forecast(
    option_chain_df: Optional[pd.DataFrame] = None,
    spot: float = 24800.0,
    max_pain: Optional[float] = None
) -> Dict[str, Any]:
    """
    Identifies Put Wall (support), Call Wall (resistance), Max Pain, and computes
    spot position percentage and location bias within the institutional derivatives corridor.
    
    Parameters:
    - option_chain_df: Raw or normalized option chain DataFrame.
    - spot: Current spot price of underlying index.
    - max_pain: Optional override for Max Pain strike price.
    
    Returns:
    - Dict with put_wall, call_wall, max_pain, spot_position_pct, location_bias, expected_corridor, etc.
    """
    spot = max(float(spot), 1.0)
    clean_df = _extract_normalized_chain(option_chain_df, spot=spot)

    if clean_df.empty:
        atm = int(round(spot / 50.0) * 50)
        call_wall = float(atm + 200)
        put_wall = float(atm - 200)
        max_pain_val = float(max_pain) if (max_pain is not None and max_pain > 0) else float(atm)
    else:
        # Support is strike with highest Put OI
        max_pe_idx = clean_df["pe_oi"].idxmax()
        put_wall = float(clean_df.loc[max_pe_idx, "strike"])
        
        # Resistance is strike with highest Call OI
        max_ce_idx = clean_df["ce_oi"].idxmax()
        call_wall = float(clean_df.loc[max_ce_idx, "strike"])
        
        if max_pain is not None and float(max_pain) > 0:
            max_pain_val = float(max_pain)
        else:
            mp_res = calculate_pcr_and_max_pain(clean_df)
            max_pain_val = float(mp_res.get("max_pain_strike", round(spot / 50.0) * 50))

    if call_wall <= put_wall:
        # Guard against inverted or anomalous wall ordering
        call_wall = max(call_wall, put_wall + 100.0)
        put_wall = min(put_wall, call_wall - 100.0)

    range_width = max(call_wall - put_wall, 1.0)
    raw_pct = ((spot - put_wall) / range_width) * 100.0
    spot_position_pct = max(0.0, min(100.0, round(raw_pct, 1)))

    if spot_position_pct <= 35.0:
        location_bias = "NEAR_SUPPORT_ACCUMULATION"
        bias_desc = f"Spot is near Put Wall support (₹{put_wall:.0f}). High probability of accumulation bounce."
    elif spot_position_pct >= 65.0:
        location_bias = "NEAR_RESISTANCE_DISTRIBUTION"
        bias_desc = f"Spot is near Call Wall resistance (₹{call_wall:.0f}). Heavy overhead writing supply."
    else:
        location_bias = "MID_RANGE_CONSOLIDATION"
        bias_desc = f"Spot is rangebound between ₹{put_wall:.0f} and ₹{call_wall:.0f} with Max Pain at ₹{max_pain_val:.0f}."

    return {
        "spot": round(float(spot), 2),
        "put_wall": round(float(put_wall), 2),
        "call_wall": round(float(call_wall), 2),
        "support_strike": round(float(put_wall), 2),
        "resistance_strike": round(float(call_wall), 2),
        "max_pain": round(float(max_pain_val), 2),
        "range_width_pts": round(float(range_width), 2),
        "spot_position_pct": spot_position_pct,
        "location_bias": location_bias,
        "expected_corridor": (round(float(put_wall), 2), round(float(call_wall), 2)),
        "bias_description": bias_desc
    }


def compute_oi_change_heatmap(
    option_chain_df: Optional[pd.DataFrame] = None,
    spot: float = 24800.0,
    range_pts: float = 500.0,
    highlight_threshold_mult: float = 2.0
) -> Dict[str, Any]:
    """
    Computes strike-level Open Interest change heatmap, hot strike additions,
    writing bias, color intensity score in [-1.0, +1.0], and range forecast.
    
    Parameters:
    - option_chain_df: Option chain DataFrame.
    - spot: Current index spot price.
    - range_pts: Range around spot to analyze (e.g. +/- 500 points).
    - highlight_threshold_mult: Multiplier above mean OI addition to flag a hot strike (default 2.0x).
    
    Returns:
    - Dict with heatmap_rows, hot_ce_strikes, hot_pe_strikes, total_ce_writing,
      total_pe_writing, writing_bias, support, resistance, range_forecast.
    """
    spot = max(float(spot), 1.0)
    clean_df = _extract_normalized_chain(option_chain_df, spot=spot)

    # Filter strikes within +/- range_pts
    filtered_df = clean_df[
        (clean_df["strike"] >= (spot - range_pts)) & 
        (clean_df["strike"] <= (spot + range_pts))
    ].copy().sort_values("strike").reset_index(drop=True)

    if filtered_df.empty:
        filtered_df = clean_df.copy().sort_values("strike").reset_index(drop=True)

    # Compute mean positive additions for thresholding
    ce_pos = filtered_df[filtered_df["ce_change_oi"] > 0]["ce_change_oi"]
    mean_ce_add = float(ce_pos.mean()) if not ce_pos.empty else float(filtered_df["ce_change_oi"].abs().mean() or 10000.0)
    
    pe_pos = filtered_df[filtered_df["pe_change_oi"] > 0]["pe_change_oi"]
    mean_pe_add = float(pe_pos.mean()) if not pe_pos.empty else float(filtered_df["pe_change_oi"].abs().mean() or 10000.0)

    ce_thresh = highlight_threshold_mult * mean_ce_add
    pe_thresh = highlight_threshold_mult * mean_pe_add

    heatmap_rows: List[Dict[str, Any]] = []
    hot_ce_strikes: List[float] = []
    hot_pe_strikes: List[float] = []

    for _, row in filtered_df.iterrows():
        k = float(row["strike"])
        ce_oi = int(row["ce_oi"])
        pe_oi = int(row["pe_oi"])
        ce_chg = int(row["ce_change_oi"])
        pe_chg = int(row["pe_change_oi"])
        ce_vol = int(row["ce_volume"])
        pe_vol = int(row["pe_volume"])
        
        net_strike_oi_delta = pe_chg - ce_chg
        is_hot_ce = bool(ce_chg > ce_thresh and ce_chg > 0)
        is_hot_pe = bool(pe_chg > pe_thresh and pe_chg > 0)

        if is_hot_ce:
            hot_ce_strikes.append(k)
        if is_hot_pe:
            hot_pe_strikes.append(k)

        abs_sum = abs(pe_chg) + abs(ce_chg)
        intensity = float(net_strike_oi_delta / (abs_sum + 1e-6)) if abs_sum > 0 else 0.0
        color_intensity = round(max(-1.0, min(1.0, intensity)), 3)

        ce_regime = "CALL_WRITING_RESISTANCE" if ce_chg > 0 else ("CALL_UNWINDING_COVER" if ce_chg < 0 else "NEUTRAL")
        pe_regime = "PUT_WRITING_SUPPORT" if pe_chg > 0 else ("PUT_UNWINDING_EXIT" if pe_chg < 0 else "NEUTRAL")

        heatmap_rows.append({
            "strike": k,
            "ce_oi": ce_oi,
            "pe_oi": pe_oi,
            "ce_change_oi": ce_chg,
            "pe_change_oi": pe_chg,
            "ce_volume": ce_vol,
            "pe_volume": pe_vol,
            "net_strike_oi_delta": net_strike_oi_delta,
            "is_hot_ce": is_hot_ce,
            "is_hot_pe": is_hot_pe,
            "color_intensity": color_intensity,
            "ce_regime": ce_regime,
            "pe_regime": pe_regime
        })

    total_ce_writing = int(sum(max(0, r["ce_change_oi"]) for r in heatmap_rows))
    total_pe_writing = int(sum(max(0, r["pe_change_oi"]) for r in heatmap_rows))

    if total_ce_writing > 1.25 * max(total_pe_writing, 1):
        writing_bias = "CALL_WRITING_HEAVY_RESISTANCE"
    elif total_pe_writing > 1.25 * max(total_ce_writing, 1):
        writing_bias = "PUT_WRITING_HEAVY_SUPPORT"
    else:
        writing_bias = "BALANCED_RANGE"

    # Support (highest PE OI strike) and Resistance (highest CE OI strike)
    max_pe_idx = clean_df["pe_oi"].idxmax() if not clean_df.empty else 0
    support = float(clean_df.loc[max_pe_idx, "strike"]) if not clean_df.empty else round(spot - 150.0, 2)
    
    max_ce_idx = clean_df["ce_oi"].idxmax() if not clean_df.empty else 0
    resistance = float(clean_df.loc[max_ce_idx, "strike"]) if not clean_df.empty else round(spot + 150.0, 2)

    range_fc = compute_oi_based_range_forecast(option_chain_df, spot=spot)

    return {
        "heatmap_rows": heatmap_rows,
        "hot_ce_strikes": hot_ce_strikes,
        "hot_pe_strikes": hot_pe_strikes,
        "total_ce_writing": total_ce_writing,
        "total_pe_writing": total_pe_writing,
        "writing_bias": writing_bias,
        "support": support,
        "resistance": resistance,
        "range_forecast": range_fc
    }


def compute_strike_level_gex_chart_data(
    option_chain_df: Optional[pd.DataFrame] = None,
    spot: float = 24800.0,
    iv: float = DEFAULT_IV,
    t_days: float = 4.0
) -> Dict[str, Any]:
    """
    Computes strike-by-strike Dealer Gamma Exposure (in ₹ Crores) using Black-Scholes gamma.
    Identifies Call Wall strike (max call GEX), Put Wall strike (max put GEX), and Zero-GEX flip level.
    
    Parameters:
    - option_chain_df: Option chain DataFrame.
    - spot: Index spot price.
    - iv: Implied Volatility (annualized, e.g. 0.12 or 12.0).
    - t_days: Days to expiry.
    
    Returns:
    - Structured dict ready for Plotly bar chart rendering:
      strikes, net_gex_per_strike, call_gex_per_strike, put_gex_per_strike,
      call_wall_strike, put_wall_strike, zero_gex_strike, net_dealer_regime.
    """
    spot = max(float(spot), 1.0)
    clean_df = _extract_normalized_chain(option_chain_df, spot=spot)

    sigma_clean = (iv / 100.0) if iv > 1.0 else iv
    sigma_clean = max(sigma_clean, 0.01)

    strikes: List[float] = []
    call_gex_per_strike: List[float] = []
    put_gex_per_strike: List[float] = []
    net_gex_per_strike: List[float] = []

    total_net_gex = 0.0
    total_call_gex = 0.0
    total_put_gex = 0.0

    for _, row in clean_df.iterrows():
        s = float(row["strike"])
        c_oi = float(row["ce_oi"])
        p_oi = float(row["pe_oi"])

        g_c = black_scholes_greeks(spot, s, t_days=t_days, sigma=sigma_clean, is_call=True)["gamma"]
        g_p = black_scholes_greeks(spot, s, t_days=t_days, sigma=sigma_clean, is_call=False)["gamma"]

        call_gex_cr = (g_c * c_oi * spot * LOT_SIZE * 0.01) / 1e7
        put_gex_cr = - (g_p * p_oi * spot * LOT_SIZE * 0.01) / 1e7
        net_gex_cr = call_gex_cr + put_gex_cr

        strikes.append(s)
        call_gex_per_strike.append(round(float(call_gex_cr), 3))
        put_gex_per_strike.append(round(float(put_gex_cr), 3))
        net_gex_per_strike.append(round(float(net_gex_cr), 3))

        total_net_gex += net_gex_cr
        total_call_gex += call_gex_cr
        total_put_gex += put_gex_cr

    if strikes:
        call_wall_idx = int(np.argmax(call_gex_per_strike))
        call_wall_strike = float(strikes[call_wall_idx])

        put_wall_idx = int(np.argmin(put_gex_per_strike))  # most negative put gex
        put_wall_strike = float(strikes[put_wall_idx])

        zero_gex_idx = int(np.argmin([abs(g) for g in net_gex_per_strike]))
        zero_gex_strike = float(strikes[zero_gex_idx])
    else:
        call_wall_strike = round(spot + 150.0, 2)
        put_wall_strike = round(spot - 150.0, 2)
        zero_gex_strike = round(spot, 2)

    net_dealer_regime = "DEALER_LONG_GAMMA" if total_net_gex >= 0 else "DEALER_SHORT_GAMMA"

    return {
        "strikes": strikes,
        "net_gex_per_strike": net_gex_per_strike,
        "call_gex_per_strike": call_gex_per_strike,
        "put_gex_per_strike": put_gex_per_strike,
        "call_wall_strike": call_wall_strike,
        "put_wall_strike": put_wall_strike,
        "zero_gex_strike": zero_gex_strike,
        "net_dealer_regime": net_dealer_regime,
        "total_net_gex_cr": round(float(total_net_gex), 2),
        "total_call_gex_cr": round(float(total_call_gex), 2),
        "total_put_gex_cr": round(float(total_put_gex), 2)
    }

