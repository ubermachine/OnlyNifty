"""OnlyNifty v5.2 Options Desk Positioning Engine.

Fuses all options microstructure and positioning analytics:
- 5-Pillar Directional Vector (D-vector)
- PCR Level Z-Score against intraday session history
- ITM vs. OTM Net Delta OI Accumulation Shift
- Max Pain Drift tracking across snapshots
- Expected vs. Actual move ratio
- Strike-Level Dealer Gamma Profile and Walls
- Multi-pillar institutional agreement count
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
import os
import json
import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_IV,
    OPTIONS_STATE_PATH,
    PCR_HISTORY_MIN_SAMPLES,
    WALL_BUFFER_PTS
)
from src.options_flow import (
    compute_short_term_directional_vector,
    compute_oi_based_range_forecast,
    compute_strike_level_gex_chart_data,
    compute_oi_change_heatmap,
    compute_vanna_charm_drift_vector,
    _extract_normalized_chain
)
from src.options_engine import black_scholes_greeks
from src.volatility_engine import VolatilityIntelligence


@dataclass
class OptionsDeskState:
    """Consolidated quantitative positioning state from the institutional options desk."""
    trend_bias: str                    # "BULLISH" | "BEARISH" | "NEUTRAL"
    trend_conviction_pct: float        # 0.0 to 100.0%
    d_vector: float                    # -1.0 to +1.0
    pcr_level: float                   # Current OI PCR
    pcr_zscore: float                  # vs session history
    pcr_momentum_score: float          # 15m derivative
    put_wall: float
    call_wall: float
    max_pain: float
    max_pain_drift_pts: float          # vs session history
    expected_move_pts: float           # straddle implied move
    actual_range_pts: float            # high - low of day
    move_ratio: float                  # actual / expected
    gamma_regime: str                  # "DEALER_LONG_GAMMA" | "DEALER_SHORT_GAMMA"
    is_positive_gamma: bool
    zero_gex_strike: float
    writing_bias: str                  # "CALL_WRITING_HEAVY_RESISTANCE" | "PUT_WRITING_HEAVY_SUPPORT" | "BALANCED_RANGE"
    itm_otm_shift: float               # -1.0 to +1.0 (+ is bullish)
    agreement_count: int               # 0..4
    data_quality: str                  # "VERIFIED" | "POSITIONING_UNVERIFIED"
    dealer_drift_score: float = 0.0    # Vanna/Charm mechanical drift vector [-1.0, +1.0]
    dwv_momentum_score: float = 0.0    # Delta-Weighted Volume flow score [-1.0, +1.0]
    gamma_flip_distance_pts: float = 0.0 # spot - zero_gex_strike (pts)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_options_history(path: str = OPTIONS_STATE_PATH) -> Dict[str, List[float]]:
    """Loads historical session snapshots of PCR and Max Pain for Z-score and drift computation."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {
                    "pcr_series": data.get("pcr_series", []),
                    "max_pain_series": data.get("max_pain_series", [])
                }
        except Exception:
            pass
    return {"pcr_series": [], "max_pain_series": []}


def save_options_history(history: Dict[str, List[float]], path: str = OPTIONS_STATE_PATH) -> None:
    """Saves session options history to disk (capped at 120 samples = ~1 full trading day)."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Cap series length
        trimmed = {
            "pcr_series": history.get("pcr_series", [])[-120:],
            "max_pain_series": history.get("max_pain_series", [])[-120:]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(trimmed, f, indent=2)
    except Exception:
        pass


def clamp_targets_to_corridor(
    entry: float,
    t1: float,
    t2: float,
    direction: str,
    put_wall: float,
    call_wall: float
) -> Tuple[float, float, float]:
    """
    Clamps targets so trades never project unrealistically past major dealer walls in positive gamma.
    Returns (clamped_t1, clamped_t2, sl_hint).
    """
    if direction == "LONG":
        max_t = call_wall if call_wall > entry else entry + 150.0
        clean_t1 = min(t1, max_t)
        clean_t2 = min(t2, max_t)
        sl_hint = put_wall - WALL_BUFFER_PTS if put_wall < entry else entry - 40.0
    else:
        min_t = put_wall if (0 < put_wall < entry) else entry - 150.0
        clean_t1 = max(t1, min_t)
        clean_t2 = max(t2, min_t)
        sl_hint = call_wall + WALL_BUFFER_PTS if call_wall > entry else entry + 40.0

    return clean_t1, clean_t2, sl_hint


def compute_options_desk_state(
    option_chain_df: Optional[pd.DataFrame],
    spot: float,
    df_ohlcv: Optional[pd.DataFrame] = None,
    prev_chain_df: Optional[pd.DataFrame] = None,
    pcr_analytics: Optional[Dict[str, Any]] = None,
    dir_flow_res: Optional[Dict[str, Any]] = None,
    range_fc_res: Optional[Dict[str, Any]] = None,
    gex_chart_res: Optional[Dict[str, Any]] = None,
    live_iv: float = DEFAULT_IV,
    hfi_score: float = 0.0,
    history: Optional[Dict[str, List[float]]] = None,
    persist_history: bool = True
) -> OptionsDeskState:
    """
    Synthesizes real-time options positioning, order flow vectors, and dealer Greeks
    into an authoritative OptionsDeskState.
    """
    data_quality = "VERIFIED"
    if option_chain_df is None or (isinstance(option_chain_df, pd.DataFrame) and option_chain_df.empty):
        data_quality = "POSITIONING_UNVERIFIED"

    # 1. 5-Pillar Directional Vector (D)
    if dir_flow_res is None:
        dir_flow_res = compute_short_term_directional_vector(
            spot=spot,
            df=df_ohlcv,
            live_iv=live_iv,
            hfi_score=hfi_score
        )
    d_vector = float(dir_flow_res.get("directional_vector", 0.0))
    
    # IMP-6: Expiry-day Charm/Vanna weight boost
    # Near expiry, second-order Greeks (Charm, Vanna) dominate mechanical price drivers.
    # Boost Vanna-Charm sub-score contribution and ATM straddle gamma sensitivity.
    vc_score = float(dir_flow_res.get("sub_scores", {}).get("vanna_charm", 0.0))
    straddle_score = float(dir_flow_res.get("sub_scores", {}).get("straddle_state", 0.0))
    is_expiry_proximity = dir_flow_res.get("is_expiry_day", False)
    if is_expiry_proximity and abs(vc_score) > 0.1:
        # On expiry day, Charm/Vanna flows are the primary mechanical driver.
        # Amplify D-vector by up to 20% toward the Charm/Vanna direction.
        charm_boost = 0.20 * np.sign(vc_score)
        d_vector = float(np.clip(d_vector + charm_boost, -1.0, 1.0))

    conviction = float(dir_flow_res.get("conviction_percentage", 0.0))
    raw_bias = dir_flow_res.get("short_term_bias", "NEUTRAL")

    if "BULLISH" in raw_bias:
        trend_bias = "BULLISH"
    elif "BEARISH" in raw_bias:
        trend_bias = "BEARISH"
    else:
        trend_bias = "NEUTRAL"

    # 2. PCR Analytics & Z-Score
    pcr_level = 1.0
    pcr_mom_score = 0.0
    max_pain = round(spot / 50.0) * 50.0
    
    if pcr_analytics:
        pcr_level = float(pcr_analytics.get("pcr_oi", 1.0))
        max_pain = float(pcr_analytics.get("max_pain_strike", max_pain))
    
    pcr_mom_score = float(dir_flow_res.get("sub_scores", {}).get("pcr_momentum", 0.0))

    # History & Z-Score
    hist = history if history is not None else load_options_history()
    pcr_series = hist.setdefault("pcr_series", [])
    max_pain_series = hist.setdefault("max_pain_series", [])

    # Calculate drift vs last snapshot
    # NOTE (IMP-9): Max Pain drift is DISPLAY-ONLY context for the Desk Verdict UI.
    # Academic evidence (Filippou et al. 2022) shows Max Pain is NOT a reliable
    # directional predictor for broad indices. It is excluded from directional scoring.
    max_pain_drift = 0.0
    if max_pain_series:
        max_pain_drift = round(max_pain - max_pain_series[-1], 2)

    pcr_series.append(pcr_level)
    max_pain_series.append(max_pain)

    pcr_zscore = 0.0
    if len(pcr_series) >= PCR_HISTORY_MIN_SAMPLES:
        pcr_arr = np.array(pcr_series, dtype=np.float64)
        # Academic finding (Blau et al. 2015): True neutral PCR baseline
        # for equities is 0.60-0.80, NOT 1.0. Using 0.70 as structural anchor.
        # Smooth with 21-sample MA to reduce noise per academic recommendation.
        if len(pcr_arr) >= 21:
            smoothed_pcr = float(pd.Series(pcr_arr).rolling(21).mean().iloc[-1])
        else:
            smoothed_pcr = float(np.mean(pcr_arr))
        std_pcr = float(np.std(pcr_arr))
        if std_pcr > 1e-4:
            from src.config import PCR_STRUCTURAL_BASELINE
            pcr_zscore = round(float((smoothed_pcr - PCR_STRUCTURAL_BASELINE) / std_pcr), 2)

    if persist_history:
        save_options_history(hist)

    # 3. GEX Walls & Gamma Regime
    if gex_chart_res is None and data_quality == "VERIFIED":
        gex_chart_res = compute_strike_level_gex_chart_data(option_chain_df, spot, live_iv, t_days=1.0)

    if gex_chart_res:
        call_wall = float(gex_chart_res.get("call_wall_strike", round(spot + 200, -2)))
        put_wall = float(gex_chart_res.get("put_wall_strike", round(spot - 200, -2)))
        if call_wall <= put_wall:
            call_wall = put_wall + 200.0
        zero_gex_strike = float(gex_chart_res.get("zero_gex_strike", spot))
        gamma_regime = gex_chart_res.get("net_dealer_regime", "DEALER_LONG_GAMMA")
        is_positive_gamma = gamma_regime.startswith("DEALER_LONG_GAMMA") or gamma_regime.startswith("POSITIVE")
    else:
        call_wall = round(spot + 200.0, -2)
        put_wall = round(spot - 200.0, -2)
        zero_gex_strike = spot
        gamma_regime = "DEALER_LONG_GAMMA"
        is_positive_gamma = True

    # 4. Range Forecast & Expected Move
    if range_fc_res is None and data_quality == "VERIFIED":
        range_fc_res = compute_oi_based_range_forecast(option_chain_df, spot, max_pain)

    if range_fc_res:
        rf_call = float(range_fc_res.get("call_wall", 0.0))
        rf_put = float(range_fc_res.get("put_wall", 0.0))
        if rf_call > rf_put > 0:
            call_wall = rf_call
            put_wall = rf_put

    # Expected Move — the conventional 1-sigma straddle-implied close-to-close move.
    sigma_daily_pts = spot * max(live_iv, 0.05) * np.sqrt(1.0 / 365.0)
    expected_move_pts = round(sigma_daily_pts * 0.8, 1)

    actual_range_pts = 0.0
    if df_ohlcv is not None and not df_ohlcv.empty:
        actual_range_pts = round(float(df_ohlcv["high"].max() - df_ohlcv["low"].min()), 1)

    # move_ratio compares a HIGH-LOW RANGE against an expected range, so it must not be
    # divided by the 1-sigma POINT move. For driftless Brownian motion the expected range
    # is E[max-min] = 2*sqrt(2/pi)*sigma ~= 1.596*sigma. Dividing the range by the 1-sigma
    # move instead made move_ratio structurally ~2.0 on a perfectly ordinary day, so any
    # "range exhausted" threshold near 1.3 fired essentially every bar.
    expected_range_pts = 1.596 * sigma_daily_pts
    move_ratio = round(actual_range_pts / max(expected_range_pts, 1.0), 2)

    # 5. OI Heatmap Writing Bias & ITM/OTM Shift
    writing_bias = "BALANCED_RANGE"
    itm_otm_shift = 0.0

    if data_quality == "VERIFIED" and option_chain_df is not None:
        try:
            hm_res = compute_oi_change_heatmap(option_chain_df, spot)
            writing_bias = hm_res.get("writing_bias", "BALANCED_RANGE")
            
            # ITM / OTM shift calculation
            norm_chain = _extract_normalized_chain(option_chain_df, spot=spot)
            if not norm_chain.empty:
                atm = round(spot / 50.0) * 50.0
                ce_itm_change = norm_chain[norm_chain["strike"] < atm]["ce_change_oi"].sum()
                pe_itm_change = norm_chain[norm_chain["strike"] > atm]["pe_change_oi"].sum()
                ce_otm_change = norm_chain[norm_chain["strike"] > atm]["ce_change_oi"].sum()
                pe_otm_change = norm_chain[norm_chain["strike"] < atm]["pe_change_oi"].sum()

                bull_delta = (ce_itm_change + pe_otm_change)
                bear_delta = (pe_itm_change + ce_otm_change)
                total_delta = abs(bull_delta) + abs(bear_delta)
                
                if total_delta > 0:
                    itm_otm_shift = round(float((bull_delta - bear_delta) / total_delta), 2)
        except Exception:
            pass

    # 6. Mechanical Dealer Drift (Vanna/Charm) & Delta-Weighted Volume (DWV)
    dealer_drift_score = 0.0
    dwv_momentum_score = 0.0
    atm_k = round(spot / 50.0) * 50.0
    
    try:
        vc_res = compute_vanna_charm_drift_vector(spot=spot, strike=atm_k, iv=live_iv, t_days=1.0)
        dealer_drift_score = float(vc_res.get("drift_score", 0.0))
    except Exception:
        pass

    if data_quality == "VERIFIED" and option_chain_df is not None:
        try:
            dwv_momentum_score = compute_delta_weighted_volume(option_chain_df, spot=spot, live_iv=live_iv)
        except Exception:
            pass

    gamma_flip_distance_pts = round(spot - zero_gex_strike, 1)

    # 7. Multi-Pillar Agreement Count (0..4)
    # Pillars: D-vector sign, PCR momentum, ITM/OTM shift, Writing bias
    votes = 0
    if trend_bias == "BULLISH":
        if d_vector > 0.1: votes += 1
        if pcr_mom_score > 0.0: votes += 1
        if itm_otm_shift > 0.0: votes += 1
        if "PUT_WRITING" in writing_bias: votes += 1
    elif trend_bias == "BEARISH":
        if d_vector < -0.1: votes += 1
        if pcr_mom_score < 0.0: votes += 1
        if itm_otm_shift < 0.0: votes += 1
        if "CALL_WRITING" in writing_bias: votes += 1
    else:
        votes = 2

    return OptionsDeskState(
        trend_bias=trend_bias,
        trend_conviction_pct=conviction,
        d_vector=d_vector,
        pcr_level=pcr_level,
        pcr_zscore=pcr_zscore,
        pcr_momentum_score=pcr_mom_score,
        put_wall=put_wall,
        call_wall=call_wall,
        max_pain=max_pain,
        max_pain_drift_pts=max_pain_drift,
        expected_move_pts=expected_move_pts,
        actual_range_pts=actual_range_pts,
        move_ratio=move_ratio,
        gamma_regime=gamma_regime,
        is_positive_gamma=is_positive_gamma,
        zero_gex_strike=zero_gex_strike,
        writing_bias=writing_bias,
        itm_otm_shift=itm_otm_shift,
        agreement_count=votes,
        data_quality=data_quality,
        dealer_drift_score=dealer_drift_score,
        dwv_momentum_score=dwv_momentum_score,
        gamma_flip_distance_pts=gamma_flip_distance_pts
    )


def compute_delta_weighted_volume(
    option_chain_df: Optional[pd.DataFrame],
    spot: float,
    live_iv: float = DEFAULT_IV
) -> float:
    """
    Computes Delta-Weighted Volume (DWV) momentum flow across strikes:
    DWV_i = Volume_CE * Delta_CE + Volume_PE * Delta_PE.
    Normalizes the aggregate institutional directional conviction into [-1.0, +1.0].
    """
    clean_df = _extract_normalized_chain(option_chain_df, spot=spot)
    if clean_df.empty:
        return 0.0

    sigma = (live_iv / 100.0) if live_iv > 1.0 else live_iv
    sigma = max(sigma, 0.01)

    total_dwv = 0.0
    for _, row in clean_df.iterrows():
        k = float(row.get("strike", spot))
        ce_vol = float(row.get("ce_volume", 0.0) or 0.0)
        pe_vol = float(row.get("pe_volume", 0.0) or 0.0)

        delta_ce = black_scholes_greeks(spot, k, t_days=1.0, sigma=sigma, is_call=True)["delta"]
        delta_pe = black_scholes_greeks(spot, k, t_days=1.0, sigma=sigma, is_call=False)["delta"]

        strike_dwv = (ce_vol * delta_ce) + (pe_vol * delta_pe)
        total_dwv += strike_dwv

    # Scale with tanh across a typical intraday index volume magnitude (50,000 contracts)
    return round(float(np.tanh(total_dwv / 50000.0)), 3)


def compute_zero_gamma_level(
    option_chain_df: Optional[pd.DataFrame],
    spot: float,
    live_iv: float = DEFAULT_IV,
    t_days: float = 1.0
) -> Dict[str, Any]:
    """
    Computes the exact Zero Gamma Flip level (ZGF) and dealer gamma regime.
    """
    gex_data = compute_strike_level_gex_chart_data(option_chain_df, spot=spot, iv=live_iv, t_days=t_days)
    zero_strike = float(gex_data.get("zero_gex_strike", spot))
    total_net_gex = float(gex_data.get("total_net_gex_cr", 0.0))
    gamma_regime = gex_data.get("net_dealer_regime", "DEALER_LONG_GAMMA")
    is_positive = total_net_gex >= 0.0 and spot >= zero_strike

    return {
        "zero_gex_strike": zero_strike,
        "gamma_flip_distance_pts": round(spot - zero_strike, 1),
        "total_net_gex_cr": total_net_gex,
        "gamma_regime": "DEALER_LONG_GAMMA" if is_positive else "DEALER_SHORT_GAMMA",
        "is_positive_gamma": is_positive,
        "call_wall_strike": float(gex_data.get("call_wall_strike", spot + 200)),
        "put_wall_strike": float(gex_data.get("put_wall_strike", spot - 200))
    }
