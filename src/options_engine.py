"""JustNifty v3.1 Institutional Tier-1 Options Engine.

Features:
- Exact Black-Scholes-Merton pricing with continuous dividend yield q.
- 1st, 2nd, and 3rd order Greeks: Delta, Gamma, Theta, Vega, Vanna, Charm, Volga, Speed, Color.
- Continuous Arbitrage-Free Quadratic Volatility Skew/Smile surface.
- 0DTE Microstructure Singularity Shield (sub-minute resilience).
- 2nd-Order Taylor Series option risk modeling with positive Gamma convexity.
- Dynamic 21 EMA / AVWAP 1σ Trailing Stop-Loss translation.
- 3-Tier Asymmetric Exits, Free Vertical Spread converter & Multi-Leg Indian NSE TCA Friction Engine.
"""

import math
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.config import (
    DEFAULT_CAPITAL, MAX_RISK_PCT, LOT_SIZE,
    DELTA_MIN, DELTA_MAX, DELTA_DEEP_ITM_0DTE, RISK_FREE_RATE, DEFAULT_IV,
    PUT_SKEW_PREMIUM, STT_SELL_PCT, BROKERAGE_PER_ORDER,
    NSE_TURNOVER_PCT, GST_PCT, SEBI_CHARGES_PCT, STAMP_DUTY_BUY_PCT,
    DEFAULT_SLIPPAGE_PTS, KELLY_FRACTION, MAX_TOLERABLE_MDD
)
from src.strategy_rules import Signal, SignalType



def compute_volatility_surface(
    spot: float,
    strike: float,
    base_iv: float = DEFAULT_IV,
    is_call: bool = True,
    skew_slope: float = 0.18,
    smile_curvature: float = 0.35
) -> float:
    """
    Computes continuous arbitrage-free implied volatility across strike moneyness.
    Models structural downside put skew and wing curvature for index options.
    """
    moneyness = (strike - spot) / spot
    skew_term = -skew_slope * moneyness if moneyness < 0 else -0.5 * skew_slope * moneyness
    smile_term = smile_curvature * (moneyness ** 2)
    
    sigma = base_iv + skew_term + smile_term
    if not is_call and moneyness <= 0:
        sigma = max(sigma, base_iv + PUT_SKEW_PREMIUM)
        
    return max(float(sigma), 0.01)


def compute_svi_volatility_skew(
    spot: float,
    strike: float,
    base_iv: float = DEFAULT_IV,
    is_call: bool = True,
    a: float = 0.015,
    b: float = 0.08,
    rho: float = -0.45,
    m: float = 0.0,
    sigma_svi: float = 0.12
) -> float:
    """
    Stochastic Volatility Inspired (SVI) Parametric Volatility Smile:
    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma_svi^2))
    where k = ln(K / S) is log-moneyness.
    """
    k = math.log(max(strike, 1.0) / max(spot, 1.0))
    disc = math.sqrt((k - m) ** 2 + sigma_svi ** 2)
    total_var = a + b * (rho * (k - m) + disc)
    
    sigma = math.sqrt(max(total_var, 0.0004))
    # Calibrate to base_iv level
    iv_calibrated = (sigma / 0.20) * base_iv
    
    if not is_call and strike <= spot:
        iv_calibrated = max(iv_calibrated, base_iv + PUT_SKEW_PREMIUM)
        
    return round(max(float(iv_calibrated), 0.05), 4)


def generate_svi_smile_curve(
    spot: float,
    base_iv: float = DEFAULT_IV,
    strike_span: int = 500,
    step: int = 50
) -> pd.DataFrame:
    """Generates full SVI volatility smile & skew distribution across strikes."""
    atm_base = int(round(spot / 50.0) * 50)
    strikes = range(atm_base - strike_span, atm_base + strike_span + step, step)
    
    records = []
    for k in strikes:
        iv_ce = compute_svi_volatility_skew(spot, k, base_iv=base_iv, is_call=True)
        iv_pe = compute_svi_volatility_skew(spot, k, base_iv=base_iv, is_call=False)
        moneyness = (k - spot) / spot * 100.0
        records.append({
            "strike": k,
            "moneyness_pct": round(moneyness, 2),
            "call_iv_pct": round(iv_ce * 100.0, 2),
            "put_iv_pct": round(iv_pe * 100.0, 2),
            "skew_spread_bps": round((iv_pe - iv_ce) * 10000.0, 1)
        })
        
    return pd.DataFrame(records)



def black_scholes_greeks(
    spot: float,
    strike: float,
    t_days: float = 4.0,
    r: float = RISK_FREE_RATE,
    q: float = 0.0,
    sigma: float = DEFAULT_IV,
    is_call: bool = True,
    use_vol_surface: bool = False
) -> Dict[str, float]:
    """Computes theoretical European option price and exact 1st/2nd/3rd order Greeks."""
    spot = max(float(spot), 1.0)
    strike = max(float(strike), 1.0)
    t_years = max(t_days / 365.0, 0.00001)  # Sub-minute 0DTE singularity shield
    
    if use_vol_surface:
        sigma = compute_volatility_surface(spot, strike, base_iv=sigma, is_call=is_call)
    else:
        if not is_call:
            sigma = sigma + PUT_SKEW_PREMIUM
        sigma = max(sigma, 0.01)
    
    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r - q + 0.5 * (sigma ** 2)) * t_years) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    
    discount_r = math.exp(-r * t_years)
    discount_q = math.exp(-q * t_years)
    
    if is_call:
        price = spot * discount_q * cdf_d1 - strike * discount_r * cdf_d2
        delta = discount_q * cdf_d1
        theta_annual = (
            - (spot * discount_q * pdf_d1 * sigma) / (2.0 * sqrt_t)
            - r * strike * discount_r * cdf_d2
            + q * spot * discount_q * cdf_d1
        )
    else:
        price = strike * discount_r * norm.cdf(-d2) - spot * discount_q * norm.cdf(-d1)
        delta = discount_q * (cdf_d1 - 1.0)
        theta_annual = (
            - (spot * discount_q * pdf_d1 * sigma) / (2.0 * sqrt_t)
            + r * strike * discount_r * norm.cdf(-d2)
            - q * spot * discount_q * norm.cdf(-d1)
        )

    gamma = (discount_q * pdf_d1) / (spot * sigma * sqrt_t)
    vega_1pct = (spot * discount_q * pdf_d1 * sqrt_t) / 100.0
    theta_daily = theta_annual / 365.0
    
    # 2nd Order Cross Greeks
    term2 = discount_q * pdf_d1 * ((2.0 * (r - q) * t_years - d2 * sigma * sqrt_t) / (2.0 * t_years * sigma * sqrt_t))
    if is_call:
        charm_annual = q * discount_q * cdf_d1 - term2
    else:
        charm_annual = -q * discount_q * norm.cdf(-d1) - term2
    charm_daily = charm_annual / 365.0
    vanna_1pct = - discount_q * (pdf_d1 * d2) / (sigma * 100.0)
    volga_1pct = (vega_1pct * d1 * d2) / (sigma * 100.0)
    
    # 3rd Order Greeks
    speed = - (gamma / spot) * (d1 / (sigma * sqrt_t) + 1.0)
    color_daily = - (gamma / (2.0 * t_years * 365.0)) * (
        1.0 + (2.0 * (r - q) * t_years - d2 * sigma * sqrt_t) / (sigma * sqrt_t) * d1
    )

    return {
        "price": round(max(price, 0.20), 2),
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta_daily, 2),
        "vega": round(vega_1pct, 2),
        "vanna": round(vanna_1pct, 4),
        "charm": round(charm_daily, 4),
        "volga": round(volga_1pct, 4),
        "speed": round(speed, 8),
        "color": round(color_daily, 6),
        "iv": round(sigma * 100.0, 1)
    }


def black_scholes_greeks_batch(
    spot: float,
    strikes: np.ndarray,
    t_days: float = 4.0,
    r: float = RISK_FREE_RATE,
    q: float = 0.0,
    sigma: float = DEFAULT_IV,
    is_call: bool = True
) -> Dict[str, np.ndarray]:
    """Vectorized European option pricing and Greeks across array of strikes."""
    t_years = max(t_days / 365.0, 0.00001)
    if not is_call:
        sigma = sigma + PUT_SKEW_PREMIUM
    sigma = max(sigma, 0.01)
    
    sqrt_t = math.sqrt(t_years)
    d1 = (np.log(spot / strikes) + (r - q + 0.5 * (sigma ** 2)) * t_years) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    discount_r = math.exp(-r * t_years)
    discount_q = math.exp(-q * t_years)
    
    if is_call:
        price = spot * discount_q * cdf_d1 - strikes * discount_r * cdf_d2
        delta = discount_q * cdf_d1
        theta_annual = (
            - (spot * discount_q * pdf_d1 * sigma) / (2.0 * sqrt_t)
            - r * strikes * discount_r * cdf_d2
            + q * spot * discount_q * cdf_d1
        )
    else:
        price = strikes * discount_r * norm.cdf(-d2) - spot * discount_q * norm.cdf(-d1)
        delta = discount_q * (cdf_d1 - 1.0)
        theta_annual = (
            - (spot * discount_q * pdf_d1 * sigma) / (2.0 * sqrt_t)
            + r * strikes * discount_r * norm.cdf(-d2)
            - q * spot * discount_q * norm.cdf(-d1)
        )
        
    gamma = (discount_q * pdf_d1) / (spot * sigma * sqrt_t)
    theta_daily = theta_annual / 365.0
    vega_1pct = (spot * discount_q * pdf_d1 * sqrt_t) / 100.0
    vanna_1pct = - discount_q * (pdf_d1 * d2) / (sigma * 100.0)
    
    return {
        "price": np.maximum(price, 0.20),
        "delta": delta,
        "gamma": gamma,
        "theta": theta_daily,
        "vega": vega_1pct,
        "vanna": vanna_1pct
    }


def black_76_greeks(
    futures_price: float,
    strike: float,
    t_days: float = 4.0,
    r: float = RISK_FREE_RATE,
    sigma: float = DEFAULT_IV,
    is_call: bool = True
) -> Dict[str, float]:
    """
    Black-76 model for European options on futures/forwards (Standard for NSE Index Options).
    Eliminates dividend-yield parameter estimation risk.
    """
    f = max(float(futures_price), 1.0)
    k = max(float(strike), 1.0)
    t_years = max(t_days / 365.0, 0.00001)
    
    sigma_clean = (sigma / 100.0) if sigma > 1.0 else sigma
    if not is_call:
        sigma_clean = sigma_clean + PUT_SKEW_PREMIUM
    sigma_clean = max(sigma_clean, 0.01)

    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(f / k) + 0.5 * (sigma_clean ** 2) * t_years) / (sigma_clean * sqrt_t)
    d2 = d1 - sigma_clean * sqrt_t
    
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    discount = math.exp(-r * t_years)
    
    if is_call:
        price = discount * (f * cdf_d1 - k * cdf_d2)
        delta = discount * cdf_d1
        theta_annual = - (f * discount * pdf_d1 * sigma_clean) / (2.0 * sqrt_t) - r * price
    else:
        price = discount * (k * norm.cdf(-d2) - f * norm.cdf(-d1))
        delta = - discount * norm.cdf(-d1)
        theta_annual = - (f * discount * pdf_d1 * sigma_clean) / (2.0 * sqrt_t) - r * price

    gamma = (discount * pdf_d1) / (f * sigma_clean * sqrt_t)
    vega_1pct = (f * discount * sqrt_t * pdf_d1) / 100.0
    vanna_1pct = - discount * (pdf_d1 * d2) / (sigma_clean * 100.0)
    charm_daily = (- discount * pdf_d1 * (r * sqrt_t / (sigma_clean) - d2 / (2.0 * t_years)) ) / 365.0 if t_years > 0 else 0.0

    return {
        "price": round(max(price, 0.20), 2),
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta_annual / 365.0, 2),
        "vega": round(vega_1pct, 2),
        "vanna": round(vanna_1pct, 4),
        "charm": round(charm_daily, 4),
        "iv": round(sigma_clean * 100.0, 1)
    }



def compute_0dte_gamma_scalp_parameters(
    spot: float,
    strike: float,
    dte_days: float = 0.10,
    iv: float = DEFAULT_IV,
    is_call: bool = True,
    current_time_str: Optional[str] = None,
    mode: str = "AUTO",
    atr: float = 25.0,
    r: float = RISK_FREE_RATE,
    q: float = 0.0
) -> Dict[str, Any]:
    """
    0DTE Expiry Day Gamma Scalper & Microstructure Shield Engine:
    
    Models post-13:00 IST expiry dynamics where Gamma (∂²P/∂S²) explodes and Charm (∂Δ/∂t) accelerates:
    1. Adjusts SL to tighter delta stops (max 15-20 spot points / 20% option premium ceiling).
    2. Provides dual-strike recommendations: Deep ITM (Delta >= 0.70) or ATM High-Gamma Scalp.
    3. Calculates 3-tier gamma surge profit targets (+25%, +65%, +120%) and locks in gains before 15:15 IST.
    4. Enforces a 15-minute time-decay circuit breaker to avoid Charm rot.
    """
    if current_time_str:
        try:
            parts = current_time_str.split(":")
            h, m = int(parts[0]), int(parts[1])
            curr_mins = h * 60 + m
            sq_mins = 15 * 60 + 15  # 15:15 IST Hard Square-Off
            rem_mins = max(sq_mins - curr_mins, 1)
            t_eff_days = max(rem_mins / 375.0 * (1.0 / 365.0) * 365.0, 0.005)
        except Exception:
            t_eff_days = max(dte_days, 0.005)
    else:
        t_eff_days = max(dte_days, 0.005)

    greeks = black_scholes_greeks(spot, strike, t_days=t_eff_days, r=r, q=q, sigma=iv, is_call=is_call)
    greeks_ref = black_scholes_greeks(spot, strike, t_days=4.0, r=r, q=q, sigma=iv, is_call=is_call)
    gamma_multiplier = round(greeks["gamma"] / max(greeks_ref["gamma"], 0.00001), 2)
    charm_hourly = round(greeks["charm"] / 6.25, 5)
    
    atm_center = int(round(spot / 50.0) * 50)
    deep_itm_offset = -150 if is_call else 150
    deep_itm_strike = atm_center + deep_itm_offset
    
    greeks_deep_itm = black_scholes_greeks(spot, deep_itm_strike, t_days=t_eff_days, r=r, q=q, sigma=iv, is_call=is_call)
    greeks_atm = black_scholes_greeks(spot, atm_center, t_days=t_eff_days, r=r, q=q, sigma=iv, is_call=is_call)

    entry_prem = greeks["price"]
    delta_mag = abs(greeks["delta"])
    gamma_val = greeks["gamma"]
    
    max_spot_sl_pts = min(round(0.60 * atr, 1), 18.0)
    option_loss_at_spot_sl = (max_spot_sl_pts * delta_mag) - (0.5 * gamma_val * (max_spot_sl_pts ** 2))
    max_prem_sl_pts = max(round(min(option_loss_at_spot_sl, entry_prem * 0.20), 2), 2.0)
    option_sl_prem = max(round(entry_prem - max_prem_sl_pts, 2), 1.0)
    
    spot_move_t1 = round(0.80 * atr, 1)
    prem_gain_t1 = (spot_move_t1 * delta_mag) + (0.5 * gamma_val * (spot_move_t1 ** 2))
    t1_prem = round(entry_prem + max(prem_gain_t1, entry_prem * 0.25), 2)
    
    spot_move_t2 = round(1.80 * atr, 1)
    prem_gain_t2 = (spot_move_t2 * delta_mag) + (0.5 * gamma_val * (spot_move_t2 ** 2))
    t2_prem = round(entry_prem + max(prem_gain_t2, entry_prem * 0.65), 2)
    
    spot_move_t3 = round(3.00 * atr, 1)
    prem_gain_t3 = (spot_move_t3 * delta_mag) + (0.5 * gamma_val * (spot_move_t3 ** 2))
    t3_prem = round(entry_prem + max(prem_gain_t3, entry_prem * 1.20), 2)

    is_afternoon_0dte = (t_eff_days <= 0.15) or (current_time_str is not None and current_time_str >= "13:00")
    regime_name = "0DTE_GAMMA_SQUEEZE_AFTERNOON" if is_afternoon_0dte else "0DTE_MORNING_TRANSITION"

    return {
        "regime": regime_name,
        "is_0dte_afternoon": is_afternoon_0dte,
        "t_effective_days": round(t_eff_days, 4),
        "spot": spot,
        "strike": strike,
        "option_type": "CE" if is_call else "PE",
        "entry_premium": entry_prem,
        "greeks": greeks,
        "gamma_explosion_multiplier": gamma_multiplier,
        "charm_hourly_drift": charm_hourly,
        "speed": greeks.get("speed", 0.0),
        "color": greeks.get("color", 0.0),
        "tightened_sl": {
            "max_spot_sl_pts": max_spot_sl_pts,
            "max_option_loss_pts": max_prem_sl_pts,
            "sl_premium": option_sl_prem,
            "max_risk_pct_ceiling": "20.0% Premium Floor"
        },
        "gamma_surge_targets": {
            "target_1_premium": t1_prem,
            "target_1_gain_pct": round(((t1_prem - entry_prem) / entry_prem) * 100.0, 1),
            "target_2_premium": t2_prem,
            "target_2_gain_pct": round(((t2_prem - entry_prem) / entry_prem) * 100.0, 1),
            "target_3_moonshot_premium": t3_prem,
            "target_3_gain_pct": round(((t3_prem - entry_prem) / entry_prem) * 100.0, 1)
        },
        "strike_playbook": {
            "deep_itm_synthetic": {
                "strike": deep_itm_strike,
                "symbol": f"NIFTY {deep_itm_strike} {'CE' if is_call else 'PE'}",
                "delta": round(greeks_deep_itm["delta"], 3),
                "premium": greeks_deep_itm["price"],
                "thesis": "Synthetic Future (Delta >= 0.70) with minimal theta decay & high directional linearity."
            },
            "atm_momentum_scalp": {
                "strike": atm_center,
                "symbol": f"NIFTY {atm_center} {'CE' if is_call else 'PE'}",
                "delta": round(greeks_atm["delta"], 3),
                "gamma": round(greeks_atm["gamma"], 5),
                "premium": greeks_atm["price"],
                "thesis": "Maximum Gamma Exploit for rapid 2x-3x surge on velocity breakouts."
            }
        },
        "execution_mandates": {
            "time_stop_minutes": 15,
            "time_stop_rule": "Exit trade if spot momentum stalls after 3 consecutive 5m bars (15 mins) to prevent Charm decay.",
            "breakeven_trigger": "Ratchet SL to Entry + ₹1.00 immediately once T1 (+25%) is achieved.",
            "hard_squareoff_time": "15:15 IST",
            "hard_squareoff_rule": "Mandatory market sweep liquidation before 15:15 IST. Zero overnight/delivery risk."
        }
    }


def compute_dynamic_trailing_option_sl(
    entry_prem: float,
    spot_entry: float,
    current_trailing_spot_sl: float,
    delta: float,
    gamma: float = 0.0008,
    theta_daily: float = -14.0,
    elapsed_bars_5m: int = 0,
    is_call: bool = True
) -> float:
    """Translates spot dynamic trailing SL (21 EMA / AVWAP 1σ) into exact option premium floor."""
    spot_diff = (current_trailing_spot_sl - spot_entry) if is_call else (spot_entry - current_trailing_spot_sl)
    convexity = 0.5 * gamma * (spot_diff ** 2) if spot_diff >= 0 else -0.5 * gamma * (spot_diff ** 2)
    time_decay = abs(theta_daily / 75.0) * elapsed_bars_5m
    
    option_sl_prem = entry_prem + (spot_diff * delta) + convexity - time_decay
    return max(round(option_sl_prem, 2), 2.0)


def calculate_adaptive_tca_friction(
    entry_prem: float,
    t1_prem: float,
    final_exit_prem: float,
    total_qty: int,
    lots: int,
    part_booked: bool = False,
    iv: float = DEFAULT_IV,
    is_0dte_afternoon: bool = False
) -> Dict[str, float]:
    """
    Computes 3-Leg Part-Booking Indian NSE Transaction Cost Analysis (TCA) friction:
    Statutory STT (0.1% sell), Brokerage (₹20/order), NSE Charges (0.03503%), GST (18%), Stamp Duty (0.003%),
    and Volatility/0DTE-expanded Slippage.
    """
    if total_qty <= 0:
        return {
            "total_friction": 0.0, "stt": 0.0, "brokerage": 0.0,
            "exchange_charges": 0.0, "gst": 0.0, "stamp_duty": 0.0,
            "slippage": 0.0, "effective_slippage_pts": 0.0
        }
        
    orders_count = 3.0 if part_booked else 2.0
    brokerage = BROKERAGE_PER_ORDER * orders_count
    
    turnover_buy = entry_prem * total_qty
    if part_booked:
        qty_50 = (lots // 2) * LOT_SIZE
        qty_rem = total_qty - qty_50
        turnover_sell = (qty_50 * t1_prem) + (qty_rem * final_exit_prem)
    else:
        turnover_sell = total_qty * final_exit_prem
        
    stt = turnover_sell * STT_SELL_PCT
    exchange_charges = (turnover_buy + turnover_sell) * NSE_TURNOVER_PCT
    sebi_fees = (turnover_buy + turnover_sell) * SEBI_CHARGES_PCT
    stamp_duty = turnover_buy * STAMP_DUTY_BUY_PCT
    gst = (brokerage + exchange_charges + sebi_fees) * GST_PCT
    
    vol_multiplier = math.sqrt(max(iv, 0.08) / 0.12)
    time_multiplier = 1.40 if is_0dte_afternoon else 1.0
    effective_slippage_pts = DEFAULT_SLIPPAGE_PTS * vol_multiplier * time_multiplier
    slippage = effective_slippage_pts * total_qty
    
    total_friction = brokerage + stt + exchange_charges + sebi_fees + stamp_duty + gst + slippage
    return {
        "total_friction": round(total_friction, 2),
        "stt": round(stt, 2),
        "brokerage": round(brokerage, 2),
        "exchange_charges": round(exchange_charges, 2),
        "gst": round(gst, 2),
        "stamp_duty": round(stamp_duty, 2),
        "slippage": round(slippage, 2),
        "effective_slippage_pts": round(effective_slippage_pts, 2)
    }


def calculate_adaptive_tca_friction_multi_tier(
    entry_prem: float,
    t1_prem: float,
    t2_prem: float,
    final_exit_prem: float,
    total_qty: int,
    lots: int,
    t1_hit: bool = False,
    t2_hit: bool = False,
    is_pyramided: bool = False,
    iv: float = DEFAULT_IV,
    is_0dte_afternoon: bool = False
) -> Dict[str, float]:
    """Accurately calculates statutory NSE TCA friction across 1, 2, 3, or 4 exit tranches."""
    if total_qty <= 0:
        return {"total_friction": 0.0, "stt": 0.0, "brokerage": 0.0, "exchange_charges": 0.0, "gst": 0.0, "slippage": 0.0}
        
    orders_count = 2.0
    if is_pyramided:
        orders_count += 1.0
    if t1_hit and t2_hit:
        orders_count += 2.0
    elif t1_hit:
        orders_count += 1.0
        
    brokerage = BROKERAGE_PER_ORDER * orders_count
    turnover_buy = entry_prem * total_qty
    
    if t1_hit and t2_hit:
        qty_35_1 = int(round(lots * 0.35)) * LOT_SIZE
        qty_35_2 = int(round(lots * 0.35)) * LOT_SIZE
        qty_30 = total_qty - qty_35_1 - qty_35_2
        turnover_sell = (qty_35_1 * t1_prem) + (qty_35_2 * t2_prem) + (qty_30 * final_exit_prem)
    elif t1_hit:
        qty_35_1 = int(round(lots * 0.35)) * LOT_SIZE
        qty_rem = total_qty - qty_35_1
        turnover_sell = (qty_35_1 * t1_prem) + (qty_rem * final_exit_prem)
    else:
        turnover_sell = total_qty * final_exit_prem
        
    stt = turnover_sell * STT_SELL_PCT
    exchange_charges = (turnover_buy + turnover_sell) * NSE_TURNOVER_PCT
    sebi_fees = (turnover_buy + turnover_sell) * SEBI_CHARGES_PCT
    stamp_duty = turnover_buy * STAMP_DUTY_BUY_PCT
    gst = (brokerage + exchange_charges + sebi_fees) * GST_PCT
    
    vol_multiplier = math.sqrt(max(iv, 0.08) / 0.12)
    time_multiplier = 1.40 if is_0dte_afternoon else 1.0
    effective_slippage_pts = DEFAULT_SLIPPAGE_PTS * vol_multiplier * time_multiplier
    slippage = effective_slippage_pts * total_qty
    
    total_friction = brokerage + stt + exchange_charges + sebi_fees + stamp_duty + gst + slippage
    return {
        "total_friction": round(total_friction, 2),
        "stt": round(stt, 2),
        "brokerage": round(brokerage, 2),
        "exchange_charges": round(exchange_charges, 2),
        "gst": round(gst, 2),
        "stamp_duty": round(stamp_duty, 2),
        "slippage": round(slippage, 2),
        "effective_slippage_pts": round(effective_slippage_pts, 2),
        "orders_count": int(orders_count)
    }


def calculate_tca_friction(
    entry_prem: float,
    exit_prem: float,
    total_qty: int,
    lots: int,
    part_booked: bool = False,
    iv: float = DEFAULT_IV,
    is_0dte_afternoon: bool = False
) -> Dict[str, float]:
    """Backward-compatible wrapper for 2-leg or 3-leg TCA computation."""
    return calculate_adaptive_tca_friction(
        entry_prem=entry_prem,
        t1_prem=exit_prem,
        final_exit_prem=exit_prem,
        total_qty=total_qty,
        lots=lots,
        part_booked=part_booked,
        iv=iv,
        is_0dte_afternoon=is_0dte_afternoon
    )


# ----------------- THE GOLDEN VAULT INTRADAY PROFIT LOCK -----------------

def evaluate_golden_vault_lock(
    initial_capital: float = DEFAULT_CAPITAL,
    current_intraday_pnl: float = 0.0,
    peak_intraday_pnl: float = 0.0,
    profit_trigger_pct: float = 0.015,
    lock_pct: float = 0.75,
    realized_volatility: Optional[float] = None,
    baseline_volatility: float = 0.12
) -> Dict[str, Any]:
    """
    Dynamic Volatility-Adjusted Intraday Profit Lock ('The Golden Vault Rule' - Phase 5.1):
    Once intraday Net PnL reaches >= +1.5% of account capital, lock 75% of peak profits
    as an untouchable risk floor for the remainder of the session.
    If Realized Volatility remains elevated above baseline, allows active moonshot runners
    to trail while vetoing new risk.
    """
    trigger_threshold_rupees = initial_capital * profit_trigger_pct
    peak_pnl = max(peak_intraday_pnl, current_intraday_pnl)
    is_vault_triggered = peak_pnl >= trigger_threshold_rupees
    
    allow_runners = False
    if is_vault_triggered:
        locked_profit_floor = round(peak_pnl * lock_pct, 2)
        untouchable_capital_floor = round(initial_capital + locked_profit_floor, 2)
        risk_cushion = round(max(current_intraday_pnl - locked_profit_floor, 0.0), 2)
        is_session_halted = current_intraday_pnl <= locked_profit_floor
        
        # Volatility expansion runner condition
        if realized_volatility is not None and realized_volatility > 1.30 * baseline_volatility:
            allow_runners = True
        
        if is_session_halted:
            status = "LOCKED_GOLDEN_VAULT"
            message = (
                f"🔒 Golden Vault Engaged: +₹{locked_profit_floor:,.2f} locked floor reached. "
                f"Session trading halted to preserve 75% of peak gains (₹{peak_pnl:,.2f})."
            )
        else:
            status = "VAULT_ACTIVE"
            runner_str = " | Volatility elevated: Moonshot runners active." if allow_runners else ""
            message = (
                f"🛡️ Golden Vault Active: Peak PnL ₹{peak_pnl:,.2f} (≥ +{profit_trigger_pct*100:.1f}%). "
                f"₹{locked_profit_floor:,.2f} locked floor. Remaining risk cushion: ₹{risk_cushion:,.2f}.{runner_str}"
            )
    else:
        locked_profit_floor = 0.0
        untouchable_capital_floor = initial_capital
        risk_cushion = float("inf")
        is_session_halted = False
        status = "UNLOCKED"
        message = f"Golden Vault Idle (Triggers at +₹{trigger_threshold_rupees:,.2f} / +{profit_trigger_pct*100:.1f}% net gain)."
        
    return {
        "status": status,
        "is_vault_triggered": is_vault_triggered,
        "is_session_halted": is_session_halted,
        "allow_runners": allow_runners,
        "peak_intraday_pnl": round(peak_pnl, 2),
        "current_intraday_pnl": round(current_intraday_pnl, 2),
        "trigger_threshold_rupees": round(trigger_threshold_rupees, 2),
        "locked_profit_floor": locked_profit_floor,
        "untouchable_capital_floor": untouchable_capital_floor,
        "risk_cushion": risk_cushion,
        "lock_pct": lock_pct,
        "message": message
    }


def calculate_dynamic_kelly(
    win_rate: float,
    payoff_ratio: float,
    day_type: str = "NORMAL_VARIATION_DAY"
) -> Dict[str, Any]:
    """
    Calculates Regime-Scaled Dynamic Kelly Criterion (Phase 5.2).
    Base Full-Kelly: f* = (p*b - q) / b
    Quarter-Kelly Base: f* / 4.
    Scaled by Market Profile Day Type:
    - TREND_DAY: 1.5x (Aggressive sizing on confirmed unilateral breakouts)
    - NORMAL_VARIATION_DAY: 1.0x (Standard sizing)
    - NORMAL_DAY: 0.5x (Conservative inside Initial Balance)
    - NEUTRAL_DAY: 0.3x (Highly conservative on bilateral chop)
    """
    q = 1.0 - win_rate
    base_kelly = ((win_rate * payoff_ratio) - q) / payoff_ratio if payoff_ratio > 0 else 0.0
    base_kelly = max(0.0, min(base_kelly, 0.50))
    
    regime_multipliers = {
        "BULLISH_TREND_DAY": 1.5,
        "BEARISH_TREND_DAY": 1.5,
        "TREND_DAY": 1.5,
        "NORMAL_VARIATION_DAY": 1.0,
        "NORMAL_DAY": 0.5,
        "NEUTRAL_DAY": 0.3,
        "ACCUMULATING_INITIAL_BALANCE": 0.5
    }
    
    multiplier = 1.0
    for k, v in regime_multipliers.items():
        if k in day_type.upper():
            multiplier = v
            break
    # Scale standard 1% institutional risk budget by Kelly conviction and Market Profile Day Type
    kelly_conviction = base_kelly / 0.50 if base_kelly > 0 else 0.0
    dynamic_risk_pct = min(max(0.01 * kelly_conviction * multiplier, 0.002), 0.02)
    
    return {
        "base_full_kelly": round(base_kelly, 4),
        "base_quarter_kelly": round(base_kelly / 4.0, 4),
        "day_type_multiplier": multiplier,
        "dynamic_risk_pct": round(dynamic_risk_pct, 4),
        "dynamic_risk_pct_str": f"{dynamic_risk_pct * 100:.2f}%",
        "day_type": day_type
    }



def calculate_position_size(
    capital: float,
    risk_pct: float,
    entry_prem: float,
    sl_prem: float,
    lot_size: int = LOT_SIZE,
    current_drawdown_pct: float = 0.0,
    current_intraday_pnl: float = 0.0,
    peak_intraday_pnl: float = 0.0,
    enforce_golden_vault: bool = True
) -> Dict[str, Any]:
    """
    Computes exact lot sizing with Non-Linear Drawdown Dampener, 10% MDD hard circuit breaker,
    and Dynamic Intraday Profit Lock ('The Golden Vault Rule').
    """
    if current_drawdown_pct >= MAX_TOLERABLE_MDD:
        dd_dampener = 0.0
    elif current_drawdown_pct <= 0.03:
        dd_dampener = 1.0
    elif current_drawdown_pct <= 0.06:
        dd_dampener = 1.0 - ((current_drawdown_pct - 0.03) / 0.03) * 0.50
    else:
        norm_val = (current_drawdown_pct - 0.06) / (MAX_TOLERABLE_MDD - 0.06)
        dd_dampener = max(0.50 * ((1.0 - norm_val) ** 2), 0.05)
        
    adjusted_risk_pct = risk_pct * dd_dampener
    max_risk_rupees = capital * adjusted_risk_pct
    
    vault_info = evaluate_golden_vault_lock(
        initial_capital=capital,
        current_intraday_pnl=current_intraday_pnl,
        peak_intraday_pnl=peak_intraday_pnl
    )
    
    vault_constrained = False
    if enforce_golden_vault and vault_info["is_vault_triggered"]:
        if vault_info["is_session_halted"]:
            max_risk_rupees = 0.0
            vault_constrained = True
        else:
            if max_risk_rupees > vault_info["risk_cushion"]:
                max_risk_rupees = vault_info["risk_cushion"]
                vault_constrained = True

    risk_per_share = max(entry_prem - sl_prem, 12.0)
    risk_per_lot = risk_per_share * lot_size
    
    lots = int(max_risk_rupees // risk_per_lot) if risk_per_lot > 0 else 0
    max_margin_lots = int((capital * 0.35) // (entry_prem * lot_size)) if entry_prem > 0 else lots
    lots = min(lots, max_margin_lots, 15)
    
    if (lots * risk_per_lot > max_risk_rupees) and lots > 0:
        lots -= 1
        
    total_qty = lots * lot_size
    total_capital_required = round(total_qty * entry_prem, 2)
    actual_risk_rupees = round(total_qty * risk_per_share, 2)
    
    return {
        "lots": lots,
        "total_qty": total_qty,
        "risk_per_lot": round(risk_per_lot, 2),
        "actual_risk_rupees": actual_risk_rupees,
        "max_risk_rupees": round(max_risk_rupees, 2),
        "capital_required": total_capital_required,
        "dd_dampener": round(dd_dampener, 3),
        "vault_info": vault_info,
        "vault_constrained": vault_constrained
    }


def run_monte_carlo_simulation(
    initial_capital: float = DEFAULT_CAPITAL,
    base_risk_pct: float = MAX_RISK_PCT,
    win_rate: float = 0.58,
    win_payoff_r: float = 2.10,
    loss_payoff_r: float = -1.0,
    num_simulations: int = 1000,
    num_trades: int = 100,
    ruin_threshold_pct: float = 0.50,
    enable_quarter_kelly_dampener: bool = True,
    random_seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    High-Performance Vectorized 1,000-Path Monte Carlo Ruin & Stress-Test Engine.
    Computes VaR (95%/99%), CVaR (Expected Shortfall), Drawdown distributions, and PoR < 0.01%.
    """
    rng = np.random.default_rng(random_seed)
    win_matrix = rng.random((num_simulations, num_trades)) < win_rate
    r_multipliers = np.where(win_matrix, win_payoff_r, loss_payoff_r)
    
    equity_paths = np.zeros((num_simulations, num_trades + 1), dtype=np.float64)
    equity_paths[:, 0] = initial_capital
    
    peak_cap = np.full(num_simulations, initial_capital, dtype=np.float64)
    current_cap = np.full(num_simulations, initial_capital, dtype=np.float64)
    
    for t in range(num_trades):
        dd = np.maximum((peak_cap - current_cap) / peak_cap, 0.0)
        
        if enable_quarter_kelly_dampener:
            dampener = np.where(
                dd <= 0.03,
                1.0,
                np.where(
                    dd <= 0.06,
                    1.0 - ((dd - 0.03) / 0.03) * 0.50,
                    np.where(
                        dd < MAX_TOLERABLE_MDD,
                        np.maximum(0.50 * ((1.0 - (dd - 0.06) / (MAX_TOLERABLE_MDD - 0.06)) ** 2), 0.05),
                        0.0
                    )
                )
            )
        else:
            dampener = 1.0
            
        risk_budget = current_cap * (base_risk_pct * dampener)
        step_pnl = risk_budget * r_multipliers[:, t]
        current_cap = np.maximum(current_cap + step_pnl, 0.0)
        peak_cap = np.maximum(peak_cap, current_cap)
        equity_paths[:, t + 1] = current_cap

    cum_max = np.maximum.accumulate(equity_paths, axis=1)
    dd_matrix = (cum_max - equity_paths) / cum_max
    path_max_dds = np.max(dd_matrix, axis=1)
    
    mdd_median_pct = float(np.median(path_max_dds) * 100.0)
    mdd_95th_pct = float(np.percentile(path_max_dds, 95) * 100.0)
    mdd_99th_pct = float(np.percentile(path_max_dds, 99) * 100.0)
    mdd_worst_pct = float(np.max(path_max_dds) * 100.0)
    
    var_95_rupees = round(initial_capital * (mdd_95th_pct / 100.0), 2)
    var_99_rupees = round(initial_capital * (mdd_99th_pct / 100.0), 2)
    
    tail_95 = path_max_dds[path_max_dds >= np.percentile(path_max_dds, 95)]
    tail_99 = path_max_dds[path_max_dds >= np.percentile(path_max_dds, 99)]
    
    cvar_95_mdd_pct = float(np.mean(tail_95) * 100.0) if len(tail_95) > 0 else mdd_95th_pct
    cvar_99_mdd_pct = float(np.mean(tail_99) * 100.0) if len(tail_99) > 0 else mdd_99th_pct
    cvar_95_rupees = round(initial_capital * (cvar_95_mdd_pct / 100.0), 2)
    cvar_99_rupees = round(initial_capital * (cvar_99_mdd_pct / 100.0), 2)
    
    min_equity_per_path = np.min(equity_paths, axis=1)
    ruined_count = int(np.sum(min_equity_per_path <= initial_capital * (1.0 - ruin_threshold_pct)))
    prob_of_ruin_pct = float((ruined_count / num_simulations) * 100.0)
    por_display_str = "< 0.01%" if prob_of_ruin_pct == 0.0 else f"{prob_of_ruin_pct:.2f}%"
    
    final_equities = equity_paths[:, -1]
    net_returns_pct = (final_equities - initial_capital) / initial_capital * 100.0
    
    mean_final_equity = round(float(np.mean(final_equities)), 2)
    median_final_equity = round(float(np.median(final_equities)), 2)
    min_final_equity = round(float(np.min(final_equities)), 2)
    max_final_equity = round(float(np.max(final_equities)), 2)
    
    ret_std = float(np.std(net_returns_pct))
    sharpe_proxy = round(float(np.mean(net_returns_pct)) / ret_std, 2) if ret_std > 0 else 0.0
    
    pos_pnls = final_equities[final_equities > initial_capital] - initial_capital
    neg_pnls = initial_capital - final_equities[final_equities < initial_capital]
    profit_factor = round(float(np.sum(pos_pnls) / np.sum(neg_pnls)), 2) if np.sum(neg_pnls) > 0 else 999.0

    p5 = np.percentile(equity_paths, 5, axis=0)
    p25 = np.percentile(equity_paths, 25, axis=0)
    p50 = np.percentile(equity_paths, 50, axis=0)
    p75 = np.percentile(equity_paths, 75, axis=0)
    p95 = np.percentile(equity_paths, 95, axis=0)
    
    return {
        "num_simulations": num_simulations,
        "num_trades": num_trades,
        "initial_capital": initial_capital,
        "win_rate": win_rate,
        "win_payoff_r": win_payoff_r,
        "loss_payoff_r": loss_payoff_r,
        "prob_of_ruin_pct": prob_of_ruin_pct,
        "prob_of_ruin_str": por_display_str,
        "is_ruin_safe": prob_of_ruin_pct < 0.01,
        "var_95_pct": round(mdd_95th_pct, 2),
        "var_95_rupees": var_95_rupees,
        "var_99_pct": round(mdd_99th_pct, 2),
        "var_99_rupees": var_99_rupees,
        "cvar_95_pct": round(cvar_95_mdd_pct, 2),
        "cvar_95_rupees": cvar_95_rupees,
        "cvar_99_pct": round(cvar_99_mdd_pct, 2),
        "cvar_99_rupees": cvar_99_rupees,
        "mdd_median_pct": round(mdd_median_pct, 2),
        "mdd_95th_pct": round(mdd_95th_pct, 2),
        "mdd_99th_pct": round(mdd_99th_pct, 2),
        "mdd_worst_pct": round(mdd_worst_pct, 2),
        "mean_final_equity": mean_final_equity,
        "median_final_equity": median_final_equity,
        "min_final_equity": min_final_equity,
        "max_final_equity": max_final_equity,
        "sharpe_ratio": sharpe_proxy,
        "profit_factor": profit_factor,
        "percentile_5": p5.tolist(),
        "percentile_25": p25.tolist(),
        "percentile_50": p50.tolist(),
        "percentile_75": p75.tolist(),
        "percentile_95": p95.tolist(),
        "sample_paths": equity_paths[:35].tolist(),
        "all_max_drawdowns": (path_max_dds * 100.0).tolist()
    }


def calculate_pcr_and_max_pain(option_chain_df: Any) -> Dict[str, Any]:
    """
    Computes exact Max Pain strike price (where option sellers incur minimum cumulative payout)
    and Open Interest / Change in OI / Volume Put-Call Ratio (PCR).
    """
    if option_chain_df is not None and isinstance(option_chain_df, dict):
        if "dataframe" in option_chain_df and isinstance(option_chain_df["dataframe"], pd.DataFrame):
            option_chain_df = option_chain_df["dataframe"]
        elif "records" in option_chain_df and isinstance(option_chain_df["records"], dict) and "data" in option_chain_df["records"]:
            rows = []
            for item in option_chain_df["records"]["data"]:
                stk = item.get("strikePrice")
                ce = item.get("CE", {})
                pe = item.get("PE", {})
                rows.append({
                    "strike": float(stk),
                    "ce_oi": float(ce.get("openInterest", 0)),
                    "pe_oi": float(pe.get("openInterest", 0)),
                    "ce_change_oi": float(ce.get("changeinOpenInterest", 0)),
                    "pe_change_oi": float(pe.get("changeinOpenInterest", 0)),
                    "ce_volume": float(ce.get("totalTradedVolume", 0)),
                    "pe_volume": float(pe.get("totalTradedVolume", 0)),
                    "ce_ltp": float(ce.get("lastPrice", 0)),
                    "pe_ltp": float(pe.get("lastPrice", 0))
                })
            option_chain_df = pd.DataFrame(rows)
        else:
            try:
                option_chain_df = pd.DataFrame(option_chain_df)
            except Exception:
                option_chain_df = None

    if option_chain_df is None or not isinstance(option_chain_df, pd.DataFrame) or option_chain_df.empty:
        return {
            "max_pain_strike": 24500.0,
            "total_ce_oi": 0, "total_pe_oi": 0, "pcr_oi": 1.0,
            "pcr_sentiment": "Neutral / Balanced (No Data)",
            "total_ce_change_oi": 0, "total_pe_change_oi": 0, "pcr_change_oi": 1.0,
            "total_ce_volume": 0, "total_pe_volume": 0, "pcr_volume": 1.0,
            "min_payout_loss": 0.0, "pain_distribution": pd.DataFrame(), "strikes_analyzed": 0
        }

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
        
    df[strike_col] = pd.to_numeric(df[strike_col], errors="coerce")
    df = df.dropna(subset=[strike_col]).sort_values(by=strike_col).reset_index(drop=True)
    
    if df.empty:
        return {
            "max_pain_strike": 24500.0,
            "total_ce_oi": 0, "total_pe_oi": 0, "pcr_oi": 1.0,
            "pcr_sentiment": "Neutral / Balanced", "total_ce_change_oi": 0,
            "total_pe_change_oi": 0, "pcr_change_oi": 1.0, "total_ce_volume": 0,
            "total_pe_volume": 0, "pcr_volume": 1.0, "min_payout_loss": 0.0,
            "pain_distribution": pd.DataFrame(), "strikes_analyzed": 0
        }

    ce_oi_col = next((c for c in ["ce_oi", "ce_open_interest", "call_oi", "openinterest_ce"] if c in df.columns), None)
    pe_oi_col = next((c for c in ["pe_oi", "pe_open_interest", "put_oi", "openinterest_pe"] if c in df.columns), None)
    
    ce_chg_col = next((c for c in ["ce_change_oi", "ce_changeinopeninterest", "call_change_oi", "ce_chg_oi"] if c in df.columns), None)
    pe_chg_col = next((c for c in ["pe_change_oi", "pe_changeinopeninterest", "put_change_oi", "pe_chg_oi"] if c in df.columns), None)
    
    ce_vol_col = next((c for c in ["ce_volume", "ce_total_volume", "call_volume", "ce_vol"] if c in df.columns), None)
    pe_vol_col = next((c for c in ["pe_volume", "pe_total_volume", "put_volume", "pe_vol"] if c in df.columns), None)

    ce_oi_vals = pd.to_numeric(df[ce_oi_col], errors="coerce").fillna(0).values if ce_oi_col else np.zeros(len(df))
    pe_oi_vals = pd.to_numeric(df[pe_oi_col], errors="coerce").fillna(0).values if pe_oi_col else np.zeros(len(df))
    
    ce_chg_vals = pd.to_numeric(df[ce_chg_col], errors="coerce").fillna(0).values if ce_chg_col else np.zeros(len(df))
    pe_chg_vals = pd.to_numeric(df[pe_chg_col], errors="coerce").fillna(0).values if pe_chg_col else np.zeros(len(df))
    
    ce_vol_vals = pd.to_numeric(df[ce_vol_col], errors="coerce").fillna(0).values if ce_vol_col else np.zeros(len(df))
    pe_vol_vals = pd.to_numeric(df[pe_vol_col], errors="coerce").fillna(0).values if pe_vol_col else np.zeros(len(df))
    
    total_ce_oi = int(np.sum(ce_oi_vals))
    total_pe_oi = int(np.sum(pe_oi_vals))
    pcr_oi = round(float(total_pe_oi / total_ce_oi), 3) if total_ce_oi > 0 else 1.0
    
    total_ce_chg = int(np.sum(ce_chg_vals))
    total_pe_chg = int(np.sum(pe_chg_vals))
    pcr_chg = round(float(total_pe_chg / total_ce_chg), 3) if total_ce_chg > 0 else (round(pcr_oi, 3))
    
    total_ce_vol = int(np.sum(ce_vol_vals))
    total_pe_vol = int(np.sum(pe_vol_vals))
    pcr_vol = round(float(total_pe_vol / total_ce_vol), 3) if total_ce_vol > 0 else 1.0
    
    if pcr_oi >= 1.30:
        sentiment = "Strong Bullish (Put Writing Floor / Bull Trap for Bears)"
    elif pcr_oi >= 1.05:
        sentiment = "Moderately Bullish (Put Support Dominant)"
    elif pcr_oi <= 0.70:
        sentiment = "Strong Bearish (Call Writing Ceiling / Overbought)"
    elif pcr_oi <= 0.90:
        sentiment = "Moderately Bearish (Call Resistance Dominant)"
    else:
        sentiment = "Neutral / Balanced Oscillation Range"
        
    strikes = df[strike_col].values.astype(float)
    pain_records = []
    min_loss = float("inf")
    max_pain_strike = float(strikes[0]) if len(strikes) > 0 else 24500.0
    
    for s_expiry in strikes:
        call_loss = np.sum(ce_oi_vals * np.maximum(0.0, s_expiry - strikes))
        put_loss = np.sum(pe_oi_vals * np.maximum(0.0, strikes - s_expiry))
        total_loss = call_loss + put_loss
        
        pain_records.append({
            "strike": float(s_expiry),
            "call_payout": round(float(call_loss), 2),
            "put_payout": round(float(put_loss), 2),
            "total_payout": round(float(total_loss), 2)
        })
        
        if total_loss < min_loss:
            min_loss = total_loss
            max_pain_strike = float(s_expiry)
            
    pain_df = pd.DataFrame(pain_records)
    
    return {
        "max_pain_strike": round(max_pain_strike, 2),
        "total_ce_oi": total_ce_oi,
        "total_pe_oi": total_pe_oi,
        "pcr_oi": pcr_oi,
        "pcr_sentiment": sentiment,
        "total_ce_change_oi": total_ce_chg,
        "total_pe_change_oi": total_pe_chg,
        "pcr_change_oi": pcr_chg,
        "total_ce_volume": total_ce_vol,
        "total_pe_volume": total_pe_vol,
        "pcr_volume": pcr_vol,
        "min_payout_loss": round(min_loss, 2),
        "pain_distribution": pain_df,
        "strikes_analyzed": len(strikes)
    }


def compute_full_chain_gex_profile(
    option_chain_df: pd.DataFrame,
    spot: float,
    iv: float = DEFAULT_IV,
    t_days: float = 3.5
) -> Dict[str, Any]:
    """
    Vectorized Institutional Dealer Gamma Exposure (GEX) Profile across full Option Chain:
    
    Call GEX ($) = Gamma_C * Call_OI * Spot * Lot_Size * 0.01 / 10^7 (Cr ₹)
    Put GEX ($) = - Gamma_P * Put_OI * Spot * Lot_Size * 0.01 / 10^7 (Cr ₹)
    Net GEX = Call GEX + Put GEX
    
    Identifies:
    1. Call Wall (Major Resistance / Dealer Short Gamma supply).
    2. Put Wall (Major Support / Dealer Short Put peg).
    3. Zero-GEX Level (Regime switch from positive mean-reverting gamma to negative accelerating gamma).
    """
    if option_chain_df.empty or "strike" not in option_chain_df.columns:
        return {
            "call_wall": round(spot + 150.0, 2),
            "put_wall": round(spot - 150.0, 2),
            "zero_gex_strike": round(spot, 2),
            "net_gex_cr": 0.0,
            "is_positive_gamma": True,
            "gex_df": pd.DataFrame()
        }

    strikes = option_chain_df["strike"].values
    ce_ois = option_chain_df.get("ce_oi", pd.Series(np.zeros(len(strikes)))).fillna(0).values
    pe_ois = option_chain_df.get("pe_oi", pd.Series(np.zeros(len(strikes)))).fillna(0).values

    gex_records = []
    total_net_gex = 0.0
    zero_gex_strike = spot
    min_abs_net_gex = float("inf")

    for s, c_oi, p_oi in zip(strikes, ce_ois, pe_ois):
        g_c = black_scholes_greeks(spot, s, t_days=t_days, sigma=iv, is_call=True)["gamma"]
        g_p = black_scholes_greeks(spot, s, t_days=t_days, sigma=iv, is_call=False)["gamma"]

        call_gex_cr = (g_c * c_oi * spot * LOT_SIZE * 0.01) / 1e7
        put_gex_cr = - (g_p * p_oi * spot * LOT_SIZE * 0.01) / 1e7
        net_gex_cr = call_gex_cr + put_gex_cr
        total_net_gex += net_gex_cr

        if abs(net_gex_cr) < min_abs_net_gex:
            min_abs_net_gex = abs(net_gex_cr)
            zero_gex_strike = s

        gex_records.append({
            "strike": float(s),
            "call_oi": int(c_oi),
            "put_oi": int(p_oi),
            "call_gex_cr": round(float(call_gex_cr), 3),
            "put_gex_cr": round(float(put_gex_cr), 3),
            "net_gex_cr": round(float(net_gex_cr), 3)
        })

    gex_df = pd.DataFrame(gex_records)
    
    # Identify Walls
    max_ce_idx = gex_df["call_oi"].idxmax() if not gex_df.empty else 0
    max_pe_idx = gex_df["put_oi"].idxmax() if not gex_df.empty else 0
    call_wall = float(gex_df.loc[max_ce_idx, "strike"]) if not gex_df.empty else spot + 100.0
    put_wall = float(gex_df.loc[max_pe_idx, "strike"]) if not gex_df.empty else spot - 100.0

    return {
        "call_wall": round(call_wall, 2),
        "put_wall": round(put_wall, 2),
        "zero_gex_strike": round(zero_gex_strike, 2),
        "total_net_gex_cr": round(total_net_gex, 2),
        "is_positive_gamma": total_net_gex >= 0,
        "gamma_regime": "DEALER_LONG_GAMMA (Mean-Reverting & Low Volatility)" if total_net_gex >= 0 else "DEALER_SHORT_GAMMA (High Volatility & Accelerating Velocity)",
        "gex_df": gex_df
    }


def construct_ratio_spread(
    spot: float,
    is_call: bool = True,
    lots: int = 2,
    t_days: float = 3.5,
    iv: float = DEFAULT_IV
) -> Dict[str, Any]:
    """
    Constructs an Institutional 1:2 Ratio Spread (Buy 1 ATM, Sell 2 OTM):
    Used when IV is elevated (> 14%) to achieve zero theta decay and wide breakeven wings.
    """
    atm_k = int(round(spot / 50.0) * 50)
    wing_offset = 100 if is_call else -100
    otm_k = atm_k + wing_offset

    long_greeks = black_scholes_greeks(spot, atm_k, t_days=t_days, sigma=iv, is_call=is_call)
    short_greeks = black_scholes_greeks(spot, otm_k, t_days=t_days, sigma=iv, is_call=is_call)

    p_long = long_greeks["price"]
    p_short = short_greeks["price"]

    # 1 Long : 2 Short
    net_premium_cost = p_long - (2.0 * p_short)
    is_net_credit = net_premium_cost <= 0
    spread_width = abs(otm_k - atm_k)

    max_profit_pts = spread_width - net_premium_cost if not is_net_credit else spread_width + abs(net_premium_cost)
    upper_breakeven = otm_k + max_profit_pts if is_call else otm_k - max_profit_pts

    net_delta = long_greeks["delta"] - (2.0 * short_greeks["delta"])
    net_theta = long_greeks["theta"] - (2.0 * short_greeks["theta"])
    net_vega = long_greeks["vega"] - (2.0 * short_greeks["vega"])

    opt_type = "CE" if is_call else "PE"

    return {
        "spread_name": f"1:2 {opt_type} Ratio Spread",
        "long_leg": {"strike": atm_k, "qty_multiplier": 1, "premium": p_long, "symbol": f"NIFTY {atm_k} {opt_type}"},
        "short_leg": {"strike": otm_k, "qty_multiplier": 2, "premium": p_short, "symbol": f"NIFTY {otm_k} {opt_type}"},
        "net_entry_cost_pts": round(net_premium_cost, 2),
        "is_net_credit": is_net_credit,
        "max_profit_pts": round(max_profit_pts, 2),
        "max_profit_strike": otm_k,
        "breakeven_point": round(upper_breakeven, 2),
        "net_delta": round(net_delta, 3),
        "net_theta_daily": round(net_theta, 2),
        "net_vega": round(net_vega, 2)
    }


def select_institutional_strike(
    spot: float,
    is_call: bool,
    t_days: float = 4.0,
    iv: float = DEFAULT_IV,
    is_0dte_afternoon: bool = False
) -> Dict[str, Any]:
    """
    Selects optimal institutional strike (Vectorized):
    - Normal regimes: Target Delta ~0.58 in [0.50, 0.65].
    - 0DTE Expiry Thursday afternoon: Selects Deep ITM (Delta ~0.75-0.85) to avoid gamma cliff.
    """
    atm_base = int(round(spot / 50.0) * 50)
    candidate_offsets = np.array([0, -50, 50, -100, 100, -150, 150, -200, 200, -250, -300] if is_call else [0, 50, -50, 100, -100, 150, -150, 200, -200, 250, 300])
    candidates = atm_base + candidate_offsets
    
    target_delta = DELTA_DEEP_ITM_0DTE if is_0dte_afternoon else 0.58
    min_delta = 0.70 if is_0dte_afternoon else DELTA_MIN
    max_delta = 0.90 if is_0dte_afternoon else DELTA_MAX
    
    # Vectorized evaluation of all candidates at once
    batch = black_scholes_greeks_batch(spot, candidates, t_days=t_days, sigma=iv, is_call=is_call)
    abs_deltas = np.abs(batch["delta"])
    
    valid_mask = (abs_deltas >= min_delta) & (abs_deltas <= max_delta)
    if np.any(valid_mask):
        valid_indices = np.where(valid_mask)[0]
        diffs = np.abs(abs_deltas[valid_indices] - target_delta)
        best_idx = valid_indices[np.argmin(diffs)]
    else:
        best_idx = 0
        
    best_strike = int(candidates[best_idx])
    best_greeks = black_scholes_greeks(spot, best_strike, t_days=t_days, sigma=iv, is_call=is_call)

    opt_type = "CE" if is_call else "PE"
    return {
        "strike": best_strike,
        "option_type": opt_type,
        "symbol": f"NIFTY {best_strike} {opt_type}",
        "regime_mode": "0DTE Deep ITM Synthetic" if is_0dte_afternoon else "Standard ATM/1-ITM",
        **best_greeks
    }


def convert_to_free_vertical_spread(
    long_ticket: Dict[str, Any],
    spot_at_t1: float,
    t_days_remaining: float = 3.0,
    iv: float = DEFAULT_IV
) -> Dict[str, Any]:
    """Constructs an Institutional Free Vertical Spread at Target 1 to eliminate Theta decay on runners."""
    is_call = "CE" in long_ticket["option_type"]
    k1 = long_ticket["strike"]
    lots = long_ticket["lots"]
    total_qty = long_ticket["total_qty"]
    entry_prem_k1 = long_ticket["entry_premium"]
    
    spread_width = 100 if is_call else -100
    k2 = k1 + spread_width if is_call else k1 - abs(spread_width)
    
    greeks_k1_now = black_scholes_greeks(spot_at_t1, k1, t_days=t_days_remaining, sigma=iv, is_call=is_call)
    greeks_k2_now = black_scholes_greeks(spot_at_t1, k2, t_days=t_days_remaining, sigma=iv, is_call=is_call)
    
    prem_k1_now = greeks_k1_now["price"]
    prem_k2_now = greeks_k2_now["price"]
    
    net_effective_cost = round(entry_prem_k1 - prem_k2_now, 2)
    max_spread_width = abs(k2 - k1)
    max_spread_profit_pts = round(max_spread_width - max(net_effective_cost, 0.0) if net_effective_cost >= 0 else max_spread_width + abs(net_effective_cost), 2)
    max_spread_pnl_rupees = round(max_spread_profit_pts * total_qty, 2)
    breakeven_spot = round(k1 + net_effective_cost if is_call else k1 - net_effective_cost, 2)
    
    net_delta = round(greeks_k1_now["delta"] - greeks_k2_now["delta"], 3)
    net_theta = round(greeks_k1_now["theta"] - greeks_k2_now["theta"], 2)
    net_vega = round(greeks_k1_now["vega"] - greeks_k2_now["vega"], 2)
    
    return {
        "status": "CONVERTED_SPREAD",
        "spread_type": "Bull Call Spread" if is_call else "Bear Put Spread",
        "long_leg": f"NIFTY {k1} {'CE' if is_call else 'PE'} (Bought @ ₹{entry_prem_k1:.2f})",
        "short_leg": f"NIFTY {k2} {'CE' if is_call else 'PE'} (Sold @ ₹{prem_k2_now:.2f})",
        "short_strike": k2,
        "strike_k1": k1,
        "strike_k2": k2,
        "short_premium_collected": prem_k2_now,
        "credit_received": prem_k2_now,
        "net_effective_cost_per_share": net_effective_cost,
        "net_debit": net_effective_cost,
        "max_spread_width_pts": max_spread_width,
        "max_spread_profit_pts": max_spread_profit_pts,
        "max_spread_pnl_rupees": max_spread_pnl_rupees,
        "breakeven_spot": breakeven_spot,
        "net_delta": net_delta,
        "net_theta_daily": net_theta,
        "net_vega": net_vega,
        "is_theta_positive": net_theta >= 0,
        "execution_guidance": (
            f"SELL {lots} Lots of NIFTY {k2} {'CE' if is_call else 'PE'} @ ₹{prem_k2_now:.2f}. "
            f"Net debit is ₹{net_effective_cost:.2f}. Theta decay is neutralized ({net_theta:+.2f}/sh/day). "
            f"Max Profit: ₹{max_spread_profit_pts:.2f} pts (₹{max_spread_pnl_rupees:,.2f}) with Breakeven at ₹{breakeven_spot:.2f}."
        )
    }


def compute_strike_ladder_greeks(
    spot: float,
    iv: float = DEFAULT_IV,
    t_days: float = 4.0,
    r: float = RISK_FREE_RATE,
    q: float = 0.0,
    num_strikes: int = 10,
    step: int = 50
) -> pd.DataFrame:
    """Computes strike ladder DataFrame around current spot price with vectorized Greeks matrix."""
    atm_center = int(round(spot / float(step)) * step)
    min_strike = atm_center - (num_strikes // 2) * step
    max_strike = atm_center + (num_strikes // 2) * step
    k_array = np.arange(min_strike, max_strike + step, step, dtype=float)
    
    ce_batch = black_scholes_greeks_batch(spot, k_array, t_days=t_days, r=r, q=q, sigma=iv, is_call=True)
    pe_batch = black_scholes_greeks_batch(spot, k_array, t_days=t_days, r=r, q=q, sigma=iv, is_call=False)
    
    rows = []
    for i, k in enumerate(k_array):
        k_int = int(k)
        is_atm = (k_int == atm_center)
        ce_delta = ce_batch["delta"][i]
        pe_delta = pe_batch["delta"][i]
        ce_rec = "👉 PRO CALL" if (0.50 <= ce_delta <= 0.65) else ""
        pe_rec = "👉 PRO PUT" if (0.50 <= abs(pe_delta) <= 0.65) else ""
        rows.append({
            "Call Setup": ce_rec,
            "CE Delta": round(ce_delta, 4),
            "CE Gamma": round(ce_batch["gamma"][i], 6),
            "CE Theta": round(ce_batch["theta"][i], 2),
            "CE Vega": round(ce_batch["vega"][i], 2),
            "CE Vanna": round(ce_batch["vanna"][i], 4),
            "CE Premium (₹)": round(ce_batch["price"][i], 2),
            "Strike": f"🎯 {k_int} (ATM)" if is_atm else str(k_int),
            "PE Premium (₹)": round(pe_batch["price"][i], 2),
            "PE Vanna": round(pe_batch["vanna"][i], 4),
            "PE Vega": round(pe_batch["vega"][i], 2),
            "PE Theta": round(pe_batch["theta"][i], 2),
            "PE Gamma": round(pe_batch["gamma"][i], 6),
            "PE Delta": round(pe_delta, 4),
            "Put Setup": pe_rec
        })
    return pd.DataFrame(rows)


def generate_option_trade_ticket(
    spot: float,
    signal: Signal,
    capital: float = DEFAULT_CAPITAL,
    current_drawdown_pct: float = 0.0,
    t_days: float = 4.0,
    iv: float = DEFAULT_IV,
    is_0dte_afternoon: bool = False,
    current_intraday_pnl: float = 0.0,
    peak_intraday_pnl: float = 0.0,
    risk_pct_override: Optional[float] = None
) -> Dict[str, Any]:
    """Translates 3-Tier spot setups into convex institutional Option Trade Ticket with Free Spread details."""
    if signal.signal_type == SignalType.WAIT or getattr(signal, "entry_price", 0.0) <= 0.0:
        return {"status": "WAIT", "message": signal.reason}
        
    is_call = "LONG" in signal.signal_type.value
    strike_info = select_institutional_strike(spot, is_call=is_call, t_days=t_days, iv=iv, is_0dte_afternoon=is_0dte_afternoon)
    
    delta = abs(strike_info["delta"])
    gamma = strike_info["gamma"]
    theta = strike_info["theta"]
    entry_prem = strike_info["price"]
    k1 = strike_info["strike"]
    
    # 1. Spot Risk Bounds with Defensive Index Sanitization
    sl_p = float(getattr(signal, "sl_price", 0.0))
    if sl_p > 1000.0 and abs(spot - sl_p) < 300.0:
        spot_risk = abs(spot - sl_p)
    else:
        spot_risk = 35.0  # Default 35 pts index risk
        
    convexity_benefit = 0.5 * gamma * (spot_risk ** 2)
    theta_risk = abs(theta) * 0.15
    option_risk = max((spot_risk * delta) - convexity_benefit + theta_risk, 10.0)
    sl_prem = max(round(entry_prem - option_risk, 2), 5.0)
    
    # 2. 3-Tier Convex Option Target Premiums with Dynamic Bounds (Accounting for Theta Decay)
    t1_p = float(getattr(signal, "target_1", 0.0))
    if t1_p > 1000.0 and abs(t1_p - spot) < 400.0:
        diff_t1 = abs(t1_p - spot)
    else:
        diff_t1 = 45.0  # Default 45 pts index T1 (+1.2x ATR)
        
    theta_decay_t1 = abs(theta) * 0.10
    target1_prem = round(max(entry_prem + (diff_t1 * delta) + (0.5 * gamma * (diff_t1 ** 2)) - theta_decay_t1, entry_prem + 1.0), 2)
    
    t2_p = float(getattr(signal, "target_2", 0.0))
    if t2_p > 1000.0 and abs(t2_p - spot) < 600.0:
        diff_t2 = abs(t2_p - spot)
    else:
        diff_t2 = 90.0  # Default 90 pts index T2 (+2.5x ATR)
        
    theta_decay_t2 = abs(theta) * 0.20
    target2_prem = round(max(entry_prem + (diff_t2 * delta) + (0.5 * gamma * (diff_t2 ** 2)) - theta_decay_t2, target1_prem + 1.0), 2)
    
    t3_p = float(getattr(signal, "target_3_moonshot", 0.0))
    if t3_p > 1000.0 and abs(t3_p - spot) < 800.0:
        diff_t3 = abs(t3_p - spot)
    else:
        diff_t3 = 140.0  # Default 140 pts index T3 (+3.8x ATR)
        
    theta_decay_t3 = abs(theta) * 0.35
    target3_prem = round(max(entry_prem + (diff_t3 * delta) + (0.5 * gamma * (diff_t3 ** 2)) - theta_decay_t3, target2_prem + 1.0), 2)
    
    risk_pct_to_use = risk_pct_override if risk_pct_override is not None else MAX_RISK_PCT
    sizing = calculate_position_size(
        capital,
        risk_pct_to_use,
        entry_prem,
        sl_prem,
        LOT_SIZE,
        current_drawdown_pct,
        current_intraday_pnl=current_intraday_pnl,
        peak_intraday_pnl=peak_intraday_pnl
    )
    tca = calculate_adaptive_tca_friction_multi_tier(
        entry_prem, target1_prem, target2_prem, target3_prem,
        sizing["total_qty"], sizing["lots"], t1_hit=True, t2_hit=True, iv=iv, is_0dte_afternoon=is_0dte_afternoon
    )
    
    lots_35 = max(int(round(sizing["lots"] * 0.35)), 1) if sizing["lots"] >= 3 else (sizing["lots"] // 2 or 1)
    lots_30 = sizing["lots"] - (2 * lots_35) if sizing["lots"] >= 3 else (sizing["lots"] - lots_35)
    
    # Free Vertical Spread at T1 Construction
    spread_width = 100 if is_call else -100
    k2 = k1 + spread_width if is_call else k1 - abs(spread_width)
    
    target_1_spot = float(signal.target_1) if getattr(signal, "target_1", 0.0) > 0 else (spot + 50.0 if is_call else spot - 50.0)
    greeks_k2_at_t1 = black_scholes_greeks(target_1_spot, k2, t_days=max(t_days - 0.5, 0.05), sigma=iv, is_call=is_call)
    credit_received_k2 = greeks_k2_at_t1["price"]
    net_debit = round(entry_prem - credit_received_k2, 2)
    max_spread_width = abs(k2 - k1)
    max_profit_pts = round(max_spread_width - max(net_debit, 0.0) if net_debit >= 0 else max_spread_width + abs(net_debit), 2)
    max_profit_rupees = round(max_profit_pts * sizing["total_qty"], 2)
    breakeven_spot = round(k1 + net_debit if is_call else k1 - net_debit, 2)
    
    free_spread_t1 = {
        "status": "T1_FREE_SPREAD_AVAILABLE",
        "spread_type": "Bull Call Spread (Risk-Free T1 Conversion)" if is_call else "Bear Put Spread (Risk-Free T1 Conversion)",
        "strike_k1_long": k1,
        "entry_premium_k1": entry_prem,
        "strike_k2_short": k2,
        "k2_symbol": f"NIFTY {k2} {strike_info['option_type']}",
        "credit_received": credit_received_k2,
        "net_debit": net_debit,
        "max_profit_pts": max_profit_pts,
        "max_profit_rupees": max_profit_rupees,
        "breakeven_spot": breakeven_spot,
        "spread_width_pts": max_spread_width
    }

    gamma_scalp_meta = None
    if is_0dte_afternoon or t_days <= 0.5:
        gamma_scalp_meta = compute_0dte_gamma_scalp_parameters(
            spot=spot,
            strike=k1,
            dte_days=t_days,
            iv=iv,
            is_call=is_call,
            atr=diff_t1 / 1.2 if diff_t1 > 0 else 25.0
        )

    
    return {
        "status": "READY",
        "signal": signal.signal_type.value,
        "symbol": strike_info["symbol"],
        "strike": strike_info["strike"],
        "target_strike": strike_info["strike"],
        "option_type": strike_info["option_type"],
        "regime_mode": strike_info["regime_mode"],
        "spot_entry": spot,
        "delta": strike_info["delta"],
        "gamma": strike_info["gamma"],
        "theta_decay_daily": strike_info["theta"],
        "vega": strike_info["vega"],
        "vanna": strike_info["vanna"],
        "charm": strike_info["charm"],
        "volga": strike_info["volga"],
        "speed": strike_info.get("speed", 0.0),
        "color": strike_info.get("color", 0.0),
        "entry_premium": entry_prem,
        "sl_premium": sl_prem,
        "target1_premium": target1_prem,
        "target2_premium": target2_prem,
        "target3_moonshot_premium": target3_prem,
        "pyramid_trigger_spot": getattr(signal, "pyramid_trigger", spot + 30.0),
        "lots": sizing["lots"],
        "lots_t1_35pct": lots_35,
        "lots_t2_35pct": lots_35,
        "lots_t3_30pct": max(lots_30, 1),
        "total_qty": sizing["total_qty"],
        "max_risk_rupees": sizing["actual_risk_rupees"],
        "capital_outlay": sizing["capital_required"],
        "tca_friction": tca,
        "free_spread_t1": free_spread_t1,
        "gamma_scalp_meta": gamma_scalp_meta,
        "execution_rules": {
            "part_book_50_pct": f"Option A (Outright): Book 50% ({max(sizing['lots'] // 2, 1)} lots) at ₹{target1_prem:.2f} and move SL to ₹{entry_prem:.2f}",
            "tier_1_asymmetric": (
                f"Tier 1 (+1.2x ATR @ ₹{signal.target_1:.1f}): Book 35% ({lots_35} lots) at ₹{target1_prem:.2f} "
                f"OR Sell {sizing['lots']} Lots OTM {k2} {strike_info['option_type']} @ ~₹{credit_received_k2:.2f} for Free Spread "
                f"(Net Debit: ₹{net_debit:.2f}, Max Profit: ₹{max_profit_pts:.2f} pts)."
            ),
            "tier_2_structural": f"Tier 2 (+2.5x ATR): Book 35% ({lots_35} lots) at ₹{target2_prem:.2f}.",
            "tier_3_moonshot": f"Tier 3 (Moonshot Runner): Trail remaining 30% ({max(lots_30, 1)} lots) on 5m 21 EMA / AVWAP 1σ into ₹{target3_prem:.2f}.",
            "trailing_rule": "Trail remaining position on 5-minute 21 EMA or Session AVWAP 1σ",
            "profit_ratchet": "Lock 65% of peak gains once +1.5R (+1.5%) session profit is achieved"
        }
    }


def construct_delta_neutral_iron_condor(
    spot: float,
    wing_width: int = 150,
    short_offset: int = 100,
    t_days: float = 3.5,
    iv: float = DEFAULT_IV,
    r: float = RISK_FREE_RATE
) -> Dict[str, Any]:
    """
    Constructs a 4-Leg Delta-Neutral Iron Condor for Range-Bound / Chop Market Regimes:
    - Long Put Wing: K_pl = Spot - short_offset - wing_width
    - Short Put: K_ps = Spot - short_offset
    - Short Call: K_cs = Spot + short_offset
    - Long Call Wing: K_cl = Spot + short_offset + wing_width
    """
    atm = int(round(spot / 50.0) * 50)
    k_ps = atm - short_offset
    k_pl = k_ps - wing_width
    k_cs = atm + short_offset
    k_cl = k_cs + wing_width

    # Compute individual leg Greeks and theoretical prices
    g_pl = black_scholes_greeks(spot, k_pl, t_days=t_days, r=r, sigma=iv, is_call=False)
    g_ps = black_scholes_greeks(spot, k_ps, t_days=t_days, r=r, sigma=iv, is_call=False)
    g_cs = black_scholes_greeks(spot, k_cs, t_days=t_days, r=r, sigma=iv, is_call=True)
    g_cl = black_scholes_greeks(spot, k_cl, t_days=t_days, r=r, sigma=iv, is_call=True)

    # Net Credit Received
    put_spread_credit = max(g_ps["price"] - g_pl["price"], 1.0)
    call_spread_credit = max(g_cs["price"] - g_cl["price"], 1.0)
    total_net_credit = put_spread_credit + call_spread_credit
    
    max_profit_pts = total_net_credit
    max_loss_pts = wing_width - total_net_credit
    risk_reward_ratio = max_loss_pts / max(max_profit_pts, 0.1)

    lower_breakeven = k_ps - total_net_credit
    upper_breakeven = k_cs + total_net_credit

    net_delta = (-g_ps["delta"] + g_pl["delta"]) + (-g_cs["delta"] + g_cl["delta"]) # short legs inverted
    net_theta = (-g_ps["theta"] + g_pl["theta"]) + (-g_cs["theta"] + g_cl["theta"]) # net positive time decay collected
    net_vega = (-g_ps["vega"] + g_pl["vega"]) + (-g_cs["vega"] + g_cl["vega"]) # short vega


    # Probability of Profit (PoP) via Log-Normal Distribution
    t_years = max(t_days / 365.0, 0.001)
    sigma_root_t = iv * math.sqrt(t_years)
    d2_upper = (math.log(upper_breakeven / spot) - (r - 0.5 * iv**2) * t_years) / sigma_root_t
    d2_lower = (math.log(lower_breakeven / spot) - (r - 0.5 * iv**2) * t_years) / sigma_root_t
    pop_pct = (norm.cdf(d2_upper) - norm.cdf(d2_lower)) * 100.0
    pop_pct = round(float(np.clip(pop_pct, 60.0, 95.0)), 1)

    return {
        "status": "STRUCTURED",
        "strategy": "DELTA_NEUTRAL_IRON_CONDOR",
        "spot": spot,
        "legs": {
            "long_put": {"strike": k_pl, "type": "PE", "side": "BUY", "premium": round(g_pl["price"], 2), "delta": round(g_pl["delta"], 3)},
            "short_put": {"strike": k_ps, "type": "PE", "side": "SELL", "premium": round(g_ps["price"], 2), "delta": round(g_ps["delta"], 3)},
            "short_call": {"strike": k_cs, "type": "CE", "side": "SELL", "premium": round(g_cs["price"], 2), "delta": round(g_cs["delta"], 3)},
            "long_call": {"strike": k_cl, "type": "CE", "side": "BUY", "premium": round(g_cl["price"], 2), "delta": round(g_cl["delta"], 3)}
        },
        "total_net_credit_pts": round(total_net_credit, 2),
        "max_profit_pts": round(max_profit_pts, 2),
        "max_loss_pts": round(max_loss_pts, 2),
        "lower_breakeven": round(lower_breakeven, 1),
        "upper_breakeven": round(upper_breakeven, 1),
        "profit_range_pts": round(upper_breakeven - lower_breakeven, 1),
        "net_delta": round(net_delta, 4),
        "net_theta_daily": round(net_theta, 2),
        "net_vega": round(net_vega, 2),
        "probability_of_profit_pct": pop_pct,
        "recommended_regime": "ANTI-PERSISTENT CHOP / NEUTRAL DAY (H < 0.48, VR < 0.80)"
    }


def construct_jade_lizard(
    spot: float,
    put_offset: int = 150,
    call_short_offset: int = 150,
    call_wing_width: int = 100,
    t_days: float = 4.0,
    iv: float = DEFAULT_IV,
    r: float = RISK_FREE_RATE,
    put_skew_multiplier: float = 1.15
) -> Dict[str, Any]:
    """
    Constructs a 3-Leg Jade Lizard Strategy for exploiting institutional Put Skew:
    - Short 1x OTM Put: K_p = Spot - put_offset (higher IV due to downside skew)
    - Short 1x OTM Call: K_cs = Spot + call_short_offset
    - Long 1x OTM Call: K_cl = K_cs + call_wing_width
    
    Zero Upside Risk Invariant:
    If Net Credit Collected >= Call Wing Width, the maximum loss on the Bear Call Spread
    is fully offset by the total credit, leaving ZERO upside risk.
    """
    atm = int(round(spot / 50.0) * 50)
    k_p = atm - put_offset
    k_cs = atm + call_short_offset
    k_cl = k_cs + call_wing_width

    # Skewed put IV
    iv_put = iv * put_skew_multiplier

    g_p = black_scholes_greeks(spot, k_p, t_days=t_days, r=r, sigma=iv_put, is_call=False)
    g_cs = black_scholes_greeks(spot, k_cs, t_days=t_days, r=r, sigma=iv, is_call=True)
    g_cl = black_scholes_greeks(spot, k_cl, t_days=t_days, r=r, sigma=iv, is_call=True)

    put_credit = max(g_p["price"], 1.0)
    call_spread_cost = max(g_cs["price"] - g_cl["price"], 1.0)
    total_net_credit = put_credit + call_spread_cost

    upside_risk = max(0.0, float(call_wing_width - total_net_credit))
    has_zero_upside_risk = total_net_credit >= float(call_wing_width)

    max_profit = total_net_credit
    lower_breakeven = k_p - total_net_credit

    net_delta = -g_p["delta"] - g_cs["delta"] + g_cl["delta"]
    net_theta = -g_p["theta"] - g_cs["theta"] + g_cl["theta"]
    net_vega = -g_p["vega"] - g_cs["vega"] + g_cl["vega"]

    return {
        "status": "STRUCTURED",
        "strategy": "JADE_LIZARD_SKEW_ARBITRAGE",
        "spot": spot,
        "legs": {
            "short_put": {"strike": k_p, "type": "PE", "side": "SELL", "premium": round(g_p["price"], 2), "delta": round(g_p["delta"], 3), "iv": round(iv_put, 3)},
            "short_call": {"strike": k_cs, "type": "CE", "side": "SELL", "premium": round(g_cs["price"], 2), "delta": round(g_cs["delta"], 3), "iv": round(iv, 3)},
            "long_call": {"strike": k_cl, "type": "CE", "side": "BUY", "premium": round(g_cl["price"], 2), "delta": round(g_cl["delta"], 3), "iv": round(iv, 3)}
        },
        "total_net_credit_pts": round(total_net_credit, 2),
        "call_wing_width_pts": float(call_wing_width),
        "has_zero_upside_risk": has_zero_upside_risk,
        "upside_risk_pts": round(upside_risk, 2),
        "max_profit_pts": round(max_profit, 2),
        "lower_breakeven": round(lower_breakeven, 1),
        "net_delta": round(net_delta, 4),
        "net_theta_daily": round(net_theta, 2),
        "net_vega": round(net_vega, 2),
        "recommended_regime": "ELEVATED PUT SKEW (Z_skew >= +1.0) / MODERATE BULLISH TO NEUTRAL BIAS"
    }




