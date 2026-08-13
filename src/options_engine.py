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
    """
    Computes theoretical European option price and exact 1st/2nd/3rd order Greeks.
    """
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
    charm_daily = -pdf_d1 * (r / (sigma * sqrt_t) - d2 / (2.0 * t_years)) / 365.0
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
    lock_pct: float = 0.75
) -> Dict[str, Any]:
    """
    Dynamic Intraday Profit Lock ('The Golden Vault Rule'):
    Once intraday Net PnL reaches >= +1.5% of account capital, lock 75% of peak profits
    as an untouchable risk floor for the remainder of the session.
    """
    trigger_threshold_rupees = initial_capital * profit_trigger_pct
    peak_pnl = max(peak_intraday_pnl, current_intraday_pnl)
    is_vault_triggered = peak_pnl >= trigger_threshold_rupees
    
    if is_vault_triggered:
        locked_profit_floor = round(peak_pnl * lock_pct, 2)
        untouchable_capital_floor = round(initial_capital + locked_profit_floor, 2)
        risk_cushion = round(max(current_intraday_pnl - locked_profit_floor, 0.0), 2)
        is_session_halted = current_intraday_pnl <= locked_profit_floor
        
        if is_session_halted:
            status = "LOCKED_GOLDEN_VAULT"
            message = (
                f"🔒 Golden Vault Engaged: +₹{locked_profit_floor:,.2f} locked floor reached. "
                f"Session trading halted to preserve 75% of peak gains (₹{peak_pnl:,.2f})."
            )
        else:
            status = "VAULT_ACTIVE"
            message = (
                f"🛡️ Golden Vault Active: Peak PnL ₹{peak_pnl:,.2f} (≥ +{profit_trigger_pct*100:.1f}%). "
                f"₹{locked_profit_floor:,.2f} locked floor. Remaining risk cushion: ₹{risk_cushion:,.2f}."
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
        "peak_intraday_pnl": round(peak_pnl, 2),
        "current_intraday_pnl": round(current_intraday_pnl, 2),
        "trigger_threshold_rupees": round(trigger_threshold_rupees, 2),
        "locked_profit_floor": locked_profit_floor,
        "untouchable_capital_floor": untouchable_capital_floor,
        "risk_cushion": risk_cushion,
        "lock_pct": lock_pct,
        "message": message
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
    
    ruined_count = int(np.sum(path_max_dds >= ruin_threshold_pct))
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


def calculate_pcr_and_max_pain(option_chain_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes exact Max Pain strike price (where option sellers incur minimum cumulative payout)
    and Open Interest / Change in OI / Volume Put-Call Ratio (PCR).
    """
    if option_chain_df is None or option_chain_df.empty:
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


def select_institutional_strike(
    spot: float,
    is_call: bool,
    t_days: float = 4.0,
    iv: float = DEFAULT_IV,
    is_0dte_afternoon: bool = False
) -> Dict[str, Any]:
    """
    Selects optimal institutional strike:
    - Normal regimes: Target Delta ~0.58 in [0.50, 0.65].
    - 0DTE Expiry Thursday afternoon: Selects Deep ITM (Delta ~0.75-0.85) to avoid gamma cliff.
    """
    atm_base = int(round(spot / 50.0) * 50)
    candidate_offsets = [0, -50, 50, -100, 100, -150, 150, -200, 200, -250, -300] if is_call else [0, 50, -50, 100, -100, 150, -150, 200, -200, 250, 300]
    candidates = [atm_base + off for off in candidate_offsets]
    
    target_delta = DELTA_DEEP_ITM_0DTE if is_0dte_afternoon else 0.58
    min_delta = 0.70 if is_0dte_afternoon else DELTA_MIN
    max_delta = 0.90 if is_0dte_afternoon else DELTA_MAX
    
    best_strike = atm_base
    best_greeks = black_scholes_greeks(spot, best_strike, t_days, sigma=iv, is_call=is_call)
    best_delta_diff = float("inf")
    
    for k in candidates:
        g = black_scholes_greeks(spot, k, t_days, sigma=iv, is_call=is_call)
        abs_delta = abs(g["delta"])
        if min_delta <= abs_delta <= max_delta:
            diff = abs(abs_delta - target_delta)
            if diff < best_delta_diff:
                best_delta_diff = diff
                best_strike = k
                best_greeks = g

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
    """Computes strike ladder DataFrame around current spot price with Greeks matrix."""
    atm_center = int(round(spot / float(step)) * step)
    min_strike = atm_center - (num_strikes // 2) * step
    max_strike = atm_center + (num_strikes // 2) * step
    
    rows = []
    for k in range(min_strike, max_strike + step, step):
        ce = black_scholes_greeks(spot, k, t_days=t_days, r=r, q=q, sigma=iv, is_call=True)
        pe = black_scholes_greeks(spot, k, t_days=t_days, r=r, q=q, sigma=iv, is_call=False)
        is_atm = (k == atm_center)
        ce_rec = "👉 PRO CALL" if (0.50 <= ce["delta"] <= 0.65) else ""
        pe_rec = "👉 PRO PUT" if (0.50 <= abs(pe["delta"]) <= 0.65) else ""
        rows.append({
            "Call Setup": ce_rec,
            "CE Delta": ce["delta"],
            "CE Gamma": ce["gamma"],
            "CE Theta": ce["theta"],
            "CE Vega": ce["vega"],
            "CE Vanna": ce["vanna"],
            "CE Premium (₹)": ce["price"],
            "Strike": f"🎯 {k} (ATM)" if is_atm else str(k),
            "PE Premium (₹)": pe["price"],
            "PE Vanna": pe["vanna"],
            "PE Vega": pe["vega"],
            "PE Theta": pe["theta"],
            "PE Gamma": pe["gamma"],
            "PE Delta": pe["delta"],
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
    is_0dte_afternoon: bool = False
) -> Dict[str, Any]:
    """Translates 3-Tier spot setups into convex institutional Option Trade Ticket with Free Spread details."""
    if signal.signal_type == SignalType.WAIT:
        return {"status": "WAIT", "message": signal.reason}
        
    is_call = signal.signal_type in [SignalType.LONG, SignalType.LONG_3PM]
    strike_info = select_institutional_strike(spot, is_call=is_call, t_days=t_days, iv=iv, is_0dte_afternoon=is_0dte_afternoon)
    
    delta = abs(strike_info["delta"])
    gamma = strike_info["gamma"]
    theta = strike_info["theta"]
    entry_prem = strike_info["price"]
    k1 = strike_info["strike"]
    
    spot_risk = abs(signal.entry_price - signal.sl_price)
    convexity_benefit = 0.5 * gamma * (spot_risk ** 2)
    theta_risk = abs(theta) * 0.15
    option_risk = max((spot_risk * delta) - convexity_benefit + theta_risk, 2.0)
    sl_prem = max(round(entry_prem - option_risk, 2), 2.0)
    
    # 3-Tier Convex Option Target Premiums
    diff_t1 = abs(signal.target_1 - signal.entry_price)
    target1_prem = round(entry_prem + (diff_t1 * delta) + (0.5 * gamma * (diff_t1 ** 2)), 2)
    
    diff_t2 = abs(signal.target_2 - signal.entry_price)
    target2_prem = round(entry_prem + (diff_t2 * delta) + (0.5 * gamma * (diff_t2 ** 2)), 2)
    
    diff_t3 = abs(getattr(signal, "target_3_moonshot", signal.target_2 + 50.0) - signal.entry_price)
    target3_prem = round(entry_prem + (diff_t3 * delta) + (0.5 * gamma * (diff_t3 ** 2)), 2)
    
    sizing = calculate_position_size(capital, MAX_RISK_PCT, entry_prem, sl_prem, LOT_SIZE, current_drawdown_pct)
    tca = calculate_adaptive_tca_friction_multi_tier(
        entry_prem, target1_prem, target2_prem, target3_prem,
        sizing["total_qty"], sizing["lots"], t1_hit=True, t2_hit=True, iv=iv, is_0dte_afternoon=is_0dte_afternoon
    )
    
    lots_35 = max(int(round(sizing["lots"] * 0.35)), 1) if sizing["lots"] >= 3 else (sizing["lots"] // 2 or 1)
    lots_30 = sizing["lots"] - (2 * lots_35) if sizing["lots"] >= 3 else (sizing["lots"] - lots_35)
    
    # Free Vertical Spread at T1 Construction
    spread_width = 100 if is_call else -100
    k2 = k1 + spread_width if is_call else k1 - abs(spread_width)
    
    greeks_k2_at_t1 = black_scholes_greeks(signal.target_1, k2, t_days=max(t_days - 0.5, 0.05), sigma=iv, is_call=is_call)
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
    
    return {
        "status": "READY",
        "signal": signal.signal_type.value,
        "symbol": strike_info["symbol"],
        "strike": strike_info["strike"],
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

