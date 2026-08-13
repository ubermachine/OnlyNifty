"""Institutional Options Engine: Black-Scholes Greeks, Strike Picker, OI Analytics, and 1% Risk Sizing."""

import math
from typing import Dict, Any, List
from scipy.stats import norm
from src.config import (
    DEFAULT_CAPITAL, MAX_RISK_PCT, LOT_SIZE,
    DELTA_MIN, DELTA_MAX, RISK_FREE_RATE, DEFAULT_IV
)
from src.strategy_rules import Signal, SignalType

def black_scholes_greeks(
    spot: float,
    strike: float,
    t_days: float,
    r: float = RISK_FREE_RATE,
    sigma: float = DEFAULT_IV,
    is_call: bool = True
) -> Dict[str, float]:
    """Calculates Black-Scholes theoretical price and 1st/2nd order Greeks (Delta, Gamma, Theta, Vega)."""
    t_years = max(t_days / 365.0, 0.0001)
    sigma = max(sigma, 0.01)
    
    d1 = (math.log(spot / strike) + (r + 0.5 * (sigma ** 2)) * t_years) / (sigma * math.sqrt(t_years))
    d2 = d1 - sigma * math.sqrt(t_years)
    
    if is_call:
        price = spot * norm.cdf(d1) - strike * math.exp(-r * t_years) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (- (spot * norm.pdf(d1) * sigma) / (2.0 * math.sqrt(t_years)) - r * strike * math.exp(-r * t_years) * norm.cdf(d2)) / 365.0
    else:
        price = strike * math.exp(-r * t_years) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        delta = -norm.cdf(-d1)
        theta = (- (spot * norm.pdf(d1) * sigma) / (2.0 * math.sqrt(t_years)) + r * strike * math.exp(-r * t_years) * norm.cdf(-d2)) / 365.0

    gamma = norm.pdf(d1) / (spot * sigma * math.sqrt(t_years))
    vega = spot * norm.pdf(d1) * math.sqrt(t_years) / 100.0

    return {
        "price": round(max(price, 0.5), 2),
        "delta": round(delta, 3),
        "gamma": round(gamma, 6),
        "theta": round(theta, 2),
        "vega": round(vega, 2),
        "iv": round(sigma * 100.0, 1)
    }

def select_institutional_strike(
    spot: float,
    is_call: bool,
    t_days: float = 4.0,
    iv: float = DEFAULT_IV
) -> Dict[str, Any]:
    """Selects the institutional strike targeting Delta between 0.50 and 0.65 (ATM to 1-strike ITM)."""
    atm_base = int(round(spot / 50.0) * 50)
    # Order candidates starting with ATM, then 1-strike ITM, 1-strike OTM, etc.
    if is_call:
        candidates = [atm_base, atm_base - 50, atm_base + 50, atm_base - 100, atm_base + 100]
    else:
        candidates = [atm_base, atm_base + 50, atm_base - 50, atm_base + 100, atm_base - 100]
        
    best_strike = atm_base
    best_greeks = black_scholes_greeks(spot, best_strike, t_days, sigma=iv, is_call=is_call)
    
    for k in candidates:
        g = black_scholes_greeks(spot, k, t_days, sigma=iv, is_call=is_call)
        abs_delta = abs(g["delta"])
        if DELTA_MIN <= abs_delta <= DELTA_MAX:
            best_strike = k
            best_greeks = g
            break

    opt_type = "CE" if is_call else "PE"
    return {
        "strike": best_strike,
        "option_type": opt_type,
        "symbol": f"NIFTY {best_strike} {opt_type}",
        **best_greeks
    }

def calculate_position_size(
    capital: float,
    risk_pct: float,
    entry_prem: float,
    sl_prem: float,
    lot_size: int = LOT_SIZE
) -> Dict[str, Any]:
    """Computes exact lot sizing strictly governed by the 1.0% capital risk rule."""
    max_risk_rupees = capital * risk_pct
    risk_per_share = max(entry_prem - sl_prem, 2.0)
    risk_per_lot = risk_per_share * lot_size
    
    lots = int(max_risk_rupees // risk_per_lot)
    total_qty = lots * lot_size
    total_capital_required = round(total_qty * entry_prem, 2)
    actual_risk_rupees = round(total_qty * risk_per_share, 2)
    
    return {
        "lots": lots,
        "total_qty": total_qty,
        "risk_per_lot": round(risk_per_lot, 2),
        "actual_risk_rupees": actual_risk_rupees,
        "max_risk_rupees": round(max_risk_rupees, 2),
        "capital_required": total_capital_required
    }

def generate_option_trade_ticket(
    spot: float,
    signal: Signal,
    capital: float = DEFAULT_CAPITAL
) -> Dict[str, Any]:
    """Translates spot technical setups into an institutional Option Trade Ticket."""
    if signal.signal_type == SignalType.WAIT:
        return {"status": "WAIT", "message": signal.reason}
        
    is_call = signal.signal_type in [SignalType.LONG, SignalType.LONG_3PM]
    strike_info = select_institutional_strike(spot, is_call=is_call)
    
    delta = abs(strike_info["delta"])
    entry_prem = strike_info["price"]
    
    # Translate Spot SL and Targets to Option Premiums using Delta
    spot_risk = abs(signal.entry_price - signal.sl_price)
    option_risk = spot_risk * delta
    sl_prem = max(round(entry_prem - option_risk, 2), 5.0)
    
    spot_target1_diff = abs(signal.target_1 - signal.entry_price)
    target1_prem = round(entry_prem + (spot_target1_diff * delta), 2)
    
    spot_target2_diff = abs(signal.target_2 - signal.entry_price)
    target2_prem = round(entry_prem + (spot_target2_diff * delta), 2)
    
    sizing = calculate_position_size(capital, MAX_RISK_PCT, entry_prem, sl_prem, LOT_SIZE)
    
    half_lots = max(sizing["lots"] // 2, 1) if sizing["lots"] > 0 else 0
    
    return {
        "status": "READY",
        "signal": signal.signal_type.value,
        "symbol": strike_info["symbol"],
        "strike": strike_info["strike"],
        "option_type": strike_info["option_type"],
        "delta": strike_info["delta"],
        "gamma": strike_info["gamma"],
        "theta_decay_daily": strike_info["theta"],
        "entry_premium": entry_prem,
        "sl_premium": sl_prem,
        "target1_premium": target1_prem,
        "target2_premium": target2_prem,
        "lots": sizing["lots"],
        "total_qty": sizing["total_qty"],
        "max_risk_rupees": sizing["actual_risk_rupees"],
        "capital_outlay": sizing["capital_required"],
        "execution_rules": {
            "part_book_50_pct": f"Book 50% ({half_lots} lots) at ₹{target1_prem} (VF T1 / 1.5% Envelope)",
            "breakeven_sl": f"Move SL on remaining lots to Break-Even (₹{entry_prem}) once Target 1 is hit",
            "trailing_rule": "Trail remaining 50% lots candle-by-candle on the 1-minute 21 EMA or AVWAP",
            "enough_rule": "Shut terminal if session gains reach ≥ 0.3% on total account capital"
        }
    }
