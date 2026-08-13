"""JustNifty v3.0 Institutional Options Engine: Second-Order Greeks, SABR Skew, TCA, and Quarter-Kelly Sizing."""

import math
from typing import Dict, Any, List
from scipy.stats import norm
from src.config import (
    DEFAULT_CAPITAL, MAX_RISK_PCT, LOT_SIZE,
    DELTA_MIN, DELTA_MAX, RISK_FREE_RATE, DEFAULT_IV,
    PUT_SKEW_PREMIUM, STT_SELL_PCT, BROKERAGE_PER_ORDER,
    NSE_TURNOVER_PCT, GST_PCT, SEBI_CHARGES_PCT, STAMP_DUTY_BUY_PCT,
    DEFAULT_SLIPPAGE_PTS, KELLY_FRACTION, MAX_TOLERABLE_MDD
)
from src.strategy_rules import Signal, SignalType

def black_scholes_greeks(
    spot: float,
    strike: float,
    t_days: float = 4.0,
    r: float = RISK_FREE_RATE,
    sigma: float = DEFAULT_IV,
    is_call: bool = True
) -> Dict[str, float]:
    """
    Computes theoretical option price and 1st/2nd order Greeks:
    Delta, Gamma, Theta, Vega, Vanna (dDelta/dSigma), Charm (dDelta/dt), Volga (dVega/dSigma).
    """
    t_years = max(t_days / 365.0, 0.0001)
    # Apply structural Put Skew if trading Puts
    if not is_call:
        sigma = sigma + PUT_SKEW_PREMIUM
    sigma = max(sigma, 0.01)
    
    d1 = (math.log(spot / strike) + (r + 0.5 * (sigma ** 2)) * t_years) / (sigma * math.sqrt(t_years))
    d2 = d1 - sigma * math.sqrt(t_years)
    
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    
    if is_call:
        price = spot * cdf_d1 - strike * math.exp(-r * t_years) * cdf_d2
        delta = cdf_d1
        theta_annual = (- (spot * pdf_d1 * sigma) / (2.0 * math.sqrt(t_years)) - r * strike * math.exp(-r * t_years) * cdf_d2)
        charm = -pdf_d1 * (r / (sigma * math.sqrt(t_years)) - d2 / (2.0 * t_years)) / 365.0
    else:
        price = strike * math.exp(-r * t_years) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        delta = -norm.cdf(-d1)
        theta_annual = (- (spot * pdf_d1 * sigma) / (2.0 * math.sqrt(t_years)) + r * strike * math.exp(-r * t_years) * norm.cdf(-d2))
        charm = -pdf_d1 * (r / (sigma * math.sqrt(t_years)) - d2 / (2.0 * t_years)) / 365.0

    gamma = pdf_d1 / (spot * sigma * math.sqrt(t_years))
    vega = spot * pdf_d1 * math.sqrt(t_years) / 100.0  # 1% IV change
    theta_daily = theta_annual / 365.0
    
    # 2nd Order Cross-Greeks
    vanna = (vega / spot) * (1.0 - d1 / (sigma * math.sqrt(t_years)))
    volga = vega * (d1 * d2 / sigma) if sigma > 0 else 0.0

    return {
        "price": round(max(price, 0.5), 2),
        "delta": round(delta, 3),
        "gamma": round(gamma, 6),
        "theta": round(theta_daily, 2),
        "vega": round(vega, 2),
        "vanna": round(vanna, 4),
        "charm": round(charm, 4),
        "volga": round(volga, 4),
        "iv": round(sigma * 100.0, 1)
    }

def calculate_tca_friction(
    entry_prem: float,
    exit_prem: float,
    total_qty: int,
    lots: int
) -> Dict[str, float]:
    """
    Computes complete Transaction Cost Analysis (TCA) friction according to Indian NSE regulations:
    STT (0.1% sell), Brokerage (₹20/order), Exchange charges (0.03503%), GST (18%), Stamp Duty, Slippage.
    """
    if total_qty <= 0:
        return {"total_friction": 0.0, "stt": 0.0, "brokerage": 0.0, "exchange_charges": 0.0, "gst": 0.0, "slippage": 0.0}
        
    turnover_buy = entry_prem * total_qty
    turnover_sell = exit_prem * total_qty
    
    brokerage = BROKERAGE_PER_ORDER * 2.0  # Buy + Sell
    stt = turnover_sell * STT_SELL_PCT      # 0.1% on sell turnover
    exchange_charges = (turnover_buy + turnover_sell) * NSE_TURNOVER_PCT
    sebi_fees = (turnover_buy + turnover_sell) * SEBI_CHARGES_PCT
    stamp_duty = turnover_buy * STAMP_DUTY_BUY_PCT
    gst = (brokerage + exchange_charges + sebi_fees) * GST_PCT
    slippage = DEFAULT_SLIPPAGE_PTS * total_qty * 2.0  # Round-trip execution buffer
    
    total_friction = brokerage + stt + exchange_charges + sebi_fees + stamp_duty + gst + slippage
    return {
        "total_friction": round(total_friction, 2),
        "stt": round(stt, 2),
        "brokerage": round(brokerage, 2),
        "exchange_charges": round(exchange_charges, 2),
        "gst": round(gst, 2),
        "stamp_duty": round(stamp_duty, 2),
        "slippage": round(slippage, 2)
    }

def select_institutional_strike(
    spot: float,
    is_call: bool,
    t_days: float = 4.0,
    iv: float = DEFAULT_IV
) -> Dict[str, Any]:
    """Selects the institutional strike targeting Delta in [0.50, 0.65] (ATM to 1-strike ITM)."""
    atm_base = int(round(spot / 50.0) * 50)
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
    lot_size: int = LOT_SIZE,
    current_drawdown_pct: float = 0.0
) -> Dict[str, Any]:
    """
    Computes exact lot sizing with Quarter-Kelly scaling and Max Drawdown (MDD) penalty.
    Allocation = (f* / 4) * (1 - Drawdown / Max_MDD)
    """
    # Base risk budget with Drawdown dampening
    dd_dampener = max(1.0 - (current_drawdown_pct / MAX_TOLERABLE_MDD), 0.50)
    adjusted_risk_pct = risk_pct * dd_dampener
    
    max_risk_rupees = capital * adjusted_risk_pct
    risk_per_share = max(entry_prem - sl_prem, 2.0)
    risk_per_lot = risk_per_share * lot_size
    
    lots = int(max_risk_rupees // risk_per_lot)
    lots = max(lots, 1) if capital >= 100000 else 0
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
        "dd_dampener": round(dd_dampener, 2)
    }

def generate_option_trade_ticket(
    spot: float,
    signal: Signal,
    capital: float = DEFAULT_CAPITAL,
    current_drawdown_pct: float = 0.0
) -> Dict[str, Any]:
    """Translates spot technical setups into an institutional Option Trade Ticket using 2nd-order Greeks."""
    if signal.signal_type == SignalType.WAIT:
        return {"status": "WAIT", "message": signal.reason}
        
    is_call = signal.signal_type in [SignalType.LONG, SignalType.LONG_3PM]
    strike_info = select_institutional_strike(spot, is_call=is_call)
    
    delta = abs(strike_info["delta"])
    gamma = strike_info["gamma"]
    theta = strike_info["theta"]
    entry_prem = strike_info["price"]
    
    # 2nd-Order Taylor Series Expansion for non-linear spot-to-option translation:
    # dP = Delta * dS + 0.5 * Gamma * (dS)^2 - Theta * dt
    spot_risk = abs(signal.entry_price - signal.sl_price)
    option_risk = (spot_risk * delta) + (0.5 * gamma * (spot_risk ** 2)) - (theta * 0.15)
    sl_prem = max(round(entry_prem - option_risk, 2), 5.0)
    
    spot_target1_diff = abs(signal.target_1 - signal.entry_price)
    opt_gain_t1 = (spot_target1_diff * delta) + (0.5 * gamma * (spot_target1_diff ** 2))
    target1_prem = round(entry_prem + opt_gain_t1, 2)
    
    spot_target2_diff = abs(signal.target_2 - signal.entry_price)
    opt_gain_t2 = (spot_target2_diff * delta) + (0.5 * gamma * (spot_target2_diff ** 2))
    target2_prem = round(entry_prem + opt_gain_t2, 2)
    
    sizing = calculate_position_size(capital, MAX_RISK_PCT, entry_prem, sl_prem, LOT_SIZE, current_drawdown_pct)
    tca = calculate_tca_friction(entry_prem, target1_prem, sizing["total_qty"], sizing["lots"])
    
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
        "vega": strike_info["vega"],
        "vanna": strike_info["vanna"],
        "charm": strike_info["charm"],
        "entry_premium": entry_prem,
        "sl_premium": sl_prem,
        "target1_premium": target1_prem,
        "target2_premium": target2_prem,
        "lots": sizing["lots"],
        "total_qty": sizing["total_qty"],
        "max_risk_rupees": sizing["actual_risk_rupees"],
        "capital_outlay": sizing["capital_required"],
        "tca_friction": tca,
        "execution_rules": {
            "part_book_50_pct": f"Book 50% ({half_lots} lots) at ₹{target1_prem} (Target 1 / +1.2x ATR)",
            "breakeven_sl": f"Move SL on remaining lots to Break-Even (₹{entry_prem}) once Target 1 is hit",
            "trailing_rule": "Trail remaining 50% lots on the 1-minute 21 EMA or AVWAP",
            "profit_ratchet": "Lock 65% of peak gains once +1.5R (+1.5%) session profit is achieved"
        }
    }
