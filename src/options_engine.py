"""JustNifty v3.0 Enhanced Institutional Options Engine: Second-Order Greeks, 0DTE Adaptation, Free Vertical Spreads & TCA."""

import math
from typing import Dict, Any, List, Optional
from scipy.stats import norm
from src.config import (
    DEFAULT_CAPITAL, MAX_RISK_PCT, LOT_SIZE,
    DELTA_MIN, DELTA_MAX, DELTA_DEEP_ITM_0DTE, RISK_FREE_RATE, DEFAULT_IV,
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
    Computes theoretical option price and exact 1st/2nd order Greeks:
    Delta, Gamma, Theta, Vega, Vanna (dDelta/dSigma per 1%), Charm (dDelta/dt daily), Volga (dVega/dSigma per 1%).
    """
    t_years = max(t_days / 365.0, 0.00005)  # Resilient to 0DTE expiry minutes
    
    # Structural Put Skew (+250 bps for OTM/ATM puts)
    if not is_call:
        sigma = sigma + PUT_SKEW_PREMIUM
    sigma = max(sigma, 0.01)
    
    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r + 0.5 * (sigma ** 2)) * t_years) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    
    if is_call:
        price = spot * cdf_d1 - strike * math.exp(-r * t_years) * cdf_d2
        delta = cdf_d1
        theta_annual = (- (spot * pdf_d1 * sigma) / (2.0 * sqrt_t) - r * strike * math.exp(-r * t_years) * cdf_d2)
    else:
        price = strike * math.exp(-r * t_years) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        delta = -norm.cdf(-d1)
        theta_annual = (- (spot * pdf_d1 * sigma) / (2.0 * sqrt_t) + r * strike * math.exp(-r * t_years) * norm.cdf(-d2))

    gamma = pdf_d1 / (spot * sigma * sqrt_t)
    vega_1pct = (spot * pdf_d1 * sqrt_t) / 100.0  # ₹ change per +1.0% IV
    theta_daily = theta_annual / 365.0
    
    # Corrected 2nd Order Cross-Greeks
    charm_daily = -pdf_d1 * (r / (sigma * sqrt_t) - d2 / (2.0 * t_years)) / 365.0
    vanna_1pct = - (pdf_d1 * d2) / (sigma * 100.0)
    volga_1pct = (vega_1pct * d1 * d2) / (sigma * 100.0)

    return {
        "price": round(max(price, 0.20), 2),
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta_daily, 2),
        "vega": round(vega_1pct, 2),
        "vanna": round(vanna_1pct, 4),
        "charm": round(charm_daily, 4),
        "volga": round(volga_1pct, 4),
        "iv": round(sigma * 100.0, 1)
    }

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
    Statutory STT (0.1% sell), Brokerage (₹20/order), NSE Charges (0.035%), GST (18%), Stamp Duty (0.003%),
    and Volatility/0DTE-expanded Slippage.
    """
    if total_qty <= 0:
        return {"total_friction": 0.0, "stt": 0.0, "brokerage": 0.0, "exchange_charges": 0.0, "gst": 0.0, "slippage": 0.0}
        
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
    # Round-trip slippage on executed quantity
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

def calculate_position_size(
    capital: float,
    risk_pct: float,
    entry_prem: float,
    sl_prem: float,
    lot_size: int = LOT_SIZE,
    current_drawdown_pct: float = 0.0
) -> Dict[str, Any]:
    """
    Computes exact lot sizing with Non-Linear Drawdown Dampener and 10% MDD hard circuit breaker.
    Enforces maximum portfolio margin allocation constraints (max 35% margin).
    """
    # 4-Tier Institutional Drawdown Curve
    if current_drawdown_pct >= MAX_TOLERABLE_MDD:
        dd_dampener = 0.0  # Terminal Circuit Breaker
    elif current_drawdown_pct <= 0.03:
        dd_dampener = 1.0
    elif current_drawdown_pct <= 0.06:
        dd_dampener = 1.0 - ((current_drawdown_pct - 0.03) / 0.03) * 0.50
    else:
        norm_val = (current_drawdown_pct - 0.06) / (MAX_TOLERABLE_MDD - 0.06)
        dd_dampener = max(0.50 * (1.0 - norm_val) ** 2, 0.05)
        
    adjusted_risk_pct = risk_pct * dd_dampener
    max_risk_rupees = capital * adjusted_risk_pct
    # Minimum 12-point option stop-loss distance to prevent over-allocation into high lot counts
    risk_per_share = max(entry_prem - sl_prem, 12.0)
    risk_per_lot = risk_per_share * lot_size
    
    lots = int(max_risk_rupees // risk_per_lot) if risk_per_lot > 0 else 0
    # Margin Constraint: Maximum 35% of total capital in any single trade
    max_margin_lots = int((capital * 0.35) // (entry_prem * lot_size)) if entry_prem > 0 else lots
    lots = min(lots, max_margin_lots, 15)  # Cap at 15 lots max on intraday trades
    
    # Strict Capital Risk: Never force lots if risk exceeds budget
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
        "dd_dampener": round(dd_dampener, 3)
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
    - Normal regimes: Minimizes distance to optimal Target Delta ~0.58 in [0.50, 0.65].
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
    max_spread_profit_pts = round(max_spread_width - net_effective_cost, 2)
    max_spread_pnl_rupees = round(max_spread_profit_pts * total_qty, 2)
    
    net_delta = round(greeks_k1_now["delta"] - greeks_k2_now["delta"], 3)
    net_theta = round(greeks_k1_now["theta"] - greeks_k2_now["theta"], 2)
    net_vega = round(greeks_k1_now["vega"] - greeks_k2_now["vega"], 2)
    
    return {
        "status": "CONVERTED_SPREAD",
        "spread_type": "Bull Call Spread" if is_call else "Bear Put Spread",
        "long_leg": f"NIFTY {k1} {'CE' if is_call else 'PE'} (Bought @ ₹{entry_prem_k1:.2f})",
        "short_leg": f"NIFTY {k2} {'CE' if is_call else 'PE'} (Sold @ ₹{prem_k2_now:.2f})",
        "short_strike": k2,
        "short_premium_collected": prem_k2_now,
        "net_effective_cost_per_share": net_effective_cost,
        "max_spread_width_pts": max_spread_width,
        "max_spread_profit_pts": max_spread_profit_pts,
        "max_spread_pnl_rupees": max_spread_pnl_rupees,
        "net_delta": net_delta,
        "net_theta_daily": net_theta,
        "net_vega": net_vega,
        "is_theta_positive": net_theta >= 0,
        "execution_guidance": f"SELL {lots} Lots of NIFTY {k2} {'CE' if is_call else 'PE'} @ ₹{prem_k2_now:.2f}. Theta decay is neutralized ({net_theta:+.2f}/sh/day)."
    }

def generate_option_trade_ticket(
    spot: float,
    signal: Signal,
    capital: float = DEFAULT_CAPITAL,
    current_drawdown_pct: float = 0.0,
    t_days: float = 4.0,
    iv: float = DEFAULT_IV,
    is_0dte_afternoon: bool = False
) -> Dict[str, Any]:
    """Translates spot setups into institutional Option Trade Ticket with convex Taylor Series modeling."""
    if signal.signal_type == SignalType.WAIT:
        return {"status": "WAIT", "message": signal.reason}
        
    is_call = signal.signal_type in [SignalType.LONG, SignalType.LONG_3PM]
    strike_info = select_institutional_strike(spot, is_call=is_call, t_days=t_days, iv=iv, is_0dte_afternoon=is_0dte_afternoon)
    
    delta = abs(strike_info["delta"])
    gamma = strike_info["gamma"]
    theta = strike_info["theta"]
    entry_prem = strike_info["price"]
    
    # Corrected 2nd-Order Taylor Series: Long options benefit from positive Gamma convexity on downside
    spot_risk = abs(signal.entry_price - signal.sl_price)
    convexity_benefit = 0.5 * gamma * (spot_risk ** 2)
    theta_risk = abs(theta) * 0.15
    option_risk = max((spot_risk * delta) - convexity_benefit + theta_risk, 2.0)
    sl_prem = max(round(entry_prem - option_risk, 2), 2.0)
    
    spot_target1_diff = abs(signal.target_1 - signal.entry_price)
    opt_gain_t1 = (spot_target1_diff * delta) + (0.5 * gamma * (spot_target1_diff ** 2))
    target1_prem = round(entry_prem + opt_gain_t1, 2)
    
    spot_target2_diff = abs(signal.target_2 - signal.entry_price)
    opt_gain_t2 = (spot_target2_diff * delta) + (0.5 * gamma * (spot_target2_diff ** 2))
    target2_prem = round(entry_prem + opt_gain_t2, 2)
    
    sizing = calculate_position_size(capital, MAX_RISK_PCT, entry_prem, sl_prem, LOT_SIZE, current_drawdown_pct)
    tca = calculate_adaptive_tca_friction(entry_prem, target1_prem, target1_prem, sizing["total_qty"], sizing["lots"], part_booked=True, iv=iv, is_0dte_afternoon=is_0dte_afternoon)
    
    half_lots = max(sizing["lots"] // 2, 1) if sizing["lots"] > 0 else 0
    
    return {
        "status": "READY",
        "signal": signal.signal_type.value,
        "symbol": strike_info["symbol"],
        "strike": strike_info["strike"],
        "option_type": strike_info["option_type"],
        "regime_mode": strike_info["regime_mode"],
        "delta": strike_info["delta"],
        "gamma": strike_info["gamma"],
        "theta_decay_daily": strike_info["theta"],
        "vega": strike_info["vega"],
        "vanna": strike_info["vanna"],
        "charm": strike_info["charm"],
        "volga": strike_info["volga"],
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
            "part_book_50_pct": f"Option A (Outright): Book 50% ({half_lots} lots) at ₹{target1_prem} and move SL to ₹{entry_prem}",
            "free_vertical_spread": f"Option B (Institutional Spread): Sell {sizing['lots']} lots of OTM Strike {strike_info['strike'] + (100 if is_call else -100)} at T1 to lock in a Free Vertical Spread",
            "trailing_rule": "Trail remaining position on 1-minute 21 EMA or Session AVWAP",
            "profit_ratchet": "Lock 65% of peak gains once +1.5R (+1.5%) session profit is achieved"
        }
    }
