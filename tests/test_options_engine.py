import pytest
from src.options_engine import (
    black_scholes_greeks,
    select_institutional_strike,
    calculate_position_size,
    generate_option_trade_ticket
)
from src.strategy_rules import Signal, SignalType

def test_black_scholes_greeks():
    # ATM Call: Spot=24500, Strike=24500, T=5 days (5/365), r=6.5%, sigma=12%
    greeks = black_scholes_greeks(spot=24500.0, strike=24500.0, t_days=5.0, r=0.065, sigma=0.12, is_call=True)
    assert 0.48 <= greeks["delta"] <= 0.58
    assert greeks["gamma"] > 0
    assert greeks["theta"] < 0
    assert greeks["price"] > 0

def test_select_institutional_strike_call():
    # Spot 24530, Long -> Deep ITM Synthetic CE (e.g. 24350 / 24400 CE) with delta ~0.65-0.85 (cuts extrinsic decay)
    res = select_institutional_strike(spot=24530.0, is_call=True)
    assert res["strike"] <= 24530
    assert 0.65 <= abs(res["delta"]) <= 0.88
    assert res["option_type"] == "CE"

def test_select_institutional_strike_put():
    # Spot 24480, Short -> Deep ITM Synthetic PE (e.g. 24650 / 24700 PE) with delta ~0.65-0.85
    res = select_institutional_strike(spot=24480.0, is_call=False)
    assert res["strike"] >= 24480
    assert 0.65 <= abs(res["delta"]) <= 0.88
    assert res["option_type"] == "PE"

def test_calculate_position_size():
    # Capital: 5,00,000, 1% Risk = 5,000. Entry: 150, SL: 110 (Risk=40). Lot size: 25 (Risk per lot = 1000)
    # Lots = 5000 / 1000 = 5 lots
    sizing = calculate_position_size(capital=500000.0, risk_pct=0.01, entry_prem=150.0, sl_prem=110.0, lot_size=25)
    assert sizing["lots"] == 5
    assert sizing["total_qty"] == 125
    assert sizing["actual_risk_rupees"] <= 5000.0

def test_generate_option_trade_ticket_long():
    signal = Signal(
        signal_type=SignalType.LONG,
        entry_price=24500.0,
        sl_price=24450.0,
        target_1=24580.0,
        target_2=24650.0,
        reason="Test Long",
        htf_aligned=True,
        fib_retracement=0.55,
        details={}
    )
    ticket = generate_option_trade_ticket(spot=24500.0, signal=signal, capital=500000.0)
    assert ticket["status"] == "READY"
    assert "CE" in ticket["symbol"]
    assert ticket["entry_premium"] > ticket["sl_premium"]
    assert ticket["target1_premium"] > ticket["entry_premium"]
    assert ticket["lots"] > 0
    assert "part_book_50_pct" in ticket["execution_rules"]
