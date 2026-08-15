"""
Unit tests for Institutional v5.7 Academic Alpha Upgrades:
- Yang-Zhang Realized Volatility
- Variance Risk Premium (VRP)
- Rough Volatility (Fractional Brownian Motion)
- Hawkes-Decay Order Flow Imbalance
- Jade Lizard Skew Arbitrage
- Index-to-Constituent Dispersion
- Gate 11 VRP Backwardation Veto
"""

import math
import pytest
import numpy as np
import pandas as pd

from src.volatility_engine import VolatilityIntelligence
from src.options_engine import construct_jade_lizard
from src.regime_switching import MarkovRegimeSwitcher
from src.indicators import compute_order_flow_imbalance
from src.institutional_flow import compute_dispersion_arbitrage_signal
from src.strategy_rules import StrategyEngine, Signal, SignalType


def test_yang_zhang_realized_volatility():
    # Construct 30 bars of synthetic OHLC data with an overnight gap
    np.random.seed(42)
    n_bars = 30
    base_price = 24500.0
    opens = [base_price]
    highs = []
    lows = []
    closes = []

    for i in range(n_bars):
        o = opens[-1] if i > 0 else base_price
        ret = np.random.normal(0, 15)
        c = o + ret
        h = max(o, c) + abs(np.random.normal(0, 10))
        l = min(o, c) - abs(np.random.normal(0, 10))
        highs.append(h)
        lows.append(l)
        closes.append(c)
        if i < n_bars - 1:
            # Introduce overnight gap between bars
            gap = np.random.normal(0, 20)
            opens.append(c + gap)

    df = pd.DataFrame({
        "open": opens[:n_bars],
        "high": highs,
        "low": lows,
        "close": closes
    })

    yz_res = VolatilityIntelligence.compute_yang_zhang_volatility(df, window=20)

    assert yz_res["is_yang_zhang"] is True
    assert 0.05 <= yz_res["realized_vol_yz"] <= 1.50
    assert yz_res["rv_overnight"] >= 0.0
    assert yz_res["rv_rogers_satchell"] >= 0.0
    assert 0.0 < yz_res["k_weight"] < 1.0


def test_variance_risk_premium_computation():
    # Case 1: Rich Volatility Premium (IV = 16%, RV_YZ = 11%)
    vrp_rich = VolatilityIntelligence.compute_variance_risk_premium(implied_vol=0.16, realized_vol_yz=0.11)
    assert vrp_rich["is_positive_vrp"] is True
    assert vrp_rich["vrp"] == 0.05
    assert vrp_rich["regime"] == "HIGH_VARIANCE_PREMIUM"

    # Case 2: Negative VRP / Backwardation (IV = 12%, RV_YZ = 18%)
    vrp_neg = VolatilityIntelligence.compute_variance_risk_premium(implied_vol=0.12, realized_vol_yz=0.18)
    assert vrp_neg["is_positive_vrp"] is False
    assert vrp_neg["vrp"] == -0.06
    assert vrp_neg["regime"] == "NEGATIVE_VRP_BACKWARDATION"


def test_rough_volatility_regime_switch():
    switcher = MarkovRegimeSwitcher(base_iv=0.135)
    prices = pd.Series(np.linspace(24000, 24500, 40))
    df = pd.DataFrame({"close": prices})

    # Test with Rough Volatility (Hurst H = 0.12 < 0.25)
    res_rough = switcher.infer_regimes(df, window=30, hurst_exponent=0.12)
    assert res_rough["is_rough_volatility"] is True
    assert res_rough["hurst_h"] == 0.12
    assert "Rough Volatility" in res_rough["advice"]

    # Test with standard Brownian motion / trend (Hurst H = 0.65)
    res_smooth = switcher.infer_regimes(df, window=30, hurst_exponent=0.65)
    assert res_smooth["is_rough_volatility"] is False


def test_hawkes_order_flow_imbalance():
    # Construct 20 bars of OHLCV
    n = 20
    df = pd.DataFrame({
        "open": np.linspace(24500, 24550, n),
        "high": np.linspace(24510, 24560, n),
        "low": np.linspace(24490, 24540, n),
        "close": np.linspace(24505, 24555, n),
        "volume": np.full(n, 1000.0)
    })

    ofi_res = compute_order_flow_imbalance(df, decay_lambda=0.15)

    assert "hawkes_ofi" in ofi_res
    assert "is_hawkes_surge" in ofi_res
    assert isinstance(ofi_res["hawkes_ofi"], float)


def test_construct_jade_lizard():
    spot = 24500.0
    res = construct_jade_lizard(
        spot=spot,
        put_offset=150,
        call_short_offset=150,
        call_wing_width=100,
        t_days=4.0,
        iv=0.14,
        put_skew_multiplier=1.20
    )

    assert res["status"] == "STRUCTURED"
    assert res["strategy"] == "JADE_LIZARD_SKEW_ARBITRAGE"
    assert res["total_net_credit_pts"] > 0
    assert "short_put" in res["legs"]
    assert "short_call" in res["legs"]
    assert "long_call" in res["legs"]
    assert res["legs"]["short_put"]["strike"] == 24350
    assert res["legs"]["short_call"]["strike"] == 24650
    assert res["legs"]["long_call"]["strike"] == 24750
    # Net Theta should be positive (time decay harvesting)
    assert res["net_theta_daily"] > 0


def test_dispersion_arbitrage_signal():
    # Case 1: Index IV (18%) >> Basket RV (12%)
    disp_sell = compute_dispersion_arbitrage_signal(nifty_iv=0.18, hfi_realized_vol=0.12)
    assert disp_sell["is_arbitrage_opportunity"] is True
    assert disp_sell["regime"] == "DISPERSION_SELL_INDEX_VOL"
    assert disp_sell["spread_zscore"] > 1.50

    # Case 2: Equilibrium (Index IV 14%, Basket RV 12%)
    disp_fair = compute_dispersion_arbitrage_signal(nifty_iv=0.14, hfi_realized_vol=0.12)
    assert disp_fair["is_arbitrage_opportunity"] is False
    assert disp_fair["regime"] == "DISPERSION_FAIR_VALUE"


def test_vrp_backwardation_gate():
    engine = StrategyEngine()
    skew_info = {"is_crash_hedging": False, "skew_zscore": 0.0}
    vpin_info = {"vpin": 0.30}
    gex_info = {"is_positive_gamma": True, "call_wall_strike": 25000.0, "put_wall_strike": 24000.0}
    htf_regime = {"htf_aligned_long": True, "htf_aligned_short": False}
    
    # Negative VRP (IV in backwardation)
    options_context = {"vrp": -0.05}

    passed, reason, audit = engine._apply_universal_gates(
        candidate_direction="LONG",
        close=24500.0,
        skew_info=skew_info,
        vpin_info=vpin_info,
        hfi_score=0.0,
        gex_info=gex_info,
        htf_regime=htf_regime,
        options_context=options_context,
        candidate_signal_type="RANGE_FADE_LONG"
    )

    assert passed is False
    assert audit["veto_gate"] == "VRP_BACKWARDATION_VETO"
    assert "Variance Risk Premium Gate" in reason
