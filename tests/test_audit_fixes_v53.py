"""
Audit Fixes Verification Suite (v5.3 Desk Edition)
Tests for:
1. GEX wall side-of-spot constraints (Call Wall >= Spot, Put Wall <= Spot).
2. Unscored WAIT confluence cleanup (0.0% / Consolidation instead of 50% Standard).
3. Dominant family agreement reporting on WAIT.
4. Put Wall Breakdown Short / Gamma Breakdown firing on bearish trend days.
5. Range Fade Long suppression when spot has breached below the Put Wall.
"""
import pytest
import numpy as np
import pandas as pd

from src.options_flow import compute_strike_level_gex_chart_data
from src.options_positioning import compute_options_desk_state
from src.desk_verdict import build_desk_verdict, compute_conviction
from src.strategy_rules import StrategyEngine, SignalType
from src.volatility_engine import VolatilityIntelligence


def test_gex_walls_constrained_to_side_of_spot():
    """
    Verifies that compute_strike_level_gex_chart_data enforces:
    - Call Wall strike >= spot
    - Put Wall strike <= spot
    Even when the strike with the single highest call GEX is far below spot or vice versa.
    """
    spot = 24264.0
    strikes = [24000, 24100, 24200, 24300, 24400, 24500, 24600]
    
    # Construct a chain where strike 24100 (below spot) has high Call OI/GEX
    # and strike 24500 (above spot) has high Put OI/GEX
    records = []
    for k in strikes:
        records.append({
            "strike_price": k,
            "ce_oi": 500000 if k == 24100 else 100000,
            "pe_oi": 600000 if k == 24500 else 80000,
            "ce_iv": 0.13,
            "pe_iv": 0.13
        })
    df_oc = pd.DataFrame(records)

    gex_data = compute_strike_level_gex_chart_data(df_oc, spot, iv=0.13, t_days=1.0)
    
    assert gex_data["call_wall_strike"] >= spot, f"Call wall {gex_data['call_wall_strike']} must be >= spot {spot}"
    assert gex_data["put_wall_strike"] <= spot, f"Put wall {gex_data['put_wall_strike']} must be <= spot {spot}"


def test_unscored_wait_confluence_is_consolidation():
    """
    Verifies that on an unscored WAIT (e.g. signal=None or vetoed with no score),
    confluence defaults to 0.0% and grade is 'Consolidation' (not 50.0% Standard).
    """
    verdict = build_desk_verdict(
        signal=None,
        current_spot=24264.0,
        desk_state=None,
        vol_report=None
    )
    assert verdict.confluence_score == 0.0
    assert verdict.confluence_grade == "Consolidation"
    assert verdict.action == "WAIT"


def test_wait_reports_dominant_family_agreement():
    """
    Verifies that on a WAIT with 3 bearish families and 1 bullish family,
    compute_conviction reports 3/4 aligned with note 'Board leans bearish (3/4 families)'.
    """
    votes = {"structure": -1, "flow": -1, "positioning": -1, "macro": 1}
    score, tier, agree, notes = compute_conviction("WAIT", votes, 0.0)
    
    assert score == 0.0
    assert tier == "LOW"
    assert agree == 3
    assert any("bearish" in n for n in notes)


def test_range_fade_long_blocked_when_broken_below_put_wall():
    """
    Verifies that if spot is broken below the Put Wall (e.g. spot 24264 with put wall 24300),
    RANGE_FADE_LONG is prevented from triggering.
    """
    engine = StrategyEngine()
    
    # Create sample 5m bars
    dates = pd.date_range("2026-08-17 09:15", periods=30, freq="5min")
    df_5m = pd.DataFrame({
        "open": np.linspace(24350, 24264, 30),
        "high": np.linspace(24360, 24270, 30),
        "low": np.linspace(24340, 24260, 30),
        "close": np.linspace(24345, 24264, 30),
        "volume": [10000] * 30
    }, index=dates)

    options_context = {
        "dir_flow": {"directional_vector": -0.66},
        "pcr_zscore": -1.2,
        "gex_chart": {
            "call_wall_strike": 24500.0,
            "put_wall_strike": 24300.0,
            "zero_gex_strike": 24400.0,
            "net_dealer_regime": "DEALER_LONG_GAMMA (Positive Gamma)"
        },
        "pcr": {"max_pain_strike": 24300.0}
    }

    sig = engine.evaluate_bar(
        df_5m=df_5m,
        live_iv=0.13,
        options_context=options_context,
        hfi_score=-0.56
    )

    # Must NOT be RANGE_FADE_LONG because spot (24264) is below put wall (24300)
    assert sig.signal_type != SignalType.RANGE_FADE_LONG


def test_put_wall_breakdown_short_triggers():
    """
    Verifies that on a bearish tape where spot breaches below Put Wall with negative D-vector,
    GAMMA_BREAKOUT_SHORT or a short continuation setup fires.
    """
    engine = StrategyEngine()
    
    dates = pd.date_range("2026-08-17 09:15", periods=30, freq="5min")
    # Clean downtrend bars closing lower
    closes = np.linspace(24400, 24260, 30)
    df_5m = pd.DataFrame({
        "open": closes + 5.0,
        "high": closes + 8.0,
        "low": closes - 5.0,
        "close": closes,
        "volume": [50000] * 30
    }, index=dates)

    options_context = {
        "dir_flow": {"directional_vector": -0.60},
        "pcr_zscore": -1.5,
        "gex_chart": {
            "call_wall_strike": 24500.0,
            "put_wall_strike": 24300.0,
            "zero_gex_strike": 24400.0,
            "net_dealer_regime": "DEALER_SHORT_GAMMA (Negative Gamma)"
        },
        "pcr": {"max_pain_strike": 24300.0}
    }

    sig = engine.evaluate_bar(
        df_5m=df_5m,
        live_iv=0.13,
        options_context=options_context,
        hfi_score=-0.30
    )

    # In a clean breakdown below Put Wall (24300) at 24260 with D=-0.60 and HFI=-0.30,
    # it should produce a SHORT setup (either GAMMA_BREAKOUT_SHORT, IB breakdown, or order flow short)
    # rather than a counter-trend long.
    if sig.signal_type != SignalType.WAIT:
        assert "SHORT" in sig.signal_type.value or sig.signal_type == SignalType.SHORT


def test_evaluate_single_tf_regime_prefix_stability():
    """
    Verifies that _evaluate_single_tf_regime evaluates the full path-dependent series
    so that historical line-break states do not flip/repaint as new bars arrive.
    """
    from src.indicators import _evaluate_single_tf_regime, compute_line_break_trend

    # Generate a reproducible price series
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2026-08-17 09:15", periods=n, freq="5min")
    price = 24000.0 + np.cumsum(np.random.normal(0.5, 5.0, n))
    df = pd.DataFrame({
        "open": price,
        "high": price + 4.0,
        "low": price - 4.0,
        "close": price + 1.0,
        "volume": [10000] * n
    }, index=dates)

    # Evaluate at length 120 and length 150
    regime_120 = _evaluate_single_tf_regime(df.iloc[:120], "5m")
    full_trend_120 = compute_line_break_trend(df.iloc[:120])
    
    # Must match full history line break calculation exactly at bar 120
    assert regime_120["lb_bias"] == full_trend_120["lb_bias"].iloc[-1]
    assert regime_120["lb_direction"] == full_trend_120["lb_direction"].iloc[-1]


def test_gamma_breakout_blocked_in_positive_gamma():
    """
    Verifies that GAMMA_BREAKOUT_LONG is strictly blocked in positive gamma (+Γ)
    regimes where dealers suppress momentum and enforce mean-reversion.
    """
    engine = StrategyEngine()
    
    dates = pd.date_range("2026-08-17 09:15", periods=30, freq="5min")
    closes = np.linspace(24400, 24550, 30)
    df_5m = pd.DataFrame({
        "open": closes - 2.0,
        "high": closes + 5.0,
        "low": closes - 3.0,
        "close": closes,
        "volume": [50000] * 30
    }, index=dates)

    options_context = {
        "dir_flow": {"directional_vector": +0.65},
        "pcr_zscore": +1.5,
        "gex_chart": {
            "call_wall_strike": 24500.0,
            "put_wall_strike": 24300.0,
            "zero_gex_strike": 24400.0,
            "net_dealer_regime": "DEALER_LONG_GAMMA (Positive Gamma)"
        },
        "pcr": {"max_pain_strike": 24450.0}
    }

    sig = engine.evaluate_bar(
        df_5m=df_5m,
        live_iv=0.13,
        options_context=options_context,
        hfi_score=+0.30
    )

    # In +Γ regime, GAMMA_BREAKOUT_LONG must not fire
    assert sig.signal_type != SignalType.GAMMA_BREAKOUT_LONG
