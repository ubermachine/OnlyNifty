"""Unit tests for OnlyNifty v3.3 Multi-Timeframe Confluence, Stacked Footprint Order Flow, 0DTE Gamma Scalper, and Bhavcopy Downloader."""

import pytest
import numpy as np
import pandas as pd
from datetime import date

from src.indicators import (
    compute_multi_timeframe_regime,
    detect_stacked_order_flow_imbalances
)
from src.strategy_rules import StrategyEngine, SignalType
from src.options_engine import (
    compute_0dte_gamma_scalp_parameters,
    calculate_adaptive_tca_friction_multi_tier,
    generate_option_trade_ticket
)
from src.data_engine import DataEngine


def test_multi_timeframe_alignment_engine():
    # Construct 100 bars of strongly trending 5m data
    dates = pd.date_range("2026-08-14 09:15", periods=100, freq="5min")
    close_prices = np.linspace(24300, 24650, 100)
    df_5m = pd.DataFrame({
        "open": close_prices - 2.0,
        "high": close_prices + 5.0,
        "low": close_prices - 4.0,
        "close": close_prices,
        "volume": np.random.randint(100000, 300000, 100)
    }, index=dates)

    htf = compute_multi_timeframe_regime(df_5m)
    assert "htf_aligned_long" in htf
    assert "htf_aligned_short" in htf
    assert htf["htf_aligned_long"] is True
    assert htf["htf_aligned_short"] is False
    assert "FULL_BULLISH_CONFLUENCE" in htf["confluence_regime"] or "HTF_BULLISH_ALIGNMENT" in htf["confluence_regime"]


def test_stacked_footprint_order_flow_imbalances_and_absorption():
    dates = pd.date_range("2026-08-14 09:15", periods=20, freq="5min")
    # Simulate buyer absorption at support 24400 (long lower wick, closing at 24410)
    opens = [24420] * 19 + [24405]
    highs = [24430] * 19 + [24415]
    lows = [24410] * 19 + [24385]  # Deep probe below 24400
    closes = [24425] * 19 + [24412] # Closed back above 24400 with 60% lower wick
    volumes = [150000] * 20

    df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}, index=dates)
    key_levels = {"CPR_PIVOT": 24400.0, "VAH": 24480.0, "VAL": 24390.0}

    of = detect_stacked_order_flow_imbalances(df, key_levels=key_levels)
    assert of["absorption_event"] is not None
    assert of["absorption_event"]["type"] == "BUYER_ABSORPTION"
    assert of["absorption_event"]["side"] == "LONG"
    assert of["order_flow_bias"] == "BULLISH_ABSORPTION"


def test_0dte_gamma_scalp_parameters():
    spot = 24500.0
    strike = 24500.0
    
    scalp = compute_0dte_gamma_scalp_parameters(
        spot=spot,
        strike=strike,
        dte_days=0.08,
        iv=0.12,
        is_call=True,
        current_time_str="13:30",
        atr=30.0
    )

    assert scalp["regime"] == "0DTE_GAMMA_SQUEEZE_AFTERNOON"
    assert scalp["is_0dte_afternoon"] is True
    assert scalp["gamma_explosion_multiplier"] >= 1.5
    assert scalp["tightened_sl"]["max_spot_sl_pts"] <= 18.0
    assert "strike_playbook" in scalp
    assert "deep_itm_synthetic" in scalp["strike_playbook"]
    assert "atm_momentum_scalp" in scalp["strike_playbook"]
    assert scalp["gamma_surge_targets"]["target_1_gain_pct"] >= 20.0


def test_historical_bhavcopy_downloader_methods():
    engine = DataEngine(use_cache=False)
    today = date(2026, 8, 14)
    
    df_fo = engine.download_fno_bhavcopy(today)
    assert df_fo is not None
    assert not df_fo.empty
    assert "strike_pr" in df_fo.columns or "strike" in df_fo.columns

    df_idx = engine.download_nifty_index_bhavcopy(today)
    assert df_idx is not None
    assert not df_idx.empty
    assert "closing_index_val" in df_idx.columns or "close" in df_idx.columns or "index_name" in df_idx.columns


def test_multi_tier_tca_friction_calculation():
    friction = calculate_adaptive_tca_friction_multi_tier(
        entry_prem=150.0,
        t1_prem=190.0,
        t2_prem=230.0,
        final_exit_prem=280.0,
        total_qty=150,
        lots=6,
        t1_hit=True,
        t2_hit=True,
        is_pyramided=False,
        iv=0.12
    )

    assert friction["total_friction"] > 0.0
    assert friction["brokerage"] >= 60.0  # 3 orders (Buy, Sell T1, Sell T2/Runner)
    assert friction["stt"] > 0.0
    assert friction["gst"] > 0.0
