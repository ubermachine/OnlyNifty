"""
Latency Profiling Script for OnlyNifty Pipeline
Measures execution time of all mathematical engines and UI subcomponents.
"""
import sys
import os
sys.path.insert(0, os.path.abspath("."))
import time
import cProfile
import pstats
import io
import numpy as np
import pandas as pd

from src.data_engine import DataEngine
from src.indicators import (
    compute_ema, compute_vakc_envelopes, compute_vwap, compute_cpr,
    compute_volume_profile, compute_hurst_exponent, compute_order_flow_imbalance,
    compute_dealer_gex, compute_multi_timeframe_regime, detect_stacked_order_flow_imbalances,
    compute_dfa_alpha, compute_vpin_toxicity
)
from src.regime_switching import KalmanFilterTrendEstimator, MarkovRegimeSwitcher
from src.volatility_engine import VolatilityIntelligence
from src.options_engine import calculate_pcr_and_max_pain
from src.options_flow import (
    compute_strike_level_gex_chart_data, compute_oi_based_range_forecast,
    compute_short_term_directional_vector
)
from src.options_positioning import compute_options_desk_state
from src.strategy_rules import StrategyEngine
from src.desk_verdict import build_desk_verdict


def profile_full_run():
    engine = DataEngine()
    df = engine.generate_synthetic_nifty(bars=150, interval_mins=5)
    spot = float(df.iloc[-1]["close"])
    
    chain_df = pd.DataFrame([
        {"strike_price": spot + i * 50, "ce_oi": 100000 + abs(i)*10000, "pe_oi": 120000 - i*5000,
         "ce_change_oi": 5000, "pe_change_oi": -2000, "ce_iv": 0.13, "pe_iv": 0.13, "ce_volume": 5000, "pe_volume": 4000}
        for i in range(-20, 21)
    ])

    timings = {}

    def measure(name, func, *args, **kwargs):
        t0 = time.perf_counter()
        res = func(*args, **kwargs)
        t1 = time.perf_counter()
        timings[name] = (t1 - t0) * 1000.0
        return res

    # Individual Indicators
    measure("compute_ema x3", lambda: [compute_ema(df["close"], p) for p in [21, 55, 200]])
    measure("compute_vakc_envelopes", compute_vakc_envelopes, df, iv=0.13)
    measure("compute_vwap", compute_vwap, df)
    measure("compute_cpr", compute_cpr, df)
    measure("compute_volume_profile", compute_volume_profile, df)
    measure("compute_hurst_exponent", compute_hurst_exponent, df["close"])
    measure("compute_order_flow_imbalance", compute_order_flow_imbalance, df)
    measure("compute_dealer_gex", compute_dealer_gex, spot)
    measure("compute_multi_timeframe_regime", compute_multi_timeframe_regime, df)
    measure("detect_stacked_order_flow_imbalances", detect_stacked_order_flow_imbalances, df)
    measure("compute_dfa_alpha", compute_dfa_alpha, df["close"])
    measure("compute_vpin_toxicity", compute_vpin_toxicity, df)
    
    # Statistical Engines
    kalman = KalmanFilterTrendEstimator()
    markov = MarkovRegimeSwitcher()
    vol_intel = VolatilityIntelligence()
    
    measure("kalman_filter_series", kalman.filter_series, df["close"])
    measure("markov_infer_regimes", markov.infer_regimes, df)
    measure("vol_intelligence_report", vol_intel.generate_vol_intelligence_report, df["close"], 0.13)

    # Options Flow & Desk State
    pcr_res = measure("calculate_pcr_and_max_pain", calculate_pcr_and_max_pain, chain_df)
    gex_res = measure("compute_strike_level_gex_chart_data", compute_strike_level_gex_chart_data, chain_df, spot, 0.13, 1.0)
    rf_res = measure("compute_oi_based_range_forecast", compute_oi_based_range_forecast, chain_df, spot, spot)
    dir_res = measure("compute_short_term_directional_vector", compute_short_term_directional_vector, spot, df, chain_df, 0.13, 0.0)
    desk_state = measure(
        "compute_options_desk_state",
        compute_options_desk_state,
        option_chain_df=chain_df,
        spot=spot,
        df_ohlcv=df,
        pcr_analytics=pcr_res,
        dir_flow_res=dir_res,
        range_fc_res=rf_res,
        gex_chart_res=gex_res,
        live_iv=0.13,
        hfi_score=0.0
    )

    # Strategy Engine & Desk Verdict
    strat = StrategyEngine()
    sig = measure("strategy_evaluate_bar", strat.evaluate_bar, df, live_iv=0.13, options_context={"dir_flow": dir_res, "gex_chart": gex_res})
    verdict = measure("build_desk_verdict", build_desk_verdict, sig, current_spot=spot, desk_state=desk_state)

    print("\n--- DETAILED COMPONENT LATENCIES (ms) ---")
    total = 0.0
    for k, v in sorted(timings.items(), key=lambda x: x[1], reverse=True):
        print(f"{k:<45}: {v:7.2f} ms")
        total += v
    print(f"\nTotal Pipeline Calculation Time: {total:.2f} ms")


if __name__ == "__main__":
    profile_full_run()
