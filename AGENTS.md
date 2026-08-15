# OnlyNifty (v5.1) — Fast Repository Map for AI Agents

> **Agent Invariant:** Read THIS file first before exploring other files. It provides 100% of the architecture, module index, data contracts, and execution commands in ~2,000 tokens.

---

## 1. Quick Start & Test Commands (PowerShell Invariants)
- **Run all tests:** `python -m pytest tests/ -v --tb=short`
- **Run single test file:** `python -m pytest tests/test_v51_institutional_flow.py -v`
- **Full Forensic 6-Pillar Verification:** `python verify_all_modules.py`
- **Streamlit App Launch:** `streamlit run app.py`
- **PowerShell Syntax Rule:** Never use `&&` (use `;`). Never pipe to `head`/`tail`/`grep` (use `Select-Object -First N`, `Select-Object -Last N`, `Select-String`).
- **Streamlit Syntax Rule:** Always use `width='stretch'` (never `use_container_width`).

---

## 2. System Architecture & End-to-End Pipeline

```
[ Market Ingestion: yfinance / jugaad-data / Synthetic ]
                           │
                           ▼
             [ DataEngine: clean_ohlcv (IST) ]
                           │
       ┌───────────────────┴───────────────────┐
       ▼                                       ▼
[ Microstructure & Technicals ]      [ Statistical & Volatility Engines ]
 • EMA (21, 55, 200)                  • Kalman Velocity Estimator
 • VAKC Keltner Envelopes             • Markov 3-State Regime Switcher
 • Session AVWAP ±2σ                  • DFA Long-Memory Alpha
 • CPR Pivots & Volume Profile        • VPIN Flow Toxicity
 • Stacked Footprint Imbalances       • Volatility Intelligence (IV-RV)
 • Heavyweight Flow Index (HFI)       • Institutional Flow Engine (FII L/S)
       └───────────────────┬───────────────────┘
                           │
                           ▼
               [ StrategyEngine.evaluate_bar ]
             (Regime-Adaptive Multi-Strategy)
                           │
                           ▼
          [ generate_option_trade_ticket ]
         (Taylor-Series Convexity & TCA Fees)
                           │
       ┌───────────────────┴───────────────────┐
       ▼                                       ▼
[ Options Flow & Microstructure ]    [ LiveSignalJournal & Lifecycle ]
 • Combined ATM Straddle Corridor     • Real-time bar-by-bar logger
 • 4-Quadrant ΔOI & Traps             • Deduplication state machine
 • Strike OI Change Heatmap           • MFE / MAE excursion tracking
 • Dealer GEX Profile & Walls         • SignalPerformanceAnalyzer
 • 5-Pillar Direction Vector (D)      • SHA-256 tamper-evident chain
       └───────────────────┬───────────────────┘
                           │
                           ▼
          [ Streamlit Cockpit & 6 Interactive Tabs ]
```

---

## 3. Module Catalog (`src/`)

| File | Primary Classes / Functions | Key Outputs & Contracts |
|---|---|---|
| **`src/config.py`** | Configuration constants | `DEFAULT_CAPITAL=500000`, `LOT_SIZE=25`, `DEFAULT_IV=0.135`, `EMA_FAST=21`, `EMA_SLOW=200`, `KELLY_FRACTION=0.25` |
| **`src/data_engine.py`** | `DataEngine` | `clean_ohlcv(df)`, `fetch_yfinance_nifty()`, `fetch_live_nse_option_chain()`, `fetch_multi_expiry_option_chain()`, `get_participant_oi_snapshot()`, `fetch_heavyweight_flow_index()`, `fetch_sectoral_pulse()`, `fetch_pre_open_gap()` |
| **`src/indicators.py`** | Technical & quantitative indicators | `compute_ema()`, `compute_vakc_envelopes()`, `compute_vwap()`, `compute_cpr()`, `compute_fibonacci_levels()`, `compute_volume_profile()`, `compute_hurst_exponent()`, `compute_order_flow_imbalance()`, `compute_dealer_gex()`, `compute_multi_timeframe_regime()`, `detect_stacked_order_flow_imbalances()`, `compute_initial_balance_and_day_type()`, `compute_vwap_multi_dispersion_and_half_life()`, `compute_dfa_alpha()`, `compute_vpin_toxicity()` |
| **`src/strategy_rules.py`** | `StrategyEngine`, `Signal`, `SignalType` | `evaluate_bar(df, live_iv)`: 4-regime router (`LOW_VOL_TRENDING` -> IB Breakout, `MEAN_REVERTING_CHOP` -> VAL/VAH Mean-Reversion, `Golden Pocket` -> Fibonacci Pullback, `High Vol` -> Convexity Scalp). Enforces Lunch Lull filter (11:30-13:00 IST). |
| **`src/volatility_engine.py`** | `VolatilityIntelligence` | `compute_realized_volatility()`, `compute_iv_rv_spread()`, `compute_iv_percentile()`, `compute_expected_vs_actual_move()`, `compute_intraday_quality_score()`, `compute_volatility_cone()`, `compute_rv_term_structure()`, `compute_iv_term_structure()`, `generate_vol_intelligence_report()` |
| **`src/institutional_flow.py`** | `InstitutionalFlowEngine` | `fetch_live_participant_oi()` (FII L/S ratio, options PCR), `compute_fii_flow_trend()` (5d accumulation/distribution/capitulation), `compute_rollover_analysis()` (spread pts, carry bias), `generate_institutional_flow_report()` |
| **`src/options_flow.py`** | Options microstructure & flow | `compute_atm_straddle_metrics()`, `compute_cumulative_oi_delta_and_traps()`, `compute_pcr_momentum_derivative()`, `compute_vanna_charm_drift_vector()`, `compute_short_term_directional_vector()`, `compute_oi_change_heatmap()`, `compute_strike_level_gex_chart_data()`, `compute_oi_based_range_forecast()` |
| **`src/options_engine.py`** | Derivatives pricing & structuring | `black_scholes_greeks()`, `select_institutional_strike()`, `generate_option_trade_ticket()`, `calculate_position_size()`, `calculate_tca_friction()`, `calculate_pcr_and_max_pain()`, `evaluate_golden_vault_lock()`, `run_monte_carlo_simulation()`, `compute_0dte_gamma_scalp_parameters()`, `construct_ratio_spread()`, `construct_delta_neutral_iron_condor()` |
| **`src/portfolio_risk.py`** | `PortfolioRiskManager` | `compute_portfolio_greeks()`, `compute_scenario_pnl_grid()` (Spot $\pm 200$, $T+0, T+1\text{d}, \text{Expiry}, \pm 3\%$ IV), `compute_var_stress_test()` (Flash Crash, Black Swan, Squeeze) |
| **`src/signal_journal.py`** | `LiveSignalJournal`, `SignalPerformanceAnalyzer`, `SignalEntry` | `log_signal()`, `update_open_trades_lifecycle()`, `seed_from_intraday_history()`, `get_journal_dataframe()`, `compute_daily_journal_summary()`, `export_csv_bytes()`, `win_rate_by_signal_type()`, `win_rate_by_time_bucket()`, `win_rate_by_regime()`, `confluence_vs_outcome_correlation()`, `streak_and_tilt_analysis()` |
| **`src/regime_switching.py`** | `KalmanFilterTrendEstimator`, `MarkovRegimeSwitcher` | Latent velocity estimation ($x_t, v_t$) and 3-state forward Markov posterior probabilities (`LOW_VOL_TRENDING`, `MEAN_REVERTING_CHOP`, `HIGH_VOL_EXPANSION`) |
| **`src/execution.py`** | `OrderManager`, `slice_institutional_order` | Chase-and-cancel limit order simulation on NSE ₹0.05 ticks, TWAP/VWAP child order slicing |
| **`src/performance_analytics.py`** | Performance ratios | Sharpe, Sortino, Calmar, Ulcer Index, Martin Ratio (UPI), SQN, Payoff, Kelly recovery protocol |
| **`src/backtest_engine.py`** | `BacktestEngine` | Historical bar-by-bar simulation, execution matching, cumulative equity curve |
| **`src/macro_engine.py`** | Macro sentiment | `compute_macro_sentiment_score()`, Brent crude, USD/INR, US 10Y Yields |
| **`src/broker_adapters.py`** | Broker interfaces | Zero-friction abstraction for live broker execution |

---

## 4. Primary Data Structures & Schemas

### `SignalEntry` (Dataclass in `src/signal_journal.py`)
```python
signal_id: str                      # "SIG-YYYYMMDD-HHMMSS-STRIKEOPT"
timestamp_ist: str                  # "YYYY-MM-DD HH:MM:SS IST"
spot_price: float                   # Nifty 50 spot price at entry
signal_type: str                    # "LONG_IB_BREAKOUT", "MEAN_REVERSION_SHORT", etc.
direction: str                      # "LONG" / "SHORT" / "WAIT"
selected_strike: int                # Strike price (e.g. 24500)
option_type: str                    # "CE" / "PE"
entry_premium: float                # Recommended option entry premium (₹)
sl_spot: float, sl_premium: float   # Stop loss levels
target_1_spot: float, target_2_spot: float, target_3_spot: float
confluence_score: float             # 0 to 100
confluence_grade: str               # "A+ Institutional" (>=85), "A Standard" (>=70)
lifecycle_status: str               # "TRIGGERED" -> "ACTIVE" -> "T1_REACHED" -> "T2_REACHED" -> "T3_MOONSHOT" / "STOPPED_OUT"
realized_r_multiple: float          # Realized R after exit
realized_pnl_rupees: float          # Net realized PnL in ₹
```

---

## 5. UI Structure (`app.py` Tabs)
1. **`tab_chart` (📈 Interactive Candlestick Chart):** Candlesticks, EMAs, VAKC, AVWAP, Fibonacci, CPR, Volume Profile (POC/VAH/VAL).
2. **`tab_journal` (📜 Live Signals Journal):** KPI Cards, Trade Lifecycle Table, Inspector, Signal Performance Attribution Expander (Win Rate by Type/Time/Regime, Confluence Correlation, Tilt Diagnostics), CSV Export.
3. **`tab_sizer` (🛡️ 1% Risk & Kelly Sizer):** Position Sizer, Golden Vault Controls, SOR Slicer, 1,000-Path Monte Carlo, Portfolio Greeks Dashboard, What-If Scenario Matrix, Volatility Cone.
4. **`tab_oi` (🏛️ Institutional Breadth & Option Chain):** HFI, Sectoral Pulse, FII/DII Derivatives Flow Intelligence, Live OI Change Heatmap, Dealer GEX Profile with Call/Put/Zero Walls, Option Chain Table.
5. **`tab_backtest` (📊 Bar-by-Bar Replay & Backtest Simulator):** Historical strategy simulation.
6. **`tab_cheatsheet` (🧠 Institutional Desk Wisdom & Master Playbook):** Trading rules & 0DTE guide.
