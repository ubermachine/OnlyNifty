# OnlyNifty (v5.3 Desk Edition) — Master Architecture & Agent Blueprint

> **Agent Invariant:** Read THIS file first before modifying or exploring the codebase. It provides 100% of the architecture, module index, mathematical formulas, data contracts, execution gates, and test commands.

---

## 1. Quick Start & Execution Commands (PowerShell Invariants)

- **Run all tests:** `python -m pytest tests/ -v --tb=short`
- **Run single test file:** `python -m pytest tests/test_options_positioning.py -v`
- **Run notifications test:** `python -m pytest tests/test_notifications.py -v`
- **Full Forensic 6-Pillar Verification:** `python verify_all_modules.py`
- **Streamlit App Launch:** `python -m streamlit run app.py`
- **Verify Syntax:** `python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('Syntax OK')"`

### Critical Shell & UI Rules
- **PowerShell Syntax Invariant:** NEVER use `&&` (use `;`). Never pipe to `head`/`tail`/`grep` (use `Select-Object -First N`, `Select-Object -Last N`, `Select-String`).
- **Streamlit Layout Invariant:** ALWAYS use `width='stretch'` (NEVER `use_container_width=True` which is deprecated in Streamlit 1.58+).
- **Streamlit Markdown HTML Invariant:** Never include 4+ spaces of leading indentation on tag lines inside multi-line f-strings passed to `st.markdown(..., unsafe_allow_html=True)`. Keep HTML tags flush to column 0 or pass through `textwrap.dedent()` to avoid triggering CommonMark code-block (`<pre><code>`) formatting.
- **Scraper / Test Verification Invariant:** Rely strictly on DOM structure, console logs, and intercepted network responses. Do not take, save, or use screenshots during automated scraping verification or test runs unless explicitly requested.

---

## 2. End-to-End System Pipeline

```
[ Market Ingestion: yfinance / jugaad-data / Synthetic ]
                           │
                           ▼
             [ DataEngine: clean_ohlcv (IST) ]
                           │
       ┌───────────────────┴───────────────────┐
       ▼                                       ▼
[ Microstructure & Technicals ]      [ Statistical & Volatility Engines ]
 • EMA (21, 55, 200)                  • Kalman Velocity Estimator (v_t)
 • VAKC Keltner Envelopes             • Markov 3-State Regime Switcher
 • Session AVWAP ±2σ                  • DFA Long-Memory Alpha (H)
 • Session-Reset CVD (09:15 IST)      • VPIN Order Flow Toxicity (BVC)
 • Passive Limit Absorption Traps     • 25-Delta Put-Call Skew (Z-Score)
 • CPR Pivots & Volume Profile        • Realized Vol Compression (VCR)
 • Heavyweight Flow Index (HFI 41.2%) • Volatility Cones & Term Structure
 • Sector Breadth Momentum (SBM)      • Institutional Flow Engine (FII L/S)
       └───────────────────┬───────────────────┘
                           │
                           ▼
            [ Options Desk Positioning Engine ]
         (D-Vector, PCR Z-Score, Max Pain Drift,
          Dealer GEX Walls, Corridor Clamping)
                           │
                           ▼
               [ StrategyEngine.evaluate_bar ]
    (4-Regime Router + Passive Absorption + Lunch Lull +
     Hard Gates: 25D Skew, HFI Flow, GEX Wall, Positioning Veto)
                           │
                           ▼
                [ build_desk_verdict ]
     (Authoritative Action, Trend Conviction, Range Corridor,
      Taylor Convexity Option Pick Ticket, Evidence Pills)
                           │
       ┌───────────────────┼───────────────────┐
       ▼                   ▼                   ▼
[ Options Flow & Micro ] [ LiveSignalJournal ] [ Telegram Webhook ]
 • ATM Straddle Corridor  • State deduplication • Async ThreadPool (<50ms)
 • 4-Quadrant ΔOI & Traps • MFE / MAE tracking  • HTML Risk Cards
 • Strike OI Heatmap      • Signal Attribution  • Free Spread guidance @ T1
 • Dealer GEX Profile     • SHA-256 tamper chain• Ring-buffer deduplication
       └───────────────────┼───────────────────┘
                           │
                           ▼
     [ Unified Main-Page Desk Verdict Cockpit & 6 Tabs ]
```

---

## 3. Comprehensive Module Catalog (`src/`)

### Core Configuration & Risk
| File | Classes / Functions | Key Contracts & Defaults |
|---|---|---|
| **`src/config.py`** | Configuration Constants | `DEFAULT_CAPITAL = 500000.0`, `LOT_SIZE = 25`, `DEFAULT_IV = 0.135`, `EMA_FAST = 21`, `EMA_SLOW = 200`, `SKEW_ZSCORE_THRESHOLD = 1.50`, `VCR_SQUEEZE_THRESHOLD = 0.15`, `SIGNAL_MIN_CONFLUENCE = 70.0`, `COOLDOWN_BARS = 12`, `POSITIONING_VETO_STRENGTH = 0.50`, `WALL_BUFFER_PTS = 25.0`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` |
| **`src/risk_state.py`** | `SessionRiskState` | Daily trade budget ($\le 3$), 2-strike loss streak halt, 1.5% Daily Loss Limit (DLL), 12-bar relative cooldown tracker (with index underflow protection), JSON persistence to `data/risk_state_today.json`. |
| **`src/decision_engine.py`** | `DecisionEngine`, `SetupCandidate`, `DecisionContext` | Pure decision engine: ranks setup candidates, evaluates universal gates, queries edge tables, enforces confluence score floor ($< 70.0 \implies \text{WAIT}$), and sizes positions. |
| **`src/edge_harness.py`** | `EdgeTable`, `EdgeStats`, `WalkForwardRunner` | Walk-forward out-of-sample edge discovery (30d train / 5d test, 60-bar lookback purge, 12-bar embargo) and automated quarantine of negative EV setups. |

### Options Positioning & Decision Verdict
| File | Classes / Functions | Key Contracts & Outputs |
|---|---|---|
| **`src/options_positioning.py`** | `OptionsDeskState`, `compute_options_desk_state()`, `clamp_targets_to_corridor()`, `load_options_history()`, `save_options_history()` | Synthesizes 5-pillar directional vector ($D \in [-1.0, +1.0]$), intraday PCR Z-score ($\ge 20$ samples), Max Pain drift ($\Delta \text{MaxPain}$ pts), ITM/OTM Delta OI shift, Expected vs. Actual move ratio ($R_{\text{move}}$), and multi-pillar agreement voting ($0 \dots 4$). Persists session state to `data/options_state.json`. Clamps option targets to Put/Call dealer walls in $+\Gamma$ regimes. |
| **`src/desk_verdict.py`** | `DeskVerdict`, `build_desk_verdict()` | Pure synthesis producing authoritative Action (`BUY_CE` 🟢, `BUY_PE` 🔴, `WAIT` 🟡) + reason, Trend Conviction %, Range Corridor $[PutWall \longleftrightarrow CallWall]$ with spot position dot $\%$, Max Pain magnet, expected move ($\pm \text{pts}$), concrete execution ticket, and 4-pillar evidence pills (`Structure`, `Flow`, `Positioning`, `Macro`). |

### Market Ingestion & Microstructure
| File | Classes / Functions | Key Contracts & Outputs |
|---|---|---|
| **`src/data_engine.py`** | `DataEngine` | `clean_ohlcv(df)` (enforces IST timezone, synthesizes volume proxy if missing), `fetch_yfinance_nifty()`, `fetch_live_nse_option_chain()`, `fetch_multi_expiry_option_chain()`, `get_participant_oi_snapshot()`, `fetch_heavyweight_flow_index()`, `fetch_sectoral_pulse()`, `fetch_pre_open_gap()` |
| **`src/indicators.py`** | Technical & Quantitative Suite | `compute_ema(series, span)`, `compute_vakc_envelopes(df)` (dynamic IV scaling), `compute_vwap(df)` (online variance $\pm 2\sigma$), `compute_cpr(df)`, `compute_volume_profile(df)` (POC, VAH, VAL), `compute_session_cvd(df)`, `detect_absorption_traps(df)`, `compute_hurst_exponent(series)` (Anis-Lloyd bias-corrected), `compute_order_flow_imbalance(df)` (OFI Z-score), `compute_dealer_gex(spot)`, `compute_multi_timeframe_regime(df)`, `detect_stacked_order_flow_imbalances(df)`, `compute_initial_balance_and_day_type(df)` (09:15-10:15 IST), `compute_vwap_multi_dispersion_and_half_life(df)`, `compute_dfa_alpha(series)`, `compute_vpin_toxicity(df)` |
| **`src/volatility_engine.py`** | `VolatilityIntelligence` | `compute_25delta_skew()`, `compute_vcr_squeeze()`, `compute_realized_volatility(close_prices)`, `compute_iv_rv_spread(iv, rv)`, `compute_iv_percentile(iv, history)`, `compute_expected_vs_actual_move()`, `compute_intraday_quality_score(bar_time)`, `compute_volatility_cone(close_prices)`, `compute_rv_term_structure()`, `compute_iv_term_structure()`, `generate_vol_intelligence_report()` |

### Strategy & Derivatives Structuring
| File | Classes / Functions | Key Contracts & Outputs |
|---|---|---|
| **`src/strategy_rules.py`** | `StrategyEngine`, `Signal`, `SignalType` | 4-Regime Router + Passive Limit Absorption + Lunch Lull Halver (11:30-13:00 IST) + Range Fade & Gamma Breakout setups. Hard universal gates: 25D Put Skew crash veto ($Z > 1.5$), Heavyweight Flow veto ($\|HFI\| > 0.20$), GEX Call Wall defense, Positioning Flow Veto ($\|D\| \ge 0.50$). |
| **`src/options_flow.py`** | Options Microstructure & Flow | `compute_atm_straddle_metrics()`, `compute_cumulative_oi_delta_and_traps()`, `compute_pcr_momentum_derivative()`, `compute_vanna_charm_drift_vector()`, `compute_short_term_directional_vector()` (5-pillar synthesis), `compute_oi_change_heatmap()`, `compute_strike_level_gex_chart_data()`, `compute_oi_based_range_forecast()` |
| **`src/options_engine.py`** | Derivatives Pricing & Risk | `black_scholes_greeks(spot, strike, dte, iv, is_call)`, `select_institutional_strike()`, `generate_option_trade_ticket()` (2nd-order Taylor series convexity), `calculate_position_size()` (Quarter-Kelly & DD dampener), `calculate_tca_friction()` (Indian STT, exchange, GST, stamp duty), `calculate_pcr_and_max_pain()`, `evaluate_golden_vault_lock()`, `run_monte_carlo_simulation()` (1,000 vectors), `compute_0dte_gamma_scalp_parameters()`, `construct_ratio_spread()`, `construct_delta_neutral_iron_condor()` |
| **`src/institutional_flow.py`** | `InstitutionalFlowEngine` | `fetch_live_participant_oi()`, `compute_fii_flow_trend()`, `compute_rollover_analysis()`, `generate_institutional_flow_report()` |
| **`src/portfolio_risk.py`** | `PortfolioRiskManager` | `compute_portfolio_greeks()`, `compute_scenario_pnl_grid()` (Spot $\pm 200$, $T+0, T+1\text{d}, \text{Expiry}, \pm 3\%$ IV), `compute_var_stress_test()` (Flash Crash, Black Swan, Squeeze) |

### Execution, Logging & Analytics
| File | Classes / Functions | Key Contracts & Outputs |
|---|---|---|
| **`src/signal_journal.py`** | `LiveSignalJournal`, `SignalPerformanceAnalyzer`, `SignalEntry` | `log_signal()`, `update_open_trades_lifecycle()`, `seed_from_intraday_history()`, `get_journal_dataframe()`, `compute_daily_journal_summary()`, `export_csv_bytes()`, `win_rate_by_signal_type()`, `win_rate_by_time_bucket()`, `win_rate_by_regime()`, `confluence_vs_outcome_correlation()`, `streak_and_tilt_analysis()` |
| **`src/notifications.py`** | `TelegramNotifier` | Async `dispatch_signal_alert()`, `format_signal_html()`, `format_lifecycle_html()`, `send_test_alert()`. Thread-safe ring buffer deduplication. |
| **`src/regime_switching.py`** | `KalmanFilterTrendEstimator`, `MarkovRegimeSwitcher` | Latent velocity estimation ($x_t, v_t$) and 3-state forward Markov posterior probabilities (`LOW_VOL_TRENDING`, `MEAN_REVERTING_CHOP`, `HIGH_VOL_EXPANSION`). |
| **`src/execution.py`** | `OrderManager`, `slice_institutional_order` | Chase-and-cancel limit order simulation on NSE ₹0.05 ticks, TWAP/VWAP child order slicing. |
| **`src/performance_analytics.py`** | Performance Ratios | Sharpe, Sortino, Calmar, Ulcer Index, Martin Ratio (UPI), SQN, Payoff, Kelly recovery protocol. |
| **`src/backtest_engine.py`** | `BacktestEngine` | Historical bar-by-bar simulation, execution matching, cumulative equity curve. |
| **`src/macro_engine.py`** | Macro Sentiment | `compute_macro_sentiment_score()`, Brent crude, USD/INR, US 10Y Yields. |
| **`src/broker_adapters.py`** | Broker Interfaces | Zero-friction abstraction for live broker execution. |

---

## 4. Primary Data Schemas & Contracts

### `SignalEntry` Dataclass (`src/signal_journal.py`)
```python
signal_id: str                      # "SIG-YYYYMMDD-HHMMSS-STRIKEOPT"
timestamp_ist: str                  # "YYYY-MM-DD HH:MM:SS IST"
spot_price: float                   # Nifty 50 spot price at entry
signal_type: str                    # "LONG_IB_BREAKOUT", "RANGE_FADE_LONG", "GAMMA_BREAKOUT_LONG", etc.
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

### `OptionsDeskState` Dataclass (`src/options_positioning.py`)
```python
d_vector: float                     # Consolidated 5-Pillar Directional Vector [-1.0, +1.0]
pcr_zscore: float                   # Intraday PCR rolling Z-score
max_pain_drift: float               # Shift in Max Pain strike vs session open (pts)
itm_otm_shift: float                # Net ITM vs OTM Delta OI accumulation shift
expected_move_pts: float            # Intraday 1-day Straddle Expected Move (pts)
call_wall: float, put_wall: float   # Dealer Call / Put Resistance & Support walls
gamma_regime: str                   # "DEALER_LONG_GAMMA" (+Γ Pin) vs "DEALER_SHORT_GAMMA" (-Γ Breakout)
agreement_score: int                # Multi-pillar confirmation votes (0 to 4)
data_quality: str                   # "VERIFIED" vs "UNVERIFIED" (Synthetic/fallback)
```

### `DeskVerdict` Dataclass (`src/desk_verdict.py`)
```python
action: str                         # "BUY_CE", "BUY_PE", "WAIT"
action_label: str                   # "BUY CALL (CE) — BREAKOUT", "NO-TRADE (WAIT) — AWAITING CONFLUENCE", etc.
reason: str                         # Authoritative 1-line plain english explanation
trend_bias: str                     # "BULLISH", "BEARISH", "NEUTRAL"
trend_conviction_pct: float         # 0.0% to 100.0%
range_corridor: Tuple[float, float] # [PutWall, CallWall]
spot_position_pct: float            # 0.0% (at Put Wall) to 100.0% (at Call Wall)
max_pain: float                     # Max Pain magnet strike
expected_move_pts: float            # Straddle expected move (pts)
option_pick: Optional[Dict]         # Concrete ticket (symbol, entry, SL, T1/T2/T3, lots, TCA) or None
evidence: Dict[str, str]            # {"structure": "...", "flow": "...", "positioning": "...", "macro": "..."}
confluence_score: float             # 0.0 to 100.0
confluence_grade: str               # "A+ Institutional", "A Standard", "B Moderate", "C Weak / Vetoed"
data_quality: str                   # "VERIFIED" vs "UNVERIFIED"
```

---

## 5. UI Architecture (`app.py`)

- **Sidebar Controls:** Risk & capital inputs (Capital ₹, Risk % per trade, Lot size, IV %, Portfolio DD %, 0DTE mode toggle), Data stream engine (Live vs Synthetic, Timeframe, Refresh cadence), **🔔 Telegram Webhook Alert expander** with credentials and "Send Test Alert" button.
- **Top Main-Page Desk Verdict Panel:**
  - **Authoritative Action Badge:** `● BUY CE CONFIRMED` 🟢 / `● BUY PE CONFIRMED` 🔴 / `● NO-TRADE (WAIT)` 🟡.
  - **Quality & Confluence Metric:** Quality grade (`VERIFIED`/`UNVERIFIED`) and Score %.
  - **Desk Reason:** Plain English justification citing exact conflicting pillars or setup rationale.
  - **Institutional Trend & Momentum Box:** Bias (`BULLISH`/`BEARISH`/`NEUTRAL`), Conviction %, $D$-Vector, Kalman Velocity ($V$), Active Regime, Day Type.
  - **Expected Range & Dealer Walls Box:** Put Wall $\longleftrightarrow$ Spot Position Dot $\%$ $\longleftrightarrow$ Call Wall, Max Pain, Expected Move ($\pm \text{pts}$), Actual Range and Expansion multiplier.
  - **Recommended Ticket Grid:** 5-box option card (Entry, SL, Target 1, Target 2, Moonshot T3, lots, TCA friction) or clean risk-preservation wait banner.
  - **Evidence Summary Pills:** `Structure`, `Flow`, `Positioning`, `Macro`.
- **Collapsible Microstructure Expanders:**
  - `📊 Stochastic Indicators & 9-Cell Confluence Engine`: 1. HTF Alignment, 2. Kalman Velocity, 3. VWAP Dispersion, 4. DFA Alpha, 5. VPIN Toxicity, 6. Dynamic Kelly, 7. IV-RV Spread, 8. Order Flow OFI, 9. Dealer GEX.
  - `🌊 5-Pillar Short-Term Directional Vector ($D_{\text{intraday}}$)`.
  - `⚡ Volatility Intelligence & Quality Banner`.
- **6 Interactive Tabs:**
  1. **`tab_chart` (📈 Interactive Candlestick Chart):** Candlesticks, 200/55/21 EMAs, VAKC Keltner Envelopes, AVWAP $\pm 2\sigma$, CPR Pivots, Volume Profile (POC/VAH/VAL).
  2. **`tab_journal` (📜 Live Signals Journal & Audit Store):** 6 KPI summary cards, Live Trade Lifecycle table, Trade Inspector, Attribution expander (Win Rate by Type/Time/Regime, Confluence Correlation, Tilt Diagnostics), CSV Export.
  3. **`tab_sizer` (🛡️ 1% Risk & Kelly Sizer):** Quarter-Kelly position sizer, Golden Vault lock (+1.5% net profit floor), Indian NSE Statutory TCA breakdown, Smart Order Routing (SOR) slicer, 1,000-Path Monte Carlo ruin simulator, Portfolio Greeks dashboard, What-If scenario matrix, Volatility Cone.
  4. **`tab_oi` (🏛️ Institutional Breadth & Option Chain):** Heavyweight Flow Index (HFI), Sector Breadth Momentum (SBM), FII/DII Participant positioning, Live OI change heatmap, Dealer GEX profile, Option Chain table.
  5. **`tab_backtest` (📊 Bar-by-Bar Replay & Backtest Simulator):** Historical bar-by-bar strategy replay with equity curve and performance metrics.
  6. **`tab_cheatsheet` (🧠 Master Alpha Playbook v4.0):** 4 Desk Scrutiny pillars (Quant, Options Structuring, CRO, Microstructure) and Top 12 Golden Rules of Institutional Index Options Trading.

---

## 6. Core Mathematical Formulations

1. **2nd-Order Taylor Series Convexity Pricing:**
   $$dP \approx \Delta \cdot dS + \frac{1}{2} \Gamma \cdot (dS)^2 + \Theta \cdot dt$$
2. **Directional Vector ($D_{\text{intraday}} \in [-1.0, +1.0]$) — equal weight across the 3 *informative* pillars:**
   $$D = \tfrac{1}{3}\left( S_{\Delta\text{OI}} + S_{\text{PCR}} + S_{\text{HFI}} \right)$$
   Equal weighting follows Timmermann (2006), which holds only among candidates of
   comparable signal-to-noise — hence the exclusions below.

   > **$S_{\text{VC}}$ (vanna/charm) and $S_{\text{Straddle}}$ are EXCLUDED from $D$.** Both were
   > measured to be deterministic functions of $(\text{spot} \bmod 50)$ with no directional content:
   > `compute_vanna_charm_drift_vector` is called with $K = \text{round}(S/50)\cdot 50$ and spot cancels
   > inside $\text{vanna} = \frac{\text{vega}}{S}(\cdot)$, so `drift_score` is identical ($-0.0880$) at
   > spot 20000 / 24500 / 24550 / 26000 — level-invariant across 6000 index points, with a constant
   > $\approx -0.045$ bearish tilt. $S_{\text{Straddle}}$ keys off `vol_state`, which compares
   > `straddle >= straddle * 1.02` and is therefore never `VOL_EXPANSION`, reducing the pillar to
   > $\pm 0.25$ on $\text{sign}(S - K_{\text{ATM}})$. Carried at 0.20 each, they made 40% of the
   > "directional" vector a sawtooth. Both remain in `component_scores` for diagnostics only.
   > `dealer_drift_score` is excluded from the desk-verdict positioning family for the same reason.
3. **Black-Scholes Intraday Expected Move Approximation:**
   $$\text{ExpMove}_{\text{pts}} \approx S_0 \cdot \sigma_{\text{IV}} \cdot \sqrt{\frac{1}{365}} \cdot 0.80$$
4. **Realized Volatility (RV) Annualized:**
   $$\text{RV}_{\text{annual}} = \sqrt{252 \cdot \frac{375}{5}} \cdot \sqrt{\frac{1}{N-1} \sum_{i=1}^N (r_i - \bar{r})^2}$$
5. **Quarter-Kelly Sizing Fraction:**
   $$f^* = \frac{p \cdot b - q}{b}, \quad f_{\text{exec}} = \max\left(0, \frac{f^*}{4} \cdot \text{Dampener}_{\text{DD}}\right)$$
6. **Golden Vault Intraday Profit Lock Floor:**
   $$\text{Locked Floor} = \begin{cases} 0.75 \cdot \text{Peak PnL}, & \text{if Peak PnL} \ge 0.015 \cdot \text{Capital} \\ 0, & \text{otherwise} \end{cases}$$

---

## 7. Universal Quality & Gating Defense Invariants

| Gate / Defense | Mathematical Threshold | Trigger Condition | Enforcement Action |
|---|---|---|---|
| **25-Delta Put Skew Gate** | $Z_{\text{skew}} > +1.50$ | Severe downside tail-risk demand | HARD VETO on all `BUY_CE` setups |
| **Heavyweight Flow Index (HFI)** | $\|HFI\| > 0.20$ opposing trade | Reliance & HDFC Bank fighting signal | HARD VETO on opposing setups |
| **Positioning Flow Veto** | $\|D_{\text{intraday}}\| \ge 0.50$ opposing | Options flow diametrically opposes chart | HARD VETO on opposing setups |
| **VPIN Flow Toxicity Gate** | $\text{VPIN} > 0.85$ | Institutional informed order toxicity | HARD VETO on all directional trades |
| **Dealer GEX Call Wall Pin** | Spot $\ge \text{CallWall} - 25\text{ pts}$ in $+\Gamma$ | Gamma pin resistance | Block breakout longs; favor Range Fade |
| **Minimum Confluence Floor** | $\text{Score} < 70.0\%$ | Sub-threshold statistical edge | Downgrades action to `WAIT` |
| **Daily Loss Limit (DLL)** | $\text{Realized PnL} \le -1.5\%$ | Daily capital defense ceiling | Session halt & terminal lock |
| **2-Strike Loss Breaker** | 2 consecutive session losses | Negative momentum streak | Session halt & terminal lock |
| **Relative Cooldown** | $< 12$ bars since last entry | High-frequency overtrading prevention | Blocks new entries until 12 bars elapse |
