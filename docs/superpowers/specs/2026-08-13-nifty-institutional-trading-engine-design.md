# Technical Design Specification: Nifty Institutional Trading Plan & Options Engine (JustNifty v2.0)

**Date:** 2026-08-13  
**Status:** Approved for Implementation Planning  
**Target Platform:** Streamlit Web Application (Python 3.10+)  
**Data Sources:** `yfinance` (^NSEI) + `jugaad-data` (NSE India Live & Derivatives)

---

## 1. Executive Summary & Objective

The objective of this project is to build an institutional-grade Nifty 50 Futures and Options trading system and Streamlit interactive dashboard. The system faithfully implements the **JustNifty v2.0 quantitative day-trading methodology** augmented with **institutional & prop desk microstructure analytics** (Order Flow, Participant-wise OI, Gamma Exposure, Black-Scholes Greeks, and algorithmic strike selection).

The system operates in two core modes:
1. **Live Market Scanner:** Real-time intraday monitoring, signal generation, institutional strike selection, and risk-sized trade ticket formulation.
2. **Historical Date Replay & Backtesting Mode:** Bar-by-bar historical replay across any trading day, accompanied by full backtest analytics, trade logs, and performance metrics.

---

## 2. Architecture & Modular Breakdown

The project follows a clean, modular, and testable architecture:

```
d:/antigravity_sandbox/Nifty/
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-08-13-nifty-institutional-trading-engine-design.md
├── src/
│   ├── __init__.py
│   ├── config.py             # Constants, lot sizes, default parameters, color themes
│   ├── data_engine.py        # yfinance + jugaad-data ingestion, IST alignment, caching
│   ├── indicators.py         # 200/55/21 EMA, Envelopes, AVWAP, CPR, Fib, VF Table, Volume Profile
│   ├── strategy_rules.py     # JustNifty v2.0 logic, HTF weighting, MA stretch, 3 PM strategy
│   ├── options_engine.py     # Black-Scholes Greeks, Strike Picker (0.50-0.65 Delta), OI/PCR, 1% Sizing
│   └── backtest_engine.py    # Replay engine, trade logger, part-booking simulator, metrics
├── tests/
│   ├── test_data_engine.py
│   ├── test_indicators.py
│   ├── test_strategy_rules.py
│   └── test_options_engine.py
├── app.py                    # Streamlit interactive UI & Plotly visualization
└── requirements.txt          # Python dependencies
```

---

## 3. Detailed Component Specifications

### 3.1 Data Layer (`src/data_engine.py`)
- **Multi-Timeframe Streaming (`yfinance`):**
  - Fetches `^NSEI` for 1m (last 7 days), 5m (last 60 days), 15m, 1h, and Daily intervals.
  - Normalizes timezone strictly to `Asia/Kolkata` (IST) and filters session hours strictly between `09:15` and `15:30`.
- **NSE India Integration (`jugaad-data`):**
  - Fetches live/latest option chain data, participant-wise open interest (FII, DII, Prop, Client), and active weekly/monthly expiry schedules.
- **Resilience & Fallback:**
  - If NSE Live endpoint rate limits or encounters captchas, the engine falls back to `yfinance` spot data + synthetic Black-Scholes option chain modeling.
  - Local caching in `.cache/` to minimize redundant network calls.

### 3.2 Technical Indicator Engine (`src/indicators.py`)
- **Moving Averages:** 200 EMA (Regime), 55 EMA (Trend), 21 EMA (Momentum/Trailing).
- **1.5% 200 EMA Envelopes:** 
  $$\text{Upper Band} = 200\,\text{EMA} \times 1.015, \quad \text{Lower Band} = 200\,\text{EMA} \times 0.985$$
- **Session VWAP & Anchored VWAP (AVWAP):**
  $$\text{AVWAP}_t = \frac{\sum_{i=k}^t (\text{Typical Price}_i \times \text{Volume}_i)}{\sum_{i=k}^t \text{Volume}_i}$$
  - Primary anchor: `09:15` session open. Secondary anchors: Previous Day High/Low, Week Open.
  - Standard Deviation Bands ($\pm 1\sigma, \pm 2\sigma$).
- **Central Pivot Range (CPR):**
  $$\text{Pivot} = \frac{H+L+C}{3}, \quad \text{BC} = \frac{H+L}{2}, \quad \text{TC} = (\text{Pivot} - \text{BC}) + \text{Pivot}$$
  $$\text{CPR Width \%} = \frac{|\text{TC} - \text{BC}|}{\text{Pivot}} \times 100$$
- **Dynamic Fibonacci Retracement:** Automatically identifies local 5m swing highs/lows and computes 23.6%, 38.2%, 50.0%, 61.8%, 78.6% levels.
- **VF Trade Table (T1–T6):** Computes step-wise targets based on opening range and ATR extensions.
- **Volume Profile:** Computes Point of Control (POC), Value Area High (VAH), and Value Area Low (VAL) for the 70% volume distribution.

### 3.3 Strategy Rule Engine (`src/strategy_rules.py`)
- **9:15–9:30 AM Freak Candle Filter:** Ignores false spikes during the first 15 minutes, establishes the true Initial Balance (IB).
- **Higher Timeframe (HTF) Hierarchy (Query 79):** Confirms alignment between Daily and Hourly trends. If Daily is Down and Hourly is Up, long entries are blocked until the 5m chart reaches the lower 1.5% Envelope / Oversold extreme.
- **Far-Away MA Crossover Nuance (Query 12):** Rejects entries if price is overextended ($>0.35\%$) from the 21/55 EMA at the time of signal; waits for pullback/mean reversion.
- **Long Setup (BUY):**
  1. Price in 50.0% – 61.8% Fibonacci zone of recent upward impulse.
  2. Price strictly **Above 200 EMA** AND **Above 09:15 AVWAP**.
  3. Bullish trigger candle (Engulfing / Hammer / Bullish Doji).
  4. Proximity within $0.35\%$ of 21/55 EMA.
- **Short Setup (SELL):**
  1. Price in 50.0% – 61.8% Fibonacci zone of recent downward impulse.
  2. Price strictly **Below 200 EMA** AND **Below 09:15 AVWAP**.
  3. Bearish trigger candle (Bearish Engulfing / Shooting Star / Bearish Doji).
  4. Proximity within $0.35\%$ of 21/55 EMA.
- **The 3:00 PM Aggressive Breakout Strategy (Page 100 / Query 39):**
  - Scans the 15:00 candle on range-bound/consolidation days.
  - Break above 15:00 High $\rightarrow$ Aggressive BUY.
  - Break below 15:00 Low $\rightarrow$ Aggressive SELL.
  - Stop loss placed at the opposite extreme of the 15:00 candle.

### 3.4 Institutional Options & Strike Selection Engine (`src/options_engine.py`)
- **Strike Selector:**
  - Delta target: $\Delta \in [0.50, 0.65]$ (ATM to 1-strike ITM).
  - Expiry selection: Nearest weekly expiry; auto-shifts to next week after 12:30 PM on Thursday expiry day.
- **Black-Scholes Pricing & Greeks:**
  - Computes theoretical Option Price, Delta ($\Delta$), Gamma ($\Gamma$), Theta ($\Theta$), and Vega ($\mathcal{V}$).
  - Projects exact Option Entry Premium, Stop-Loss Premium, and Target Premiums (T1–T6).
- **Open Interest & Microstructure Analysis:**
  - Real-time $\Delta\text{OI}$ tagging: Long Build-up, Short Build-up, Short Covering, Long Unwinding.
  - Put-Call Ratio (PCR) and Max Pain calculation.
- **Institutional 1% Risk Sizing & Part-Booking:**
  - $\text{Risk Amount} = \text{Account Capital} \times 0.01$.
  - $\text{Lots} = \lfloor \text{Risk Amount} / ((\text{Entry Premium} - \text{SL Premium}) \times \text{Lot Size}) \rfloor$.
  - **Part-Booking Plan:**
    - 50% lots booked at **VF Table T1** or **1.5% 200 EMA Envelope extreme**.
    - Stop-Loss moved to **Break-Even (Entry Premium)** immediately upon 50% fill.
    - Remaining 50% trailed on **1-min 21 EMA** or **AVWAP**.
    - **"Enough" Rule:** Triggers session shutdown banner when daily profit $\ge 0.2\% - 0.5\%$.

### 3.5 Backtesting & Simulation Engine (`src/backtest_engine.py`)
- Historical bar-by-bar replay across selected date ranges.
- Full execution simulation including 50% part-booking, breakeven SL adjustment, and 21 EMA / AVWAP trailing.
- Performance statistics: Win Rate, Profit Factor, Max Drawdown, Total Return (₹ and %), Risk:Reward ratio distribution, and exportable CSV trade logs.

### 3.6 Streamlit Application (`app.py`)
- **Header & Metric Bar:** Current Nifty Spot, Futures Basis, CPR Width & Regime, Session AVWAP, India VIX, and PCR.
- **Interactive Multi-Pane Plotly Chart:**
  - Candlesticks with 200/55/21 EMAs, 1.5% Envelopes, AVWAP $\pm 1\sigma, \pm 2\sigma$, Fib Retracement overlays, and VF Target levels.
  - Subplots for RSI and Volume Profile.
- **Institutional Signal & Trade Ticket Panel:**
  - Active Signal (LONG / SHORT / WAIT / 3PM).
  - Recommended Option Strike (CE/PE), Expiry, Entry Premium, SL Premium, and T1–T6 Targets.
  - Position Sizing Calculator (Capital input, Max Risk %, Calculated Lots & Quantity).
- **Institutional Market Intelligence Tab:**
  - Participant-wise OI summary (FII, Prop, DII, Retail).
  - Option Chain Table with Greeks ($\Delta, \Gamma, \Theta, \text{IV}$) and $\Delta\text{OI}$ flags.
- **Backtest / Date Replay Tab:**
  - Date picker to replay any historical session bar-by-bar with interactive slider.
  - Backtest performance report and trade log table.

---

## 4. Verification & Testing Strategy

1. **Unit Tests (`tests/`):**
   - Test data fetchers, timezone converters, and caching mechanisms.
   - Test indicator math against known reference calculations (EMAs, AVWAP, CPR, Envelopes, Fibs).
   - Test Black-Scholes Greeks calculations against theoretical benchmarks.
   - Test strategy rule triggers (Long, Short, 3 PM, Freak candle filter, MA stretch filter).
2. **Integration & Visual Verification:**
   - Launch Streamlit application locally and verify interactive chart rendering, indicator toggles, live/historical mode switching, and trade ticket generation.

---

## 5. Next Steps

Upon review and approval of this specification, we will invoke the `writing-plans` skill to generate a structured implementation plan with step-by-step development and verification milestones.
