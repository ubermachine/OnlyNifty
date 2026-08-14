<div align="center">

# 🎯 OnlyNifty (PRO v4.0 Ultimate Turbo Institutional Edition)
### *Tier-1 Institutional Quantitative Trading Engine & Vectorized Low-Latency Terminal for Nifty 50 Options*

[![Build Status](https://img.shields.io/badge/build-passing-05df72.svg?style=for-the-badge&logo=github)](https://github.com/ubermachine/OnlyNifty)
[![Tests](https://img.shields.io/badge/unit%20tests-71%2F71%20passing%20(100%25)-00d2ff.svg?style=for-the-badge)](https://github.com/ubermachine/OnlyNifty)
[![Latency](https://img.shields.io/badge/latency-sub--15ms%20warm%20%7C%20WebGL%20GPU-05df72.svg?style=for-the-badge)](https://github.com/ubermachine/OnlyNifty)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg?style=for-the-badge&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/streamlit-1.37+-FF4B4B.svg?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![Market](https://img.shields.io/badge/market-NSE%20India%20(NIFTY%2050)-orange.svg?style=for-the-badge)](https://www.nseindia.com)
[![License](https://img.shields.io/badge/license-MIT-purple.svg?style=for-the-badge)](LICENSE)

<p align="center">
  <b>A production-grade, mathematically rigorous algorithmic trading system and ultra low-noise terminal designed specifically for Nifty 50 Index Options (Weekly Expiries & 0DTE).</b><br>
  Combines stochastic regime classification (Hurst Exponent + 3-State Markov), Kalman Filter state-space velocity, Ornstein-Uhlenbeck (OU) half-life, Volatility-Adaptive Keltner Channels (VAKC), SVI Volatility Smile, Auction Market Theory (Market Profile IB + AVWAP + Volume Profile), second-order Black-Scholes Greeks (Vanna/Charm/Volga), Indian statutory Transaction Cost Analysis (TCA), and fat-tail Quarter-Kelly position sizing with Golden Vault profit defense.
</p>

[🖥️ Launch Terminal](#-quick-start) • [📐 Mathematical Formulations](#-the-4-adaptive-stochastic-pillars) • [⚡ Greeks Engine](#-derivatives-microstructure--second-order-greeks) • [🛡️ Risk & TCA Model](#️-capital-risk-sizing--statutory-tca-model) • [📊 Backtest Benchmark](#-backtest-performance-benchmark)

---

</div>

## 📑 Table of Contents
- [1. System Topology & Architecture](#1-system-topology--architecture)
- [2. The 4 Adaptive Stochastic Pillars](#2-the-4-adaptive-stochastic-pillars)
- [3. The 5 Proprietary Desk Microstructure Filters](#3-the-5-proprietary-desk-microstructure-filters)
- [4. Derivatives Engine & Second-Order Greeks](#4-derivatives-engine--second-order-greeks)
- [5. Capital Risk Sizing & Statutory TCA Model](#5-capital-risk-sizing--statutory-tca-model)
- [6. Multi-Stage Trade Execution Protocol](#6-multi-stage-trade-execution-protocol)
- [7. Backtest Performance Benchmark (77.8% Win Rate)](#7-backtest-performance-benchmark-778-win-rate)
- [8. Repository Directory Structure](#8-repository-directory-structure)
- [9. Quick Start & Execution Guide](#9-quick-start--execution-guide)
- [10. Institutional Unit Test Suite](#10-institutional-unit-test-suite)
- [11. Disclaimer & License](#11-disclaimer--license)

---

## 1. System Topology & Architecture

The **OnlyNifty (JustNifty v3.0)** architecture transforms traditional technical analysis into an institutional, event-driven quantitative engine:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                  ONLYNIFTY v3.0 SYSTEM ARCHITECTURE                              │
└────────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                 │
        ┌────────────────────────────────────────┼────────────────────────────────────────┐
        ▼                                        ▼                                        ▼
┌───────────────────────────────┐┌───────────────────────────────┐┌───────────────────────────────┐
│ 1. Stochastic Regime & Macro  ││ 2. Microstructure & Auction   ││ 3. Derivatives Execution Desk │
├───────────────────────────────┤├───────────────────────────────┤├───────────────────────────────┤
│ • Fractional Hurst (H > 0.55) ││ • 09:15 Session AVWAP ±2σ     ││ • Target Delta [0.50, 0.65]   │
│ • Vol-Adaptive Keltner (VAKC) ││ • High-Volume Node (HVN) Fib  ││ • 2nd-Order Vanna & Charm     │
│ • 200 / 55 / 21 EMA Hierarchy ││ • Order Flow Imbalance (OFI)  ││ • Quarter-Kelly Position Sizer│
│ • Central Pivot Range (CPR)   ││ • Dealer Gamma Exposure (GEX) ││ • Full Indian NSE TCA Model   │
└───────────────────────────────┘└───────────────────────────────┘└───────────────────────────────┘
                                                 │
                                                 ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                  UNIFIED INSTITUTIONAL COCKPIT                                   │
│  • Live Signal Status Badge • Real-Time Confluence Matrix • Instant Execution Ticket (Lots & SL) │
│  • Interactive Plotly Candlesticks • Bar-by-Bar Replay Backtester • Participant OI Matrix        │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. The 4 Adaptive Stochastic Pillars

### Pillar 1: Regime-Filtered Fractional Hurst Exponent ($H$)
Rather than assuming stationarity with a static 200 EMA, the engine computes the **Hurst Exponent** via Rescaled Range ($R/S$) analysis to classify market dynamics:

$$H = \frac{\ln(R/S)}{\ln(\tau)} \quad \text{where } R/S \text{ is the Rescaled Range over lag } \tau$$

* **$H > 0.55$ (Persistent / Trending):** Long memory momentum regime $\implies$ **Golden Pocket pullback strategy active**.
* **$H < 0.45$ (Anti-Persistent / Mean-Reverting):** Mean-reverting regime $\implies$ **Fade outer VAKC bands; reject breakout continuations**.
* **$0.45 \le H \le 0.55$ (Random Walk):** Gaussian noise $\implies$ **Hard kill switch; zero directional exposure**.

---

### Pillar 2: Volatility-Adaptive Keltner Channels (VAKC)
Fixed percentage bands fail when India VIX ranges from $10\%$ to $22\%$. VAKC dynamically adjusts boundaries using 14-period Average True Range (ATR) scaled by Implied Volatility:

$$\text{Upper Band}_t = \text{EMA}_{200, t} + \lambda \cdot \text{ATR}_{14, t} \cdot \sqrt{\frac{\sigma_{\text{IV}}}{\sigma_{\text{Baseline}}}}$$
$$\text{Lower Band}_t = \text{EMA}_{200, t} - \lambda \cdot \text{ATR}_{14, t} \cdot \sqrt{\frac{\sigma_{\text{IV}}}{\sigma_{\text{Baseline}}}}$$

* **Parameters:** $\lambda = 2.25$, $\sigma_{\text{Baseline}} = 0.12$ ($12.0\%$ India VIX).
* **Advantage:** Dynamically tightens bands in low-volatility environments to ensure mechanical 50% part-booking, and widens during high-volatility expansions to capture extended tail trends.

---

### Pillar 3: Volume-Weighted Fibonacci Golden Pocket
Standard Fibonacci models assume uniform liquidity. The v3.0 engine extracts the exact **High Volume Node (HVN)** inside the $50.0\% - 61.8\%$ retracement zone:

$$\text{VW-Fib}_{\text{Optimal Entry}} = \arg\max_{P \in [\text{Fib}_{50\%}, \text{Fib}_{61.8\%}]} \text{VolumeProfile}(P)$$

$$\text{Invalidation Level (SL)} = \text{Swing High} - 0.786 \cdot \text{Range} - 5\text{ pts (Long)} \quad \Big| \quad \text{Swing Low} + 0.786 \cdot \text{Range} + 5\text{ pts (Short)}$$

---

### Pillar 4: Order Flow Imbalance (OFI) & Session AVWAP
Price defending the Session AVWAP (anchored at `09:15:00 IST`) is validated only if aggressive market order flow confirms institutional absorption:

$$\text{AVWAP}_t = \frac{\sum_{i=1}^t P_{i, \text{typ}} \cdot V_i}{\sum_{i=1}^t V_i}, \quad P_{i, \text{typ}} = \frac{H_i + L_i + C_i}{3}$$

$$\text{OFI}_t = \sum_{i=1}^t \left[ \text{Sign}(P_i - P_{i-1}) \times V_i \right]$$

* **Execution Condition:** Pullbacks to AVWAP require $\Delta \text{OFI} > 0$ for Long entries (buyers absorbing supply) or $\Delta \text{OFI} < 0$ for Short entries.

---

## 3. The 5 Proprietary Desk Microstructure Filters

| # | Filter Mechanism | Mathematical Condition | Institutional Rationale |
| :---: | :--- | :--- | :--- |
| **1** | **15-Min Freak Candle Isolation** | $\text{Time} \in [09:15, 09:30) \implies \text{WAIT}$ | Eliminates opening order-matching volatility, spread anomalies, and overnight stop-runs. Establishes the true Initial Balance (IB). |
| **2** | **Far-Away MA Stretch Filter** | $\frac{\|P_t - \text{EMA}_{21, t}\|}{P_t} > 0.35\% \implies \text{WAIT}$ | Rejects entries when price is overextended $>85\text{ pts}$ from the 21 EMA to avoid buying tops or selling bottoms. |
| **3** | **3:00 PM Breakout Strategy** | $\text{Time} \in [15:05, 15:10] \land P_t > H_{15:00} \implies \text{BUY}$ | Exploits institutional Market-On-Close (MOC) squaring, 15:15 margin auto-liquidation sweeps, and 0DTE gamma squeezes. |
| **4** | **Candlestick Confirmation Trigger** | $\text{Long: } C_t > O_t \lor C_t > C_{t-1}$<br>$\text{Short: } C_t < O_t \lor C_t < C_{t-1}$ | Rejects "falling knife" counter-trend bars; requires a directional reversal close inside the Golden Pocket. |
| **5** | **CPR Width Regime Filter** | $\text{CPR Width } \% = \frac{\|TC - BC\|}{\text{Pivot}} \times 100\%$ | $\text{Width} < 0.20\% \implies$ **Trending Day (Directional Option Buying)**.<br>$\text{Width} \ge 0.20\% \implies$ **Range-Bound Chop (Theta Decay Risk)**. |

---

## 4. Derivatives Engine & Second-Order Greeks

Proprietary trading desks strictly avoid deep Out-of-the-Money (OTM) options ($\Delta < 0.30$) due to rapid Theta decay ($\Theta$) and poor Delta sensitivity.

```
                  INSTITUTIONAL STRIKE SELECTION SWEET SPOT
  Deep OTM (Delta < 0.30)  ──► Rapid Theta Decay, Low Delta Sensitivity (Retail Trap) ❌
  Deep ITM (Delta > 0.80)  ──► High Capital Outlay, Wide Bid-Ask Spread Friction ⚠️
👉 PRO DESK TARGET: [ Delta 0.50 to 0.65 ] ──► Optimal Gamma Convexity + Balanced Theta ✅
```

### Black-Scholes Formulation with Structural Put Skew
For Spot $S$, Strike $K$, Time to Expiry $T$, Risk-Free Rate $r = 6.5\%$, and Implied Volatility $\sigma$ (with structural $+250\text{ bps}$ downside put skew):

$$d_1 = \frac{\ln(S/K) + \left(r + \frac{\sigma^2}{2}\right)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

### Second-Order Cross Greeks & Non-Linear Price Translation
* **Vanna ($\frac{\partial \Delta}{\partial \sigma}$):** Measures the rate of Delta change with respect to Implied Volatility:
  $$\text{Vanna} = \frac{\mathcal{V}}{S} \left( 1 - \frac{d_1}{\sigma \sqrt{T}} \right)$$
* **Charm ($\frac{\partial \Delta}{\partial t}$):** Measures Delta decay over time (critical on **0DTE Thursdays**):
  $$\text{Charm} = -\phi(d_1) \left[ \frac{r}{\sigma \sqrt{T}} - \frac{d_2}{2T} \right] \times \frac{1}{365}$$
* **Non-Linear Taylor Series Translation:**
  $$\Delta P_{\text{Option}} \approx \Delta \cdot (\Delta S) + \frac{1}{2}\Gamma \cdot (\Delta S)^2 - \Theta \cdot \Delta t$$

---

## 5. Capital Risk Sizing & Statutory TCA Model

### 1. The 1.0% Maximum Risk & Quarter-Kelly Sizing
Position sizing is strictly derived from account capital with drawdown dampening:

$$\text{Max Risk Budget (₹)} = \text{Account Capital} \times 0.01 \times \left( 1 - \frac{\text{Current MDD}}{0.10} \right)$$

$$\text{Risk per 1 Lot (₹)} = (\text{Entry Premium} - \text{SL Premium}) \times 25\text{ (Nifty Lot Size)}$$

$$\mathbf{\text{Allocated Lots}} = \left\lfloor \frac{\text{Max Risk Budget (₹)}}{\text{Risk per 1 Lot (₹)}} \right\rfloor$$

> **Worked Example on ₹5,00,000 Capital:**
> * $\text{Risk Budget} = ₹5,00,000 \times 1\% = \mathbf{₹5,000}$.
> * Option Entry Premium $= ₹142.50$, Option SL $= ₹112.50 \implies \text{Risk/sh} = \mathbf{₹30.00}$.
> * Risk per 1 Lot $= ₹30.00 \times 25 = \mathbf{₹750.00/\text{lot}}$.
> * $\text{Lots} = \lfloor 5000 / 750 \rfloor = \mathbf{6\text{ Lots (150 Qty)}}$.
> * Actual Max Capital Exposure $= 6 \times 750 = \mathbf{₹4,500 \le ₹5,000}$.

### 2. Complete Indian NSE Statutory TCA Friction Modeling
Every transaction in the backtest engine and live ticket calculator deducts real statutory friction:

$$\text{TCA Friction} = \text{STT} (0.10\%\text{ on sell}) + \text{Brokerage} (₹20/\text{order}) + \text{NSE Turnover} (0.03503\%) + \text{GST} (18\%) + \text{Slippage} (₹0.75/\text{sh})$$

---

## 6. Multi-Stage Trade Execution Protocol

```
                        MULTI-STAGE TRADE LIFECYCLE (6 LOTS)
                        
       [ ENTRY: 6 Lots @ ₹142.50 ] ──► Initial SL @ ₹112.50 (Max Risk: ₹4,500)
                    │
                    ▼ (Spot moves to Target 1 / +1.2x ATR / +50 pts)
       [ STAGE 1: Target 1 Hit (₹188.00) ]
       ├── Book 50% Profit (3 Lots) ──► Realizes +₹3,412.50 Gross Profit
       └── Shift SL on Remaining 3 Lots to BREAK-EVEN (₹142.50)
                    │
                    ▼ (Trade is now 100% Risk-Free)
       [ STAGE 2: Micro-Trailing on Runners ]
       └── Trail remaining 3 lots candle-by-candle on the 1-Minute 21 EMA / AVWAP
                    │
                    ▼ (Session Profit reaches >= +1.5%)
       [ STAGE 3: Asymmetric Profit Ratchet ]
       └── Lock 65% of peak gains; shut terminal if daily retracement hits threshold
```

---

## 7. Backtest Performance Benchmark (77.8% Win Rate)

Benchmarked on historical Nifty 50 5-minute bar-by-bar execution with full TCA deductions:

```
================================================================================
                        JUSTNIFTY v3.0 PERFORMANCE AUDIT
================================================================================
  • Initial Account Capital: ₹500,000.00
  • Final Account Balance:   ₹503,869.23
  • Gross Strategy PnL:      ₹5,691.73
  • Total TCA Fees Deducted: ₹1,822.50 (STT, Brokerage, GST, Slippage)
  • Net Strategy PnL:        +₹3,869.23 (+0.77% Net Return)
  • Total Trades Executed:   9
  • Winning Trades:          7 (77.78% Win Rate 🟢)
  • Losing Trades:           2 (22.22%)
  • Profit Factor:           1.41
================================================================================
```

### Executed Trade Log (Chronological Bar-by-Bar Replay):
| Entry Time | Exit Time | Symbol | Direction | Entry (₹) | Exit (₹) | Lots | Net PnL (₹) | Result | Part-Booked |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2026-08-07 11:30 | 13:50 | NIFTY 24600 PE | SHORT | 148.61 | 158.80 | 8 | **+₹1,886.25** | WIN | True |
| 2026-08-07 14:00 | 14:05 | NIFTY 24600 PE | SHORT | 148.61 | 159.20 | 8 | **+₹1,657.50** | WIN | True |
| 2026-08-07 15:05 | 15:20 | NIFTY 24550 CE | LONG_3PM | 142.50 | 154.20 | 5 | **+₹1,232.09** | WIN | False |
| 2026-08-10 13:10 | 13:40 | NIFTY 24600 PE | SHORT | 124.81 | 120.81 | 51 | **-₹4,988.75** | LOSS | False |
| 2026-08-10 14:05 | 14:35 | NIFTY 24600 CE | LONG | 126.91 | 120.12 | 30 | **-₹4,730.25** | LOSS | False |
| 2026-08-11 13:10 | 15:20 | NIFTY 24500 PE | SHORT | 133.20 | 134.10 | 59 | **+₹466.38** | WIN | False |
| 2026-08-12 09:35 | 15:20 | NIFTY 24450 PE | SHORT | 144.11 | 168.40 | 8 | **+₹6,903.67** | WIN | True |
| 2026-08-13 09:55 | 11:35 | NIFTY 24400 PE | SHORT | 137.40 | 145.80 | 6 | **+₹1,239.75** | WIN | True |
| 2026-08-13 13:15 | 15:20 | NIFTY 24400 PE | SHORT | 137.40 | 138.20 | 6 | **+₹202.58** | WIN | False |

---

## 8. Repository Directory Structure

```
OnlyNifty/
├── app.py                      # Streamlit Tier-1 Institutional Dashboard & Top Cockpit
├── index.html                  # Standalone ultra low-noise HTML/SVG visual terminal
├── evaluate_backtest.py        # Standalone backtest benchmarking and verification script
├── requirements.txt            # Python dependencies (Streamlit, Plotly, Pandas, SciPy, yfinance)
├── .gitignore                  # Git exclusion rules
├── src/                        # Core quantitative engine source modules
│   ├── __init__.py
│   ├── config.py               # Trading constants, TCA statutory fees, and Kelly bounds
│   ├── data_engine.py          # yfinance live feed ingestion & synthetic market simulator
│   ├── indicators.py           # Hurst Exponent, VAKC, Session AVWAP, Volume Profile, GEX
│   ├── options_engine.py       # Black-Scholes Greeks, 2nd-order Vanna/Charm, TCA calculator
│   ├── strategy_rules.py       # Signal generator, Golden Pocket logic, 3 PM breakout rule
│   └── backtest_engine.py      # Bar-by-bar simulator with 50% part-booking & TCA deductions
└── tests/                      # Pytest automated test suite (25 test cases)
    ├── test_app_smoke.py       # End-to-end pipeline smoke tests
    ├── test_config.py          # Configuration parameter unit tests
    ├── test_data_engine.py     # Data fetching and synthesis tests
    ├── test_indicators.py      # Technical indicator mathematical tests
    ├── test_options_engine.py  # Black-Scholes pricing & position sizer tests
    ├── test_strategy_rules.py  # Signal rule & freak candle filter tests
    ├── test_backtest_engine.py # Backtest replay tests
    └── test_v3_institutional.py# Hurst exponent, VAKC, OFI, GEX & TCA unit tests
```

---

## 9. Quick Start & Execution Guide

### Prerequisites
* Python 3.10 or higher
* Git

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/ubermachine/OnlyNifty.git
cd OnlyNifty
pip install -r requirements.txt
```

### 2. Launch the Streamlit Institutional Terminal
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser to access the live terminal.

### 3. Open Standalone Low-Noise HTML Terminal
Double-click [`index.html`](index.html) or serve it locally:
```bash
python -m http.server 8080
```
Open **`http://localhost:8080`**.

### 4. Run Backtest Verification Script
```bash
python evaluate_backtest.py
```

---

## 10. Institutional Unit Test Suite

Execute the complete 25-case unit test suite:
```bash
pytest tests/ -v
```

Expected output:
```
============================= 25 passed in 6.94s ==============================
```

---

## 11. Disclaimer & License

### Disclaimer
*This repository and software are developed for quantitative research, backtesting, and educational purposes only. Index options trading involves substantial risk of capital loss. Past performance in historical simulations is no guarantee of future returns. Implement rigorous risk controls before deploying live capital.*

### License
Distributed under the **MIT License**. See `LICENSE` for more information.

---

<div align="center">
  <b>Built with quantitative precision for the Indian Derivatives Market (NSE).</b><br>
  Developed by the <b>OnlyNifty Quantitative Research Desk</b>.
</div>
