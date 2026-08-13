"""Streamlit Interactive Application for Nifty Institutional Trading Plan & Options Engine (JustNifty v2.0)."""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime

from src.config import (
    DEFAULT_CAPITAL, MAX_RISK_PCT, LOT_SIZE, ENVELOPE_PCT,
    DEFAULT_IV, ENOUGH_PROFIT_PCT
)
from src.data_engine import DataEngine
from src.indicators import (
    compute_ema, compute_envelopes, compute_vwap, compute_cpr,
    compute_fibonacci_levels, compute_vf_trade_table, compute_volume_profile
)
from src.strategy_rules import StrategyEngine, SignalType
from src.options_engine import (
    generate_option_trade_ticket, select_institutional_strike, black_scholes_greeks
)
from src.backtest_engine import BacktestEngine

st.set_page_config(
    page_title="Nifty Institutional Trading Plan & Options Engine (JustNifty v2.0)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e222d;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #2a2e39;
    }
    .signal-badge-long {
        background-color: #00c853;
        color: white;
        padding: 4px 12px;
        border-radius: 4px;
        font-weight: bold;
    }
    .signal-badge-short {
        background-color: #d50000;
        color: white;
        padding: 4px 12px;
        border-radius: 4px;
        font-weight: bold;
    }
    .signal-badge-wait {
        background-color: #ffab00;
        color: black;
        padding: 4px 12px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎯 Nifty Institutional Trading Plan & Options Engine")
st.caption("Quantitative Execution Framework based on JustNifty v2.0 (EMAs, 1.5% Envelopes, AVWAP, Fib Golden Pocket, 3 PM Strategy & Delta 0.50-0.65 Strike Selection)")

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.header("⚙️ Institutional Risk & Setup")
account_capital = st.sidebar.number_input("Account Capital (₹)", min_value=50000.0, max_value=50000000.0, value=DEFAULT_CAPITAL, step=50000.0)
risk_pct = st.sidebar.slider("Max Capital Risk per Trade (%)", min_value=0.25, max_value=2.0, value=1.0, step=0.05) / 100.0
contract_lot_size = st.sidebar.number_input("Nifty Lot Size", min_value=25, max_value=75, value=LOT_SIZE, step=25)
iv_input = st.sidebar.slider("Expected Implied Volatility / India VIX (%)", min_value=8.0, max_value=30.0, value=DEFAULT_IV * 100.0, step=0.5) / 100.0

st.sidebar.markdown("---")
st.sidebar.header("🕹️ Mode & Timeframe")
data_mode = st.sidebar.radio("Data Stream Source", ["Live / Latest Market Feed (yfinance + NSE)", "Synthetic Market Simulation"])
timeframe = st.sidebar.selectbox("Intraday Timeframe", ["5m (Primary Execution)", "1m (Micro Trailing)"], index=0)
tf_str = "5m" if "5m" in timeframe else "1m"

# Cached helper functions for instant UI interactions
@st.cache_data(ttl=60)
def load_market_data(mode_choice: str, tf: str) -> pd.DataFrame:
    engine = DataEngine(use_cache=True)
    if mode_choice == "Live / Latest Market Feed (yfinance + NSE)":
        return engine.fetch_yfinance_nifty(interval=tf, period="5d")
    else:
        return engine.generate_synthetic_nifty(bars=150, interval_mins=5 if tf == "5m" else 1)

@st.cache_data(ttl=60)
def get_institutional_oi_data() -> pd.DataFrame:
    engine = DataEngine(use_cache=True)
    return engine.get_participant_oi_snapshot()

data_engine = DataEngine(use_cache=True)
strategy_engine = StrategyEngine()

with st.spinner("Streaming Nifty 50 Market Data..."):
    df_raw = load_market_data(data_mode, tf_str)

if df_raw.empty or len(df_raw) < 15:
    st.error("Insufficient market data received. Switching to synthetic fallback...")
    df_raw = data_engine.generate_synthetic_nifty(bars=150, interval_mins=5)

# Indicator Calculations
df = df_raw.copy()
df["ema21"] = compute_ema(df["close"], 21)
df["ema55"] = compute_ema(df["close"], 55)
df["ema200"] = compute_ema(df["close"], 200)
df["env_upper"], df["env_lower"] = compute_envelopes(df["ema200"], ENVELOPE_PCT)
df["vwap"], df["vwap_upper"], df["vwap_lower"] = compute_vwap(df)

current_spot = float(df.iloc[-1]["close"])
prev_spot = float(df.iloc[-2]["close"]) if len(df) > 1 else current_spot
cpr = compute_cpr(df)
vol_profile = compute_volume_profile(df)
vf_table = compute_vf_trade_table(float(df.iloc[0]["open"]), atr=float(df["high"].max() - df["low"].min()) / 4.0)

# Evaluate Current Signal
signal = strategy_engine.evaluate_bar(df)
trade_ticket = generate_option_trade_ticket(current_spot, signal, account_capital)

# ----------------- TOP METRIC BAR -----------------
m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
spot_delta = current_spot - prev_spot
m_col1.metric("Nifty 50 Spot", f"₹{current_spot:,.2f}", f"{spot_delta:+.2f} ({spot_delta/prev_spot*100:+.2f}%)")
m_col2.metric("Market Regime", cpr["regime"].split("(")[0].strip(), f"CPR: {cpr['width_pct']:.2f}%")
m_col3.metric("Session AVWAP", f"₹{float(df.iloc[-1]['vwap']):,.2f}", f"Diff: {current_spot - float(df.iloc[-1]['vwap']):+.1f}")
m_col4.metric("200 EMA (5m)", f"₹{float(df.iloc[-1]['ema200']):,.2f}", f"Diff: {current_spot - float(df.iloc[-1]['ema200']):+.1f}")
m_col5.metric("Value Area POC", f"₹{vol_profile['poc']:,.2f}", f"VAH: {vol_profile['vah']:.0f} | VAL: {vol_profile['val']:.0f}")
m_col6.metric("Active Setup", signal.signal_type.value, "1% Risk Ready" if trade_ticket.get("status") == "READY" else "Waiting")

# ----------------- MAIN INTERFACE TABS -----------------
tab_chart, tab_ticket, tab_oi, tab_backtest, tab_cheatsheet = st.tabs([
    "📈 Interactive Candlestick Chart",
    "🎟️ Institutional Option Ticket & Strike Sizer",
    "🏛️ Institutional Participant OI & Option Chain",
    "📊 Bar-by-Bar Replay & Backtest Engine",
    "📖 JustNifty v2.0 Algorithm & Rules Reference"
])

# ----- TAB 1: CHART -----
with tab_chart:
    st.subheader("Nifty 50 Multi-Indicator Interactive Chart")
    
    # Overlay Toggles
    c_t1, c_t2, c_t3, c_t4, c_t5 = st.columns(5)
    show_emas = c_t1.checkbox("Show 200/55/21 EMAs", value=True)
    show_env = c_t2.checkbox("Show 1.5% Envelopes", value=True)
    show_vwap = c_t3.checkbox("Show Session AVWAP ±2σ", value=True)
    show_fib = c_t4.checkbox("Show Fib Golden Pocket", value=True)
    show_cpr = c_t5.checkbox("Show CPR Pivot Levels", value=False)
    
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.78, 0.22], vertical_spacing=0.03,
        subplot_titles=("Price Action & Confluence Overlays", "Intraday Volume")
    )
    
    # Candlestick Trace
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Nifty 50",
        increasing_line_color="#00e676", decreasing_line_color="#ff1744"
    ), row=1, col=1)
    
    if show_emas:
        fig.add_trace(go.Scatter(x=df.index, y=df["ema200"], name="200 EMA (Regime)", line=dict(color="#00e676", width=2.2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["ema55"], name="55 EMA (Trend)", line=dict(color="#ff9100", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["ema21"], name="21 EMA (Trailing)", line=dict(color="#2979ff", width=1.5)), row=1, col=1)
        
    if show_env:
        fig.add_trace(go.Scatter(x=df.index, y=df["env_upper"], name="1.5% Upper Env (Extreme)", line=dict(color="#ff1744", width=1.2, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["env_lower"], name="1.5% Lower Env (Extreme)", line=dict(color="#00e676", width=1.2, dash="dot")), row=1, col=1)
        
    if show_vwap:
        fig.add_trace(go.Scatter(x=df.index, y=df["vwap"], name="Session AVWAP", line=dict(color="#d500f9", width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["vwap_upper"], name="+2σ AVWAP Band", line=dict(color="rgba(213,0,249,0.4)", width=1, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["vwap_lower"], name="-2σ AVWAP Band", line=dict(color="rgba(213,0,249,0.4)", width=1, dash="dash")), row=1, col=1)
        
    if show_cpr and cpr["pivot"] > 0:
        fig.add_hline(y=cpr["pivot"], line_dash="dash", line_color="#ffd600", annotation_text="CPR Pivot", row=1, col=1)
        fig.add_hline(y=cpr["tc"], line_dash="dot", line_color="#ffd600", annotation_text="CPR TC", row=1, col=1)
        fig.add_hline(y=cpr["bc"], line_dash="dot", line_color="#ffd600", annotation_text="CPR BC", row=1, col=1)
        
    if show_fib and len(df) >= 30:
        s_high = df["high"].tail(30).max()
        s_low = df["low"].tail(30).min()
        fib = compute_fibonacci_levels(s_high, s_low, is_uptrend=current_spot > float(df["ema200"].iloc[-1]))
        fig.add_hrect(
            y0=min(fib["fib_500"], fib["fib_618"]),
            y1=max(fib["fib_500"], fib["fib_618"]),
            fillcolor="rgba(255, 215, 0, 0.15)",
            line_width=1,
            line_color="gold",
            annotation_text="Golden Pocket (50% - 61.8%)",
            annotation_position="top left",
            row=1, col=1
        )
        
    # Volume subplot
    colors = ["#00e676" if c >= o else "#ff1744" for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume", marker_color=colors, opacity=0.7), row=2, col=1)
    
    fig.update_layout(
        height=700,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# ----- TAB 2: OPTION TRADE TICKET -----
with tab_ticket:
    st.subheader("Institutional Option Execution & Position Sizing Ticket")
    
    status_color = "green" if "LONG" in signal.signal_type.value else ("red" if "SHORT" in signal.signal_type.value else "orange")
    st.markdown(f"### Current Status: <span style='color:{status_color}; font-weight:bold;'>{signal.signal_type.value}</span>", unsafe_allow_html=True)
    st.info(f"**Diagnostic Evaluation:** {signal.reason}")
    
    if trade_ticket.get("status") == "READY":
        st.success("✅ **High-Conviction Institutional Setup Confirmed.** Complete Trade Ticket Generated Below:")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Recommended Strike", trade_ticket["symbol"])
        c2.metric("Target Delta (Δ)", f"{trade_ticket['delta']:.2f}")
        c3.metric("Gamma (Γ)", f"{trade_ticket['gamma']:.5f}")
        c4.metric("Daily Theta Decay (Θ)", f"₹{trade_ticket['theta_decay_daily']:.2f}/share")
        
        st.markdown("#### 🎯 Execution Price Levels (Option Premiums Translated via Delta)")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Option Entry Premium", f"₹{trade_ticket['entry_premium']:.2f}")
        p2.metric("Option Stop-Loss", f"₹{trade_ticket['sl_premium']:.2f}", f"Risk: -₹{trade_ticket['entry_premium'] - trade_ticket['sl_premium']:.2f}/sh", delta_color="inverse")
        p3.metric("Target 1 (50% Part-Book)", f"₹{trade_ticket['target1_premium']:.2f}", f"+₹{trade_ticket['target1_premium'] - trade_ticket['entry_premium']:.2f}/sh")
        p4.metric("Target 2 (Runner Target)", f"₹{trade_ticket['target2_premium']:.2f}", f"+₹{trade_ticket['target2_premium'] - trade_ticket['entry_premium']:.2f}/sh")
        
        st.markdown("#### 🛡️ Institutional 1% Capital Risk Allocation")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Allocated Position Size", f"{trade_ticket['lots']} Lots ({trade_ticket['total_qty']} Qty)")
        s2.metric("Total 1% Risk Exposure", f"₹{trade_ticket['max_risk_rupees']:,.2f}", f"{risk_pct*100:.2f}% of ₹{account_capital:,.0f}")
        s3.metric("Total Capital Outlay", f"₹{trade_ticket['capital_outlay']:,.2f}", f"{trade_ticket['capital_outlay']/account_capital*100:.1f}% Margin")
        s4.metric("Reward to Risk Ratio", f"{(trade_ticket['target1_premium'] - trade_ticket['entry_premium'])/(trade_ticket['entry_premium'] - trade_ticket['sl_premium']):.2f} : 1")
        
        st.markdown("#### 📜 Exact Institutional Trade Management Plan")
        for key, val in trade_ticket["execution_rules"].items():
            st.markdown(f"- **{key.replace('_', ' ').title()}:** {val}")
    else:
        st.warning("⚠️ **No Active Execution Trigger At This Moment.**")
        st.markdown("""
        **Why is the trade engine waiting?**
        - Institutional option buying requires strict confluence: (1) Above/Below 200 EMA, (2) Aligned with 09:15 Session AVWAP, (3) Pullback into the 50.0% to 61.8% Fibonacci Golden Pocket, and (4) Price not overextended from the 21 EMA.
        - Monitor the chart for pullback confirmation into the Golden Pocket, or watch for the **3:00 PM Aggressive Breakout Setup** between 15:00 and 15:10 IST.
        """)

# ----- TAB 3: INSTITUTIONAL OI & OPTION CHAIN -----
with tab_oi:
    st.subheader("🏛️ Institutional Participant-Wise Open Interest (FII / Prop Desks vs Retail)")
    st.dataframe(get_institutional_oi_data(), use_container_width=True)
    
    st.subheader("🔍 Algorithmic Strike Selection Ladder & Greeks Matrix")
    atm_center = int(round(current_spot / 50.0) * 50)
    chain_rows = []
    
    for k in range(atm_center - 250, atm_center + 300, 50):
        ce_greeks = black_scholes_greeks(current_spot, k, t_days=4.0, sigma=iv_input, is_call=True)
        pe_greeks = black_scholes_greeks(current_spot, k, t_days=4.0, sigma=iv_input, is_call=False)
        
        is_atm = (k == atm_center)
        ce_rec = "👉 PRO CALL BUY" if (0.50 <= ce_greeks["delta"] <= 0.65) else ""
        pe_rec = "👉 PRO PUT BUY" if (0.50 <= abs(pe_greeks["delta"]) <= 0.65) else ""
        
        chain_rows.append({
            "CE Delta": ce_greeks["delta"],
            "CE Theta": ce_greeks["theta"],
            "CE Premium (₹)": ce_greeks["price"],
            "Call Setup": ce_rec,
            "Strike": f"🎯 {k} (ATM)" if is_atm else str(k),
            "Put Setup": pe_rec,
            "PE Premium (₹)": pe_greeks["price"],
            "PE Theta": pe_greeks["theta"],
            "PE Delta": pe_greeks["delta"]
        })
        
    st.dataframe(pd.DataFrame(chain_rows), use_container_width=True)
    
    st.markdown("#### 💡 The VF Trade Table Targets (T1 to T6)")
    vf_cols = st.columns(6)
    for i in range(1, 7):
        vf_cols[i-1].metric(f"Level T{i}", f"L: {vf_table[f'T{i}_Long']:.0f}", f"S: {vf_table[f'T{i}_Short']:.0f}")

# ----- TAB 4: BACKTESTING & REPLAY -----
with tab_backtest:
    st.subheader("📊 Bar-by-Bar Replay & Strategy Backtest Engine")
    st.markdown("Simulates the complete JustNifty v2.0 execution model (Golden Pocket entries, 50% part-booking at T1 / Envelope, breakeven SL adjustment, and 21 EMA / AVWAP trailing).")
    
    if st.button("🚀 Run Backtest on Active Dataset", use_container_width=True):
        bt_engine = BacktestEngine(initial_capital=account_capital)
        results = bt_engine.run_backtest(df)
        
        s1, s2, s3, s4 = st.columns(4)
        pnl_color = "normal" if results.summary["pnl_rupees"] >= 0 else "inverse"
        s1.metric("Net Strategy PnL", f"₹{results.summary['pnl_rupees']:,.2f}", f"{results.summary['return_pct']:+.2f}%", delta_color=pnl_color)
        s2.metric("Win Rate", f"{results.summary['win_rate']:.1f}%", f"{results.summary['wins']}W / {results.summary['losses']}L")
        s3.metric("Total Trades Executed", results.summary["total_trades"])
        s4.metric("Final Account Balance", f"₹{results.summary['final_capital']:,.2f}")
        
        if results.trade_log:
            st.markdown("#### 📜 Executed Trade Log")
            df_trades = pd.DataFrame(results.trade_log)
            st.dataframe(df_trades, use_container_width=True)
            
            # Equity curve chart
            st.markdown("#### 📈 Account Equity Curve")
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(y=results.equity_curve, mode="lines+markers", line=dict(color="#00e676", width=2), name="Equity (₹)"))
            fig_eq.update_layout(template="plotly_dark", height=350, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_eq, use_container_width=True)
        else:
            st.info("No completed trade setups triggered within this specific historical slice.")

# ----- TAB 5: CHEATSHEET & RULES -----
with tab_cheatsheet:
    st.subheader("📖 JustNifty v2.0 Algorithm & Institutional Rules Cheatsheet")
    st.markdown(r"""
    ### 1. The 4 Core Technical Pillars
    1. **Price Action:** HH/HL for Uptrends, LH/LL for Downtrends. Candlestick trigger confirmations (Engulfing / Hammer / Doji).
    2. **Retracement:** 50.0% to 61.8% Fibonacci Golden Pocket.
    3. **Moving Averages:** 200 EMA (Regime), 55 EMA (Trend), 21 EMA (Momentum/Trailing).
    4. **Envelopes:** 1.5% 200 EMA bands for spotting extreme exhaustion and mechanical 50% part-booking.

    ### 2. The Missing 15% Mechanisms Incorporated in v2.0
    - **Session AVWAP (09:15 Anchor):** Above AVWAP $\rightarrow$ Buyers in profit (Bullish only). Below AVWAP $\rightarrow$ Sellers in profit (Bearish only).
    - **15-Min Freak Candle Isolation:** First 15 minutes (09:15 - 09:30) ignored for breakout entries to establish true Initial Balance.
    - **Far-Away MA Crossover Filter (Query 12):** If price is $>0.35\%$ stretched from 21/55 EMA, reject market entry; wait for mean-reversion pullback.
    - **Higher Timeframe Weightage (Query 79):** Daily/Hourly hierarchy takes precedence until 5m reaches extreme oversold/overbought zones.
    - **3:00 PM Aggressive Breakout (Page 100):** On range-bound days, trade the breakout of 15:00 candle High/Low for fast 80-160 point expansion.

    ### 3. Institutional Strike Selection & Risk Management
    - **Strike Window:** Target Delta $\Delta \in [0.50, 0.65]$ (ATM to 1-strike ITM). Never buy deep OTM lottery options.
    - **The 1% Rule:** Max allowable loss per trade strictly capped at $1.0\%$ of total account capital.
    - **50% Part-Booking:** Mechanically book 50% lots at VF Table T1 or the 1.5% 200 EMA Envelope. Move SL to Break-Even immediately.
    - **Runner Trailing:** Trail the remaining 50% lots on the 1-minute 21 EMA or AVWAP.
    - **The "Enough" Rule:** If session gains reach $\ge 0.3\%$ on capital, lock profits and shut the terminal.
    """)
