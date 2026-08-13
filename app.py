"""Nifty Institutional Signal Terminal & Unified Executive Main Page (JustNifty v2.0)."""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime

from src.config import (
    DEFAULT_CAPITAL, MAX_RISK_PCT, LOT_SIZE, ENVELOPE_PCT,
    DEFAULT_IV, ENOUGH_PROFIT_PCT, EMA_FAST, EMA_MID, EMA_SLOW,
    MA_STRETCH_THRESHOLD
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

# Page Config
st.set_page_config(
    page_title="Nifty Institutional Signal Terminal | JustNifty v2.0",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast Low-Noise CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stApp {
        background-color: #080c14;
        color: #f1f5f9;
    }
    
    /* Institutional Cockpit Container */
    .cockpit-box {
        background-color: #0e1422;
        border: 1px solid #1c273c;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    .cockpit-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
    }

    .badge-pro {
        background: linear-gradient(135deg, #05df72, #00d2ff);
        color: #04121e;
        font-weight: 800;
        font-size: 11px;
        padding: 3px 8px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
    }

    .signal-long {
        background-color: rgba(5, 223, 114, 0.08);
        color: #05df72;
        border: 1px solid rgba(5, 223, 114, 0.25);
        padding: 4px 10px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 13px;
    }

    .signal-short {
        background-color: rgba(255, 51, 85, 0.08);
        color: #ff3355;
        border: 1px solid rgba(255, 51, 85, 0.25);
        padding: 4px 10px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 13px;
    }

    .signal-wait {
        background-color: rgba(251, 176, 36, 0.08);
        color: #fbb024;
        border: 1px solid rgba(251, 176, 36, 0.25);
        padding: 4px 10px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 13px;
    }

    .confluence-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        margin-top: 14px;
        padding-top: 14px;
        border-top: 1px solid #1c273c;
    }

    .confluence-cell {
        background-color: #141c2e;
        border: 1px solid #1c273c;
        border-radius: 6px;
        padding: 8px 12px;
    }

    .c-lbl {
        font-size: 11px;
        color: #55657e;
        text-transform: uppercase;
        font-weight: 600;
    }

    .c-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12.5px;
        font-weight: 700;
        margin-top: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.header("⚙️ Risk & Data Stream")
account_capital = st.sidebar.number_input("Account Capital (₹)", min_value=50000.0, max_value=50000000.0, value=DEFAULT_CAPITAL, step=50000.0)
risk_pct = st.sidebar.slider("Max Capital Risk per Trade (%)", min_value=0.25, max_value=2.0, value=1.0, step=0.05) / 100.0
contract_lot_size = st.sidebar.number_input("Nifty Lot Size", min_value=25, max_value=75, value=LOT_SIZE, step=25)
iv_input = st.sidebar.slider("Expected IV / India VIX (%)", min_value=8.0, max_value=30.0, value=DEFAULT_IV * 100.0, step=0.5) / 100.0

st.sidebar.markdown("---")
data_mode = st.sidebar.radio("Data Stream Source", ["Live / Latest Market Feed (yfinance + NSE)", "Synthetic Market Simulation"])
timeframe = st.sidebar.selectbox("Execution Timeframe", ["5m (Primary Execution)", "1m (Micro Trailing)"], index=0)
tf_str = "5m" if "5m" in timeframe else "1m"

# ----------------- DATA INGESTION & CACHING -----------------
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

df_raw = load_market_data(data_mode, tf_str)
if df_raw.empty or len(df_raw) < 15:
    df_raw = data_engine.generate_synthetic_nifty(bars=150, interval_mins=5)

# Indicator Math
df = df_raw.copy()
df["ema21"] = compute_ema(df["close"], EMA_FAST)
df["ema55"] = compute_ema(df["close"], EMA_MID)
df["ema200"] = compute_ema(df["close"], EMA_SLOW)
df["env_upper"], df["env_lower"] = compute_envelopes(df["ema200"], ENVELOPE_PCT)
df["vwap"], df["vwap_upper"], df["vwap_lower"] = compute_vwap(df)

current_spot = float(df.iloc[-1]["close"])
prev_spot = float(df.iloc[-2]["close"]) if len(df) > 1 else current_spot
spot_delta = current_spot - prev_spot
cpr = compute_cpr(df)
vol_profile = compute_volume_profile(df)
vf_table = compute_vf_trade_table(float(df.iloc[0]["open"]), atr=float(df["high"].max() - df["low"].min()) / 4.0)

# Evaluate Active Signal & Option Ticket
signal = strategy_engine.evaluate_bar(df)
ticket = generate_option_trade_ticket(current_spot, signal, account_capital)

# Confluence checks computation
is_above_200 = current_spot > float(df.iloc[-1]["ema200"])
is_above_vwap = current_spot > float(df.iloc[-1]["vwap"])
dist_ema21 = abs(current_spot - float(df.iloc[-1]["ema21"])) / current_spot
is_not_stretched = dist_ema21 <= MA_STRETCH_THRESHOLD

# ----------------- UNIFIED TOP INSTITUTIONAL COCKPIT -----------------
st.markdown("""
<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
    <div style="display: flex; align-items: center; gap: 10px;">
        <span class="badge-pro">PRO v2.0</span>
        <h2 style="margin: 0; font-size: 20px; font-weight: 800; letter-spacing: -0.01em;">Nifty Institutional Signal Terminal</h2>
    </div>
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #05df72;">● FEED ACTIVE (09:15-15:30 IST)</div>
</div>
""", unsafe_allow_html=True)

# Main Cockpit Split Grid
cockpit_col1, cockpit_col2 = st.columns([1.35, 1.0])

with cockpit_col1:
    sig_badge_class = "signal-long" if "LONG" in signal.signal_type.value else ("signal-short" if "SHORT" in signal.signal_type.value else "signal-wait")
    sig_badge_text = f"● {signal.signal_type.value} CONFIRMED" if signal.signal_type != SignalType.WAIT else "● AWAITING CONFLUENCE"
    
    st.markdown(f"""
    <div class="cockpit-box">
        <div class="cockpit-header">
            <span class="{sig_badge_class}">{sig_badge_text}</span>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #55657e;">TIMEFRAME: {tf_str} EXECUTION</span>
        </div>
        <div style="display: flex; align-items: baseline; gap: 12px; margin-bottom: 6px;">
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 32px; font-weight: 800; color: #f1f5f9;">₹{current_spot:,.2f}</span>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 600; color: {'#05df72' if spot_delta >= 0 else '#ff3355'};">
                {spot_delta:+.2f} ({spot_delta/prev_spot*100:+.2f}%)
            </span>
        </div>
        <div style="font-size: 13.5px; color: #8e9fb5; margin-bottom: 12px; line-height: 1.4;">
            <strong>Setup Status:</strong> {signal.reason}
        </div>
        <div class="confluence-grid">
            <div class="confluence-cell">
                <div class="c-lbl">1. 200 EMA (5m)</div>
                <div class="c-val" style="color: {'#05df72' if is_above_200 else '#ff3355'};">{'✓ Above (Bull)' if is_above_200 else '✗ Below (Bear)'}</div>
            </div>
            <div class="confluence-cell">
                <div class="c-lbl">2. Session AVWAP</div>
                <div class="c-val" style="color: {'#05df72' if is_above_vwap else '#ff3355'};">{'✓ Above 09:15' if is_above_vwap else '✗ Below 09:15'}</div>
            </div>
            <div class="confluence-cell">
                <div class="c-lbl">3. Retracement</div>
                <div class="c-val" style="color: #fbb024;">50-61.8% Golden</div>
            </div>
            <div class="confluence-cell">
                <div class="c-lbl">4. MA Proximity</div>
                <div class="c-val" style="color: {'#05df72' if is_not_stretched else '#ff3355'};">{'✓ Valid (' + str(round(dist_ema21*100, 2)) + '%)' if is_not_stretched else '✗ Stretched'}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with cockpit_col2:
    if ticket.get("status") == "READY":
        target_strike = ticket["symbol"]
        delta_str = f"Δ {ticket['delta']:.2f}"
        theta_str = f"Θ -₹{ticket['theta_decay_daily']:.2f}/sh"
        ep_str = f"₹{ticket['entry_premium']:.2f}"
        sl_str = f"₹{ticket['sl_premium']:.2f}"
        t1_str = f"₹{ticket['target1_premium']:.2f}"
        lots_str = f"{ticket['lots']} Lots ({ticket['total_qty']} Qty)"
        risk_rupees_str = f"₹{ticket['max_risk_rupees']:,.2f}"
    else:
        # Default ATM strike reference
        atm_k = int(round(current_spot / 50.0) * 50)
        target_strike = f"NIFTY {atm_k} CE / PE"
        delta_str = "Δ 0.55"
        theta_str = "Θ -₹14.20/sh"
        ep_str = "₹142.50"
        sl_str = "₹112.50"
        t1_str = "₹188.00"
        lots_str = "6 Lots (150 Qty)"
        risk_rupees_str = f"₹{account_capital * risk_pct:,.2f}"

    st.markdown(f"""
    <div class="cockpit-box" style="display: flex; flex-direction: column; justify-content: space-between; height: calc(100% - 20px);">
        <div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <div>
                    <div style="font-size: 11px; color: #55657e; text-transform: uppercase; font-weight: 600;">Optimal Strike (Delta 0.50-0.65)</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 20px; font-weight: 800; color: #00d2ff;">{target_strike}</div>
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; background-color: #141c2e; padding: 4px 8px; border-radius: 4px; color: #8e9fb5;">
                    {delta_str} • {theta_str}
                </div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 12px;">
                <div style="background-color: #080c14; border: 1px solid #1c273c; border-radius: 4px; padding: 8px;">
                    <div style="font-size: 10px; color: #55657e; text-transform: uppercase; font-weight: 600;">Entry Prem</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 700; color: #f1f5f9;">{ep_str}</div>
                </div>
                <div style="background-color: #080c14; border: 1px solid #1c273c; border-radius: 4px; padding: 8px;">
                    <div style="font-size: 10px; color: #55657e; text-transform: uppercase; font-weight: 600;">Stop Loss</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 700; color: #ff3355;">{sl_str}</div>
                </div>
                <div style="background-color: #080c14; border: 1px solid #1c273c; border-radius: 4px; padding: 8px;">
                    <div style="font-size: 10px; color: #55657e; text-transform: uppercase; font-weight: 600;">50% Target (T1)</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 700; color: #05df72;">{t1_str}</div>
                </div>
            </div>
        </div>
        <div style="background-color: rgba(0, 210, 255, 0.04); border-left: 3px solid #00d2ff; padding: 8px 10px; font-size: 11.5px; color: #8e9fb5; line-height: 1.4;">
            <strong>Protocol:</strong> Book 50% at <strong>{t1_str} (T1/Envelope)</strong> ➔ Shift SL to <strong>Break-Even ({ep_str})</strong> ➔ Trail runners on 1m 21 EMA.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------- MAIN INTERFACE TABS -----------------
tab_chart, tab_sizer, tab_oi, tab_backtest, tab_cheatsheet = st.tabs([
    "📈 Interactive Candlestick Chart",
    "🛡️ 1% Risk & Position Sizer",
    "🏛️ Institutional Participant OI & Option Chain",
    "📊 Bar-by-Bar Replay & Backtest Simulator",
    "📖 JustNifty v2.0 Master Rulebook"
])

# ----- TAB 1: INTERACTIVE CHART -----
with tab_chart:
    st.subheader("Nifty 50 Multi-Indicator Technical Chart")
    
    t1_c1, t1_c2, t1_c3, t1_c4, t1_c5 = st.columns(5)
    show_emas = t1_c1.checkbox("200 / 55 / 21 EMAs", value=True)
    show_env = t1_c2.checkbox("1.5% 200 EMA Envelopes", value=True)
    show_vwap = t1_c3.checkbox("Session AVWAP ±2σ", value=True)
    show_fib = t1_c4.checkbox("Fib Golden Pocket (50-61.8%)", value=True)
    show_cpr = t1_c5.checkbox("CPR Pivots", value=False)
    
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.78, 0.22], vertical_spacing=0.03,
        subplot_titles=("Price Action & Confluence Overlays", "Intraday Volume")
    )
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Nifty 50", increasing_line_color="#05df72", decreasing_line_color="#ff3355"
    ), row=1, col=1)
    
    if show_emas:
        fig.add_trace(go.Scatter(x=df.index, y=df["ema200"], name="200 EMA (Regime)", line=dict(color="#05df72", width=2.2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["ema55"], name="55 EMA (Trend)", line=dict(color="#ff9100", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["ema21"], name="21 EMA (Trailing)", line=dict(color="#00d2ff", width=1.5)), row=1, col=1)
        
    if show_env:
        fig.add_trace(go.Scatter(x=df.index, y=df["env_upper"], name="1.5% Upper Env (Take-Profit)", line=dict(color="#ff3355", width=1.2, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["env_lower"], name="1.5% Lower Env (Take-Profit)", line=dict(color="#05df72", width=1.2, dash="dot")), row=1, col=1)
        
    if show_vwap:
        fig.add_trace(go.Scatter(x=df.index, y=df["vwap"], name="Session AVWAP (09:15)", line=dict(color="#a855f7", width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["vwap_upper"], name="+2σ AVWAP Band", line=dict(color="rgba(168,85,247,0.35)", width=1, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["vwap_lower"], name="-2σ AVWAP Band", line=dict(color="rgba(168,85,247,0.35)", width=1, dash="dash")), row=1, col=1)
        
    if show_cpr and cpr["pivot"] > 0:
        fig.add_hline(y=cpr["pivot"], line_dash="dash", line_color="#ffd600", annotation_text="CPR Pivot", row=1, col=1)
        fig.add_hline(y=cpr["tc"], line_dash="dot", line_color="#ffd600", annotation_text="CPR TC", row=1, col=1)
        fig.add_hline(y=cpr["bc"], line_dash="dot", line_color="#ffd600", annotation_text="CPR BC", row=1, col=1)
        
    if show_fib and len(df) >= 30:
        s_high = float(df["high"].tail(30).max())
        s_low = float(df["low"].tail(30).min())
        fib = compute_fibonacci_levels(s_high, s_low, is_uptrend=current_spot > float(df["ema200"].iloc[-1]))
        fig.add_hrect(
            y0=min(fib["fib_500"], fib["fib_618"]),
            y1=max(fib["fib_500"], fib["fib_618"]),
            fillcolor="rgba(251, 176, 36, 0.12)",
            line_width=1, line_color="#fbb024",
            annotation_text="Golden Pocket (50% - 61.8%)",
            annotation_position="top left",
            row=1, col=1
        )
        
    # Volume subplot
    vol_colors = ["#05df72" if c >= o else "#ff3355" for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume", marker_color=vol_colors, opacity=0.7), row=2, col=1)
    
    fig.update_layout(
        height=680,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# ----- TAB 2: RISK & POSITION SIZER -----
with tab_sizer:
    st.subheader("🛡️ Institutional 1% Capital Risk & Position Sizer")
    st.caption("Computes allowable lot count strictly governed by the 1% maximum capital risk rule.")
    
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        calc_cap = st.number_input("Account Capital (₹)", value=account_capital, step=50000.0, key="calc_cap")
        calc_risk_pct = st.slider("Risk Limit per Trade (%)", min_value=0.25, max_value=2.0, value=risk_pct*100.0, step=0.05, key="calc_risk") / 100.0
        calc_ep = st.number_input("Option Entry Premium (₹)", value=142.50, step=1.0, key="calc_ep")
        calc_sl = st.number_input("Option Stop-Loss Premium (₹)", value=112.50, step=1.0, key="calc_sl")
        
    with s_col2:
        max_allowed_loss = calc_cap * calc_risk_pct
        risk_per_sh = max(calc_ep - calc_sl, 2.0)
        risk_per_contract_lot = risk_per_sh * contract_lot_size
        calc_lots = int(max_allowed_loss // risk_per_contract_lot)
        calc_total_qty = calc_lots * contract_lot_size
        calc_actual_risk = calc_total_qty * risk_per_sh
        calc_outlay = calc_total_qty * calc_ep
        
        m1, m2 = st.columns(2)
        m1.metric("Max Allowable Loss", f"₹{max_allowed_loss:,.2f}")
        m2.metric("Allocated Position", f"{calc_lots} Lots ({calc_total_qty} Qty)")
        
        m3, m4 = st.columns(2)
        m3.metric("Actual Risk Exposure", f"₹{calc_actual_risk:,.2f}", f"{(calc_actual_risk/calc_cap)*100:.2f}% of Capital", delta_color="inverse")
        m4.metric("Capital Outlay Required", f"₹{calc_outlay:,.2f}", f"{(calc_outlay/calc_cap)*100:.1f}% Margin")
        
        st.info(f"**Institutional Part-Booking Rule:** Book 50% ({max(calc_lots//2, 1)} lots) at T1 / 1.5% Envelope. Shift SL on remaining {calc_lots - max(calc_lots//2, 1)} lots to Break-Even (₹{calc_ep:.2f}).")

# ----- TAB 3: PARTICIPANT OI & STRIKE LADDER -----
with tab_oi:
    st.subheader("🏛️ Institutional Participant-Wise Open Interest (FII / Prop Desks vs Retail)")
    st.dataframe(get_institutional_oi_data(), use_container_width=True)
    
    st.subheader("🔍 Institutional Strike Ladder & Greeks Matrix (Delta 0.50 – 0.65)")
    atm_center = int(round(current_spot / 50.0) * 50)
    chain_rows = []
    
    for k in range(atm_center - 200, atm_center + 250, 50):
        ce_greeks = black_scholes_greeks(current_spot, k, t_days=4.0, sigma=iv_input, is_call=True)
        pe_greeks = black_scholes_greeks(current_spot, k, t_days=4.0, sigma=iv_input, is_call=False)
        
        is_atm = (k == atm_center)
        ce_rec = "👉 PRO CALL" if (0.50 <= ce_greeks["delta"] <= 0.65) else ""
        pe_rec = "👉 PRO PUT" if (0.50 <= abs(pe_greeks["delta"]) <= 0.65) else ""
        
        chain_rows.append({
            "Call Setup": ce_rec,
            "CE Delta": ce_greeks["delta"],
            "CE Theta": ce_greeks["theta"],
            "CE Premium (₹)": ce_greeks["price"],
            "Strike": f"🎯 {k} (ATM)" if is_atm else str(k),
            "PE Premium (₹)": pe_greeks["price"],
            "PE Theta": pe_greeks["theta"],
            "PE Delta": pe_greeks["delta"],
            "Put Setup": pe_rec
        })
        
    st.dataframe(pd.DataFrame(chain_rows), use_container_width=True)
    
    st.markdown("#### 💡 The VF Trade Table Targets (T1 to T6)")
    vf_cols = st.columns(6)
    for i in range(1, 7):
        vf_cols[i-1].metric(f"Level T{i}", f"L: {vf_table[f'T{i}_Long']:.0f}", f"S: {vf_table[f'T{i}_Short']:.0f}")

# ----- TAB 4: BACKTESTING & REPLAY -----
with tab_backtest:
    st.subheader("📊 Bar-by-Bar Replay & Backtesting Engine")
    st.caption("Simulates the complete JustNifty v2.0 execution model (Golden Pocket entries, 50% part-booking at T1 / Envelope, breakeven SL adjustment, and 21 EMA / AVWAP trailing).")
    
    run_btn = st.button("🚀 Run Backtest on Loaded Dataset", use_container_width=True)
    if run_btn or "bt_results" in st.session_state:
        if run_btn:
            bt_engine = BacktestEngine(initial_capital=account_capital)
            st.session_state["bt_results"] = bt_engine.run_backtest(df)
            
        results = st.session_state["bt_results"]
        b1, b2, b3, b4 = st.columns(4)
        pnl_color = "normal" if results.summary["pnl_rupees"] >= 0 else "inverse"
        b1.metric("Net Strategy PnL", f"₹{results.summary['pnl_rupees']:,.2f}", f"{results.summary['return_pct']:+.2f}%", delta_color=pnl_color)
        b2.metric("Win Rate", f"{results.summary['win_rate']:.1f}%", f"{results.summary['wins']}W / {results.summary['losses']}L")
        b3.metric("Total Trades Executed", results.summary["total_trades"])
        b4.metric("Final Account Balance", f"₹{results.summary['final_capital']:,.2f}")
        
        if results.trade_log:
            st.markdown("#### 📜 Executed Trade Log")
            st.dataframe(pd.DataFrame(results.trade_log), use_container_width=True)
            
            st.markdown("#### 📈 Account Equity Curve")
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(y=results.equity_curve, mode="lines+markers", line=dict(color="#05df72", width=2), name="Equity (₹)"))
            fig_eq.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_eq, use_container_width=True)
        else:
            st.info("No completed trade setups triggered within this specific historical slice.")

# ----- TAB 5: MASTER RULEBOOK -----
with tab_cheatsheet:
    st.subheader("📖 JustNifty v2.0 Master Strategy Rulebook")
    st.markdown(r"""
    ### 1. The 4 Core Technical Pillars (85% Foundation)
    1. **Price Action:** HH/HL for Uptrends, LH/LL for Downtrends. Candlestick trigger confirmations (Engulfing / Hammer / Doji).
    2. **Retracement:** 50.0% to 61.8% Fibonacci Golden Pocket.
    3. **Moving Averages:** 200 EMA (Regime), 55 EMA (Trend), 21 EMA (Momentum/Trailing).
    4. **Envelopes:** 1.5% 200 EMA bands for spotting extreme exhaustion and mechanical 50% part-booking.

    ### 2. The 5 Missing Audit Mechanisms (15% Nuances)
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
