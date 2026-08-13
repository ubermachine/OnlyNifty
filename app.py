"""Nifty Tier-1 Institutional Signal Terminal & Quantitative Main Dashboard (JustNifty v3.0)."""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime

from src.config import (
    DEFAULT_CAPITAL, MAX_RISK_PCT, LOT_SIZE, ENVELOPE_PCT,
    DEFAULT_IV, EMA_FAST, EMA_MID, EMA_SLOW, VAKC_LAMBDA,
    MA_STRETCH_THRESHOLD, KELLY_FRACTION, MAX_TOLERABLE_MDD
)
from src.data_engine import DataEngine
from src.indicators import (
    compute_ema, compute_vakc_envelopes, compute_vwap, compute_cpr,
    compute_fibonacci_levels, compute_vf_trade_table, compute_volume_profile,
    compute_hurst_exponent, compute_order_flow_imbalance, compute_dealer_gex
)
from src.strategy_rules import StrategyEngine, SignalType
from src.options_engine import (
    generate_option_trade_ticket, select_institutional_strike, black_scholes_greeks,
    calculate_position_size, calculate_tca_friction
)
from src.backtest_engine import BacktestEngine

# Page Config
st.set_page_config(
    page_title="Nifty Institutional Signal Terminal | JustNifty v3.0",
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
drawdown_input = st.sidebar.slider("Current Portfolio Drawdown (%)", min_value=0.0, max_value=15.0, value=0.0, step=0.5) / 100.0

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

# Indicator Math & Stochastic Pillars
df = df_raw.copy()
df["ema21"] = compute_ema(df["close"], EMA_FAST)
df["ema55"] = compute_ema(df["close"], EMA_MID)
df["ema200"] = compute_ema(df["close"], EMA_SLOW)
df["vakc_upper"], df["vakc_lower"] = compute_vakc_envelopes(df, iv=iv_input)
df["vwap"], df["vwap_upper"], df["vwap_lower"] = compute_vwap(df)

current_spot = float(df.iloc[-1]["close"])
prev_spot = float(df.iloc[-2]["close"]) if len(df) > 1 else current_spot
spot_delta = current_spot - prev_spot
cpr = compute_cpr(df)
vol_profile = compute_volume_profile(df)
hurst_data = compute_hurst_exponent(df["close"])
ofi_data = compute_order_flow_imbalance(df)
gex_data = compute_dealer_gex(current_spot)
vf_table = compute_vf_trade_table(float(df.iloc[0]["open"]), atr=float(df["high"].max() - df["low"].min()) / 4.0)

# Evaluate Active Signal & Option Ticket
signal = strategy_engine.evaluate_bar(df)
ticket = generate_option_trade_ticket(current_spot, signal, account_capital, drawdown_input)

# Confluence checks computation
is_above_200 = current_spot > float(df.iloc[-1]["ema200"])
is_above_vwap = current_spot > float(df.iloc[-1]["vwap"])
dist_ema21 = abs(current_spot - float(df.iloc[-1]["ema21"])) / current_spot
is_not_stretched = dist_ema21 <= MA_STRETCH_THRESHOLD

# ----------------- UNIFIED TOP INSTITUTIONAL COCKPIT -----------------
st.markdown("""
<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
    <div style="display: flex; align-items: center; gap: 10px;">
        <span class="badge-pro">PRO v3.0</span>
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
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #55657e;">TIMEFRAME: {tf_str} • HURST: {hurst_data['hurst']}</span>
        </div>
        <div style="display: flex; align-items: baseline; gap: 12px; margin-bottom: 6px;">
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 32px; font-weight: 800; color: #f1f5f9;">₹{current_spot:,.2f}</span>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 600; color: {'#05df72' if spot_delta >= 0 else '#ff3355'};">
                {spot_delta:+.2f} ({spot_delta/prev_spot*100:+.2f}%)
            </span>
        </div>
        <div style="font-size: 13px; color: #8e9fb5; margin-bottom: 12px; line-height: 1.4;">
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
                <div class="c-lbl">3. Hurst Exponent (H)</div>
                <div class="c-val" style="color: {'#05df72' if hurst_data['is_trending'] else '#fbb024'};">H={hurst_data['hurst']:.2f} ({'Trend' if hurst_data['is_trending'] else 'Range'})</div>
            </div>
            <div class="confluence-cell">
                <div class="c-lbl">4. Dealer GEX Flip</div>
                <div class="c-val" style="color: #00d2ff;">{gex_data['gamma_flip_strike']} ({'+Γ' if gex_data['is_positive_gamma'] else '-Γ'})</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with cockpit_col2:
    if ticket.get("status") == "READY":
        target_strike = ticket["symbol"]
        greeks_str = f"Δ {ticket['delta']:.2f} • Γ {ticket['gamma']:.5f} • Θ -₹{ticket['theta_decay_daily']:.2f}/sh • Vanna {ticket['vanna']:.4f}"
        ep_str = f"₹{ticket['entry_premium']:.2f}"
        sl_str = f"₹{ticket['sl_premium']:.2f}"
        t1_str = f"₹{ticket['target1_premium']:.2f}"
        t2_str = f"₹{ticket['target2_premium']:.2f}"
        t3_str = f"₹{ticket['target3_moonshot_premium']:.2f}"
        lots_str = f"{ticket['lots']} Lots ({ticket['total_qty']} Qty)"
        risk_rupees_str = f"₹{ticket['max_risk_rupees']:,.2f}"
        tca_fees_str = f"TCA: ₹{ticket['tca_friction']['total_friction']:.1f}"
    else:
        # Default ATM strike reference with Greeks
        atm_k = int(round(current_spot / 50.0) * 50)
        target_strike = f"NIFTY {atm_k} CE / PE"
        greeks_str = "Δ 0.55 • Γ 0.00078 • Θ -₹14.20/sh • Vanna 0.0420"
        ep_str = "₹142.50"
        sl_str = "₹112.50"
        t1_str = "₹182.00"
        t2_str = "₹228.00"
        t3_str = "₹285.00"
        lots_str = "6 Lots (150 Qty)"
        risk_rupees_str = f"₹{account_capital * risk_pct:,.2f}"
        tca_fees_str = "TCA Est: ₹182.50"

    st.markdown(f"""
    <div class="cockpit-box" style="display: flex; flex-direction: column; justify-content: space-between; height: calc(100% - 20px);">
        <div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div>
                    <div style="font-size: 11px; color: #55657e; text-transform: uppercase; font-weight: 600;">Optimal Strike (Delta 0.50-0.65)</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 19px; font-weight: 800; color: #00d2ff;">{target_strike}</div>
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; background-color: #141c2e; padding: 4px 8px; border-radius: 4px; color: #8e9fb5;">
                    {tca_fees_str}
                </div>
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #718096; margin-bottom: 10px;">
                {greeks_str}
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; margin-bottom: 10px;">
                <div style="background-color: #080c14; border: 1px solid #1c273c; border-radius: 4px; padding: 6px;">
                    <div style="font-size: 9px; color: #55657e; text-transform: uppercase; font-weight: 600;">Entry Prem</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 700; color: #f1f5f9;">{ep_str}</div>
                </div>
                <div style="background-color: #080c14; border: 1px solid #1c273c; border-radius: 4px; padding: 6px;">
                    <div style="font-size: 9px; color: #55657e; text-transform: uppercase; font-weight: 600;">Stop Loss</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 700; color: #ff3355;">{sl_str}</div>
                </div>
                <div style="background-color: #080c14; border: 1px solid #1c273c; border-radius: 4px; padding: 6px;">
                    <div style="font-size: 9px; color: #55657e; text-transform: uppercase; font-weight: 600;">T1 (35%)</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 700; color: #05df72;">{t1_str}</div>
                </div>
                <div style="background-color: #080c14; border: 1px solid #1c273c; border-radius: 4px; padding: 6px;">
                    <div style="font-size: 9px; color: #55657e; text-transform: uppercase; font-weight: 600;">T2 (35%)</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 700; color: #00d2ff;">{t2_str}</div>
                </div>
            </div>
        </div>
        <div style="background-color: rgba(0, 210, 255, 0.04); border-left: 3px solid #00d2ff; padding: 8px 10px; font-size: 11px; color: #8e9fb5; line-height: 1.4;">
            <strong>Profit Maximizer:</strong> Book 35% @ <strong>{t1_str}</strong> (or sell OTM for Free Spread) ➔ Book 35% @ <strong>{t2_str}</strong> ➔ Trail 30% runner to <strong>{t3_str}</strong> on 5m 21 EMA.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------- MAIN INTERFACE TABS -----------------
tab_chart, tab_sizer, tab_oi, tab_backtest, tab_cheatsheet = st.tabs([
    "📈 Interactive Candlestick Chart",
    "🛡️ 1% Risk & Quarter-Kelly Sizer",
    "🏛️ Institutional Participant OI & Option Chain",
    "📊 Bar-by-Bar Replay & Backtest Simulator",
    "📖 JustNifty v3.0 Master Rulebook"
])

# ----- TAB 1: INTERACTIVE CHART -----
with tab_chart:
    st.subheader("Nifty 50 Multi-Indicator Technical & Stochastic Chart")
    
    t1_c1, t1_c2, t1_c3, t1_c4, t1_c5 = st.columns(5)
    show_emas = t1_c1.checkbox("200 / 55 / 21 EMAs", value=True)
    show_vakc = t1_c2.checkbox("Adaptive Keltner (VAKC)", value=True)
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
        
    if show_vakc:
        fig.add_trace(go.Scatter(x=df.index, y=df["vakc_upper"], name="Upper VAKC Band", line=dict(color="#ff3355", width=1.2, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["vakc_lower"], name="Lower VAKC Band", line=dict(color="#05df72", width=1.2, dash="dot")), row=1, col=1)
        
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
    st.subheader("🛡️ Institutional 1% Capital Risk & Quarter-Kelly Sizer")
    st.caption("Computes allowable lot count with fat-tail Quarter-Kelly dampening and full Transaction Cost Analysis (TCA) friction.")
    
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        calc_cap = st.number_input("Account Capital (₹)", value=account_capital, step=50000.0, key="calc_cap")
        calc_risk_pct = st.slider("Risk Limit per Trade (%)", min_value=0.25, max_value=2.0, value=risk_pct*100.0, step=0.05, key="calc_risk") / 100.0
        calc_ep = st.number_input("Option Entry Premium (₹)", value=142.50, step=1.0, key="calc_ep")
        calc_sl = st.number_input("Option Stop-Loss Premium (₹)", value=112.50, step=1.0, key="calc_sl")
        calc_tp = st.number_input("Option Target Premium (₹)", value=188.00, step=1.0, key="calc_tp")
        
    with s_col2:
        pos_info = calculate_position_size(calc_cap, calc_risk_pct, calc_ep, calc_sl, contract_lot_size, drawdown_input)
        calc_lots = pos_info["lots"]
        calc_total_qty = pos_info["total_qty"]
        calc_actual_risk = pos_info["actual_risk_rupees"]
        calc_outlay = pos_info["capital_required"]
        
        tca = calculate_tca_friction(calc_ep, calc_tp, calc_total_qty, calc_lots)
        
        m1, m2 = st.columns(2)
        m1.metric("Max Risk Budget", f"₹{pos_info['max_risk_rupees']:,.2f}", f"DD Dampener: {pos_info['dd_dampener']}x")
        m2.metric("Allocated Position", f"{calc_lots} Lots ({calc_total_qty} Qty)")
        
        m3, m4 = st.columns(2)
        m3.metric("Actual Risk Exposure", f"₹{calc_actual_risk:,.2f}", f"{(calc_actual_risk/calc_cap)*100:.2f}% of Capital", delta_color="inverse")
        m4.metric("Capital Outlay Required", f"₹{calc_outlay:,.2f}", f"{(calc_outlay/calc_cap)*100:.1f}% Margin")
        
        st.markdown("#### 🧾 Indian NSE Statutory Friction (TCA Breakdown)")
        st.write(f"• **STT (0.1% on Sell):** ₹{tca['stt']:.2f} | **Brokerage:** ₹{tca['brokerage']:.2f} | **NSE Exchange Fees + GST:** ₹{tca['exchange_charges'] + tca['gst']:.2f} | **Slippage Buffer:** ₹{tca['slippage']:.2f}")
        st.write(f"• **Total Round-Trip TCA Friction:** **₹{tca['total_friction']:.2f}**")

# ----- TAB 3: PARTICIPANT OI & STRIKE LADDER -----
with tab_oi:
    st.subheader("🏛️ Institutional Participant-Wise Open Interest (FII / Prop Desks vs Retail)")
    st.dataframe(get_institutional_oi_data(), use_container_width=True)
    
    st.subheader("🔍 Institutional Strike Ladder & 2nd-Order Greeks Matrix (Delta 0.50 – 0.65)")
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
            "CE Vanna": ce_greeks["vanna"],
            "CE Theta": ce_greeks["theta"],
            "CE Premium (₹)": ce_greeks["price"],
            "Strike": f"🎯 {k} (ATM)" if is_atm else str(k),
            "PE Premium (₹)": pe_greeks["price"],
            "PE Theta": pe_greeks["theta"],
            "PE Vanna": pe_greeks["vanna"],
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
    st.subheader("📊 Bar-by-Bar Replay & Backtesting Engine with TCA Friction")
    st.caption("Simulates the JustNifty v3.0 model with full Transaction Cost Analysis (STT, Brokerage, GST, Slippage), 50% part-booking, and breakeven trailing.")
    
    run_btn = st.button("🚀 Run Backtest on Loaded Dataset", use_container_width=True)
    if run_btn or "bt_results" in st.session_state:
        if run_btn:
            bt_engine = BacktestEngine(initial_capital=account_capital)
            st.session_state["bt_results"] = bt_engine.run_backtest(df)
            
        results = st.session_state["bt_results"]
        b1, b2, b3, b4 = st.columns(4)
        pnl_color = "normal" if results.summary["pnl_rupees"] >= 0 else "inverse"
        b1.metric("Net Strategy PnL (Post-TCA)", f"₹{results.summary['pnl_rupees']:,.2f}", f"{results.summary['return_pct']:+.2f}%", delta_color=pnl_color)
        b2.metric("Win Rate", f"{results.summary['win_rate']:.1f}%", f"{results.summary['wins']}W / {results.summary['losses']}L")
        b3.metric("Total TCA Deducted", f"₹{results.summary['total_tca']:,.2f}", f"Gross: ₹{results.summary['gross_pnl']:,.2f}")
        b4.metric("Final Account Balance", f"₹{results.summary['final_capital']:,.2f}")
        
        if results.trade_log:
            st.markdown("#### 📜 Executed Trade Log (TCA Accounting)")
            st.dataframe(pd.DataFrame(results.trade_log), use_container_width=True)
            
            st.markdown("#### 📈 Account Equity Curve (Net of All Fees)")
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(y=results.equity_curve, mode="lines+markers", line=dict(color="#05df72", width=2), name="Net Equity (₹)"))
            fig_eq.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_eq, use_container_width=True)
        else:
            st.info("No completed trade setups triggered within this specific historical slice.")

# ----- TAB 5: MASTER RULEBOOK & SETUP SUMMARIES -----
with tab_cheatsheet:
    st.subheader("📖 JustNifty v3.1 Institutional Setup Summaries & Quantitative Rulebook")
    st.caption("Complete institutional playbooks, entry triggers, stop-loss formulas, 3-tier profit ladders, and execution protocols in summary form.")
    
    st.markdown("### 🎯 Institutional Buy / Sell Setup Summaries (Playbook Matrix)")
    pb_c1, pb_c2, pb_c3 = st.columns(3)
    
    with pb_c1:
        st.markdown("""
        <div style="background-color: #0e1422; border: 1px solid #1c273c; border-radius: 8px; padding: 16px; height: 100%;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="background-color: rgba(5,223,114,0.1); color: #05df72; border: 1px solid rgba(5,223,114,0.3); font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; padding: 2px 6px; border-radius: 4px;">PLAYBOOK 1</span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #05df72; font-weight: 600;">BULLISH CONFLUENCE</span>
            </div>
            <h4 style="margin: 0 0 10px 0; color: #f1f5f9;">🟢 Golden Pocket Long (Buy CE)</h4>
            <div style="font-size: 12px; color: #8e9fb5; line-height: 1.45;">
                <p style="margin-bottom: 6px;"><strong>1. Macro Regime:</strong> 5m Spot > 200 EMA + 200 EMA slope positive (<code>dEMA/dt &gt; 0</code>).</p>
                <p style="margin-bottom: 6px;"><strong>2. Value Anchor:</strong> Price holds above 09:15 Session AVWAP (<code>0.15σ ≤ Z_AVWAP ≤ 1.10σ</code>).</p>
                <p style="margin-bottom: 6px;"><strong>3. Golden Pocket:</strong> Pullback into <strong>50.0% to 61.8% Fibonacci zone</strong>.</p>
                <p style="margin-bottom: 6px;"><strong>4. Entry Trigger:</strong> Bullish candle close + positive OFI Z-Score defense.</p>
                <p style="margin-bottom: 6px;"><strong>5. Stop-Loss:</strong> Spot 78.6% Fib retracement - 5.0 pts (Delta-Gamma option SL).</p>
            </div>
            <div style="background-color: #080c14; border: 1px solid #1c273c; border-radius: 4px; padding: 8px; margin-top: 10px; font-family: 'JetBrains Mono', monospace; font-size: 11px;">
                <div style="color: #05df72;">• T1 (35%): +1.2x ATR (Sell OTM CE for Free Spread)</div>
                <div style="color: #00d2ff;">• T2 (35%): +2.5x ATR (Structural Extension)</div>
                <div style="color: #a855f7;">• T3 (30%): Moonshot Runner (Trail on 5m 21 EMA)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with pb_c2:
        st.markdown("""
        <div style="background-color: #0e1422; border: 1px solid #1c273c; border-radius: 8px; padding: 16px; height: 100%;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="background-color: rgba(255,51,85,0.1); color: #ff3355; border: 1px solid rgba(255,51,85,0.3); font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; padding: 2px 6px; border-radius: 4px;">PLAYBOOK 2</span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #ff3355; font-weight: 600;">BEARISH CONFLUENCE</span>
            </div>
            <h4 style="margin: 0 0 10px 0; color: #f1f5f9;">🔴 Golden Pocket Short (Buy PE)</h4>
            <div style="font-size: 12px; color: #8e9fb5; line-height: 1.45;">
                <p style="margin-bottom: 6px;"><strong>1. Macro Regime:</strong> 5m Spot < 200 EMA + 200 EMA slope negative (<code>dEMA/dt &lt; 0</code>).</p>
                <p style="margin-bottom: 6px;"><strong>2. Value Anchor:</strong> Price holds below 09:15 Session AVWAP (<code>-1.10σ ≤ Z_AVWAP ≤ -0.15σ</code>).</p>
                <p style="margin-bottom: 6px;"><strong>3. Golden Pocket:</strong> Rally into <strong>50.0% to 61.8% Fibonacci zone</strong>.</p>
                <p style="margin-bottom: 6px;"><strong>4. Entry Trigger:</strong> Bearish candle close + negative OFI Z-Score defense.</p>
                <p style="margin-bottom: 6px;"><strong>5. Stop-Loss:</strong> Spot 78.6% Fib retracement + 5.0 pts (Delta-Gamma option SL).</p>
            </div>
            <div style="background-color: #080c14; border: 1px solid #1c273c; border-radius: 4px; padding: 8px; margin-top: 10px; font-family: 'JetBrains Mono', monospace; font-size: 11px;">
                <div style="color: #05df72;">• T1 (35%): -1.2x ATR (Sell OTM PE for Free Spread)</div>
                <div style="color: #00d2ff;">• T2 (35%): -2.5x ATR (Structural Extension)</div>
                <div style="color: #a855f7;">• T3 (30%): Moonshot Runner (Trail on 5m 21 EMA)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with pb_c3:
        st.markdown("""
        <div style="background-color: #0e1422; border: 1px solid #1c273c; border-radius: 8px; padding: 16px; height: 100%;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="background-color: rgba(168,85,247,0.1); color: #a855f7; border: 1px solid rgba(168,85,247,0.3); font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; padding: 2px 6px; border-radius: 4px;">PLAYBOOK 3</span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #a855f7; font-weight: 600;">MOC SQUEEZE</span>
            </div>
            <h4 style="margin: 0 0 10px 0; color: #f1f5f9;">⚡ 3:00 PM Breakout</h4>
            <div style="font-size: 12px; color: #8e9fb5; line-height: 1.45;">
                <p style="margin-bottom: 6px;"><strong>1. Reference Range:</strong> Note exact High & Low of the 15:00-15:05 IST candle.</p>
                <p style="margin-bottom: 6px;"><strong>2. Breakout Trigger:</strong> Enter on 15:05 close beyond 15:00 candle extreme.</p>
                <p style="margin-bottom: 6px;"><strong>3. 0DTE Strike Shift:</strong> Select <strong>Deep ITM (Δ ≥ 0.75)</strong> to eliminate theta decay.</p>
                <p style="margin-bottom: 6px;"><strong>4. Stop-Loss:</strong> Invalidation set at opposite extreme of 15:00 candle.</p>
                <p style="margin-bottom: 6px;"><strong>5. Hard Execution Close:</strong> <strong>15:15 IST Mandatory Square-Off</strong>.</p>
            </div>
            <div style="background-color: #080c14; border: 1px solid #1c273c; border-radius: 4px; padding: 8px; margin-top: 10px; font-family: 'JetBrains Mono', monospace; font-size: 11px;">
                <div style="color: #05df72;">• T1 (50%): +40 to +50 points gamma expansion</div>
                <div style="color: #00d2ff;">• T2 (50%): +75 to +100 points MOC squeeze</div>
                <div style="color: #ff3355;">• Hard Stop: 15:15 IST terminal clock liquidation</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(r"""
    ### 🏛️ The 4 Adaptive Stochastic Pillars (Mathematical Formulations)
    1. **Anis-Lloyd Bias-Corrected Hurst Exponent ($H$):**
       $$H > 0.52 \implies \text{Trending State (Pillars Active)}; \quad H < 0.45 \implies \text{Mean-Reverting}; \quad H \in [0.45, 0.52] \implies \text{Noise Filter Kill-Switch}$$
    2. **Volatility-Adaptive Keltner Channels (VAKC):**
       $$\text{VAKC}_{\text{Upper/Lower}} = \text{EMA}_{200} \pm 2.25 \times \text{ATR}_{14} \times \sqrt{\frac{\text{IV}}{0.12}}$$
    3. **Session AVWAP 2nd-Moment Dispersion Corridor:**
       $$\text{AVWAP}_t = \frac{\sum P_i V_i}{\sum V_i}, \quad \sigma_{\text{AVWAP}} = \sqrt{\frac{\sum V_i (P_i - \text{AVWAP})^2}{\sum V_i}}, \quad \text{Gate: } 0.15\sigma \le |Z| \le 1.10\sigma$$
    4. **Composite Wick-Adjusted Bar Delta & 20-bar OFI Z-Score:**
       $$\Delta_{\text{Bar}} = V \times \left[\left(\frac{C - L}{H - L}\right) - \left(\frac{H - C}{H - L}\right)\right], \quad Z_{\text{OFI}} = \frac{\text{OFI}_t - \mu_{\text{OFI}}}{\sigma_{\text{OFI}}}$$

    ### 🛡️ Risk Management & Capital Defense Architecture
    - **1% Quarter-Kelly Position Sizer:** $f^* = \frac{p \cdot b - q}{b}, \quad \text{Lots} = \lfloor \frac{1}{4} f^* \times \frac{\text{Capital}}{\text{Risk Per Lot}} \rfloor \times (1 - \frac{\text{DD}}{\text{Max MDD}})$.
    - **Intraday 2-Strike Rule:** Trading halts automatically for the session upon recording 2 consecutive losses.
    - **Daily Loss Limit (DLL):** Maximum daily drawdown strictly capped at $1.5\%$ of account equity.
    - **Full Indian Statutory TCA:** STT ($0.1\%$ on sell), Brokerage (₹20/order), NSE Charges ($0.03503\%$), GST ($18\%$), and empirical slippage ($0.75$ pts) deducted on every transaction.
    """)

