"""
Script to apply v5.1 UI upgrades to app.py.
"""
import ast
import re

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update imports
old_imports = """from src.signal_journal import LiveSignalJournal
from src.options_flow import (
    compute_atm_straddle_metrics,
    compute_cumulative_oi_delta_and_traps,
    compute_pcr_momentum_derivative,
    compute_vanna_charm_drift_vector,
    compute_short_term_directional_vector
)
from src.volatility_engine import VolatilityIntelligence"""

new_imports = """from src.signal_journal import LiveSignalJournal, SignalPerformanceAnalyzer
from src.options_flow import (
    compute_atm_straddle_metrics,
    compute_cumulative_oi_delta_and_traps,
    compute_pcr_momentum_derivative,
    compute_vanna_charm_drift_vector,
    compute_short_term_directional_vector,
    compute_oi_change_heatmap,
    compute_strike_level_gex_chart_data,
    compute_oi_based_range_forecast
)
from src.volatility_engine import VolatilityIntelligence
from src.institutional_flow import InstitutionalFlowEngine
from src.portfolio_risk import PortfolioRiskManager"""

assert old_imports in content, "old_imports not found in app.py"
content = content.replace(old_imports, new_imports)

# 2. Add Pre-Open, Breadth Ticker & Signal Toast in Cockpit area
cockpit_marker = """# ----------------- UNIFIED TOP INSTITUTIONAL COCKPIT -----------------"""
cockpit_additions = """# ----------------- UNIFIED TOP INSTITUTIONAL COCKPIT -----------------
# 09:00 - 09:30 Pre-Open Gap Intelligence
now_ist = datetime.now(IST)
pre_open_data = data_engine.fetch_pre_open_gap()
if pre_open_data and (pre_open_data.get("pChange", 0.0) != 0.0 or (now_ist.hour == 9 and now_ist.minute < 30)):
    po_gap = pre_open_data.get("pChange", 0.0)
    po_iep = pre_open_data.get("iep", current_spot)
    po_adv = pre_open_data.get("advances", 0)
    po_dec = pre_open_data.get("declines", 0)
    po_border = "#05df72" if po_gap >= 0 else "#ff3355"
    st.markdown(f'''
    <div style="background-color: rgba(14, 20, 34, 0.7); border-left: 3px solid {po_border}; border-radius: 4px; padding: 6px 12px; margin-bottom: 8px; font-size: 11px; color: #94a3b8; display: flex; justify-content: space-between; align-items: center;">
        <div>
            <strong style="color: #f1f5f9;">🌅 09:08 AM Pre-Open Discovery:</strong> IEP: <strong style="color: #f1f5f9;">₹{po_iep:,.2f}</strong> ({po_gap:+.2f}%) • Breadth: <span style="color:#05df72;">{po_adv} Adv</span> / <span style="color:#ff3355;">{po_dec} Dec</span>
        </div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #00d2ff;">
            STRATEGY: {'GAP-AND-GO (MOMENTUM CONTINUATION)' if abs(po_gap) >= 0.6 else 'MEAN-REVERSION GAP FILL' if abs(po_gap) >= 0.25 else 'BALANCED OPEN (RANGE TRADING)'}
        </div>
    </div>
    ''', unsafe_allow_html=True)

# Market Breadth & Daily Range Ticker
hfi_adv = hfi_res.get("advances", 0)
hfi_dec = hfi_res.get("declines", 0)
day_range_pts = float(df['high'].max() - df['low'].min())
vol_ratio = float(df['volume'].iloc[-1] / max(float(df['volume'].mean()), 1.0)) * 100.0
st.markdown(f'''
<div style="background-color: #0b101b; border: 1px solid #162032; border-radius: 6px; padding: 4px 12px; margin-bottom: 8px; font-size: 11px; color: #8e9fb5; display: flex; justify-content: space-between; align-items: center; font-family: 'JetBrains Mono', monospace;">
    <div><strong>🏛️ BREADTH:</strong> <span style="color:#05df72;">{hfi_adv}↑</span> / <span style="color:#ff3355;">{hfi_dec}↓</span> (Top 5: 41.2% Wt) | <strong>DAY RANGE:</strong> {day_range_pts:.1f} pts ({day_range_pts/current_spot*100:.2f}%)</div>
    <div><strong>VOL SURGE:</strong> <span style="color:{'#05df72' if vol_ratio >= 120 else '#94a3b8'};">{vol_ratio:.0f}% of Avg</span> | <strong>REGIME MEMORY (DFA α):</strong> {dfa_res['dfa_alpha']:.3f}</div>
</div>
''', unsafe_allow_html=True)

# Real-Time Signal Alert Toast
if signal.signal_type != SignalType.WAIT:
    last_toast_sig = st.session_state.get("last_toast_signal_id", "")
    current_sig_key = f"{signal.signal_type.value}_{last_bar_ts}"
    if last_toast_sig != current_sig_key:
        st.session_state["last_toast_signal_id"] = current_sig_key
        st.toast(f"🎯 {signal.signal_type.value} @ ₹{current_spot:,.2f} | Strike: {ticket['target_strike']} {ticket['option_type']}", icon="🚨")
"""

assert cockpit_marker in content, "cockpit_marker not found"
content = content.replace(cockpit_marker, cockpit_additions)

# 3. Add SignalPerformanceAnalyzer to Tab 2
tab2_export_marker = """    # Export Toolbar
    st.markdown("---")"""

tab2_analytics_code = """    # Institutional Signal Performance & Execution Analytics Suite (v5.1)
    with st.expander("📊 Institutional Signal Performance & Execution Analytics (Historical & Real-Time)", expanded=False):
        perf_analyzer = SignalPerformanceAnalyzer(journal_engine.entries)
        perf_rep = perf_analyzer.generate_performance_report()
        perf_summary = perf_rep["summary"]
        
        # Summary Row
        p_c1, p_c2, p_c3, p_c4, p_c5 = st.columns(5)
        p_c1.metric("Closed Trades", f"{perf_summary['total_closed_trades']}", f"{perf_summary['winning_trades']}W / {perf_summary['losing_trades']}L")
        p_c2.metric("Win Rate", f"{perf_summary['win_rate_pct']:.1f}%", "Target: > 65%")
        p_c3.metric("Profit Factor", f"{perf_summary['profit_factor']:.2f}", f"SQN: {perf_summary['system_quality_number_sqn']:.2f}")
        p_c4.metric("Avg R-Multiple", f"{perf_summary['avg_r_multiple']:+.2f}R", f"Total: {perf_summary['total_r_multiple']:+.1f}R")
        p_c5.metric("Net Realized PnL", f"₹{perf_summary['total_realized_pnl']:+,.2f}")
        
        st.markdown("---")
        
        pa_col1, pa_col2, pa_col3 = st.columns(3)
        with pa_col1:
            st.markdown("**🎯 Win Rate by Signal Type**")
            df_by_sig = perf_analyzer.win_rate_by_signal_type()
            if not df_by_sig.empty:
                st.dataframe(df_by_sig[["signal_type", "total_trades", "win_rate_pct", "avg_r_multiple", "profit_factor"]], hide_index=True, width="stretch")
            else:
                st.caption("No closed trades recorded yet.")
                
        with pa_col2:
            st.markdown("**⏰ Win Rate by Intraday Time Bucket**")
            df_by_time = perf_analyzer.win_rate_by_time_bucket()
            if not df_by_time.empty:
                st.dataframe(df_by_time[["time_bucket", "total_trades", "win_rate_pct", "avg_r_multiple", "profit_factor"]], hide_index=True, width="stretch")
            else:
                st.caption("No closed trades recorded yet.")
                
        with pa_col3:
            st.markdown("**🏛️ Win Rate by Market Regime**")
            df_by_reg = perf_analyzer.win_rate_by_regime()
            if not df_by_reg.empty:
                st.dataframe(df_by_reg[["regime", "total_trades", "win_rate_pct", "avg_r_multiple", "profit_factor"]], hide_index=True, width="stretch")
            else:
                st.caption("No closed trades recorded yet.")
                
        st.markdown("---")
        
        # Confluence Correlation & Tilt Diagnostics
        corr_col, tilt_col = st.columns([1.2, 1.0])
        with corr_col:
            st.markdown("**📈 Confluence Score vs Outcome Correlation (Pearson $r$)**")
            conf_corr = perf_rep["confluence_correlation"]
            st.caption(f"Pearson Correlation: **{conf_corr['pearson_r']:+.3f}** (Strength: `{conf_corr['correlation_strength']}`) | p-value: `{conf_corr['p_value']:.4f}`")
            df_buckets = pd.DataFrame(conf_corr["buckets"])
            if not df_buckets.empty:
                st.dataframe(df_buckets, hide_index=True, width="stretch")
                
        with tilt_col:
            st.markdown("**🧠 Behavioral Tilt & Streak Diagnostics**")
            stk_tilt = perf_rep["streak_and_tilt"]
            tilt_bg = "rgba(255, 51, 85, 0.12)" if stk_tilt["tilt_detected"] else "rgba(5, 223, 114, 0.08)"
            tilt_border = "#ff3355" if stk_tilt["tilt_detected"] else "#05df72"
            st.markdown(f'''
            <div style="background-color: {tilt_bg}; border: 1px solid {tilt_border}; border-radius: 6px; padding: 10px; font-size: 11px; color: #f1f5f9;">
                <div><strong>Current Streak:</strong> {stk_tilt['current_streak_count']} {stk_tilt['current_streak_type']}s (Max Win: {stk_tilt['max_win_streak']}, Max Loss: {stk_tilt['max_loss_streak']})</div>
                <div style="margin-top: 4px;"><strong>Tilt Risk Level:</strong> <span style="font-weight:700; color:{tilt_border};">{stk_tilt['tilt_warning_level']}</span></div>
                <div style="margin-top: 4px;"><strong>Trade Interval:</strong> {stk_tilt['avg_trade_interval_minutes']:.1f}m (Loss Streak: {stk_tilt['loss_streak_interval_minutes']:.1f}m, Accel: {stk_tilt['frequency_acceleration_ratio']:.2f}x)</div>
                <div style="margin-top: 4px;"><strong>Recommended Action:</strong> <code>{stk_tilt['recommended_action']}</code></div>
            </div>
            ''', unsafe_allow_html=True)

    # Export Toolbar
    st.markdown("---")"""

assert tab2_export_marker in content, "tab2_export_marker not found"
content = content.replace(tab2_export_marker, tab2_analytics_code)

# 4. Add Portfolio Greeks, Scenario P&L, Volatility Cone in Tab 3
tab3_end_marker = """# ----- TAB 3: PARTICIPANT OI & LIVE OPTION CHAIN -----"""
tab3_additions = """    # ----------------- PORTFOLIO GREEKS & SCENARIO ANALYSIS SECTION (v5.1) -----------------
    st.markdown("---")
    st.markdown("### 📊 Real-Time Portfolio Greeks & What-If Scenario Matrix")
    st.caption("Aggregates 1st, 2nd, and 3rd order Greeks across all active positions and simulates full non-linear revaluation PnL across spot shifts, time decay, and IV shocks.")
    
    port_risk_mgr = PortfolioRiskManager(lot_size=int(contract_lot_size))
    # Collect active trade tickets
    active_sigs = [e.to_dict() for e in journal_engine.entries if e.is_active()]
    if not active_sigs:
        active_sigs = [{
            "selected_strike": ticket["target_strike"],
            "option_type": ticket["option_type"],
            "lots_suggested": calc_lots,
            "direction": "LONG" if "CE" in ticket["option_type"] else "SHORT",
            "entry_premium": calc_ep,
            "sl_premium": calc_sl
        }]
        
    port_greeks = port_risk_mgr.compute_portfolio_greeks(active_sigs, spot=current_spot, iv=iv_input, t_days=3.5)
    
    # 4 Portfolio Greeks KPI Cards
    pg1, pg2, pg3, pg4 = st.columns(4)
    pg1.metric("Net Delta (Δ)", f"{port_greeks['net_delta']:+.2f}", f"Notional: ₹{port_greeks['net_notional_delta_rupees']:+,.0f}")
    pg2.metric("Net Gamma (Γ)", f"{port_greeks['net_gamma']:.6f}", f"{port_greeks['directional_bias']}")
    pg3.metric("Net Daily Theta (Θ)", f"₹{port_greeks['net_theta_daily_rupees']:+,.1f}/day", "Time Decay Flow")
    pg4.metric("Net Vega (ν)", f"₹{port_greeks['net_vega_rupees']:+,.1f}/1%", f"Vanna: {port_greeks['net_vanna']:+.4f}")
    
    # Scenario Grid
    scenario_res = port_risk_mgr.compute_scenario_pnl_grid(active_sigs, spot=current_spot, iv=iv_input, t_days=3.5)
    df_scen = scenario_res["scenario_dataframe"]
    
    scen_c1, scen_c2 = st.columns([1.2, 1.0])
    with scen_c1:
        st.markdown("**What-If Spot Revaluation Curve (T+0 vs T+1d vs Expiry):**")
        fig_scen = go.Figure()
        fig_scen.add_trace(go.Scatter(x=df_scen["spot"], y=df_scen["pnl_t0"], mode="lines", name="T+0 (Immediate)", line=dict(color="#00d2ff", width=2.5)))
        fig_scen.add_trace(go.Scatter(x=df_scen["spot"], y=df_scen["pnl_t1d"], mode="lines", name="T+1 Day Decay", line=dict(color="#fbb024", width=1.5, dash="dash")))
        fig_scen.add_trace(go.Scatter(x=df_scen["spot"], y=df_scen["pnl_expiry"], mode="lines", name="Expiry Payoff", line=dict(color="#05df72", width=2)))
        fig_scen.add_hline(y=0.0, line_color="#475569", line_dash="dot", line_width=1)
        fig_scen.add_vline(x=current_spot, line_color="#e2e8f0", line_dash="dash", annotation_text=f"Spot ₹{current_spot:.0f}", annotation_position="top")
        fig_scen.update_layout(
            paper_bgcolor="#080c14",
            plot_bgcolor="#0e1422",
            height=280,
            margin=dict(l=10, r=10, t=25, b=10),
            xaxis=dict(title="Nifty Spot Price", gridcolor="#162032"),
            yaxis=dict(title="Net PnL (₹)", gridcolor="#162032"),
            legend=dict(orientation="h", y=1.1, x=0.0, font=dict(size=10, color="#94a3b8"))
        )
        st.plotly_chart(fig_scen, width="stretch")
        
    with scen_c2:
        st.markdown("**Scenario Stress Matrix (₹ PnL at Key Spot Shifts):**")
        st.dataframe(df_scen[["spot_shift", "spot", "pnl_t0", "pnl_t1d", "pnl_expiry", "pnl_iv_plus3", "pnl_iv_minus3"]], hide_index=True, width="stretch", height=280)
        
    # Volatility Cone & RV Term Structure Expander
    with st.expander("📉 Multi-Horizon Volatility Cone & Realized Volatility Term Structure", expanded=False):
        vol_cone_res = vol_engine.compute_volatility_cone(df["close"])
        rv_term_res = vol_engine.compute_rv_term_structure(df["close"])
        
        vc1, vc2 = st.columns([1.2, 1.0])
        with vc1:
            st.markdown("**Realized Volatility Percentile Distribution (Quantile Cone):**")
            st.dataframe(vol_cone_res["cone_dataframe"], hide_index=True, width="stretch")
        with vc2:
            st.markdown(f"**Term Structure Regime: `{rv_term_res['classification']}`**")
            st.write(f"• **5-Period RV:** {rv_term_res['rv_5']*100:.1f}% | **20-Period RV:** {rv_term_res['rv_20']*100:.1f}% | **60-Period RV:** {rv_term_res['rv_60']*100:.1f}%")
            st.write(f"• **Compression Ratio (RV5 / RV20):** **{rv_term_res['compression_ratio']:.2f}** (Slope: {rv_term_res['term_structure_slope']*100:+.2f}%)")
            if rv_term_res["compression_breakout_signal"]:
                st.warning(f"⚡ {rv_term_res['breakout_commentary']}")
            else:
                st.info(f"ℹ️ {rv_term_res['breakout_commentary']}")

# ----- TAB 3: PARTICIPANT OI & LIVE OPTION CHAIN -----"""

assert tab3_end_marker in content, "tab3_end_marker not found"
content = content.replace(tab3_end_marker, tab3_additions)

# 5. Add FII Flow Intelligence & OI Heatmap & GEX Chart to Tab 4
fii_marker = """    st.markdown("#### 🏛️ Participant-Wise Open Interest (FII / Prop Desks vs Retail)")"""
fii_code = """    # FII / DII Institutional Flow Intelligence Engine (v5.1)
    flow_engine = InstitutionalFlowEngine()
    inst_report = flow_engine.generate_institutional_flow_report(data_engine)
    fii_snap = inst_report["current_snapshot"]
    fii_trend = inst_report["flow_trend"]
    fii_roll = inst_report["rollover_analysis"]
    
    st.markdown("#### 🌊 Institutional Derivatives Flow Intelligence (FII vs DII)")
    fii_c1, fii_c2, fii_c3, fii_c4 = st.columns(4)
    fii_c1.metric("FII Futures Long / Short Ratio", f"{fii_snap['fii_ls_ratio']:.2f}", f"5D Change: {fii_trend['ls_ratio_change_today']:+.2f}")
    fii_c2.metric("FII 5-Day Flow Trend", f"{fii_trend['trend']}", f"{fii_trend['consecutive_days']} Consecutive Days")
    fii_c3.metric("FII Options PCR", f"{fii_snap['fii_options_pcr']:.2f}", f"DII: {fii_snap['dii_net_bias'].replace('_',' ')}")
    fii_c4.metric("Institutional Consensus", f"{inst_report['macro_bias_score']:+.2f}", f"{inst_report['institutional_consensus_bias'].replace('_',' ')}")
    
    st.markdown(f'''
    <div style="background-color: #0d1527; border: 1px solid #1c2e4a; border-radius: 6px; padding: 8px 12px; margin-bottom: 12px; font-size: 11px; color: #94a3b8;">
        <strong style="color: #00d2ff;">Macro Flow Summary:</strong> {inst_report['flow_summary']}
        {' • <strong>Rollover:</strong> ' + fii_roll['rollover_signal'].replace('_',' ') if fii_roll['is_expiry_week'] else ''}
    </div>
    ''', unsafe_allow_html=True)

    st.markdown("#### 🏛️ Participant-Wise Open Interest (FII / Prop Desks vs Retail)")"""

assert fii_marker in content, "fii_marker not found"
content = content.replace(fii_marker, fii_code)

# 6. Add Live OI Change Heatmap and Strike-Level GEX Bar Chart inside Official Option Chain mode
pcr_marker = """        pcr_c4.metric("Exact Max Pain Strike", f"₹{pcr_analytics['max_pain_strike']:,.0f}", f"Underlying: ₹{underlying_val:,.1f}", delta_color="normal")"""
pcr_additions = """        pcr_c4.metric("Exact Max Pain Strike", f"₹{pcr_analytics['max_pain_strike']:,.0f}", f"Underlying: ₹{underlying_val:,.1f}", delta_color="normal")
        
        # ----------------- LIVE OI CHANGE HEATMAP & STRIKE GEX CHART (v5.1) -----------------
        st.markdown("---")
        st.markdown("#### 🔥 Strike-Level Institutional OI Change Heatmap & Dealer Gamma Exposure (GEX)")
        
        oi_hm_res = compute_oi_change_heatmap(oc_filtered, spot=current_spot, range_pts=500.0)
        gex_chart_res = compute_strike_level_gex_chart_data(oc_filtered, spot=current_spot, iv=iv_input, t_days=3.5)
        range_fc_res = compute_oi_based_range_forecast(oc_filtered, spot=current_spot, max_pain=pcr_analytics['max_pain_strike'])
        
        # Expected Range Corridor Banner
        st.markdown(f'''
        <div style="background-color: #0c1424; border: 1px solid #1f2e4d; border-radius: 6px; padding: 8px 14px; margin-bottom: 12px; font-size: 12px; color: #8e9fb5; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <strong>🎯 OI Expected Range:</strong> <span style="color:#05df72; font-weight:700;">Put Wall ₹{range_fc_res['put_wall']:,.0f} (Support)</span> ⟷ <span style="color:#ff3355; font-weight:700;">Call Wall ₹{range_fc_res['call_wall']:,.0f} (Resistance)</span> | Width: <strong>{range_fc_res['range_width_pts']} pts</strong>
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px;">
                Spot Position: <strong style="color:#00d2ff;">{range_fc_res['spot_position_pct']:.1f}%</strong> ({range_fc_res['location_bias'].replace('_',' ')})
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        hm_col1, hm_col2 = st.columns([1.1, 1.0])
        
        with hm_col1:
            st.markdown(f"**Live Strike OI Change Heatmap (Bias: `{oi_hm_res['writing_bias']}`):**")
            df_hm = pd.DataFrame(oi_hm_res["heatmap_rows"])
            if not df_hm.empty:
                # Plotly Heatmap Matrix
                z_matrix = [
                    df_hm["ce_change_oi"].values,
                    df_hm["pe_change_oi"].values,
                    df_hm["net_strike_oi_delta"].values
                ]
                fig_hm = go.Figure(data=go.Heatmap(
                    z=z_matrix,
                    x=[f"₹{s:.0f}" for s in df_hm["strike"]],
                    y=["CE ΔOI (Call Writing)", "PE ΔOI (Put Writing)", "Net ΔOI (PE - CE)"],
                    colorscale=[[0.0, "#ff3355"], [0.5, "#1c273c"], [1.0, "#05df72"]],
                    zmid=0.0,
                    showscale=True,
                    colorbar=dict(title=dict(text="Contracts", font=dict(size=9, color="#94a3b8")), thickness=8)
                ))
                fig_hm.update_layout(
                    paper_bgcolor="#080c14",
                    plot_bgcolor="#0e1422",
                    height=240,
                    margin=dict(l=10, r=10, t=20, b=10),
                    xaxis=dict(gridcolor="#162032", tickangle=-45, tickfont=dict(size=8, color="#94a3b8")),
                    yaxis=dict(tickfont=dict(size=9, color="#94a3b8"))
                )
                st.plotly_chart(fig_hm, width="stretch")
            else:
                st.caption("No heatmap strikes available in range.")
                
        with hm_col2:
            st.markdown(f"**Dealer Gamma Exposure (GEX) by Strike (Regime: `{gex_chart_res['net_dealer_regime']}`):**")
            if gex_chart_res["strikes"]:
                fig_gex = go.Figure()
                fig_gex.add_trace(go.Bar(
                    x=gex_chart_res["strikes"],
                    y=gex_chart_res["net_gex_per_strike"],
                    name="Net GEX (₹ Cr)",
                    marker_color=["#05df72" if g >= 0 else "#ff3355" for g in gex_chart_res["net_gex_per_strike"]]
                ))
                fig_gex.add_vline(x=current_spot, line_color="#e2e8f0", line_dash="dash", line_width=1.5, annotation_text=f"Spot ₹{current_spot:.0f}", annotation_position="top")
                if gex_chart_res["zero_gex_strike"] > 0:
                    fig_gex.add_vline(x=gex_chart_res["zero_gex_strike"], line_color="#00d2ff", line_dash="dot", line_width=1, annotation_text=f"Zero-Γ", annotation_position="bottom")
                fig_gex.update_layout(
                    paper_bgcolor="#080c14",
                    plot_bgcolor="#0e1422",
                    height=240,
                    margin=dict(l=10, r=10, t=20, b=10),
                    xaxis=dict(title="Strike Price", gridcolor="#162032", tickfont=dict(size=9, color="#94a3b8")),
                    yaxis=dict(title="Net GEX (₹ Cr)", gridcolor="#162032", tickfont=dict(size=9, color="#94a3b8"))
                )
                st.plotly_chart(fig_gex, width="stretch")
            else:
                st.caption("No GEX strikes available.")"""

assert pcr_marker in content, "pcr_marker not found"
content = content.replace(pcr_marker, pcr_additions)

# Validate syntax
ast.parse(content)
print("AST validation passed! Writing back to app.py...")

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)

print("app.py updated successfully!")
