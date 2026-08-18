import sys
import os
sys.path.insert(0, os.path.abspath("."))
import pandas as pd
from src.data_engine import DataEngine
from src.indicators import compute_vwap, compute_initial_balance_and_day_type, compute_session_cvd, compute_cpr

def analyze_session():
    engine = DataEngine()
    df = engine.fetch_yfinance_nifty(interval='5m', period='5d')

    if df is None or df.empty:
        print("No data available")
        return

    today_str = str(df.index[-1].date())
    df_today = df[df.index.astype(str).str.startswith(today_str)]
    
    open_p = float(df_today.iloc[0]["open"])
    high_p = float(df_today["high"].max())
    low_p = float(df_today["low"].min())
    curr_p = float(df_today.iloc[-1]["close"])
    total_range = high_p - low_p
    
    ib_df = df_today.iloc[:12] if len(df_today) >= 12 else df_today
    ib_high = float(ib_df["high"].max())
    ib_low = float(ib_df["low"].min())
    
    vwap_series, upper_2sd, lower_2sd = compute_vwap(df_today)
    curr_vwap = float(vwap_series.iloc[-1])
    
    print("=" * 60)
    print(f"SESSION REPLAY & DECONSTRUCTION: {today_str}")
    print("=" * 60)
    print(f"Open:        Rs. {open_p:,.2f}")
    print(f"High:        Rs. {high_p:,.2f} ({open_p - high_p:+.2f} pts from open)")
    print(f"Low:         Rs. {low_p:,.2f}  ({low_p - open_p:+.2f} pts from open)")
    print(f"Current:     Rs. {curr_p:,.2f} ({curr_p - open_p:+.2f} pts from open)")
    print(f"Day Range:   {total_range:.1f} pts ({total_range/open_p*100:.2f}%)")
    print(f"IB High/Low: Rs. {ib_high:,.2f} / Rs. {ib_low:,.2f} (IB Range: {ib_high - ib_low:.1f} pts)")
    print(f"Current VWAP:Rs. {curr_vwap:,.2f}")
    
    print("\n--- 30-MINUTE CANDLE REPLAY ---")
    for i in range(0, len(df_today), 6):
        sub = df_today.iloc[i:i+6]
        ts_start = str(sub.index[0])[11:16]
        ts_end = str(sub.index[-1])[11:16]
        c_open = sub.iloc[0]["open"]
        c_high = sub["high"].max()
        c_low = sub["low"].min()
        c_close = sub.iloc[-1]["close"]
        c_vol = sub["volume"].sum()
        delta_pts = c_close - c_open
        direction = "[BULL]" if delta_pts > 0 else "[BEAR]"
        print(f"{ts_start} - {ts_end} IST | {direction} ({delta_pts:+6.1f} pts) | O: {c_open:7.1f} H: {c_high:7.1f} L: {c_low:7.1f} C: {c_close:7.1f} | Vol: {c_vol:8,.0f}")

if __name__ == "__main__":
    analyze_session()
