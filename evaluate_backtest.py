import pandas as pd
from src.data_engine import DataEngine
from src.backtest_engine import BacktestEngine
from src.indicators import compute_ema, compute_vakc_envelopes, compute_vwap
from src.config import EMA_FAST, EMA_MID, EMA_SLOW

engine = DataEngine(use_cache=True)
df_live = engine.fetch_yfinance_nifty(interval="5m", period="5d")

if df_live.empty or len(df_live) < 25:
    print("Generating synthetic dataset...")
    df_live = engine.generate_synthetic_nifty(bars=250, interval_mins=5)

df = df_live.copy()
df["ema21"] = compute_ema(df["close"], EMA_FAST)
df["ema55"] = compute_ema(df["close"], EMA_MID)
df["ema200"] = compute_ema(df["close"], EMA_SLOW)
df["vakc_upper"], df["vakc_lower"] = compute_vakc_envelopes(df)
df["vwap"], df["vwap_upper"], df["vwap_lower"] = compute_vwap(df)

bt = BacktestEngine(initial_capital=500000.0)
results = bt.run_backtest(df)

print("="*60)
print("             JUSTNIFTY v3.0 INSTITUTIONAL BACKTEST REPORT")
print("="*60)
print(f"Total Trades: {results.summary['total_trades']}")
print(f"Wins: {results.summary['wins']} | Losses: {results.summary['losses']}")
print(f"Win Rate: {results.summary['win_rate']}%")
print(f"Gross PnL: Rs {results.summary['gross_pnl']:,.2f}")
print(f"Total TCA Friction: Rs {results.summary['total_tca']:,.2f} (STT, Brokerage, GST, Slippage)")
print(f"Net PnL (Post-TCA): Rs {results.summary['pnl_rupees']:,.2f} ({results.summary['return_pct']:+.2f}%)")
print(f"Final Capital: Rs {results.summary['final_capital']:,.2f}")
print("="*60)

for t in results.trade_log:
    print(f"{t['entry_time']} -> {t['exit_time']} | {t['symbol']} | {t['signal']} | Net: Rs {t['net_pnl']:,.2f} (Gross: Rs {t['gross_pnl']:,.2f}, TCA: Rs {t['tca_fees']:.2f}) | {t['result']} | Part-Booked: {t['part_booked']}")
