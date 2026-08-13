import pandas as pd
from src.data_engine import DataEngine
from src.backtest_engine import BacktestEngine
from src.indicators import compute_ema, compute_envelopes, compute_vwap
from src.config import EMA_FAST, EMA_MID, EMA_SLOW, ENVELOPE_PCT

engine = DataEngine(use_cache=True)
df_live = engine.fetch_yfinance_nifty(interval="5m", period="5d")

if df_live.empty or len(df_live) < 25:
    print("Generating synthetic dataset...")
    df_live = engine.generate_synthetic_nifty(bars=250, interval_mins=5)

df = df_live.copy()
df["ema21"] = compute_ema(df["close"], EMA_FAST)
df["ema55"] = compute_ema(df["close"], EMA_MID)
df["ema200"] = compute_ema(df["close"], EMA_SLOW)
df["env_upper"], df["env_lower"] = compute_envelopes(df["ema200"], ENVELOPE_PCT)
df["vwap"], df["vwap_upper"], df["vwap_lower"] = compute_vwap(df)

bt = BacktestEngine(initial_capital=500000.0)
results = bt.run_backtest(df)

print("="*50)
print(f"Total Trades: {results.summary['total_trades']}")
print(f"Wins: {results.summary['wins']} | Losses: {results.summary['losses']}")
print(f"Win Rate: {results.summary['win_rate']}%")
print(f"Net PnL: Rs {results.summary['pnl_rupees']:,.2f} ({results.summary['return_pct']:+.2f}%)")
print(f"Final Capital: Rs {results.summary['final_capital']:,.2f}")
print("="*50)

for t in results.trade_log:
    print(f"{t['entry_time']} -> {t['exit_time']} | {t['symbol']} | {t['signal']} | PnL: Rs {t['pnl']:,.2f} ({t['result']}) | Part-Booked: {t['part_booked']}")
