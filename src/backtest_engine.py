"""Bar-by-bar historical replay and backtesting engine for JustNifty v2.0."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd
from src.config import DEFAULT_CAPITAL, LOT_SIZE
from src.strategy_rules import StrategyEngine, SignalType
from src.options_engine import generate_option_trade_ticket

@dataclass
class BacktestResults:
    summary: Dict[str, Any]
    trade_log: List[Dict[str, Any]]
    equity_curve: List[float]

class BacktestEngine:
    def __init__(self, initial_capital: float = DEFAULT_CAPITAL):
        self.initial_capital = initial_capital
        self.strategy = StrategyEngine()

    def run_backtest(self, df_5m: pd.DataFrame) -> BacktestResults:
        """Executes a chronological bar-by-bar simulation with 50% part-booking and breakeven trailing."""
        if df_5m.empty or len(df_5m) < 25:
            return BacktestResults(
                summary={
                    "initial_capital": self.initial_capital,
                    "final_capital": self.initial_capital,
                    "pnl_rupees": 0.0,
                    "return_pct": 0.0,
                    "total_trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "win_rate": 0.0
                },
                trade_log=[],
                equity_curve=[self.initial_capital]
            )

        capital = self.initial_capital
        equity_curve = [capital]
        trade_log = []
        
        in_trade = False
        active_ticket: Optional[Dict[str, Any]] = None
        part_booked = False
        entry_time = None
        entry_idx = 0
        
        for i in range(25, len(df_5m)):
            bar = df_5m.iloc[i]
            close = float(bar["close"])
            high = float(bar["high"])
            low = float(bar["low"])
            bar_time_str = bar.name.strftime("%H:%M") if hasattr(bar.name, "strftime") else "12:00"
            
            if not in_trade:
                signal = self.strategy.evaluate_bar(df_5m, current_idx=i)
                if signal.signal_type in [SignalType.LONG, SignalType.SHORT, SignalType.LONG_3PM, SignalType.SHORT_3PM]:
                    ticket = generate_option_trade_ticket(close, signal, capital)
                    if ticket.get("status") == "READY" and ticket.get("lots", 0) > 0:
                        in_trade = True
                        active_ticket = ticket
                        part_booked = False
                        entry_idx = i
                        entry_time = bar.name
            else:
                delta = abs(active_ticket["delta"])
                entry_prem = float(active_ticket["entry_premium"])
                sl_prem = float(active_ticket["sl_premium"])
                t1_prem = float(active_ticket["target1_premium"])
                lots = int(active_ticket["lots"])
                is_call = "CE" in active_ticket["option_type"]
                
                # Approximate current high/low option price from spot
                spot_entry = df_5m.iloc[entry_idx]["close"]
                if is_call:
                    opt_high = entry_prem + ((high - spot_entry) * delta)
                    opt_low = entry_prem - ((spot_entry - low) * delta)
                else:
                    opt_high = entry_prem + ((spot_entry - low) * delta)
                    opt_low = entry_prem - ((high - spot_entry) * delta)

                # 1. Check 50% Part-Booking hit
                if not part_booked and opt_high >= t1_prem:
                    part_booked = True
                    booked_lots = max(lots // 2, 1)
                    pnl_50 = (t1_prem - entry_prem) * booked_lots * LOT_SIZE
                    capital += pnl_50
                    # Shift SL to Break-Even (entry premium)
                    active_ticket["sl_premium"] = entry_prem
                    
                # 2. Check Stop-Loss hit or EOD Square-off (15:20)
                is_eod = (i == len(df_5m) - 1) or (bar_time_str >= "15:20")
                sl_hit = opt_low <= float(active_ticket["sl_premium"])
                
                if sl_hit or is_eod:
                    exit_prem = float(active_ticket["sl_premium"]) if sl_hit else entry_prem + ((close - spot_entry) * delta if is_call else (spot_entry - close) * delta)
                    booked_lots = max(lots // 2, 1) if part_booked else 0
                    remaining_lots = max(lots - booked_lots, 0)
                    
                    pnl_rem = (exit_prem - entry_prem) * remaining_lots * LOT_SIZE
                    capital += pnl_rem
                    
                    pnl_first_half = (t1_prem - entry_prem) * booked_lots * LOT_SIZE if part_booked else 0.0
                    trade_total_pnl = pnl_first_half + pnl_rem
                    
                    trade_log.append({
                        "entry_time": entry_time.strftime("%Y-%m-%d %H:%M") if hasattr(entry_time, "strftime") else str(entry_time),
                        "exit_time": bar.name.strftime("%Y-%m-%d %H:%M") if hasattr(bar.name, "strftime") else str(bar.name),
                        "symbol": active_ticket["symbol"],
                        "signal": active_ticket["signal"],
                        "entry_prem": round(entry_prem, 2),
                        "exit_prem": round(exit_prem, 2),
                        "lots": lots,
                        "pnl": round(trade_total_pnl, 2),
                        "result": "WIN" if trade_total_pnl > 0 else "LOSS",
                        "part_booked": part_booked
                    })
                    
                    equity_curve.append(round(capital, 2))
                    in_trade = False
                    active_ticket = None

        wins = [t for t in trade_log if t["pnl"] > 0]
        losses = [t for t in trade_log if t["pnl"] <= 0]
        win_rate = (len(wins) / len(trade_log) * 100.0) if trade_log else 0.0
        total_pnl = capital - self.initial_capital
        
        summary = {
            "initial_capital": round(self.initial_capital, 2),
            "final_capital": round(capital, 2),
            "pnl_rupees": round(total_pnl, 2),
            "return_pct": round((total_pnl / self.initial_capital) * 100.0, 2),
            "total_trades": len(trade_log),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 2)
        }

        return BacktestResults(summary=summary, trade_log=trade_log, equity_curve=equity_curve)
