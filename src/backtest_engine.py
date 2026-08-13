"""JustNifty v3.0 Institutional Bar-by-Bar Simulator with 3-Leg TCA Friction, 2-Strike Circuit Breaker, and 0DTE Adaptation."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd
from src.config import (
    DEFAULT_CAPITAL, LOT_SIZE, DAILY_LOSS_LIMIT_PCT,
    MAX_CONSECUTIVE_LOSSES_DAY, MAX_TRADES_PER_DAY
)
from src.strategy_rules import StrategyEngine, SignalType
from src.options_engine import generate_option_trade_ticket, calculate_adaptive_tca_friction

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
        """Executes a chronological bar-by-bar simulation with TCA friction deductions, 2-strike kill-switch, and 0DTE deep-ITM adaptation."""
        if df_5m.empty or len(df_5m) < 25:
            return BacktestResults(
                summary={
                    "initial_capital": self.initial_capital,
                    "final_capital": self.initial_capital,
                    "gross_pnl": 0.0,
                    "total_tca": 0.0,
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
        total_tca_accumulated = 0.0
        gross_pnl_accumulated = 0.0
        
        in_trade = False
        active_ticket: Optional[Dict[str, Any]] = None
        part_booked = False
        entry_time = None
        entry_idx = 0
        peak_capital = capital
        
        # Stateful Intraday Circuit Breaker Tracking
        current_day_str = None
        daily_trades_count = 0
        daily_consecutive_losses = 0
        daily_realized_loss = 0.0
        
        for i in range(25, len(df_5m)):
            bar = df_5m.iloc[i]
            close = float(bar["close"])
            high = float(bar["high"])
            low = float(bar["low"])
            bar_time_str = bar.name.strftime("%H:%M") if hasattr(bar.name, "strftime") else "12:00"
            bar_date_str = bar.name.strftime("%Y-%m-%d") if hasattr(bar.name, "strftime") else "2026-08-01"
            
            # Reset daily counters on new session
            if bar_date_str != current_day_str:
                current_day_str = bar_date_str
                daily_trades_count = 0
                daily_consecutive_losses = 0
                daily_realized_loss = 0.0
                
            # Current Drawdown for Quarter-Kelly Dampener
            current_dd_pct = max((peak_capital - capital) / peak_capital, 0.0) if peak_capital > 0 else 0.0
            
            # Check if session is circuit-breaker locked (2-Strike Rule or Daily Loss Limit)
            is_session_locked = (
                daily_consecutive_losses >= MAX_CONSECUTIVE_LOSSES_DAY or
                daily_realized_loss >= (capital * DAILY_LOSS_LIMIT_PCT) or
                daily_trades_count >= MAX_TRADES_PER_DAY
            )
            
            # 0DTE Expiry Thursday Afternoon Detection
            is_thursday = hasattr(bar.name, "weekday") and bar.name.weekday() == 3
            is_0dte_afternoon = is_thursday and (bar_time_str >= "12:30")
            
            if not in_trade:
                if not is_session_locked:
                    signal = self.strategy.evaluate_bar(df_5m, current_idx=i)
                    if signal.signal_type in [SignalType.LONG, SignalType.SHORT, SignalType.LONG_3PM, SignalType.SHORT_3PM]:
                        ticket = generate_option_trade_ticket(
                            close, signal, capital, current_dd_pct,
                            t_days=0.15 if is_0dte_afternoon else 4.0,
                            is_0dte_afternoon=is_0dte_afternoon
                        )
                        if ticket.get("status") == "READY" and ticket.get("lots", 0) > 0:
                            in_trade = True
                            active_ticket = ticket
                            part_booked = False
                            entry_idx = i
                            entry_time = bar.name
                            daily_trades_count += 1
            else:
                delta = abs(active_ticket["delta"])
                entry_prem = float(active_ticket["entry_premium"])
                sl_prem = float(active_ticket["sl_premium"])
                t1_prem = float(active_ticket["target1_premium"])
                lots = int(active_ticket["lots"])
                is_call = "CE" in active_ticket["option_type"]
                
                # Approximate current high/low option price from spot with Gamma convexity
                spot_entry = df_5m.iloc[entry_idx]["close"]
                gamma = float(active_ticket.get("gamma", 0.0008))
                
                if is_call:
                    dS_high = high - spot_entry
                    dS_low = low - spot_entry
                    opt_high = entry_prem + (dS_high * delta) + (0.5 * gamma * (dS_high ** 2))
                    opt_low = entry_prem + (dS_low * delta) - (0.5 * gamma * (dS_low ** 2))
                else:
                    dS_high = spot_entry - low
                    dS_low = spot_entry - high
                    opt_high = entry_prem + (dS_high * delta) + (0.5 * gamma * (dS_high ** 2))
                    opt_low = entry_prem + (dS_low * delta) - (0.5 * gamma * (dS_low ** 2))

                # 1. Check 50% Part-Booking hit
                if not part_booked and opt_high >= t1_prem:
                    part_booked = True
                    booked_lots = max(lots // 2, 1)
                    pnl_50 = (t1_prem - entry_prem) * booked_lots * LOT_SIZE
                    capital += pnl_50
                    gross_pnl_accumulated += pnl_50
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
                    gross_pnl_accumulated += pnl_rem
                    
                    pnl_first_half = (t1_prem - entry_prem) * booked_lots * LOT_SIZE if part_booked else 0.0
                    gross_trade_pnl = pnl_first_half + pnl_rem
                    
                    # 3-Leg Part-Booking Indian Market TCA Friction
                    tca = calculate_adaptive_tca_friction(
                        entry_prem, t1_prem, exit_prem,
                        lots * LOT_SIZE, lots,
                        part_booked=part_booked,
                        is_0dte_afternoon=is_0dte_afternoon
                    )
                    friction = tca["total_friction"]
                    net_trade_pnl = gross_trade_pnl - friction
                    
                    capital -= friction
                    total_tca_accumulated += friction
                    
                    # Update session circuit-breaker state
                    if net_trade_pnl <= 0:
                        daily_consecutive_losses += 1
                        daily_realized_loss += abs(net_trade_pnl)
                    else:
                        daily_consecutive_losses = 0
                        
                    if capital > peak_capital:
                        peak_capital = capital
                    
                    trade_log.append({
                        "entry_time": entry_time.strftime("%Y-%m-%d %H:%M") if hasattr(entry_time, "strftime") else str(entry_time),
                        "exit_time": bar.name.strftime("%Y-%m-%d %H:%M") if hasattr(bar.name, "strftime") else str(bar.name),
                        "symbol": active_ticket["symbol"],
                        "signal": active_ticket["signal"],
                        "entry_prem": round(entry_prem, 2),
                        "exit_prem": round(exit_prem, 2),
                        "lots": lots,
                        "pnl": round(net_trade_pnl, 2),  # Compatibility key
                        "gross_pnl": round(gross_trade_pnl, 2),
                        "tca_fees": round(friction, 2),
                        "net_pnl": round(net_trade_pnl, 2),
                        "result": "WIN" if net_trade_pnl > 0 else "LOSS",
                        "part_booked": part_booked
                    })
                    
                    equity_curve.append(round(capital, 2))
                    in_trade = False
                    active_ticket = None

        wins = [t for t in trade_log if t["net_pnl"] > 0]
        losses = [t for t in trade_log if t["net_pnl"] <= 0]
        win_rate = (len(wins) / len(trade_log) * 100.0) if trade_log else 0.0
        net_total_pnl = capital - self.initial_capital
        
        summary = {
            "initial_capital": round(self.initial_capital, 2),
            "final_capital": round(capital, 2),
            "gross_pnl": round(gross_pnl_accumulated, 2),
            "total_tca": round(total_tca_accumulated, 2),
            "pnl_rupees": round(net_total_pnl, 2),
            "return_pct": round((net_total_pnl / self.initial_capital) * 100.0, 2),
            "total_trades": len(trade_log),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 2)
        }

        return BacktestResults(summary=summary, trade_log=trade_log, equity_curve=equity_curve)
