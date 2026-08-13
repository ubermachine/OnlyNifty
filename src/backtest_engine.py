"""JustNifty v3.1 Institutional Bar-by-Bar Simulator with 3-Tier Exits, Dynamic Trailing Ratchets, and 0DTE Adaptation."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd
from src.config import (
    DEFAULT_CAPITAL, LOT_SIZE, DAILY_LOSS_LIMIT_PCT,
    MAX_CONSECUTIVE_LOSSES_DAY, MAX_TRADES_PER_DAY
)
from src.strategy_rules import StrategyEngine, SignalType
from src.options_engine import (
    generate_option_trade_ticket,
    calculate_adaptive_tca_friction_multi_tier,
    compute_dynamic_trailing_option_sl
)
from src.performance_analytics import compute_institutional_performance_suite


@dataclass
class BacktestResults:
    summary: Dict[str, Any]
    trade_log: List[Dict[str, Any]]
    equity_curve: List[float]

class BacktestEngine:
    def __init__(self, initial_capital: float = DEFAULT_CAPITAL):
        self.initial_capital = initial_capital
        self.strategy = StrategyEngine()

    def run_backtest(
        self,
        df_5m: pd.DataFrame,
        enable_3tier: bool = True,
        enable_dynamic_trailing: bool = True,
        enable_pyramiding: bool = False
    ) -> BacktestResults:
        """Executes high-fidelity bar-by-bar institutional simulation with 3-tier exits, dynamic trailing, and 4-leg TCA friction."""
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
        t1_booked = False
        t2_booked = False
        pyramided = False
        
        entry_time = None
        entry_idx = 0
        peak_capital = capital
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
            
            if bar_date_str != current_day_str:
                current_day_str = bar_date_str
                daily_trades_count = 0
                daily_consecutive_losses = 0
                daily_realized_loss = 0.0
                
            current_dd_pct = max((peak_capital - capital) / peak_capital, 0.0) if peak_capital > 0 else 0.0
            
            is_session_locked = (
                daily_consecutive_losses >= MAX_CONSECUTIVE_LOSSES_DAY or
                daily_realized_loss >= (capital * DAILY_LOSS_LIMIT_PCT) or
                daily_trades_count >= MAX_TRADES_PER_DAY
            )
            
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
                            t1_booked = False
                            t2_booked = False
                            pyramided = False
                            entry_idx = i
                            entry_time = bar.name
                            daily_trades_count += 1
            else:
                delta = abs(active_ticket["delta"])
                entry_prem = float(active_ticket["entry_premium"])
                t1_prem = float(active_ticket["target1_premium"])
                t2_prem = float(active_ticket["target2_premium"])
                t3_prem = float(active_ticket["target3_moonshot_premium"])
                lots = int(active_ticket["lots"])
                is_call = "CE" in active_ticket["option_type"]
                spot_entry = float(active_ticket.get("spot_entry", df_5m.iloc[entry_idx]["close"]))
                gamma = float(active_ticket.get("gamma", 0.0008))
                
                # Approximate current option high/low via second-order Taylor series
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
                
                lots_t1 = active_ticket.get("lots_t1_35pct", max(lots // 2, 1))
                lots_t2 = active_ticket.get("lots_t2_35pct", max(lots // 2, 1))
                lots_t3 = active_ticket.get("lots_t3_30pct", max(lots - lots_t1 - lots_t2, 1))

                # 1. TIER 1 EXIT (35% at +1.2x ATR)
                if not t1_booked and opt_high >= t1_prem:
                    t1_booked = True
                    pnl_t1 = (t1_prem - entry_prem) * lots_t1 * LOT_SIZE
                    capital += pnl_t1
                    gross_pnl_accumulated += pnl_t1

                # 2. TIER 2 EXIT (35% at +2.5x ATR)
                if enable_3tier and t1_booked and not t2_booked and opt_high >= t2_prem:
                    t2_booked = True
                    pnl_t2 = (t2_prem - entry_prem) * lots_t2 * LOT_SIZE
                    capital += pnl_t2
                    gross_pnl_accumulated += pnl_t2

                # 3. DYNAMIC TRAILING SL UPDATE (21 EMA / AVWAP 1σ Floor)
                if enable_dynamic_trailing and t1_booked:
                    ema21_now = float(df_5m["close"].iloc[:i+1].ewm(span=21, adjust=False).mean().iloc[-1])
                    elapsed_bars = i - entry_idx
                    dynamic_opt_sl = compute_dynamic_trailing_option_sl(
                        entry_prem=entry_prem,
                        spot_entry=spot_entry,
                        current_trailing_spot_sl=ema21_now,
                        delta=delta,
                        gamma=gamma,
                        theta_daily=float(active_ticket.get("theta_decay_daily", -14.0)),
                        elapsed_bars_5m=elapsed_bars,
                        is_call=is_call
                    )
                    active_ticket["sl_premium"] = max(dynamic_opt_sl, entry_prem)
                elif t1_booked and not enable_dynamic_trailing:
                    active_ticket["sl_premium"] = entry_prem

                # 4. CHECK STOP-LOSS OR EOD SQUARE-OFF
                is_eod = (i == len(df_5m) - 1) or (bar_time_str >= "15:20")
                sl_hit = opt_low <= float(active_ticket["sl_premium"])
                
                if sl_hit or is_eod:
                    exit_prem = float(active_ticket["sl_premium"]) if sl_hit else entry_prem + ((close - spot_entry) * delta if is_call else (spot_entry - close) * delta)
                    
                    if enable_3tier:
                        if t1_booked and t2_booked:
                            rem_lots = max(lots - lots_t1 - lots_t2, 0)
                        elif t1_booked:
                            rem_lots = max(lots - lots_t1, 0)
                        else:
                            rem_lots = lots
                    else:
                        booked_lots = lots_t1 if t1_booked else 0
                        rem_lots = max(lots - booked_lots, 0)
                        
                    pnl_rem = (exit_prem - entry_prem) * rem_lots * LOT_SIZE
                    capital += pnl_rem
                    gross_pnl_accumulated += pnl_rem
                    
                    pnl_realized_prior = 0.0
                    if t1_booked:
                        pnl_realized_prior += (t1_prem - entry_prem) * lots_t1 * LOT_SIZE
                    if t2_booked and enable_3tier:
                        pnl_realized_prior += (t2_prem - entry_prem) * lots_t2 * LOT_SIZE
                        
                    gross_trade_pnl = pnl_realized_prior + pnl_rem
                    
                    tca = calculate_adaptive_tca_friction_multi_tier(
                        entry_prem=entry_prem,
                        t1_prem=t1_prem,
                        t2_prem=t2_prem,
                        final_exit_prem=exit_prem,
                        total_qty=lots * LOT_SIZE,
                        lots=lots,
                        t1_hit=t1_booked,
                        t2_hit=t2_booked,
                        is_pyramided=pyramided,
                        is_0dte_afternoon=is_0dte_afternoon
                    )
                    friction = tca["total_friction"]
                    net_trade_pnl = gross_trade_pnl - friction
                    
                    capital -= friction
                    total_tca_accumulated += friction
                    
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
                        "pnl": round(net_trade_pnl, 2),
                        "gross_pnl": round(gross_trade_pnl, 2),
                        "tca_fees": round(friction, 2),
                        "net_pnl": round(net_trade_pnl, 2),
                        "result": "WIN" if net_trade_pnl > 0 else "LOSS",
                        "t1_booked": t1_booked,
                        "t2_booked": t2_booked,
                        "part_booked": t1_booked
                    })
                    
                    equity_curve.append(round(capital, 2))
                    in_trade = False
                    active_ticket = None

        wins = [t for t in trade_log if t["net_pnl"] > 0]
        losses = [t for t in trade_log if t["net_pnl"] <= 0]
        
        perf_metrics = compute_institutional_performance_suite(
            equity_curve=equity_curve,
            trade_log=trade_log,
            initial_capital=self.initial_capital,
            trading_days=max(len(df_5m) // 75, 1)
        )
        
        summary = {
            **perf_metrics,
            "gross_pnl": round(gross_pnl_accumulated, 2),
            "total_tca": round(total_tca_accumulated, 2),
            "total_trades": len(trade_log),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(perf_metrics["win_rate_pct"], 2)
        }

        return BacktestResults(summary=summary, trade_log=trade_log, equity_curve=equity_curve)

