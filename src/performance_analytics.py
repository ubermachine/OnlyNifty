"""OnlyNifty v3.3 Institutional Quantitative Performance & Risk Analytics Suite.

Provides exact institutional metrics:
- Sharpe Ratio (Annualized, zero-risk benchmark)
- Sortino Ratio (Downside deviation risk-adjusted return)
- Calmar Ratio (CAGR / Max Drawdown)
- Ulcer Index & Martin Ratio (Ulcer Performance Index)
- Profit Factor & Win/Loss Payoff Ratio
- Max Consecutive Losses & Wins Distribution
- Capital Drawdown Recovery Protocol (Dynamic Kelly Scaling)
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd


# ----------------- PERFORMANCE RATIOS -----------------

def calculate_sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = 0.0,
    annualization_factor: float = 252.0
) -> float:
    """Computes annualized Sharpe Ratio with zero-risk benchmark or custom R_f."""
    if not returns or len(returns) < 2:
        return 0.0
    arr = np.array(returns, dtype=np.float64)
    std = float(np.std(arr, ddof=1))
    if std <= 1e-9:
        return 0.0
    excess_mean = float(np.mean(arr)) - (risk_free_rate / annualization_factor)
    sharpe = (excess_mean / std) * math.sqrt(annualization_factor)
    return round(float(sharpe), 3)


def calculate_sortino_ratio(
    returns: List[float],
    mar: float = 0.0,
    annualization_factor: float = 252.0
) -> float:
    """Computes annualized Sortino Ratio focusing on downside semi-deviation."""
    if not returns or len(returns) < 2:
        return 0.0
    arr = np.array(returns, dtype=np.float64)
    mean_ret = float(np.mean(arr))
    downside = np.minimum(0.0, arr - mar)
    downside_var = float(np.mean(downside ** 2))
    if downside_var <= 1e-12:
        return 999.0 if mean_ret > mar else 0.0
    downside_dev = math.sqrt(downside_var)
    sortino = ((mean_ret - mar) / downside_dev) * math.sqrt(annualization_factor)
    return round(float(sortino), 3)


def calculate_calmar_ratio(
    equity_curve: List[float],
    trading_days: int = 252,
    annualization_factor: float = 252.0
) -> Dict[str, float]:
    """Computes Calmar Ratio (CAGR / Max Drawdown)."""
    if not equity_curve or len(equity_curve) < 2:
        return {"calmar_ratio": 0.0, "cagr_pct": 0.0, "max_drawdown_pct": 0.0, "max_drawdown_rupees": 0.0}

    arr = np.array(equity_curve, dtype=np.float64)
    initial = arr[0]
    final = arr[-1]
    
    cum_max = np.maximum.accumulate(arr)
    drawdowns_rupees = cum_max - arr
    drawdowns_pct = np.where(cum_max > 0, drawdowns_rupees / cum_max, 0.0)
    
    mdd_pct = float(np.max(drawdowns_pct))
    mdd_rupees = float(np.max(drawdowns_rupees))
    
    total_return = (final - initial) / initial if initial > 0 else 0.0
    days = max(trading_days, 1)
    
    # Compound Annual Growth Rate (CAGR)
    if total_return > -1.0 and initial > 0:
        cagr = ((final / initial) ** (annualization_factor / days)) - 1.0
    else:
        cagr = -1.0
        
    cagr_pct = cagr * 100.0
    mdd_pct_display = mdd_pct * 100.0
    
    if mdd_pct <= 1e-6:
        calmar = 999.0 if cagr > 0 else 0.0
    else:
        calmar = cagr / mdd_pct

    return {
        "calmar_ratio": round(float(calmar), 3),
        "cagr_pct": round(float(cagr_pct), 2),
        "max_drawdown_pct": round(float(mdd_pct_display), 2),
        "max_drawdown_rupees": round(float(mdd_rupees), 2)
    }


def calculate_ulcer_index_and_martin_ratio(
    equity_curve: List[float],
    cagr_pct: Optional[float] = None,
    risk_free_rate_pct: float = 0.0
) -> Dict[str, float]:
    """Computes Peter Martin's Ulcer Index (UI) and Martin Ratio (Ulcer Performance Index - UPI)."""
    if not equity_curve or len(equity_curve) < 2:
        return {"ulcer_index": 0.0, "martin_ratio": 0.0}

    arr = np.array(equity_curve, dtype=np.float64)
    cum_max = np.maximum.accumulate(arr)
    
    # Drawdown percentage series from running high-water mark
    dd_pct_series = np.where(cum_max > 0, 100.0 * (arr - cum_max) / cum_max, 0.0)
    squared_drawdowns = dd_pct_series ** 2
    ulcer_index = math.sqrt(float(np.mean(squared_drawdowns)))
    
    if cagr_pct is None:
        total_ret_pct = ((arr[-1] - arr[0]) / arr[0]) * 100.0 if arr[0] > 0 else 0.0
        annualized_return_pct = total_ret_pct
    else:
        annualized_return_pct = cagr_pct

    excess_return = annualized_return_pct - risk_free_rate_pct
    if ulcer_index <= 1e-6:
        martin_ratio = 999.0 if excess_return > 0 else 0.0
    else:
        martin_ratio = excess_return / ulcer_index

    return {
        "ulcer_index": round(float(ulcer_index), 3),
        "martin_ratio": round(float(martin_ratio), 3)
    }


def calculate_profit_factor(
    trade_pnls: List[float]
) -> Dict[str, float]:
    """Computes Gross Profit, Gross Loss, and Profit Factor."""
    if not trade_pnls:
        return {"profit_factor": 0.0, "gross_profit": 0.0, "gross_loss": 0.0}

    pnls = np.array(trade_pnls, dtype=np.float64)
    wins = pnls[pnls > 0]
    losses = np.abs(pnls[pnls < 0])
    
    gross_profit = float(np.sum(wins))
    gross_loss = float(np.sum(losses))
    
    if gross_loss <= 1e-6:
        pf = 999.0 if gross_profit > 0 else 0.0
    else:
        pf = gross_profit / gross_loss

    return {
        "profit_factor": round(float(pf), 3),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2)
    }


def calculate_payoff_ratio(
    trade_pnls: List[float]
) -> Dict[str, float]:
    """Computes Average Win/Loss Payoff Ratio and individual expectancy values."""
    if not trade_pnls:
        return {"payoff_ratio": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "win_rate_pct": 0.0}

    pnls = np.array(trade_pnls, dtype=np.float64)
    wins = pnls[pnls > 0]
    losses = np.abs(pnls[pnls < 0])
    
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    win_rate = (len(wins) / len(pnls)) * 100.0
    
    if avg_loss <= 1e-6:
        payoff = 999.0 if avg_win > 0 else 0.0
    else:
        payoff = avg_win / avg_loss

    return {
        "payoff_ratio": round(float(payoff), 3),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "win_rate_pct": round(win_rate, 2)
    }


def calculate_consecutive_streaks_distribution(
    trade_pnls: List[float]
) -> Dict[str, Any]:
    """Analyzes streak dynamics: Max Consecutive Wins/Losses, Current Streak, and full distribution."""
    if not trade_pnls:
        return {
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
            "current_streak": 0,
            "avg_win_streak": 0.0,
            "avg_loss_streak": 0.0,
            "win_streak_distribution": {},
            "loss_streak_distribution": {}
        }

    win_streaks: List[int] = []
    loss_streaks: List[int] = []
    
    current_type: Optional[str] = None
    current_count = 0

    for pnl in trade_pnls:
        if pnl > 0:
            outcome = "W"
        elif pnl < 0:
            outcome = "L"
        else:
            continue

        if outcome == current_type:
            current_count += 1
        else:
            if current_type == "W":
                win_streaks.append(current_count)
            elif current_type == "L":
                loss_streaks.append(current_count)
            current_type = outcome
            current_count = 1

    if current_type == "W":
        win_streaks.append(current_count)
    elif current_type == "L":
        loss_streaks.append(current_count)

    max_wins = max(win_streaks) if win_streaks else 0
    max_losses = max(loss_streaks) if loss_streaks else 0
    avg_wins = float(np.mean(win_streaks)) if win_streaks else 0.0
    avg_losses = float(np.mean(loss_streaks)) if loss_streaks else 0.0
    
    # Signed current streak (+k for wins, -k for losses)
    if current_type == "W":
        signed_current_streak = current_count
    elif current_type == "L":
        signed_current_streak = -current_count
    else:
        signed_current_streak = 0

    # Frequency Distributions
    win_dist: Dict[int, int] = {}
    for w in win_streaks:
        win_dist[w] = win_dist.get(w, 0) + 1
        
    loss_dist: Dict[int, int] = {}
    for l in loss_streaks:
        loss_dist[l] = loss_dist.get(l, 0) + 1

    return {
        "max_consecutive_wins": max_wins,
        "max_consecutive_losses": max_losses,
        "current_streak": signed_current_streak,
        "avg_win_streak": round(avg_wins, 2),
        "avg_loss_streak": round(avg_losses, 2),
        "win_streak_distribution": dict(sorted(win_dist.items())),
        "loss_streak_distribution": dict(sorted(loss_dist.items()))
    }


def calculate_value_at_risk_and_cvar(
    returns: List[float],
    initial_capital: float = 500000.0,
    confidence: float = 0.95
) -> Dict[str, float]:
    """
    Computes Parametric & Historical Value-at-Risk (VaR) and Expected Shortfall (CVaR).
    
    VaR_95%: Maximum expected loss at 95% confidence over a single trade/day.
    CVaR_95%: Expected loss conditional on breaching VaR threshold (Tail Risk).
    """
    if len(returns) < 5:
        return {
            "var_95_pct": 1.0, "var_95_rupees": initial_capital * 0.01,
            "cvar_95_pct": 1.5, "cvar_95_rupees": initial_capital * 0.015,
            "var_99_pct": 2.0, "var_99_rupees": initial_capital * 0.02,
            "cvar_99_pct": 2.8, "cvar_99_rupees": initial_capital * 0.028
        }

    arr = np.array(returns, dtype=np.float64)
    # Historical VaR
    var_95_hist = - float(np.percentile(arr, (1.0 - 0.95) * 100.0))
    var_99_hist = - float(np.percentile(arr, (1.0 - 0.99) * 100.0))

    # Parametric Gaussian VaR
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.01
    var_95_param = - (mu - 1.645 * sigma)
    var_99_param = - (mu - 2.326 * sigma)

    var_95 = max(round(max(var_95_hist, var_95_param) * 100.0, 2), 0.1)
    var_99 = max(round(max(var_99_hist, var_99_param) * 100.0, 2), 0.2)

    # Conditional VaR (Expected Shortfall)
    tail_losses_95 = [r for r in arr if r <= - (var_95 / 100.0)]
    cvar_95 = abs(float(np.mean(tail_losses_95)) * 100.0) if tail_losses_95 else var_95 * 1.3

    tail_losses_99 = [r for r in arr if r <= - (var_99 / 100.0)]
    cvar_99 = abs(float(np.mean(tail_losses_99)) * 100.0) if tail_losses_99 else var_99 * 1.4

    return {
        "var_95_pct": round(var_95, 2),
        "var_95_rupees": round((var_95 / 100.0) * initial_capital, 2),
        "cvar_95_pct": round(cvar_95, 2),
        "cvar_95_rupees": round((cvar_95 / 100.0) * initial_capital, 2),
        "var_99_pct": round(var_99, 2),
        "var_99_rupees": round((var_99 / 100.0) * initial_capital, 2),
        "cvar_99_pct": round(cvar_99, 2),
        "cvar_99_rupees": round((cvar_99 / 100.0) * initial_capital, 2)
    }



def compute_institutional_performance_suite(
    equity_curve: List[float],
    trade_log: List[Dict[str, Any]],
    initial_capital: float,
    trading_days: int = 252,
    risk_free_rate: float = 0.0
) -> Dict[str, Any]:
    """Unified Orchestrator: Computes all institutional risk & performance metrics."""
    trade_pnls = [float(t.get("net_pnl", t.get("pnl", 0.0))) for t in trade_log]
    
    # Percentage returns for Sharpe & Sortino
    if len(equity_curve) > 1:
        eq_arr = np.array(equity_curve, dtype=np.float64)
        pct_returns = (np.diff(eq_arr) / eq_arr[:-1]).tolist()
    else:
        pct_returns = []

    sharpe = calculate_sharpe_ratio(pct_returns, risk_free_rate=risk_free_rate)
    sortino = calculate_sortino_ratio(pct_returns, mar=risk_free_rate)
    calmar_data = calculate_calmar_ratio(equity_curve, trading_days=trading_days)
    ulcer_data = calculate_ulcer_index_and_martin_ratio(equity_curve, cagr_pct=calmar_data["cagr_pct"])
    pf_data = calculate_profit_factor(trade_pnls)
    payoff_data = calculate_payoff_ratio(trade_pnls)
    streak_data = calculate_consecutive_streaks_distribution(trade_pnls)
    var_cvar_data = calculate_value_at_risk_and_cvar(pct_returns, initial_capital=initial_capital)
    
    net_pnl = round(equity_curve[-1] - initial_capital, 2) if equity_curve else 0.0
    return {
        "initial_capital": round(initial_capital, 2),
        "final_capital": round(equity_curve[-1], 2) if equity_curve else initial_capital,
        "pnl_rupees": net_pnl,
        "return_pct": round((net_pnl / initial_capital) * 100.0, 2) if initial_capital > 0 else 0.0,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar_data["calmar_ratio"],
        "cagr_pct": calmar_data["cagr_pct"],
        "max_drawdown_pct": calmar_data["max_drawdown_pct"],
        "max_drawdown_rupees": calmar_data["max_drawdown_rupees"],
        "ulcer_index": ulcer_data["ulcer_index"],
        "martin_ratio": ulcer_data["martin_ratio"],
        "profit_factor": pf_data["profit_factor"],
        "gross_profit": pf_data["gross_profit"],
        "gross_loss": pf_data["gross_loss"],
        "payoff_ratio": payoff_data["payoff_ratio"],
        "avg_win_rupees": payoff_data["avg_win"],
        "avg_loss_rupees": payoff_data["avg_loss"],
        "win_rate_pct": payoff_data["win_rate_pct"],
        "max_consecutive_wins": streak_data["max_consecutive_wins"],
        "max_consecutive_losses": streak_data["max_consecutive_losses"],
        "current_streak": streak_data["current_streak"],
        "avg_win_streak": streak_data["avg_win_streak"],
        "avg_loss_streak": streak_data["avg_loss_streak"],
        "win_streak_distribution": streak_data["win_streak_distribution"],
        "loss_streak_distribution": streak_data["loss_streak_distribution"],
        "var_95_pct": var_cvar_data["var_95_pct"],
        "var_95_rupees": var_cvar_data["var_95_rupees"],
        "cvar_95_pct": var_cvar_data["cvar_95_pct"],
        "cvar_95_rupees": var_cvar_data["cvar_95_rupees"],
        "var_99_pct": var_cvar_data["var_99_pct"],
        "var_99_rupees": var_cvar_data["var_99_rupees"],
        "cvar_99_pct": var_cvar_data["cvar_99_pct"],
        "cvar_99_rupees": var_cvar_data["cvar_99_rupees"]
    }



# ----------------- CAPITAL DRAWDOWN RECOVERY PROTOCOL -----------------

@dataclass
class DrawdownRecoveryState:
    current_equity: float
    peak_equity: float
    current_drawdown_pct: float
    trough_drawdown_pct: float
    recovery_progress_ratio: float
    consecutive_wins_since_trough: int
    recovery_stage: str
    dampener: float
    kelly_fraction: float
    effective_risk_pct: float
    is_halted: bool
    status_message: str


class DynamicKellyRecoveryProtocol:
    """Mathematical Model for Progressive Risk Re-escalation after Drawdown Recovery."""

    def __init__(
        self,
        initial_capital: float = 500000.0,
        base_risk_pct: float = 0.01,
        kelly_fraction: float = 0.25,
        max_tolerable_mdd: float = 0.10,
        confirmation_wins_required: int = 2,
        convex_exponent_beta: float = 1.5
    ):
        self.initial_capital = initial_capital
        self.base_risk_pct = base_risk_pct
        self.kelly_fraction = kelly_fraction
        self.max_tolerable_mdd = max_tolerable_mdd
        self.k_req = confirmation_wins_required
        self.beta = convex_exponent_beta
        
        self.peak_equity = initial_capital
        self.trough_drawdown_pct = 0.0
        self.consecutive_wins_since_trough = 0
        self.in_recovery_mode = False

    def compute_deescalation_dampener(self, dd: float) -> float:
        """Non-linear de-escalation curve when moving deeper into drawdown."""
        if dd >= self.max_tolerable_mdd:
            return 0.0
        elif dd <= 0.03:
            return 1.0
        elif dd <= 0.06:
            return 1.0 - ((dd - 0.03) / 0.03) * 0.50
        else:
            norm_val = (dd - 0.06) / (self.max_tolerable_mdd - 0.06)
            return max(0.50 * ((1.0 - norm_val) ** 2), 0.05)

    def evaluate_capital_state(
        self,
        current_equity: float,
        recent_trade_pnl: Optional[float] = None,
        rolling_win_rate: float = 0.58,
        rolling_payoff_ratio: float = 2.10
    ) -> DrawdownRecoveryState:
        """Evaluates equity state and returns dynamic risk multiplier with hysteresis."""
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            self.trough_drawdown_pct = 0.0
            self.consecutive_wins_since_trough = 0
            self.in_recovery_mode = False

        current_dd = max((self.peak_equity - current_equity) / self.peak_equity, 0.0) if self.peak_equity > 0 else 0.0

        # Update streak tracker
        if recent_trade_pnl is not None:
            if recent_trade_pnl > 0:
                self.consecutive_wins_since_trough += 1
            else:
                self.consecutive_wins_since_trough = 0

        # Update Trough
        if current_dd > self.trough_drawdown_pct:
            self.trough_drawdown_pct = current_dd
            self.consecutive_wins_since_trough = 0
            self.in_recovery_mode = False

        # Compute Recovery Ratio (rho)
        if self.trough_drawdown_pct > 0.005 and current_dd < self.trough_drawdown_pct:
            self.in_recovery_mode = True
            recovery_rho = (self.trough_drawdown_pct - current_dd) / self.trough_drawdown_pct
            recovery_rho = min(max(recovery_rho, 0.0), 1.0)
        else:
            recovery_rho = 0.0

        delta_down = self.compute_deescalation_dampener(current_dd)
        delta_trough = self.compute_deescalation_dampener(self.trough_drawdown_pct)

        # Re-escalation Logic with Confirmation Gate
        if current_dd >= self.max_tolerable_mdd:
            dampener = 0.0
            stage = "CIRCUIT_BREAKER_HALT"
            msg = f"🛑 10% Hard MDD Breached ({current_dd*100:.2f}%). Trading halted."
            is_halted = True
        elif not self.in_recovery_mode:
            dampener = delta_down
            is_halted = False
            if current_dd <= 0.01:
                stage = "ALL_TIME_HIGH"
                msg = "🟢 Account at All-Time High. Full Quarter-Kelly active."
            else:
                stage = "DRAWDOWN_EXPANSION"
                msg = f"⚠️ Drawdown Expanding ({current_dd*100:.2f}%). Defensive contraction to {dampener*100:.1f}% risk."
        else:
            # Progressive Re-escalation with Hysteresis Confirmation
            psi_gate = min(1.0, (self.consecutive_wins_since_trough / self.k_req) ** 0.8) if self.k_req > 0 else 1.0
            delta_up = delta_trough + (1.0 - delta_trough) * (recovery_rho ** self.beta)
            recovery_dampener = delta_trough + (delta_up - delta_trough) * psi_gate
            dampener = min(recovery_dampener, delta_down)
            is_halted = False
            
            if recovery_rho >= 0.90:
                stage = "RECOVERY_CONFIRMED"
                msg = f"🚀 Drawdown {recovery_rho*100:.1f}% Recovered. Re-escalated to {dampener*100:.1f}% Kelly."
            else:
                stage = "PROGRESSIVE_RECOVERY"
                msg = f"🔄 Re-escalating ({recovery_rho*100:.1f}% of Trough). Win Streak: {self.consecutive_wins_since_trough}/{self.k_req}. Dampener: {dampener*100:.1f}%."

        # Dynamic Kelly fraction
        if rolling_payoff_ratio > 0:
            full_kelly = max(0.0, (rolling_win_rate * rolling_payoff_ratio - (1.0 - rolling_win_rate)) / rolling_payoff_ratio)
        else:
            full_kelly = 0.0
            
        scaled_kelly = self.kelly_fraction * full_kelly * dampener
        effective_risk = self.base_risk_pct * dampener

        return DrawdownRecoveryState(
            current_equity=round(current_equity, 2),
            peak_equity=round(self.peak_equity, 2),
            current_drawdown_pct=round(current_dd * 100.0, 2),
            trough_drawdown_pct=round(self.trough_drawdown_pct * 100.0, 2),
            recovery_progress_ratio=round(recovery_rho, 3),
            consecutive_wins_since_trough=self.consecutive_wins_since_trough,
            recovery_stage=stage,
            dampener=round(dampener, 3),
            kelly_fraction=round(scaled_kelly, 4),
            effective_risk_pct=round(effective_risk * 100.0, 3),
            is_halted=is_halted,
            status_message=msg
        )
