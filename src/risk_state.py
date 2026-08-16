"""
SessionRiskState — Institutional Risk Rails & Live Budget Enforcement.
Enforces 3 trades/day, 2-strike loss streak halt, 1.5% DLL, and cooldown in the live execution loop.
"""

from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, Optional
import os
import json
from datetime import datetime
import pytz

from src.config import (
    MAX_TRADES_PER_DAY,
    MAX_CONSECUTIVE_LOSSES_DAY,
    DAILY_LOSS_LIMIT_PCT,
    DEFAULT_CAPITAL,
    COOLDOWN_BARS,
    MAX_OPEN_TRADES
)

IST = pytz.timezone("Asia/Kolkata")
DEFAULT_RISK_STATE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "risk_state_today.json")


@dataclass
class SessionRiskState:
    date: str = ""
    trades_today: int = 0
    consecutive_losses: int = 0
    realized_pnl_today: float = 0.0
    open_trades_count: int = 0
    last_entry_bar_idx: int = -999
    locked: bool = False
    lock_reason: str = ""
    max_trades_per_day: int = MAX_TRADES_PER_DAY
    max_consecutive_losses: int = MAX_CONSECUTIVE_LOSSES_DAY
    daily_loss_limit_pct: float = DAILY_LOSS_LIMIT_PCT
    account_capital: float = DEFAULT_CAPITAL
    cooldown_bars: int = COOLDOWN_BARS
    max_open_trades: int = MAX_OPEN_TRADES

    persistence_file: Optional[str] = None

    def __post_init__(self):
        if not self.date:
            self.date = datetime.now(IST).strftime("%Y-%m-%d")
        if not self.persistence_file:
            self.persistence_file = DEFAULT_RISK_STATE_PATH

    @property
    def daily_loss_limit_rupees(self) -> float:
        return self.account_capital * self.daily_loss_limit_pct

    def check_day_reset(self, current_date_str: Optional[str] = None) -> bool:
        """Resets daily counters if date has rolled over to a new session."""
        today_str = current_date_str or datetime.now(IST).strftime("%Y-%m-%d")
        if self.date != today_str:
            self.date = today_str
            self.trades_today = 0
            self.consecutive_losses = 0
            self.realized_pnl_today = 0.0
            self.open_trades_count = 0
            self.last_entry_bar_idx = -999
            self.locked = False
            self.lock_reason = ""
            return True
        return False

    def can_take_new_trade(self, current_bar_idx: int = 0) -> Tuple[bool, str]:
        """Evaluates whether session budget, circuit breakers, and cooldown allow a new trade entry."""
        self.check_day_reset()

        if self.locked:
            return False, f"Session Locked: {self.lock_reason}"

        if self.trades_today >= self.max_trades_per_day:
            return False, f"Session Budget Exceeded: {self.trades_today}/{self.max_trades_per_day} trades executed today."

        if self.open_trades_count >= self.max_open_trades:
            return False, f"Concurrency Limit: {self.open_trades_count} active position already open."

        if self.consecutive_losses >= self.max_consecutive_losses:
            self.locked = True
            self.lock_reason = f"2-Strike Loss Circuit Breaker ({self.consecutive_losses} consecutive losses)."
            return False, self.lock_reason

        if self.realized_pnl_today <= -self.daily_loss_limit_rupees:
            self.locked = True
            self.lock_reason = f"Daily Loss Limit Breached (PnL: ₹{self.realized_pnl_today:.2f} <= -₹{self.daily_loss_limit_rupees:.2f})."
            return False, self.lock_reason

        # Cooldown enforcement
        if self.last_entry_bar_idx >= 0 and current_bar_idx >= 0:
            if current_bar_idx >= self.last_entry_bar_idx:
                bars_since_last = current_bar_idx - self.last_entry_bar_idx
                if bars_since_last < self.cooldown_bars:
                    return False, f"Cooldown Active: {bars_since_last}/{self.cooldown_bars} bars elapsed since last entry."

        return True, "ALLOWED"

    def record_entry(self, bar_idx: int = 0, persist_path: Optional[str] = None) -> None:
        """Records a new trade entry."""
        self.check_day_reset()
        self.trades_today += 1
        self.open_trades_count += 1
        self.last_entry_bar_idx = bar_idx
        save_path = persist_path or self.persistence_file
        if save_path:
            self.save_to_disk(save_path)

    def record_trade_entry(self, bar_idx: int = 0, persist_path: Optional[str] = None) -> None:
        """Alias for record_entry."""
        self.record_entry(bar_idx, persist_path)

    def record_exit(self, realized_pnl: float, was_win: bool, persist_path: Optional[str] = None) -> None:
        """Records trade closure, updates streak counters, and checks loss breakers."""
        self.check_day_reset()
        self.open_trades_count = max(0, self.open_trades_count - 1)
        self.realized_pnl_today += realized_pnl

        if was_win:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.max_consecutive_losses:
                self.locked = True
                self.lock_reason = f"2-Strike Loss Circuit Breaker ({self.consecutive_losses} consecutive losses)."

        if self.realized_pnl_today <= -self.daily_loss_limit_rupees:
            self.locked = True
            self.lock_reason = f"Daily Loss Limit Breached (PnL: ₹{self.realized_pnl_today:.2f} <= -₹{self.daily_loss_limit_rupees:.2f})."

        save_path = persist_path or self.persistence_file
        if save_path:
            self.save_to_disk(save_path)

    def record_trade_exit(
        self,
        realized_pnl: float,
        r_multiple: float = 0.0,
        is_loss: Optional[bool] = None,
        was_win: Optional[bool] = None,
        persist_path: Optional[str] = None
    ) -> None:
        """Flexible alias for record_exit."""
        if was_win is None:
            if is_loss is not None:
                was_win = not is_loss
            else:
                was_win = realized_pnl > 0 or r_multiple > 0
        self.record_exit(realized_pnl, was_win, persist_path)

    def sync_from_journal(
        self,
        entries: Any,
        current_bar_idx: int = -1,
        bar_index_of: Optional[Any] = None,
        persist: bool = True,
        persist_path: Optional[str] = None
    ) -> "SessionRiskState":
        """
        Rebuilds today's risk counters from the signal journal, which is the source of
        truth for what was actually traded.

        Why derive instead of increment: record_entry/record_exit were never called from
        the live app, so trades_today / consecutive_losses / realized_pnl_today /
        open_trades_count sat at their defaults forever and can_take_new_trade() always
        returned ALLOWED — the trade budget, 2-strike halt, daily loss limit and
        concurrency cap never fired once. Streamlit also re-executes the whole script on
        every autorefresh, so incremental counters would double-count on each rerun.
        Recomputing from the journal is idempotent and rerun-safe.

        `bar_index_of` is an optional callable mapping an entry's bar_timestamp to a bar
        index, used to enforce the cooldown; when absent the cooldown is left untouched.
        """
        self.check_day_reset()
        today = self.date

        def _is_today(e: Any) -> bool:
            ts = str(getattr(e, "bar_timestamp", "") or getattr(e, "timestamp_ist", ""))
            return ts.startswith(today)

        # Seeded (replayed-history) rows are audit context, never real executions.
        live = [
            e for e in (entries or [])
            if _is_today(e)
            and not getattr(e, "is_seed", False)
            and str(getattr(e, "signal_type", "")) not in ("WAIT", "")
            and int(getattr(e, "selected_strike", 0) or 0) > 0
        ]

        self.trades_today = len(live)
        self.open_trades_count = sum(1 for e in live if hasattr(e, "is_active") and e.is_active())

        closed = [e for e in live if hasattr(e, "is_active") and not e.is_active()]
        self.realized_pnl_today = round(
            sum(float(getattr(e, "realized_pnl_rupees", 0.0) or 0.0) for e in closed), 2
        )

        # Trailing consecutive losses, newest first.
        # Scratches / breakeven trades (R >= -0.05 and PnL >= -50) do not count as losses.
        streak = 0
        for e in reversed(closed):
            r_mult = float(getattr(e, "realized_r_multiple", 0.0) or 0.0)
            pnl_val = float(getattr(e, "realized_pnl_rupees", 0.0) or 0.0)
            if r_mult > 0 or pnl_val > 0:
                break
            elif r_mult >= -0.05 and pnl_val >= -50.0:
                # Breakeven scratch: ignores and continues checking earlier trades
                continue
            else:
                streak += 1
        self.consecutive_losses = streak

        if callable(bar_index_of) and live:
            idx = bar_index_of(getattr(live[-1], "bar_timestamp", None))
            if idx is not None and idx >= 0:
                self.last_entry_bar_idx = int(idx)

        # Re-arm the breakers off the freshly derived numbers.
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.locked = True
            self.lock_reason = f"2-Strike Loss Circuit Breaker ({self.consecutive_losses} consecutive losses)."
        elif self.realized_pnl_today <= -self.daily_loss_limit_rupees:
            self.locked = True
            self.lock_reason = f"Daily Loss Limit Breached (PnL: ₹{self.realized_pnl_today:.2f} <= -₹{self.daily_loss_limit_rupees:.2f})."

        if persist:
            save_path = persist_path or self.persistence_file
            if save_path:
                self.save_to_disk(save_path)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionRiskState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save_to_disk(self, filepath: str = DEFAULT_RISK_STATE_PATH) -> None:
        try:
            dir_name = os.path.dirname(filepath)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            tmp_path = f"{filepath}.tmp.{os.getpid()}"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
            os.replace(tmp_path, filepath)
        except Exception:
            pass

    @classmethod
    def load_from_disk(cls, filepath: str = DEFAULT_RISK_STATE_PATH) -> "SessionRiskState":
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                state = cls.from_dict(data)
                state.check_day_reset()
                return state
            except Exception:
                pass
        return cls()
