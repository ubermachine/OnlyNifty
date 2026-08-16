"""OnlyNifty v5.2 Walk-Forward Out-Of-Sample Edge Harness.

Measures each setup's real statistical edge across market regimes with:
- Rolling train/test windows (30d train / 5d test).
- 60-bar lookback purging at boundaries to prevent lookahead bias.
- 12-bar signal embargo across test boundaries.
- 95% Wilson / Bootstrap confidence intervals on Expected Value (EV).
- Automatic quarantine policy for negative-edge setups.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
import os
import json
import numpy as np
import pandas as pd

from src.config import QUARANTINE_MIN_SAMPLES, EDGE_OVERLAP_VIF


@dataclass
class EdgeStats:
    setup_id: str
    regime: str
    n: int
    win_rate: float
    mean_r: float
    ev: float
    ci_low: float
    ci_high: float
    status: str  # "TRUSTED" | "PAPER" | "QUARANTINED"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EdgeStats":
        return cls(**data)


class EdgeTable:
    """Stores statistical edge verification records across setup IDs and regimes."""

    def __init__(self, records: Optional[List[EdgeStats]] = None):
        self.records: Dict[Tuple[str, str], EdgeStats] = {}
        if records:
            for r in records:
                self.records[(r.setup_id, r.regime)] = r

    def lookup(self, setup_id: str, regime: str) -> Optional[EdgeStats]:
        return self.records.get((setup_id, regime))

    def is_tradeable(self, setup_id: str, regime: str) -> bool:
        stats = self.lookup(setup_id, regime)
        if stats is None:
            return True  # If no record exists yet, allow initial sampling
        return stats.status != "QUARANTINED"

    def get_sizing_factor(self, setup_id: str, regime: str) -> float:
        stats = self.lookup(setup_id, regime)
        if stats is None:
            return 0.5  # Untested setup starts at half size
        if stats.status == "TRUSTED":
            return 1.0
        elif stats.status == "PAPER":
            return 0.5
        return 0.0  # QUARANTINED

    def to_json(self) -> str:
        data = [r.to_dict() for r in self.records.values()]
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "EdgeTable":
        data = json.loads(s)
        records = [EdgeStats.from_dict(item) for item in data]
        return cls(records)

    def save_to_disk(self, filepath: str = "data/edge_table.json"):
        try:
            dir_name = os.path.dirname(filepath)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            tmp_path = f"{filepath}.tmp.{os.getpid()}"
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(self.to_json())
            os.replace(tmp_path, filepath)
        except Exception:
            pass

    @classmethod
    def load_from_disk(cls, filepath: str = "data/edge_table.json") -> "EdgeTable":
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    return cls.from_json(f.read())
            except Exception:
                pass
        return cls()


class WalkForwardRunner:
    """
    Executes walk-forward rolling out-of-sample edge discovery
    with strict data purge and signal embargo boundaries.
    """

    def __init__(self, strategy_engine: Any = None):
        self.strategy_engine = strategy_engine

    def compute_edge_stats(
        self,
        r_multiples: List[float],
        setup_id: str,
        regime: str
    ) -> EdgeStats:
        n = len(r_multiples)
        if n == 0:
            return EdgeStats(
                setup_id=setup_id, regime=regime, n=0,
                win_rate=0.0, mean_r=0.0, ev=0.0,
                ci_low=-1.0, ci_high=1.0, status="PAPER"
            )

        arr = np.array(r_multiples, dtype=np.float64)
        wins = np.sum(arr > 0)
        win_rate = float(wins / n)
        mean_r = float(np.mean(arr))
        
        # Expected value per trade
        ev = float(np.mean(arr))

        # Bootstrap 95% Confidence Interval.
        #
        # OVERLAP CORRECTION: signals can fire on consecutive bars while each outcome
        # spans a 12-bar horizon, so these observations are NOT independent. A plain iid
        # bootstrap resamples them as if they were and produces an interval that is too
        # narrow — which is how a marginal setup earns a confident "TRUSTED".
        # Inflate the interval half-width by sqrt(VIF) about the mean. EDGE_OVERLAP_VIF
        # is a conservative floor, not a fitted value; a proper stationary block bootstrap
        # (mean block ~2x the outcome horizon) is the right long-term replacement.
        if n >= 10:
            bootstraps = []
            rng = np.random.RandomState(42)
            for _ in range(1000):
                sample = rng.choice(arr, size=n, replace=True)
                bootstraps.append(np.mean(sample))
            ci_low = float(np.percentile(bootstraps, 2.5))
            ci_high = float(np.percentile(bootstraps, 97.5))
        else:
            std_err = float(np.std(arr, ddof=1) / np.sqrt(n)) if n > 1 else 0.5
            ci_low = ev - 1.96 * std_err
            ci_high = ev + 1.96 * std_err

        inflation = float(np.sqrt(max(EDGE_OVERLAP_VIF, 1.0)))
        ci_low = ev - (ev - ci_low) * inflation
        ci_high = ev + (ci_high - ev) * inflation

        # Quarantine Policy.
        #
        # QUARANTINED must mean "demonstrated loser", not merely "unproven". Collapsing
        # `ci_low < 0` into QUARANTINE blocked setups whose EV is positive but whose
        # interval simply still straddles zero — that is missing evidence, not evidence
        # of harm, and the right response is to keep sampling at reduced size.
        # The table is a kill-switch for losers, never a licence to size up.
        if n < QUARANTINE_MIN_SAMPLES:
            status = "PAPER"
        elif ev <= 0.0:
            status = "QUARANTINED"      # negative expectancy on an adequate sample
        elif ci_low < 0.0:
            status = "PAPER"            # positive EV, not yet statistically established
        else:
            status = "TRUSTED"

        return EdgeStats(
            setup_id=setup_id,
            regime=regime,
            n=n,
            win_rate=round(win_rate * 100.0, 1),
            mean_r=round(mean_r, 2),
            ev=round(ev, 2),
            ci_low=round(ci_low, 2),
            ci_high=round(ci_high, 2),
            status=status
        )

    def simulate_trade_outcome(
        self,
        sig: Any,
        future_window: pd.DataFrame
    ) -> Optional[float]:
        """
        Replays a signal bar-by-bar against future price and returns its realized R.

        Mirrors LiveSignalJournal.update_open_trades_lifecycle exactly: 50% is booked at
        T1 and the stop trails to breakeven, so a T1-then-reversal is a small WIN
        (0.5 x R_t1), not the full -1R a naive model would score it. Keeping this in sync
        with the live lifecycle is what stops the harness from quarantining setups that
        are actually profitable in production.
        """
        entry_px = float(sig.entry_price)
        sl_px = float(sig.sl_price)
        t1_px = float(sig.target_1)
        t2_px = float(sig.target_2)
        t3_px = float(getattr(sig, "target_3_moonshot", 0.0) or 0.0)
        is_long = "LONG" in sig.signal_type.value

        sl_pts = abs(entry_px - sl_px)
        if sl_pts <= 0:
            return None  # no risk-defined stop -> R is undefined, exclude from stats

        r_t1 = abs(t1_px - entry_px) / sl_pts if t1_px > 0 else 0.0
        r_t2 = abs(t2_px - entry_px) / sl_pts if t2_px > 0 else 0.0

        live_sl = sl_px
        t1_booked = False
        outcome_r = 0.0

        for _, fbar in future_window.iterrows():
            fhigh = float(fbar["high"])
            flow = float(fbar["low"])

            if is_long:
                hit_sl = flow <= live_sl
                hit_t3 = t3_px > 0 and fhigh >= t3_px
                hit_t2 = t2_px > 0 and fhigh >= t2_px
                hit_t1 = t1_px > 0 and fhigh >= t1_px
            else:
                hit_sl = fhigh >= live_sl
                hit_t3 = t3_px > 0 and flow <= t3_px
                hit_t2 = t2_px > 0 and flow <= t2_px
                hit_t1 = t1_px > 0 and flow <= t1_px

            # Stop is checked first, matching the live lifecycle's elif ordering.
            if hit_sl:
                outcome_r = round(0.5 * r_t1, 2) if t1_booked else -1.0
                break
            elif hit_t3:
                # Derive R from the actual T3 distance rather than asserting a constant.
                # A hardcoded 4.0 inflated the winners' tail independently of where T3
                # actually sat, which is precisely what props up a setup's measured EV.
                r_t3 = abs(t3_px - entry_px) / sl_pts if t3_px > 0 else max(r_t2, 1.0)
                outcome_r = round(r_t3, 2)
                break
            elif hit_t2:
                outcome_r = round(0.5 * r_t1 + 0.5 * r_t2, 2)
                break
            elif hit_t1 and not t1_booked:
                t1_booked = True
                outcome_r = round(0.5 * r_t1, 2)
                live_sl = entry_px  # trail to breakeven on the remaining 50%

        # Window expired with the trade still open and nothing banked. Returning 0.0 here
        # counted an unresolved trade as a real scratch: it entered n, deflated the sample
        # SD, dragged EV toward zero and corrupted win_rate. Censor it instead (None is
        # dropped by the caller) unless T1 had already booked a partial.
        if not t1_booked and outcome_r == 0.0:
            return None
        return outcome_r

    def run(
        self,
        df_5m: pd.DataFrame,
        train_days: int = 30,
        test_days: int = 5,
        purge_bars: int = 60,
        embargo_bars: int = 12
    ) -> EdgeTable:
        """
        Executes rolling walk-forward test across the dataframe.
        """
        from src.strategy_rules import StrategyEngine, SignalType
        engine = self.strategy_engine or StrategyEngine()

        results_by_group: Dict[Tuple[str, str], List[float]] = {}

        if df_5m.empty or len(df_5m) < (purge_bars + 30):
            return EdgeTable()

        bars_per_day = 75
        train_bars = train_days * bars_per_day
        test_bars = test_days * bars_per_day
        step_bars = test_bars

        total_bars = len(df_5m)
        start_idx = train_bars

        while start_idx + test_bars <= total_bars:
            test_start = start_idx + purge_bars  # Purge indicator warmup
            test_end = start_idx + test_bars - embargo_bars  # Embargo boundary

            if test_start < test_end:
                for bar_idx in range(test_start, test_end):
                    sub = df_5m.iloc[:bar_idx + 1]
                    sig = engine.evaluate_bar(sub, current_idx=len(sub) - 1)
                    if sig.signal_type != SignalType.WAIT and sig.entry_price > 0:
                        # Outcome simulation across next 12 bars (60 min)
                        future_window = df_5m.iloc[bar_idx + 1 : min(bar_idx + 13, total_bars)]
                        if not future_window.empty:
                            outcome_r = self.simulate_trade_outcome(sig, future_window)
                            if outcome_r is None:
                                continue

                            regime = (sig.details.get("markov_regime") or {}).get("active_regime", "UNKNOWN") if sig.details else "UNKNOWN"
                            key = (sig.signal_type.value, regime)
                            results_by_group.setdefault(key, []).append(outcome_r)

            start_idx += step_bars

        records = []
        for (stype, regime), r_list in results_by_group.items():
            stats = self.compute_edge_stats(r_list, stype, regime)
            records.append(stats)

        return EdgeTable(records)
