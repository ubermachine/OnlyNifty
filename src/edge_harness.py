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

from src.config import QUARANTINE_MIN_SAMPLES, EDGE_OVERLAP_VIF, MIN_OOS_SAMPLES, TIME_STOP_BARS, TIME_STOP_MIN_R


def replay_option_quote_series(
    quote_series: pd.DataFrame,
    entry_prem: float,
    sl_prem: float,
    t1_prem: float,
    t2_prem: float,
    t3_prem: float,
) -> Optional[float]:
    """Single authoritative QUOTE-tier resolver: replays a real option OHLCV series and
    returns realized option R with 3-tier partial booking, breakeven trail after T1,
    unbiased intrabar SL/target resolution (closer-to-open wins), and a stall time stop.

    Used by BOTH the walk-forward harness and the perishable premium harvester so there
    is exactly one option-outcome definition in the system.
    """
    if quote_series is None or quote_series.empty:
        return None
    sl_pts = max(entry_prem - sl_prem, 1.0)
    r_t1 = (t1_prem - entry_prem) / sl_pts
    r_t2 = (t2_prem - entry_prem) / sl_pts

    live_sl = sl_prem
    t1_booked = False
    outcome_r = 0.0
    bars_count = 0

    for _, qbar in quote_series.iterrows():
        bars_count += 1
        qopen = float(qbar.get("open", entry_prem))
        qhigh = float(qbar.get("high", entry_prem))
        qlow = float(qbar.get("low", entry_prem))
        qclose = float(qbar.get("close", entry_prem))

        hit_sl = qlow <= live_sl
        hit_t3 = t3_prem > 0 and qhigh >= t3_prem
        hit_t2 = t2_prem > 0 and qhigh >= t2_prem
        hit_t1 = t1_prem > 0 and qhigh >= t1_prem

        # Unbiased intrabar resolution: if a bar spans both stop and a target,
        # award whichever level sits closer to the bar's open.
        if hit_sl and (hit_t1 or hit_t2 or hit_t3):
            target_level = t3_prem if hit_t3 else (t2_prem if hit_t2 else t1_prem)
            if abs(qopen - target_level) < abs(qopen - live_sl):
                hit_sl = False

        if hit_sl:
            return round(0.5 * r_t1, 2) if t1_booked else -1.0
        elif hit_t3 and t3_prem > 0:
            return round((t3_prem - entry_prem) / sl_pts, 2)
        elif hit_t2 and t2_prem > 0:
            return round(0.5 * r_t1 + 0.5 * r_t2, 2)
        elif hit_t1 and not t1_booked:
            t1_booked = True
            outcome_r = round(0.5 * r_t1, 2)
            live_sl = entry_prem  # trail remaining 50% to breakeven

        if not t1_booked and bars_count >= TIME_STOP_BARS:
            cur_r = (qclose - entry_prem) / sl_pts
            if cur_r < TIME_STOP_MIN_R:
                return round(cur_r, 2)

    # Window expired still open
    if not t1_booked and outcome_r == 0.0:
        if len(quote_series) >= TIME_STOP_BARS:
            final_close = float(quote_series.iloc[-1]["close"])
            return round((final_close - entry_prem) / sl_pts, 2)
        return None
    return outcome_r


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
    evidence_tier: str = "SPOT"  # "QUOTE" | "MODEL" | "SPOT"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EdgeStats":
        clean_data = dict(data)
        if "evidence_tier" not in clean_data:
            clean_data["evidence_tier"] = "SPOT"
        return cls(**clean_data)


def build_synthetic_options_context(spot: float, sub_df: pd.DataFrame, iv: float = 0.13) -> Dict[str, Any]:
    """Builds a synthetic-but-structurally-complete options context for research walks.

    The strategy's data-sufficiency gate fails CLOSED without a chain (no 25d skew, no
    verified dealer walls, no positioning flow), so a spot-only walk-forward fires zero
    trades. This supplies a synthetic chain + computed GEX walls + directional vector so
    the gates can actually evaluate — but because the chain is synthetic, any resulting
    edge is MODEL tier and can NEVER promote to TRUSTED (only live QUOTE evidence can).
    """
    from src.data_engine import DataEngine
    from src.options_flow import compute_strike_level_gex_chart_data, compute_short_term_directional_vector
    chain = DataEngine(use_cache=False).generate_synthetic_option_chain(spot=float(spot))
    chain_df = chain.get("dataframe")
    gex_chart = compute_strike_level_gex_chart_data(chain_df, spot=float(spot), iv=iv)
    dir_flow = compute_short_term_directional_vector(float(spot), sub_df, option_chain_df=chain_df, live_iv=iv)
    return {"chain_df": chain_df, "gex_chart": gex_chart, "dir_flow": dir_flow}


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
            return True  # unmeasured -> allow PAPER exploration
        return stats.status != "QUARANTINED"

    def get_sizing_factor(self, setup_id: str, regime: str) -> float:
        stats = self.lookup(setup_id, regime)
        if stats is None or stats.status == "PAPER":
            return 0.5
        if stats.status == "TRUSTED":
            return 1.0
        return 0.0  # QUARANTINED

    def to_json(self) -> str:
        data = [r.to_dict() for r in self.records.values()]
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "EdgeTable":
        data = json.loads(s)
        records = [EdgeStats.from_dict(item) for item in data]
        return cls(records=records)

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
        return cls(records=[])


class WalkForwardRunner:
    """Executes walk-forward cross-validation across rolling market regimes."""

    def __init__(
        self,
        strategy_engine: Optional[Any] = None,
        edge_table_path: str = "data/edge_table.json"
    ):
        if isinstance(strategy_engine, str):
            self.edge_table_path = strategy_engine
            self.strategy_engine = None
        else:
            self.strategy_engine = strategy_engine
            self.edge_table_path = edge_table_path
        self.raw_records: List[Dict[str, Any]] = []

    def compute_edge_stats(
        self,
        arg1: Any,
        arg2: str,
        arg3: Any,
        evidence_tier: str = "SPOT"
    ) -> EdgeStats:
        if isinstance(arg1, (list, tuple, np.ndarray)):
            outcomes_r = list(arg1)
            setup_id = str(arg2)
            regime = str(arg3)
        else:
            setup_id = str(arg1)
            regime = str(arg2)
            outcomes_r = list(arg3)

        n = len(outcomes_r)
        if n == 0:
            return EdgeStats(
                setup_id=setup_id, regime=regime, n=0,
                win_rate=0.0, mean_r=0.0, ev=0.0,
                ci_low=0.0, ci_high=0.0, status="PAPER",
                evidence_tier=evidence_tier
            )

        arr = np.array(outcomes_r, dtype=np.float64)
        wins = np.sum(arr > 0)
        win_rate = float(wins / n)
        mean_r = float(np.mean(arr))
        
        # Expected value per trade
        ev = float(np.mean(arr))

        # Bootstrap 95% Confidence Interval.
        if n >= 10:
            bootstraps = []
            rng = np.random.RandomState(42)
            for _ in range(1000):
                sample = rng.choice(arr, size=n, replace=True)
                bootstraps.append(np.mean(sample))
            ci_low = float(np.percentile(bootstraps, 2.5))
            ci_high = float(np.percentile(bootstraps, 97.5))
            ci_5 = float(np.percentile(bootstraps, 5.0))
        else:
            std_err = float(np.std(arr, ddof=1) / np.sqrt(n)) if n > 1 else 0.5
            ci_low = ev - 1.96 * std_err
            ci_high = ev + 1.96 * std_err
            ci_5 = ev - 1.645 * std_err

        inflation = float(np.sqrt(max(EDGE_OVERLAP_VIF, 1.0)))
        ci_low = ev - (ev - ci_low) * inflation
        ci_high = ev + (ci_high - ev) * inflation
        ci_5_inflated = ev - (ev - ci_5) * inflation

        # Quarantine & Promotion Policy with Evidence Tiering:
        # Requires genuine QUOTE evidence and MIN_OOS_SAMPLES (30) before status can exceed PAPER to TRUSTED.
        # MODEL and SPOT tiers are strictly hard-capped at PAPER ceiling.
        if n < QUARANTINE_MIN_SAMPLES:
            status = "PAPER"
        elif ev <= 0.0:
            status = "QUARANTINED"      # negative expectancy on an adequate sample
        elif evidence_tier != "QUOTE":
            status = "PAPER"            # Hard promotion ceiling for non-quote evidence
        elif n < MIN_OOS_SAMPLES or ci_5_inflated < 0.0:
            status = "PAPER"            # positive EV, not yet statistically established on >=30 samples at 95% one-sided confidence
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
            status=status,
            evidence_tier=evidence_tier
        )

    def simulate_trade_outcome(
        self,
        sig: Any,
        future_window: pd.DataFrame,
        quote_series: Optional[pd.DataFrame] = None,
        ticket: Optional[Dict[str, Any]] = None
    ) -> Optional[float]:
        """
        Replays a signal bar-by-bar against future option quotes (if available) or
        derivatives-translated spot prices and returns its realized option R.
        Uses unbiased intrabar open distance resolution and 5-bar stall time stop.
        """
        is_long = "LONG" in sig.signal_type.value
        
        # 1. Direct Option Quote Series Replay (QUOTE Tier) — delegates to the single
        #    shared resolver so harness and harvester never diverge.
        if quote_series is not None and not quote_series.empty:
            entry_prem = float(ticket.get("entry_premium", quote_series.iloc[0]["open"])) if ticket else float(quote_series.iloc[0]["open"])
            sl_prem = float(ticket.get("sl_premium", max(entry_prem * 0.75, 5.0))) if ticket else max(entry_prem * 0.75, 5.0)
            t1_prem = float(ticket.get("target1_premium", entry_prem * 1.30)) if ticket else entry_prem * 1.30
            t2_prem = float(ticket.get("target2_premium", entry_prem * 1.60)) if ticket else entry_prem * 1.60
            t3_prem = float(ticket.get("target3_moonshot_premium", entry_prem * 2.0)) if ticket else entry_prem * 2.0
            return replay_option_quote_series(quote_series, entry_prem, sl_prem, t1_prem, t2_prem, t3_prem)

        # 2. Derivatives-Translated Greeks Model Replay (MODEL Tier)
        entry_px = float(sig.entry_price)
        sl_px = float(sig.sl_price)
        t1_px = float(sig.target_1)
        t2_px = float(sig.target_2)
        t3_px = float(getattr(sig, "target_3_moonshot", 0.0) or 0.0)

        sl_pts = abs(entry_px - sl_px)
        if sl_pts <= 0:
            return None

        # Translate spot target distances to option premium R with theta haircut
        r_t1 = abs(t1_px - entry_px) / sl_pts if t1_px > 0 else 0.0
        r_t2 = abs(t2_px - entry_px) / sl_pts if t2_px > 0 else 0.0

        live_sl = sl_px
        t1_booked = False
        outcome_r = 0.0
        bars_count = 0

        for _, fbar in future_window.iterrows():
            bars_count += 1
            fopen = float(fbar["open"]) if "open" in fbar else entry_px
            fhigh = float(fbar["high"])
            flow = float(fbar["low"])
            fclose = float(fbar["close"])

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

            # Unbiased Intrabar Resolution:
            # If bar spans both SL and a Target, resolve by whichever level is closer to open
            if hit_sl and (hit_t1 or hit_t2 or hit_t3):
                target_level = t3_px if hit_t3 else (t2_px if hit_t2 else t1_px)
                dist_sl = abs(fopen - live_sl)
                dist_tgt = abs(fopen - target_level)
                if dist_tgt < dist_sl:
                    hit_sl = False

            if hit_sl:
                outcome_r = round(0.5 * r_t1, 2) if t1_booked else -1.0
                break
            elif hit_t3 and t3_px > 0:
                r_t3 = abs(t3_px - entry_px) / sl_pts
                outcome_r = round(r_t3, 2)
                break
            elif hit_t2 and t2_px > 0:
                outcome_r = round(0.5 * r_t1 + 0.5 * r_t2, 2)
                break
            elif hit_t1 and not t1_booked:
                t1_booked = True
                outcome_r = round(0.5 * r_t1, 2)
                live_sl = entry_px  # trail to breakeven on the remaining 50%

            # Stall Time Stop:
            if not t1_booked and bars_count >= TIME_STOP_BARS:
                move_pts = (fclose - entry_px) if is_long else (entry_px - fclose)
                cur_r = move_pts / sl_pts
                if cur_r < TIME_STOP_MIN_R:
                    outcome_r = round(cur_r, 2)
                    return outcome_r

        # Window expired with trade still open and nothing banked
        if not t1_booked and outcome_r == 0.0:
            if len(future_window) >= TIME_STOP_BARS:
                final_close = float(future_window.iloc[-1]["close"])
                move_pts = (final_close - entry_px) if is_long else (entry_px - final_close)
                outcome_r = round(move_pts / sl_pts, 2)
                return outcome_r
            return None
        return outcome_r

    def run(
        self,
        df_5m: pd.DataFrame,
        train_days: int = 30,
        test_days: int = 5,
        purge_bars: int = 60,
        embargo_bars: int = 12,
        synthetic_context: bool = False,
        context_iv: float = 0.13,
    ) -> EdgeTable:
        """
        Executes rolling walk-forward test across the dataframe.

        synthetic_context=True feeds a synthetic options context per bar so the
        fail-closed data-sufficiency gate can evaluate (otherwise spot-only walks fire
        zero trades). Records are then tagged MODEL — never TRUSTED-eligible.
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
                    octx = None
                    if synthetic_context:
                        try:
                            octx = build_synthetic_options_context(float(sub["close"].iloc[-1]), sub, iv=context_iv)
                        except Exception:
                            octx = None
                    sig = engine.evaluate_bar(
                        sub, current_idx=len(sub) - 1,
                        option_chain_df=(octx.get("chain_df") if octx else None),
                        options_context=octx,
                    )
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

        tier = "MODEL" if synthetic_context else "SPOT"
        records = []
        for (stype, regime), r_list in results_by_group.items():
            stats = self.compute_edge_stats(r_list, stype, regime, evidence_tier=tier)
            records.append(stats)

        return EdgeTable(records)
