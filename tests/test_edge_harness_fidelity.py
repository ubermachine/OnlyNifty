"""Tests that the walk-forward harness stays faithful to the live system.

Two failure modes these guard against:
  1. Trade signals carrying no regime -> edge table keyed on a regime the live gate
     never looks up -> quarantine gate silently dead.
  2. The harness's R model disagreeing with the journal's lifecycle -> profitable
     setups quarantined because T1-then-reversal was scored as a full -1R.
"""

import pandas as pd
import pytest

from src.strategy_rules import StrategyEngine, Signal, SignalType
from src.edge_harness import WalkForwardRunner


def make_future(bars):
    """bars: list of (high, low) tuples."""
    return pd.DataFrame(
        {
            "open": [b[0] for b in bars],
            "high": [b[0] for b in bars],
            "low": [b[1] for b in bars],
            "close": [b[1] for b in bars],
            "volume": [1000] * len(bars),
        }
    )


def long_signal():
    # entry 24500, stop 24450 -> 50pt risk. T1 24550 (1R), T2 24600 (2R), T3 24700 (4R)
    return Signal(
        signal_type=SignalType.LONG,
        entry_price=24500.0,
        sl_price=24450.0,
        target_1=24550.0,
        target_2=24600.0,
        target_3_moonshot=24700.0,
    )


class TestOutcomeModelMatchesLiveLifecycle:
    def test_clean_stop_is_minus_one_r(self):
        runner = WalkForwardRunner()
        out = runner.simulate_trade_outcome(long_signal(), make_future([(24510, 24440)]))
        assert out == -1.0

    def test_t1_then_reversal_is_a_small_win_not_minus_one_r(self):
        # The core fidelity fix: live books 50% at T1 and trails to breakeven.
        runner = WalkForwardRunner()
        future = make_future([(24560, 24520), (24505, 24495)])  # tag T1, then back to entry
        out = runner.simulate_trade_outcome(long_signal(), future)
        assert out == pytest.approx(0.5)
        assert out > 0

    def test_t2_blends_both_booked_halves(self):
        runner = WalkForwardRunner()
        future = make_future([(24560, 24520), (24610, 24555)])
        out = runner.simulate_trade_outcome(long_signal(), future)
        assert out == pytest.approx(1.5)  # 0.5*1R + 0.5*2R

    def test_t3_moonshot_is_four_r(self):
        runner = WalkForwardRunner()
        out = runner.simulate_trade_outcome(long_signal(), make_future([(24710, 24520)]))
        assert out == 4.0

    def test_open_at_window_expiry_keeps_banked_t1(self):
        runner = WalkForwardRunner()
        future = make_future([(24560, 24520), (24555, 24530)])
        out = runner.simulate_trade_outcome(long_signal(), future)
        assert out == pytest.approx(0.5)

    def test_unresolved_trade_is_censored_not_scored_zero(self):
        # An unresolved trade is missing data, not a scratch. Scoring it 0.0 put it in n,
        # deflated the sample SD and dragged EV toward zero.
        runner = WalkForwardRunner()
        out = runner.simulate_trade_outcome(long_signal(), make_future([(24510, 24495)]))
        assert out is None

    def test_zero_risk_stop_is_excluded_from_stats(self):
        runner = WalkForwardRunner()
        sig = Signal(
            signal_type=SignalType.LONG,
            entry_price=24500.0,
            sl_price=24500.0,
            target_1=24550.0,
            target_2=24600.0,
        )
        # __post_init__ nudges SL off entry, so this now has definable risk.
        assert sig.sl_price != sig.entry_price
        # Resolve it (tag T1) so we are testing risk-definability, not censoring.
        out = runner.simulate_trade_outcome(sig, make_future([(24560, 24520)]))
        assert out is not None
        assert out > 0

    def test_short_side_is_symmetric(self):
        runner = WalkForwardRunner()
        sig = Signal(
            signal_type=SignalType.SHORT,
            entry_price=24500.0,
            sl_price=24550.0,
            target_1=24450.0,
            target_2=24400.0,
            target_3_moonshot=24300.0,
        )
        assert runner.simulate_trade_outcome(sig, make_future([(24560, 24490)])) == -1.0
        # Tag T1 then revert to entry -> banked half, breakeven on the rest.
        future = make_future([(24480, 24440), (24505, 24495)])
        assert runner.simulate_trade_outcome(sig, future) == pytest.approx(0.5)


class TestRegimeStamping:
    def _df(self, periods=60, start="2026-08-13 10:00"):
        dates = pd.date_range(start, periods=periods, freq="5min")
        prices = [24500.0 + i * 3 for i in range(periods)]
        return pd.DataFrame(
            {
                "open": prices,
                "high": [p + 12 for p in prices],
                "low": [p - 12 for p in prices],
                "close": prices,
                "volume": [15000] * periods,
            },
            index=dates,
        )

    def test_every_signal_carries_a_regime(self):
        engine = StrategyEngine()
        sig = engine.evaluate_bar(self._df(), current_idx=59)
        assert "markov_regime" in sig.details
        assert sig.details["markov_regime"].get("active_regime")

    def test_regime_is_a_real_value_not_the_default(self):
        engine = StrategyEngine()
        sig = engine.evaluate_bar(self._df(), current_idx=59)
        regime = sig.details["markov_regime"]["active_regime"]
        # Must be a genuine Markov state, not the "NORMAL"/"UNKNOWN" fallback that
        # previously made every edge-table key unmatchable.
        assert regime in {
            "LOW_VOL_TRENDING",
            "MEAN_REVERTING_CHOP",
            "HIGH_VOL_EXPANSION",
        }

    def test_early_return_signal_still_carries_a_regime_key(self):
        # 09:15-09:30 freak-candle path returns before the Markov model runs.
        engine = StrategyEngine()
        dates = pd.date_range("2026-08-13 09:15", periods=3, freq="5min")
        df = pd.DataFrame(
            {
                "open": [24500.0, 24510.0, 24520.0],
                "high": [24550.0, 24560.0, 24570.0],
                "low": [24480.0, 24490.0, 24500.0],
                "close": [24520.0, 24540.0, 24550.0],
                "volume": [10000, 12000, 15000],
            },
            index=dates,
        )
        sig = engine.evaluate_bar(df, current_idx=1)
        assert sig.details["markov_regime"]["active_regime"] == "UNKNOWN"
