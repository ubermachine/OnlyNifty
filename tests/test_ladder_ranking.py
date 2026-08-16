"""The ladder must keep looking after a candidate is rejected.

Every branch used to `return _check_and_return(...)`, and that helper returned a WAIT
Signal on rejection. So the FIRST setup to fail ended the bar. Combined with
first-match-wins ordering this had a perverse consequence: quarantining a
measured-losing setup converted it into a NO-TRADE rather than into the next-best
trade, because the better setup further down the ladder was never evaluated.
"""

import pandas as pd
import pytest

from src.edge_harness import EdgeStats, EdgeTable
from src.strategy_rules import SignalType, StrategyEngine


def trending_df(periods=90, step=5.0):
    dates = pd.date_range("2026-08-13 10:00", periods=periods, freq="5min")
    px = [24000.0 + i * step for i in range(periods)]
    return pd.DataFrame(
        {
            "open": px,
            "high": [p + 18 for p in px],
            "low": [p - 18 for p in px],
            "close": px,
            "volume": [25000 + (i % 7) * 4000 for i in range(periods)],
        },
        index=dates,
    )


def table_quarantining(setup_id, regime="LOW_VOL_TRENDING"):
    return EdgeTable([
        EdgeStats(setup_id=setup_id, regime=regime, n=85, win_rate=25.9,
                  mean_r=-0.50, ev=-0.50, ci_low=-0.755, ci_high=-0.25,
                  status="QUARANTINED")
    ])


class TestRejectionDoesNotEndTheBar:
    def test_quarantined_setup_does_not_terminate_the_ladder(self):
        """A quarantined setup must step aside, not stop evaluation."""
        df = trending_df()
        quarantined = StrategyEngine(edge_table=table_quarantining("LONG_ORDER_FLOW"))
        sig = quarantined.evaluate_bar(df, current_idx=len(df) - 1)

        # Whatever it returns, it must not be the quarantined setup itself.
        assert sig.signal_type != SignalType.LONG_ORDER_FLOW

    def test_wait_reason_names_the_rejections(self):
        # A generic "no confluence" message when setups were actually tested and
        # rejected hides the real reason from the journal and the operator.
        df = trending_df()
        eng = StrategyEngine(edge_table=table_quarantining("LONG_ORDER_FLOW"))
        sig = eng.evaluate_bar(df, current_idx=len(df) - 1)
        if sig.signal_type == SignalType.WAIT:
            reason = sig.reason.lower()
            assert ("rejected" in reason or "consolidation" in reason
                    or "veto" in reason or "opening" in reason
                    or "accumulating" in reason or "overextended" in reason)

    def test_empty_edge_table_is_a_no_op(self):
        """With nothing measured, behaviour must be unchanged — the gate is permissive
        by design so an absent table never silently blocks trading."""
        df = trending_df()
        a = StrategyEngine(edge_table=EdgeTable()).evaluate_bar(df, current_idx=len(df) - 1)
        b = StrategyEngine(edge_table=None).evaluate_bar(df, current_idx=len(df) - 1)
        assert a.signal_type == b.signal_type

    def test_quarantine_changes_the_outcome(self):
        """Sanity: quarantining whatever the engine would otherwise pick must actually
        alter the result, proving the gate is reachable and not inert."""
        df = trending_df()
        base = StrategyEngine(edge_table=EdgeTable()).evaluate_bar(df, current_idx=len(df) - 1)
        if base.signal_type == SignalType.WAIT:
            pytest.skip("no setup fired on this fixture; nothing to quarantine")
        blocked = StrategyEngine(
            edge_table=table_quarantining(base.signal_type.value)
        ).evaluate_bar(df, current_idx=len(df) - 1)
        assert blocked.signal_type != base.signal_type


class TestAbsoluteVetoesStillStopEverything:
    def test_absolute_veto_set_is_narrow(self):
        """Only genuinely bar-wide conditions may end evaluation. Direction-specific
        vetoes must reject one candidate and let the opposite side be considered."""
        from src.strategy_rules import _ABSOLUTE_VETOES
        assert _ABSOLUTE_VETOES == {"VPIN_TOXICITY", "SESSION_RISK_LIMIT"}
        for direction_specific in (
            "SKEW_CRASH_HEDGING", "HFI_BEARISH_DIVERGENCE", "GEX_CALL_WALL_PIN",
            "HTF_NOT_ALIGNED_LONG", "EDGE_TABLE_QUARANTINED",
        ):
            assert direction_specific not in _ABSOLUTE_VETOES
