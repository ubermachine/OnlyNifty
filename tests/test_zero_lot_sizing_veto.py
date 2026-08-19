"""A "READY" ticket can still size to 0 lots (Kelly risk budget below this option's
per-lot stop cost — deep-ITM premiums make risk_per_lot large, and lunch-lull / event-risk
sizing caps can compound it below one lot).

Found analyzing two real losing trades on 2026-08-19: both had total_qty=0 and
capital_risk_rupees=0.0 yet were journaled as executed TRIGGERED trades. The lifecycle
tracker then fell back to a fabricated floor quantity (25 shares) to compute a PnL for a
position that was never sized, and capital_risk_rupees=0.0 made
`realized_r_multiple = pnl / max(capital_risk_rupees, 1.0)` collapse into raw rupees
mislabeled as R — corrupting the daily "Net Realized R-Multiple" aggregate
(-1281.25 + -1141.75 summed to a nonsensical "-2423.00R").

backtest_engine.py already guards on `ticket.get("lots", 0) > 0`; log_signal must match it.
"""

from src.signal_journal import LiveSignalJournal
from src.strategy_rules import Signal, SignalType


def _short_signal():
    return Signal(SignalType.SHORT, entry_price=24095.45, sl_price=24145.45,
                  target_1=24065.45, target_2=24032.95, target_3_moonshot=23994.33)


def _quote_ticket(lots):
    t = {
        "status": "READY", "symbol": "NSE:NIFTY26AUG24450PE", "strike": 24450,
        "option_type": "PE", "pricing_source": "MARKET_QUOTE",
        "entry_premium": 341.65, "sl_premium": 290.4,
        "target1_premium": 392.9, "target2_premium": 444.15, "target3_moonshot_premium": 521.03,
    }
    if lots is not None:
        t["lots"] = lots
        t["total_qty"] = lots * 75
        t["actual_risk_rupees"] = round((341.65 - 290.4) * lots * 75, 2)
    return t


class TestZeroLotVeto:
    def test_zero_lots_is_not_journaled_as_executed(self):
        journal = LiveSignalJournal(persistence_file=None)
        entry = journal.log_signal(_short_signal(), _quote_ticket(lots=0), 24095.45,
                                    bar_timestamp="2026-08-19 09:30")
        assert entry is not None
        assert entry.direction == "WAIT"
        assert entry.lifecycle_status == "AWAITING_SETUP"
        assert entry.selected_strike == 0
        assert entry.total_qty == 0
        assert entry.capital_risk_rupees == 0.0
        assert "0 lots" in entry.trigger_reason

    def test_zero_lots_excluded_from_daily_stats(self):
        journal = LiveSignalJournal(persistence_file=None)
        journal.log_signal(_short_signal(), _quote_ticket(lots=0), 24095.45,
                           bar_timestamp="2026-08-19 09:30")
        summary = journal.compute_daily_journal_summary(target_date="2026-08-19")
        # Must never be counted as an actionable/closed/losing trade.
        assert summary["actionable_trades"] == 0
        assert summary["short_trades"] == 0
        assert summary["losing_trades"] == 0
        assert summary["total_r_multiple"] == 0.0

    def test_nonzero_lots_still_journaled_as_executed(self):
        journal = LiveSignalJournal(persistence_file=None)
        entry = journal.log_signal(_short_signal(), _quote_ticket(lots=5), 24095.45,
                                    bar_timestamp="2026-08-19 09:30")
        assert entry is not None
        assert entry.direction == "SHORT"
        assert entry.lifecycle_status == "TRIGGERED"
        assert entry.total_qty == 375

    def test_lots_key_absent_preserves_legacy_behavior(self):
        # Callers/tests that never populate "lots" at all (e.g. minimal WAIT-adjacent
        # fixtures) must not be treated as a zero-lot veto — only an EXPLICIT lots<=0
        # from a real sizing computation should veto.
        journal = LiveSignalJournal(persistence_file=None)
        entry = journal.log_signal(_short_signal(), _quote_ticket(lots=None), 24095.45,
                                    bar_timestamp="2026-08-19 09:30")
        assert entry is not None
        assert entry.direction == "SHORT"
        assert entry.lifecycle_status == "TRIGGERED"
