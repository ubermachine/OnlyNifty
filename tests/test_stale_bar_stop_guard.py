"""A trade must never be resolved against a bar that predates its own entry.

Real incident, 2026-08-19 (cost: two fabricated losses + a locked session):
The 5m candle frame lagged the live quote. Entries were taken off a live spot at 09:30
and 12:01, but df.iloc[-1] was still the 09:15 OPENING bar, whose high (24172.85) was the
high of the day. update_open_trades_lifecycle checked both stops against that high and
marked both trades STOPPED_OUT — on a price that occurred BEFORE either trade existed and
was never revisited afterwards.

Ground truth from settled 5m bars that day:
  Trade 1  SHORT @24095.45  SL 24145.45  T1 24065.45
           real max high after entry = 24108.45  -> stop NOT touched
           real min low  after entry = 24065.10  -> T1 WAS reached (a winner)
  Trade 2  SHORT @24072.65  SL 24087.65
           real max high after entry = 24082.20  -> stop NOT touched

Both were recorded as losses, which then tripped the 2-strike circuit breaker and blocked
trading for the remainder of the session.
"""

from src.signal_journal import LiveSignalJournal
from src.strategy_rules import Signal, SignalType


OPENING_BAR_HIGH = 24172.85   # 09:15 bar high = day's high


def _short_trade(journal, entry_bar, spot, sl, t1):
    sig = Signal(SignalType.SHORT, entry_price=spot, sl_price=sl,
                 target_1=t1, target_2=t1 - 30.0, target_3_moonshot=t1 - 60.0)
    ticket = {
        "status": "READY", "symbol": "NSE:NIFTY26AUG24450PE", "strike": 24450,
        "option_type": "PE", "pricing_source": "MARKET_QUOTE", "lots": 5, "total_qty": 375,
        "actual_risk_rupees": 19218.75,
        "entry_premium": 341.65, "sl_premium": 290.40,
        "target1_premium": 392.90, "target2_premium": 444.15,
        "target3_moonshot_premium": 521.03,
    }
    return journal.log_signal(sig, ticket, spot, bar_timestamp=entry_bar)


class TestStaleBarStopGuard:
    def test_stop_not_triggered_by_bar_preceding_entry(self):
        """The exact 2026-08-19 trade-1 scenario."""
        j = LiveSignalJournal(persistence_file=None)
        e = _short_trade(j, "2026-08-19 09:30", 24095.45, 24145.45, 24065.45)
        assert e.lifecycle_status == "TRIGGERED"

        # Frame is stale: last candle is still the 09:15 opening bar (high = day's high).
        j.update_open_trades_lifecycle(
            current_spot=24118.70, current_high=OPENING_BAR_HIGH,
            current_low=24102.50, bar_time_str="09:15", current_open=24152.05,
        )
        assert e.lifecycle_status == "TRIGGERED", (
            "trade was resolved against the 09:15 opening bar, which precedes its 09:30 entry"
        )
        assert e.realized_pnl_rupees == 0.0

    def test_real_forward_bar_reaches_t1(self):
        """Once the frame catches up, the true post-entry bars are honoured -> T1, not a stop."""
        j = LiveSignalJournal(persistence_file=None)
        e = _short_trade(j, "2026-08-19 09:30", 24095.45, 24145.45, 24065.45)
        j.update_open_trades_lifecycle(
            current_spot=24118.70, current_high=OPENING_BAR_HIGH,
            current_low=24102.50, bar_time_str="09:15", current_open=24152.05,
        )
        # 09:35 bar: high 24080.05, low 24065.10 -> reaches T1 (24065.45), never the stop.
        j.update_open_trades_lifecycle(
            current_spot=24079.55, current_high=24080.05,
            current_low=24065.10, bar_time_str="09:35", current_open=24076.40,
        )
        assert e.lifecycle_status != "STOPPED_OUT"
        assert e.touched_t1 is True

    def test_trade_two_scenario_also_survives(self):
        """Trade 2: stale 09:15 high must not stop a 12:01 entry."""
        j = LiveSignalJournal(persistence_file=None)
        e = _short_trade(j, "2026-08-19 12:01", 24072.65, 24087.65, 24042.65)
        j.update_open_trades_lifecycle(
            current_spot=24118.70, current_high=OPENING_BAR_HIGH,
            current_low=24102.50, bar_time_str="09:15", current_open=24152.05,
        )
        assert e.lifecycle_status == "TRIGGERED"

    def test_same_bar_as_entry_still_evaluates(self):
        """Guard blocks only EARLIER bars; the entry bar itself must still resolve."""
        j = LiveSignalJournal(persistence_file=None)
        e = _short_trade(j, "2026-08-19 09:30", 24095.45, 24145.45, 24065.45)
        j.update_open_trades_lifecycle(
            current_spot=24160.0, current_high=24150.0,
            current_low=24090.0, bar_time_str="09:30", current_open=24095.0,
        )
        assert e.lifecycle_status == "STOPPED_OUT"
