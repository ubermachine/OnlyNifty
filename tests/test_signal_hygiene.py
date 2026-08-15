"""Tests for Signal target/stop hygiene and journal structural deduplication.

Covers the Phase 4 §7.2.3 (structure_epoch dedup) and §7.2.4 (stop/target hygiene)
guarantees, which previously had fields and config constants but no enforcement.
"""

import pytest

from src.strategy_rules import Signal, SignalType
from src.signal_journal import LiveSignalJournal, SignalLifecycleStatus


READY_TICKET = {
    "status": "READY",
    "strike": 24500,
    "entry_premium": 150.0,
    "sl_premium": 120.0,
    "target1_premium": 195.0,
    "target2_premium": 240.0,
    "actual_risk_rupees": 5000.0,
}


class TestSignalTargetHygiene:
    def test_long_inverted_moonshot_is_corrected_beyond_t2(self):
        sig = Signal(
            signal_type=SignalType.LONG,
            entry_price=24500.0,
            sl_price=24460.0,
            target_1=24560.0,
            target_2=24620.0,
            target_3_moonshot=24580.0,  # inverted: inside T2
        )
        assert sig.target_3_moonshot > sig.target_2

    def test_short_inverted_moonshot_is_corrected_beyond_t2(self):
        sig = Signal(
            signal_type=SignalType.SHORT,
            entry_price=24500.0,
            sl_price=24540.0,
            target_1=24440.0,
            target_2=24380.0,
            target_3_moonshot=24420.0,  # inverted: above T2 on a short
        )
        assert sig.target_3_moonshot < sig.target_2

    def test_valid_moonshot_is_left_untouched(self):
        sig = Signal(
            signal_type=SignalType.LONG,
            entry_price=24500.0,
            sl_price=24460.0,
            target_1=24560.0,
            target_2=24620.0,
            target_3_moonshot=24700.0,
        )
        assert sig.target_3_moonshot == 24700.0

    def test_zero_moonshot_still_autofills(self):
        sig = Signal(
            signal_type=SignalType.LONG,
            entry_price=24500.0,
            sl_price=24460.0,
            target_1=24560.0,
            target_2=24620.0,
            target_3_moonshot=0.0,
        )
        assert sig.target_3_moonshot > sig.target_2

    def test_wait_signal_targets_are_not_rewritten(self):
        sig = Signal(
            signal_type=SignalType.WAIT,
            entry_price=24500.0,
            sl_price=0.0,
            target_1=0.0,
            target_2=0.0,
            target_3_moonshot=0.0,
        )
        assert sig.target_3_moonshot == 0.0


class TestSignalStopHygiene:
    def test_long_sl_equal_to_entry_is_nudged_below(self):
        sig = Signal(
            signal_type=SignalType.LONG,
            entry_price=24500.0,
            sl_price=24500.0,  # not a stop at all
            target_1=24560.0,
            target_2=24620.0,
        )
        assert sig.sl_price < sig.entry_price

    def test_short_sl_equal_to_entry_is_nudged_above(self):
        sig = Signal(
            signal_type=SignalType.SHORT,
            entry_price=24500.0,
            sl_price=24500.0,
            target_1=24440.0,
            target_2=24380.0,
        )
        assert sig.sl_price > sig.entry_price

    def test_valid_stop_is_left_untouched(self):
        sig = Signal(
            signal_type=SignalType.LONG,
            entry_price=24500.0,
            sl_price=24460.0,
            target_1=24560.0,
            target_2=24620.0,
        )
        assert sig.sl_price == 24460.0


class TestStructureEpochDedup:
    def test_same_setup_instance_across_bars_logs_once(self):
        journal = LiveSignalJournal(persistence_file=None)
        sig = Signal(SignalType.LONG, 24500.0, 24460.0, 24560.0, 24620.0, 24680.0)

        first = journal.log_signal(sig, READY_TICKET, 24500.0, bar_timestamp="2026-08-15 10:00")
        # Same structural setup, later bar — previously logged a fresh duplicate entry.
        second = journal.log_signal(sig, READY_TICKET, 24502.0, bar_timestamp="2026-08-15 10:05")

        assert first is not None
        assert second is None
        assert len(journal.entries) == 1

    def test_materially_different_setup_logs_a_new_entry(self):
        journal = LiveSignalJournal(persistence_file=None)
        sig_a = Signal(SignalType.LONG, 24500.0, 24460.0, 24560.0, 24620.0, 24680.0)
        sig_b = Signal(SignalType.LONG, 24700.0, 24650.0, 24760.0, 24820.0, 24880.0)

        first = journal.log_signal(sig_a, READY_TICKET, 24500.0, bar_timestamp="2026-08-15 10:00")
        second = journal.log_signal(sig_b, READY_TICKET, 24700.0, bar_timestamp="2026-08-15 11:00")

        assert first is not None
        assert second is not None
        assert len(journal.entries) == 2

    def test_opposite_direction_is_not_deduped(self):
        journal = LiveSignalJournal(persistence_file=None)
        sig_long = Signal(SignalType.LONG, 24500.0, 24460.0, 24560.0, 24620.0, 24680.0)
        sig_short = Signal(SignalType.SHORT, 24500.0, 24540.0, 24440.0, 24380.0, 24320.0)

        journal.log_signal(sig_long, READY_TICKET, 24500.0, bar_timestamp="2026-08-15 10:00")
        second = journal.log_signal(sig_short, READY_TICKET, 24500.0, bar_timestamp="2026-08-15 10:05")

        assert second is not None
        assert len(journal.entries) == 2

    def test_same_structure_may_retrigger_after_the_trade_closes(self):
        # Dedup suppresses re-logging while a trade is OPEN; once it closes the same
        # structure is a legitimate new trade (re-entry pacing is the cooldown's job).
        journal = LiveSignalJournal(persistence_file=None)
        sig = Signal(SignalType.LONG, 24500.0, 24460.0, 24560.0, 24620.0, 24680.0)

        first = journal.log_signal(sig, READY_TICKET, 24500.0, bar_timestamp="2026-08-15 10:00")
        assert first is not None

        blocked = journal.log_signal(sig, READY_TICKET, 24500.0, bar_timestamp="2026-08-15 10:05")
        assert blocked is None

        # Close the trade out, then the same structure may fire again.
        first.lifecycle_status = SignalLifecycleStatus.STOPPED_OUT.value
        reentry = journal.log_signal(sig, READY_TICKET, 24500.0, bar_timestamp="2026-08-15 11:00")
        assert reentry is not None
        assert len(journal.entries) == 2

    def test_explicit_structure_epoch_is_respected(self):
        journal = LiveSignalJournal(persistence_file=None)
        sig_a = Signal(SignalType.LONG, 24500.0, 24460.0, 24560.0, 24620.0, 24680.0)
        sig_b = Signal(SignalType.LONG, 24900.0, 24850.0, 24960.0, 25020.0, 25080.0)

        journal.log_signal(
            sig_a, READY_TICKET, 24500.0, bar_timestamp="2026-08-15 10:00", structure_epoch="EPOCH-1"
        )
        # Different price structure but caller declares the same epoch -> collapsed.
        second = journal.log_signal(
            sig_b, READY_TICKET, 24900.0, bar_timestamp="2026-08-15 11:00", structure_epoch="EPOCH-1"
        )

        assert second is None
        assert len(journal.entries) == 1
