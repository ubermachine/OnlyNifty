"""Perishable premium harvester + measured signal conviction — QUOTE-tier edge loop."""

import os
import tempfile
from datetime import datetime, timedelta

import pandas as pd
import pytz
import pytest

from src.premium_harvester import PremiumHarvester
from src.edge_harness import replay_option_quote_series, EdgeTable, EdgeStats
from src.signal_edge import SignalEdgeWeighter

IST = pytz.timezone("Asia/Kolkata")


def _tmp_harvester():
    d = tempfile.mkdtemp()
    return PremiumHarvester(
        pending_path=os.path.join(d, "pend.jsonl"),
        resolved_path=os.path.join(d, "res.jsonl"),
    )


def _quote_ticket(**over):
    t = dict(
        status="READY", fyers_symbol="NSE:NIFTY2681824300CE", pricing_source="MARKET_QUOTE",
        signal="LONG", entry_premium=100.0, sl_premium=85.0,
        target1_premium=118.0, target2_premium=140.0, target3_moonshot_premium=170.0,
        strike=24300, option_type="CE", spot_entry=24300.0,
    )
    t.update(over)
    return t


def _series(prems):
    base = pd.Timestamp("2026-08-18 09:15").tz_localize(IST)
    idx = pd.DatetimeIndex([base + timedelta(minutes=5 * i) for i in range(len(prems))])
    return pd.DataFrame([dict(open=p, high=p + 3, low=p - 3, close=p, volume=1000) for p in prems], index=idx)


class TestSharedResolver:
    def test_win_books_partial_then_target(self):
        # crosses T1(118) and T2(140): 0.5*r_t1 + 0.5*r_t2
        r = replay_option_quote_series(_series([100, 120, 145]), 100, 85, 118, 140, 170)
        assert r is not None and r > 0

    def test_stop_is_minus_one(self):
        r = replay_option_quote_series(_series([100, 95, 80]), 100, 85, 118, 140, 170)
        assert r == -1.0

    def test_intrabar_stop_closer_to_open_wins(self):
        # a single bar spanning both stop and target: open near stop -> stop resolves
        base = pd.Timestamp("2026-08-18 09:15").tz_localize(IST)
        bar = pd.DataFrame([dict(open=86, high=125, low=80, close=90, volume=1)], index=[base])
        r = replay_option_quote_series(bar, 100, 85, 118, 140, 170)
        assert r == -1.0


class TestCapture:
    def test_only_live_quote_tickets_captured(self):
        h = _tmp_harvester()
        assert h.capture(_quote_ticket(), "LONG", "R", "LONG", "2026-08-18 10:00") is not None
        # theoretical ticket has no resolvable contract -> rejected
        assert h.capture(_quote_ticket(pricing_source="THEORETICAL_MODEL", fyers_symbol=""),
                         "LONG", "R", "LONG", "2026-08-18 10:00") is None

    def test_dedup_same_contract_same_bar(self):
        h = _tmp_harvester()
        h.capture(_quote_ticket(), "LONG", "R", "LONG", "2026-08-18 10:00")
        assert h.capture(_quote_ticket(), "LONG", "R", "LONG", "2026-08-18 10:00") is None


class TestResolveAndAggregate:
    def test_resolve_from_live_fetch(self):
        h = _tmp_harvester()
        h.capture(_quote_ticket(), "LONG", "LOW_VOL_TRENDING", "LONG", "2026-08-18 10:00",
                  expiry_epoch=int(pd.Timestamp("2026-08-18 15:30").tz_localize(IST).timestamp()))

        def fetch(sym, res, rf, rt):
            return _series([95, 98, 100, 105, 112, 120, 145, 130, 150, 160, 120, 110])

        now = pd.Timestamp("2026-08-18 15:45").tz_localize(IST).to_pydatetime()
        newly = h.resolve(fetch_history_fn=fetch, now=now)
        assert len(newly) == 1 and newly[0].outcome_r is not None
        et = h.build_edge_table()
        rec = et.lookup("LONG", "LOW_VOL_TRENDING")
        assert rec is not None and rec.evidence_tier == "QUOTE"

    def test_archive_resolves_after_expiry_when_live_dead(self):
        h = _tmp_harvester()

        class FakeArchive:
            def load_window(self, exp, strike, ot, start_ts, bars=0):
                ts = pd.Timestamp(start_ts)
                # full-session series so bars exist strictly after the entry bar
                s = _series([100, 100, 100, 105, 112, 120, 145, 130, 150, 160, 120, 110])
                return s[s.index > ts]

        h.capture(_quote_ticket(), "LONG", "R", "LONG", "2026-08-18 09:15",
                  expiry_epoch=int(pd.Timestamp("2026-08-18 15:30").tz_localize(IST).timestamp()),
                  expiry_label="2026-08-18")

        def dead(*a, **k):
            raise RuntimeError("Invalid symbol provided")  # expired contract

        now = datetime(2026, 8, 25, 12, 0, tzinfo=IST)  # a week after expiry
        newly = h.resolve(fetch_history_fn=dead, archive=FakeArchive(), now=now)
        assert len(newly) == 1 and newly[0].outcome_r is not None


class TestMeasuredConviction:
    def _stat(self, sid, reg, n, ev, ci, tier, status):
        return EdgeStats(sid, reg, n, 60.0, ev, ev, ci, ci + 0.5, status, tier)

    def test_quote_outranks_rosier_spot(self):
        spot = EdgeTable([self._stat("LONG", "R", 100, 0.6, 0.3, "SPOT", "TRUSTED")])
        quote = EdgeTable([self._stat("LONG", "R", 35, 0.4, 0.1, "QUOTE", "TRUSTED")])
        c = SignalEdgeWeighter(spot, quote).conviction_for("LONG", "R")
        assert c.evidence_tier == "QUOTE"

    def test_spot_capped_at_b(self):
        w = SignalEdgeWeighter(EdgeTable([self._stat("X", "R", 200, 1.0, 0.8, "SPOT", "TRUSTED")]))
        c = w.conviction_for("X", "R")
        assert c.conviction <= 0.75 and c.grade == "B"

    def test_quarantined_zero_size(self):
        w = SignalEdgeWeighter(EdgeTable([self._stat("Z", "R", 85, -0.4, -0.6, "QUOTE", "QUARANTINED")]))
        assert w.conviction_for("Z", "R").conviction == 0.0

    def test_unmeasured_half_size(self):
        assert SignalEdgeWeighter(EdgeTable()).conviction_for("NEW", "R").conviction == 0.5
