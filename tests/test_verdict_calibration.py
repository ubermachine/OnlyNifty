"""The verdict score must report what it has been worth — and admit when it knows nothing.

Measured basis: the confluence score carries no information over the range we could test
(non-overlapping n=50, r=-0.0047, t=-0.03). But live signals score 78-100 and that band has
never been measured at all. So the requirement is not "ignore the score" — it is "never let
an unmeasured score imply an endorsement".
"""

import pytest

from src.verdict_calibration import (
    VerdictCalibrator,
    Calibration,
    band_for,
    family_attribution,
    CALIBRATED_MIN_N,
    SPARSE_MIN_N,
)


def entry(score, r, direction="LONG", status="STOPPED_OUT", votes=None):
    return {
        "confluence_score": score, "realized_r_multiple": r, "direction": direction,
        "lifecycle_status": status, "family_votes": votes or {},
    }


class TestBanding:
    @pytest.mark.parametrize("score,band", [(30, "<45"), (50, "45-55"), (60, "55-70"),
                                            (78, "70-85"), (100, "85+")])
    def test_band_boundaries(self, score, band):
        assert band_for(score) == band


class TestHonestyAboutIgnorance:
    def test_unmeasured_band_is_uncalibrated_not_endorsed(self):
        # Only low-band trades exist; the live 78-100 band has nothing.
        c = VerdictCalibrator([(50.0, -1.0)] * 40)
        hi = c.calibrate(90.0)
        assert hi.status == "UNCALIBRATED"
        assert hi.n == 0
        assert hi.is_actionable is False
        assert "never been measured" in hi.note

    def test_empty_calibrator_never_claims_knowledge(self):
        c = VerdictCalibrator([])
        for s in (10, 50, 78, 100):
            assert c.calibrate(s).status == "UNCALIBRATED"
            assert c.calibrate(s).is_actionable is False

    def test_sparse_band_is_flagged_not_trusted(self):
        c = VerdictCalibrator([(60.0, 0.5)] * (SPARSE_MIN_N + 2))
        r = c.calibrate(60.0)
        assert r.status == "SPARSE"
        assert r.is_actionable is False
        assert "not established" in r.note

    def test_calibrated_only_past_threshold(self):
        c = VerdictCalibrator([(60.0, 0.5)] * CALIBRATED_MIN_N)
        r = c.calibrate(60.0)
        assert r.status == "CALIBRATED"
        assert r.is_actionable is True


class TestMeasurement:
    def test_reports_realized_not_asserted(self):
        # A band full of losers must report as a loser regardless of a high score.
        c = VerdictCalibrator([(80.0, -1.0)] * CALIBRATED_MIN_N)
        r = c.calibrate(80.0)
        assert r.mean_r == pytest.approx(-1.0)
        assert r.win_rate == 0.0
        assert r.status == "CALIBRATED"

    def test_from_entries_excludes_open_trades(self):
        rows = [entry(60, 1.0, status="STOPPED_OUT"), entry(60, 99.0, status="ACTIVE"),
                entry(60, 99.0, status="TRIGGERED")]
        c = VerdictCalibrator.from_entries(rows)
        assert c.calibrate(60).n == 1  # open trades have no outcome yet

    def test_from_entries_excludes_wait_rows(self):
        rows = [entry(60, 1.0), entry(0, 0.0, direction="WAIT", status="AWAITING_SETUP")]
        assert VerdictCalibrator.from_entries(rows).calibrate(60).n == 1


class TestFamilyAttribution:
    def test_measures_each_pillar_separately(self):
        rows = []
        # "flow" votes with direction and wins; "macro" votes against and the trade still wins
        for _ in range(SPARSE_MIN_N + 1):
            rows.append(entry(60, 1.0, votes={"flow": 1, "macro": -1}))
        for _ in range(SPARSE_MIN_N + 1):
            rows.append(entry(60, -1.0, votes={"flow": -1, "macro": 1}))
        att = family_attribution(rows)
        assert att["flow"]["n_with"] > 0 and att["flow"]["n_against"] > 0
        # flow tracked direction correctly -> positive edge
        assert att["flow"]["edge"] > 0
        # macro was inverted -> negative edge
        assert att["macro"]["edge"] < 0

    def test_absent_votes_yield_uncalibrated(self):
        att = family_attribution([entry(60, 1.0, votes={})])
        assert all(v["status"] == "UNCALIBRATED" for v in att.values())


class TestCorruptDataRejection:
    """The journal contains rows at -256.55R and -88.46R from the zero-lot bug
    (R computed as raw rupees / a 0.0 risk floored to 1.0). Ingesting them would poison
    the exact numbers this module exists to make honest."""

    def test_impossible_r_values_are_rejected(self):
        good = [(80.0, 1.0)] * 12
        corrupt = [(80.0, -256.55), (80.0, -88.46), (80.0, 1281.25)]
        c = VerdictCalibrator(good + corrupt)
        assert c.rejected_corrupt == 3
        cal = c.calibrate(80.0)
        assert cal.n == 12
        assert cal.mean_r == pytest.approx(1.0)   # untainted by the corrupt rows

    def test_rejection_is_surfaced_not_silent(self):
        c = VerdictCalibrator([(80.0, 1.0)] * 5 + [(80.0, -999.0)])
        assert "corrupt rows rejected" in c.coverage_summary()

    def test_legitimate_extremes_survive(self):
        c = VerdictCalibrator([(80.0, 3.5)] * 12)   # a real T3 moonshot
        assert c.rejected_corrupt == 0
        assert c.calibrate(80.0).mean_r == pytest.approx(3.5)
