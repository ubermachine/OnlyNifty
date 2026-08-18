import os
import glob
import pytest
from src.signal_journal import (
    LiveSignalJournal,
    SignalEntry,
    compute_sha256_record_hash,
    verify_chain_integrity,
    verify_archive_file
)

def test_archive_files_tamper_evident_integrity():
    archive_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "archive")
    if not os.path.exists(archive_dir):
        pytest.skip("No data/archive directory present")

    files = glob.glob(os.path.join(archive_dir, "*.jsonl"))
    if not files:
        pytest.skip("No .jsonl archive files to verify")

    for f in files:
        res = verify_archive_file(f)
        assert res["is_valid"] is True, f"Tamper verification failed for {f}: {res['errors']}"
        assert res["broken_links"] == 0
        assert res["content_mismatches"] == 0
        assert res["total_records"] > 0

def test_chain_tamper_detection():
    # Construct a synthetic 3-element chain
    last_hash = "GENESIS_ROOT_HASH_0000000000000000"
    entries = []
    for i in range(3):
        e = SignalEntry(
            signal_id=f"SIG-TEST-{i}",
            timestamp_ist=f"2026-08-18 10:{i:02d}:00 IST",
            timestamp_utc_ms=1000 + i,
            bar_timestamp=f"2026-08-18 10:{i:02d}",
            spot_price=24500.0 + i,
            signal_type="LONG_ORDER_FLOW",
            direction="LONG",
            trigger_reason="Test trigger",
            selected_strike=24500,
            option_type="CE",
            symbol="NIFTY24500CE",
            entry_premium=100.0,
            sl_spot=24450.0,
            sl_premium=80.0,
            sl_points_spot=50.0,
            sl_risk_premium_pts=20.0,
            target_1_spot=24550.0,
            target_1_premium=130.0,
            target_2_spot=24600.0,
            target_2_premium=160.0,
            target_3_spot=24650.0,
            target_3_premium=200.0,
            r_multiple_t1=1.0,
            r_multiple_t2=2.0,
            confluence_score=85.0,
            confluence_grade="A",
            regime_summary="Normal",
            kalman_velocity=1.0,
            kalman_zscore=0.5,
            markov_regime="Trend",
            htf_alignment="Aligned",
            is_0dte=False,
            lots_suggested=2,
            total_qty=50,
            capital_risk_rupees=5000.0,
            tca_friction_est=150.0,
            lifecycle_status="TRIGGERED",
            is_seed=False,
            setup_id="TEST",
            structure_epoch=0,
            gate_audit={},
            evidence={},
            greeks_snapshot={},
            notes="",
            prev_hash=last_hash,
            record_hash="",
            conviction_score=75.0,
            conviction_tier="HIGH",
            family_votes={},
            family_agreement=3,
            directional_score=0.5,
            schema_version=2
        )
        e.record_hash = compute_sha256_record_hash(last_hash, e.to_dict())
        last_hash = e.record_hash
        entries.append(e)

    # Valid chain check
    valid_res = verify_chain_integrity(entries)
    assert valid_res["is_valid"] is True
    assert valid_res["broken_links"] == 0
    assert valid_res["content_mismatches"] == 0

    # Tamper with content of record 1
    entries[1].spot_price = 99999.0
    tampered_res = verify_chain_integrity(entries)
    assert tampered_res["is_valid"] is False
    assert tampered_res["content_mismatches"] == 1
