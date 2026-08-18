import pytest
import os
import json
from src.cloud_storage import (
    get_database_url,
    init_db,
    upsert_signal,
    upsert_signals_batch,
    fetch_signals_by_date,
    fetch_recent_signals,
    sync_local_archive_to_cloud
)

def test_database_url_retrieval():
    url = get_database_url()
    assert url is not None
    assert "neon.tech" in url or "postgresql" in url

def test_init_db_and_upsert_signal():
    url = get_database_url()
    if not url:
        pytest.skip("Neon DATABASE_URL not available")

    # Test single signal upsert
    test_sig = {
        "signal_id": "TEST-SIG-UNIT-TEST-001",
        "timestamp_ist": "2026-08-18 10:00:00 IST",
        "spot_price": 24500.0,
        "signal_type": "LONG_ORDER_FLOW",
        "direction": "LONG",
        "selected_strike": 24500,
        "option_type": "CE",
        "entry_premium": 150.0,
        "sl_spot": 24460.0,
        "target_1_spot": 24550.0,
        "confluence_score": 85.0,
        "lifecycle_status": "TRIGGERED",
        "realized_r_multiple": 0.0,
        "realized_pnl_rupees": 0.0,
        "record_hash": "TEST_HASH_12345",
        "prev_hash": "GENESIS_ROOT_HASH_0000000000000000"
    }

    ok = upsert_signal(test_sig)
    assert ok is True

    # Test fetch by date
    signals = fetch_signals_by_date("2026-08-18")
    assert isinstance(signals, list)
    matching = [s for s in signals if s.get("signal_id") == "TEST-SIG-UNIT-TEST-001"]
    assert len(matching) >= 1
    assert matching[0]["selected_strike"] == 24500

    # Test update status (upsert idempotency)
    test_sig["lifecycle_status"] = "T1_REACHED"
    test_sig["realized_r_multiple"] = 1.0
    test_sig["realized_pnl_rupees"] = 5000.0
    ok2 = upsert_signal(test_sig)
    assert ok2 is True

    # Verify update
    signals2 = fetch_signals_by_date("2026-08-18")
    updated = [s for s in signals2 if s.get("signal_id") == "TEST-SIG-UNIT-TEST-001"][0]
    assert updated["lifecycle_status"] == "T1_REACHED"
    assert updated["realized_r_multiple"] == 1.0

def test_fetch_recent_signals():
    url = get_database_url()
    if not url:
        pytest.skip("Neon DATABASE_URL not available")

    recent = fetch_recent_signals(limit=10)
    assert isinstance(recent, list)
    assert len(recent) >= 1
