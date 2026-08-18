"""
OnlyNifty v5.3 Cloud Storage Adapter — Serverless Neon PostgreSQL Sync.

Provides asynchronous, non-blocking cloud persistence for the institutional signal journal
and audit store. Enables zero-maintenance durability across Streamlit Community Cloud container
restarts while maintaining cryptographic tamper-evident SHA-256 hash chains.
"""

import os
import json
import glob
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("OnlyNifty.CloudStorage")
IST = timezone(timedelta(hours=5, minutes=30))

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="neon_sync")
_db_initialized = False


def get_database_url() -> Optional[str]:
    """Retrieves the PostgreSQL connection string from environment or Streamlit secrets."""
    # 1. Environment variable
    db_url = os.environ.get("DATABASE_URL") or os.environ.get("NEON_DATABASE_URL")
    if db_url:
        return db_url

    # 2. Streamlit secrets (if running inside Streamlit)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "DATABASE_URL" in st.secrets:
            return str(st.secrets["DATABASE_URL"])
        if hasattr(st, "secrets") and "connections" in st.secrets and "postgresql" in st.secrets["connections"]:
            return str(st.secrets["connections"]["postgresql"].get("url", ""))
    except Exception:
        pass

    # 3. Read .streamlit/secrets.toml directly from disk
    secrets_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".streamlit", "secrets.toml")
    if os.path.exists(secrets_path):
        try:
            with open(secrets_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("DATABASE_URL") and "=" in line:
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val:
                            return val
                    if line.startswith("url") and "=" in line:
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val and "postgresql" in val:
                            return val
        except Exception:
            pass

    return None


def get_connection():
    """Returns a psycopg2 connection to Neon PostgreSQL."""
    import psycopg2
    db_url = get_database_url()
    if not db_url:
        return None
    return psycopg2.connect(db_url, sslmode="require")


def init_db() -> bool:
    """Initializes the signals_journal_audit table and indices on Neon PostgreSQL."""
    global _db_initialized
    if _db_initialized:
        return True

    db_url = get_database_url()
    if not db_url:
        return False

    try:
        import psycopg2
        with psycopg2.connect(db_url, sslmode="require") as conn:
            with conn.cursor() as cur:
                cur.execute("""
                CREATE TABLE IF NOT EXISTS signals_journal_audit (
                    record_hash TEXT PRIMARY KEY,
                    signal_id TEXT,
                    timestamp_ist TEXT,
                    session_date TEXT,
                    spot_price DOUBLE PRECISION,
                    signal_type TEXT,
                    direction TEXT,
                    selected_strike INT,
                    option_type TEXT,
                    entry_premium DOUBLE PRECISION,
                    sl_spot DOUBLE PRECISION,
                    target_1_spot DOUBLE PRECISION,
                    confluence_score DOUBLE PRECISION,
                    lifecycle_status TEXT,
                    realized_r_multiple DOUBLE PRECISION DEFAULT 0.0,
                    realized_pnl_rupees DOUBLE PRECISION DEFAULT 0.0,
                    prev_hash TEXT,
                    payload JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_signals_session_date ON signals_journal_audit (session_date);
                CREATE INDEX IF NOT EXISTS idx_signals_signal_id ON signals_journal_audit (signal_id);
                CREATE INDEX IF NOT EXISTS idx_signals_created_at ON signals_journal_audit (created_at);
                """)
                conn.commit()
        _db_initialized = True
        logger.info("Neon PostgreSQL signals_journal_audit table initialized successfully.")
        return True
    except Exception as e:
        logger.warning(f"Could not initialize Neon PostgreSQL table: {e}")
        return False


def upsert_signal(entry_dict: Dict[str, Any]) -> bool:
    """Upserts a single SignalEntry record to Neon PostgreSQL synchronously."""
    if not entry_dict or not entry_dict.get("signal_id"):
        return False

    db_url = get_database_url()
    if not db_url:
        return False

    init_db()

    sig_id = entry_dict["signal_id"]
    ts_ist = entry_dict.get("timestamp_ist", "")
    sess_date = entry_dict.get("bar_timestamp", "")[:10] or (ts_ist[:10] if len(ts_ist) >= 10 else datetime.now(IST).strftime("%Y-%m-%d"))
    spot = float(entry_dict.get("spot_price", 0.0))
    sig_type = str(entry_dict.get("signal_type", ""))
    direction = str(entry_dict.get("direction", "WAIT"))
    strike = int(entry_dict.get("selected_strike", 0))
    opt_type = str(entry_dict.get("option_type", "N/A"))
    entry_prem = float(entry_dict.get("entry_premium", 0.0))
    sl_spot = float(entry_dict.get("sl_spot", 0.0))
    t1_spot = float(entry_dict.get("target_1_spot", 0.0))
    conf_score = float(entry_dict.get("confluence_score", 0.0))
    status = str(entry_dict.get("lifecycle_status", "TRIGGERED"))
    r_mult = float(entry_dict.get("realized_r_multiple", 0.0))
    pnl = float(entry_dict.get("realized_pnl_rupees", 0.0))
    rec_hash = str(entry_dict.get("record_hash", "")) or f"{sig_id}_{entry_dict.get('bar_timestamp', '')}"
    prev_hash = str(entry_dict.get("prev_hash", ""))
    payload_json = json.dumps(entry_dict, default=str)

    try:
        import psycopg2
        with psycopg2.connect(db_url, sslmode="require") as conn:
            with conn.cursor() as cur:
                cur.execute("""
                INSERT INTO signals_journal_audit (
                    record_hash, signal_id, timestamp_ist, session_date, spot_price, signal_type,
                    direction, selected_strike, option_type, entry_premium, sl_spot,
                    target_1_spot, confluence_score, lifecycle_status, realized_r_multiple,
                    realized_pnl_rupees, prev_hash, payload, updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
                )
                ON CONFLICT (record_hash) DO UPDATE SET
                    lifecycle_status = EXCLUDED.lifecycle_status,
                    realized_r_multiple = EXCLUDED.realized_r_multiple,
                    realized_pnl_rupees = EXCLUDED.realized_pnl_rupees,
                    payload = EXCLUDED.payload,
                    updated_at = NOW();
                """, (
                    rec_hash, sig_id, ts_ist, sess_date, spot, sig_type,
                    direction, strike, opt_type, entry_prem, sl_spot,
                    t1_spot, conf_score, status, r_mult,
                    pnl, prev_hash, payload_json
                ))
                conn.commit()
        return True
    except Exception as e:
        logger.warning(f"Failed to upsert signal {sig_id} to Neon: {e}")
        return False


def upsert_signal_async(entry_dict: Dict[str, Any]) -> None:
    """Non-blocking fire-and-forget sync to Neon PostgreSQL."""
    _executor.submit(upsert_signal, entry_dict)


def upsert_signals_batch(entries: List[Dict[str, Any]]) -> int:
    """Fast batch upsert of multiple SignalEntry records."""
    if not entries:
        return 0

    db_url = get_database_url()
    if not db_url:
        return 0

    init_db()
    count = 0

    try:
        import psycopg2
        with psycopg2.connect(db_url, sslmode="require") as conn:
            with conn.cursor() as cur:
                for entry_dict in entries:
                    sig_id = entry_dict.get("signal_id")
                    if not sig_id:
                        continue
                    ts_ist = entry_dict.get("timestamp_ist", "")
                    sess_date = entry_dict.get("bar_timestamp", "")[:10] or (ts_ist[:10] if len(ts_ist) >= 10 else datetime.now(IST).strftime("%Y-%m-%d"))
                    spot = float(entry_dict.get("spot_price", 0.0))
                    sig_type = str(entry_dict.get("signal_type", ""))
                    direction = str(entry_dict.get("direction", "WAIT"))
                    strike = int(entry_dict.get("selected_strike", 0))
                    opt_type = str(entry_dict.get("option_type", "N/A"))
                    entry_prem = float(entry_dict.get("entry_premium", 0.0))
                    sl_spot = float(entry_dict.get("sl_spot", 0.0))
                    t1_spot = float(entry_dict.get("target_1_spot", 0.0))
                    conf_score = float(entry_dict.get("confluence_score", 0.0))
                    status = str(entry_dict.get("lifecycle_status", "TRIGGERED"))
                    r_mult = float(entry_dict.get("realized_r_multiple", 0.0))
                    pnl = float(entry_dict.get("realized_pnl_rupees", 0.0))
                    rec_hash = str(entry_dict.get("record_hash", "")) or f"{sig_id}_{entry_dict.get('bar_timestamp', '')}"
                    prev_hash = str(entry_dict.get("prev_hash", ""))
                    payload_json = json.dumps(entry_dict, default=str)

                    cur.execute("""
                    INSERT INTO signals_journal_audit (
                        record_hash, signal_id, timestamp_ist, session_date, spot_price, signal_type,
                        direction, selected_strike, option_type, entry_premium, sl_spot,
                        target_1_spot, confluence_score, lifecycle_status, realized_r_multiple,
                        realized_pnl_rupees, prev_hash, payload, updated_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
                    )
                    ON CONFLICT (record_hash) DO UPDATE SET
                        lifecycle_status = EXCLUDED.lifecycle_status,
                        realized_r_multiple = EXCLUDED.realized_r_multiple,
                        realized_pnl_rupees = EXCLUDED.realized_pnl_rupees,
                        payload = EXCLUDED.payload,
                        updated_at = NOW();
                    """, (
                        rec_hash, sig_id, ts_ist, sess_date, spot, sig_type,
                        direction, strike, opt_type, entry_prem, sl_spot,
                        t1_spot, conf_score, status, r_mult,
                        pnl, prev_hash, payload_json
                    ))
                    count += 1
                conn.commit()
        return count
    except Exception as e:
        logger.warning(f"Batch upsert failed: {e}")
        return count


def fetch_signals_by_date(session_date: str) -> List[Dict[str, Any]]:
    """Loads all signals for a specific date from Neon PostgreSQL."""
    db_url = get_database_url()
    if not db_url:
        return []

    init_db()
    results = []
    try:
        import psycopg2
        with psycopg2.connect(db_url, sslmode="require") as conn:
            with conn.cursor() as cur:
                cur.execute("""
                SELECT payload FROM signals_journal_audit
                WHERE session_date = %s
                ORDER BY created_at ASC;
                """, (session_date,))
                rows = cur.fetchall()
                for r in rows:
                    if r[0]:
                        item = r[0] if isinstance(r[0], dict) else json.loads(r[0])
                        results.append(item)
    except Exception as e:
        logger.warning(f"Failed to fetch signals for {session_date}: {e}")
    return results


def fetch_recent_signals(limit: int = 100) -> List[Dict[str, Any]]:
    """Loads the most recent signals across all sessions from Neon PostgreSQL."""
    db_url = get_database_url()
    if not db_url:
        return []

    init_db()
    results = []
    try:
        import psycopg2
        with psycopg2.connect(db_url, sslmode="require") as conn:
            with conn.cursor() as cur:
                cur.execute("""
                SELECT payload FROM signals_journal_audit
                ORDER BY created_at DESC
                LIMIT %s;
                """, (limit,))
                rows = cur.fetchall()
                for r in rows:
                    if r[0]:
                        item = r[0] if isinstance(r[0], dict) else json.loads(r[0])
                        results.append(item)
    except Exception as e:
        logger.warning(f"Failed to fetch recent signals: {e}")
    return results


def sync_local_archive_to_cloud(archive_dir: str = "data/archive") -> int:
    """Scans local data/archive/*.jsonl files and backfills any un-synced entries into Neon PostgreSQL."""
    if not os.path.exists(archive_dir):
        return 0

    all_entries = []
    for filepath in glob.glob(os.path.join(archive_dir, "*.jsonl")):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        all_entries.append(json.loads(line))
        except Exception:
            pass

    if all_entries:
        synced = upsert_signals_batch(all_entries)
        logger.info(f"Synced {synced} historical archive entries to Neon PostgreSQL.")
        return synced
    return 0
