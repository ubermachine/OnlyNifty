"""
OnlyNifty v4.0 Live Institutional Signals Journal & Audit Store.

Features:
- Full-fidelity institutional trade signal data schema with dual timestamps (UTC epoch ms + IST).
- High-frequency auto-refresh deduplication (bar-timestamp and state-transition gating).
- Event-driven trade lifecycle state machine (TRIGGERED -> T1 -> T2 -> T3 -> STOPPED_OUT -> EOD).
- Real-time MFE (Max Favorable Excursion) & MAE (Max Adverse Excursion) tracking.
- Dual-tier persistence: st.session_state + Atomic Disk JSON/CSV synchronization.
- Institutional KPI analytics suite (Win Rate, Payoff, Realized R-Multiple, SQN, Net TCA PnL).
- Cryptographic SHA-256 tamper-evident chaining for regulatory/prop compliance.
"""

import os
import glob
import json
import uuid
import hashlib
import threading
from datetime import datetime, timezone, timedelta
from enum import Enum
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from src.strategy_rules import Signal, SignalType
from src.config import LOT_SIZE, TIME_STOP_BARS, TIME_STOP_MIN_R

IST = timezone(timedelta(hours=5, minutes=30))


class SignalLifecycleStatus(str, Enum):
    TRIGGERED = "TRIGGERED"
    ACTIVE = "ACTIVE"
    T1_REACHED = "T1_REACHED"
    T2_REACHED = "T2_REACHED"
    T3_MOONSHOT = "T3_MOONSHOT"
    STOPPED_OUT = "STOPPED_OUT"
    TIME_STOPPED = "TIME_STOPPED"
    EOD_SQUAREOFF = "EOD_SQUAREOFF"
    CANCELLED = "CANCELLED"


def compute_sha256_record_hash(prev_hash: str, entry_dict: Dict[str, Any]) -> str:
    """Computes a cryptographic SHA-256 fingerprint chained to the previous record hash."""
    clean_dict = {k: v for k, v in entry_dict.items() if k not in ("prev_hash", "record_hash")}
    serialized = json.dumps(clean_dict, sort_keys=True, default=str)
    payload = f"{prev_hash}|{serialized}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class SignalEntry:
    signal_id: str
    timestamp_ist: str
    timestamp_utc_ms: int
    bar_timestamp: str
    spot_price: float
    signal_type: str
    direction: str  # "LONG", "SHORT", "WAIT"
    trigger_reason: str
    selected_strike: int
    option_type: str  # "CE", "PE", "N/A"
    symbol: str
    entry_premium: float
    sl_spot: float
    sl_premium: float
    sl_points_spot: float
    sl_risk_premium_pts: float
    target_1_spot: float
    target_1_premium: float
    target_2_spot: float
    target_2_premium: float
    target_3_spot: float
    target_3_premium: float
    r_multiple_t1: float
    r_multiple_t2: float
    confluence_score: float
    confluence_grade: str  # "A+ Institutional", "A Standard", "B Tactical"
    regime_summary: str
    kalman_velocity: float
    kalman_zscore: float
    markov_regime: str
    htf_alignment: str
    is_0dte: bool
    lots_suggested: int
    total_qty: int
    capital_risk_rupees: float
    tca_friction_est: float
    lifecycle_status: str = SignalLifecycleStatus.TRIGGERED.value
    realized_r_multiple: float = 0.0
    realized_pnl_rupees: float = 0.0
    realized_pnl_net: float = 0.0
    is_seed: bool = False
    setup_id: str = ""
    structure_epoch: str = ""
    gate_audit: Dict[str, Any] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)
    exit_timestamp_ist: Optional[str] = None
    exit_spot: Optional[float] = None
    exit_premium: Optional[float] = None
    peak_favorable_excursion_pts: float = 0.0
    peak_adverse_excursion_pts: float = 0.0
    greeks_snapshot: Dict[str, float] = field(default_factory=dict)
    notes: str = ""
    prev_hash: str = ""
    record_hash: str = ""
    bars_held: int = 0
    last_counted_bar: str = ""
    touched_t1: bool = False
    touched_t2: bool = False
    touched_t3: bool = False
    conviction_score: float = 0.0
    conviction_tier: str = "LOW"
    family_votes: Dict[str, int] = field(default_factory=dict)
    family_agreement: int = 0
    directional_score: float = 0.0
    schema_version: int = 2

    def is_active(self) -> bool:
        return self.lifecycle_status in [
            SignalLifecycleStatus.TRIGGERED.value,
            SignalLifecycleStatus.ACTIVE.value,
            SignalLifecycleStatus.T1_REACHED.value,
            SignalLifecycleStatus.T2_REACHED.value
        ]

    def __getitem__(self, item: str) -> Any:
        if item == "status":
            return self.lifecycle_status
        if item == "is_actionable":
            return self.signal_type != "WAIT"
        if item == "lots":
            return self.lots_suggested
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(item)

    def __setitem__(self, key: str, value: Any):
        if key == "status":
            self.lifecycle_status = value
        elif hasattr(self, key):
            setattr(self, key, value)
        else:
            setattr(self, key, value)

    def __contains__(self, item: str) -> bool:
        return hasattr(self, item) or item in ["status", "is_actionable"]

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SignalEntry":
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


def calculate_confluence_score(
    signal_obj: Any,
    df_5m: pd.DataFrame,
    htf_data: Optional[Dict[str, Any]] = None,
    kalman_vel: float = 0.0,
    kalman_z: float = 0.0,
    regime_state: Optional[Dict[str, Any]] = None,
    ofi_data: Optional[Dict[str, Any]] = None,
    gex_data: Optional[Dict[str, Any]] = None,
    vol_profile: Optional[Dict[str, Any]] = None,
    options_context: Optional[Dict[str, Any]] = None
) -> Tuple[float, str]:
    """Computes a rigorous 0-100 Institutional Confluence Score."""
    if signal_obj is None or getattr(signal_obj, "signal_type", None) is None:
        return 0.0, "Consolidation"

    sig_type = signal_obj.signal_type.value if hasattr(signal_obj.signal_type, "value") else str(signal_obj.signal_type)
    if "WAIT" in sig_type:
        return 0.0, "Consolidation"

    is_long = "LONG" in sig_type
    close = float(df_5m.iloc[-1]["close"]) if not df_5m.empty else 24500.0
    ema200 = float(df_5m["ema200"].iloc[-1]) if "ema200" in df_5m.columns else close
    ema55 = float(df_5m["ema55"].iloc[-1]) if "ema55" in df_5m.columns else close
    vwap = float(df_5m["vwap"].iloc[-1]) if "vwap" in df_5m.columns else close

    score = 0.0

    # 1. Higher Timeframe Confluence (20 Pts)
    if htf_data:
        if is_long and htf_data.get("htf_aligned_long", False):
            score += 20.0
        elif not is_long and htf_data.get("htf_aligned_short", False):
            score += 20.0
        elif htf_data.get("confluence_regime") in ["STRONG_BULLISH", "STRONG_BEARISH"]:
            score += 12.0
        else:
            score += 6.0
    else:
        score += 10.0

    # 2. Macro Referee & Intermediate EMA Filter (15 Pts)
    if is_long:
        if close > ema200: score += 10.0
        if close > ema55: score += 5.0
    else:
        if close < ema200: score += 10.0
        if close < ema55: score += 5.0

    # 3. Session AVWAP & Value Area (15 Pts)
    if vol_profile:
        vah = vol_profile.get("vah", close + 50.0)
        val = vol_profile.get("val", close - 50.0)
        if is_long:
            if close >= vwap: score += 10.0
            if close >= val: score += 5.0
        else:
            if close <= vwap: score += 10.0
            if close <= vah: score += 5.0
    else:
        if is_long and close >= vwap: score += 12.0
        elif not is_long and close <= vwap: score += 12.0

    # 4. Latent Kalman Velocity & Momentum (15 Pts)
    if is_long and kalman_vel > 0:
        score += 10.0
        if kalman_z >= 1.0: score += 5.0
    elif not is_long and kalman_vel < 0:
        score += 10.0
        if kalman_z <= -1.0: score += 5.0

    # 5. Order Flow Imbalance & Stacked Footprint (15 Pts)
    if ofi_data:
        if is_long and ofi_data.get("buyer_defense", False):
            score += 15.0
        elif not is_long and ofi_data.get("seller_defense", False):
            score += 15.0
        elif ofi_data.get("ofi_zscore", 0.0) != 0.0:
            score += 8.0
        
        # Hawkes Process OFI surge bonus
        if ofi_data.get("is_hawkes_surge", False):
            score += 5.0
    else:
        score += 8.0

    # 6. Fibonacci Golden Pocket / Trigger Setup (10 Pts)
    fib_ret = getattr(signal_obj, "fib_retracement", 0.0)
    if 0.45 <= fib_ret <= 0.65:
        score += 10.0
    elif "3PM" in sig_type or "ORDER_FLOW" in sig_type or "RANGE_FADE" in sig_type or "GAMMA_BREAKOUT" in sig_type:
        score += 10.0
    else:
        score += 5.0

    # 7. Dealer GEX & Markov Regime Alignment (5 Pts)
    if regime_state and "Trend" in str(regime_state.get("active_regime", "")):
        score += 3.0
    if regime_state and regime_state.get("is_rough_volatility", False) and ("BREAKOUT" in sig_type or "3PM" in sig_type):
        # Rough volatility fractal persistence accelerates breakout momentum
        score += 5.0
    if gex_data:
        if gex_data.get("is_positive_gamma", True):
            score += 2.0
        elif "BREAKOUT" in sig_type:
            # Dealer Short Gamma boosts breakout momentum (+10% Gamma Squeeze edge)
            score += 10.0

    # 8. Options Desk Positioning Votes (Up to +20 Pts Bonus when Verified)
    if options_context:
        dir_flow = options_context.get("dir_flow", {})
        d_vec = float(dir_flow.get("directional_vector", 0.0))
        if is_long and d_vec >= 0.2:
            score += 10.0
        elif not is_long and d_vec <= -0.2:
            score += 10.0
        elif (is_long and d_vec <= -0.2) or (not is_long and d_vec >= 0.2):
            score -= 10.0

        pcr_mom = float(dir_flow.get("sub_scores", {}).get("pcr_momentum", 0.0))
        if (is_long and pcr_mom > 0) or (not is_long and pcr_mom < 0):
            score += 5.0

        # DWV (Delta-Weighted Volume) Flow Alignment
        dwv = float(options_context.get("dwv_score", 0.0))
        if (is_long and dwv > 0.15) or (not is_long and dwv < -0.15):
            score += 5.0

        max_p = float(options_context.get("pcr", {}).get("max_pain_strike", close))
        if (is_long and close < max_p) or (not is_long and close > max_p):
            score += 5.0

    final_score = min(max(round(score, 1), 0.0), 100.0)

    if final_score >= 75.0:
        grade = "A+ Institutional"
    elif final_score >= 55.0:
        grade = "A Standard"
    elif final_score >= 45.0:
        grade = "B Tactical"
    else:
        grade = "C Weak / Vetoed"

    return final_score, grade


class LiveSignalJournal:
    """
    Thread-safe and deduplicated daily trading signal journal.
    Tracks every institutional signal generated during the session with execution details,
    BSM Greeks, target tranches, risk budgets, and automated lifecycle outcome tracking.
    """

    def __init__(
        self,
        persistence_file: Optional[str] = "data/signals_journal_today.json",
        archive_dir: Optional[str] = "data/archive"
    ):
        self.persistence_file = persistence_file
        self.archive_dir = archive_dir
        self.entries: List[SignalEntry] = []
        self._last_hash: str = "GENESIS_ROOT_HASH_0000000000000000"
        self._pending_lifecycle_events: List[Tuple[SignalEntry, str, float, float]] = []
        self._lifecycle_lock = threading.Lock()
        self._cleanup_orphaned_temps()
        self.reload_from_disk()

    def _cleanup_orphaned_temps(self) -> None:
        """Unlinks any stale atomic-write temp files older than 60 seconds."""
        if not self.persistence_file:
            return
        try:
            import time
            now = time.time()
            parent_dir = os.path.dirname(self.persistence_file) or "."
            pattern = os.path.join(parent_dir, "*.tmp.*")
            for tmp_file in glob.glob(pattern):
                try:
                    if (now - os.path.getmtime(tmp_file)) > 60:
                        os.unlink(tmp_file)
                except Exception:
                    pass
        except Exception:
            pass

    def _archive_previous_days(self) -> None:
        """
        Flushes entries belonging to previous calendar dates into immutable JSONL daily archives
        (data/archive/signals-YYYY-MM-DD.jsonl) and retains only today's session entries in active memory.
        Uses bar_timestamp as the authoritative session date and record_hash as unique dedup key.
        """
        if not self.entries or not self.archive_dir:
            return

        today_ist = datetime.now(IST).strftime("%Y-%m-%d")
        by_date: Dict[str, List[SignalEntry]] = {}
        today_entries: List[SignalEntry] = []

        for e in self.entries:
            # Authoritative session date: bar_timestamp first, fallback to timestamp_ist
            e_date = None
            if e.bar_timestamp and len(e.bar_timestamp) >= 10:
                e_date = e.bar_timestamp[:10]
            elif e.timestamp_ist and len(e.timestamp_ist) >= 10:
                e_date = e.timestamp_ist[:10]

            if e_date and e_date < today_ist:
                by_date.setdefault(e_date, []).append(e)
            else:
                today_entries.append(e)

        if not by_date:
            return

        try:
            os.makedirs(self.archive_dir, exist_ok=True)
            for date_str, old_entries in by_date.items():
                archive_file = os.path.join(self.archive_dir, f"signals-{date_str}.jsonl")
                
                # Content-addressed deduplication by record_hash
                existing_hashes = set()
                if os.path.exists(archive_file):
                    try:
                        with open(archive_file, "r", encoding="utf-8") as f:
                            for line in f:
                                if line.strip():
                                    rec = json.loads(line)
                                    h = rec.get("record_hash") or f"{rec.get('signal_id')}_{rec.get('bar_timestamp')}"
                                    existing_hashes.add(h)
                    except Exception:
                        pass

                with open(archive_file, "a", encoding="utf-8") as f:
                    for entry in old_entries:
                        h = entry.record_hash or f"{entry.signal_id}_{entry.bar_timestamp}"
                        if h not in existing_hashes:
                            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
                            existing_hashes.add(h)

            self.entries = today_entries
            self._persist_to_disk()
        except Exception:
            pass

    def reload_from_disk(self) -> None:
        """Reloads entries from disk persistence file if present and archives previous days."""
        if self.persistence_file and os.path.exists(self.persistence_file):
            try:
                with open(self.persistence_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    self.entries = [SignalEntry.from_dict(item) for item in raw]
                if self.entries:
                    self._last_hash = self.entries[-1].record_hash or self._last_hash
                self._archive_previous_days()
            except Exception:
                self.entries = []

        # If local disk has no entries (e.g. fresh ephemeral container on Streamlit Cloud), restore from Neon
        if not self.entries:
            try:
                today_ist = datetime.now(IST).strftime("%Y-%m-%d")
                from src.cloud_storage import fetch_signals_by_date
                cloud_records = fetch_signals_by_date(today_ist)
                if cloud_records:
                    self.entries = [SignalEntry.from_dict(item) for item in cloud_records]
                    if self.entries:
                        self._last_hash = self.entries[-1].record_hash or self._last_hash
                    self._persist_to_disk()
            except Exception:
                pass

    def log_signal(
        self,
        signal: Signal,
        ticket: Dict[str, Any],
        current_spot: float,
        bar_timestamp: Optional[str] = None,
        regime_info: Optional[Dict[str, Any]] = None,
        confluence_score: float = 1.0,
        htf_data: Optional[Dict[str, Any]] = None,
        kalman_vel: float = 0.0,
        kalman_z: float = 0.0,
        ofi_data: Optional[Dict[str, Any]] = None,
        gex_data: Optional[Dict[str, Any]] = None,
        vol_profile: Optional[Dict[str, Any]] = None,
        df_context: Optional[pd.DataFrame] = None,
        is_0dte: bool = False,
        is_seed: bool = False,
        setup_id: str = "",
        structure_epoch: str = "",
        gate_audit: Optional[Dict[str, Any]] = None,
        evidence: Optional[Dict[str, Any]] = None,
        options_context: Optional[Dict[str, Any]] = None
    ) -> Optional[SignalEntry]:
        """Logs a generated signal with deduplication and state-transition filtering."""
        now = datetime.now(timezone.utc)
        utc_ms = int(now.timestamp() * 1000)
        now_ist = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
        bar_time_str = str(bar_timestamp) if bar_timestamp else now_ist[:16]
        sig_type_str = signal.signal_type.value

        is_actionable = signal.signal_type != SignalType.WAIT and ticket.get("status") == "READY"
        direction = "LONG" if "LONG" in sig_type_str else ("SHORT" if "SHORT" in sig_type_str else "WAIT")

        if not is_actionable:
            if ticket.get("status") != "WAIT":
                return None
            # Filter duplicate consecutive WAIT logs only when the veto reason is structurally identical.
            # Normalizing reasons avoids logging duplicate rows on minor score flutter (e.g. 54.3 vs 54.8).
            if self.entries and self.entries[-1].signal_type in ["WAIT", "AWAITING_SETUP", "NO_TRADE"]:
                def _normalize_reason(r: str) -> str:
                    if not r:
                        return ""
                    r_clean = r.strip()
                    if "Confluence Veto:" in r_clean or "Awaiting confluence" in r_clean:
                        return "CONFLUENCE_VETO"
                    if "Data Sufficiency Gate:" in r_clean:
                        return "DATA_SUFFICIENCY_VETO"
                    if "Opening 15-min range" in r_clean or "Freak" in r_clean:
                        return "FREAK_CANDLE_ISOLATION"
                    return r_clean

                last_reason = getattr(self.entries[-1], "trigger_reason", "")
                curr_reason = getattr(signal, "reason", "")
                if _normalize_reason(last_reason) == _normalize_reason(curr_reason):
                    return None
        else:
            # Structural fingerprint: setup_id + direction + entry/SL band (10pt buckets).
            # Replaces the old bar_timestamp+direction dedup, which logged a fresh entry
            # every single bar a setup stayed valid. Same instance -> same epoch -> collapsed;
            # a materially different entry/SL (a genuinely new instance) still gets logged.
            if not structure_epoch:
                epoch_setup = setup_id or sig_type_str
                epoch_payload = f"{epoch_setup}_{direction}_{round(signal.entry_price, -1)}_{round(signal.sl_price, -1)}"
                structure_epoch = hashlib.sha256(epoch_payload.encode("utf-8")).hexdigest()[:16]

            # Only an OPEN trade on the same structure suppresses a re-log. Once the
            # trade has closed, the same structure may legitimately re-trigger (re-entry
            # frequency is governed separately by the session cooldown rails).
            for existing in self.entries:
                if (
                    existing.structure_epoch == structure_epoch
                    and existing.direction == direction
                    and existing.is_active()
                ):
                    return None

        strike = int(ticket.get("strike", int(round(current_spot / 50.0) * 50)))
        opt_type = ticket.get("option_type", "CE" if direction == "LONG" else ("PE" if direction == "SHORT" else "N/A"))
        symbol = ticket.get("symbol", f"NIFTY {strike} {opt_type}" if is_actionable else "N/A")

        # Compute / Preserve Confluence Score
        if not is_actionable:
            c_score = 0.0
            c_grade = "Consolidation"
        elif signal and signal.details and "confluence_score" in signal.details:
            c_score = float(signal.details["confluence_score"])
            c_grade = str(signal.details.get("confluence_grade", "A Standard"))
        elif confluence_score is not None and confluence_score > 1.0:
            c_score = round(float(confluence_score), 1)
            c_grade = "A+ Institutional" if c_score >= 75.0 else ("A Standard" if c_score >= 55.0 else ("B Tactical" if c_score >= 45.0 else "C Weak / Vetoed"))
        elif df_context is not None and not df_context.empty:
            c_score, c_grade = calculate_confluence_score(
                signal, df_context, htf_data, kalman_vel, kalman_z, regime_info, ofi_data, gex_data, vol_profile, options_context=options_context
            )
        else:
            c_score = round(confluence_score * 100.0, 1) if (confluence_score is not None and 0.0 < confluence_score <= 1.0) else (float(confluence_score) if confluence_score is not None else 0.0)
            c_grade = "A+ Institutional" if c_score >= 75.0 else ("A Standard" if c_score >= 55.0 else ("B Tactical" if c_score >= 45.0 else "C Weak / Vetoed"))

        entry_prem = float(ticket.get("entry_premium", 140.0 if is_actionable else 0.0))
        sl_prem = float(ticket.get("sl_premium", 110.0 if is_actionable else 0.0))
        t1_prem = float(ticket.get("target1_premium", 180.0 if is_actionable else 0.0))
        t2_prem = float(ticket.get("target2_premium", 225.0 if is_actionable else 0.0))
        t3_prem = float(ticket.get("target3_moonshot_premium", 280.0 if is_actionable else 0.0))

        sl_spot = float(signal.sl_price) if signal.sl_price else 0.0
        t1_spot = float(signal.target_1) if signal.target_1 else 0.0
        t2_spot = float(signal.target_2) if signal.target_2 else 0.0
        t3_spot = float(getattr(signal, "target_3_moonshot", t2_spot + 50.0)) if t2_spot > 0 else 0.0

        sl_pts_spot = abs(current_spot - sl_spot) if sl_spot > 0 else 0.0
        r_t1 = round(abs(t1_spot - current_spot) / max(sl_pts_spot, 1.0), 2) if (t1_spot > 0 and sl_pts_spot > 0) else 0.0
        r_t2 = round(abs(t2_spot - current_spot) / max(sl_pts_spot, 1.0), 2) if (t2_spot > 0 and sl_pts_spot > 0) else 0.0

        lots = int(ticket.get("lots", 6 if is_actionable else 0))
        total_qty = int(ticket.get("total_qty", lots * LOT_SIZE))
        risk_rupees = float(ticket.get("actual_risk_rupees", ticket.get("max_risk_rupees", 5000.0 if is_actionable else 0.0)))
        tca_friction = float(ticket.get("tca_friction", {}).get("total_friction", 180.0) if isinstance(ticket.get("tca_friction"), dict) else (180.0 if is_actionable else 0.0))

        if is_seed and bar_timestamp:
            clean_ts = bar_timestamp.replace("-", "").replace(" ", "").replace(":", "")
            sig_date = clean_ts[:8] if len(clean_ts) >= 8 else datetime.now(IST).strftime("%Y%m%d")
            sig_time = (clean_ts[8:12] + "00") if len(clean_ts) >= 12 else datetime.now(IST).strftime("%H%M%S")
            sig_id = f"SIG-{sig_date}-{sig_time}-{strike}{opt_type}"
            now_ist = f"{bar_timestamp}:00 IST" if len(bar_timestamp) == 16 else (f"{bar_timestamp} IST" if "IST" not in bar_timestamp else bar_timestamp)
        else:
            sig_id = f"SIG-{datetime.now(IST).strftime('%Y%m%d')}-{datetime.now(IST).strftime('%H%M%S')}-{strike}{opt_type}"

        conviction_score = float(ticket.get("conviction_score", 0.0) or getattr(signal, "conviction_score", 0.0) or 0.0)
        conviction_tier = str(ticket.get("conviction_tier", "") or getattr(signal, "conviction_tier", "LOW") or "LOW")
        family_votes = dict(ticket.get("family_votes", {}) or getattr(signal, "family_votes", {}) or {})
        family_agreement = int(ticket.get("family_agreement", 0) or getattr(signal, "family_agreement", 0) or 0)
        directional_score = float(ticket.get("directional_score", 0.0) or getattr(signal, "directional_score", 0.0) or 0.0)

        gate_audit_resolved = gate_audit or (ticket.get("gate_audit") if isinstance(ticket, dict) else None) or (signal.details.get("gate_audit") if signal and signal.details else {}) or {}
        evidence_resolved = evidence or (ticket.get("evidence") if isinstance(ticket, dict) else None) or (signal.details.get("evidence") if signal and signal.details else {}) or {}

        entry = SignalEntry(
            signal_id=sig_id,
            timestamp_ist=now_ist,
            timestamp_utc_ms=utc_ms,
            bar_timestamp=bar_time_str,
            spot_price=round(current_spot, 2),
            signal_type=sig_type_str,
            direction=direction,
            trigger_reason=signal.reason,
            selected_strike=strike if is_actionable else 0,
            option_type=opt_type,
            symbol=symbol,
            entry_premium=entry_prem,
            sl_spot=sl_spot,
            sl_premium=sl_prem,
            sl_points_spot=sl_pts_spot,
            sl_risk_premium_pts=abs(entry_prem - sl_prem),
            target_1_spot=t1_spot,
            target_1_premium=t1_prem,
            target_2_spot=t2_spot,
            target_2_premium=t2_prem,
            target_3_spot=t3_spot,
            target_3_premium=t3_prem,
            r_multiple_t1=r_t1,
            r_multiple_t2=r_t2,
            confluence_score=c_score,
            confluence_grade=c_grade,
            regime_summary=f"Kalman V={kalman_vel:+.1f} | {regime_info.get('active_regime', 'Normal') if regime_info else 'Normal'}",
            kalman_velocity=round(kalman_vel, 2),
            kalman_zscore=round(kalman_z, 2),
            markov_regime=regime_info.get("active_regime", "Trend") if regime_info else "Trend",
            htf_alignment=f"1H: {htf_data.get('tf_1h', {}).get('bias', 'N/A')[:4]} | 15m: {htf_data.get('tf_15m', {}).get('bias', 'N/A')[:4]}" if htf_data else "HTF Aligned",
            is_0dte=is_0dte,
            lots_suggested=lots,
            total_qty=total_qty,
            capital_risk_rupees=risk_rupees,
            tca_friction_est=tca_friction,
            lifecycle_status=SignalLifecycleStatus.TRIGGERED.value if is_actionable else "AWAITING_SETUP",
            is_seed=is_seed,
            setup_id=setup_id or sig_type_str,
            structure_epoch=structure_epoch,
            gate_audit=gate_audit_resolved,
            evidence=evidence_resolved,
            greeks_snapshot={
                "delta": ticket.get("delta", 0.55 if is_actionable else 0.0),
                "gamma": ticket.get("gamma", 0.0008 if is_actionable else 0.0),
                "theta": ticket.get("theta_decay_daily", -12.0 if is_actionable else 0.0),
                "vanna": ticket.get("vanna", 0.04 if is_actionable else 0.0)
            },
            notes="Institutional setup triggered & registered in audit log." if is_actionable else "Consolidation / Awaiting confluence trigger.",
            prev_hash=self._last_hash,
            record_hash="",
            conviction_score=conviction_score,
            conviction_tier=conviction_tier,
            family_votes=family_votes,
            family_agreement=family_agreement,
            directional_score=directional_score,
            schema_version=2
        )

        rec_hash = compute_sha256_record_hash(self._last_hash, entry.to_dict())
        entry.record_hash = rec_hash
        self._last_hash = rec_hash
        self.entries.append(entry)
        self._persist_to_disk()
        try:
            from src.cloud_storage import upsert_signal_async
            upsert_signal_async(entry.to_dict())
        except Exception:
            pass
        return entry

    def update_open_trades_lifecycle(
        self,
        current_spot: float,
        current_high: float,
        current_low: float,
        bar_time_str: str = "12:00",
        current_open: Optional[float] = None
    ) -> int:
        """Evaluates all active trades against the current bar high/low/close prices with 50% partial booking at T1."""
        updates_count = 0
        now_ist = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")

        for entry in self.entries:
            if not entry.is_active():
                continue

            if bar_time_str and bar_time_str != entry.last_counted_bar:
                entry.bars_held += 1
                entry.last_counted_bar = bar_time_str

            direction = entry.direction
            sl_spot = entry.sl_spot
            t1_spot = entry.target_1_spot
            t2_spot = entry.target_2_spot
            t3_spot = entry.target_3_spot

            # Track MFE / MAE
            if direction == "LONG":
                entry.peak_favorable_excursion_pts = max(entry.peak_favorable_excursion_pts, current_high - entry.spot_price)
                entry.peak_adverse_excursion_pts = max(entry.peak_adverse_excursion_pts, entry.spot_price - current_low)
            else:
                entry.peak_favorable_excursion_pts = max(entry.peak_favorable_excursion_pts, entry.spot_price - current_low)
                entry.peak_adverse_excursion_pts = max(entry.peak_adverse_excursion_pts, current_high - entry.spot_price)

            # Task 14: Track Level Touches independent of terminal status
            if direction == "LONG":
                if t1_spot > 0 and current_high >= t1_spot: entry.touched_t1 = True
                if t2_spot > 0 and current_high >= t2_spot: entry.touched_t2 = True
                if t3_spot > 0 and current_high >= t3_spot: entry.touched_t3 = True
            else:
                if t1_spot > 0 and current_low <= t1_spot: entry.touched_t1 = True
                if t2_spot > 0 and current_low <= t2_spot: entry.touched_t2 = True
                if t3_spot > 0 and current_low <= t3_spot: entry.touched_t3 = True

            # Check 15:15 Hard Squareoff
            if bar_time_str >= "15:15" and entry.is_active():
                entry.lifecycle_status = SignalLifecycleStatus.SQUARED_OFF.value
                entry.exit_timestamp_ist = now_ist
                entry.exit_spot = current_spot
                sl_pts = max(abs(entry.spot_price - entry.sl_spot), 1.0)
                move_pts = (current_spot - entry.spot_price) if direction == "LONG" else (entry.spot_price - current_spot)
                squared_r = round(move_pts / sl_pts, 2)
                if entry.lifecycle_status == SignalLifecycleStatus.T1_REACHED.value or entry.realized_r_multiple > 0:
                    squared_r = round(max(squared_r, entry.realized_r_multiple), 2)
                entry.realized_r_multiple = squared_r
                entry.realized_pnl_rupees = round(entry.capital_risk_rupees * squared_r, 2)
                entry.realized_pnl_net = round(entry.realized_pnl_rupees - entry.tca_friction_est, 2)
                entry.notes += f" | 15:15 IST Mandatory Squareoff @ {squared_r:+.2f}R."
                updates_count += 1
                with self._lifecycle_lock:
                    self._pending_lifecycle_events.append((entry, "SQUARED_OFF", current_spot, float(getattr(entry, "exit_premium", 0.0))))
                continue

            # Task 02: Stall Time Stop (Stagnation Exit before T1)
            if entry.bars_held >= TIME_STOP_BARS and entry.lifecycle_status in [
                SignalLifecycleStatus.TRIGGERED.value,
                SignalLifecycleStatus.ACTIVE.value
            ]:
                sl_pts = max(abs(entry.spot_price - entry.sl_spot), 1.0)
                move_pts = (current_spot - entry.spot_price) if direction == "LONG" else (entry.spot_price - current_spot)
                cur_r = move_pts / sl_pts
                if cur_r < TIME_STOP_MIN_R:
                    entry.lifecycle_status = SignalLifecycleStatus.TIME_STOPPED.value
                    entry.realized_r_multiple = round(cur_r, 2)
                    entry.realized_pnl_rupees = round(entry.capital_risk_rupees * entry.realized_r_multiple, 2)
                    entry.realized_pnl_net = round(entry.realized_pnl_rupees - entry.tca_friction_est, 2)
                    entry.exit_timestamp_ist = now_ist
                    entry.exit_spot = current_spot
                    entry.notes += f" | Time stop @ {entry.bars_held} bars ({entry.realized_r_multiple:+.2f}R)."
                    updates_count += 1
                    with self._lifecycle_lock:
                        self._pending_lifecycle_events.append((entry, "TIME_STOPPED", current_spot, float(getattr(entry, "entry_premium", 0.0))))
                    continue

            if direction == "LONG":
                hit_sl = current_low <= sl_spot and sl_spot > 0
                hit_t3 = current_high >= t3_spot and t3_spot > 0
                hit_t2 = current_high >= t2_spot and t2_spot > 0 and entry.lifecycle_status != SignalLifecycleStatus.T2_REACHED.value
                hit_t1 = current_high >= t1_spot and t1_spot > 0 and entry.lifecycle_status == SignalLifecycleStatus.TRIGGERED.value

                # Task 06: Unbiased Intrabar Resolution
                if hit_sl and (hit_t1 or hit_t2 or hit_t3):
                    open_ref = current_open if current_open is not None else entry.spot_price
                    target_level = t3_spot if hit_t3 else (t2_spot if hit_t2 else t1_spot)
                    if abs(open_ref - target_level) < abs(open_ref - sl_spot):
                        hit_sl = False

                if hit_sl:
                    if entry.lifecycle_status == SignalLifecycleStatus.T2_REACHED.value:
                        entry.lifecycle_status = SignalLifecycleStatus.STOPPED_OUT.value
                        entry.exit_timestamp_ist = now_ist
                        entry.exit_spot = sl_spot
                        entry.exit_premium = entry.target_1_premium
                        entry.notes += f" | Trailed SL (T1) hit on remaining 25%."
                        entry.realized_pnl_net = round(entry.realized_pnl_rupees - entry.tca_friction_est, 2)
                        with self._lifecycle_lock:
                            self._pending_lifecycle_events.append((entry, "STOPPED_OUT", sl_spot, entry.target_1_premium))
                    elif entry.lifecycle_status == SignalLifecycleStatus.T1_REACHED.value:
                        entry.lifecycle_status = SignalLifecycleStatus.STOPPED_OUT.value
                        entry.exit_timestamp_ist = now_ist
                        entry.exit_spot = sl_spot
                        entry.exit_premium = entry.entry_premium
                        entry.notes += f" | Breakeven SL hit on remaining 50%."
                        entry.realized_pnl_net = round(entry.realized_pnl_rupees - entry.tca_friction_est, 2)
                        with self._lifecycle_lock:
                            self._pending_lifecycle_events.append((entry, "STOPPED_OUT", sl_spot, entry.entry_premium))
                    else:
                        entry.lifecycle_status = SignalLifecycleStatus.STOPPED_OUT.value
                        entry.realized_r_multiple = -1.0
                        entry.realized_pnl_rupees = - entry.capital_risk_rupees
                        entry.realized_pnl_net = - entry.capital_risk_rupees - entry.tca_friction_est
                        entry.exit_timestamp_ist = now_ist
                        entry.exit_spot = sl_spot
                        entry.exit_premium = entry.sl_premium
                        entry.notes += f" | SL Hit @ ₹{current_low:.1f}"
                        with self._lifecycle_lock:
                            self._pending_lifecycle_events.append((entry, "STOPPED_OUT", sl_spot, entry.sl_premium))
                    updates_count += 1
                elif hit_t3:
                    entry.lifecycle_status = SignalLifecycleStatus.T3_MOONSHOT.value
                    _sl_pts = max(abs(entry.spot_price - entry.sl_spot), 1.0)
                    _t3_r = round(abs(t3_spot - entry.spot_price) / _sl_pts, 2)
                    entry.realized_r_multiple = _t3_r
                    entry.realized_pnl_rupees = round(entry.capital_risk_rupees * _t3_r, 2)
                    entry.realized_pnl_net = round(entry.realized_pnl_rupees - entry.tca_friction_est, 2)
                    entry.exit_timestamp_ist = now_ist
                    entry.exit_spot = t3_spot
                    entry.exit_premium = entry.target_3_premium
                    entry.notes += f" | T3 Moonshot Hit @ ₹{current_high:.1f}"
                    with self._lifecycle_lock:
                        self._pending_lifecycle_events.append((entry, "T3_MOONSHOT", t3_spot, entry.target_3_premium))
                    updates_count += 1
                elif hit_t2:
                    entry.lifecycle_status = SignalLifecycleStatus.T2_REACHED.value
                    entry.realized_r_multiple = round(0.5 * entry.r_multiple_t1 + 0.5 * entry.r_multiple_t2, 2)
                    entry.realized_pnl_rupees = round(entry.capital_risk_rupees * entry.realized_r_multiple, 2)
                    entry.realized_pnl_net = round(entry.realized_pnl_rupees - entry.tca_friction_est, 2)
                    entry.sl_spot = entry.target_1_spot if entry.target_1_spot > 0 else entry.spot_price  # Trail SL to T1
                    entry.exit_spot = t2_spot
                    entry.exit_premium = entry.target_2_premium
                    entry.notes += f" | T2 Hit @ ₹{current_high:.1f}. Trailed SL to T1."
                    with self._lifecycle_lock:
                        self._pending_lifecycle_events.append((entry, "T2_REACHED", current_high, entry.target_2_premium))
                    updates_count += 1
                elif hit_t1:
                    entry.lifecycle_status = SignalLifecycleStatus.T1_REACHED.value
                    entry.realized_r_multiple = round(entry.r_multiple_t1 * 0.5, 2)
                    entry.realized_pnl_rupees = round(0.5 * entry.capital_risk_rupees * entry.r_multiple_t1, 2)
                    entry.realized_pnl_net = round(entry.realized_pnl_rupees - (0.5 * entry.tca_friction_est), 2)
                    entry.sl_spot = entry.spot_price  # Trail SL to Breakeven
                    entry.notes += f" | T1 Hit. 50% booked, SL trailed to entry."
                    with self._lifecycle_lock:
                        self._pending_lifecycle_events.append((entry, "T1_REACHED", current_high, entry.target_1_premium))
                    updates_count += 1

            elif direction == "SHORT":
                hit_sl = current_high >= sl_spot and sl_spot > 0
                hit_t3 = current_low <= t3_spot and t3_spot > 0
                hit_t2 = current_low <= t2_spot and t2_spot > 0 and entry.lifecycle_status != SignalLifecycleStatus.T2_REACHED.value
                hit_t1 = current_low <= t1_spot and t1_spot > 0 and entry.lifecycle_status == SignalLifecycleStatus.TRIGGERED.value

                # Task 06: Unbiased Intrabar Resolution
                if hit_sl and (hit_t1 or hit_t2 or hit_t3):
                    open_ref = current_open if current_open is not None else entry.spot_price
                    target_level = t3_spot if hit_t3 else (t2_spot if hit_t2 else t1_spot)
                    if abs(open_ref - target_level) < abs(open_ref - sl_spot):
                        hit_sl = False

                if hit_sl:
                    if entry.lifecycle_status == SignalLifecycleStatus.T2_REACHED.value:
                        entry.lifecycle_status = SignalLifecycleStatus.STOPPED_OUT.value
                        entry.exit_timestamp_ist = now_ist
                        entry.exit_spot = sl_spot
                        entry.exit_premium = entry.target_1_premium
                        entry.notes += f" | Trailed SL (T1) hit on remaining 25%."
                        entry.realized_pnl_net = round(entry.realized_pnl_rupees - entry.tca_friction_est, 2)
                        with self._lifecycle_lock:
                            self._pending_lifecycle_events.append((entry, "STOPPED_OUT", sl_spot, entry.target_1_premium))
                    elif entry.lifecycle_status == SignalLifecycleStatus.T1_REACHED.value:
                        entry.lifecycle_status = SignalLifecycleStatus.STOPPED_OUT.value
                        entry.exit_timestamp_ist = now_ist
                        entry.exit_spot = sl_spot
                        entry.exit_premium = entry.entry_premium
                        entry.notes += f" | Breakeven SL hit on remaining 50%."
                        entry.realized_pnl_net = round(entry.realized_pnl_rupees - entry.tca_friction_est, 2)
                        with self._lifecycle_lock:
                            self._pending_lifecycle_events.append((entry, "STOPPED_OUT", sl_spot, entry.entry_premium))
                    else:
                        entry.lifecycle_status = SignalLifecycleStatus.STOPPED_OUT.value
                        entry.realized_r_multiple = -1.0
                        entry.realized_pnl_rupees = - entry.capital_risk_rupees
                        entry.realized_pnl_net = - entry.capital_risk_rupees - entry.tca_friction_est
                        entry.exit_timestamp_ist = now_ist
                        entry.exit_spot = sl_spot
                        entry.exit_premium = entry.sl_premium
                        entry.notes += f" | SL Hit @ ₹{current_high:.1f}"
                        with self._lifecycle_lock:
                            self._pending_lifecycle_events.append((entry, "STOPPED_OUT", sl_spot, entry.sl_premium))
                    updates_count += 1
                elif hit_t3:
                    entry.lifecycle_status = SignalLifecycleStatus.T3_MOONSHOT.value
                    _sl_pts = max(abs(entry.spot_price - entry.sl_spot), 1.0)
                    _t3_r = round(abs(t3_spot - entry.spot_price) / _sl_pts, 2)
                    entry.realized_r_multiple = _t3_r
                    entry.realized_pnl_rupees = round(entry.capital_risk_rupees * _t3_r, 2)
                    entry.realized_pnl_net = round(entry.realized_pnl_rupees - entry.tca_friction_est, 2)
                    entry.exit_timestamp_ist = now_ist
                    entry.exit_spot = t3_spot
                    entry.exit_premium = entry.target_3_premium
                    entry.notes += f" | T3 Moonshot Hit @ ₹{current_low:.1f}"
                    with self._lifecycle_lock:
                        self._pending_lifecycle_events.append((entry, "T3_MOONSHOT", t3_spot, entry.target_3_premium))
                    updates_count += 1
                elif hit_t2:
                    entry.lifecycle_status = SignalLifecycleStatus.T2_REACHED.value
                    entry.realized_r_multiple = round(0.5 * entry.r_multiple_t1 + 0.5 * entry.r_multiple_t2, 2)
                    entry.realized_pnl_rupees = round(entry.capital_risk_rupees * entry.realized_r_multiple, 2)
                    entry.realized_pnl_net = round(entry.realized_pnl_rupees - entry.tca_friction_est, 2)
                    entry.sl_spot = entry.target_1_spot if entry.target_1_spot > 0 else entry.spot_price  # Trail SL to T1
                    entry.exit_spot = t2_spot
                    entry.exit_premium = entry.target_2_premium
                    entry.notes += f" | T2 Hit @ ₹{current_low:.1f}. Trailed SL to T1."
                    with self._lifecycle_lock:
                        self._pending_lifecycle_events.append((entry, "T2_REACHED", current_low, entry.target_2_premium))
                    updates_count += 1
                elif hit_t1:
                    entry.lifecycle_status = SignalLifecycleStatus.T1_REACHED.value
                    entry.realized_r_multiple = round(entry.r_multiple_t1 * 0.5, 2)
                    entry.realized_pnl_rupees = round(0.5 * entry.capital_risk_rupees * entry.r_multiple_t1, 2)
                    entry.realized_pnl_net = round(entry.realized_pnl_rupees - (0.5 * entry.tca_friction_est), 2)
                    entry.sl_spot = entry.spot_price
                    entry.notes += f" | T1 Hit. 50% booked, SL trailed to entry."
                    with self._lifecycle_lock:
                        self._pending_lifecycle_events.append((entry, "T1_REACHED", current_low, entry.target_1_premium))
                    updates_count += 1

        if updates_count > 0:
            self._persist_to_disk()
            try:
                from src.cloud_storage import upsert_signal_async
                with self._lifecycle_lock:
                    for ev_entry, _, _, _ in self._pending_lifecycle_events:
                        upsert_signal_async(ev_entry.to_dict())
            except Exception:
                pass

        return updates_count

    def drain_lifecycle_events(self) -> List[Tuple[SignalEntry, str, float, float]]:
        """
        Drains and returns all queued lifecycle transition events since the last call.
        Each event is a tuple of (entry, status_event, current_spot, current_premium).
        """
        with self._lifecycle_lock:
            events = list(self._pending_lifecycle_events)
            self._pending_lifecycle_events.clear()
            return events

    def seed_from_intraday_history(
        self,
        df: pd.DataFrame,
        strategy_engine: Any,
        live_iv: float = 0.135,
        capital: float = 500000.0
    ) -> int:
        """
        Scans through the loaded intraday dataframe and populates historical signals and their trade outcomes.
        Tags seeded entries with is_seed=True and maintains strict idempotency.
        """
        from src.options_engine import generate_option_trade_ticket
        from src.strategy_rules import SignalType
        
        if df.empty or len(df) < 15:
            return 0
            
        seeded_count = 0
        today = df.index[-1].date() if hasattr(df.index, "date") else None
        today_start_idx = next((i for i, ts in enumerate(df.index) if hasattr(ts, "date") and ts.date() == today), max(0, len(df) - 80))
        start_idx = max(15, today_start_idx)
        
        for i in range(start_idx, len(df)):
            sub_df = df.iloc[:i+1]
            cur_spot = float(sub_df.iloc[-1]["close"])
            cur_high = float(sub_df.iloc[-1]["high"])
            cur_low = float(sub_df.iloc[-1]["low"])
            bar_ts = sub_df.index[-1].strftime("%Y-%m-%d %H:%M") if hasattr(sub_df.index[-1], "strftime") else str(sub_df.index[-1])
            bar_t_str = sub_df.index[-1].strftime("%H:%M") if hasattr(sub_df.index[-1], "strftime") else "12:00"
            
            # Update previous open trades first
            self.update_open_trades_lifecycle(cur_spot, cur_high, cur_low, bar_time_str=bar_t_str)
            
            # Deduplication: do not re-evaluate / re-seed if bar already processed
            if any(e.bar_timestamp == bar_ts for e in self.entries):
                continue

            # Evaluate signal on this bar using full df history context
            sig = strategy_engine.evaluate_bar(df, current_idx=i, live_iv=live_iv)
            if sig.signal_type != SignalType.WAIT:
                tkt = generate_option_trade_ticket(cur_spot, sig, capital, 0.0, iv=live_iv)
                if tkt.get("status") == "READY":
                    # Prefer the score computed at decision time (Phase 1 pre-decision scoring);
                    # only a few ungated branches (e.g. the 3PM breakout) skip it, so fall back
                    # to computing it now rather than a hardcoded placeholder.
                    if sig.details and "confluence_score" in sig.details:
                        c_score = sig.details["confluence_score"]
                    else:
                        c_score, _ = calculate_confluence_score(sig, sub_df)
                    # Real regime for this historical bar, from the same Markov switcher
                    # instance the engine used internally, not a hardcoded placeholder.
                    try:
                        regime_snapshot = strategy_engine.markov_switcher.infer_regimes(sub_df)
                    except Exception:
                        regime_snapshot = {"active_regime": "UNKNOWN"}
                    entry = self.log_signal(
                        signal=sig,
                        ticket=tkt,
                        current_spot=cur_spot,
                        bar_timestamp=bar_ts,
                        regime_info=regime_snapshot,
                        confluence_score=c_score,
                        df_context=sub_df,
                        is_seed=True,
                        setup_id=sig.signal_type.value,
                        gate_audit=sig.details.get("gate_audit", {}) if sig.details else {}
                    )
                    if entry:
                        seeded_count += 1
                        
        # Final pass update
        if len(df) > 0:
            final_t_str = df.index[-1].strftime("%H:%M") if hasattr(df.index[-1], "strftime") else "15:30"
            self.update_open_trades_lifecycle(
                current_spot=float(df.iloc[-1]["close"]),
                current_high=float(df.iloc[-1]["high"]),
                current_low=float(df.iloc[-1]["low"]),
                bar_time_str=final_t_str
            )

        # Clear historical replay lifecycle events so seeding does not trigger live alerts
        with self._lifecycle_lock:
            self._pending_lifecycle_events.clear()

        return seeded_count

    def get_journal_dataframe(
        self,
        target_date: Optional[str] = None,
        actionable_only: bool = False,
        scope: str = "auto"
    ) -> pd.DataFrame:
        """
        Returns a Pandas DataFrame formatted for table inspection, optionally filtered by date and actionable status.
        
        Args:
            target_date: Optional date string ('YYYY-MM-DD').
            actionable_only: If True, include only LONG and SHORT setups.
            scope: 'today' (enforce today's date), 'all' (return full history), or 'auto' (target_date if passed, else all).
        """
        if not self.entries:
            return pd.DataFrame()

        def _entry_date(e: SignalEntry) -> str:
            ts = str(getattr(e, "bar_timestamp", "") or getattr(e, "timestamp_ist", ""))
            return ts[:10] if len(ts) >= 10 else ""

        entries = self.entries
        today_str = datetime.now(IST).strftime("%Y-%m-%d")

        if scope == "today":
            target = target_date or today_str
            entries = [e for e in entries if _entry_date(e) == target]
        elif scope == "all":
            if target_date and target_date != "ALL":
                entries = [e for e in entries if _entry_date(e) == target_date]
        else:  # "auto"
            if target_date and target_date != "ALL":
                entries = [e for e in entries if _entry_date(e) == target_date]

        if actionable_only:
            entries = [e for e in entries if e.direction in ["LONG", "SHORT"]]

        if not entries:
            return pd.DataFrame()

        rows = []
        for e in entries:
            bar_t = str(getattr(e, "bar_timestamp", "") or "")
            if " " in bar_t:
                bar_display_time = bar_t.split(" ")[1]
            elif ":" in bar_t:
                bar_display_time = bar_t
            else:
                bar_display_time = e.timestamp_ist.split(" ")[1] if " " in e.timestamp_ist else e.timestamp_ist

            rows.append({
                "Signal ID": e.signal_id,
                "Time (IST)": bar_display_time,
                "Direction": e.direction,
                "Signal Type": e.signal_type,
                "Symbol": e.symbol,
                "Spot Entry": f"₹{e.spot_price:,.2f}",
                "Entry Prem (₹)": f"₹{e.entry_premium:.2f}",
                "Stop Loss (₹)": f"₹{e.sl_premium:.2f} ({e.sl_spot:.1f} spot)",
                "Target 1 (₹)": f"₹{e.target_1_premium:.2f} ({e.target_1_spot:.1f} spot)",
                "Target 2 (₹)": f"₹{e.target_2_premium:.2f} ({e.target_2_spot:.1f} spot)",
                "Target 3 (₹)": f"₹{e.target_3_premium:.2f}",
                "Confluence": f"{e.confluence_score:.0f}%",
                "Grade": e.confluence_grade,
                "Lots": f"{e.lots_suggested} ({e.total_qty} Qty)",
                "Risk (₹)": f"₹{e.capital_risk_rupees:,.2f}",
                "Status": e.lifecycle_status,
                "Realized R": f"{e.realized_r_multiple:+.2f}R" if not e.is_active() else "Active",
                "Net PnL (₹)": f"₹{e.realized_pnl_rupees:+,.2f}" if not e.is_active() else "--",
                "MFE (Pts)": f"+{e.peak_favorable_excursion_pts:.1f}",
                "MAE (Pts)": f"-{e.peak_adverse_excursion_pts:.1f}",
                "HTF Bias": e.htf_alignment,
                "Trigger Rationale": e.trigger_reason,
                "Exit Time": e.exit_timestamp_ist.split(" ")[1] if e.exit_timestamp_ist and " " in e.exit_timestamp_ist else (e.exit_timestamp_ist or "--")
            })

        return pd.DataFrame(rows)

    def compute_daily_journal_summary(self, target_date: Optional[str] = None, scope: str = "auto") -> Dict[str, Any]:
        """
        Computes daily signal statistics and performance metrics with date scoping.
        
        Args:
            target_date: Optional date string ('YYYY-MM-DD').
            scope: 'today' (enforce today's IST date), 'all' (all dates), or 'auto' (today if exists, else latest/all).
        """
        if not self.entries:
            return {
                "total_signals": 0,
                "actionable_trades": 0,
                "long_trades": 0,
                "short_trades": 0,
                "active_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate_pct": 0.0,
                "total_realized_pnl": 0.0,
                "total_r_multiple": 0.0,
                "avg_r_multiple": 0.0,
                "avg_confluence_score": 0.0,
                "system_quality_number_sqn": 0.0,
                "profit_factor": 0.0,
                "session_date": target_date or "N/A"
            }

        today_ist = datetime.now(IST).strftime("%Y-%m-%d")

        def _entry_date(e: SignalEntry) -> str:
            ts = str(getattr(e, "bar_timestamp", "") or getattr(e, "timestamp_ist", ""))
            return ts[:10] if len(ts) >= 10 else ""

        if target_date == "today" or scope == "today":
            effective_date = today_ist
            filtered = [e for e in self.entries if _entry_date(e) == effective_date]
        elif target_date:
            effective_date = target_date
            filtered = [e for e in self.entries if _entry_date(e) == effective_date]
        elif scope == "all":
            effective_date = "ALL_DATES"
            filtered = self.entries
        else:  # scope == "auto"
            today_entries = [e for e in self.entries if _entry_date(e) == today_ist]
            if today_entries:
                effective_date = today_ist
                filtered = today_entries
            else:
                dates = [_entry_date(e) for e in self.entries if _entry_date(e)]
                latest_date = max(dates) if dates else today_ist
                effective_date = latest_date
                filtered = [e for e in self.entries if _entry_date(e) == latest_date]

        total = len(filtered)
        longs = sum(1 for e in filtered if e.direction == "LONG")
        shorts = sum(1 for e in filtered if e.direction == "SHORT")
        active = sum(1 for e in filtered if e.direction in ["LONG", "SHORT"] and e.is_active())

        actionable_entries = [e for e in filtered if e.direction in ["LONG", "SHORT"]]
        closed = [e for e in actionable_entries if not e.is_active()]
        wins = [e for e in closed if e.realized_r_multiple > 0]
        losses = [e for e in closed if e.realized_r_multiple <= 0]

        win_rate = (len(wins) / len(closed) * 100.0) if closed else 0.0
        total_pnl = sum(e.realized_pnl_rupees for e in closed)
        total_r = sum(e.realized_r_multiple for e in closed)
        avg_r = (total_r / len(closed)) if closed else 0.0
        avg_conf = float(np.mean([e.confluence_score for e in actionable_entries])) if actionable_entries else 0.0

        gross_gains = sum(e.realized_pnl_rupees for e in wins)
        gross_losses = abs(sum(e.realized_pnl_rupees for e in losses))
        pf = round(gross_gains / gross_losses, 2) if gross_losses > 0 else (99.0 if gross_gains > 0 else 0.0)

        r_arr = np.array([e.realized_r_multiple for e in closed], dtype=np.float64) if closed else np.array([0.0])
        r_std = float(np.std(r_arr, ddof=1)) if len(r_arr) > 1 else 0.01
        sqn = float(np.sqrt(len(closed)) * (avg_r / max(r_std, 0.001))) if closed else 0.0

        return {
            "total_signals": total,
            "actionable_trades": len(actionable_entries),
            "long_trades": longs,
            "short_trades": shorts,
            "active_trades": active,
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate_pct": round(win_rate, 1),
            "total_realized_pnl": round(total_pnl, 2),
            "total_r_multiple": round(total_r, 2),
            "avg_r_multiple": round(avg_r, 2),
            "avg_confluence_score": round(avg_conf, 1),
            "system_quality_number_sqn": round(sqn, 2),
            "profit_factor": pf,
            "session_date": effective_date
        }

    def clear_journal(self):
        """Clears current journal entries and resets persistence file."""
        self.entries = []
        self._last_hash = "GENESIS_ROOT_HASH_0000000000000000"
        self._persist_to_disk()

    def export_csv_bytes(self, target_date: Optional[str] = None, scope: str = "all") -> bytes:
        """Exports the journal entries to CSV format as bytes for browser downloading."""
        df = self.get_journal_dataframe(target_date=target_date, actionable_only=False, scope=scope)
        if df.empty:
            return b"Timestamp,Signal_Type,Direction,Spot_Price,Symbol,Status\n"
        return df.to_csv(index=False).encode("utf-8")

    def cluster_context(
        self,
        direction: str,
        setup_id: Optional[str] = None,
        now_ist: str = "",
        lookback_min: int = 90
    ) -> Dict[str, Any]:
        """
        Task 04 & Setup-Level Attribution: Evaluates signal sequence cluster position 
        and prior-signal MFE decay across direction and specific setup_id.
        """
        actionable = [e for e in self.entries if e.direction in ["LONG", "SHORT"]]
        if not actionable:
            return {
                "index": 1,
                "count": 0,
                "prior_mfe_pts": [],
                "prior_mfe_median": 0.0,
                "prior_went_negative": 0,
                "setup_index": 1,
                "setup_count": 0,
                "setup_mfe_median": 0.0,
                "setup_went_negative": 0
            }

        same_dir = [e for e in actionable if e.direction == direction]
        mfes = [float(e.peak_favorable_excursion_pts) for e in same_dir]
        neg_count = sum(1 for e in same_dir if e.realized_r_multiple <= 0 and not e.is_active())
        median_mfe = float(np.median(mfes)) if mfes else 0.0

        # Setup-specific cluster context
        if setup_id:
            same_setup = [
                e for e in actionable 
                if (getattr(e, "setup_id", "") == setup_id or e.signal_type == setup_id)
            ]
            setup_mfes = [float(e.peak_favorable_excursion_pts) for e in same_setup]
            setup_neg = sum(1 for e in same_setup if e.realized_r_multiple <= 0 and not e.is_active())
            setup_med_mfe = float(np.median(setup_mfes)) if setup_mfes else 0.0
            setup_idx = len(same_setup) + 1
            setup_cnt = len(same_setup)
        else:
            setup_mfes = []
            setup_neg = 0
            setup_med_mfe = 0.0
            setup_idx = 1
            setup_cnt = 0

        return {
            "index": len(same_dir) + 1,
            "count": len(same_dir),
            "prior_mfe_pts": mfes,
            "prior_mfe_median": round(median_mfe, 1),
            "prior_went_negative": neg_count,
            "setup_index": setup_idx,
            "setup_count": setup_cnt,
            "setup_mfe_median": round(setup_med_mfe, 1),
            "setup_went_negative": setup_neg
        }

    def _persist_to_disk(self):
        """Saves current journal to JSON file atomically."""
        if not self.persistence_file:
            return
        try:
            os.makedirs(os.path.dirname(self.persistence_file), exist_ok=True)
            temp_file = self.persistence_file + f".tmp.{uuid.uuid4().hex[:6]}"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump([e.to_dict() for e in self.entries], f, indent=2)
            os.replace(temp_file, self.persistence_file)
        except Exception:
            pass


class SignalPerformanceAnalyzer:
    """
    Institutional Signal Performance & Execution Analytics Suite.
    
    Provides deep-dive trade attribution, win rate decompositions across setup types,
    intraday execution time buckets, macro/Markov regimes, confluence correlation analysis,
    and behavioral tilt / losing streak diagnostics.
    """

    def __init__(
        self,
        entries: Optional[List[SignalEntry]] = None,
        include_seeds: bool = False,
        min_schema_version: Optional[int] = None
    ):
        raw = entries or []
        # Task 05: Exclude schema v1 legacy records from performance analytics when v2 exists
        has_v2 = any(getattr(e, "schema_version", 1) >= 2 for e in raw)
        target_min_ver = min_schema_version if min_schema_version is not None else (2 if has_v2 else 1)
        self.raw_entries: List[SignalEntry] = [
            e for e in raw if getattr(e, "schema_version", 1) >= target_min_ver
        ]
        # Filter completed / closed actionable trades
        wait_val = SignalType.WAIT.value if hasattr(SignalType.WAIT, "value") else "WAIT"
        self.closed_entries: List[SignalEntry] = [
            e for e in self.raw_entries
            if not e.is_active() and e.signal_type not in ["WAIT", wait_val]
            and (include_seeds or not getattr(e, "is_seed", False))
        ]
        self.entries: List[SignalEntry] = self.closed_entries

    def t1_to_t2_conversion_rate(self) -> Dict[str, Any]:
        """
        Task 14: Measures empirical conversion rate of trades reaching T1 that then reached T2.
        """
        t1_touches = sum(
            1 for e in self.raw_entries
            if getattr(e, "touched_t1", False) or e.lifecycle_status in [
                SignalLifecycleStatus.T1_REACHED.value,
                SignalLifecycleStatus.T2_REACHED.value,
                SignalLifecycleStatus.T3_MOONSHOT.value
            ]
        )
        t2_touches = sum(
            1 for e in self.raw_entries
            if getattr(e, "touched_t2", False) or e.lifecycle_status in [
                SignalLifecycleStatus.T2_REACHED.value,
                SignalLifecycleStatus.T3_MOONSHOT.value
            ]
        )
        t3_touches = sum(
            1 for e in self.raw_entries
            if getattr(e, "touched_t3", False) or e.lifecycle_status == SignalLifecycleStatus.T3_MOONSHOT.value
        )

        conversion_pct = (t2_touches / t1_touches * 100.0) if t1_touches > 0 else 0.0
        return {
            "t1_touches": t1_touches,
            "t2_touches": t2_touches,
            "t3_touches": t3_touches,
            "conversion_rate_pct": round(conversion_pct, 1),
            "verdict": "LADDER_SOUND" if conversion_pct >= 55.0 else ("MARGINAL" if conversion_pct >= 35.0 else "CANNOT_PAY")
        }

    def win_rate_by_signal_type(self) -> pd.DataFrame:
        """Computes trade counts, win rate, average R-multiple, and PnL grouped by signal type."""
        columns = [
            "signal_type", "total_trades", "winning_trades", "losing_trades",
            "win_rate_pct", "avg_r_multiple", "total_pnl_rupees", "profit_factor", "avg_confluence"
        ]
        if not self.closed_entries:
            return pd.DataFrame(columns=columns)

        groups: Dict[str, List[SignalEntry]] = {}
        for e in self.closed_entries:
            stype = str(e.signal_type)
            groups.setdefault(stype, []).append(e)

        records = []
        for stype, group in sorted(groups.items(), key=lambda x: len(x[1]), reverse=True):
            total = len(group)
            wins = sum(1 for e in group if e.realized_r_multiple > 0)
            losses = sum(1 for e in group if e.realized_r_multiple <= 0)
            win_rate = round((wins / total) * 100.0, 1) if total > 0 else 0.0
            avg_r = round(float(np.mean([e.realized_r_multiple for e in group])), 2)
            total_pnl = round(float(sum(e.realized_pnl_rupees for e in group)), 2)
            gross_gain = sum(e.realized_pnl_rupees for e in group if e.realized_pnl_rupees > 0)
            gross_loss = abs(sum(e.realized_pnl_rupees for e in group if e.realized_pnl_rupees < 0))
            pf = round(gross_gain / gross_loss, 2) if gross_loss > 0 else (99.0 if gross_gain > 0 else 0.0)
            avg_conf = round(float(np.mean([e.confluence_score for e in group])), 1)

            records.append({
                "signal_type": stype,
                "total_trades": total,
                "winning_trades": wins,
                "losing_trades": losses,
                "win_rate_pct": win_rate,
                "avg_r_multiple": avg_r,
                "total_pnl_rupees": total_pnl,
                "profit_factor": pf,
                "avg_confluence": avg_conf
            })

        return pd.DataFrame(records)

    def win_rate_by_time_bucket(self, bucket_minutes: int = 30) -> pd.DataFrame:
        """Computes trade attribution and win rate broken down into intraday time windows."""
        columns = [
            "time_bucket", "total_trades", "winning_trades", "losing_trades",
            "win_rate_pct", "avg_r_multiple", "total_pnl_rupees", "profit_factor"
        ]
        if not self.closed_entries:
            return pd.DataFrame(columns=columns)

        def _get_bucket_key(entry: SignalEntry) -> Tuple[int, str]:
            raw_ts = entry.timestamp_ist or entry.bar_timestamp or "09:15"
            cleaned = raw_ts.replace("IST", "").strip()
            time_part = cleaned.split(" ")[1] if " " in cleaned else cleaned
            parts = time_part.split(":")
            try:
                hour = int(parts[0])
                minute = int(parts[1]) if len(parts) > 1 else 0
            except Exception:
                hour, minute = 9, 15
            
            total_mins = hour * 60 + minute
            start_mins = (total_mins // bucket_minutes) * bucket_minutes
            end_mins = start_mins + bucket_minutes
            start_str = f"{start_mins // 60:02d}:{start_mins % 60:02d}"
            end_str = f"{end_mins // 60:02d}:{end_mins % 60:02d}"
            label = f"{start_str} - {end_str}"
            return start_mins, label

        buckets: Dict[Tuple[int, str], List[SignalEntry]] = {}
        for e in self.closed_entries:
            k = _get_bucket_key(e)
            buckets.setdefault(k, []).append(e)

        records = []
        for (start_min, label), group in sorted(buckets.items(), key=lambda x: x[0][0]):
            total = len(group)
            wins = sum(1 for e in group if e.realized_r_multiple > 0)
            losses = sum(1 for e in group if e.realized_r_multiple <= 0)
            win_rate = round((wins / total) * 100.0, 1) if total > 0 else 0.0
            avg_r = round(float(np.mean([e.realized_r_multiple for e in group])), 2)
            total_pnl = round(float(sum(e.realized_pnl_rupees for e in group)), 2)
            gross_gain = sum(e.realized_pnl_rupees for e in group if e.realized_pnl_rupees > 0)
            gross_loss = abs(sum(e.realized_pnl_rupees for e in group if e.realized_pnl_rupees < 0))
            pf = round(gross_gain / gross_loss, 2) if gross_loss > 0 else (99.0 if gross_gain > 0 else 0.0)

            records.append({
                "time_bucket": label,
                "total_trades": total,
                "winning_trades": wins,
                "losing_trades": losses,
                "win_rate_pct": win_rate,
                "avg_r_multiple": avg_r,
                "total_pnl_rupees": total_pnl,
                "profit_factor": pf
            })

        return pd.DataFrame(records)

    def win_rate_by_regime(self) -> pd.DataFrame:
        """Computes win rate and performance metrics segmented by market/Markov regime."""
        columns = [
            "regime", "total_trades", "winning_trades", "losing_trades",
            "win_rate_pct", "avg_r_multiple", "total_pnl_rupees", "profit_factor"
        ]
        if not self.closed_entries:
            return pd.DataFrame(columns=columns)

        groups: Dict[str, List[SignalEntry]] = {}
        for e in self.closed_entries:
            regime = e.markov_regime or "Normal"
            groups.setdefault(regime, []).append(e)

        records = []
        for regime, group in sorted(groups.items(), key=lambda x: len(x[1]), reverse=True):
            total = len(group)
            wins = sum(1 for e in group if e.realized_r_multiple > 0)
            losses = sum(1 for e in group if e.realized_r_multiple <= 0)
            win_rate = round((wins / total) * 100.0, 1) if total > 0 else 0.0
            avg_r = round(float(np.mean([e.realized_r_multiple for e in group])), 2)
            total_pnl = round(float(sum(e.realized_pnl_rupees for e in group)), 2)
            gross_gain = sum(e.realized_pnl_rupees for e in group if e.realized_pnl_rupees > 0)
            gross_loss = abs(sum(e.realized_pnl_rupees for e in group if e.realized_pnl_rupees < 0))
            pf = round(gross_gain / gross_loss, 2) if gross_loss > 0 else (99.0 if gross_gain > 0 else 0.0)

            records.append({
                "regime": regime,
                "total_trades": total,
                "winning_trades": wins,
                "losing_trades": losses,
                "win_rate_pct": win_rate,
                "avg_r_multiple": avg_r,
                "total_pnl_rupees": total_pnl,
                "profit_factor": pf
            })

        return pd.DataFrame(records)

    def score_vs_outcome_correlation(self, score_field: str = "confluence_score") -> Dict[str, Any]:
        """
        Computes Pearson correlation (r, p-value) between score (confluence_score or conviction_score)
        and trade outcomes, and bins performance into 5 score buckets ([0-50, 50-65, 65-75, 75-85, 85-100]).
        """
        bucket_defs = [
            ("0-50", 0.0, 50.0),
            ("50-65", 50.0, 65.0),
            ("65-75", 65.0, 75.0),
            ("75-85", 75.0, 85.0),
            ("85-100", 85.0, 100.01)
        ]

        if not self.closed_entries:
            empty_buckets = [
                {"bucket": name, "count": 0, "win_rate_pct": 0.0, "avg_r_multiple": 0.0, "total_pnl_rupees": 0.0}
                for name, _, _ in bucket_defs
            ]
            return {
                "pearson_r": 0.0,
                "p_value": 1.0,
                "buckets": empty_buckets,
                "bucket_dict": {b["bucket"]: b for b in empty_buckets},
                "correlation_strength": "NEUTRAL",
                "statistically_significant": False,
                "sample_size": 0
            }

        scores = [float(getattr(e, score_field, 0.0) or 0.0) for e in self.closed_entries]
        r_multiples = [float(e.realized_r_multiple) for e in self.closed_entries]

        # Pearson correlation
        if len(scores) >= 2 and np.std(scores) > 1e-6 and np.std(r_multiples) > 1e-6:
            try:
                r_val, p_val = stats.pearsonr(scores, r_multiples)
                r_val = float(r_val) if not np.isnan(r_val) else 0.0
                p_val = float(p_val) if not np.isnan(p_val) else 1.0
            except Exception:
                r_matrix = np.corrcoef(scores, r_multiples)
                r_val = float(r_matrix[0, 1]) if not np.isnan(r_matrix[0, 1]) else 0.0
                p_val = 0.05 if abs(r_val) > 0.5 else 0.5
        else:
            r_val, p_val = 0.0, 1.0

        if r_val >= 0.5:
            strength = "STRONG_POSITIVE"
        elif r_val >= 0.2:
            strength = "MODERATE_POSITIVE"
        elif r_val <= -0.2:
            strength = "NEGATIVE"
        else:
            strength = "WEAK_NEUTRAL"

        # Bucket performance
        bucket_results = []
        bucket_dict = {}
        for name, low, high in bucket_defs:
            if name == "85-100":
                b_entries = [e for e in self.closed_entries if low <= float(getattr(e, score_field, 0.0) or 0.0) <= 100.0]
            else:
                b_entries = [e for e in self.closed_entries if low <= float(getattr(e, score_field, 0.0) or 0.0) < high]

            cnt = len(b_entries)
            if cnt > 0:
                wins = sum(1 for e in b_entries if e.realized_r_multiple > 0)
                wr = round((wins / cnt) * 100.0, 1)
                avg_r = round(float(np.mean([e.realized_r_multiple for e in b_entries])), 2)
                pnl = round(float(sum(e.realized_pnl_rupees for e in b_entries)), 2)
            else:
                wr, avg_r, pnl = 0.0, 0.0, 0.0

            b_data = {
                "bucket": name,
                "count": cnt,
                "win_rate_pct": wr,
                "avg_r_multiple": avg_r,
                "total_pnl_rupees": pnl
            }
            bucket_results.append(b_data)
            bucket_dict[name] = b_data

        return {
            "pearson_r": round(r_val, 4),
            "p_value": round(p_val, 4),
            "buckets": bucket_results,
            "bucket_dict": bucket_dict,
            "correlation_strength": strength,
            "statistically_significant": bool(p_val < 0.05) if len(scores) >= 5 else False,
            "sample_size": len(self.closed_entries)
        }

    def confluence_vs_outcome_correlation(self) -> Dict[str, Any]:
        """Backward-compatible proxy to score_vs_outcome_correlation using confluence_score."""
        return self.score_vs_outcome_correlation("confluence_score")

    def streak_and_tilt_analysis(self) -> Dict[str, Any]:
        """
        Analyzes consecutive win/loss streaks and evaluates behavioral tilt risk.
        Flags tilt when 3+ consecutive losses coincide with elevated trading frequency.
        """
        if not self.closed_entries:
            return {
                "current_streak_type": "NONE",
                "current_streak_count": 0,
                "max_win_streak": 0,
                "max_loss_streak": 0,
                "consecutive_losses": 0,
                "tilt_detected": False,
                "tilt_warning_level": "NORMAL",
                "avg_trade_interval_minutes": 0.0,
                "loss_streak_interval_minutes": 0.0,
                "frequency_acceleration_ratio": 1.0,
                "recommended_action": "NORMAL_TRADING"
            }

        ordered = sorted(self.closed_entries, key=lambda x: getattr(x, "timestamp_utc_ms", 0))
        outcomes = [1 if e.realized_r_multiple > 0 else -1 for e in ordered]

        max_win = 0
        max_loss = 0
        cur_type = "NONE"
        cur_count = 0

        temp_type = "NONE"
        temp_count = 0

        for outcome in outcomes:
            o_type = "WIN" if outcome > 0 else "LOSS"
            if o_type == temp_type:
                temp_count += 1
            else:
                temp_type = o_type
                temp_count = 1

            if temp_type == "WIN" and temp_count > max_win:
                max_win = temp_count
            elif temp_type == "LOSS" and temp_count > max_loss:
                max_loss = temp_count

        cur_type = temp_type
        cur_count = temp_count
        consecutive_losses = cur_count if cur_type == "LOSS" else 0

        timestamps = []
        for e in ordered:
            if hasattr(e, "timestamp_utc_ms") and e.timestamp_utc_ms > 0:
                timestamps.append(e.timestamp_utc_ms / (1000.0 * 60.0))
            else:
                timestamps.append(float(len(timestamps) * 5.0))

        intervals = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))] if len(timestamps) > 1 else []
        avg_interval = float(np.mean(intervals)) if intervals else 30.0

        loss_streak_intervals = []
        if consecutive_losses >= 2:
            loss_streak_timestamps = timestamps[-consecutive_losses:]
            loss_streak_intervals = [loss_streak_timestamps[i] - loss_streak_timestamps[i-1] for i in range(1, len(loss_streak_timestamps))]

        loss_interval_mean = float(np.mean(loss_streak_intervals)) if loss_streak_intervals else avg_interval
        freq_ratio = (avg_interval / max(loss_interval_mean, 1.0)) if avg_interval > 0 else 1.0

        tilt_flag = False
        tilt_level = "NORMAL"
        action = "NORMAL_TRADING"

        if consecutive_losses >= 4:
            tilt_flag = True
            tilt_level = "CRITICAL"
            action = "MANDATORY_SESSION_HALT_COOLDOWN"
        elif consecutive_losses >= 3:
            if freq_ratio >= 1.3 or loss_interval_mean <= 15.0 or len(ordered) <= 4:
                tilt_flag = True
                tilt_level = "ELEVATED"
                action = "MANDATORY_COOLDOWN_30M_SIZE_REDUCTION"
            else:
                tilt_flag = False
                tilt_level = "WARNING"
                action = "REDUCE_SIZE_STRICT_A_PLUS_ONLY"

        return {
            "current_streak_type": cur_type,
            "current_streak_count": cur_count,
            "max_win_streak": max_win,
            "max_loss_streak": max_loss,
            "consecutive_losses": consecutive_losses,
            "tilt_detected": tilt_flag,
            "tilt_warning_level": tilt_level,
            "avg_trade_interval_minutes": round(avg_interval, 1),
            "loss_streak_interval_minutes": round(loss_interval_mean, 1),
            "frequency_acceleration_ratio": round(freq_ratio, 2),
            "recommended_action": action
        }

    def generate_performance_report(self) -> Dict[str, Any]:
        """Compiles the full institutional performance and execution audit report."""
        df_sig = self.win_rate_by_signal_type()
        df_time = self.win_rate_by_time_bucket()
        df_regime = self.win_rate_by_regime()
        conf_corr = self.confluence_vs_outcome_correlation()
        streak_tilt = self.streak_and_tilt_analysis()

        total_closed = len(self.closed_entries)
        wins = sum(1 for e in self.closed_entries if e.realized_r_multiple > 0)
        losses = sum(1 for e in self.closed_entries if e.realized_r_multiple <= 0)
        win_rate = round((wins / total_closed) * 100.0, 1) if total_closed > 0 else 0.0

        total_r = round(float(sum(e.realized_r_multiple for e in self.closed_entries)), 2)
        avg_r = round(total_r / total_closed, 2) if total_closed > 0 else 0.0
        total_pnl = round(float(sum(e.realized_pnl_rupees for e in self.closed_entries)), 2)

        gross_gains = sum(e.realized_pnl_rupees for e in self.closed_entries if e.realized_pnl_rupees > 0)
        gross_losses = abs(sum(e.realized_pnl_rupees for e in self.closed_entries if e.realized_pnl_rupees < 0))
        pf = round(gross_gains / gross_losses, 2) if gross_losses > 0 else (99.0 if gross_gains > 0 else 0.0)

        r_arr = np.array([e.realized_r_multiple for e in self.closed_entries], dtype=np.float64) if self.closed_entries else np.array([0.0])
        r_std = float(np.std(r_arr, ddof=1)) if len(r_arr) > 1 else 0.01
        sqn = round(float(np.sqrt(total_closed) * (avg_r / max(r_std, 0.001))), 2) if total_closed > 0 else 0.0

        return {
            "summary": {
                "total_closed_trades": total_closed,
                "winning_trades": wins,
                "losing_trades": losses,
                "win_rate_pct": win_rate,
                "profit_factor": pf,
                "total_r_multiple": total_r,
                "avg_r_multiple": avg_r,
                "total_realized_pnl": total_pnl,
                "system_quality_number_sqn": sqn
            },
            "by_signal_type": df_sig.to_dict(orient="records") if not df_sig.empty else [],
            "by_time_bucket": df_time.to_dict(orient="records") if not df_time.empty else [],
            "by_regime": df_regime.to_dict(orient="records") if not df_regime.empty else [],
            "confluence_correlation": conf_corr,
            "streak_and_tilt": streak_tilt,
            "t1_to_t2_conversion": self.t1_to_t2_conversion_rate(),
            "report_timestamp": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
        }

