"""OnlyNifty Perishable Premium Harvester.

The system's edge was historically measured in *spot* points while it trades decaying
option premium. Fyers serves genuine option OHLCV — but ONLY for contracts that are still
live; expired weeklies return "Invalid symbol". Premium history is therefore perishable:
the only way to build QUOTE-tier evidence is to capture the exact broker contract at signal
time and resolve its outcome from that same contract's history while it is still alive.

Flow:
  1. capture(ticket, ...) at signal time  -> append a PENDING record (contract symbol + the
     real market-quote entry/stop/target premiums) to data/premium_pending.jsonl.
  2. resolve(fetch_history_fn) on every later run -> for each matured, still-live pending,
     fetch the contract's own 5m history, replay the bars strictly AFTER the entry bar via
     the single shared resolver, and bank a realized option R at evidence_tier="QUOTE".
  3. build_edge_table() folds resolved outcomes into QUOTE-tier EdgeStats — the only tier
     allowed to promote a setup to TRUSTED.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import pytz

from src.edge_harness import replay_option_quote_series, WalkForwardRunner, EdgeTable
from src.config import TIME_STOP_BARS

IST = pytz.timezone("Asia/Kolkata")

PENDING_PATH = "data/premium_pending.jsonl"
RESOLVED_PATH = "data/premium_resolved.jsonl"

# A capture matures for resolution once this many 5m bars of the contract's own life have
# elapsed past entry — enough to let the shared resolver's tiers and stall-stop play out.
MATURITY_BARS = TIME_STOP_BARS + 1
BAR_MINUTES = 5


@dataclass
class PendingCapture:
    capture_id: str                 # dedup key: f"{fyers_symbol}@{bar_timestamp}"
    fyers_symbol: str               # NSE:NIFTY...CE — perishable; captured while live
    bar_timestamp: str              # "YYYY-MM-DD HH:MM" of the entry bar (IST)
    captured_at: str                # wall-clock capture time (IST)
    setup_id: str
    regime: str
    direction: str
    entry_premium: float
    sl_premium: float
    target1_premium: float
    target2_premium: float
    target3_premium: float
    expiry_epoch: int = 0
    expiry_label: str = ""          # for durable archive lookup (survives expiry)
    strike: float = 0.0
    option_type: str = ""
    spot_at_entry: float = 0.0
    resolved: bool = False
    outcome_r: Optional[float] = None
    resolved_at: str = ""
    resolve_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PendingCapture":
        known = {k: d.get(k) for k in cls.__dataclass_fields__}  # tolerate schema drift
        return cls(**known)


class PremiumHarvester:
    def __init__(self, pending_path: str = PENDING_PATH, resolved_path: str = RESOLVED_PATH):
        self.pending_path = pending_path
        self.resolved_path = resolved_path

    # ---------------------------------------------------------------- capture
    def capture(
        self,
        ticket: Dict[str, Any],
        setup_id: str,
        regime: str,
        direction: str,
        bar_timestamp: str,
        expiry_epoch: int = 0,
        expiry_label: str = "",
    ) -> Optional[PendingCapture]:
        """Records a live contract for later QUOTE-tier resolution.

        No-op (returns None) unless the ticket carries a genuine broker symbol from live
        MARKET_QUOTE pricing — a theoretical-model ticket has nothing resolvable.
        """
        if not ticket or ticket.get("status") != "READY":
            return None
        fy = str(ticket.get("fyers_symbol") or "")
        if not fy.startswith("NSE:"):
            return None
        if ticket.get("pricing_source") != "MARKET_QUOTE":
            return None

        cap_id = f"{fy}@{bar_timestamp}"
        existing = {c.capture_id for c in self._load_pending()}
        if cap_id in existing:
            return None  # dedup: one capture per contract per entry bar

        rec = PendingCapture(
            capture_id=cap_id,
            fyers_symbol=fy,
            bar_timestamp=bar_timestamp,
            captured_at=datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
            setup_id=setup_id or ticket.get("signal", "UNKNOWN"),
            regime=regime or "UNKNOWN",
            direction=direction,
            entry_premium=float(ticket.get("entry_premium", 0.0)),
            sl_premium=float(ticket.get("sl_premium", 0.0)),
            target1_premium=float(ticket.get("target1_premium", 0.0)),
            target2_premium=float(ticket.get("target2_premium", 0.0)),
            target3_premium=float(ticket.get("target3_moonshot_premium", 0.0)),
            expiry_epoch=int(expiry_epoch or 0),
            expiry_label=str(expiry_label or ""),
            strike=float(ticket.get("strike", 0.0)),
            option_type=str(ticket.get("option_type", "")),
            spot_at_entry=float(ticket.get("spot_entry", 0.0)),
        )
        self._append_jsonl(self.pending_path, rec.to_dict())
        return rec

    # --------------------------------------------------------------- resolve
    def resolve(
        self,
        fetch_history_fn: Optional[Callable[[str, str, str, str], pd.DataFrame]] = None,
        now: Optional[datetime] = None,
        archive: Optional[Any] = None,
    ) -> List[PendingCapture]:
        """Resolves every matured pending from real premium OHLCV.

        Resolution source precedence:
          1. `archive` (PremiumArchive) — durable parquet that SURVIVES expiry; preferred.
          2. `fetch_history_fn` — live re-fetch; works only while the contract is alive.

        Passing the archive is what lets a trade be resolved even after its contract expires,
        because the candles were persisted to disk while it was still live.
        Returns the list of records newly resolved on this pass.
        """
        now = now or datetime.now(IST)
        pendings = self._load_pending()
        if not pendings:
            return []

        newly: List[PendingCapture] = []
        changed = False
        for rec in pendings:
            if rec.resolved:
                continue
            if not self._is_mature(rec, now):
                continue

            expired = rec.expiry_epoch > 0 and now.timestamp() >= rec.expiry_epoch
            entry_dt = self._parse_bar(rec.bar_timestamp)
            if entry_dt is None:
                rec.resolved, rec.resolve_note = True, "unparseable bar_timestamp"
                changed = True
                continue

            fwd = None
            # 1. Durable archive first (survives expiry).
            if archive is not None and rec.expiry_label and rec.option_type and rec.strike:
                try:
                    w = archive.load_window(rec.expiry_label, rec.strike, rec.option_type, entry_dt, bars=0)
                    if w is not None and not w.empty:
                        fwd = w
                except Exception:
                    fwd = None

            # 2. Live re-fetch fallback (only meaningful while contract is alive).
            if (fwd is None or fwd.empty) and fetch_history_fn is not None:
                try:
                    hist = fetch_history_fn(
                        rec.fyers_symbol, "5",
                        entry_dt.date().isoformat(),
                        (now + timedelta(days=1)).date().isoformat(),
                    )
                    fwd = self._forward_bars(hist, entry_dt)
                except Exception as ex:
                    if expired:
                        rec.resolved, rec.outcome_r, rec.resolve_note = True, None, f"contract expired unresolved: {str(ex)[:60]}"
                        changed = True
                    continue
            if fwd is None or fwd.empty:
                if expired:
                    rec.resolved, rec.outcome_r, rec.resolve_note = True, None, "no forward bars before expiry"
                    changed = True
                continue

            outcome = replay_option_quote_series(
                fwd, rec.entry_premium, rec.sl_premium,
                rec.target1_premium, rec.target2_premium, rec.target3_premium,
            )
            if outcome is None and not expired:
                continue  # still developing; try again next run

            rec.resolved = True
            rec.outcome_r = outcome
            rec.resolved_at = now.strftime("%Y-%m-%d %H:%M:%S IST")
            rec.resolve_note = f"resolved on {len(fwd)} forward bars" + ("" if not expired else " (expiry-forced)")
            self._append_jsonl(self.resolved_path, rec.to_dict())
            newly.append(rec)
            changed = True

        if changed:
            self._rewrite_pending(pendings)
        return newly

    # ------------------------------------------------------- edge aggregation
    def build_edge_table(self) -> EdgeTable:
        """Aggregates resolved QUOTE outcomes into QUOTE-tier EdgeStats by (setup, regime)."""
        runner = WalkForwardRunner()
        groups: Dict[tuple, List[float]] = {}
        for rec in self._load_pending():
            if rec.resolved and rec.outcome_r is not None:
                groups.setdefault((rec.setup_id, rec.regime), []).append(rec.outcome_r)
        records = [
            runner.compute_edge_stats(r_list, sid, reg, evidence_tier="QUOTE")
            for (sid, reg), r_list in groups.items()
        ]
        return EdgeTable(records)

    def summary(self) -> Dict[str, Any]:
        pend = self._load_pending()
        resolved = [p for p in pend if p.resolved and p.outcome_r is not None]
        return {
            "total_captured": len(pend),
            "resolved": len(resolved),
            "open": sum(1 for p in pend if not p.resolved),
            "quote_mean_r": round(sum(p.outcome_r for p in resolved) / len(resolved), 3) if resolved else 0.0,
            "quote_win_rate": round(100.0 * sum(1 for p in resolved if p.outcome_r > 0) / len(resolved), 1) if resolved else 0.0,
        }

    # ------------------------------------------------------------- internals
    def _is_mature(self, rec: PendingCapture, now: datetime) -> bool:
        entry_dt = self._parse_bar(rec.bar_timestamp)
        if entry_dt is None:
            return True  # let resolve() close it out
        return now >= entry_dt + timedelta(minutes=BAR_MINUTES * MATURITY_BARS)

    @staticmethod
    def _parse_bar(bar_ts: str) -> Optional[datetime]:
        try:
            dt = pd.to_datetime(bar_ts)
            dt = dt.tz_localize(IST) if dt.tzinfo is None else dt.tz_convert(IST)
            return dt.to_pydatetime()
        except Exception:
            return None

    @staticmethod
    def _forward_bars(hist: pd.DataFrame, entry_dt: datetime) -> Optional[pd.DataFrame]:
        if hist is None or hist.empty:
            return None
        idx = hist.index
        try:
            if idx.tz is None:
                hist = hist.copy()
                hist.index = idx.tz_localize(IST)
        except Exception:
            return None
        return hist[hist.index > entry_dt]

    def _load_pending(self) -> List[PendingCapture]:
        if not os.path.exists(self.pending_path):
            return []
        out: List[PendingCapture] = []
        with open(self.pending_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(PendingCapture.from_dict(json.loads(line)))
                    except Exception:
                        continue
        return out

    def _rewrite_pending(self, records: List[PendingCapture]) -> None:
        self._ensure_dir(self.pending_path)
        tmp = f"{self.pending_path}.tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r.to_dict()) + "\n")
        os.replace(tmp, self.pending_path)

    def _append_jsonl(self, path: str, obj: Dict[str, Any]) -> None:
        self._ensure_dir(path)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")

    @staticmethod
    def _ensure_dir(path: str) -> None:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
