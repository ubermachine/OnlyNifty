"""
OnlyNifty v5.2 Asynchronous Telegram Webhook & Institutional Alert Dispatcher.

Features:
- High-performance, non-blocking ThreadPoolExecutor (<50ms dispatch overhead).
- Thread-safe LRU ring-buffer deduplication to prevent duplicate alerts during auto-refresh.
- High-density HTML risk card formatting (Spot, Strike, Premium, SL, 3-Tier Targets, Greeks, TCA).
- Real-time Trade Lifecycle update dispatches (T1 Reached, T2 Reached, Moonshot, Stop Loss Hit).
- Safe timeout resilience & one-click UI test alert capability.
"""

import os
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any, Tuple, Set
import urllib.request
import urllib.parse
import urllib.error
from collections import deque

from src.config import (
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_PARSE_MODE,
    TELEGRAM_TIMEOUT_SECONDS, TELEGRAM_MIN_CONFLUENCE_SCORE, LOT_SIZE
)

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Singleton institutional Telegram alert dispatcher for live trade signals."""

    _instance: Optional["TelegramNotifier"] = None
    _init_lock = threading.Lock()

    def __init__(self, max_workers: int = 2):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tg_notify")
        self._sent_keys: deque = deque(maxlen=200)
        self._sent_set: Set[str] = set()
        self._lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "TelegramNotifier":
        """Thread-safe singleton accessor."""
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _is_duplicate(self, dedupe_key: str) -> bool:
        """Check and record deduplication key under thread safety."""
        with self._lock:
            if dedupe_key in self._sent_set:
                return True
            if len(self._sent_keys) >= self._sent_keys.maxlen:
                oldest = self._sent_keys.popleft()
                self._sent_set.discard(oldest)
            self._sent_keys.append(dedupe_key)
            self._sent_set.add(dedupe_key)
            return False

    def _send_telegram_post(self, token: str, chat_id: str, text: str, parse_mode: str = "HTML") -> Tuple[bool, str]:
        """Synchronous HTTP POST to Telegram Bot API with timeout."""
        if not token or not chat_id:
            return False, "Telegram Bot Token or Chat ID not configured."

        url = f"https://api.telegram.org/bot{token.strip()}/sendMessage"
        payload = {
            "chat_id": chat_id.strip(),
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "OnlyNifty-v52-Bot/1.0"}
        )

        try:
            with urllib.request.urlopen(req, timeout=TELEGRAM_TIMEOUT_SECONDS) as response:
                res_body = response.read().decode("utf-8")
                res_json = json.loads(res_body)
                if res_json.get("ok"):
                    return True, "Alert dispatched successfully."
                return False, f"Telegram API Error: {res_json.get('description', 'Unknown error')}"
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8", errors="ignore")
            logger.error("Telegram HTTPError %s: %s", e.code, err_msg)
            return False, f"HTTP Error {e.code}: {err_msg}"
        except Exception as e:
            logger.error("Telegram dispatch failed: %s", str(e))
            return False, f"Network Exception: {str(e)}"

    def format_signal_html(self, entry: Any) -> str:
        """Constructs high-density institutional HTML trade alert card."""
        is_long = getattr(entry, "direction", "WAIT") == "LONG"
        action_emoji = "🟢" if is_long else "🔴"
        action_text = "BUY CE (CALL)" if is_long else "BUY PE (PUT)"
        grade = getattr(entry, "confluence_grade", "A Standard")
        conf_score = getattr(entry, "confluence_score", 0.0)
        conf_pct = conf_score * 100 if conf_score <= 1.0 else conf_score

        spot = getattr(entry, "spot_price", 0.0)
        strike = getattr(entry, "selected_strike", 0)
        opt_type = getattr(entry, "option_type", "CE")
        premium = getattr(entry, "entry_premium", 0.0)
        
        sl_spot = getattr(entry, "sl_spot", 0.0)
        sl_prem = getattr(entry, "sl_premium", 0.0)
        t1_spot = getattr(entry, "target_1_spot", 0.0)
        t1_prem = getattr(entry, "target_1_premium", 0.0)
        t2_spot = getattr(entry, "target_2_spot", 0.0)
        t2_prem = getattr(entry, "target_2_premium", 0.0)
        t3_spot = getattr(entry, "target_3_spot", 0.0)
        t3_prem = getattr(entry, "target_3_premium", 0.0)
        
        lots = getattr(entry, "lots_suggested", 2)
        qty = getattr(entry, "total_qty", lots * LOT_SIZE)
        risk_rs = getattr(entry, "capital_risk_rupees", 0.0)
        tca_rs = getattr(entry, "tca_friction_est", 45.0)
        ts_ist = getattr(entry, "timestamp_ist", "")
        reason = getattr(entry, "trigger_reason", "")
        regime = getattr(entry, "regime_summary", "CONFLUENCE")
        is_0dte = getattr(entry, "is_0dte", False)

        otm_hedge_strike = strike + 100 if is_long else strike - 100

        html = f"""<b>{action_emoji} ONLYNIFTY INSTITUTIONAL SIGNAL ALERT</b>
━━━━━━━━━━━━━━━━━━━━
<b>Action:</b> <code>{action_text}</code>
<b>Contract:</b> <code>NIFTY {strike} {opt_type}</code> {'🔥 <i>[0DTE]</i>' if is_0dte else ''}
<b>Grade:</b> <b>{grade}</b> (Confluence: <b>{conf_pct:.0f}%</b>)
<b>Regime:</b> <code>{regime}</code>

<b>📍 Execution Parameters:</b>
• <b>Spot Entry:</b> ₹{spot:,.2f}
• <b>Est. Premium:</b> ₹{premium:,.2f}
• <b>Recommended Sizing:</b> {lots} Lots ({qty} Qty)
• <b>Capital at Risk:</b> ₹{risk_rs:,.2f} (Est. TCA: ₹{tca_rs:.1f})

<b>🛑 Stop Loss:</b>
• <b>Spot SL:</b> ₹{sl_spot:,.2f} | <b>Prem SL:</b> ₹{sl_prem:,.2f}

<b>🎯 Asymmetric Targets:</b>
• <b>Target 1 (1.2R):</b> Spot ₹{t1_spot:,.2f} (Prem ₹{t1_prem:,.2f})
• <b>Target 2 (2.5R):</b> Spot ₹{t2_spot:,.2f} (Prem ₹{t2_prem:,.2f})
• <b>Target 3 (Moonshot):</b> Spot ₹{t3_spot:,.2f} (Prem ₹{t3_prem:,.2f})

💡 <b>Free Spread Protocol:</b> At Target 1, sell <code>NIFTY {otm_hedge_strike} {opt_type}</code> to lock +θ and zero risk.
<b>Rationale:</b> <i>{reason}</i>
━━━━━━━━━━━━━━━━━━━━
⏰ <i>{ts_ist} | Generated by OnlyNifty v5.2</i>"""
        return html

    def format_lifecycle_html(self, entry: Any, status_event: str, current_spot: float, current_prem: float) -> str:
        """Constructs trade lifecycle milestone notification card."""
        strike = getattr(entry, "selected_strike", 0)
        opt_type = getattr(entry, "option_type", "CE")
        r_mult = getattr(entry, "realized_r_multiple", 0.0)
        pnl_rs = getattr(entry, "realized_pnl_rupees", 0.0)

        if status_event == "T1_REACHED":
            emoji = "🎯"
            title = "TARGET 1 REACHED (+1.2R)"
            note = "Lock in partial profits & activate Free Spread Converter."
        elif status_event == "T2_REACHED":
            emoji = "🚀"
            title = "TARGET 2 REACHED (+2.5R)"
            note = "Trail Stop Loss to T1. Let runner ride to Target 3 Moonshot."
        elif status_event == "T3_MOONSHOT":
            emoji = "🌕"
            title = "T3 MOONSHOT COMPLETED"
            note = "Full target achieved. Close out remaining runner lots."
        elif status_event == "STOPPED_OUT":
            emoji = "🛑"
            title = "STOP LOSS TRIGGERED"
            note = "Capital defense protocol engaged. Trade closed."
        else:
            emoji = "ℹ️"
            title = f"LIFECYCLE UPDATE: {status_event}"
            note = ""

        html = f"""<b>{emoji} ONLYNIFTY TRADE UPDATE: {title}</b>
━━━━━━━━━━━━━━━━━━━━
<b>Position:</b> <code>NIFTY {strike} {opt_type}</code>
<b>Current Spot:</b> ₹{current_spot:,.2f}
<b>Current Premium:</b> ₹{current_prem:,.2f}
<b>Realized R-Multiple:</b> <b>{r_mult:+.2f}R</b>
<b>Net PnL:</b> <b>{'₹{:,.2f}'.format(pnl_rs) if pnl_rs != 0.0 else 'Active / Locked'}</b>

<i>{note}</i>
━━━━━━━━━━━━━━━━━━━━
⏰ <i>OnlyNifty v5.2 Lifecycle Engine</i>"""
        return html

    def dispatch_signal_alert(
        self,
        entry: Any,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        blocking: bool = False
    ) -> bool:
        """
        Dispatches a signal alert asynchronously or synchronously.
        Enforces deduplication, confluence floor, and non-actionable filtering.
        """
        token = (bot_token or TELEGRAM_BOT_TOKEN or os.getenv("TELEGRAM_BOT_TOKEN", "")).strip()
        cid = (chat_id or TELEGRAM_CHAT_ID or os.getenv("TELEGRAM_CHAT_ID", "")).strip()

        if not token or not cid:
            return False

        # Filter non-actionable or low confluence
        direction = getattr(entry, "direction", "WAIT")
        if direction in ["WAIT", "NEUTRAL"]:
            return False

        conf = float(getattr(entry, "confluence_score", 0.0))
        if 0.0 < conf <= 1.0:
            conf = conf * 100.0

        if conf < TELEGRAM_MIN_CONFLUENCE_SCORE:
            return False

        # Deduplication key: signal_id + bar_timestamp
        sig_id = getattr(entry, "signal_id", "")
        bar_ts = getattr(entry, "bar_timestamp", "")
        dedupe_key = f"{sig_id}_{bar_ts}_{direction}"

        if self._is_duplicate(dedupe_key):
            logger.debug("Suppressed duplicate Telegram alert for key: %s", dedupe_key)
            return False

        text = self.format_signal_html(entry)

        if blocking:
            success, msg = self._send_telegram_post(token, cid, text)
            return success
        else:
            self.executor.submit(self._send_telegram_post, token, cid, text)
            return True

    def dispatch_lifecycle_alert(
        self,
        entry: Any,
        status_event: str,
        current_spot: float,
        current_prem: float,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        blocking: bool = False
    ) -> bool:
        """Dispatches a trade lifecycle transition alert."""
        token = (bot_token or TELEGRAM_BOT_TOKEN or os.getenv("TELEGRAM_BOT_TOKEN", "")).strip()
        cid = (chat_id or TELEGRAM_CHAT_ID or os.getenv("TELEGRAM_CHAT_ID", "")).strip()

        if not token or not cid:
            return False

        sig_id = getattr(entry, "signal_id", "")
        dedupe_key = f"LIFECYCLE_{sig_id}_{status_event}"
        if self._is_duplicate(dedupe_key):
            return False

        text = self.format_lifecycle_html(entry, status_event, current_spot, current_prem)

        if blocking:
            success, msg = self._send_telegram_post(token, cid, text)
            return success
        else:
            self.executor.submit(self._send_telegram_post, token, cid, text)
            return True

    def send_test_alert(self, bot_token: str, chat_id: str) -> Tuple[bool, str]:
        """Synchronous ping to verify credentials from the UI."""
        test_msg = """<b>🚀 ONLYNIFTY TELEGRAM WEBHOOK TEST</b>
━━━━━━━━━━━━━━━━━━━━
<b>Status:</b> 🟢 <b>Connected & Active</b>
<b>Version:</b> <code>OnlyNifty v5.2 Institutional Edition</code>
<b>Features Enabled:</b>
• Non-Blocking Async Dispatch (<50ms)
• Thread-Safe Ring-Buffer Deduplication
• 3-Tier Targets & Free Spread Converter
• Real-Time Trade Lifecycle State Machine

<i>Your Telegram alert stream is configured and ready for live trading!</i>"""
        return self._send_telegram_post(bot_token, chat_id, test_msg)
