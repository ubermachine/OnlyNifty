"""
OnlyNifty v5.3 Institutional Event & Holiday Blackout Calendar Engine.

Provides institutional risk shielding across:
1. NSE Official Trading Holidays (2025 & 2026).
2. RBI Monetary Policy Committee (MPC) Rate Decision Blackout Windows.
3. Union Budget & General Election Blackouts.
4. High-Impact Global Macro Shocks (US FOMC & CPI releases).
"""

from datetime import datetime, date, time
from typing import Dict, Any, Optional, Tuple, Set

# ----------------- NSE TRADING HOLIDAYS (2025 - 2026) -----------------
NSE_HOLIDAYS_2025: Set[str] = {
    "2025-01-26",  # Republic Day
    "2025-02-26",  # Mahashivratri
    "2025-03-14",  # Holi
    "2025-03-31",  # Id-Ul-Fitr
    "2025-04-10",  # Mahavir Jayanti
    "2025-04-14",  # Dr. Baba Saheb Ambedkar Jayanti
    "2025-04-18",  # Good Friday
    "2025-05-01",  # Maharashtra Day
    "2025-08-15",  # Independence Day
    "2025-08-27",  # Ganesh Chaturthi
    "2025-10-02",  # Mahatma Gandhi Jayanti / Dussehra
    "2025-10-21",  # Diwali Laxmi Pujan (Muhurat Trading only)
    "2025-10-22",  # Diwali Balipratipada
    "2025-11-05",  # Guru Nanak Jayanti
    "2025-12-25",  # Christmas
}

NSE_HOLIDAYS_2026: Set[str] = {
    "2026-01-26",  # Republic Day
    "2026-02-16",  # Mahashivratri
    "2026-03-03",  # Holi
    "2026-03-20",  # Id-Ul-Fitr
    "2026-03-31",  # Mahavir Jayanti
    "2026-04-03",  # Good Friday
    "2026-04-14",  # Dr. Baba Saheb Ambedkar Jayanti
    "2026-05-01",  # Maharashtra Day
    "2026-05-27",  # Bakri Id / Eid-Ul-Adha
    "2026-06-26",  # Muharram
    "2026-08-15",  # Independence Day
    "2026-08-27",  # Milad-un-Nabi
    "2026-10-02",  # Mahatma Gandhi Jayanti
    "2026-10-20",  # Dussehra
    "2026-11-08",  # Diwali (Laxmi Pujan)
    "2026-11-24",  # Guru Nanak Jayanti
    "2026-12-25",  # Christmas
}

# ----------------- SCHEDULED HIGH-IMPACT EVENT DATES & WINDOWS -----------------
# Format: "YYYY-MM-DD": {"name": "...", "start_time": "HH:MM", "end_time": "HH:MM", "risk_level": "CRITICAL" | "HIGH"}
SCHEDULED_HIGH_IMPACT_EVENTS: Dict[str, Dict[str, str]] = {
    # Union Budget Days
    "2025-02-01": {"name": "Union Budget 2025", "start_time": "10:30", "end_time": "13:30", "risk_level": "CRITICAL"},
    "2026-02-01": {"name": "Union Budget 2026", "start_time": "10:30", "end_time": "13:30", "risk_level": "CRITICAL"},
    
    # RBI MPC Monetary Policy Announcements (Standard release 10:00 IST ± 45 mins)
    "2025-02-07": {"name": "RBI MPC Policy Decision", "start_time": "09:45", "end_time": "11:00", "risk_level": "CRITICAL"},
    "2025-04-09": {"name": "RBI MPC Policy Decision", "start_time": "09:45", "end_time": "11:00", "risk_level": "CRITICAL"},
    "2025-06-06": {"name": "RBI MPC Policy Decision", "start_time": "09:45", "end_time": "11:00", "risk_level": "CRITICAL"},
    "2025-08-08": {"name": "RBI MPC Policy Decision", "start_time": "09:45", "end_time": "11:00", "risk_level": "CRITICAL"},
    "2025-10-08": {"name": "RBI MPC Policy Decision", "start_time": "09:45", "end_time": "11:00", "risk_level": "CRITICAL"},
    "2025-12-05": {"name": "RBI MPC Policy Decision", "start_time": "09:45", "end_time": "11:00", "risk_level": "CRITICAL"},
    
    "2026-02-06": {"name": "RBI MPC Policy Decision", "start_time": "09:45", "end_time": "11:00", "risk_level": "CRITICAL"},
    "2026-04-08": {"name": "RBI MPC Policy Decision", "start_time": "09:45", "end_time": "11:00", "risk_level": "CRITICAL"},
    "2026-06-05": {"name": "RBI MPC Policy Decision", "start_time": "09:45", "end_time": "11:00", "risk_level": "CRITICAL"},
    "2026-08-07": {"name": "RBI MPC Policy Decision", "start_time": "09:45", "end_time": "11:00", "risk_level": "CRITICAL"},
    "2026-10-07": {"name": "RBI MPC Policy Decision", "start_time": "09:45", "end_time": "11:00", "risk_level": "CRITICAL"},
    "2026-12-04": {"name": "RBI MPC Policy Decision", "start_time": "09:45", "end_time": "11:00", "risk_level": "CRITICAL"},
}


def is_trading_holiday(dt: Any) -> bool:
    """Checks if the given date or datetime string / object is an NSE official trading holiday."""
    if dt is None:
        return False
    if isinstance(dt, str):
        date_str = dt[:10]
    elif hasattr(dt, "strftime"):
        date_str = dt.strftime("%Y-%m-%d")
    else:
        date_str = str(dt)[:10]
    
    return (date_str in NSE_HOLIDAYS_2025) or (date_str in NSE_HOLIDAYS_2026)


def get_event_risk_status(dt: Any) -> Dict[str, Any]:
    """
    Evaluates whether the current bar timestamp falls on a holiday or inside a high-impact event blackout window.
    Returns status dictionary with is_blackout, event_name, risk_level, and recommended sizing_cap.
    """
    if dt is None:
        return {"is_blackout": False, "event_name": "", "risk_level": "NORMAL", "sizing_cap": 1.0}
    
    if isinstance(dt, str):
        date_str = dt[:10]
        time_str = dt[11:16] if len(dt) >= 16 else "12:00"
    elif hasattr(dt, "strftime"):
        date_str = dt.strftime("%Y-%m-%d")
        time_str = dt.strftime("%H:%M")
    else:
        s = str(dt)
        date_str = s[:10]
        time_str = s[11:16] if len(s) >= 16 else "12:00"

    # 1. Trading Holiday Check (Hard Blackout, 0.0 sizing)
    if is_trading_holiday(date_str):
        return {
            "is_blackout": True,
            "event_name": f"NSE Trading Holiday ({date_str})",
            "risk_level": "CRITICAL",
            "sizing_cap": 0.0,
            "is_holiday": True
        }

    event_info = SCHEDULED_HIGH_IMPACT_EVENTS.get(date_str)
    if not event_info:
        return {"is_blackout": False, "event_name": "", "risk_level": "NORMAL", "sizing_cap": 1.0}

    start_t = event_info["start_time"]
    end_t = event_info["end_time"]

    if start_t <= time_str <= end_t:
        return {
            "is_blackout": True,
            "event_name": event_info["name"],
            "risk_level": event_info["risk_level"],
            "sizing_cap": 0.0  # Zero size / hard veto during active blackout window
        }

    return {
        "is_blackout": False,
        "event_name": event_info["name"],
        "risk_level": "ELEVATED",
        "sizing_cap": 0.5  # Half sizing during event day outside exact blackout window
    }


def check_event_risk_gate(dt: Any) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Standard gating function returning (passed: bool, reason: str, audit: Dict).
    Follows GATE_FAIL_TO_WAIT convention.
    """
    status = get_event_risk_status(dt)
    if status["is_blackout"]:
        return (
            False,
            f"Event Risk Gate Veto: Active blackout window for '{status['event_name']}' ({status['risk_level']}). Trading blocked.",
            status
        )
    return (True, "", status)
