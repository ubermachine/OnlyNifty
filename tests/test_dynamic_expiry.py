import pytest
from datetime import datetime, timezone, timedelta
import pytz
import pandas as pd
from src.options_engine import calculate_time_to_expiry_days
from src.config import NIFTY_WEEKLY_EXPIRY_WEEKDAY

IST = pytz.timezone("Asia/Kolkata")


def test_calculate_time_to_expiry_from_epoch():
    now = datetime(2026, 8, 18, 10, 0, 0, tzinfo=IST)
    expiry_dt = datetime(2026, 8, 20, 15, 30, 0, tzinfo=IST)
    expiry_epoch = int(expiry_dt.timestamp())
    
    t_days = calculate_time_to_expiry_days(timestamp=now, expiry_epoch=expiry_epoch)
    expected_days = (expiry_dt - now).total_seconds() / 86400.0
    assert abs(t_days - expected_days) < 0.01


def test_calculate_time_to_expiry_from_date_str():
    now = datetime(2026, 8, 18, 9, 30, 0, tzinfo=IST)
    t_days = calculate_time_to_expiry_days(timestamp=now, expiry_date="2026-08-18")
    assert 0.20 <= t_days <= 0.30


def test_calculate_time_to_expiry_weekday_fallback():
    tue_10am = datetime(2026, 8, 18, 10, 0, 0, tzinfo=IST)
    t_days = calculate_time_to_expiry_days(timestamp=tue_10am, expiry_weekday=1)
    assert 0.20 <= t_days <= 0.25
