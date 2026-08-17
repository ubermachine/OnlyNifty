"""
Unit tests for OnlyNifty v5.2 Telegram Webhook & Alert Dispatcher (src/notifications.py).
"""

import pytest
from unittest.mock import patch, MagicMock
from src.notifications import TelegramNotifier
from src.signal_journal import SignalEntry, SignalLifecycleStatus


@pytest.fixture
def sample_signal_entry():
    return SignalEntry(
        signal_id="SIG-20260815-103000-24500CE",
        timestamp_ist="2026-08-15 10:30:00 IST",
        timestamp_utc_ms=1723708200000,
        bar_timestamp="2026-08-15 10:30",
        spot_price=24520.0,
        signal_type="LONG",
        direction="LONG",
        trigger_reason="Bullish IB Breakout with Heavyweight Confluence",
        selected_strike=24500,
        option_type="CE",
        symbol="NIFTY 24500 CE",
        entry_premium=145.0,
        sl_spot=24480.0,
        sl_premium=118.0,
        sl_points_spot=40.0,
        sl_risk_premium_pts=27.0,
        target_1_spot=24570.0,
        target_1_premium=182.0,
        target_2_spot=24620.0,
        target_2_premium=225.0,
        target_3_spot=24700.0,
        target_3_premium=295.0,
        r_multiple_t1=1.25,
        r_multiple_t2=2.60,
        confluence_score=90.0,
        confluence_grade="A+ Institutional",
        regime_summary="TREND_EXPANSION",
        kalman_velocity=1.85,
        kalman_zscore=2.10,
        markov_regime="LOW_VOL_TRENDING",
        htf_alignment="BULLISH_ALIGNED",
        is_0dte=False,
        lots_suggested=3,
        total_qty=75,
        capital_risk_rupees=2025.0,
        tca_friction_est=62.5
    )


def test_telegram_singleton():
    t1 = TelegramNotifier.get_instance()
    t2 = TelegramNotifier.get_instance()
    assert t1 is t2


def test_format_signal_html(sample_signal_entry):
    notifier = TelegramNotifier.get_instance()
    html = notifier.format_signal_html(sample_signal_entry)
    
    assert "ONLYNIFTY INSTITUTIONAL SIGNAL ALERT" in html
    assert "BUY CE (CALL)" in html
    assert "NIFTY 24500 CE" in html
    assert "₹24,520.00" in html
    assert "₹145.00" in html
    assert "Target 1" in html
    assert "Free Spread Protocol" in html
    assert "A+ Institutional" in html


def test_format_lifecycle_html(sample_signal_entry):
    notifier = TelegramNotifier.get_instance()
    html = notifier.format_lifecycle_html(sample_signal_entry, "T1_REACHED", current_spot=24575.0, current_prem=185.0)
    
    assert "TARGET 1 REACHED" in html
    assert "NIFTY 24500 CE" in html
    assert "₹24,575.00" in html
    assert "₹185.00" in html


def test_deduplication_lock(sample_signal_entry):
    notifier = TelegramNotifier.get_instance()
    notifier._sent_set.clear()
    notifier._sent_keys.clear()

    key = "TEST_DEDUPE_KEY_123"
    assert notifier._is_duplicate(key) is False
    assert notifier._is_duplicate(key) is True  # Second attempt is duplicate


def test_dispatch_signal_without_tokens(sample_signal_entry):
    notifier = TelegramNotifier.get_instance()
    # When no token or chat id is provided, dispatch returns False cleanly without raising
    res = notifier.dispatch_signal_alert(sample_signal_entry, bot_token="", chat_id="", blocking=True)
    assert res is False


@patch("urllib.request.urlopen")
def test_dispatch_signal_mocked_success(mock_urlopen, sample_signal_entry):
    notifier = TelegramNotifier.get_instance()
    notifier._sent_set.clear()
    notifier._sent_keys.clear()

    mock_response = MagicMock()
    mock_response.read.return_value = b'{"ok": true, "result": {"message_id": 1001}}'
    mock_urlopen.return_value.__enter__.return_value = mock_response

    res = notifier.dispatch_signal_alert(
        sample_signal_entry,
        bot_token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
        chat_id="987654321",
        blocking=True
    )
    assert res is True


def test_confluence_floor_filtering(sample_signal_entry):
    notifier = TelegramNotifier.get_instance()
    notifier._sent_set.clear()
    notifier._sent_keys.clear()

    # Sub-70 score on 0-100 scale should be filtered
    sample_signal_entry.confluence_score = 38.0
    res_low = notifier.dispatch_signal_alert(
        sample_signal_entry,
        bot_token="123456:ABC-DEF",
        chat_id="987654321",
        blocking=True
    )
    assert res_low is False

    # Sub-70 score on legacy 0-1 scale (0.50 -> 50.0) should also be filtered
    sample_signal_entry.confluence_score = 0.50
    res_legacy_low = notifier.dispatch_signal_alert(
        sample_signal_entry,
        bot_token="123456:ABC-DEF",
        chat_id="987654321",
        blocking=True
    )
    assert res_legacy_low is False


@patch("urllib.request.urlopen")
def test_dispatch_lifecycle_alert_mocked(mock_urlopen, sample_signal_entry):
    notifier = TelegramNotifier.get_instance()
    notifier._sent_set.clear()
    notifier._sent_keys.clear()

    mock_response = MagicMock()
    mock_response.read.return_value = b'{"ok": true, "result": {"message_id": 1002}}'
    mock_urlopen.return_value.__enter__.return_value = mock_response

    res = notifier.dispatch_lifecycle_alert(
        sample_signal_entry,
        status_event="T1_REACHED",
        current_spot=24570.0,
        current_prem=182.0,
        bot_token="123456:ABC-DEF",
        chat_id="987654321",
        blocking=True
    )
    assert res is True

    # Duplicate lifecycle event for same signal and status should be suppressed
    res_dup = notifier.dispatch_lifecycle_alert(
        sample_signal_entry,
        status_event="T1_REACHED",
        current_spot=24575.0,
        current_prem=185.0,
        bot_token="123456:ABC-DEF",
        chat_id="987654321",
        blocking=True
    )
    assert res_dup is False


@patch("urllib.request.urlopen")
def test_send_test_alert_html_safety(mock_urlopen):
    notifier = TelegramNotifier.get_instance()
    mock_response = MagicMock()
    mock_response.read.return_value = b'{"ok": true, "result": {"message_id": 1003}}'
    mock_urlopen.return_value.__enter__.return_value = mock_response

    success, msg = notifier.send_test_alert("123456:ABC-DEF", "987654321")
    assert success is True
    # Verify the payload text has &lt;50ms and no unescaped <50ms
    call_args = mock_urlopen.call_args
    req = call_args[0][0]
    payload_str = req.data.decode("utf-8")
    assert "<50ms" not in payload_str
    assert "&lt;50ms" in payload_str


def test_format_signal_html_escapes_special_characters(sample_signal_entry):
    notifier = TelegramNotifier.get_instance()
    sample_signal_entry.trigger_reason = "Price < 21 EMA and Vol > 1.5x & Skew < 0"
    html_out = notifier.format_signal_html(sample_signal_entry)
    
    assert "Price &lt; 21 EMA and Vol &gt; 1.5x &amp; Skew &lt; 0" in html_out
