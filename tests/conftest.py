"""Shared pytest fixtures.

Chief job: keep the test suite from writing to PRODUCTION state files.

SessionRiskState.__post_init__ defaults persistence_file to data/risk_state_today.json,
so any test calling record_entry/record_exit silently overwrote the live trading session's
risk counters. That is how data/risk_state_today.json came to hold trades_today=1 /
realized_pnl_today=1000.0 — fixture values from test_session_risk_state.py, sitting in the
file the running app loads its circuit breakers from.
"""

import pytest

import src.risk_state as risk_state
import src.options_positioning as options_positioning


@pytest.fixture(autouse=True)
def isolate_state_files(tmp_path, monkeypatch):
    """Redirect every persisted-state path to a per-test temp dir."""
    monkeypatch.setattr(
        risk_state, "DEFAULT_RISK_STATE_PATH", str(tmp_path / "risk_state_today.json"), raising=False
    )
    monkeypatch.setattr(
        options_positioning, "OPTIONS_STATE_PATH", str(tmp_path / "options_state.json"), raising=False
    )
    yield
