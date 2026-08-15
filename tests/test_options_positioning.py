"""Comprehensive unit tests for Options Desk Positioning Engine (src/options_positioning.py)."""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from src.options_positioning import (
    OptionsDeskState,
    compute_options_desk_state,
    clamp_targets_to_corridor,
    load_options_history,
    save_options_history
)
from src.data_engine import DataEngine

@pytest.fixture
def sample_df_5m():
    dates = pd.date_range("2026-08-15 09:15", periods=50, freq="5min", tz="Asia/Kolkata")
    np.random.seed(42)
    close = 24500.0 + np.cumsum(np.random.randn(50) * 8.0)
    df = pd.DataFrame({
        "open": close - 2.0,
        "high": close + 5.0,
        "low": close - 5.0,
        "close": close,
        "volume": np.random.randint(10000, 50000, size=50)
    }, index=dates)
    return df

@pytest.fixture
def sample_option_chain():
    de = DataEngine()
    return de.fetch_live_nse_option_chain("NIFTY")

def test_compute_options_desk_state_structure(sample_option_chain, sample_df_5m):
    spot = 24500.0
    state = compute_options_desk_state(
        option_chain_df=sample_option_chain,
        spot=spot,
        df_ohlcv=sample_df_5m,
        live_iv=0.14,
        hfi_score=0.10
    )
    assert isinstance(state, OptionsDeskState)
    assert state.put_wall > 0
    assert state.call_wall > state.put_wall
    assert state.data_quality in ["VERIFIED", "POSITIONING_UNVERIFIED"]
    assert -1.0 <= state.d_vector <= 1.0
    assert state.expected_move_pts > 0

def test_clamp_targets_to_corridor_long():
    entry = 24500.0
    t1 = 24550.0
    t2 = 24700.0  # Exceeds call wall 24600
    call_wall = 24600.0
    put_wall = 24400.0
    
    clean_t1, clean_t2, sl_hint = clamp_targets_to_corridor(
        entry=entry,
        t1=t1,
        t2=t2,
        direction="LONG",
        put_wall=put_wall,
        call_wall=call_wall
    )
    assert clean_t2 <= call_wall
    assert clean_t1 <= clean_t2
    assert sl_hint < entry

def test_clamp_targets_to_corridor_short():
    entry = 24500.0
    t1 = 24450.0
    t2 = 24300.0  # Below put wall 24400
    call_wall = 24600.0
    put_wall = 24400.0
    
    clean_t1, clean_t2, sl_hint = clamp_targets_to_corridor(
        entry=entry,
        t1=t1,
        t2=t2,
        direction="SHORT",
        put_wall=put_wall,
        call_wall=call_wall
    )
    assert clean_t2 >= put_wall
    assert clean_t1 >= clean_t2
    assert sl_hint > entry

def test_pcr_history_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = os.path.join(tmpdir, "test_options_state.json")
        
        sample_history = {
            "pcr_series": [1.05, 1.08, 1.12, 1.15, 1.20],
            "max_pain_series": [24500.0, 24500.0, 24550.0]
        }
        save_options_history(
            history=sample_history,
            path=state_file
        )
        
        loaded = load_options_history(path=state_file)
        assert len(loaded["pcr_series"]) == 5
        assert loaded["pcr_series"][-1] == 1.20
        assert len(loaded["max_pain_series"]) == 3

def test_compute_delta_weighted_volume(sample_option_chain):
    from src.options_positioning import compute_delta_weighted_volume
    dwv = compute_delta_weighted_volume(sample_option_chain, spot=24500.0, live_iv=0.14)
    assert -1.0 <= dwv <= 1.0
    assert isinstance(dwv, float)

def test_compute_zero_gamma_level(sample_option_chain):
    from src.options_positioning import compute_zero_gamma_level
    zg = compute_zero_gamma_level(sample_option_chain, spot=24500.0, live_iv=0.14)
    assert "zero_gex_strike" in zg
    assert "gamma_regime" in zg
    assert "is_positive_gamma" in zg
    assert zg["zero_gex_strike"] > 0

