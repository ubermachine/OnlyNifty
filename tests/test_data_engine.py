import pandas as pd
import pytest
from src.data_engine import DataEngine

def test_data_engine_synthetic_ohlcv():
    engine = DataEngine(use_cache=False)
    df = engine.generate_synthetic_nifty(bars=100, interval_mins=5)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    for col in ["open", "high", "low", "close", "volume"]:
        assert col in df.columns
    assert (df["high"] >= df["low"]).all()
    assert (df["high"] >= df["open"]).all()
    assert (df["high"] >= df["close"]).all()
    assert (df["low"] <= df["open"]).all()
    assert (df["low"] <= df["close"]).all()

def test_clean_and_localize_df():
    engine = DataEngine(use_cache=False)
    raw_df = pd.DataFrame({
        "Open": [24500.0, 24520.0],
        "High": [24550.0, 24560.0],
        "Low": [24480.0, 24510.0],
        "Close": [24520.0, 24540.0],
        "Volume": [10000, 12000]
    }, index=pd.date_range("2026-08-13 09:15", periods=2, freq="5min"))
    cleaned = engine.clean_ohlcv(raw_df)
    assert "open" in cleaned.columns
    assert "volume" in cleaned.columns
    assert str(cleaned.index.tz) in ["Asia/Kolkata", "UTC+05:30"] or cleaned.index.tz is not None

def test_participant_oi_snapshot():
    engine = DataEngine(use_cache=False)
    df_oi = engine.get_participant_oi_snapshot()
    assert isinstance(df_oi, pd.DataFrame)
    assert "FII" in df_oi.index
    assert "Pro (Prop Desks)" in df_oi.index
    assert "Futures Long" in df_oi.columns
