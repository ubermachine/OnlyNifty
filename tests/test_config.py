from src.config import (
    DEFAULT_CAPITAL,
    MAX_RISK_PCT,
    LOT_SIZE,
    EMA_FAST,
    EMA_MID,
    EMA_SLOW,
    ENVELOPE_PCT,
    DELTA_MIN,
    DELTA_MAX,
    FIB_GOLDEN_MIN,
    FIB_GOLDEN_MAX,
    MA_STRETCH_THRESHOLD
)

def test_config_constants():
    assert DEFAULT_CAPITAL == 500000.0
    assert MAX_RISK_PCT == 0.01
    # NSE revised the Nifty contract 25 -> 75 (SEBI Oct-2024 min contract value ~Rs.15L).
    # This must track the live contract master: a stale value silently scales every
    # position size and R-multiple by the ratio of the error.
    assert LOT_SIZE == 75
    assert EMA_FAST == 21
    assert EMA_MID == 55
    assert EMA_SLOW == 200
    assert ENVELOPE_PCT == 0.015
    assert DELTA_MIN == 0.50
    assert DELTA_MAX == 0.65
    assert FIB_GOLDEN_MIN == 0.50
    assert FIB_GOLDEN_MAX == 0.618
    assert MA_STRETCH_THRESHOLD == 0.0035
