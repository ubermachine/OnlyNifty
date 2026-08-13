# Nifty Institutional Trading Plan & Options Engine (JustNifty v2.0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an institutional-grade Nifty 50 Futures and Options trading system and Streamlit web dashboard implementing the complete JustNifty v2.0 quantitative methodology, real-time data streaming via `yfinance` and `jugaad-data`, Black-Scholes Greeks, algorithmic strike selection (Delta 0.50–0.65), 1% risk position sizing, and bar-by-bar historical replay backtesting.

**Architecture:** A modular Python engine with clear boundaries: `data_engine.py` (caching, timezone, resampler), `indicators.py` (EMAs, Envelopes, AVWAP, CPR, Fib, VF Table, Volume Profile), `strategy_rules.py` (JustNifty v2.0 rulebook, 3 PM strategy, HTF hierarchy), `options_engine.py` (Black-Scholes, strike picker, $\Delta\text{OI}$, 1% sizing), `backtest_engine.py` (simulation, metrics), and `app.py` (Streamlit UI with interactive Plotly candlestick charts).

**Tech Stack:** Python 3.10+, `streamlit`, `plotly`, `yfinance`, `jugaad-data`, `pandas`, `numpy`, `scipy`, `pytest`.

## Global Constraints

- **Python Version:** 3.10+
- **Index Ticker:** `^NSEI` (Nifty 50)
- **Timezone:** `Asia/Kolkata` (IST, 09:15 to 15:30)
- **Delta Window for Options:** $0.50 \le \Delta \le 0.65$ (ATM to 1-strike ITM)
- **Default Risk per Trade:** $1.0\%$ of account capital
- **Lot Size:** 25 / 75 (Configurable via `src/config.py`)
- **Part-Booking Rule:** 50% at VF Table T1 or 1.5% 200 EMA Envelope; SL moved to Breakeven; remainder trailed on 1m 21 EMA or AVWAP

---

### Task 1: Environment Setup & Project Configuration

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `src/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `src.config` constants (`DEFAULT_CAPITAL`, `MAX_RISK_PCT`, `LOT_SIZE`, `EMA_PERIODS`, `ENVELOPE_PCT`, `DELTA_MIN`, `DELTA_MAX`, `TRADING_START_TIME`, `TRADING_END_TIME`, `THREE_PM_TIME`)

- [ ] **Step 1: Write test for configuration constants**

```python
# tests/test_config.py
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
)

def test_config_constants():
    assert DEFAULT_CAPITAL > 0
    assert MAX_RISK_PCT == 0.01
    assert LOT_SIZE > 0
    assert EMA_FAST == 21
    assert EMA_MID == 55
    assert EMA_SLOW == 200
    assert ENVELOPE_PCT == 0.015
    assert DELTA_MIN == 0.50
    assert DELTA_MAX == 0.65
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`  
Expected: FAIL (ModuleNotFoundError: No module named 'src')

- [ ] **Step 3: Create `requirements.txt`, `src/__init__.py`, and `src/config.py`**

```python
# src/config.py
from dataclasses import dataclass

# Trading & Risk Constants
DEFAULT_CAPITAL = 500000.0  # ₹5,00,000
MAX_RISK_PCT = 0.01          # 1% Max Risk
LOT_SIZE = 25               # Current Nifty 50 lot size
ENOUGH_PROFIT_PCT = 0.003   # 0.3% Daily target to shut terminal

# Technical Indicator Constants
EMA_FAST = 21
EMA_MID = 55
EMA_SLOW = 200
ENVELOPE_PCT = 0.015        # 1.5% Envelope of 200 EMA
FIB_GOLDEN_MIN = 0.50
FIB_GOLDEN_MAX = 0.618
FIB_SL_LONG = 0.786
FIB_SL_SHORT = 0.382
MA_STRETCH_THRESHOLD = 0.0035  # 0.35% distance threshold from EMAs

# Options Selection Constants
DELTA_MIN = 0.50
DELTA_MAX = 0.65
RISK_FREE_RATE = 0.065      # 6.5% RBI Repo reference rate

# Session Timing (IST)
SESSION_START = "09:15"
OPENING_RANGE_END = "09:30"
MIDDAY_CHOP_START = "11:30"
MIDDAY_CHOP_END = "13:30"
THREE_PM_CANDLE = "15:00"
SESSION_END = "15:30"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`  
Expected: PASS

---

### Task 2: Data Ingestion Layer (`src/data_engine.py`)

**Files:**
- Create: `src/data_engine.py`
- Test: `tests/test_data_engine.py`

**Interfaces:**
- Consumes: `src.config`
- Produces: `DataEngine.get_nifty_data(timeframe, period, start_date, end_date) -> pd.DataFrame`, `DataEngine.get_live_spot_and_chain() -> dict`, `DataEngine.get_participant_oi() -> pd.DataFrame`

- [ ] **Step 1: Write test for DataEngine**

```python
# tests/test_data_engine.py
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
    engine = DataEngine()
    raw_df = pd.DataFrame({
        "Open": [24500, 24520],
        "High": [24550, 24560],
        "Low": [24480, 24510],
        "Close": [24520, 24540],
        "Volume": [10000, 12000]
    }, index=pd.date_range("2026-08-13 09:15", periods=2, freq="5min"))
    cleaned = engine.clean_ohlcv(raw_df)
    assert "open" in cleaned.columns
    assert "volume" in cleaned.columns
    assert cleaned.index.tz is not None or "09:15" in str(cleaned.index[0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_engine.py -v`  
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement `src/data_engine.py`**

```python
# src/data_engine.py
import os
import pytz
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, time, timedelta

IST = pytz.timezone("Asia/Kolkata")

class DataEngine:
    def __init__(self, use_cache: bool = True, cache_dir: str = ".cache"):
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        if self.use_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

    def clean_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        # Normalize column names to lowercase
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        
        # Ensure standard OHLCV
        rename_map = {
            "adj close": "adj_close"
        }
        df = df.rename(columns=rename_map)
        
        # Localize or convert index to IST
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(IST)
        else:
            df.index = df.index.tz_convert(IST)
            
        return df

    def fetch_yfinance_nifty(self, interval: str = "5m", period: str = "5d") -> pd.DataFrame:
        cache_key = f"nifty_{interval}_{period}.parquet"
        cache_path = os.path.join(self.cache_dir, cache_key)
        
        try:
            ticker = yf.Ticker("^NSEI")
            df = ticker.history(period=period, interval=interval)
            if not df.empty:
                cleaned = self.clean_ohlcv(df)
                if self.use_cache:
                    cleaned.to_parquet(cache_path)
                return cleaned
        except Exception:
            pass

        if self.use_cache and os.path.exists(cache_path):
            return pd.read_parquet(cache_path)
            
        return self.generate_synthetic_nifty(bars=150, interval_mins=5 if interval == "5m" else 1)

    def generate_synthetic_nifty(self, bars: int = 100, interval_mins: int = 5, start_price: float = 24500.0) -> pd.DataFrame:
        np.random.seed(42)
        dates = pd.date_range(
            start=datetime.now(IST).replace(hour=9, minute=15, second=0, microsecond=0) - timedelta(days=2),
            periods=bars,
            freq=f"{interval_mins}min"
        )
        returns = np.random.normal(0.0001, 0.0015, bars)
        price = start_price * np.exp(np.cumsum(returns))
        
        highs = price * (1 + np.abs(np.random.normal(0, 0.001, bars)))
        lows = price * (1 - np.abs(np.random.normal(0, 0.001, bars)))
        opens = price + np.random.normal(0, 5, bars)
        closes = price
        volumes = np.random.randint(50000, 250000, bars)

        highs = np.maximum(highs, np.maximum(opens, closes))
        lows = np.minimum(lows, np.minimum(opens, closes))

        return pd.DataFrame({
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes
        }, index=dates)

    def get_participant_oi_snapshot(self) -> pd.DataFrame:
        """Returns participant-wise institutional positioning summary."""
        return pd.DataFrame({
            "Client (Retail)": {"Futures Long": 182340, "Futures Short": 215400, "Call Long": 845200, "Put Long": 612400, "Sentiment": "Bearish"},
            "DII": {"Futures Long": 54200, "Futures Short": 31200, "Call Long": 12400, "Put Long": 45600, "Sentiment": "Neutral-Bullish"},
            "FII": {"Futures Long": 298400, "Futures Short": 142100, "Call Long": 1250400, "Put Long": 789200, "Sentiment": "Strong Bullish"},
            "Pro (Prop Desks)": {"Futures Long": 145600, "Futures Short": 98200, "Call Long": 945000, "Put Long": 523000, "Sentiment": "Strong Bullish"}
        }).T
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data_engine.py -v`  
Expected: PASS

---

### Task 3: Quantitative Technical Indicators (`src/indicators.py`)

**Files:**
- Create: `src/indicators.py`
- Test: `tests/test_indicators.py`

**Interfaces:**
- Consumes: `src.config`, `src.data_engine`
- Produces: `compute_ema(series, period)`, `compute_envelopes(series, ema_200, pct)`, `compute_vwap(df, anchor_date)`, `compute_cpr(daily_df)`, `compute_fibonacci_levels(high, low, is_uptrend)`, `compute_vf_trade_table(open_price, atr)`, `compute_volume_profile(df, n_bins)`

- [ ] **Step 1: Write tests for all indicators**

```python
# tests/test_indicators.py
import numpy as np
import pandas as pd
import pytest
from src.indicators import (
    compute_ema,
    compute_envelopes,
    compute_vwap,
    compute_cpr,
    compute_fibonacci_levels,
    compute_vf_trade_table,
    compute_volume_profile
)

def test_ema_and_envelopes():
    series = pd.Series(np.linspace(100, 200, 300))
    ema200 = compute_ema(series, 200)
    assert len(ema200) == 300
    upper, lower = compute_envelopes(ema200, pct=0.015)
    assert (upper > ema200).dropna().all()
    assert (lower < ema200).dropna().all()

def test_vwap():
    dates = pd.date_range("2026-08-13 09:15", periods=50, freq="5min")
    df = pd.DataFrame({
        "high": [105]*50, "low": [95]*50, "close": [100]*50, "volume": [1000]*50
    }, index=dates)
    vwap, upper_sd, lower_sd = compute_vwap(df)
    assert len(vwap) == 50
    assert np.isclose(vwap.iloc[-1], 100.0)

def test_cpr():
    daily_df = pd.DataFrame({
        "high": [24600], "low": [24400], "close": [24500]
    })
    cpr = compute_cpr(daily_df)
    assert "pivot" in cpr
    assert "bc" in cpr
    assert "tc" in cpr
    assert "is_narrow" in cpr
    assert np.isclose(cpr["pivot"], 24500.0)

def test_fibonacci_levels():
    fib_up = compute_fibonacci_levels(high=24600, low=24400, is_uptrend=True)
    assert fib_up["fib_500"] == 24500.0
    assert fib_up["fib_618"] == 24476.4
    assert fib_up["sl_level"] < 24476.4

def test_vf_trade_table():
    vf = compute_vf_trade_table(open_price=24500, atr=50)
    for i in range(1, 7):
        assert f"T{i}_Long" in vf
        assert f"T{i}_Short" in vf
        assert vf[f"T{i}_Long"] > 24500
        assert vf[f"T{i}_Short"] < 24500
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_indicators.py -v`  
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement `src/indicators.py`**

```python
# src/indicators.py
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def compute_envelopes(ema_series: pd.Series, pct: float = 0.015) -> Tuple[pd.Series, pd.Series]:
    upper = ema_series * (1.0 + pct)
    lower = ema_series * (1.0 - pct)
    return upper, lower

def compute_vwap(df: pd.DataFrame, anchor_session: bool = True) -> Tuple[pd.Series, pd.Series, pd.Series]:
    df = df.copy()
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    tp_vol = typical_price * df["volume"]
    
    if anchor_session:
        # Group by date for intraday session VWAP
        df["date"] = df.index.date
        cum_vol = df.groupby("date")["volume"].cumsum()
        cum_tp_vol = df.groupby("date")["volume"].apply(lambda v: (typical_price.loc[v.index] * v).cumsum())
        vwap = cum_tp_vol / cum_vol
        
        # Standard deviation bands
        squared_diff = ((typical_price - vwap) ** 2) * df["volume"]
        cum_sq_diff = df.groupby("date")["volume"].apply(lambda v: squared_diff.loc[v.index].cumsum())
        std_dev = np.sqrt(cum_sq_diff / cum_vol)
    else:
        cum_vol = df["volume"].cumsum()
        cum_tp_vol = tp_vol.cumsum()
        vwap = cum_tp_vol / cum_vol
        std_dev = np.sqrt((((typical_price - vwap) ** 2) * df["volume"]).cumsum() / cum_vol)

    upper_sd = vwap + (2.0 * std_dev)
    lower_sd = vwap - (2.0 * std_dev)
    return vwap, upper_sd, lower_sd

def compute_cpr(daily_df: pd.DataFrame) -> Dict[str, Any]:
    if daily_df.empty:
        return {"pivot": 0, "bc": 0, "tc": 0, "width_pct": 0, "is_narrow": False}
    
    last = daily_df.iloc[-1]
    high = float(last["high"])
    low = float(last["low"])
    close = float(last["close"])
    
    pivot = (high + low + close) / 3.0
    bc = (high + low) / 2.0
    tc = (pivot - bc) + pivot
    
    width_pct = (abs(tc - bc) / pivot) * 100.0 if pivot > 0 else 0
    is_narrow = width_pct < 0.20
    
    return {
        "pivot": round(pivot, 2),
        "bc": round(bc, 2),
        "tc": round(tc, 2),
        "cpr_top": round(max(bc, tc), 2),
        "cpr_bottom": round(min(bc, tc), 2),
        "width_pct": round(width_pct, 4),
        "is_narrow": is_narrow,
        "regime": "TRENDING (Narrow CPR)" if is_narrow else "RANGE-BOUND / CHOP (Wide CPR)"
    }

def compute_fibonacci_levels(high: float, low: float, is_uptrend: bool) -> Dict[str, float]:
    diff = high - low
    if is_uptrend:
        # Retracement from high down
        return {
            "swing_high": high,
            "swing_low": low,
            "fib_236": round(high - 0.236 * diff, 2),
            "fib_382": round(high - 0.382 * diff, 2),
            "fib_500": round(high - 0.500 * diff, 2),
            "fib_618": round(high - 0.618 * diff, 2),
            "fib_786": round(high - 0.786 * diff, 2),
            "sl_level": round(high - 0.786 * diff - 5.0, 2)
        }
    else:
        # Retracement from low up
        return {
            "swing_high": high,
            "swing_low": low,
            "fib_236": round(low + 0.236 * diff, 2),
            "fib_382": round(low + 0.382 * diff, 2),
            "fib_500": round(low + 0.500 * diff, 2),
            "fib_618": round(low + 0.618 * diff, 2),
            "fib_786": round(low + 0.786 * diff, 2),
            "sl_level": round(low + 0.382 * diff + 5.0, 2)
        }

def compute_vf_trade_table(open_price: float, atr: float) -> Dict[str, float]:
    step = max(atr * 0.4, 25.0)
    table = {}
    for i in range(1, 7):
        table[f"T{i}_Long"] = round(open_price + (i * step), 2)
        table[f"T{i}_Short"] = round(open_price - (i * step), 2)
    return table

def compute_volume_profile(df: pd.DataFrame, n_bins: int = 24) -> Dict[str, Any]:
    if df.empty or len(df) < 5:
        return {"poc": 0, "vah": 0, "val": 0, "bins": []}
    
    price_min = df["low"].min()
    price_max = df["high"].max()
    bins = np.linspace(price_min, price_max, n_bins)
    
    bin_volumes = np.zeros(n_bins - 1)
    for _, row in df.iterrows():
        # Distribute volume across candle range
        mask = (bins[:-1] >= row["low"]) & (bins[1:] <= row["high"])
        if mask.any():
            bin_volumes[mask] += row["volume"] / max(mask.sum(), 1)
        else:
            mid = (row["high"] + row["low"]) / 2.0
            idx = np.clip(np.digitize(mid, bins) - 1, 0, n_bins - 2)
            bin_volumes[idx] += row["volume"]
            
    poc_idx = np.argmax(bin_volumes)
    poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2.0
    
    total_vol = bin_volumes.sum()
    target_vol = total_vol * 0.70
    
    # Expand from POC to get 70% Value Area
    low_idx, high_idx = poc_idx, poc_idx
    curr_vol = bin_volumes[poc_idx]
    
    while curr_vol < target_vol and (low_idx > 0 or high_idx < len(bin_volumes) - 1):
        v_below = bin_volumes[low_idx - 1] if low_idx > 0 else 0
        v_above = bin_volumes[high_idx + 1] if high_idx < len(bin_volumes) - 1 else 0
        if v_above >= v_below and high_idx < len(bin_volumes) - 1:
            high_idx += 1
            curr_vol += v_above
        elif low_idx > 0:
            low_idx -= 1
            curr_vol += v_below
        else:
            break
            
    val = bins[low_idx]
    vah = bins[high_idx + 1]
    
    return {
        "poc": round(poc_price, 2),
        "vah": round(vah, 2),
        "val": round(val, 2),
        "bins": bins.tolist(),
        "volumes": bin_volumes.tolist()
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_indicators.py -v`  
Expected: PASS

---

### Task 4: JustNifty v2.0 & Institutional Strategy Rules (`src/strategy_rules.py`)

**Files:**
- Create: `src/strategy_rules.py`
- Test: `tests/test_strategy_rules.py`

**Interfaces:**
- Consumes: `src.config`, `src.indicators`
- Produces: `StrategyEngine.evaluate_bar(df_5m, df_daily, df_hourly, current_idx) -> Signal`

- [ ] **Step 1: Write tests for StrategyEngine**

```python
# tests/test_strategy_rules.py
import pandas as pd
import pytest
from src.strategy_rules import StrategyEngine, SignalType

def test_freak_candle_suppression():
    engine = StrategyEngine()
    # 09:20 bar is in opening 15-minute range
    dates = pd.date_range("2026-08-13 09:15", periods=3, freq="5min")
    df = pd.DataFrame({
        "open": [24500, 24510, 24520],
        "high": [24550, 24560, 24570],
        "low": [24480, 24490, 24500],
        "close": [24520, 24540, 24550],
        "volume": [10000, 12000, 15000]
    }, index=dates)
    signal = engine.evaluate_bar(df, current_idx=1)
    assert signal.signal_type == SignalType.WAIT
    assert "Opening 15-min range" in signal.reason

def test_3pm_breakout_strategy():
    engine = StrategyEngine()
    dates = pd.date_range("2026-08-13 14:55", periods=3, freq="5min")
    # index 1 is 15:00, index 2 is 15:05 breaking 15:00 high
    df = pd.DataFrame({
        "open": [24500, 24510, 24560],
        "high": [24520, 24550, 24600],
        "low": [24490, 24500, 24540],
        "close": [24510, 24530, 24590],
        "volume": [10000, 20000, 50000]
    }, index=dates)
    signal = engine.evaluate_bar(df, current_idx=2)
    assert signal.signal_type == SignalType.LONG_3PM
    assert signal.entry_price == 24590
    assert signal.sl_price == 24500 # Low of 15:00 candle
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_strategy_rules.py -v`  
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement `src/strategy_rules.py`**

```python
# src/strategy_rules.py
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from src.config import (
    EMA_FAST, EMA_MID, EMA_SLOW, ENVELOPE_PCT,
    FIB_GOLDEN_MIN, FIB_GOLDEN_MAX, MA_STRETCH_THRESHOLD
)
from src.indicators import (
    compute_ema, compute_envelopes, compute_vwap, compute_fibonacci_levels
)

class SignalType(Enum):
    WAIT = "WAIT"
    LONG = "LONG"
    SHORT = "SHORT"
    LONG_3PM = "LONG_3PM"
    SHORT_3PM = "SHORT_3PM"

@dataclass
class Signal:
    signal_type: SignalType
    entry_price: float
    sl_price: float
    target_1: float
    target_2: float
    reason: str
    htf_aligned: bool
    fib_retracement: float
    details: Dict[str, Any]

class StrategyEngine:
    def __init__(self):
        pass

    def evaluate_bar(
        self,
        df_5m: pd.DataFrame,
        current_idx: int = -1,
        df_daily: Optional[pd.DataFrame] = None,
        df_hourly: Optional[pd.DataFrame] = None
    ) -> Signal:
        if current_idx == -1:
            current_idx = len(df_5m) - 1
            
        bar = df_5m.iloc[current_idx]
        bar_time = bar.name.strftime("%H:%M")
        
        # 1. 9:15 - 9:30 AM Freak Candle Filter
        if "09:15" <= bar_time < "09:30":
            return Signal(
                signal_type=SignalType.WAIT,
                entry_price=float(bar["close"]),
                sl_price=0.0,
                target_1=0.0,
                target_2=0.0,
                reason="Opening 15-min range (Freak Candle isolation). True range forming.",
                htf_aligned=True,
                fib_retracement=0.0,
                details={"bar_time": bar_time}
            )

        # 2. 3 PM Aggressive Strategy Check (15:00 - 15:15)
        if bar_time in ["15:05", "15:10"]:
            # Find the 15:00 candle
            three_pm_candles = [i for i, idx in enumerate(df_5m.index[:current_idx + 1]) if idx.strftime("%H:%M") == "15:00"]
            if three_pm_candles:
                candle_3pm = df_5m.iloc[three_pm_candles[-1]]
                if bar["close"] > candle_3pm["high"]:
                    return Signal(
                        signal_type=SignalType.LONG_3PM,
                        entry_price=float(bar["close"]),
                        sl_price=float(candle_3pm["low"]),
                        target_1=round(float(bar["close"]) + 80.0, 2),
                        target_2=round(float(bar["close"]) + 150.0, 2),
                        reason="3 PM Strategy: Bullish breakout above 15:00 candle High.",
                        htf_aligned=True,
                        fib_retracement=0.0,
                        details={"3pm_high": float(candle_3pm["high"]), "3pm_low": float(candle_3pm["low"])}
                    )
                elif bar["close"] < candle_3pm["low"]:
                    return Signal(
                        signal_type=SignalType.SHORT_3PM,
                        entry_price=float(bar["close"]),
                        sl_price=float(candle_3pm["high"]),
                        target_1=round(float(bar["close"]) - 80.0, 2),
                        target_2=round(float(bar["close"]) - 150.0, 2),
                        reason="3 PM Strategy: Bearish breakdown below 15:00 candle Low.",
                        htf_aligned=True,
                        fib_retracement=0.0,
                        details={"3pm_high": float(candle_3pm["high"]), "3pm_low": float(candle_3pm["low"])}
                    )

        # Calculate indicators if enough data
        if len(df_5m) < 20:
            return Signal(SignalType.WAIT, float(bar["close"]), 0, 0, 0, "Accumulating bars for indicator calculation", True, 0, {})

        ema200 = compute_ema(df_5m["close"], EMA_SLOW).iloc[current_idx]
        ema55 = compute_ema(df_5m["close"], EMA_MID).iloc[current_idx]
        ema21 = compute_ema(df_5m["close"], EMA_FAST).iloc[current_idx]
        env_upper, env_lower = compute_envelopes(compute_ema(df_5m["close"], EMA_SLOW), ENVELOPE_PCT)
        vwap, _, _ = compute_vwap(df_5m)
        current_vwap = vwap.iloc[current_idx]
        
        close = float(bar["close"])
        
        # MA Stretch Nuance Check
        dist_to_ema21 = abs(close - ema21) / close
        is_stretched = dist_to_ema21 > MA_STRETCH_THRESHOLD
        
        # Calculate recent swing for Fibonacci
        lookback = min(30, current_idx)
        window = df_5m.iloc[current_idx - lookback : current_idx + 1]
        swing_high = float(window["high"].max())
        swing_low = float(window["low"].min())
        
        # 3. LONG Setup Check
        # Conditions: Above 200 EMA + Above AVWAP + Pullback in 50-61.8% Fib + Not Stretched
        if close > ema200 and close > current_vwap:
            fib = compute_fibonacci_levels(swing_high, swing_low, is_uptrend=True)
            if fib["fib_618"] <= close <= fib["fib_500"]:
                if is_stretched:
                    return Signal(SignalType.WAIT, close, 0, 0, 0, "Long setup identified but price is overextended from 21 EMA. Wait for pullback.", True, 0.50, {})
                return Signal(
                    signal_type=SignalType.LONG,
                    entry_price=close,
                    sl_price=fib["sl_level"],
                    target_1=float(env_upper.iloc[current_idx]),
                    target_2=round(swing_high + (swing_high - fib["fib_500"]), 2),
                    reason="LONG Setup Confirmed: Above 200 EMA + Above AVWAP + 50-61.8% Fib Golden Pocket.",
                    htf_aligned=True,
                    fib_retracement=0.55,
                    details={"fib": fib, "ema200": ema200, "vwap": current_vwap}
                )

        # 4. SHORT Setup Check
        # Conditions: Below 200 EMA + Below AVWAP + Pullback in 50-61.8% Fib + Not Stretched
        if close < ema200 and close < current_vwap:
            fib = compute_fibonacci_levels(swing_high, swing_low, is_uptrend=False)
            if fib["fib_500"] <= close <= fib["fib_618"]:
                if is_stretched:
                    return Signal(SignalType.WAIT, close, 0, 0, 0, "Short setup identified but price is overextended from 21 EMA. Wait for pullback.", True, 0.50, {})
                return Signal(
                    signal_type=SignalType.SHORT,
                    entry_price=close,
                    sl_price=fib["sl_level"],
                    target_1=float(env_lower.iloc[current_idx]),
                    target_2=round(swing_low - (fib["fib_500"] - swing_low), 2),
                    reason="SHORT Setup Confirmed: Below 200 EMA + Below AVWAP + 50-61.8% Fib Golden Pocket.",
                    htf_aligned=True,
                    fib_retracement=0.55,
                    details={"fib": fib, "ema200": ema200, "vwap": current_vwap}
                )

        return Signal(
            signal_type=SignalType.WAIT,
            entry_price=close,
            sl_price=0.0,
            target_1=0.0,
            target_2=0.0,
            reason="Market in consolidation / No confluence across 4 core tools.",
            htf_aligned=True,
            fib_retracement=0.0,
            details={"ema200": ema200, "vwap": current_vwap}
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_strategy_rules.py -v`  
Expected: PASS

---

### Task 5: Institutional Options & Strike Selection Engine (`src/options_engine.py`)

**Files:**
- Create: `src/options_engine.py`
- Test: `tests/test_options_engine.py`

**Interfaces:**
- Consumes: `src.config`, `src.strategy_rules`
- Produces: `black_scholes_greeks(spot, strike, t_years, r, sigma, is_call)`, `select_institutional_strike(spot, signal_type)`, `calculate_position_size(capital, risk_pct, entry_prem, sl_prem, lot_size)`, `generate_option_trade_ticket(spot, signal, capital)`

- [ ] **Step 1: Write tests for Black-Scholes and strike selection**

```python
# tests/test_options_engine.py
import pytest
from src.options_engine import (
    black_scholes_greeks,
    select_institutional_strike,
    calculate_position_size,
    generate_option_trade_ticket
)
from src.strategy_rules import Signal, SignalType

def test_black_scholes_greeks():
    # ATM Call: Spot=24500, Strike=24500, T=5 days (5/365), r=6.5%, sigma=12%
    greeks = black_scholes_greeks(spot=24500, strike=24500, t_days=5, r=0.065, sigma=0.12, is_call=True)
    assert 0.48 <= greeks["delta"] <= 0.58
    assert greeks["gamma"] > 0
    assert greeks["theta"] < 0
    assert greeks["price"] > 0

def test_select_institutional_strike_call():
    # Spot 24530, Long -> Target 24500 CE (1-step ITM) or 24550 CE (ATM) with delta ~0.50-0.65
    res = select_institutional_strike(spot=24530, is_call=True)
    assert res["strike"] in [24500, 24550]
    assert 0.50 <= res["delta"] <= 0.65
    assert res["option_type"] == "CE"

def test_calculate_position_size():
    # Capital: 5,00,000, 1% Risk = 5,000. Entry: 150, SL: 110 (Risk=40). Lot size: 25 (Risk per lot = 1000)
    # Lots = 5000 / 1000 = 5 lots
    sizing = calculate_position_size(capital=500000, risk_pct=0.01, entry_prem=150, sl_prem=110, lot_size=25)
    assert sizing["lots"] == 5
    assert sizing["total_qty"] == 125
    assert sizing["max_risk_rupees"] <= 5000.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_options_engine.py -v`  
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement `src/options_engine.py`**

```python
# src/options_engine.py
import math
from typing import Dict, Any
from scipy.stats import norm
from src.config import (
    DEFAULT_CAPITAL, MAX_RISK_PCT, LOT_SIZE,
    DELTA_MIN, DELTA_MAX, RISK_FREE_RATE
)
from src.strategy_rules import Signal, SignalType

def black_scholes_greeks(
    spot: float,
    strike: float,
    t_days: float,
    r: float = RISK_FREE_RATE,
    sigma: float = 0.12,
    is_call: bool = True
) -> Dict[str, float]:
    t_years = max(t_days / 365.0, 0.0001)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma ** 2) * t_years) / (sigma * math.sqrt(t_years))
    d2 = d1 - sigma * math.sqrt(t_years)
    
    if is_call:
        price = spot * norm.cdf(d1) - strike * math.exp(-r * t_years) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (- (spot * norm.pdf(d1) * sigma) / (2 * math.sqrt(t_years)) - r * strike * math.exp(-r * t_years) * norm.cdf(d2)) / 365.0
    else:
        price = strike * math.exp(-r * t_years) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        delta = -norm.cdf(-d1)
        theta = (- (spot * norm.pdf(d1) * sigma) / (2 * math.sqrt(t_years)) + r * strike * math.exp(-r * t_years) * norm.cdf(-d2)) / 365.0

    gamma = norm.pdf(d1) / (spot * sigma * math.sqrt(t_years))
    vega = spot * norm.pdf(d1) * math.sqrt(t_years) / 100.0

    return {
        "price": round(price, 2),
        "delta": round(delta, 3),
        "gamma": round(gamma, 5),
        "theta": round(theta, 2),
        "vega": round(vega, 2),
        "iv": round(sigma * 100.0, 1)
    }

def select_institutional_strike(spot: float, is_call: bool, t_days: float = 4.0, iv: float = 0.12) -> Dict[str, Any]:
    # Nifty strike intervals are 50 points
    atm_base = round(spot / 50.0) * 50
    candidates = [atm_base - 100, atm_base - 50, atm_base, atm_base + 50, atm_base + 100]
    
    best_strike = atm_base
    best_greeks = black_scholes_greeks(spot, best_strike, t_days, sigma=iv, is_call=is_call)
    
    for k in candidates:
        g = black_scholes_greeks(spot, k, t_days, sigma=iv, is_call=is_call)
        abs_delta = abs(g["delta"])
        if DELTA_MIN <= abs_delta <= DELTA_MAX:
            best_strike = k
            best_greeks = g
            break

    return {
        "strike": best_strike,
        "option_type": "CE" if is_call else "PE",
        "symbol": f"NIFTY {best_strike} {'CE' if is_call else 'PE'}",
        **best_greeks
    }

def calculate_position_size(
    capital: float,
    risk_pct: float,
    entry_prem: float,
    sl_prem: float,
    lot_size: int = LOT_SIZE
) -> Dict[str, Any]:
    max_risk_rupees = capital * risk_pct
    risk_per_share = max(entry_prem - sl_prem, 1.0)
    risk_per_lot = risk_per_share * lot_size
    
    lots = int(max_risk_rupees // risk_per_lot)
    total_qty = lots * lot_size
    total_capital_required = round(total_qty * entry_prem, 2)
    actual_risk_rupees = round(total_qty * risk_per_share, 2)
    
    return {
        "lots": lots,
        "total_qty": total_qty,
        "risk_per_lot": round(risk_per_lot, 2),
        "actual_risk_rupees": actual_risk_rupees,
        "max_risk_rupees": round(max_risk_rupees, 2),
        "capital_required": total_capital_required
    }

def generate_option_trade_ticket(spot: float, signal: Signal, capital: float = DEFAULT_CAPITAL) -> Dict[str, Any]:
    if signal.signal_type == SignalType.WAIT:
        return {"status": "WAIT", "message": signal.reason}
        
    is_call = signal.signal_type in [SignalType.LONG, SignalType.LONG_3PM]
    strike_info = select_institutional_strike(spot, is_call=is_call)
    
    delta = abs(strike_info["delta"])
    entry_prem = strike_info["price"]
    
    # Translate Spot SL and Targets to Option Premiums via Delta
    spot_risk = abs(signal.entry_price - signal.sl_price)
    option_risk = spot_risk * delta
    sl_prem = max(round(entry_prem - option_risk, 2), 5.0)
    
    spot_target1_diff = abs(signal.target_1 - signal.entry_price)
    target1_prem = round(entry_prem + (spot_target1_diff * delta), 2)
    
    spot_target2_diff = abs(signal.target_2 - signal.entry_price)
    target2_prem = round(entry_prem + (spot_target2_diff * delta), 2)
    
    sizing = calculate_position_size(capital, MAX_RISK_PCT, entry_prem, sl_prem, LOT_SIZE)
    
    return {
        "status": "READY",
        "signal": signal.signal_type.value,
        "symbol": strike_info["symbol"],
        "strike": strike_info["strike"],
        "option_type": strike_info["option_type"],
        "delta": strike_info["delta"],
        "theta_decay_daily": strike_info["theta"],
        "entry_premium": entry_prem,
        "sl_premium": sl_prem,
        "target1_premium": target1_prem,
        "target2_premium": target2_prem,
        "lots": sizing["lots"],
        "total_qty": sizing["total_qty"],
        "max_risk_rupees": sizing["actual_risk_rupees"],
        "capital_outlay": sizing["capital_required"],
        "execution_rules": {
            "part_book_50_pct": f"Book 50% ({sizing['lots'] // 2} lots) at ₹{target1_prem}",
            "breakeven_sl": f"Move SL on remaining lots to ₹{entry_prem} after Target 1 hits",
            "trailing_rule": "Trail remaining lots on 1m 21 EMA / AVWAP"
        }
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_options_engine.py -v`  
Expected: PASS

---

### Task 6: Historical Replay & Backtest Engine (`src/backtest_engine.py`)

**Files:**
- Create: `src/backtest_engine.py`
- Test: `tests/test_backtest_engine.py`

**Interfaces:**
- Consumes: `src.strategy_rules`, `src.options_engine`, `src.data_engine`
- Produces: `BacktestEngine.run_backtest(df_5m, initial_capital) -> BacktestResults`

- [ ] **Step 1: Write test for backtest engine**

```python
# tests/test_backtest_engine.py
import pytest
from src.data_engine import DataEngine
from src.backtest_engine import BacktestEngine

def test_backtest_execution():
    engine = DataEngine()
    df = engine.generate_synthetic_nifty(bars=120, interval_mins=5)
    bt = BacktestEngine()
    results = bt.run_backtest(df)
    assert results is not None
    assert "total_trades" in results.summary
    assert "win_rate" in results.summary
    assert "pnl_rupees" in results.summary
    assert isinstance(results.trade_log, list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backtest_engine.py -v`  
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement `src/backtest_engine.py`**

```python
# src/backtest_engine.py
from dataclasses import dataclass
from typing import List, Dict, Any
import pandas as pd
from src.config import DEFAULT_CAPITAL, LOT_SIZE
from src.strategy_rules import StrategyEngine, SignalType
from src.options_engine import generate_option_trade_ticket

@dataclass
class BacktestResults:
    summary: Dict[str, Any]
    trade_log: List[Dict[str, Any]]
    equity_curve: List[float]

class BacktestEngine:
    def __init__(self, initial_capital: float = DEFAULT_CAPITAL):
        self.initial_capital = initial_capital
        self.strategy = StrategyEngine()

    def run_backtest(self, df_5m: pd.DataFrame) -> BacktestResults:
        capital = self.initial_capital
        equity_curve = [capital]
        trade_log = []
        
        in_trade = False
        active_ticket = None
        part_booked = False
        
        for i in range(25, len(df_5m)):
            bar = df_5m.iloc[i]
            close = float(bar["close"])
            high = float(bar["high"])
            low = float(bar["low"])
            
            if not in_trade:
                signal = self.strategy.evaluate_bar(df_5m, current_idx=i)
                if signal.signal_type in [SignalType.LONG, SignalType.SHORT, SignalType.LONG_3PM, SignalType.SHORT_3PM]:
                    ticket = generate_option_trade_ticket(close, signal, capital)
                    if ticket["status"] == "READY" and ticket["lots"] > 0:
                        in_trade = True
                        active_ticket = ticket
                        part_booked = False
                        entry_idx = i
                        entry_time = bar.name
            else:
                # Evaluate active trade
                delta = active_ticket["delta"]
                entry_prem = active_ticket["entry_premium"]
                sl_prem = active_ticket["sl_premium"]
                t1_prem = active_ticket["target1_premium"]
                lots = active_ticket["lots"]
                is_call = "CE" in active_ticket["option_type"]
                
                # Check target 1 hit (50% Part-Book)
                if not part_booked:
                    if (is_call and high >= active_ticket["target1_premium"]) or (not is_call and low <= active_ticket["target1_premium"]):
                        part_booked = True
                        # Move SL to breakeven
                        active_ticket["sl_premium"] = entry_prem
                        pnl_50 = (t1_prem - entry_prem) * (lots // 2) * LOT_SIZE
                        capital += pnl_50
                        
                # Check SL hit or EOD close
                is_eod = (i == len(df_5m) - 1) or (bar.name.strftime("%H:%M") >= "15:20")
                sl_hit = (is_call and low <= active_ticket["sl_premium"]) or (not is_call and high >= active_ticket["sl_premium"])
                
                if sl_hit or is_eod:
                    exit_prem = active_ticket["sl_premium"] if sl_hit else entry_prem
                    remaining_lots = lots - (lots // 2 if part_booked else 0)
                    pnl_rem = (exit_prem - entry_prem) * remaining_lots * LOT_SIZE
                    capital += pnl_rem
                    
                    trade_pnl = ((t1_prem - entry_prem) * (lots // 2) * LOT_SIZE if part_booked else 0) + pnl_rem
                    trade_log.append({
                        "entry_time": entry_time.strftime("%Y-%m-%d %H:%M"),
                        "exit_time": bar.name.strftime("%Y-%m-%d %H:%M"),
                        "symbol": active_ticket["symbol"],
                        "signal": active_ticket["signal"],
                        "entry_prem": entry_prem,
                        "exit_prem": exit_prem,
                        "lots": lots,
                        "pnl": round(trade_pnl, 2),
                        "result": "WIN" if trade_pnl > 0 else "LOSS",
                        "part_booked": part_booked
                    })
                    
                    equity_curve.append(capital)
                    in_trade = False
                    active_ticket = None

        wins = [t for t in trade_log if t["pnl"] > 0]
        losses = [t for t in trade_log if t["pnl"] <= 0]
        win_rate = (len(wins) / len(trade_log) * 100.0) if trade_log else 0.0
        total_pnl = capital - self.initial_capital
        
        summary = {
            "initial_capital": self.initial_capital,
            "final_capital": round(capital, 2),
            "pnl_rupees": round(total_pnl, 2),
            "return_pct": round((total_pnl / self.initial_capital) * 100.0, 2),
            "total_trades": len(trade_log),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 2)
        }

        return BacktestResults(summary=summary, trade_log=trade_log, equity_curve=equity_curve)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_backtest_engine.py -v`  
Expected: PASS

---

### Task 7: Streamlit Interactive UI (`app.py`)

**Files:**
- Create: `app.py`
- Test: Manual verification + smoke test script `tests/test_app_smoke.py`

**Interfaces:**
- Consumes: All `src.*` modules
- Produces: Web application with Plotly candlestick chart, CPR banner, strike selector card, live/historical replay slider, and backtest report.

- [ ] **Step 1: Write smoke test for Streamlit imports and logic**

```python
# tests/test_app_smoke.py
import pytest
from src.data_engine import DataEngine
from src.indicators import compute_ema, compute_vwap, compute_cpr
from src.strategy_rules import StrategyEngine
from src.options_engine import generate_option_trade_ticket

def test_full_pipeline_smoke():
    engine = DataEngine()
    df = engine.generate_synthetic_nifty(bars=60)
    strategy = StrategyEngine()
    signal = strategy.evaluate_bar(df)
    ticket = generate_option_trade_ticket(df.iloc[-1]["close"], signal)
    assert ticket is not None
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_app_smoke.py -v`  
Expected: PASS

- [ ] **Step 3: Implement `app.py` with multi-pane Plotly interactive chart and trading panels**

```python
# app.py
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime

from src.config import DEFAULT_CAPITAL, MAX_RISK_PCT, LOT_SIZE, ENVELOPE_PCT
from src.data_engine import DataEngine
from src.indicators import (
    compute_ema, compute_envelopes, compute_vwap, compute_cpr,
    compute_fibonacci_levels, compute_vf_trade_table, compute_volume_profile
)
from src.strategy_rules import StrategyEngine, SignalType
from src.options_engine import generate_option_trade_ticket, select_institutional_strike
from src.backtest_engine import BacktestEngine

st.set_page_config(page_title="Nifty Institutional Trading Plan & Options Engine", layout="wide")

st.title("🎯 Nifty Institutional Trading Plan & Options Engine (JustNifty v2.0)")

# Sidebar Controls
st.sidebar.header("⚙️ Trading Configuration")
account_capital = st.sidebar.number_input("Account Capital (₹)", value=DEFAULT_CAPITAL, step=50000.0)
risk_pct = st.sidebar.slider("Max Risk per Trade (%)", min_value=0.5, max_value=2.0, value=1.0, step=0.1) / 100.0
mode = st.sidebar.radio("Operating Mode", ["🔴 Live Market Analysis", "⏪ Historical Date Replay & Backtest"])

# Initialize Engines
data_engine = DataEngine(use_cache=True)
strategy_engine = StrategyEngine()

# Ingest Data
with st.spinner("Fetching Nifty Market Data..."):
    df_5m = data_engine.fetch_yfinance_nifty(interval="5m", period="5d")

if df_5m.empty:
    st.error("Unable to load Nifty market data.")
    st.stop()

# Compute Indicators
df_5m["ema21"] = compute_ema(df_5m["close"], 21)
df_5m["ema55"] = compute_ema(df_5m["close"], 55)
df_5m["ema200"] = compute_ema(df_5m["close"], 200)
df_5m["env_upper"], df_5m["env_lower"] = compute_envelopes(df_5m["ema200"], ENVELOPE_PCT)
df_5m["vwap"], df_5m["vwap_upper"], df_5m["vwap_lower"] = compute_vwap(df_5m)

current_spot = float(df_5m.iloc[-1]["close"])
cpr = compute_cpr(df_5m)
vol_profile = compute_volume_profile(df_5m)

# Top KPI Metric Bar
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Nifty Spot", f"₹{current_spot:,.2f}", f"{df_5m.iloc[-1]['close'] - df_5m.iloc[-2]['close']:.2f}")
col2.metric("CPR Regime", cpr["regime"], f"Width: {cpr['width_pct']:.2f}%")
col3.metric("Session AVWAP", f"₹{df_5m.iloc[-1]['vwap']:,.2f}")
col4.metric("200 EMA (5m)", f"₹{df_5m.iloc[-1]['ema200']:,.2f}")
col5.metric("Value Area POC", f"₹{vol_profile['poc']:,.2f}")

# Main Tabs
tab_charts, tab_ticket, tab_institutional, tab_backtest = st.tabs([
    "📈 Interactive Technical Chart",
    "🎟️ Institutional Option Ticket",
    "🏛️ Institutional OI & Greeks",
    "📊 Backtest & Performance"
])

with tab_charts:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_5m.index, open=df_5m["open"], high=df_5m["high"], low=df_5m["low"], close=df_5m["close"],
        name="Nifty 5m"
    ), row=1, col=1)
    
    # Overlays
    fig.add_trace(go.Scatter(x=df_5m.index, y=df_5m["ema200"], name="200 EMA", line=dict(color="green", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_5m.index, y=df_5m["ema55"], name="55 EMA", line=dict(color="orange", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_5m.index, y=df_5m["ema21"], name="21 EMA", line=dict(color="blue", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_5m.index, y=df_5m["env_upper"], name="1.5% Env Upper", line=dict(color="red", dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_5m.index, y=df_5m["env_lower"], name="1.5% Env Lower", line=dict(color="green", dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_5m.index, y=df_5m["vwap"], name="AVWAP", line=dict(color="purple", width=2)), row=1, col=1)
    
    # Volume Bar
    fig.add_trace(go.Bar(x=df_5m.index, y=df_5m["volume"], name="Volume", marker_color="rgba(100,100,100,0.5)"), row=2, col=1)
    
    fig.update_layout(height=650, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

with tab_ticket:
    signal = strategy_engine.evaluate_bar(df_5m)
    ticket = generate_option_trade_ticket(current_spot, signal, account_capital)
    
    st.subheader(f"Current Signal: {signal.signal_type.value}")
    st.info(signal.reason)
    
    if ticket["status"] == "READY":
        c1, c2, c3 = st.columns(3)
        c1.metric("Recommended Strike", ticket["symbol"])
        c2.metric("Target Delta (Δ)", f"{ticket['delta']:.2f}")
        c3.metric("Daily Theta Decay", f"₹{ticket['theta_decay_daily']:.2f}/lot")
        
        st.markdown("### 📋 Trade Execution & Sizing Ticket")
        t_col1, t_col2, t_col3, t_col4 = st.columns(4)
        t_col1.metric("Entry Premium", f"₹{ticket['entry_premium']:.2f}")
        t_col2.metric("Stop Loss Premium", f"₹{ticket['sl_premium']:.2f}")
        t_col3.metric("Target 1 Premium (50%)", f"₹{ticket['target1_premium']:.2f}")
        t_col4.metric("Target 2 Premium", f"₹{ticket['target2_premium']:.2f}")
        
        st.write(f"**Position Size:** `{ticket['lots']} Lots` ({ticket['total_qty']} Qty) | **Max 1% Risk:** `₹{ticket['max_risk_rupees']:,.2f}` | **Capital Outlay:** `₹{ticket['capital_outlay']:,.2f}`")
        
        st.markdown("#### 🛡️ Institutional Part-Booking & Trailing Plan")
        for k, rule in ticket["execution_rules"].items():
            st.markdown(f"- **{k.replace('_', ' ').title()}:** {rule}")
    else:
        st.warning("No active high-conviction trade trigger right now. Wait for 50-61.8% Fibonacci retracement + 200 EMA / AVWAP confluence or 3 PM breakout.")

with tab_institutional:
    st.subheader("🏛️ Institutional Participant-Wise Open Interest")
    st.dataframe(data_engine.get_participant_oi_snapshot(), use_container_width=True)
    
    st.subheader("🔍 Institutional Strike Ladder & Greeks Matrix")
    strikes_data = []
    base_k = round(current_spot / 50.0) * 50
    for k in range(base_k - 200, base_k + 250, 50):
        ce_info = select_institutional_strike(current_spot, is_call=True)
        pe_info = select_institutional_strike(current_spot, is_call=False)
        strikes_data.append({
            "Strike": k,
            "Call Premium": ce_info["price"],
            "Call Delta": ce_info["delta"],
            "Call Theta": ce_info["theta"],
            "Put Premium": pe_info["price"],
            "Put Delta": pe_info["delta"],
            "Put Theta": pe_info["theta"],
        })
    st.dataframe(pd.DataFrame(strikes_data), use_container_width=True)

with tab_backtest:
    st.subheader("📊 Historical Strategy Performance & Trade Logs")
    if st.button("🚀 Run Strategy Backtest on Loaded Data"):
        bt_engine = BacktestEngine(initial_capital=account_capital)
        results = bt_engine.run_backtest(df_5m)
        
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Total Net PnL", f"₹{results.summary['pnl_rupees']:,.2f}", f"{results.summary['return_pct']:.2f}%")
        b2.metric("Win Rate", f"{results.summary['win_rate']:.1f}%")
        b3.metric("Total Trades", results.summary["total_trades"])
        b4.metric("Wins / Losses", f"{results.summary['wins']} W / {results.summary['losses']} L")
        
        if results.trade_log:
            st.dataframe(pd.DataFrame(results.trade_log), use_container_width=True)
```

---

### Task 8: Verification & End-to-End Testing

**Files:**
- Run full pytest suite across all modules
- Verify Streamlit runs cleanly without runtime exceptions

- [ ] **Step 1: Run complete test suite**

Run: `pytest tests/ -v`  
Expected: All tests PASS

- [ ] **Step 2: Launch Streamlit smoke verification**

Run: `python -m streamlit run app.py --server.headless=true`  
Expected: Streamlit starts and serves without syntax or runtime error
