"""Tests verifying that 3 PM breakout setups are routed through the full pipeline:
- Universal gates (VPIN, 25d Skew, HFI, Data sufficiency)
- Stop loss hygiene (bounded between MIN_ATR and STOP_MAX_POINTS cap)
- Walk-forward edge table quarantine
- Confluence score calculation & floor enforcement
"""

import pandas as pd
import pytest
from src.strategy_rules import StrategyEngine, SignalType
from src.edge_harness import EdgeTable, EdgeStats
from src.config import STOP_MAX_POINTS


def _make_3pm_df(n_bars=20, bar_1500_range=(24510.0, 24540.0), bar_1505_close=24555.0):
    dates = pd.date_range("2026-08-13 13:30", periods=n_bars, freq="5min")
    opens = [24500.0 + i * 2 for i in range(n_bars)]
    closes = [24500.0 + i * 2 + (2.0 if i % 2 == 0 else -1.0) for i in range(n_bars)]
    highs = [max(o, c) + 5.0 for o, c in zip(opens, closes)]
    lows = [min(o, c) - 5.0 for o, c in zip(opens, closes)]
    volumes = [10000] * (n_bars - 2) + [20000, 50000]

    # Bar 18 is 15:00, Bar 19 is 15:05
    lows[18] = bar_1500_range[0]
    highs[18] = bar_1500_range[1]
    closes[18] = (bar_1500_range[0] + bar_1500_range[1]) / 2.0

    opens[19] = closes[18]
    closes[19] = bar_1505_close
    highs[19] = max(bar_1505_close + 5.0, highs[18] + 5.0)
    lows[19] = min(opens[19] - 5.0, closes[18] - 5.0)

    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes
    }, index=dates)
    return df


def _make_valid_opt_ctx():
    return {
        "chain_df": pd.DataFrame({"strike": [24550, 24550], "type": ["CE", "PE"], "iv": [0.13, 0.13]}),
        "gex_chart": {"call_wall_strike": 25000.0, "put_wall_strike": 24000.0, "net_dealer_regime": "DEALER_LONG_GAMMA"},
        "dir_flow": {"directional_vector": 0.25}
    }


class Test3PMPipelineRouting:
    def test_3pm_basic_execution(self):
        engine = StrategyEngine()
        df = _make_3pm_df(n_bars=20, bar_1500_range=(24510.0, 24540.0), bar_1505_close=24555.0)
        signal = engine.evaluate_bar(df, current_idx=19, options_context=_make_valid_opt_ctx())
        
        assert signal.signal_type == SignalType.LONG_3PM
        assert signal.entry_price == 24555.0
        assert signal.sl_price == 24510.0  # 15:00 low
        assert signal.target_1 > signal.entry_price

    def test_3pm_respects_stop_loss_cap(self):
        """If 15:00 candle was 120 pts wide, stop loss must be capped at STOP_MAX_POINTS (60 pts)."""
        engine = StrategyEngine()
        # 15:00 candle low is 24400, high is 24540. Entry is 24555.
        # Raw distance = 155 pts! Must be capped to 60 pts.
        df = _make_3pm_df(n_bars=20, bar_1500_range=(24400.0, 24540.0), bar_1505_close=24555.0)
        signal = engine.evaluate_bar(df, current_idx=19, options_context=_make_valid_opt_ctx())
        
        assert signal.signal_type == SignalType.LONG_3PM
        assert signal.entry_price == 24555.0
        sl_dist = abs(signal.entry_price - signal.sl_price)
        assert sl_dist <= STOP_MAX_POINTS
        assert signal.sl_price == 24555.0 - STOP_MAX_POINTS

    def test_3pm_quarantined_by_edge_table(self):
        """When LONG_3PM is quarantined in edge table, it must be blocked."""
        table = EdgeTable()
        # Quarantine LONG_3PM in all regimes
        for regime in ["LOW_VOL_TRENDING", "MEAN_REVERTING_CHOP", "HIGH_VOL_EXPANSION"]:
            table.records[("LONG_3PM", regime)] = EdgeStats(
                setup_id="LONG_3PM",
                regime=regime,
                n=30,
                win_rate=0.20,
                mean_r=-0.45,
                ev=-0.45,
                ci_low=-0.70,
                ci_high=-0.20,
                status="QUARANTINED"
            )

        engine = StrategyEngine(edge_table=table)
        df = _make_3pm_df(n_bars=20)
        signal = engine.evaluate_bar(df, current_idx=19, options_context=_make_valid_opt_ctx())
        
        # Must not fire LONG_3PM when quarantined
        assert signal.signal_type != SignalType.LONG_3PM

    def test_3pm_gated_by_universal_vpin_toxicity(self):
        """When VPIN indicates extreme informed toxicity, 3PM setup must be gated."""
        engine = StrategyEngine()
        df = _make_3pm_df(n_bars=20)
        # Artificially inject 100% one-sided volume imbalance to trigger VPIN > 0.85
        df["open"] = [24500.0 + i for i in range(len(df))]
        df["close"] = [24505.0 + i for i in range(len(df))]
        df["volume"] = [500000 for _ in range(len(df))]
        signal = engine.evaluate_bar(df, current_idx=19, options_context=_make_valid_opt_ctx())
        
        # If VPIN fires or gates block, signal must be WAIT, not un-gated LONG_3PM
        assert signal.signal_type == SignalType.WAIT
