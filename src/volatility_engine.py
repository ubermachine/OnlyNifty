"""
OnlyNifty v5.0 Volatility Intelligence Engine.

Computes the institutional volatility edge metrics that separate profitable
options traders from the rest:

1. Rolling IV-RV Spread & Z-Score
2. IV Percentile Rank (where is current IV relative to history?)
3. Variance Risk Premium (VRP) estimation
4. Vol Regime Classification: BUY_VOL / SELL_VOL / NEUTRAL
5. Expected Move vs Actual Move accuracy tracking
6. Intraday Seasonality quality scoring
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from src.config import DEFAULT_IV


class VolatilityIntelligence:
    """
    Institutional Volatility Edge Engine.

    The #1 edge in options trading is the Implied Volatility vs Realized Volatility spread.
    - When IV > RV (positive VRP): Sell premium (Iron Condors, Credit Spreads, Strangles)
    - When RV > IV (negative VRP): Buy options (Directional, Straddles) — cheap convexity
    - When IV ≈ RV: Use delta-directional strategies with standard structure
    """

    def __init__(self, baseline_iv: float = DEFAULT_IV):
        self.baseline_iv = baseline_iv
        self.historical_iv_values: List[float] = []
        self.expected_move_log: List[Dict[str, float]] = []

    # ────────────────── 1. Realized Volatility ──────────────────

    @staticmethod
    def compute_realized_volatility(
        close_prices: pd.Series,
        window: int = 20,
        annualization_factor: float = 252 * 75  # 5m bars: 75 bars/day * 252 days
    ) -> Dict[str, float]:
        """
        Computes Yang-Zhang or Close-to-Close realized volatility from intraday closes.
        Returns annualized RV and raw bar-level standard deviation.
        """
        if len(close_prices) < max(window, 5):
            return {"realized_vol": 0.12, "rv_raw_std": 0.0, "window": window}

        log_returns = np.diff(np.log(np.maximum(close_prices.values[-window:], 1.0)))
        if len(log_returns) < 3:
            return {"realized_vol": 0.12, "rv_raw_std": 0.0, "window": window}

        rv_raw = float(np.std(log_returns, ddof=1))
        rv_annualized = rv_raw * np.sqrt(annualization_factor)

        return {
            "realized_vol": round(rv_annualized, 4),
            "rv_raw_std": round(rv_raw, 6),
            "window": window
        }

    # ────────────────── 2. IV-RV Spread ──────────────────

    @staticmethod
    def compute_iv_rv_spread(
        implied_vol: float,
        realized_vol: float
    ) -> Dict[str, Any]:
        """
        Computes the core IV-RV spread that determines whether options are overpriced
        (sell premium) or underpriced (buy convexity).

        Spread > 0: IV overprices risk → Sell options (Variance Risk Premium is positive)
        Spread < 0: IV underprices risk → Buy options (cheap gamma/convexity)
        """
        spread = implied_vol - realized_vol
        spread_ratio = spread / max(realized_vol, 0.01)

        # Classify the spread regime
        if spread_ratio > 0.25:
            regime = "STRONG_SELL_VOL"
            advice = "IV significantly overpriced. Sell premium: Iron Condors, Credit Spreads, or Ratio Spreads."
            structure_recommendation = "SELL_PREMIUM"
        elif spread_ratio > 0.10:
            regime = "MILD_SELL_VOL"
            advice = "IV moderately overpriced. Favor credit structures or debit spreads over outright buys."
            structure_recommendation = "SELL_PREMIUM"
        elif spread_ratio < -0.20:
            regime = "STRONG_BUY_VOL"
            advice = "IV significantly underpriced relative to actual moves. Buy options: Straddles, directional calls/puts."
            structure_recommendation = "BUY_CONVEXITY"
        elif spread_ratio < -0.08:
            regime = "MILD_BUY_VOL"
            advice = "IV slightly underpriced. Outright option buying has structural edge."
            structure_recommendation = "BUY_DIRECTIONAL"
        else:
            regime = "NEUTRAL_VOL"
            advice = "IV and RV approximately balanced. Use directional delta strategies with standard structure."
            structure_recommendation = "DIRECTIONAL_DELTA"

        return {
            "iv": round(implied_vol, 4),
            "rv": round(realized_vol, 4),
            "spread": round(spread, 4),
            "spread_pct": round(spread * 100, 2),
            "spread_ratio": round(spread_ratio, 3),
            "vol_regime": regime,
            "structure_recommendation": structure_recommendation,
            "advice": advice
        }

    # ────────────────── 3. IV Percentile Rank ──────────────────

    def compute_iv_percentile(
        self,
        current_iv: float,
        iv_history: Optional[List[float]] = None,
        lookback: int = 20
    ) -> Dict[str, Any]:
        """
        Computes where current IV sits relative to recent history.
        IV Percentile = % of past readings below current IV.

        - IV Percentile > 80%: IV is historically high → favor selling
        - IV Percentile < 20%: IV is historically low → favor buying
        """
        if iv_history and len(iv_history) >= 5:
            history = iv_history[-lookback:]
        elif len(self.historical_iv_values) >= 5:
            history = self.historical_iv_values[-lookback:]
        else:
            # Not enough history — estimate from baseline
            history = [self.baseline_iv * (0.85 + 0.30 * i / 19) for i in range(20)]

        count_below = sum(1 for h in history if h < current_iv)
        percentile = round((count_below / len(history)) * 100, 1)

        iv_min = round(min(history), 4)
        iv_max = round(max(history), 4)
        iv_mean = round(float(np.mean(history)), 4)
        iv_std = round(float(np.std(history)), 4)

        # Z-score of current IV relative to history
        iv_zscore = round((current_iv - iv_mean) / max(iv_std, 0.001), 2)

        if percentile >= 80:
            rank_label = "HIGH_IV"
            rank_advice = "IV is in the top 20% of recent history. Premium is expensive — favor selling."
        elif percentile >= 60:
            rank_label = "ABOVE_AVERAGE_IV"
            rank_advice = "IV is above average. Debit spreads preferred over outright buys."
        elif percentile <= 20:
            rank_label = "LOW_IV"
            rank_advice = "IV is in the bottom 20%. Options are cheap — favor buying convexity."
        elif percentile <= 40:
            rank_label = "BELOW_AVERAGE_IV"
            rank_advice = "IV is below average. Outright option buying has favorable pricing."
        else:
            rank_label = "AVERAGE_IV"
            rank_advice = "IV is in the normal range. Standard directional strategies appropriate."

        # Update running history
        self.historical_iv_values.append(current_iv)
        if len(self.historical_iv_values) > 500:
            self.historical_iv_values = self.historical_iv_values[-200:]

        return {
            "iv_percentile": percentile,
            "iv_zscore": iv_zscore,
            "rank_label": rank_label,
            "rank_advice": rank_advice,
            "iv_min": iv_min,
            "iv_max": iv_max,
            "iv_mean": iv_mean,
            "iv_std": iv_std,
            "history_length": len(history)
        }

    # ────────────────── 4. Expected Move vs Actual Move ──────────────────

    @staticmethod
    def compute_expected_vs_actual_move(
        atm_straddle_premium: float,
        session_high: float,
        session_low: float,
        session_open: float
    ) -> Dict[str, Any]:
        """
        Tracks whether the market moved more or less than the ATM straddle predicted.

        Expected Move = ATM Straddle Premium (in spot-equivalent points)
        Actual Move = max(|High - Open|, |Low - Open|)

        If Actual consistently < Expected: Market overprices options → sell premium
        If Actual consistently > Expected: Market underprices options → buy options
        """
        expected_move = atm_straddle_premium
        actual_move_high = abs(session_high - session_open)
        actual_move_low = abs(session_low - session_open)
        actual_move = max(actual_move_high, actual_move_low)
        total_range = session_high - session_low

        if expected_move > 0:
            move_ratio = actual_move / expected_move
            range_ratio = total_range / (2.0 * expected_move)
        else:
            move_ratio = 1.0
            range_ratio = 1.0

        if move_ratio < 0.65:
            accuracy = "OVERPRICED"
            edge = "Straddle sellers won. Options were overpriced relative to actual movement."
        elif move_ratio > 1.35:
            accuracy = "UNDERPRICED"
            edge = "Straddle buyers won. Actual move exceeded expected — vol was underpriced."
        else:
            accuracy = "FAIR"
            edge = "ATM straddle was approximately fairly priced for today's range."

        return {
            "expected_move_pts": round(expected_move, 2),
            "actual_move_pts": round(actual_move, 2),
            "total_range_pts": round(total_range, 2),
            "move_ratio": round(move_ratio, 3),
            "range_ratio": round(range_ratio, 3),
            "accuracy": accuracy,
            "edge_insight": edge
        }

    # ────────────────── 5. Intraday Seasonality Quality Score ──────────────────

    @staticmethod
    def compute_intraday_quality_score(bar_time: str) -> Dict[str, Any]:
        """
        Assigns a trading quality score based on time-of-day seasonality patterns
        in Nifty 50 intraday markets.

        Returns a quality_score (0.0 to 1.0) and a human-readable description.
        """
        hour_min = bar_time[:5] if len(bar_time) >= 5 else "12:00"

        # Define quality zones
        quality_map = {
            # (start, end): (score, label, description)
            ("09:15", "09:30"): (0.0, "FREAK_CANDLE", "Opening range isolation — no trades"),
            ("09:30", "10:15"): (1.0, "PRIME_WINDOW_1", "Initial Balance formation — best breakout window"),
            ("10:15", "11:30"): (0.85, "PULLBACK_WINDOW", "First pullback — Golden Pocket / mean-reversion entries"),
            ("11:30", "13:00"): (0.35, "LUNCH_LULL", "Lowest volume, choppy price action — reduce size, tighten stops"),
            ("13:00", "14:00"): (0.75, "EUROPEAN_OVERLAP", "European open influence — re-entry window"),
            ("14:00", "14:30"): (0.80, "LATE_ACCUMULATION", "Pre-close accumulation — watch for IB expansion"),
            ("14:30", "15:15"): (0.95, "PRIME_WINDOW_2", "Institutional MOC flow — strongest trend moves"),
            ("15:15", "15:30"): (0.10, "SQUAREOFF_CHAOS", "Square-off chaos — wide spreads, unpredictable"),
        }

        for (start, end), (score, label, desc) in quality_map.items():
            if start <= hour_min < end:
                return {
                    "quality_score": score,
                    "quality_label": label,
                    "description": desc,
                    "is_lunch_lull": label == "LUNCH_LULL",
                    "is_prime_window": score >= 0.85,
                    "sizing_multiplier": round(max(0.3, min(1.0, score)), 2),
                    "bar_time": bar_time
                }

        # Default (outside market hours)
        return {
            "quality_score": 0.0,
            "quality_label": "OUTSIDE_HOURS",
            "description": "Outside regular trading hours",
            "is_lunch_lull": False,
            "is_prime_window": False,
            "sizing_multiplier": 0.0,
            "bar_time": bar_time
        }

    # ────────────────── 6. Unified Vol Intelligence Report ──────────────────

    def generate_vol_intelligence_report(
        self,
        close_prices: pd.Series,
        current_iv: float,
        atm_straddle_premium: float = 0.0,
        session_high: float = 0.0,
        session_low: float = 0.0,
        session_open: float = 0.0,
        bar_time: str = "12:00",
        iv_history: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Master orchestrator: generates the complete Volatility Intelligence Report
        combining all edge metrics into a single actionable output.
        """
        # 1. Realized Volatility
        rv_data = self.compute_realized_volatility(close_prices, window=20)

        # 2. IV-RV Spread
        spread_data = self.compute_iv_rv_spread(current_iv, rv_data["realized_vol"])

        # 3. IV Percentile
        pctile_data = self.compute_iv_percentile(current_iv, iv_history)

        # 4. Expected vs Actual Move
        if atm_straddle_premium > 0 and session_high > 0:
            move_data = self.compute_expected_vs_actual_move(
                atm_straddle_premium, session_high, session_low, session_open
            )
        else:
            move_data = {
                "expected_move_pts": 0.0, "actual_move_pts": 0.0,
                "move_ratio": 1.0, "accuracy": "N/A", "edge_insight": "No straddle data available"
            }

        # 5. Intraday Quality Score
        quality_data = self.compute_intraday_quality_score(bar_time)

        # 6. Composite Vol Regime Decision
        # Weighted synthesis of IV-RV spread + IV percentile + expected move accuracy
        sell_vol_score = 0.0
        buy_vol_score = 0.0

        # IV-RV spread signal (strongest weight)
        if spread_data["spread_ratio"] > 0.15:
            sell_vol_score += 0.50
        elif spread_data["spread_ratio"] < -0.10:
            buy_vol_score += 0.50
        else:
            sell_vol_score += 0.10
            buy_vol_score += 0.10

        # IV percentile signal
        if pctile_data["iv_percentile"] >= 75:
            sell_vol_score += 0.30
        elif pctile_data["iv_percentile"] <= 25:
            buy_vol_score += 0.30
        else:
            sell_vol_score += 0.10
            buy_vol_score += 0.10

        # Expected move accuracy signal
        if move_data["accuracy"] == "OVERPRICED":
            sell_vol_score += 0.20
        elif move_data["accuracy"] == "UNDERPRICED":
            buy_vol_score += 0.20
        else:
            sell_vol_score += 0.05
            buy_vol_score += 0.05

        if sell_vol_score > buy_vol_score + 0.15:
            composite_regime = "SELL_VOL"
            composite_advice = (
                "📉 Volatility is overpriced. Favor premium-selling structures: "
                "Iron Condors, Credit Spreads, or Ratio Spreads. "
                "If taking directional trades, use debit spreads to offset theta."
            )
        elif buy_vol_score > sell_vol_score + 0.15:
            composite_regime = "BUY_VOL"
            composite_advice = (
                "📈 Volatility is underpriced. Options are cheap — buy convexity: "
                "Outright Calls/Puts for directional, or Straddles/Strangles for non-directional. "
                "Gamma will reward you if the market moves."
            )
        else:
            composite_regime = "NEUTRAL_VOL"
            composite_advice = (
                "⚖️ Volatility is fairly priced. Use standard directional delta strategies. "
                "No structural edge in buying or selling premium."
            )

        return {
            "realized_vol": rv_data,
            "iv_rv_spread": spread_data,
            "iv_percentile": pctile_data,
            "expected_vs_actual": move_data,
            "intraday_quality": quality_data,
            "composite_vol_regime": composite_regime,
            "composite_advice": composite_advice,
            "sell_vol_score": round(sell_vol_score, 2),
            "buy_vol_score": round(buy_vol_score, 2)
        }
