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
from typing import Dict, Any, Optional, List, Tuple, Union
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

    # ────────────────── 7. Volatility Cone Analysis ──────────────────

    @staticmethod
    def compute_volatility_cone(
        close_prices: Union[pd.Series, np.ndarray, List[float]],
        windows: Optional[List[int]] = None,
        percentiles: Optional[List[int]] = None,
        annualization_factor: float = 252 * 75
    ) -> Dict[str, Any]:
        """
        Constructs an institutional Volatility Cone across multiple lookback windows
        ([5, 10, 20, 60]). Computes historical rolling realized volatility distributions
        and percentile quantiles ([10, 25, 50, 75, 90]) to pinpoint volatility coiling/expansion.
        """
        if windows is None:
            windows = [5, 10, 20, 60]
        if percentiles is None:
            percentiles = [10, 25, 50, 75, 90]

        if isinstance(close_prices, pd.DataFrame):
            prices_arr = close_prices["close"].values if "close" in close_prices.columns else close_prices.iloc[:, 0].values
        elif isinstance(close_prices, pd.Series):
            prices_arr = close_prices.values
        else:
            prices_arr = np.array(close_prices, dtype=np.float64)

        prices_arr = np.maximum(prices_arr, 1.0)
        n_points = len(prices_arr)

        cone_table = []
        cone_data = {}

        if n_points < 3:
            for w in windows:
                w_dict = {
                    "window": w, "min": 0.08, "p10": 0.10, "p25": 0.12,
                    "p50": 0.14, "p75": 0.16, "p90": 0.18, "max": 0.22,
                    "mean": 0.14, "current_rv": 0.14, "rv_percentile": 50.0
                }
                cone_data[w] = w_dict
                cone_table.append(w_dict)
            return {
                "windows": windows,
                "percentiles": percentiles,
                "cone_data": cone_data,
                "cone_dataframe": pd.DataFrame(cone_table),
                "summary": "Insufficient price history; returning baseline volatility cone."
            }

        log_returns = np.diff(np.log(prices_arr))

        for w in windows:
            effective_w = min(w, len(log_returns))
            if effective_w < 2:
                roll_rvs = np.array([0.14])
            else:
                roll_rvs = []
                for i in range(effective_w, len(log_returns) + 1):
                    window_slice = log_returns[i - effective_w : i]
                    std_val = float(np.std(window_slice, ddof=1)) * np.sqrt(annualization_factor)
                    roll_rvs.append(std_val)
                roll_rvs = np.array(roll_rvs) if roll_rvs else np.array([0.14])

            current_rv = float(roll_rvs[-1]) if len(roll_rvs) > 0 else 0.14
            min_val = float(np.min(roll_rvs))
            max_val = float(np.max(roll_rvs))
            mean_val = float(np.mean(roll_rvs))

            pctile_values = {}
            for p in percentiles:
                pctile_values[f"p{p}"] = round(float(np.percentile(roll_rvs, p)), 4)

            count_below = sum(1 for v in roll_rvs if v < current_rv)
            rv_rank = round((count_below / max(len(roll_rvs), 1)) * 100.0, 1)

            w_record = {
                "window": w,
                "min": round(min_val, 4),
                **pctile_values,
                "max": round(max_val, 4),
                "mean": round(mean_val, 4),
                "current_rv": round(current_rv, 4),
                "rv_percentile": rv_rank
            }
            cone_data[w] = w_record
            cone_table.append(w_record)

        df_cone = pd.DataFrame(cone_table)
        short_w = windows[0]
        short_pct = cone_data[short_w]["rv_percentile"]
        if short_pct <= 20.0:
            summary_txt = f"Extreme Volatility Compression: Short-term ({short_w}-period) RV is in bottom {short_pct:.0f}% quantile."
        elif short_pct >= 80.0:
            summary_txt = f"Extreme Volatility Expansion: Short-term ({short_w}-period) RV is in top {100-short_pct:.0f}% quantile."
        else:
            summary_txt = f"Normal Volatility Regime: Short-term ({short_w}-period) RV is at {short_pct:.0f}th percentile."

        return {
            "windows": windows,
            "percentiles": percentiles,
            "cone_data": cone_data,
            "cone_dataframe": df_cone,
            "summary": summary_txt
        }

    # ────────────────── 8. RV Term Structure & Breakout Signal ──────────────────

    @staticmethod
    def compute_rv_term_structure(
        close_prices: Union[pd.Series, np.ndarray, List[float]],
        annualization_factor: float = 252 * 75
    ) -> Dict[str, Any]:
        """
        Calculates 5d, 10d, 20d, 60d (or period equivalent) Realized Volatility term structure.
        Classifies term structure as 'INVERTED_EXPANDING', 'NORMAL_COMPRESSING', or 'FLAT',
        and generates an institutional Compression Breakout Signal.
        """
        if isinstance(close_prices, pd.DataFrame):
            prices = close_prices["close"] if "close" in close_prices.columns else close_prices.iloc[:, 0]
        elif isinstance(close_prices, list):
            prices = pd.Series(close_prices)
        else:
            prices = close_prices

        rv_5 = VolatilityIntelligence.compute_realized_volatility(prices, window=5, annualization_factor=annualization_factor)["realized_vol"]
        rv_10 = VolatilityIntelligence.compute_realized_volatility(prices, window=10, annualization_factor=annualization_factor)["realized_vol"]
        rv_20 = VolatilityIntelligence.compute_realized_volatility(prices, window=20, annualization_factor=annualization_factor)["realized_vol"]
        rv_60 = VolatilityIntelligence.compute_realized_volatility(prices, window=60, annualization_factor=annualization_factor)["realized_vol"]

        slope = round(rv_60 - rv_5, 4)
        compression_ratio = round(rv_5 / max(rv_20, 0.001), 3)

        if rv_5 > rv_20 * 1.10 or rv_5 > rv_60 * 1.15:
            classification = "INVERTED_EXPANDING"
            commentary = "Short-term RV significantly exceeds baseline. Volatility is expanding rapidly (shock/trend momentum)."
        elif rv_5 < rv_20 * 0.90 or rv_5 < rv_60 * 0.85:
            classification = "NORMAL_COMPRESSING"
            commentary = "Short-term RV below baseline. Volatility is compressing (range consolidation / mean-reversion)."
        else:
            classification = "FLAT"
            commentary = "Realized volatility term structure is balanced and flat across maturities."

        is_breakout = bool(compression_ratio < 0.70 or (rv_5 < 0.09 and compression_ratio < 0.85))
        if is_breakout:
            breakout_msg = "VOLATILITY_COILING_ALERT: RV5 is severely suppressed relative to RV20 (ratio < 0.70). High probability of sharp expansion."
        else:
            breakout_msg = "Normal volatility dispersion; no active coiling squeeze detected."

        return {
            "rv_5": round(rv_5, 4),
            "rv_10": round(rv_10, 4),
            "rv_20": round(rv_20, 4),
            "rv_60": round(rv_60, 4),
            "term_structure_slope": slope,
            "compression_ratio": compression_ratio,
            "classification": classification,
            "compression_breakout_signal": is_breakout,
            "breakout_commentary": breakout_msg,
            "commentary": commentary
        }

    # ────────────────── 9. IV Term Structure & Contango/Backwardation ──────────────────

    @staticmethod
    def compute_iv_term_structure(
        expiry_iv_pairs: Union[List[Tuple[Any, float]], List[Dict[str, Any]], Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Computes Implied Volatility term structure across multiple expiries.
        Determines near vs far IV spread, classifies Contango vs Backwardation,
        and provides structural options trading recommendations.
        """
        parsed_pairs: List[Tuple[str, float]] = []

        if isinstance(expiry_iv_pairs, dict):
            for k, v in expiry_iv_pairs.items():
                parsed_pairs.append((str(k), float(v)))
        elif isinstance(expiry_iv_pairs, list):
            for item in expiry_iv_pairs:
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    parsed_pairs.append((str(item[0]), float(item[1])))
                elif isinstance(item, dict):
                    exp = item.get("expiry") or item.get("dte") or item.get("name") or "Expiry"
                    iv = item.get("iv") or item.get("implied_vol") or 0.14
                    parsed_pairs.append((str(exp), float(iv)))

        if not parsed_pairs:
            parsed_pairs = [("Near Expiry", 0.135), ("Far Expiry", 0.145)]

        near_label, near_iv = parsed_pairs[0]
        far_label, far_iv = parsed_pairs[-1]

        iv_spread = round(far_iv - near_iv, 4)
        spread_bps = round(iv_spread * 10000.0, 1)

        if iv_spread > 0.005:
            term_structure = "CONTANGO"
            implication = (
                "Normal Contango: Longer-dated options carry higher IV. "
                "Front-month options suffer less Vega drag; favorable for calendar spreads and long near-dated convexity."
            )
        elif iv_spread < -0.005:
            term_structure = "BACKWARDATION"
            implication = (
                "Inverted Backwardation: Front-month IV is elevated due to immediate event risk / panic. "
                "Premium is rich on the near-term curve; favor selling near-dated premium or debit diagonal spreads."
            )
        else:
            term_structure = "FLAT"
            implication = (
                "Flat IV Term Structure: Implied volatility is uniform across maturities. "
                "Standard directional structures without term spread edge."
            )

        curve = [{"expiry": p[0], "iv": round(p[1], 4), "iv_pct": round(p[1] * 100.0, 2)} for p in parsed_pairs]

        return {
            "near_expiry": near_label,
            "near_iv": round(near_iv, 4),
            "far_expiry": far_label,
            "far_iv": round(far_iv, 4),
            "iv_spread": iv_spread,
            "spread_bps": spread_bps,
            "term_structure": term_structure,
            "curve": curve,
            "implication": implication
        }

