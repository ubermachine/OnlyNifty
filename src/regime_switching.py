"""
OnlyNifty v3.4 Markov Regime-Switching & Kalman Filter Velocity Engine.

Provides:
1. KalmanFilterTrendEstimator: Latent state-space price & velocity estimation with zero-lag noise reduction.
2. MarkovRegimeSwitcher: 3-State Gaussian Hidden Markov Model for dynamic volatility regime inference.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd


class KalmanFilterTrendEstimator:
    """
    1D Constant Velocity State-Space Kalman Filter:
    State vector: x_t = [position, velocity]^T
    State Transition: F = [[1, dt], [0, 1]]
    Measurement Matrix: H = [[1, 0]]
    Process Noise Covariance: Q
    Measurement Noise Covariance: R
    """
    def __init__(self, process_noise_std: float = 0.8, measurement_noise_std: float = 3.5, dt: float = 1.0):
        self.dt = dt
        self.F = np.array([[1.0, self.dt],
                           [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        
        # Discrete-time process noise covariance
        q_pos = (process_noise_std ** 2) * (self.dt ** 3) / 3.0
        q_vel = (process_noise_std ** 2) * (self.dt)
        q_pos_vel = (process_noise_std ** 2) * (self.dt ** 2) / 2.0
        self.Q = np.array([[q_pos, q_pos_vel],
                           [q_pos_vel, q_vel]])
        
        self.R = np.array([[measurement_noise_std ** 2]])
        
        # State estimate and error covariance
        self.x = np.zeros((2, 1))
        self.P = np.eye(2) * 50.0
        self.is_initialized = False

    def update(self, price: float) -> Tuple[float, float, float]:
        """
        Updates the filter with a new observed price and returns (filtered_price, velocity, velocity_zscore).
        """
        if not self.is_initialized:
            self.x = np.array([[price], [0.0]])
            self.P = np.eye(2) * 20.0
            self.is_initialized = True
            return price, 0.0, 0.0

        # Predict Step
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Measurement Update (Innovation)
        z = np.array([[price]])
        y = z - self.H @ x_pred  # Innovation
        S = self.H @ P_pred @ self.H.T + self.R # Innovation Covariance
        K = P_pred @ self.H.T @ np.linalg.inv(S) # Optimal Kalman Gain

        self.x = x_pred + K @ y
        I = np.eye(2)
        self.P = (I - K @ self.H) @ P_pred

        filtered_price = float(self.x[0, 0])
        velocity = float(self.x[1, 0])
        vel_variance = float(self.P[1, 1])
        velocity_zscore = float(velocity / np.sqrt(max(vel_variance, 1e-6)))

        return filtered_price, velocity, velocity_zscore

    def filter_series(self, prices: pd.Series) -> pd.DataFrame:
        """Runs the Kalman filter over an entire price series."""
        filtered_prices = []
        velocities = []
        vel_zscores = []

        # Reset filter state for fresh series run
        self.is_initialized = False
        for p in prices:
            fp, vel, z = self.update(float(p))
            filtered_prices.append(fp)
            velocities.append(vel)
            vel_zscores.append(z)

        return pd.DataFrame({
            "kalman_price": filtered_prices,
            "kalman_velocity": velocities,
            "kalman_vel_zscore": vel_zscores
        }, index=prices.index)


class MarkovRegimeSwitcher:
    """
    3-State Gaussian Hidden Markov Volatility & Trend Model:
    - State 0: LOW_VOL_TRENDING (High persistence, low return volatility, directional drift)
    - State 1: HIGH_VOL_EXPANSION (High return variance, wide breakout moves)
    - State 2: MEAN_REVERTING_CHOP (Near-zero drift, fast mean-reversion, choppy wicks)
    """
    def __init__(self):
        # 3 States: [0: LOW_VOL_TREND, 1: HIGH_VOL_EXPANSION, 2: MEAN_REVERTING_CHOP]
        self.state_names = ["LOW_VOL_TRENDING", "HIGH_VOL_EXPANSION", "MEAN_REVERTING_CHOP"]
        
        # State Prior Probabilities
        self.pi = np.array([0.45, 0.20, 0.35])
        
        # Transition Probability Matrix A (Rows = from_state, Cols = to_state)
        # Highly persistent regimes
        self.A = np.array([
            [0.85, 0.08, 0.07], # From Low-Vol Trend
            [0.15, 0.70, 0.15], # From High-Vol Expansion
            [0.10, 0.10, 0.80]  # From Mean-Reverting Chop
        ])
        
        # Emission Parameters (Mean drift %, Return Volatility % per 5m bar)
        self.means = np.array([0.0003, 0.0000, 0.0000])
        self.sigmas = np.array([0.0008, 0.0025, 0.0012])

    def compute_emission_density(self, ret: float, state_idx: int) -> float:
        """Gaussian probability density of observing return `ret` given hidden state `state_idx`."""
        mu = self.means[state_idx]
        sigma = self.sigmas[state_idx]
        denom = np.sqrt(2.0 * np.pi) * sigma
        exp_term = np.exp(-0.5 * ((ret - mu) / sigma) ** 2)
        return max(exp_term / denom, 1e-12)

    def infer_regimes(self, df_5m: pd.DataFrame, window: int = 30) -> Dict[str, Any]:
        """
        Runs Forward Algorithm to compute smoothed posterior state probabilities P(S_t = k | Y_1:t).
        """
        if df_5m.empty or len(df_5m) < 10:
            return {
                "active_regime": "LOW_VOL_TRENDING",
                "state_probabilities": {"LOW_VOL_TRENDING": 0.5, "HIGH_VOL_EXPANSION": 0.2, "MEAN_REVERTING_CHOP": 0.3},
                "kelly_multiplier": 1.0,
                "target_scaling": "NORMAL_3TIER",
                "entropy": 0.50
            }

        prices = df_5m["close"].values
        recent_prices = prices[-window:] if len(prices) >= window else prices
        returns = np.diff(np.log(np.maximum(recent_prices, 1.0)))

        if len(returns) == 0:
            return {
                "active_regime": "LOW_VOL_TRENDING",
                "state_probabilities": {"LOW_VOL_TRENDING": 0.5, "HIGH_VOL_EXPANSION": 0.2, "MEAN_REVERTING_CHOP": 0.3},
                "kelly_multiplier": 1.0,
                "target_scaling": "NORMAL_3TIER",
                "entropy": 0.50
            }

        # Forward recursion
        alpha = self.pi * np.array([self.compute_emission_density(returns[0], k) for k in range(3)])
        alpha /= max(np.sum(alpha), 1e-12)

        for t in range(1, len(returns)):
            ret = returns[t]
            emissions = np.array([self.compute_emission_density(ret, k) for k in range(3)])
            alpha_next = (alpha @ self.A) * emissions
            sum_alpha = np.sum(alpha_next)
            alpha = alpha_next / max(sum_alpha, 1e-12)

        p_low_vol = float(alpha[0])
        p_high_vol = float(alpha[1])
        p_chop = float(alpha[2])

        best_state_idx = int(np.argmax(alpha))
        active_regime = self.state_names[best_state_idx]

        # Shannon Entropy as regime uncertainty metric: H = - sum(p * log(p))
        entropy = - float(np.sum(alpha * np.log(np.maximum(alpha, 1e-12))))

        # Adaptive Kelly & Target Scalers
        if active_regime == "LOW_VOL_TRENDING":
            kelly_multiplier = 1.0
            target_scaling = "RUNNER_EXPANSION_T3"
            advice = "Full 1% Quarter-Kelly sizing active. Trail 30% runner to upper VAKC / AVWAP +2σ."
        elif active_regime == "HIGH_VOL_EXPANSION":
            kelly_multiplier = 0.75
            target_scaling = "ACCELERATED_FREE_SPREAD"
            advice = "Elevated Volatility. Convert immediately to Free Vertical Spread at T1 (+1.2x ATR)."
        else: # MEAN_REVERTING_CHOP
            kelly_multiplier = 0.50
            target_scaling = "SCALPING_T1_ONLY"
            advice = "Mean-reverting chop detected. Halve risk (0.5% Kelly) or take quick 15-20pt scalp."

        return {
            "active_regime": active_regime,
            "state_probabilities": {
                "LOW_VOL_TRENDING": round(p_low_vol, 3),
                "HIGH_VOL_EXPANSION": round(p_high_vol, 3),
                "MEAN_REVERTING_CHOP": round(p_chop, 3)
            },
            "kelly_multiplier": kelly_multiplier,
            "target_scaling": target_scaling,
            "entropy": round(entropy, 3),
            "advice": advice
        }
