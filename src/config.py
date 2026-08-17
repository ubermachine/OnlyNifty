from typing import Tuple
import pytz

IST = pytz.timezone("Asia/Kolkata")

# ----------------- CAPITAL & FRACTIONAL KELLY SIZING -----------------
DEFAULT_CAPITAL: float = 500000.0       # ₹5,00,000 baseline reference institutional capital
MAX_RISK_PCT: float = 0.01               # Strict 1.0% max risk per trade baseline
# !! VERIFY AGAINST YOUR BROKER'S CONTRACT MASTER BEFORE TRADING !!
# NSE revised the Nifty lot size 25 -> 75 (SEBI Oct-2024 circular raising the minimum
# F&O contract value to ~Rs.15 lakh). This value scales EVERY position-size, R-multiple
# and margin figure in the system: if it reads 25 while the real contract is 75, each
# ticket recommends 3x the intended lots and silently risks 3x the capital budget.
# Erring high under-trades; erring low over-risks — hence the conservative default.
LOT_SIZE: int = 75                      # Nifty 50 derivative contract lot size
KELLY_FRACTION: float = 0.25            # Quarter-Kelly allocation for fat-tail safety
MAX_TOLERABLE_MDD: float = 0.10         # 10.0% Maximum tolerable portfolio drawdown
DAILY_LOSS_LIMIT_PCT: float = 0.015     # 1.5% Hard Daily Loss Limit (DLL) circuit breaker
MAX_CONSECUTIVE_LOSSES_DAY: int = 2     # 2-Strike Rule: Halt intraday trading after 2 consecutive losses
MAX_TRADES_PER_DAY: int = 3             # Daily trade frequency ceiling to prevent overtrading & TCA drag
PROFIT_RATCHET_TRIGGER: float = 0.015   # Lock in 65% of peak gains once +1.5R (+1.5%) achieved

# ----------------- ADAPTIVE STOCHASTIC PILLARS -----------------
# 1. Fractional Hurst Exponent (H)
HURST_TRENDING_MIN: float = 0.52        # H >= 0.52: Persistent / Trending regime (Anis-Lloyd corrected)
HURST_MEAN_REV_MAX: float = 0.45        # H < 0.45: Anti-Persistent / Mean-Reverting regime
HURST_WINDOW: int = 60                  # Rolling lookback window for Rescaled Range (R/S)

# 2. Volatility-Adaptive Keltner Channels (VAKC)
VAKC_LAMBDA: float = 2.25               # ATR multiplier for adaptive dispersion bands
VAKC_ATR_SPAN: int = 14                 # Average True Range lookback span (Wilder RMA)
EMA_FAST: int = 21                      # Momentum & micro-trailing EMA
EMA_MID: int = 55                       # Intermediate trend filter EMA
EMA_SLOW: int = 200                     # Primary market regime filter EMA
ENVELOPE_PCT: float = 0.015             # 1.5% 200 EMA extreme band fallback

# 2b. 6-Line Break & Monotonic EMA Stack Trend Engine
LINE_BREAK_COUNT: int = 6               # 6-Block price reversal threshold (asymmetric trend confirmation)
LINE_BREAK_EMA_PERIODS: Tuple[int, int, int] = (15, 20, 50)  # Monotonic EMA filter stack

# 3. Volume-Weighted Fibonacci Golden Pocket
FIB_GOLDEN_MIN: float = 0.50            # 50.0% Retracement boundary
FIB_GOLDEN_MAX: float = 0.618           # 61.8% Retracement boundary
FIB_SL_LONG: float = 0.786              # 78.6% Retracement boundary + 5 points for Long SL
FIB_SL_SHORT: float = 0.786             # 78.6% Retracement boundary + 5 points for Short SL
MA_STRETCH_THRESHOLD: float = 0.0035    # 0.35% distance threshold from 21/55 EMA (Query 12 filter)

# 4. Order Flow Imbalance (OFI) & Cumulative Volume Delta (CVD)
OFI_ZSCORE_MIN: float = 0.65            # Minimum rolling Z-score OFI required to validate AVWAP defense
VPIN_TOXICITY_THRESHOLD: float = 0.75   # Volume-Synchronized Probability of Informed Trading limit (0.75 institutional consensus)

# ----------------- DERIVATIVES & SECOND-ORDER GREEKS -----------------
DELTA_MIN: float = 0.65                 # Minimum Delta for Deep ITM directional options (cuts extrinsic decay)
DELTA_MAX: float = 0.85                 # Maximum Delta for institutional directional buying
DELTA_TARGET: float = 0.75              # Target institutional execution Delta
DELTA_DEEP_ITM_0DTE: float = 0.80       # Target Delta on 0DTE after 12:30 (Deep ITM Synthetic)
RISK_FREE_RATE: float = 0.065           # 6.5% RBI reference repo rate
DEFAULT_IV: float = 0.12                # 12.0% India VIX / IV baseline
PUT_SKEW_PREMIUM: float = 0.025         # +250 bps structural downside put skew (PRICING ONLY — never applied to gamma, see below)

# Adverse IV drift penalties in target projections (Vega crush modeling)
IV_ADVERSE_DRIFT_T1: float = 0.015      # -1.5 vol pts IV crush at Target 1
IV_ADVERSE_DRIFT_T2: float = 0.020      # -2.0 vol pts IV crush at Target 2
IV_ADVERSE_DRIFT_T3: float = 0.025      # -2.5 vol pts IV crush at Target 3

# NSE has revised the Nifty weekly expiry day more than once (historically Thursday,
# later moved to Tuesday). VERIFY THIS AGAINST THE CURRENT NSE CIRCULAR before trading —
# it gates all 0DTE/expiry-pin/charm logic. Python weekday(): Mon=0, Tue=1, Wed=2, Thu=3.
NIFTY_WEEKLY_EXPIRY_WEEKDAY: int = 1    # 1 = Tuesday (CONFIRM against current NSE contract spec)

# ----------------- TRANSACTION COST ANALYSIS (TCA) - NSE FRICTION -----------------
STT_SELL_PCT: float = 0.001             # 0.10% Securities Transaction Tax on sell turnover (Oct 2024 mandate)
BROKERAGE_PER_ORDER: float = 20.0       # ₹20 flat per executed order
NSE_TURNOVER_PCT: float = 0.0003503     # 0.03503% NSE exchange turnover charges
GST_PCT: float = 0.18                   # 18% GST on (Brokerage + Exchange Fees)
SEBI_CHARGES_PCT: float = 0.000001      # ₹10 per Crore
STAMP_DUTY_BUY_PCT: float = 0.00003     # 0.003% Stamp duty on buy turnover
DEFAULT_SLIPPAGE_PTS: float = 0.75      # ₹0.75 baseline per share slippage

# ----------------- RISK MANAGEMENT & MONTE CARLO UPGRADES (v3.2) -----------------
GOLDEN_VAULT_TRIGGER_PCT: float = 0.015   # +1.5% Intraday Net PnL activates the Golden Vault
GOLDEN_VAULT_LOCK_PCT: float = 0.75       # 75% of peak intraday gains permanently locked
MONTE_CARLO_PATHS: int = 1000             # 1,000 Vectorized Monte Carlo simulation paths
MONTE_CARLO_HORIZON_TRADES: int = 100     # 100 consecutive trade stress horizon
RUIN_MDD_THRESHOLD: float = 0.50          # 50.0% Max Drawdown ruin barrier

# ----------------- GAP & VOLATILITY EXTENSIONS (v3.2) -----------------
GAP_THRESHOLD_LARGE_PCT: float = 0.0050     # 0.50% threshold for Large Gap-Up / Gap-Down
GAP_DECAY_HALF_LIFE_MINS: float = 45.0      # Half-life for morning gap decay in minutes
VAKC_ELASTICITY_GAMMA: float = 0.50         # Core diffusion scaling exponent
VAKC_ELASTICITY_ALPHA: float = 0.35         # Fat-tail high-vol expansion weight
VAKC_ELASTICITY_BETA: float = 0.60          # Regime transition smoothness parameter
VAKC_ELASTICITY_MIN: float = 0.70           # Min elasticity floor (VIX ~ 7)
VAKC_ELASTICITY_MAX: float = 2.50           # Max elasticity ceiling (VIX ~ 35)
VALUE_AREA_PCT: float = 0.70                # 70.0% standard Value Area volume bracket
VA_REJECTION_WICK_MIN_RATIO: float = 0.35   # Minimum wick ratio (35%) to confirm Value Area rejection

# ----------------- SESSION TIMINGS (IST) -----------------
SESSION_START: str = "09:15"
OPENING_RANGE_END: str = "09:30"
EUROPEAN_OPEN_START: str = "13:00"
THREE_PM_CANDLE: str = "15:00"
HARD_SQUAREOFF_TIME: str = "15:15"      # Mandatory institutional intraday sweep close
SESSION_END: str = "15:30"

# ----------------- TOP 10 HEAVYWEIGHT CONSTITUENTS (55% WEIGHT) -----------------
TOP_10_NIFTY_CONSTITUENTS = [
    "HDFCBANK.NS", "RELIANCE.NS", "ICICIBANK.NS", "INFY.NS", "ITC.NS",
    "TCS.NS", "LT.NS", "AXISBANK.NS", "KOTAKBANK.NS", "BHARTIARTL.NS"
]

# ----------------- TELEGRAM NOTIFICATION WEBHOOK (v5.2) -----------------
import os
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_PARSE_MODE: str = "HTML"
TELEGRAM_TIMEOUT_SECONDS: float = 5.0
TELEGRAM_MIN_CONFLUENCE_SCORE: float = 70.0  # Standard Grade alert floor (>=85% is A+ Institutional)

# ----------------- QUANTITATIVE MICROSTRUCTURE & SKEW (v5.2) -----------------
SKEW_ZSCORE_THRESHOLD: float = 1.50         # Z-Score boundary for 25-Delta Put Skew spikes (Crash Risk Gate)
GEX_WALL_BUFFER_PTS: float = 15.0           # Spot distance to Call/Put Wall to trigger Pinning / Fade Gate
VCR_SQUEEZE_THRESHOLD: float = 0.15         # Realized Volatility Ratio (5d / 60d) compression threshold
CVD_ABSORPTION_THRESHOLD: float = 1.20      # Ratio of Delta expansion to Price displacement for Absorption

# ----------------- SIGNAL QUALITY & UNIVERSAL GATES (Phase 1-5) -----------------
SIGNAL_MIN_CONFLUENCE: float = 70.0         # Pre-decision score floor; below -> WAIT (Veto, not label)
COOLDOWN_BARS: int = 12                     # 60 min of 5m bars between fresh entries
MAX_OPEN_TRADES: int = 1                    # Single active position concurrency limit
GATE_FAIL_TO_WAIT: bool = True              # Missing gate data -> safe WAIT, never trade
# How many core gate inputs (25d skew, dealer walls, positioning flow) must be missing
# before the desk stands aside entirely.
#
# All three derive from the option chain and therefore fail together: the real-world
# condition this guards is "the chain is gone", which shows up as all three missing at
# once. Partial degradation (e.g. walls unverified while skew is real) is handled by
# POSITIONING_UNVERIFIED_SIZE_CAP reducing size rather than by blocking, so the desk
# stays useful when data is merely thin and stands aside only when it is truly blind.
GATE_MIN_MISSING_TO_BLOCK: int = 3

LUNCH_LULL_SIZE_FACTOR: float = 0.5         # Halved sizing during 11:30-13:00 IST lunch lull
STOP_MIN_ATR_FRACTION: float = 0.5          # SL >= 0.5 x ATR14
STOP_MAX_POINTS: float = 60.0               # Absolute SL cap (index points)
STOP_NOISE_BAND_MULT: float = 2.0           # SL >= 2 x rolling 5m bar range sigma
VPIN_TOXICITY_THRESHOLD: float = 0.75       # Toxic order flow veto boundary (0.75 consensus)
MAX_CONSECUTIVE_LOSSES_DAY: int = 2         # 2-strike daily circuit breaker
DAILY_LOSS_LIMIT_PCT: float = 0.015         # 1.5% max daily account loss limit
QUARANTINE_MIN_SAMPLES: int = 30            # Minimum sample size before setup can be TRUSTED

# Walk-forward observations overlap: signals may fire on consecutive bars while each
# outcome spans a 12-bar horizon, so trades are serially dependent. An iid bootstrap
# understates the CI and lets a marginal setup earn a confident TRUSTED. This variance
# inflation factor widens the interval. 2.0 is a conservative floor derived from the
# observed firing rate; it is NOT fitted. Replace with a stationary block bootstrap
# (mean block ~2x the outcome horizon) when the sample supports it.
EDGE_OVERLAP_VIF: float = 2.0

# Desk verdict conviction floor. Conviction was previously computed AFTER the action was
# chosen and used only to prefix the label, so a LOW-conviction setup still produced a
# full ticket. This makes it a veto. Deliberately set at the MODERATE boundary rather
# than higher: the weights feeding conviction are reasoned defaults, not fitted, so the
# floor should reject the clearly-weak rather than pretend to fine-grained selectivity.
MIN_CONVICTION_TO_TRADE: float = 45.0

# Net family evidence (weighted, [-1, +1]) that must oppose a trade before it counts as
# a named conflict. The four families are not fully independent (measured N_eff ~1.8),
# so this is set well above the noise floor.
EVIDENCE_OPPOSITION_THRESHOLD: float = 0.35



# ----------------- OPTIONS DESK & POSITIONING FUSION (v5.2) -----------------
POSITIONING_VETO_STRENGTH: float = 0.5      # D-vector threshold opposing chart setup to trigger WAIT
WALL_BUFFER_PTS: float = 25.0               # Buffer around Call/Put walls for range fade and pinning
POSITIONING_UNVERIFIED_SIZE_CAP: float = 0.5 # Max size factor when options positioning is unverified/synthetic
PCR_Z_CONTRARIAN_THRESHOLD: float = 2.0     # Z-Score threshold on PCR session history for contrarian fade
PCR_STRUCTURAL_BASELINE: float = 0.70       # Academic neutral PCR baseline (Blau et al. 2015), NOT 1.0
PCR_HISTORY_MIN_SAMPLES: int = 20           # Minimum session snapshots required before PCR Z-Score is verified
OPTIONS_STATE_PATH: str = "data/options_state.json"

# ----------------- SIGNAL IMPROVEMENT GATES (v5.3 Research-Driven) -----------------
IV_RANK_SPREAD_THRESHOLD: float = 0.50       # IV Percentile above which debit spreads preferred over naked longs
IV_RANK_CONVEXITY_THRESHOLD: float = 0.20    # IV Percentile below which naked long convexity has structural edge
VAL_BUFFER_ATR_MULT: float = 0.15            # ATR multiplier for VAL/VAH proximity buffer (replaces hardcoded 5pts)
GEX_WALL_BUFFER_ATR_MULT: float = 0.40       # ATR multiplier for GEX Wall proximity buffer (replaces hardcoded 15pts)
CPR_BUFFER_ATR_MULT: float = 0.10            # ATR multiplier for CPR level proximity buffer
TERM_STRUCTURE_BACKWARDATION_THRESHOLD: float = -0.02  # IV spread threshold for crisis backwardation detection
TERM_STRUCTURE_CRISIS_SIZE_MULT: float = 0.25          # Sizing multiplier during term structure inversion
GAMMA_SQUEEZE_TARGET_MULT: float = 2.0       # Target multiplier for gamma squeeze breakout trades
EXPIRY_PIN_MIN_DISTANCE_PTS: float = 50.0    # Minimum Max Pain distance to trigger expiry pin trade

# ----------------- REGIME-CONDITIONAL EVIDENCE WEIGHTS (v5.3) -----------------
# Dynamically adapts the 4 independent evidence family weights to the Markov / Volatility regime:
# 1. LOW_VOL_TRENDING: Structure and Directional Flow dominate
# 2. MEAN_REVERTING_CHOP: Options Desk Positioning (GEX Walls, Max Pain, Mean Reversion) dominates
# 3. HIGH_VOL_EXPANSION / Crisis: Macro shocks and Order Flow Toxicity dominate
REGIME_EVIDENCE_WEIGHTS = {
    "LOW_VOL_TRENDING": {
        "structure": 0.40,
        "flow": 0.30,
        "positioning": 0.20,
        "macro": 0.10
    },
    "MEAN_REVERTING_CHOP": {
        "structure": 0.15,
        "flow": 0.20,
        "positioning": 0.45,
        "macro": 0.20
    },
    "HIGH_VOL_EXPANSION": {
        "structure": 0.15,
        "flow": 0.35,
        "positioning": 0.15,
        "macro": 0.35
    },
    "DEFAULT": {
        "structure": 0.30,
        "flow": 0.25,
        "positioning": 0.30,
        "macro": 0.15
    }
}

