"""JustNifty v3.0 Tier-1 Institutional Hedge Fund Configuration Constants."""
import pytz

IST = pytz.timezone("Asia/Kolkata")

# ----------------- CAPITAL & FRACTIONAL KELLY SIZING -----------------
DEFAULT_CAPITAL: float = 500000.0       # ₹5,00,000 baseline reference institutional capital
MAX_RISK_PCT: float = 0.01               # Strict 1.0% max risk per trade baseline
LOT_SIZE: int = 25                      # Nifty 50 derivative contract lot size
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

# 3. Volume-Weighted Fibonacci Golden Pocket
FIB_GOLDEN_MIN: float = 0.50            # 50.0% Retracement boundary
FIB_GOLDEN_MAX: float = 0.618           # 61.8% Retracement boundary
FIB_SL_LONG: float = 0.786              # 78.6% Retracement boundary + 5 points for Long SL
FIB_SL_SHORT: float = 0.786             # 78.6% Retracement boundary + 5 points for Short SL
MA_STRETCH_THRESHOLD: float = 0.0035    # 0.35% distance threshold from 21/55 EMA (Query 12 filter)

# 4. Order Flow Imbalance (OFI) & Cumulative Volume Delta (CVD)
OFI_ZSCORE_MIN: float = 0.65            # Minimum rolling Z-score OFI required to validate AVWAP defense
VPIN_TOXICITY_THRESHOLD: float = 0.65   # Volume-Synchronized Probability of Informed Trading limit

# ----------------- DERIVATIVES & SECOND-ORDER GREEKS -----------------
DELTA_MIN: float = 0.50                 # Minimum Delta for ATM / 1-ITM directional options
DELTA_MAX: float = 0.65                 # Maximum Delta for institutional directional buying
DELTA_DEEP_ITM_0DTE: float = 0.78       # Target Delta on Thursday 0DTE after 12:30 (Deep ITM Synthetic)
RISK_FREE_RATE: float = 0.065           # 6.5% RBI reference repo rate
DEFAULT_IV: float = 0.12                # 12.0% India VIX / IV baseline
PUT_SKEW_PREMIUM: float = 0.025         # +250 bps structural downside put skew

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

