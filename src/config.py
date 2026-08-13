"""JustNifty v3.0 Tier-1 Institutional Hedge Fund Configuration Constants."""

# ----------------- CAPITAL & FRACTIONAL KELLY SIZING -----------------
DEFAULT_CAPITAL: float = 500000.0       # ₹5,00,000 baseline reference institutional capital
MAX_RISK_PCT: float = 0.01               # Strict 1.0% max risk per trade baseline
LOT_SIZE: int = 25                      # Nifty 50 derivative contract lot size
KELLY_FRACTION: float = 0.25            # Quarter-Kelly allocation for fat-tail safety
MAX_TOLERABLE_MDD: float = 0.10         # 10.0% Maximum tolerable portfolio drawdown
PROFIT_RATCHET_TRIGGER: float = 0.015   # Lock in 65% of peak gains once +1.5R (+1.5%) achieved

# ----------------- ADAPTIVE STOCHASTIC PILLARS -----------------
# 1. Fractional Hurst Exponent (H)
HURST_TRENDING_MIN: float = 0.55        # H > 0.55: Persistent / Trending regime
HURST_MEAN_REV_MAX: float = 0.45        # H < 0.45: Anti-Persistent / Mean-Reverting regime
HURST_WINDOW: int = 50                  # Rolling lookback window for Rescaled Range (R/S)

# 2. Volatility-Adaptive Keltner Channels (VAKC)
VAKC_LAMBDA: float = 2.25               # ATR multiplier for adaptive dispersion bands
VAKC_ATR_SPAN: int = 14                 # Average True Range lookback span
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
OFI_THRESHOLD: float = 0.0              # OFI > 0 confirms buyer defense at AVWAP
VPIN_TOXICITY_THRESHOLD: float = 0.65   # Volume-Synchronized Probability of Informed Trading limit

# ----------------- DERIVATIVES & SECOND-ORDER GREEKS -----------------
DELTA_MIN: float = 0.50                 # Minimum Delta for ATM / 1-ITM directional options
DELTA_MAX: float = 0.65                 # Maximum Delta for institutional directional buying
DELTA_DEEP_ITM_0DTE: float = 0.75       # Target Delta on Thursday 0DTE after 12:30 to avoid gamma cliff
RISK_FREE_RATE: float = 0.065           # 6.5% RBI reference repo rate
DEFAULT_IV: float = 0.12                # 12.0% India VIX / IV baseline
PUT_SKEW_PREMIUM: float = 0.025         # +250 bps structural downside put skew

# ----------------- TRANSACTION COST ANALYSIS (TCA) - NSE FRICTION -----------------
STT_SELL_PCT: float = 0.001             # 0.10% Securities Transaction Tax on sell turnover
BROKERAGE_PER_ORDER: float = 20.0       # ₹20 flat per executed order
NSE_TURNOVER_PCT: float = 0.0003503     # 0.03503% NSE exchange turnover charges
GST_PCT: float = 0.18                   # 18% GST on (Brokerage + Exchange Fees)
SEBI_CHARGES_PCT: float = 0.000001      # ₹10 per Crore
STAMP_DUTY_BUY_PCT: float = 0.00003     # 0.003% Stamp duty on buy turnover
DEFAULT_SLIPPAGE_PTS: float = 0.75      # ₹0.75 per share realistic round-trip slippage

# ----------------- SESSION TIMINGS (IST) -----------------
SESSION_START: str = "09:15"
OPENING_RANGE_END: str = "09:30"
EUROPEAN_OPEN_START: str = "13:00"
THREE_PM_CANDLE: str = "15:00"
HARD_SQUAREOFF_TIME: str = "15:15"      # Mandatory institutional intraday sweep close
SESSION_END: str = "15:30"
