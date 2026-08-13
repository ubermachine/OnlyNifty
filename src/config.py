"""Trading and configuration constants for JustNifty v2.0 institutional trading engine."""

# Capital and Risk Management Constants
DEFAULT_CAPITAL: float = 500000.0  # ₹5,00,000 reference institutional capital
MAX_RISK_PCT: float = 0.01          # Strict 1% risk per trade
LOT_SIZE: int = 25                 # Nifty 50 derivative contract lot size
ENOUGH_PROFIT_PCT: float = 0.003   # 0.3% Daily target threshold to lock profit & stop trading

# Technical Indicator Parameters
EMA_FAST: int = 21                 # Momentum & trailing stop MA
EMA_MID: int = 55                  # Intermediate trend filter MA
EMA_SLOW: int = 200                # Primary market regime filter MA
ENVELOPE_PCT: float = 0.015        # 1.5% 200 EMA extreme bands for mechanical part-booking

# Fibonacci Golden Pocket & Invalidation Levels
FIB_GOLDEN_MIN: float = 0.50       # 50.0% Retracement boundary
FIB_GOLDEN_MAX: float = 0.618      # 61.8% Retracement boundary
FIB_SL_LONG: float = 0.786         # 78.6% Retracement boundary + 5 points for Long SL
FIB_SL_SHORT: float = 0.382        # 38.2% Retracement boundary + 5 points for Short SL
MA_STRETCH_THRESHOLD: float = 0.0035  # 0.35% distance threshold from 21/55 EMA (Query 12 filter)

# Institutional Options Strike Selection & Greeks
DELTA_MIN: float = 0.50            # Minimum Delta for ATM/1-strike ITM options
DELTA_MAX: float = 0.65            # Maximum Delta for institutional directional buying
RISK_FREE_RATE: float = 0.065      # 6.5% RBI reference repo rate
DEFAULT_IV: float = 0.12           # 12.0% India VIX / IV baseline

# Session Timing in Asia/Kolkata (IST)
SESSION_START: str = "09:15"
OPENING_RANGE_END: str = "09:30"
MIDDAY_CHOP_START: str = "11:30"
MIDDAY_CHOP_END: str = "13:30"
THREE_PM_CANDLE: str = "15:00"
SESSION_END: str = "15:30"
