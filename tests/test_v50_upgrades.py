import pytest
import numpy as np
import pandas as pd
from src.volatility_engine import VolatilityIntelligence

def test_realized_volatility_computation():
    # Create a trending price series with known std dev
    # We'll just provide a series of prices that go up steadily
    prices = pd.Series([100.0, 101.0, 102.01, 103.0301, 104.060401, 105.101005])
    # log returns will be approx 0.00995 constant
    # With a bit of noise to not have 0 std
    prices = pd.Series([100.0 * (1.01 ** i) for i in range(20)])
    # Add a bit of noise
    np.random.seed(42)
    noise = np.random.normal(1, 0.005, 20)
    prices = prices * noise
    
    rv_data = VolatilityIntelligence.compute_realized_volatility(prices, window=20)
    assert 'realized_vol' in rv_data
    assert rv_data['realized_vol'] > 0
    assert rv_data['window'] == 20

def test_iv_rv_spread_sell_vol():
    iv = 0.18
    rv = 0.10
    result = VolatilityIntelligence.compute_iv_rv_spread(iv, rv)
    assert result['spread'] > 0
    assert 'SELL' in result['vol_regime']
    assert result['structure_recommendation'] == 'SELL_PREMIUM'

def test_iv_rv_spread_buy_vol():
    iv = 0.08
    rv = 0.15
    result = VolatilityIntelligence.compute_iv_rv_spread(iv, rv)
    assert result['spread'] < 0
    assert 'BUY' in result['vol_regime']
    assert 'BUY' in result['structure_recommendation']

def test_iv_rv_spread_neutral():
    iv = 0.12
    rv = 0.12
    result = VolatilityIntelligence.compute_iv_rv_spread(iv, rv)
    assert result['vol_regime'] == 'NEUTRAL_VOL'

def test_iv_percentile_high_iv():
    engine = VolatilityIntelligence()
    iv_history = [0.08, 0.09, 0.10, 0.11, 0.12] * 4 # 20 items, max 0.12
    result = engine.compute_iv_percentile(0.20, iv_history)
    assert result['iv_percentile'] >= 80

def test_iv_percentile_low_iv():
    engine = VolatilityIntelligence()
    iv_history = [0.15, 0.18, 0.20, 0.22, 0.25] * 4 # 20 items, min 0.15
    result = engine.compute_iv_percentile(0.08, iv_history)
    assert result['iv_percentile'] <= 20

def test_expected_vs_actual_move_overpriced():
    # Expected move=150 pts, actual move=80 pts
    result = VolatilityIntelligence.compute_expected_vs_actual_move(
        atm_straddle_premium=150.0,
        session_high=10080.0,
        session_low=10000.0,
        session_open=10000.0
    )
    # Actual move is max(|10080-10000|, |10000-10000|) = 80
    assert result['accuracy'] == 'OVERPRICED'

def test_expected_vs_actual_move_underpriced():
    # Expected move=80 pts, actual move=180 pts
    result = VolatilityIntelligence.compute_expected_vs_actual_move(
        atm_straddle_premium=80.0,
        session_high=10180.0,
        session_low=9900.0, # not needed for actual move from open
        session_open=10000.0
    )
    # Actual move = 180
    assert result['accuracy'] == 'UNDERPRICED'

def test_intraday_quality_prime_window():
    result = VolatilityIntelligence.compute_intraday_quality_score('09:45')
    assert result['quality_score'] >= 0.85
    assert result['is_prime_window'] is True

def test_intraday_quality_lunch_lull():
    result = VolatilityIntelligence.compute_intraday_quality_score('12:30')
    assert result['quality_score'] <= 0.40
    assert result['is_lunch_lull'] is True

def test_intraday_quality_squareoff():
    result = VolatilityIntelligence.compute_intraday_quality_score('15:20')
    assert result['quality_score'] <= 0.15

def test_full_vol_intelligence_report():
    engine = VolatilityIntelligence()
    # Mock prices to avoid errors
    prices = pd.Series([100.0 * (1.001 ** i) for i in range(20)])
    iv_history = [0.15] * 20
    
    report = engine.generate_vol_intelligence_report(
        close_prices=prices,
        current_iv=0.18,
        atm_straddle_premium=100.0,
        session_high=10100.0,
        session_low=9950.0,
        session_open=10000.0,
        bar_time='10:00',
        iv_history=iv_history
    )
    
    assert 'realized_vol' in report
    assert 'iv_rv_spread' in report
    assert 'iv_percentile' in report
    assert 'expected_vs_actual' in report
    assert 'intraday_quality' in report
    assert 'composite_vol_regime' in report

def test_vol_regime_composite_sell():
    engine = VolatilityIntelligence()
    prices = pd.Series([100.0] * 20) # extremely low RV (0)
    iv_history = [0.10, 0.11, 0.12] * 7 # max 0.12
    report = engine.generate_vol_intelligence_report(
        close_prices=prices,
        current_iv=0.20, # High IV -> strong sell score, percentile 100
        atm_straddle_premium=150.0,
        session_high=10050.0, # Actual move 50
        session_low=10000.0,
        session_open=10000.0, # Accuracy OVERPRICED
        bar_time='10:00',
        iv_history=iv_history
    )
    assert report['composite_vol_regime'] == 'SELL_VOL'
