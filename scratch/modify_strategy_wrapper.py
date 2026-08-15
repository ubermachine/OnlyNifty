import re

def modify_strategy():
    with open('d:/antigravity_sandbox/Nifty/src/strategy_rules.py', 'r') as f:
        content = f.read()

    # 1. Imports
    if 'VolatilityIntelligence' not in content:
        content = content.replace(
            'from src.macro_engine import GlobalMacroEngine',
            'from src.macro_engine import GlobalMacroEngine\nfrom src.volatility_engine import VolatilityIntelligence'
        )
    if 'compute_initial_balance_and_day_type' not in content:
        content = content.replace(
            'compute_volume_synchronized_gamma_tracker\n)',
            'compute_volume_synchronized_gamma_tracker,\n    compute_initial_balance_and_day_type\n)'
        )

    # 2. Init
    if 'self.vol_intelligence = VolatilityIntelligence()' not in content:
        content = content.replace(
            'self.cointegrator = MultiAssetKalmanCointegrator()',
            'self.cointegrator = MultiAssetKalmanCointegrator()\n        self.vol_intelligence = VolatilityIntelligence()'
        )

    # 3. Rename evaluate_bar to _evaluate_bar_core
    content = content.replace('    def evaluate_bar(', '    def _evaluate_bar_core(')
    
    # 4. Insert new evaluate_bar wrapper
    wrapper_code = '''
    def evaluate_bar(
        self,
        df_5m: pd.DataFrame,
        current_idx: int = -1,
        df_daily: Optional[pd.DataFrame] = None,
        df_hourly: Optional[pd.DataFrame] = None,
        df_15m: Optional[pd.DataFrame] = None,
        df_1h: Optional[pd.DataFrame] = None,
        live_iv: float = DEFAULT_IV,
        live_vix: Optional[float] = None,
        pre_open_gap: Optional[Dict[str, Any]] = None,
        prev_close: Optional[float] = None
    ) -> Signal:
        self._last_vol_report = None
        self._last_lunch_lull = False
        
        signal = self._evaluate_bar_core(
            df_5m, current_idx, df_daily, df_hourly, df_15m, df_1h,
            live_iv, live_vix, pre_open_gap, prev_close
        )
        
        if getattr(self, '_last_vol_report', None) is not None:
            if signal.details is None:
                signal.details = {}
            signal.details['vol_report'] = self._last_vol_report
            
        if getattr(self, '_last_lunch_lull', False) and signal.signal_type != SignalType.WAIT:
            signal.reason += " [LUNCH LULL: Reduced confidence, halved sizing recommended]"
            
        return signal

'''
    # insert before _evaluate_bar_core
    content = content.replace('    def _evaluate_bar_core(', wrapper_code + '    def _evaluate_bar_core(')

    # 5. Lunch Lull
    if '# Lunch Lull Filter' not in content:
        lunch_lull_code = '''
        # Lunch Lull Filter
        lunch_lull_active = False
        if "11:30" <= bar_time <= "13:00":
            intraday_quality = self.vol_intelligence.compute_intraday_quality_score(bar_time)
            lunch_lull_active = True
        self._last_lunch_lull = lunch_lull_active
'''
        content = content.replace(
            'if len(sub_df) < 15:\n            return Signal(SignalType.WAIT, close, 0.0, 0.0, 0.0, 0.0, 0.0, "Accumulating bars for indicator stability", True, 0.0, {})\n',
            'if len(sub_df) < 15:\n            return Signal(SignalType.WAIT, close, 0.0, 0.0, 0.0, 0.0, 0.0, "Accumulating bars for indicator stability", True, 0.0, {})\n' + lunch_lull_code
        )

    # 6. Vol Report
    if '# Vol Intelligence: IV-RV Spread & Regime' not in content:
        vol_code = '''
        # Vol Intelligence: IV-RV Spread & Regime
        vol_report = self.vol_intelligence.generate_vol_intelligence_report(
            close_prices=sub_df['close'],
            current_iv=live_iv,
            bar_time=bar_time
        )
        vol_regime = vol_report['composite_vol_regime']
        intraday_quality = vol_report['intraday_quality']
        self._last_vol_report = vol_report
'''
        content = content.replace(
            'cointegration = self.cointegrator.evaluate_spread_divergence(sub_df["close"])',
            'cointegration = self.cointegrator.evaluate_spread_divergence(sub_df["close"])\n' + vol_code
        )

    # 7. Add Strategies
    if '# 4.2 Mean-Reversion Strategy' not in content:
        strategy_code = '''
        # 4.2 Mean-Reversion Strategy (Active in MEAN_REVERTING_CHOP regime)
        if markov_info['active_regime'] == 'MEAN_REVERTING_CHOP':
            # LONG
            if close <= vp_info.get('val', 0) + 5.0 and close > ema200 and ofi_info['buyer_defense'] and close > bar_open:
                return Signal(
                    signal_type=SignalType.LONG,
                    entry_price=close,
                    sl_price=round(close - 0.8 * atr_14, 2),
                    target_1=round(vp_info.get('poc', close + 20), 2),
                    target_2=round(vp_info.get('vah', close + 40), 2),
                    reason="Mean-Reversion Long: Price at VAL in Choppy Regime. OFI confirms buyer defense. Quick scalp to POC.",
                    htf_aligned=True,
                    details={}
                )
            # SHORT
            if close >= vp_info.get('vah', 99999) - 5.0 and close < ema200 and ofi_info['seller_defense'] and close < bar_open:
                return Signal(
                    signal_type=SignalType.SHORT,
                    entry_price=close,
                    sl_price=round(close + 0.8 * atr_14, 2),
                    target_1=round(vp_info.get('poc', close - 20), 2),
                    target_2=round(vp_info.get('val', close - 40), 2),
                    reason="Mean-Reversion Short: Price at VAH in Choppy Regime. OFI confirms seller defense. Quick scalp to POC.",
                    htf_aligned=True,
                    details={}
                )

        # 4.3 IB Breakout Strategy (Active in LOW_VOL_TRENDING regime after 10:15 IST)
        if markov_info['active_regime'] == 'LOW_VOL_TRENDING' and bar_time >= '10:15':
            ib_state = compute_initial_balance_and_day_type(sub_df)
            avg_vol = float(sub_df['volume'].tail(10).mean()) if 'volume' in sub_df.columns else 0.0
            curr_vol = float(bar.get('volume', 0))
            
            # LONG
            if close > ib_state.get('ib_high', 99999) and htf_aligned_long and curr_vol > avg_vol:
                return Signal(
                    signal_type=SignalType.LONG,
                    entry_price=close,
                    sl_price=ib_state.get('ib_low', close - atr_14),
                    target_1=round(close + 1.5 * atr_14, 2),
                    target_2=round(close + 3.0 * atr_14, 2),
                    target_3_moonshot=round(close + 5.0 * atr_14, 2),
                    pyramid_trigger=round(close + 1.0 * atr_14, 2),
                    reason="IB Breakout Long: Price cleared Initial Balance High in Trending Regime. HTF aligned. Volume confirmed.",
                    htf_aligned=True,
                    details={'ib_state': ib_state}
                )
            # SHORT
            if close < ib_state.get('ib_low', 0) and htf_aligned_short and curr_vol > avg_vol:
                return Signal(
                    signal_type=SignalType.SHORT,
                    entry_price=close,
                    sl_price=ib_state.get('ib_high', close + atr_14),
                    target_1=round(close - 1.5 * atr_14, 2),
                    target_2=round(close - 3.0 * atr_14, 2),
                    target_3_moonshot=round(close - 5.0 * atr_14, 2),
                    pyramid_trigger=round(close - 1.0 * atr_14, 2),
                    reason="IB Breakdown Short: Price broke Initial Balance Low in Trending Regime. HTF aligned. Volume confirmed.",
                    htf_aligned=True,
                    details={'ib_state': ib_state}
                )
'''
        content = content.replace(
            '# 5. Auction Market Theory (AMT) Value Area Trigger Check',
            strategy_code + '\n        # 5. Auction Market Theory (AMT) Value Area Trigger Check'
        )

    with open('d:/antigravity_sandbox/Nifty/src/strategy_rules.py', 'w') as f:
        f.write(content)
    print("Done")

modify_strategy()
