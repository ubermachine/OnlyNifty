"""Regression tests for the options-flow -> desk-state data contract.

These lock in a class of bug where a producer and consumer disagreed on key names, so
every read silently returned its default and whole pillars of the desk quietly became
constants. Nothing raised; the numbers just stopped meaning anything.
"""

import numpy as np
import pandas as pd
import pytest

from src.options_engine import black_scholes_greeks
from src.options_flow import (
    compute_cumulative_oi_delta_and_traps,
    compute_short_term_directional_vector,
    compute_strike_level_gex_chart_data,
)
from src.options_positioning import compute_options_desk_state


def trending_df(periods=60, step=4.0):
    dates = pd.date_range("2026-08-13 10:00", periods=periods, freq="5min")
    px = [24500.0 + i * step for i in range(periods)]
    return pd.DataFrame(
        {
            "open": px,
            "high": [p + 15 for p in px],
            "low": [p - 15 for p in px],
            "close": px,
            "volume": [20000] * periods,
        },
        index=dates,
    )


def symmetric_chain(spot=24500.0, n=9, oi=100000):
    strikes = [spot + (i - n // 2) * 50 for i in range(n)]
    return pd.DataFrame(
        {
            "strike": strikes,
            "ce_oi": [oi] * n,
            "pe_oi": [oi] * n,
            "ce_change_oi": [0] * n,
            "pe_change_oi": [0] * n,
            "ce_volume": [5000] * n,
            "pe_volume": [5000] * n,
        }
    )


class TestDirectionalVectorContract:
    """The keys options_positioning and signal_journal actually read must exist."""

    @pytest.mark.parametrize(
        "key", ["short_term_bias", "conviction_percentage", "sub_scores", "is_expiry_day"]
    )
    def test_consumer_keys_are_present(self, key):
        res = compute_short_term_directional_vector(
            spot=24500.0, df=trending_df(), live_iv=0.13, hfi_score=0.3
        )
        assert key in res, f"consumers read '{key}' — missing means a silent default"

    @pytest.mark.parametrize(
        "sub_key", ["vanna_charm", "straddle_state", "pcr_momentum"]
    )
    def test_sub_score_keys_are_present(self, sub_key):
        res = compute_short_term_directional_vector(
            spot=24500.0, df=trending_df(), live_iv=0.13, hfi_score=0.3
        )
        assert sub_key in res["sub_scores"]

    def test_aliases_agree_with_primary_keys(self):
        res = compute_short_term_directional_vector(
            spot=24500.0, df=trending_df(), live_iv=0.13, hfi_score=0.3
        )
        assert res["short_term_bias"] == res["bias"]
        assert res["conviction_percentage"] == res["conviction_pct"]

    def test_conviction_is_not_pinned_to_zero(self):
        # Was permanently 0.0 because the consumer read a key that did not exist.
        strong = compute_short_term_directional_vector(
            spot=24500.0, df=trending_df(), live_iv=0.13, hfi_score=0.9
        )
        st = compute_options_desk_state(
            option_chain_df=None, spot=24500.0, df_ohlcv=trending_df(),
            dir_flow_res=strong, live_iv=0.13, hfi_score=0.9, persist_history=False,
        )
        assert st.trend_conviction_pct > 0.0

    def test_trend_bias_can_leave_neutral(self):
        strong = compute_short_term_directional_vector(
            spot=24500.0, df=trending_df(), live_iv=0.13, hfi_score=0.95
        )
        st = compute_options_desk_state(
            option_chain_df=None, spot=24500.0, df_ohlcv=trending_df(),
            dir_flow_res=strong, live_iv=0.13, hfi_score=0.95, persist_history=False,
        )
        assert st.trend_bias == "BULLISH"


class TestMissingDataFailsNeutral:
    def test_absent_chain_gives_no_directional_opinion(self):
        # Previously fabricated a +0.45 bullish "LONG_BUILDUP" pulse from nothing.
        res = compute_cumulative_oi_delta_and_traps(None, spot=24500.0)
        assert res["net_oi_pulse_score"] == 0.0
        assert res["active_quadrant"] == "NO_DATA"
        assert res["data_quality"] == "UNVERIFIED"

    def test_d_vector_is_not_structurally_bullish_without_a_chain(self):
        df = trending_df()
        neutral = compute_short_term_directional_vector(
            spot=24500.0, df=df, live_iv=0.13, hfi_score=0.0
        )
        # With no chain and no heavyweight tilt, D must sit near zero.
        assert abs(neutral["directional_vector"]) < 0.10


class TestGammaParity:
    def test_call_and_put_gamma_are_equal_at_same_strike(self):
        # Put-call parity. The skew premium is a PRICING adjustment and must never
        # make a put's gamma differ from a call's at the same strike.
        call = black_scholes_greeks(24500.0, 24500.0, t_days=1.0, sigma=0.13, is_call=True)
        put = black_scholes_greeks(24500.0, 24500.0, t_days=1.0, sigma=0.13, is_call=False)
        assert call["gamma"] == pytest.approx(put["gamma"], rel=1e-9), (
            "gamma must be identical for C and P at the same strike"
        )

    def test_symmetric_chain_does_not_manufacture_long_gamma(self):
        res = compute_strike_level_gex_chart_data(
            symmetric_chain(), spot=24500.0, iv=0.13, t_days=1.0
        )
        # A perfectly symmetric chain must net to ~zero GEX, not a confident +gamma call.
        assert abs(res["total_net_gex_cr"]) < 1e-6


class TestGammaFlipLevel:
    def test_flip_is_not_parked_on_an_outer_wing(self):
        # argmin(|per-strike GEX|) always returned an outer strike because gamma -> 0
        # in the wings, making gamma_flip_distance_pts meaningless.
        spot = 24500.0
        res = compute_strike_level_gex_chart_data(
            symmetric_chain(spot=spot, n=13), spot=spot, iv=0.13, t_days=1.0
        )
        strikes = res["strikes"]
        flip = res["zero_gex_strike"]
        assert flip not in (min(strikes), max(strikes)), "flip landed on a chain wing"
        assert abs(flip - spot) < 400.0


class TestMoveRatioScaling:
    def test_move_ratio_centres_near_one_on_a_normal_day(self):
        # move_ratio compares a HIGH-LOW RANGE to an expected RANGE. Dividing by the
        # 1-sigma point move made it structurally ~2.0 on an ordinary day, so any
        # "range exhausted" rule near 1.3 fired essentially always.
        spot, iv = 24500.0, 0.13
        sigma_daily = spot * iv * np.sqrt(1.0 / 365.0)
        expected_range = 1.596 * sigma_daily

        dates = pd.date_range("2026-08-13 09:15", periods=40, freq="5min")
        half = expected_range / 2.0
        df = pd.DataFrame(
            {
                "open": [spot] * 40,
                "high": [spot + half] * 40,
                "low": [spot - half] * 40,
                "close": [spot] * 40,
                "volume": [20000] * 40,
            },
            index=dates,
        )
        st = compute_options_desk_state(
            option_chain_df=None, spot=spot, df_ohlcv=df,
            live_iv=iv, persist_history=False,
        )
        assert st.move_ratio == pytest.approx(1.0, abs=0.15)
