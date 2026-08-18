import pytest
import numpy as np
import pandas as pd
from src.indicators import compute_kaufman_efficiency_ratio
from src.desk_verdict import compute_conviction, build_desk_verdict, DeskVerdict
from src.signal_journal import LiveSignalJournal
from src.strategy_rules import Signal, SignalType
from src.options_positioning import OptionsDeskState

def test_kaufman_efficiency_ratio_discrimination():
    # 1. Pure Trend series (Aug 18 style)
    trend_prices = [24000.0 + i * 10.0 for i in range(30)] # Net move = 290, total path = 290 -> ER = 1.0
    dates = pd.date_range("2026-08-18 09:15:00", periods=30, freq="5min", tz="Asia/Kolkata")
    df_trend = pd.DataFrame({"close": trend_prices}, index=dates)
    
    res_trend = compute_kaufman_efficiency_ratio(df_trend)
    assert res_trend["efficiency_ratio"] >= 0.15
    assert res_trend["regime"] == "STRONG_TREND"
    assert res_trend["is_trending"] is True

    # 2. Pure Chop series (Aug 17 style)
    chop_prices = [24000.0 + (20.0 if i % 2 == 0 else -20.0) for i in range(30)]
    df_chop = pd.DataFrame({"close": chop_prices}, index=dates)
    
    res_chop = compute_kaufman_efficiency_ratio(df_chop)
    assert res_chop["efficiency_ratio"] < 0.08
    assert res_chop["regime"] == "CHOP_OR_REVERSAL"
    assert res_chop["is_trending"] is False


def test_conviction_efficiency_ratio_decoupled_from_live_scoring():
    votes = {"structure": 1, "flow": 1, "positioning": 1, "macro": 0}
    
    # Intraday ER is noted for diagnostic review but decoupled from live score modification
    score_trend, _, _, notes_trend = compute_conviction(
        action="BUY_CE",
        votes=votes,
        confluence_score=70.0,
        efficiency_ratio=0.20,
        setup_id="LONG_ORDER_FLOW"
    )
    assert any("session ER: 0.200" in n for n in notes_trend)

    score_no_er, _, _, _ = compute_conviction(
        action="BUY_CE",
        votes=votes,
        confluence_score=70.0,
        efficiency_ratio=None,
        setup_id="LONG_ORDER_FLOW"
    )
    # Score should be identical, confirming decoupling from live execution gating
    assert score_trend == score_no_er


def test_ranked_opportunity_board_synthesis_and_contract_levels():
    # Test Bug 1 fix: raw dict with short keys 'entry', 'sl', 't1', 't2', 't3'
    # Test Status Precedence: HTF veto prose with edge_status=QUARANTINED -> EDGE_QUARANTINED
    # Test Dedup: 2 identical candidates with 1pt SL difference collapsed to 1 row
    sig = Signal(
        signal_type=SignalType.LONG,
        entry_price=24500.0,
        sl_price=24460.0,
        target_1=24550.0,
        target_2=24600.0,
        target_3_moonshot=24680.0,
        reason="Test Long Fired",
        details={
            "rejected_candidates": [
                {
                    "signal_type": "SHORT_ORDER_FLOW",
                    "direction": "SHORT",
                    "entry": 24214.05,
                    "sl": 24204.0,
                    "t1": 24244.05,
                    "t2": 24276.55,
                    "t3": 24320.0,
                    "confluence": 65.0,
                    "veto_gate": "HTF_NOT_ALIGNED_LONG",
                    "reason": "HTF Confluence Veto: Long not supported by 1H EMA200",
                    "edge_status": "QUARANTINED"
                },
                {
                    "signal_type": "SHORT_ORDER_FLOW",
                    "direction": "SHORT",
                    "entry": 24214.05,
                    "sl": 24203.0,  # 1-pt difference should be collapsed by 10pt dedup bucket
                    "t1": 24244.05,
                    "t2": 24276.55,
                    "t3": 24320.0,
                    "confluence": 65.0,
                    "veto_gate": "HTF_NOT_ALIGNED_LONG",
                    "reason": "HTF Confluence Veto: Long not supported by 1H EMA200",
                    "edge_status": "QUARANTINED"
                },
                {
                    "signal_type": "RANGE_FADE_SHORT",
                    "direction": "SHORT",
                    "entry": 24500.0,
                    "sl": 24530.0,
                    "t1": 24470.0,
                    "t2": 24440.0,
                    "confluence": 50.0,
                    "veto_gate": "CONFLUENCE_FLOOR",
                    "reason": "Confluence 50 < 70",
                    "edge_status": "UNMEASURED"
                }
            ]
        }
    )

    ticket = {
        "status": "READY",
        "strike": 24500,
        "option_type": "CE",
        "entry_premium": 150.0,
        "sl_premium": 120.0,
        "target_1_premium": 190.0,
        "target_2_premium": 220.0,
        "target_3_premium": 260.0,
        "contracts": 25,
        "tca_cost_total": 45.0
    }

    verdict = build_desk_verdict(
        signal=sig,
        ticket=ticket,
        current_spot=24500.0,
        options_context={"efficiency_ratio": 0.18, "efficiency_regime": "STRONG_TREND"}
    )

    assert verdict.efficiency_ratio == 0.18
    assert verdict.efficiency_regime == "STRONG_TREND"
    
    # Verify deduplication collapsed 2 SHORT_ORDER_FLOW candidates into 1
    short_of_list = [opp for opp in verdict.ranked_opportunities if opp["setup_id"] == "SHORT_ORDER_FLOW"]
    assert len(short_of_list) == 1
    
    short_of = short_of_list[0]
    assert short_of["entry_price"] == 24214.05
    assert short_of["sl_price"] == 24204.0
    assert short_of["target_1"] == 24244.05
    assert short_of["target_2"] == 24276.55
    assert short_of["r_multiple_t1"] > 0.0
    # Strict precedence: edge_status=QUARANTINED takes priority over prose containing "Confluence"
    assert short_of["status"] == "EDGE_QUARANTINED"
    assert short_of["edge_status"] == "QUARANTINED"

    rf = next(opp for opp in verdict.ranked_opportunities if opp["setup_id"] == "RANGE_FADE_SHORT")
    assert rf["entry_price"] == 24500.0
    assert rf["sl_price"] == 24530.0
    assert rf["target_1"] == 24470.0
    assert rf["status"] == "CONFLUENCE_FLOOR"


def test_setup_level_cluster_attribution():
    journal = LiveSignalJournal(persistence_file=None)
    
    sig = Signal(
        signal_type=SignalType.RANGE_FADE_SHORT,
        entry_price=24500.0,
        sl_price=24530.0,
        target_1=24460.0,
        target_2=24420.0,
        target_3_moonshot=24380.0,
        reason="Test Fade"
    )
    ticket = {
        "status": "READY",
        "strike": 24500,
        "option_type": "PE",
        "entry_premium": 120.0,
        "sl_premium": 100.0,
        "target1_premium": 150.0,
        "target2_premium": 170.0,
        "target3_moonshot_premium": 200.0,
        "lots": 1,
        "tca_cost_total": 45.0
    }
    
    # Log 3 RANGE_FADE_SHORT trades
    for i in range(3):
        entry = journal.log_signal(
            signal=sig,
            ticket=ticket,
            confluence_score=72.0,
            current_spot=24500.0,
            bar_timestamp=f"2026-08-18 10:0{i}:00",
            setup_id="RANGE_FADE_SHORT",
            is_seed=True,
            structure_epoch=f"EPOCH_RANGE_FADE_{i}"
        )
        if entry:
            entry.lifecycle_status = "STOPPED_OUT"
            entry.realized_r_multiple = -1.0
            entry.peak_favorable_excursion_pts = 8.0 + i * 2.0

    cl = journal.cluster_context(direction="SHORT", setup_id="RANGE_FADE_SHORT")
    assert cl["setup_index"] == 4
    assert cl["setup_count"] == 3
    assert cl["setup_mfe_median"] == 10.0
    assert cl["setup_went_negative"] == 3
