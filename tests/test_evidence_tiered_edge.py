import pytest
import pandas as pd
import numpy as np
from src.edge_harness import EdgeStats, EdgeTable, WalkForwardRunner
from src.strategy_rules import Signal, SignalType


def test_evidence_tier_promotion_ceiling():
    runner = WalkForwardRunner()
    
    # 1. 40 samples with high EV on QUOTE tier -> TRUSTED
    r_winner = [1.5] * 30 + [-1.0] * 10
    stats_quote = runner.compute_edge_stats(r_winner, "ABSORPTION_LONG", "LOW_VOL_TRENDING", evidence_tier="QUOTE")
    assert stats_quote.status == "TRUSTED"
    assert stats_quote.evidence_tier == "QUOTE"
    
    # 2. Same 40 samples on MODEL tier -> CAPPED at PAPER
    stats_model = runner.compute_edge_stats(r_winner, "ABSORPTION_LONG", "LOW_VOL_TRENDING", evidence_tier="MODEL")
    assert stats_model.status == "PAPER"
    assert stats_model.evidence_tier == "MODEL"

    # 3. Same 40 samples on SPOT tier -> CAPPED at PAPER
    stats_spot = runner.compute_edge_stats(r_winner, "ABSORPTION_LONG", "LOW_VOL_TRENDING", evidence_tier="SPOT")
    assert stats_spot.status == "PAPER"
    assert stats_spot.evidence_tier == "SPOT"


def test_evidence_tier_quarantine_applies_to_all_tiers():
    runner = WalkForwardRunner()
    r_loser = [-1.0] * 30 + [0.5] * 10
    
    # QUARANTINED applies across all tiers on negative expectancy
    stats_q = runner.compute_edge_stats(r_loser, "BAD_SETUP", "CHOP", evidence_tier="QUOTE")
    assert stats_q.status == "QUARANTINED"
    
    stats_m = runner.compute_edge_stats(r_loser, "BAD_SETUP", "CHOP", evidence_tier="MODEL")
    assert stats_m.status == "QUARANTINED"


def test_edge_stats_serialization_with_evidence_tier():
    stat = EdgeStats("TEST_SETUP", "TREND", 40, 70.0, 0.8, 0.8, 0.2, 1.4, "TRUSTED", evidence_tier="QUOTE")
    d = stat.to_dict()
    assert d["evidence_tier"] == "QUOTE"
    
    # Reloading older dict without evidence_tier defaults to SPOT
    old_dict = {
        "setup_id": "LEGACY", "regime": "CHOP", "n": 35,
        "win_rate": 60.0, "mean_r": 0.5, "ev": 0.5,
        "ci_low": 0.1, "ci_high": 0.9, "status": "TRUSTED"
    }
    reloaded_old = EdgeStats.from_dict(old_dict)
    assert reloaded_old.evidence_tier == "SPOT"


def test_simulate_trade_outcome_with_quote_series():
    runner = WalkForwardRunner()
    sig = Signal(SignalType.LONG, entry_price=24500.0, sl_price=24465.0, target_1=24545.0, target_2=24590.0)
    
    ticket = {
        "entry_premium": 150.0,
        "sl_premium": 125.0,  # 25 pt stop
        "target1_premium": 185.0,  # +35 pt target (+1.4R)
        "target2_premium": 215.0
    }
    
    # Quote series hits T1 on bar 2
    quote_df = pd.DataFrame([
        {"open": 150.0, "high": 160.0, "low": 145.0, "close": 155.0},
        {"open": 156.0, "high": 190.0, "low": 154.0, "close": 188.0},
        {"open": 188.0, "high": 220.0, "low": 185.0, "close": 218.0}
    ])
    
    r_outcome = runner.simulate_trade_outcome(sig, pd.DataFrame(), quote_series=quote_df, ticket=ticket)
    assert r_outcome is not None
    assert r_outcome > 0.5  # Won trade
