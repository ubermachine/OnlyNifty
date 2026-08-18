import pytest
import pandas as pd
import numpy as np
from src.edge_harness import EdgeStats, EdgeTable, WalkForwardRunner


def test_edge_stats_quarantine_policy():
    runner = WalkForwardRunner()

    # 1. Under minimum samples -> PAPER
    r_small = [1.5, -1.0, 2.0]
    stats_small = runner.compute_edge_stats(r_small, "IB_BREAKOUT_LONG", "LOW_VOL_TRENDING")
    assert stats_small.status == "PAPER"

    # 2. 35 samples with negative mean EV -> QUARANTINED
    r_loser = [-1.0] * 25 + [1.0] * 10
    stats_loser = runner.compute_edge_stats(r_loser, "CHOP_BREAKOUT_LONG", "MEAN_REVERTING_CHOP")
    assert stats_loser.status == "QUARANTINED"
    assert stats_loser.ev < 0

    # 3. 40 samples with high positive mean EV and QUOTE tier -> TRUSTED
    r_winner = [1.8] * 28 + [-1.0] * 12
    stats_winner = runner.compute_edge_stats(r_winner, "ABSORPTION_LONG", "MEAN_REVERTING_CHOP", evidence_tier="QUOTE")
    assert stats_winner.status == "TRUSTED"
    assert stats_winner.ev > 0


def test_edge_table_json_serialization():
    stat1 = EdgeStats("IB_BREAKOUT_LONG", "TRENDING", 35, 65.0, 1.2, 1.2, 0.4, 2.0, "TRUSTED")
    stat2 = EdgeStats("BAD_SETUP", "CHOP", 40, 20.0, -0.6, -0.6, -0.9, -0.3, "QUARANTINED")
    table = EdgeTable([stat1, stat2])

    assert table.is_tradeable("IB_BREAKOUT_LONG", "TRENDING") is True
    assert table.is_tradeable("BAD_SETUP", "CHOP") is False
    assert table.get_sizing_factor("IB_BREAKOUT_LONG", "TRENDING") == 1.0
    assert table.get_sizing_factor("BAD_SETUP", "CHOP") == 0.0

    json_str = table.to_json()
    reloaded = EdgeTable.from_json(json_str)
    assert reloaded.is_tradeable("IB_BREAKOUT_LONG", "TRENDING") is True
    assert reloaded.is_tradeable("BAD_SETUP", "CHOP") is False
