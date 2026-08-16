"""Measurement script for Effective Number of Independent Signals (N_eff).

Quantifies:
1. Correlation matrix and N_eff across the 4 evidence families:
   - Structure (HTF regime + EMAs)
   - Flow (Delta-weighted volume + Smart Money flow score)
   - Positioning (Options desk D-vector + Gamma regime)
   - Macro (Futures basis + Global cues MSS + VRP)

2. Sub-pillar collinearity inside positioning:
   - Pairwise correlation of (d_vector, itm_otm_shift, pcr_momentum_score, writing_bias)
   - Variance explained by PC1

Formula for N_eff:
   N_eff = (Tr(C))^2 / Tr(C^2) = K^2 / sum(lambda_i^2)
   N_eff_entropy = exp(-sum(p_i * ln(p_i))) where p_i = lambda_i / K
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from datetime import timezone, timedelta

from src.data_engine import DataEngine
from src.strategy_rules import StrategyEngine
from src.desk_verdict import compute_evidence_families, build_desk_verdict
from src.options_positioning import compute_options_desk_state
from src.volatility_engine import VolatilityIntelligence


def compute_n_eff_from_cov(corr_matrix: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """Computes N_eff (eigenvalue concentration) and N_eff_entropy from correlation matrix."""
    eigvals = np.linalg.eigvalsh(corr_matrix)
    eigvals = np.maximum(eigvals, 1e-10)
    K = len(eigvals)
    
    # 1. Herfindahl-Hirschman / Participation Ratio formulation
    n_eff_pr = float((K ** 2) / np.sum(eigvals ** 2))
    
    # 2. Shannon Entropy formulation
    p = eigvals / float(np.sum(eigvals))
    entropy = -float(np.sum(p * np.log(p)))
    n_eff_entropy = float(np.exp(entropy))
    
    return round(n_eff_pr, 3), round(n_eff_entropy, 3), np.sort(eigvals)[::-1]


def run_n_eff_measurement(n_bars: int = 300) -> Dict[str, Any]:
    """Runs a multi-session simulation across varying regimes to measure empirical N_eff."""
    engine = DataEngine()
    strat = StrategyEngine()
    vol_intel = VolatilityIntelligence()
    IST = timezone(timedelta(hours=5, minutes=30))

    np.random.seed(42)
    # Generate 4 distinct market regimes (Trend Up, Trend Down, Chop, Vol Expansion)
    df_ohlcv = engine.generate_synthetic_nifty(bars=n_bars, interval_mins=5, start_price=24500.0)
    df_ohlcv.index = pd.date_range("2026-08-10 09:15", periods=n_bars, freq="5min", tz=IST)

    family_records = []
    positioning_sub_records = []

    for i in range(25, n_bars):
        sub_df = df_ohlcv.iloc[:i+1]
        spot = float(sub_df["close"].iloc[-1])
        
        # Synthetic options chain with varying flow
        chain_df = engine.generate_synthetic_option_chain(spot)
        
        # Microstructure indicators
        vol_report = vol_intel.generate_vol_intelligence_report(sub_df["close"], current_iv=0.135)
        hfi_score = float(np.clip(np.random.normal(0.0, 0.25), -1.0, 1.0))
        
        # Cross-market macro inputs
        basis_score = float(np.clip(np.random.normal(0.05, 0.3), -1.0, 1.0))
        macro_score = float(np.clip(np.random.normal(-0.05, 0.35), -1.0, 1.0))
        vrp_val = float(vol_report.get("vrp_data", {}).get("vrp", 0.015))
        
        opt_ctx = {
            "chain_df": chain_df,
            "flow_score": float(np.clip(50.0 + hfi_score * 30.0 + np.random.normal(0, 5), 0, 100)),
            "futures_basis": {"bias_score": basis_score, "data_quality": "VERIFIED", "basis_pts": basis_score * 40.0, "annualised_basis_pct": basis_score * 6.5},
            "macro_report": {"macro_sentiment_score": macro_score},
            "vrp": vrp_val
        }

        desk_state = compute_options_desk_state(
            option_chain_df=chain_df,
            spot=spot,
            df_ohlcv=sub_df,
            live_iv=0.135,
            hfi_score=hfi_score,
            persist_history=False
        )

        htf_data = {
            "htf_aligned_long": bool(sub_df["close"].iloc[-1] > sub_df["close"].iloc[-10]),
            "htf_aligned_short": bool(sub_df["close"].iloc[-1] < sub_df["close"].iloc[-10]),
            "confluence_regime": "ALIGNED_BULLISH" if sub_df["close"].iloc[-1] > sub_df["close"].iloc[-10] else "ALIGNED_BEARISH"
        }
        regime_data = {"active_regime": "LOW_VOL_TRENDING" if abs(hfi_score) > 0.3 else "MEAN_REVERTING_CHOP"}

        votes, why, directional = compute_evidence_families(
            desk_state=desk_state,
            htf_data=htf_data,
            regime_state=regime_data,
            vol_report=vol_report,
            options_context=opt_ctx
        )

        family_records.append(votes)
        
        # Sub-pillars within positioning
        wb_val = 1.0 if "PUT_WRITING" in str(desk_state.writing_bias) else (-1.0 if "CALL_WRITING" in str(desk_state.writing_bias) else 0.0)
        positioning_sub_records.append({
            "d_vector": desk_state.d_vector,
            "itm_otm_shift": desk_state.itm_otm_shift,
            "pcr_mom": desk_state.pcr_momentum_score,
            "writing_bias": wb_val
        })

    df_fam = pd.DataFrame(family_records)
    corr_fam = df_fam.corr().fillna(0.0).values
    n_eff_pr, n_eff_ent, eigvals_fam = compute_n_eff_from_cov(corr_fam)

    df_pos = pd.DataFrame(positioning_sub_records)
    corr_pos = df_pos.corr().fillna(0.0).values
    pos_n_eff_pr, pos_n_eff_ent, eigvals_pos = compute_n_eff_from_cov(corr_pos)
    
    # Variance explained by PC1 in positioning sub-pillars
    pc1_var_explained = float(eigvals_pos[0] / np.sum(eigvals_pos))

    return {
        "family_names": list(df_fam.columns),
        "family_correlation": np.round(corr_fam, 3).tolist(),
        "family_eigenvalues": np.round(eigvals_fam, 3).tolist(),
        "family_n_eff_pr": n_eff_pr,
        "family_n_eff_entropy": n_eff_ent,
        "pos_sub_pillars": list(df_pos.columns),
        "pos_sub_correlation": np.round(corr_pos, 3).tolist(),
        "pos_pc1_variance_explained": round(pc1_var_explained, 3),
        "pos_sub_n_eff": pos_n_eff_pr
    }


if __name__ == "__main__":
    results = run_n_eff_measurement(n_bars=300)
    print("================================================================")
    print("EMPIRICAL N_EFF & INDEPENDENCE MEASUREMENT REPORT")
    print("================================================================")
    print(f"Evidence Families: {results['family_names']}")
    print(f"Family Correlation Matrix:\n{pd.DataFrame(results['family_correlation'], index=results['family_names'], columns=results['family_names'])}")
    print(f"\nFamily Eigenvalues: {results['family_eigenvalues']}")
    print(f"-> N_eff (Participation Ratio): {results['family_n_eff_pr']:.2f} / 4.00")
    print(f"-> N_eff (Shannon Entropy):     {results['family_n_eff_entropy']:.2f} / 4.00")
    print("\n----------------------------------------------------------------")
    print("POSITIONING SUB-PILLARS REDUNDANCY ANALYSIS:")
    print(f"Sub-pillars: {results['pos_sub_pillars']}")
    print(f"Sub-pillar Correlation Matrix:\n{pd.DataFrame(results['pos_sub_correlation'], index=results['pos_sub_pillars'], columns=results['pos_sub_pillars'])}")
    print(f"-> PC1 Variance Explained: {results['pos_pc1_variance_explained'] * 100:.1f}%")
    print(f"-> Sub-pillar N_eff:        {results['pos_sub_n_eff']:.2f} / 4.00")
    print("================================================================")
