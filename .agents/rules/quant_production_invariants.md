# Quantitative Production Invariants & Decision Architecture

As an autonomous agent operating in this repository, you must permanently uphold the following quantitative tenets and defensive engineering rules:

## 1. Pure Session Kaufman Efficiency Ratio (KER) Invariant
- **Formula**:
  $$ER_{\text{session}} = \frac{|P_{\text{current}} - P_{\text{open\_session}}|}{\sum_{i=1}^{N_{\text{session}}} |P_i - P_{i-1}|}$$
- **Anti-Pattern (Blend Corruption)**: Never blend short rolling windows (e.g. 14 bars) with session ER when $\ge 6$ bars exist. A short rolling window has no multi-hour reversals and hovers near $\approx 0.42$ on both trend and chop days, washing out the true 5× discrimination ($ER=0.037$ on chop vs $0.189$ on trend).
- **Execution Invariant (Decoupled from Live Conviction)**:
  - **Do NOT feed ER into live intraday conviction scoring or setup bonuses/penalties.**
  - **Reason**: Session ER is anchored to the session open and is naturally linear ($0.42$–$0.48$, reading `STRONG_TREND`) throughout the entire morning of a V-reversal chop day before the afternoon reversal occurs. Injecting it into live scoring awards premature momentum bonuses to losing morning trades.
- **Display & Post-Session Analytics Usage**:
  - ER is reserved strictly for display in the **Environment Quality Banner** (with a convergence progress indicator `N/75 bars`) and for **post-session journal attribution & walk-forward analysis**, where completed-session hindsight is legitimate.
  - Regime Labels:
    - $ER \ge 0.15 \implies \text{STRONG\_TREND}$
    - $0.08 \le ER < 0.15 \implies \text{MODERATE\_TREND}$
    - $ER < 0.08 \implies \text{CHOP\_OR\_REVERSAL}$

## 2. Dictionary Contract & Key-Name Defenses
- Setup candidate dictionaries and signal structures may carry abbreviated keys (`entry`, `sl`, `t1`, `t2`, `t3`) or verbose keys (`entry_price`, `sl_price`, `target_1`, etc.).
- Extraction layers (`src/desk_verdict.py`, `app.py`) must defensively extract both (`rej.get("entry_price") or rej.get("entry") or spot`) so price levels never collapse to `0.00`.

## 3. Operator Decision-Support: The Ranked Opportunity Board
- Do not architect the system as a single-verdict bottleneck that silences or deletes candidates.
- Always surface ALL candidate setups evaluated on the bar (Active Fired, Gate Vetoed, Edge Quarantined, Confluence Floor) into a single ranked opportunity board ordered descending by conviction score with non-zero levels, $R_{\text{T1}}$, edge status, and cluster sequence context.
- Prioritize `EDGE_QUARANTINED` before `CONFLUENCE_FLOOR` and match strictly on `veto_gate` enums (never regex/match prose reason strings).
- Deduplicate candidate rows using 10-point buckets (`round(entry, -1)`, `round(sl, -1)`) to collapse trivial $\le 1\text{ pt}$ SL noise.

## 4. Method Signature Keyword Invariant
- Always invoke multi-parameter engine methods (`cluster_context`, `log_signal`, `build_desk_verdict`) with explicit keyword arguments (`journal_engine.cluster_context(direction=..., now_ist=...)`) to avoid silent argument sliding when signatures change.
