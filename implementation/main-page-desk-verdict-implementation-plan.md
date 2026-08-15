# Main-Page Desk Verdict — Detailed Implementation Plan

> **Status:** Approved blueprint (pending implementation)
> **Date:** 2026-08-15
> **Scope:** Fuse every analytics the app computes into ONE main-page decision: **BUY / SELL / WAIT + trend + range + which option to buy**. Extends `implementation/signal-quality-implementation-plan.md` (this is its options-desk + main-page layer).
> **Audience:** The implementation agent(s) executing this plan.
> **Deliverable of this plan:** code changes described below (not a document-only request).

---

## 1. Goal, Success Metrics & Design Principles

### 1.1 Goal
The main page (Streamlit cockpit, `app.py:513-687`) must display **one fused, actionable verdict** computed from everything the app already produces:

- **WHAT TO DO** — BUY CE / BUY PE / WAIT, with the reason.
- **TREND** — bullish / bearish / neutral with conviction %.
- **RANGE** — expected corridor (support → resistance), max-pain magnet, straddle expected move, spot's position inside it.
- **WHICH OPTION** — concrete ticket: strike, CE/PE, entry premium, SL, T1/T2/T3, lots.

### 1.2 Success metrics

| # | Metric | Target |
|---|--------|--------|
| M1 | Main page always renders a verdict | 100% of refreshes — including WAIT with reasons (never blank, never hardcoded demo prices) |
| M2 | Trades against positioning veto | 0 (D-vector opposes chart setup ⇒ WAIT) |
| M3 | Hardcoded ticket fallbacks (`app.py:644-647`, ₹228/₹285 strings) | 0 remaining — replaced by a real `No setup` state |
| M4 | Inert gates | 0 — skew gate and GEX wall gate must fire on real chain data |
| M5 | Journal audit | Every trade row carries the options snapshot (walls, PCR, max pain, D, gamma regime) |
| M6 | Tests | New `test_options_positioning.py`, `test_desk_verdict.py` green + full suite green |

### 1.3 Design principles
1. **One pure verdict function** — `build_desk_verdict()` is the single source of truth; the UI only renders it. No Streamlit imports inside it (testable, reusable by Telegram later).
2. **Default answer is WAIT.** Conflicting evidence (e.g., chart LONG vs bearish options vector) resolves to WAIT with the conflict named.
3. **Fail-closed positioning.** Missing option chain ⇒ positioning-based setups vetoed; chart setups capped at 0.5× with `POSITIONING_UNVERIFIED` audit tag. Never a silent guess.
4. **Evidence families.** The verdict summarizes 4 independent families: structure (chart/TA), flow (CVD/OFI), positioning (options), macro (HFI/skew/vol).

---

## 2. Current-State Audit (verified problem map)

Anchors verified this session; re-grep before editing (parallel sessions are active).

### 2.1 The main page shows two separable verdicts
- `app.py:513-514` — "Main Cockpit Split Grid" (`cockpit_col1, cockpit_col2`).
- `cockpit_col1` (`:516-548`): signal badge (`LONG/SHORT/WAIT CONFIRMED`), spot, setup status, and a **separate** options flow badge rendering `dir_flow_res` (`:535-548`: D vector, conviction, playbook, Tgt/SL). Then the 9-cell confluence grid (`:549+`).
- `cockpit_col2` (`:649-687`): "Optimal Strike (Delta 0.50-0.65)" ticket box — Entry Prem / SL / T1 (35%) / T2 (35%) / Profit Maximizer — with **hardcoded fallback strings** (`:644-647`) when the ticket is absent.
- **Problem:** the chart signal, the options direction badge, and the strike ticket are computed independently and can contradict each other on screen. The options-desk data (walls, max pain, PCR, gamma, expected move) never reaches the main-page conclusion.

### 2.2 Options analytics are compute-and-display-only
- `compute_short_term_directional_vector` — 5-pillar trend vector (options_flow.py:304-410): `D = 0.30×ΔOI pulse + 0.25×vanna/charm + 0.20×PCR momentum + 0.15×straddle + 0.10×HFI`, with bias/action/target/stop. Computed at `app.py:397`; rendered only in the badge (`:535-548`) and tab metrics (`:1510`). **Never consumed by any decision logic.**
- `compute_oi_based_range_forecast` (options_flow.py:486-557) — put wall / call wall / max pain / corridor / location bias. Called at `app.py:1601` (tab_oi) and internally at options_flow.py:664. **Display-only.**
- tab_oi consumers: `pcr_analytics` `app.py:1586`, `oi_hm_res` `:1599`, `gex_chart_res` `:1600`, `range_fc_res` `:1601` — all rendered as metrics/charts, none passed to `evaluate_bar`.

### 2.3 Two gates are silently inert in production
- **Skew crash gate inert:** `app.py:350-355` calls `evaluate_bar(df, live_iv, hfi_score)` with **no option chain**. `compute_25delta_skew` (`strategy_rules.py:446`) therefore uses a synthetic fallback that always returns `is_crash_hedging=False` (`volatility_engine.py:626-639`). The LONG-blocking veto can never fire live.
- **GEX wall gate inert:** `_apply_universal_gates` reads `call_wall_strike` / `put_wall_strike` / `is_positive_gamma` (`strategy_rules.py:200-210`) but its input `gex_info` comes from `compute_dealer_gex(close)` (`strategy_rules.py:423`) — a price-only heuristic returning `net_gex_crores/is_positive_gamma/gamma_flip_strike` with **no wall keys** ⇒ defaults 999999/0 ⇒ pin veto never fires.
- The chain-based function that DOES have the keys is `compute_strike_level_gex_chart_data` (options_flow.py:679-764: `call_wall_strike`, `put_wall_strike`, `zero_gex_strike`, `net_dealer_regime`). `compute_full_chain_gex_profile` (options_engine.py:1039) returns them too but is **dead code** (zero call sites).

### 2.4 Missing trader-grade signals (to build)
- PCR **level** z-score vs session history (only the 15-min momentum derivative exists, options_flow.py:227-253).
- ITM/OTM OI shift (institutional accumulation vs writing).
- Max-pain **drift** across the session (only single-snapshot exists).
- Expected-vs-actual move ratio in decisions (function exists at volatility_engine.py:184-230, display-only).
- VCR computed (`strategy_rules.py:447`) but **never read** for a decision (sizing opportunity).

### 2.5 Baseline (why this matters)
Journal evidence: 18.2% realized win rate, 39/54 traded entries self-labeled "C Weak / Vetoed" — decisions are being made without the positioning evidence that already exists in the codebase.

---

## 3. Architecture — `src/desk_verdict.py`

New module, pure functions + dataclasses only. No pandas required in the verdict itself (accepts precomputed dicts).

### 3.1 `DeskVerdict` dataclass
```python
@dataclass
class DeskVerdict:
    action: str                          # "BUY_CE" | "BUY_PE" | "WAIT"
    action_label: str                    # human headline, e.g. "BUY 24300 CE @ ₹148 | Target Call Wall"
    reason: str                          # the why, incl. veto/conflict names
    trend_bias: str                      # "BULLISH" | "BEARISH" | "NEUTRAL"
    trend_conviction_pct: float          # 0-100
    range_corridor: tuple[float, float]  # (support, resistance)
    max_pain: float
    expected_move_pts: float
    spot_position_pct: float             # 0-100 within corridor
    option_pick: Optional[Dict[str, Any]]  # strike, option_type, entry_premium,
                                           # sl_spot, t1/t2/t3, lots, r_t1, r_t2
    evidence: Dict[str, str]             # family -> one-line verdict
    gate_audit: List[Dict[str, Any]]     # [{gate, value, passed, note}]
    conflicts: List[str]                 # named disagreements resolved to WAIT
    confluence_score: float
    confluence_grade: str
    data_quality: str                    # "VERIFIED" | "POSITIONING_UNVERIFIED"
```

### 3.2 `build_desk_verdict()` signature
```python
def build_desk_verdict(
    signal: Signal,                              # post-gate decisioned signal
    ticket: Optional[Dict[str, Any]],            # generate_option_trade_ticket output or None
    desk_state: Optional[OptionsDeskState],      # from src/options_positioning.py
    vol_report: Dict[str, Any],                  # generate_vol_intelligence_report output
    regime_state: Dict[str, Any],                # markov active_regime etc.
    htf_data: Dict[str, Any],                    # tf_1h/tf_15m biases + aligned flags
    edge_stats: Optional[EdgeStats],             # from src/edge_harness (may be None)
    session_state: Optional[SessionRiskState],   # from src/risk_state (may be None)
) -> DeskVerdict
```

### 3.3 Fusion logic (the decision table)

| Condition | Verdict |
|---|---|
| Signal is trade AND desk_state verified AND direction agrees with D (sign match or \|D\| < veto strength) | BUY_CE / BUY_PE with `option_pick` from ticket (strike clamped into corridor) |
| `desk_state.d_vector <= -POSITIONING_VETO_STRENGTH` AND signal LONG (or reverse) | WAIT — conflict `POSITIONING_OPPOSES_CHART` |
| Signal WAIT but desk trend strong (\|D\| ≥ 0.5) AND wall-fade condition met (spot ≤ put_wall+WALL_BUFFER_PTS, positive gamma) | BUY_CE (RANGE_FADE) — desk-only trade, strike = ATM |
| Signal WAIT but negative gamma AND \|D\| ≥ 0.5 AND spot beyond corridor wall in D's direction | BUY_CE/PE (GAMMA_BREAKOUT) |
| PCR z ≥ +PCR_Z_CONTRARIAN_THRESHOLD (extreme puts) AND spot near put wall AND positive gamma | BUY_CE (contrarian fade panic) |
| PCR z ≤ −PCR_Z_CONTRARIAN_THRESHOLD (extreme calls) AND spot near call wall | BUY_PE (contrarian fade euphoria) |
| `desk_state is None` (no chain) | Chart trades capped 0.5× with `POSITIONING_UNVERIFIED`; desk-only trades WAIT |
| Session state locked / edge quarantined | WAIT — reason from `session_state.can_take_new_trade()` / edge status |

Rules: conflicts list every disagreeing pair; if `conflicts` non-empty and no WAIT already, downgrade to WAIT. TREND = majority of (D sign, Kalman velocity sign, Markov regime, HTF bias, PCR momentum) with conviction = agreement fraction. RANGE = desk corridor if verified, else VAKC ±2σ bands.

---

## 4. Step-by-Step Code Changes

All edits additive; new code in new files. Re-grep anchors before editing (parallel sessions active).

### 4.1 Wire options data into the decision path
- **`src/strategy_rules.py`** — add optional kwarg to `evaluate_bar` (`:119-133`):
  `options_context: Optional[Dict[str, Any]] = None` — thread through to `_evaluate_bar_core` and store as `self._last_options_context` + attach to every returned Signal's `details["options_context"]` (trade AND WAIT signals).
- **`app.py`** (`:350-355`) — assemble the dict from already-computed values before calling `evaluate_bar`:
  ```python
  options_context = {
      "chain_df": oc_filtered,                    # filtered chain (as in tab_oi :1578-1580)
      "pcr": pcr_analytics,                       # :1586
      "range_fc": range_fc_res,                   # :1601
      "dir_flow": dir_flow_res,                   # :397
      "gex_chart": gex_chart_res,                 # :1600
      "expected_move": vol_report["expected_move_pts"],
      "iv_percentile": vol_report["iv_percentile"],
      "fii": inst_report["current_snapshot"],     # :1471
  }
  signal = strategy_engine.evaluate_bar(df, live_iv=iv_input, hfi_score=..., options_context=options_context)
  ```
  (Build `options_context` defensively with try/except → `None` on any missing upstream computation.)
- Journal: `log_signal` already receives `signal` — the snapshot rides in `signal.details`; also pass `options_context` explicitly for the journal row if `log_signal` has no access to details (check `signal_journal.py:273` signature; if needed add optional `options_context=None` param, defaulted — backward compatible).

### 4.2 Fix the skew gate (real chain, fail-closed fallback)
- **`src/strategy_rules.py:446`** — change skew computation to prefer the chain from `options_context`:
  ```python
  chain = options_context.get("chain_df") if options_context else option_chain_df
  skew_info = compute_25delta_skew(chain, spot=close, iv_baseline=live_iv)
  skew_info["data_quality"] = "VERIFIED" if (chain is not None and not chain.empty) else "SYNTHETIC"
  ```
- **`_apply_universal_gates`** (`:153-230`): when `data_quality == "SYNTHETIC"`, do NOT silently pass — set audit tag `POSITIONING_UNVERIFIED` and record `skew_verified=False` in the gate audit; the sizing cap (4.6) applies downstream. When verified and `Z > SKEW_ZSCORE_THRESHOLD`, the existing LONG veto (`:184-187`) fires for real.

### 4.3 Fix the GEX wall gate (chain walls instead of price heuristic)
- **`src/strategy_rules.py:423`** — replace the wall-source:
  ```python
  if options_context and options_context.get("gex_chart"):
      g = options_context["gex_chart"]
      gex_info = {
          "call_wall_strike": g.get("call_wall_strike", 999999.0),
          "put_wall_strike": g.get("put_wall_strike", 0.0),
          "zero_gex_strike": g.get("zero_gex_strike"),
          "is_positive_gamma": g.get("net_dealer_regime", "").startswith("POSITIVE"),
      }
  else:
      gex_info = compute_dealer_gex(close)   # price-only fallback
      gex_info["walls_verified"] = False     # existing pin gate (:200-210) skips + tags POSITIONING_UNVERIFIED
  ```
- The existing gate block (`:200-210`) needs no logic change once the keys exist — verify key names against options_flow.py:752-764 and normalize.

### 4.4 NEW `src/options_positioning.py`
```python
@dataclass
class OptionsDeskState:
    trend_bias: str                    # BULLISH/BEARISH/NEUTRAL
    trend_conviction_pct: float
    d_vector: float                    # from compute_short_term_directional_vector
    pcr_level: float                   # pcr_oi
    pcr_zscore: float                  # vs session history (0.0 until warm)
    pcr_momentum_score: float
    put_wall: float; call_wall: float; max_pain: float
    max_pain_drift_pts: float          # vs previous snapshot
    expected_move_pts: float
    actual_range_pts: float
    move_ratio: float                  # expected vs actual (volatility_engine.py:184-230)
    gamma_regime: str                  # POSITIVE/NEGATIVE/FLIP
    is_positive_gamma: bool
    zero_gex_strike: float
    writing_bias: str                  # CALL_WRITING_DOMINANT / PUT_WRITING_DOMINANT / BALANCED
    itm_otm_shift: float               # + = bullish build (see formula)
    agreement_count: int               # 0..4
    data_quality: str                  # VERIFIED / POSITIONING_UNVERIFIED

def compute_options_desk_state(
    option_chain_df, spot,
    prev_chain_df=None, pcr_analytics=None, dir_flow_res=None,
    range_fc_res=None, gex_chart_res=None,
    live_iv: float = DEFAULT_IV, hfi_score: float = 0.0,
    history: Optional[Dict[str, List[float]]] = None,   # persisted PCR/max-pain series
) -> OptionsDeskState

def load_options_history(path: str = OPTIONS_STATE_PATH) -> Dict[str, List[float]]
def save_options_history(history: Dict[str, List[float]], path: str = OPTIONS_STATE_PATH) -> None
```
**Computation spec:**
- Reuse: `dir_flow_res` (D + components), `range_fc_res` (walls/max-pain/corridor), `pcr_analytics` (pcr_oi/pcr_change_oi/max_pain_strike), `gex_chart_res` (gamma regime/walls/zero-gamma), `compute_expected_vs_actual_move` (straddle vs day range), `compute_oi_change_heatmap` writing_bias.
- **PCR z-score (new):** append current `pcr_oi` to `history["pcr_series"]` (cap 120 samples ≈ 1 session of 15-min snaps); `z = (pcr − mean)/std`; return 0.0 when `len < PCR_HISTORY_MIN_SAMPLES`.
- **ITM/OTM OI shift (new):** with normalized chain (cols per `_extract_normalized_chain`, options_flow.py:417-483): `atm = round(spot/50)*50`; `shift = (ΣΔOI CE < atm + ΣΔOI PE > atm) − (ΣΔOI CE > atm + ΣΔOI PE < atm)` normalized by total |ΔOI| → [−1, +1], + = bullish.
- **Max-pain drift (new):** `max_pain − history["max_pain_series"][-1]` (if history exists), push current.
- **Agreement count:** count of these agreeing in sign with trend_bias: D sign, PCR momentum sign, ITM/OTM shift sign, writing bias (put-writing = bullish), max-pain location (spot < max_pain = bullish magnet).
- Persist `{"pcr_series": [...], "max_pain_series": [...]}` to `data/options_state.json` every N refreshes (e.g., ≥5 min apart, timestamp-guarded).

### 4.5 Strategy rules — trend+range → buy/sell setups
- **New SignalType members** (additive, `strategy_rules.py:29-38`): `RANGE_FADE_LONG`, `RANGE_FADE_SHORT`, `GAMMA_BREAKOUT_LONG`, `GAMMA_BREAKOUT_SHORT`.
- **Insertion point:** inside `_evaluate_bar_core`, after the existing branch ladder and before the final WAIT return — using the `_check_and_return` closure pattern (`:492-506`) so every new branch inherits universal gates.
- **RANGE_FADE_LONG** (needs `desk_state`): `spot ≤ put_wall + WALL_BUFFER_PTS` AND `is_positive_gamma` AND (`d_vector ≥ 0.2` OR `pcr_zscore ≥ PCR_Z_CONTRARIAN_THRESHOLD`). Entry = close; SL = `put_wall − WALL_BUFFER_PTS`; T1 = `max_pain`; T2 = `call_wall`; strike = ATM CE (delta 0.5-0.6 via `select_institutional_strike`).
- **RANGE_FADE_SHORT:** mirror at `call_wall − WALL_BUFFER_PTS` (SL above wall, T1 max_pain, T2 put_wall, ATM PE).
- **GAMMA_BREAKOUT_LONG/SHORT:** `not is_positive_gamma` AND `|d_vector| ≥ 0.5` AND `close` beyond the corridor wall in D's direction AND OFI/CVD confirms (existing `ofi_info`). T1/T2 = 1.2/2.5×ATR; strike = ATM.
- **Positioning veto (in `_apply_universal_gates`):** if desk_state present and verified: `d_vector ≤ −POSITIONING_VETO_STRENGTH` → block LONG (`POSITIONING_OPPOSES_CHART`); `≥ +POSITIONING_VETO_STRENGTH` → block SHORT. Wall proximity + positive gamma → block fading INTO the wall (RANGE_FADE logic inverted guard).
- **Corridor clamping helper** (in `src/options_positioning.py`):
  ```python
  def clamp_targets_to_corridor(entry, t1, t2, direction, put_wall, call_wall) -> tuple[float, float, float]:
      # LONG: T1/T2 capped at call_wall (never beyond); SHORT: capped at put_wall
      # returns (t1, t2, sl_hint) — SL beyond the opposite wall for fade setups
  ```
- **VCR sizing:** in `_evaluate_bar_core` after `vcr_info` (`:447`): `if vcr_info.get("vcr_ratio", 1.0) < VCR_SQUEEZE_THRESHOLD: signal.details["size_factor"] = 0.5` — `app.py` multiplies lots by `signal.details.get("size_factor", 1.0)` before `generate_option_trade_ticket` (`:353`).

### 4.6 Confluence + config
- **`calculate_confluence_score`** (signal_journal.py:138-245 region): add positioning votes ONLY when desk_state verified: D sign ±10, PCR momentum ±5, PCR level contrarian ±5, max-pain location ±5. Unverified ⇒ skip votes (no inflation), tag `POSITIONING_UNVERIFIED`.
- **`src/config.py` additions:**
  ```python
  POSITIONING_VETO_STRENGTH: float = 0.5
  WALL_BUFFER_PTS: float = 25.0
  POSITIONING_UNVERIFIED_SIZE_CAP: float = 0.5
  PCR_Z_CONTRARIAN_THRESHOLD: float = 2.0
  PCR_HISTORY_MIN_SAMPLES: int = 20
  OPTIONS_STATE_PATH: str = "data/options_state.json"
  ```
- **Sizing caps:** missing chain ⇒ chart setups `min(size_factor, POSITIONING_UNVERIFIED_SIZE_CAP)`; desk-only setups (RANGE_FADE/GAMMA) require VERIFIED.

---

## 5. Main-Page UI Spec (`app.py:513-687` replacement)

Replace the current cockpit split content with a **Desk Verdict panel** (keep the same HTML card styling language already used):

```
┌─ DESK VERDICT ────────────────────────────────────────────────┐
│ [BUY 24300 CE]  (badge: green/red/amber for BUY_CE/BUY_PE/WAIT)│
│ Reason line (veto names, conflict names, or setup reason)      │
├────────────────────────────┬──────────────────────────────────┤
│ TREND  ▲ BULLISH 78%       │ RANGE  ₹24,270 ──●── ₹24,420     │
│ (D +0.62 · Kalman +1.2 ·   │ Max Pain ₹24,350 (magnet)        │
│  Markov TRENDING · PCR ↗)  │ Exp. Move ±₹95                    │
├────────────────────────────┴──────────────────────────────────┤
│ WHICH OPTION: NIFTY 24300 CE | Entry ₹148 | SL ₹131           │
│ T1 ₹162 | T2 ₹186 | T3 ₹206 | Lots 15 (375) | TCA ₹499        │
├───────────────────────────────────────────────────────────────┤
│ EVIDENCE: [Structure ✓ Trend] [Flow ✓ Buyer defense]           │
│ [Positioning ✓ Put wall defense] [Macro ✓ HFI +0.12]           │
│ GATES: Skew Z +0.4 ✓ | VPIN 0.42 ✓ | GEX wall ✓ | HFI ✓ | ... │
└───────────────────────────────────────────────────────────────┘
```

- **WAIT state:** amber card; `reason` + `conflicts` always shown (e.g., "SPLIT DESK: Chart LONG vs Options D = −0.63").
- **No hardcoded prices:** when `option_pick is None` render "No setup — awaiting confluence" (delete the ₹228/₹285 fallback strings at `:644-647`).
- **Drill-downs** (expanders under the verdict): the existing 9-cell confluence grid (`:549+`) and the dir-flow badge details (`:535-548`) move here unchanged — evidence views, not decisions.
- **Tabs** (`:690+`) remain as-is: evidence drill-downs (chart, journal, sizer, OI, backtest, cheatsheet).
- **Journal:** verdict snapshot persisted on each entry (via `signal.details["options_context"]` + `desk_verdict` dict passed to `log_signal`).
- Streamlit invariant: `width='stretch'` everywhere.

---

## 6. Test Plan

| File | Tests |
|---|---|
| `tests/test_options_positioning.py` | PCR z-score math (warm vs cold start, cap 120); ITM/OTM shift sign correctness (bullish vs bearish synthetic chains); max-pain drift across snapshots; agreement_count; data_quality=VERIFIED/POSITIONING_UNVERIFIED; state persistence round-trip (temp file) |
| `tests/test_desk_verdict.py` | Fusion: chart LONG + bearish D (≥ veto) ⇒ WAIT with conflict named; BUY_CE when agreed; RANGE_FADE triggers (wall + positive gamma + contrarian PCR z); GAMMA_BREAKOUT trigger (negative gamma + corridor break); strike clamped inside corridor (T2 ≤ call_wall for LONG); WAIT always has non-empty reason; missing desk_state ⇒ 0.5× + POSITIONING_UNVERIFIED (never hardcoded prices); session-locked ⇒ WAIT |
| `tests/test_strategy_rules.py` (extend) | Skew veto fires with real chain (Z > 1.5 blocks LONG); GEX wall veto fires with chain walls (LONG near call_wall blocked in +Γ); positioning veto blocks opposite-direction setups; RANGE_FADE/GAMMA_BREAKOUT branches emit correct SignalTypes; VCR squeeze sets size_factor 0.5; options_context kwarg backward compatible (None = old behavior + POSITIONING_UNVERIFIED tags) |
| Regression | `python -m pytest tests/ -v --tb=short` ; `python verify_all_modules.py` |

Update any existing test that hardcodes synthetic `is_crash_hedging=False` expectations — document each change in the test diff.

---

## 7. Rollout Order, Risks & Non-Goals

### 7.1 Rollout (each step independently green)
1. `src/options_positioning.py` (module + unit tests) — pure, no integration risk.
2. `src/desk_verdict.py` (verdict + unit tests).
3. Gate fixes 4.2 + 4.3 + options_context threading 4.1 (strategy_rules + app.py).
4. Decision rules 4.5 + confluence/config 4.6.
5. Main-page UI (Section 5).
6. Full regression + journal audit check.

### 7.2 Risk register
| Risk | Mitigation |
|---|---|
| Parallel sessions editing app.py/strategy_rules.py (12 modified / 18 untracked) | Re-grep anchors before every edit; additive-only changes; new code in new files |
| Real-chain gates fire for the first time ⇒ fewer signals, backtest-vs-live drift | Correct behavior; document in tests + changelog; monitor gate audit counts |
| `decision_engine.py` exists but is unwired (duplicate gate system) | Verdict targets the LIVE path (`_apply_universal_gates`); flag merge decision to user — do not edit decision_engine.py |
| PCR z cold start | Neutral (0.0) until `PCR_HISTORY_MIN_SAMPLES=20`; verdict excludes PCR votes when unverified |
| Chain fetch failure/stale | `POSITIONING_UNVERIFIED` path: desk setups WAIT, chart setups 0.5× — never a guess |
| Verdict must render even when ticket/signal missing | Pure function tolerates None inputs; UI renders "No setup" state (no fake prices) |
| Existing tests assume synthetic skew behavior | Explicitly update those tests and list them in the diff |

### 7.3 Non-goals
- No broker/execution wiring; no Telegram changes (in-flight elsewhere).
- No edits to `decision_engine.py`, `edge_harness.py`, `risk_state.py`, `notifications.py` (other sessions' in-flight work — reference only).
- No new external data dependencies; no new pip packages.
- No redesign of the tabs — they remain evidence drill-downs.

---

*Execution source of truth for the Desk Verdict layer. See `implementation/signal-quality-implementation-plan.md` for the broader signal-quality phases this layer extends.*
