# OnlyNifty — Signal Quality & Noise Reduction: Implementation Plan

> **Status:** Approved blueprint (pending implementation)
> **Date:** 2026-08-15
> **Scope:** Signal pipeline only (`strategy_rules`, `signal_journal`, `app.py`, `config`, backtest/validation). No broker execution, no UI redesign, no Telegram notifier changes.
> **Audience:** The implementation agent(s) that will execute this plan.

---

## 0. Executive Summary

The system generates many signals, but few of them are *decisions*. The journal (56 entries over 2 sessions) shows a **18.2% realized win rate** (8 wins / 44 closed), a **median R of −1.0**, net **−₹17,117**, and — most damning — **39 of 54 traded entries carried the system's own "C Weak / Vetoed" grade**. The confluence score is computed *after* the trade decision, the defensive gates guard only the last two branches of the setup ladder, the live path enforces none of the risk rails that made the backtest look good, and the journal itself is corrupted by the restart-seeder replaying a whole day in 12 seconds.

This plan converts the pipeline from "setup ladder that emits trades" to **"decision engine whose default answer is no-trade"**, following institutional doctrine:

1. A signal is only a signal if it carries a measured, out-of-sample edge.
2. Every trade must pass *universal* hard gates — evaluated before any setup can fire.
3. Confluence score must be an *input* to the decision (veto below 70), never a display label.
4. Position size is a function of signal quality, regime, and volatility.
5. The journal must be ground truth — seeds, duplicates, and friction must not poison the statistics that close the feedback loop.
6. The headline performance number must come from walk-forward out-of-sample validation, not an in-sample run.

---

## 1. Goal & Success Metrics

### Goal
Trade Nifty index options with **fewer, higher-quality signals** — institution-style: ≤3 trades/day, every trade passing universal gates, each setup carrying a measured conditional edge, and an honest audit trail.

### Success metrics (all verifiable post-implementation)

| # | Metric | Target |
|---|--------|--------|
| M1 | Trade tickets per day (live path) | ≤ 3 (hard budget, enforced in code) |
| M2 | Tickets with confluence score < 70 | 0 (veto, not label) |
| M3 | Tickets failing any hard gate (skew/VPIN/HFI/GEX/session-lock) | 0 — every ticket's `details` carries the gate audit snapshot |
| M4 | Stop distance | ≥ max(0.5×ATR14, 2×5m noise band) and ≤ 60 index pts; never SL == entry |
| M5 | Lunch-lull (11:30–13:00 IST) tickets | Sizing actually halved (×0.5), not just a note |
| M6 | Journal entries | 100% carry `setup_id` + evidence snapshot; seeded entries flagged and excluded from stats |
| M7 | Per-setup OOS edge table | n ≥ 30 closed trades before a setup is "TRUSTED"; quarantine if 95% CI lower bound < 0 |
| M8 | Headline benchmark | Regenerated from walk-forward OOS; README wording updated to match |
| M9 | Realized expectancy (target after Phase 3+) | Median R > 0 and profit factor ≥ 1.2 net of TCA per trusted setup |

---

## 2. Diagnosis — Why the Current Signals Are Noisy (Evidence)

All findings verified by full-file audit (`src/strategy_rules.py`, `src/signal_journal.py`, `app.py`, `src/backtest_engine.py`) and full paging of the 56-entry journal (`data/signals_journal_today.json`).

### 2.1 Decision-before-quality (the central defect)
- The confluence score is computed **post-hoc, for display only**, inside `signal_journal.py` (`:138–245`). `app.py:402` even passes a hardcoded `confluence_score=1.0` to the journal.
- **No branch** in `strategy_rules.py` reads or applies the score. Zero grep hits for `confluence_score` in `strategy_rules.py`.
- **Evidence:** 39/54 traded entries were graded "C Weak / Vetoed" (scores 28–60); 33 of 39 stopped out at −1R. Example: `SIG-20260815-121546-24300CE` — LONG_ORDER_FLOW, score 45, "C Weak / Vetoed", traded 15 lots, STOPPED_OUT −1R (−₹4,999).
- The system literally labels its own trades "Vetoed" and trades them anyway.

### 2.2 Gate inversion (priority ladder)
- The defensive gates — 25Δ skew crash veto (`:617–627`), HFI veto (`:630–640`, `:700–710`), GEX call-wall pin (`:643–653`), HTF veto (`:655–665`, `:712–722`) — exist **only inside the final Fibonacci LONG/SHORT branches**.
- The aggressive branches that fire **first** skip every gate:
  - 3PM breakout chase — `:202–228`
  - Passive limit absorption trap — `:318–346`
  - Liquidity sweep traps — `:349–378`
  - Mean-reversion (CHOP) — `:381–406`
  - IB breakout — `:408–441` (regime + volume gated only)
  - AMT value-area rejection — `:443–473`
  - Spring reclaim / distribution thrust — `:475–514`
  - Order-flow absorption — `:516–546`
  - Stacked OFI imbalance — `:548–576`
- Result: the noisiest setups get first claim on capital; the carefully gated ones almost never fire. In the journal, `*_ORDER_FLOW` (mostly ungated) is the most-traded family.

### 2.3 Fake alignment
- `htf_aligned=True` is **hardcoded** on trade returns at `strategy_rules.py` lines 329, 343, 361, 375, 392, 404, 425, 439, 456, 470, 511, 529, 543, 559, 573, 686, 743 (plus the dataclass default at `:49`).
- Journal rows say "HTF Aligned" where alignment was never checked.

### 2.4 Risk rails exist only where they can't hurt the narrative
- `MAX_TRADES_PER_DAY=3`, 2-strike rule, and 1.5% DLL are enforced **only in `backtest_engine.py:93–95`**.
- The live path (`app.py`, `strategy_rules.py`) has zero enforcement; `self.session_losses` (`strategy_rules.py:102`) is dead code.
- **Consequence:** the README's 77.8% backtest is a *different system* than what runs live. The backtest enforces limits, the live app doesn't — the claim is untransferable by construction.

### 2.5 Stop & lifecycle hygiene
- 10–11 pt stops on a ~24,300 index (≈0.04%) — **inside the 5m noise band** → 36 STOPPED_OUTs.
- 11 journal entries with SL 225–311 pts away (`sl_spot=24620.55`) — effectively no stop; ~14 with SL == entry.
- 11 entries with inverted moonshot targets (T3 inside T1).
- ~14 entries: "T1 Hit. SL trailed to entry. | SL Hit" — winners of +38…+70 pts given back to −1R by trail-to-entry on 5m noise.
- TCA friction is added into `realized_pnl_rupees`, so every loss exceeds −1R (e.g., ₹4,500 risk → −₹5,010).

### 2.6 Audit-trail corruption (the seeder)
- `signal_journal.py:501–557` (`seed_from_intraday_history`) replays the entire day's bars on restart, re-logs every signal with **hardcoded** `regime_info="TRENDING_EXPANSION"` (`:542`) and `confluence_score=85.0` (`:543`).
- **Evidence:** 49 entries were written in a 12-second burst (12:15:46–12:15:57 IST) carrying Aug-14 bar timestamps under Aug-15 wall-clock timestamps; 30 duplicate signal_ids across 11 prefixes; 10 TRIGGERED entries left unresolved.
- The `SignalPerformanceAnalyzer` (win rate by type/regime/score, `:726/:868/:898`) is fed this corrupted data — the feedback loop cannot work.

### 2.7 Validation theater
- The 77.8% README benchmark comes from `evaluate_backtest.py`: one in-sample run over ≤5 days of yfinance data, **falling back to synthetic data** on fetch failure (`:10–12`). No walk-forward, no out-of-sample split, no purging, no confidence intervals, no per-setup stats.

### 2.8 Journal evidence table (2 sessions, 56 entries)

| Dimension | Result |
|---|---|
| Entries | 56 (Aug-14: 4 live; Aug-15: 52, of which 49 are the 12-second seeder replay) |
| Signal types | WAIT 2 · LONG 9 · SHORT 27 · LONG_ORDER_FLOW 9 · SHORT_ORDER_FLOW 8 · SHORT_3PM 1 |
| Grades | "C Weak / Vetoed" **39** · "B Tactical" 13 · "A Standard" 2 · Consolidation 2 |
| Lifecycle | STOPPED_OUT 36 · TRIGGERED 10 · T3_MOONSHOT 4 · T2_REACHED 4 · AWAITING_SETUP 2 |
| Realized R (44 closed) | mean −0.07 · median −1.0 · **win rate 18.2%** (8/44) |
| Win rate by type | LONG 0/9 · SHORT 2/17 · LONG_OF 3/9 · SHORT_OF 3/8 · SHORT_3PM 0/1 |
| Win rate by grade | A 1/2 · B 1/5 (20%) · **C 6/39 (15.4%)** |
| PnL | Gross wins +₹137,275 (8) · gross losses −₹154,392 (36) · **net −₹17,117** |

Direction also flip-flops bar-to-bar (LONG_ORDER_FLOW and SHORT on consecutive 5m bars within the same replay).

---

## 3. Design Principles (Institutional Doctrine)

1. **Default answer is no-trade.** 0–3 trades/day is normal. Every signal must *earn* the right to exist; WAIT is the fall-through, not an exception.
2. **Independent evidence families, not correlated votes.** EMA21/55/200, AVWAP, VAKC, fib — all functions of the same price series — count as **one** family (structure). A trade needs agreement across ≥3 families: (a) *structure*, (b) *order flow* (CVD/OFI/absorption), (c) *positioning* (OI delta, GEX, PCR, FII/DII), (d) *macro/vol* (HFI, skew, VCR).
3. **Universal gates, evaluated first.** Crash-skew, VPIN toxicity, HFI, GEX-wall, session-lock, budget, cooldown — checked **before any setup** for **every setup**, with gate audit snapshots written into the signal's `details`.
4. **Regime-conditional edge tables.** A setup is only tradeable in the regimes where it has a measured edge. CHOP → fades; TRENDING → continuation; HIGH_VOL_EXPANSION → half size or stand down.
5. **Sizing as signal quality.** `size = Quarter-Kelly × (score/100) × regime_multiplier × VCR_factor × lunch_factor`. Score < 70 ⇒ size 0.
6. **Honest validation.** Walk-forward with purge/embargo; OOS-only reporting; per-setup confidence intervals; quarantine on lower CI bound < 0. If a claim can't survive that, it doesn't ship.
7. **Journal = ground truth.** Every row carries what justified it; seeds/duplicates/friction never pollute the statistics that drive the feedback loop.
8. **Gates fail closed.** Missing gate data (no option chain, no HFI, no macro) ⇒ WAIT. Never a trade.

---

## 4. Phase 1 — Kill-Switch Rails & Universal Gates

*Highest leverage, smallest change. Cuts the noise immediately on the existing ladder.*

### 4.1 Files touched
- `src/strategy_rules.py` — hoist gates; honest alignment; pre-decision scoring hook
- `src/signal_journal.py` — move scoring to a reusable pre-decision function
- `app.py` — enforce session budget / 2-strike / DLL / one-open-trade / cooldown before ticket generation
- `src/config.py` — new thresholds (below)

### 4.2 Changes

**4.2.1 Hoist the gates above the ladder** (`strategy_rules.py::_evaluate_bar_core`)
- Insert a `_apply_universal_gates(...)` step immediately after indicators are computed (~`:296`) and **before** any setup branch. It evaluates, in order:
  1. 25Δ put-skew crash veto (`SKEW_ZSCORE_THRESHOLD`): if `Z > 1.5` → block LONGs (block ALL if `GATE_FAIL_TO_WAIT` and data missing).
  2. VPIN toxicity (`VPIN_TOXICITY_THRESHOLD = 0.65`, currently computed at `:258` but never used): if `vpin > 0.65` → block everything.
  3. HFI veto: `|hfi| > 0.2` against direction (currently only in fib branches `:630–640`, `:700–710`).
  4. GEX call/put-wall pin (`GEX_WALL_BUFFER_PTS`): block trades toward a wall within buffer (currently `:643–653` only).
  5. Session lock: day-changed reset + 2-strike + DLL (moved from backtest-only logic).
- Every block returns a `WAIT` signal whose `reason` and `details` name the gate and its value — so the journal shows *why* nothing traded.
- The per-branch gate copies in `:617–722` become redundant and are **deleted** (keeps one source of truth).

**4.2.2 Honest HTF alignment**
- Delete all 17 hardcoded `htf_aligned=True` trade-return sites (lines listed in §2.3). Each branch must pass the computed `htf_aligned_long` / `htf_aligned_short` from `compute_multi_timeframe_regime` (`:292–294`).
- Rule: a LONG branch requires `htf_aligned_long`; a SHORT branch requires `htf_aligned_short`; mismatch → downgrade to WAIT with reason `HTF veto`.
- Keep the dataclass default but make it `htf_aligned: bool = False` so unset ≠ aligned.

**4.2.3 Pre-decision confluence score**
- Extract the scoring logic from `signal_journal.py:138–245` into `signal_journal.py::compute_confluence_score(signal, htf_regime, indicators, order_flow, ...) -> float` (pure function, no state).
- `strategy_rules.py` calls it **before** returning any trade Signal and stores it in `Signal.details["confluence_score"]`.
- Add `SIGNAL_MIN_CONFLUENCE = 70.0` (config). Score < 70 ⇒ return WAIT (`reason`: `Confluence veto: score {x} < {floor}`). This turns "C Weak / Vetoed" from a label into a veto.
- The journal's display path keeps recomputing for compatibility, but the pre-decision value is authoritative and stored.

**4.2.4 Real lunch-lull halving**
- `app.py` ticket path: when `strategy_engine._last_lunch_lull` is set (`:136–137`, `:233–238`), multiply lots by `LUNCH_LULL_SIZE_FACTOR = 0.5` in `generate_option_trade_ticket` sizing inputs. The advisory string stays, but the halving becomes real.

**4.2.5 Live session budget (port the backtest rails)**
- New `SessionRiskState` dataclass (in `strategy_rules.py` or a new `src/risk_state.py`): `date`, `trades_today`, `consecutive_losses`, `realized_pnl_today`, `locked`, `lock_reason`.
- `app.py`, before `generate_option_trade_ticket`: consult state → refuse ticket when `trades_today >= MAX_TRADES_PER_DAY`, `consecutive_losses >= MAX_CONSECUTIVE_LOSSES_DAY`, `realized_pnl_today <= -DAILY_LOSS_LIMIT_PCT × capital`, or one open trade exists (`MAX_OPEN_TRADES = 1`), or within `COOLDOWN_BARS = 12` (60 min of 5m bars) of the last entry.
- Update state on lifecycle transitions (STOPPED_OUT / target reached) via the journal's lifecycle hook. Persist state next to the journal file so a restart keeps the budget.

### 4.3 New config keys (`src/config.py`)
```python
SIGNAL_MIN_CONFLUENCE: float = 70.0     # score floor; below → WAIT
COOLDOWN_BARS: int = 12                 # 60 min of 5m bars between fresh entries
MAX_OPEN_TRADES: int = 1                # one open trade at a time
GATE_FAIL_TO_WAIT: bool = True          # missing gate data → WAIT, never trade
LUNCH_LULL_SIZE_FACTOR: float = 0.5     # real halving, 11:30–13:00 IST
STOP_MIN_ATR_FRACTION: float = 0.5      # SL ≥ 0.5 × ATR14
STOP_MAX_POINTS: float = 60.0           # absolute SL cap (index pts)
STOP_NOISE_BAND_MULT: float = 2.0       # SL ≥ 2 × rolling 5m bar-range σ
```

### 4.4 Tests (Phase 1)
- `tests/test_signal_quality_gates.py`: skew spike blocks LONG (and blocks all with missing chain + `GATE_FAIL_TO_WAIT`); VPIN > 0.65 blocks everything; HFI > +0.2 blocks SHORT; GEX wall within buffer blocks approach; score 45 ⇒ WAIT; score 85 passes; missing hfi/chain ⇒ WAIT.
- `tests/test_session_risk_state.py`: 3rd trade refused; 2 consecutive losses lock the session; DLL breach locks; cooldown blocks re-entry; state persists across restart (temp file round-trip).
- Lunch halving: ticket lots halved when `_last_lunch_lull` set.

---

## 5. Phase 2 — Decision Architecture (Setup → Decision Separation)

*Restructure so gates, budgets, and edge tables apply uniformly. Prerequisite for Phase 3.*

### 5.1 Files
- New: `src/decision_engine.py`
- `src/strategy_rules.py` — ladder becomes pure detection
- `app.py` — consumes `DecisionEngine`

### 5.2 New types (`src/decision_engine.py`)
```python
@dataclass
class SetupCandidate:
    setup_id: str                 # e.g. "IB_BREAKOUT_LONG", "CHOP_FADE_SHORT"
    signal_type: SignalType
    direction: str                # "LONG" | "SHORT"
    entry_price: float
    sl_price: float
    targets: tuple[float, float, float]     # T1, T2, T3
    pyramid_trigger: float
    reason: str
    details: dict
    evidence: dict                # family votes: {"structure": 0|1, "flow": 0|1,
                                  #               "positioning": 0|1, "macro": 0|1}

@dataclass
class DecisionContext:
    markov_regime: str
    htf_aligned_long: bool
    htf_aligned_short: bool
    confluence_score: float
    skew_z: float; vpin: float; hfi_score: float
    gex_walls: dict; vcr: float
    lunch_lull: bool
    session_state: SessionRiskState
    live_iv: float
```

### 5.3 `DecisionEngine`
```python
class DecisionEngine:
    def __init__(self, edge_table: "EdgeTable | None" = None): ...

    def decide(self, candidates: list[SetupCandidate], ctx: DecisionContext) -> Signal:
        # 1. Hard gates (skew, VPIN, HFI, GEX, session lock, DLL, 2-strike) → WAIT + gate audit in details
        # 2. Budget & cooldown (MAX_TRADES_PER_DAY, MAX_OPEN_TRADES, COOLDOWN_BARS)
        # 3. Regime edge-table lookup (Phase 3): untrusted/quarantined setup → WAIT
        # 4. Confluence floor: score < SIGNAL_MIN_CONFLUENCE → WAIT
        # 5. Rank candidates (edge EV desc, then score desc); take the best ONE
        # 6. Size: Quarter-Kelly × (score/100) × regime_mult × VCR_factor × lunch_factor
        #    (score 70–79 → 0.5×, 80–89 → 0.75×, ≥90 → 1.0× — never above Kelly fraction)
        # Returns at most one trade Signal per call.
```

### 5.4 Ladder refactor (`strategy_rules.py`)
- Rename the core into `detect_setups(self, ...) -> list[SetupCandidate]` — **pure detection, no early trade returns, no gates**. Each of the ~10 existing branches appends a `SetupCandidate` with its `setup_id` and evidence votes instead of `return Signal(...)`.
- Keep `evaluate_bar(...) -> Signal` as a backward-compatible wrapper: `detect_setups → DecisionEngine.decide`, with a module-level fallback `DecisionEngine` when none is injected (so existing tests that call `evaluate_bar` still pass, modulo the new gates which tests will update).
- Delete the per-branch gate copies (`:617–722`) — gates now live only in `DecisionEngine`.

### 5.5 Tests (Phase 2)
- `tests/test_decision_engine.py`: only one ticket per call even with 3 candidates; ranking picks highest-EV candidate; sizing math (Kelly × multipliers, lunch halving, score tiers); gate audit appears in returned WAIT details; all gates evaluate before any setup logic runs.

---

## 6. Phase 3 — Walk-Forward Edge Harness

*Measure each setup's real, out-of-sample edge; auto-quarantine losers. This is what makes "quality" measurable instead of asserted.*

### 6.1 Files
- New: `src/edge_harness.py`
- Extend: `evaluate_backtest.py` (or new `scripts/walkforward_benchmark.py`) to replace the in-sample benchmark
- `README.md` — regenerate the performance badge from OOS results

### 6.2 New module (`src/edge_harness.py`)
```python
@dataclass
class EdgeStats:
    setup_id: str; regime: str
    n: int                      # OOS closed trades
    win_rate: float; mean_r: float; ev: float
    ci_low: float; ci_high: float    # 95% (bootstrap or Wilson for win rate)
    status: str                 # "TRUSTED" | "PAPER" | "QUARANTINED"

class EdgeTable:
    def lookup(self, setup_id: str, regime: str) -> EdgeStats | None: ...
    def is_tradeable(self, setup_id: str, regime: str) -> bool: ...
    def to_json(self) -> str: ...
    @classmethod
    def from_json(cls, s: str) -> "EdgeTable": ...

class WalkForwardRunner:
    def run(self, df_5m, *, train_days=30, test_days=5,
            purge_bars=60, embargo_bars=12) -> EdgeTable:
        # Rolling train/test windows over the same DecisionEngine code path.
        # purge_bars: drop indicators-lookback bars at each boundary (no leakage).
        # embargo_bars: skip signals overlapping the boundary.
        # Aggregate per (setup_id, regime): n, win_rate, mean_r, ev, CI.
```
**Quarantine policy (avoids premature kills):**
- `n < QUARANTINE_MIN_SAMPLES (30)` → `PAPER` (tradeable at half size only if score ≥ 80).
- `ci_low < 0` → `QUARANTINED` (no trades; alert-only).
- `ci_low ≥ 0` and `n ≥ 30` → `TRUSTED` (full size).
- Persist `EdgeTable` to `data/edge_table.json`; `DecisionEngine` loads it and applies §5.3 step 3.

### 6.3 Validation rules encoded
- The harness drives the **same** `detect_setups → DecisionEngine` path used live (the §2.4 trap must never recur: backtest ≠ live).
- OOS-only reporting; no parameter re-tuning on test windows; deflated stats reported alongside raw.

### 6.4 Tests (Phase 3)
- `tests/test_edge_harness.py`: synthetic trending/chop datasets produce non-empty EdgeTable; **no-lookahead assertion** (a signal logged at bar *t* may not use any data with index > *t*); purge/embargo respected at boundaries; a deliberately bad setup (inverted entries) lands `QUARANTINED`; JSON round-trip preserves statuses.

---

## 7. Phase 4 — Journal & Lifecycle Integrity

*Ground truth. Without this, Phases 3's edge tables learn from garbage.*

### 7.1 Files
- `src/signal_journal.py` (primary)
- `app.py` (lifecycle hook wiring)

### 7.2 Changes

**7.2.1 Seed hygiene**
- `seed_from_intraday_history` (`:501–557`): set `is_seed=True` on every entry; **skip** entries already present for the same bar+setup (idempotent); remove the hardcoded `regime_info="TRENDING_EXPANSION"` (`:542`) and `confluence_score=85.0` (`:543`) — use the real computed values; skip lifecycle finalization for seeded entries (no re-trading yesterday's history).
- `SignalEntry` gains `is_seed: bool = False`.

**7.2.2 Analyzer excludes seeds**
- `SignalPerformanceAnalyzer` (win-rate-by-type `:726`, confluence correlation `:868`, buckets `:898`, daily summary `:649`) and `LiveSignalJournal.get_journal_dataframe()` accept `include_seeds: bool = False` (default) and filter `is_seed`.

**7.2.3 Dedup by structure epoch**
- Replace the bar+direction dedup (`:300–310`) with a `structure_epoch` key: a hash of the relevant swing structure (e.g., swing-high/low fingerprint + setup_id) so the *same setup instance* can't re-log every bar. WAIT entries: keep the existing collapse rule but store them in a separate stream (`state_events`) rather than the signal list, so WAITs never dilute trade stats.

**7.2.4 Stop hygiene** (in the ladder/DecisionEngine, not the journal)
- SL = structure level + `max(STOP_MIN_ATR_FRACTION × ATR14, STOP_NOISE_BAND_MULT × σ_5m_bar_range)`; clamp to `[entry ± 5pts, entry ± STOP_MAX_POINTS]`; never SL == entry; never the 24620.55-style far stops.
- Fix inverted moonshot: `T3` must extend beyond `T2` in the trade direction (assert in `Signal.__post_init__`).

**7.2.5 Lifecycle: book T1, don't give it back**
- On T1 touch: **book 50%**, move SL to entry, keep 50% running to T2/T3. Remove "trail-to-entry on full position" (the `T1 Hit. SL trailed to entry. | SL Hit` giveback pattern, ~14 occurrences).
- Record `realized_r_multiple` from premium P&L **excluding** `tca_friction_est` (new field `realized_pnl_net` keeps friction; `realized_r_multiple` stays the pure edge measure).
- Resolve the 10 lingering TRIGGERED entries: a lifecycle pass that force-closes open entries at `HARD_SQUAREOFF_TIME` (15:15 IST) with an explicit `SQUARED_OFF` status.

**7.2.6 Evidence snapshot on every trade row**
- Each trade entry stores `setup_id`, the gate audit (which gates ran, values, pass/fail), `confluence_score` (pre-decision), and `evidence` votes. Review becomes: "why did this trade happen?" — answerable from the row alone.

### 7.3 Tests (Phase 4)
- `tests/test_journal_integrity.py`: seeder is idempotent (no duplicate ids); seeded entries excluded from all analyzer outputs by default; structure-epoch dedup prevents same-setup re-logging on consecutive bars; T1-booking splits and trails correctly; realized R excludes TCA friction; `Signal.__post_init__` rejects inverted T3 and SL == entry; square-off sweep closes TRIGGERED entries at 15:15.

---

## 8. Phase 5 — Signal Diet (Canonical Edges)

*After Phase 3 measures truth, cut the ladder down to edges that earn their keep.*

### 8.1 Target: 4 canonical edges (each with its own regime-conditional edge table)

| Edge | Setup IDs | Regime | Evidence families |
|---|---|---|---|
| **ORB / IB breakout** | `IB_BREAKOUT_LONG/SHORT` | LOW_VOL_TRENDING, after 10:15 | structure (IB range) + flow (volume > avg) + positioning (OI delta) |
| **Flow-confirmed pullback continuation** | `FLOW_CONTINUATION_LONG/SHORT` (merges fib golden-pocket `:612–751`, spring/thrust `:475–514`) | LOW_VOL_TRENDING / TRENDING_EXPANSION | structure (AVWAP/EMA21 tag) + flow (CVD/OFI agree) + macro (HFI aligned) |
| **Chop mean-reversion fade** | `CHOP_FADE_LONG/SHORT` (from `:381–406`, tightened) | MEAN_REVERTING_CHOP only | structure (VAL/VAH) + flow (OFI defense) + vol (VCR not squeezed) |
| **Expiry-day gamma pin** | `GAMMA_PIN_LONG/SHORT` (new) | Any, expiry day (0DTE Thursday) | positioning (dealer GEX wall/zero-gamma) + flow (CVD divergence) + vol (IV percentile) |

### 8.2 Demotions (paper-only until proven)
- **3PM breakout chase** (`:193–228`): demote to `PAPER`. It is a classic retail late-day chase; journal record 0/1. Reinstates only if the EdgeTable shows `ci_low ≥ 0` with n ≥ 30.
- **Liquidity sweep traps** (`:349–378`): paper-only, same criterion.
- **Absorption traps** (`:318–346`, `:516–546`) and **stacked OFI** (`:548–576`): folded into `FLOW_CONTINUATION` candidates; they must *strengthen* the flow vote, not fire standalone.
- **AMT value-area** (`:443–473`): keep as a `CHOP_FADE` trigger, not a standalone edge.

### 8.3 Endgame additions (only if the harness supports them)
- Vol-premium sleeve (IV percentile + RV:IV spread from the existing `VolatilityIntelligence`) as a *fifth* edge class for non-directional premium capture — gated on the vol cone inputs already computed.
- FII/DII positioning edge: HFI + participant OI (`institutional_flow.py`) as a positioning vote that can veto or upgrade, never trade standalone.

### 8.4 Tests (Phase 5)
- Each canonical edge emits its `setup_id` and evidence votes correctly; demoted branches produce `PAPER`-only entries (journal flag) and never reach `DecisionEngine` sizing; `EdgeTable.is_tradeable` governs reinstatement.

---

## 9. New Module & Config Reference (Consolidated)

| Item | Location | Purpose |
|---|---|---|
| `SessionRiskState` | `src/risk_state.py` (new) | Live 2-strike / DLL / budget / cooldown; persisted |
| `SetupCandidate`, `DecisionContext`, `DecisionEngine` | `src/decision_engine.py` (new) | Universal gating + budget + edge lookup + sizing |
| `EdgeStats`, `EdgeTable`, `WalkForwardRunner` | `src/edge_harness.py` (new) | Walk-forward OOS edge measurement, quarantine |
| `compute_confluence_score()` | `src/signal_journal.py` (extracted from `:138–245`) | Pre-decision score, reused by engine + journal |
| New config keys | `src/config.py` (§4.3) | `SIGNAL_MIN_CONFLUENCE`, `COOLDOWN_BARS`, `MAX_OPEN_TRADES`, `GATE_FAIL_TO_WAIT`, `LUNCH_LULL_SIZE_FACTOR`, `STOP_*`, quarantine keys |
| `data/edge_table.json` | data dir | Persisted EdgeTable |
| `data/risk_state_today.json` | data dir | Persisted session risk state |

**Backward compatibility contract:** `StrategyEngine.evaluate_bar(...) -> Signal` signature unchanged; `LiveSignalJournal.log_signal(...)` signature unchanged; new fields defaulted so old JSON loads. `SignalEntry` gains `is_seed`, `setup_id`, `structure_epoch`, `gate_audit`, `evidence` — all optional/defaulted.

---

## 10. Test Plan (Summary)

| Suite | Proves |
|---|---|
| `tests/test_signal_quality_gates.py` | Every gate blocks its case; missing data ⇒ WAIT |
| `tests/test_session_risk_state.py` | Budget/2-strike/DLL/cooldown enforcement + persistence |
| `tests/test_decision_engine.py` | One ticket/bar, ranking, sizing math, audit trail |
| `tests/test_edge_harness.py` | No-lookahead, purge/embargo, quarantine, round-trip |
| `tests/test_journal_integrity.py` | Seeds excluded, epoch dedup, T1 booking, R ex-friction, stop hygiene |
| Existing suite | Full `python -m pytest tests/ -v --tb=short` + `python verify_all_modules.py` must stay green (tests updated only where the new gates intentionally change behavior — each such change documented in the test diff) |

---

## 11. Rollout Order

**Recommended execution sequence (differs from numbering by intent):**

1. **Phase 1** — immediate noise cut on the current ladder; live budget enforced.
2. **Phase 4** — journal hygiene *before* measurement (clean stats are a Phase 3 precondition).
3. **Phase 2** — decision architecture (a Phase 3 precondition: the harness must drive the new decision path).
4. **Phase 3** — walk-forward edge tables; README benchmark regenerated from OOS.
5. **Phase 5** — signal diet driven by the resulting EdgeTable.

Each phase ships independently with its tests green and the full suite passing.

---

## 12. Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| The 77.8% README claim fails honest walk-forward | Reputational | Expected — update README messaging to cite the OOS report; frame as "measured, not promised" |
| Tight gates shrink sample sizes | Edge tables fill slowly | Quarantine uses lower CI bound; PAPER tier allows half-size data collection at score ≥ 80 |
| Gate data missing (HFI/chain/macro) in some data modes | Over-blocking | `GATE_FAIL_TO_WAIT=True` (safe direction); UI shows which gate blocked and why |
| In-flight work conflicts (11 modified + 8 untracked files incl. `notifications.py`) | Merge collisions | Diff `main` before editing; do not touch `src/notifications.py` / `tests/test_notifications.py` / `tests/test_v52_microstructure.py` |
| `notifications.py` may also consume `evaluate_bar` output | Ungated alerts continue | Verify its call sites during Phase 1; alerts reuse the same decisioned Signal, never a raw setup |
| Line numbers drift as work lands | Stale anchors | All citations here pair line numbers with stable anchors (function names, branch descriptions) |
| Overtightening turns the system into a no-trade bot | Missed opportunity | Success metric M9 (expectancy) is the arbiter; if 0 trades occur over a full week with gates green, relax the confluence floor to 65 — never below 60 |

---

## 13. Non-Goals (explicitly out of scope)

- No broker/execution wiring, no live order routing.
- No Telegram notifier changes (in-flight work).
- No UI redesign beyond surfacing gate-audit reasons in existing expanders.
- No parameter re-tuning on historical data outside the Phase 3 harness protocol.
- No new external data dependencies; all gates reuse already-computed inputs (VPIN, VCR, GEX, skew, HFI).

---

*End of plan. The single source of truth for "why" is §2; the "what" is §4–§8; the "proof" is §10.*
