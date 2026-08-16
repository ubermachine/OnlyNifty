"""
Smart Execution & Smart Order Routing (SOR) Engine (Phase 4).
Provides:
1. OrderManager: State-Machine Limit Order "Chase & Cancel" Execution with strict NSE tick rules (₹0.05)
   and maximum slippage tolerance abort protection.
2. TWAP / VWAP Child Order Slicer: Institutional lot slicing to mask footprints and prevent HFT front-running.
"""

from enum import Enum
from typing import Dict, List, Any, Optional
import time
import numpy as np

from src.config import LOT_SIZE

class OrderState(Enum):
    PENDING = "PENDING"
    PASSIVE = "PASSIVE"             # State 1: Limit at Best Ask
    AGGRESSIVE = "AGGRESSIVE"       # State 2: Limit at Best Ask + 1 Tick (₹0.05)
    MAX_SLIPPAGE = "MAX_SLIPPAGE"   # State 3: Limit at Best Ask + 2 Ticks (₹0.10)
    FILLED = "FILLED"
    ABORTED = "ABORTED"             # Exceeded maximum allowable slippage ceiling


class OrderManager:
    """
    NSE Options Limit Order State-Machine Execution Manager.
    Executes trades passively at Best Ask, ratcheting aggressively by tick increments (₹0.05)
    before aborting if market impact exceeds slippage ceiling.
    """

    def __init__(
        self,
        tick_size: float = 0.05,
        max_slippage_pts: float = 3.0,
        passive_timeout_ms: int = 500,
        aggressive_timeout_ms: int = 1000
    ):
        self.tick_size = tick_size
        self.max_slippage_pts = max_slippage_pts
        self.passive_timeout_ms = passive_timeout_ms
        self.aggressive_timeout_ms = aggressive_timeout_ms

    def simulate_chase_and_cancel_execution(
        self,
        target_symbol: str,
        side: str,
        initial_best_ask: float,
        simulated_market_drift_ticks: int = 1,
        fill_latency_ms: int = 350
    ) -> Dict[str, Any]:
        """
        Simulates the 3-state limit order execution protocol against live book depth.
        """
        current_ask = initial_best_ask
        state = OrderState.PASSIVE
        order_price = current_ask
        elapsed_ms = 0
        state_log = []

        # State 1: Passive Limit Order at Best Ask
        state_log.append(f"T+0ms: [PASSIVE] Placed Limit {side} @ ₹{order_price:.2f} (Best Ask)")

        if fill_latency_ms <= self.passive_timeout_ms and simulated_market_drift_ticks == 0:
            state = OrderState.FILLED
            fill_price = order_price
            elapsed_ms = fill_latency_ms
            state_log.append(f"T+{elapsed_ms}ms: [FILLED] Passive Limit filled @ ₹{fill_price:.2f} (Zero Slippage)")
        else:
            # State 2: Aggressive Ratchet (+1 Tick)
            elapsed_ms = self.passive_timeout_ms
            current_ask += simulated_market_drift_ticks * self.tick_size
            order_price = initial_best_ask + self.tick_size
            state = OrderState.AGGRESSIVE
            state_log.append(f"T+{elapsed_ms}ms: [AGGRESSIVE] Unfilled. Cancel & Replaced @ ₹{order_price:.2f} (+1 Tick)")

            if fill_latency_ms <= self.aggressive_timeout_ms and simulated_market_drift_ticks <= 1:
                state = OrderState.FILLED
                fill_price = order_price
                elapsed_ms = min(fill_latency_ms, self.aggressive_timeout_ms)
                state_log.append(f"T+{elapsed_ms}ms: [FILLED] Aggressive Limit filled @ ₹{fill_price:.2f} (Slippage: +₹{fill_price - initial_best_ask:.2f})")
            else:
                # State 3: Max Slippage (+2 Ticks) or Abort Check
                elapsed_ms = self.aggressive_timeout_ms
                current_ask += simulated_market_drift_ticks * self.tick_size
                order_price = initial_best_ask + 2 * self.tick_size

                slippage = current_ask - initial_best_ask
                if slippage > self.max_slippage_pts:
                    state = OrderState.ABORTED
                    fill_price = 0.0
                    state_log.append(f"T+{elapsed_ms}ms: [ABORTED] Price surged by ₹{slippage:.2f} > Max Tolerance (₹{self.max_slippage_pts:.2f}). Order Killed.")
                else:
                    state = OrderState.FILLED
                    fill_price = order_price
                    state_log.append(f"T+{elapsed_ms}ms: [FILLED] Max Slippage Order filled @ ₹{fill_price:.2f} (Slippage: +₹{fill_price - initial_best_ask:.2f})")

        realized_slippage = round(fill_price - initial_best_ask, 2) if state == OrderState.FILLED else 0.0

        return {
            "symbol": target_symbol,
            "side": side,
            "final_state": state.value,
            "initial_best_ask": initial_best_ask,
            "fill_price": fill_price,
            "realized_slippage_pts": realized_slippage,
            "execution_duration_ms": elapsed_ms,
            "state_log": state_log,
            "is_successful": state == OrderState.FILLED
        }


def slice_institutional_order(
    total_lots: int,
    lot_size: int = LOT_SIZE,   # was hardcoded 65 — never a Nifty lot size (that was FinNifty's)
    slice_count: int = 4,
    interval_seconds: int = 30,
    algo: str = "VWAP"
) -> List[Dict[str, Any]]:
    """
    Slices large institutional lot sizes (e.g. >= 10 lots) into time-distributed child orders
    using TWAP or VWAP volume curve weighting to minimize market impact.
    """
    if total_lots <= 2:
        return [{
            "child_order_id": 1,
            "lots": total_lots,
            "qty": total_lots * lot_size,
            "weight_pct": 100.0,
            "scheduled_offset_sec": 0,
            "algo": "DIRECT_FILL"
        }]

    effective_slices = min(slice_count, total_lots)
    
    if algo == "VWAP":
        # Institutional U-shaped or front-loaded liquidity weights
        base_weights = np.array([0.35, 0.25, 0.20, 0.20])[:effective_slices]
        weights = base_weights / np.sum(base_weights)
    else: # TWAP (Uniform)
        weights = np.ones(effective_slices) / effective_slices

    lots_allocated = np.round(weights * total_lots).astype(int)
    # Correct rounding discrepancies
    diff = total_lots - np.sum(lots_allocated)
    lots_allocated[0] += diff
    lots_allocated = np.maximum(lots_allocated, 1)

    child_orders = []
    for i, lots in enumerate(lots_allocated):
        child_orders.append({
            "child_order_id": i + 1,
            "lots": int(lots),
            "qty": int(lots * lot_size),
            "weight_pct": round(float(lots / total_lots) * 100, 1),
            "scheduled_offset_sec": i * interval_seconds,
            "algo": f"{algo}_CHILD_SLICE"
        })

    return child_orders
