"""
Causal order-lifecycle simulator.

Unlike a purely random data generator, this module builds each order's
ground-truth prep time (`actual_kpt`) out of a real kitchen workflow DAG
(see workflow_graph.py) whose step durations are driven by two *causal*
covariates:

    - active_orders: kitchen congestion, estimated with a min-heap
      interval-overlap sweep (classic "meeting rooms" style problem).
    - complexity: per-order dish complexity.

The merchant-marked "Food Order Ready" (FOR) timestamp is then generated
as the true ready time plus (a) a fixed per-merchant bias, (b) a
congestion-correlated distortion that a simple per-merchant average
CANNOT remove, and (c) irreducible random noise. This is what creates
genuine headroom for the Composite Readiness Score to beat a naive
bias-corrected baseline: only a congestion-aware signal can claw back
component (b).
"""

from __future__ import annotations

import heapq
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .config import SimulationConfig
from .workflow_graph import build_order_workflow


def _estimate_active_orders(order_times_by_merchant: dict, nominal_minutes: float) -> dict:
    """Min-heap interval-overlap sweep: how many orders are "in flight"
    at each order's arrival time, per merchant.

    For each merchant, orders are processed in arrival order. A min-heap
    of nominal finish-times represents orders currently being cooked.
    Before processing a new order we pop every finished order (O(log n)
    per pop), and the heap size left over is the concurrency estimate.
    This is the same technique used to solve "minimum meeting rooms"
    interval-scheduling problems, applied here to kitchen throughput.

    Returns a dict keyed by the original record index -> active_orders.
    """
    active_orders_by_index: dict[int, int] = {}

    for merchant_id, entries in order_times_by_merchant.items():
        # entries: list of (order_time, original_index), arrival-sorted.
        heap: list[datetime] = []
        for order_time, idx in entries:
            while heap and heap[0] <= order_time:
                heapq.heappop(heap)
            active_orders_by_index[idx] = len(heap)
            heapq.heappush(heap, order_time + timedelta(minutes=nominal_minutes))

    return active_orders_by_index


def _sliding_window_rider_cluster(
    rider_times_by_merchant: dict, window_minutes: float
) -> dict:
    """Sliding-window rider density: for each rider arrival, count how
    many OTHER rider arrivals at the same merchant fall within
    +/- window_minutes/2. High density means riders are piling up
    waiting at the merchant -- a proxy for merchant-side congestion
    that is independent of the kitchen's own order count.

    Implemented as a classic two-pointer sliding window over arrival
    times sorted ascending: O(n) per merchant after the sort.
    """
    cluster_by_index: dict[int, float] = {}
    half_window = timedelta(minutes=window_minutes / 2)

    for merchant_id, entries in rider_times_by_merchant.items():
        # entries: list of (rider_time, original_index), time-sorted.
        times = [e[0] for e in entries]
        n = len(times)
        left = 0
        for right in range(n):
            while times[right] - times[left] > half_window * 2:
                left += 1
            # Window [left, right] all lie within `window_minutes` of
            # each other by construction; density = count - 1 (exclude self).
            density = right - left
            idx = entries[right][1]
            cluster_by_index[idx] = density

    if not cluster_by_index:
        return cluster_by_index

    max_density = max(cluster_by_index.values()) or 1
    return {k: v / max_density for k, v in cluster_by_index.items()}


def _sample_order_offsets(config: SimulationConfig, rng: np.random.Generator) -> np.ndarray:
    """Sample arrival offsets (minutes) with realistic lunch/dinner rush
    clustering rather than a flat uniform spread. A meaningful fraction
    of orders land inside short, high-density rush windows, which is
    what makes per-merchant concurrency (active_orders) actually vary --
    a flat uniform spread over a wide window (the original prototype's
    approach) produces near-zero overlap and no real congestion signal.
    """
    total_span = config.simulation_span_minutes
    rush_windows = config.rush_windows_minutes  # list of (start, width)

    is_rush = rng.random(config.num_orders) < config.rush_order_fraction
    offsets = np.empty(config.num_orders)

    n_rush = int(is_rush.sum())
    if n_rush:
        window_choice = rng.integers(0, len(rush_windows), size=n_rush)
        starts = np.array([rush_windows[w][0] for w in window_choice])
        widths = np.array([rush_windows[w][1] for w in window_choice])
        offsets[is_rush] = starts + rng.random(n_rush) * widths

    n_off = config.num_orders - n_rush
    if n_off:
        offsets[~is_rush] = rng.random(n_off) * total_span

    return offsets


def generate_orders(config: SimulationConfig) -> pd.DataFrame:
    """Simulate `config.num_orders` order lifecycles across
    `config.num_merchants` merchants with causally-linked signals.
    """
    rng = np.random.default_rng(config.random_seed)

    start_time = datetime.now()
    merchant_ids = rng.integers(1, config.num_merchants + 1, size=config.num_orders)
    order_offsets = _sample_order_offsets(config, rng)
    order_times = [start_time + timedelta(minutes=float(m)) for m in order_offsets]
    complexities = rng.uniform(1, 5, size=config.num_orders)

    # Fixed per-merchant behavioral bias (drawn once per merchant).
    merchant_bias = {
        m: rng.normal(config.merchant_bias_mean, config.merchant_bias_std)
        for m in range(1, config.num_merchants + 1)
    }

    # --- Congestion estimation (heap sweep), grouped by merchant ---------
    by_merchant_order_times: dict[int, list] = {}
    for idx, (m, t) in enumerate(zip(merchant_ids, order_times)):
        by_merchant_order_times.setdefault(int(m), []).append((t, idx))
    for m in by_merchant_order_times:
        by_merchant_order_times[m].sort(key=lambda e: e[0])

    active_orders_map = _estimate_active_orders(
        by_merchant_order_times, config.nominal_service_minutes
    )
    active_orders = np.array([active_orders_map[i] for i in range(config.num_orders)])

    records = []
    for i in range(config.num_orders):
        m = int(merchant_ids[i])
        order_time = order_times[i]
        complexity = complexities[i]
        n_active = int(active_orders[i])

        graph = build_order_workflow(
            complexity=complexity,
            active_orders=n_active,
            congestion_scale=config.congestion_scale_per_active_order,
            complexity_scale_divisor=config.complexity_scale_divisor,
            rng=rng,
        )
        actual_kpt = max(4.0, graph.critical_path_length())
        actual_ready_time = order_time + timedelta(minutes=actual_kpt)

        # FOR marking = fixed bias + congestion-correlated distortion
        # (NOT removable by a per-merchant mean) + residual noise.
        rush_distortion = config.rush_distortion_per_active_order * n_active
        marking_noise = rng.normal(0, config.marking_noise_std)
        for_time = actual_ready_time + timedelta(
            minutes=merchant_bias[m] + rush_distortion + marking_noise
        )

        rider_offset = rng.normal(
            config.rider_arrival_noise_mean, config.rider_arrival_noise_std
        )
        rider_arrival_time = actual_ready_time + timedelta(minutes=rider_offset)

        records.append(
            {
                "merchant_id": m,
                "order_time": order_time,
                "actual_ready_time": actual_ready_time,
                "for_time": for_time,
                "rider_arrival_time": rider_arrival_time,
                "actual_kpt": actual_kpt,
                "active_orders": n_active,
                "complexity": complexity,
            }
        )

    df = pd.DataFrame(records)

    # --- Rider Wait Cluster Detection (sliding window) --------------------
    by_merchant_rider_times: dict[int, list] = {}
    for idx, row in df.iterrows():
        by_merchant_rider_times.setdefault(int(row["merchant_id"]), []).append(
            (row["rider_arrival_time"], idx)
        )
    for m in by_merchant_rider_times:
        by_merchant_rider_times[m].sort(key=lambda e: e[0])

    wait_cluster_map = _sliding_window_rider_cluster(
        by_merchant_rider_times, config.rider_cluster_window_minutes
    )
    df["wait_cluster"] = df.index.map(wait_cluster_map).astype(float)

    return df
