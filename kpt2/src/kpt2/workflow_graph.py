"""
Graph-based dependency resolution for kitchen prep workflows.

A kitchen order is not a single "prep time" number -- it's a small
directed acyclic graph (DAG) of steps (pull ingredients, chop, marinate,
cook, plate, quality check...) where some steps can run in parallel and
others must wait on upstream steps. The true minimum completion time for
an order is the length of the LONGEST path through that DAG (the
"critical path"), not the sum of every step.

This module implements the two classic graph algorithms that make that
computation possible, from scratch (no networkx dependency):

    1. Kahn's algorithm for topological sorting (BFS + in-degree
       counting via a queue -- collections.deque).
    2. A single dynamic-programming pass over the topological order to
       compute the longest path ("critical path method" / CPM), which is
       the standard technique used in project-scheduling software.

`build_order_workflow` wires a concrete 8-step kitchen template using
these primitives and is the function the rest of the pipeline calls.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


class GraphCycleError(ValueError):
    """Raised when a workflow graph is not a valid DAG."""


@dataclass
class WorkflowGraph:
    """A directed acyclic graph of named steps with per-node durations.

    Internally stored as an adjacency list (dict[str, list[str]]), which
    is the standard O(V + E)-friendly representation for sparse graphs
    like a cooking workflow (a handful of nodes, a handful of edges).
    """

    nodes: list[str] = field(default_factory=list)
    durations: dict[str, float] = field(default_factory=dict)
    adjacency: dict[str, list[str]] = field(default_factory=dict)

    def add_node(self, name: str, duration: float) -> None:
        if name not in self.adjacency:
            self.nodes.append(name)
            self.adjacency[name] = []
        self.durations[name] = duration

    def add_edge(self, upstream: str, downstream: str) -> None:
        """`downstream` depends on `upstream` completing first."""
        self.adjacency.setdefault(upstream, [])
        self.adjacency.setdefault(downstream, [])
        self.adjacency[upstream].append(downstream)

    def topological_order(self) -> list[str]:
        """Kahn's algorithm: BFS using a queue of zero-in-degree nodes.

        Returns nodes in an order where every node appears after all of
        its dependencies. Raises GraphCycleError if the graph has a
        cycle (which would mean an invalid, unresolvable workflow).
        """
        in_degree = {n: 0 for n in self.adjacency}
        for u in self.adjacency:
            for v in self.adjacency[u]:
                in_degree[v] += 1

        queue = deque([n for n, d in in_degree.items() if d == 0])
        order: list[str] = []

        while queue:
            u = queue.popleft()
            order.append(u)
            for v in self.adjacency[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        if len(order) != len(self.adjacency):
            raise GraphCycleError(
                "Workflow graph contains a cycle; cannot resolve a valid "
                "prep order."
            )
        return order

    def critical_path_length(self) -> float:
        """Longest path through the DAG using topological-order DP.

        earliest_finish[v] = durations[v] + max(earliest_finish[u] for u
        in predecessors(v)), computed in one linear pass over the
        topological order. This is the Critical Path Method (CPM) used
        in project scheduling, applied here to kitchen prep steps.
        """
        order = self.topological_order()
        earliest_finish: dict[str, float] = {}

        for node in order:
            best_predecessor_finish = 0.0
            # Because we process in topological order, every predecessor
            # of `node` has already been finalized by the time we reach it.
            for u in self.adjacency:
                if node in self.adjacency[u]:
                    best_predecessor_finish = max(
                        best_predecessor_finish, earliest_finish[u]
                    )
            earliest_finish[node] = self.durations[node] + best_predecessor_finish

        return max(earliest_finish.values())


# ---------------------------------------------------------------------------
# Concrete kitchen workflow template
# ---------------------------------------------------------------------------
#
#   order_received -> ingredient_pull -> prep_chop -----\
#                                     \-> marinate -------> cook_primary -> cook_secondary(*) -> plate -> quality_check -> handoff
#
# (*) cook_secondary is only included for higher-complexity dishes.

_BASE_STEP_MINUTES = {
    "order_received": 0.0,
    "ingredient_pull": 1.4,
    "prep_chop": 2.6,
    "marinate": 3.4,
    "cook_primary": 6.5,
    "cook_secondary": 3.8,
    "plate": 1.3,
    "quality_check": 0.9,
    "handoff": 0.0,
}

# Steps whose duration scales with kitchen congestion (shared equipment:
# stovetops, ovens, expo line) vs. steps that don't (handoff is instant).
_CONGESTION_SENSITIVE_STEPS = {"cook_primary", "cook_secondary", "plate"}

# Steps whose duration scales with dish complexity (prep-heavy steps).
_COMPLEXITY_SENSITIVE_STEPS = {"prep_chop", "marinate", "cook_secondary"}


def build_order_workflow(
    complexity: float,
    active_orders: int,
    congestion_scale: float,
    complexity_scale_divisor: float,
    rng,
    include_secondary_cook: bool | None = None,
) -> WorkflowGraph:
    """Construct the per-order kitchen workflow DAG.

    Parameters
    ----------
    complexity : dish complexity score (roughly 1-5).
    active_orders : concurrent orders in the kitchen (congestion proxy).
    congestion_scale / complexity_scale_divisor : scaling knobs from config.
    rng : a numpy.random.Generator for reproducible per-step noise.
    include_secondary_cook : force-include/exclude the optional step;
        if None, decided from `complexity`.
    """
    if include_secondary_cook is None:
        include_secondary_cook = complexity > 2.6

    graph = WorkflowGraph()
    congestion_factor = 1.0 + congestion_scale * active_orders
    complexity_factor = 1.0 + complexity / complexity_scale_divisor

    for step, base in _BASE_STEP_MINUTES.items():
        if step == "cook_secondary" and not include_secondary_cook:
            continue
        duration = base
        if step in _CONGESTION_SENSITIVE_STEPS:
            duration *= congestion_factor
        if step in _COMPLEXITY_SENSITIVE_STEPS:
            duration *= complexity_factor
        # Small per-step execution noise (kitchen staff aren't robots).
        duration = max(0.0, duration + rng.normal(0, 0.25 if base > 0 else 0))
        graph.add_node(step, duration)

    edges = [
        ("order_received", "ingredient_pull"),
        ("ingredient_pull", "prep_chop"),
        ("ingredient_pull", "marinate"),
        ("prep_chop", "cook_primary"),
        ("marinate", "cook_primary"),
    ]
    if include_secondary_cook:
        edges += [
            ("cook_primary", "cook_secondary"),
            ("cook_secondary", "plate"),
        ]
    else:
        edges += [("cook_primary", "plate")]
    edges += [
        ("plate", "quality_check"),
        ("quality_check", "handoff"),
    ]

    for u, v in edges:
        graph.add_edge(u, v)

    return graph


def _build_calibration_noise():
    """A model analyst never knows the true per-step base durations or
    scaling constants exactly -- they estimate them from historical
    data, with some error. This fixed (seeded, reproducible) per-step
    multiplicative miscalibration stands in for that estimation error,
    so the "predictive" workflow estimate below is deliberately NOT
    identical to the ground-truth generating function -- it is a
    plausible, imperfect model of it, the same way a real fitted model
    would be.
    """
    import numpy as np

    calib_rng = np.random.default_rng(7)  # fixed, independent of sim seed
    return {
        step: float(calib_rng.uniform(0.82, 1.18)) for step in _BASE_STEP_MINUTES
    }


_STEP_CALIBRATION_NOISE = _build_calibration_noise()
_CONGESTION_SCALE_CALIBRATION = 0.83  # analyst under-estimates congestion sensitivity
_COMPLEXITY_DIVISOR_CALIBRATION = 1.2  # analyst over-estimates complexity divisor


def estimate_workflow_duration(
    complexity: float,
    active_orders: int,
    congestion_scale: float,
    complexity_scale_divisor: float,
    estimation_noise_std: float = 1.3,
    noise_seed: int | None = None,
) -> float:
    """Imperfectly-calibrated critical-path estimate.

    Used as a *predictive* signal (fed into the Composite Readiness
    Score), built from the same DAG structure as the ground-truth
    simulator but with per-step base durations and scaling constants
    perturbed by a fixed calibration error -- standing in for the fact
    that a real workflow-timing model is fitted from historical
    averages, not handed the true generating parameters.

    On top of that fixed miscalibration, an independent per-order
    Gaussian estimation error (`estimation_noise_std`) is added. This
    matters: a *fixed* multiplicative bias can be perfectly undone by a
    downstream linear model, which would make this signal look
    unrealistically perfect. Genuine per-order noise cannot be
    regressed away, which keeps the resulting improvement over baseline
    defensible rather than an artifact of over-idealized simulation.
    """
    import numpy as np

    class _CalibratedRNG:
        def normal(self, *_args, **_kwargs):
            return 0.0

    graph = build_order_workflow(
        complexity=complexity,
        active_orders=active_orders,
        congestion_scale=congestion_scale * _CONGESTION_SCALE_CALIBRATION,
        complexity_scale_divisor=complexity_scale_divisor * _COMPLEXITY_DIVISOR_CALIBRATION,
        rng=_CalibratedRNG(),
    )
    for step in graph.nodes:
        graph.durations[step] *= _STEP_CALIBRATION_NOISE.get(step, 1.0)

    estimate = graph.critical_path_length()

    noise_rng = np.random.default_rng(noise_seed)
    estimate += noise_rng.normal(0, estimation_noise_std)

    return max(1.0, estimate)
