import numpy as np
import pytest

from kpt2.workflow_graph import (
    GraphCycleError,
    WorkflowGraph,
    build_order_workflow,
    estimate_workflow_duration,
)


def test_topological_order_respects_dependencies():
    g = WorkflowGraph()
    for name in ["a", "b", "c", "d"]:
        g.add_node(name, duration=1.0)
    g.add_edge("a", "b")
    g.add_edge("a", "c")
    g.add_edge("b", "d")
    g.add_edge("c", "d")

    order = g.topological_order()
    positions = {n: i for i, n in enumerate(order)}

    assert positions["a"] < positions["b"]
    assert positions["a"] < positions["c"]
    assert positions["b"] < positions["d"]
    assert positions["c"] < positions["d"]


def test_critical_path_takes_longest_not_sum():
    g = WorkflowGraph()
    g.add_node("start", 0.0)
    g.add_node("short_branch", 1.0)
    g.add_node("long_branch", 5.0)
    g.add_node("end", 0.0)
    g.add_edge("start", "short_branch")
    g.add_edge("start", "long_branch")
    g.add_edge("short_branch", "end")
    g.add_edge("long_branch", "end")

    # Critical path = start(0) -> long_branch(5) -> end(0) = 5, NOT the
    # sum of every node (which would incorrectly be 6).
    assert g.critical_path_length() == pytest.approx(5.0)


def test_cycle_raises():
    g = WorkflowGraph()
    g.add_node("a", 1.0)
    g.add_node("b", 1.0)
    g.add_edge("a", "b")
    g.add_edge("b", "a")

    with pytest.raises(GraphCycleError):
        g.topological_order()


def test_build_order_workflow_is_a_valid_dag():
    rng = np.random.default_rng(1)
    for complexity in [1.0, 2.5, 3.5, 5.0]:
        graph = build_order_workflow(
            complexity=complexity,
            active_orders=3,
            congestion_scale=0.045,
            complexity_scale_divisor=5.0,
            rng=rng,
        )
        # Should not raise (i.e. must be acyclic).
        order = graph.topological_order()
        assert "order_received" == order[0]
        assert "handoff" == order[-1]
        assert graph.critical_path_length() > 0


def test_higher_congestion_increases_critical_path():
    low = estimate_workflow_duration(
        complexity=3.0,
        active_orders=0,
        congestion_scale=0.045,
        complexity_scale_divisor=5.0,
        estimation_noise_std=0.0,
        noise_seed=1,
    )
    high = estimate_workflow_duration(
        complexity=3.0,
        active_orders=15,
        congestion_scale=0.045,
        complexity_scale_divisor=5.0,
        estimation_noise_std=0.0,
        noise_seed=1,
    )
    assert high > low
