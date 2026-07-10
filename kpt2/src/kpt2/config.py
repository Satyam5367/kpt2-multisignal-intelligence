"""
Central configuration for the KPT2 simulation and inference pipeline.

Keeping every tunable constant in one place makes the causal simulation
auditable: every number that affects "ground truth" generation vs.
"observed signal" generation is declared here, not buried in function
bodies.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    num_orders: int = 3000
    num_merchants: int = 60
    random_seed: int = 42

    # Merchant behavioral bias: fixed per-merchant offset (minutes) between
    # when a merchant marks "Food Order Ready" (FOR) and when the rider
    # actually arrives. Drawn once per merchant.
    merchant_bias_mean: float = 2.0
    merchant_bias_std: float = 1.5

    # Rush-hour marking distortion (minutes per unit of active_orders).
    # This is the part of the noise that a simple per-merchant mean
    # correction CANNOT remove, because it varies order-to-order with
    # kitchen congestion rather than being a fixed merchant constant.
    rush_distortion_per_active_order: float = -0.18

    # Residual random marking noise (minutes) left after all systematic
    # effects above. No signal can recover this component; it sets a
    # floor on achievable MAE.
    marking_noise_std: float = 1.1

    # Rider arrival noise around the true ready time (minutes).
    rider_arrival_noise_mean: float = 3.0
    rider_arrival_noise_std: float = 2.0

    # Congestion scaling applied to workflow step durations.
    congestion_scale_per_active_order: float = 0.045

    # Complexity scaling applied to prep-heavy workflow steps.
    complexity_scale_divisor: float = 5.0

    # Nominal (uncongested) duration used ONLY for the heap-based interval
    # overlap sweep that estimates active_orders per merchant. This is a
    # deliberate simplification: real concurrency is estimated from a
    # nominal service time rather than solved circularly.
    nominal_service_minutes: float = 18.0

    # Total simulated time span (minutes) that off-peak orders are
    # spread across.
    simulation_span_minutes: float = 1000.0

    # Lunch/dinner rush windows as (start_minute, width_minutes) within
    # the simulation span. A configurable fraction of orders (below)
    # land inside one of these short, high-density windows instead of
    # being spread uniformly, which is what produces realistic kitchen
    # concurrency (active_orders) instead of near-constant near-zero
    # overlap.
    rush_windows_minutes: tuple = ((150.0, 90.0), (550.0, 90.0), (850.0, 60.0))
    rush_order_fraction: float = 0.55

    # Sliding-window size (minutes) used for Rider Wait Cluster Detection.
    rider_cluster_window_minutes: float = 6.0

    # Rolling history window (# past orders per merchant) used for the
    # historical_pattern signal.
    history_window_size: int = 20

    # Drift monitoring threshold: merchants whose per-order offset std
    # exceeds this many minutes are flagged as "unstable".
    drift_std_threshold: float = 3.0

    # Irreducible per-order noise (minutes) on the workflow-graph
    # duration estimate, representing real model estimation error that
    # no downstream regression can remove.
    workflow_estimate_noise_std: float = 2.3


DEFAULT_CONFIG = SimulationConfig()
