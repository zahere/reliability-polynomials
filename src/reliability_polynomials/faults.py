"""Fault model analysis and crossover computation.

Implements the three theorems from the SSM paper:
- Theorem 1: Coordination transparency classification
- Theorem 2: Fault-dependent crossover formulas
- Theorem 3: Effective connectivity under different fault types
"""

from __future__ import annotations

import math

from reliability_polynomials.types import (
    ConsensusProtocol,
    CrossoverResult,
    FaultConfig,
    FaultType,
)

# ---------------------------------------------------------------------------
# Theorem 2: Crossover formulas
# ---------------------------------------------------------------------------


def crash_stop_crossover(config: FaultConfig) -> CrossoverResult:
    """Compute crossover for crash-stop faults.

    p* ≈ delta_coord

    Crash-stop agents produce zero output. Mesh pays coordination overhead
    but recovers work; hierarchy loses the coordinator with probability p.
    Mesh dominates when p exceeds the coordination cost.
    """
    p_star = config.delta_coord
    return CrossoverResult(
        fault_type=FaultType.CRASH_STOP,
        p_star=p_star,
        formula="p* = delta_coord",
        parameters={"delta_coord": config.delta_coord},
    )


def byzantine_crossover(config: FaultConfig) -> CrossoverResult:
    """Compute crossover for Byzantine faults.

    p* = delta_coord / (alpha * (2 + tau_d))

    where alpha = (mu - mu_byz) / mu

    Byzantine agents produce plausible but incorrect output. The detection
    delay tau_d means corrupted output propagates before detection, making
    the crossover much earlier than crash-stop.
    """
    alpha = (config.mu - config.mu_byz) / config.mu
    p_star = config.delta_coord / (alpha * (2 + config.tau_d))
    return CrossoverResult(
        fault_type=FaultType.BYZANTINE,
        p_star=p_star,
        formula="p* = delta_coord / (alpha * (2 + tau_d))",
        parameters={
            "delta_coord": config.delta_coord,
            "alpha": alpha,
            "tau_d": float(config.tau_d),
            "mu": config.mu,
            "mu_byz": config.mu_byz,
        },
    )


def degradation_crossover(config: FaultConfig) -> CrossoverResult:
    """Compute crossover for degradation faults.

    p* = infinity (no crossover)

    Degraded agents produce lower-quality but nonzero output. Both topologies
    suffer equally from quality reduction, but mesh always pays coordination
    overhead. Hierarchy dominates for pure degradation.
    """
    return CrossoverResult(
        fault_type=FaultType.DEGRADATION,
        p_star=math.inf,
        formula="p* = infinity (mesh always pays overhead)",
        parameters={"delta_coord": config.delta_coord},
    )


def intermittent_crossover(config: FaultConfig) -> CrossoverResult:
    """Compute crossover for intermittent faults.

    p* = delta_coord^(1/k)

    Intermittent agents fail transiently. With k retries, the effective
    failure probability is p^k, pushing the crossover much higher than
    crash-stop.
    """
    k = config.k_retries
    p_star = config.delta_coord ** (1.0 / k)
    return CrossoverResult(
        fault_type=FaultType.INTERMITTENT,
        p_star=p_star,
        formula="p* = delta_coord^(1/k)",
        parameters={
            "delta_coord": config.delta_coord,
            "k_retries": float(k),
        },
    )


def correlated_crossover(config: FaultConfig) -> CrossoverResult:
    """Compute crossover for correlated faults.

    p* = p*_crash + rho * (1 - p*_crash) * delta_coord

    Correlated failures (e.g., shared LLM provider outage) shift the
    crossover slightly above crash-stop due to increased effective failure
    probability.
    """
    p_crash = config.delta_coord
    p_star = p_crash + config.rho * (1.0 - p_crash) * config.delta_coord
    return CrossoverResult(
        fault_type=FaultType.CORRELATED,
        p_star=p_star,
        formula="p* = p*_crash + rho * (1 - p*_crash) * delta_coord",
        parameters={
            "delta_coord": config.delta_coord,
            "rho": config.rho,
            "p_crash": p_crash,
        },
    )


def all_crossovers(config: FaultConfig | None = None) -> list[CrossoverResult]:
    """Compute crossovers for all five fault types.

    Returns results ordered by p*:
    p*_byz < p*_crash < p*_corr < p*_int < p*_deg = infinity

    Args:
        config: Fault configuration. Defaults to paper parameters.
    """
    if config is None:
        config = FaultConfig()

    results = [
        byzantine_crossover(config),
        crash_stop_crossover(config),
        correlated_crossover(config),
        intermittent_crossover(config),
        degradation_crossover(config),
    ]
    return results


# ---------------------------------------------------------------------------
# Theorem 1: Coordination transparency
# ---------------------------------------------------------------------------


def is_coordination_transparent(fault_type: FaultType) -> bool:
    """Determine if a fault type is coordination-transparent.

    A fault type is coordination-transparent if the supervision mechanism
    does not depend on fault-specific detection. Crash-stop, degradation,
    and intermittent (per-round) faults are transparent; Byzantine and
    correlated faults require specialized detection.

    Args:
        fault_type: The fault type to classify.

    Returns:
        True if the fault type is coordination-transparent.
    """
    transparent = {
        FaultType.CRASH_STOP,
        FaultType.DEGRADATION,
        FaultType.INTERMITTENT,
    }
    return fault_type in transparent


def fault_classification() -> list[dict[str, str | bool]]:
    """Return the full fault classification table (Corollary B.1).

    Returns:
        List of dicts with fault_type, transparent, detection, mechanism,
        and crossover_behavior.
    """
    return [
        {
            "fault_type": FaultType.CRASH_STOP,
            "transparent": True,
            "detection": "Heartbeat / timeout",
            "mechanism": "Work redistribution",
            "crossover_behavior": "p* = delta_coord",
        },
        {
            "fault_type": FaultType.BYZANTINE,
            "transparent": False,
            "detection": "CUSUM / output validation",
            "mechanism": "Quarantine + rerouteing",
            "crossover_behavior": "p* = delta_coord / (alpha * (2 + tau_d))",
        },
        {
            "fault_type": FaultType.DEGRADATION,
            "transparent": True,
            "detection": "Quality monitoring",
            "mechanism": "Graceful load shedding",
            "crossover_behavior": "No crossover (p* = infinity)",
        },
        {
            "fault_type": FaultType.INTERMITTENT,
            "transparent": True,
            "detection": "Retry tracking",
            "mechanism": "Exponential retry",
            "crossover_behavior": "p* = delta_coord^(1/k)",
        },
        {
            "fault_type": FaultType.CORRELATED,
            "transparent": False,
            "detection": "Provider health correlation",
            "mechanism": "Provider-aware rerouting",
            "crossover_behavior": "p* > p*_crash (correlation-adjusted)",
        },
    ]


# ---------------------------------------------------------------------------
# Theorem 3: Effective connectivity
# ---------------------------------------------------------------------------


def effective_connectivity(
    fault_type: FaultType,
    vertex_connectivity: int,
    n_agents: int,
    protocol: ConsensusProtocol = ConsensusProtocol.PBFT,
) -> int:
    """Compute effective fault tolerance under a given fault type.

    - Crash-stop: tolerance = vertex connectivity kappa(G)
    - Byzantine: tolerance = f(protocol), independent of graph
    - Degradation: tolerance = n (all agents can degrade without disconnection)

    Args:
        fault_type: The fault model.
        vertex_connectivity: Graph vertex connectivity kappa(G).
        n_agents: Total number of agents.
        protocol: Consensus protocol (for Byzantine tolerance).

    Returns:
        Maximum number of simultaneous faults the system can tolerate.
    """
    if fault_type == FaultType.CRASH_STOP:
        return vertex_connectivity

    if fault_type == FaultType.BYZANTINE:
        return byzantine_tolerance(n_agents, protocol)

    if fault_type == FaultType.DEGRADATION:
        return n_agents

    if fault_type == FaultType.INTERMITTENT:
        return vertex_connectivity

    if fault_type == FaultType.CORRELATED:
        return max(0, vertex_connectivity - 1)

    return vertex_connectivity


def byzantine_tolerance(
    n_agents: int,
    protocol: ConsensusProtocol = ConsensusProtocol.PBFT,
) -> int:
    """Compute Byzantine fault tolerance for a given protocol.

    - PBFT: floor((n-1)/3)
    - Majority vote: floor((n-1)/2)
    - Gossip: 0 (no Byzantine resilience)

    Args:
        n_agents: Total number of agents.
        protocol: Consensus protocol.

    Returns:
        Maximum number of Byzantine agents tolerated.
    """
    if protocol == ConsensusProtocol.PBFT:
        return (n_agents - 1) // 3
    if protocol == ConsensusProtocol.MAJORITY:
        return (n_agents - 1) // 2
    if protocol == ConsensusProtocol.GOSSIP:
        return 0
    return 0
