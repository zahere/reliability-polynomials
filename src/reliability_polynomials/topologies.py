"""Topology-dependent weight computation for reliability polynomials.

Computes quality weight vectors w_k for three supervision topologies:
- Mesh: fully connected peer supervision with work redistribution
- Hierarchy: hub-and-spoke with single coordinator
- None (unsupervised): independent agents, no recovery
"""

from __future__ import annotations

from reliability_polynomials.polynomial import ReliabilityPolynomial
from reliability_polynomials.types import TopologyConfig, TopologyKind, WeightVector


def mesh_weights(config: TopologyConfig) -> WeightVector:
    """Compute quality weights for mesh (peer supervision) topology.

    w_k = (1 - delta_coord) * [(n-k) + k * gamma(k) * (1 - delta_redist)] / n
    w_n = 0 (all agents failed)

    The mesh maintains quality through peer-to-peer work redistribution.
    When k agents fail, surviving agents absorb their work with gamma(k)
    degradation and delta_redist overhead.
    """
    n = config.n_agents
    weights: list[float] = []

    for k in range(n + 1):
        if k == n:
            weights.append(0.0)
            continue

        surviving = n - k
        gamma_k = config.gamma_fn(k) if k > 0 else 1.0
        redistributed = k * gamma_k * (1.0 - config.delta_redist)
        effective = (surviving + redistributed) / n
        w_k = (1.0 - config.delta_coord) * effective
        weights.append(w_k)

    return WeightVector(
        weights=tuple(weights),
        topology=TopologyKind.MESH,
        n_agents=n,
    )


def hierarchy_weights(config: TopologyConfig) -> WeightVector:
    """Compute quality weights for hierarchy (hub-and-spoke) topology.

    w_k = (n-k)/n * [(n-k) + k * gamma(k) * (1 - delta_redist)] / n

    Hierarchy has a single coordinator. If the coordinator fails (modeled as
    the (n-k)/n factor), the entire system loses coordination capability.
    """
    n = config.n_agents
    weights: list[float] = []

    for k in range(n + 1):
        if k == n:
            weights.append(0.0)
            continue

        surviving_frac = (n - k) / n
        gamma_k = config.gamma_fn(k) if k > 0 else 1.0
        redistributed = k * gamma_k * (1.0 - config.delta_redist)
        effective = ((n - k) + redistributed) / n
        w_k = surviving_frac * effective
        weights.append(w_k)

    return WeightVector(
        weights=tuple(weights),
        topology=TopologyKind.HIERARCHY,
        n_agents=n,
    )


def unsupervised_weights(config: TopologyConfig) -> WeightVector:
    """Compute quality weights for unsupervised (no mesh) topology.

    w_k = (n-k)/n

    With no supervision, failed agents' work is simply lost. System quality
    degrades linearly with the fraction of surviving agents.
    """
    n = config.n_agents
    weights = tuple((n - k) / n for k in range(n + 1))
    return WeightVector(
        weights=weights,
        topology=TopologyKind.NONE,
        n_agents=n,
    )


def custom_weights(weights: tuple[float, ...] | list[float]) -> WeightVector:
    """Create a weight vector from explicit values.

    Args:
        weights: Sequence of n+1 quality values [w_0, ..., w_n].
    """
    w = tuple(float(x) for x in weights)
    return WeightVector(
        weights=w,
        topology=TopologyKind.CUSTOM,
        n_agents=len(w) - 1,
    )


def build_polynomial(kind: TopologyKind, config: TopologyConfig) -> ReliabilityPolynomial:
    """Factory: build a reliability polynomial for the given topology.

    Args:
        kind: Topology type (MESH, HIERARCHY, NONE).
        config: Topology configuration.

    Returns:
        ReliabilityPolynomial with computed weights.
    """
    if kind == TopologyKind.MESH:
        return ReliabilityPolynomial(mesh_weights(config))
    if kind == TopologyKind.HIERARCHY:
        return ReliabilityPolynomial(hierarchy_weights(config))
    if kind == TopologyKind.NONE:
        return ReliabilityPolynomial(unsupervised_weights(config))
    msg = f"Cannot build polynomial for {kind}; use custom_weights() instead"
    raise ValueError(msg)


def compare_topologies(
    config: TopologyConfig,
    p_values: list[float] | None = None,
) -> dict[str, list[float]]:
    """Evaluate all three topologies across a range of failure probabilities.

    Args:
        config: Topology configuration.
        p_values: Failure probabilities to evaluate at. Defaults to
            [0.0, 0.05, 0.1, ..., 0.5].

    Returns:
        Dictionary mapping topology name to list of quality values.
    """
    if p_values is None:
        p_values = [i / 20.0 for i in range(11)]

    mesh_poly = build_polynomial(TopologyKind.MESH, config)
    hier_poly = build_polynomial(TopologyKind.HIERARCHY, config)
    none_poly = build_polynomial(TopologyKind.NONE, config)

    return {
        "p": p_values,
        "mesh": [mesh_poly.evaluate(p) for p in p_values],
        "hierarchy": [hier_poly.evaluate(p) for p in p_values],
        "none": [none_poly.evaluate(p) for p in p_values],
    }


def quality_table(config: TopologyConfig | None = None) -> dict[str, tuple[float, ...]]:
    """Reproduce the paper's Table B.1: weight vectors for each topology.

    Args:
        config: Optional configuration. Defaults to paper parameters
            (n=5, delta_coord=delta_redist=0.05, gamma=sqrt).

    Returns:
        Dictionary mapping topology name to weight tuple (w_0, ..., w_n).
    """
    if config is None:
        config = TopologyConfig(n_agents=5, delta_coord=0.05, delta_redist=0.05, gamma="sqrt")

    return {
        "mesh": mesh_weights(config).weights,
        "hierarchy": hierarchy_weights(config).weights,
        "none": unsupervised_weights(config).weights,
    }
