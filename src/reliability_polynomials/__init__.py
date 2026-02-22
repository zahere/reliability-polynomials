"""Reliability Polynomials — generalized network reliability for quality-weighted systems.

Every network reliability library assumes binary survival: a component either works
or it doesn't. In LLM multi-agent systems, failure is NOT binary — a "failed" agent
might produce low-quality but not zero output. This library computes generalized
reliability polynomials where coefficients encode how much quality survives.

Quick start::

    from reliability_polynomials import ReliabilityPolynomial, mesh_weights, TopologyConfig

    config = TopologyConfig(n_agents=5, delta_coord=0.05, delta_redist=0.05)
    poly = ReliabilityPolynomial(mesh_weights(config))
    print(poly.evaluate(0.1))  # Expected quality at 10% failure rate
"""

from reliability_polynomials.faults import (
    all_crossovers,
    byzantine_crossover,
    byzantine_tolerance,
    correlated_crossover,
    crash_stop_crossover,
    degradation_crossover,
    effective_connectivity,
    fault_classification,
    intermittent_crossover,
    is_coordination_transparent,
)
from reliability_polynomials.graphs import (
    Graph,
    complete_graph,
    grid_graph,
    ring_graph,
    star_graph,
)
from reliability_polynomials.polynomial import ReliabilityPolynomial
from reliability_polynomials.topologies import (
    build_polynomial,
    compare_topologies,
    custom_weights,
    hierarchy_weights,
    mesh_weights,
    quality_table,
    unsupervised_weights,
)
from reliability_polynomials.types import (
    ConnectivityResult,
    ConsensusProtocol,
    CrossoverResult,
    FaultConfig,
    FaultType,
    PolynomialAnalysis,
    TopologyConfig,
    TopologyKind,
    WeightVector,
)

__all__ = [
    # Core
    "ReliabilityPolynomial",
    # Graph
    "Graph",
    "complete_graph",
    "ring_graph",
    "star_graph",
    "grid_graph",
    # Topologies
    "mesh_weights",
    "hierarchy_weights",
    "unsupervised_weights",
    "custom_weights",
    "build_polynomial",
    "compare_topologies",
    "quality_table",
    # Faults
    "crash_stop_crossover",
    "byzantine_crossover",
    "degradation_crossover",
    "intermittent_crossover",
    "correlated_crossover",
    "all_crossovers",
    "is_coordination_transparent",
    "fault_classification",
    "effective_connectivity",
    "byzantine_tolerance",
    # Types
    "TopologyKind",
    "FaultType",
    "ConsensusProtocol",
    "WeightVector",
    "TopologyConfig",
    "FaultConfig",
    "CrossoverResult",
    "PolynomialAnalysis",
    "ConnectivityResult",
]

__version__ = "0.1.0"
