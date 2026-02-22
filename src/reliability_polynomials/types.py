"""Core types for reliability polynomial computation.

Enums, frozen dataclasses, and configuration objects used across the library.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


class TopologyKind(StrEnum):
    """Supervision topology type."""

    MESH = "mesh"
    HIERARCHY = "hierarchy"
    NONE = "none"
    CUSTOM = "custom"


class FaultType(StrEnum):
    """Fault model classification."""

    CRASH_STOP = "crash_stop"
    BYZANTINE = "byzantine"
    DEGRADATION = "degradation"
    INTERMITTENT = "intermittent"
    CORRELATED = "correlated"


class ConsensusProtocol(StrEnum):
    """Consensus protocol for Byzantine tolerance computation."""

    PBFT = "pbft"
    MAJORITY = "majority"
    GOSSIP = "gossip"


@dataclass(frozen=True)
class WeightVector:
    """Immutable quality weight vector for a topology.

    weights[k] = expected system quality when exactly k of n agents have failed.
    By convention, weights[0] is quality with no failures and weights[n] is
    quality when all agents have failed.
    """

    weights: tuple[float, ...]
    topology: TopologyKind
    n_agents: int

    def __post_init__(self) -> None:
        if len(self.weights) != self.n_agents + 1:
            msg = f"Expected {self.n_agents + 1} weights, got {len(self.weights)}"
            raise ValueError(msg)
        for i, w in enumerate(self.weights):
            if not 0.0 <= w <= 1.0:
                msg = f"Weight w_{i}={w} outside [0, 1]"
                raise ValueError(msg)

    def __getitem__(self, k: int) -> float:
        return self.weights[k]

    def __len__(self) -> int:
        return len(self.weights)


@dataclass
class TopologyConfig:
    """Configuration for topology weight computation.

    Args:
        n_agents: Number of agents in the system.
        delta_coord: Coordination overhead fraction (supervision/rerouting cost).
        delta_redist: Redistribution overhead fraction (load-balancing cost).
        gamma: Quality degradation model under concurrent load.
            "sqrt" for 1/sqrt(k), "constant" for 1.0, or a callable(k) -> float.
    """

    n_agents: int = 5
    delta_coord: float = 0.05
    delta_redist: float = 0.05
    gamma: str | Callable[[int], float] = "sqrt"

    def __post_init__(self) -> None:
        if self.n_agents < 1:
            msg = f"n_agents must be >= 1, got {self.n_agents}"
            raise ValueError(msg)
        if not 0.0 <= self.delta_coord <= 1.0:
            msg = f"delta_coord must be in [0, 1], got {self.delta_coord}"
            raise ValueError(msg)
        if not 0.0 <= self.delta_redist <= 1.0:
            msg = f"delta_redist must be in [0, 1], got {self.delta_redist}"
            raise ValueError(msg)
        if isinstance(self.gamma, str) and self.gamma not in ("sqrt", "constant"):
            msg = f"gamma must be 'sqrt', 'constant', or callable, got '{self.gamma}'"
            raise ValueError(msg)

    def gamma_fn(self, k: int) -> float:
        """Evaluate the gamma function at k concurrent failures."""
        if callable(self.gamma):
            return self.gamma(k)
        if self.gamma == "sqrt":
            return 1.0 / k**0.5 if k > 0 else 1.0
        # constant
        return 1.0


@dataclass
class FaultConfig:
    """Configuration for fault model crossover computation.

    Args:
        delta_coord: Coordination overhead (same as TopologyConfig).
        mu: Expected honest agent quality. Range (0, 1).
        mu_byz: Expected Byzantine agent quality. Range [0, mu).
        tau_d: Detection delay for Byzantine faults (observations).
        k_retries: Retry budget for intermittent faults.
        rho: Correlation coefficient for correlated faults. Range [0, 1].
        n_providers: Number of LLM providers (for correlated fault model).
    """

    delta_coord: float = 0.05
    mu: float = 0.65
    mu_byz: float = 0.50
    tau_d: int = 12
    k_retries: int = 3
    rho: float = 0.15
    n_providers: int = 3

    def __post_init__(self) -> None:
        if not 0.0 < self.delta_coord <= 1.0:
            msg = f"delta_coord must be in (0, 1], got {self.delta_coord}"
            raise ValueError(msg)
        if not 0.0 < self.mu <= 1.0:
            msg = f"mu must be in (0, 1], got {self.mu}"
            raise ValueError(msg)
        if not 0.0 <= self.mu_byz < self.mu:
            msg = f"mu_byz must be in [0, mu), got {self.mu_byz}"
            raise ValueError(msg)
        if self.tau_d < 0:
            msg = f"tau_d must be >= 0, got {self.tau_d}"
            raise ValueError(msg)
        if self.k_retries < 1:
            msg = f"k_retries must be >= 1, got {self.k_retries}"
            raise ValueError(msg)
        if not 0.0 <= self.rho <= 1.0:
            msg = f"rho must be in [0, 1], got {self.rho}"
            raise ValueError(msg)


@dataclass(frozen=True)
class CrossoverResult:
    """Result of a crossover analysis between topologies.

    The crossover point p* is the failure probability where mesh supervision
    begins outperforming hierarchy.
    """

    fault_type: FaultType
    p_star: float
    formula: str
    parameters: dict[str, float]


@dataclass(frozen=True)
class PolynomialAnalysis:
    """Analysis of a reliability polynomial at a given failure probability."""

    expected_quality: float
    derivative: float
    coefficients: tuple[float, ...]
    critical_failure_rate: float | None


@dataclass(frozen=True)
class ConnectivityResult:
    """Graph connectivity analysis result."""

    vertex_connectivity: int
    algebraic_connectivity: float
    crash_tolerance: int
    byzantine_tolerance: int
    is_connected: bool
