# Reliability Polynomials

[![CI](https://github.com/zahere/reliability-polynomials/actions/workflows/ci.yml/badge.svg)](https://github.com/zahere/reliability-polynomials/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: BSD-3](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](LICENSE)

**Generalized reliability polynomials for quality-weighted network analysis.**

Every network reliability library assumes binary survival: a component either works or it doesn't. In LLM multi-agent systems, failure is **not binary** — a "failed" agent might produce low-quality but not zero output, a degraded model might still return plausible answers, and a Byzantine agent might inject subtly wrong results that look correct. This library computes generalized reliability polynomials where coefficients encode *how much quality survives*, not just whether the network is connected.

## Why Binary Reliability Fails for LLM Agents

| Property | Classical Reliability | This Library |
|----------|----------------------|--------------|
| Failure model | Binary (works / broken) | Quality-weighted (0.0 to 1.0) |
| Coefficient meaning | Probability of connectivity | Expected system quality |
| Agent output on failure | Zero (no output) | Degraded, Byzantine, or intermittent |
| Supervision overhead | Not modeled | Explicit (delta_coord, delta_redist) |
| Topology comparison | Same polynomial | Different weight vectors per topology |

## Quick Start

```python
from reliability_polynomials import ReliabilityPolynomial, TopologyConfig, mesh_weights

config = TopologyConfig(n_agents=5, delta_coord=0.05, delta_redist=0.05)
poly = ReliabilityPolynomial(mesh_weights(config))
print(poly.evaluate(0.1))  # 0.9355 — expected quality at 10% failure rate
```

## The Generalized Reliability Polynomial

The core mathematical object:

```
Res_τ(p) = Σ C(n,k) · p^k · (1-p)^{n-k} · w_k(τ)
```

where `p` is the per-agent failure probability, `n` is the number of agents, and `w_k(τ)` is the expected system quality when exactly `k` agents have failed under topology `τ`. The classical reliability polynomial is the special case where `w_k ∈ {0, 1}`.

### Quality Weight Table (n=5, δ=0.05)

| Topology | w₀ | w₁ | w₂ | w₃ | w₄ | w₅ |
|----------|------|------|------|------|------|------|
| Mesh | 0.950 | 0.941 | 0.825 | 0.693 | 0.551 | 0.000 |
| Hierarchy | 1.000 | 0.792 | 0.521 | 0.292 | 0.116 | 0.000 |
| Unsupervised | 1.000 | 0.800 | 0.600 | 0.400 | 0.200 | 0.000 |

Mesh pays 5% coordination overhead at `p=0` but retains 55.1% quality even with 4 of 5 agents failed. Hierarchy collapses to 11.6%.

```python
from reliability_polynomials import quality_table

table = quality_table()  # Reproduces the table above
```

## Three Theorems

### Theorem 1: Coordination Transparency

Some fault types can be handled by the same supervision mechanism regardless of topology; others require specialized detection.

```python
from reliability_polynomials import FaultType, is_coordination_transparent

is_coordination_transparent(FaultType.CRASH_STOP)    # True — heartbeat suffices
is_coordination_transparent(FaultType.BYZANTINE)      # False — needs CUSUM/validation
is_coordination_transparent(FaultType.DEGRADATION)    # True — quality monitoring
```

### Theorem 2: Fault-Dependent Crossover

The failure probability `p*` where mesh supervision begins outperforming hierarchy depends on the fault type:

```
p*_byzantine < p*_crash < p*_correlated < p*_intermittent < p*_degradation = ∞
```

```python
from reliability_polynomials import FaultConfig, all_crossovers

results = all_crossovers(FaultConfig(delta_coord=0.05, mu=0.65, tau_d=12))
for r in results:
    print(f"{r.fault_type.value:>14}: p* = {r.p_star:.4f}")
#      byzantine: p* = 0.0155
#     crash_stop: p* = 0.0500
#     correlated: p* = 0.0571
#   intermittent: p* = 0.3684
#    degradation: p* = inf
```

### Theorem 3: Effective Connectivity

Graph connectivity means different things under different fault models. A complete graph K₅ has vertex connectivity 4, but with gossip consensus it tolerates **zero** Byzantine agents.

```python
from reliability_polynomials import complete_graph, ConsensusProtocol

g = complete_graph(5)
result = g.analyze(ConsensusProtocol.PBFT)
print(result.vertex_connectivity)    # 4
print(result.byzantine_tolerance)    # 1 (PBFT: floor((n-1)/3))
print(result.crash_tolerance)        # 2

result = g.analyze(ConsensusProtocol.GOSSIP)
print(result.byzantine_tolerance)    # 0 (gossip has no Byzantine resilience)
```

## Topology Models

Three supervision topologies with configurable parameters:

```python
from reliability_polynomials import (
    TopologyConfig, TopologyKind, build_polynomial, compare_topologies
)

config = TopologyConfig(n_agents=5, delta_coord=0.05, delta_redist=0.05, gamma="sqrt")

mesh = build_polynomial(TopologyKind.MESH, config)
hier = build_polynomial(TopologyKind.HIERARCHY, config)
none = build_polynomial(TopologyKind.NONE, config)

# Find where mesh overtakes hierarchy
crossover = mesh.crossover(hier)
print(f"Mesh dominates hierarchy for p > {crossover:.4f}")

# Compare all three across failure rates
comparison = compare_topologies(config)
```

The `gamma` parameter models quality degradation under concurrent load:
- `"sqrt"`: `γ(k) = 1/√k` — coupled tasks with minimal context
- `"constant"`: `γ(k) = 1` — independent tasks or full context sharing
- Any `callable(k) -> float` for custom models

## Graph Analysis

Lightweight graph representation with connectivity and spectral analysis:

```python
from reliability_polynomials import complete_graph, ring_graph, star_graph, grid_graph

# Factory functions
k5 = complete_graph(5)    # K₅: κ=4, λ₂=5
ring = ring_graph(10)     # C₁₀: κ=2, λ₂≈0.382
star = star_graph(5)      # S₅: κ=1 (hub is single point of failure)
grid = grid_graph(3, 3)   # 3×3 lattice

# Full analysis
result = k5.analyze()
print(f"Vertex connectivity: {result.vertex_connectivity}")
print(f"Algebraic connectivity: {result.algebraic_connectivity:.4f}")
print(f"Crash tolerance: {result.crash_tolerance}")
print(f"Byzantine tolerance (PBFT): {result.byzantine_tolerance}")
```

## Installation

```bash
pip install git+https://github.com/zahere/reliability-polynomials.git
```

**Zero dependencies.** Only uses the Python standard library (`math`, `dataclasses`, `enum`, `itertools`).

## Benchmarks

```bash
python benchmarks/polynomial_evaluation.py
```

Evaluation is O(n) per point using log-space Bernstein basis computation, stable up to n=100+ without overflow.

## Origin

This library implements the core theoretical framework from
"When Does Topology Matter? Fault-Dependent Resilience in Multi-Agent
LLM Systems" (in preparation) — developed during research that grew
out of production work on [AgentiCraft](https://agenticraft.ai),
an enterprise multi-agent platform.

The framework emerged from a recurring production problem: existing
multi-agent frameworks provide no principled basis for topology
selection. This library gives you the analytical tools to make those
decisions rigorously, based on the fault model your system actually
faces rather than convention or intuition.

A companion library implementing the statistical circuit breaker
from the same research is available at
[stochastic-circuit-breaker](https://github.com/zahere/stochastic-circuit-breaker).

## Author

**Zaher Khateeb** — AI/ML Engineer, Founder of [AgentiCraft](https://agenticraft.ai)

Research focus: fault-dependent resilience in multi-agent LLM systems,
stochastic service mesh architecture, formal verification for
distributed agent coordination.

[linkedin.com/in/zahere](https://www.linkedin.com/in/zahere/) ·
[agenticraft.ai](https://agenticraft.ai)

## References

- Colbourn, C. J. (1987). *The Combinatorics of Network Reliability*. Oxford University Press.
- Moore, E. F., & Shannon, C. E. (1956). Reliable circuits using less reliable relays. *Journal of the Franklin Institute*, 262(3), 191-208.
- Bernstein, S. N. (1912). Proof of the theorem of Weierstrass based on the calculus of probabilities. *Communications of the Kharkov Mathematical Society*, 13, 1-2.
- Castro, M., & Liskov, B. (1999). Practical Byzantine fault tolerance. *OSDI*.
- Page, E. S. (1954). Continuous inspection schemes. *Biometrika*, 41(1/2), 100-115.

## License

[BSD-3-Clause](LICENSE)
