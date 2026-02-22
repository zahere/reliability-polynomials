"""Benchmark polynomial evaluation speed across different agent counts.

Measures evaluation time for n=5 to n=100 to verify numerical stability
and acceptable performance without numpy.
"""

from __future__ import annotations

import time

from reliability_polynomials import ReliabilityPolynomial, TopologyConfig, mesh_weights


def benchmark_evaluation(n: int, n_evals: int = 1000) -> float:
    """Benchmark polynomial evaluation for n agents.

    Returns average time per evaluation in microseconds.
    """
    config = TopologyConfig(n_agents=n, delta_coord=0.05, delta_redist=0.05, gamma="sqrt")
    poly = ReliabilityPolynomial(mesh_weights(config))

    p_values = [i / (n_evals - 1) for i in range(n_evals)]

    start = time.perf_counter()
    for p in p_values:
        poly.evaluate(p)
    elapsed = time.perf_counter() - start

    return elapsed / n_evals * 1e6  # microseconds


def benchmark_derivative(n: int, n_evals: int = 1000) -> float:
    """Benchmark derivative computation."""
    config = TopologyConfig(n_agents=n, delta_coord=0.05, delta_redist=0.05, gamma="sqrt")
    poly = ReliabilityPolynomial(mesh_weights(config))

    p_values = [i / (n_evals - 1) for i in range(n_evals)]

    start = time.perf_counter()
    for p in p_values:
        poly.derivative(p)
    elapsed = time.perf_counter() - start

    return elapsed / n_evals * 1e6


def benchmark_critical_rate(n: int, n_evals: int = 100) -> float:
    """Benchmark critical failure rate bisection."""
    config = TopologyConfig(n_agents=n, delta_coord=0.05, delta_redist=0.05, gamma="sqrt")
    poly = ReliabilityPolynomial(mesh_weights(config))

    start = time.perf_counter()
    for _ in range(n_evals):
        poly.critical_failure_rate(0.5)
    elapsed = time.perf_counter() - start

    return elapsed / n_evals * 1e6


def main() -> None:
    print("Reliability Polynomial Benchmarks")
    print("=" * 65)
    print(f"{'n':>5}  {'evaluate (us)':>14}  {'derivative (us)':>16}  {'critical (us)':>14}")
    print("-" * 65)

    for n in [5, 10, 20, 50, 100]:
        eval_us = benchmark_evaluation(n)
        deriv_us = benchmark_derivative(n)
        crit_us = benchmark_critical_rate(n)
        print(f"{n:5d}  {eval_us:14.1f}  {deriv_us:16.1f}  {crit_us:14.1f}")

    # Verify numerical stability at n=100
    print("\nNumerical stability check (n=100):")
    config = TopologyConfig(n_agents=100, delta_coord=0.05, delta_redist=0.05, gamma="sqrt")
    poly = ReliabilityPolynomial(mesh_weights(config))
    for p in [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        q = poly.evaluate(p)
        print(f"  p={p:.2f}: quality={q:.6f}")


if __name__ == "__main__":
    main()
