"""Basic usage of reliability polynomials.

Demonstrates building a polynomial, evaluating quality at different failure
rates, and finding the critical failure rate.
"""

from reliability_polynomials import (
    ReliabilityPolynomial,
    TopologyConfig,
    mesh_weights,
    unsupervised_weights,
)


def main() -> None:
    # Build a mesh reliability polynomial (n=5 agents)
    config = TopologyConfig(n_agents=5, delta_coord=0.05, delta_redist=0.05, gamma="sqrt")
    poly = ReliabilityPolynomial(mesh_weights(config))

    print("Mesh Reliability Polynomial (n=5, gamma=sqrt)")
    print("=" * 50)
    print(f"Weights: {poly.weights.weights}")
    print()

    # Evaluate at various failure rates
    print(f"{'p':>6}  {'Quality':>8}  {'dQ/dp':>8}")
    print("-" * 28)
    for p_int in range(0, 55, 5):
        p = p_int / 100.0
        q = poly.evaluate(p)
        d = poly.derivative(p)
        print(f"{p:6.2f}  {q:8.4f}  {d:8.4f}")

    # Find critical failure rate
    threshold = 0.7
    p_star = poly.critical_failure_rate(threshold)
    print(f"\nCritical failure rate (quality < {threshold}): ", end="")
    if p_star is not None:
        print(f"p* = {p_star:.4f}")
    else:
        print("never (quality always above threshold)")

    # Compare with unsupervised
    none_poly = ReliabilityPolynomial(unsupervised_weights(config))
    crossover = poly.crossover(none_poly)
    print(f"\nCrossover vs unsupervised: p* = {crossover:.4f}" if crossover else "No crossover")

    # Full analysis
    analysis = poly.analyze(p=0.1, threshold=0.5)
    print("\nAnalysis at p=0.1:")
    print(f"  Expected quality: {analysis.expected_quality:.4f}")
    print(f"  Quality derivative: {analysis.derivative:.4f}")
    print(f"  Critical failure rate (50%): {analysis.critical_failure_rate}")


if __name__ == "__main__":
    main()
