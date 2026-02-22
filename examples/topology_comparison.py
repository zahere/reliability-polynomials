"""Compare topology models across failure probabilities.

Reproduces the paper's quality table and shows how mesh, hierarchy, and
unsupervised topologies degrade differently under increasing failure rates.
"""

from reliability_polynomials import (
    TopologyConfig,
    TopologyKind,
    build_polynomial,
    compare_topologies,
    quality_table,
)


def main() -> None:
    config = TopologyConfig(n_agents=5, delta_coord=0.05, delta_redist=0.05, gamma="sqrt")

    # Table B.1: Weight vectors
    print("Quality Weight Table (Paper Table B.1)")
    print("=" * 60)
    table = quality_table(config)
    header = f"{'Topology':<12}" + "".join(f"{'w_' + str(k):>8}" for k in range(6))
    print(header)
    print("-" * 60)
    for name, weights in table.items():
        row = f"{name:<12}" + "".join(f"{w:8.3f}" for w in weights)
        print(row)

    # Compare across failure rates
    print("\n\nExpected Quality vs Failure Probability")
    print("=" * 60)
    p_values = [i / 20.0 for i in range(11)]
    comparison = compare_topologies(config, p_values)
    header = f"{'p':>6}" + f"{'Mesh':>10}" + f"{'Hierarchy':>10}" + f"{'None':>10}"
    print(header)
    print("-" * 36)
    for i, p in enumerate(comparison["p"]):
        print(
            f"{p:6.2f}"
            f"{comparison['mesh'][i]:10.4f}"
            f"{comparison['hierarchy'][i]:10.4f}"
            f"{comparison['none'][i]:10.4f}"
        )

    # Find crossover between mesh and hierarchy
    mesh_poly = build_polynomial(TopologyKind.MESH, config)
    hier_poly = build_polynomial(TopologyKind.HIERARCHY, config)
    crossover = mesh_poly.crossover(hier_poly)
    if crossover is not None:
        print(f"\nMesh-Hierarchy crossover at p* = {crossover:.4f}")
        print(f"  Mesh quality:      {mesh_poly.evaluate(crossover):.4f}")
        print(f"  Hierarchy quality:  {hier_poly.evaluate(crossover):.4f}")

    # Dominance check
    print(f"\nMesh dominates hierarchy for p > {crossover:.4f}:")
    print(f"  At p=0.3: mesh={mesh_poly.evaluate(0.3):.4f}, hier={hier_poly.evaluate(0.3):.4f}")


if __name__ == "__main__":
    main()
