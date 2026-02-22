"""Crossover analysis for all five fault types.

Demonstrates the complete crossover ordering from the SSM paper:
p*_byz < p*_crash < p*_corr < p*_int < p*_deg = infinity
"""

from reliability_polynomials import (
    FaultConfig,
    FaultType,
    all_crossovers,
    fault_classification,
    is_coordination_transparent,
)


def main() -> None:
    config = FaultConfig(
        delta_coord=0.05,
        mu=0.65,
        mu_byz=0.50,
        tau_d=12,
        k_retries=3,
        rho=0.15,
    )

    # All crossovers in order
    print("Fault-Type Crossover Ordering (Theorem 2)")
    print("=" * 70)
    results = all_crossovers(config)

    print(f"{'Fault Type':<16} {'p*':>10} {'Transparent':>12}  {'Formula'}")
    print("-" * 70)
    for r in results:
        transparent = is_coordination_transparent(r.fault_type)
        p_str = f"{r.p_star:.4f}" if r.p_star < float("inf") else "infinity"
        t_str = "yes" if transparent else "no"
        print(f"{r.fault_type.value:<16} {p_str:>10} {t_str:>12}  {r.formula}")

    # Verify ordering
    print("\nOrdering verification:")
    for i in range(len(results) - 1):
        a, b = results[i], results[i + 1]
        symbol = "<" if a.p_star < b.p_star else ">="
        print(
            f"  p*_{a.fault_type.value} ({a.p_star:.4f}) {symbol} "
            f"p*_{b.fault_type.value} ({b.p_star if b.p_star < float('inf') else 'inf'})"
        )

    # Full classification table
    print("\n\nFault Classification Table (Corollary B.1)")
    print("=" * 80)
    table = fault_classification()
    print(f"{'Type':<14} {'Transparent':<12} {'Detection':<25} {'Mechanism'}")
    print("-" * 80)
    for row in table:
        print(
            f"{str(row['fault_type']):<14} "
            f"{'yes' if row['transparent'] else 'no':<12} "
            f"{row['detection']:<25} "
            f"{row['mechanism']}"
        )

    # Practical interpretation
    print("\n\nPractical Interpretation")
    print("=" * 60)
    print(f"With delta_coord={config.delta_coord} (5% supervision overhead):")
    print()
    for r in results:
        ft = r.fault_type
        if ft == FaultType.BYZANTINE:
            print(f"  Byzantine: Mesh needed as soon as p > {r.p_star:.4f}")
            print(f"    (detection delay tau_d={config.tau_d} makes this urgent)")
        elif ft == FaultType.CRASH_STOP:
            print(f"  Crash-stop: Mesh beneficial when p > {r.p_star:.4f}")
        elif ft == FaultType.CORRELATED:
            print(f"  Correlated: Provider-shared failures shift to p > {r.p_star:.4f}")
        elif ft == FaultType.INTERMITTENT:
            retries = config.k_retries
            print(f"  Intermittent: {retries} retries push threshold to p > {r.p_star:.4f}")
        elif ft == FaultType.DEGRADATION:
            print("  Degradation: Mesh never beneficial (always pays overhead)")


if __name__ == "__main__":
    main()
