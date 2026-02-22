"""Tests for ReliabilityPolynomial core object."""

from __future__ import annotations

import math

import pytest

from reliability_polynomials.polynomial import ReliabilityPolynomial
from reliability_polynomials.types import TopologyKind, WeightVector


class TestConstruction:
    """Test polynomial construction from various inputs."""

    def test_from_weight_vector(self) -> None:
        wv = WeightVector(weights=(1.0, 0.8, 0.5, 0.0), topology=TopologyKind.MESH, n_agents=3)
        poly = ReliabilityPolynomial(wv)
        assert poly.n == 3
        assert poly.weights is wv

    def test_from_sequence(self) -> None:
        poly = ReliabilityPolynomial([1.0, 0.8, 0.5, 0.0])
        assert poly.n == 3
        assert poly.weights.topology == TopologyKind.CUSTOM

    def test_from_tuple(self) -> None:
        poly = ReliabilityPolynomial((1.0, 0.0))
        assert poly.n == 1

    def test_single_agent(self) -> None:
        poly = ReliabilityPolynomial([1.0, 0.0])
        assert poly.n == 1
        assert poly.degree == 1

    def test_weight_validation_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            WeightVector(weights=(1.0, 1.5, 0.0), topology=TopologyKind.CUSTOM, n_agents=2)

    def test_weight_validation_rejects_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="Expected"):
            WeightVector(weights=(1.0, 0.5), topology=TopologyKind.CUSTOM, n_agents=3)


class TestEvaluation:
    """Test polynomial evaluation at various failure probabilities."""

    def test_p_zero_returns_w0(self) -> None:
        poly = ReliabilityPolynomial([0.95, 0.8, 0.5, 0.2, 0.0])
        assert poly.evaluate(0.0) == pytest.approx(0.95)

    def test_p_one_returns_wn(self) -> None:
        poly = ReliabilityPolynomial([1.0, 0.8, 0.5, 0.2, 0.0])
        assert poly.evaluate(1.0) == pytest.approx(0.0)

    def test_p_one_nonzero_wn(self) -> None:
        poly = ReliabilityPolynomial([1.0, 0.5, 0.3])
        assert poly.evaluate(1.0) == pytest.approx(0.3)

    def test_midpoint_binary_weights(self) -> None:
        # w = [1, 0]: Res(p) = (1-p)*1 + p*0 = 1-p
        poly = ReliabilityPolynomial([1.0, 0.0])
        assert poly.evaluate(0.5) == pytest.approx(0.5)

    def test_uniform_weights(self) -> None:
        # All weights = 0.7: Res(p) = 0.7 for all p
        poly = ReliabilityPolynomial([0.7, 0.7, 0.7])
        assert poly.evaluate(0.0) == pytest.approx(0.7)
        assert poly.evaluate(0.5) == pytest.approx(0.7)
        assert poly.evaluate(1.0) == pytest.approx(0.7)

    def test_expected_quality_alias(self) -> None:
        poly = ReliabilityPolynomial([1.0, 0.5, 0.0])
        assert poly.expected_quality(0.3) == poly.evaluate(0.3)

    def test_out_of_range_raises(self) -> None:
        poly = ReliabilityPolynomial([1.0, 0.0])
        with pytest.raises(ValueError, match="must be in"):
            poly.evaluate(-0.1)
        with pytest.raises(ValueError, match="must be in"):
            poly.evaluate(1.1)

    def test_monotone_decreasing_weights_give_decreasing_quality(self) -> None:
        poly = ReliabilityPolynomial([1.0, 0.9, 0.7, 0.4, 0.1, 0.0])
        values = [poly.evaluate(p / 10.0) for p in range(11)]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1] - 1e-12


class TestDerivative:
    """Test polynomial derivative computation."""

    def test_constant_polynomial_zero_derivative(self) -> None:
        poly = ReliabilityPolynomial([0.5, 0.5, 0.5])
        assert poly.derivative(0.3) == pytest.approx(0.0, abs=1e-12)

    def test_linear_polynomial(self) -> None:
        # w = [1, 0]: Res(p) = 1-p, dRes/dp = -1
        poly = ReliabilityPolynomial([1.0, 0.0])
        assert poly.derivative(0.5) == pytest.approx(-1.0)

    def test_derivative_matches_numerical(self) -> None:
        poly = ReliabilityPolynomial([0.95, 0.85, 0.6, 0.3, 0.1, 0.0])
        p = 0.15
        h = 1e-7
        numerical = (poly.evaluate(p + h) - poly.evaluate(p - h)) / (2 * h)
        analytical = poly.derivative(p)
        assert analytical == pytest.approx(numerical, rel=1e-4)

    def test_derivative_at_boundary(self) -> None:
        poly = ReliabilityPolynomial([1.0, 0.5, 0.0])
        # Should not raise
        d0 = poly.derivative(0.0)
        d1 = poly.derivative(1.0)
        assert isinstance(d0, float)
        assert isinstance(d1, float)

    def test_zero_degree_polynomial(self) -> None:
        poly = ReliabilityPolynomial([0.8])
        assert poly.derivative(0.5) == 0.0


class TestCriticalFailureRate:
    """Test bisection for critical failure rate."""

    def test_linear_threshold(self) -> None:
        # Res(p) = 1-p, threshold=0.5 => p*=0.5
        poly = ReliabilityPolynomial([1.0, 0.0])
        p_star = poly.critical_failure_rate(threshold=0.5)
        assert p_star is not None
        assert p_star == pytest.approx(0.5, abs=1e-6)

    def test_never_below_threshold(self) -> None:
        poly = ReliabilityPolynomial([0.9, 0.9, 0.9])
        assert poly.critical_failure_rate(threshold=0.5) is None

    def test_starts_below_threshold(self) -> None:
        poly = ReliabilityPolynomial([0.3, 0.1, 0.0])
        assert poly.critical_failure_rate(threshold=0.5) == 0.0

    def test_critical_rate_consistent_with_evaluate(self) -> None:
        poly = ReliabilityPolynomial([0.95, 0.85, 0.6, 0.3, 0.1, 0.0])
        p_star = poly.critical_failure_rate(threshold=0.7)
        assert p_star is not None
        assert poly.evaluate(p_star) == pytest.approx(0.7, abs=1e-6)


class TestCrossover:
    """Test crossover detection between polynomials."""

    def test_identical_no_crossover(self) -> None:
        poly = ReliabilityPolynomial([1.0, 0.5, 0.0])
        # Identical polynomials: crossover at 0.0 (diff_0 = 0)
        result = poly.crossover(poly)
        assert result is not None
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_crossing_polynomials(self) -> None:
        # poly_a starts higher, poly_b ends higher
        poly_a = ReliabilityPolynomial([0.9, 0.3, 0.0])
        poly_b = ReliabilityPolynomial([0.5, 0.5, 0.5])
        p_star = poly_a.crossover(poly_b)
        assert p_star is not None
        assert poly_a.evaluate(p_star) == pytest.approx(poly_b.evaluate(p_star), abs=1e-6)

    def test_no_crossover_one_dominates(self) -> None:
        poly_a = ReliabilityPolynomial([1.0, 0.9, 0.8])
        poly_b = ReliabilityPolynomial([0.5, 0.4, 0.3])
        assert poly_a.crossover(poly_b) is None


class TestDominance:
    """Test dominance checking."""

    def test_higher_weights_dominate(self) -> None:
        poly_a = ReliabilityPolynomial([1.0, 0.9, 0.8])
        poly_b = ReliabilityPolynomial([0.9, 0.8, 0.7])
        assert poly_a.dominates(poly_b)

    def test_lower_weights_dont_dominate(self) -> None:
        poly_a = ReliabilityPolynomial([0.5, 0.4, 0.3])
        poly_b = ReliabilityPolynomial([1.0, 0.9, 0.8])
        assert not poly_a.dominates(poly_b)

    def test_crossing_neither_dominates(self) -> None:
        poly_a = ReliabilityPolynomial([0.9, 0.3, 0.0])
        poly_b = ReliabilityPolynomial([0.5, 0.5, 0.5])
        assert not poly_a.dominates(poly_b)
        assert not poly_b.dominates(poly_a)


class TestStandardForm:
    """Test Bernstein to power basis conversion."""

    def test_linear_conversion(self) -> None:
        # w = [1, 0]: Res(p) = 1-p = 1 + (-1)*p
        poly = ReliabilityPolynomial([1.0, 0.0])
        coeffs = poly.to_standard_form()
        assert coeffs[0] == pytest.approx(1.0)
        assert coeffs[1] == pytest.approx(-1.0)

    def test_constant_conversion(self) -> None:
        poly = ReliabilityPolynomial([0.7, 0.7, 0.7])
        coeffs = poly.to_standard_form()
        assert coeffs[0] == pytest.approx(0.7)
        assert coeffs[1] == pytest.approx(0.0, abs=1e-12)
        assert coeffs[2] == pytest.approx(0.0, abs=1e-12)

    def test_roundtrip_evaluation(self) -> None:
        """Standard form should evaluate to same values as Bernstein."""
        poly = ReliabilityPolynomial([0.95, 0.8, 0.5, 0.2, 0.0])
        coeffs = poly.to_standard_form()
        for p_int in range(11):
            p = p_int / 10.0
            # Evaluate standard form
            std_val = sum(c * p**i for i, c in enumerate(coeffs))
            assert poly.evaluate(p) == pytest.approx(std_val, abs=1e-10)


class TestLargeN:
    """Test numerical stability with large n."""

    def test_n100_no_overflow(self) -> None:
        n = 100
        weights = [max(0.0, 1.0 - k / n) for k in range(n + 1)]
        poly = ReliabilityPolynomial(weights)
        # Should not overflow or produce NaN
        val = poly.evaluate(0.1)
        assert math.isfinite(val)
        assert 0.0 <= val <= 1.0

    def test_n50_derivative_finite(self) -> None:
        n = 50
        weights = [max(0.0, 1.0 - 1.5 * k / n) for k in range(n + 1)]
        poly = ReliabilityPolynomial(weights)
        d = poly.derivative(0.2)
        assert math.isfinite(d)


class TestAnalyze:
    """Test the analyze() convenience method."""

    def test_analyze_returns_all_fields(self) -> None:
        poly = ReliabilityPolynomial([0.95, 0.8, 0.5, 0.0])
        analysis = poly.analyze(p=0.1, threshold=0.6)
        assert 0.0 <= analysis.expected_quality <= 1.0
        assert math.isfinite(analysis.derivative)
        assert len(analysis.coefficients) == 4
        assert analysis.critical_failure_rate is None or analysis.critical_failure_rate >= 0.0


class TestEquality:
    """Test equality comparison."""

    def test_equal_polynomials(self) -> None:
        a = ReliabilityPolynomial([1.0, 0.5, 0.0])
        b = ReliabilityPolynomial([1.0, 0.5, 0.0])
        assert a == b

    def test_unequal_polynomials(self) -> None:
        a = ReliabilityPolynomial([1.0, 0.5, 0.0])
        b = ReliabilityPolynomial([1.0, 0.3, 0.0])
        assert a != b

    def test_not_equal_to_other_type(self) -> None:
        a = ReliabilityPolynomial([1.0, 0.5, 0.0])
        assert a != "not a polynomial"


class TestRepr:
    """Test string representation."""

    def test_repr_format(self) -> None:
        poly = ReliabilityPolynomial([1.0, 0.5, 0.0])
        r = repr(poly)
        assert "ReliabilityPolynomial" in r
        assert "n=2" in r


class TestBernsteinBasis:
    """Test static Bernstein basis computation."""

    def test_b00_is_one(self) -> None:
        assert ReliabilityPolynomial.bernstein_basis(0, 0, 0.5) == pytest.approx(1.0)

    def test_b_n_n_at_one(self) -> None:
        assert ReliabilityPolynomial.bernstein_basis(5, 5, 1.0) == pytest.approx(1.0)

    def test_b_0_n_at_zero(self) -> None:
        assert ReliabilityPolynomial.bernstein_basis(5, 0, 0.0) == pytest.approx(1.0)

    def test_partition_of_unity(self) -> None:
        """All Bernstein basis polynomials sum to 1 for any p."""
        n = 5
        p = 0.37
        total = sum(ReliabilityPolynomial.bernstein_basis(n, k, p) for k in range(n + 1))
        assert total == pytest.approx(1.0)

    def test_out_of_range_k(self) -> None:
        assert ReliabilityPolynomial.bernstein_basis(5, -1, 0.5) == 0.0
        assert ReliabilityPolynomial.bernstein_basis(5, 6, 0.5) == 0.0
