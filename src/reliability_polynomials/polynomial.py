"""Core reliability polynomial object.

The generalized reliability polynomial Res_τ(p) = Σ C(n,k) p^k (1-p)^{n-k} w_k(τ)
extends classical network reliability by replacing binary survival with quality-weighted
coefficients. Each w_k encodes how much quality the system retains when exactly k of n
agents have failed.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from reliability_polynomials.types import PolynomialAnalysis, TopologyKind, WeightVector

if TYPE_CHECKING:
    from collections.abc import Sequence


class ReliabilityPolynomial:
    """Generalized reliability polynomial in Bernstein basis.

    Res(p) = Σ_{k=0}^{n} C(n,k) p^k (1-p)^{n-k} w_k

    where p is the per-agent failure probability and w_k is the expected system
    quality when exactly k agents have failed.

    Args:
        weights: Quality weight vector. Either a WeightVector or a sequence of
            floats [w_0, w_1, ..., w_n] where w_k is quality with k failures.
    """

    __slots__ = ("_n", "_weights")

    def __init__(self, weights: WeightVector | Sequence[float]) -> None:
        if isinstance(weights, WeightVector):
            self._weights = weights
        else:
            w = tuple(float(x) for x in weights)
            self._weights = WeightVector(
                weights=w,
                topology=TopologyKind.CUSTOM,
                n_agents=len(w) - 1,
            )
        self._n = self._weights.n_agents

    @property
    def n(self) -> int:
        """Number of agents."""
        return self._n

    @property
    def weights(self) -> WeightVector:
        """Quality weight vector."""
        return self._weights

    @property
    def degree(self) -> int:
        """Polynomial degree (= number of agents)."""
        return self._n

    @staticmethod
    def bernstein_basis(n: int, k: int, p: float) -> float:
        """Compute Bernstein basis polynomial B_{k,n}(p).

        B_{k,n}(p) = C(n,k) p^k (1-p)^{n-k}

        Uses log-space computation for numerical stability with large n.
        """
        if k < 0 or k > n:
            return 0.0
        if p == 0.0:
            return 1.0 if k == 0 else 0.0
        if p == 1.0:
            return 1.0 if k == n else 0.0

        # Log-space: log(C(n,k)) + k*log(p) + (n-k)*log(1-p)
        log_binom = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        log_term = log_binom + k * math.log(p) + (n - k) * math.log(1.0 - p)
        return math.exp(log_term)

    def evaluate(self, p: float) -> float:
        """Evaluate the reliability polynomial at failure probability p.

        Args:
            p: Per-agent failure probability in [0, 1].

        Returns:
            Expected system quality Res(p) in [0, 1].
        """
        if not 0.0 <= p <= 1.0:
            msg = f"p must be in [0, 1], got {p}"
            raise ValueError(msg)

        # Compute each term in log-space, then sum with fsum for precision
        n = self._n
        terms: list[float] = []
        for k in range(n + 1):
            w_k = self._weights[k]
            if w_k == 0.0:
                continue
            basis = self.bernstein_basis(n, k, p)
            terms.append(basis * w_k)

        return math.fsum(terms)

    def derivative(self, p: float) -> float:
        """Compute the derivative dRes/dp at failure probability p.

        Uses the Bernstein derivative identity:
        dRes/dp = n * Σ_{k=0}^{n-1} C(n-1,k) p^k (1-p)^{n-1-k} (w_{k+1} - w_k)

        Args:
            p: Per-agent failure probability in [0, 1].

        Returns:
            Rate of quality change with respect to failure probability.
        """
        if not 0.0 <= p <= 1.0:
            msg = f"p must be in [0, 1], got {p}"
            raise ValueError(msg)

        if self._n == 0:
            return 0.0

        n = self._n
        terms: list[float] = []
        for k in range(n):
            delta_w = self._weights[k + 1] - self._weights[k]
            if delta_w == 0.0:
                continue
            basis = self.bernstein_basis(n - 1, k, p)
            terms.append(basis * delta_w)

        return n * math.fsum(terms)

    def expected_quality(self, p: float) -> float:
        """Alias for evaluate(p). Returns expected system quality."""
        return self.evaluate(p)

    def critical_failure_rate(self, threshold: float = 0.5, tol: float = 1e-9) -> float | None:
        """Find the failure probability where quality drops below threshold.

        Uses bisection on [0, 1] to find p* such that Res(p*) = threshold.

        Args:
            threshold: Quality threshold in (0, 1).
            tol: Bisection tolerance.

        Returns:
            The critical failure rate p*, or None if quality never drops below
            the threshold (or starts below it).
        """
        if self.evaluate(0.0) < threshold:
            return 0.0
        if self.evaluate(1.0) >= threshold:
            return None

        lo, hi = 0.0, 1.0
        while hi - lo > tol:
            mid = (lo + hi) / 2.0
            if self.evaluate(mid) > threshold:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    def crossover(self, other: ReliabilityPolynomial, tol: float = 1e-9) -> float | None:
        """Find the crossover point where self and other have equal quality.

        Uses bisection to find p* where self(p*) = other(p*), leveraging the
        intermediate value theorem on the difference self(p) - other(p).

        Args:
            other: Another reliability polynomial to compare against.
            tol: Bisection tolerance.

        Returns:
            The crossover failure rate p*, or None if no crossover exists in (0, 1).
        """
        # Sample the difference at endpoints
        diff_0 = self.evaluate(0.0) - other.evaluate(0.0)
        diff_1 = self.evaluate(1.0) - other.evaluate(1.0)

        if diff_0 == 0.0:
            return 0.0

        # Check for sign change (IVT)
        if diff_0 * diff_1 > 0:
            # No guaranteed crossover; scan for one
            for i in range(1, 101):
                p = i / 100.0
                diff_p = self.evaluate(p) - other.evaluate(p)
                if diff_0 * diff_p <= 0:
                    return self._bisect_crossover(other, (i - 1) / 100.0, p, tol)
            return None

        return self._bisect_crossover(other, 0.0, 1.0, tol)

    def _bisect_crossover(
        self,
        other: ReliabilityPolynomial,
        lo: float,
        hi: float,
        tol: float,
    ) -> float:
        """Bisection helper for crossover finding."""
        diff_lo = self.evaluate(lo) - other.evaluate(lo)
        while hi - lo > tol:
            mid = (lo + hi) / 2.0
            diff_mid = self.evaluate(mid) - other.evaluate(mid)
            if diff_lo * diff_mid <= 0:
                hi = mid
            else:
                lo = mid
                diff_lo = diff_mid
        return (lo + hi) / 2.0

    def dominates(
        self,
        other: ReliabilityPolynomial,
        p_range: tuple[float, float] = (0.0, 1.0),
        n_samples: int = 1000,
    ) -> bool:
        """Check if self(p) >= other(p) across the given range.

        Args:
            other: Polynomial to compare against.
            p_range: Range of failure probabilities to check.
            n_samples: Number of sample points.

        Returns:
            True if self dominates other across the entire range.
        """
        lo, hi = p_range
        for i in range(n_samples + 1):
            p = lo + (hi - lo) * i / n_samples
            if self.evaluate(p) < other.evaluate(p) - 1e-12:
                return False
        return True

    def to_standard_form(self) -> tuple[float, ...]:
        """Convert from Bernstein basis to standard power basis.

        Returns coefficients (a_0, a_1, ..., a_n) such that
        Res(p) = a_0 + a_1*p + a_2*p^2 + ... + a_n*p^n.

        Uses the identity: B_{k,n}(p) = C(n,k) Σ_{j=0}^{n-k} (-1)^j C(n-k,j) p^{k+j}
        so that a_i = Σ_{k=0}^{i} (-1)^{i-k} C(n,i) C(i,k) w_k.
        """
        n = self._n
        coeffs = [0.0] * (n + 1)
        for i in range(n + 1):
            terms: list[float] = []
            for k in range(i + 1):
                sign = (-1) ** (i - k)
                binom_ni = math.comb(n, i)
                binom_ik = math.comb(i, k)
                terms.append(sign * binom_ni * binom_ik * self._weights[k])
            coeffs[i] = math.fsum(terms)
        return tuple(coeffs)

    def analyze(self, p: float = 0.1, threshold: float = 0.5) -> PolynomialAnalysis:
        """Comprehensive analysis at a given failure probability.

        Args:
            p: Failure probability for evaluation.
            threshold: Quality threshold for critical failure rate.

        Returns:
            PolynomialAnalysis with quality, derivative, coefficients, and
            critical failure rate.
        """
        return PolynomialAnalysis(
            expected_quality=self.evaluate(p),
            derivative=self.derivative(p),
            coefficients=self._weights.weights,
            critical_failure_rate=self.critical_failure_rate(threshold),
        )

    def __repr__(self) -> str:
        return f"ReliabilityPolynomial(n={self._n}, topology={self._weights.topology.value})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ReliabilityPolynomial):
            return NotImplemented
        return self._weights.weights == other._weights.weights
