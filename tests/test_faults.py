"""Tests for fault model analysis and crossover computation."""

from __future__ import annotations

import math

import pytest

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
from reliability_polynomials.types import ConsensusProtocol, FaultConfig, FaultType


class TestCrashStopCrossover:
    """Test crash-stop crossover formula."""

    def test_default_params(self) -> None:
        """p* = delta_coord = 0.05 with default config."""
        config = FaultConfig()
        result = crash_stop_crossover(config)
        assert result.p_star == pytest.approx(0.05)

    def test_exact_equals_delta_coord(self) -> None:
        config = FaultConfig(delta_coord=0.1)
        result = crash_stop_crossover(config)
        assert result.p_star == pytest.approx(0.1)

    def test_fault_type(self) -> None:
        result = crash_stop_crossover(FaultConfig())
        assert result.fault_type == FaultType.CRASH_STOP


class TestByzantineCrossover:
    """Test Byzantine crossover formula."""

    def test_less_than_crash(self) -> None:
        """Byzantine crossover must be less than crash-stop crossover."""
        config = FaultConfig()
        byz = byzantine_crossover(config)
        crash = crash_stop_crossover(config)
        assert byz.p_star < crash.p_star

    def test_paper_value(self) -> None:
        """p* ≈ 0.0155 with paper parameters (delta=0.05, mu=0.65, tau_d=12)."""
        config = FaultConfig(delta_coord=0.05, mu=0.65, mu_byz=0.50, tau_d=12)
        result = byzantine_crossover(config)
        assert result.p_star == pytest.approx(0.0155, abs=1e-3)

    def test_alpha_in_parameters(self) -> None:
        config = FaultConfig()
        result = byzantine_crossover(config)
        alpha = (config.mu - config.mu_byz) / config.mu
        assert result.parameters["alpha"] == pytest.approx(alpha)

    def test_higher_tau_d_lowers_crossover(self) -> None:
        """Longer detection delay means Byzantine is worse earlier."""
        config_low = FaultConfig(tau_d=5)
        config_high = FaultConfig(tau_d=20)
        assert byzantine_crossover(config_high).p_star < byzantine_crossover(config_low).p_star


class TestIntermittentCrossover:
    """Test intermittent crossover formula."""

    def test_paper_value(self) -> None:
        """p* = 0.05^(1/3) ≈ 0.368 with k=3."""
        config = FaultConfig(delta_coord=0.05, k_retries=3)
        result = intermittent_crossover(config)
        assert result.p_star == pytest.approx(0.05 ** (1 / 3), abs=1e-4)

    def test_k1_equals_crash(self) -> None:
        """With k=1 retry, intermittent = crash-stop."""
        config = FaultConfig(k_retries=1)
        result = intermittent_crossover(config)
        assert result.p_star == pytest.approx(config.delta_coord)

    def test_higher_k_raises_crossover(self) -> None:
        """More retries push crossover higher."""
        config_k2 = FaultConfig(k_retries=2)
        config_k5 = FaultConfig(k_retries=5)
        assert intermittent_crossover(config_k5).p_star > intermittent_crossover(config_k2).p_star


class TestDegradationCrossover:
    """Test degradation crossover formula."""

    def test_is_infinity(self) -> None:
        result = degradation_crossover(FaultConfig())
        assert result.p_star == math.inf

    def test_fault_type(self) -> None:
        result = degradation_crossover(FaultConfig())
        assert result.fault_type == FaultType.DEGRADATION


class TestCorrelatedCrossover:
    """Test correlated crossover formula."""

    def test_greater_than_crash(self) -> None:
        """Correlated crossover > crash-stop crossover."""
        config = FaultConfig(rho=0.15)
        corr = correlated_crossover(config)
        crash = crash_stop_crossover(config)
        assert corr.p_star > crash.p_star

    def test_zero_correlation_equals_crash(self) -> None:
        """With rho=0, correlated = crash-stop."""
        config = FaultConfig(rho=0.0)
        corr = correlated_crossover(config)
        crash = crash_stop_crossover(config)
        assert corr.p_star == pytest.approx(crash.p_star)


class TestCrossoverOrdering:
    """Verify the complete crossover ordering."""

    def test_full_ordering(self) -> None:
        """p*_byz < p*_crash < p*_corr < p*_int < p*_deg = inf."""
        results = all_crossovers()
        p_stars = [r.p_star for r in results]
        # Results are pre-ordered in the function
        assert p_stars[0] < p_stars[1]  # byz < crash
        assert p_stars[1] < p_stars[2]  # crash < corr
        assert p_stars[2] < p_stars[3]  # corr < int
        assert p_stars[3] < p_stars[4]  # int < deg
        assert p_stars[4] == math.inf  # deg = inf

    def test_fault_types_in_order(self) -> None:
        results = all_crossovers()
        types = [r.fault_type for r in results]
        assert types == [
            FaultType.BYZANTINE,
            FaultType.CRASH_STOP,
            FaultType.CORRELATED,
            FaultType.INTERMITTENT,
            FaultType.DEGRADATION,
        ]


class TestCoordinationTransparency:
    """Test Theorem 1: coordination transparency."""

    def test_crash_transparent(self) -> None:
        assert is_coordination_transparent(FaultType.CRASH_STOP) is True

    def test_degradation_transparent(self) -> None:
        assert is_coordination_transparent(FaultType.DEGRADATION) is True

    def test_intermittent_transparent(self) -> None:
        assert is_coordination_transparent(FaultType.INTERMITTENT) is True

    def test_byzantine_not_transparent(self) -> None:
        assert is_coordination_transparent(FaultType.BYZANTINE) is False

    def test_correlated_not_transparent(self) -> None:
        assert is_coordination_transparent(FaultType.CORRELATED) is False


class TestFaultClassification:
    """Test full classification table."""

    def test_has_all_fault_types(self) -> None:
        table = fault_classification()
        types = {row["fault_type"] for row in table}
        assert FaultType.CRASH_STOP in types
        assert FaultType.BYZANTINE in types
        assert FaultType.DEGRADATION in types
        assert FaultType.INTERMITTENT in types
        assert FaultType.CORRELATED in types

    def test_table_length(self) -> None:
        assert len(fault_classification()) == 5


class TestEffectiveConnectivity:
    """Test Theorem 3: effective connectivity."""

    def test_crash_equals_kappa(self) -> None:
        assert effective_connectivity(FaultType.CRASH_STOP, 4, 5) == 4

    def test_byzantine_pbft(self) -> None:
        # K5 with PBFT: floor((5-1)/3) = 1
        assert effective_connectivity(FaultType.BYZANTINE, 4, 5, ConsensusProtocol.PBFT) == 1

    def test_byzantine_gossip_zero(self) -> None:
        # K5 with gossip: 0 Byzantine tolerance (even though kappa=4)
        assert effective_connectivity(FaultType.BYZANTINE, 4, 5, ConsensusProtocol.GOSSIP) == 0

    def test_degradation_equals_n(self) -> None:
        assert effective_connectivity(FaultType.DEGRADATION, 4, 5) == 5


class TestByzantineTolerance:
    """Test Byzantine tolerance for different protocols."""

    def test_pbft_5_agents(self) -> None:
        assert byzantine_tolerance(5, ConsensusProtocol.PBFT) == 1

    def test_pbft_7_agents(self) -> None:
        assert byzantine_tolerance(7, ConsensusProtocol.PBFT) == 2

    def test_majority_5_agents(self) -> None:
        assert byzantine_tolerance(5, ConsensusProtocol.MAJORITY) == 2

    def test_gossip_always_zero(self) -> None:
        assert byzantine_tolerance(100, ConsensusProtocol.GOSSIP) == 0


class TestFaultConfigValidation:
    """Test FaultConfig validation."""

    def test_mu_byz_ge_mu_raises(self) -> None:
        with pytest.raises(ValueError, match="mu_byz"):
            FaultConfig(mu=0.65, mu_byz=0.70)

    def test_negative_tau_d_raises(self) -> None:
        with pytest.raises(ValueError, match="tau_d"):
            FaultConfig(tau_d=-1)

    def test_zero_retries_raises(self) -> None:
        with pytest.raises(ValueError, match="k_retries"):
            FaultConfig(k_retries=0)

    def test_rho_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="rho"):
            FaultConfig(rho=1.5)
