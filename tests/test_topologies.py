"""Tests for topology weight computation."""

from __future__ import annotations

import pytest

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
from reliability_polynomials.types import TopologyConfig, TopologyKind


class TestMeshWeights:
    """Test mesh topology weight computation."""

    def test_w0_equals_one_minus_delta_coord(self) -> None:
        config = TopologyConfig(n_agents=5, delta_coord=0.05, delta_redist=0.05)
        wv = mesh_weights(config)
        assert wv[0] == pytest.approx(0.95)

    def test_wn_is_zero(self) -> None:
        config = TopologyConfig(n_agents=5, delta_coord=0.05, delta_redist=0.05)
        wv = mesh_weights(config)
        assert wv[5] == 0.0

    def test_weights_monotone_decreasing_sqrt(self) -> None:
        config = TopologyConfig(n_agents=5, gamma="sqrt")
        wv = mesh_weights(config)
        for k in range(len(wv) - 1):
            assert wv[k] >= wv[k + 1] - 1e-12

    def test_constant_gamma_higher_than_sqrt(self) -> None:
        """Constant gamma (no degradation) should give higher weights than sqrt."""
        config_const = TopologyConfig(n_agents=5, gamma="constant")
        config_sqrt = TopologyConfig(n_agents=5, gamma="sqrt")
        wv_const = mesh_weights(config_const)
        wv_sqrt = mesh_weights(config_sqrt)
        for k in range(1, 5):  # Skip k=0 (same) and k=5 (both 0)
            assert wv_const[k] >= wv_sqrt[k] - 1e-12

    def test_custom_gamma_callable(self) -> None:
        config = TopologyConfig(n_agents=3, gamma=lambda k: 0.5)
        wv = mesh_weights(config)
        assert len(wv) == 4
        assert wv[3] == 0.0

    def test_topology_kind_is_mesh(self) -> None:
        config = TopologyConfig(n_agents=5)
        wv = mesh_weights(config)
        assert wv.topology == TopologyKind.MESH


class TestHierarchyWeights:
    """Test hierarchy topology weight computation."""

    def test_w0_is_one(self) -> None:
        """Hierarchy w_0 = (n/n) * (n/n) = 1.0 (no overhead like mesh)."""
        config = TopologyConfig(n_agents=5, delta_coord=0.05, delta_redist=0.05)
        wv = hierarchy_weights(config)
        assert wv[0] == pytest.approx(1.0)

    def test_wn_is_zero(self) -> None:
        config = TopologyConfig(n_agents=5)
        wv = hierarchy_weights(config)
        assert wv[5] == 0.0

    def test_coordinator_failure_modeled(self) -> None:
        """With k failures, probability (n-k)/n that coordinator survives."""
        config = TopologyConfig(n_agents=5, delta_coord=0.0, delta_redist=0.0, gamma="constant")
        wv = hierarchy_weights(config)
        # w_1 = (4/5) * [(4 + 1*1.0*1.0) / 5] = 0.8 * 1.0 = 0.8
        assert wv[1] == pytest.approx(0.8)

    def test_topology_kind_is_hierarchy(self) -> None:
        config = TopologyConfig(n_agents=5)
        wv = hierarchy_weights(config)
        assert wv.topology == TopologyKind.HIERARCHY


class TestUnsupervisedWeights:
    """Test unsupervised topology weight computation."""

    def test_exact_formula(self) -> None:
        config = TopologyConfig(n_agents=5)
        wv = unsupervised_weights(config)
        for k in range(6):
            assert wv[k] == pytest.approx((5 - k) / 5)

    def test_w0_is_one(self) -> None:
        config = TopologyConfig(n_agents=10)
        wv = unsupervised_weights(config)
        assert wv[0] == pytest.approx(1.0)

    def test_wn_is_zero(self) -> None:
        config = TopologyConfig(n_agents=10)
        wv = unsupervised_weights(config)
        assert wv[10] == pytest.approx(0.0)

    def test_topology_kind_is_none(self) -> None:
        config = TopologyConfig(n_agents=5)
        wv = unsupervised_weights(config)
        assert wv.topology == TopologyKind.NONE


class TestPaperTableB1:
    """Reproduce paper Table B.1 values (n=5, delta=0.05, gamma=sqrt)."""

    @pytest.fixture()
    def table(self) -> dict[str, tuple[float, ...]]:
        return quality_table()

    def test_mesh_w0(self, table: dict[str, tuple[float, ...]]) -> None:
        assert table["mesh"][0] == pytest.approx(0.950, abs=1e-3)

    def test_mesh_w1(self, table: dict[str, tuple[float, ...]]) -> None:
        # w_1 = 0.95 * [(4 + 1*(1/1)*0.95) / 5] = 0.95 * 0.990 = 0.9405
        assert table["mesh"][1] == pytest.approx(0.941, abs=2e-3)

    def test_mesh_w5(self, table: dict[str, tuple[float, ...]]) -> None:
        assert table["mesh"][5] == pytest.approx(0.0)

    def test_none_exact(self, table: dict[str, tuple[float, ...]]) -> None:
        expected = (1.0, 0.8, 0.6, 0.4, 0.2, 0.0)
        for k in range(6):
            assert table["none"][k] == pytest.approx(expected[k], abs=1e-10)

    def test_hierarchy_w0(self, table: dict[str, tuple[float, ...]]) -> None:
        assert table["hierarchy"][0] == pytest.approx(1.0, abs=1e-3)

    def test_all_topologies_present(self, table: dict[str, tuple[float, ...]]) -> None:
        assert set(table.keys()) == {"mesh", "hierarchy", "none"}

    def test_all_weight_lengths(self, table: dict[str, tuple[float, ...]]) -> None:
        for weights in table.values():
            assert len(weights) == 6  # n=5 => 6 weights


class TestCustomWeights:
    """Test custom weight vector creation."""

    def test_from_list(self) -> None:
        wv = custom_weights([1.0, 0.5, 0.0])
        assert wv.topology == TopologyKind.CUSTOM
        assert wv.n_agents == 2

    def test_from_tuple(self) -> None:
        wv = custom_weights((0.9, 0.7, 0.3, 0.0))
        assert wv.n_agents == 3


class TestBuildPolynomial:
    """Test factory function."""

    def test_mesh(self) -> None:
        config = TopologyConfig(n_agents=5)
        poly = build_polynomial(TopologyKind.MESH, config)
        assert isinstance(poly, ReliabilityPolynomial)

    def test_hierarchy(self) -> None:
        config = TopologyConfig(n_agents=5)
        poly = build_polynomial(TopologyKind.HIERARCHY, config)
        assert isinstance(poly, ReliabilityPolynomial)

    def test_none(self) -> None:
        config = TopologyConfig(n_agents=5)
        poly = build_polynomial(TopologyKind.NONE, config)
        assert isinstance(poly, ReliabilityPolynomial)

    def test_custom_raises(self) -> None:
        config = TopologyConfig(n_agents=5)
        with pytest.raises(ValueError, match="Cannot build"):
            build_polynomial(TopologyKind.CUSTOM, config)


class TestCompareTopologies:
    """Test topology comparison."""

    def test_returns_all_keys(self) -> None:
        config = TopologyConfig(n_agents=5)
        result = compare_topologies(config)
        assert set(result.keys()) == {"p", "mesh", "hierarchy", "none"}

    def test_default_p_values(self) -> None:
        config = TopologyConfig(n_agents=5)
        result = compare_topologies(config)
        assert len(result["p"]) == 11

    def test_custom_p_values(self) -> None:
        config = TopologyConfig(n_agents=5)
        result = compare_topologies(config, p_values=[0.0, 0.5, 1.0])
        assert len(result["mesh"]) == 3


class TestConfigValidation:
    """Test configuration validation."""

    def test_negative_agents_raises(self) -> None:
        with pytest.raises(ValueError, match="n_agents"):
            TopologyConfig(n_agents=0)

    def test_invalid_delta_coord_raises(self) -> None:
        with pytest.raises(ValueError, match="delta_coord"):
            TopologyConfig(delta_coord=1.5)

    def test_invalid_gamma_string_raises(self) -> None:
        with pytest.raises(ValueError, match="gamma"):
            TopologyConfig(gamma="invalid")
