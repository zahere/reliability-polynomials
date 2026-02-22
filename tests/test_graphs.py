"""Tests for graph analysis."""

from __future__ import annotations

import math

import pytest

from reliability_polynomials.graphs import (
    Graph,
    complete_graph,
    grid_graph,
    ring_graph,
    star_graph,
)
from reliability_polynomials.types import ConsensusProtocol


class TestCompleteGraph:
    """Test complete graph K_n."""

    def test_k5_edges(self) -> None:
        g = complete_graph(5)
        assert g.n_edges == 5 * 4 // 2  # 10

    def test_k5_connectivity(self) -> None:
        g = complete_graph(5)
        assert g.vertex_connectivity() == 4

    def test_k5_algebraic_connectivity(self) -> None:
        """lambda_2(K_n) = n."""
        g = complete_graph(5)
        lambda2 = g.algebraic_connectivity()
        assert lambda2 == pytest.approx(5.0, abs=0.1)

    def test_k3_edges(self) -> None:
        g = complete_graph(3)
        assert g.n_edges == 3

    def test_k3_connectivity(self) -> None:
        g = complete_graph(3)
        assert g.vertex_connectivity() == 2

    def test_k1_single_node(self) -> None:
        g = complete_graph(1)
        assert g.n_nodes == 1
        assert g.n_edges == 0

    def test_k10_edges(self) -> None:
        g = complete_graph(10)
        assert g.n_edges == 10 * 9 // 2

    def test_k10_connectivity(self) -> None:
        g = complete_graph(10)
        assert g.vertex_connectivity() == 9


class TestRingGraph:
    """Test ring (cycle) graph C_n."""

    def test_ring5_edges(self) -> None:
        g = ring_graph(5)
        assert g.n_edges == 5

    def test_ring5_connectivity(self) -> None:
        g = ring_graph(5)
        assert g.vertex_connectivity() == 2

    def test_ring5_algebraic_connectivity(self) -> None:
        """lambda_2(C_n) = 2(1 - cos(2*pi/n))."""
        g = ring_graph(5)
        expected = 2.0 * (1.0 - math.cos(2.0 * math.pi / 5.0))
        lambda2 = g.algebraic_connectivity()
        assert lambda2 == pytest.approx(expected, abs=0.1)

    def test_ring10_algebraic_connectivity(self) -> None:
        g = ring_graph(10)
        expected = 2.0 * (1.0 - math.cos(2.0 * math.pi / 10.0))
        lambda2 = g.algebraic_connectivity()
        assert lambda2 == pytest.approx(expected, abs=0.1)

    def test_ring3_is_triangle(self) -> None:
        g = ring_graph(3)
        assert g.n_edges == 3
        assert g.vertex_connectivity() == 2

    def test_ring_requires_3_nodes(self) -> None:
        with pytest.raises(ValueError, match="n >= 3"):
            ring_graph(2)


class TestStarGraph:
    """Test star graph S_n."""

    def test_star5_edges(self) -> None:
        g = star_graph(5)
        assert g.n_edges == 4

    def test_star5_connectivity(self) -> None:
        """Removing the hub disconnects all leaves."""
        g = star_graph(5)
        assert g.vertex_connectivity() == 1

    def test_star_hub_degree(self) -> None:
        g = star_graph(5)
        assert g.degree("0") == 4

    def test_star_leaf_degree(self) -> None:
        g = star_graph(5)
        assert g.degree("1") == 1

    def test_star_requires_2_nodes(self) -> None:
        with pytest.raises(ValueError, match="n >= 2"):
            star_graph(1)


class TestGridGraph:
    """Test grid (lattice) graph."""

    def test_3x3_edges(self) -> None:
        g = grid_graph(3, 3)
        # 3*3=9 nodes, edges: 3*2 horizontal + 2*3 vertical = 12
        assert g.n_edges == 12

    def test_3x3_nodes(self) -> None:
        g = grid_graph(3, 3)
        assert g.n_nodes == 9

    def test_2x2_edges(self) -> None:
        g = grid_graph(2, 2)
        assert g.n_edges == 4

    def test_1x1_no_edges(self) -> None:
        g = grid_graph(1, 1)
        assert g.n_nodes == 1
        assert g.n_edges == 0

    def test_invalid_dimensions(self) -> None:
        with pytest.raises(ValueError, match="rows >= 1"):
            grid_graph(0, 3)


class TestGraphBasics:
    """Test basic graph operations."""

    def test_add_node(self) -> None:
        g = Graph()
        g.add_node("a")
        assert "a" in g.nodes

    def test_add_edge(self) -> None:
        g = Graph()
        g.add_edge("a", "b")
        assert g.n_nodes == 2
        assert g.n_edges == 1

    def test_self_loop_raises(self) -> None:
        g = Graph()
        with pytest.raises(ValueError, match="Self-loops"):
            g.add_edge("a", "a")

    def test_neighbors(self) -> None:
        g = Graph()
        g.add_edge("a", "b")
        g.add_edge("a", "c")
        assert g.neighbors("a") == frozenset({"b", "c"})

    def test_neighbors_unknown_node_raises(self) -> None:
        g = Graph()
        with pytest.raises(KeyError):
            g.neighbors("nonexistent")

    def test_degree(self) -> None:
        g = Graph()
        g.add_edge("a", "b")
        g.add_edge("a", "c")
        assert g.degree("a") == 2
        assert g.degree("b") == 1


class TestConnectivity:
    """Test BFS connectivity checking."""

    def test_connected_graph(self) -> None:
        g = complete_graph(5)
        assert g.is_connected()

    def test_disconnected_graph(self) -> None:
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        assert not g.is_connected()

    def test_single_node_connected(self) -> None:
        g = Graph()
        g.add_node("a")
        assert g.is_connected()

    def test_empty_graph_connected(self) -> None:
        g = Graph()
        assert g.is_connected()

    def test_connected_with_exclusion(self) -> None:
        """Removing hub from star disconnects it."""
        g = star_graph(4)
        assert not g.is_connected(exclude=frozenset({"0"}))

    def test_disconnected_vertex_connectivity_zero(self) -> None:
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        assert g.vertex_connectivity() == 0


class TestAlgebraicConnectivity:
    """Test lambda_2 computation."""

    def test_single_node_zero(self) -> None:
        g = Graph()
        g.add_node("a")
        assert g.algebraic_connectivity() == 0.0

    def test_disconnected_zero(self) -> None:
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        assert g.algebraic_connectivity() == 0.0

    def test_k2_lambda2(self) -> None:
        """K_2: lambda_2 = 2."""
        g = complete_graph(2)
        assert g.algebraic_connectivity() == pytest.approx(2.0, abs=0.1)


class TestGraphAnalyze:
    """Test full graph analysis."""

    def test_k5_pbft(self) -> None:
        g = complete_graph(5)
        result = g.analyze(ConsensusProtocol.PBFT)
        assert result.vertex_connectivity == 4
        assert result.byzantine_tolerance == 1
        assert result.is_connected

    def test_k5_gossip(self) -> None:
        """K5 with gossip: kappa=4 but Byzantine tolerance=0."""
        g = complete_graph(5)
        result = g.analyze(ConsensusProtocol.GOSSIP)
        assert result.vertex_connectivity == 4
        assert result.byzantine_tolerance == 0

    def test_star_analysis(self) -> None:
        g = star_graph(5)
        result = g.analyze()
        assert result.vertex_connectivity == 1
        assert result.crash_tolerance == 1


class TestRepr:
    """Test graph string representation."""

    def test_repr_format(self) -> None:
        g = complete_graph(5)
        r = repr(g)
        assert "Graph" in r
        assert "nodes=5" in r
        assert "edges=10" in r
