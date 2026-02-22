"""Lightweight graph analysis for reliability computation.

Provides undirected graph representation with vertex connectivity (kappa)
and algebraic connectivity (lambda_2) computation. Designed for small agent
networks (n <= 20) where brute-force algorithms are acceptable.
"""

from __future__ import annotations

import itertools
import math

from reliability_polynomials.types import ConnectivityResult, ConsensusProtocol


class Graph:
    """Undirected, unweighted graph for network reliability analysis.

    Nodes are strings. Edges are undirected pairs stored as frozensets internally.
    """

    __slots__ = ("_adjacency", "_nodes")

    def __init__(self) -> None:
        self._nodes: set[str] = set()
        self._adjacency: dict[str, set[str]] = {}

    @property
    def nodes(self) -> frozenset[str]:
        """All nodes in the graph."""
        return frozenset(self._nodes)

    @property
    def edges(self) -> set[tuple[str, str]]:
        """All edges as (u, v) pairs where u < v lexicographically."""
        seen: set[tuple[str, str]] = set()
        for u, neighbors in self._adjacency.items():
            for v in neighbors:
                edge = (min(u, v), max(u, v))
                seen.add(edge)
        return seen

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return len(self._nodes)

    @property
    def n_edges(self) -> int:
        """Number of edges."""
        return len(self.edges)

    def add_node(self, node: str) -> None:
        """Add a node to the graph."""
        self._nodes.add(node)
        if node not in self._adjacency:
            self._adjacency[node] = set()

    def add_edge(self, u: str, v: str) -> None:
        """Add an undirected edge between u and v."""
        if u == v:
            msg = f"Self-loops not allowed: {u}"
            raise ValueError(msg)
        self.add_node(u)
        self.add_node(v)
        self._adjacency[u].add(v)
        self._adjacency[v].add(u)

    def neighbors(self, node: str) -> frozenset[str]:
        """Return the neighbors of a node."""
        if node not in self._adjacency:
            msg = f"Node '{node}' not in graph"
            raise KeyError(msg)
        return frozenset(self._adjacency[node])

    def degree(self, node: str) -> int:
        """Return the degree of a node."""
        return len(self._adjacency[node])

    def is_connected(self, exclude: frozenset[str] | None = None) -> bool:
        """Check if the graph is connected, optionally excluding nodes.

        Uses BFS from an arbitrary non-excluded node.

        Args:
            exclude: Set of nodes to remove before checking connectivity.
        """
        excluded = exclude or frozenset()
        active = self._nodes - excluded

        if len(active) <= 1:
            return True

        start = next(iter(active))
        visited: set[str] = set()
        queue = [start]

        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            for neighbor in self._adjacency[node]:
                if neighbor not in excluded and neighbor not in visited:
                    queue.append(neighbor)

        return len(visited) == len(active)

    def vertex_connectivity(self) -> int:
        """Compute vertex connectivity kappa(G).

        The minimum number of nodes whose removal disconnects the graph.
        Uses brute-force enumeration over subsets (acceptable for n <= 20).

        Returns:
            Vertex connectivity. Returns n-1 for complete graphs,
            0 for disconnected graphs.
        """
        n = self.n_nodes
        if n <= 1:
            return 0

        if not self.is_connected():
            return 0

        # Check if complete graph
        if self.n_edges == n * (n - 1) // 2:
            return n - 1

        # Try removing k nodes for increasing k
        node_list = sorted(self._nodes)
        for k in range(1, n):
            for subset in itertools.combinations(node_list, k):
                excluded = frozenset(subset)
                if not self.is_connected(exclude=excluded):
                    return k

        return n - 1

    def algebraic_connectivity(self, max_iter: int = 200, tol: float = 1e-10) -> float:
        """Compute algebraic connectivity lambda_2 (Fiedler value).

        Uses Jacobi eigenvalue algorithm on the graph Laplacian L = D - A.
        This finds ALL eigenvalues of the symmetric matrix, then returns the
        second smallest. Suitable for small graphs (n <= 20).

        Args:
            max_iter: Maximum Jacobi sweeps.
            tol: Convergence tolerance for off-diagonal norm.

        Returns:
            Second-smallest eigenvalue of the Laplacian. Returns 0.0 for
            disconnected graphs or single-node graphs.
        """
        n = self.n_nodes
        if n <= 1:
            return 0.0

        if not self.is_connected():
            return 0.0

        # Build Laplacian L = D - A
        nodes = sorted(self._nodes)
        idx = {node: i for i, node in enumerate(nodes)}

        a: list[list[float]] = [[0.0] * n for _ in range(n)]
        for u in nodes:
            i = idx[u]
            a[i][i] = float(self.degree(u))
            for v in self._adjacency[u]:
                j = idx[v]
                a[i][j] = -1.0

        # Jacobi eigenvalue algorithm for symmetric matrices
        for _ in range(max_iter * n):
            # Find largest off-diagonal element
            max_val = 0.0
            p, q = 0, 1
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(a[i][j]) > max_val:
                        max_val = abs(a[i][j])
                        p, q = i, j

            if max_val < tol:
                break

            # Compute rotation angle
            if abs(a[p][p] - a[q][q]) < 1e-15:
                theta = math.pi / 4.0
            else:
                theta = 0.5 * math.atan2(2.0 * a[p][q], a[p][p] - a[q][q])

            c = math.cos(theta)
            s = math.sin(theta)

            # Apply Jacobi rotation: A' = J^T A J
            # Store rows p and q
            row_p = [a[p][j] for j in range(n)]
            row_q = [a[q][j] for j in range(n)]

            for j in range(n):
                if j in (p, q):
                    continue
                a[p][j] = c * row_p[j] + s * row_q[j]
                a[j][p] = a[p][j]
                a[q][j] = -s * row_p[j] + c * row_q[j]
                a[j][q] = a[q][j]

            a[p][p] = c * c * row_p[p] + 2.0 * s * c * row_p[q] + s * s * row_q[q]
            a[q][q] = s * s * row_p[p] - 2.0 * s * c * row_p[q] + c * c * row_q[q]
            a[p][q] = 0.0
            a[q][p] = 0.0

        # Eigenvalues are on the diagonal
        eigenvalues = sorted(a[i][i] for i in range(n))

        # lambda_2 is the second smallest eigenvalue
        if len(eigenvalues) < 2:
            return 0.0

        return max(0.0, eigenvalues[1])

    def analyze(
        self,
        protocol: ConsensusProtocol = ConsensusProtocol.PBFT,
    ) -> ConnectivityResult:
        """Full connectivity analysis.

        Args:
            protocol: Consensus protocol for Byzantine tolerance computation.

        Returns:
            ConnectivityResult with vertex/algebraic connectivity, fault
            tolerance bounds, and connection status.
        """
        from reliability_polynomials.faults import byzantine_tolerance

        kappa = self.vertex_connectivity()
        lambda2 = self.algebraic_connectivity()
        byz_tol = byzantine_tolerance(self.n_nodes, protocol)
        crash_tol = min(kappa, (self.n_nodes - 1) // 2) if kappa > 0 else 0

        return ConnectivityResult(
            vertex_connectivity=kappa,
            algebraic_connectivity=lambda2,
            crash_tolerance=crash_tol,
            byzantine_tolerance=byz_tol,
            is_connected=self.is_connected(),
        )

    def __repr__(self) -> str:
        return f"Graph(nodes={self.n_nodes}, edges={self.n_edges})"


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def complete_graph(n: int) -> Graph:
    """Create a complete graph K_n.

    Every pair of nodes is connected. Vertex connectivity = n-1.
    """
    g = Graph()
    nodes = [str(i) for i in range(n)]
    for node in nodes:
        g.add_node(node)
    for i in range(n):
        for j in range(i + 1, n):
            g.add_edge(nodes[i], nodes[j])
    return g


def ring_graph(n: int) -> Graph:
    """Create a ring (cycle) graph C_n.

    Each node connected to its two neighbors. Vertex connectivity = 2.
    """
    if n < 3:
        msg = f"Ring graph requires n >= 3, got {n}"
        raise ValueError(msg)
    g = Graph()
    nodes = [str(i) for i in range(n)]
    for node in nodes:
        g.add_node(node)
    for i in range(n):
        g.add_edge(nodes[i], nodes[(i + 1) % n])
    return g


def star_graph(n: int) -> Graph:
    """Create a star graph S_n with one hub and n-1 leaves.

    Hub is node "0", leaves are "1" through "n-1". Vertex connectivity = 1.
    """
    if n < 2:
        msg = f"Star graph requires n >= 2, got {n}"
        raise ValueError(msg)
    g = Graph()
    hub = "0"
    g.add_node(hub)
    for i in range(1, n):
        leaf = str(i)
        g.add_edge(hub, leaf)
    return g


def grid_graph(rows: int, cols: int) -> Graph:
    """Create a grid (lattice) graph with given dimensions.

    Nodes are labeled "row,col". Interior nodes have degree 4.
    """
    if rows < 1 or cols < 1:
        msg = f"Grid requires rows >= 1 and cols >= 1, got {rows}x{cols}"
        raise ValueError(msg)
    g = Graph()
    for r in range(rows):
        for c in range(cols):
            node = f"{r},{c}"
            g.add_node(node)
            if r > 0:
                g.add_edge(node, f"{r - 1},{c}")
            if c > 0:
                g.add_edge(node, f"{r},{c - 1}")
    return g
