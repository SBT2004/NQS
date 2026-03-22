from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from math import cos, pi, sin
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


NodePair = tuple[int, int]


@dataclass(frozen=True, order=True)
class Edge:
    i: int
    j: int
    color: str


class Graph(ABC):
    """Base graph class with common ordering, neighbor, and edge-iterator logic."""

    def __init__(self, n_nodes: int, pbc: bool = True) -> None:
        if n_nodes <= 0:
            raise ValueError("Graph size must be positive.")
        self._n_nodes = n_nodes
        self.pbc = pbc

    @property
    def n_nodes(self) -> int:
        return self._n_nodes

    def get_neighbors(self, node: int, n: int = 1) -> tuple[int, ...]:
        self._validate_node(node)
        if n <= 0:
            raise ValueError("Neighbor order n must be positive.")
        return self._get_neighbors_impl(node, n)

    def adjacency(self, n: int = 1) -> dict[int, tuple[int, ...]]:
        return {node: self.get_neighbors(node, n) for node in range(self.n_nodes)}

    def iter_neighbor_pairs(self, n: int = 1) -> Iterator[NodePair]:
        seen: set[NodePair] = set()
        for node in range(self.n_nodes):
            for neighbor in self.get_neighbors(node, n):
                pair = (min(node, neighbor), max(node, neighbor))
                if pair[0] == pair[1] or pair in seen:
                    continue
                seen.add(pair)
                yield pair

    def iter_edges(self, color: str, n: int = 1) -> Iterator[Edge]:
        for i, j in self.iter_neighbor_pairs(n):
            yield Edge(i=i, j=j, color=color)

    @abstractmethod
    def _get_neighbors_impl(self, node: int, n: int) -> tuple[int, ...]:
        """Return the nth-neighbor list for a validated node."""

    def _make_edge(self, i: int, j: int, color: str) -> Edge | None:
        self._validate_node(i)
        self._validate_node(j)
        a, b = sorted((i, j))
        if a == b:
            return None
        return Edge(i=a, j=b, color=color)

    def _validate_node(self, node: int) -> None:
        if node < 0 or node >= self.n_nodes:
            raise ValueError(f"Node {node} is outside the graph.")

    def to_networkx(
        self,
        edge_specs: dict[int, str] | tuple[tuple[int, str], ...] = ((1, "#1f77b4"),),
    ) -> Any:
        """Build a networkx graph with edge color and neighbor-order metadata."""

        nx = _require_networkx()
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(range(self.n_nodes))
        for neighbor_order, color in _normalize_edge_specs(edge_specs):
            for edge in self.iter_edges(color=color, n=neighbor_order):
                nx_graph.add_edge(edge.i, edge.j, color=edge.color, neighbor_order=neighbor_order)
        return nx_graph

    def draw(
        self,
        *,
        edge_specs: dict[int, str] | tuple[tuple[int, str], ...] = ((1, "#1f77b4"),),
        ax: Axes | None = None,
        node_size: int = 420,
        node_color: str = "#F7FAFC",
        node_edge_color: str = "#1A202C",
        font_size: int = 9,
        title: str | None = None,
    ) -> tuple[Figure, Axes]:
        """Render the graph with networkx and matplotlib."""

        plt = _require_matplotlib_pyplot()
        nx = _require_networkx()
        specs = _normalize_edge_specs(edge_specs)
        nx_graph = self.to_networkx(edge_specs=specs)
        positions = _default_positions(self)

        if ax is None:
            figure_obj, axis_obj = plt.subplots(figsize=_default_figure_size(self))
            figure = cast("Figure", figure_obj)
            axis = cast("Axes", axis_obj)
        else:
            axis = ax
            figure = cast("Figure", axis.figure)

        nx.draw_networkx_nodes(
            nx_graph,
            pos=positions,
            ax=axis,
            node_size=node_size,
            node_color=node_color,
            edgecolors=node_edge_color,
            linewidths=1.1,
        )
        nx.draw_networkx_labels(nx_graph, pos=positions, ax=axis, font_size=font_size, font_color="#1A202C")

        for neighbor_order, color in specs:
            straight_edges, wrapped_edges = _partition_edges_for_drawing(self, nx_graph, neighbor_order)
            if straight_edges:
                nx.draw_networkx_edges(
                    nx_graph,
                    pos=positions,
                    ax=axis,
                    edgelist=straight_edges,
                    edge_color=color,
                    width=2.3,
                )
            if wrapped_edges:
                nx.draw_networkx_edges(
                    nx_graph,
                    pos=positions,
                    ax=axis,
                    edgelist=wrapped_edges,
                    edge_color=color,
                    width=2.1,
                    alpha=0.9,
                    arrows=True,
                    arrowstyle="-",
                    connectionstyle="arc3,rad=0.18",
                )

        if title is not None:
            axis.set_title(title)
        axis.set_axis_off()
        axis.set_aspect("equal")
        figure.tight_layout()
        return figure, axis


class Chain1D(Graph):
    """1D chain with nearest-neighbor bonds."""

    def __init__(self, length: int, pbc: bool = True) -> None:
        self.length = length
        super().__init__(n_nodes=length, pbc=pbc)

    @property
    def shape(self) -> tuple[int]:
        return (self.length,)

    def coord_to_index(self, coord: int) -> int:
        self._validate_node(coord)
        return coord

    def index_to_coord(self, index: int) -> int:
        self._validate_node(index)
        return index

    def _get_neighbors_impl(self, node: int, n: int) -> tuple[int, ...]:
        neighbors: set[int] = set()

        if self.pbc:
            neighbors.add((node + n) % self.length)
            neighbors.add((node - n) % self.length)
        else:
            if node + n < self.length:
                neighbors.add(node + n)
            if node - n >= 0:
                neighbors.add(node - n)

        neighbors.discard(node)
        return tuple(sorted(neighbors))


class SquareLattice(Graph):
    """2D square lattice with colored nearest- and next-nearest-neighbor edges."""

    def __init__(self, Lx: int, Ly: int | None = None, pbc: bool = True, ordering: str = "row_major") -> None:
        self.Lx = Lx
        self.Ly = Lx if Ly is None else Ly
        self.ordering = ordering

        if self.Lx <= 0 or self.Ly <= 0:
            raise ValueError("Lattice dimensions must be positive.")
        if ordering not in {"row_major", "column_major"}:
            raise ValueError("ordering must be either 'row_major' or 'column_major'.")

        super().__init__(n_nodes=self.Lx * self.Ly, pbc=pbc)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.Lx, self.Ly)

    def coord_to_index(self, coord: tuple[int, int]) -> int:
        x, y = coord
        if not (0 <= x < self.Lx and 0 <= y < self.Ly):
            raise ValueError(f"Coordinate {coord} is outside the lattice.")
        if self.ordering == "row_major":
            return y * self.Lx + x
        return x * self.Ly + y

    def index_to_coord(self, index: int) -> tuple[int, int]:
        self._validate_node(index)
        if self.ordering == "row_major":
            return (index % self.Lx, index // self.Lx)
        return (index // self.Ly, index % self.Ly)

    def _get_neighbors_impl(self, node: int, n: int) -> tuple[int, ...]:
        x, y = self.index_to_coord(node)
        if n == 1:
            offsets = ((1, 0), (-1, 0), (0, 1), (0, -1))
        elif n == 2:
            offsets = ((1, 1), (-1, 1), (1, -1), (-1, -1))
        else:
            raise ValueError("SquareLattice currently supports only first and second neighbors.")

        neighbors: set[int] = set()
        for dx, dy in offsets:
            coord = self._resolve_coord(x + dx, y + dy)
            if coord is not None:
                neighbors.add(self.coord_to_index(coord))
        neighbors.discard(node)
        return tuple(sorted(neighbors))

    def _resolve_coord(self, x: int, y: int) -> tuple[int, int] | None:
        if self.pbc:
            return (x % self.Lx, y % self.Ly)
        if 0 <= x < self.Lx and 0 <= y < self.Ly:
            return (x, y)
        return None


def _normalize_edge_specs(
    edge_specs: dict[int, str] | tuple[tuple[int, str], ...],
) -> tuple[tuple[int, str], ...]:
    items = edge_specs.items() if isinstance(edge_specs, dict) else edge_specs
    normalized = tuple((int(neighbor_order), color) for neighbor_order, color in items)
    if not normalized:
        raise ValueError("edge_specs must contain at least one neighbor-order/color entry.")
    for neighbor_order, _ in normalized:
        if neighbor_order <= 0:
            raise ValueError("neighbor-order entries in edge_specs must be positive.")
    return normalized


def _default_positions(graph: Graph) -> dict[int, tuple[float, float]]:
    if isinstance(graph, Chain1D):
        if graph.pbc:
            return {
                node: (
                    cos((2.0 * pi * node / graph.length) - (pi / 2.0)),
                    sin((2.0 * pi * node / graph.length) - (pi / 2.0)),
                )
                for node in range(graph.n_nodes)
            }
        return {node: (float(node), 0.0) for node in range(graph.n_nodes)}

    if isinstance(graph, SquareLattice):
        return {
            node: (float(coord[0]), float(graph.Ly - 1 - coord[1]))
            for node in range(graph.n_nodes)
            for coord in (graph.index_to_coord(node),)
        }

    return {node: (float(node), 0.0) for node in range(graph.n_nodes)}


def _default_figure_size(graph: Graph) -> tuple[float, float]:
    if isinstance(graph, Chain1D):
        return (7.0, 5.0) if graph.pbc else (9.0, 1.8)
    if isinstance(graph, SquareLattice):
        return (6.4, 6.0)
    return (6.0, 4.0)


def _partition_edges_for_drawing(graph: Graph, nx_graph: Any, neighbor_order: int) -> tuple[list[NodePair], list[NodePair]]:
    straight_edges: list[NodePair] = []
    wrapped_edges: list[NodePair] = []
    for i, j, metadata in nx_graph.edges(data=True):
        if metadata.get("neighbor_order") != neighbor_order:
            continue
        pair = (i, j)
        if isinstance(graph, SquareLattice) and graph.pbc and _is_wrapped_square_edge(graph, i, j):
            wrapped_edges.append(pair)
        else:
            straight_edges.append(pair)
    return straight_edges, wrapped_edges


def _is_wrapped_square_edge(graph: SquareLattice, i: int, j: int) -> bool:
    xi, yi = graph.index_to_coord(i)
    xj, yj = graph.index_to_coord(j)
    dx = abs(xi - xj)
    dy = abs(yi - yj)
    return dx > 1 or dy > 1


def _require_networkx() -> Any:
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError("Graph drawing requires the optional dependency 'networkx'.") from exc
    return nx


def _require_matplotlib_pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Graph drawing requires the optional dependency 'matplotlib'.") from exc
    return plt
