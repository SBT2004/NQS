from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass


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

    def iter_neighbor_pairs(self, n: int = 1) -> Iterable[tuple[int, int]]:
        seen: set[tuple[int, int]] = set()
        for node in range(self.n_nodes):
            for neighbor in self.get_neighbors(node, n):
                pair = (min(node, neighbor), max(node, neighbor))
                if pair[0] == pair[1] or pair in seen:
                    continue
                seen.add(pair)
                yield pair

    def iter_edges(self, color: str, n: int = 1) -> Iterable[Edge]:
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
