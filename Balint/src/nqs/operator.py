from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Number
from typing import TYPE_CHECKING, Iterable, Protocol, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from .graph import Graph, SquareLattice
from .hilbert import SpinHilbert, SpinState, StateInput

if TYPE_CHECKING:
    import netket as nk
    from netket.utils.types import Array as NetKetArray
else:
    NetKetArray = object


LocalMatrix: TypeAlias = npt.NDArray[np.complex128]
ConnectedElement: TypeAlias = tuple[SpinState, complex]
ConnectedElementBits: TypeAlias = tuple[int, complex]


class SupportsConnectedElements(Protocol):
    hilbert: SpinHilbert

    def connected_elements(self, state: StateInput) -> list[ConnectedElement]:
        ...

    def connected_elements_bits(self, state: int | StateInput) -> list[ConnectedElementBits]:
        ...


def identity() -> LocalMatrix:
    return np.eye(2, dtype=np.complex128)


def sigmax() -> LocalMatrix:
    return np.array([[0, 1], [1, 0]], dtype=np.complex128)


def sigmay() -> LocalMatrix:
    return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)


def sigmaz() -> LocalMatrix:
    return np.array([[1, 0], [0, -1]], dtype=np.complex128)


def projector_zero() -> LocalMatrix:
    return np.array([[1, 0], [0, 0]], dtype=np.complex128)


def projector_one() -> LocalMatrix:
    return np.array([[0, 0], [0, 1]], dtype=np.complex128)


def local_matrix(matrix: npt.ArrayLike) -> LocalMatrix:
    arr = np.asarray(matrix, dtype=np.complex128)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Local matrices must be square.")
    return arr


def kron_product(*matrices: npt.ArrayLike) -> LocalMatrix:
    if not matrices:
        raise ValueError("kron_product requires at least one matrix.")
    result = local_matrix(matrices[0])
    for matrix in matrices[1:]:
        result = np.kron(result, local_matrix(matrix))
    return np.asarray(result, dtype=np.complex128)


def _pair_sites(i: int, j: int) -> tuple[int, int]:
    if i == j:
        raise ValueError("Two-site terms require distinct sites.")
    return (i, j) if i < j else (j, i)


def sx_term(site: int, coefficient: complex = 1.0) -> "LocalTerm":
    return LocalTerm((site,), sigmax(), coefficient=coefficient)


def szsz_term(i: int, j: int, coefficient: complex = 1.0) -> "LocalTerm":
    return LocalTerm(_pair_sites(i, j), kron_product(sigmaz(), sigmaz()), coefficient=coefficient)


def sxsx_term(i: int, j: int, coefficient: complex = 1.0) -> "LocalTerm":
    return LocalTerm(_pair_sites(i, j), kron_product(sigmax(), sigmax()), coefficient=coefficient)


def sysy_term(i: int, j: int, coefficient: complex = 1.0) -> "LocalTerm":
    return LocalTerm(_pair_sites(i, j), kron_product(sigmay(), sigmay()), coefficient=coefficient)


def heisenberg_term(i: int, j: int, coefficient: complex = 1.0) -> "LocalTerm":
    matrix = kron_product(sigmax(), sigmax()) + kron_product(sigmay(), sigmay()) + kron_product(sigmaz(), sigmaz())
    return LocalTerm(_pair_sites(i, j), matrix, coefficient=coefficient)


def collect_terms(*term_groups: Iterable["LocalTerm"]) -> list["LocalTerm"]:
    terms: list[LocalTerm] = []
    for group in term_groups:
        terms.extend(group)
    return terms


@dataclass(frozen=True)
class LocalTerm:
    sites: tuple[int, ...]
    matrix: LocalMatrix
    coefficient: complex = 1.0

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.complex128)
        object.__setattr__(self, "sites", tuple(self.sites))
        object.__setattr__(self, "matrix", matrix)

        if any(not isinstance(site, int) for site in self.sites):
            raise TypeError("LocalTerm sites must be integers.")
        if tuple(sorted(self.sites)) != self.sites:
            raise ValueError("LocalTerm sites must be sorted in ascending order.")
        if len(set(self.sites)) != len(self.sites):
            raise ValueError("LocalTerm sites must be unique.")
        if not isinstance(self.coefficient, Number):
            raise TypeError("LocalTerm coefficient must be numeric.")

        local_dim = 1 << len(self.sites)
        if matrix.shape != (local_dim, local_dim):
            raise ValueError(
                f"LocalTerm matrix must have shape {(local_dim, local_dim)} for {len(self.sites)} sites."
            )


class Operator:
    """Linear operator on a spin Hilbert space defined as a sum of local terms."""

    def __init__(self, hilbert: SpinHilbert, terms: Sequence[LocalTerm]) -> None:
        self.hilbert = hilbert
        self.terms = tuple(terms)
        self._validate_terms()

    def connected_elements(
        self,
        state: StateInput,
    ) -> list[ConnectedElement]:
        """Array-returning convenience wrapper around ``connected_elements_bits``."""

        return [
            (self.hilbert.index_to_state(index), value)
            for index, value in self.connected_elements_bits(state)
        ]

    def connected_elements_bits(
        self,
        state: int | StateInput,
    ) -> list[ConnectedElementBits]:
        state_bits = self._state_bits(state)
        contributions: dict[int, complex] = {}

        for term in self.terms:
            local_index = self._local_index_bits(state_bits, term.sites)
            column = term.matrix[:, local_index]
            nonzero_rows = np.flatnonzero(np.abs(column) > 0)

            for row in nonzero_rows:
                global_index = self._write_local_bits(state_bits, term.sites, int(row))
                value = complex(term.coefficient * column[row])
                contributions[global_index] = contributions.get(global_index, 0.0) + value

        return [
            (index, value)
            for index, value in sorted(contributions.items(), key=lambda item: item[0])
            if value != 0
        ]

    def _validate_terms(self) -> None:
        for term in self.terms:
            if any(site < 0 or site >= self.hilbert.size for site in term.sites):
                raise ValueError("LocalTerm site is outside the Hilbert space.")

    def to_netket(self) -> "nk.operator.LocalOperator":
        """Convert to NetKet's LocalOperator.

        This compatibility helper intentionally performs a lazy NetKet import so
        the ordinary project-owned runtime path stays NetKet-free. Calling this
        method still requires NetKet to be installed.
        """

        try:
            import netket as nk
        except ImportError as exc:
            raise ImportError("Operator.to_netket() requires NetKet to be installed.") from exc

        netket_hilbert = nk.hilbert.Spin(s=0.5, N=self.hilbert.size)
        operators = cast(
            list[NetKetArray],
            [np.asarray(term.matrix, dtype=np.complex128) * term.coefficient for term in self.terms],
        )
        acting_on = [list(term.sites) for term in self.terms]
        return nk.operator.LocalOperator(netket_hilbert, operators=operators, acting_on=acting_on)

    @staticmethod
    def _local_index(state: SpinState, sites: tuple[int, ...]) -> int:
        return sum(int(state[site]) << offset for offset, site in enumerate(sites))

    @staticmethod
    def _write_local_state(state: SpinState, sites: tuple[int, ...], local_index: int) -> None:
        for offset, site in enumerate(sites):
            state[site] = (local_index >> offset) & 1

    def _state_bits(self, state: int | StateInput) -> int:
        if isinstance(state, (int, np.integer)):
            state_bits = int(state)
            if state_bits < 0 or state_bits >= self.hilbert.n_states:
                raise ValueError(f"Bitmap state must lie in [0, {self.hilbert.n_states}).")
            return state_bits
        return self.hilbert.state_to_index(state)

    @staticmethod
    def _local_index_bits(state_bits: int, sites: tuple[int, ...]) -> int:
        return sum(((state_bits >> site) & 1) << offset for offset, site in enumerate(sites))

    @staticmethod
    def _write_local_bits(state_bits: int, sites: tuple[int, ...], local_index: int) -> int:
        updated_bits = state_bits
        for site in sites:
            updated_bits &= ~(1 << site)
        for offset, site in enumerate(sites):
            updated_bits |= ((local_index >> offset) & 1) << site
        return updated_bits


def tfim(hilbert: SpinHilbert, graph: Graph, J: float, h: float) -> Operator:
    zz_terms = [szsz_term(edge.i, edge.j, coefficient=-J) for edge in graph.iter_edges("J", n=1)]
    x_terms = [sx_term(site, coefficient=-h) for site in range(hilbert.size)]
    return Operator(hilbert, collect_terms(zz_terms, x_terms))


def j1_j2(hilbert: SpinHilbert, graph: SquareLattice, J1: float, J2: float) -> Operator:
    j1_terms = [heisenberg_term(edge.i, edge.j, coefficient=J1) for edge in graph.iter_edges("J1", n=1)]
    j2_terms = [heisenberg_term(edge.i, edge.j, coefficient=J2) for edge in graph.iter_edges("J2", n=2)]
    return Operator(hilbert, collect_terms(j1_terms, j2_terms))
