from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
from typing import Any, Iterable, cast

import netket as nk
import numpy as np

from .graph import Graph, SquareLattice
from .hilbert import SpinHilbert


def identity() -> np.ndarray:
    return np.eye(2, dtype=np.complex128)


def sigmax() -> np.ndarray:
    return np.array([[0, 1], [1, 0]], dtype=np.complex128)


def sigmay() -> np.ndarray:
    return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)


def sigmaz() -> np.ndarray:
    return np.array([[1, 0], [0, -1]], dtype=np.complex128)


def projector_zero() -> np.ndarray:
    return np.array([[1, 0], [0, 0]], dtype=np.complex128)


def projector_one() -> np.ndarray:
    return np.array([[0, 0], [0, 1]], dtype=np.complex128)


def local_matrix(matrix: np.ndarray | list[list[complex]]) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.complex128)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Local matrices must be square.")
    return arr


def kron_product(*matrices: np.ndarray | list[list[complex]]) -> np.ndarray:
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
    matrix: np.ndarray
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

    def __init__(self, hilbert: SpinHilbert, terms: list[LocalTerm] | tuple[LocalTerm, ...]) -> None:
        self.hilbert = hilbert
        self.terms = tuple(terms)
        self._validate_terms()

    def connected_elements(
        self,
        state: np.ndarray | list[int] | tuple[int, ...],
    ) -> list[tuple[np.ndarray, complex]]:
        sigma = self.hilbert.validate_state(state)
        contributions: dict[int, complex] = {}

        for term in self.terms:
            local_index = self._local_index(sigma, term.sites)
            column = term.matrix[:, local_index]
            nonzero_rows = np.flatnonzero(np.abs(column) > 0)

            for row in nonzero_rows:
                sigma_prime = sigma.copy()
                self._write_local_state(sigma_prime, term.sites, int(row))
                global_index = self.hilbert.state_to_index(sigma_prime)
                value = complex(term.coefficient * column[row])
                contributions[global_index] = contributions.get(global_index, 0.0) + value

        return [
            (self.hilbert.index_to_state(index), value)
            for index, value in sorted(contributions.items(), key=lambda item: item[0])
            if value != 0
        ]

    def _validate_terms(self) -> None:
        for term in self.terms:
            if any(site < 0 or site >= self.hilbert.size for site in term.sites):
                raise ValueError("LocalTerm site is outside the Hilbert space.")

    def to_netket(self) -> nk.operator.LocalOperator:
        netket_hilbert = nk.hilbert.Spin(s=0.5, N=self.hilbert.size)
        operators = cast(list[Any], [np.asarray(term.matrix, dtype=np.complex128) * term.coefficient for term in self.terms])
        acting_on = [list(term.sites) for term in self.terms]
        return nk.operator.LocalOperator(netket_hilbert, operators=operators, acting_on=acting_on)

    @staticmethod
    def _local_index(state: np.ndarray, sites: tuple[int, ...]) -> int:
        return sum(int(state[site]) << offset for offset, site in enumerate(sites))

    @staticmethod
    def _write_local_state(state: np.ndarray, sites: tuple[int, ...], local_index: int) -> None:
        for offset, site in enumerate(sites):
            state[site] = (local_index >> offset) & 1


def tfim(hilbert: SpinHilbert, graph: Graph, J: float, h: float) -> Operator:
    zz_terms = [szsz_term(edge.i, edge.j, coefficient=-J) for edge in graph.iter_edges("J", n=1)]
    x_terms = [sx_term(site, coefficient=-h) for site in range(hilbert.size)]
    return Operator(hilbert, collect_terms(zz_terms, x_terms))


def j1_j2(hilbert: SpinHilbert, graph: SquareLattice, J1: float, J2: float) -> Operator:
    j1_terms = [heisenberg_term(edge.i, edge.j, coefficient=J1) for edge in graph.iter_edges("J1", n=1)]
    j2_terms = [heisenberg_term(edge.i, edge.j, coefficient=J2) for edge in graph.iter_edges("J2", n=2)]
    return Operator(hilbert, collect_terms(j1_terms, j2_terms))
