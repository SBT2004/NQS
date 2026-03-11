from __future__ import annotations

from dataclasses import dataclass
from numbers import Number

import numpy as np

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

    @staticmethod
    def _local_index(state: np.ndarray, sites: tuple[int, ...]) -> int:
        return sum(int(state[site]) << offset for offset, site in enumerate(sites))

    @staticmethod
    def _write_local_state(state: np.ndarray, sites: tuple[int, ...], local_index: int) -> None:
        for offset, site in enumerate(sites):
            state[site] = (local_index >> offset) & 1
