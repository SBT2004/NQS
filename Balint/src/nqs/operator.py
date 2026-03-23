from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from numbers import Number
from typing import TYPE_CHECKING, Iterable, Literal, Protocol, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from .graph import Graph, SquareLattice
from .hilbert import SpinHilbert, SpinState, SpinStateBatch, StateInput

if TYPE_CHECKING:
    import netket as nk
    from netket.utils.types import Array as NetKetArray
else:
    NetKetArray = object


LocalMatrix: TypeAlias = npt.NDArray[np.complex128]
ConnectedElement: TypeAlias = tuple[SpinState, complex]
ConnectedElementBits: TypeAlias = tuple[int, complex]
FastTermKind: TypeAlias = Literal["sx", "szsz", "sxsx", "sysy", "heisenberg"]
SUPPORTED_FAST_TERM_KINDS = frozenset({"sx", "szsz", "sxsx", "sysy", "heisenberg"})


@dataclass(frozen=True)
class BatchedConnectedElements:
    sample_indices: npt.NDArray[np.int64]
    connected_states: SpinStateBatch
    coefficients: npt.NDArray[np.complex128]


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
    return LocalTerm((site,), sigmax(), coefficient=coefficient, _fast_kind="sx")


def szsz_term(i: int, j: int, coefficient: complex = 1.0) -> "LocalTerm":
    return LocalTerm(_pair_sites(i, j), kron_product(sigmaz(), sigmaz()), coefficient=coefficient, _fast_kind="szsz")


def sxsx_term(i: int, j: int, coefficient: complex = 1.0) -> "LocalTerm":
    return LocalTerm(_pair_sites(i, j), kron_product(sigmax(), sigmax()), coefficient=coefficient, _fast_kind="sxsx")


def sysy_term(i: int, j: int, coefficient: complex = 1.0) -> "LocalTerm":
    return LocalTerm(_pair_sites(i, j), kron_product(sigmay(), sigmay()), coefficient=coefficient, _fast_kind="sysy")


def heisenberg_term(i: int, j: int, coefficient: complex = 1.0) -> "LocalTerm":
    matrix = kron_product(sigmax(), sigmax()) + kron_product(sigmay(), sigmay()) + kron_product(sigmaz(), sigmaz())
    return LocalTerm(_pair_sites(i, j), matrix, coefficient=coefficient, _fast_kind="heisenberg")


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
    _fast_kind: FastTermKind | None = field(default=None, repr=False, compare=False)

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
        if self._fast_kind is not None:
            if self._fast_kind not in SUPPORTED_FAST_TERM_KINDS:
                raise ValueError(f"Unsupported LocalTerm fast kind: {self._fast_kind!r}.")
            expected_site_count = 1 if self._fast_kind == "sx" else 2
            if len(self.sites) != expected_site_count:
                raise ValueError(f"LocalTerm fast kind {self._fast_kind!r} requires {expected_site_count} site(s).")


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
            for global_index, value in self._connected_elements_bits_for_term(state_bits, term):
                contributions[global_index] = contributions.get(global_index, 0.0) + value

        return [
            (index, value)
            for index, value in sorted(contributions.items(), key=lambda item: item[0])
            if value != 0
        ]

    def connected_elements_batched(
        self,
        states: StateInput,
    ) -> BatchedConnectedElements:
        state_array = np.asarray(states, dtype=np.uint8)
        if state_array.ndim == 1:
            state_array = state_array.reshape(1, -1)
        if state_array.ndim != 2 or state_array.shape[1] != self.hilbert.size:
            raise ValueError(f"Expected shape (batch, {self.hilbert.size}), got {state_array.shape}.")
        if np.any((state_array != 0) & (state_array != 1)):
            raise ValueError("Spin states must only contain 0 or 1.")

        sample_indices = np.arange(state_array.shape[0], dtype=np.int64)
        connected_states_chunks: list[SpinStateBatch] = []
        sample_index_chunks: list[npt.NDArray[np.int64]] = []
        coefficient_chunks: list[npt.NDArray[np.complex128]] = []

        for term in self.terms:
            term_sample_indices, term_connected_states, term_coefficients = self._connected_elements_batched_for_term(
                state_array,
                sample_indices,
                term,
            )
            if term_sample_indices.size == 0:
                continue
            connected_states_chunks.append(term_connected_states)
            sample_index_chunks.append(term_sample_indices)
            coefficient_chunks.append(term_coefficients)

        if not connected_states_chunks:
            return BatchedConnectedElements(
                sample_indices=np.zeros(0, dtype=np.int64),
                connected_states=np.zeros((0, self.hilbert.size), dtype=np.uint8),
                coefficients=np.zeros(0, dtype=np.complex128),
            )

        return BatchedConnectedElements(
            sample_indices=np.concatenate(sample_index_chunks),
            connected_states=np.concatenate(connected_states_chunks, axis=0),
            coefficients=np.concatenate(coefficient_chunks),
        )

    def iter_matrix_elements(self) -> Iterator[tuple[int, int, complex]]:
        """Yield nonzero matrix elements as ``(row, column, value)`` triples."""

        for column_index in range(self.hilbert.n_states):
            for row_index, value in self.connected_elements_bits(column_index):
                yield row_index, column_index, value

    def _validate_terms(self) -> None:
        for term in self.terms:
            if any(site < 0 or site >= self.hilbert.size for site in term.sites):
                raise ValueError("LocalTerm site is outside the Hilbert space.")

    def _connected_elements_bits_for_term(
        self,
        state_bits: int,
        term: LocalTerm,
    ) -> list[ConnectedElementBits]:
        if term._fast_kind is not None:
            return self._fast_connected_elements_bits(state_bits, term)
        return self._matrix_connected_elements_bits(state_bits, term)

    def _fast_connected_elements_bits(
        self,
        state_bits: int,
        term: LocalTerm,
    ) -> list[ConnectedElementBits]:
        coefficient = complex(term.coefficient)
        if coefficient == 0:
            return []
        if term._fast_kind == "sx":
            site = term.sites[0]
            return [(state_bits ^ (1 << site), coefficient)]

        i, j = term.sites
        bit_i = (state_bits >> i) & 1
        bit_j = (state_bits >> j) & 1
        equal_bits = bit_i == bit_j
        flipped_bits = state_bits ^ ((1 << i) | (1 << j))

        if term._fast_kind == "szsz":
            return [(state_bits, coefficient if equal_bits else -coefficient)]
        if term._fast_kind == "sxsx":
            return [(flipped_bits, coefficient)]
        if term._fast_kind == "sysy":
            return [(flipped_bits, -coefficient if equal_bits else coefficient)]
        if term._fast_kind == "heisenberg":
            elements: list[ConnectedElementBits] = [(state_bits, coefficient if equal_bits else -coefficient)]
            if not equal_bits:
                elements.append((flipped_bits, 2.0 * coefficient))
            return elements
        raise ValueError(f"Unsupported fast-term kind: {term._fast_kind!r}.")

    def _matrix_connected_elements_bits(
        self,
        state_bits: int,
        term: LocalTerm,
    ) -> list[ConnectedElementBits]:
        local_index = self._local_index_bits(state_bits, term.sites)
        column = term.matrix[:, local_index]
        nonzero_rows = np.flatnonzero(np.abs(column) > 0)
        return [
            (
                self._write_local_bits(state_bits, term.sites, int(row)),
                complex(term.coefficient * column[row]),
            )
            for row in nonzero_rows
        ]

    def _connected_elements_batched_for_term(
        self,
        state_array: SpinStateBatch,
        sample_indices: npt.NDArray[np.int64],
        term: LocalTerm,
    ) -> tuple[npt.NDArray[np.int64], SpinStateBatch, npt.NDArray[np.complex128]]:
        if term._fast_kind is not None:
            return self._fast_connected_elements_batched(state_array, sample_indices, term)
        return self._matrix_connected_elements_batched(state_array, sample_indices, term)

    def _fast_connected_elements_batched(
        self,
        state_array: SpinStateBatch,
        sample_indices: npt.NDArray[np.int64],
        term: LocalTerm,
    ) -> tuple[npt.NDArray[np.int64], SpinStateBatch, npt.NDArray[np.complex128]]:
        coefficient = complex(term.coefficient)
        if coefficient == 0:
            return self._empty_batched_connected_elements()
        if term._fast_kind == "sx":
            connected_states = state_array.copy()
            connected_states[:, term.sites[0]] ^= 1
            coefficients = np.full(sample_indices.shape, coefficient, dtype=np.complex128)
            return sample_indices.copy(), connected_states, coefficients

        i, j = term.sites
        equal_mask = state_array[:, i] == state_array[:, j]

        if term._fast_kind == "szsz":
            coefficients = np.where(equal_mask, coefficient, -coefficient).astype(np.complex128, copy=False)
            return sample_indices.copy(), state_array.copy(), coefficients

        if term._fast_kind == "sxsx":
            connected_states = state_array.copy()
            connected_states[:, i] ^= 1
            connected_states[:, j] ^= 1
            coefficients = np.full(sample_indices.shape, coefficient, dtype=np.complex128)
            return sample_indices.copy(), connected_states, coefficients

        if term._fast_kind == "sysy":
            connected_states = state_array.copy()
            connected_states[:, i] ^= 1
            connected_states[:, j] ^= 1
            coefficients = np.where(equal_mask, -coefficient, coefficient).astype(np.complex128, copy=False)
            return sample_indices.copy(), connected_states, coefficients

        if term._fast_kind == "heisenberg":
            diagonal_states = state_array.copy()
            diagonal_coefficients = np.where(equal_mask, coefficient, -coefficient).astype(np.complex128, copy=False)

            differing_mask = ~equal_mask
            if not np.any(differing_mask):
                return sample_indices.copy(), diagonal_states, diagonal_coefficients

            offdiagonal_indices = sample_indices[differing_mask]
            offdiagonal_states = state_array[differing_mask].copy()
            offdiagonal_states[:, i] ^= 1
            offdiagonal_states[:, j] ^= 1
            offdiagonal_coefficients = np.full(offdiagonal_indices.shape, 2.0 * coefficient, dtype=np.complex128)

            return (
                np.concatenate((sample_indices, offdiagonal_indices)),
                np.concatenate((diagonal_states, offdiagonal_states), axis=0),
                np.concatenate((diagonal_coefficients, offdiagonal_coefficients)),
            )

        raise ValueError(f"Unsupported fast-term kind: {term._fast_kind!r}.")

    def _matrix_connected_elements_batched(
        self,
        state_array: SpinStateBatch,
        sample_indices: npt.NDArray[np.int64],
        term: LocalTerm,
    ) -> tuple[npt.NDArray[np.int64], SpinStateBatch, npt.NDArray[np.complex128]]:
        sites = np.asarray(term.sites, dtype=np.intp)
        local_weights = (1 << np.arange(len(term.sites), dtype=np.int64)).reshape(1, -1)
        local_columns = np.sum(
            state_array[:, sites].astype(np.int64, copy=False) * local_weights,
            axis=1,
            dtype=np.int64,
        )

        connected_states_chunks: list[SpinStateBatch] = []
        sample_index_chunks: list[npt.NDArray[np.int64]] = []
        coefficient_chunks: list[npt.NDArray[np.complex128]] = []

        for column_index in range(term.matrix.shape[1]):
            matching_samples = sample_indices[local_columns == column_index]
            if matching_samples.size == 0:
                continue

            column = term.matrix[:, column_index]
            nonzero_rows = np.flatnonzero(np.abs(column) > 0)
            if nonzero_rows.size == 0:
                continue

            repeated_states = np.repeat(state_array[matching_samples], nonzero_rows.size, axis=0)
            repeated_sample_indices = np.repeat(matching_samples, nonzero_rows.size)
            local_row_bits = ((nonzero_rows[:, None] >> np.arange(len(term.sites))) & 1).astype(np.uint8)
            repeated_states[:, sites] = np.tile(local_row_bits, (matching_samples.size, 1))
            repeated_coefficients = np.tile(
                np.asarray(term.coefficient * column[nonzero_rows], dtype=np.complex128),
                matching_samples.size,
            )

            connected_states_chunks.append(repeated_states)
            sample_index_chunks.append(repeated_sample_indices)
            coefficient_chunks.append(repeated_coefficients)

        if not connected_states_chunks:
            return self._empty_batched_connected_elements()

        return (
            np.concatenate(sample_index_chunks),
            np.concatenate(connected_states_chunks, axis=0),
            np.concatenate(coefficient_chunks),
        )

    def _empty_batched_connected_elements(
        self,
    ) -> tuple[npt.NDArray[np.int64], SpinStateBatch, npt.NDArray[np.complex128]]:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros((0, self.hilbert.size), dtype=np.uint8),
            np.zeros(0, dtype=np.complex128),
        )

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
