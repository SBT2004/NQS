from __future__ import annotations

from typing import TypedDict

import numpy as np
from scipy.sparse import coo_array, csr_array
from scipy.sparse.linalg import eigsh

from .operator import Operator


class ExactGroundStateResult(TypedDict):
    ground_state: np.ndarray
    ground_energy: float


def sparse_operator_matrix(operator: Operator) -> csr_array:
    """Build a sparse CSR matrix for a project-owned operator."""

    hilbert = operator.hilbert
    dimension = hilbert.n_states
    rows: list[int] = []
    columns: list[int] = []
    values: list[complex] = []

    for row_index, column_index, value in operator.iter_matrix_elements():
        rows.append(row_index)
        columns.append(column_index)
        values.append(value)

    matrix = coo_array(
        (np.asarray(values, dtype=np.complex128), (rows, columns)),
        shape=(dimension, dimension),
        dtype=np.complex128,
    ).tocsr()
    matrix.sum_duplicates()
    return matrix


def solve_sparse_ground_state(sparse_matrix: csr_array) -> ExactGroundStateResult:
    """Solve a sparse Hermitian ground state without densifying the matrix."""

    eigenvalues, eigenvectors = eigsh(sparse_matrix, k=1, which="SA")
    ground_energy = float(eigenvalues[0].real)
    ground_state = np.asarray(eigenvectors[:, 0], dtype=np.complex128)
    dominant_amplitude = ground_state[np.argmax(np.abs(ground_state))]
    if dominant_amplitude != 0:
        ground_state *= np.exp(-1j * np.angle(dominant_amplitude))
    return {
        "ground_state": ground_state,
        "ground_energy": ground_energy,
    }


def exact_ground_state(operator: Operator) -> ExactGroundStateResult:
    """Return the sparse-solver ground-state result for production ED paths."""

    return solve_sparse_ground_state(sparse_operator_matrix(operator))


def exact_ground_state_energy(operator: Operator) -> float:
    return float(exact_ground_state(operator)["ground_energy"])


__all__ = [
    "ExactGroundStateResult",
    "exact_ground_state",
    "exact_ground_state_energy",
    "solve_sparse_ground_state",
    "sparse_operator_matrix",
]
