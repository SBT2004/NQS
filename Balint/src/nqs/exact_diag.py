from __future__ import annotations

from typing import TypedDict

import numpy as np

from .operator import Operator


class ExactDiagResult(TypedDict):
    matrix: np.ndarray
    eigenvalues: np.ndarray
    ground_state: np.ndarray
    ground_energy: float


def operator_matrix(operator: Operator) -> np.ndarray:
    """Build the dense matrix for a small project-owned operator.

    This scales exponentially with Hilbert-space size, so it is only suitable
    for small-system exact-diagonalization checks.
    """

    hilbert = operator.hilbert
    dimension = hilbert.n_states
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    for column_index in range(dimension):
        for row_index, value in operator.connected_elements_bits(column_index):
            matrix[row_index, column_index] += value
    return matrix


def exact_ground_state(operator: Operator) -> ExactDiagResult:
    matrix = operator_matrix(operator)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(eigenvalues.real)
    ordered_values = eigenvalues[order]
    ordered_vectors = eigenvectors[:, order]
    ground_state = ordered_vectors[:, 0]
    return {
        "matrix": matrix,
        "eigenvalues": ordered_values.real,
        "ground_state": ground_state,
        "ground_energy": float(ordered_values[0].real),
    }


def exact_ground_state_energy(operator: Operator) -> float:
    return float(exact_ground_state(operator)["ground_energy"])
