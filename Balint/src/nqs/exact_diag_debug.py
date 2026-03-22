from __future__ import annotations

import numpy as np

from .exact_diag import sparse_operator_matrix
from .operator import Operator


def dense_debug_operator_matrix(operator: Operator) -> np.ndarray:
    """Return a dense Hamiltonian matrix for explicit demo/debug use only."""

    return np.asarray(sparse_operator_matrix(operator).toarray(), dtype=np.complex128)


__all__ = ["dense_debug_operator_matrix"]
