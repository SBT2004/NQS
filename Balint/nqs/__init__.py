"""Compatibility package that re-exports the implementation from ``src.nqs``.

Demo notebooks live outside ``src/``, so this shim makes ``import nqs`` work
from the Balint workspace root without relying on IDE-specific source-root
configuration.
"""

from src.nqs import Adam
from src.nqs import CNN
from src.nqs import FFNN
from src.nqs import Chain1D
from src.nqs import Edge
from src.nqs import Graph
from src.nqs import LocalTerm
from src.nqs import NetKetSampler
from src.nqs import Operator
from src.nqs import RBM
from src.nqs import SpinHilbert
from src.nqs import SquareLattice
from src.nqs import VMC
from src.nqs import VariationalState
from src.nqs import energy_loss
from src.nqs import identity
from src.nqs import local_matrix
from src.nqs import projector_one
from src.nqs import projector_zero
from src.nqs import sigmax
from src.nqs import sigmay
from src.nqs import sigmaz
from src.nqs import states_from_netket
from src.nqs import states_to_netket

__all__ = [
    "Adam",
    "CNN",
    "FFNN",
    "Edge",
    "Graph",
    "Chain1D",
    "LocalTerm",
    "NetKetSampler",
    "Operator",
    "RBM",
    "SpinHilbert",
    "SquareLattice",
    "VMC",
    "VariationalState",
    "energy_loss",
    "identity",
    "local_matrix",
    "projector_one",
    "projector_zero",
    "sigmax",
    "sigmay",
    "sigmaz",
    "states_from_netket",
    "states_to_netket",
]
