"""Explicit NetKet-only comparison helpers for the user-facing benchmark notebook."""

from __future__ import annotations

from typing import Any, cast

import netket as nk
import numpy as np

from nqs import Operator, SpinHilbert, SquareLattice


def _heisenberg_matrix() -> np.ndarray:
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    return np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz)


def build_netket_tfim_operator(
    *,
    graph: SquareLattice,
    J: float,
    h: float,
) -> Any:
    nk_hilbert = nk.hilbert.Spin(s=0.5, N=graph.n_nodes)
    nk_graph = nk.graph.Grid(extent=[graph.Lx, graph.Ly], pbc=graph.pbc)
    ising_ctor = cast(Any, getattr(nk.operator, "IsingJax"))
    return ising_ctor(hilbert=nk_hilbert, graph=nk_graph, h=h, J=-J)


def build_netket_j1j2_operator(
    graph: SquareLattice,
    *,
    J1: float,
    J2: float,
) -> nk.operator.LocalOperator:
    """Build a NetKet LocalOperator for explicit comparison-only checks."""

    nk_hilbert = nk.hilbert.Spin(s=0.5, N=graph.n_nodes)
    heisenberg_matrix = _heisenberg_matrix()
    j1_edges = list(graph.iter_edges("J1", n=1))
    j2_edges = list(graph.iter_edges("J2", n=2))
    native_terms = cast(
        list[Any],
        [heisenberg_matrix * J1 for _ in j1_edges] + [heisenberg_matrix * J2 for _ in j2_edges],
    )
    native_sites = [[edge.i, edge.j] for edge in j1_edges] + [[edge.i, edge.j] for edge in j2_edges]
    return nk.operator.LocalOperator(nk_hilbert, operators=native_terms, acting_on=native_sites)


def exact_netket_ground_energy_from_operator(operator: Any) -> float:
    return float(nk.exact.lanczos_ed(operator, k=1, compute_eigenvectors=False)[0])


def exact_netket_tfim_ground_energy(
    *,
    lattice_shape: tuple[int, int],
    pbc: bool,
    J: float,
    h: float,
) -> float:
    graph = SquareLattice(lattice_shape[0], lattice_shape[1], pbc=pbc)
    return exact_netket_ground_energy_from_operator(build_netket_tfim_operator(graph=graph, J=J, h=h))


def exact_netket_j1j2_ground_energy(
    *,
    lattice_shape: tuple[int, int],
    pbc: bool,
    J1: float,
    J2: float,
) -> float:
    graph = SquareLattice(lattice_shape[0], lattice_shape[1], pbc=pbc)
    return exact_netket_ground_energy_from_operator(build_netket_j1j2_operator(graph, J1=J1, J2=J2))


def exact_project_operator_ground_energy(operator: Operator) -> float:
    return float(nk.exact.lanczos_ed(operator.to_netket(), k=1, compute_eigenvectors=False)[0])


def build_project_j1j2_system(
    *,
    lattice_shape: tuple[int, int],
    pbc: bool,
    J1: float,
    J2: float,
) -> tuple[SpinHilbert, SquareLattice, Operator]:
    from nqs import j1_j2

    graph = SquareLattice(lattice_shape[0], lattice_shape[1], pbc=pbc)
    hilbert = SpinHilbert(graph.n_nodes)
    return hilbert, graph, j1_j2(hilbert, graph, J1=J1, J2=J2)


__all__ = [
    "build_netket_j1j2_operator",
    "build_netket_tfim_operator",
    "build_project_j1j2_system",
    "exact_netket_ground_energy_from_operator",
    "exact_netket_j1j2_ground_energy",
    "exact_netket_tfim_ground_energy",
    "exact_project_operator_ground_energy",
]
