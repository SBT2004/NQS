from __future__ import annotations

import sys
import unittest
from pathlib import Path

import netket as nk
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from demos.netket_reference import (  # noqa: E402
    build_netket_j1j2_operator,
    build_netket_tfim_operator,
    exact_netket_j1j2_ground_energy,
    exact_netket_tfim_ground_energy,
    exact_project_operator_ground_energy,
)
from nqs.graph import SquareLattice  # noqa: E402
from nqs.hilbert import SpinHilbert  # noqa: E402
from nqs.operator import j1_j2, tfim  # noqa: E402


class NetKetCompatibilityTests(unittest.TestCase):
    def test_operator_to_netket_returns_local_operator(self) -> None:
        graph = SquareLattice(2, 2, pbc=False)
        hilbert = SpinHilbert(graph.n_nodes)
        operator = j1_j2(hilbert, graph, J1=1.0, J2=0.5)

        adapted = operator.to_netket()

        self.assertIsInstance(adapted, nk.operator.LocalOperator)

    def test_j1j2_reference_helper_builds_netket_operator_and_energy(self) -> None:
        graph = SquareLattice(2, 2, pbc=False)

        operator = build_netket_j1j2_operator(graph, J1=1.0, J2=0.5)
        energy = exact_netket_j1j2_ground_energy(lattice_shape=(2, 2), pbc=False, J1=1.0, J2=0.5)

        self.assertIsInstance(operator, nk.operator.LocalOperator)
        self.assertTrue(np.isfinite(energy))

    def test_project_operator_reference_energy_matches_native_netket_builder(self) -> None:
        graph = SquareLattice(2, 2, pbc=False)
        hilbert = SpinHilbert(graph.n_nodes)
        operator = j1_j2(hilbert, graph, J1=1.0, J2=0.5)

        project_energy = exact_project_operator_ground_energy(operator)
        native_energy = exact_netket_j1j2_ground_energy(lattice_shape=(2, 2), pbc=False, J1=1.0, J2=0.5)

        self.assertAlmostEqual(project_energy, native_energy, places=10)

    def test_project_tfim_reference_energy_matches_native_netket_builder(self) -> None:
        graph = SquareLattice(2, 2, pbc=False)
        hilbert = SpinHilbert(graph.n_nodes)
        operator = tfim(hilbert, graph, J=1.0, h=0.8)

        project_energy = exact_project_operator_ground_energy(operator)
        native_energy = exact_netket_tfim_ground_energy(lattice_shape=(2, 2), pbc=False, J=1.0, h=0.8)

        self.assertAlmostEqual(project_energy, native_energy, places=10)

    def test_tfim_reference_helper_builds_netket_operator_and_energy(self) -> None:
        graph = SquareLattice(2, 2, pbc=False)

        operator = build_netket_tfim_operator(graph=graph, J=1.0, h=0.8)
        energy = exact_netket_tfim_ground_energy(lattice_shape=(2, 2), pbc=False, J=1.0, h=0.8)

        self.assertTrue(hasattr(operator, "to_sparse"))
        self.assertTrue(np.isfinite(energy))


if __name__ == "__main__":
    unittest.main()
