import sys
import unittest
from pathlib import Path

import netket as nk
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.nqs import LocalTerm, Operator, SpinHilbert, SquareLattice, heisenberg_term, j1_j2, local_matrix, sigmax, sigmaz
from src.nqs.operator import tfim


class OperatorTests(unittest.TestCase):
    def test_single_site_sigmax_flips_only_target_bit(self) -> None:
        hilbert = SpinHilbert(4)
        op = Operator(hilbert, [LocalTerm((3,), sigmax())])

        connected = op.connected_elements([0, 1, 0, 1])

        self.assertEqual(len(connected), 1)
        np.testing.assert_array_equal(connected[0][0], np.array([0, 1, 0, 0], dtype=np.uint8))
        self.assertEqual(connected[0][1], 1.0)

    def test_single_site_diagonal_operator_returns_same_state(self) -> None:
        hilbert = SpinHilbert(4)
        op = Operator(hilbert, [LocalTerm((1,), sigmaz())])

        connected = op.connected_elements([0, 1, 0, 1])

        self.assertEqual(len(connected), 1)
        np.testing.assert_array_equal(connected[0][0], np.array([0, 1, 0, 1], dtype=np.uint8))
        self.assertEqual(connected[0][1], -1.0)

    def test_two_site_operator_acts_only_on_its_support(self) -> None:
        hilbert = SpinHilbert(4)
        swap = local_matrix(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ]
        )
        op = Operator(hilbert, [LocalTerm((1, 3), swap)])

        connected = op.connected_elements([1, 0, 0, 1])

        self.assertEqual(len(connected), 1)
        np.testing.assert_array_equal(connected[0][0], np.array([1, 1, 0, 0], dtype=np.uint8))
        self.assertEqual(connected[0][1], 1.0)

    def test_sum_of_terms_merges_connected_outputs(self) -> None:
        hilbert = SpinHilbert(3)
        op = Operator(
            hilbert,
            [
                LocalTerm((0,), sigmax(), coefficient=2.0),
                LocalTerm((0,), sigmax(), coefficient=-0.5),
            ],
        )

        connected = op.connected_elements([0, 1, 1])

        self.assertEqual(len(connected), 1)
        np.testing.assert_array_equal(connected[0][0], np.array([1, 1, 1], dtype=np.uint8))
        self.assertEqual(connected[0][1], 1.5)

    def test_reject_repeated_sites(self) -> None:
        with self.assertRaises(ValueError):
            LocalTerm((1, 1), np.eye(4))

    def test_reject_wrong_matrix_shape(self) -> None:
        with self.assertRaises(ValueError):
            LocalTerm((0, 1), np.eye(2))

    def test_reject_site_outside_hilbert_space(self) -> None:
        hilbert = SpinHilbert(2)
        with self.assertRaises(ValueError):
            Operator(hilbert, [LocalTerm((2,), sigmax())])

    def test_heisenberg_term_matches_expected_two_site_matrix(self) -> None:
        term = heisenberg_term(0, 1, coefficient=2.0)
        sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        expected = np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz)

        self.assertEqual(term.sites, (0, 1))
        np.testing.assert_allclose(term.matrix, expected)
        self.assertEqual(term.coefficient, 2.0)

    def test_heisenberg_term_rejects_repeated_site(self) -> None:
        with self.assertRaises(ValueError):
            heisenberg_term(1, 1)

    def test_j1_j2_builder_creates_expected_term_counts(self) -> None:
        graph = SquareLattice(2, 2, pbc=False)
        hilbert = SpinHilbert(graph.n_nodes)

        operator = j1_j2(hilbert, graph, J1=1.0, J2=0.5)

        self.assertEqual(len(operator.terms), 6)
        self.assertEqual(sum(1 for term in operator.terms if term.coefficient == 1.0), 4)
        self.assertEqual(sum(1 for term in operator.terms if term.coefficient == 0.5), 2)
        self.assertEqual({term.sites for term in operator.terms}, {(0, 1), (0, 2), (1, 3), (2, 3), (0, 3), (1, 2)})

    def test_tfim_builder_creates_nearest_neighbor_and_field_terms(self) -> None:
        graph = SquareLattice(2, 2, pbc=False)
        hilbert = SpinHilbert(graph.n_nodes)

        operator = tfim(hilbert, graph, J=1.5, h=0.25)

        self.assertEqual(len(operator.terms), 8)
        self.assertEqual(sum(1 for term in operator.terms if term.coefficient == -1.5), 4)
        self.assertEqual(sum(1 for term in operator.terms if term.coefficient == -0.25), 4)
        self.assertEqual(sum(1 for term in operator.terms if len(term.sites) == 2), 4)
        self.assertEqual(sum(1 for term in operator.terms if len(term.sites) == 1), 4)

    def test_operator_to_netket_returns_local_operator(self) -> None:
        graph = SquareLattice(2, 2, pbc=False)
        hilbert = SpinHilbert(graph.n_nodes)
        operator = j1_j2(hilbert, graph, J1=1.0, J2=0.5)

        adapted = operator.to_netket()

        self.assertIsInstance(adapted, nk.operator.LocalOperator)


if __name__ == "__main__":
    unittest.main()
