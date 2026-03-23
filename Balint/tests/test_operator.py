import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nqs.exact_diag import exact_ground_state, exact_ground_state_energy, sparse_operator_matrix
from nqs.exact_diag_debug import dense_debug_operator_matrix
from nqs.graph import SquareLattice
from nqs.hilbert import SpinHilbert
from nqs.operator import LocalTerm, Operator, heisenberg_term, j1_j2, local_matrix, sigmax, sigmaz, sx_term, tfim


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

    def test_builtin_sx_term_matches_matrix_fallback_for_all_basis_states(self) -> None:
        hilbert = SpinHilbert(4)
        fast_operator = Operator(hilbert, [sx_term(2, coefficient=1.5)])
        matrix_operator = Operator(hilbert, [LocalTerm((2,), sigmax(), coefficient=1.5)])

        for state_bits in range(hilbert.n_states):
            with self.subTest(state_bits=state_bits):
                self.assertEqual(
                    fast_operator.connected_elements_bits(state_bits),
                    matrix_operator.connected_elements_bits(state_bits),
                )

    def test_bitmap_connected_elements_match_array_path(self) -> None:
        hilbert = SpinHilbert(4)
        op = Operator(
            hilbert,
            [
                LocalTerm((0,), sigmax(), coefficient=2.0),
                LocalTerm((1, 3), local_matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])),
            ],
        )
        state = np.array([1, 0, 0, 1], dtype=np.uint8)
        state_bits = hilbert.state_to_index(state)

        connected_arrays = [(hilbert.state_to_index(connected_state), value) for connected_state, value in op.connected_elements(state)]
        connected_bits = op.connected_elements_bits(state_bits)

        self.assertEqual(connected_bits, connected_arrays)

    def test_bitmap_connected_elements_reject_out_of_range_state(self) -> None:
        hilbert = SpinHilbert(3)
        op = Operator(hilbert, [LocalTerm((0,), sigmax())])

        with self.assertRaises(ValueError):
            op.connected_elements_bits(8)

    def test_batched_connected_elements_match_per_sample_contract(self) -> None:
        hilbert = SpinHilbert(4)
        op = Operator(
            hilbert,
            [
                LocalTerm((0,), sigmax(), coefficient=2.0),
                LocalTerm((1, 3), local_matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])),
            ],
        )
        samples = np.array(
            [
                [1, 0, 0, 1],
                [0, 1, 1, 0],
            ],
            dtype=np.uint8,
        )

        batched = op.connected_elements_batched(samples)
        actual = [
            (int(sample_index), tuple(state.tolist()), complex(value))
            for sample_index, state, value in zip(
                batched.sample_indices,
                batched.connected_states,
                batched.coefficients,
            )
        ]
        expected = [
            (sample_index, tuple(connected_state.tolist()), complex(value))
            for sample_index, sample in enumerate(samples)
            for connected_state, value in op.connected_elements(sample)
        ]

        self.assertCountEqual(actual, expected)

    def test_reject_repeated_sites(self) -> None:
        with self.assertRaises(ValueError):
            LocalTerm((1, 1), np.eye(4, dtype=np.complex128))

    def test_reject_wrong_matrix_shape(self) -> None:
        with self.assertRaises(ValueError):
            LocalTerm((0, 1), np.eye(2, dtype=np.complex128))

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

    def test_heisenberg_fast_path_matches_matrix_fallback_for_all_basis_states(self) -> None:
        hilbert = SpinHilbert(3)
        fast_term = heisenberg_term(0, 2, coefficient=0.75)
        fast_operator = Operator(hilbert, [fast_term])
        matrix_operator = Operator(hilbert, [LocalTerm(fast_term.sites, fast_term.matrix, coefficient=fast_term.coefficient)])

        for state_bits in range(hilbert.n_states):
            with self.subTest(state_bits=state_bits):
                self.assertEqual(
                    fast_operator.connected_elements_bits(state_bits),
                    matrix_operator.connected_elements_bits(state_bits),
                )

    def test_mixed_fast_and_matrix_batched_paths_match_matrix_only_reference(self) -> None:
        hilbert = SpinHilbert(4)
        fast_heisenberg = heisenberg_term(1, 2, coefficient=0.5)
        swap = local_matrix(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ]
        )
        fast_operator = Operator(
            hilbert,
            [
                sx_term(0, coefficient=-0.7),
                fast_heisenberg,
                LocalTerm((0, 3), swap, coefficient=0.25),
            ],
        )
        matrix_operator = Operator(
            hilbert,
            [
                LocalTerm((0,), sigmax(), coefficient=-0.7),
                LocalTerm(fast_heisenberg.sites, fast_heisenberg.matrix, coefficient=fast_heisenberg.coefficient),
                LocalTerm((0, 3), swap, coefficient=0.25),
            ],
        )
        samples = np.array(
            [
                [0, 0, 1, 1],
                [1, 1, 0, 0],
                [1, 0, 1, 0],
            ],
            dtype=np.uint8,
        )

        fast_batched = fast_operator.connected_elements_batched(samples)
        matrix_batched = matrix_operator.connected_elements_batched(samples)
        fast_entries = [
            (int(sample_index), tuple(state.tolist()), complex(value))
            for sample_index, state, value in zip(
                fast_batched.sample_indices,
                fast_batched.connected_states,
                fast_batched.coefficients,
            )
        ]
        matrix_entries = [
            (int(sample_index), tuple(state.tolist()), complex(value))
            for sample_index, state, value in zip(
                matrix_batched.sample_indices,
                matrix_batched.connected_states,
                matrix_batched.coefficients,
            )
        ]

        self.assertCountEqual(fast_entries, matrix_entries)

    def test_exact_diag_matrix_matches_operator_action(self) -> None:
        hilbert = SpinHilbert(2)
        operator = Operator(hilbert, [LocalTerm((0,), sigmax()), LocalTerm((1,), sigmaz())])

        matrix = dense_debug_operator_matrix(operator)
        expected = np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, -1, 1],
                [0, 0, 1, -1],
            ],
            dtype=np.complex128,
        )

        np.testing.assert_allclose(matrix, expected)

    def test_sparse_exact_diag_matrix_matches_dense_helper(self) -> None:
        hilbert = SpinHilbert(3)
        operator = Operator(
            hilbert,
            [
                LocalTerm((0,), sigmax(), coefficient=-0.7),
                LocalTerm((1, 2), np.kron(sigmaz(), sigmaz()), coefficient=1.3),
            ],
        )

        dense_matrix = dense_debug_operator_matrix(operator)
        sparse_matrix = sparse_operator_matrix(operator)

        np.testing.assert_allclose(sparse_matrix.toarray(), dense_matrix)
        np.testing.assert_allclose(sparse_matrix.toarray(), sparse_matrix.toarray().conj().T)

    def test_sparse_exact_ground_state_matches_dense_reference(self) -> None:
        hilbert = SpinHilbert(3)
        graph = SquareLattice(3, 1, pbc=False)
        operator = tfim(hilbert, graph, J=1.0, h=0.8)

        expected_eigenvalues, expected_eigenvectors = np.linalg.eigh(dense_debug_operator_matrix(operator))
        result = exact_ground_state(operator)

        self.assertAlmostEqual(result["ground_energy"], float(expected_eigenvalues[0].real))
        overlap = np.vdot(expected_eigenvectors[:, 0], result["ground_state"])
        self.assertAlmostEqual(float(np.abs(overlap)), 1.0, places=10)
        self.assertAlmostEqual(float(np.linalg.norm(result["ground_state"])), 1.0, places=10)

    def test_sparse_exact_ground_energies_match_dense_reference_on_small_tfim_systems(self) -> None:
        cases = (
            ((3, 1), False, 0.8),
            ((4, 1), False, 1.0),
            ((2, 2), True, 0.5),
        )

        for lattice_shape, pbc, field_strength in cases:
            with self.subTest(lattice_shape=lattice_shape, pbc=pbc, h=field_strength):
                graph = SquareLattice(*lattice_shape, pbc=pbc)
                hilbert = SpinHilbert(graph.n_nodes)
                operator = tfim(hilbert, graph, J=1.0, h=field_strength)

                expected = float(np.linalg.eigvalsh(dense_debug_operator_matrix(operator))[0].real)
                actual = exact_ground_state_energy(operator)

                self.assertAlmostEqual(actual, expected, places=10)


if __name__ == "__main__":
    unittest.main()
