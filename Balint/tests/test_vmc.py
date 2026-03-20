import sys
import unittest
from pathlib import Path

import jax
import netket as nk
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nqs import (
    Adam,
    CNN,
    FFNN,
    RBM,
    NetKetSampler,
    SpinHilbert,
    VMC,
    VariationalState,
    energy_loss,
    states_from_netket,
    states_to_netket,
)
from nqs.graph import SquareLattice
from nqs.operator import Operator, collect_terms, j1_j2, sx_term, szsz_term


def _tfim_operator(hilbert: SpinHilbert, graph: SquareLattice, J: float = 1.0, h: float = 0.8) -> Operator:
    interaction_terms = [szsz_term(edge.i, edge.j, coefficient=-J) for edge in graph.iter_edges("J1", n=1)]
    field_terms = [sx_term(site, coefficient=-h) for site in range(hilbert.size)]
    return Operator(hilbert, collect_terms(interaction_terms, field_terms))


def _exact_ground_energy(project_operator: Operator) -> float:
    hilbert = project_operator.hilbert
    dimension = hilbert.n_states
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    for column_index, state in enumerate(hilbert.all_states()):
        for connected_state, value in project_operator.connected_elements(state):
            row_index = hilbert.state_to_index(connected_state)
            matrix[row_index, column_index] += value
    return float(np.min(np.linalg.eigvalsh(matrix).real))


class ModelTests(unittest.TestCase):
    def test_netket_state_conversion_roundtrip(self) -> None:
        states = np.array([[0, 1, 0, 1], [1, 1, 0, 0]], dtype=np.uint8)
        jax_states = jax.numpy.asarray(states)
        netket_states = states_to_netket(jax_states)
        np.testing.assert_array_equal(np.asarray(netket_states), np.array([[-1.0, 1.0, -1.0, 1.0], [1.0, 1.0, -1.0, -1.0]]))
        np.testing.assert_array_equal(np.asarray(states_from_netket(netket_states)), states)

    def test_rbm_log_psi_shape(self) -> None:
        hilbert = SpinHilbert(4)
        model = RBM(alpha=2)
        params = model.init(jax.random.PRNGKey(0), hilbert)
        values = model.log_psi(params, np.array([[0, 1, 0, 1], [1, 1, 0, 0]], dtype=np.uint8))
        self.assertEqual(values.shape, (2,))

    def test_ffnn_log_psi_shape(self) -> None:
        hilbert = SpinHilbert(4)
        model = FFNN(hidden_dims=(8, 4))
        params = model.init(jax.random.PRNGKey(1), hilbert)
        values = model.log_psi(params, np.array([[0, 1, 0, 1]], dtype=np.uint8))
        self.assertEqual(values.shape, (1,))

    def test_cnn_log_psi_shape(self) -> None:
        hilbert = SpinHilbert(16)
        model = CNN(spatial_shape=(4, 4))
        params = model.init(jax.random.PRNGKey(2), hilbert)
        values = model.log_psi(params, np.zeros((2, 16), dtype=np.uint8))
        self.assertEqual(values.shape, (2,))


class VMCTests(unittest.TestCase):
    def _make_sampler(self, hilbert: SpinHilbert, seed: int) -> NetKetSampler:
        return NetKetSampler(hilbert=hilbert, n_samples=16, n_discard_per_chain=2, n_chains=4, seed=seed)

    def test_jax_gradient_matches_param_structure(self) -> None:
        hilbert = SpinHilbert(4)
        model = RBM(alpha=1)
        params = model.init(jax.random.PRNGKey(0), hilbert)
        sampler = self._make_sampler(hilbert, seed=0)
        state = VariationalState(model=model, params=params, sampler=sampler)
        nk_hilbert = sampler.netket_hilbert
        operator = nk.operator.IsingJax(  # pyright: ignore[reportCallIssue]
            hilbert=nk_hilbert,
            graph=nk.graph.Chain(length=4, pbc=False),
            h=1.0,
        )
        optimizer = Adam(learning_rate=1e-2)

        loss_value, grads = optimizer.compute_gradients(
            lambda current_params: energy_loss(state, operator, current_params),
            state.parameters,
        )

        self.assertTrue(np.isfinite(np.asarray(loss_value)))
        self.assertEqual(set(grads.keys()), set(state.parameters.keys()))

    def test_variational_state_and_driver_update_parameters(self) -> None:
        hilbert = SpinHilbert(4)
        model = RBM(alpha=1)
        params = model.init(jax.random.PRNGKey(3), hilbert)
        sampler = self._make_sampler(hilbert, seed=3)
        state = VariationalState(model=model, params=params, sampler=sampler)
        operator = nk.operator.IsingJax(  # pyright: ignore[reportCallIssue]
            hilbert=sampler.netket_hilbert,
            graph=nk.graph.Chain(length=4, pbc=False),
            h=1.0,
        )
        driver = VMC(operator=operator, variational_state=state, optimizer=Adam(learning_rate=1e-2))

        before = jax.tree_util.tree_leaves(state.parameters)
        history = driver.run(2)
        after = jax.tree_util.tree_leaves(state.parameters)

        self.assertEqual(len(history), 2)
        self.assertIn("energy", history[0])
        self.assertTrue(any(not np.allclose(np.asarray(a), np.asarray(b)) for a, b in zip(before, after)))

    def test_small_system_exact_backend_matches_ed_to_within_target(self) -> None:
        hilbert = SpinHilbert(4)
        graph = SquareLattice(2, 2, pbc=True)
        project_operator = _tfim_operator(hilbert, graph)
        exact_energy = _exact_ground_energy(project_operator)
        model = RBM(alpha=2)
        params = model.init(jax.random.PRNGKey(7), hilbert)
        sampler = NetKetSampler(hilbert=hilbert, n_samples=128, n_discard_per_chain=8, n_chains=8, seed=7)
        state = VariationalState(model=model, params=params, sampler=sampler)
        driver = VMC(
            operator=project_operator.to_netket(),
            variational_state=state,
            optimizer=Adam(learning_rate=1e-2),
        )

        for _ in range(120):
            driver.step()

        optimized_energy = float(np.asarray(state.energy(project_operator.to_netket())))
        self.assertLess(abs(optimized_energy - exact_energy), 0.01)
        self.assertEqual(np.asarray(state.exact_statevector()).shape, (hilbert.n_states,))

    def test_j1j2_architectures_match_ed_with_phase_capable_ansatze(self) -> None:
        hilbert = SpinHilbert(4)
        graph = SquareLattice(2, 2, pbc=True)
        project_operator = j1_j2(hilbert, graph, J1=1.0, J2=0.5)
        exact_energy = _exact_ground_energy(project_operator)
        architecture_runs = (
            ("RBM", RBM(alpha=4), 800, 1e-2),
            ("FFNN", FFNN(hidden_dims=(32, 16)), 400, 1e-2),
            ("CNN", CNN(spatial_shape=(2, 2), channels=(16, 8), kernel_size=(2, 2)), 400, 1e-2),
        )

        for label, model, n_steps, learning_rate in architecture_runs:
            with self.subTest(model=label):
                params = model.init(jax.random.PRNGKey(0), hilbert)
                sampler = NetKetSampler(hilbert=hilbert, n_samples=256, n_discard_per_chain=16, n_chains=8, seed=0)
                state = VariationalState(model=model, params=params, sampler=sampler)
                driver = VMC(
                    operator=project_operator.to_netket(),
                    variational_state=state,
                    optimizer=Adam(learning_rate=learning_rate),
                )

                for _ in range(n_steps):
                    driver.step()

                optimized_energy = float(np.asarray(state.energy(project_operator.to_netket())))
                self.assertLess(abs(optimized_energy - exact_energy), 0.01)


if __name__ == "__main__":
    unittest.main()
