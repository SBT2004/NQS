import sys
import unittest
from pathlib import Path
from typing import cast

import jax
import netket as nk
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.nqs import (
    Adam,
    CNN,
    FFNN,
    RBM,
    NetKetSampler,
    SpinHilbert,
    VMC,
    build_variational_state,
    build_vmc_driver,
    energy_loss,
    exact_ground_state_energy,
    states_from_netket,
    states_to_netket,
)
from src.nqs import Operator, SquareLattice, collect_terms, j1_j2, sx_term, szsz_term


def _tfim_operator(hilbert: SpinHilbert, graph: SquareLattice, J: float = 1.0, h: float = 0.8) -> Operator:
    interaction_terms = [szsz_term(edge.i, edge.j, coefficient=-J) for edge in graph.iter_edges("J1", n=1)]
    field_terms = [sx_term(site, coefficient=-h) for site in range(hilbert.size)]
    return Operator(hilbert, collect_terms(interaction_terms, field_terms))


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

    def _make_driver(self, hilbert: SpinHilbert, seed: int) -> VMC:
        model = RBM(alpha=1)
        sampler = self._make_sampler(hilbert, seed=seed)
        operator = nk.operator.IsingJax(  # pyright: ignore[reportCallIssue]
            hilbert=sampler.netket_hilbert,
            graph=nk.graph.Chain(length=hilbert.size, pbc=False),
            h=1.0,
        )
        _, driver = build_vmc_driver(
            model=model,
            hilbert=hilbert,
            operator=operator,
            learning_rate=1e-2,
            seed=seed,
            n_samples=16,
            n_discard_per_chain=2,
            n_chains=4,
        )
        return driver

    def test_jax_gradient_matches_param_structure(self) -> None:
        hilbert = SpinHilbert(4)
        model = RBM(alpha=1)
        state = build_variational_state(
            model=model,
            hilbert=hilbert,
            seed=0,
            n_samples=16,
            n_discard_per_chain=2,
            n_chains=4,
        )
        nk_hilbert = state.sampler.netket_hilbert
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
        driver = self._make_driver(hilbert, seed=3)
        before = jax.tree_util.tree_leaves(driver.variational_state.parameters)
        history = driver.run(2)
        after = jax.tree_util.tree_leaves(driver.variational_state.parameters)

        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["step"], 0)
        self.assertEqual(history[1]["step"], 1)
        self.assertIn("energy", history[0])
        self.assertTrue(any(not np.allclose(np.asarray(a), np.asarray(b)) for a, b in zip(before, after)))

    def test_vmc_run_records_callback_outputs_with_cadence(self) -> None:
        hilbert = SpinHilbert(4)
        driver = self._make_driver(hilbert, seed=4)

        history = driver.run(
            3,
            callback=lambda step, current_driver: {
                "marker": step,
                "energy_snapshot": float(np.asarray(current_driver.variational_state.energy(current_driver.operator))),
            },
            callback_every=2,
        )
        first_observables = cast(dict[str, object], history[0]["observables"])
        last_observables = cast(dict[str, object], history[2]["observables"])

        self.assertEqual(first_observables["marker"], 0)
        self.assertNotIn("observables", history[1])
        self.assertEqual(last_observables["marker"], 2)

    def test_vmc_run_rejects_nonpositive_callback_every(self) -> None:
        hilbert = SpinHilbert(4)
        driver = self._make_driver(hilbert, seed=5)

        with self.assertRaises(ValueError):
            driver.run(1, callback=lambda step, _: {"step": step}, callback_every=0)

    def test_small_system_exact_backend_matches_ed_to_within_target(self) -> None:
        hilbert = SpinHilbert(4)
        graph = SquareLattice(2, 2, pbc=True)
        project_operator = _tfim_operator(hilbert, graph)
        exact_energy = exact_ground_state_energy(project_operator)
        model = RBM(alpha=2)
        state, driver = build_vmc_driver(
            model=model,
            hilbert=hilbert,
            operator=project_operator.to_netket(),
            learning_rate=1e-2,
            seed=7,
            n_samples=128,
            n_discard_per_chain=8,
            n_chains=8,
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
        exact_energy = exact_ground_state_energy(project_operator)
        architecture_runs = (
            ("RBM", RBM(alpha=4), 800, 1e-2),
            ("FFNN", FFNN(hidden_dims=(32, 16)), 400, 1e-2),
            ("CNN", CNN(spatial_shape=(2, 2), channels=(16, 8), kernel_size=(2, 2)), 400, 1e-2),
        )

        for label, model, n_steps, learning_rate in architecture_runs:
            with self.subTest(model=label):
                state, driver = build_vmc_driver(
                    model=model,
                    hilbert=hilbert,
                    operator=project_operator.to_netket(),
                    learning_rate=learning_rate,
                    seed=0,
                    n_samples=256,
                    n_discard_per_chain=16,
                    n_chains=8,
                )

                for _ in range(n_steps):
                    driver.step()

                optimized_energy = float(np.asarray(state.energy(project_operator.to_netket())))
                self.assertLess(abs(optimized_energy - exact_energy), 0.01)


if __name__ == "__main__":
    unittest.main()
