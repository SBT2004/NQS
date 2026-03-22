import sys
import unittest
import warnings
from pathlib import Path
from typing import cast
from unittest import mock

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nqs.driver import VMC
import nqs.expectation as expectation_module
from nqs.expectation import ProjectExpectationBackend
from nqs.exact_diag import exact_ground_state_energy, sparse_operator_matrix
from nqs.exact_diag_debug import dense_debug_operator_matrix
from nqs.graph import Chain1D, SquareLattice
from nqs.hilbert import SpinHilbert
from nqs.loss import energy_loss
from nqs.models import CNN, FFNN, RBM
from nqs.operator import Operator, collect_terms, j1_j2, sx_term, szsz_term
from nqs.optimizer import Adam
from nqs.runtime_types import states_from_signed_spins, states_to_signed_spins
from nqs.sampler import MetropolisLocal
from nqs.vmc_setup import build_variational_state, build_vmc_experiment


def _tfim_operator(hilbert: SpinHilbert, graph: SquareLattice, J: float = 1.0, h: float = 0.8) -> Operator:
    interaction_terms = [szsz_term(edge.i, edge.j, coefficient=-J) for edge in graph.iter_edges("J1", n=1)]
    field_terms = [sx_term(site, coefficient=-h) for site in range(hilbert.size)]
    return Operator(hilbert, collect_terms(interaction_terms, field_terms))


def _chain_tfim_operator(hilbert: SpinHilbert, h: float = 1.0) -> Operator:
    graph = Chain1D(length=hilbert.size, pbc=False)
    interaction_terms = [szsz_term(edge.i, edge.j, coefficient=-1.0) for edge in graph.iter_edges("J", n=1)]
    field_terms = [sx_term(site, coefficient=-h) for site in range(hilbert.size)]
    return Operator(hilbert, collect_terms(interaction_terms, field_terms))


def _reference_local_energies(
    operator: Operator,
    model: RBM | FFNN | CNN,
    params: dict[str, object],
    samples: np.ndarray,
) -> np.ndarray:
    sample_array = np.asarray(samples, dtype=np.uint8).reshape(-1, operator.hilbert.size)
    original_log_values = np.asarray(
        model.log_psi(params, jax.numpy.asarray(sample_array, dtype=jax.numpy.uint8)),
        dtype=np.complex128,
    ).reshape(-1)
    local_energies = np.zeros(sample_array.shape[0], dtype=np.complex128)

    for sample_index, state in enumerate(sample_array):
        connected = operator.connected_elements(state)
        connected_states = np.stack([connected_state for connected_state, _ in connected], axis=0)
        coefficients = np.asarray([value for _, value in connected], dtype=np.complex128)
        connected_log_values = np.asarray(
            model.log_psi(params, jax.numpy.asarray(connected_states, dtype=jax.numpy.uint8)),
            dtype=np.complex128,
        ).reshape(-1)
        local_energies[sample_index] = np.sum(
            coefficients * np.exp(connected_log_values - original_log_values[sample_index])
        )

    return local_energies


class ModelTests(unittest.TestCase):
    def test_signed_spin_state_conversion_roundtrip(self) -> None:
        states = np.array([[0, 1, 0, 1], [1, 1, 0, 0]], dtype=np.uint8)
        jax_states = jax.numpy.asarray(states)
        signed_states = states_to_signed_spins(jax_states)
        np.testing.assert_array_equal(np.asarray(signed_states), np.array([[-1.0, 1.0, -1.0, 1.0], [1.0, 1.0, -1.0, -1.0]]))
        np.testing.assert_array_equal(np.asarray(states_from_signed_spins(signed_states)), states)

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
    def _make_sampler(self, hilbert: SpinHilbert, seed: int) -> MetropolisLocal:
        return MetropolisLocal(hilbert=hilbert, n_samples=16, n_discard_per_chain=2, n_chains=4, seed=seed)

    def _architecture_cases(self) -> tuple[tuple[str, RBM | FFNN | CNN, SpinHilbert], ...]:
        return (
            ("RBM", RBM(alpha=1), SpinHilbert(4)),
            ("FFNN", FFNN(hidden_dims=(8, 4)), SpinHilbert(4)),
            ("CNN", CNN(spatial_shape=(2, 2), channels=(4,), kernel_size=(2, 2)), SpinHilbert(4)),
        )

    def _make_driver(self, hilbert: SpinHilbert, seed: int) -> VMC:
        operator = _chain_tfim_operator(hilbert, h=1.0)
        _, _, driver = build_vmc_experiment(
            hilbert=hilbert,
            operator=operator,
            learning_rate=1e-2,
            seed=seed,
            n_samples=16,
            n_discard_per_chain=2,
            n_chains=4,
            model=RBM(alpha=1),
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
        operator = _chain_tfim_operator(hilbert, h=1.0)
        optimizer = Adam(learning_rate=1e-2)

        loss_value, grads = optimizer.compute_gradients(
            lambda current_params: energy_loss(state, operator, current_params),
            state.parameters,
        )

        self.assertTrue(np.isfinite(np.asarray(loss_value)))
        self.assertEqual(set(grads.keys()), set(state.parameters.keys()))

    def test_bitmap_local_energies_match_reference_array_path(self) -> None:
        hilbert = SpinHilbert(4)
        model = RBM(alpha=1)
        params = model.init(jax.random.PRNGKey(0), hilbert)
        backend = ProjectExpectationBackend(
            model=model,
            sampler=self._make_sampler(hilbert, seed=11),
            params=params,
        )
        operator = _chain_tfim_operator(hilbert, h=1.0)
        samples = np.array(
            [
                [0, 0, 0, 0],
                [1, 0, 1, 0],
                [1, 1, 0, 1],
            ],
            dtype=np.uint8,
        )

        actual = np.asarray(
            backend._local_energies(operator, params, jax.numpy.asarray(samples, dtype=jax.numpy.uint8))
        )
        expected = _reference_local_energies(operator, model, params, samples)

        np.testing.assert_allclose(actual, expected)

    def test_sampler_sample_with_params_matches_sample_contract_across_architectures(self) -> None:
        for label, model, hilbert in self._architecture_cases():
            with self.subTest(model=label):
                params = model.init(jax.random.PRNGKey(0), hilbert)
                reference_sampler = self._make_sampler(hilbert, seed=21)
                compiled_sampler = self._make_sampler(hilbert, seed=21)

                def log_psi_fn(states: jax.Array) -> jax.Array:
                    return model.log_psi(params, states)

                reference_first = np.asarray(reference_sampler.sample(log_psi_fn))
                compiled_first = np.asarray(compiled_sampler.sample_with_params(model.log_psi, params))
                np.testing.assert_array_equal(compiled_first, reference_first)

                reference_second = np.asarray(reference_sampler.sample(log_psi_fn))
                compiled_second = np.asarray(compiled_sampler.sample_with_params(model.log_psi, params))
                np.testing.assert_array_equal(compiled_second, reference_second)

                np.testing.assert_array_equal(
                    np.asarray(compiled_sampler._chain_states),
                    np.asarray(reference_sampler._chain_states),
                )

    def test_sampler_independent_sample_with_params_matches_sample_contract_across_architectures(self) -> None:
        for label, model, hilbert in self._architecture_cases():
            with self.subTest(model=label):
                params = model.init(jax.random.PRNGKey(1), hilbert)
                reference_sampler = self._make_sampler(hilbert, seed=22)
                compiled_sampler = self._make_sampler(hilbert, seed=22)

                def log_psi_fn(states: jax.Array) -> jax.Array:
                    return model.log_psi(params, states)

                for seed_offset in (0, 3):
                    reference_samples = np.asarray(
                        reference_sampler.independent_sample(log_psi_fn, seed_offset=seed_offset)
                    )
                    compiled_samples = np.asarray(
                        compiled_sampler.independent_sample_with_params(
                            model.log_psi,
                            params,
                            seed_offset=seed_offset,
                        )
                    )
                    np.testing.assert_array_equal(compiled_samples, reference_samples)

                self.assertIsNone(reference_sampler._chain_states)
                self.assertIsNone(compiled_sampler._chain_states)

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
        _, state, driver = build_vmc_experiment(
            hilbert=hilbert,
            operator=project_operator,
            learning_rate=1e-2,
            seed=7,
            n_samples=128,
            n_discard_per_chain=8,
            n_chains=8,
            model=model,
        )

        for _ in range(120):
            driver.step()

        optimized_energy = float(np.asarray(state.energy(project_operator)))
        self.assertLess(abs(optimized_energy - exact_energy), 0.01)
        self.assertEqual(np.asarray(state.exact_statevector()).shape, (hilbert.n_states,))

    def test_exact_backend_expectation_avoids_complex128_truncation_warning(self) -> None:
        hilbert = SpinHilbert(4)
        operator = _chain_tfim_operator(hilbert, h=1.0)
        model = RBM(alpha=1)
        params = model.init(jax.random.PRNGKey(0), hilbert)
        backend = ProjectExpectationBackend(
            model=model,
            sampler=self._make_sampler(hilbert, seed=12),
            params=params,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = backend.expect(operator)

        self.assertTrue(np.isfinite(np.asarray(result.mean)))
        self.assertFalse(
            any("Explicitly requested dtype <class 'jax.numpy.complex128'>" in str(warning.message) for warning in caught)
        )

    def test_exact_backend_expectation_matches_dense_reference_without_dense_runtime_path(self) -> None:
        hilbert = SpinHilbert(4)
        operator = _chain_tfim_operator(hilbert, h=1.0)
        model = RBM(alpha=1)
        params = model.init(jax.random.PRNGKey(0), hilbert)
        backend = ProjectExpectationBackend(
            model=model,
            sampler=self._make_sampler(hilbert, seed=13),
            params=params,
        )

        psi = np.asarray(backend.exact_statevector(), dtype=np.complex128)
        expected = np.vdot(psi, dense_debug_operator_matrix(operator) @ psi)

        with mock.patch.object(
            expectation_module,
            "sparse_operator_matrix",
            wraps=sparse_operator_matrix,
        ) as sparse_builder:
            actual = np.asarray(backend.expect(operator).mean)

        np.testing.assert_allclose(actual, expected)
        self.assertGreaterEqual(sparse_builder.call_count, 1)

    def test_exact_backend_expect_with_params_matches_dense_reference(self) -> None:
        hilbert = SpinHilbert(4)
        operator = _chain_tfim_operator(hilbert, h=1.0)
        model = RBM(alpha=1)
        params = model.init(jax.random.PRNGKey(0), hilbert)
        backend = ProjectExpectationBackend(
            model=model,
            sampler=self._make_sampler(hilbert, seed=14),
            params=params,
        )

        psi = np.asarray(backend._exact_statevector_for_params(params), dtype=np.complex128)
        expected = np.vdot(psi, dense_debug_operator_matrix(operator) @ psi)
        actual = np.asarray(backend.expect_with_params(operator, params).mean)

        np.testing.assert_allclose(actual, expected)

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
                _, state, driver = build_vmc_experiment(
                    hilbert=hilbert,
                    operator=project_operator,
                    learning_rate=learning_rate,
                    seed=0,
                    n_samples=256,
                    n_discard_per_chain=16,
                    n_chains=8,
                    model=model,
                )

                best_energy = float(np.asarray(state.energy(project_operator)))
                for _ in range(n_steps):
                    driver.step()
                    best_energy = min(best_energy, float(np.asarray(state.energy(project_operator))))

                self.assertLess(abs(best_energy - exact_energy), 0.01)


if __name__ == "__main__":
    unittest.main()
