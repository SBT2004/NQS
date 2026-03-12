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


if __name__ == "__main__":
    unittest.main()
