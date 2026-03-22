import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree


class AdamOptimizer:
    def __init__(self, wf, ham, sampler, lr=1e-2):
        self.wf = wf
        self.ham = ham
        self.sampler = sampler
        self.opt = optax.adam(lr)

        flat, _ = wf.flatten_params(wf.params)
        self.state = self.opt.init(flat)

    def step(self, key, params):
        samples = self.sampler.sample_chain(key, params)
        energies = self.ham.energy(params, samples)
        E_mean = jnp.mean(energies)

        def grad_logpsi(p, s):
            g = jax.grad(self.wf.logpsi)(p, s)
            return ravel_pytree(g)[0]

        O = jax.vmap(grad_logpsi, in_axes=(None, 0))(params, samples)
        O_mean = jnp.mean(O, axis=0)
        grad_E = 2.0 * jnp.mean((energies - E_mean)[:, None] * (O - O_mean), axis=0)

        flat, unravel = ravel_pytree(params)
        updates, self.state = self.opt.update(grad_E, self.state, flat)
        flat = optax.apply_updates(flat, updates)

        return unravel(flat), E_mean, samples


class SROptimizer:
    def __init__(self, wf, ham, sampler, lr=0.05, diag_shift=1e-2):
        self.wf = wf
        self.ham = ham
        self.sampler = sampler
        self.lr = lr
        self.diag_shift = diag_shift

    def step(self, key, params):
        samples = self.sampler.sample_chain(key, params)
        energies = self.ham.energy(params, samples)
        E_mean = jnp.mean(energies)

        def grad_logpsi(p, s):
            g = jax.grad(self.wf.logpsi)(p, s)
            return ravel_pytree(g)[0]

        O = jax.vmap(grad_logpsi, in_axes=(None, 0))(params, samples)
        O_mean = jnp.mean(O, axis=0, keepdims=True)
        Oc = O - O_mean

        grad_E = 2.0 * jnp.mean((energies - E_mean)[:, None] * Oc, axis=0)

        S = (Oc.T @ Oc) / O.shape[0]
        S = S + self.diag_shift * jnp.eye(S.shape[0])

        delta = -self.lr * jnp.linalg.solve(S, grad_E)

        flat, unravel = ravel_pytree(params)
        flat = flat + delta

        return unravel(flat), E_mean, samples