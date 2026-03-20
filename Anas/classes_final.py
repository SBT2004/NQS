import jax
import jax.numpy as jnp
from jax import random
from functools import partial
import optax


# =========================
# Neural Quantum State
# =========================

class NeuralQuantumState:
    def __init__(self, architecture, params, L):
        self.architecture = architecture
        self.params = params
        self.L = L

    @partial(jax.jit, static_argnames=("self",))
    def logpsi(self, params, sigma):
        return self.architecture.forward(params, sigma)

    def vmap_logpsi(self, params, sigmas):
        return jax.vmap(self.logpsi, in_axes=(None, 0))(params, sigmas)

    def flatten_params(self, params):
        flat, unravel = jax.flatten_util.ravel_pytree(params)
        return flat, unravel


# =========================
# RBM
# =========================

class RBM:
    def __init__(self, L, hidden):
        self.L = L
        self.hidden = hidden

    def init_params(self, key):
        k1, k2, k3 = random.split(key, 3)
        W = random.normal(k1, (self.hidden, self.L)) * 0.1
        a = jnp.zeros(self.L)
        b = jnp.zeros(self.hidden)
        return (W, a, b)

    @partial(jax.jit, static_argnames=("self",))
    def forward(self, params, sigma):
        W, a, b = params
        visible = jnp.dot(a, sigma)
        hidden = jnp.sum(jnp.log(2.0 * jnp.cosh(b + W @ sigma)))
        return visible + hidden


# =========================
# FFN
# =========================

class FFN:
    def __init__(self, L, hidden_layers):
        self.L = L
        self.hidden_layers = hidden_layers

    def init_params(self, key):
        layer_sizes = [self.L] + self.hidden_layers + [1]
        keys = random.split(key, len(layer_sizes) - 1)
        params = []
        for k, (m, n) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):
            W = random.normal(k, (n, m)) * jnp.sqrt(2.0 / m)
            b = jnp.zeros(n)
            params.append((W, b))
        return params

    @partial(jax.jit, static_argnames=("self",))
    def forward(self, params, sigma):
        x = sigma
        for W, b in params[:-1]:
            x = jnp.tanh(W @ x + b)
        W, b = params[-1]
        return (W @ x + b)[0]


# =========================
# CNN
# =========================

class CNN:
    def __init__(self, L, channels=16, kernel=3):
        self.L = L
        self.channels = channels
        self.kernel = kernel

    def init_params(self, key):
        k1, k2 = random.split(key)
        conv = random.normal(k1, (self.channels, 1, self.kernel)) * 0.5
        dense = random.normal(k2, (self.channels * self.L, 1)) * 0.5
        bias = jnp.zeros(1)
        return (conv, dense, bias)

    @partial(jax.jit, static_argnames=("self",))
    def forward(self, params, sigma):
        conv, dense, bias = params
        x = sigma.reshape(1, 1, self.L).astype(jnp.float32)

        x = jax.lax.conv_general_dilated(
            x,
            conv,
            window_strides=(1,),
            padding="SAME",
            dimension_numbers=("NCH", "OIH", "NCH"),
        )

        x = jnp.tanh(x)
        x = x.reshape(-1)
        return (dense.T @ x + bias)[0]


# =========================
# Sampler
# =========================

class Sampler:
    def __init__(self, nchains, nsamples_per_chain, neq, nskip, wavefunction):
        self.nchains = nchains
        self.nsamples_per_chain = nsamples_per_chain
        self.neq = neq
        self.nskip = nskip
        self.wavefunction = wavefunction
        self.L = wavefunction.L

    @partial(jax.jit, static_argnames=("self",))
    def step(self, vals):
        sigma_o, logpsi_o, params, key = vals
        key, subkey1, subkey2 = random.split(key, 3)

        sites = random.randint(subkey1, (self.nchains,), 0, self.L)
        sigma_n = sigma_o.at[jnp.arange(self.nchains), sites].multiply(-1)

        logpsi_n = jax.vmap(self.wavefunction.logpsi, in_axes=(None, 0))(params, sigma_n)
        log_ratio = 2.0 * (logpsi_n - logpsi_o)

        accept = jnp.log(random.uniform(subkey2, (self.nchains,))) < log_ratio

        sigma_o = jnp.where(accept[:, None], sigma_n, sigma_o)
        logpsi_o = jnp.where(accept, logpsi_n, logpsi_o)

        return sigma_o, logpsi_o, params, key

    def sample_chain(self, key, params):
        key_init, key_mc = random.split(key)
        sigma0 = random.choice(key_init, jnp.array([-1, 1]), shape=(self.nchains, self.L))
        logpsi0 = jax.vmap(self.wavefunction.logpsi, in_axes=(None, 0))(params, sigma0)

        vals = (sigma0, logpsi0, params, key_mc)

        for _ in range(self.neq):
            vals = self.step(vals)

        samples = []
        for _ in range(self.nsamples_per_chain):
            for _ in range(self.nskip):
                vals = self.step(vals)
            samples.append(vals[0])

        samples = jnp.stack(samples)
        return samples.reshape(-1, self.L)


# =========================
# TFIM
# =========================

class TFIM:
    def __init__(self, wavefunction, J, g):
        self.wavefunction = wavefunction
        self.J = J
        self.g = g
        self.L = wavefunction.L

    @partial(jax.jit, static_argnames=("self",))
    def local_energy(self, params, sigma):
        zz = -self.J * jnp.sum(sigma * jnp.roll(sigma, -1))
        logpsi_sigma = self.wavefunction.logpsi(params, sigma)

        def flip(i):
            sigma_flip = sigma.at[i].set(-sigma[i])
            logpsi_flip = self.wavefunction.logpsi(params, sigma_flip)
            return jnp.exp(logpsi_flip - logpsi_sigma)

        flip_energy = jnp.sum(jax.vmap(flip)(jnp.arange(self.L)))
        return zz - self.g * flip_energy

    def energy(self, params, samples):
        return jax.vmap(self.local_energy, in_axes=(None, 0))(params, samples)


# =========================
# Observables
# =========================

class Observables:
    def __init__(self, wavefunction):
        self.wavefunction = wavefunction
        self.L = wavefunction.L

    def _swap_estimator_from_perm(self, params, samples, perm, subsystem_size):
        LA = subsystem_size

        sigma = samples
        sigma_p = samples[perm]

        sigma_swap = jnp.concatenate([sigma[:, :LA], sigma_p[:, LA:]], axis=1)
        sigma_p_swap = jnp.concatenate([sigma_p[:, :LA], sigma[:, LA:]], axis=1)

        logpsi = self.wavefunction.vmap_logpsi(params, sigma)
        logpsi_p = self.wavefunction.vmap_logpsi(params, sigma_p)
        logpsi_swap = self.wavefunction.vmap_logpsi(params, sigma_swap)
        logpsi_p_swap = self.wavefunction.vmap_logpsi(params, sigma_p_swap)

        log_ratio = logpsi_swap + logpsi_p_swap - logpsi - logpsi_p
        return jnp.exp(log_ratio)

    def renyi2_entropy_swap(self, params, samples, key, subsystem_size=None, n_pairings=8, eps=1e-12):
        """
        Rényi-2 entropy from the swap trick:
            S2(A) = -log <Swap_A>

        A is taken to be the first `subsystem_size` spins.
        If subsystem_size is None, use half chain.

        n_pairings:
            number of independent random pairings/permutations to average over.
        """
        if subsystem_size is None:
            subsystem_size = self.L // 2

        N = samples.shape[0]
        keys = random.split(key, n_pairings)

        swap_means = []
        for k in keys:
            perm = random.permutation(k, N)
            swap_vals = self._swap_estimator_from_perm(params, samples, perm, subsystem_size)
            swap_means.append(jnp.mean(swap_vals))

        swap_mean = jnp.mean(jnp.stack(swap_means))
        return -jnp.log(swap_mean + eps)


# =========================
# Adam optimizer
# =========================

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
            return jax.flatten_util.ravel_pytree(g)[0]

        O = jax.vmap(grad_logpsi, in_axes=(None, 0))(params, samples)
        O_mean = jnp.mean(O, axis=0)

        grad_E = 2.0 * jnp.mean((energies - E_mean)[:, None] * (O - O_mean), axis=0)

        flat, unravel = jax.flatten_util.ravel_pytree(params)
        updates, self.state = self.opt.update(grad_E, self.state, flat)
        flat = optax.apply_updates(flat, updates)

        return unravel(flat), E_mean, samples


# =========================
# SR optimizer
# =========================

class SROptimizer:
    def __init__(self, wf, ham, sampler, lr=0.05, diag_shift=1e-3):
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
            return jax.flatten_util.ravel_pytree(g)[0]

        O = jax.vmap(grad_logpsi, in_axes=(None, 0))(params, samples)
        O_mean = jnp.mean(O, axis=0, keepdims=True)
        Oc = O - O_mean

        grad_E = 2.0 * jnp.mean((energies - E_mean)[:, None] * Oc, axis=0)

        S = (Oc.T @ Oc) / O.shape[0]
        S = S + self.diag_shift * jnp.eye(S.shape[0])

        delta = -self.lr * jnp.linalg.solve(S, grad_E)

        flat, unravel = jax.flatten_util.ravel_pytree(params)
        flat = flat + delta

        return unravel(flat), E_mean, samples