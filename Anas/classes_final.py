import jax
import jax.numpy as jnp
from jax import random
from functools import partial
import optax
import netket as nk


# =========================
# Neural Quantum State
# =========================

class NeuralQuantumState:

    def __init__(self, architecture, params, L):
        self.architecture = architecture
        self.params = params
        self.L = L

    @partial(jax.jit, static_argnames=('self',))
    def logpsi(self, params, sigma):
        return self.architecture.forward(params, sigma)

    @partial(jax.jit, static_argnames=('self',))
    def psi(self, params, sigma):
        return jnp.exp(self.logpsi(params, sigma))

    def flatten_params(self, params):
        flat, unravel = jax.flatten_util.ravel_pytree(params)
        return flat, unravel


# =========================
# Feed Forward Network
# =========================

class FFN:

    def __init__(self, L, hidden_layers):
        self.L = L
        self.hidden_layers = hidden_layers

    def init_params(self, key):
        layer_sizes = [self.L] + self.hidden_layers + [1]
        keys = random.split(key, len(layer_sizes)-1)

        params = []
        for k,(m,n) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):
            W = random.normal(k,(n,m))*jnp.sqrt(2/m)
            b = jnp.zeros(n)
            params.append((W,b))

        return params

    @partial(jax.jit, static_argnames=('self',))
    def forward(self, params, sigma):
        x = sigma
        for W,b in params[:-1]:
            x = jnp.tanh(W@x + b)
        W,b = params[-1]
        return (W@x + b)[0]


# =========================
# RBM
# =========================

class RBM:

    def __init__(self, L, hidden):
        self.L = L
        self.hidden = hidden

    def init_params(self, key):
        k1,k2,k3 = random.split(key,3)
        W = random.normal(k1,(self.hidden,self.L))*0.01
        a = jnp.zeros(self.L)
        b = jnp.zeros(self.hidden)
        return (W,a,b)

    @partial(jax.jit, static_argnames=('self',))
    def forward(self, params, sigma):
        W,a,b = params
        visible = jnp.dot(a,sigma)
        hidden = jnp.sum(jnp.log(2*jnp.cosh(b + W@sigma)))
        return visible + hidden


# =========================
# CNN
# =========================

class CNN:

    def __init__(self, L, channels=16, kernel=3):
        self.L = L
        self.channels = channels
        self.kernel = kernel

    def init_params(self, key):
        k1,k2,k3 = random.split(key,3)
        conv = random.normal(k1,(self.channels,1,self.kernel))*0.1
        dense = random.normal(k2,(self.channels*self.L,1))*0.1
        bias = jnp.zeros(1)
        return (conv,dense,bias)

    @partial(jax.jit, static_argnames=('self',))
    def forward(self, params, sigma):
        conv,dense,bias = params
        x = sigma.reshape(1,1,self.L)
        x = jax.lax.conv_general_dilated(
            x, conv,
            window_strides=(1,),
            padding="SAME",
            dimension_numbers=("NCH","OIH","NCH")
        )
        x = jnp.tanh(x)
        x = x.reshape(-1)
        return (dense.T @ x + bias)[0]


# =========================
# NetKet Sampler
# =========================

class NetKetSampler:

    def __init__(self, wavefunction, n_chains):
        self.wavefunction = wavefunction
        self.L = wavefunction.L
        self.n_chains = n_chains

        self.hilbert = nk.hilbert.Spin(s=1/2, N=self.L)
        self.sampler = nk.sampler.MetropolisLocal(
            self.hilbert,
            n_chains=n_chains
        )

    def sample_chains(self, key, params, n_samples, n_chains, burn=200):

        nk.random.seed(int(jax.random.randint(key, (), 0, 1e6)))

        def logpsi_fn(sigma):
            return self.wavefunction.logpsi(params, jnp.array(sigma))

        logpsi_batch = jax.vmap(logpsi_fn)

        state = self.sampler.init_state()

        samples = []
        for _ in range(n_samples + burn):
            state, σ = self.sampler.sample(logpsi_batch, state)
            samples.append(σ)

        samples = jnp.array(samples)[burn:]
        samples = samples.reshape(-1, self.L)

        # convert {0,1} → {-1,1}
        samples = 2 * samples - 1

        return samples


# =========================
# TFIM Hamiltonian
# =========================

class TFIM:

    def __init__(self, wavefunction, J, g):
        self.wavefunction = wavefunction
        self.J = J
        self.g = g
        self.L = wavefunction.L

    @partial(jax.jit, static_argnames=('self',))
    def local_energy(self, params, sigma):

        zz = -self.J * jnp.sum(sigma * jnp.roll(sigma,-1))

        logpsi_sigma = self.wavefunction.logpsi(params,sigma)

        def flip_term(i):
            sigma_flip = sigma.at[i].set(-sigma[i])
            logpsi_flip = self.wavefunction.logpsi(params,sigma_flip)
            return jnp.exp(logpsi_flip - logpsi_sigma)

        flip_energy = jax.vmap(flip_term)(jnp.arange(self.L)).sum()

        return zz - self.g * flip_energy

    @partial(jax.jit, static_argnames=('self',))
    def energy(self, params, samples):
        return jax.vmap(self.local_energy, in_axes=(None,0))(params,samples)


# =========================
# XXZ Hamiltonian
# =========================

class XXZ:

    def __init__(self, wavefunction, J=1.0, Delta=1.0):
        self.wavefunction = wavefunction
        self.J = J
        self.Delta = Delta
        self.L = wavefunction.L

    @partial(jax.jit, static_argnames=('self',))
    def local_energy(self, params, sigma):

        zz = self.J * self.Delta * jnp.sum(
            sigma * jnp.roll(sigma, -1)
        )

        logpsi_sigma = self.wavefunction.logpsi(params, sigma)

        def flip_pair(i):
            j = (i + 1) % self.L
            cond = sigma[i] != sigma[j]

            def flipped():
                sigma_new = sigma.at[i].set(-sigma[i])
                sigma_new = sigma_new.at[j].set(-sigma[j])
                logpsi_new = self.wavefunction.logpsi(params, sigma_new)
                return jnp.exp(logpsi_new - logpsi_sigma)

            return jnp.where(cond, flipped(), 0.0)

        flip_terms = jax.vmap(flip_pair)(jnp.arange(self.L)).sum()

        return zz - self.J * flip_terms

    @partial(jax.jit, static_argnames=('self',))
    def energy(self, params, samples):
        return jax.vmap(self.local_energy, in_axes=(None,0))(params,samples)


# =========================
# Observables
# =========================

class Observables:

    def __init__(self, wavefunction, sampler):
        self.wavefunction = wavefunction
        self.sampler = sampler

    def renyi2_entropy(self, key, params, n_samples, n_chains, LA, n_disorder=4):

        entropies = []

        for _ in range(n_disorder):

            key, k1, k2 = random.split(key, 3)

            s1 = self.sampler.sample_chains(k1, params, n_samples, n_chains)
            s2 = self.sampler.sample_chains(k2, params, n_samples, n_chains)

            s1p = jnp.concatenate([s2[:, :LA], s1[:, LA:]], axis=1)
            s2p = jnp.concatenate([s1[:, :LA], s2[:, LA:]], axis=1)

            logpsi = self.wavefunction.logpsi

            logpsi_s1  = jax.vmap(logpsi, in_axes=(None, 0))(params, s1)
            logpsi_s2  = jax.vmap(logpsi, in_axes=(None, 0))(params, s2)
            logpsi_s1p = jax.vmap(logpsi, in_axes=(None, 0))(params, s1p)
            logpsi_s2p = jax.vmap(logpsi, in_axes=(None, 0))(params, s2p)

            log_ratio = logpsi_s1p + logpsi_s2p - logpsi_s1 - logpsi_s2

            max_log = jnp.max(log_ratio)
            swap = jnp.exp(max_log) * jnp.mean(jnp.exp(log_ratio - max_log))

            S2 = -jnp.log(jnp.abs(swap))

            entropies.append(S2)

        return jnp.mean(jnp.array(entropies))


# =========================
# Optimizer (with entropy tracking)
# =========================

class Optimizer:

    def __init__(self, wavefunction, hamiltonian, sampler, lr=1e-3):

        self.wavefunction = wavefunction
        self.hamiltonian = hamiltonian
        self.sampler = sampler

        self.optimizer = optax.adam(lr)

        flat_params, _ = wavefunction.flatten_params(wavefunction.params)
        self.opt_state = self.optimizer.init(flat_params)

    def step(self, key, params, n_samples, n_chains):

        samples = self.sampler.sample_chains(
            key, params, n_samples, n_chains
        )

        energies = self.hamiltonian.energy(params, samples)
        meanE = jnp.mean(energies)

        def loss_fn(p):
            return jnp.mean(self.hamiltonian.energy(p, samples))

        grads = jax.grad(loss_fn)(params)

        flat_params, unravel = jax.flatten_util.ravel_pytree(params)
        flat_grads,_ = jax.flatten_util.ravel_pytree(grads)

        updates, self.opt_state = self.optimizer.update(
            flat_grads, self.opt_state
        )

        flat_params = optax.apply_updates(flat_params, updates)
        params = unravel(flat_params)

        return params, meanE

    def optimize(self, key, params, n_steps, n_samples, n_chains,
                 observables=None, entropy_every=10, LA=None):

        energies = []
        entropies = []

        for step in range(n_steps):

            key, subkey = random.split(key)
            params, E = self.step(subkey, params, n_samples, n_chains)

            energies.append(E)

            # ---- entropy tracking ----
            if observables is not None and step % entropy_every == 0:
                key, subkey = random.split(key)
                S2 = observables.renyi2_entropy(
                    subkey, params, n_samples, n_chains, LA
                )
                entropies.append(S2)
                print(f"step {step} | E = {E:.6f} | S2 = {S2:.6f}")
            else:
                print(f"step {step} | E = {E:.6f}")

        return params, jnp.array(energies), jnp.array(entropies)