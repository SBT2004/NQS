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

    @partial(jax.jit, static_argnames=('self',))
    def logpsi(self, params, sigma):
        return self.architecture.forward(params, sigma)

    @partial(jax.jit, static_argnames=('self',))
    def psi(self, params, sigma):
        return jnp.exp(self.logpsi(params, sigma))

    def flatten_params(self, params):
        flat, _ = jax.flatten_util.ravel_pytree(params)
        return flat


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
        #kept k2 and k3 in case random initialization is needed for a and b
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
            x,
            conv,
            window_strides=(1,),
            padding="SAME",
            dimension_numbers=("NCH","OIH","NCH")
        )

        x = jnp.tanh(x)
        x = x.reshape(-1)

        return (dense.T @ x + bias)[0]


# =========================
# Sampler (MULTI-CHAIN)
# =========================

class Sampler:

    def __init__(self, wavefunction):
        self.wavefunction = wavefunction
        self.L = wavefunction.L


    def metropolis_steps(self,key,params,sigma):

        key1,key2 = random.split(key)

        site = random.randint(key1,(),0,self.L)

        sigma_new = sigma.at[site].set(-sigma[site])

        logpsi_old = self.wavefunction.logpsi(params,sigma)
        logpsi_new = self.wavefunction.logpsi(params,sigma_new)

        log_ratio = 2 * (logpsi_new - logpsi_old)

        accept = jnp.log(random.uniform(key2)) < jnp.minimum(0,log_ratio)

        sigma = jnp.where(accept,sigma_new,sigma)

        return sigma



    # ----- vectorized over chains -----
    @partial(jax.jit, static_argnames=('self',))
    def metropolis_step(self, keys, params, sigmas):

        return jax.vmap(self.metropolis_steps, in_axes=(0,None,0))(
            keys, params, sigmas
        )


    # ----- full sampling -----
    def sample_chains(self, key, params, n_samples, n_chains, burn=200):

        # RANDOM INITIAL STATES (important!)
        key, subkey = random.split(key)
        sigmas = random.choice(
            subkey,
            jnp.array([-1, 1]),
            shape=(n_chains, self.L)
        )


        def step_fn(carry, _):
            key, sigmas = carry

            key, subkey = random.split(key)
            step_keys = random.split(subkey, n_chains)

            sigmas = self.metropolis_step(step_keys, params, sigmas)

            return (key, sigmas), sigmas


        (key, _), all_sigmas = jax.lax.scan(
            step_fn,
            (key, sigmas),
            None,
            length=n_samples + burn
        )

        samples = all_sigmas[burn:]
        samples = samples.reshape(-1, self.L)

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
# Optimizer
# =========================

class Optimizer:

    def __init__(self, wavefunction, hamiltonian, sampler, lr=1e-3):

        self.wavefunction = wavefunction
        self.hamiltonian = hamiltonian
        self.sampler = sampler

        self.optimizer = optax.adam(lr)

        flat_params = wavefunction.flatten_params(wavefunction.params)
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


    def optimize(self, key, params, n_steps, n_samples, n_chains):

        energies = []

        for step in range(n_steps):

            key, subkey = random.split(key)

            params, E = self.step(subkey, params, n_samples, n_chains)

            energies.append(E)

            print("step",step,"energy",E)

        return params, jnp.array(energies)


# =========================
# Observables
# =========================

class Observables:

    def __init__(self, wavefunction):
        self.wavefunction = wavefunction

    @partial(jax.jit, static_argnames=('self','LA'))
    def renyi2_entropy(self, params, samples1, samples2, LA):

        s1 = samples1
        s2 = samples2

        s1p = jnp.concatenate([s2[:, :LA], s1[:, LA:]], axis=1)
        s2p = jnp.concatenate([s1[:, :LA], s2[:, LA:]], axis=1)

        logpsi = self.wavefunction.logpsi

        logpsi_s1  = jax.vmap(logpsi, in_axes=(None, 0))(params, s1)
        logpsi_s2  = jax.vmap(logpsi, in_axes=(None, 0))(params, s2)
        logpsi_s1p = jax.vmap(logpsi, in_axes=(None, 0))(params, s1p)
        logpsi_s2p = jax.vmap(logpsi, in_axes=(None, 0))(params, s2p)

        log_ratio = logpsi_s1p + logpsi_s2p - logpsi_s1 - logpsi_s2

        max_log = jnp.max(log_ratio)

        swap_estimator = jnp.exp(max_log) * jnp.mean(jnp.exp(log_ratio - max_log))

        return -jnp.log(jnp.abs(swap_estimator))