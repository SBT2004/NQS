import jax
import jax.numpy as jnp
from functools import partial


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