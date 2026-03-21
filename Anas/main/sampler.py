import jax
import jax.numpy as jnp
from jax import random
from functools import partial


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