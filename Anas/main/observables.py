import jax
import jax.numpy as jnp
from jax import random


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

    def entropy_profile(self, params, samples, key, subsystem_sizes, n_pairings=8):
        out = {}
        keys = random.split(key, len(subsystem_sizes))
        for k, LA in zip(keys, subsystem_sizes):
            out[int(LA)] = self.renyi2_entropy_swap(
                params=params,
                samples=samples,
                key=k,
                subsystem_size=int(LA),
                n_pairings=n_pairings,
            )
        return out