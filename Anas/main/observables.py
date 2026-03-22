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

        # clipping to avoid extreme overflow
        log_ratio = jnp.clip(log_ratio, -30.0, 30.0)

        return log_ratio    


    def renyi2_entropy_swap(
        self,
        params,
        samples,
        key,
        subsystem_size=None,
        n_pairings=8,
        eps=1e-12,
    ):
        if subsystem_size is None:
            subsystem_size = self.L // 2

        N = samples.shape[0]
        keys = random.split(key, n_pairings)

        swap_means = []
        for k in keys:
            perm = random.permutation(k, N)
            log_vals = self._swap_estimator_from_perm(params, samples, perm, subsystem_size)

            # log-sum-exp trick for stability
            max_log = jnp.max(log_vals)
            stable_vals = jnp.exp(log_vals - max_log)
            mean_val = jnp.mean(stable_vals) * jnp.exp(max_log)

            swap_means.append(mean_val)

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
    
    def energy_variance(self, hamiltonian, params, samples):
        energies = hamiltonian.energy(params, samples)
        mean_E = jnp.mean(energies)
        mean_E2 = jnp.mean(energies ** 2)
        return mean_E2 - mean_E ** 2
    
    def magnetization_z(self, samples):
        """
        <m_z> where m_z = (1/L) sum_i sigma_i
        """
        mz_per_sample = jnp.mean(samples, axis=1)
        return jnp.mean(mz_per_sample)

    def abs_magnetization_z(self, samples):
        """
        <|m_z|>
        Often more informative than <m_z> because symmetry can force <m_z> ~ 0.
        """
        mz_per_sample = jnp.mean(samples, axis=1)
        return jnp.mean(jnp.abs(mz_per_sample))

    def spin_spin_correlation(self, samples, r):
        """
        C(r) = (1/L) sum_i <sigma_i sigma_{i+r}>
        periodic boundary conditions
        """
        corr_per_sample = jnp.mean(samples * jnp.roll(samples, -r, axis=1), axis=1)
        return jnp.mean(corr_per_sample)

    def correlation_profile(self, samples, r_values):
        """
        Returns dict: r -> C(r)
        """
        out = {}
        for r in r_values:
            out[int(r)] = self.spin_spin_correlation(samples, int(r))
        return out
    
    def enumerate_spin_basis(self):
        states = ((jnp.arange(2**self.L)[:, None] >> jnp.arange(self.L)) & 1)
        states = 2 * states - 1
        return states.astype(jnp.int32)

    def normalized_statevector(self, params, eps=1e-12):
        basis = self.enumerate_spin_basis()
        logpsi = self.wavefunction.vmap_logpsi(params, basis)
        psi = jnp.exp(logpsi)
        norm = jnp.linalg.norm(psi)
        return psi / (norm + eps)

    def reduced_density_matrix_exact(self, params, subsystem_size):
        LA = int(subsystem_size)
        LB = self.L - LA

        psi = self.normalized_statevector(params)
        psi_matrix = psi.reshape((2**LA, 2**LB))
        rho_A = psi_matrix @ jnp.conjugate(psi_matrix.T)
        return rho_A

    def renyi2_entropy_exact(self, params, subsystem_size, eps=1e-12):
        rho_A = self.reduced_density_matrix_exact(params, subsystem_size)
        tr_rho2 = jnp.real(jnp.trace(rho_A @ rho_A))
        return -jnp.log(tr_rho2 + eps)
