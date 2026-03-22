from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse as jsparse

from .exact_diag import sparse_operator_matrix
from .operator import Operator
from .runtime_types import SupportsLogPsi
from .sampler import LogPsiApplyFn, MetropolisLocal, SampleBatch

ParamTree = Any


def _jax_complex_dtype() -> jnp.dtype:
    return jnp.complex128 if bool(jax.config.read("jax_enable_x64")) else jnp.complex64


@dataclass(frozen=True)
class ExpectationResult:
    mean: Any


class SupportsExpectationBackend(Protocol):
    def set_parameters(self, params: ParamTree) -> None:
        ...

    def sample(self) -> jax.Array:
        ...

    def sample_with_log_values(self) -> SampleBatch:
        ...

    def independent_sample(self, seed_offset: int = 0) -> jax.Array:
        ...

    def independent_sample_with_log_values(self, seed_offset: int = 0) -> SampleBatch:
        ...

    def exact_statevector(self) -> jax.Array:
        ...

    def expect(self, operator: Any) -> ExpectationResult:
        ...

    def expect_and_grad(self, operator: Any) -> tuple[ExpectationResult, ParamTree]:
        ...

    def expect_with_params(self, operator: Any, params: ParamTree) -> ExpectationResult:
        ...


@dataclass
class ProjectExpectationBackend:
    """Expectation backend using project-owned exact and sampled execution."""

    model: SupportsLogPsi
    sampler: MetropolisLocal
    params: ParamTree
    exact_backend_max_states: int = 4096
    _all_states: jax.Array | None = field(default=None, init=False, repr=False)
    _exact_operator_cache_operator: Operator | None = field(default=None, init=False, repr=False)
    _exact_operator_cache_value: jsparse.BCOO | None = field(default=None, init=False, repr=False)
    _sampler_log_psi_apply: LogPsiApplyFn = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._sampler_log_psi_apply = self.model.log_psi
        if self.sampler.hilbert.n_states <= self.exact_backend_max_states:
            self._all_states = jnp.asarray(self.sampler.hilbert.all_states(), dtype=jnp.uint8)

    def set_parameters(self, params: ParamTree) -> None:
        self.params = params

    def sample(self) -> jax.Array:
        return self.sample_with_log_values().states

    def sample_with_log_values(self) -> SampleBatch:
        return self.sampler.sample_with_params_and_log_values(
            self._sampler_log_psi_apply,
            self.params,
        )

    def independent_sample(self, seed_offset: int = 0) -> jax.Array:
        return self.independent_sample_with_log_values(seed_offset=seed_offset).states

    def independent_sample_with_log_values(self, seed_offset: int = 0) -> SampleBatch:
        return self.sampler.independent_sample_with_params_and_log_values(
            self._sampler_log_psi_apply,
            self.params,
            seed_offset=seed_offset,
        )

    def _exact_statevector_for_params(self, params: ParamTree) -> jax.Array:
        if self._all_states is None:
            raise ValueError("exact_statevector is only available for small full-summation states.")
        complex_dtype = _jax_complex_dtype()
        log_values = jnp.asarray(self.model.log_psi(params, self._all_states), dtype=complex_dtype)
        real_shift = jnp.max(jnp.real(log_values))
        amplitudes = jnp.exp(log_values - real_shift)
        norm = jnp.linalg.norm(amplitudes)
        if jnp.isclose(norm, 0.0):
            raise ValueError("Wavefunction amplitudes must have non-zero norm.")
        return amplitudes / norm

    def exact_statevector(self) -> jax.Array:
        return self._exact_statevector_for_params(self.params)

    def _require_project_operator(self, operator: Any) -> Operator:
        if not isinstance(operator, Operator):
            raise TypeError(
                "Project-owned runtime expectations require nqs.operator.Operator instances."
            )
        return operator

    def _exact_sparse_operator(self, operator: Operator) -> jsparse.BCOO:
        if self._exact_operator_cache_operator is not operator or self._exact_operator_cache_value is None:
            self._exact_operator_cache_operator = operator
            self._exact_operator_cache_value = jsparse.BCOO.from_scipy_sparse(
                sparse_operator_matrix(operator)
            )
        return self._exact_operator_cache_value

    def _exact_expectation_mean(self, operator: Operator, params: ParamTree) -> jax.Array:
        psi = self._exact_statevector_for_params(params)
        return jnp.vdot(psi, self._exact_sparse_operator(operator) @ psi)

    def _local_energies(
        self,
        operator: Operator,
        params: ParamTree,
        samples: jax.Array,
        sample_log_values: jax.Array | None = None,
    ) -> jax.Array:
        sample_array = np.asarray(samples, dtype=np.uint8).reshape(-1, operator.hilbert.size)
        if sample_log_values is None:
            original_log_values = np.asarray(
                self.model.log_psi(params, jnp.asarray(sample_array, dtype=jnp.uint8)),
                dtype=np.complex128,
            ).reshape(-1)
        else:
            original_log_values = np.asarray(sample_log_values, dtype=np.complex128).reshape(-1)
        batched_connected = operator.connected_elements_batched(sample_array)
        local_energies = np.zeros(sample_array.shape[0], dtype=np.complex128)
        if batched_connected.sample_indices.size == 0:
            return jnp.asarray(local_energies)

        connected_log_values = np.asarray(
            self.model.log_psi(params, jnp.asarray(batched_connected.connected_states, dtype=jnp.uint8)),
            dtype=np.complex128,
        ).reshape(-1)
        weighted_contributions = batched_connected.coefficients * np.exp(
            connected_log_values - original_log_values[batched_connected.sample_indices]
        )
        # np.bincount gives a compact grouped sum from connected-state rows back
        # to the original sample index; it only accepts real weights, so the
        # complex accumulation is split into real and imaginary parts.
        local_energies += np.bincount(
            batched_connected.sample_indices,
            weights=np.real(weighted_contributions),
            minlength=sample_array.shape[0],
        )
        local_energies += 1j * np.bincount(
            batched_connected.sample_indices,
            weights=np.imag(weighted_contributions),
            minlength=sample_array.shape[0],
        )
        return jnp.asarray(local_energies)

    def expect(self, operator: Any) -> ExpectationResult:
        project_operator = self._require_project_operator(operator)
        if self._all_states is not None:
            return ExpectationResult(mean=self._exact_expectation_mean(project_operator, self.params))
        sample_batch = self.sample_with_log_values()
        local_energies = self._local_energies(
            project_operator,
            self.params,
            sample_batch.states,
            sample_log_values=sample_batch.log_values,
        )
        return ExpectationResult(mean=jnp.mean(local_energies))

    def expect_and_grad(self, operator: Any) -> tuple[ExpectationResult, ParamTree]:
        project_operator = self._require_project_operator(operator)
        if self._all_states is not None:
            def exact_energy(params: ParamTree) -> jax.Array:
                return jnp.real(self._exact_expectation_mean(project_operator, params))

            energy, grads = jax.value_and_grad(exact_energy)(self.params)
            return ExpectationResult(mean=energy), grads

        sample_batch = self.sample_with_log_values()
        local_energies = jax.lax.stop_gradient(
            self._local_energies(
                project_operator,
                self.params,
                sample_batch.states,
                sample_log_values=sample_batch.log_values,
            )
        )
        energy = jnp.real(jnp.mean(local_energies))

        def surrogate_loss(params: ParamTree) -> jax.Array:
            log_values = jnp.asarray(self.model.log_psi(params, sample_batch.states))
            centered_local_energies = local_energies - jnp.mean(local_energies)
            return 2.0 * jnp.real(
                jnp.mean(jnp.conj(log_values) * centered_local_energies)
            )

        grads = jax.grad(surrogate_loss)(self.params)
        return ExpectationResult(mean=energy), grads

    def expect_with_params(
        self,
        operator: Any,
        params: ParamTree,
    ) -> ExpectationResult:
        project_operator = self._require_project_operator(operator)
        if self._all_states is not None:
            return ExpectationResult(mean=self._exact_expectation_mean(project_operator, params))

        sample_batch = self.sampler.independent_sample_with_params_and_log_values(
            self._sampler_log_psi_apply,
            params,
            seed_offset=0,
        )
        local_energies = self._local_energies(
            project_operator,
            params,
            sample_batch.states,
            sample_log_values=sample_batch.log_values,
        )
        return ExpectationResult(mean=jnp.mean(local_energies))


__all__ = [
    "ExpectationResult",
    "ProjectExpectationBackend",
    "SupportsExpectationBackend",
]
