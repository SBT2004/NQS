from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from .exact_diag import operator_matrix
from .operator import Operator
from .runtime_types import SupportsLogPsi
from .sampler import MetropolisLocal

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

    def independent_sample(self, seed_offset: int = 0) -> jax.Array:
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

    def __post_init__(self) -> None:
        if self.sampler.hilbert.n_states <= self.exact_backend_max_states:
            self._all_states = jnp.asarray(self.sampler.hilbert.all_states(), dtype=jnp.uint8)

    def set_parameters(self, params: ParamTree) -> None:
        self.params = params

    def _log_value_fn(self, params: ParamTree):
        return lambda states: self.model.log_psi(params, states)

    def sample(self) -> jax.Array:
        return self.sampler.sample(self._log_value_fn(self.params))

    def independent_sample(self, seed_offset: int = 0) -> jax.Array:
        return self.sampler.independent_sample(
            self._log_value_fn(self.params),
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

    def _exact_expectation_mean(self, operator: Operator, params: ParamTree) -> jax.Array:
        dense_matrix = jnp.asarray(operator_matrix(operator), dtype=_jax_complex_dtype())
        psi = self._exact_statevector_for_params(params)
        return jnp.vdot(psi, dense_matrix @ psi)

    def _local_energies(
        self,
        operator: Operator,
        params: ParamTree,
        samples: jax.Array,
    ) -> jax.Array:
        sample_array = np.asarray(samples, dtype=np.uint8).reshape(-1, operator.hilbert.size)
        sample_bits = operator.hilbert.states_to_bits(sample_array)
        original_log_values = np.asarray(
            self.model.log_psi(params, jnp.asarray(sample_array, dtype=jnp.uint8)),
            dtype=np.complex128,
        ).reshape(-1)
        local_energies = np.zeros(sample_array.shape[0], dtype=np.complex128)

        for sample_index, state_bits in enumerate(sample_bits):
            connected = operator.connected_elements_bits(int(state_bits))
            if not connected:
                continue
            connected_bits = np.asarray([connected_state_bits for connected_state_bits, _ in connected], dtype=object)
            connected_states = operator.hilbert.bits_to_states(connected_bits)
            coefficients = np.asarray([value for _, value in connected], dtype=np.complex128)
            connected_log_values = np.asarray(
                self.model.log_psi(params, jnp.asarray(connected_states, dtype=jnp.uint8)),
                dtype=np.complex128,
            ).reshape(-1)
            local_energies[sample_index] = np.sum(
                coefficients * np.exp(connected_log_values - original_log_values[sample_index])
            )

        return jnp.asarray(local_energies)

    def expect(self, operator: Any) -> ExpectationResult:
        project_operator = self._require_project_operator(operator)
        if self._all_states is not None:
            return ExpectationResult(mean=self._exact_expectation_mean(project_operator, self.params))
        samples = self.sample()
        local_energies = self._local_energies(project_operator, self.params, samples)
        return ExpectationResult(mean=jnp.mean(local_energies))

    def expect_and_grad(self, operator: Any) -> tuple[ExpectationResult, ParamTree]:
        project_operator = self._require_project_operator(operator)
        if self._all_states is not None:
            def exact_energy(params: ParamTree) -> jax.Array:
                return jnp.real(self._exact_expectation_mean(project_operator, params))

            energy, grads = jax.value_and_grad(exact_energy)(self.params)
            return ExpectationResult(mean=energy), grads

        samples = self.sample()
        local_energies = jax.lax.stop_gradient(
            self._local_energies(project_operator, self.params, samples)
        )
        energy = jnp.real(jnp.mean(local_energies))

        def surrogate_loss(params: ParamTree) -> jax.Array:
            log_values = jnp.asarray(self.model.log_psi(params, samples))
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

        samples = self.sampler.independent_sample(
            self._log_value_fn(params),
            seed_offset=0,
        )
        local_energies = self._local_energies(project_operator, params, samples)
        return ExpectationResult(mean=jnp.mean(local_energies))


__all__ = [
    "ExpectationResult",
    "ProjectExpectationBackend",
    "SupportsExpectationBackend",
]
