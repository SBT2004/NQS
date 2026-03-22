from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from .expectation import ProjectExpectationBackend, SupportsExpectationBackend
from .runtime_types import SupportsLogPsi
from .sampler import MetropolisLocal

ParamTree = Any


@dataclass
class VariationalState:
    """Project-owned variational state backed by a project-owned expectation backend."""

    model: SupportsLogPsi
    params: ParamTree
    sampler: MetropolisLocal
    exact_backend_max_states: int = 4096

    def __post_init__(self) -> None:
        self._expectation_backend: SupportsExpectationBackend = ProjectExpectationBackend(
            model=self.model,
            sampler=self.sampler,
            params=self.params,
            exact_backend_max_states=self.exact_backend_max_states,
        )

    @property
    def parameters(self) -> ParamTree:
        return self.params

    def replace_parameters(self, new_params: ParamTree) -> None:
        self.params = new_params
        self._expectation_backend.set_parameters(self.params)

    def log_value(self, states: jax.Array) -> jax.Array:
        return self.model.log_psi(self.params, states)

    def sample(self) -> jax.Array:
        return self._expectation_backend.sample()

    def independent_sample(self, seed_offset: int = 0) -> jax.Array:
        return self._expectation_backend.independent_sample(seed_offset=seed_offset)

    def exact_statevector(self) -> jax.Array:
        return self._expectation_backend.exact_statevector()

    def expect(self, operator: Any):
        return self._expectation_backend.expect(operator)

    def expect_and_grad(self, operator: Any):
        return self._expectation_backend.expect_and_grad(operator)

    def expect_with_params(
        self,
        operator: Any,
        params: ParamTree,
    ):
        return self._expectation_backend.expect_with_params(operator, params)

    def energy(self, operator: Any) -> jax.Array:
        return jnp.real(self.expect(operator).mean)

    def energy_with_params(
        self,
        operator: Any,
        params: ParamTree,
    ) -> jax.Array:
        return jnp.real(self.expect_with_params(operator, params).mean)
