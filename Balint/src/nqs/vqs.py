from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import netket as nk

from .netket_adapter import NetKetSampler, SupportsLogPsi, states_from_netket

ParamTree = Any


@dataclass
class VariationalState:
    """Project-owned variational state backed by a NetKet MCState."""

    model: SupportsLogPsi
    params: ParamTree
    sampler: NetKetSampler

    def __post_init__(self) -> None:
        # The wrapped MCState is an implementation detail. We create it once
        # from the current model parameters and replace it whenever parameters
        # change.
        self._mc_state = self.sampler.create_mc_state(self.model, self.params)

    @property
    def parameters(self) -> ParamTree:
        return self.params

    def replace_parameters(self, new_params: ParamTree) -> None:
        # During ordinary sampled VMC updates we are outside JAX tracing, so we
        # can update the live MCState in place and keep the Markov chains warm.
        # This is both faster and better behaved than restarting the sampler
        # every optimization step.
        self.params = new_params
        self._mc_state.parameters = self.params

    def log_value(self, states: jax.Array) -> jax.Array:
        return self.model.log_psi(self.params, states)

    def sample(self) -> jax.Array:
        # NetKet exposes samples using its own encoding. We convert them back to
        # the project's {0, 1} convention before returning them to callers.
        return states_from_netket(self._mc_state.samples)

    def expect(self, operator: nk.operator.AbstractOperator) -> nk.stats.Stats:
        # The expectation call is intentionally thin: operators are delegated to
        # NetKet in this phase, while state ownership remains ours.
        self._mc_state.parameters = self.params
        return self._mc_state.expect(operator)

    def expect_and_grad(self, operator: nk.operator.AbstractOperator) -> tuple[nk.stats.Stats, ParamTree]:
        # NetKet already knows how to compute sampled VMC gradients for the
        # wrapped MCState. Reusing that path avoids rebuilding backend state
        # inside jax.value_and_grad for every optimization step.
        self._mc_state.parameters = self.params
        return self._mc_state.expect_and_grad(operator)

    def expect_with_params(
        self,
        operator: nk.operator.AbstractOperator,
        params: ParamTree,
    ) -> nk.stats.Stats:
        # JAX autodiff requires the differentiated function to avoid mutating
        # shared Python state. Creating a temporary backend state for candidate
        # parameters keeps the loss function pure from JAX's perspective.
        temporary_state = self.sampler.create_mc_state(self.model, params)
        return temporary_state.expect(operator)

    def energy(self, operator: nk.operator.AbstractOperator) -> jax.Array:
        # NetKet returns a structured statistics object; we extract the scalar
        # mean because that is the quantity used by the optimizer.
        return jnp.real(self.expect(operator).mean)

    def energy_with_params(
        self,
        operator: nk.operator.AbstractOperator,
        params: ParamTree,
    ) -> jax.Array:
        return jnp.real(self.expect_with_params(operator, params).mean)
