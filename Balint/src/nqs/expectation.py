from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import jax

from .netket_adapter import NetKetSampler, SupportsLogPsi, states_from_netket

ParamTree = Any


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
class NetKetExpectationBackend:
    """Project-owned expectation backend implemented with NetKet MCState."""

    model: SupportsLogPsi
    sampler: NetKetSampler
    params: ParamTree
    exact_backend_max_states: int = 4096

    def __post_init__(self) -> None:
        self._mc_state = self.sampler.create_mc_state(self.model, self.params)
        self._fullsum_state = (
            self.sampler.create_fullsum_state(self.model, self.params)
            if self.sampler.hilbert.n_states <= self.exact_backend_max_states
            else None
        )

    def set_parameters(self, params: ParamTree) -> None:
        self.params = params
        self._mc_state.parameters = params
        if self._fullsum_state is not None:
            self._fullsum_state.variables = {"params": params}

    def sample(self) -> jax.Array:
        return states_from_netket(self._mc_state.samples)

    def independent_sample(self, seed_offset: int = 0) -> jax.Array:
        temporary_state = self.sampler.create_mc_state(
            self.model,
            self.params,
            seed=self.sampler.seed + seed_offset + 1,
        )
        return states_from_netket(temporary_state.samples)

    def exact_statevector(self) -> jax.Array:
        if self._fullsum_state is None:
            raise ValueError("exact_statevector is only available for small full-summation states.")
        self._fullsum_state.variables = {"params": self.params}
        return self._fullsum_state.to_array()

    def expect(self, operator: Any) -> ExpectationResult:
        if self._fullsum_state is not None:
            self._fullsum_state.variables = {"params": self.params}
            stats = self._fullsum_state.expect(operator)
            return ExpectationResult(mean=stats.mean)
        self._mc_state.parameters = self.params
        stats = self._mc_state.expect(operator)
        return ExpectationResult(mean=stats.mean)

    def expect_and_grad(self, operator: Any) -> tuple[ExpectationResult, ParamTree]:
        if self._fullsum_state is not None:
            self._fullsum_state.variables = {"params": self.params}
            stats, grads = self._fullsum_state.expect_and_grad(operator)
            return ExpectationResult(mean=stats.mean), grads
        self._mc_state.parameters = self.params
        stats, grads = self._mc_state.expect_and_grad(operator)
        return ExpectationResult(mean=stats.mean), grads

    def expect_with_params(
        self,
        operator: Any,
        params: ParamTree,
    ) -> ExpectationResult:
        # The autodiff fallback needs a functional-looking evaluation for
        # candidate parameter trees. NetKet's MCState is stateful, so the safe
        # option here is a temporary cold backend state rather than mutating the
        # warm sampler owned by the main training path.
        temporary_state = self.sampler.create_mc_state(self.model, params)
        stats = temporary_state.expect(operator)
        return ExpectationResult(mean=stats.mean)


__all__ = [
    "ExpectationResult",
    "NetKetExpectationBackend",
    "SupportsExpectationBackend",
]
