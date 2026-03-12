from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import netket as nk

from .loss import energy_loss

ParamTree = Any


class SupportsOptimizer(Protocol):
    def init(self, params: ParamTree) -> object:
        ...

    def compute_gradients(
        self,
        loss_fn,
        params: ParamTree,
        *args,
        **kwargs,
    ) -> tuple[jax.Array, Any]:
        ...

    def update(
        self,
        grads: Any,
        params: ParamTree,
        opt_state: Any,
    ) -> tuple[ParamTree, Any]:
        ...


class SupportsVariationalState(Protocol):
    @property
    def parameters(self) -> ParamTree:
        ...

    def replace_parameters(self, new_params: ParamTree) -> None:
        ...

    def expect_and_grad(self, operator: nk.operator.AbstractOperator) -> tuple[nk.stats.Stats, Any]:
        ...


@dataclass
class VMC:
    """Minimal project-owned VMC driver."""

    operator: nk.operator.AbstractOperator
    variational_state: SupportsVariationalState
    optimizer: SupportsOptimizer
    opt_state: object = field(init=False)

    def __post_init__(self) -> None:
        # The optimizer state depends on the initial parameters, so it is
        # created once when the driver is built.
        self.opt_state = self.optimizer.init(self.variational_state.parameters)

    def step(self) -> dict[str, object]:
        # Prefer the sampled VMC gradient path when the variational state can
        # provide it directly. This keeps the demo on an actual Monte Carlo
        # training path instead of rebuilding backend objects inside autodiff.
        if hasattr(self.variational_state, "expect_and_grad"):
            stats, grads = self.variational_state.expect_and_grad(self.operator)
            energy = jnp.real(stats.mean)
        else:
            def loss_fn(params):
                # This nested function captures the fixed operator and
                # variational state, leaving JAX to differentiate only with
                # respect to params.
                return energy_loss(self.variational_state, self.operator, params)

            # Step 1: evaluate the current scalar objective and its gradient.
            energy, grads = self.optimizer.compute_gradients(loss_fn, self.variational_state.parameters)
        # Step 2: convert gradients into an updated parameter set.
        new_params, self.opt_state = self.optimizer.update(
            grads,
            self.variational_state.parameters,
            self.opt_state,
        )
        # Step 3: store the new parameters back into the project-owned state.
        self.variational_state.replace_parameters(new_params)
        return {"energy": energy, "grads": grads}

    def run(self, n_iter: int) -> list[dict[str, object]]:
        # We return a simple history list so demos can inspect energies without
        # needing a separate logger abstraction yet.
        history: list[dict[str, object]] = []
        for _ in range(n_iter):
            history.append(self.step())
        return history
