from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import jax
import jax.numpy as jnp

from .expectation import ExpectationResult
from .loss import energy_loss

ParamTree = Any
Callback = Callable[[int, Any], dict[str, object]]


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

    def expect_and_grad(self, operator: Any) -> tuple[ExpectationResult, Any]:
        ...


@dataclass
class VMC:
    """Minimal project-owned VMC driver."""

    operator: Any
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
            result, grads = self.variational_state.expect_and_grad(self.operator)
            energy = jnp.real(result.mean)
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

    def run(
        self,
        n_iter: int,
        callback: Callback | None = None,
        callback_every: int = 1,
    ) -> list[dict[str, object]]:
        if callback_every <= 0:
            raise ValueError("callback_every must be positive.")

        history: list[dict[str, object]] = []
        for step in range(n_iter):
            step_result = dict(self.step())
            step_result["step"] = step
            if callback is not None and step % callback_every == 0:
                step_result["observables"] = dict(callback(step, self))
            history.append(step_result)
        return history
