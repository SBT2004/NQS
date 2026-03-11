from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import optax

ParamTree = Any


@dataclass
class Adam:
    """Project-owned Adam optimizer wrapper using JAX gradients."""

    learning_rate: float = 1e-3

    def __post_init__(self) -> None:
        # Optax provides the update-rule mechanics, but the project keeps its
        # own API so later changes to the optimizer backend do not leak into the
        # rest of the codebase.
        self._optimizer = optax.adam(self.learning_rate)

    def init(self, params: ParamTree) -> optax.OptState:
        # Optimizers like Adam keep extra running statistics in addition to the
        # parameters themselves. Optax calls that optimizer state.
        return self._optimizer.init(params)

    def compute_gradients(
        self,
        loss_fn: Callable[..., jax.Array],
        params: ParamTree,
        *args,
        **kwargs,
    ) -> tuple[jax.Array, Any]:
        # jax.value_and_grad returns both the scalar loss and its gradient in one
        # pass. The gradient has the same nested structure ("pytree") as params.
        value, grads = jax.value_and_grad(loss_fn)(params, *args, **kwargs)
        return value, grads

    def update(
        self,
        grads: Any,
        params: ParamTree,
        opt_state: optax.OptState,
    ) -> tuple[ParamTree, optax.OptState]:
        # Optax splits updating into two steps:
        # 1. turn gradients into parameter updates
        # 2. apply those updates to the parameter tree
        updates, new_opt_state = self._optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state
