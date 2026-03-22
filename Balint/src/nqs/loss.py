from __future__ import annotations

from typing import Any

import jax.numpy as jnp

ParamTree = Any


def energy_loss(
    variational_state,
    operator: Any,
    params: ParamTree,
) -> jnp.ndarray:
    # The loss function is deliberately tiny: one place that says "for these
    # parameters, what is the current energy estimate?" This makes it easy to
    # pass into jax.value_and_grad later.
    return variational_state.energy_with_params(operator, params)
