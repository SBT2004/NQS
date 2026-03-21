from __future__ import annotations

from typing import Any, Protocol

import jax
import jax.numpy as jnp


ParamTree = Any


class SupportsLogPsi(Protocol):
    def log_psi(self, params: ParamTree, states: jax.Array) -> jax.Array:
        ...


def states_to_signed_spins(states: jax.Array) -> jax.Array:
    # The signed spin convention for spin-1/2 states is {-1, +1}, while the
    # project's local bitstring convention is {0, 1}.
    return 2.0 * states.astype(jnp.float32) - 1.0


def states_from_signed_spins(states: jax.Array) -> jax.Array:
    # Mapping back through a sign/threshold check is enough because the signed
    # spin values are exactly two-valued for this Hilbert space.
    return (states > 0).astype(jnp.uint8)


__all__ = [
    "ParamTree",
    "SupportsLogPsi",
    "states_from_signed_spins",
    "states_to_signed_spins",
]
