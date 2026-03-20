from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import netket as nk

from .hilbert import SpinHilbert

ParamTree = Any


class SupportsLogPsi(Protocol):
    def log_psi(self, params: ParamTree, states: jax.Array) -> jax.Array:
        ...


def states_to_netket(states: jax.Array) -> jax.Array:
    # NetKet's spin-1/2 convention in this project is {-1, +1}, while our
    # local project convention is {0, 1}. This helper isolates that mismatch.
    return 2.0 * states.astype(jnp.float32) - 1.0


def states_from_netket(states: jax.Array) -> jax.Array:
    # Mapping back through a sign/threshold check is enough because NetKet's
    # sampled spin values are exactly two-valued for this Hilbert space.
    return (states > 0).astype(jnp.uint8)


@dataclass
class NetKetSampler:
    """Temporary NetKet-backed sampler and expectation backend."""

    hilbert: SpinHilbert
    n_samples: int = 64
    n_discard_per_chain: int = 8
    n_chains: int = 16
    seed: int = 0

    def __post_init__(self) -> None:
        # This adapter is the only place that constructs a NetKet Hilbert space.
        # The rest of the project should continue thinking in terms of our own
        # SpinHilbert class.
        self.netket_hilbert = nk.hilbert.Spin(s=0.5, N=self.hilbert.size)
        self.sampler = nk.sampler.MetropolisLocal(self.netket_hilbert, n_chains=self.n_chains)

    def _make_apply_fun(self, model: SupportsLogPsi):
        def apply_fun(
            variables: dict[str, Any],
            sigma: jax.Array,
            mutable: Any = False,
            **_: Any,
        ) -> jax.Array | tuple[jax.Array, dict[str, Any]]:
            # NetKet calls this function with its own spin convention. We
            # translate into our project convention, evaluate our JAX model, and
            # return the resulting log-amplitudes back to NetKet.
            project_states = states_from_netket(sigma)
            output = model.log_psi(variables["params"], project_states)
            if mutable:
                return output, {}
            return output
        return apply_fun

    def create_mc_state(
        self,
        model: SupportsLogPsi,
        params: ParamTree,
        *,
        seed: int | None = None,
    ) -> nk.vqs.MCState:
        apply_fun = self._make_apply_fun(model)
        state_seed = self.seed if seed is None else seed

        # MCState is used only as a temporary backend object. The public API is
        # still our own VariationalState wrapper.
        return nk.vqs.MCState(
            self.sampler,
            apply_fun=apply_fun,
            variables={"params": params},
            n_samples=self.n_samples,
            n_discard_per_chain=self.n_discard_per_chain,
            seed=state_seed,
            sampler_seed=state_seed,
        )

    def create_fullsum_state(
        self,
        model: SupportsLogPsi,
        params: ParamTree,
    ) -> nk.vqs.FullSumState:
        return nk.vqs.FullSumState(
            self.netket_hilbert,
            apply_fun=self._make_apply_fun(model),
            variables={"params": params},
        )
