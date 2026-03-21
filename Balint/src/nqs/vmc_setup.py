from __future__ import annotations

from typing import Any

import jax

from .driver import VMC
from .netket_adapter import NetKetSampler
from .optimizer import Adam
from .vqs import VariationalState


def build_variational_state(
    *,
    model: Any,
    hilbert: Any,
    seed: int,
    n_samples: int,
    n_discard_per_chain: int,
    n_chains: int,
    params: Any | None = None,
) -> VariationalState:
    model_params = params if params is not None else model.init(jax.random.PRNGKey(seed), hilbert)
    sampler = NetKetSampler(
        hilbert=hilbert,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
        n_chains=n_chains,
        seed=seed,
    )
    return VariationalState(model=model, params=model_params, sampler=sampler)


def build_vmc_driver(
    *,
    model: Any,
    hilbert: Any,
    operator: Any,
    learning_rate: float,
    seed: int,
    n_samples: int,
    n_discard_per_chain: int,
    n_chains: int,
    params: Any | None = None,
) -> tuple[VariationalState, VMC]:
    state = build_variational_state(
        model=model,
        hilbert=hilbert,
        seed=seed,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
        n_chains=n_chains,
        params=params,
    )
    driver = VMC(
        operator=operator,
        variational_state=state,
        optimizer=Adam(learning_rate=learning_rate),
    )
    return state, driver
