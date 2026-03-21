from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax

from .driver import VMC
from .models import CNN, FFNN, RBM
from .optimizer import Adam
from .sampler import MetropolisLocal
from .vqs import VariationalState


def build_model(
    *,
    model_name: str,
    model_kwargs: Mapping[str, Any] | None = None,
    lattice_shape: tuple[int, int] | None = None,
) -> Any:
    kwargs = {} if model_kwargs is None else dict(model_kwargs)
    normalized_name = model_name.upper()
    if normalized_name == "RBM":
        return RBM(**kwargs)
    if normalized_name == "FFNN":
        return FFNN(**kwargs)
    if normalized_name == "CNN":
        if lattice_shape is None:
            raise ValueError("lattice_shape is required when building a CNN model.")
        kwargs.setdefault("spatial_shape", lattice_shape)
        return CNN(**kwargs)
    raise ValueError(f"Unsupported model_name: {model_name}")


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
    sampler = MetropolisLocal(
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


def build_vmc_experiment(
    *,
    hilbert: Any,
    operator: Any,
    learning_rate: float,
    seed: int,
    n_samples: int,
    n_discard_per_chain: int,
    n_chains: int,
    params: Any | None = None,
    model: Any | None = None,
    model_name: str | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
    lattice_shape: tuple[int, int] | None = None,
) -> tuple[Any, VariationalState, VMC]:
    experiment_model = model
    if experiment_model is None:
        if model_name is None:
            raise ValueError("build_vmc_experiment requires either model or model_name.")
        experiment_model = build_model(
            model_name=model_name,
            model_kwargs=model_kwargs,
            lattice_shape=lattice_shape,
        )

    state, driver = build_vmc_driver(
        model=experiment_model,
        hilbert=hilbert,
        operator=operator,
        learning_rate=learning_rate,
        seed=seed,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
        n_chains=n_chains,
        params=params,
    )
    return experiment_model, state, driver
