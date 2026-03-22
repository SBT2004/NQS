from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from math import ceil
from typing import Any, Callable

import jax
import jax.numpy as jnp

from .hilbert import SpinHilbert


LogPsiFn = Callable[[jax.Array], jax.Array]
LogPsiApplyFn = Callable[[Any, jax.Array], jax.Array]


@dataclass(frozen=True)
class SampleBatch:
    """Sampled configurations with the log-values used to accept them."""

    states: jax.Array
    log_values: jax.Array


def _metropolis_step(
    log_psi_apply: LogPsiApplyFn,
    params: Any,
    *,
    states: jax.Array,
    log_values: jax.Array,
    rng_key: jax.Array,
    n_chains: int,
    n_sites: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    site_key, accept_key, next_key = jax.random.split(rng_key, 3)
    proposal_sites = jax.random.randint(
        site_key,
        shape=(n_chains,),
        minval=0,
        maxval=n_sites,
    )
    proposal_rows = jnp.arange(n_chains)
    proposed_states = states.at[proposal_rows, proposal_sites].set(
        1 - states[proposal_rows, proposal_sites]
    )
    proposed_log_values = jnp.asarray(log_psi_apply(params, proposed_states))
    log_ratio = 2.0 * jnp.real(proposed_log_values - log_values)
    log_uniform = jnp.log(
        jax.random.uniform(
            accept_key,
            shape=(n_chains,),
            minval=jnp.finfo(jnp.float32).tiny,
            maxval=1.0,
        )
    )
    accepted = log_uniform < jnp.minimum(log_ratio, 0.0)
    next_states = jnp.where(accepted[:, None], proposed_states, states)
    next_log_values = jnp.where(accepted, proposed_log_values, log_values)
    return next_states, next_log_values, next_key


def _scan_samples(
    log_psi_apply: LogPsiApplyFn,
    params: Any,
    *,
    initial_states: jax.Array,
    rng_key: jax.Array,
    n_chains: int,
    n_sites: int,
    n_discard_per_chain: int,
    steps_per_chain: int,
    n_samples: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    states = jnp.asarray(initial_states, dtype=jnp.uint8)
    log_values = jnp.asarray(log_psi_apply(params, states))

    def thermalize_step(
        carry: tuple[jax.Array, jax.Array, jax.Array],
        _,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
        current_states, current_log_values, current_rng_key = carry
        next_states, next_log_values, next_rng_key = _metropolis_step(
            log_psi_apply,
            params,
            states=current_states,
            log_values=current_log_values,
            rng_key=current_rng_key,
            n_chains=n_chains,
            n_sites=n_sites,
        )
        return (next_states, next_log_values, next_rng_key), None

    def sample_step(
        carry: tuple[jax.Array, jax.Array, jax.Array],
        _,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
        current_states, current_log_values, current_rng_key = carry
        next_states, next_log_values, next_rng_key = _metropolis_step(
            log_psi_apply,
            params,
            states=current_states,
            log_values=current_log_values,
            rng_key=current_rng_key,
            n_chains=n_chains,
            n_sites=n_sites,
        )
        return (next_states, next_log_values, next_rng_key), (next_states, next_log_values)

    carry = (states, log_values, rng_key)
    if n_discard_per_chain > 0:
        carry, _ = jax.lax.scan(
            thermalize_step,
            carry,
            xs=None,
            length=n_discard_per_chain,
        )
    carry, collected = jax.lax.scan(
        sample_step,
        carry,
        xs=None,
        length=steps_per_chain,
    )
    final_states, _, final_key = carry
    collected_states, collected_log_values = collected
    flat_samples = collected_states.reshape((steps_per_chain * n_chains, n_sites))[:n_samples]
    flat_log_values = collected_log_values.reshape((steps_per_chain * n_chains,))[:n_samples]
    return flat_samples, flat_log_values, final_states, final_key


def _scan_samples_unary(
    log_psi_fn: LogPsiFn,
    *,
    initial_states: jax.Array,
    rng_key: jax.Array,
    n_chains: int,
    n_sites: int,
    n_discard_per_chain: int,
    steps_per_chain: int,
    n_samples: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    def apply_unary(_: None, states: jax.Array) -> jax.Array:
        return log_psi_fn(states)

    return _scan_samples(
        apply_unary,
        None,
        initial_states=initial_states,
        rng_key=rng_key,
        n_chains=n_chains,
        n_sites=n_sites,
        n_discard_per_chain=n_discard_per_chain,
        steps_per_chain=steps_per_chain,
        n_samples=n_samples,
    )


@dataclass
class MetropolisLocal:
    """Project-owned local Metropolis sampler for spin-1/2 states."""

    hilbert: SpinHilbert
    n_samples: int = 64
    n_discard_per_chain: int = 8
    n_chains: int = 16
    seed: int = 0
    _rng_key: jax.Array = field(init=False, repr=False)
    _chain_states: jax.Array | None = field(default=None, init=False, repr=False)
    _compiled_draw_samples: Callable[..., tuple[jax.Array, jax.Array, jax.Array, jax.Array]] = field(
        init=False,
        repr=False,
    )
    _compiled_draw_samples_unary: Callable[..., tuple[jax.Array, jax.Array, jax.Array, jax.Array]] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        if self.n_discard_per_chain < 0:
            raise ValueError("n_discard_per_chain must be non-negative.")
        if self.n_chains <= 0:
            raise ValueError("n_chains must be positive.")
        self._rng_key = jax.random.PRNGKey(self.seed)
        self._compiled_draw_samples = jax.jit(
            partial(
                _scan_samples,
                n_chains=self.n_chains,
                n_sites=self.hilbert.size,
                n_discard_per_chain=self.n_discard_per_chain,
                steps_per_chain=self._steps_per_chain(),
                n_samples=self.n_samples,
            ),
            static_argnames=("log_psi_apply",),
        )
        self._compiled_draw_samples_unary = jax.jit(
            partial(
                _scan_samples_unary,
                n_chains=self.n_chains,
                n_sites=self.hilbert.size,
                n_discard_per_chain=self.n_discard_per_chain,
                steps_per_chain=self._steps_per_chain(),
                n_samples=self.n_samples,
            ),
            static_argnames=("log_psi_fn",),
        )

    def _steps_per_chain(self) -> int:
        return max(1, ceil(self.n_samples / self.n_chains))

    def _random_states(self, rng_key: jax.Array) -> jax.Array:
        return jax.random.bernoulli(
            rng_key,
            p=0.5,
            shape=(self.n_chains, self.hilbert.size),
        ).astype(jnp.uint8)

    def _draw_samples(
        self,
        log_psi_fn: LogPsiFn,
        *,
        initial_states: jax.Array,
        rng_key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        return self._compiled_draw_samples_unary(
            log_psi_fn=log_psi_fn,
            initial_states=initial_states,
            rng_key=rng_key,
        )

    def _draw_samples_with_params(
        self,
        log_psi_apply: LogPsiApplyFn,
        params: Any,
        *,
        initial_states: jax.Array,
        rng_key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        return self._compiled_draw_samples(
            log_psi_apply=log_psi_apply,
            params=params,
            initial_states=initial_states,
            rng_key=rng_key,
        )

    def sample(self, log_psi_fn: LogPsiFn) -> jax.Array:
        return self.sample_with_log_values(log_psi_fn).states

    def sample_with_log_values(self, log_psi_fn: LogPsiFn) -> SampleBatch:
        init_key, sample_key = jax.random.split(self._rng_key)
        if self._chain_states is None:
            self._chain_states = self._random_states(init_key)
        samples, log_values, final_states, final_key = self._draw_samples(
            log_psi_fn,
            initial_states=self._chain_states,
            rng_key=sample_key,
        )
        self._chain_states = final_states
        self._rng_key = final_key
        return SampleBatch(states=samples, log_values=log_values)

    def sample_with_params(
        self,
        log_psi_apply: LogPsiApplyFn,
        params: Any,
    ) -> jax.Array:
        return self.sample_with_params_and_log_values(log_psi_apply, params).states

    def sample_with_params_and_log_values(
        self,
        log_psi_apply: LogPsiApplyFn,
        params: Any,
    ) -> SampleBatch:
        init_key, sample_key = jax.random.split(self._rng_key)
        if self._chain_states is None:
            self._chain_states = self._random_states(init_key)
        samples, log_values, final_states, final_key = self._draw_samples_with_params(
            log_psi_apply,
            params,
            initial_states=self._chain_states,
            rng_key=sample_key,
        )
        self._chain_states = final_states
        self._rng_key = final_key
        return SampleBatch(states=samples, log_values=log_values)

    def independent_sample(
        self,
        log_psi_fn: LogPsiFn,
        *,
        seed_offset: int = 0,
    ) -> jax.Array:
        return self.independent_sample_with_log_values(
            log_psi_fn,
            seed_offset=seed_offset,
        ).states

    def independent_sample_with_log_values(
        self,
        log_psi_fn: LogPsiFn,
        *,
        seed_offset: int = 0,
    ) -> SampleBatch:
        rng_key = jax.random.PRNGKey(self.seed + seed_offset + 1)
        init_key, sample_key = jax.random.split(rng_key)
        initial_states = self._random_states(init_key)
        samples, log_values, _, _ = self._draw_samples(
            log_psi_fn,
            initial_states=initial_states,
            rng_key=sample_key,
        )
        return SampleBatch(states=samples, log_values=log_values)

    def independent_sample_with_params(
        self,
        log_psi_apply: LogPsiApplyFn,
        params: Any,
        *,
        seed_offset: int = 0,
    ) -> jax.Array:
        return self.independent_sample_with_params_and_log_values(
            log_psi_apply,
            params,
            seed_offset=seed_offset,
        ).states

    def independent_sample_with_params_and_log_values(
        self,
        log_psi_apply: LogPsiApplyFn,
        params: Any,
        *,
        seed_offset: int = 0,
    ) -> SampleBatch:
        rng_key = jax.random.PRNGKey(self.seed + seed_offset + 1)
        init_key, sample_key = jax.random.split(rng_key)
        initial_states = self._random_states(init_key)
        samples, log_values, _, _ = self._draw_samples_with_params(
            log_psi_apply,
            params,
            initial_states=initial_states,
            rng_key=sample_key,
        )
        return SampleBatch(states=samples, log_values=log_values)

__all__ = [
    "LogPsiFn",
    "LogPsiApplyFn",
    "MetropolisLocal",
    "SampleBatch",
]
