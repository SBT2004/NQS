from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict

from .hilbert import SpinHilbert

StateBatch = jax.Array | np.ndarray
ParamTree = Any


def _states_to_pm1(states: jax.Array) -> jax.Array:
    # Our Hilbert space stores spins as {0, 1}, but neural-network formulas are
    # usually written for {-1, +1}. Keeping the conversion in one helper makes
    # the rest of the model code easier to read.
    return 2.0 * states.astype(jnp.float32) - 1.0


def _complex_log_output(log_amplitude: jax.Array, raw_phase: jax.Array) -> jax.Array:
    phase = jnp.pi * jnp.tanh(raw_phase)
    return log_amplitude + 1j * phase


def _dense_complex_head(output: jax.Array) -> jax.Array:
    """Assume `output[..., 0]` is log amplitude and `output[..., 1]` is raw phase."""

    return _complex_log_output(output[..., 0], output[..., 1])


class _RBMModule(nn.Module):
    hidden_features: int

    @nn.compact
    def __call__(self, states: jax.Array) -> jax.Array:
        # In Flax, parameters are created the first time the module is called.
        # The @nn.compact decorator allows defining layers and parameters inside
        # __call__ instead of splitting construction and forward pass manually.
        x = _states_to_pm1(states)
        # self.param creates a trainable parameter managed by Flax/JAX. The
        # returned value behaves like an array, but it is tracked in the
        # parameter pytree so JAX can differentiate with respect to it.
        visible_bias = self.param("visible_bias", nn.initializers.zeros, (x.shape[-1],))
        phase_bias = self.param("phase_bias", nn.initializers.zeros, (x.shape[-1],))
        # nn.Dense is a standard linear layer: x @ W + b.
        hidden = nn.Dense(self.hidden_features)(x)
        phase_hidden = nn.Dense(self.hidden_features, name="phase_hidden")(x)
        # This is the usual RBM log-amplitude formula for a visible
        # configuration after summing out hidden spins analytically.
        log_amplitude = jnp.dot(x, visible_bias) + jnp.sum(jnp.log(jnp.cosh(hidden)), axis=-1)
        raw_phase = jnp.dot(x, phase_bias) + jnp.sum(phase_hidden, axis=-1)
        return _complex_log_output(log_amplitude, raw_phase)


class _FFNNModule(nn.Module):
    hidden_dims: tuple[int, ...]
    activation: Callable[[jax.Array], jax.Array]

    @nn.compact
    def __call__(self, states: jax.Array) -> jax.Array:
        x = _states_to_pm1(states)
        # This loop builds a simple multilayer perceptron. In Flax, creating
        # layers inside a loop is normal as long as the loop structure is fixed.
        for width in self.hidden_dims:
            x = nn.Dense(width)(x)
            x = self.activation(x)
        # We return one scalar per configuration because the wavefunction model
        # needs one log-amplitude per sampled spin state.
        output = nn.Dense(2)(x)
        return _dense_complex_head(output)


class _CNNModule(nn.Module):
    spatial_shape: tuple[int, int]
    channels: tuple[int, ...]
    kernel_size: tuple[int, int]
    activation: Callable[[jax.Array], jax.Array]

    @nn.compact
    def __call__(self, states: jax.Array) -> jax.Array:
        x = _states_to_pm1(states)
        batch = x.shape[0]
        # CNNs expect image-like tensors, so we reshape a flat spin string into
        # (batch, height, width, channels). The last dimension is 1 because the
        # spin configuration is a single scalar field.
        x = x.reshape((batch, self.spatial_shape[0], self.spatial_shape[1], 1))
        for channel in self.channels:
            x = nn.Conv(features=channel, kernel_size=self.kernel_size, padding="SAME")(x)
            x = self.activation(x)
        # After the convolution stack we flatten back to a vector so a final
        # dense layer can produce the scalar log-amplitude.
        x = x.reshape((batch, -1))
        output = nn.Dense(2)(x)
        return _dense_complex_head(output)


@dataclass
class BaseModel:
    """Project-owned model interface for neural quantum states."""

    module: nn.Module

    def init(self, rng_key: jax.Array, hilbert: SpinHilbert) -> ParamTree:
        # JAX/Flax separates parameter creation from later evaluation. To create
        # parameters we run the module once on a dummy input with the right
        # shape. The returned dictionary contains all variables; we only keep
        # the trainable "params" collection here.
        dummy_state = jnp.zeros((1, hilbert.size), dtype=jnp.float32)
        variables = self.module.init(rng_key, dummy_state)
        params = variables["params"]
        if isinstance(params, FrozenDict):
            return dict(params)
        return params

    def apply(self, params: ParamTree, states: StateBatch) -> jax.Array:
        # jnp.atleast_2d lets us support both a single state and a batch of
        # states through the same code path. This keeps the high-level API
        # simple for demos.
        batched_states = jnp.atleast_2d(states)
        # Flax expects parameters wrapped inside {"params": ...} because that is
        # the standard variable-tree format used by module.apply.
        result = self.module.apply({"params": params}, batched_states)
        return jnp.asarray(result)

    def log_psi(self, params: ParamTree, states: StateBatch) -> jax.Array:
        # At this stage the raw network output is directly interpreted as the
        # logarithm of the wavefunction amplitude.
        return self.apply(params, states)


class RBM(BaseModel):
    def __init__(self, alpha: int = 2) -> None:
        self.alpha = alpha
        # We store a placeholder module first and replace it during init once we
        # know the Hilbert-space size. RBM hidden size depends on the number of
        # visible spins, so it cannot be finalized at construction time alone.
        super().__init__(module=_RBMModule(hidden_features=1))

    def init(self, rng_key: jax.Array, hilbert: SpinHilbert) -> ParamTree:
        module = _RBMModule(hidden_features=max(1, self.alpha * hilbert.size))
        # The RBM hidden layer size depends on the Hilbert size, so the actual
        # Flax module is finalized here once that size is known.
        self.module = module
        return super().init(rng_key, hilbert)


class FFNN(BaseModel):
    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (32, 16),
        activation: Callable[[jax.Array], jax.Array] = jax.nn.tanh,
    ) -> None:
        self.hidden_dims = hidden_dims
        self.activation = activation
        super().__init__(module=_FFNNModule(hidden_dims=hidden_dims, activation=activation))


class CNN(BaseModel):
    def __init__(
        self,
        spatial_shape: tuple[int, int],
        channels: tuple[int, ...] = (8, 4),
        kernel_size: tuple[int, int] = (3, 3),
        activation: Callable[[jax.Array], jax.Array] = jax.nn.tanh,
    ) -> None:
        self.spatial_shape = spatial_shape
        self.channels = channels
        self.kernel_size = kernel_size
        self.activation = activation
        super().__init__(
            module=_CNNModule(
                spatial_shape=spatial_shape,
                channels=channels,
                kernel_size=kernel_size,
                activation=activation,
            )
        )
