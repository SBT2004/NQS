import jax
import jax.numpy as jnp
from jax import random
from jax.flatten_util import ravel_pytree
from functools import partial


class NeuralQuantumState:
    def __init__(self, architecture, params, L):
        self.architecture = architecture
        self.params = params
        self.L = L

    @partial(jax.jit, static_argnames=("self",))
    def logpsi(self, params, sigma):
        return self.architecture.forward(params, sigma)

    def vmap_logpsi(self, params, sigmas):
        return jax.vmap(self.logpsi, in_axes=(None, 0))(params, sigmas)

    def flatten_params(self, params):
        flat, unravel = ravel_pytree(params)
        return flat, unravel


class RBM:
    def __init__(self, L, hidden=20, init_scale=0.1):
        self.L = L
        self.hidden = hidden
        self.init_scale = init_scale

    def init_params(self, key):
        k1, k2, k3 = random.split(key, 3)
        W = random.normal(k1, (self.hidden, self.L)) * self.init_scale
        a = jnp.zeros(self.L)
        b = jnp.zeros(self.hidden)
        return (W, a, b)

    @partial(jax.jit, static_argnames=("self",))
    def forward(self, params, sigma):
        W, a, b = params
        visible = jnp.dot(a, sigma)
        hidden = jnp.sum(jnp.log(2.0 * jnp.cosh(b + W @ sigma)))
        return visible + hidden


class FFN:
    def __init__(self, L, hidden_layers=(32, 32), init_scale=1.0):
        self.L = L
        self.hidden_layers = list(hidden_layers)
        self.init_scale = init_scale

    def init_params(self, key):
        layer_sizes = [self.L] + self.hidden_layers + [1]
        keys = random.split(key, len(layer_sizes) - 1)
        params = []
        for k, (m, n) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):
            W = random.normal(k, (n, m)) * self.init_scale * jnp.sqrt(2.0 / m)
            b = jnp.zeros(n)
            params.append((W, b))
        return params

    @partial(jax.jit, static_argnames=("self",))
    def forward(self, params, sigma):
        x = sigma
        for W, b in params[:-1]:
            x = jnp.tanh(W @ x + b)
        W, b = params[-1]
        return (W @ x + b)[0]


class CNN:
    def __init__(self, L, channels=16, kernel=3, n_conv_layers=1, init_scale=0.5):
        self.L = L
        self.channels = channels
        self.kernel = kernel
        self.n_conv_layers = n_conv_layers
        self.init_scale = init_scale

    def init_params(self, key):
        keys = random.split(key, self.n_conv_layers + 1)

        conv_params = []

        W0 = random.normal(keys[0], (self.channels, 1, self.kernel)) * self.init_scale
        b0 = jnp.zeros(self.channels)
        conv_params.append((W0, b0))

        for k in keys[1:-1]:
            W = random.normal(k, (self.channels, self.channels, self.kernel)) * self.init_scale
            b = jnp.zeros(self.channels)
            conv_params.append((W, b))

        dense = random.normal(keys[-1], (self.channels * self.L, 1)) * self.init_scale
        bias = jnp.zeros(1)

        return (conv_params, dense, bias)

    @partial(jax.jit, static_argnames=("self",))
    def forward(self, params, sigma):
        conv_params, dense, bias = params
        x = sigma.reshape(1, 1, self.L).astype(jnp.float32)

        for W, b in conv_params:
            x = jax.lax.conv_general_dilated(
                x,
                W,
                window_strides=(1,),
                padding="SAME",
                dimension_numbers=("NCH", "OIH", "NCH"),
            )
            x = x + b[None, :, None]
            x = jnp.tanh(x)

        x = x.reshape(-1)
        return (dense.T @ x + bias)[0]


def count_parameters(params):
    flat, _ = ravel_pytree(params)
    return int(flat.size)