import jax
import jax.numpy as jnp
from functools import partial
from jax.flatten_util import ravel_pytree


class NQS_own():
    def __init__(self,
                 layers: list[int],
                 activation: callable,
                 contraints: callable
                 ):
        self.layers = layers  # dimensions of each layer, including input and output for FFNN ()
        self.activation = activation  # nonlinear activation function
        self.contraints = contraints # function to transform output to comform to constraints, e.g. for TFIM exp



class NeuralQuantumState(object):
    """Constructs the internal neural network and defines the variational wave function."""

    def __init__(self,
                 layers: list[int],
                 activation: callable,
                 conf: float
                 ):

        self.layers = layers  # dimensions of each layer, including input and output
        self.activation = activation  # nonlinear activation function
        self.c = conf  # confinement hyperparameter

    def build(self, key):
        """Returns the weights and biases of the internal neural network"""

        params = []

        for l in range(1, len(self.layers)):
            # define the input and output dimensions of layer l
            dim_in = self.layers[l - 1]
            dim_out = self.layers[l]

            # define Gaussian widths of random initialization
            sigma_W = jnp.sqrt(2 / (dim_in + dim_out))
            sigma_b = 0.001

            # randomly initialize the weights and biases of layer l
            key, subkey_W, subkey_b = jax.random.split(key, num=3)
            W = sigma_W * jax.random.normal(subkey_W, (dim_in, dim_out))
            b = sigma_b * jax.random.normal(subkey_b, (dim_out,))

            # add the weights and biases of layer l to the parameters of the network
            params.append((W, b))

        # flatten parameters
        flat_params = self.flatten_params(params)
        num_params = flat_params.shape[0]
        print("Number of trainable parameters = ", num_params)

        return params

    @partial(jax.jit, static_argnames=('self'))
    def apply_net(self, params, x):
        """ Passes a single value x through the network. Possible to use in batches. """

        # apply hidden layers
        for W, b in params[:-1]:
            x = self.activation(jnp.dot(x, W) + b)

        # apply output layer
        W, b = params[-1]
        return jnp.dot(x, W) + b

    @partial(jax.jit, static_argnames=('self'))
    def logpsi(self, params, x):
        """ Defines the logarithm of the wave function in terms of the neural network. """
        F = self.apply_net(params, x)[0, 0]  # get the output of the neural network
        return F - self.c * jnp.sum(x ** 2)  # apply constraint on boundary conditions

    @partial(jax.jit, static_argnames=('self'))
    def vmap_logpsi(self, params, x):
        """ Helper function for computing logpsi in batches. """
        vmap_logpsi = jax.vmap(self.logpsi, in_axes=(None, 0))(params, x)
        return vmap_logpsi

    @partial(jax.jit, static_argnames=('self'))
    def unflatten_params(self, flat_params):
        """ Helper function for constructing a pytree of parameters from a flat parameter vector. """
        params = self.unravel(flat_params)
        return params

    @partial(jax.jit, static_argnames=('self'))
    def flatten_params(self, params):
        """ Helper function for flattening a pytree into a parameter vector. """
        flat_params, self.unravel = ravel_pytree(params)
        return flat_params
