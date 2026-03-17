import jax
import jax.numpy as jnp
from jax import random
from functools import partial
import optax

# Neural Quantum State (architecture wrapper)

class NeuralQuantumState:

    def __init__(self, architecture, params, L):

        self.architecture = architecture
        self.params = params
        self.L = L


    @partial(jax.jit, static_argnames=('self',))
    def logpsi(self, params, sigma):

        return self.architecture.forward(params, sigma)


    @partial(jax.jit, static_argnames=('self',))
    def psi(self, params, sigma):

        return jnp.exp(self.logpsi(params, sigma))


    def flatten_params(self, params):

        flat, _ = jax.flatten_util.ravel_pytree(params)

        return flat


# Feed Forward Network

class FFN:

    def __init__(self, L, hidden_layers):

        self.L = L
        self.hidden_layers = hidden_layers


    def init_params(self, key):

        layer_sizes = [self.L] + self.hidden_layers + [1]

        keys = random.split(key, len(layer_sizes)-1)

        params = []

        for k,(m,n) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):

            W = random.normal(k,(n,m))*jnp.sqrt(2/m)
            b = jnp.zeros(n)

            params.append((W,b))

        return params


    @partial(jax.jit, static_argnames=('self',))
    def forward(self, params, sigma):

        x = sigma

        for W,b in params[:-1]: #exclude output layer

            x = jnp.tanh(W@x + b)

        W,b = params[-1]

        return (W@x + b)[0] #the [0] is to return a float instead of a list


# Restricted Boltzmann Machine

class RBM:

    def __init__(self, L, hidden):

        self.L = L
        self.hidden = hidden


    def init_params(self, key):

        k1,k2,k3 = random.split(key,3) 
        #kept k2 and k3 in case random initialization is needed for a and b

        W = random.normal(k1,(self.hidden,self.L))*0.01
        a = jnp.zeros(self.L)
        b = jnp.zeros(self.hidden)

        return (W,a,b)


    @partial(jax.jit, static_argnames=('self',))
    def forward(self, params, sigma):

        W,a,b = params

        visible = jnp.dot(a,sigma)

        hidden = jnp.sum(jnp.log(2*jnp.cosh(b + W@sigma)))

        return visible + hidden


# Convolutional Neural Network

class CNN:

    def __init__(self, L, channels=16, kernel=3):

        self.L = L
        self.channels = channels #output channels
        self.kernel = kernel


    def init_params(self, key):

        k1,k2,k3 = random.split(key,3)
        #kept k3 in case random initialization is needed for the biases
        conv = random.normal(k1,(self.channels,1,self.kernel))*0.1
        dense = random.normal(k2,(self.channels*self.L,1))*0.1
        bias = jnp.zeros(1)

        return (conv,dense,bias)


    @partial(jax.jit, static_argnames=('self',))
    def forward(self, params, sigma):

        conv,dense,bias = params

        x = sigma.reshape(1,1,self.L) #signle input channel

        x = jax.lax.conv_general_dilated(
            x,
            conv,
            window_strides=(1,),
            padding="SAME",
            dimension_numbers=("NCH","OIH","NCH")
        )

        x = jnp.tanh(x)

        x = x.reshape(-1)

        return (dense.T @ x + bias)[0]


# Metropolis Sampler

class Sampler:

    def __init__(self, wavefunction):

        self.wavefunction = wavefunction 
        self.L = wavefunction.L


    def metropolis_step(self,key,params,sigma):

        key1,key2 = random.split(key)

        site = random.randint(key1,(),0,self.L)

        sigma_new = sigma.at[site].set(-sigma[site])

        psi_old = self.wavefunction.psi(params,sigma)
        psi_new = self.wavefunction.psi(params,sigma_new)

        ratio = jnp.abs(psi_new)**2 / jnp.abs(psi_old)**2

        accept = random.uniform(key2) < jnp.minimum(1.0,ratio)

        sigma = jnp.where(accept,sigma_new,sigma)

        return sigma


    def sample_chain(self,key,params,sigma0,n_samples,burn=200):

        sigma = sigma0
        samples = []

        for i in range(n_samples + burn):

            key,subkey = random.split(key)

            sigma = self.metropolis_step(subkey,params,sigma)

            if i >= burn:

                samples.append(sigma)

        return jnp.stack(samples)


# TFIM Hamiltonian

class TFIM:

    def __init__(self, wavefunction, J, g):

        self.wavefunction = wavefunction
        self.J = J
        self.g = g
        self.L = wavefunction.L


    @partial(jax.jit, static_argnames=('self',))
    def local_energy(self, params, sigma):

        zz = -self.J * jnp.sum(sigma * jnp.roll(sigma,-1))

        psi_sigma = self.wavefunction.psi(params,sigma)

        flip_energy = 0.0

        for i in range(self.L):

            sigma_flip = sigma.at[i].set(-sigma[i])

            psi_flip = self.wavefunction.psi(params,sigma_flip)

            flip_energy += psi_flip / psi_sigma

        return zz - self.g * flip_energy


    @partial(jax.jit, static_argnames=('self',))
    def energy(self, params, samples):

        return jax.vmap(self.local_energy,in_axes=(None,0))(params,samples) 
        #batched collection of local energies for all samples

# Adam Optimizer

class Optimizer:

    def __init__(self, wavefunction, hamiltonian, sampler, lr=1e-3):

        self.wavefunction = wavefunction
        self.hamiltonian = hamiltonian
        self.sampler = sampler

        self.optimizer = optax.adam(lr)

        flat_params = wavefunction.flatten_params(wavefunction.params)

        self.opt_state = self.optimizer.init(flat_params)


    def step(self,key,params,sigma0,n_samples):

        samples = self.sampler.sample_chain(key,params,sigma0,n_samples)

        energies = self.hamiltonian.energy(params,samples)

        meanE = jnp.mean(energies)

        def loss_fn(p):

            return jnp.mean(self.hamiltonian.energy(p,samples))
            #new loss for each batch/set of samples

        grads = jax.grad(loss_fn)(params)

        flat_params,unravel = jax.flatten_util.ravel_pytree(params)
        flat_grads,_ = jax.flatten_util.ravel_pytree(grads)

        updates,self.opt_state = self.optimizer.update(flat_grads,self.opt_state)

        flat_params = optax.apply_updates(flat_params,updates)

        params = unravel(flat_params)

        return params,meanE


    def optimize(self,key,params,sigma0,n_steps,n_samples):

        energies = []

        for step in range(n_steps):

            key,subkey = random.split(key)

            params,E = self.step(subkey,params,sigma0,n_samples)

            energies.append(E)

            print("step",step,"energy",E)

        return params,jnp.array(energies)


# Observables

class Observables:

    def __init__(self,wavefunction):

        self.wavefunction = wavefunction


    def renyi2(self,params,samples,LA):

        N = samples.shape[0]

        swap = 0

        for i in range(N):

            s1 = samples[i]
            s2 = samples[(i+1)%N]

            s1p = jnp.concatenate([s2[:LA],s1[LA:]])
            s2p = jnp.concatenate([s1[:LA],s2[LA:]])

            psi_num = self.wavefunction.psi(params,s1p)*self.wavefunction.psi(params,s2p)
            psi_den = self.wavefunction.psi(params,s1)*self.wavefunction.psi(params,s2)

            swap += psi_num/psi_den

        swap /= N

        return -jnp.log(jnp.abs(swap))



