import jax
import jax.numpy as jnp


def enumerate_spin_basis(L):
    states = ((jnp.arange(2**L)[:, None] >> jnp.arange(L)) & 1)
    states = 2 * states - 1
    return states.astype(jnp.int32)


def exact_tfim_hamiltonian(L, J, g):
    dim = 2 ** L
    states = enumerate_spin_basis(L)

    H = jnp.zeros((dim, dim), dtype=jnp.float32)

    diag = -J * jnp.sum(states * jnp.roll(states, -1, axis=1), axis=1)
    H = H.at[jnp.arange(dim), jnp.arange(dim)].set(diag)

    for i in range(L):
        flipped = states.at[:, i].set(-states[:, i])
        bits = ((flipped + 1) // 2).astype(jnp.int32)
        idx = jnp.sum(bits * (2 ** jnp.arange(L)), axis=1)
        H = H.at[jnp.arange(dim), idx].add(-g)

    return H


def exact_tfim_ground_energy(L, J, g):
    H = exact_tfim_hamiltonian(L, J, g)
    evals = jnp.linalg.eigvalsh(H)
    return evals[0]


def exact_tfim_ground_state(L, J, g):
    H = exact_tfim_hamiltonian(L, J, g)
    evals, evecs = jnp.linalg.eigh(H)
    return evals[0], evecs[:, 0]
import jax.numpy as jnp

def reduced_density_matrix_from_statevector(state, L, subsystem_size):
    """
    Build rho_A from a pure statevector on L spins by tracing out B.

    state:
        shape (2^L,)
    subsystem_size:
        size of A, taken as the first subsystem_size spins
    """
    LA = int(subsystem_size)
    LB = L - LA

    psi = state.reshape((2**LA, 2**LB))
    rho_A = psi @ jnp.conjugate(psi.T)
    return rho_A


def von_neumann_entropy_from_statevector(state, L, subsystem_size, eps=1e-12):
    rho_A = reduced_density_matrix_from_statevector(state, L, subsystem_size)
    evals = jnp.linalg.eigvalsh(rho_A)
    evals = jnp.clip(jnp.real(evals), eps, 1.0)
    return -jnp.sum(evals * jnp.log(evals))


def renyi2_entropy_from_statevector(state, L, subsystem_size, eps=1e-12):
    rho_A = reduced_density_matrix_from_statevector(state, L, subsystem_size)
    tr_rho2 = jnp.real(jnp.trace(rho_A @ rho_A))
    return -jnp.log(tr_rho2 + eps)


def exact_tfim_entropies(L, J, g, subsystem_size):
    """
    Returns exact ground-state energy, exact von Neumann entropy,
    and exact Renyi-2 entropy for the TFIM ground state.
    """
    E0, psi0 = exact_tfim_ground_state(L, J, g)
    SvN = von_neumann_entropy_from_statevector(psi0, L, subsystem_size)
    S2 = renyi2_entropy_from_statevector(psi0, L, subsystem_size)

    return {
        "energy": E0,
        "SvN": SvN,
        "S2": S2,
    }