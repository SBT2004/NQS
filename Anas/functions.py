import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

def construct_basis(L):
    """
    Construct the computational basis for L spin-1/2 sites.
    Returns a list of integers from 0 to 2^L - 1.
    """
    return list(range(2**L))

def construct_spin_operators(L):
    """
    Construct σ^x_j and σ^z_j as sparse matrices for all sites j = 0,...,L-1.
    """
    dim = 2**L
    sx_list = []
    sz_list = []

    for j in range(L):
        # Initialize σ^x_j and σ^z_j as sparse matrices
        sx = lil_matrix((dim, dim), dtype=float)
        sz = lil_matrix((dim, dim), dtype=float)

        for state in range(dim):
            # Determine if spin j is up (1) or down (0)
            spin_j = (state >> j) & 1

            # --- σ^z_j ---
            # Acts as +1 on |0> (down), -1 on |1> (up)
            sz[state, state] = 1 if spin_j == 0 else -1

            # --- σ^x_j ---
            # Flips the spin at site j
            flipped_state = state ^ (1 << j)
            sx[flipped_state, state] = 1

        sx_list.append(csr_matrix(sx))
        sz_list.append(csr_matrix(sz))

    return sx_list, sz_list

def build_hamiltonian(L, J, g):
    sx_list, sz_list = construct_spin_operators(L)
    dim = 2**L
    H = lil_matrix((dim, dim), dtype=float)

    # Interaction term: σ^z_j σ^z_{j+1}
    for j in range(L-1):
        H -= J * (sz_list[j].dot(sz_list[j + 1]))

    # Transverse field term: σ^x_j
    for j in range(L):
        H -= g * sx_list[j]

    return csr_matrix(H)

def von_neumann_entropy(psi, L, LA):
    """
    Compute von Neumann entropy of subsystem A of size LA.
    
    psi : state vector of size 2^L
    L   : total system size
    LA  : size of subsystem A
    """
    LB = L - LA
    psi_AB = psi.reshape((2**LA, 2**LB))       # reshape to bipartition
    rho_A = psi_AB @ psi_AB.conj().T           # reduced density matrix
    eigvals = np.linalg.eigvalsh(rho_A)        # eigenvalues
    eigvals = eigvals[eigvals > 1e-12]         # remove numerical zeros
    return -np.sum(eigvals * np.log(eigvals))

def renyi2_entropy(psi, L, LA):
    LB = L - LA
    psi_AB = psi.reshape((2**LA, 2**LB))
    rho_A = psi_AB @ psi_AB.conj().T
    eigvals = np.linalg.eigvalsh(rho_A)
    eigvals = eigvals[eigvals > 1e-12]
    return -np.log(np.sum(eigvals**2))