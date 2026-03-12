import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

def construct_basis(L):
    return list(range(2**L))

"""Local spin operators"""
def sigma_z(state, j):
    return 1 if ((state >> j) & 1) == 0 else -1

def sigma_x_flip(state, j):
    return state ^ (1 << j)

def sigma_plus(state, j):
    if ((state >> j) & 1) == 0:
        return state ^ (1 << j)
    return None

def sigma_minus(state, j):
    if ((state >> j) & 1) == 1:
        return state ^ (1 << j)
    return None


def von_neumann_entropy(psi, L, LA):
    """Compute von Neumann entropy of subsystem A of size LA.

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