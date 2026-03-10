import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

class SpinChain1D:
    def __init__(self, L):
        self.L = L
        self.dim = 2**L
        self.H = lil_matrix((self.dim, self.dim), dtype=float)

    # ZZ interaction
    def add_zz(self, J):
        for j in range(self.L-1):
            for s in range(self.dim):
                val = self.sigma_z(s,j) * self.sigma_z(s,j+1)
                self.H[s,s] += J * val

    # XX + YY interaction
    def add_xx_yy(self, J):
        for j in range(self.L-1):
            for s in range(self.dim):
                sp = self.sigma_plus(s,j)
                sm = self.sigma_minus(s,j+1)
                if sp is not None and sm is not None:
                    self.H[sp ^ (1 << (j+1)), s] += J

    # Transverse field
    def add_x_field(self, g):
        for j in range(self.L):
            for s in range(self.dim):
                flipped = self.sigma_x_flip(s,j)
                self.H[flipped,s] += g
                
    # Longitudinal field
    def add_z_field(self, h):
        for j in range(self.L):
            for s in range(self.dim):
                self.H[s,s] += h * self.sigma_z(s,j)

    def build(self):
        return csr_matrix(self.H)