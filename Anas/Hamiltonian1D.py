from scipy.sparse import lil_matrix, csr_matrix
from functions import sigma_z, sigma_plus, sigma_minus, sigma_x_flip

class SpinChain1D:
    def __init__(self, L):
        self.L = L
        self.dim = 2**L
        self.H = lil_matrix((self.dim, self.dim), dtype=float)

    # ZZ interaction
    def add_zz(self, J):
        for j in range(self.L-1):
            for s in range(self.dim):
                val = sigma_z(s,j) * sigma_z(s,j+1)
                self.H[s,s] += J * val

    # XX + YY interaction
    def add_xx_yy(self, J):
        for j in range(self.L-1):
            for s in range(self.dim):
                sp = sigma_plus(s,j)
                sm = sigma_minus(s,j+1)
                if sp is not None and sm is not None:
                    self.H[sp ^ (1 << (j+1)), s] += J

    # Transverse field
    def add_x_field(self, g):
        for j in range(self.L):
            for s in range(self.dim):
                flipped = sigma_x_flip(s,j)
                self.H[flipped,s] += g

    # Longitudinal field
    def add_z_field(self, h):
        for j in range(self.L):
            for s in range(self.dim):
                self.H[s,s] += h * sigma_z(s,j)

    def build(self):
        return csr_matrix(self.H)