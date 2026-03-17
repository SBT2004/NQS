from scipy.sparse import lil_matrix, csr_matrix
from functions import sigma_z, sigma_plus, sigma_minus, sigma_x_flip

# General graph Hamiltonian

class SpinGraph:
    def __init__(self, N):
        self.N = N
        self.dim = 2**N
        self.H = lil_matrix((self.dim, self.dim), dtype=float)

    # ZZ interaction
    def add_zz(self, edges, J):
        for i, j in edges:
            for s in range(self.dim):
                val = sigma_z(s, i) * sigma_z(s, j)
                self.H[s, s] += J * val

    # XX + YY interaction
    def add_xx_yy(self, edges, J):
        for i, j in edges:
            for s in range(self.dim):
                sp = sigma_plus(s, i)
                sm = sigma_minus(s, j)

                if sp is not None and sm is not None:
                    new_state = sp ^ (1 << j)
                    self.H[new_state, s] += J

                sm = sigma_minus(s, i)
                sp = sigma_plus(s, j)

                if sm is not None and sp is not None:
                    new_state = sm ^ (1 << j)
                    self.H[new_state, s] += J

    # Transverse field
    def add_x_field(self, g):
        for j in range(self.N):
            for s in range(self.dim):
                flipped = sigma_x_flip(s, j)
                self.H[flipped, s] += g

    # Longitudinal field
    def add_z_field(self, h):
        for j in range(self.N):
            for s in range(self.dim):
                self.H[s, s] += h * sigma_z(s, j)

    def build(self):
        return csr_matrix(self.H)
