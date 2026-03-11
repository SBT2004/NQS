from scipy.sparse import lil_matrix, csr_matrix

class SpinGraph:
    def __init__(self, N):
        self.N = N              # number of sites
        self.dim = 2**N
        self.H = lil_matrix((self.dim, self.dim), dtype=float)

    # --- local operators (same as before) ---
    def sigma_z(self, state, j):
        return 1 if ((state >> j) & 1) == 0 else -1

    def sigma_x_flip(self, state, j):
        return state ^ (1 << j)

    def sigma_plus(self, state, j):
        if ((state >> j) & 1) == 0:
            return state ^ (1 << j)
        return None

    def sigma_minus(self, state, j):
        if ((state >> j) & 1) == 1:
            return state ^ (1 << j)
        return None

    # --- interaction terms over arbitrary edges ---
    def add_zz(self, edges, J):
        for i,j in edges:
            for s in range(self.dim):
                val = self.sigma_z(s,i) * self.sigma_z(s,j)
                self.H[s,s] += J * val

    def add_xx_yy(self, edges, J):
        for i,j in edges:
            for s in range(self.dim):
                sp = self.sigma_plus(s,i)
                sm = self.sigma_minus(s,j)
                if sp is not None and sm is not None:
                    self.H[sp ^ (1 << j), s] += J

    def add_x_field(self, g):
        for j in range(self.N):
            for s in range(self.dim):
                flipped = self.sigma_x_flip(s,j)
                self.H[flipped,s] += g

    def add_z_field(self, h):
        for j in range(self.N):
            for s in range(self.dim):
                self.H[s,s] += h * self.sigma_z(s,j)

    def build(self):
        return csr_matrix(self.H)