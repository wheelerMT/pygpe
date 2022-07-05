from pygpe.shared.grid import Grid
import cupy as cp


class Wavefunction:

    def __init__(self, grid: Grid):
        self.grid = grid

        self.component = cp.empty(grid.shape, dtype='complex128')

        self.atom_num = 0
