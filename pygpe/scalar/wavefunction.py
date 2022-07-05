from pygpe.shared.grid import Grid
import cupy as cp


class Wavefunction:

    def __init__(self, grid: Grid):
        self.grid = grid

        self.wavefunction = cp.empty(grid.shape, dtype='complex128')

        self.atom_num = 0

    def set_wavefunction(self, wavefunction: cp.ndarray) -> None:
        """Sets the wavefunction to the specified state.

        :param wavefunction:  The array to set the wavefunction as.
        """
        self.wavefunction = wavefunction
       