from pygpe.shared.grid import Grid
import cupy as cp


class Wavefunction:

    def __init__(self, grid: Grid):
        self.grid = grid

        self.wavefunction = cp.empty(grid.shape, dtype='complex128')
        self.fourier_wavefunction = cp.empty(grid.shape, dtype='complex128')  # Fourier component

        self.atom_num = 0

    def set_wavefunction(self, wavefunction: cp.ndarray) -> None:
        """Sets the wavefunction to the specified state.

        :param wavefunction:  The array to set the wavefunction as.
        """
        self.wavefunction = wavefunction

    def add_noise(self, mean: float, std_dev: float) -> None:
        """Adds noise to the wavefunction using a normal distribution.

        :param mean: The mean of the normal distribution.
        :param std_dev: The standard deviation of the normal distribution.
        """
        self.wavefunction += self._generate_complex_normal_dist(mean, std_dev)

    def _generate_complex_normal_dist(self, mean: float, std_dev: float) -> cp.ndarray:
        """Returns a ndarray of complex values containing results from
        a normal distribution.
        """
        return cp.random.normal(mean, std_dev, size=self.grid.shape) + 1j * cp.random.normal(mean, std_dev,
                                                                                             size=self.grid.shape)
