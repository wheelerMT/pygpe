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
        self._update_atom_number()

    def add_noise(self, mean: float, std_dev: float) -> None:
        """Adds noise to the wavefunction using a normal distribution.

        :param mean: The mean of the normal distribution.
        :param std_dev: The standard deviation of the normal distribution.
        """
        self.wavefunction += self._generate_complex_normal_dist(mean, std_dev)
        self._update_atom_number()

    def _generate_complex_normal_dist(self, mean: float, std_dev: float) -> cp.ndarray:
        """Returns a ndarray of complex values containing results from
        a normal distribution.
        """
        return cp.random.normal(mean, std_dev, size=self.grid.shape) + 1j * cp.random.normal(mean, std_dev,
                                                                                             size=self.grid.shape)

    def _update_atom_number(self) -> None:
        self.atom_num = self.grid.grid_spacing_product * cp.sum(cp.abs(self.wavefunction) ** 2)

    def fft(self) -> None:
        """Fourier transforms real-space component and updates Fourier-space component."""
        self.fourier_wavefunction = cp.fft.fftn(self.wavefunction)

    def ifft(self) -> None:
        """Inverse Fourier transforms Fourier-space component and updates real-space component."""
        self.wavefunction = cp.fft.ifftn(self.fourier_wavefunction)

    def density(self) -> cp.ndarray:
        """Returns the density of the system."""
        return cp.abs(self.wavefunction) ** 2
