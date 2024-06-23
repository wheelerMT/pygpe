from pygpe.shared.grid import Grid
from pygpe.shared.wavefunction import _Wavefunction

try:
    import cupy as cp  # type: ignore
except ImportError:
    import numpy as cp


class ScalarWavefunction(_Wavefunction):
    """Represents the scalar BEC wavefunction.
    This class contains the wavefunction array, in addition to various useful
    functions for manipulating and using the wavefunction.

    :param grid: The numerical grid.
    :type grid: :class:`Grid`

    :ivar component: The real-space wavefunction array.
    :ivar fourier_component: The Fourier-space wavefunction array.
    :ivar atom_num: The atom number of the condensate.
    :ivar grid: Reference to the grid object of the simulation.
    """

    def __init__(self, grid: Grid):
        """Constructs the wavefunction object."""
        super().__init__(grid)

        self.component = cp.zeros(grid.shape, dtype="complex128")
        self.fourier_component = cp.zeros(
            grid.shape, dtype="complex128"
        )  # Fourier component

        self.atom_num = 0

    def set_wavefunction(self, wavefunction: cp.ndarray) -> None:
        """Sets the wavefunction to the specified state.

        :param wavefunction:  The array to set the wavefunction as.
        :type wavefunction: `cupy.ndarray`
        """
        self.component = wavefunction
        self._update_atom_number()

    def add_noise(self, mean: float, std_dev: float) -> None:
        """Adds noise to the wavefunction using a normal distribution.

        :param mean: The mean of the normal distribution.
        :type mean: float
        :param std_dev: The standard deviation of the normal distribution.
        :type std_dev: float
        """
        self.component += super()._generate_complex_normal_dist(mean, std_dev)
        self._update_atom_number()

    def apply_phase(self, phase: cp.ndarray, **kwargs) -> None:
        """Applies a phase to the wavefunction.

        :param phase: The phase to apply.
        :type phase: `cupy.ndarray`
        """
        self.component *= cp.exp(1j * phase)

    def _update_atom_number(self) -> None:
        self.atom_num = self.grid.grid_spacing_product * cp.sum(
            cp.abs(self.component) ** 2
        )

    def fft(self) -> None:
        """Fourier transforms real-space component and updates Fourier-space
        component.
        """
        self.fourier_component = cp.fft.fftn(self.component)

    def ifft(self) -> None:
        """Inverse Fourier transforms Fourier-space component and updates
        real-space component.
        """
        self.component = cp.fft.ifftn(self.fourier_component)

    def density(self) -> cp.ndarray:
        """

        :return: An array of the condensate density.
        :rtype: `ndarray`
        """
        return cp.abs(self.component) ** 2
