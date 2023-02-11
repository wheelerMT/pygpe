from abc import ABC, abstractmethod
from grid import Grid
import cupy as cp


class Wavefunction(ABC):
    def __init__(self, grid: Grid) -> None:
        """The default constructor for the abstract `Wavefunction` class, to be
        inherited by subclasses of `Wavefunction`.

        :param grid: _description_
        :type grid: Grid
        """
        self.grid = grid

    @abstractmethod
    def set_wavefunction(self) -> None:
        """Sets the components of the wavefunction to the specified
        array(s).
        """
        pass

    @abstractmethod
    def add_noise(self, *args, mean: float, std_dev: float) -> None:
        """Adds noise to the specified component(s), drawn from a normal
        distribution.

        :param mean: Mean of the normal distribution.
        :type mean: float
        :param std_dev: Standard deviation of the normal distribution.
        :type std_dev: float
        """
        pass

    def _generate_complex_normal_dist(
        self, mean: float, std_dev: float
    ) -> cp.ndarray:
        """Returns a `cp.ndarray` of complex values containing results from
        a normal distribution.
        """
        return cp.random.normal(
            mean, std_dev, size=self.grid.shape
        ) + 1j * cp.random.normal(mean, std_dev, size=self.grid.shape)

    @abstractmethod
    def apply_phase(self, phase: cp.ndarray, **kwargs) -> None:
        """Applies a phase to the specified component(s).

        :param phase: Array of the condensate phase.
        :type phase: cp.ndarray
        """
        pass

    @abstractmethod
    def fft(self) -> None:
        """Computes the forward Fourier transform on all wavefunction
        components.
        """
        pass

    @abstractmethod
    def ifft(self) -> None:
        """Computes the backward Fourier transform on all k-space wavefunction
        components.
        """
        pass

    @abstractmethod
    def density(self) -> cp.ndarray:
        """Computes the total density of the condensate.

        :return: Total density of the condensate.
        :rtype: cp.ndarray
        """
        pass
