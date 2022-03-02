from pygpe.shared.grid import Grid
import cupy as cp


class Wavefunction:

    def __init__(self, grid: Grid):
        self.grid = grid

        self.plus_component = cp.empty(grid.shape, dtype='complex64')
        self.zero_component = cp.empty(grid.shape, dtype='complex64')
        self.minus_component = cp.empty(grid.shape, dtype='complex64')
        self.fourier_plus_component = cp.empty(grid.shape, dtype='complex64')
        self.fourier_zero_component = cp.empty(grid.shape, dtype='complex64')
        self.fourier_minus_component = cp.empty(grid.shape, dtype='complex64')

    def set_initial_state(self, ground_state: str) -> None:
        """Sets the components of the wavefunction according to
        the ground state we wish to be in.

        :param ground_state: The ground state of the wavefunction.
        """
        if ground_state.lower() == "polar":
            self.plus_component = cp.zeros(self.grid.shape, dtype='complex64')
            self.zero_component = cp.ones(self.grid.shape, dtype='complex64')
            self.minus_component = cp.zeros(self.grid.shape, dtype='complex64')
        elif ground_state.lower() == "empty":
            self.plus_component = cp.zeros(self.grid.shape, dtype='complex64')
            self.zero_component = cp.zeros(self.grid.shape, dtype='complex64')
            self.minus_component = cp.zeros(self.grid.shape, dtype='complex64')
        else:
            raise ValueError(f"{ground_state} is not a supported ground state")

    def add_noise_to_components(self, components: str, mean: float, std_dev: float) -> None:
        """Adds noise to the specified wavefunction components
        using a normal distribution.

        :param components: The components to add noise to.
        :param mean: The mean of the normal distribution.
        :param std_dev: The standard deviation of the normal distribution.
        """
        if components.lower() == "outer":
            self.plus_component += self._generate_complex_normal_dist(mean, std_dev)
            self.minus_component += self._generate_complex_normal_dist(mean, std_dev)
        else:
            raise ValueError(f"{components} is not a supported configuration")

    def _generate_complex_normal_dist(self, mean: float, std_dev: float) -> cp.ndarray:
        """Returns a ndarray of complex values containing results from
        a normal distribution.
        """
        return cp.random.normal(mean, std_dev, size=self.grid.shape) \
               + 1j * cp.random.normal(mean, std_dev, size=self.grid.shape)
