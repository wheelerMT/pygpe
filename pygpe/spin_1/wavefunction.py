from pygpe.shared.grid import Grid2D
import cupy as cp


class Wavefunction2D:

    def __init__(self, grid: Grid2D):
        self.grid = grid

        self.plus_component = cp.empty((grid.num_points_x, grid.num_points_y), dtype='complex64')
        self.zero_component = cp.empty((grid.num_points_x, grid.num_points_y), dtype='complex64')
        self.minus_component = cp.empty((grid.num_points_x, grid.num_points_y), dtype='complex64')
        self.fourier_plus_component = cp.empty((grid.num_points_x, grid.num_points_y), dtype='complex64')
        self.fourier_zero_component = cp.empty((grid.num_points_x, grid.num_points_y), dtype='complex64')
        self.fourier_minus_component = cp.empty((grid.num_points_x, grid.num_points_y), dtype='complex64')

    def set_initial_state(self, ground_state: str) -> None:
        """Sets the components of the wavefunction according to
        the ground state we wish to be in.

        :param ground_state: The ground state of the wavefunction.
        """
        if ground_state.lower() == "polar":
            self.plus_component = cp.zeros((self.grid.num_points_x, self.grid.num_points_y), dtype='complex64')
            self.zero_component = cp.ones((self.grid.num_points_x, self.grid.num_points_y), dtype='complex64')
            self.minus_component = cp.zeros((self.grid.num_points_x, self.grid.num_points_y), dtype='complex64')
        elif ground_state.lower() == "empty":
            self.plus_component = cp.zeros((self.grid.num_points_x, self.grid.num_points_y), dtype='complex64')
            self.zero_component = cp.zeros((self.grid.num_points_x, self.grid.num_points_y), dtype='complex64')
            self.minus_component = cp.zeros((self.grid.num_points_x, self.grid.num_points_y), dtype='complex64')
        else:
            raise ValueError(f"Argument \"{ground_state}\" is not a supported ground state")

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
            raise ValueError(f"Argument \"{components}\" is not a supported configuration")

    def _generate_complex_normal_dist(self, mean: float, std_dev: float) -> cp.ndarray:
        """Returns a ndarray of complex values containing results from
        a normal distribution.
        """
        return cp.random.normal(mean, std_dev, size=(self.grid.num_points_x, self.grid.num_points_y)) \
               + 1j * cp.random.normal(mean, std_dev, size=(self.grid.num_points_x, self.grid.num_points_y))

    def fft(self) -> None:
        """Performs a fast Fourier transform on each wavefunction component."""
        self.fourier_plus_component = cp.fft.fft2(self.plus_component)
        self.fourier_zero_component = cp.fft.fft2(self.zero_component)
        self.fourier_minus_component = cp.fft.fft2(self.minus_component)

    def ifft(self) -> None:
        """Performs an inverse Fourier transform on each wavefunction component."""
        self.plus_component = cp.fft.ifft2(self.fourier_plus_component)
        self.zero_component = cp.fft.ifft2(self.fourier_zero_component)
        self.minus_component = cp.fft.ifft2(self.fourier_minus_component)
