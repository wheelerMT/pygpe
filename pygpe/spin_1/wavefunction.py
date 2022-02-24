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
        if ground_state.lower() == "polar":
            self.plus_component = cp.zeros((self.grid.num_points_x, self.grid.num_points_y), dtype='complex64')
            self.zero_component = cp.ones((self.grid.num_points_x, self.grid.num_points_y), dtype='complex64')
            self.minus_component = cp.zeros((self.grid.num_points_x, self.grid.num_points_y), dtype='complex64')
        else:
            raise ValueError(f"Argument \"{ground_state}\" is not a supported type")

    def add_noise_to_components(self, components: str, mean: float, std_dev: float) -> None:
        if components.lower() == "outer":
            self.plus_component += self._generate_complex_normal_dist(mean, std_dev)
            self.zero_component += self._generate_complex_normal_dist(mean, std_dev)
            self.minus_component += self._generate_complex_normal_dist(mean, std_dev)
        else:
            raise ValueError(f"Argument \"{components}\" is not a supported configuration")

    def _generate_complex_normal_dist(self, mean: float, std_dev: float) -> cp.ndarray:
        return cp.random.normal(mean, std_dev, size=(self.grid.num_points_x, self.grid.num_points_y)) \
               + 1j * cp.random.normal(mean, std_dev, size=(self.grid.num_points_x, self.grid.num_points_y))

    def fft(self) -> None:
        self.fourier_plus_component = cp.fft.fft2(self.plus_component)
        self.fourier_zero_component = cp.fft.fft2(self.zero_component)
        self.fourier_minus_component = cp.fft.fft2(self.minus_component)

    def ifft(self) -> None:
        self.plus_component = cp.fft.ifft2(self.fourier_plus_component)
        self.zero_component = cp.fft.ifft2(self.fourier_zero_component)
        self.minus_component = cp.fft.ifft2(self.fourier_minus_component)
