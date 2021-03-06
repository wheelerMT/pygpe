from pygpe.shared.grid import Grid
import cupy as cp


class Wavefunction:

    def __init__(self, grid: Grid):
        self.grid = grid

        self.plus_component = cp.empty(grid.shape, dtype='complex128')
        self.zero_component = cp.empty(grid.shape, dtype='complex128')
        self.minus_component = cp.empty(grid.shape, dtype='complex128')
        self.fourier_plus_component = cp.empty(grid.shape, dtype='complex128')
        self.fourier_zero_component = cp.empty(grid.shape, dtype='complex128')
        self.fourier_minus_component = cp.empty(grid.shape, dtype='complex128')

        self.atom_num_plus = 0
        self.atom_num_zero = 0
        self.atom_num_minus = 0

    def set_ground_state(self, ground_state: str, params: dict) -> None:
        """Sets the components of the wavefunction according to
        the ground state we wish to be in.

        :param ground_state: The ground state of the wavefunction.
        :param params: Dictionary containing condensate parameters.
        """
        ground_states = {
            "polar": _polar_initial_state,
            "ferromagnetic": _ferromagnetic_initial_state,
            "antiferromagnetic": _antiferromagnetic_initial_state
        }

        ground_states[ground_state.lower()](self, params)

        self._update_atom_numbers()

    def set_custom_components(self, plus_component: cp.ndarray = None, zero_component: cp.ndarray = None,
                              minus_component: cp.ndarray = None) -> None:
        """Sets the wavefunction components to the specified arrays.

        :param plus_component: Plus component of the wavefunction.
        :param zero_component: Zero component of the wavefunction.
        :param minus_component: Minus component of the wavefunction.
        """
        if plus_component is not None:
            self.plus_component = plus_component
        if zero_component is not None:
            self.zero_component = zero_component
        if minus_component is not None:
            self.minus_component = minus_component

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

        self._update_atom_numbers()

    def _generate_complex_normal_dist(self, mean: float, std_dev: float) -> cp.ndarray:
        """Returns a ndarray of complex values containing results from
        a normal distribution.
        """
        return cp.random.normal(mean, std_dev, size=self.grid.shape) + 1j * cp.random.normal(mean, std_dev,
                                                                                             size=self.grid.shape)

    def _update_atom_numbers(self):
        self.atom_num_plus = self.grid.grid_spacing_product * cp.sum(cp.abs(self.plus_component) ** 2)
        self.atom_num_zero = self.grid.grid_spacing_product * cp.sum(cp.abs(self.zero_component) ** 2)
        self.atom_num_minus = self.grid.grid_spacing_product * cp.sum(cp.abs(self.minus_component) ** 2)

    def fft(self):
        """Fourier transforms real-space components and updates Fourier-space components."""
        self.fourier_plus_component = cp.fft.fftn(self.plus_component)
        self.fourier_zero_component = cp.fft.fftn(self.zero_component)
        self.fourier_minus_component = cp.fft.fftn(self.minus_component)

    def ifft(self):
        """Inverse Fourier transforms Fourier-space components and updates real-space components."""
        self.plus_component = cp.fft.ifftn(self.fourier_plus_component)
        self.zero_component = cp.fft.ifftn(self.fourier_zero_component)
        self.minus_component = cp.fft.ifftn(self.fourier_minus_component)


def _polar_initial_state(wfn: Wavefunction, params: dict):
    """Sets wavefunction components to (easy-axis) polar state."""
    wfn.plus_component = cp.zeros(wfn.grid.shape, dtype='complex128')
    wfn.zero_component = cp.sqrt(params["n0"]) * cp.ones(wfn.grid.shape, dtype='complex128')
    wfn.minus_component = cp.zeros(wfn.grid.shape, dtype='complex128')


def _ferromagnetic_initial_state(wfn: Wavefunction, params: dict):
    """Sets wavefunction components to ferromagnetic state."""
    wfn.plus_component = cp.sqrt(params["n0"]) * cp.ones(wfn.grid.shape, dtype='complex128')
    wfn.zero_component = cp.zeros(wfn.grid.shape, dtype='complex128')
    wfn.minus_component = cp.zeros(wfn.grid.shape, dtype='complex128')


def _antiferromagnetic_initial_state(wfn: Wavefunction, params: dict):
    """Sets wavefunction components to antiferromagnetic state."""
    p = params["p"]  # Linear Zeeman
    c2 = params["c2"]  # Spin-dependent interaction strength
    n = params["n0"]

    wfn.plus_component = cp.sqrt(n) * cp.sqrt((1 + p / c2) / 2) * cp.ones(wfn.grid.shape, dtype='complex128')
    wfn.zero_component = cp.zeros(wfn.grid.shape, dtype='complex128')
    wfn.minus_component = cp.sqrt(n) * cp.sqrt((1 + p / c2) / 2) * cp.ones(wfn.grid.shape, dtype='complex128')
