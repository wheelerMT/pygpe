from pygpe.shared.grid import Grid
from pygpe.shared.wavefunction import _Wavefunction

try:
    import cupy as cp  # type: ignore
except ImportError:
    import numpy as cp


class SpinHalfWavefunction(_Wavefunction):
    def __init__(self, grid: Grid):
        """Constructs the wavefunction object."""
        super().__init__(grid)

        self.plus_component = cp.zeros(grid.shape, dtype="complex128")
        self.minus_component = cp.zeros(grid.shape, dtype="complex128")
        self.fourier_plus_component = cp.zeros(grid.shape, dtype="complex128")
        self.fourier_minus_component = cp.zeros(grid.shape, dtype="complex128")

        self.atom_num_plus = 0
        self.atom_num_minus = 0

    def set_wavefunction(
        self, plus_component: cp.ndarray, minus_component: cp.ndarray
    ) -> None:
        """Set the wavefunction components to the specified arrays.

        :param plus_component: Plus component array.
        :param minus_component: Minus component array.
        """
        self.plus_component = plus_component
        self.minus_component = minus_component

        self.fft()
        self._update_atom_numbers()

    def _update_atom_numbers(self) -> None:
        """Updates atom number variables after change in wavefunction."""
        self.atom_num_plus = self.grid.grid_spacing_product * cp.sum(
            cp.abs(self.plus_component) ** 2
        )
        self.atom_num_minus = self.grid.grid_spacing_product * cp.sum(
            cp.abs(self.minus_component) ** 2
        )

    def add_noise(self, components: str, mean: float, std_dev: float) -> None:
        """Adds noise to the specified wavefunction components
        using a normal distribution.

        :param components: "plus", "minus", or "all": The components to add
            noise to.
        :param mean: The mean of the normal distribution.
        :param std_dev: The standard deviation of the normal distribution.
        """
        match components.lower():
            case "plus":
                self.plus_component += super()._generate_complex_normal_dist(
                    mean, std_dev
                )
            case "minus":
                self.minus_component += super()._generate_complex_normal_dist(
                    mean, std_dev
                )
            case "all":
                self.plus_component += super()._generate_complex_normal_dist(
                    mean, std_dev
                )
                self.minus_component += super()._generate_complex_normal_dist(
                    mean, std_dev
                )
            case _:
                raise ValueError(f"{components} is not a supported configuration")

    def apply_phase(self, phase: cp.ndarray, components: str = "all") -> None:
        """Applies a given phase to the specified condensate components.

        :param phase: Array containing the required phase.
        :param components: "plus", "minus", or "all". String specifying which
            component(s) to apply the phase to.
        """
        match components:
            case "plus":
                self.plus_component *= cp.exp(1j * phase)
            case "minus":
                self.minus_component *= cp.exp(1j * phase)
            case "all":
                self.plus_component *= cp.exp(1j * phase)
                self.minus_component *= cp.exp(1j * phase)
            case _:
                raise ValueError(f"Components type {components} is unsupported")

    def fft(self) -> None:
        """Fourier transforms real-space components and updates Fourier-space
        components.
        """
        self.fourier_plus_component = cp.fft.fftn(self.plus_component)
        self.fourier_minus_component = cp.fft.fftn(self.minus_component)

    def ifft(self) -> None:
        """Inverse Fourier transforms Fourier-space components and updates
        real-space components."""
        self.plus_component = cp.fft.ifftn(self.fourier_plus_component)
        self.minus_component = cp.fft.ifftn(self.fourier_minus_component)

    def density(self, components: str) -> cp.ndarray | tuple[cp.ndarray, cp.ndarray]:
        """Calculates the density of the specified component(s).

        :param components: "plus", "minus", or "all". String specifying which
            component(s) to calculate the density of.
        :return: Respective density array. If "all" is specified as the
            component, then both the plus and minus component densities are
            returned as a tuple, respectively.
        """
        match components.lower():
            case "plus":
                return cp.abs(self.plus_component) ** 2
            case "minus":
                return cp.abs(self.minus_component) ** 2
            case "all":
                return (
                    cp.abs(self.plus_component) ** 2,
                    cp.abs(self.minus_component) ** 2,
                )
            case _:
                raise ValueError(f"Components type {components} is unsupported")
