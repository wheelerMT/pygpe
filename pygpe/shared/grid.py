import numpy as np
import cupy as cp


class Grid:
    """An object representing the numerical grid."""

    def __init__(self, points: int | tuple[int, ...], grid_spacings: float | tuple[float, ...]):
        """Constructs the grid object.

        :param points: The number of points in each spatial dimension.
        :param grid_spacings: The numerical spacing between grid points in each spatial dimension.
        """

        self.shape = points
        if isinstance(points, tuple):
            if len(points) != len(grid_spacings):
                raise ValueError(f"{points} and {grid_spacings} are not of same length")
            if len(points) > 3:
                raise ValueError(f"{points} is not a valid dimensionality")
            self.ndim = len(points)
            self.total_num_points = 1
            for point in points:
                self.total_num_points *= point
        else:
            self.ndim = 1
            self.total_num_points = points

        if self.ndim == 1:
            self._generate_1d_grids(points, grid_spacings)
        elif self.ndim == 2:
            self._generate_2d_grids(points, grid_spacings)
        elif self.ndim == 3:
            self._generate_3d_grids(points, grid_spacings)

    def _generate_1d_grids(self, points: int, grid_spacing: float):
        """Generates meshgrids for a 1D grid."""
        self.num_points_x = points
        self.grid_spacing_x = grid_spacing
        self.grid_spacing_product = self.grid_spacing_x

        self.length_x = self.num_points_x * self.grid_spacing_x
        self.x_mesh = np.arange(-self.num_points_x // 2, self.num_points_x // 2) * self.grid_spacing_x

        self.fourier_spacing_x = np.pi / (self.num_points_x // 2 * self.grid_spacing_x)
        self.fourier_x_mesh = np.fft.fftshift(
            np.arange(-self.num_points_x // 2, self.num_points_x // 2) * self.fourier_spacing_x)

        # Defined on device for use in evolution
        self.wave_number = cp.asarray(self.fourier_x_mesh ** 2)

    def _generate_2d_grids(self, points: tuple[int, ...], grid_spacings: tuple[float, ...]):
        """Generates meshgrids for a 2D grid."""
        self.num_points_x, self.num_points_y = points
        self.grid_spacing_x, self.grid_spacing_y = grid_spacings
        self.grid_spacing_product = self.grid_spacing_x * self.grid_spacing_y

        self.length_x = self.num_points_x * self.grid_spacing_x
        self.length_y = self.num_points_y * self.grid_spacing_y

        x = np.arange(-self.num_points_x // 2, self.num_points_x // 2) * self.grid_spacing_x
        y = np.arange(-self.num_points_y // 2, self.num_points_y // 2) * self.grid_spacing_y
        self.x_mesh, self.y_mesh = np.meshgrid(x, y)

        # Generate Fourier space variables
        self.fourier_spacing_x = np.pi / (self.num_points_x // 2 * self.grid_spacing_x)
        self.fourier_spacing_y = np.pi / (self.num_points_y // 2 * self.grid_spacing_y)

        fourier_x = np.arange(-self.num_points_x // 2, self.num_points_x // 2) * self.fourier_spacing_x
        fourier_y = np.arange(-self.num_points_y // 2, self.num_points_y // 2) * self.fourier_spacing_y

        self.fourier_x_mesh, self.fourier_y_mesh = np.meshgrid(fourier_x, fourier_y)
        self.fourier_x_mesh = np.fft.fftshift(self.fourier_x_mesh)
        self.fourier_y_mesh = np.fft.fftshift(self.fourier_y_mesh)

        # Defined on device for use in evolution
        self.wave_number = cp.asarray(self.fourier_x_mesh ** 2 + self.fourier_y_mesh ** 2)

    def _generate_3d_grids(self, points: tuple[int, ...], grid_spacings: tuple[float, ...]):
        """Generates meshgrids for a 3D grid."""
        self.num_points_x, self.num_points_y, self.num_points_z = points
        self.grid_spacing_x, self.grid_spacing_y, self.grid_spacing_z = grid_spacings
        self.grid_spacing_product = self.grid_spacing_x * self.grid_spacing_y * self.grid_spacing_z

        self.length_x = self.num_points_x * self.grid_spacing_x
        self.length_y = self.num_points_y * self.grid_spacing_y
        self.length_z = self.num_points_z * self.grid_spacing_z

        x = np.arange(-self.num_points_x // 2, self.num_points_x // 2) * self.grid_spacing_x
        y = np.arange(-self.num_points_y // 2, self.num_points_y // 2) * self.grid_spacing_y
        z = np.arange(-self.num_points_z // 2, self.num_points_z // 2) * self.grid_spacing_z
        self.x_mesh, self.y_mesh, self.z_mesh = np.meshgrid(x, y, z)

        # Generate Fourier space variables
        self.fourier_spacing_x = np.pi / (self.num_points_x // 2 * self.grid_spacing_x)
        self.fourier_spacing_y = np.pi / (self.num_points_y // 2 * self.grid_spacing_y)
        self.fourier_spacing_z = np.pi / (self.num_points_z // 2 * self.grid_spacing_z)

        fourier_x = np.arange(-self.num_points_x // 2, self.num_points_x // 2) * self.fourier_spacing_x
        fourier_y = np.arange(-self.num_points_y // 2, self.num_points_y // 2) * self.fourier_spacing_y
        fourier_z = np.arange(-self.num_points_z // 2, self.num_points_z // 2) * self.fourier_spacing_z

        self.fourier_x_mesh, self.fourier_y_mesh, self.fourier_z_mesh = np.meshgrid(fourier_x, fourier_y, fourier_z)
        self.fourier_x_mesh = np.fft.fftshift(self.fourier_x_mesh)
        self.fourier_y_mesh = np.fft.fftshift(self.fourier_y_mesh)
        self.fourier_z_mesh = np.fft.fftshift(self.fourier_z_mesh)

        # Defined on device for use in evolution
        self.wave_number = cp.asarray(self.fourier_x_mesh ** 2 + self.fourier_y_mesh ** 2 + self.fourier_z_mesh ** 2)
