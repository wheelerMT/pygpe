try:
    import cupy as cp  # type: ignore
except ImportError:
    import numpy as cp


def _check_valid_tuple(points: tuple, grid_spacings: tuple) -> None:
    if len(points) != len(grid_spacings):
        raise ValueError(f"{points} and {grid_spacings} are not of same length")
    if len(points) > 3:
        raise ValueError(f"{points} is not a valid dimensionality")
    for point in points:
        if not isinstance(point, int):
            raise ValueError(f"{points} contains non-integer values")


class Grid:
    """An object representing the numerical grid.
    It contains information on the number of grid points, the shape, the
    dimensionality, and lengths of the grid.

    :param points: Number of points in each spatial dimension.
    :type points: int or tuple of ints
    :param grid_spacings: Numerical spacing between grid points in each
        spatial dimension.
    :type grid_spacings: float or tuple of floats

    :ivar shape: Shape of the grid.
    :ivar ndim: Dimensionality of the grid.
    :ivar total_num_points: Total number of grid points across all dimensions.

    :ivar num_points_x: Number of points in the x-direction.
    :ivar num_points_y: (2D and 3D only) Number of points in the y-direction.
    :ivar num_points_z: (3D only) Number of points in the z-direction.
    :ivar length_x: Length of the grid in the x-direction.
    :ivar length_y: (2D and 3D only) Length of the grid in the y-direction.
    :ivar length_z: (3D only) Length of the grid in the z-direction.
    :ivar x_mesh: The x meshgrid. The dimensionality matches that of `ndim`.
    :ivar y_mesh: (2D and 3D only) The y meshgrid. The dimensionality matches
        that of `ndim`.
    :ivar z_mesh: (3D only) The z meshgrid. The dimensionality matches that of
        `ndim`.
    :ivar grid_spacing_x: Grid spacing in the x-direction.
    :ivar grid_spacing_y: (2D and 3D only) Grid spacing in the y-direction.
    :ivar grid_spacing_z: (3D only) Grid spacing in the z-direction.
    :ivar grid_spacing_product: The product of the grid spacing for each
        dimension.
    :ivar fourier_x_mesh: The Fourier-space x meshgrid. The dimensionality
        matches that of `ndim`.
    :ivar fourier_y_mesh: (2D and 3D only) The Fourier-space y meshgrid. The
        dimensionality matches that of `ndim`.
    :ivar fourier_z_mesh: (3D only) The Fourier-space z meshgrid. The
        dimensionality matches that of `ndim`.
    :ivar fourier_spacing_x: Fourier grid spacing in the x-direction.
    :ivar fourier_spacing_y: (2D and 3D only) Fourier grid spacing in the
        y-direction.
    :ivar fourier_spacing_z: (3D only) Fourier grid spacing in the z-direction.
    """

    def __init__(
        self,
        points: int | tuple[int, ...],
        grid_spacings: float | tuple[float, ...],
    ):
        """Constructs the grid object."""

        self.shape = points
        if isinstance(points, tuple):
            _check_valid_tuple(points, grid_spacings)

            self.ndim = len(points)
            self.total_num_points = 1
            for point in points:
                self.total_num_points *= point
        elif isinstance(points, int):
            self.ndim = 1
            self.total_num_points = points
        else:
            raise ValueError(
                f"{points} is of unsupported type. Use int or tuple of ints."
            )

        if self.ndim == 1:
            self._generate_1d_grids(points, grid_spacings)
        elif self.ndim == 2:
            self._generate_2d_grids(points, grid_spacings)
        elif self.ndim == 3:
            self._generate_3d_grids(points, grid_spacings)

    def _generate_1d_grids(self, points: int, grid_spacing: float):
        """Generates meshgrid for a 1D grid."""
        self.num_points_x = points
        self.grid_spacing_x = grid_spacing
        self.grid_spacing_product = self.grid_spacing_x

        self.length_x = self.num_points_x * self.grid_spacing_x
        self.x_mesh = (
            cp.arange(-self.num_points_x // 2, self.num_points_x // 2)
            * self.grid_spacing_x
        )

        self.fourier_spacing_x = cp.pi / (self.num_points_x // 2 * self.grid_spacing_x)
        self.fourier_x_mesh = cp.fft.fftshift(
            cp.arange(-self.num_points_x // 2, self.num_points_x // 2)
            * self.fourier_spacing_x
        )

        self.wave_number = self.fourier_x_mesh**2

    def _generate_2d_grids(
        self, points: tuple[int, ...], grid_spacings: tuple[float, ...]
    ):
        """Generates meshgrid for a 2D grid."""
        self.num_points_x, self.num_points_y = points
        self.grid_spacing_x, self.grid_spacing_y = grid_spacings
        self.grid_spacing_product = self.grid_spacing_x * self.grid_spacing_y

        self.length_x = self.num_points_x * self.grid_spacing_x
        self.length_y = self.num_points_y * self.grid_spacing_y

        x = (
            cp.arange(-self.num_points_x // 2, self.num_points_x // 2)
            * self.grid_spacing_x
        )
        y = (
            cp.arange(-self.num_points_y // 2, self.num_points_y // 2)
            * self.grid_spacing_y
        )
        self.x_mesh, self.y_mesh = cp.meshgrid(x, y, indexing="ij")

        # Generate Fourier space variables
        self.fourier_spacing_x = cp.pi / (self.num_points_x // 2 * self.grid_spacing_x)
        self.fourier_spacing_y = cp.pi / (self.num_points_y // 2 * self.grid_spacing_y)

        fourier_x = (
            cp.arange(-self.num_points_x // 2, self.num_points_x // 2)
            * self.fourier_spacing_x
        )
        fourier_y = (
            cp.arange(-self.num_points_y // 2, self.num_points_y // 2)
            * self.fourier_spacing_y
        )

        self.fourier_x_mesh, self.fourier_y_mesh = cp.meshgrid(
            fourier_x, fourier_y, indexing="ij"
        )
        self.fourier_x_mesh = cp.fft.fftshift(self.fourier_x_mesh)
        self.fourier_y_mesh = cp.fft.fftshift(self.fourier_y_mesh)

        self.wave_number = self.fourier_x_mesh**2 + self.fourier_y_mesh**2

    def _generate_3d_grids(
        self, points: tuple[int, ...], grid_spacings: tuple[float, ...]
    ):
        """Generates meshgrid for a 3D grid."""
        self.num_points_x, self.num_points_y, self.num_points_z = points
        (
            self.grid_spacing_x,
            self.grid_spacing_y,
            self.grid_spacing_z,
        ) = grid_spacings
        self.grid_spacing_product = (
            self.grid_spacing_x * self.grid_spacing_y * self.grid_spacing_z
        )

        self.length_x = self.num_points_x * self.grid_spacing_x
        self.length_y = self.num_points_y * self.grid_spacing_y
        self.length_z = self.num_points_z * self.grid_spacing_z

        x = (
            cp.arange(-self.num_points_x // 2, self.num_points_x // 2)
            * self.grid_spacing_x
        )
        y = (
            cp.arange(-self.num_points_y // 2, self.num_points_y // 2)
            * self.grid_spacing_y
        )
        z = (
            cp.arange(-self.num_points_z // 2, self.num_points_z // 2)
            * self.grid_spacing_z
        )
        self.x_mesh, self.y_mesh, self.z_mesh = cp.meshgrid(x, y, z, indexing="ij")

        # Generate Fourier space variables
        self.fourier_spacing_x = cp.pi / (self.num_points_x // 2 * self.grid_spacing_x)
        self.fourier_spacing_y = cp.pi / (self.num_points_y // 2 * self.grid_spacing_y)
        self.fourier_spacing_z = cp.pi / (self.num_points_z // 2 * self.grid_spacing_z)

        fourier_x = (
            cp.arange(-self.num_points_x // 2, self.num_points_x // 2)
            * self.fourier_spacing_x
        )
        fourier_y = (
            cp.arange(-self.num_points_y // 2, self.num_points_y // 2)
            * self.fourier_spacing_y
        )
        fourier_z = (
            cp.arange(-self.num_points_z // 2, self.num_points_z // 2)
            * self.fourier_spacing_z
        )

        (
            self.fourier_x_mesh,
            self.fourier_y_mesh,
            self.fourier_z_mesh,
        ) = cp.meshgrid(fourier_x, fourier_y, fourier_z, indexing="ij")
        self.fourier_x_mesh = cp.fft.fftshift(self.fourier_x_mesh)
        self.fourier_y_mesh = cp.fft.fftshift(self.fourier_y_mesh)
        self.fourier_z_mesh = cp.fft.fftshift(self.fourier_z_mesh)

        self.wave_number = (
            self.fourier_x_mesh**2 + self.fourier_y_mesh**2 + self.fourier_z_mesh**2
        )
