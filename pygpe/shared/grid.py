import numpy as np
import cupy as cp
from typing import Tuple


class Grid:
    def __init__(self, points: int | Tuple[int, ...], grid_spacings: float | Tuple[float, ...]):

        self.shape = points
        if isinstance(points, tuple):
            self.ndim = len(points)
            self.total_num_points = sum(points)

        else:
            self.ndim = 1
            self.total_num_points = points

        # TODO: implement error checking to check dimensions of grid_spacings match points

        if self.ndim == 1:
            self._generate_1d_grids(points, grid_spacings)
        elif self.ndim == 2:
            self._generate_2d_grids(points, grid_spacings)
        elif self.ndim == 3:
            self._generate_3d_grids(points, grid_spacings)
        else:
            raise ValueError(f"{points} is not a valid dimensionality")

    def _generate_1d_grids(self, points: int, grid_spacing: float):
        raise NotImplementedError

    def _generate_2d_grids(self, points: Tuple[int, ...], grid_spacings: Tuple[float, ...]):
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

    def _generate_3d_grids(self, points: Tuple[int, ...], grid_spacings: Tuple[float, ...]):
        raise NotImplementedError
