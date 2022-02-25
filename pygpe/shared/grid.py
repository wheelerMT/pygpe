import cupy as cp
from abc import ABC, abstractmethod
from typing import Tuple


class Grid(ABC):
    pass


class Grid2D(Grid):
    def __init__(self, points: Tuple[int, int], grid_spacings: Tuple[float, float]):
        self.num_points_x, self.num_points_y = points
        self.grid_spacing_x, self.grid_spacing_y = grid_spacings
        self.length_x = self.num_points_x * self.grid_spacing_x
        self.length_y = self.num_points_y * self.grid_spacing_y
        
        x = cp.arange(-self.num_points_x // 2, self.num_points_x // 2) * self.grid_spacing_x
        y = cp.arange(-self.num_points_y // 2, self.num_points_y // 2) * self.grid_spacing_y
        self.x_mesh, self.y_mesh = cp.meshgrid(x, y)

        # Generate Fourier space variables
        self.fourier_spacing_x = cp.pi / (self.num_points_x // 2 * self.grid_spacing_x)
        self.fourier_spacing_y = cp.pi / (self.num_points_y // 2 * self.grid_spacing_y)

        fourier_x = cp.arange(-self.num_points_x // 2, self.num_points_x // 2) * self.fourier_spacing_x
        fourier_y = cp.arange(-self.num_points_y // 2, self.num_points_y // 2) * self.fourier_spacing_y
        self.fourier_x_mesh, self.fourier_y_mesh = cp.meshgrid(fourier_x, fourier_y)
        self.fourier_x_mesh = cp.fft.fftshift(self.fourier_x_mesh)
        self.fourier_y_mesh = cp.fft.fftshift(self.fourier_y_mesh)
