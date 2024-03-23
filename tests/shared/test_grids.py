import pytest

import pygpe.shared.grid as grid


def test_correct_mesh_shape_1d():
    """
    Tests to see whether the meshgrid matches the
    shape of points passed in for a 1D grid.
    """

    shape = 64
    grid1d = grid.Grid(shape, 0.5)

    assert grid1d.x_mesh.shape[0] == shape
    assert grid1d.fourier_x_mesh.shape[0] == shape


def test_correct_lengths_1d():
    """Tests to see whether the length of each grid
    dimension gives the expected length for a 1D grid.
    """

    grid1d = grid.Grid(64, 0.5)
    assert grid1d.length_x == 32.0


def test_correct_mesh_shape_2d():
    """
    Tests to see whether the meshgrid matches the
    shape of points passed in for a 2D grid.
    """

    shape = (64, 64)
    grid2d = grid.Grid(shape, (0.5, 0.5))

    assert grid2d.x_mesh.shape == shape
    assert grid2d.y_mesh.shape == shape
    assert grid2d.fourier_x_mesh.shape == shape
    assert grid2d.fourier_y_mesh.shape == shape


def test_correct_lengths_2d():
    """Tests to see whether the length of each grid
    dimension gives the expected length for a 2D grid.
    """

    grid2d = grid.Grid((64, 64), (0.5, 0.5))

    assert grid2d.length_x == 32.0
    assert grid2d.length_y == 32.0


def test_correct_fourier_shift_2d():
    """Tests to see whether Fourier space
    meshes are correctly shifted to the center of
    the spectrum for a 2D grid.
    """

    grid2d = grid.Grid((64, 64), (0.5, 0.5))

    for element in grid2d.fourier_x_mesh[0, :]:
        assert element == 0
    for element in grid2d.fourier_y_mesh[:, 0]:
        assert element == 0


def test_correct_mesh_shape_3d():
    """
    Tests to see whether the meshgrid matches the
    shape of points passed in for a 3D grid.
    """

    shape = (32, 32, 32)
    grid3d = grid.Grid(shape, (0.5, 0.5, 0.5))

    assert grid3d.x_mesh.shape == shape
    assert grid3d.y_mesh.shape == shape
    assert grid3d.z_mesh.shape == shape
    assert grid3d.fourier_x_mesh.shape == shape
    assert grid3d.fourier_y_mesh.shape == shape
    assert grid3d.fourier_z_mesh.shape == shape


def test_correct_lengths_3d():
    """Tests to see whether the length of each grid
    dimension gives the expected length for a 3D grid.
    """

    grid3d = grid.Grid((32, 32, 32), (0.5, 0.5, 0.5))
    assert grid3d.length_x == 16.0
    assert grid3d.length_y == 16.0
    assert grid3d.length_z == 16.0


def test_handles_incorrect_dimension():
    with pytest.raises(ValueError):
        grid.Grid((64, 64, 64, 64), (0.5, 0.5, 0.5, 0.5))


def test_handles_incorrect_type_1d():
    with pytest.raises(ValueError):
        grid.Grid(64.0, 0.5)
