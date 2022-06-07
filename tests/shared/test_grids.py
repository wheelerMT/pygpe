import unittest
import pygpe.shared.grid as grid


class TestGrid(unittest.TestCase):

    def test_correct_mesh_shape_1d(self):
        """
        Tests to see whether meshgrids match the
        shape of points passed in for a 1D grid.
        """

        shape = 64
        grid1d = grid.Grid(shape, 0.5)

        self.assertEqual(grid1d.x_mesh.shape[0], shape)
        self.assertEqual(grid1d.fourier_x_mesh.shape[0], shape)

    def test_correct_lengths_1d(self):
        """Tests to see whether the length of each grid
        dimension gives the expected length for a 1D grid.
        """

        grid1d = grid.Grid(64, 0.5)
        self.assertEqual(grid1d.length_x, 32.)

    def test_correct_mesh_shape_2d(self):
        """
        Tests to see whether meshgrids match the
        shape of points passed in for a 2D grid.
        """

        shape = (64, 64)
        grid2d = grid.Grid(shape, (0.5, 0.5))

        self.assertEqual(grid2d.x_mesh.shape, shape)
        self.assertEqual(grid2d.y_mesh.shape, shape)
        self.assertEqual(grid2d.fourier_x_mesh.shape, shape)
        self.assertEqual(grid2d.fourier_y_mesh.shape, shape)

    def test_correct_lengths_2d(self):
        """Tests to see whether the length of each grid
        dimension gives the expected length for a 2D grid.
        """

        grid2d = grid.Grid((64, 64), (0.5, 0.5))
        self.assertEqual(grid2d.length_x, 32.)
        self.assertEqual(grid2d.length_y, 32.)

    def test_correct_fourier_shift_2d(self):
        """Tests to see whether Fourier space
        meshes are correctly shifted to the center of
        the spectrum for a 2D grid.
        """

        grid2d = grid.Grid((64, 64), (0.5, 0.5))
        for element in grid2d.fourier_x_mesh[:, 0]:
            self.assertEqual(element, 0)
        for element in grid2d.fourier_y_mesh[0, :]:
            self.assertEqual(element, 0)

    def test_correct_mesh_shape_3d(self):
        """
        Tests to see whether meshgrids match the
        shape of points passed in for a 3D grid.
        """

        shape = (32, 32, 32)
        grid3d = grid.Grid(shape, (0.5, 0.5, 0.5))

        self.assertEqual(grid3d.x_mesh.shape, shape)
        self.assertEqual(grid3d.y_mesh.shape, shape)
        self.assertEqual(grid3d.z_mesh.shape, shape)
        self.assertEqual(grid3d.fourier_x_mesh.shape, shape)
        self.assertEqual(grid3d.fourier_y_mesh.shape, shape)
        self.assertEqual(grid3d.fourier_z_mesh.shape, shape)

    def test_correct_lengths_3d(self):
        """Tests to see whether the length of each grid
        dimension gives the expected length for a 3D grid.
        """

        grid3d = grid.Grid((32, 32, 32), (0.5, 0.5, 0.5))
        self.assertEqual(grid3d.length_x, 16.)
        self.assertEqual(grid3d.length_y, 16.)
        self.assertEqual(grid3d.length_z, 16.)
