import unittest
import pygpe.shared.grid as grid


class TestGrid(unittest.TestCase):

    def test_correct_mesh_shape(self):
        """
        Tests to see whether meshgrids match the
        shape of points passed in.
        """

        shape = (64, 64)
        grid2d = grid.Grid2D(shape, (0.5, 0.5))

        self.assertEqual(grid2d.x_mesh.shape, shape)
        self.assertEqual(grid2d.y_mesh.shape, shape)
        self.assertEqual(grid2d.fourier_x_mesh.shape, shape)
        self.assertEqual(grid2d.fourier_y_mesh.shape, shape)

    def test_correct_lengths(self):
        """Tests to see whether the length of each grid
        dimension gives the expected length.
        """

        grid2d = grid.Grid2D((64, 64), (0.5, 0.5))
        self.assertEqual(grid2d.length_x, 32.)
        self.assertEqual(grid2d.length_y, 32.)

    def test_correct_fourier_shift(self):
        """Tests to see whether Fourier space
        meshes are correctly shifted to the center of
        the spectrum.
        """

        grid2d = grid.Grid2D((64, 64), (0.5, 0.5))
        for element in grid2d.fourier_x_mesh[:, 0]:
            self.assertEqual(element, 0)
        for element in grid2d.fourier_y_mesh[0, :]:
            self.assertEqual(element, 0)
