import unittest
import pygpe.shared.grid as grid


class TestGrid(unittest.TestCase):

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
