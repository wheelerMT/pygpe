import unittest
import cupy as cp
from typing import Tuple
from pygpe.shared.grid import Grid
from pygpe.scalar.wavefunction import Wavefunction


def generate_wavefunction2d(points: Tuple[int, int], grid_spacing: Tuple[float, float]) -> Wavefunction:
    """Generates and returns a Wavefunction2D object specified

    :param points: The number of grid points in the x and y dimension, respectively.
    :param grid_spacing: The spacing of grid points in the x and y dimension, respectively.
    :return: The Wavefunction2D object.
    """
    return Wavefunction(Grid(points, grid_spacing))


class TestWavefunction2D(unittest.TestCase):

    def test_set_wavefunction(self):
        """Tests whether the wavefunction array is set correctly."""

        wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
        array = cp.ones((64, 64))
        wavefunction.set_wavefunction(array)

        self.assertEqual(wavefunction.wavefunction.all(), 1.)

    def test_adding_noise(self):
        """Tests whether adding noise to empty wavefunction correctly
        makes the wavefunction non-zero.
        """
        wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
        wavefunction.add_noise(0, 1e-2)

        self.assertNotEqual(wavefunction.wavefunction.all(), 0.)
