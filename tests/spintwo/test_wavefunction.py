import unittest
import cupy as cp
from typing import Tuple
from pygpe.shared.grid import Grid
from pygpe.spintwo.wavefunction import Wavefunction


def generate_wavefunction2d(points: Tuple[int, int], grid_spacing: Tuple[float, float]) -> Wavefunction:
    """Generates and returns a Wavefunction2D object specified

    :param points: The number of grid points in the x and y dimension, respectively.
    :param grid_spacing: The spacing of grid points in the x and y dimension, respectively.
    :return: The Wavefunction2D object.
    """
    return Wavefunction(Grid(points, grid_spacing))


class TestWavefunction2D(unittest.TestCase):
    def test_uniaxial_initial_state(self):
        """Tests whether the polar initial state is set correctly."""
        wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
        params = {"n0": 1}
        wavefunction.set_ground_state("UN", params)

        cp.testing.assert_array_equal(wavefunction.plus2_component, cp.zeros((64, 64), dtype='complex128'))
        cp.testing.assert_array_equal(wavefunction.plus1_component, cp.zeros((64, 64), dtype='complex128'))
        cp.testing.assert_array_equal(wavefunction.zero_component, cp.ones((64, 64), dtype='complex128'))
        cp.testing.assert_array_equal(wavefunction.minus1_component, cp.zeros((64, 64), dtype='complex128'))
        cp.testing.assert_array_equal(wavefunction.minus2_component, cp.zeros((64, 64), dtype='complex128'))
