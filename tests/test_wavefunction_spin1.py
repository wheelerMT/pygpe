import unittest
import cupy as cp
from pygpe.shared.grid import Grid2D
from pygpe.spin_1.wavefunction import Wavefunction2D


class TestWavefunctionSpin1(unittest.TestCase):

    def test_polar_initial_state(self):
        grid = Grid2D((64, 64), (0.5, 0.5))
        wavefunction = Wavefunction2D(grid)
        wavefunction.set_initial_state("Polar")

        self.assertEqual(wavefunction.plus_component.all(), 0.)
        self.assertEqual(wavefunction.zero_component.all(), 1.)
        self.assertEqual(wavefunction.minus_component.all(), 0.)

    def test_set_initial_state_raises_error(self):
        grid = Grid2D((64, 64), (0.5, 0.5))
        wavefunction = Wavefunction2D(grid)
        with self.assertRaises(ValueError):
            wavefunction.set_initial_state("garbage")

    def test_fft_normalised(self):
        grid = Grid2D((64, 64), (0.5, 0.5))
        wavefunction_1 = Wavefunction2D(grid)
        wavefunction_1.set_initial_state("polar")
        wavefunction_2 = Wavefunction2D(grid)
        wavefunction_2.set_initial_state("polar")

        wavefunction_2.fft()
        wavefunction_2.ifft()

        # assert_array_equal returns None if arrays are equal
        self.assertIsNone(cp.testing.assert_array_equal(wavefunction_1.plus_component, wavefunction_2.plus_component))
        self.assertIsNone(cp.testing.assert_array_equal(wavefunction_1.zero_component, wavefunction_2.zero_component))
        self.assertIsNone(cp.testing.assert_array_equal(wavefunction_1.minus_component, wavefunction_2.minus_component))
