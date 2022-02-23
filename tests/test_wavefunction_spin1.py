import unittest
from pygpe.shared.grid import Grid2D
from pygpe.spin_1.wavefunction import Wavefunction2D


class TestWavefunctionSpin1(unittest.TestCase):

    def test_polar_initial_state(self):
        grid = Grid2D((64, 64), (0.5, 0.5))
        wavefunction = Wavefunction2D(grid)
        wavefunction.set_initial_state("Polar")

        self.assertEqual(wavefunction.plus_component, complex(0., 0.))
        self.assertEqual(wavefunction.zero_component, complex(1., 0.))
        self.assertEqual(wavefunction.minus_component, complex(0., 0.))
