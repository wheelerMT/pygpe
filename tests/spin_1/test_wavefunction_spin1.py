import unittest
from typing import Tuple
from pygpe.shared.grid import Grid
from pygpe.spin_1.wavefunction import Wavefunction


def generate_wavefunction2d(points: Tuple[int, int], grid_spacing: Tuple[float, float]) -> Wavefunction:
    """Generates and returns a Wavefunction2D object specified

    :param points: The number of grid points in the x and y dimension, respectively.
    :param grid_spacing: The spacing of grid points in the x and y dimension, respectively.
    :return: The Wavefunction2D object.
    """
    return Wavefunction(Grid(points, grid_spacing))


class TestWavefunction2D(unittest.TestCase):

    def test_polar_initial_state(self):
        """Tests whether the polar initial state is set correctly."""
        wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
        wavefunction.set_ground_state("Polar")

        self.assertEqual(wavefunction.plus_component.all(), 0.)
        self.assertEqual(wavefunction.zero_component.all(), 1.)
        self.assertEqual(wavefunction.minus_component.all(), 0.)

    def test_ferromagnetic_initial_state(self):
        """Tests whether the polar initial state is set correctly."""
        wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
        wavefunction.set_ground_state("ferromagnetic")

        self.assertEqual(wavefunction.plus_component.all(), 1.)
        self.assertEqual(wavefunction.zero_component.all(), 0.)
        self.assertEqual(wavefunction.minus_component.all(), 0.)

    def test_empty_initial_state(self):
        """Tests whether the empty initial state correctly sets
        all wavefunction components to zero.
        """
        wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
        wavefunction.set_ground_state("empty")

        self.assertEqual(wavefunction.plus_component.all(), 0.)
        self.assertEqual(wavefunction.zero_component.all(), 0.)
        self.assertEqual(wavefunction.minus_component.all(), 0.)

    def test_set_initial_state_raises_error(self):
        """Tests that an unsupported/invalid initial state returns an error."""
        wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
        with self.assertRaises(ValueError):
            wavefunction.set_ground_state("garbage")

    def test_adding_noise_outer(self):
        """Tests whether adding noise to empty outer components correctly
        makes those components non-zero.
        """
        wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
        wavefunction.set_ground_state("empty")
        wavefunction.add_noise_to_components("outer", 0, 1e-2)

        self.assertNotEqual(wavefunction.plus_component.all(), 0.)
        self.assertEqual(wavefunction.zero_component.all(), 0.)
        self.assertNotEqual(wavefunction.minus_component.all(), 0.)
