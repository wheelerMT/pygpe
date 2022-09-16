import unittest
import cupy as cp
from typing import Tuple
from pygpe.shared.grid import Grid
from pygpe.spinone.wavefunction import Wavefunction


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
        params = {"n0": 1}
        wavefunction.set_ground_state("Polar", params)

        self.assertEqual(wavefunction.plus_component.all(), 0.)
        self.assertEqual(wavefunction.zero_component.all(), 1.)
        self.assertEqual(wavefunction.minus_component.all(), 0.)

    def test_ferromagnetic_initial_state(self):
        """Tests whether the ferromagnetic initial state is set correctly."""
        wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
        params = {"n0": 1}
        wavefunction.set_ground_state("ferromagnetic", params)

        self.assertEqual(wavefunction.plus_component.all(), 1.)
        self.assertEqual(wavefunction.zero_component.all(), 0.)
        self.assertEqual(wavefunction.minus_component.all(), 0.)

    def test_antiferromagnetic_initial_state(self):
        """Tests whether the antiferromagnetic initial state is set correctly."""
        wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
        params = {"n0": 1,
                  "p": 0.5,
                  "c2": 0.5}
        wavefunction.set_ground_state("antiferromagnetic", params)

        self.assertEqual(wavefunction.plus_component.all(), 1.)
        self.assertEqual(wavefunction.zero_component.all(), 0.)
        self.assertEqual(wavefunction.minus_component.all(), 0.)

    def test_broken_axisymmetry_initial_state(self):
        """Tests whether the broken-axisymmetry initial state is set correctly."""
        wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
        params = {"n0": 1,
                  "p": 0.,
                  "q": 0.5,
                  "c2": -0.5}
        wavefunction.set_ground_state("BA", params)

        cp.testing.assert_allclose(wavefunction.plus_component, cp.sqrt(2) / 4)
        cp.testing.assert_allclose(wavefunction.zero_component, cp.sqrt(3) / 2)
        cp.testing.assert_allclose(wavefunction.minus_component, cp.sqrt(2) / 4)

    def test_custom_wavefunction_components(self):
        """Tests whether a custom wavefunction is set correctly."""
        wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
        plus_component = 1.67 * cp.ones((64, 64))
        zero_component = 1e4 * cp.ones((64, 64))
        minus_component = cp.zeros((64, 64))

        wavefunction.set_custom_components(plus_component, zero_component, minus_component)

        cp.testing.assert_array_equal(wavefunction.plus_component, plus_component)
        cp.testing.assert_array_equal(wavefunction.zero_component, zero_component)
        cp.testing.assert_array_equal(wavefunction.minus_component, minus_component)
        
    def test_set_initial_state_raises_error(self):
        """Tests that an unsupported/invalid initial state returns an error."""
        wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
        with self.assertRaises(KeyError):
            wavefunction.set_ground_state("garbage", params={})

    def test_adding_noise_outer(self):
        """Tests whether adding noise to empty outer components correctly
        makes those components non-zero.
        """
        wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
        wavefunction.add_noise_to_components("outer", 0, 1e-2)

        self.assertNotEqual(wavefunction.plus_component.all(), 0.)
        self.assertNotEqual(wavefunction.minus_component.all(), 0.)

    def test_adding_noise_all(self):
        """Tests whether adding noise to all components correctly
        makes them non-zero.
        """
        wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
        wavefunction.add_noise_to_components("all", 0, 1e-2)

        self.assertNotEqual(wavefunction.plus_component.all(), 0.)
        self.assertNotEqual(wavefunction.zero_component.all(), 0.)
        self.assertNotEqual(wavefunction.minus_component.all(), 0.)

    def test_phase_all(self):
        """Tests that a phase applied to all components is applied correctly."""
        wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
        wavefunction.set_custom_components(cp.ones((64, 64), dtype='complex128'), cp.ones((64, 64), dtype='complex128'),
                                           cp.ones((64, 64), dtype='complex128'))

        phase = cp.random.uniform(size=(64, 64), dtype=cp.float64)
        wavefunction.apply_phase(phase, 'all')

        cp.testing.assert_allclose(cp.angle(wavefunction.plus_component), phase)
        cp.testing.assert_allclose(cp.angle(wavefunction.zero_component), phase)
        cp.testing.assert_allclose(cp.angle(wavefunction.minus_component), phase)

    def test_phase_multiple_components(self):
        """Tests that a phase is applied correctly to multiple components."""
        wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
        wavefunction.set_custom_components(cp.ones((64, 64), dtype='complex128'), cp.ones((64, 64), dtype='complex128'),
                                           cp.ones((64, 64), dtype='complex128'))

        phase = cp.random.uniform(size=(64, 64), dtype=cp.float64)
        wavefunction.apply_phase(phase, ['plus', 'minus'])

        cp.testing.assert_allclose(cp.angle(wavefunction.plus_component), phase)
        cp.testing.assert_allclose(cp.angle(wavefunction.minus_component), phase)

    def test_phase_single(self):
        """Tests that a phase is applied correctly to a single component."""
        wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
        wavefunction.set_custom_components(cp.ones((64, 64), dtype='complex128'), cp.ones((64, 64), dtype='complex128'),
                                           cp.ones((64, 64), dtype='complex128'))

        phase = cp.random.uniform(size=(64, 64), dtype=cp.float64)
        wavefunction.apply_phase(phase, 'zero')

        cp.testing.assert_allclose(cp.angle(wavefunction.zero_component), phase)
       