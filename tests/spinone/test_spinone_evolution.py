import unittest
import cupy as cp
from pygpe.shared.grid import Grid
from pygpe.spinone.wavefunction import Wavefunction
import pygpe.spinone.evolution as evo


class TestEvolution2D(unittest.TestCase):
    def test_spin_vectors_polar(self):
        """Tests whether the perpendicular and z-component spin vectors are
        correct for a polar wavefunction.
        """
        wavefunction_polar = Wavefunction(Grid((64, 64), (0.5, 0.5)))
        wavefunction_polar.set_ground_state("polar", params={"n0": 1.0})

        f_perp, fz = evo._calculate_spins(wavefunction_polar)

        self.assertEqual(f_perp.all(), 0.0)
        self.assertEqual(fz.all(), 0.0)

    def test_density(self):
        """Tests to see if density is one given a normalised spinor."""
        wavefunction = Wavefunction(Grid((64, 64), (0.5, 0.5)))
        wavefunction.set_ground_state("polar", params={"n0": 1.0})

        self.assertEqual(evo._calculate_density(wavefunction).all(), 1.0)

    def test_renormalise(self):
        """Tests whether wavefunction correctly gets re-normalised after being
        modified.
        """
        wavefunction_1 = Wavefunction(Grid((64, 64), (0.5, 0.5)))
        wavefunction_1.set_ground_state("polar", params={"n0": 1.0})
        wavefunction_1.add_noise_to_components("outer", 0.0, 1e-2)
        wavefunction_1.fft()

        wavefunction_2 = wavefunction_1
        wavefunction_2.plus_component += cp.random.uniform(size=(64, 64))
        wavefunction_2.zero_component += cp.random.uniform(size=(64, 64))
        wavefunction_2.minus_component += cp.random.uniform(size=(64, 64))
        evo._renormalise_wavefunction(wavefunction_2)

        cp.testing.assert_array_equal(
            wavefunction_2.plus_component, wavefunction_1.plus_component
        )
        cp.testing.assert_array_equal(
            wavefunction_2.zero_component, wavefunction_1.zero_component
        )
        cp.testing.assert_array_equal(
            wavefunction_2.minus_component, wavefunction_1.minus_component
        )

    def test_atom_number(self):
        """Tests to see if atom number of wavefunction is calculated correctly."""
        wavefunction = Wavefunction(Grid((64, 64), (0.5, 0.5)))
        wavefunction.set_ground_state("polar", params={"n0": 1.0})

        self.assertEqual(sum(evo._calculate_atom_num(wavefunction)), 1024)
