import unittest
import cupy as cp
from pygpe.shared.grid import Grid
from pygpe.spin_1.wavefunction import Wavefunction
import pygpe.spin_1.evolution as evo


class TestEvolution2D(unittest.TestCase):
    def test_spin_vectors_polar(self):
        """Tests whether the perpendicular and z-component spin vectors are
        correct for a polar wavefunction.
        """
        wavefunction_polar = Wavefunction(Grid((64, 64), (0.5, 0.5)))
        wavefunction_polar.set_initial_state("polar")

        f_perp, fz = evo._calculate_spins(wavefunction_polar)

        self.assertEqual(f_perp.all(), 0.)
        self.assertEqual(fz.all(), 0.)

    def test_density(self):
        """Tests to see if density is one given a normalised spinor."""
        wavefunction = Wavefunction(Grid((64, 64), (0.5, 0.5)))
        wavefunction.set_initial_state("polar")

        self.assertEqual(evo._calculate_density(wavefunction).all(), 1.)

    def test_renormalise(self):
        """Tests whether wavefunction correctly gets re-normalised after being
        modified.
        """
