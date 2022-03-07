import unittest
import cupy as cp
from pygpe.shared.grid import Grid
from pygpe.spin_1.wavefunction import Wavefunction
import pygpe.spin_1.evolution as evo


class TestEvolution2D(unittest.TestCase):
    def test_fft_normalised(self):
        """Tests whether performing a forward followed by a backwards
        fast Fourier transform on the wavefunction retains the same input.
        """
        wavefunction_1 = Wavefunction(Grid((64, 64), (0.5, 0.5)))
        wavefunction_1.set_initial_state("polar")
        wavefunction_2 = Wavefunction(Grid((64, 64), (0.5, 0.5)))
        wavefunction_2.set_initial_state("polar")

        evo._fft(wavefunction_2)
        evo._ifft(wavefunction_2)

        # assert_array_equal returns None if arrays are equal
        self.assertIsNone(cp.testing.assert_array_equal(wavefunction_1.plus_component, wavefunction_2.plus_component))
        self.assertIsNone(cp.testing.assert_array_equal(wavefunction_1.zero_component, wavefunction_2.zero_component))
        self.assertIsNone(cp.testing.assert_array_equal(wavefunction_1.minus_component, wavefunction_2.minus_component))
