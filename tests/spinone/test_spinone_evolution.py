import numpy as np

import pygpe.spinone.evolution as evo
from pygpe.shared.grid import Grid
from pygpe.spinone.wavefunction import SpinOneWavefunction


def test_spin_vectors_polar():
    """Tests whether the perpendicular and z-component spin vectors are
    correct for a polar wavefunction.
    """
    wavefunction_polar = SpinOneWavefunction(Grid((64, 64), (0.5, 0.5)))
    wavefunction_polar.set_ground_state("polar", params={"n0": 1.0})

    f_perp, f_z = evo._calculate_spins(wavefunction_polar)

    np.testing.assert_array_equal(f_perp, np.zeros(wavefunction_polar.grid.shape))
    np.testing.assert_array_equal(f_z, np.zeros(wavefunction_polar.grid.shape))


def test_density():
    """Tests to see if density is one given a normalised spinor."""
    wavefunction = SpinOneWavefunction(Grid((64, 64), (0.5, 0.5)))
    wavefunction.set_ground_state("polar", params={"n0": 1.0})

    np.testing.assert_array_equal(
        evo._calculate_density(wavefunction), np.ones(wavefunction.grid.shape)
    )


def test_renormalise():
    """Tests whether wavefunction correctly gets re-normalised after being
    modified.
    """
    wavefunction_1 = SpinOneWavefunction(Grid((64, 64), (0.5, 0.5)))
    wavefunction_1.set_ground_state("polar", params={"n0": 1.0})
    wavefunction_1.add_noise("outer", 0.0, 1e-2)
    wavefunction_1.fft()

    wavefunction_2 = wavefunction_1
    wavefunction_2.plus_component += np.random.uniform(size=(64, 64))
    wavefunction_2.zero_component += np.random.uniform(size=(64, 64))
    wavefunction_2.minus_component += np.random.uniform(size=(64, 64))
    evo._renormalise_wavefunction(wavefunction_2)

    np.testing.assert_array_equal(
        wavefunction_2.plus_component, wavefunction_1.plus_component
    )
    np.testing.assert_array_equal(
        wavefunction_2.zero_component, wavefunction_1.zero_component
    )
    np.testing.assert_array_equal(
        wavefunction_2.minus_component, wavefunction_1.minus_component
    )


def test_atom_number():
    """Tests to see if atom number of wavefunction is calculated correctly."""
    wavefunction = SpinOneWavefunction(Grid((64, 64), (0.5, 0.5)))
    wavefunction.set_ground_state("polar", params={"n0": 1.0})

    assert evo._calculate_atom_num(wavefunction) == 1024
