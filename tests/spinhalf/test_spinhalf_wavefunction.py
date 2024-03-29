import numpy as np
import pytest

from pygpe.shared.grid import Grid
from pygpe.spinhalf.wavefunction import SpinHalfWavefunction


def generate_wavefunction2d(
    points: tuple[int, int], grid_spacing: tuple[float, float]
) -> SpinHalfWavefunction:
    """Generates and returns a Wavefunction2D object specified

    :param points: The number of grid points in the x and y dimension,
    respectively.
    :param grid_spacing: The spacing of grid points in the x and y dimension,
    respectively.
    :return: The Wavefunction2D object.
    """
    return SpinHalfWavefunction(Grid(points, grid_spacing))


def test_set_wavefunction():
    """Tests whether the wavefunction components are set correctly."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))

    plus_component = np.ones((64, 64), dtype="complex128")
    minus_component = 0.345 * np.ones((64, 64), dtype="complex128")
    wavefunction.set_wavefunction(plus_component, minus_component)

    np.testing.assert_array_equal(wavefunction.plus_component, plus_component)
    np.testing.assert_array_equal(wavefunction.minus_component, minus_component)


def test_update_atom_numbers():
    """Tests whether the atom numbers are updated correctly for a specified
    initial wavefunction.
    """
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))

    plus_component = 0.5 * np.ones((64, 64), dtype="complex128")
    minus_component = 2 * np.ones((64, 64), dtype="complex128")
    wavefunction.set_wavefunction(plus_component, minus_component)

    assert wavefunction.atom_num_plus == 256.0 * np.ones(1)
    assert wavefunction.atom_num_minus == 4096.0 * np.ones(1)


def test_adding_noise_plus():
    """Tests whether adding noise to the plus component correctly
    makes that component non-zero.
    """
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))

    plus_component = np.zeros((64, 64), dtype="complex128")
    minus_component = np.zeros((64, 64), dtype="complex128")
    wavefunction.set_wavefunction(plus_component, minus_component)
    wavefunction.add_noise("plus", mean=0, std_dev=1e-2)

    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(
            wavefunction.plus_component, np.zeros(wavefunction.grid.shape)
        )
    np.testing.assert_array_equal(
        wavefunction.minus_component, np.zeros(wavefunction.grid.shape)
    )


def test_adding_noise_minus():
    """Tests whether adding noise to the minus component correctly
    makes that component non-zero.
    """
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))

    plus_component = np.zeros((64, 64), dtype="complex128")
    minus_component = np.zeros((64, 64), dtype="complex128")
    wavefunction.set_wavefunction(plus_component, minus_component)
    wavefunction.add_noise("minus", mean=0, std_dev=1e-2)

    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(
            wavefunction.minus_component, np.zeros(wavefunction.grid.shape)
        )
    np.testing.assert_array_equal(
        wavefunction.plus_component, np.zeros(wavefunction.grid.shape)
    )


def test_adding_noise_all():
    """Tests whether adding noise to both components correctly
    makes those components non-zero.
    """
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))

    plus_component = np.zeros((64, 64), dtype="complex128")
    minus_component = np.zeros((64, 64), dtype="complex128")
    wavefunction.set_wavefunction(plus_component, minus_component)
    wavefunction.add_noise("all", mean=0, std_dev=1e-2)

    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(
            wavefunction.plus_component, np.zeros(wavefunction.grid.shape)
        )
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(
            wavefunction.minus_component, np.zeros(wavefunction.grid.shape)
        )


def test_phase_all():
    """Tests that a phase applied to all components is applied correctly."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))

    plus_component = np.ones((64, 64), dtype="complex128")
    minus_component = np.ones((64, 64), dtype="complex128")
    wavefunction.set_wavefunction(plus_component, minus_component)

    phase = np.random.uniform(size=(64, 64))
    wavefunction.apply_phase(phase, components="all")

    np.testing.assert_allclose(np.angle(wavefunction.plus_component), phase)
    np.testing.assert_allclose(np.angle(wavefunction.minus_component), phase)


def test_phase_plus():
    """Tests that a phase applied to the plus component is applied
    correctly.
    """
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))

    plus_component = np.ones((64, 64), dtype="complex128")
    minus_component = np.ones((64, 64), dtype="complex128")
    wavefunction.set_wavefunction(plus_component, minus_component)

    phase = np.random.uniform(size=(64, 64))
    wavefunction.apply_phase(phase, components="plus")

    np.testing.assert_allclose(np.angle(wavefunction.plus_component), phase)
    np.testing.assert_allclose(
        np.angle(wavefunction.minus_component),
        np.zeros(wavefunction.grid.shape),
    )


def test_phase_minus():
    """Tests that a phase applied to the plus component is applied
    correctly.
    """
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))

    plus_component = np.ones((64, 64), dtype="complex128")
    minus_component = np.ones((64, 64), dtype="complex128")
    wavefunction.set_wavefunction(plus_component, minus_component)

    phase = np.random.uniform(size=(64, 64))
    wavefunction.apply_phase(phase, components="minus")

    np.testing.assert_allclose(np.angle(wavefunction.minus_component), phase)
    np.testing.assert_allclose(
        np.angle(wavefunction.plus_component),
        np.zeros(wavefunction.grid.shape),
    )


def test_phase_handles_unknown():
    """Tests whether the apply phase function correctly handles an unknown
    component.
    """
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    phase = np.random.uniform(size=(64, 64))

    with pytest.raises(ValueError):
        wavefunction.apply_phase(phase, components="garbage")


def test_density():
    """Tests whether the density function returns the correct density."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))

    plus_component = 0.5 * np.ones((64, 64), dtype="complex128")
    minus_component = 2 * np.ones((64, 64), dtype="complex128")
    wavefunction.set_wavefunction(plus_component, minus_component)

    plus_density = wavefunction.density("plus")
    minus_density = wavefunction.density("minus")
    np.testing.assert_allclose(plus_density, 0.25 * np.ones(wavefunction.grid.shape))
    np.testing.assert_allclose(minus_density, 4 * np.ones(wavefunction.grid.shape))

    plus_density, minus_density = wavefunction.density("all")
    np.testing.assert_allclose(plus_density, 0.25 * np.ones(wavefunction.grid.shape))
    np.testing.assert_allclose(minus_density, 4 * np.ones(wavefunction.grid.shape))


def test_density_handles_unknown():
    """Tests whether the density function correctly handles an unknown
    component.
    """
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))

    with pytest.raises(ValueError):
        wavefunction.density("garbage")
