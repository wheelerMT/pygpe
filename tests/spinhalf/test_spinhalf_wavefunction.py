import pytest
import cupy as cp
from pygpe.shared.grid import Grid
from pygpe.spinhalf.wavefunction import Wavefunction


def generate_wavefunction2d(
    points: tuple[int, int], grid_spacing: tuple[float, float]
) -> Wavefunction:
    """Generates and returns a Wavefunction2D object specified

    :param points: The number of grid points in the x and y dimension, respectively.
    :param grid_spacing: The spacing of grid points in the x and y dimension, respectively.
    :return: The Wavefunction2D object.
    """
    return Wavefunction(Grid(points, grid_spacing))


def test_set_wavefunction_components():
    """Tests whether the wavefunction components are set correctly."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))

    plus_component = cp.ones((64, 64), dtype="complex128")
    minus_component = 0.345 * cp.ones((64, 64), dtype="complex128")
    wavefunction.set_wavefunction_components(plus_component, minus_component)

    cp.testing.assert_array_equal(wavefunction.plus_component, plus_component)
    cp.testing.assert_array_equal(wavefunction.minus_component, minus_component)


def test_update_atom_numbers():
    """Tests whether the atom numbers are updated correctly for a specified
    initial wavefunction.
    """
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))

    plus_component = 0.5 * cp.ones((64, 64), dtype="complex128")
    minus_component = 2 * cp.ones((64, 64), dtype="complex128")
    wavefunction.set_wavefunction_components(plus_component, minus_component)

    assert wavefunction.atom_num_plus == 256.0 * cp.ones(1)
    assert wavefunction.atom_num_minus == 4096.0 * cp.ones(1)


def test_adding_noise_plus():
    """Tests whether adding noise to the plus component correctly
    makes that component non-zero.
    """
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))

    plus_component = cp.zeros((64, 64), dtype="complex128")
    minus_component = cp.zeros((64, 64), dtype="complex128")
    wavefunction.set_wavefunction_components(plus_component, minus_component)
    wavefunction.add_noise_to_components("plus", mean=0, std_dev=1e-2)

    with pytest.raises(AssertionError):
        cp.testing.assert_array_equal(
            wavefunction.plus_component, cp.zeros(wavefunction.grid.shape)
        )
    cp.testing.assert_array_equal(
        wavefunction.minus_component, cp.zeros(wavefunction.grid.shape)
    )


def test_adding_noise_minus():
    """Tests whether adding noise to the minus component correctly
    makes that component non-zero.
    """
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))

    plus_component = cp.zeros((64, 64), dtype="complex128")
    minus_component = cp.zeros((64, 64), dtype="complex128")
    wavefunction.set_wavefunction_components(plus_component, minus_component)
    wavefunction.add_noise_to_components("minus", mean=0, std_dev=1e-2)

    with pytest.raises(AssertionError):
        cp.testing.assert_array_equal(
            wavefunction.minus_component, cp.zeros(wavefunction.grid.shape)
        )
    cp.testing.assert_array_equal(
        wavefunction.plus_component, cp.zeros(wavefunction.grid.shape)
    )


def test_adding_noise_all():
    """Tests whether adding noise to both components correctly
    makes those components non-zero.
    """
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))

    plus_component = cp.zeros((64, 64), dtype="complex128")
    minus_component = cp.zeros((64, 64), dtype="complex128")
    wavefunction.set_wavefunction_components(plus_component, minus_component)
    wavefunction.add_noise_to_components("all", mean=0, std_dev=1e-2)

    with pytest.raises(AssertionError):
        cp.testing.assert_array_equal(
            wavefunction.plus_component, cp.zeros(wavefunction.grid.shape)
        )
    with pytest.raises(AssertionError):
        cp.testing.assert_array_equal(
            wavefunction.minus_component, cp.zeros(wavefunction.grid.shape)
        )


def test_phase_all():
    """Tests that a phase applied to all components is applied correctly."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))

    plus_component = cp.ones((64, 64), dtype="complex128")
    minus_component = cp.ones((64, 64), dtype="complex128")
    wavefunction.set_wavefunction_components(plus_component, minus_component)

    phase = cp.random.uniform(size=(64, 64), dtype=cp.float64)
    wavefunction.apply_phase(phase, components="all")

    cp.testing.assert_allclose(cp.angle(wavefunction.plus_component), phase)
    cp.testing.assert_allclose(cp.angle(wavefunction.minus_component), phase)


def test_phase_plus():
    """Tests that a phase applied to the plus component is applied correctly."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))

    plus_component = cp.ones((64, 64), dtype="complex128")
    minus_component = cp.ones((64, 64), dtype="complex128")
    wavefunction.set_wavefunction_components(plus_component, minus_component)

    phase = cp.random.uniform(size=(64, 64), dtype=cp.float64)
    wavefunction.apply_phase(phase, components="plus")

    cp.testing.assert_allclose(cp.angle(wavefunction.plus_component), phase)
    cp.testing.assert_allclose(cp.angle(wavefunction.minus_component), cp.zeros(wavefunction.grid.shape))


def test_phase_minus():
    """Tests that a phase applied to the plus component is applied correctly."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))

    plus_component = cp.ones((64, 64), dtype="complex128")
    minus_component = cp.ones((64, 64), dtype="complex128")
    wavefunction.set_wavefunction_components(plus_component, minus_component)

    phase = cp.random.uniform(size=(64, 64), dtype=cp.float64)
    wavefunction.apply_phase(phase, components="minus")

    cp.testing.assert_allclose(cp.angle(wavefunction.minus_component), phase)
    cp.testing.assert_allclose(cp.angle(wavefunction.plus_component), cp.zeros(wavefunction.grid.shape))


def test_phase_handles_unknown():
    """Tests whether the apply phase function correctly handles an unknown component."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    phase = cp.random.uniform(size=(64, 64), dtype=cp.float64)

    with pytest.raises(ValueError):
        wavefunction.apply_phase(phase, components="garbage")


def test_density():
    """Tests whether the density function returns the correct density."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))

    plus_component = 0.5 * cp.ones((64, 64), dtype="complex128")
    minus_component = 2 * cp.ones((64, 64), dtype="complex128")
    wavefunction.set_wavefunction_components(plus_component, minus_component)

    plus_density = wavefunction.density("plus")
    minus_density = wavefunction.density("minus")
    cp.testing.assert_allclose(plus_density, 0.25 * cp.ones(wavefunction.grid.shape))
    cp.testing.assert_allclose(minus_density, 4 * cp.ones(wavefunction.grid.shape))

    plus_density, minus_density = wavefunction.density("all")
    cp.testing.assert_allclose(plus_density, 0.25 * cp.ones(wavefunction.grid.shape))
    cp.testing.assert_allclose(minus_density, 4 * cp.ones(wavefunction.grid.shape))


def test_density_handles_unknown():
    """Tests whether the density function correctly handles an unknown component."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))

    with pytest.raises(ValueError):
        wavefunction.density("garbage")
