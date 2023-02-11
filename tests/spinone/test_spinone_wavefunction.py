import pytest
import cupy as cp
from typing import Tuple
from pygpe.shared.grid import Grid
from pygpe.spinone.wavefunction import SpinOneWavefunction


def generate_wavefunction2d(
    points: Tuple[int, int], grid_spacing: Tuple[float, float]
) -> SpinOneWavefunction:
    """Generates and returns a Wavefunction2D object specified

    :param points: The number of grid points in the x and y dimension,
        respectively.
    :param grid_spacing: The spacing of grid points in the x and y dimension,
        respectively.
    :return: The Wavefunction2D object.
    """
    return SpinOneWavefunction(Grid(points, grid_spacing))


def test_polar_initial_state():
    """Tests whether the polar initial state is set correctly."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    params = {"n0": 1}
    wavefunction.set_ground_state("polar", params)

    cp.testing.assert_array_equal(
        wavefunction.plus_component, cp.zeros(wavefunction.grid.shape)
    )
    cp.testing.assert_array_equal(
        wavefunction.zero_component, cp.ones(wavefunction.grid.shape)
    )
    cp.testing.assert_array_equal(
        wavefunction.minus_component, cp.zeros(wavefunction.grid.shape)
    )


def test_ferromagnetic_initial_state():
    """Tests whether the ferromagnetic initial state is set correctly."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    params = {"n0": 1}
    wavefunction.set_ground_state("ferromagnetic", params)

    cp.testing.assert_array_equal(
        wavefunction.plus_component, cp.ones(wavefunction.grid.shape)
    )
    cp.testing.assert_array_equal(
        wavefunction.zero_component, cp.zeros(wavefunction.grid.shape)
    )
    cp.testing.assert_array_equal(
        wavefunction.minus_component, cp.zeros(wavefunction.grid.shape)
    )


def test_antiferromagnetic_initial_state():
    """Tests whether the antiferromagnetic initial state is set correctly."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    params = {"n0": 1, "p": 0.25, "c2": 0.5}
    wavefunction.set_ground_state("antiferromagnetic", params)

    cp.testing.assert_allclose(
        wavefunction.plus_component,
        cp.sqrt(0.75) * cp.ones(wavefunction.grid.shape),
    )
    cp.testing.assert_allclose(
        wavefunction.zero_component, cp.zeros(wavefunction.grid.shape)
    )
    cp.testing.assert_allclose(
        wavefunction.minus_component,
        cp.sqrt(0.25) * cp.ones(wavefunction.grid.shape),
    )


def test_broken_axisymmetry_initial_state():
    """Tests whether the broken-axisymmetry initial state is set correctly."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    params = {"n0": 1, "p": 0.0, "q": 0.5, "c2": -0.5}
    wavefunction.set_ground_state("BA", params)

    cp.testing.assert_allclose(wavefunction.plus_component, cp.sqrt(2) / 4)
    cp.testing.assert_allclose(wavefunction.zero_component, cp.sqrt(3) / 2)
    cp.testing.assert_allclose(wavefunction.minus_component, cp.sqrt(2) / 4)


def test_custom_wavefunction_components():
    """Tests whether a custom wavefunction is set correctly."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    plus_component = 1.67 * cp.ones((64, 64))
    zero_component = 1e4 * cp.ones((64, 64))
    minus_component = cp.zeros((64, 64))

    wavefunction.set_wavefunction(
        plus_component, zero_component, minus_component
    )

    cp.testing.assert_array_equal(wavefunction.plus_component, plus_component)
    cp.testing.assert_array_equal(wavefunction.zero_component, zero_component)
    cp.testing.assert_array_equal(
        wavefunction.minus_component, minus_component
    )


def test_set_initial_state_raises_error():
    """Tests that an unsupported/invalid initial state returns an error."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    with pytest.raises(KeyError):
        wavefunction.set_ground_state("garbage", params={})


def test_adding_noise_outer():
    """Tests whether adding noise to empty outer components correctly
    makes those components non-zero.
    """
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    zeros = cp.zeros(wavefunction.grid.shape, dtype="complex128")
    wavefunction.set_wavefunction(
        cp.zeros_like(zeros), cp.zeros_like(zeros), cp.zeros_like(zeros)
    )
    wavefunction.add_noise("outer", 0, 1e-2)

    with pytest.raises(AssertionError):
        cp.testing.assert_array_equal(
            wavefunction.plus_component, cp.zeros(wavefunction.grid.shape)
        )
    with pytest.raises(AssertionError):
        cp.testing.assert_array_equal(
            wavefunction.minus_component, cp.zeros(wavefunction.grid.shape)
        )


def test_adding_noise_all():
    """Tests whether adding noise to all components correctly
    makes them non-zero.
    """
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    zeros = cp.zeros(wavefunction.grid.shape, dtype="complex128")
    wavefunction.set_wavefunction(
        cp.zeros_like(zeros), cp.zeros_like(zeros), cp.zeros_like(zeros)
    )
    wavefunction.add_noise("all", 0, 1e-2)

    with pytest.raises(AssertionError):
        cp.testing.assert_array_equal(
            wavefunction.plus_component, cp.zeros(wavefunction.grid.shape)
        )
    with pytest.raises(AssertionError):
        cp.testing.assert_array_equal(
            wavefunction.zero_component, cp.zeros(wavefunction.grid.shape)
        )
    with pytest.raises(AssertionError):
        cp.testing.assert_array_equal(
            wavefunction.minus_component, cp.zeros(wavefunction.grid.shape)
        )


def test_adding_noise_zero():
    """Tests whether adding noise to the middle component correctly
    makes it non-zero.
    """
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    zeros = cp.zeros(wavefunction.grid.shape, dtype="complex128")
    wavefunction.set_wavefunction(
        cp.zeros_like(zeros), cp.zeros_like(zeros), cp.zeros_like(zeros)
    )
    wavefunction.add_noise("zero", 0, 1e-2)

    with pytest.raises(AssertionError):
        cp.testing.assert_array_equal(
            wavefunction.zero_component, cp.zeros(wavefunction.grid.shape)
        )


def test_adding_noise_list():
    """Tests whether adding noise to specified components given as a list
    correctly makes those components non-zero."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    zeros = cp.zeros(wavefunction.grid.shape, dtype="complex128")
    wavefunction.set_wavefunction(
        cp.zeros_like(zeros), cp.zeros_like(zeros), cp.zeros_like(zeros)
    )
    wavefunction.add_noise(["plus", "zero"], 0, 1e-2)

    with pytest.raises(AssertionError):
        cp.testing.assert_array_equal(
            wavefunction.plus_component, cp.zeros(wavefunction.grid.shape)
        )
    with pytest.raises(AssertionError):
        cp.testing.assert_array_equal(
            wavefunction.zero_component, cp.zeros(wavefunction.grid.shape)
        )


def test_phase_all():
    """Tests that a phase applied to all components is applied correctly."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    wavefunction.set_wavefunction(
        cp.ones((64, 64), dtype="complex128"),
        cp.ones((64, 64), dtype="complex128"),
        cp.ones((64, 64), dtype="complex128"),
    )

    phase = cp.random.uniform(size=(64, 64), dtype=cp.float64)
    wavefunction.apply_phase(phase, "all")

    cp.testing.assert_allclose(cp.angle(wavefunction.plus_component), phase)
    cp.testing.assert_allclose(cp.angle(wavefunction.zero_component), phase)
    cp.testing.assert_allclose(cp.angle(wavefunction.minus_component), phase)


def test_phase_multiple_components():
    """Tests that a phase is applied correctly to multiple components."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    wavefunction.set_wavefunction(
        cp.ones((64, 64), dtype="complex128"),
        cp.ones((64, 64), dtype="complex128"),
        cp.ones((64, 64), dtype="complex128"),
    )

    phase = cp.random.uniform(size=(64, 64), dtype=cp.float64)
    wavefunction.apply_phase(phase, ["plus", "minus"])

    cp.testing.assert_allclose(cp.angle(wavefunction.plus_component), phase)
    cp.testing.assert_allclose(cp.angle(wavefunction.minus_component), phase)


def test_phase_single():
    """Tests that a phase is applied correctly to a single component."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    wavefunction.set_wavefunction(
        cp.ones((64, 64), dtype="complex128"),
        cp.ones((64, 64), dtype="complex128"),
        cp.ones((64, 64), dtype="complex128"),
    )

    phase = cp.random.uniform(size=(64, 64), dtype=cp.float64)
    wavefunction.apply_phase(phase, "zero")

    cp.testing.assert_allclose(cp.angle(wavefunction.zero_component), phase)


def test_density():
    """Tests that the condensate density is calculated correctly."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    wavefunction.set_ground_state("polar", {"n0": 1})

    cp.testing.assert_allclose(
        wavefunction.density(), cp.ones(wavefunction.grid.shape, dtype="float")
    )
