import pytest
import cupy as cp
from typing import Tuple
from pygpe.shared.grid import Grid
from pygpe.scalar.wavefunction import ScalarWavefunction


def generate_wavefunction2d(
    points: Tuple[int, int], grid_spacing: Tuple[float, float]
) -> ScalarWavefunction:
    """Generates and returns a Wavefunction2D object specified

    :param points: The number of grid points in the x and y dimension,
        respectively.
    :param grid_spacing: The spacing of grid points in the x and y dimension,
        respectively.
    :return: The Wavefunction2D object.
    """
    return ScalarWavefunction(Grid(points, grid_spacing))


def test_set_wavefunction():
    """Tests whether the wavefunction array is set correctly."""

    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    array = cp.ones((64, 64))
    wavefunction.set_wavefunction(array)

    cp.testing.assert_array_equal(wavefunction.component, array)


def test_adding_noise():
    """Tests whether adding noise to empty wavefunction correctly
    makes the wavefunction non-zero.
    """
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    wavefunction.add_noise(0, 1e-2)

    with pytest.raises(AssertionError):
        cp.testing.assert_array_equal(
            wavefunction.component, cp.zeros(wavefunction.grid.shape)
        )


def test_fft():
    """Tests whether performing a forward fft followed by an inverse
    fft returns the same result (up to numerical error).
    """
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    wavefunction.add_noise(0, 1e-2)

    before_fft = wavefunction.component
    wavefunction.fft()
    wavefunction.ifft()

    cp.testing.assert_allclose(wavefunction.component, before_fft)


def test_density():
    """Tests whether the atomic density is calculated correctly."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    wavefunction.set_wavefunction(5 * cp.ones((64, 64)))

    cp.testing.assert_array_equal(
        wavefunction.density(), 25 * cp.ones((64, 64))
    )


def test_phase():
    """Tests whether the specified phase gets applied correctly."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    wavefunction.set_wavefunction(5 * cp.ones((64, 64), dtype="complex128"))

    phase = cp.random.uniform(0, 1, size=(64, 64))
    wavefunction.apply_phase(phase)

    cp.testing.assert_allclose(phase, cp.angle(wavefunction.component))
