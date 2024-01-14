import numpy as np
import pytest
from pygpe.shared.grid import Grid
from pygpe.spintwo.wavefunction import SpinTwoWavefunction


def generate_wavefunction2d(
    points: tuple[int, int], grid_spacing: tuple[float, float]
) -> SpinTwoWavefunction:
    """Generates and returns a Wavefunction2D object specified

    :param points: The number of grid points in the x and y dimension,
        respectively.
    :param grid_spacing: The spacing of grid points in the x and y dimension,
        respectively.
    :return: The Wavefunction2D object.
    """
    return SpinTwoWavefunction(Grid(points, grid_spacing))


def test_uniaxial_initial_state():
    """Tests whether the uniaxial nematic initial state is set correctly."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    params = {"n0": 1}
    wavefunction.set_ground_state("UN", params)

    np.testing.assert_array_equal(
        wavefunction.plus2_component, np.zeros((64, 64), dtype="complex128")
    )
    np.testing.assert_array_equal(
        wavefunction.plus1_component, np.zeros((64, 64), dtype="complex128")
    )
    np.testing.assert_array_equal(
        wavefunction.zero_component, np.ones((64, 64), dtype="complex128")
    )
    np.testing.assert_array_equal(
        wavefunction.minus1_component, np.zeros((64, 64), dtype="complex128")
    )
    np.testing.assert_array_equal(
        wavefunction.minus2_component, np.zeros((64, 64), dtype="complex128")
    )


def test_biaxial_initial_state():
    """Tests whether the biaxial nematic initial state is set correctly."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    params = {"n0": 1}
    wavefunction.set_ground_state("BN", params)

    np.testing.assert_array_equal(
        wavefunction.plus2_component,
        1 / np.sqrt(2.0) * np.ones((64, 64), dtype="complex128"),
    )
    np.testing.assert_array_equal(
        wavefunction.plus1_component, np.zeros((64, 64), dtype="complex128")
    )
    np.testing.assert_array_equal(
        wavefunction.zero_component, np.zeros((64, 64), dtype="complex128")
    )
    np.testing.assert_array_equal(
        wavefunction.minus1_component, np.zeros((64, 64), dtype="complex128")
    )
    np.testing.assert_array_equal(
        wavefunction.minus2_component,
        1 / np.sqrt(2.0) * np.ones((64, 64), dtype="complex128"),
    )


def test_f2p_initial_state():
    """Tests whether the ferromagnetic-2 (with spin up) initial state is set
    correctly.
    """
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    params = {"n0": 1}
    wavefunction.set_ground_state("F2p", params)

    np.testing.assert_array_equal(
        wavefunction.plus2_component, np.ones((64, 64), dtype="complex128")
    )
    np.testing.assert_array_equal(
        wavefunction.plus1_component, np.zeros((64, 64), dtype="complex128")
    )
    np.testing.assert_array_equal(
        wavefunction.zero_component, np.zeros((64, 64), dtype="complex128")
    )
    np.testing.assert_array_equal(
        wavefunction.minus1_component, np.zeros((64, 64), dtype="complex128")
    )
    np.testing.assert_array_equal(
        wavefunction.minus2_component, np.zeros((64, 64), dtype="complex128")
    )


def test_f2m_initial_state():
    """Tests whether the ferromagnetic-2 (with spin down) initial state is set
    correctly.
    """
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    params = {"n0": 1}
    wavefunction.set_ground_state("F2m", params)

    np.testing.assert_array_equal(
        wavefunction.plus2_component, np.zeros((64, 64), dtype="complex128")
    )
    np.testing.assert_array_equal(
        wavefunction.plus1_component, np.zeros((64, 64), dtype="complex128")
    )
    np.testing.assert_array_equal(
        wavefunction.zero_component, np.zeros((64, 64), dtype="complex128")
    )
    np.testing.assert_array_equal(
        wavefunction.minus1_component, np.zeros((64, 64), dtype="complex128")
    )
    np.testing.assert_array_equal(
        wavefunction.minus2_component, np.ones((64, 64), dtype="complex128")
    )


def test_f1p_initial_state():
    """Tests whether the ferromagnetic-1 (with spin up) initial state is set
    correctly.
    """
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    params = {"n0": 1}
    wavefunction.set_ground_state("F1p", params)

    np.testing.assert_array_equal(
        wavefunction.plus2_component, np.zeros((64, 64), dtype="complex128")
    )
    np.testing.assert_array_equal(
        wavefunction.plus1_component, np.ones((64, 64), dtype="complex128")
    )
    np.testing.assert_array_equal(
        wavefunction.zero_component, np.zeros((64, 64), dtype="complex128")
    )
    np.testing.assert_array_equal(
        wavefunction.minus1_component, np.zeros((64, 64), dtype="complex128")
    )
    np.testing.assert_array_equal(
        wavefunction.minus2_component, np.zeros((64, 64), dtype="complex128")
    )


def test_f1m_initial_state():
    """Tests whether the ferromagnetic-1 (with spin down) initial state is set
    correctly.
    """
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    params = {"n0": 1}
    wavefunction.set_ground_state("F1m", params)

    np.testing.assert_array_equal(
        wavefunction.plus2_component, np.zeros((64, 64), dtype="complex128")
    )
    np.testing.assert_array_equal(
        wavefunction.plus1_component, np.zeros((64, 64), dtype="complex128")
    )
    np.testing.assert_array_equal(
        wavefunction.zero_component, np.zeros((64, 64), dtype="complex128")
    )
    np.testing.assert_array_equal(
        wavefunction.minus1_component, np.ones((64, 64), dtype="complex128")
    )
    np.testing.assert_array_equal(
        wavefunction.minus2_component, np.zeros((64, 64), dtype="complex128")
    )


def test_cyclic_initial_state():
    """Tests whether the cyclic initial state is set correctly."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    params = {"n0": 1, "c2": 0.5, "p": 0.0, "q": 0}
    wavefunction.set_ground_state("cyclic", params)

    np.testing.assert_array_equal(
        wavefunction.plus2_component,
        np.sqrt(1 / 3) * np.ones((64, 64), dtype="complex128"),
    )
    np.testing.assert_array_equal(
        wavefunction.plus1_component, np.zeros((64, 64), dtype="complex128")
    )
    np.testing.assert_array_equal(
        wavefunction.zero_component, np.zeros((64, 64), dtype="complex128")
    )
    np.testing.assert_array_equal(
        wavefunction.minus1_component,
        np.sqrt(2 / 3) * np.ones((64, 64), dtype="complex128"),
    )
    np.testing.assert_array_equal(
        wavefunction.minus2_component, np.zeros((64, 64), dtype="complex128")
    )


def test_custom_components():
    """Tests whether custom wavefunction components are set correctly."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    plus2_comp = np.ones(wavefunction.grid.shape, dtype="complex128")
    plus1_comp = np.zeros(wavefunction.grid.shape, dtype="complex128")
    zero_comp = np.zeros(wavefunction.grid.shape, dtype="complex128")
    minus1_comp = np.sqrt(1 / 3) * np.ones(
        wavefunction.grid.shape, dtype="complex128"
    )
    minus2_comp = 5 * np.ones(wavefunction.grid.shape, dtype="complex128")
    wavefunction.set_wavefunction(
        plus2_comp, plus1_comp, zero_comp, minus1_comp, minus2_comp
    )

    np.testing.assert_array_equal(wavefunction.plus2_component, plus2_comp)
    np.testing.assert_array_equal(wavefunction.plus1_component, plus1_comp)
    np.testing.assert_array_equal(wavefunction.zero_component, zero_comp)
    np.testing.assert_array_equal(wavefunction.minus1_component, minus1_comp)
    np.testing.assert_array_equal(wavefunction.minus2_component, minus2_comp)


def test_adding_noise_list():
    """Tests whether adding noise to specified empty components correctly
    makes those components non-zero.
    """
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    zeros = np.zeros(wavefunction.grid.shape, dtype="complex128")
    wavefunction.set_wavefunction(
        np.zeros_like(zeros),
        np.zeros_like(zeros),
        np.zeros_like(zeros),
        np.zeros_like(zeros),
        np.zeros_like(zeros),
    )
    wavefunction.add_noise(["plus2", "plus1"], 0, 1e-2)

    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(
            wavefunction.plus2_component, np.zeros(wavefunction.grid.shape)
        )
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(
            wavefunction.plus1_component, np.zeros(wavefunction.grid.shape)
        )


def test_adding_noise_all():
    """Tests whether adding noise to all empty components correctly
    makes those components non-zero.
    """
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    zeros = np.zeros(wavefunction.grid.shape, dtype="complex128")
    wavefunction.set_wavefunction(
        np.zeros_like(zeros),
        np.zeros_like(zeros),
        np.zeros_like(zeros),
        np.zeros_like(zeros),
        np.zeros_like(zeros),
    )
    wavefunction.add_noise("all", 0, 1e-2)

    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(
            wavefunction.plus2_component, np.zeros(wavefunction.grid.shape)
        )
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(
            wavefunction.plus1_component, np.zeros(wavefunction.grid.shape)
        )
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(
            wavefunction.zero_component, np.zeros(wavefunction.grid.shape)
        )
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(
            wavefunction.minus1_component, np.zeros(wavefunction.grid.shape)
        )
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(
            wavefunction.minus2_component, np.zeros(wavefunction.grid.shape)
        )


def test_phase_all():
    """Tests that a phase applied to all components is applied correctly."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    wavefunction.set_wavefunction(
        np.ones((64, 64), dtype="complex128"),
        np.ones((64, 64), dtype="complex128"),
        np.ones((64, 64), dtype="complex128"),
        np.ones((64, 64), dtype="complex128"),
        np.ones((64, 64), dtype="complex128"),
    )

    phase = np.random.uniform(size=(64, 64))
    wavefunction.apply_phase(phase, "all")

    np.testing.assert_allclose(np.angle(wavefunction.plus2_component), phase)
    np.testing.assert_allclose(np.angle(wavefunction.plus1_component), phase)
    np.testing.assert_allclose(np.angle(wavefunction.zero_component), phase)
    np.testing.assert_allclose(np.angle(wavefunction.minus1_component), phase)
    np.testing.assert_allclose(np.angle(wavefunction.minus2_component), phase)


def test_phase_multiple_components():
    """Tests that a phase is applied correctly to multiple components."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    wavefunction.set_wavefunction(
        np.ones((64, 64), dtype="complex128"),
        np.ones((64, 64), dtype="complex128"),
        np.ones((64, 64), dtype="complex128"),
        np.ones((64, 64), dtype="complex128"),
        np.ones((64, 64), dtype="complex128"),
    )

    phase = np.random.uniform(size=(64, 64))
    wavefunction.apply_phase(phase, ["plus1", "minus1"])

    np.testing.assert_allclose(np.angle(wavefunction.plus1_component), phase)
    np.testing.assert_allclose(np.angle(wavefunction.minus1_component), phase)


def test_phase_single():
    """Tests that a phase is applied correctly to a single component."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    wavefunction.set_wavefunction(
        np.ones((64, 64), dtype="complex128"),
        np.ones((64, 64), dtype="complex128"),
        np.ones((64, 64), dtype="complex128"),
        np.ones((64, 64), dtype="complex128"),
        np.ones((64, 64), dtype="complex128"),
    )

    phase = np.random.uniform(size=(64, 64))
    wavefunction.apply_phase(phase, "zero")

    np.testing.assert_allclose(np.angle(wavefunction.zero_component), phase)


def test_density():
    """Tests that the condensate density is calculated correctly."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    wavefunction.set_ground_state("UN", {"n0": 1})

    np.testing.assert_allclose(
        wavefunction.density(), np.ones(wavefunction.grid.shape, dtype="float")
    )
