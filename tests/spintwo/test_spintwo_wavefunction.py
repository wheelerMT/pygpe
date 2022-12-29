import cupy as cp
from typing import Tuple
from pygpe.shared.grid import Grid
from pygpe.spintwo.wavefunction import Wavefunction


def generate_wavefunction2d(
    points: Tuple[int, int], grid_spacing: Tuple[float, float]
) -> Wavefunction:
    """Generates and returns a Wavefunction2D object specified

    :param points: The number of grid points in the x and y dimension, respectively.
    :param grid_spacing: The spacing of grid points in the x and y dimension, respectively.
    :return: The Wavefunction2D object.
    """
    return Wavefunction(Grid(points, grid_spacing))


def test_uniaxial_initial_state():
    """Tests whether the uniaxial nematic initial state is set correctly."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    params = {"n0": 1}
    wavefunction.set_ground_state("UN", params)

    cp.testing.assert_array_equal(
        wavefunction.plus2_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.plus1_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.zero_component, cp.ones((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.minus1_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.minus2_component, cp.zeros((64, 64), dtype="complex128")
    )


def test_biaxial_initial_state():
    """Tests whether the biaxial nematic initial state is set correctly."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    params = {"n0": 1}
    wavefunction.set_ground_state("BN", params)

    cp.testing.assert_array_equal(
        wavefunction.plus2_component,
        1 / cp.sqrt(2.0) * cp.ones((64, 64), dtype="complex128"),
    )
    cp.testing.assert_array_equal(
        wavefunction.plus1_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.zero_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.minus1_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.minus2_component,
        1 / cp.sqrt(2.0) * cp.ones((64, 64), dtype="complex128"),
    )


def test_f2p_initial_state():
    """Tests whether the ferromagnetic-2 (with spin up) initial state is set correctly."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    params = {"n0": 1}
    wavefunction.set_ground_state("F2p", params)

    cp.testing.assert_array_equal(
        wavefunction.plus2_component, cp.ones((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.plus1_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.zero_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.minus1_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.minus2_component, cp.zeros((64, 64), dtype="complex128")
    )


def test_f2m_initial_state():
    """Tests whether the ferromagnetic-2 (with spin down) initial state is set correctly."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    params = {"n0": 1}
    wavefunction.set_ground_state("F2m", params)

    cp.testing.assert_array_equal(
        wavefunction.plus2_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.plus1_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.zero_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.minus1_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.minus2_component, cp.ones((64, 64), dtype="complex128")
    )


def test_f1p_initial_state():
    """Tests whether the ferromagnetic-1 (with spin up) initial state is set correctly."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    params = {"n0": 1}
    wavefunction.set_ground_state("F1p", params)

    cp.testing.assert_array_equal(
        wavefunction.plus2_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.plus1_component, cp.ones((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.zero_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.minus1_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.minus2_component, cp.zeros((64, 64), dtype="complex128")
    )


def test_f1m_initial_state():
    """Tests whether the ferromagnetic-1 (with spin down) initial state is set correctly."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    params = {"n0": 1}
    wavefunction.set_ground_state("F1m", params)

    cp.testing.assert_array_equal(
        wavefunction.plus2_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.plus1_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.zero_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.minus1_component, cp.ones((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.minus2_component, cp.zeros((64, 64), dtype="complex128")
    )


def test_cyclic_initial_state():
    """Tests whether the cyclic initial state is set correctly."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    params = {"n0": 1, "c2": 0.5, "p": 0.0, "q": 0}
    wavefunction.set_ground_state("cyclic", params)

    cp.testing.assert_array_equal(
        wavefunction.plus2_component,
        cp.sqrt(1 / 3) * cp.ones((64, 64), dtype="complex128"),
    )
    cp.testing.assert_array_equal(
        wavefunction.plus1_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.zero_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.minus1_component,
        cp.sqrt(2 / 3) * cp.ones((64, 64), dtype="complex128"),
    )
    cp.testing.assert_array_equal(
        wavefunction.minus2_component, cp.zeros((64, 64), dtype="complex128")
    )


def test_custom_components():
    """Tests whether custom wavefunction components are set correctly."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    plus2_comp = cp.ones(wavefunction.grid.shape, dtype="complex128")
    plus1_comp = cp.zeros(wavefunction.grid.shape, dtype="complex128")
    zero_comp = cp.zeros(wavefunction.grid.shape, dtype="complex128")
    minus1_comp = cp.sqrt(1 / 3) * cp.ones(wavefunction.grid.shape, dtype="complex128")
    minus2_comp = 5 * cp.ones(wavefunction.grid.shape, dtype="complex128")
    wavefunction.set_custom_components(
        plus2_comp, plus1_comp, zero_comp, minus1_comp, minus2_comp
    )

    cp.testing.assert_array_equal(wavefunction.plus2_component, plus2_comp)
    cp.testing.assert_array_equal(wavefunction.plus1_component, plus1_comp)
    cp.testing.assert_array_equal(wavefunction.zero_component, zero_comp)
    cp.testing.assert_array_equal(wavefunction.minus1_component, minus1_comp)
    cp.testing.assert_array_equal(wavefunction.minus2_component, minus2_comp)
