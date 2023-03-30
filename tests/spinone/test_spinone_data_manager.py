import cupy as cp
import h5py
from pygpe.shared.grid import Grid
from pygpe.spinone.data_manager import DataManager
import pygpe.shared.data_manager_paths as dmp
from pygpe.spinone.wavefunction import SpinOneWavefunction

FILENAME = "spinone_test.hdf5"
FILE_PATH = "data"


def generate_wavefunction(
    points: tuple[int, int] = (64, 64),
    grid_spacing: tuple[float, float] = (0.5, 0.5),
) -> SpinOneWavefunction:
    """Generates a 2D `Wavefunction` object for use in testing.

    :param points: Number of grid points in each dimension,
        defaults to (64, 64).
    :type points: tuple[int, int], optional
    :param grid_spacing: Grid spacing in each dimension, defaults to
        (0.5, 0.5).
    :type grid_spacing: tuple[float, float], optional.
    :return: `SpinOneWavefunction` object.
    :rtype: SpinOneWavefunction.
    """
    return SpinOneWavefunction(Grid(points, grid_spacing))


def generate_parameters() -> dict:
    """Generates the spin-1 BEC parameters dictionary for use in testing.

    :return: The generated dictionary.
    :rtype: dict.
    """
    spinone_parameter_types = [
        "c0",
        "c2",
        "p",
        "q",
        "n0",
        "nt",
        "dt",
        "t",
    ]
    params = {}
    for key in spinone_parameter_types:
        params[key] = hash(key)

    return params


def test_data_manager_creation():
    """Tests whether the DataManager gets constructed without errors."""

    wavefunction = generate_wavefunction()
    params = generate_parameters()
    DataManager(FILENAME, FILE_PATH, wavefunction, params)


def test_correct_parameters():
    """Tests whether condensate parameters are correctly saved to file."""

    wavefunction = generate_wavefunction()
    params = generate_parameters()
    DataManager(FILENAME, FILE_PATH, wavefunction, params)

    with h5py.File(f"{FILE_PATH}/{FILENAME}", "r") as file:
        for key, value in params.items():
            assert value == file[f"{dmp.PARAMETERS}/{key}"][...]


def test_correct_wavefunction():
    """Tests whether the condensate wavefunction is correctly saved to file."""
    wavefunction = generate_wavefunction()
    params = generate_parameters()
    DataManager(FILENAME, FILE_PATH, wavefunction, params)

    with h5py.File(f"{FILE_PATH}/{FILENAME}", "r") as file:
        saved_wavefunction_plus = cp.asarray(
            file[f"{dmp.SPIN1_WAVEFUNCTION_PLUS}"][:, :, 0]
        )
        cp.testing.assert_array_almost_equal(
            wavefunction.plus_component, saved_wavefunction_plus
        )
        saved_wavefunction_zero = cp.asarray(
            file[f"{dmp.SPIN1_WAVEFUNCTION_ZERO}"][:, :, 0]
        )
        cp.testing.assert_array_almost_equal(
            wavefunction.zero_component, saved_wavefunction_zero
        )
        saved_wavefunction_minus = cp.asarray(
            file[f"{dmp.SPIN1_WAVEFUNCTION_MINUS}"][:, :, 0]
        )
        cp.testing.assert_array_almost_equal(
            wavefunction.minus_component, saved_wavefunction_minus
        )
