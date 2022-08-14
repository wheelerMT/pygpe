import numpy as np
import cupy as cp
from pygpe.shared.grid import Grid
from pygpe.spinone.wavefunction import Wavefunction
from typing import List, Tuple

"""Contains set of functions that can add various types of vortices to the wavefunction."""


def _generate_positions(grid: Grid, num_vortices: int, threshold: float) -> iter:
    """Generates and returns a list of positions that are separated by at least
    `threshold`.
    """
    print(f"Attempting to find {num_vortices} positions...")
    max_iter = 10000
    vortex_positions = []

    iterations = 0
    while len(vortex_positions) < num_vortices:
        if iterations > max_iter:
            print(f"WARNING: Number of iterations exceeded maximum, "
                  f"returning with only {len(vortex_positions)} positions\n")
            return vortex_positions

        position = np.random.uniform(-grid.length_x / 2, grid.length_x / 2), \
                   np.random.uniform(-grid.length_y / 2, grid.length_y / 2)

        if _position_sufficiently_far(position, vortex_positions, threshold):
            vortex_positions.append(position)

        iterations += 1

    print(f"Successfully found {num_vortices} positions in {iterations} iterations!")
    return iter(vortex_positions)


def _position_sufficiently_far(position: Tuple, accepted_positions: List[Tuple], threshold: float) -> bool:
    """Tests that the given `position` is at least `threshold` away from all the positions
    currently in `accepted_positions`.
    """
    # Special case where accepted_positions is empty
    if not accepted_positions:
        return True

    for accepted_pos in accepted_positions:
        if abs(position[0] - accepted_pos[0]) > threshold:
            if abs(position[1] - accepted_pos[1]) > threshold:
                return True
    return False


def _heaviside(array: np.ndarray) -> np.ndarray:
    """Computes the heaviside function on a given array and returns the result."""
    return np.where(array < 0, np.zeros(array.shape), np.ones(array.shape))


def add_singly_quantised_vortices(wavefunction: Wavefunction, num_vortices: int, threshold: float) -> None:
    """Constructs a 2D phase profile containing a number of singly quantised vortices,
    which is then applied to each wavefunction component.

    :param wavefunction: The 2D wavefunction of the system.
    :param num_vortices: The total number of vortices.
    :param threshold: The minimum distance allowed between any two vortices.
    """
    vortex_positions_iter = _generate_positions(wavefunction.grid, num_vortices, threshold)

    phase = np.zeros((wavefunction.grid.num_points_x, wavefunction.grid.num_points_y), dtype='float32')

    for _ in range(num_vortices // 2):
        phase_temp = np.zeros((wavefunction.grid.num_points_x, wavefunction.grid.num_points_y), dtype='float32')
        x_pos_minus, y_pos_minus = next(vortex_positions_iter)  # Negative circulation vortex
        x_pos_plus, y_pos_plus = next(vortex_positions_iter)  # Positive circulation vortex

        # Aux variables
        y_minus = 2 * np.pi / wavefunction.grid.length_y * (wavefunction.grid.y_mesh - y_pos_minus)
        x_minus = 2 * np.pi / wavefunction.grid.length_x * (wavefunction.grid.x_mesh - x_pos_minus)
        y_plus = 2 * np.pi / wavefunction.grid.length_y * (wavefunction.grid.y_mesh - y_pos_plus)
        x_plus = 2 * np.pi / wavefunction.grid.length_x * (wavefunction.grid.x_mesh - x_pos_plus)

        heaviside_x_plus = _heaviside(x_plus)
        heaviside_x_minus = _heaviside(x_minus)

        for nn in np.arange(-5, 6):
            phase_temp += np.arctan(np.tanh((y_minus + 2 * np.pi * nn) / 2) * np.tan((x_minus - np.pi) / 2)) \
                          - np.arctan(np.tanh((y_plus + 2 * np.pi * nn) / 2) * np.tan((x_plus - np.pi) / 2)) \
                          + np.pi * (heaviside_x_plus - heaviside_x_minus)
        phase_temp -= 2 * np.pi * (wavefunction.grid.y_mesh - wavefunction.grid.y_mesh.min()) \
                      * (x_pos_plus - x_pos_minus) / (wavefunction.grid.length_y * wavefunction.grid.length_x)
        phase += phase_temp

    wavefunction.plus_component *= cp.exp(1j * cp.asarray(phase))
    wavefunction.zero_component *= cp.exp(1j * cp.asarray(phase))
    wavefunction.minus_component *= cp.exp(1j * cp.asarray(phase))
