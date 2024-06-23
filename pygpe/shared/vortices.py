try:
    import cupy as cp  # type: ignore
except ImportError:
    import numpy as cp
from pygpe.shared.grid import Grid


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
            print(
                f"WARNING: Number of iterations exceeded maximum, "
                f"returning with only {len(vortex_positions)} positions\n"
            )
            return vortex_positions

        position = cp.random.uniform(
            -grid.length_x / 2, grid.length_x / 2
        ), cp.random.uniform(-grid.length_y / 2, grid.length_y / 2)

        if _position_sufficiently_far(position, vortex_positions, threshold):
            vortex_positions.append(position)

        iterations += 1

    print(
        f"Successfully found {num_vortices} positions " f"in {iterations} iterations!"
    )
    return iter(vortex_positions)


def _position_sufficiently_far(
    position: tuple, accepted_positions: list[tuple], threshold: float
) -> bool:
    """Tests that the given `position` is at least `threshold` away from all
    the positions currently in `accepted_positions`.
    """
    # Special case where accepted_positions is empty
    if not accepted_positions:
        return True

    for accepted_pos in accepted_positions:
        if abs(position[0] - accepted_pos[0]) > threshold:
            if abs(position[1] - accepted_pos[1]) > threshold:
                return True
    return False


def _heaviside(array: cp.ndarray) -> cp.ndarray:
    """Computes the heaviside function on a given array and returns the
    result.
    """
    return cp.where(array < 0, cp.zeros(array.shape), cp.ones(array.shape))


def _calculate_vortex_contribution(grid, x_pos, y_pos, circulation):
    """Calculates the phase contribution from a single vortex at specified position.

    :param grid: The grid on which the phase is calculated.
    :param x_pos: x position of the vortex.
    :param y_pos: y position of the vortex.
    :param circulation: Circulation of the vortex, +1 or -1.
    """
    y = 2 * cp.pi / grid.length_y * (grid.y_mesh - y_pos)
    x = 2 * cp.pi / grid.length_x * (grid.x_mesh - x_pos)
    phase_contribution = cp.arctan2(y, x)

    if circulation == -1:
        phase_contribution = -phase_contribution

    return phase_contribution


def vortex_phase_profile(grid: Grid, num_vortices: int, threshold: float) -> cp.ndarray:
    """Constructs a 2D phase profile consisting of 2pi phase windings.
    This phase can be applied to a wavefunction to generate different types of
    vortices.

    :param grid: The 2D grid of the system.
    :type grid: :class:`Grid`
    :param num_vortices: The total number of vortices to be included in the
        phase profile.
    :type num_vortices: int
    :param threshold: The minimum distance allowed between any two vortices.
    :type threshold: float
    """
    vortex_positions_iter = _generate_positions(grid, num_vortices, threshold)

    phase = cp.zeros((grid.num_points_x, grid.num_points_y), dtype="float32")

    for _ in range(num_vortices // 2):
        phase_temp = cp.zeros((grid.num_points_x, grid.num_points_y), dtype="float32")
        x_pos_minus, y_pos_minus = next(
            vortex_positions_iter
        )  # Negative circulation vortex
        x_pos_plus, y_pos_plus = next(
            vortex_positions_iter
        )  # Positive circulation vortex

        # Aux variables
        y_minus = 2 * cp.pi / grid.length_y * (grid.y_mesh - y_pos_minus)
        x_minus = 2 * cp.pi / grid.length_x * (grid.x_mesh - x_pos_minus)
        y_plus = 2 * cp.pi / grid.length_y * (grid.y_mesh - y_pos_plus)
        x_plus = 2 * cp.pi / grid.length_x * (grid.x_mesh - x_pos_plus)

        heaviside_x_plus = _heaviside(x_plus)
        heaviside_x_minus = _heaviside(x_minus)

        for nn in cp.arange(-5, 6):
            phase_temp += (
                cp.arctan(
                    cp.tanh((y_minus + 2 * cp.pi * nn) / 2)
                    * cp.tan((x_minus - cp.pi) / 2)
                )
                - cp.arctan(
                    cp.tanh((y_plus + 2 * cp.pi * nn) / 2)
                    * cp.tan((x_plus - cp.pi) / 2)
                )
                + cp.pi * (heaviside_x_plus - heaviside_x_minus)
            )
        phase_temp -= (
            2
            * cp.pi
            * (grid.y_mesh - grid.y_mesh.min())
            * (x_pos_plus - x_pos_minus)
            / (grid.length_y * grid.length_x)
        )
        phase += phase_temp

    return phase


def add_dipole_pair(grid: Grid, threshold: float) -> cp.ndarray:
    """Creates a phase profile with a central dipole pair separated along the y-axis by the specified threshold.

    :param grid: The 2D grid of the system.
    :type grid: :class:`Grid`
    :param threshold: The distance between the two vortices.
    :type threshold: float
    """
    # Central position along x and specific positions along y
    x_pos = 0  # Central along the x-axis
    y_pos = 0  # Central along the y-axis

    # Calculate positions for negative and positive circulation vortices
    y_pos_minus = y_pos - threshold / 2
    y_pos_plus = y_pos + threshold / 2

    phase = cp.zeros((grid.num_points_x, grid.num_points_y), dtype="float32")

    # Calculate phase contributions from both vortices
    phase += _calculate_vortex_contribution(
        grid, x_pos, y_pos_minus, -1
    )  # Negative circulation
    phase += _calculate_vortex_contribution(
        grid, x_pos, y_pos_plus, 1
    )  # Positive circulation

    return phase
