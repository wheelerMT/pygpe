import cupy as cp
from pygpe.shared.grid import Grid2D
from typing import List, Tuple


class Phase2D:

    def __init__(self, grid: Grid2D):
        self.grid = grid

        self.phase_plus = cp.empty((grid.num_points_x, grid.num_points_y), dtype='float32')
        self.phase_zero = cp.empty((grid.num_points_x, grid.num_points_y), dtype='float32')
        self.phase_minus = cp.empty((grid.num_points_x, grid.num_points_y), dtype='float32')

    def add_singly_quantised_vortices(self, num_vortices: int, threshold: float) -> None:
        """Constructs a phase profile containing a number of vortices of singly quantised vortices,
        which is then applied to each spinor component.

        :param num_vortices: The total number of vortices.
        :param threshold: The minimum distance allowed between any two vortices.
        """
        vortex_positions_iter = self._generate_positions(num_vortices, threshold)

        phase = cp.empty((self.grid.num_points_x, self.grid.num_points_y), dtype='float32')

        for _ in range(num_vortices // 2):
            phase_temp = cp.zeros((self.grid.num_points_x, self.grid.num_points_y), dtype='float32')
            x_pos_minus, y_pos_minus = next(vortex_positions_iter)  # Negative circulation vortex
            x_pos_plus, y_pos_plus = next(vortex_positions_iter)  # Positive circulation vortex

            # Aux variables
            y_minus = 2 * cp.pi / self.grid.length_y * (self.grid.y_mesh - y_pos_minus)
            x_minus = 2 * cp.pi / self.grid.length_x * (self.grid.x_mesh - x_pos_minus)
            y_plus = 2 * cp.pi / self.grid.length_y * (self.grid.y_mesh - y_pos_plus)
            x_plus = 2 * cp.pi / self.grid.length_x * (self.grid.x_mesh - x_pos_plus)

            heaviside_x_plus = self._heaviside(x_plus)
            heaviside_x_minus = self._heaviside(x_minus)

            for nn in cp.arange(-5, 6):
                phase_temp += cp.arctan(cp.tanh((y_minus + 2 * cp.pi * nn) / 2) * cp.tan((x_minus - cp.pi) / 2)) \
                              - cp.arctan(cp.tanh((y_plus + 2 * cp.pi * nn) / 2) * cp.tan((x_plus - cp.pi) / 2)) \
                              + cp.pi * (heaviside_x_plus - heaviside_x_minus)
            phase_temp -= 2 * cp.pi * (self.grid.y_mesh - self.grid.y_mesh.min()) \
                          * (x_pos_plus - x_pos_minus) / (self.grid.length_y * self.grid.length_x)
            phase += phase_temp

        return phase

    def _generate_positions(self, num_vortices: int, threshold: float) -> iter:
        """Generates and returns a list of positions that are separated by at least
        `threshold`.
        """
        print(f"Attempting to find {num_vortices} positions...")
        max_iter = 10000
        vortex_positions = []

        iterations = 0
        while len(vortex_positions) < num_vortices:
            position = cp.random.uniform(-self.grid.length_x / 2, self.grid.length_x / 2), \
                       cp.random.uniform(-self.grid.length_y / 2, self.grid.length_y / 2)

            if self._position_sufficiently_far(position, vortex_positions, threshold):
                vortex_positions.append(position)

            iterations += 1
            if iterations > max_iter:
                print(f"WARNING: Number of iterations exceeded maximum, "
                      f"returning with only {len(vortex_positions)} positions\n")
                return vortex_positions

        print(f"Successfully found {num_vortices} positions!")
        return iter(vortex_positions)

    @staticmethod
    def _position_sufficiently_far(position: Tuple, accepted_positions: List[Tuple], threshold: float) -> bool:
        """Tests that the given `position` is at least `threshold` away from all the positions
        currently in `accepted_positions`.
        """
        for accepted_pos in accepted_positions:
            if abs(position[0] - accepted_pos[0]) > threshold:
                if abs(position[1] - accepted_pos[1]) > threshold:
                    return True
        return False

    @staticmethod
    def _heaviside(array: cp.ndarray):
        return cp.where(array < 0, cp.zeros(array.shape), cp.ones(array.shape))
