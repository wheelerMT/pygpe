import h5py
import cupy as cp
from pygpe.shared.grid import Grid
from pygpe.spin_1.wavefunction import Wavefunction


class DataManager:
    def __init__(self, filename: str, data_path: str):
        self.filename = filename
        self.data_path = data_path
        self.time_index = 0

        # Create file
        h5py.File(f'{self.data_path}/{self.filename}', 'w')

    def save_initial_parameters(self, grid: Grid, wfn: Wavefunction, parameters: dict) -> None:
        """Saves the initial grid, wavefunction and parameters to a HDF5 file.

        :param grid: The grid object of the system.
        :param wfn: The wavefunction of the system.
        :param parameters: The parameter object of the system.
        """

        self._save_initial_grid_params(grid)
        self._save_initial_wfn(wfn)
        self._save_params(parameters)

    def _save_initial_grid_params(self, grid: Grid) -> None:
        """Creates new datasets in file for grid-related parameters and saves
        initial values.
        """
        if grid.ndim == 1:
            with h5py.File(f'{self.data_path}/{self.filename}', 'w') as file:
                file.create_dataset('grid/nx', data=grid.num_points_x)
                file.create_dataset('grid/dx', data=grid.grid_spacing_x)
        elif grid.ndim == 2:
            with h5py.File(f'{self.data_path}/{self.filename}', 'w') as file:
                file.create_dataset('grid/nx', data=grid.num_points_x)
                file.create_dataset('grid/ny', data=grid.num_points_y)
                file.create_dataset('grid/dx', data=grid.grid_spacing_x)
                file.create_dataset('grid/dy', data=grid.grid_spacing_y)
        elif grid.ndim == 3:
            raise NotImplementedError

    def _save_initial_wfn(self, wfn: Wavefunction) -> None:
        """Creates new datasets in file for the wavefunction and saves
        initial values.
        """
        if wfn.grid.ndim == 1:
            with h5py.File(f'{self.data_path}/{self.filename}', 'w') as file:
                file.create_dataset('wavefunction/psi_plus', (wfn.grid.num_points_x, 1),
                                    maxshape=(wfn.grid.num_points_x, None), dtype='complex128')
                file.create_dataset('wavefunction/psi_zero', (wfn.grid.num_points_x, 1),
                                    maxshape=(wfn.grid.num_points_x, None), dtype='complex128')
                file.create_dataset('wavefunction/psi_minus', (wfn.grid.num_points_x, 1),
                                    maxshape=(wfn.grid.num_points_x, None), dtype='complex128')
        elif wfn.grid.ndim == 2:
            with h5py.File(f'{self.data_path}/{self.filename}', 'w') as file:
                file.create_dataset('wavefunction/psi_plus', (wfn.grid.num_points_x, wfn.grid.num_points_y, 1),
                                    maxshape=(wfn.grid.num_points_x, wfn.grid.num_points_y, None), dtype='complex128')
                file.create_dataset('wavefunction/psi_zero', (wfn.grid.num_points_x, wfn.grid.num_points_y, 1),
                                    maxshape=(wfn.grid.num_points_x, wfn.grid.num_points_y, None), dtype='complex128')
                file.create_dataset('wavefunction/psi_minus', (wfn.grid.num_points_x, wfn.grid.num_points_y, 1),
                                    maxshape=(wfn.grid.num_points_x, wfn.grid.num_points_y, None), dtype='complex128')
        elif wfn.grid.ndim == 3:
            raise NotImplementedError

    def _save_params(self, parameters: dict) -> None:
        """Creates new datasets in file for condensate & time parameters
        and saves initial values.
        """
        with h5py.File(f'{self.data_path}/{self.filename}', 'r+') as file:
            # Condensate and trap parameters
            file.create_dataset('parameters/c0', data=parameters["c0"])
            file.create_dataset('parameters/c2', data=parameters["c2"])
            file.create_dataset('parameters/p', data=parameters["p"])
            file.create_dataset('parameters/q', data=parameters["q"])
            file.create_dataset('parameters/trap', data=parameters["trap"])

            # Time-related parameters
            file.create_dataset('parameters/dt', data=parameters["dt"])
            file.create_dataset('parameters/nt', data=parameters["nt"])

    def save_wfn(self, wfn: Wavefunction) -> None:
        """Saves the current wavefunction data to the dataset.

        :param wfn: The wavefunction of the system.
        """
        wfn.ifft()  # Update real-space wavefunction before saving
        if wfn.grid.ndim == 1:
            with h5py.File(f'{self.data_path}/{self.filename}', 'r+') as data:
                new_psi_plus = data['wavefunction/psi_plus']
                new_psi_plus.resize((wfn.grid.num_points_x, self.time_index + 1))
                new_psi_plus[:, self.time_index] = wfn.plus_component

                new_psi_zero = data['wavefunction/psi_zero']
                new_psi_zero.resize((wfn.grid.num_points_x, self.time_index + 1))
                new_psi_zero[:, self.time_index] = wfn.zero_component

                new_psi_minus = data['wavefunction/psi_minus']
                new_psi_minus.resize((wfn.grid.num_points_x, self.time_index + 1))
                new_psi_minus[:, self.time_index] = wfn.minus_component
        elif wfn.grid.ndim == 2:
            with h5py.File(f'{self.data_path}/{self.filename}', 'r+') as data:
                new_psi_plus = data['wavefunction/psi_plus']
                new_psi_plus.resize((wfn.grid.num_points_x, wfn.grid.num_points_y, self.time_index + 1))
                new_psi_plus[:, :, self.time_index] = cp.asnumpy(wfn.plus_component)

                new_psi_zero = data['wavefunction/psi_zero']
                new_psi_zero.resize((wfn.grid.num_points_x, wfn.grid.num_points_y, self.time_index + 1))
                new_psi_zero[:, :, self.time_index] = cp.asnumpy(wfn.zero_component)

                new_psi_minus = data['wavefunction/psi_minus']
                new_psi_minus.resize((wfn.grid.num_points_x, wfn.grid.num_points_y, self.time_index + 1))
                new_psi_minus[:, :, self.time_index] = cp.asnumpy(wfn.minus_component)

        elif wfn.grid.ndim == 3:
            raise NotImplementedError
