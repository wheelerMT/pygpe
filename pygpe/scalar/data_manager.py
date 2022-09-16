import h5py
import cupy as cp
from pygpe.shared.grid import Grid
from pygpe.scalar.wavefunction import Wavefunction


class DataManager:
    def __init__(self, filename: str, data_path: str):
        self.filename = filename
        self.data_path = data_path
        self._time_index = 0

        # Create file
        h5py.File(f'{self.data_path}/{self.filename}', 'w')

    def save_initial_parameters(self, grid: Grid, wfn: Wavefunction, parameters: dict) -> None:
        """Saves the initial grid, wavefunction and parameters to a HDF5 file.

        :param grid: The grid object of the system.
        :param wfn: The wavefunction of the system.
        :param parameters: The parameter dictionary.
        """
        self._save_grid_params(grid)
        self._save_initial_wfn(wfn)
        self._save_params(parameters)

    def _save_grid_params(self, grid: Grid) -> None:
        """Saves grid parameters to dataset."""
        if grid.ndim == 1:
            with h5py.File(f'{self.data_path}/{self.filename}', 'r+') as file:
                file.create_dataset('grid/nx', data=grid.num_points_x)
                file.create_dataset('grid/dx', data=grid.grid_spacing_x)
        elif grid.ndim == 2:
            with h5py.File(f'{self.data_path}/{self.filename}', 'r+') as file:
                file.create_dataset('grid/nx', data=grid.num_points_x)
                file.create_dataset('grid/ny', data=grid.num_points_y)
                file.create_dataset('grid/dx', data=grid.grid_spacing_x)
                file.create_dataset('grid/dy', data=grid.grid_spacing_y)
        elif grid.ndim == 3:
            with h5py.File(f'{self.data_path}/{self.filename}', 'r+') as file:
                file.create_dataset('grid/nx', data=grid.num_points_x)
                file.create_dataset('grid/ny', data=grid.num_points_y)
                file.create_dataset('grid/nz', data=grid.num_points_z)
                file.create_dataset('grid/dx', data=grid.grid_spacing_x)
                file.create_dataset('grid/dy', data=grid.grid_spacing_y)
                file.create_dataset('grid/dz', data=grid.grid_spacing_z)

    def _save_initial_wfn(self, wfn: Wavefunction) -> None:
        """Saves initial wavefunction to dataset."""
        if wfn.grid.ndim == 1:
            with h5py.File(f'{self.data_path}/{self.filename}', 'r+') as file:
                file.create_dataset('wavefunction', (wfn.grid.shape, 1), maxshape=(wfn.grid.shape, None),
                                    dtype='complex128')
        else:
            with h5py.File(f'{self.data_path}/{self.filename}', 'r+') as file:
                file.create_dataset('wavefunction', (*wfn.grid.shape, 1), maxshape=(*wfn.grid.shape, None),
                                    dtype='complex128')

    def _save_params(self, parameters: dict) -> None:
        """Saves condensate parameters to dataset."""
        with h5py.File(f'{self.data_path}/{self.filename}', 'r+') as file:
            for key in parameters:
                file.create_dataset(f'parameters/{key}', data=parameters[key])

    def save_wavefunction(self, wfn: Wavefunction) -> None:
        """Saves the current wavefunction data to the dataset.

        :param wfn: The wavefunction of the system.
        """
        wfn.ifft()  # Update real-space wavefunction before saving
        if wfn.grid.ndim == 1:
            with h5py.File(f'{self.data_path}/{self.filename}', 'r+') as data:
                new_psi = data['wavefunction']
                new_psi.resize((wfn.grid.num_points_x, self._time_index + 1))
                new_psi[:, self._time_index] = cp.asnumpy(wfn.wavefunction)
        else:
            with h5py.File(f'{self.data_path}/{self.filename}', 'r+') as data:
                new_psi = data['wavefunction']
                new_psi.resize((*wfn.grid.shape, self._time_index + 1))
                new_psi[..., self._time_index] = cp.asnumpy(wfn.wavefunction)

        self._time_index += 1
