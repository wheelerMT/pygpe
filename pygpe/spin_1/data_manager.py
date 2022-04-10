import h5py
from pygpe.shared.grid import Grid
from pygpe.spin_1.wavefunction import Wavefunction


class DataManager:
    def __init__(self, filename: str, grid: Grid, wfn: Wavefunction, params: dict):
        self.filename = filename
        self.data_path = f'data/{self.filename}'

        self._save_initial_grid_params(grid)
        self._save_initial_wfn(wfn)
        self._save_params(params)

    def _save_initial_grid_params(self, grid: Grid) -> None:
        if grid.ndim == 1:
            with h5py.File(f'{self.filename}.hdf5', 'w') as file:
                file.create_dataset('grid/nx', data=grid.num_points_x)
                file.create_dataset('grid/dx', data=grid.grid_spacing_x)
        elif grid.ndim == 2:
            with h5py.File(f'{self.filename}.hdf5', 'w') as file:
                file.create_dataset('grid/nx', data=grid.num_points_x)
                file.create_dataset('grid/ny', data=grid.num_points_y)
                file.create_dataset('grid/dx', data=grid.grid_spacing_x)
                file.create_dataset('grid/dy', data=grid.grid_spacing_y)
        elif grid.ndim == 3:
            raise NotImplementedError

    def _save_initial_wfn(self, wfn: Wavefunction) -> None:
        if wfn.grid.ndim == 1:
            with h5py.File(f'{self.filename}.hdf5', 'w') as file:
                file.create_dataset('wavefunction/psi_plus', (wfn.grid.num_points_x, 1),
                                    maxshape=(wfn.grid.num_points_x, None), dtype='complex128')
                file.create_dataset('wavefunction/psi_0', (wfn.grid.num_points_x, 1),
                                    maxshape=(wfn.grid.num_points_x, None), dtype='complex128')
                file.create_dataset('wavefunction/psi_minus', (wfn.grid.num_points_x, 1),
                                    maxshape=(wfn.grid.num_points_x, None), dtype='complex128')
        elif wfn.grid.ndim == 2:
            with h5py.File(f'{self.filename}.hdf5', 'w') as file:
                file.create_dataset('wavefunction/psi_plus', (wfn.grid.num_points_x, 1),
                                    maxshape=(wfn.grid.num_points_x, wfn.grid.num_points_y), dtype='complex128')
                file.create_dataset('wavefunction/psi_0', (wfn.grid.num_points_x, 1),
                                    maxshape=(wfn.grid.num_points_x, wfn.grid.num_points_y), dtype='complex128')
                file.create_dataset('wavefunction/psi_minus', (wfn.grid.num_points_x, 1),
                                    maxshape=(wfn.grid.num_points_x, wfn.grid.num_points_y), dtype='complex128')
        elif wfn.grid.ndim == 3:
            raise NotImplementedError

    def _save_params(self, params: dict):
        with h5py.File(f'{self.filename}.hdf5', 'r+') as file:
            file.create_dataset('parameters', data=params)
