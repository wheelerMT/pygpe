import h5py
import cupy as cp
from pygpe.shared.grid import Grid
from pygpe.spintwo.wavefunction import Wavefunction


class DataManager:
    """This object handles all the data of the simulation, including the wavefunction, grid, and parameter data.

    :param filename: The name of the data file.
    :type filename: str
    :param data_path: The relative path to the folder containing the data file.
    :type data_path: str

    :ivar filename: The name of the data file.
    :ivar data_path: The relative path to the folder containing the data file.
    """

    def __init__(self, filename: str, data_path: str):
        self.filename = filename
        self.data_path = data_path
        self._time_index = 0

        # Create file
        h5py.File(f'{self.data_path}/{self.filename}', 'w')

    def save_initial_parameters(self, grid: Grid, wfn: Wavefunction, parameters: dict) -> None:
        """Saves the initial grid, wavefunction and parameters to a HDF5 file.

        :param grid: The grid object of the system.
        :type grid: :class:`Grid`
        :param wfn: The wavefunction of the system.
        :type wfn: :class:`Wavefunction`
        :param parameters: The parameter dictionary.
        :type parameters: dict
        """

        self._save_initial_grid_params(grid)
        self._save_initial_wfn(wfn)
        self._save_params(parameters)

    def _save_initial_grid_params(self, grid: Grid) -> None:
        """Creates new datasets in file for grid-related parameters and saves
        initial values.
        """
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
        """Creates new datasets in file for the wavefunction."""
        if wfn.grid.ndim == 1:
            with h5py.File(f'{self.data_path}/{self.filename}', 'r+') as file:
                file.create_dataset('wavefunction/psi_plus2', (wfn.grid.shape, 1), maxshape=(wfn.grid.shape, None),
                                    dtype='complex128')
                file.create_dataset('wavefunction/psi_plus1', (wfn.grid.shape, 1), maxshape=(wfn.grid.shape, None),
                                    dtype='complex128')
                file.create_dataset('wavefunction/psi_zero', (wfn.grid.shape, 1), maxshape=(wfn.grid.shape, None),
                                    dtype='complex128')
                file.create_dataset('wavefunction/psi_minus1', (wfn.grid.shape, 1), maxshape=(wfn.grid.shape, None),
                                    dtype='complex128')
                file.create_dataset('wavefunction/psi_minus2', (wfn.grid.shape, 1), maxshape=(wfn.grid.shape, None),
                                    dtype='complex128')
        else:
            with h5py.File(f'{self.data_path}/{self.filename}', 'r+') as file:
                file.create_dataset('wavefunction/psi_plus2', (*wfn.grid.shape, 1), maxshape=(*wfn.grid.shape, None),
                                    dtype='complex128')
                file.create_dataset('wavefunction/psi_plus1', (*wfn.grid.shape, 1), maxshape=(*wfn.grid.shape, None),
                                    dtype='complex128')
                file.create_dataset('wavefunction/psi_zero', (*wfn.grid.shape, 1), maxshape=(*wfn.grid.shape, None),
                                    dtype='complex128')
                file.create_dataset('wavefunction/psi_minus1', (*wfn.grid.shape, 1), maxshape=(*wfn.grid.shape, None),
                                    dtype='complex128')
                file.create_dataset('wavefunction/psi_minus2', (*wfn.grid.shape, 1), maxshape=(*wfn.grid.shape, None),
                                    dtype='complex128')

    def _save_params(self, parameters: dict) -> None:
        """Creates new datasets in file for condensate & time parameters
        and saves initial values.
        """
        with h5py.File(f'{self.data_path}/{self.filename}', 'r+') as file:
            for key in parameters:
                file.create_dataset(f'parameters/{key}', data=parameters[key])

    def save_wavefunction(self, wfn: Wavefunction) -> None:
        """Saves the current wavefunction data to the dataset.

        :param wfn: The wavefunction of the system.
        :type wfn: :class:`Wavefunction`
        """
        wfn.ifft()  # Update real-space wavefunction before saving
        if wfn.grid.ndim == 1:
            with h5py.File(f'{self.data_path}/{self.filename}', 'r+') as data:
                new_psi_plus2 = data['wavefunction/psi_plus2']
                new_psi_plus2.resize((wfn.grid.num_points_x, self._time_index + 1))
                new_psi_plus2[:, self._time_index] = cp.asnumpy(wfn.plus2_component)

                new_psi_plus1 = data['wavefunction/psi_plus1']
                new_psi_plus1.resize((wfn.grid.num_points_x, self._time_index + 1))
                new_psi_plus1[:, self._time_index] = cp.asnumpy(wfn.plus1_component)

                new_psi_zero = data['wavefunction/psi_zero']
                new_psi_zero.resize((wfn.grid.num_points_x, self._time_index + 1))
                new_psi_zero[:, self._time_index] = cp.asnumpy(wfn.zero_component)

                new_psi_minus1 = data['wavefunction/psi_minus1']
                new_psi_minus1.resize((wfn.grid.num_points_x, self._time_index + 1))
                new_psi_minus1[:, self._time_index] = cp.asnumpy(wfn.minus1_component)

                new_psi_minus2 = data['wavefunction/psi_minus2']
                new_psi_minus2.resize((wfn.grid.num_points_x, self._time_index + 1))
                new_psi_minus2[:, self._time_index] = cp.asnumpy(wfn.minus2_component)
        else:
            with h5py.File(f'{self.data_path}/{self.filename}', 'r+') as data:
                new_psi_plus2 = data['wavefunction/psi_plus2']
                new_psi_plus2.resize((*wfn.grid.shape, self._time_index + 1))
                new_psi_plus2[..., self._time_index] = cp.asnumpy(wfn.plus2_component)

                new_psi_plus1 = data['wavefunction/psi_plus1']
                new_psi_plus1.resize((*wfn.grid.shape, self._time_index + 1))
                new_psi_plus1[..., self._time_index] = cp.asnumpy(wfn.plus1_component)

                new_psi_zero = data['wavefunction/psi_zero']
                new_psi_zero.resize((*wfn.grid.shape, self._time_index + 1))
                new_psi_zero[..., self._time_index] = cp.asnumpy(wfn.zero_component)

                new_psi_minus1 = data['wavefunction/psi_minus1']
                new_psi_minus1.resize((*wfn.grid.shape, self._time_index + 1))
                new_psi_minus1[..., self._time_index] = cp.asnumpy(wfn.minus1_component)

                new_psi_minus2 = data['wavefunction/psi_minus2']
                new_psi_minus2.resize((*wfn.grid.shape, self._time_index + 1))
                new_psi_minus2[..., self._time_index] = cp.asnumpy(wfn.minus2_component)

        self._time_index += 1
