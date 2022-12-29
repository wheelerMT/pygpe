import h5py
import cupy as cp
from pygpe.shared.grid import Grid
from pygpe.shared import data_manager_paths as dmp
from pygpe.scalar.wavefunction import Wavefunction


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
        """Constructs the DataManager object."""
        self.filename = filename
        self.data_path = data_path
        self._time_index = 0

        # Create file
        h5py.File(f"{self.data_path}/{self.filename}", "w")

    def save_initial_parameters(
        self, grid: Grid, wfn: Wavefunction, parameters: dict
    ) -> None:
        """Saves the initial grid, wavefunction and parameters to a HDF5 file.

        :param grid: The grid object of the system.
        :type grid: :class:`Grid`
        :param wfn: The wavefunction of the system.
        :type wfn: :class:`Wavefunction`
        :param parameters: The parameter dictionary.
        :type parameters: dict
        """
        self._save_grid_params(grid)
        self._save_initial_wfn(wfn)
        self._save_params(parameters)

    def _save_grid_params(self, grid: Grid) -> None:
        """Saves grid parameters to dataset."""
        if grid.ndim == 1:
            with h5py.File(f"{self.data_path}/{self.filename}", "r+") as file:
                file.create_dataset(dmp.GRID_NX, data=grid.num_points_x)
                file.create_dataset(dmp.GRID_DX, data=grid.grid_spacing_x)
        elif grid.ndim == 2:
            with h5py.File(f"{self.data_path}/{self.filename}", "r+") as file:
                file.create_dataset(dmp.GRID_NX, data=grid.num_points_x)
                file.create_dataset(dmp.GRID_NY, data=grid.num_points_y)
                file.create_dataset(dmp.GRID_DX, data=grid.grid_spacing_x)
                file.create_dataset(dmp.GRID_DY, data=grid.grid_spacing_y)
        elif grid.ndim == 3:
            with h5py.File(f"{self.data_path}/{self.filename}", "r+") as file:
                file.create_dataset(dmp.GRID_NX, data=grid.num_points_x)
                file.create_dataset(dmp.GRID_NY, data=grid.num_points_y)
                file.create_dataset(dmp.GRID_NZ, data=grid.num_points_z)
                file.create_dataset(dmp.GRID_DX, data=grid.grid_spacing_x)
                file.create_dataset(dmp.GRID_DY, data=grid.grid_spacing_y)
                file.create_dataset(dmp.GRID_DZ, data=grid.grid_spacing_z)

    def _save_initial_wfn(self, wfn: Wavefunction) -> None:
        """Saves initial wavefunction to dataset."""
        if wfn.grid.ndim == 1:
            with h5py.File(f"{self.data_path}/{self.filename}", "r+") as file:
                file.create_dataset(
                    dmp.SCALAR_WAVEFUNCTION,
                    (wfn.grid.shape, 1),
                    maxshape=(wfn.grid.shape, None),
                    dtype="complex128",
                )
        else:
            with h5py.File(f"{self.data_path}/{self.filename}", "r+") as file:
                file.create_dataset(
                    dmp.SCALAR_WAVEFUNCTION,
                    (*wfn.grid.shape, 1),
                    maxshape=(*wfn.grid.shape, None),
                    dtype="complex128",
                )

    def _save_params(self, parameters: dict) -> None:
        """Saves condensate parameters to dataset."""
        with h5py.File(f"{self.data_path}/{self.filename}", "r+") as file:
            for key in parameters:
                file.create_dataset(f"{dmp.PARAMETERS}/{key}", data=parameters[key])

    def save_wavefunction(self, wfn: Wavefunction) -> None:
        """Saves the current wavefunction data to the dataset.

        :param wfn: The wavefunction of the system.
        :type wfn: :class:`Wavefunction`
        """
        wfn.ifft()  # Update real-space wavefunction before saving
        if wfn.grid.ndim == 1:
            with h5py.File(f"{self.data_path}/{self.filename}", "r+") as data:
                new_psi = data[dmp.SCALAR_WAVEFUNCTION]
                new_psi.resize((wfn.grid.num_points_x, self._time_index + 1))
                new_psi[:, self._time_index] = cp.asnumpy(wfn.component)
        else:
            with h5py.File(f"{self.data_path}/{self.filename}", "r+") as data:
                new_psi = data[dmp.SCALAR_WAVEFUNCTION]
                new_psi.resize((*wfn.grid.shape, self._time_index + 1))
                new_psi[..., self._time_index] = cp.asnumpy(wfn.component)

        self._time_index += 1
