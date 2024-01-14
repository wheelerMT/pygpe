from abc import ABC, abstractmethod
from pathlib import Path

import h5py

import pygpe.shared.data_manager_paths as dmp
from pygpe.shared.grid import Grid
from pygpe.shared.wavefunction import _Wavefunction


class _DataManager(ABC):
    """Defines the abstract DataManager base class.
    Each system's DataManager inherits from this class and provides overrides
    for the abstract methods.
    """

    def __init__(
        self, filename: str, data_path: str, wfn: _Wavefunction, params: dict
    ) -> None:
        """The default constructor for the abstract `DataManager` class, to be
        inherited by sucblasses of `DataManager`.

        :param filename: Filename.
        :type filename: str
        :param data_path: Path to file.
        :type data_path: str
        :param wfn: Wavefunction object.
        :type wfn: _Wavefunction
        :param params: Parametesr of the system.
        :type params: dict
        """
        self.filename = filename
        self.data_path = Path(f"./{data_path}")
        self.data_path_and_file = self.data_path / self.filename
        self._time_index = 0

        # Create file and save initial parameters
        h5py.File(self.data_path_and_file, "w")
        self._save_grid_params(wfn.grid)
        self._save_params(params)

    def _save_grid_params(self, grid: Grid) -> None:
        """Saves grid parameters to dataset."""
        with h5py.File(self.data_path_and_file, "r+") as file:
            if grid.ndim == 1:
                file.create_dataset(dmp.GRID_NX, data=grid.num_points_x)
                file.create_dataset(dmp.GRID_DX, data=grid.grid_spacing_x)
            elif grid.ndim == 2:
                file.create_dataset(dmp.GRID_NX, data=grid.num_points_x)
                file.create_dataset(dmp.GRID_NY, data=grid.num_points_y)
                file.create_dataset(dmp.GRID_DX, data=grid.grid_spacing_x)
                file.create_dataset(dmp.GRID_DY, data=grid.grid_spacing_y)
            elif grid.ndim == 3:
                file.create_dataset(dmp.GRID_NX, data=grid.num_points_x)
                file.create_dataset(dmp.GRID_NY, data=grid.num_points_y)
                file.create_dataset(dmp.GRID_NZ, data=grid.num_points_z)
                file.create_dataset(dmp.GRID_DX, data=grid.grid_spacing_x)
                file.create_dataset(dmp.GRID_DY, data=grid.grid_spacing_y)
                file.create_dataset(dmp.GRID_DZ, data=grid.grid_spacing_z)

    def _save_params(self, parameters: dict) -> None:
        """Saves condensate parameters to dataset."""
        with h5py.File(self.data_path / self.filename, "r+") as file:
            for key in parameters:
                file.create_dataset(f"{dmp.PARAMETERS}/{key}", data=parameters[key])

    @abstractmethod
    def _save_initial_wfn(self, wfn: _Wavefunction) -> None:
        """Saves initial wavefunction to dataset."""
        pass

    @abstractmethod
    def save_wavefunction(self, wfn: _Wavefunction) -> None:
        """Saves current wavefunction data to the dataset.

        :param wfn: Wavefunction of the system.
        :type wfn: _Wavefunction
        """
        pass
