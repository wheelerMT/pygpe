import h5py

from pygpe.shared import data_manager_paths as dmp
from pygpe.shared.utils import handle_array
from pygpe.shared.data_manager import _DataManager
from pygpe.scalar.wavefunction import ScalarWavefunction


class DataManager(_DataManager):
    """This object handles all the data of the simulation, including the
    wavefunction, grid, and parameter data.

    :param filename: The name of the data file.
    :type filename: str
    :param data_path: The relative path to the folder containing the data file.
    :type data_path: str

    :ivar filename: The name of the data file.
    :ivar data_path: The relative path to the folder containing the data file.
    """

    def __init__(
        self,
        filename: str,
        data_path: str,
        wfn: ScalarWavefunction,
        params: dict,
    ):
        """Constructs the DataManager object."""
        super().__init__(filename, data_path, wfn, params)
        self._save_initial_wfn(wfn)

    def _save_initial_wfn(self, wfn: ScalarWavefunction) -> None:
        """Saves initial wavefunction to dataset."""
        with h5py.File(self.data_path_and_file, "r+") as file:
            if wfn.grid.ndim == 1:
                file.create_dataset(
                    dmp.SCALAR_WAVEFUNCTION,
                    (wfn.grid.shape, 1),
                    maxshape=(wfn.grid.shape, None),
                    dtype="complex128",
                )
            else:
                file.create_dataset(
                    dmp.SCALAR_WAVEFUNCTION,
                    (*wfn.grid.shape, 1),
                    maxshape=(*wfn.grid.shape, None),
                    dtype="complex128",
                )

    def save_wavefunction(self, wfn: ScalarWavefunction) -> None:
        """Saves the current wavefunction data to the dataset.

        :param wfn: The wavefunction of the system.
        :type wfn: :class:`Wavefunction`
        """
        wfn.ifft()  # Update real-space wavefunction before saving
        with h5py.File(self.data_path_and_file, "r+") as data:
            if wfn.grid.ndim == 1:
                new_psi = data[dmp.SCALAR_WAVEFUNCTION]
                new_psi.resize((wfn.grid.num_points_x, self._time_index + 1))
                new_psi[:, self._time_index] = handle_array(wfn.component)
            else:
                new_psi = data[dmp.SCALAR_WAVEFUNCTION]
                new_psi.resize((*wfn.grid.shape, self._time_index + 1))
                new_psi[..., self._time_index] = handle_array(wfn.component)

        self._time_index += 1
