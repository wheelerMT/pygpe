import h5py
import cupy as cp
from pygpe.shared import data_manager_paths as dmp
from pygpe.shared.data_manager import _DataManager
from pygpe.spinhalf.wavefunction import SpinHalfWavefunction


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
        wfn: SpinHalfWavefunction,
        params: dict,
    ):
        """Constructs the DataManager object."""
        super().__init__(filename, data_path, wfn, params)
        self._save_initial_wfn(wfn)

    def _save_initial_wfn(self, wfn: SpinHalfWavefunction) -> None:
        """Creates new datasets in file for the wavefunction and saves
        initial values.
        """
        with h5py.File(f"{self.data_path}/{self.filename}", "r+") as file:
            if wfn.grid.ndim == 1:
                file.create_dataset(
                    dmp.SPINHALF_WAVEFUNCTION_PLUS,
                    (wfn.grid.shape, 1),
                    maxshape=(wfn.grid.shape, None),
                    dtype="complex128",
                )
                file.create_dataset(
                    dmp.SPINHALF_WAVEFUNCTION_MINUS,
                    (wfn.grid.shape, 1),
                    maxshape=(wfn.grid.shape, None),
                    dtype="complex128",
                )
            else:
                file.create_dataset(
                    dmp.SPINHALF_WAVEFUNCTION_PLUS,
                    (*wfn.grid.shape, 1),
                    maxshape=(*wfn.grid.shape, None),
                    dtype="complex128",
                )
                file.create_dataset(
                    dmp.SPINHALF_WAVEFUNCTION_MINUS,
                    (*wfn.grid.shape, 1),
                    maxshape=(*wfn.grid.shape, None),
                    dtype="complex128",
                )

    def save_wavefunction(self, wfn: SpinHalfWavefunction) -> None:
        """Saves the current wavefunction data to the dataset.

        :param wfn: The wavefunction of the system.
        :type wfn: :class:`Wavefunction`
        """
        wfn.ifft()  # Update real-space wavefunction before saving
        with h5py.File(f"{self.data_path}/{self.filename}", "r+") as data:
            if wfn.grid.ndim == 1:
                new_psi_plus = data[dmp.SPINHALF_WAVEFUNCTION_PLUS]
                new_psi_plus.resize(
                    (wfn.grid.num_points_x, self._time_index + 1)
                )
                new_psi_plus[:, self._time_index] = cp.asnumpy(
                    wfn.plus_component
                )

                new_psi_minus = data[dmp.SPINHALF_WAVEFUNCTION_MINUS]
                new_psi_minus.resize(
                    (wfn.grid.num_points_x, self._time_index + 1)
                )
                new_psi_minus[:, self._time_index] = cp.asnumpy(
                    wfn.minus_component
                )
            else:
                new_psi_plus = data[dmp.SPINHALF_WAVEFUNCTION_PLUS]
                new_psi_plus.resize((*wfn.grid.shape, self._time_index + 1))
                new_psi_plus[..., self._time_index] = cp.asnumpy(
                    wfn.plus_component
                )

                new_psi_minus = data[dmp.SPINHALF_WAVEFUNCTION_MINUS]
                new_psi_minus.resize((*wfn.grid.shape, self._time_index + 1))
                new_psi_minus[..., self._time_index] = cp.asnumpy(
                    wfn.minus_component
                )

        self._time_index += 1
