import h5py

from pygpe.shared.data_manager import _DataManager
from pygpe.shared.utils import handle_array
from pygpe.shared import data_manager_paths as dmp
from pygpe.spinone.wavefunction import SpinOneWavefunction


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
        wfn: SpinOneWavefunction,
        params: dict,
    ):
        """Constructs the DataManager object."""
        super().__init__(filename, data_path, wfn, params)
        self._save_initial_wfn(wfn)

    def _save_initial_wfn(self, wfn: SpinOneWavefunction) -> None:
        """Creates new datasets in file for the wavefunction and saves
        initial values.
        """
        with h5py.File(self.data_path_and_file, "r+") as file:
            if wfn.grid.ndim == 1:
                file.create_dataset(
                    dmp.SPIN1_WAVEFUNCTION_PLUS,
                    (wfn.grid.shape, 1),
                    maxshape=(wfn.grid.shape, None),
                    dtype="complex128",
                )
                file.create_dataset(
                    dmp.SPIN1_WAVEFUNCTION_ZERO,
                    (wfn.grid.shape, 1),
                    maxshape=(wfn.grid.shape, None),
                    dtype="complex128",
                )
                file.create_dataset(
                    dmp.SPIN1_WAVEFUNCTION_MINUS,
                    (wfn.grid.shape, 1),
                    maxshape=(wfn.grid.shape, None),
                    dtype="complex128",
                )
            else:
                file.create_dataset(
                    dmp.SPIN1_WAVEFUNCTION_PLUS,
                    (*wfn.grid.shape, 1),
                    maxshape=(*wfn.grid.shape, None),
                    dtype="complex128",
                )
                file.create_dataset(
                    dmp.SPIN1_WAVEFUNCTION_ZERO,
                    (*wfn.grid.shape, 1),
                    maxshape=(*wfn.grid.shape, None),
                    dtype="complex128",
                )
                file.create_dataset(
                    dmp.SPIN1_WAVEFUNCTION_MINUS,
                    (*wfn.grid.shape, 1),
                    maxshape=(*wfn.grid.shape, None),
                    dtype="complex128",
                )

    def save_wavefunction(self, wfn: SpinOneWavefunction) -> None:
        """Saves the current wavefunction data to the dataset.

        :param wfn: The wavefunction of the system.
        :type wfn: :class:`Wavefunction`
        """
        wfn.ifft()  # Update real-space wavefunction before saving
        with h5py.File(self.data_path_and_file, "r+") as data:
            if wfn.grid.ndim == 1:
                new_psi_plus = data[dmp.SPIN1_WAVEFUNCTION_PLUS]
                new_psi_plus.resize((wfn.grid.num_points_x, self._time_index + 1))
                new_psi_plus[:, self._time_index] = handle_array(wfn.plus_component)

                new_psi_zero = data[dmp.SPIN1_WAVEFUNCTION_ZERO]
                new_psi_zero.resize((wfn.grid.num_points_x, self._time_index + 1))
                new_psi_zero[:, self._time_index] = handle_array(wfn.zero_component)

                new_psi_minus = data[dmp.SPIN1_WAVEFUNCTION_MINUS]
                new_psi_minus.resize((wfn.grid.num_points_x, self._time_index + 1))
                new_psi_minus[:, self._time_index] = handle_array(wfn.minus_component)
            else:
                new_psi_plus = data[dmp.SPIN1_WAVEFUNCTION_PLUS]
                new_psi_plus.resize((*wfn.grid.shape, self._time_index + 1))
                new_psi_plus[..., self._time_index] = handle_array(wfn.plus_component)

                new_psi_zero = data[dmp.SPIN1_WAVEFUNCTION_ZERO]
                new_psi_zero.resize((*wfn.grid.shape, self._time_index + 1))
                new_psi_zero[..., self._time_index] = handle_array(wfn.zero_component)

                new_psi_minus = data[dmp.SPIN1_WAVEFUNCTION_MINUS]
                new_psi_minus.resize((*wfn.grid.shape, self._time_index + 1))
                new_psi_minus[..., self._time_index] = handle_array(wfn.minus_component)

        self._time_index += 1
