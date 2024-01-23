import h5py

from pygpe.shared.data_manager import _DataManager
from pygpe.shared.utils import handle_array
from pygpe.shared import data_manager_paths as dmp
from pygpe.spintwo.wavefunction import SpinTwoWavefunction


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
        wfn: SpinTwoWavefunction,
        params: dict,
    ):
        """Constructs the DataManager object."""
        super().__init__(filename, data_path, wfn, params)
        self._save_initial_wfn(wfn)

    def _save_initial_wfn(self, wfn: SpinTwoWavefunction) -> None:
        """Creates new datasets in file for the wavefunction."""
        with h5py.File(self.data_path_and_file, "r+") as file:
            if wfn.grid.ndim == 1:
                file.create_dataset(
                    dmp.SPIN2_WAVEFUNCTION_PLUS_TWO,
                    (wfn.grid.shape, 1),
                    maxshape=(wfn.grid.shape, None),
                    dtype="complex128",
                )
                file.create_dataset(
                    dmp.SPIN2_WAVEFUNCTION_PLUS_ONE,
                    (wfn.grid.shape, 1),
                    maxshape=(wfn.grid.shape, None),
                    dtype="complex128",
                )
                file.create_dataset(
                    dmp.SPIN2_WAVEFUNCTION_ZERO,
                    (wfn.grid.shape, 1),
                    maxshape=(wfn.grid.shape, None),
                    dtype="complex128",
                )
                file.create_dataset(
                    dmp.SPIN2_WAVEFUNCTION_MINUS_ONE,
                    (wfn.grid.shape, 1),
                    maxshape=(wfn.grid.shape, None),
                    dtype="complex128",
                )
                file.create_dataset(
                    dmp.SPIN2_WAVEFUNCTION_MINUS_TWO,
                    (wfn.grid.shape, 1),
                    maxshape=(wfn.grid.shape, None),
                    dtype="complex128",
                )
            else:
                file.create_dataset(
                    dmp.SPIN2_WAVEFUNCTION_PLUS_TWO,
                    (*wfn.grid.shape, 1),
                    maxshape=(*wfn.grid.shape, None),
                    dtype="complex128",
                )
                file.create_dataset(
                    dmp.SPIN2_WAVEFUNCTION_PLUS_ONE,
                    (*wfn.grid.shape, 1),
                    maxshape=(*wfn.grid.shape, None),
                    dtype="complex128",
                )
                file.create_dataset(
                    dmp.SPIN2_WAVEFUNCTION_ZERO,
                    (*wfn.grid.shape, 1),
                    maxshape=(*wfn.grid.shape, None),
                    dtype="complex128",
                )
                file.create_dataset(
                    dmp.SPIN2_WAVEFUNCTION_MINUS_ONE,
                    (*wfn.grid.shape, 1),
                    maxshape=(*wfn.grid.shape, None),
                    dtype="complex128",
                )
                file.create_dataset(
                    dmp.SPIN2_WAVEFUNCTION_MINUS_TWO,
                    (*wfn.grid.shape, 1),
                    maxshape=(*wfn.grid.shape, None),
                    dtype="complex128",
                )

    def save_wavefunction(self, wfn: SpinTwoWavefunction) -> None:
        """Saves the current wavefunction data to the dataset.

        :param wfn: The wavefunction of the system.
        :type wfn: :class:`Wavefunction`
        """
        wfn.ifft()  # Update real-space wavefunction before saving
        with h5py.File(self.data_path_and_file, "r+") as data:
            if wfn.grid.ndim == 1:
                new_psi_plus2 = data[dmp.SPIN2_WAVEFUNCTION_PLUS_TWO]
                new_psi_plus2.resize((wfn.grid.num_points_x, self._time_index + 1))
                new_psi_plus2[:, self._time_index] = handle_array(wfn.plus2_component)

                new_psi_plus1 = data[dmp.SPIN2_WAVEFUNCTION_PLUS_ONE]
                new_psi_plus1.resize((wfn.grid.num_points_x, self._time_index + 1))
                new_psi_plus1[:, self._time_index] = handle_array(wfn.plus1_component)

                new_psi_zero = data[dmp.SPIN2_WAVEFUNCTION_ZERO]
                new_psi_zero.resize((wfn.grid.num_points_x, self._time_index + 1))
                new_psi_zero[:, self._time_index] = handle_array(wfn.zero_component)

                new_psi_minus1 = data[dmp.SPIN2_WAVEFUNCTION_MINUS_ONE]
                new_psi_minus1.resize((wfn.grid.num_points_x, self._time_index + 1))
                new_psi_minus1[:, self._time_index] = handle_array(wfn.minus1_component)

                new_psi_minus2 = data[dmp.SPIN2_WAVEFUNCTION_MINUS_TWO]
                new_psi_minus2.resize((wfn.grid.num_points_x, self._time_index + 1))
                new_psi_minus2[:, self._time_index] = handle_array(wfn.minus2_component)
            else:
                new_psi_plus2 = data[dmp.SPIN2_WAVEFUNCTION_PLUS_TWO]
                new_psi_plus2.resize((*wfn.grid.shape, self._time_index + 1))
                new_psi_plus2[..., self._time_index] = handle_array(wfn.plus2_component)

                new_psi_plus1 = data[dmp.SPIN2_WAVEFUNCTION_PLUS_ONE]
                new_psi_plus1.resize((*wfn.grid.shape, self._time_index + 1))
                new_psi_plus1[..., self._time_index] = handle_array(wfn.plus1_component)

                new_psi_zero = data[dmp.SPIN2_WAVEFUNCTION_ZERO]
                new_psi_zero.resize((*wfn.grid.shape, self._time_index + 1))
                new_psi_zero[..., self._time_index] = handle_array(wfn.zero_component)

                new_psi_minus1 = data[dmp.SPIN2_WAVEFUNCTION_MINUS_ONE]
                new_psi_minus1.resize((*wfn.grid.shape, self._time_index + 1))
                new_psi_minus1[..., self._time_index] = handle_array(
                    wfn.minus1_component
                )

                new_psi_minus2 = data[dmp.SPIN2_WAVEFUNCTION_MINUS_TWO]
                new_psi_minus2.resize((*wfn.grid.shape, self._time_index + 1))
                new_psi_minus2[..., self._time_index] = handle_array(
                    wfn.minus2_component
                )

        self._time_index += 1
