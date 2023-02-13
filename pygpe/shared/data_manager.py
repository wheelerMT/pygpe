import h5py
import cupy as cp
import pygpe.shared.data_manager_paths as dmp
from pygpe.shared.grid import Grid
from pygpe.shared.wavefunction import _Wavefunction
from pygpe.scalar.wavefunction import ScalarWavefunction
from pygpe.spinhalf.wavefunction import SpinHalfWavefunction
from pygpe.spinone.wavefunction import SpinOneWavefunction
from pygpe.spintwo.wavefunction import SpinTwoWavefunction


class DataManager:
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
        wfn: _Wavefunction,
        parameters: dict,
    ) -> None:
        """Constructs the DataManager object."""

        self.filename = filename
        self.data_path = data_path
        self._time_index = 0

        self._save_grid_params(wfn.grid)
        self._save_params(parameters)
        match wfn:
            case ScalarWavefunction():
                self._save_initial_scalar_wavefunction(wfn)
            case SpinHalfWavefunction():
                self._save_initial_spin_half_wavefunction(wfn)
            case SpinOneWavefunction():
                self._save_initial_spin_one_wavefunction(wfn)
            case SpinTwoWavefunction():
                self._save_initial_spin_two_wavefunction(wfn)

    def _save_grid_params(self, grid: Grid) -> None:
        """Saves grid parameters to dataset."""
        with h5py.File(f"{self.data_path}/{self.filename}", "r+") as file:
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
        """Creates new datasets in file for condensate & time parameters
        and saves initial values.
        """
        with h5py.File(f"{self.data_path}/{self.filename}", "r+") as file:
            for key in parameters:
                file.create_dataset(f"parameters/{key}", data=parameters[key])

    def _save_initial_scalar_wavefunction(
        self, wfn: ScalarWavefunction
    ) -> None:
        """Creates new datasets in file for a scalar wavefunction and saves
        initial values.
        """
        with h5py.File(f"{self.data_path}/{self.filename}", "r+") as file:
            _create_dataset(file, dmp.SCALAR_WAVEFUNCTION, wfn)

    def _save_initial_spin_half_wavefunction(
        self, wfn: SpinHalfWavefunction
    ) -> None:
        """Creates new datasets in file for a spin-half wavefunction and saves
        initial values.
        """
        with h5py.File(f"{self.data_path}/{self.filename}", "r+") as file:
            _create_dataset(file, dmp.SPINHALF_WAVEFUNCTION_PLUS, wfn)
            _create_dataset(file, dmp.SPINHALF_WAVEFUNCTION_MINUS, wfn)

    def _save_initial_spin_one_wavefunction(
        self, wfn: SpinOneWavefunction
    ) -> None:
        """Creates new datasets in file for a spin-1 wavefunction and saves
        initial values.
        """
        with h5py.File(f"{self.data_path}/{self.filename}", "r+") as file:
            _create_dataset(file, dmp.SPIN1_WAVEFUNCTION_PLUS, wfn)
            _create_dataset(file, dmp.SPIN1_WAVEFUNCTION_ZERO, wfn)
            _create_dataset(file, dmp.SPIN1_WAVEFUNCTION_MINUS, wfn)

    def _save_initial_spin_two_wavefunction(
        self, wfn: SpinTwoWavefunction
    ) -> None:
        """Creates new datasets in file for a spin-1 wavefunction and saves
        initial values.
        """
        with h5py.File(f"{self.data_path}/{self.filename}", "r+") as file:
            _create_dataset(file, dmp.SPIN2_WAVEFUNCTION_PLUS_TWO, wfn)
            _create_dataset(file, dmp.SPIN2_WAVEFUNCTION_PLUS_ONE, wfn)
            _create_dataset(file, dmp.SPIN2_WAVEFUNCTION_ZERO, wfn)
            _create_dataset(file, dmp.SPIN2_WAVEFUNCTION_MINUS_ONE, wfn)
            _create_dataset(file, dmp.SPIN2_WAVEFUNCTION_MINUS_TWO, wfn)

    def save_wavefunction(self, wfn: _Wavefunction) -> None:
        """Saves the current wavefunction data to the dataset.

        :param wfn: The wavefunction of the system.
        :type wfn: :class:`_Wavefunction`
        """
        match wfn:
            case ScalarWavefunction():
                self._save_scalar_wavefunction(wfn)
            case SpinHalfWavefunction():
                self._save_spin_half_wavefunction(wfn)
            case SpinOneWavefunction():
                self._save_spin_one_wavefunction(wfn)
            case SpinTwoWavefunction():
                self._save_spin_two_wavefunction(wfn)

    def _save_scalar_wavefunction(self, wfn: ScalarWavefunction) -> None:
        wfn.ifft()  # Update real-space wavefunction before saving
        with h5py.File(f"{self.data_path}/{self.filename}", "r+") as data:
            if wfn.grid.ndim == 1:
                new_psi = data[dmp.SCALAR_WAVEFUNCTION]
                new_psi.resize((wfn.grid.num_points_x, self._time_index + 1))
                new_psi[:, self._time_index] = cp.asnumpy(wfn.component)
            else:
                new_psi = data[dmp.SCALAR_WAVEFUNCTION]
                new_psi.resize((*wfn.grid.shape, self._time_index + 1))
                new_psi[..., self._time_index] = cp.asnumpy(wfn.component)
        self._time_index += 1

    def _save_spin_half_wavefunction(self, wfn: SpinHalfWavefunction) -> None:
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

    def _save_spin_one_wavefunction(self, wfn: SpinOneWavefunction) -> None:
        wfn.ifft()  # Update real-space wavefunction before saving
        with h5py.File(f"{self.data_path}/{self.filename}", "r+") as data:
            if wfn.grid.ndim == 1:
                new_psi_plus = data[dmp.SPIN1_WAVEFUNCTION_PLUS]
                new_psi_plus.resize(
                    (wfn.grid.num_points_x, self._time_index + 1)
                )
                new_psi_plus[:, self._time_index] = cp.asnumpy(
                    wfn.plus_component
                )

                new_psi_zero = data[dmp.SPIN1_WAVEFUNCTION_ZERO]
                new_psi_zero.resize(
                    (wfn.grid.num_points_x, self._time_index + 1)
                )
                new_psi_zero[:, self._time_index] = cp.asnumpy(
                    wfn.zero_component
                )

                new_psi_minus = data[dmp.SPIN1_WAVEFUNCTION_MINUS]
                new_psi_minus.resize(
                    (wfn.grid.num_points_x, self._time_index + 1)
                )
                new_psi_minus[:, self._time_index] = cp.asnumpy(
                    wfn.minus_component
                )
            else:
                new_psi_plus = data[dmp.SPIN1_WAVEFUNCTION_PLUS]
                new_psi_plus.resize((*wfn.grid.shape, self._time_index + 1))
                new_psi_plus[..., self._time_index] = cp.asnumpy(
                    wfn.plus_component
                )

                new_psi_zero = data[dmp.SPIN1_WAVEFUNCTION_ZERO]
                new_psi_zero.resize((*wfn.grid.shape, self._time_index + 1))
                new_psi_zero[..., self._time_index] = cp.asnumpy(
                    wfn.zero_component
                )

                new_psi_minus = data[dmp.SPIN1_WAVEFUNCTION_MINUS]
                new_psi_minus.resize((*wfn.grid.shape, self._time_index + 1))
                new_psi_minus[..., self._time_index] = cp.asnumpy(
                    wfn.minus_component
                )
        self._time_index += 1

    def _save_spin_two_wavefunction(self, wfn: SpinTwoWavefunction) -> None:
        wfn.ifft()  # Update real-space wavefunction before saving
        with h5py.File(f"{self.data_path}/{self.filename}", "r+") as data:
            if wfn.grid.ndim == 1:
                new_psi_plus2 = data[dmp.SPIN2_WAVEFUNCTION_PLUS_TWO]
                new_psi_plus2.resize(
                    (wfn.grid.num_points_x, self._time_index + 1)
                )
                new_psi_plus2[:, self._time_index] = cp.asnumpy(
                    wfn.plus2_component
                )

                new_psi_plus1 = data[dmp.SPIN2_WAVEFUNCTION_PLUS_ONE]
                new_psi_plus1.resize(
                    (wfn.grid.num_points_x, self._time_index + 1)
                )
                new_psi_plus1[:, self._time_index] = cp.asnumpy(
                    wfn.plus1_component
                )

                new_psi_zero = data[dmp.SPIN2_WAVEFUNCTION_ZERO]
                new_psi_zero.resize(
                    (wfn.grid.num_points_x, self._time_index + 1)
                )
                new_psi_zero[:, self._time_index] = cp.asnumpy(
                    wfn.zero_component
                )

                new_psi_minus1 = data[dmp.SPIN2_WAVEFUNCTION_MINUS_ONE]
                new_psi_minus1.resize(
                    (wfn.grid.num_points_x, self._time_index + 1)
                )
                new_psi_minus1[:, self._time_index] = cp.asnumpy(
                    wfn.minus1_component
                )

                new_psi_minus2 = data[dmp.SPIN2_WAVEFUNCTION_MINUS_TWO]
                new_psi_minus2.resize(
                    (wfn.grid.num_points_x, self._time_index + 1)
                )
                new_psi_minus2[:, self._time_index] = cp.asnumpy(
                    wfn.minus2_component
                )
            else:
                new_psi_plus2 = data[dmp.SPIN2_WAVEFUNCTION_PLUS_TWO]
                new_psi_plus2.resize((*wfn.grid.shape, self._time_index + 1))
                new_psi_plus2[..., self._time_index] = cp.asnumpy(
                    wfn.plus2_component
                )

                new_psi_plus1 = data[dmp.SPIN2_WAVEFUNCTION_PLUS_ONE]
                new_psi_plus1.resize((*wfn.grid.shape, self._time_index + 1))
                new_psi_plus1[..., self._time_index] = cp.asnumpy(
                    wfn.plus1_component
                )

                new_psi_zero = data[dmp.SPIN2_WAVEFUNCTION_ZERO]
                new_psi_zero.resize((*wfn.grid.shape, self._time_index + 1))
                new_psi_zero[..., self._time_index] = cp.asnumpy(
                    wfn.zero_component
                )

                new_psi_minus1 = data[dmp.SPIN2_WAVEFUNCTION_MINUS_ONE]
                new_psi_minus1.resize((*wfn.grid.shape, self._time_index + 1))
                new_psi_minus1[..., self._time_index] = cp.asnumpy(
                    wfn.minus1_component
                )

                new_psi_minus2 = data[dmp.SPIN2_WAVEFUNCTION_MINUS_TWO]
                new_psi_minus2.resize((*wfn.grid.shape, self._time_index + 1))
                new_psi_minus2[..., self._time_index] = cp.asnumpy(
                    wfn.minus2_component
                )

        self._time_index += 1


def _create_dataset(
    file: h5py.File, dataset_path: str, wfn: _Wavefunction
) -> None:
    if wfn.grid.ndim == 1:
        file.create_dataset(
            dataset_path,
            (wfn.grid.shape, 1),
            maxshape=(wfn.grid.shape, None),
            dtype="complex128",
        )
    else:
        file.create_dataset(
            dataset_path,
            (*wfn.grid.shape, 1),
            maxshape=(wfn.grid.shape, None),
            dtype="complex128",
        )
