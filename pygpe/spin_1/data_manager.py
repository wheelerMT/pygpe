import h5py
from pygpe.shared.grid import Grid
from pygpe.spin_1.wavefunction import Wavefunction


class DataManager:
    def __init__(self, filename: str, grid: Grid, wfn: Wavefunction):
        self.filename = filename
        self.data_path = f'data/{self.filename}'

        self._save_initial_grid_params(grid)
        self._save_initial_wfn(wfn)

    def _save_initial_grid_params(self, grid: Grid) -> None:
        pass

    def _save_initial_wfn(self, wfn: Wavefunction) -> None:
        pass
   