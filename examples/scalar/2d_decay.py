import time

try:
    import cupy as cp  # type: ignore
except ImportError:
    import numpy as cp
import matplotlib.pyplot as plt
from pygpe.shared.utils import handle_array

from pygpe.scalar import Grid, ScalarWavefunction, DataManager, step_wavefunction
from pygpe.shared.vortices import add_dipole_pair

# Generate grid
points = (128, 128)
grid_spacings = (0.5, 0.5)
grid = Grid(points, grid_spacings)

# Set up initial wavefunction with uniform density
psi = ScalarWavefunction(grid)
psi.set_wavefunction(cp.ones(grid.shape, dtype="complex128"))
psi.add_noise(mean=0.0, std_dev=1e-2)  # Add noise to wavefunction

# Generate phase with a dipole pair seperated 2.0 units apart
phase = add_dipole_pair(grid, 2.0)
psi.apply_phase(phase)  # Apply phase to wavefunction

# Define condensate parameters (small dissipation added)
params = {"g": 1, "trap": 0, "nt": 10000, "dt": 1e-2, "t": 0, "gamma": 0.01}

# Create DataManager
data = DataManager("scalar_data.hdf5", "data", psi, params)

psi.fft()  # FFT to ensure k-space wavefunction is up-to-date
start_time = time.time()  # Start timer
for i in range(params["nt"]):
    # Evolve wavefunction
    step_wavefunction(psi, params)

    if i % 10 == 0:  # Save wavefunction data and print current time
        data.save_wavefunction(psi)
        print(f't = {params["t"]}')

    params["t"] += params["dt"]  # Increment time count

print(f'Evolution of {params["nt"]} steps took {time.time() - start_time} seconds!')

# Show the last frame
plt.imshow(handle_array(psi.density()), vmin=0, vmax=1)
plt.show()
