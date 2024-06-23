import time
import os

try:
    import cupy as cp  # type: ignore
except ImportError:
    import numpy as cp
import matplotlib.pyplot as plt
import imageio

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

# Generate phase with a dipole pair seperated 2.0 unuts apart
phase = add_dipole_pair(grid, 2.0)
psi.apply_phase(phase)  # Apply phase to wavefunction

# Define condensate parameters (small disspation added)
params = {"g": 1, "trap": 0, "nt": 10000, "dt": 1e-2, "t": 0, "gamma": 0.01}

# Create DataManager
data = DataManager("scalar_data.hdf5", "data", psi, params)

psi.fft()  # FFT to ensure k-space wavefunction is up-to-date
start_time = time.time()  # Start timer

# Prepare directory for frames
frames_dir = "frames"
os.makedirs(frames_dir, exist_ok=True)

frame_files = []
for i in range(params["nt"]):
    # Evolve wavefunction
    step_wavefunction(psi, params)

    if i % 100 == 0:  # Save wavefunction data and create a frame
        frame_path = f"{frames_dir}/frame_{i:04d}.png"
        frame_files.append(frame_path)
        plt.figure(figsize=(6, 6))
        plt.imshow(psi.density(), vmin=0, vmax=1)
        plt.colorbar()
        plt.title(f't = {params["t"]:.2f}')
        plt.savefig(frame_path)
        plt.close()

    params["t"] += params["dt"]  # Increment time count

print(f'Evolution of {params["nt"]} steps took {time.time() - start_time} seconds!')

# Create movie using imageio
with imageio.get_writer("wavefunction_evolution.mp4", fps=10) as writer:
    for filename in frame_files:
        image = imageio.imread(filename)
        writer.append_data(image)

# Clean up frames
import shutil

shutil.rmtree(frames_dir)

# Show the last frame
plt.imshow(cp.asnumpy(psi.density()), vmin=0, vmax=1)
plt.colorbar()
plt.show()
