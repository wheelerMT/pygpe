import pygpe.scalar as gpe
import matplotlib.pyplot as plt
from pygpe.scalar.evolution import *
from pygpe.shared.vortices import vortex_phase_profile
import time

# Generate grid
points = (512, 512)
grid_spacings = (0.5, 0.5)
grid = gpe.Grid(points, grid_spacings)

# Set up initial wavefunction with uniform density
psi = gpe.Wavefunction(grid)
psi.set_wavefunction(cp.ones(grid.shape, dtype='complex128'))
psi.add_noise(mean=0., std_dev=1e-2)  # Add noise to wavefunction

# Generate phase that contains 100 phase windings spaced at least 1 spatial unit apart
phase = vortex_phase_profile(grid, 100, 1.)
psi.apply_phase(cp.asarray(phase))  # Apply phase to wavefunction

# Define condensate parameters
params = {"g": 1,
          "nt": 1000,
          "dt": -1j * 1e-2,
          "t": 0}

psi.fft()  # FFT to ensure k-space wavefunction is up-to-date
start_time = time.time()  # Start timer
for i in range(params["nt"]):
    # Evolve wavefunction
    kinetic_step(psi, params)
    psi.ifft()
    potential_step(psi, params)
    psi.fft()
    kinetic_step(psi, params)

    renormalise_wavefunction(psi)  # Re-normalise since we are using imaginary time

    params["t"] += params["dt"]  # Increment time count
print(f'Evolution of {params["nt"]} steps took {time.time() - start_time}!')

# Plot the density
plt.imshow(cp.asnumpy(psi.density()), vmin=0, vmax=1)
plt.show()
