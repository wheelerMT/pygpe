import time

import cupy as cp
import matplotlib.pyplot as plt

import pygpe.shared.vortices as vort
import pygpe.spinone as gpe

# Generate grid object
points = (512, 512)
grid_spacings = (0.5, 0.5)
grid = gpe.Grid(points, grid_spacings)

# Condensate parameters
params = {
    "c0": 10,
    "c2": 0.5,
    "p": 0.0,
    "q": 0.0,
    "trap": 0.0,
    "n0": 1,
    # Time params
    "dt": -1j * 1e-2,
    "nt": 1000,
    "t": 0,
}

# Generate wavefunction object, set initial state and add noise
psi = gpe.SpinOneWavefunction(grid)
psi.set_ground_state("polar", params)
psi.add_noise("outer", 0.0, 1e-2)

phase = cp.asarray(
    vort.vortex_phase_profile(grid, 100, 1)
)  # Get 100 phase windings with a min distance of 1
psi.apply_phase(phase)  # Apply phase to all spinor components

# Generate DataManager to store data for simulation
data = gpe.DataManager("spin_one_data.hdf5", "data", psi, params)

psi.fft()  # Ensures k-space wavefunction components are up-to-date before evolution
start_time = time.time()
for i in range(params["nt"]):
    # Perform the evolution
    gpe.step_wavefunction(psi, params)

    if i % 10 == 0:  # Save data every 10 time steps
        data.save_wavefunction(psi)
        print(params["t"])
    params["t"] += params["dt"]
print(f'Evolution of {params["nt"]} steps took {time.time() - start_time}!')

# Plot density and phase of zero component
fig, ax = plt.subplots(1, 2, figsize=(10, 6))
ax[0].pcolormesh(
    cp.asnumpy(grid.x_mesh),
    cp.asnumpy(grid.y_mesh),
    abs(cp.asnumpy(psi.zero_component)) ** 2,
)
ax[1].pcolormesh(
    cp.asnumpy(grid.x_mesh),
    cp.asnumpy(grid.y_mesh),
    cp.asnumpy(cp.angle(psi.zero_component)),
    cmap="jet",
)
plt.show()
