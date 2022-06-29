import pygpe.spin_1 as gpe
import pygpe.shared.vortices as vort

# Generate grid object
points = (512, 512)
grid_spacings = (0.5, 0.5)
grid = gpe.Grid(points, grid_spacings)

# Condensate parameters
params = {
    "c0": 10,
    "c2": 0.5,
    "p": 0.,
    "q": 0.,
    "trap": 0.,

    # Time params
    "dt": -1j * 1e-2,
    "nt": 100,
    "t": 0
}

# Generate wavefunction object, set initial state and add noise
psi = gpe.Wavefunction(grid)
psi.set_ground_state("polar")
psi.add_noise_to_components("outer", 0., 1e-2)

vort.add_singly_quantised_vortices(psi, 100, 1)  # Add SQVs to phase profile of wavefunction

# Generate DataManager to store data for simulation
data = gpe.DataManager(filename='test.hdf5', data_path='../data/')
data.save_initial_parameters(grid, psi, params)

psi.fft()  # Ensures k-space wavefunction components are up-to-date before evolution
for i in range(params["nt"]):
    # Perform the evolution
    gpe.kinetic_zeeman_step(psi, params)
    psi.ifft()
    gpe.interaction_step(psi, params)
    psi.fft()
    gpe.kinetic_zeeman_step(psi, params)
    gpe.renormalise_wavefunction(psi)

    data.save_wfn(psi)  # Save data
