<p align="center"><img src="docs/pygpe.png" alt="logo" ></p>

<h4 align="center">A fast and easy to use Gross-Pitaevskii equation solver.</h4>

## Description

PyGPE is a CUDA-accelerated Python library for solving the Gross-Pitaevskii equations for use in simulating
Bose-Einstein condensate systems.

- Documentation: https://wheelermt.github.io/pygpe-docs/

### Supported features

- Scalar, spin-1, and spin-2 BEC systems.
- 1D, 2D, and 3D grid lattices.
- HDF5 data saving system.
- Method for generating vortices within the system.

### Requirements

- Python (3.9 and above),
- [h5py](https://github.com/h5py/h5py) (>=3.6.0),
- CUDA Toolkit (>=11.2)
- [CuPy](https://github.com/cupy/cupy) (>=10.2.0).

### Installation

Installation is through pip:

    pip install pygpe

Requirements are installed automatically **except** CUDA Toolkit.
Ensure you have the required version of CUDA Toolkit (11.2>=) installed before attempting to install PyGPE.

## Examples

See [examples](examples) folder for various examples on the usage of the library.
Below is an animation of superfluid turbulence in a scalar BEC simulated using PyGPE on a $512^2$ lattice
for $N_t=200000$ time steps taking **~5 minutes** to complete on an RTX 2060.

<p align="center"><img src="docs/animation.gif" alt="logo" > </p>
