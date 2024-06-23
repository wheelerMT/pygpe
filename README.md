<p align="center"><img src="docs/pygpe.png" alt="logo" ></p>

<h4 align="center">A fast and easy to use Gross-Pitaevskii equation solver.</h4>

## Description

PyGPE is a CUDA-accelerated Python library for solving the Gross-Pitaevskii equations for use in simulating
Bose-Einstein condensate systems.

- Documentation: https://wheelermt.github.io/pygpe-docs/

### Supported features

- Scalar, two-component, spin-1, and spin-2 BEC systems.
- 1D, 2D, and 3D grid lattices.
- GPU support.
- HDF5 data saving system.
- Method for generating vortices within the system.

### Requirements

- Python (3.10 and above),
- [h5py](https://github.com/h5py/h5py) (^3.6.0),
- [numpy](https://numpy.org/) (^2.0.0),
- Matplotlib (^3.8.2)

If using a GPU:
  - CUDA Toolkit (>=11.2)
  - [CuPy](https://github.com/cupy/cupy) (>=10.2.0).

## Installation

The simplest way to begin using PyGPE is through pip:

    pip install pygpe

By default, PyGPE will use the CPU to perform calculations.
However, if a CUDA-capable GPU is detected, PyGPE will automatically utilise it for drastic
speed-ups in computation time.

## Examples

See [examples](examples) folder for various examples on the usage of the library.
Below is an animation of superfluid turbulence in a scalar BEC simulated using PyGPE on a $512^2$ lattice
for $N_t=200000$ time steps taking **~5 minutes** to complete on an RTX 2060.

<p align="center"><img src="docs/animation.gif" alt="logo" > </p>
