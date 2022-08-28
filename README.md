<p align="center"><img src="docs/pygpe.png" alt="logo" ></p>

<h4 align="center">A fast and easy to use Gross-Pitaevskii equation solver.</h4>


## Description

PyGPE is a CUDA-accelerated Python library for solving the Gross-Pitaevskii equations for use in simulating Bose-Einstein condensate
systems accelerated using CUDA.

### Planned supported features
- Scalar, spin-1/2, spin-1, and spin-2 systems.
- 1D, 2D, and 3D grid lattices.
- Phase profile constructors that allow for vortices.
- HDF5 data saving system.
- A suite of diagnostics functions for calculating/using useful quantities.

### Requirements

- Python (3.9+),
- [h5py](https://github.com/h5py/h5py) (3.6.0+),
- [CuPy](https://github.com/cupy/cupy) (10.2.0+).

## Examples

See [examples](examples) folder for various examples on the usage of the library.
Below is an animation of superfluid turbulence in a scalar BEC simulated using PyGPE taking **~60 seconds** on an RTX 2060.

<p align="center"><img src="docs/animation.gif" alt="logo" > </p>
