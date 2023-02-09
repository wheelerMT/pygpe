"""
This file contains string constants that specify the paths of data stored in
the DataManager classes.
You can modify any of these constants if you wish to change the specified path
within the data HDF5 files.
"""

# Grid points
GRID_NX = "grid/nx"
GRID_NY = "grid/ny"
GRID_NZ = "grid/nz"

# Grid spacings
GRID_DX = "grid/dx"
GRID_DY = "grid/dy"
GRID_DZ = "grid/dz"

# Parameters root
PARAMETERS = "parameters"

# Scalar wavefunction
SCALAR_WAVEFUNCTION = "wavefunction"

# Spin-1/2 wavefunction
SPINHALF_WAVEFUNCTION_PLUS = "wavefunction/psi_plus"
SPINHALF_WAVEFUNCTION_MINUS = "wavefunction/psi_minus"

# Spin-1 wavefunction
SPIN1_WAVEFUNCTION_PLUS = "wavefunction/psi_plus"
SPIN1_WAVEFUNCTION_ZERO = "wavefunction/psi_zero"
SPIN1_WAVEFUNCTION_MINUS = "wavefunction/psi_minus"

# Spin-2 wavefunction
SPIN2_WAVEFUNCTION_PLUS_TWO = "wavefunction/psi_plus2"
SPIN2_WAVEFUNCTION_PLUS_ONE = "wavefunction/psi_plus1"
SPIN2_WAVEFUNCTION_ZERO = "wavefunction/psi_zero"
SPIN2_WAVEFUNCTION_MINUS_ONE = "wavefunction/psi_minus1"
SPIN2_WAVEFUNCTION_MINUS_TWO = "wavefunction/psi_minus2"
