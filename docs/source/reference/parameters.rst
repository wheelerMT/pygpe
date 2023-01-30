******************************
Condensate and time parameters
******************************

All of the condensate, trap and time parameters are stored in a Python dictionary.

.. warning::
    The dictionary **must** contain the specific keys listed below relating to these parameters to ensure the evolution
    functions work correctly.

Below is a list of the required parameters for each system.

Scalar dictionary
=================
The parameters required for the scalar system are in the table below

.. list-table:: Scalar system parameters
    :header-rows: 1

    * - Dictionary key
      - `dtype`
      - Description

    * - g
      - `float` or `cupy.ndarray`
      - Interaction strength
    * - trap
      - `float` or `cupy.ndarray`
      - Trapping potential
    * - nt
      - `int`
      - Number of time steps
    * - dt
      - `float` or `complex`
      - Numerical time step
    * - t
      - `float`
      - Current simulation time

Two-component dictionary
========================
The parameters required for the two-component system are in the table below

.. list-table:: Two-component system parameters
    :header-rows: 1

    * - Dictionary key
      - `dtype`
      - Description

    * - g_plus
      - `float` or `cupy.ndarray`
      - Interaction strength of plus component
    * - g_minus
      - `float` or `cupy.ndarray`
      - Interaction strength of minus component
    * - g_pm
      - `float` or `cupy.ndarray`
      - Inter-component interaction strength
    * - trap
      - `float` or `cupy.ndarray`
      - Trapping potential
    * - nt
      - `int`
      - Number of time steps
    * - dt
      - `float` or `complex`
      - Numerical time step
    * - t
      - `float`
      - Current simulation time

Spin-1 dictionary
=================
The parameters required for the spin-1 system are in the table below

.. list-table:: Spin-1 system parameters
    :header-rows: 1

    * - Dictionary key
      - `dtype`
      - Description

    * - c0
      - `float` or `cupy.ndarray`
      - Spin-independent interaction strength
    * - c2
      - `float` or `cupy.ndarray`
      - Spin-dependent interaction strength
    * - p
      - `float` or `cupy.ndarray`
      - Linear Zeeman shift
    * - q
      - `float` or `cupy.ndarray`
      - Quadratic Zeeman shift
    * - n0
      - `float` or `cupy.ndarray`
      - Condensate background density
    * - nt
      - `int`
      - Number of time steps
    * - dt
      - `float` or `complex`
      - Numerical time step
    * - t
      - `float`
      - Current simulation time

Spin-2 dictionary
=================
The parameters required for the spin-2 system are in the table below

.. list-table:: Spin-2 system parameters
    :header-rows: 1

    * - Dictionary key
      - `dtype`
      - Description

    * - c0
      - `float` or `cupy.ndarray`
      - Spin-independent interaction strength
    * - c2
      - `float` or `cupy.ndarray`
      - Spin-dependent interaction strength
    * - c4
      - `float` or `cupy.ndarray`
      - Spin-singlet interaction strength
    * - p
      - `float` or `cupy.ndarray`
      - Linear Zeeman shift
    * - q
      - `float` or `cupy.ndarray`
      - Quadratic Zeeman shift
    * - n0
      - `float` or `cupy.ndarray`
      - Condensate background density
    * - nt
      - `int`
      - Number of time steps
    * - dt
      - `float` or `complex`
      - Numerical time step
    * - t
      - `float`
      - Current simulation time