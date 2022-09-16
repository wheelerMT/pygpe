**********************
PyGPE: An introduction
**********************

.. py:currentmodule:: pygpe

Welcome to PyGPE!
^^^^^^^^^^^^^^^^^

PyGPE is an open source Python library for use in simulating the dynamics
of Bose-Einstein condensate (BEC) systems.
It offers an easy-to-use API that makes simulating such dynamics painless.
PyGPE solves the Gross-Pitaevskii equations using CuPy meaning above all else,
PyGPE is **fast**.

Installing PyGPE
^^^^^^^^^^^^^^^^

TODO: ADD PIP INSTRUCTIONS

How to import PyGPE
^^^^^^^^^^^^^^^^^^^

Since PyGPE supports multiple types of BEC systems, we need to first select
the system we want to work with.
PyGPE currently supports scalar, spin-1/2, spin-1, and spin-2 systems.
To access the correct functions & classes for a given system type, we
import the relevant module into Python like::

    import pygpe.scalar as gpe

We shorten the import name to `gpe` for better readability.
The table below lists the system types and their respective import statements

.. list-table::
    :widths: 25 25
    :header-rows: 1

    * - System type
      - Import statement
    * - Scalar BEC
      - :code:`import pygpe.scalar`
    * - Spin-1 BEC
      - :code:`import pygpe.spinone`
    * - Spin-2 BEC
      - :code:`import pygpe.spintwo`

.. warning::
    Importing multiple different systems into the same project can have
    disastrous side effects.
    Ensure you are only importing **one** of the above modules in your project.

Using PyGPE
^^^^^^^^^^^

The use of PyGPE can be broken down into a few simple steps:
    - Set up numerical grid.
    - Define condensate and time parameters.
    - Set up wavefunction & set initial state.
    - Set up DataManager (if using).
    - Evolve the system.

Setting up the grid
-------------------

PyGPE offers a Grid class that handles all the details of the numerical grid.
It supports 1D, 2D and 3D grids.
To create a grid we first define the number of grid points per
dimension and their respective grid spacings, then generate a Grid object::

    import pygpe.spinone as gpe

    grid_points = (64, 64, 64)
    grid_spacings = (0.5, 0.5, 0.5)
    grid = gpe.Grid(grid_points, grid_spacings)  # Creates our grid object

The above code generates a 3D grid with 64 points and a grid spacing of 0.5 in
each dimension.
To create grids of different dimensionality you only need change the grid_points
and grid_spacings to match the desired dimensionality.
For example, to create a 2D grid we would instead have
:code:`grid_points = (64, 64)` and :code:`grid_spacings = (0.5, 0.5)`.
Similarly, for 1D we would simply have :code:`grid_points = 64` and
:code:`grid_spacing = 0.5`.

The grid object is stored in the parameter you specified, in our case `grid`.
We can access useful attributes about our grid::

    print(grid.ndim)  # 3
    print(grid.total_num_points)  # 262144

.. note::
   The grid class is shared between all system types, so what works here
   for the spin-1 system will work for all other systems.

Defining condensate and time parameters
---------------------------------------

PyGPE uses a simple dictionary to keep track of condensate and time parameters.
For a spin-1 system we can define it as::

    params = {
    "c0": 10,  # Spin-independent interaction
    "c2": 0.5,  # Spin-independent interaction
    "p": 0.,  # Linear Zeeman shift
    "q": 0.,  # Quadratic Zeeman shift
    "trap": 0.,  # Trapping potential
    "n0": 1,  # Background density

    # Time params
    "dt": 1e-2,  # Numerical time step
    "nt": 1000,  # Number of time steps
    "t": 0  # Current time
    }

Each system requires specific parameters to be defined in order for the evolution functions to work correctly.
See :doc:`../reference/parameters` for more details on parameters and their definitions.

Setting up the wavefunction
---------------------------

Now that we have a grid class, we can use this to set up our wavefunction.
Setting up the initial wavefunction class is easy, we just need to pass in the
grid we have constructed.
Then we can use the class methods to manipulate the wavefunction into the
desired initial state::

    wavefunction = gpe.Wavefunction(grid)
    wavefunction.set_ground_state("polar")
    wavefunction.add_noise_to_components(components="outer", mean=0., std=1e-2)

This first creates a wavefunction in a polar state :math:`\psi=(0,1,0)^T` then
subsequently adds numerical noise drawn from a normal distribution with mean
:math:`\mu=0` and variance :math:`\sigma=10^{-2}` to the outer
(:math:`\psi_\pm`) components.

Setting up the data manager
---------------------------

PyGPE provides an easy way to save data throughout the simulation.
Once the initial grid, wavefunction and condensate parameters have been defined we an instantiate a DataManager class,
which saves all the initial details of the system.
To do this, we write::

    data = gpe.DataManager(filename='data.hdf5', data_path='../../data/')
    data.save_initial_parameters(grid, wavefunction, params)

The constructor takes two parameters: `filename` and the path where we want to save the data, `data_path`.
We then call `data.save_initial_parameters` to save our initial grid, wavefunction and parameters to the dataset.
Finally, to save the current wavefunction to the dataset we simply write::

    data.save_wavefunction(wavefunction)

For more detail on how the DataManager class works see LINK TO DATA MANAGER API.

Evolving the wavefunction
-------------------------

Now that everything is set up, we get to the important part: evolving the wavefunction.
PyGPE provides a simple function for evolving stepping the wavefunction forward one time step.
To step the wavefunction forward for a set number of time steps we include a for loop::

    for i in range(params["nt"]):
        gpe.step_wavefunction(wavefunction)

That's it! All the evolution happens behind the `step_wavefunction` method.

Imaginary/complex time evolution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Imaginary time evolution is an extremely useful way of computing ground states of Bose-Einstein condensate systems and
PyGPE readily supports it.
To use imaginary time evolution we simply have to define an imaginary time step in our parameters dictionary::

    params = {"dt": -1j * 1e-2}

PyGPE handles re-normalizing the wavefunction automatically.
To switch back to real time, re-define the time step as a float.