.. currentmodule:: pygpe.spinhalf.data_manager

Two-component BEC DataManager
=============================

The two-component BEC DataManager handles the wavefunction, grid, and parameter data of the simulation.
To instantiate a DataManager object we simply call the constructor

.. autosummary::
   :toctree: generated/

   DataManager

We provide the `filename` and `data_path` at the creation of the object, which is then stored throughout the simulation.

Class methods
^^^^^^^^^^^^^

Before evolution
----------------
The following function should be called before any evolution takes place:

.. autosummary::
   :toctree: generated/

   DataManager.save_initial_parameters

This function saves the grid, parameters, and initial state data.

.. note::

    It is necessary to call this function as it generates the wavefunction datasets which allow us to save wavefunction
    data during the simulation.

During evolution
----------------
The function that saves the current wavefunction data to the file is

.. autosummary::
   :toctree: generated/

   DataManager.save_wavefunction

You can call this function as often as you'd like depending on the time resolution you require.

.. note::

    Since the wavefunction arrays are defined on a CUDA device, the data has to be passed over to the CPU to be saved.
    This slows down the simulation time, so only save data as often as you need to.