.. currentmodule:: pygpe.spintwo.data_manager

Spin-2 BEC DataManager
======================

The spin-2 BEC DataManager handles the wavefunction, grid, and parameter data
of the simulation.
To instantiate a DataManager object we simply call the constructor

.. autosummary::
   :toctree: generated/

   DataManager

We provide the `filename` and `data_path` at the creation of the object, and
additionally pass in the `Wavefunction` object and `params` dictionary, which
is then stored throughout the simulation.

Class methods
^^^^^^^^^^^^^

During evolution
----------------
The function that saves the current wavefunction data to the file is

.. autosummary::
   :toctree: generated/

   DataManager.save_wavefunction

You can call this function as often as you'd like depending on the time
resolution you require.

.. note::

    Since the wavefunction arrays are defined on a CUDA device, the data has
    to be passed over to the CPU to be saved.
    This slows down the simulation time, so only save data as often as you
    need to.
