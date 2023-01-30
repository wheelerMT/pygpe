.. currentmodule:: pygpe.spinhalf.evolution

Two-component BEC evolution functions
=====================================

The evolution of the condensate is handle by one simple function

.. autosummary::
   :toctree: generated/

   step_wavefunction

This function propagates the wavefunction forward one time step.
To evolve the wavefunction for :math:`N_t` time steps, simply call this function in a loop :math:`N_t` times.
The evolution functions are purposely hidden behind this interface function to simplify the user experience.

The evolution functions are implemented using a second-order algorithm.
See `here <https://iopscience.iop.org/article/10.1088/0305-4470/39/12/L02/meta>`_ for more details on the numerical
implementation.
