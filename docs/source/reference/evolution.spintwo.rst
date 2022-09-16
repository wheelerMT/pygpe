.. currentmodule:: pygpe.spinone.evolution

Spin-2 BEC evolution functions
==============================

The evolution of the condensate is handle by one simple function

.. autosummary::
   :toctree: generated/

   step_wavefunction

This function propagates the wavefunction forward one time step.
To evolve the wavefunction for :math:`N_t` time steps, simply call this function in a loop :math:`N_t` times.
The evolution functions are purposely hidden behind this interface function to simplify the user experience.

The evolution functions are implemented using a second-order symplectic integrator.
See `here <https://journals.aps.org/pre/abstract/10.1103/PhysRevE.95.013311>`_ for more details on the numerical
implementation.

.. warning::

    The evolution functions are constructed so that the Fourier-space part is computed first.
    Ensure that you update the Fourier-space arrays prior to any evolution by calling the :code:`fft()` method
    of the :class:`Wavefunction` class.
