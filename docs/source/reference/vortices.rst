.. currentmodule:: pygpe.shared.vortices

Vortices
========

PyGPE provides a function for constructing a 2D phase profile for use in generating vortices.
To access the function, we must import the `vortices` module::

    import pygpe.vortices as vort

The function signature is

.. autosummary::
   :toctree: generated/

   vortex_phase_profile

.. note::
   The number of vortices must be a multiple of 2 to allow for an equal number of vortices and anti-vortices.

The vortices have a minimal spacing which is specified by the user, in units of dimensionless length.

This returns a `cupy.ndarray` which can then be applied to a wavefunction via the :code:`apply_phase()` method to
generate different types of vortices.
