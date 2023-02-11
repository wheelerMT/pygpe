.. currentmodule:: pygpe.scalar.wavefunction

Scalar BEC wavefunction
=======================

The scalar BEC wavefunction class contains the wavefunction array plus an assortment of useful functions for
manipulating and using the wavefunction.

Constructing the Wavefunction object is done through the constructor

.. autosummary::
    :toctree: generated/

    ScalarWavefunction

Here, the parameter `grid` is a :class:`Grid` object defined prior to instantiating the Wavefunction class.

Wavefunction methods
^^^^^^^^^^^^^^^^^^^^

Initial state
-------------
Below are the methods associated with the initial state.

.. autosummary::
   :toctree: generated/

   ScalarWavefunction.set_wavefunction
   ScalarWavefunction.add_noise
   ScalarWavefunction.apply_phase

The `set_wavefunction` method is used to set the initial state to whatever we desire.
The `add_noise` method adds noise to each grid point of the wavefunction where the noise is drawn from a uniform
distribution with the mean and standard deviation specified in the function signature.
Finally, `apply_phase` applies a user-defined phase to the wavefunction as :math:`\psi\rightarrow\psi e^{i\phi}` for
phase :math:`\phi`.

Other methods
-------------
The methods below fall under the miscellaneous category and are self-explanatory.

.. autosummary::
   :toctree: generated/

   ScalarWavefunction.fft
   ScalarWavefunction.ifft
   ScalarWavefunction.density

Attributes
----------
See :class:`ScalarWavefunction` for list of class attributes (variables).