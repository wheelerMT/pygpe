.. currentmodule:: pygpe.spinhalf.wavefunction

Two-component BEC wavefunction
==============================

The two-component BEC wavefunction class contains the wavefunction arrays plus an assortment of useful functions for
manipulating and using the wavefunction.

Constructing the Wavefunction object is done through the constructor

.. autosummary::
    :toctree: generated/

    SpinHalfWavefunction

Here, the parameter `grid` is a :class:`Grid` object defined prior to instantiating the Wavefunction class.

Wavefunction methods
^^^^^^^^^^^^^^^^^^^^

Initial state
-------------
Below are the methods associated with the initial state.

.. autosummary::
   :toctree: generated/

   SpinHalfWavefunction.set_wavefunction
   SpinHalfWavefunction.add_noise
   SpinHalfWavefunction.apply_phase

The `set_wavefunction` method is used to set the initial state of the two-component system.

The `add_noise` method adds noise to each grid point of the wavefunction for the specified components.
The noise is drawn from a uniform distribution with the mean and standard deviation specified in the function signature.
Finally, `apply_phase` applies a user-defined phase to the specified wavefunction components as
:math:`\psi_m\rightarrow\psi_m e^{i\phi}` for phase :math:`\phi` and component :math:`m \in \{1, 2\}`.

Other methods
-------------
The methods below fall under the miscellaneous category and are self-explanatory.

.. autosummary::
   :toctree: generated/

   SpinHalfWavefunction.fft
   SpinHalfWavefunction.ifft
   SpinHalfWavefunction.density

Attributes
----------
See :class:`SpinHalfWavefunction` for list of class attributes (variables).