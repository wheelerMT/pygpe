.. currentmodule:: pygpe.spinone.wavefunction

Spin-1 BEC wavefunction
=======================

The spin-1 BEC wavefunction class contains the wavefunction arrays plus an assortment of useful functions for
manipulating and using the wavefunction.

Constructing the Wavefunction object is done through the constructor

.. autosummary::
    :toctree: generated/

    SpinOneWavefunction

Here, the parameter `grid` is a :class:`Grid` object defined prior to instantiating the Wavefunction class.

Wavefunction methods
^^^^^^^^^^^^^^^^^^^^

Initial state
-------------
Below are the methods associated with the initial state.

.. autosummary::
   :toctree: generated/

   SpinOneWavefunction.set_ground_state
   SpinOneWavefunction.set_wavefunction
   SpinOneWavefunction.add_noise
   SpinOneWavefunction.apply_phase

The `set_ground_state` method is used to set the initial state to a specified ground state of the spin-1 system.
It is typically the way we start shaping the condensate.
The supported ground states are listed below

.. list-table:: Supported spin-1 ground states
    :header-rows: 1

    * - Ground state
      - Wavefunction
      - Description

    * - "polar"
      - :math:`\psi=(0, 1, 0)^T`
      - (Easy-axis) polar ground state.

    * - "ferromagnetic"
      - :math:`\psi=(1, 0, 0)^T`
      - Ferromagnetic ground state with spin pointing up.

    * - "antiferromagnetic"
      - :math:`\psi=(\sqrt{(1 + p / c_2) / 2}, 0, \sqrt{(1 - p / c_2) / 2})^T`
      - Antiferromagnetic ground state.

    * - "BA"
      - See below.
      - Broken-axisymmetry ground state.

The broken-axisymmetry wavefunction components are

.. math::

    \psi_{\pm 1} = \frac{q \pm p}{2q} \sqrt{\frac{-p^2+q^2+2c_2nq}{2c_2nq}}, \\

    \psi_0 = \sqrt{\frac{(q^2-p^2)(-p^2-q^2+2c_2nq)}{4c_2nq^3}}.


If more flexibility is required, the `set_wavefunction` method allows us to set specific components to specified
arrays.
The `add_noise` method adds noise to each grid point of the wavefunction for the specified components.
The noise is drawn from a uniform distribution with the mean and standard deviation specified in the function signature.
Finally, `apply_phase` applies a user-defined phase to the specified wavefunction components as
:math:`\psi_m\rightarrow\psi_m e^{i\phi}` for phase :math:`\phi` and component :math:`m \in \{+, 0, -\}`.

Other methods
-------------
The methods below fall under the miscellaneous category and are self-explanatory.

.. autosummary::
   :toctree: generated/

   SpinOneWavefunction.fft
   SpinOneWavefunction.ifft
   SpinOneWavefunction.density

Attributes
----------
See :class:`SpinOneWavefunction` for list of class attributes (variables).