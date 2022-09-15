.. currentmodule:: pygpe.spintwo.wavefunction

Spin-2 BEC wavefunction
=======================

The spin-2 BEC wavefunction class contains the wavefunction arrays plus an assortment of useful functions for
manipulating and using the wavefunction.

Constructing the Wavefunction object is done through the constructor

.. autosummary::
    :toctree: generated/

    Wavefunction

Here, the parameter `grid` is a :class:`Grid` object defined prior to instantiating the Wavefunction class.

Wavefunction methods
^^^^^^^^^^^^^^^^^^^^

Initial state
-------------
Below are the methods associated with the initial state.

.. autosummary::
   :toctree: generated/

   Wavefunction.set_ground_state
   Wavefunction.set_custom_components
   Wavefunction.add_noise_to_components
   Wavefunction.apply_phase

The `set_ground_state` method is used to set the initial state to a specified ground state of the spin-2 system.
It is typically the way we start shaping the condensate.
The supported ground states are listed below

.. list-table:: Supported spin-2 ground states
    :header-rows: 1

    * - Ground state
      - Wavefunction
      - Description

    * - "UN"
      - :math:`\psi=(0, 0, 1, 0, 0)^T`
      - Uniaxial-nematic ground state.

    * - "BN"
      - :math:`\psi=(1, 0, 0, 0, 1)^T/\sqrt{2}`
      - Biaxial-nematic ground state.

    * - "F2p"
      - :math:`\psi=(1, 0, 0, 0, 0)^T`
      - Ferromagnetic ground state with total spin 2, pointing up.

    * - "F2m"
      - :math:`\psi=(0, 0, 0, 0, 1)^T`
      - Ferromagnetic ground state with total spin 2, pointing down.

    * - "F1p"
      - :math:`\psi=(0, 1, 0, 0, 0)^T`
      - Ferromagnetic ground state with total spin 1, pointing up.

    * - "F1m"
      - :math:`\psi=(0, 0, 0, 1, 0)^T`
      - Ferromagnetic ground state with total spin 1, pointing down.

    * - "cyclic"
      - :math:`\psi=(\sqrt{(1 + f_z) / 3}, 0, 0, \sqrt{(2 - f_z) / 3}, 0)^T`
      - Cyclic ground state where :math:`f_z=p+q/c_2n`.

If more flexibility is required, the `set_custom_components` method allows us to set specific components to specified
arrays.
The `add_noise_to_components` method adds noise to each grid point of the wavefunction for the specified components.
The noise is drawn from a uniform distribution with the mean and standard deviation specified in the function signature.
Finally, `apply_phase` applies a user-defined phase to the specified wavefunction components as
:math:`\psi_m\rightarrow\psi_m e^{i\phi}` for phase :math:`\phi` and component :math:`m \in \{2, 1, 0 , -1, -2\}`.

Other methods
-------------
The methods below fall under the miscellaneous category and are self-explanatory.

.. autosummary::
   :toctree: generated/

   Wavefunction.fft
   Wavefunction.ifft

Attributes
----------
See :class:`Wavefunction` for list of class attributes (variables).