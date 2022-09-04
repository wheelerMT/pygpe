PyGPE documentation
===================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   user/wavefunction
   user/evolution

**Version:** |release|

PyGPE is a Python library for simulating the dynamics of Bose-Einstein condensate systems in 1D, 2D and 3D systems.
It offers an intuitive :doc:`user/wavefunction` class for managing and manipulating the wavefunction of the system.
What's more, the evolution of the Gross-Pitaevskii equations is handled using CuPy, making PyGPE extremely fast at
simulating dynamics.
PyGPE currently supports scalar, spin-1/2, spin-1 and spin-2 systems.

.. grid:: 2

    .. grid-item-card::
        :img-top: _static/getting_started.png

        Getting started
        ^^^^^^^^^^^^^^^
        Check out the starter guide for getting to grips with
        the basic use cases of PyGPE.

        +++
        .. button-ref:: user/starter_guide
            :expand:
            :color: secondary
            :click-parent:

            To the starter guide

    .. grid-item-card::
        :img-top: _static/api_ref.png

        API Reference
        ^^^^^^^^^^^^^
        See the API reference for detailed descriptions of the
        classes and functions contained in PyGPE.

        +++
        .. button-ref:: reference/api_reference
            :expand:
            :color: secondary
            :click-parent:

            To the API reference