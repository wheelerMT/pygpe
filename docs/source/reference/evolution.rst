*********************************
Evolution functions
*********************************

PyGPE provides fast and efficient routines for propagating the Gross-Pitaevskii equations.
Rather than inundating the user with various functions, we provide one simple function to propagate the wavefunction
forward one time step.
You then have the freedom to evolve for as long or as little as desired, with the flexibility to modify system details
after each time step if you so desire.

See the respective pages below for details on the implementation.

.. toctree::
    :maxdepth: 1

    evolution.scalar
    evolution.spinone
    evolution.spintwo
