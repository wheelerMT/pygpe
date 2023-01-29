The DataManager class
=====================

PyGPE provides a class (DataManager) for handling the data of the simulation.
We use HDF5 filetypes for their speed and simplicity.
This means we only need *one* file to manage *all* the data of the system.
For more information on HDF5 and how it works, see their `website <https://www.hdfgroup.org/solutions/hdf5/>`_.

.. toctree::
    :maxdepth: 1

    datamanager.scalar
    datamanager.spinhalf
    datamanager.spinone
    datamanager.spintwo
