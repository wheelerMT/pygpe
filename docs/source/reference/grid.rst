.. currentmodule:: pygpe.shared.grid

The :class:`Grid` class
=======================

The Grid class is an object that stores all the details of the numerical grid, such as the number of grid points, grid
spacings etc.

**Constructing a Grid object**

A Grid object is constructed using the constructor

.. autosummary::
   :toctree: generated/

   Grid

The parameter `points` represents the number of points in each spatial dimension.
For 1D systems, this is simply an integer specifying the number of points.
For dimensions equal to 2 or more, this is a tuple :math:`(N_x, N_y)` or :math:`(N_x, N_y, N_z)` representing the
points in the :math:`x`, :math:`y` and :math:`z` directions, respectively.

Similarly, *grid_spacings* represents the numerical spacing between points for each spatial dimension.