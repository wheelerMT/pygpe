try:
    import cupy as cp  # type: ignore
except ImportError:
    import numpy as np


def handle_array(arr):
    """
    Converts a CuPy array to the equivalent NumPy array.
    If the array passed in is already a NumPy array, the array will
    just be returned.

    :param arr: The array to be converted
    :type arr: cupy.ndarray or numpy.ndarray
    """
    if isinstance(arr, np.ndarray):
        return arr
    return cp.asnumpy(arr)
