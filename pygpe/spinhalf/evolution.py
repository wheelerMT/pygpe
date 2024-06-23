try:
    import cupy as cp  # type: ignore
except ImportError:
    import numpy as cp
from pygpe.spinhalf.wavefunction import SpinHalfWavefunction


def step_wavefunction(wfn: SpinHalfWavefunction, params: dict) -> None:
    """Propagates the wavefunction forward one time step.

    :param wfn: The wavefunction of the system.
    :type wfn: :class:`Wavefunction`
    :param params: The parameters of the system.
    :type params: dict
    """
    _kinetic_step(wfn, params)
    wfn.ifft()
    _potential_step(wfn, params)
    wfn.fft()
    _kinetic_step(wfn, params)
    if isinstance(params["dt"], complex):
        _renormalise_wavefunction(wfn)


def _kinetic_step(wfn: SpinHalfWavefunction, pm: dict) -> None:
    """Computes the kinetic energy subsystem for half a time step.

    :param wfn: The wavefunction of the system.
    :param pm: The parameters' dictionary.
    """
    wfn.fourier_plus_component *= cp.exp(-0.25 * 1j * pm["dt"] * wfn.grid.wave_number)
    wfn.fourier_minus_component *= cp.exp(-0.25 * 1j * pm["dt"] * wfn.grid.wave_number)


def _potential_step(wfn: SpinHalfWavefunction, pm: dict) -> None:
    """Computes the potential subsystem for a full time step.

    :param wfn: The wavefunction of the system.
    :param pm: The parameters' dictionary.
    """
    current_plus_component = wfn.plus_component
    current_minus_component = wfn.minus_component

    wfn.plus_component *= cp.exp(
        -1j
        * pm["dt"]
        * (
            pm["trap"]
            + pm["g_plus"] * cp.abs(current_plus_component) ** 2
            + pm["g_pm"] * cp.abs(current_minus_component)
        )
    )
    wfn.minus_component *= cp.exp(
        -1j
        * pm["dt"]
        * (
            pm["trap"]
            + pm["g_minus"] * cp.abs(current_minus_component) ** 2
            + pm["g_pm"] * cp.abs(current_plus_component)
        )
    )


def _renormalise_wavefunction(wfn: SpinHalfWavefunction) -> None:
    """Re-normalises the wavefunction to the correct atom number.

    :param wfn: The wavefunction of the system.
    """
    wfn.ifft()
    correct_atom_plus, correct_atom_minus = (
        wfn.atom_num_plus,
        wfn.atom_num_minus,
    )
    current_atom_plus, current_atom_minus = _calculate_atom_num(wfn)
    wfn.plus_component *= cp.sqrt(correct_atom_plus / current_atom_plus)
    wfn.minus_component *= cp.sqrt(correct_atom_minus / current_atom_minus)
    wfn.fft()


def _calculate_atom_num(wfn: SpinHalfWavefunction) -> tuple[int, int]:
    """Calculates the atom number of each wavefunction component.

    :param wfn: The wavefunction of the system.
    :return: The atom numbers of the plus, zero, and minus components,
        respectively.
    """
    atom_num_plus = wfn.grid.grid_spacing_product * cp.sum(
        cp.abs(wfn.plus_component) ** 2
    )
    atom_num_minus = wfn.grid.grid_spacing_product * cp.sum(
        cp.abs(wfn.minus_component) ** 2
    )

    return atom_num_plus, atom_num_minus
