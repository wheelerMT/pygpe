try:
    import cupy as cp  # type: ignore
except ImportError:
    import numpy as cp
from pygpe.spinone.wavefunction import SpinOneWavefunction


def step_wavefunction(wfn: SpinOneWavefunction, params: dict) -> None:
    """Propagates the wavefunction forward one time step.

    :param wfn: The wavefunction of the system.
    :type wfn: :class:`Wavefunction`
    :param params: The parameters of the system.
    :type params: dict
    """
    _kinetic_zeeman_step(wfn, params)
    wfn.ifft()
    _interaction_step(wfn, params)
    wfn.fft()
    _kinetic_zeeman_step(wfn, params)
    if isinstance(params["dt"], complex):
        _renormalise_wavefunction(wfn)


def _kinetic_zeeman_step(wfn: SpinOneWavefunction, pm: dict) -> None:
    """Computes the kinetic-zeeman subsystem for half a time step.

    :param wfn: The wavefunction of the system.
    :param pm: The parameter. dictionary.
    """
    wfn.fourier_plus_component *= cp.exp(
        -0.25 * 1j * pm["dt"] * (wfn.grid.wave_number + 2 * pm["q"])
    )
    wfn.fourier_zero_component *= cp.exp(-0.25 * 1j * pm["dt"] * wfn.grid.wave_number)
    wfn.fourier_minus_component *= cp.exp(
        -0.25 * 1j * pm["dt"] * (wfn.grid.wave_number + 2 * pm["q"])
    )


def _interaction_step(wfn: SpinOneWavefunction, pm: dict) -> None:
    """Computes the interaction subsystem for a full time step.

    :param wfn: The wavefunction of the system.
    :param pm: The parameters' dictionary.
    """
    spin_perp, spin_z = _calculate_spins(wfn)
    spin_mag = cp.sqrt(abs(spin_perp) ** 2 + spin_z**2)
    dens = _calculate_density(wfn)

    # Trig terms needed in solution
    cos_term = cp.cos(pm["c2"] * spin_mag * pm["dt"])
    sin_term = cp.nan_to_num(1j * cp.sin(pm["c2"] * spin_mag * pm["dt"]) / spin_mag)

    plus_comp_temp = cos_term * wfn.plus_component - sin_term * (
        spin_z * wfn.plus_component
        + cp.conj(spin_perp) / cp.sqrt(2) * wfn.zero_component
    )
    zero_comp_temp = cos_term * wfn.zero_component - sin_term / cp.sqrt(2) * (
        spin_perp * wfn.plus_component + cp.conj(spin_perp) * wfn.minus_component
    )
    minus_comp_temp = cos_term * wfn.minus_component - sin_term * (
        spin_perp / cp.sqrt(2) * wfn.zero_component - spin_z * wfn.minus_component
    )

    wfn.plus_component = plus_comp_temp * cp.exp(
        -1j * pm["dt"] * (pm["trap"] - pm["p"] + pm["c0"] * dens)
    )
    wfn.zero_component = zero_comp_temp * cp.exp(
        -1j * pm["dt"] * (pm["trap"] + pm["c0"] * dens)
    )
    wfn.minus_component = minus_comp_temp * cp.exp(
        -1j * pm["dt"] * (pm["trap"] + pm["p"] + pm["c0"] * dens)
    )


def _calculate_spins(
    wfn: SpinOneWavefunction,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Calculates the perpendicular and longitudinal spins.

    :param wfn: The wavefunction of the system.
    :return: The perpendicular & longitudinal spin, respectively.
    """
    spin_perp = cp.sqrt(2.0) * (
        cp.conj(wfn.plus_component) * wfn.zero_component
        + cp.conj(wfn.zero_component) * wfn.minus_component
    )
    spin_z = cp.abs(wfn.plus_component) ** 2 - cp.abs(wfn.minus_component) ** 2

    return spin_perp, spin_z


def _calculate_density(wfn: SpinOneWavefunction) -> cp.ndarray:
    """Calculates the total condensate density.

    :param wfn: The wavefunction of the system.
    :return: The total atomic density.
    """
    return (
        cp.abs(wfn.plus_component) ** 2
        + cp.abs(wfn.zero_component) ** 2
        + cp.abs(wfn.minus_component) ** 2
    )


def _renormalise_wavefunction(wfn: SpinOneWavefunction) -> None:
    """Re-normalises the wavefunction to the correct atom number.

    :param wfn: The wavefunction of the system.
    """
    wfn.ifft()
    correct_atom_num = wfn.atom_num_plus + wfn.atom_num_zero + wfn.atom_num_minus
    current_atom_num = _calculate_atom_num(wfn)
    wfn.plus_component *= cp.sqrt(correct_atom_num / current_atom_num)
    wfn.zero_component *= cp.sqrt(correct_atom_num / current_atom_num)
    wfn.minus_component *= cp.sqrt(correct_atom_num / current_atom_num)
    wfn.fft()


def _calculate_atom_num(wfn: SpinOneWavefunction) -> float:
    """Calculates the total atom number of the system.

    :param wfn: The wavefunction of the system.
    :return: The total atom number.
    """
    atom_num_plus = wfn.grid.grid_spacing_product * cp.sum(
        cp.abs(wfn.plus_component) ** 2
    )
    atom_num_zero = wfn.grid.grid_spacing_product * cp.sum(
        cp.abs(wfn.zero_component) ** 2
    )
    atom_num_minus = wfn.grid.grid_spacing_product * cp.sum(
        cp.abs(wfn.minus_component) ** 2
    )

    return atom_num_plus + atom_num_zero + atom_num_minus
