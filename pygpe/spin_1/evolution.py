import cupy as cp
from pygpe.spin_1.wavefunction import Wavefunction
from pygpe.spin_1.parameters import Parameters


def imaginary_time(wfn: Wavefunction, params: Parameters):
    _fft(wfn)
    for i in range(params.nt):
        _propagate_wavefunction(wfn, params)
        _renormalise_wavefunction(wfn)
    _ifft(wfn)


def _fft(wavefunction: Wavefunction) -> None:
    """Performs a fast Fourier transform on each wavefunction component."""
    wavefunction.fourier_plus_component = cp.fft.fftn(wavefunction.plus_component)
    wavefunction.fourier_zero_component = cp.fft.fftn(wavefunction.zero_component)
    wavefunction.fourier_minus_component = cp.fft.fftn(wavefunction.minus_component)


def _ifft(wavefunction: Wavefunction) -> None:
    """Performs an inverse Fourier transform on each wavefunction component."""
    wavefunction.plus_component = cp.fft.ifftn(wavefunction.fourier_plus_component)
    wavefunction.zero_component = cp.fft.ifftn(wavefunction.fourier_zero_component)
    wavefunction.minus_component = cp.fft.ifftn(wavefunction.fourier_minus_component)


def _kinetic_zeeman_step(wfn: Wavefunction, params: Parameters) -> None:
    """Computes the kinetic-zeeman subsystem for half a time step.

    :param wfn: The wavefunction of the system.
    :param params: The parameter class of the system.
    """
    wfn.fourier_plus_component *= cp.exp(-0.25 * 1j * params.dt * (wfn.grid.wave_number + 2 * params.q))
    wfn.fourier_zero_component *= cp.exp(-0.25 * 1j * params.dt * wfn.grid.wave_number)
    wfn.fourier_minus_component *= cp.exp(-0.25 * 1j * params.dt * (wfn.grid.wave_number + 2 * params.q))


def _interaction_step(wfn: Wavefunction, params: Parameters) -> None:
    """Computes the interaction subsystem for a full time step.

    :param wfn: The wavefunction of the system.
    :param params: The parameter class of the system.
    """
    spin_perp, spin_z = _calculate_spins(wfn)
    spin_mag = cp.sqrt(abs(spin_perp) ** 2 + spin_z ** 2)
    dens = _calculate_density(wfn)

    # Trig terms needed in solution
    cos_term = cp.cos(params.c2 * spin_mag * params.dt)
    sin_term = cp.nan_to_num(1j * cp.sin(params.c2 * spin_mag * params.dt))

    plus_comp_temp = cos_term * wfn.plus_component - sin_term * (
            spin_z * wfn.plus_component + cp.conj(spin_perp) / cp.sqrt(2) * wfn.zero_component)
    zero_comp_temp = cos_term * wfn.zero_component - sin_term / cp.sqrt(2) * (
            spin_perp * wfn.plus_component + cp.conj(spin_perp) * wfn.minus_component)
    minus_comp_temp = cos_term * wfn.minus_component - sin_term * (
            spin_perp / cp.sqrt(2) * wfn.zero_component - spin_z * wfn.minus_component)

    wfn.plus_component = plus_comp_temp * cp.exp(-1j * params.dt * (params.trap - params.p + params.c0 * dens))
    wfn.zero_component = zero_comp_temp * cp.exp(-1j * params.dt * (params.trap + params.c0 * dens))
    wfn.minus_component = minus_comp_temp * cp.exp(-1j * params.dt * (params.trap + params.p + params.c0 * dens))


def _calculate_spins(wfn: Wavefunction) -> tuple[cp.ndarray, cp.ndarray]:
    """Calculates the perpendicular and longitudinal spins.

    :param wfn: The wavefunction of the system.
    :return: The perpendicular & longitudinal spin, respectively.
    """
    spin_perp = cp.sqrt(2.) * (
            cp.conj(wfn.plus_component) * wfn.zero_component + cp.conj(wfn.zero_component) * wfn.minus_component)
    spin_z = cp.abs(wfn.plus_component) ** 2 - cp.abs(wfn.minus_component) ** 2

    return spin_perp, spin_z


def _calculate_density(wfn: Wavefunction) -> cp.ndarray:
    """Calculates the total condensate density.

    :param wfn: The wavefunction of the system.
    :return: The total atomic density.
    """
    return cp.abs(wfn.plus_component) ** 2 + cp.abs(wfn.zero_component) ** 2 + cp.abs(wfn.minus_component) ** 2


def _propagate_wavefunction(wfn: Wavefunction, params: Parameters) -> None:
    """Propagates the wavefunction forward in time by a time step.

    :param wfn: The wavefunction of the system.
    :param params: The parameter class of the system.
    """
    _kinetic_zeeman_step(wfn, params)
    _ifft(wfn)
    _interaction_step(wfn, params)
    _fft(wfn)
    _kinetic_zeeman_step(wfn, params)


def _renormalise_wavefunction(wfn: Wavefunction) -> None:
    """Re-normalises the wavefunction to the correct atom number.

    :param wfn: The wavefunction of the system.
    """
    _ifft(wfn)
    correct_atom_plus, correct_atom_zero, correct_atom_minus = wfn.atom_num_plus, wfn.atom_num_zero, wfn.atom_num_minus
    current_atom_plus, current_atom_zero, current_atom_minus = _calculate_atom_num(wfn)
    wfn.plus_component *= cp.sqrt(correct_atom_plus / current_atom_plus)
    wfn.zero_component *= cp.sqrt(correct_atom_zero / current_atom_zero)
    wfn.minus_component *= cp.sqrt(correct_atom_minus / current_atom_minus)
    _fft(wfn)


def _calculate_atom_num(wfn: Wavefunction) -> tuple[int, int, int]:
    """Calculates the atom number of each wavefunction component.

    :param wfn: The wavefunction of the system.
    :return: The atom numbers of the plus, zero, and minus components, respectively.
    """
    atom_num_plus = wfn.grid.grid_spacing_product * cp.sum(cp.abs(wfn.plus_component) ** 2)
    atom_num_zero = wfn.grid.grid_spacing_product * cp.sum(cp.abs(wfn.zero_component) ** 2)
    atom_num_minus = wfn.grid.grid_spacing_product * cp.sum(cp.abs(wfn.minus_component) ** 2)

    return atom_num_plus, atom_num_zero, atom_num_minus
