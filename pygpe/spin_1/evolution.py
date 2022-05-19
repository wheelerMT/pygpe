import cupy as cp
from pygpe.spin_1.wavefunction import Wavefunction


def kinetic_zeeman_step(wfn: Wavefunction, pm: dict) -> None:
    """Computes the kinetic-zeeman subsystem for half a time step.

    :param wfn: The wavefunction of the system.
    :param pm: The parameter class of the system.
    """
    wfn.fourier_plus_component *= cp.exp(-0.25 * 1j * pm["dt"] * (wfn.grid.wave_number + 2 * pm["q"]))
    wfn.fourier_zero_component *= cp.exp(-0.25 * 1j * pm["dt"] * wfn.grid.wave_number)
    wfn.fourier_minus_component *= cp.exp(-0.25 * 1j * pm["dt"] * (wfn.grid.wave_number + 2 * pm["q"]))


def interaction_step(wfn: Wavefunction, pm: dict) -> None:
    """Computes the interaction subsystem for a full time step.

    :param wfn: The wavefunction of the system.
    :param pm: The parameter class of the system.
    """
    spin_perp, spin_z = _calculate_spins(wfn)
    spin_mag = cp.sqrt(abs(spin_perp) ** 2 + spin_z ** 2)
    dens = _calculate_density(wfn)

    # Trig terms needed in solution
    cos_term = cp.cos(pm["c2"] * spin_mag * pm["dt"])
    sin_term = cp.nan_to_num(1j * cp.sin(pm["c2"] * spin_mag * pm["dt"]))

    plus_comp_temp = cos_term * wfn.plus_component - sin_term * (
            spin_z * wfn.plus_component + cp.conj(spin_perp) / cp.sqrt(2) * wfn.zero_component)
    zero_comp_temp = cos_term * wfn.zero_component - sin_term / cp.sqrt(2) * (
            spin_perp * wfn.plus_component + cp.conj(spin_perp) * wfn.minus_component)
    minus_comp_temp = cos_term * wfn.minus_component - sin_term * (
            spin_perp / cp.sqrt(2) * wfn.zero_component - spin_z * wfn.minus_component)

    wfn.plus_component = plus_comp_temp * cp.exp(-1j * pm["dt"] * (pm["trap"] - pm["p"] + pm["c0"] * dens))
    wfn.zero_component = zero_comp_temp * cp.exp(-1j * pm["dt"] * (pm["trap"] + pm["c0"] * dens))
    wfn.minus_component = minus_comp_temp * cp.exp(-1j * pm["dt"] * (pm["trap"] + pm["p"] + pm["c0"] * dens))


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


def renormalise_wavefunction(wfn: Wavefunction) -> None:
    """Re-normalises the wavefunction to the correct atom number.

    :param wfn: The wavefunction of the system.
    """
    wfn.ifft()
    correct_atom_plus, correct_atom_zero, correct_atom_minus = wfn.atom_num_plus, wfn.atom_num_zero, wfn.atom_num_minus
    current_atom_plus, current_atom_zero, current_atom_minus = _calculate_atom_num(wfn)
    wfn.plus_component *= cp.sqrt(correct_atom_plus / current_atom_plus)
    wfn.zero_component *= cp.sqrt(correct_atom_zero / current_atom_zero)
    wfn.minus_component *= cp.sqrt(correct_atom_minus / current_atom_minus)
    wfn.fft()


def _calculate_atom_num(wfn: Wavefunction) -> tuple[int, int, int]:
    """Calculates the atom number of each wavefunction component.

    :param wfn: The wavefunction of the system.
    :return: The atom numbers of the plus, zero, and minus components, respectively.
    """
    atom_num_plus = wfn.grid.grid_spacing_product * cp.sum(cp.abs(wfn.plus_component) ** 2)
    atom_num_zero = wfn.grid.grid_spacing_product * cp.sum(cp.abs(wfn.zero_component) ** 2)
    atom_num_minus = wfn.grid.grid_spacing_product * cp.sum(cp.abs(wfn.minus_component) ** 2)

    return atom_num_plus, atom_num_zero, atom_num_minus
