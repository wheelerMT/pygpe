import cupy as cp
from pygpe.scalar.wavefunction import Wavefunction


def kinetic_step(wfn: Wavefunction, pm: dict) -> None:
    """Computes the kinetic energy subsystem for half a time step.

    :param wfn: The wavefunction of the system.
    :param pm: The parameter dictionary.
    """
    wfn.fourier_wavefunction *= cp.exp(-0.25 * 1j * pm["dt"] * wfn.grid.wave_number)


def potential_step(wfn: Wavefunction, pm: dict) -> None:
    """Computes the potential subsystem for a full time step.

    :param wfn: The wavefunction of the system.
    :param pm: The parameter dictionary.
    """
    wfn.wavefunction *= cp.exp(-1j * pm["dt"] * (pm["g"] * cp.abs(wfn.wavefunction) ** 2))
