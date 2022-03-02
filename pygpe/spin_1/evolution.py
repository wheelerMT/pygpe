import cupy as cp
from pygpe.spin_1.wavefunction import Wavefunction
from pygpe.spin_1.parameters import Parameters


def fft(wavefunction: Wavefunction) -> None:
    """Performs a fast Fourier transform on each wavefunction component."""
    wavefunction.fourier_plus_component = cp.fft.fftn(wavefunction.plus_component)
    wavefunction.fourier_zero_component = cp.fft.fftn(wavefunction.zero_component)
    wavefunction.fourier_minus_component = cp.fft.fftn(wavefunction.minus_component)


def ifft(wavefunction: Wavefunction) -> None:
    """Performs an inverse Fourier transform on each wavefunction component."""
    wavefunction.plus_component = cp.fft.ifftn(wavefunction.fourier_plus_component)
    wavefunction.zero_component = cp.fft.ifftn(wavefunction.fourier_zero_component)
    wavefunction.minus_component = cp.fft.ifftn(wavefunction.fourier_minus_component)


def kinetic_zeeman_step(wfn: Wavefunction, params: Parameters) -> None:
    """Computes the kinetic-zeeman subsystem for half a time step.

    :param wfn: The wavefunction of the system.
    :param params: The parameter class of the system.
    """
    wfn.fourier_plus_component *= cp.exp(-0.25 * 1j * params.dt * (wfn.grid.wave_number + 2 * params.q))
    wfn.fourier_zero_component *= cp.exp(-0.25 * 1j * params.dt * wfn.grid.wave_number)
    wfn.fourier_minus_component *= cp.exp(-0.25 * 1j * params.dt * (wfn.grid.wave_number + 2 * params.q))


class Evolution:
    pass
