import cupy as cp
from pygpe.spin_1.wavefunction import Wavefunction


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
