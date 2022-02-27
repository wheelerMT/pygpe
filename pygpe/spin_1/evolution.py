import cupy as cp
from pygpe.spin_1.wavefunction import Wavefunction2D


def fft2(wavefunction: Wavefunction2D) -> None:
    """Performs a fast Fourier transform on each wavefunction component."""
    wavefunction.fourier_plus_component = cp.fft.fft2(wavefunction.plus_component)
    wavefunction.fourier_zero_component = cp.fft.fft2(wavefunction.zero_component)
    wavefunction.fourier_minus_component = cp.fft.fft2(wavefunction.minus_component)


def ifft2(wavefunction: Wavefunction2D) -> None:
    """Performs an inverse Fourier transform on each wavefunction component."""
    wavefunction.plus_component = cp.fft.ifft2(wavefunction.fourier_plus_component)
    wavefunction.zero_component = cp.fft.ifft2(wavefunction.fourier_zero_component)
    wavefunction.minus_component = cp.fft.ifft2(wavefunction.fourier_minus_component)
