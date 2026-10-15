"""Field drift, conserved-field inner product, and translation fit."""
import numpy as np

def drift(phi, psi, L=200.0, d12=.2, d22=.1):
    n = len(phi)
    k = 2*np.pi*np.fft.fftfreq(n, d=L/n)
    k2 = k*k
    ph, ps = np.fft.fft(phi), np.fft.fft(psi)
    chemical_phi = -phi + phi**3 - np.fft.ifft(-k2*ph).real + d12*psi
    fphi = np.fft.ifft(-k2*np.fft.fft(chemical_phi)).real
    fpsi = np.fft.ifft(-k2*np.fft.fft(-d12*phi+d22*psi)).real
    ux_phi = np.fft.ifft(1j*k*ph).real
    ux_psi = np.fft.ifft(1j*k*ps).real
    return fphi, fpsi, ux_phi, ux_psi, k

def inner_hminus1(a, b, k, max_mode=None):
    ah, bh = np.fft.fft(a), np.fft.fft(b)
    weight = np.zeros_like(k)
    weight[k != 0] = 1/(k[k != 0]**2)
    if max_mode is not None:
        modes = np.abs(np.fft.fftfreq(len(k))*len(k))
        weight[modes > max_mode] = 0
    return float(np.real(np.sum(np.conj(ah)*bh*weight))/len(k)**2)

def one_residual(f1, f2, x1, x2, k, max_mode=None):
    inn = lambda a, b: inner_hminus1(a, b, k, max_mode)
    ff = inn(f1, f1)+inn(f2, f2)
    fx = inn(f1, x1)+inn(f2, x2)
    xx = inn(x1, x1)+inn(x2, x2)
    speed = -fx/xx
    r1, r2 = f1+speed*x1, f2+speed*x2
    rr = inn(r1, r1)+inn(r2, r2)
    return float(speed), float(np.sqrt(rr/ff)), float(np.sqrt(ff))
