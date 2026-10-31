"""Original-frequency tilt moments survive optical folding."""

import numpy as np
from numpy import fft
from mastsel import pupilTiltFit
from tiptop.pupil_fit import SpectralPupilFit


def test_coarse_moments_preserve_true_fit_of_aliased_wave():
    pupil=np.ones((8,8));source=SpectralPupilFit(pupil,16,8)
    target=SpectralPupilFit(pupil,8,4)
    power=np.zeros((16,16));power[0,5]=power[0,-5]=16**2/2
    covariance,cross,tilt=source.covariance_moments(power,output_n=8)
    y,x=np.mgrid[:8,:8]
    # Fit the original q=5 wave on the original pupil, independently of folding.
    fine_x=8-4+np.arange(8)
    phase=2*np.pi*5*fine_x/16
    fit=pupilTiltFit(pupil)
    coefficients=np.array([np.einsum('iyx,yx->i',fit,np.broadcast_to(f(phase),(8,8))) for f in (np.cos,np.sin)])
    coarse_phase=2*np.pi*5*x/8
    waves=np.stack((np.cos(coarse_phase),np.sin(coarse_phase)))
    waves-=np.einsum('ai,iyx->ayx',coefficients,target.ramps)
    expected=np.einsum('ayx,auv->yxuv',waves,waves).reshape(64,64)
    rows=np.arange(64);ys=rows//8;xs=rows%8;r=target.ramps.reshape(2,-1);v=cross[:,ys,xs]
    actual=covariance[(ys[:,None]-ys[None])%8,(xs[:,None]-xs[None])%8]
    actual-=r.T@v;actual-=v.T@r;actual+=r.T@tilt@r
    np.testing.assert_allclose(actual,expected,atol=2e-14,rtol=2e-14)


def test_coherence_matches_independent_dense_gaussian_covariance(monkeypatch):
    from tiptop import finite_pupil
    # Explicitly exercise the same NumPy pair consumer used without Numba.
    monkeypatch.setattr(finite_pupil, '_native_pairs', finite_pupil._native_pairs_numpy)
    pupil = np.ones((8,8)); fit = SpectralPupilFit(pupil,16,8.)
    rng = np.random.default_rng(114)
    power = rng.uniform(0,.01,(16,16))
    covariance, cross, tilt = fit.covariance_moments(power)
    wave = np.zeros((16,16),complex)
    wave[4:12,4:12] = np.exp(.3j*np.sin(np.arange(8))[None])
    actual, pupil_power, variance = fit.coherence_from_covariance(covariance,cross,tilt,wave)
    ys,xs = np.nonzero(wave); ramps = fit.ramps[:,ys,xs]; moments = cross[:,ys,xs]
    projected = covariance[(ys[:,None]-ys[None])%16,(xs[:,None]-xs[None])%16].copy()
    projected -= ramps.T@moments; projected -= moments.T@ramps
    projected += ramps.T@tilt@ramps
    diagonal = np.diag(projected)
    sf = diagonal[:,None]+diagonal[None,:]-2*projected
    terms = wave[ys,xs,None]*wave[ys,xs][None].conj()*np.exp(-.5*sf)
    expected = np.zeros((16,16),complex)
    np.add.at(expected,((ys[:,None]-ys[None])%16,(xs[:,None]-xs[None])%16),terms)
    np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=2e-13)
    np.testing.assert_allclose(variance,np.mean(diagonal),atol=1e-14)
    assert abs(actual[0,0]-pupil_power)<1e-12
