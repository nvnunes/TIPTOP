"""Nonstationary Gaussian optics preserve flux and native diffraction."""

import numpy as np
import pytest
from scipy import fft
from tiptop import finite_pupil
from tiptop.finite_pupil import FinitePupilSampling


def test_zero_covariance_is_exact_native_diffraction():
    y,x = np.mgrid[:32,:32];pupil=((x-15.5)**2+(y-15.5)**2<15**2).astype(float)
    sampling=FinitePupilSampling(pupil,128,32,8.)
    result, record=sampling.psf(np.zeros((len(sampling.xs),)*2))
    exact=abs(fft.fftshift(fft.fft2(sampling.native_pupil)))**2
    exact/=exact.sum()
    np.testing.assert_allclose(result,exact,atol=2e-15,rtol=1e-10)
    assert abs(record['full_flux']-1)<1e-12


def test_native_sampling_matches_explicit_gaussian_wave():
    pupil=np.ones((8,8));s=FinitePupilSampling(pupil,32,8,4.,optical_n=32)
    r=np.column_stack((.1*np.cos(2*np.pi*s.xy[:,0]),.1*np.sin(2*np.pi*s.xy[:,0])))
    image,_=s.psf(r@r.T)
    sf=((r[:,None]-r[None])**2).sum(-1)
    coherence=np.zeros((32,32))
    np.add.at(coherence,((s.ys[:,None]-s.ys[None])%32,(s.xs[:,None]-s.xs[None])%32),np.exp(-sf/2))
    expected=fft.fftshift(fft.fft2(coherence)).real/(32**2*len(s.xs))
    np.testing.assert_allclose(image,expected,atol=2e-15)


def test_compact_lag_cache_and_fallback_match_periodic_pairs(monkeypatch):
    pupil = np.ones((8, 8)); pupil[3:5, 3:5] = 0
    cached = FinitePupilSampling(pupil, 32, 8, 4., optical_n=32)
    monkeypatch.setattr(finite_pupil, '_PAIR_LAG_CACHE_BYTES', 0)
    blocked = FinitePupilSampling(pupil, 32, 8, 4., optical_n=32)
    assert cached._pair_lags is not None and blocked._pair_lags is None
    rng = np.random.default_rng(17)
    for size in (7, 32):
        grid = rng.normal(size=(size, size))
        dy = (cached.ys[:, None]-cached.ys[None]) % size
        dx = (cached.xs[:, None]-cached.xs[None]) % size
        expected = grid[dy, dx]
        np.testing.assert_array_equal(cached.covariance_from_lags(grid), expected)
        np.testing.assert_array_equal(blocked.covariance_from_lags(grid), expected)
    r = np.sin(cached.xy[:, 0])[:, None]*.1
    a, _ = cached.psf(r@r.T)
    b, _ = blocked.psf(r@r.T)
    np.testing.assert_array_equal(a, b)


def test_compact_optics_preserves_wrapped_lag_collisions():
    sampling = FinitePupilSampling(np.ones((16, 16)), 16, 16, 4., optical_n=16)
    r = np.column_stack((.1*np.sin(sampling.xy[:, 0]), .2*np.cos(sampling.xy[:, 1])))
    actual, _ = sampling.psf(r@r.T)
    sf = ((r[:, None]-r[None])**2).sum(-1)
    coherence = np.zeros((16, 16))
    np.add.at(coherence, ((sampling.ys[:, None]-sampling.ys[None]) % 16,
        (sampling.xs[:, None]-sampling.xs[None]) % 16), np.exp(-sf/2))
    expected = fft.fftshift(fft.fft2(coherence)).real/(16**2*len(sampling.xs))
    np.testing.assert_allclose(actual, expected, atol=2e-15)


@pytest.mark.parametrize('dtype', [np.float32, np.int64])
def test_non_double_covariance_preserves_weight_precision(dtype):
    sampling = FinitePupilSampling(np.ones((8, 8)), 32, 8, 4., optical_n=32)
    r = .1*np.sin(sampling.xy[:, 0])[:, None]
    covariance = (r@r.T).astype(dtype)
    actual, _ = sampling.psf(covariance)
    diagonal = np.diag(covariance)
    sf = diagonal[:, None]+diagonal[None]-2*covariance
    weights = np.exp(-.5*sf)*sampling.amplitude[:, None]*sampling.amplitude[None]
    coherence = np.zeros((32, 32))
    np.add.at(coherence, ((sampling.ys[:, None]-sampling.ys[None]) % 32,
        (sampling.xs[:, None]-sampling.xs[None]) % 32), weights)
    expected = fft.fftshift(fft.fft2(coherence)).real/(32**2*np.sum(sampling.native_pupil**2))
    np.testing.assert_allclose(actual, expected, atol=2e-15)
