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


@pytest.mark.parametrize('native_n', [96, 128])
def test_static_only_matches_native_fourier_optics(native_n):
    y,x=np.mgrid[:32,:32];p=(((x-15.5)**2+(y-15.5)**2<15**2)&((x-15.5)**2+(y-15.5)**2>3**2)).astype(float)
    phase=.6*np.cos((x-15.5)/5)+.4*np.sin((y-15.5)/3)+.3*(x-15.5)/16
    s=FinitePupilSampling(p,native_n,32,8.,static_phase=phase)
    image,_=s.psf(np.zeros((len(s.xs),)*2))
    from mastsel import congrid,padOrCropCentered,pupilTiltFit
    c=np.einsum('iyx,yx->i',pupilTiltFit(p),phase)
    yy,xx=np.mgrid[:native_n,:native_n]
    fine=padOrCropCentered(congrid(phase,[32,32]),native_n,np)
    fine-=c[0]*(2*(xx-native_n/2)/32+1/32)+c[1]*(2*(yy-native_n/2)/32+1/32)
    exact=abs(fft.fftshift(fft.fft2(s.native_pupil*np.exp(1j*fine))))**2
    exact/=exact.sum()
    np.testing.assert_allclose(image,exact,atol=3e-15,rtol=2e-10)
    blocks=((first,np.zeros((min(31,len(s.xs)-first),len(s.xs)-first)))
        for first in range(0,len(s.xs),31))
    blocked,_=s.psf_blocks(np.zeros(len(s.xs)),blocks)
    np.testing.assert_allclose(blocked,exact,atol=3e-15,rtol=2e-10)


def test_zero_static_phase_preserves_existing_pixels_bitwise():
    pupil=np.ones((8,8));a=FinitePupilSampling(pupil,32,8,4.)
    b=FinitePupilSampling(pupil,32,8,4.,static_phase=np.zeros_like(pupil))
    r=.1*np.sin(a.xy[:,0])[:,None];c=r@r.T
    np.testing.assert_array_equal(a.psf(c)[0],b.psf(c)[0])


def test_complex_gaussian_pairs_include_static_dynamic_correlation():
    p=np.ones((8,8));y,x=np.mgrid[:8,:8];phase=.5*np.sin(x)+.2*y
    s=FinitePupilSampling(p,32,8,4.,optical_n=32,static_phase=phase)
    r=np.column_stack((.2*np.sin(s.xy[:,0]),.15*np.cos(s.xy[:,1])))
    image,_=s.psf(r@r.T)
    sf=((r[:,None]-r[None])**2).sum(-1)
    w=np.exp(-sf/2+1j*(s.static_phase[:,None]-s.static_phase[None]))
    coherence=np.zeros((32,32),complex)
    np.add.at(coherence,((s.ys[:,None]-s.ys[None])%32,(s.xs[:,None]-s.xs[None])%32),w)
    exact=fft.fftshift(fft.fft2(coherence)).real/(32**2*np.sum(s.native_pupil**2))
    np.testing.assert_allclose(image,exact,atol=2e-15)


def test_static_phase_input_is_read_only_without_padding():
    pupil = np.ones((8, 8)); y, x = np.mgrid[:8, :8]
    phase = .1*x+.2*y+.3*np.cos(x)
    before = phase.copy()
    sampling = FinitePupilSampling(pupil,8,8,4.,optical_n=8,static_phase=phase)
    np.testing.assert_array_equal(phase,before)
    from mastsel import pupilTiltFit
    coefficients = np.einsum('iyx,yx->i',pupilTiltFit(pupil),before)
    projected = before-coefficients[0]*2*(x-3.5)/8-coefficients[1]*2*(y-3.5)/8
    expected = abs(fft.fftshift(fft.fft2(pupil*np.exp(1j*projected))))**2
    expected /= expected.sum()
    actual,_ = sampling.psf(np.zeros((len(sampling.xs),)*2))
    np.testing.assert_allclose(actual,expected,atol=2e-15)


def test_static_phase_requires_native_quadrature():
    pupil = np.ones((8,8))
    with pytest.raises(ValueError,match='requires native'):
        FinitePupilSampling(pupil,32,8,4.,optical_n=24,static_phase=np.ones_like(pupil))
    assert FinitePupilSampling(pupil,32,8,4.,static_phase=np.zeros_like(pupil)).n == 32
    assert FinitePupilSampling(pupil,32,8,4.).n == 32
    assert FinitePupilSampling(pupil,32,8,4.,optical_n=24).n == 24


@pytest.mark.parametrize('backend',['numpy','numba'])
@pytest.mark.parametrize('zero_map',[False,True])
def test_native_blocks_without_static_phase_match_dense(backend,zero_map,monkeypatch):
    kernel = getattr(finite_pupil,'_native_pairs_'+backend)
    if kernel is None:pytest.skip('Optional Numba is unavailable')
    monkeypatch.setattr(finite_pupil,'_native_pairs',kernel)
    pupil=np.ones((8,8));pupil[3:5,3:5]=0
    phase=np.zeros_like(pupil) if zero_map else None
    sample=FinitePupilSampling(pupil,32,8,4.,static_phase=phase)
    r=np.column_stack((.1*np.sin(sample.xy[:,0]),.2*np.cos(sample.xy[:,1])))
    covariance=r@r.T
    expected,expected_checks=sample.psf(covariance)
    blocks=((first,covariance[first:first+7,first:]) for first in range(0,len(covariance),7))
    actual,checks=sample.psf_blocks(np.diag(covariance),blocks)
    np.testing.assert_allclose(actual,expected,atol=2e-15,rtol=2e-11)
    np.testing.assert_allclose(checks['residual_phase_variance'],
        expected_checks['residual_phase_variance'],atol=1e-13)


@pytest.mark.parametrize('backend',['numpy','numba'])
def test_native_pair_backends_match_full_ordered_pairs(backend,monkeypatch):
    kernel = getattr(finite_pupil,'_native_pairs_'+backend)
    if kernel is None:pytest.skip('Optional Numba is unavailable')
    monkeypatch.setattr(finite_pupil,'_native_pairs',kernel)
    pupil=np.ones((8,8));pupil[3:5,3:5]=0;pupil[0,0]=.7
    y,x=np.mgrid[:8,:8]
    sample=FinitePupilSampling(pupil,32,8,4.,static_phase=.3*np.sin(x)+.2*y)
    r=np.column_stack((.1*np.sin(sample.xy[:,0]),.2*np.cos(sample.xy[:,1])))
    covariance=r@r.T
    actual,checks=sample.psf(covariance)
    sf=((r[:,None]-r[None])**2).sum(-1)
    weights=np.exp(-sf/2+1j*(sample.static_phase[:,None]-sample.static_phase[None]))
    weights *= sample.amplitude[:,None]*sample.amplitude[None]
    coherence=np.zeros((32,32),complex)
    np.add.at(coherence,((sample.ys[:,None]-sample.ys[None])%32,
        (sample.xs[:,None]-sample.xs[None])%32),weights)
    expected=fft.fftshift(fft.fft2(coherence)).real/sample._normalization
    np.testing.assert_allclose(actual,expected,atol=2e-15,rtol=2e-11)
    variance=sample.intensity@np.diag(covariance)-sample.intensity@covariance@sample.intensity
    np.testing.assert_allclose(checks['residual_phase_variance'],variance,atol=1e-13)
    blocks=((first,covariance[first:first+7,first:]) for first in range(0,len(covariance),7))
    blocked,_=sample.psf_blocks(np.diag(covariance),blocks)
    np.testing.assert_allclose(blocked,expected,atol=2e-15,rtol=2e-11)


def test_covariance_block_slices_match_full_lag_sampling():
    sampling=FinitePupilSampling(np.ones((8,8)),32,8,4.,optical_n=32)
    rng=np.random.default_rng(27)
    for size in (7,32):
        lags=rng.normal(size=(size,size))
        full=sampling.covariance_from_lags(lags)
        for r,c in ((slice(3,9),slice(7,20)),(slice(0,1),slice(None)),(slice(-5,None),slice(-4,None))):
            np.testing.assert_array_equal(sampling.covariance_from_lags(lags,r,c),full[r,c])


def test_native_blocks_reject_missing_or_overlapping_rows():
    sample=FinitePupilSampling(np.ones((4,4)),16,4,4.,static_phase=np.ones((4,4)))
    diagonal=np.ones(len(sample.xs))
    with pytest.raises(ValueError,match='every optical pupil point'):
        sample.psf_blocks(diagonal,[])
    with pytest.raises(ValueError,match='consecutive upper row strips'):
        sample.psf_blocks(diagonal,[(1,np.ones((3,len(diagonal)-1)))])


@pytest.mark.parametrize('backend',['numpy','numba'])
def test_native_static_pairs_preserve_wrapped_lag_collisions(backend,monkeypatch):
    kernel=getattr(finite_pupil,'_native_pairs_'+backend)
    if kernel is None:pytest.skip('Optional Numba is unavailable')
    monkeypatch.setattr(finite_pupil,'_native_pairs',kernel)
    pupil=np.ones((8,8));y,x=np.mgrid[:8,:8]
    sample=FinitePupilSampling(pupil,8,8,4.,static_phase=.2*np.cos(x))
    r=np.sin(sample.xy[:,0])[:,None]*.1;covariance=r@r.T
    actual,_=sample.psf(covariance)
    sf=((r[:,None]-r[None])**2).sum(-1)
    weights=np.exp(-sf/2+1j*(sample.static_phase[:,None]-sample.static_phase[None]))
    coherence=np.zeros((8,8),complex)
    np.add.at(coherence,((sample.ys[:,None]-sample.ys[None])%8,
        (sample.xs[:,None]-sample.xs[None])%8),weights)
    expected=fft.fftshift(fft.fft2(coherence)).real/sample._normalization
    np.testing.assert_allclose(actual,expected,atol=2e-15)
    blocks=((first,covariance[first:first+7,first:]) for first in range(0,len(covariance),7))
    blocked,_=sample.psf_blocks(np.diag(covariance),blocks)
    np.testing.assert_allclose(blocked,expected,atol=2e-15)
