"""Compact finite-map optics retain static diffraction and covariance pairing."""
import numpy as np
import pytest
from scipy import fft

from tiptop.finite_static import FiniteStaticOptics
from tiptop.finite_pupil import FinitePupilSampling
from tiptop.static_map import OriginalPupilSampling


def inputs():
    y, x = np.mgrid[:16, :16]
    radius = (x-7.5)**2 + (y-7.5)**2
    pupil = ((radius < 7**2) & (radius > 2**2)).astype(float)
    phase = .5*np.sin(x/2) + .4*np.cos(y/3) + .2*(x-7.5)/8
    return pupil, phase


def strips(covariance):
    for first in range(0, len(covariance), 13):
        yield first, covariance[first:first+13, first:]


@pytest.mark.parametrize('scale', [0., .5, 1., 2.])
def test_static_limit_is_original_complex_pupil_diffraction(scale):
    pupil, phase = inputs()
    phase *= scale
    optics = FiniteStaticOptics(pupil, phase, 4., 32, 8, 16.)
    s = optics.sampling
    covariance = np.zeros((len(s.xs), len(s.xs)))
    actual, checks = optics.psf_blocks(np.diag(covariance), strips(covariance), complete=True)
    original = OriginalPupilSampling(pupil, phase, 4., 16., 32)
    wave = np.zeros((64, 64), complex)
    wave[original.ys, original.xs] = original.amplitude * original._static_phasor
    expected = abs(fft.fftshift(fft.fft2(wave)))**2/(64**2*np.sum(abs(wave)**2))
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-10)
    assert abs(checks['full_flux']-1) < 2e-12


def test_resolved_map_retains_nonstationary_gaussian_pairs():
    pupil, phase = inputs()
    optics = FiniteStaticOptics(pupil, phase, 4., 64, 16, 16.)
    direct = FinitePupilSampling(pupil, 64, 16, 16., static_phase=phase)
    s = optics.sampling
    patterns = np.column_stack((.2*np.sin(s.xy[:, 0]), .15*np.cos(s.xy[:, 1])))
    covariance = patterns @ patterns.T
    actual, checks = optics.psf_blocks(np.diag(covariance), strips(covariance))
    expected, original_checks = direct.psf_blocks(np.diag(covariance), strips(covariance))
    np.testing.assert_allclose(actual, expected, atol=3e-14, rtol=3e-9)
    assert abs(checks['residual_phase_variance']-original_checks['residual_phase_variance']) < 1e-15


def test_missing_covariance_rows_are_rejected():
    pupil, phase = inputs()
    optics = FiniteStaticOptics(pupil, phase, 4., 32, 8, 16.)
    with pytest.raises(ValueError, match='Incomplete'):
        optics.psf_blocks(np.zeros(len(optics.sampling.xs)), iter(()))


def test_zero_map_keeps_native_off_pixels_bitwise():
    from types import SimpleNamespace
    from tiptop.ho_optics import finite_pupil_fields
    pupil, _ = inputs()
    sampling = FinitePupilSampling(pupil, 64, 16, 16.)
    patterns = .2*np.sin(sampling.xy[:, 0, None])
    covariance = patterns @ patterns.T
    residual = SimpleNamespace(diagonal=np.diag(covariance), blocks=lambda:strips(covariance))
    native = SimpleNamespace(freq=SimpleNamespace(wvlRef=1.e-6), getErrorBreakDown=False,
                             finite_aperture_residual=lambda s, f:(residual, {}),
                             ao=SimpleNamespace(tel=SimpleNamespace(D=4.)),
                             _finite_aperture_backend=lambda:backend)
    backend=SimpleNamespace()
    expected = sampling.psf_blocks(np.diag(covariance),strips(covariance))[0]
    for phase in (None, np.zeros_like(pupil)):
        images, _, scale = finite_pupil_fields(native,pupil,phase,64,16,16.,[0])
        np.testing.assert_array_equal(images[0].sampling,expected)
        assert scale == 1.


def test_preparation_reuses_only_same_native_inputs(tmp_path):
    from tiptop.ho_optics import finite_pupil_fields
    from test_finite_two_stage import coupled_model,sampling
    native=coupled_model(tmp_path/'case')
    sample=sampling(native)
    pupil=native.ao.tel.pupil
    arguments=(native,pupil,None,sample.n,sample.native_side,sample.period,[0])
    first=finite_pupil_fields(*arguments)[0][0].sampling
    context=native._finitePupilPreparation
    second=finite_pupil_fields(*arguments)[0][0].sampling
    assert native._finitePupilPreparation is context
    np.testing.assert_array_equal(second,first)
    # A new atmosphere must replace both P3 preparation and the optical entry.
    native.ao.atm.wSpeed[0]+=1.
    finite_pupil_fields(*arguments)
    assert native._finitePupilPreparation is not context
    assert native._finitePupilPreparation.backend is not context.backend


def test_preparation_optical_key_covers_content_and_dimensions():
    from types import SimpleNamespace
    from tiptop.ho_optics import _finite_pupil_preparation
    pupil,_=inputs();backend=object()
    native=SimpleNamespace(ao=SimpleNamespace(tel=SimpleNamespace(D=4.)),
        freq=SimpleNamespace(wvlRef=1e-6),_finite_aperture_backend=lambda:backend)
    def prepare(p=pupil,phase=None,n=32,side=8,period=16.):
        return _finite_pupil_preparation(native,p,phase,n,side,period)
    initial=prepare();assert prepare(pupil.copy()) is initial
    phase=np.zeros_like(pupil);phase[2,2]=.1
    current=prepare(phase=phase);assert current is not initial
    phase[2,2]=.2;assert prepare(phase=phase) is not current
    current=prepare();changed=pupil.copy();changed[2,2]=.5
    assert prepare(p=changed) is not current
    current=prepare();native.freq.wvlRef=2e-6
    assert prepare() is not current
    current=prepare();assert prepare(n=64,side=16) is not current
    current=prepare();assert prepare(period=18.) is not current
    current=prepare();backend=object();assert prepare() is not current
