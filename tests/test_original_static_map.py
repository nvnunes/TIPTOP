"""Original phase sampling, conserved crops and fast optical cache behavior."""

import numpy as np
import pytest
from scipy import fft
from tiptop.static_map import OriginalPupilSampling,AliasStaticOptics


def inputs():
    y,x=np.mgrid[:16,:16]
    pupil=(((x-7.5)**2+(y-7.5)**2<7**2)&((x-7.5)**2+(y-7.5)**2>2**2)).astype(float)
    phase=.5*np.sin(x/2)+.4*np.cos(y/3)+.2*(x-7.5)/8
    return pupil,phase


@pytest.mark.parametrize('map_on',[False,True])
def test_original_static_limit_matches_independent_pupil_fft(map_on):
    pupil,phase=inputs()
    if not map_on:phase=np.zeros_like(phase)
    original=OriginalPupilSampling(pupil,phase,4.,16.,32)
    optics=AliasStaticOptics(pupil,phase,4.,16.,32,8,1.)
    full,checks=optics.psf(np.zeros((32,32)),complete=True)
    y,x=np.mgrid[:16,:16];ramps=np.stack(((x-7.5)/8,(y-7.5)/8))
    valid=pupil>0
    coefficients=np.linalg.lstsq(ramps[:,valid].T,phase[valid],rcond=None)[0]
    fitted=phase-np.einsum('i,iyx->yx',coefficients,ramps)
    wave=np.zeros((64,64),complex);wave[24:40,24:40]=pupil*np.exp(1j*fitted)
    expected=abs(fft.fftshift(fft.fft2(wave)))**2/(64**2*np.sum(pupil**2))
    np.testing.assert_allclose(full,expected,atol=2e-14,rtol=2e-10)
    crop,_=optics.psf(np.zeros((32,32)))
    np.testing.assert_allclose(crop,original.detector_crop(full),atol=2e-13,rtol=2e-9)
    assert abs(full.sum()-1)<2e-12 and checks['minimum_pixel']>-1e-11
    assert crop.sum()<1 and optics.high_cache_builds==1


def test_high_covariance_cache_tracks_spectrum_content():
    pupil,phase=inputs();optics=AliasStaticOptics(pupil,phase,4.,16.,32,8,.1)
    k=np.fft.fftfreq(32)*32
    power=32**2*.1*(1+k[:,None]**2+k[None,:]**2)**(-11/6)
    first,_=optics.psf(power);assert optics.high_cache_builds==1
    repeated,_=optics.psf(power.copy())
    np.testing.assert_array_equal(first,repeated);assert optics.high_cache_builds==1
    changed,_=optics.psf(2*power)
    assert optics.high_cache_builds==2 and not np.array_equal(first,changed)


def test_invalid_static_otf_is_rejected():
    pupil,phase=inputs();optics=AliasStaticOptics(pupil,phase,4.,16.,32,8,1.)
    with pytest.raises(ValueError,match='unit-DC'):
        optics.correction.from_otf(np.zeros((24,24)))


def test_narrow_spectrum_that_breaks_otf_positivity_is_rejected():
    # The ratio approximation need not be a positive-definite coherence.
    # Preserve this counterexample: the implementation must not clip it away.
    pupil,phase=inputs();optics=AliasStaticOptics(pupil,phase,4.,16.,32,8,.1)
    power=np.zeros((32,32));power[0,4]=power[0,-4]=32**2*.01
    with pytest.raises(ValueError,match='failed positivity/flux'):
        optics.psf(power)
