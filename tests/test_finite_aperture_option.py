"""Finite-aperture opt-in is explicit and rejects incompatible output modes."""

import pytest
import numpy as np
from types import SimpleNamespace
from tiptop.baseSimulation import baseSimulation

from test_optional_products import _simulation


@pytest.mark.parametrize('model', [None, 'standard', 'alias_aware', 'finite_aperture'])
def test_reconstruction_model_is_selected_before_execution(tmp_path, model):
    sensor = {'TipTiltRemoval': "'pupil_fit'"}
    if model is not None:
        sensor['ReconstructionModel'] = repr(model)
    simulation = _simulation(tmp_path, False, fullFluxNormalization=True,
        config_updates={'sensor_HO': sensor})
    assert simulation.reconstructionModel == (model or 'standard')
    assert simulation.finiteApertureReconstruction is (model == 'finite_aperture')
    assert simulation.fao is None


@pytest.mark.parametrize('lo,method,full_flux,save_psds,message', [
    (True, 'pupil_fit', True, False, 'HO-only'),
    (False, 'none', True, False, 'pupil_fit'),
    (False, 'pupil_fit', False, False, 'fullFluxNormalization'),
    (False, 'pupil_fit', True, True, 'diagonal PSD'),
])
def test_incompatible_finite_aperture_outputs_fail_early(tmp_path, lo, method,
                                                       full_flux, save_psds, message):
    with pytest.raises(ValueError, match=message):
        _simulation(tmp_path, lo, fullFluxNormalization=full_flux,
            savePSDs=save_psds, config_updates={'sensor_HO': {
                'TipTiltRemoval': repr(method), 'ReconstructionModel': "'finite_aperture'"}})


@pytest.mark.parametrize('value', ['1', "'True'"])
def test_finite_aperture_rejects_non_boolean_values(tmp_path, value):
    with pytest.raises(ValueError, match='ReconstructionModel must be'):
        _simulation(tmp_path, False, fullFluxNormalization=True,
            config_updates={'sensor_HO': {
                'TipTiltRemoval': "'pupil_fit'", 'ReconstructionModel': value}})


@pytest.mark.parametrize('key', ['AliasAwareReconstruction', 'FiniteApertureReconstruction',
    'TomographicAliasing', 'SymmetricAliasing', 'SuperResolution'])
def test_retired_false_switch_is_rejected_by_tiptop(tmp_path, key):
    with pytest.raises(ValueError, match='Remove them and set ReconstructionModel'):
        _simulation(tmp_path, False, config_updates={'sensor_HO': {key: 'False'}})


@pytest.mark.parametrize('model', ['standard', 'alias_aware', 'finite_aperture'])
@pytest.mark.parametrize('section,key,value', [
    ('RTC','TemporalModel_HO', "'integrator'"),
    ('RTC','TemporalModel_HO', "'polc'"),
    ('sensor_HO','ExposureTime','None'),
    ('sensor_HO','ExposureTime','0.0'),
])
def test_tiptop_rejects_retired_timing_before_native_execution(tmp_path, model, section, key, value):
    config = {'sensor_HO': {'ReconstructionModel': repr(model), 'TipTiltRemoval': "'pupil_fit'"}}
    config.setdefault(section, {})[key] = value
    with pytest.raises(ValueError, match='Remove these keys. ReconstructionModel owns'):
        _simulation(tmp_path, False, fullFluxNormalization=True, config_updates=config)


@pytest.mark.parametrize('name,value,message',[
    ('nPixPSF',8,'native detector'),
    ('overSamp',2.,'native detector'),
    ('LOisOn',True,'HO-only'),
    ('jitter_FWHM',1.,'without jitter'),
])
def test_native_static_map_preserves_supported_scope(name,value,message):
    simulation=object.__new__(baseSimulation)
    simulation.nWvl=1;simulation.wvlRef=1e-6
    simulation.fao=SimpleNamespace(freq=SimpleNamespace(kxAO_=np.array([[0.,0.],[1.,1.]])))
    simulation.opdMap=np.ones((4,4))
    simulation.tipTiltRemoval='pupil_fit';simulation.fullFluxNormalization=True
    simulation.LOisOn=False;simulation.jitter_FWHM=None
    simulation.N=16;simulation.nPixPSF=16;simulation.overSamp=1.
    setattr(simulation,name,value)
    with pytest.raises(ValueError,match=message):
        simulation._finite_aperture_psfs()


def test_finite_static_optics_preserves_rms_independently_of_standard_option():
    """Model-owned resampling is identical for omitted, false and true flags."""
    from mastsel import pupilTiltFit, congrid, padOrCropCentered
    from scipy import fft
    from tiptop.static_phase import static_phase_rms_scale
    y,x=np.mgrid[:16,:16]
    pupil=(((x-7.5)**2+(y-7.5)**2<7**2)&((x-7.5)**2+(y-7.5)**2>2**2)).astype(float)
    phase=.5*np.sin(1.9*x)*np.cos(.8*y)+.03*x
    wavelength=1.2e-6
    expected_scale=static_phase_rms_scale(pupil,phase,8)

    def zero_residual(sampling, field):
        size=len(sampling.xs)
        residual=SimpleNamespace(diagonal=np.zeros(size),
            blocks=lambda: iter([(0,np.zeros((size,size)))]))
        return residual,{}

    images=[]
    for setting in (None,False,True):
        simulation=object.__new__(baseSimulation)
        simulation.nWvl=1;simulation.wvlRef=wavelength
        simulation.fao=SimpleNamespace(freq=SimpleNamespace(kxAO_=np.array([[0.,0.],[.25,.25]])),
            ao=SimpleNamespace(tel=SimpleNamespace(pupil=pupil)),finite_aperture_residual=zero_residual)
        simulation.opdMap=phase*wavelength/(2*np.pi*1e-9)
        simulation.tipTiltRemoval='pupil_fit';simulation.fullFluxNormalization=True
        simulation.LOisOn=False;simulation.jitter_FWHM=None
        simulation.N=32;simulation.nPixPSF=32;simulation.overSamp=1.
        simulation.sx=8;simulation.nPointings=1;simulation.grid_diameter=4.
        if setting is not None:
            simulation.staticOnRmsCompensation=setting
        image=simulation._finite_aperture_psfs()[0].sampling
        np.testing.assert_allclose(simulation.finiteApertureStaticRmsScale,expected_scale,rtol=2e-15)
        images.append(image)
    np.testing.assert_array_equal(images[0],images[1])
    np.testing.assert_array_equal(images[0],images[2])

    # Independent static-only diffraction after the compensated original fit.
    phase=phase*expected_scale
    coefficients=np.einsum('iyx,yx->i',pupilTiltFit(pupil),phase)
    coarse=congrid(phase,[8,8]);a=np.arange(8)-4
    coarse=coarse-coefficients[0]*(2*a[None]/8+1/16)
    coarse=coarse-coefficients[1]*(2*a[:,None]/8+1/16)
    wave=padOrCropCentered(congrid(pupil,[8,8])*np.exp(1j*coarse),32,np)
    expected=abs(fft.fftshift(fft.fft2(wave)))**2
    expected/=expected.sum()
    np.testing.assert_allclose(images[0],expected,rtol=1e-11,atol=2e-14)
