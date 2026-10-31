"""Measured static RMS compensation, with unchanged default map optics."""

import numpy as np
import pytest

from mastsel import psdSetToPsfSet
from tiptop.static_phase import static_phase_rms_scale, rms_compensated_psfs


def inputs():
    y, x = np.mgrid[:16, :16]
    pupil = (((x-7.5)**2+(y-7.5)**2 < 7**2) &
             ((x-7.5)**2+(y-7.5)**2 > 2**2)).astype(float)
    opd = 80*np.sin(1.9*x)*np.cos(.8*y)+4*x
    return pupil, opd


@pytest.mark.parametrize('per_wavelength', [False, True])
def test_measured_compensation_matches_explicit_map_scaling(per_wavelength):
    pupil, opd = inputs()
    before = opd.copy()
    common = dict(nPixPsf=32, full_flux=True, tip_tilt_removal='pupil_fit')
    wavelengths = [1.2e-6, 2.2e-6]
    sides = [8, 12] if per_wavelength else [8, 8]
    psd = np.zeros((32, 32))
    kwargs = dict(nPixPup=sides if per_wavelength else 8,
                  freq_range=[8., 8.] if per_wavelength else 8.,
                  dk=[1., 1.] if per_wavelength else 1.)
    result = rms_compensated_psfs([[psd], [psd]] if per_wavelength else [psd],
        pupil, wavelengths, opdMap=opd, **common, **kwargs)
    for i, (wavelength, side) in enumerate(zip(wavelengths, sides)):
        factor = static_phase_rms_scale(pupil, opd, side)
        expected = psdSetToPsfSet([[psd]] if per_wavelength else [psd], pupil,
            [wavelength], nPixPup=side, freq_range=8., dk=1.,
            opdMap=opd*factor, **common)[0]
        # The shared-grid interface changes detector scale with wavelength;
        # compare its complete two-wave call against the same explicitly scaled map.
        if not per_wavelength:
            expected = psdSetToPsfSet([psd], pupil, wavelengths, nPixPup=side,
                freq_range=8., dk=1., opdMap=opd*factor, **common)[i][0]
        np.testing.assert_array_equal(result[i][0].sampling, expected.sampling)
    np.testing.assert_array_equal(opd, before)


@pytest.mark.parametrize('map_kind', ['none', 'zero', 'constant'])
def test_constant_or_absent_map_preserves_existing_mastsel_result(map_kind):
    pupil, _ = inputs()
    opd = {'none': None, 'zero': np.zeros_like(pupil),
           'constant': np.full_like(pupil, 30.)}[map_kind]
    kwargs = dict(nPixPup=8, freq_range=8., dk=1., nPixPsf=32,
        opdMap=opd, full_flux=True, tip_tilt_removal='pupil_fit')
    power = np.zeros((32, 32))
    native = psdSetToPsfSet([power], pupil, 1.2e-6, **kwargs)[0]
    compensated = rms_compensated_psfs([power], pupil, 1.2e-6, **kwargs)[0]
    np.testing.assert_array_equal(native.sampling, compensated.sampling)


def test_compensation_requires_supported_pupil_fit_contract():
    pupil, opd = inputs()
    with pytest.raises(ValueError, match='requires pupil_fit'):
        rms_compensated_psfs([np.zeros((32,32))], pupil, 1.2e-6,
            nPixPup=8, freq_range=8., dk=1., nPixPsf=32, opdMap=opd,
            tip_tilt_removal='none')


def test_static_rms_scale_restores_measured_resampling_variance():
    from mastsel import congrid
    y,x=np.mgrid[:16,:16];pupil=((x-7.5)**2+(y-7.5)**2<7**2).astype(float)
    phase=.5*np.sin(2*x)+.3*np.cos(y)+.1*x
    before=phase.copy()
    factor=static_phase_rms_scale(pupil,phase,8)
    def variance(values,p):
        w=p**2;mean=np.sum(w*values)/w.sum()
        return np.sum(w*(values-mean)**2)/w.sum()
    np.testing.assert_allclose(variance(congrid(phase*factor,[8,8]),congrid(pupil,[8,8])),
        variance(phase,pupil),rtol=2e-15)
    np.testing.assert_array_equal(phase,before)
    assert factor>0 and not np.isclose(factor,1)


@pytest.mark.parametrize('level',[0.,.3])
def test_static_rms_scale_leaves_constant_phase_unchanged(level):
    assert static_phase_rms_scale(np.ones((8,8)),np.full((8,8),level),4)==1.


def test_static_rms_scale_rejects_complete_variance_loss():
    y,x=np.mgrid[:8,:8]
    with pytest.raises(ValueError,match='loses all phase variance'):
        static_phase_rms_scale(np.ones((8,8)),(-1.)**(x+y),2)




def test_rms_ratio_is_independent_of_phase_units():
    pupil, opd = inputs()
    expected = static_phase_rms_scale(pupil, opd, 8)
    actual = static_phase_rms_scale(pupil, opd*(2*np.pi*1e-9/1.2e-6), 8)
    np.testing.assert_allclose(actual, expected, rtol=2e-15)


def test_compensation_preserves_static_tilt_removal():
    pupil, _ = inputs()
    y, x = np.mgrid[:16,:16]
    options = dict(nPixPup=8, freq_range=8., dk=1., nPixPsf=32,
                   full_flux=True, tip_tilt_removal='pupil_fit')
    power = np.zeros((32,32))
    actual = rms_compensated_psfs([power],pupil,1.2e-6,
                                  opdMap=30*(x-7.5)+20*(y-7.5),**options)[0]
    expected = psdSetToPsfSet([power],pupil,1.2e-6,**options)[0]
    np.testing.assert_allclose(actual.sampling,expected.sampling,rtol=1e-11,atol=1e-14)


@pytest.mark.parametrize('value', [1, 'True', None, [True]])
def test_rms_option_rejects_non_boolean_config(tmp_path, value):
    from test_optional_products import _simulation
    with pytest.raises(ValueError, match='StaticOnRmsCompensation must be a boolean'):
        _simulation(tmp_path, False, config_updates={
            'sensor_HO': {'TipTiltRemoval': "'pupil_fit'"},
            'telescope': {'StaticOnRmsCompensation': repr(value)}})


def test_rms_option_rejects_unsupported_tilt_treatment(tmp_path):
    from test_optional_products import _simulation
    with pytest.raises(ValueError, match="requires TipTiltRemoval='pupil_fit'"):
        _simulation(tmp_path, False, config_updates={
            'sensor_HO': {'TipTiltRemoval': "'spectral'"},
            'telescope': {'StaticOnRmsCompensation': 'True'}})


def test_rms_option_defaults_off_and_changes_only_static_optics(tmp_path):
    from astropy.io import fits
    from test_optional_products import _simulation
    y, x = np.mgrid[:64,:64]
    path = tmp_path/'static.fits'
    fits.writeto(path,70*np.sin(2*x)*np.cos(1.1*y))
    results = []
    for label, flag in [('omitted',None), ('disabled',False), ('enabled',True)]:
        updates = {'sensor_HO': {'TipTiltRemoval': "'pupil_fit'"},
                   'telescope': {'PathStaticOn': repr(str(path))}}
        if flag is not None:
            updates['telescope']['StaticOnRmsCompensation'] = repr(flag)
        sim = _simulation(tmp_path/label,False,fullFluxNormalization=True,config_updates=updates)
        sim.doOverallSimulation(skipMerit=True,skipPSF1D=True)
        results.append(sim)
    omitted, disabled, enabled = results
    assert not omitted.staticOnRmsCompensation and not disabled.staticOnRmsCompensation
    assert enabled.staticOnRmsCompensation
    np.testing.assert_array_equal(omitted.cubeResultsArray,disabled.cubeResultsArray)
    for sim in results:
        np.testing.assert_array_equal(sim.opdMap,fits.getdata(path))
        for key in ('W','PSD','opticalPSD'):
            np.testing.assert_array_equal(getattr(sim.fao,key),getattr(omitted.fao,key))
    assert not np.array_equal(enabled.cubeResultsArray,disabled.cubeResultsArray)
