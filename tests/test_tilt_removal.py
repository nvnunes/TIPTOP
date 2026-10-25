"""Selectable native tilt removal through TIPTOP's native INI contract."""

import numpy as np
import pytest

from test_optional_products import _simulation


def _run(path, lo_on, option=None):
    updates = None if option is None else {'sensor_HO': {'TipTiltRemoval': repr(option)}}
    simulation = _simulation(path, lo_on, fullFluxNormalization=True, config_updates=updates)
    simulation.doOverallSimulation(skipMerit=True, skipPSF1D=True)
    return simulation


@pytest.mark.parametrize('lo_on', [False, True], ids=['ho-only', 'ho-lo'])
def test_default_matches_explicit_lo_dependent_filter(tmp_path, lo_on):
    default = _run(tmp_path/'default', lo_on)
    explicit = _run(tmp_path/'explicit', lo_on, 'spectral' if lo_on else 'none')
    for field in ('PSD', 'cubeResultsArray', 'HO_res', 'psf_ol_array', 'psf_dl_array'):
        np.testing.assert_array_equal(getattr(default, field), getattr(explicit, field))
    if lo_on:
        np.testing.assert_array_equal(default.LO_res, explicit.LO_res)
    assert default.fao.applyTiltFilter is lo_on


def test_filter_can_remove_tilt_without_creating_lo_model(tmp_path):
    unfiltered = _run(tmp_path/'unfiltered', False, 'none')
    filtered = _run(tmp_path/'filtered', False, 'spectral')
    assert not filtered.LOisOn and filtered.mLO is None
    assert filtered.fao.applyTiltFilter is True
    # No extra error is configured; the existing spectral filter is the only change.
    mask = np.asarray(filtered.fao.TiltFilter())
    np.testing.assert_allclose(filtered.PSD, unfiltered.PSD*mask[None], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(filtered.cubeResultsArray.sum(axis=(-2,-1)), 1., atol=1e-12)
    assert np.all(filtered.HO_res < unfiltered.HO_res)
    assert np.all(filtered.cubeResultsArray.max(axis=(-2,-1)) > unfiltered.cubeResultsArray.max(axis=(-2,-1)))


def test_explicit_false_retains_lo_calculation(tmp_path):
    simulation = _run(tmp_path, True, 'none')
    assert simulation.LOisOn and simulation.mLO is not None
    assert simulation.fao.applyTiltFilter is False
    assert np.all(np.isfinite(simulation.LO_res))


@pytest.mark.parametrize('value', ['bad', True, 1, ['pupil_fit']])
def test_invalid_tilt_filter_value_is_rejected(tmp_path, value):
    with pytest.raises(ValueError, match='sensor_HO.TipTiltRemoval'):
        _simulation(tmp_path, False, config_updates={'sensor_HO': {'TipTiltRemoval': repr(value)}})


@pytest.mark.parametrize('lo_on', [False, True])
def test_auto_matches_omitted_setting(tmp_path, lo_on):
    automatic = _run(tmp_path/'automatic', lo_on, 'auto')
    default = _run(tmp_path/'default', lo_on)
    np.testing.assert_array_equal(automatic.PSD, default.PSD)
    np.testing.assert_array_equal(automatic.cubeResultsArray, default.cubeResultsArray)


@pytest.mark.parametrize('lo_on', [False, True])
@pytest.mark.parametrize('ncpa_on', [False, True])
def test_pupil_fit_runs_native_optics_and_keeps_lo_independent(tmp_path, lo_on, ncpa_on):
    from astropy.io import fits
    updates = {'sensor_HO': {'TipTiltRemoval': repr('pupil_fit')}}
    if ncpa_on:
        y,x = np.mgrid[:64,:64]
        path = tmp_path/'static.fits'
        fits.writeto(path, 60*np.sin(x/8)*np.cos(y/8))
        updates['telescope'] = {'PathStaticOn': repr(str(path))}
    simulation = _simulation(tmp_path, lo_on, fullFluxNormalization=True, config_updates=updates)
    simulation.doOverallSimulation(skipMerit=True, skipPSF1D=True)
    assert simulation.fao.applyTiltFilter is False
    assert simulation.LOisOn is lo_on
    assert (simulation.mLO is not None) is lo_on
    assert np.all(np.isfinite(simulation.cubeResultsArray))
    np.testing.assert_allclose(simulation.cubeResultsArray.sum(axis=(-2,-1)), 1., atol=2e-6)
    variance = np.array([image.residual_phase_variance for image in simulation.psfLongExpPointingsArr])
    np.testing.assert_allclose(simulation.HO_res, np.sqrt(variance)*simulation.wvlRef*1e9/(2*np.pi))
    if lo_on:
        assert np.all(np.isfinite(simulation.LO_res))
        assert np.all(np.isfinite(simulation.NGS_SR_field))


def test_pupil_fit_supports_native_multiwavelength_grids(tmp_path):
    simulation = _simulation(tmp_path, False, fullFluxNormalization=True,
                              exactMultiWavelengthPSD=True, config_updates={
                                  'sensor_HO': {'TipTiltRemoval': repr('pupil_fit')},
                                  'sources_science': {'Wavelength': '[1.2e-6, 2.2e-6]'}})
    simulation.getHoErrorBreakDown = False
    simulation.doOverallSimulation(skipMerit=True, skipPSF1D=True)
    assert simulation.cubeResultsArray.shape[:2] == (2,3)
    np.testing.assert_allclose(simulation.cubeResultsArray.sum(axis=(-2,-1)),1.,atol=1e-12)


def test_retired_filter_boolean_fails_clearly(tmp_path):
    with pytest.raises(ValueError, match='replaced by sensor_HO.TipTiltRemoval'):
        _simulation(tmp_path,False,config_updates={'sensor_HO':{'FilterTipTilt':'True'}})
