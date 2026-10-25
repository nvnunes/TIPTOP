"""Optional science metrics and profiles preserve native single-stage PSFs."""

import importlib
from configparser import ConfigParser
from pathlib import Path

import numpy as np
import pytest

from tiptop.baseSimulation import baseSimulation


def _simulation(path, lo_on, *, config_updates=None, **kwargs):
    path.mkdir(exist_ok=True, parents=True)
    parser = ConfigParser()
    parser.optionxform = str
    parser.read(Path(__file__).parents[1] / 'tiptop/perfTest/MAVIStest.ini')
    changes = {
        'telescope': {'Resolution': '64'},
        'sources_science': {'Wavelength': '[1.65e-6]', 'Zenith': '[0, 15, 15]',
                            'Azimuth': '[0, 0, 90]'},
        'sources_HO': {'Zenith': '[17.5, 17.5, 17.5]', 'Azimuth': '[0, 120, 240]'},
        'sensor_science': {'PixelScale': '20.0', 'FieldOfView': '64'},
        'sensor_HO': {'NumberLenslets': '[12, 12, 12]', 'NumberPhotons': '[75, 75, 75]'},
        'DM': {'NumberActuators': '[13]', 'DmPitchs': '[0.6666666666666666]',
               'InfCoupling': '[0.2]', 'DmHeights': '[0]',
               'OptimizationZenith': '[0, 15, 15]', 'OptimizationAzimuth': '[0, 0, 90]',
               'OptimizationWeight': '[1, 1, 1]', 'NumberReconstructedLayers': '3'},
    }
    for section, values in changes.items():
        for key, value in values.items():
            parser.set(section, key, value)
    if not lo_on:
        parser.remove_section('sensor_LO')
        parser.remove_section('sources_LO')
    for section, values in (config_updates or {}).items():
        for key, value in values.items():
            parser.set(section, key, value)
    with (path / 'optional.ini').open('w') as stream:
        parser.write(stream)
    return baseSimulation(str(path), 'optional', str(path), 'unused', doPlot=False,
                          getHoErrorBreakDown=True, **kwargs)


@pytest.fixture(scope='module', params=[False, True], ids=['ho-only', 'ho-lo'])
def native_case(request, tmp_path_factory):
    path = tmp_path_factory.mktemp('optional-products')
    baseline = _simulation(path / 'baseline', request.param)
    baseline.doOverallSimulation()
    return path, request.param, baseline


@pytest.mark.parametrize('skip_merit,skip_profile',
                         [(False, False), (True, False), (False, True), (True, True)])
def test_optional_products_preserve_psfs_and_residuals(native_case, monkeypatch,
                                                      skip_merit, skip_profile):
    path, lo_on, baseline = native_case
    simulation = _simulation(path / f'{skip_merit}-{skip_profile}', lo_on)
    if skip_merit:
        module = importlib.import_module('tiptop.baseSimulation')
        get_fwhm = module.getFWHM

        def ngs_fwhm_only(*args, **kwargs):
            assert 'rebin' not in kwargs, 'Unused science FWHM was computed'
            return get_fwhm(*args, **kwargs)

        monkeypatch.setattr(module, 'getFWHM', ngs_fwhm_only)
    if skip_profile:
        def unexpected_profile():
            pytest.fail('Unused science radial profiles were computed')
        monkeypatch.setattr(simulation, 'computePSF1D', unexpected_profile)

    simulation.doOverallSimulation(skipMerit=skip_merit, skipPSF1D=skip_profile)

    for name in ('cubeResultsArray', 'PSD', 'HO_res', 'psf_ol_array', 'psf_dl_array'):
        np.testing.assert_array_equal(getattr(simulation, name), getattr(baseline, name))
    for actual, expected in zip(simulation.results, baseline.results):
        np.testing.assert_array_equal(actual.sampling, expected.sampling)
    if lo_on:
        for name in ('LO_res', 'NGS_EE_field', 'NGS_FWHM_mas_field'):
            np.testing.assert_array_equal(getattr(simulation, name), getattr(baseline, name))
    if skip_merit:
        assert simulation.pointings_FWHM_mas is None
    else:
        np.testing.assert_array_equal(simulation.pointings_FWHM_mas,
                                      baseline.pointings_FWHM_mas)
    if skip_profile:
        assert simulation.psf1d_data is None
    else:
        np.testing.assert_array_equal(simulation.psf1d_data, baseline.psf1d_data)


def test_default_call_restores_products_after_a_skipped_call(tmp_path):
    simulation = _simulation(tmp_path, False)
    simulation.doOverallSimulation(skipMerit=True, skipPSF1D=True)
    pixels = simulation.cubeResultsArray.copy()
    simulation.doOverallSimulation()
    np.testing.assert_array_equal(simulation.cubeResultsArray, pixels)
    assert simulation.pointings_FWHM_mas is not None
    assert simulation.psf1d_data is not None
