"""Full-plane science scaling through native TIPTOP optics and convolution."""

import importlib

import numpy as np
import pytest
from astropy.io import fits

from test_optional_products import _simulation


@pytest.mark.parametrize('lo_on,jitter', [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize('opd_on', [False, True])
def test_full_flux_survives_convolution_and_profiles(tmp_path, monkeypatch, lo_on, jitter, opd_on):
    module = importlib.import_module('tiptop.baseSimulation')
    convolve = module.convolve
    normalizations = []

    def record_convolution(*args, **kwargs):
        normalizations.append(kwargs.get('normalize_input', True))
        return convolve(*args, **kwargs)

    monkeypatch.setattr(module, 'convolve', record_convolution)
    updates = {'telescope': {}}
    if jitter:
        updates['telescope']['jitter_FWHM'] = '10.0'
    if opd_on:
        y, x = np.mgrid[:64, :64]
        path = tmp_path / 'static.fits'
        fits.writeto(path, 60 * np.sin(x / 8) * np.cos(y / 8))
        updates['telescope']['PathStaticOn'] = repr(str(path))
    simulation = _simulation(tmp_path, lo_on, fullFluxNormalization=True,
                              config_updates=updates)
    simulation.doOverallSimulation(skipMerit=True)
    # Hold the computed PSD and WFS state fixed while requesting a smaller
    # science field; changing the INI field also changes P3's Fourier grid.
    simulation.nPixPSF = 32
    lo_data = {'cov_ellipses': simulation.cov_ellipses} if lo_on else {}
    simulation._generate_final_PSF(None, lo_data)
    simulation._finalize_full_field_results(None)
    for index, (ho, final) in enumerate(zip(simulation.psfLongExpPointingsArr,
                                          simulation.results)):
        np.testing.assert_allclose(final.sampling.sum(), ho.sampling.sum(), rtol=2e-6)
        assert 0 < final.sampling.sum() < 1
        np.testing.assert_array_equal(simulation.cubeResultsArray[index], final.sampling)
    assert simulation.psf1d_data is not None
    if lo_on or jitter:
        assert normalizations and all(value is False for value in normalizations)
    else:
        assert normalizations == []


@pytest.mark.parametrize('wavelength', [0.485e-6, 0.67e-6, 0.865e-6, 1.215e-6, 1.65e-6, 2.195e-6])
def test_full_flux_overlapping_optical_pixels_are_field_independent(tmp_path, wavelength):
    simulation = _simulation(tmp_path, False, fullFluxNormalization=True,
                              config_updates={'sources_science': {'Wavelength': repr([wavelength])}})
    simulation.doOverallSimulation(skipMerit=True, skipPSF1D=True)
    wide = np.asarray([image.sampling.copy() for image in simulation.results])
    simulation.nPixPSF = 32
    simulation._generate_final_PSF(None, {})
    crop = np.asarray([image.sampling for image in simulation.results])
    np.testing.assert_array_equal(crop, wide[:, 16:48, 16:48])
    assert np.all((crop.sum(axis=(-2, -1)) > 0) & (crop.sum(axis=(-2, -1)) < 1))


def test_full_flux_can_skip_profiles_without_changing_pixels(tmp_path):
    simulation = _simulation(tmp_path, False, fullFluxNormalization=True,
                              config_updates={'sensor_science': {'FieldOfView': '32'}})
    simulation.doOverallSimulation(skipMerit=True)
    expected = simulation.cubeResultsArray.copy()
    simulation.doOverallSimulation(skipMerit=True, skipPSF1D=True)
    np.testing.assert_array_equal(simulation.cubeResultsArray, expected)
