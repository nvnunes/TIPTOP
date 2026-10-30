"""Frequency-local shifted sampling and alias-aware reconstruction use native optics."""

import numpy as np
import pytest

from test_optional_products import _simulation


def test_tomographic_floor_reaches_native_p3(tmp_path):
    simulation = _simulation(tmp_path, False, fullFluxNormalization=True,
        config_updates={'sensor_HO': {'TomographicRegularizationFloor': '1e-6'}})
    simulation.doOverallSimulation(skipMerit=True, skipPSF1D=True)
    assert simulation.fao.tomoRelRegFloor == 1e-6
    from p3.aoSystem.fourierModel import fourierModel
    assert fourierModel.tomoRelRegFloor == 1e-12
    assert np.all(np.isfinite(simulation.cubeResultsArray))
    np.testing.assert_allclose(simulation.cubeResultsArray.sum(axis=(-2, -1)), 1., atol=1e-12)


@pytest.mark.parametrize('noise', [[0., 0., 0.], [.01, .02, .03], [0., .02, .03], [None]])
def test_xy_alias_aware_polc_uses_native_pupil_fit_optics(tmp_path, noise):
    simulation = _simulation(tmp_path, False, fullFluxNormalization=True,
        config_updates={
            'sensor_HO': {
                'TipTiltRemoval': "'pupil_fit'", 'NoiseVariance': repr(noise),
                'GridOffsets': '[[-.23, -.23], [.23, -.23], [0., .23]]',
                'SuperResolution': 'False', 'TomographicAliasing': 'True',
                'AliasAwareReconstruction': 'True', 'SymmetricAliasing': 'True',
                'TomographicRegularizationFloor': '1e-6', 'ExposureTime': '0.0'},
            'RTC': {'TemporalModel_HO': "'polc'"}})
    simulation.doOverallSimulation(skipMerit=True, skipPSF1D=True)
    model = simulation.fao
    assert model.aliasAwareReconstruction and model.symmetricAliasing
    assert model.tomographicAliasing and not model.superResolution
    assert model.W.shape[-1] == 2*model.nGs and model.tomoRelRegFloor == 1e-6
    assert model.temporalModel == 'polc' and model.exposureTime == 0
    assert model.computeOpticalPSD and simulation.tipTiltRemoval == 'pupil_fit'
    assert not simulation.LOisOn and simulation.mLO is None
    np.testing.assert_array_equal(model._measurement_noise_variances(),
                                  np.repeat(model.ao.wfs.processing.noiseVar, 2))
    # Reduced-memory native execution releases the component PSD arrays.
    assert bool(np.any(model.wfeN)) == (noise != [0., 0., 0.])
    np.testing.assert_array_equal(simulation.PSD, model.opticalPSD.transpose())
    assert np.isfinite(model.PSD).all() and np.isfinite(simulation.PSD).all()
    np.testing.assert_allclose(simulation.cubeResultsArray.sum(axis=(-2, -1)), 1., atol=1e-12)
