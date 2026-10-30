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
