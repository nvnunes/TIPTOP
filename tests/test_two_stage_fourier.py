"""Native Fourier science mirrors retain TIPTOP's normal HO optical path."""
from configparser import ConfigParser
from tiptop.baseSimulation import baseSimulation
import numpy as np
import pytest
from test_optional_products import _simulation


_SETTINGS = {
    'sensor_HO': {'ReconstructionModel': "'alias_aware'", 'TipTiltRemoval': "'pupil_fit'",
                  'NoiseVariance': '[0., 0., 0.]', 'TomographicRegularizationFloor': '1e-6'},
    'DM_science': { 'NumberActuators': '[12]', 'DmPitchs': '[0.8]',
                  'DmHeights': '[0.]'},
    'RTC_science': {'LoopGain_HO': '1.', 'LoopDelaySteps_HO': '2'},
}


def build(path, lo_on, model="alias_aware", backend="numpy"):
    # Reuse the established small physical fixture; add the new INI section
    # before constructing the actual test simulation.
    _simulation(path, lo_on, fullFluxNormalization=True,
                config_updates={'sensor_HO': _SETTINGS['sensor_HO'],
                                'DM': {'NumberReconstructedLayers': '10'}})
    config = ConfigParser()
    config.optionxform = str
    config.read(path/'optional.ini')
    config['sensor_HO']['ReconstructionModel'] = repr(model)
    config['DM_science'] = _SETTINGS['DM_science']
    config['RTC_science'] = _SETTINGS['RTC_science']
    with (path/'optional.ini').open('w') as stream:
        config.write(stream)
    return baseSimulation(str(path), 'optional', str(path), 'unused',
                          fullFluxNormalization=True, getHoErrorBreakDown=False,
                          finiteApertureBackend=backend)


def test_native_coupled_psds_reach_existing_optics(tmp_path):
    simulation = build(tmp_path, False)
    simulation.doOverallSimulation(skipMerit=True, skipPSF1D=True)
    assert simulation.fao.twoStageEnabled
    assert simulation.fao.reconstructionModel == 'alias_aware'
    assert simulation.fao._finiteApertureReconstructor is None
    np.testing.assert_array_equal(simulation.PSD, simulation.fao.opticalPSD.transpose())
    assert np.isfinite(simulation.cubeResultsArray).all()
    np.testing.assert_allclose(simulation.cubeResultsArray.sum((-1,-2)), 1., atol=1e-12)


def test_unvalidated_lo_combination_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='HO-only'):
        build(tmp_path, True)


@pytest.mark.parametrize('backend', ['numpy', 'cuda', 'mps'])
def test_finite_backend_selection_reaches_p3(tmp_path, monkeypatch, backend):
    import importlib
    simulation = build(tmp_path, False, model='finite_aperture', backend=backend)
    module = importlib.import_module('tiptop.baseSimulation')
    original = module.fourierModel

    class PreparedModel(Exception):
        pass

    def construct(*args, **kwargs):
        # Construct real P3 state without executing device products.
        model = original(*args, **kwargs)
        assert model.finiteApertureBackend == backend
        assert model.twoStageEnabled
        raise PreparedModel

    monkeypatch.setattr(module, 'fourierModel', construct)
    with pytest.raises(PreparedModel):
        simulation._prepare_static_PSF_state(None)
