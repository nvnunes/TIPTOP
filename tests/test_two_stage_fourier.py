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
                                'telescope': {'glFocusOnNGS': 'False'},
                                'sensor_science': {'PixelScale': repr(1.65e-6*206264806.24709636/16)},
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


@pytest.mark.parametrize('model', ['alias_aware', 'finite_aperture'])
def test_lo_uses_common_ngs_and_preserves_science_ho(tmp_path, model):
    science = build(tmp_path/'science', True, model=model)
    science.doOverallSimulation(skipMerit=True, skipPSF1D=True)
    ho_only = build(tmp_path/'ho', False, model=model)
    ho_only.doOverallSimulation(skipMerit=True, skipPSF1D=True)
    assert science.twoStageEnabled and science.LOisOn
    assert not science.ngsOpticsProvenance['science_mirror_applied']
    for actual, expected in zip(science.psfLongExpPointingsArr, ho_only.psfLongExpPointingsArr):
        np.testing.assert_array_equal(actual.sampling, expected.sampling)
    assert np.isfinite(science.cubeResultsArray).all()
    np.testing.assert_allclose(science.cubeResultsArray.sum((-1,-2)), 1., atol=1e-12)
    assert np.linalg.eigvalsh(science._Ctot_temp).min() >= -1e-10


@pytest.mark.parametrize('model', ['alias_aware', 'finite_aperture'])
def test_science_off_recovers_ltao_lo(tmp_path, model):
    from configparser import ConfigParser
    instances=[]
    for name in ('disabled', 'absent'):
        path=tmp_path/name
        build(path, True, model=model)
        config=ConfigParser();config.optionxform=str;config.read(path/'optional.ini')
        if name=='disabled':
            config['RTC_science']['LoopGain_HO']='0.'
        else:
            config.remove_section('DM_science');config.remove_section('RTC_science')
        with (path/'optional.ini').open('w') as stream:config.write(stream)
        simulation=baseSimulation(str(path),'optional',str(path),'unused',
            fullFluxNormalization=True,getHoErrorBreakDown=False)
        simulation.doOverallSimulation(skipMerit=True,skipPSF1D=True)
        instances.append(simulation)
    for actual,expected in zip(instances[0].results,instances[1].results):
        np.testing.assert_array_equal(actual.sampling,expected.sampling)
    np.testing.assert_array_equal(instances[0]._Ctot_temp,instances[1]._Ctot_temp)
    for actual,expected in zip(instances[0].ngsPsfFields,instances[1].ngsPsfFields):
        np.testing.assert_array_equal(actual.sampling,expected.sampling)


@pytest.mark.parametrize('model', ['alias_aware', 'finite_aperture'])
@pytest.mark.parametrize('change,message', [
    ('normalization','fullFluxNormalization'), ('tilt','pupil_fit'),
    ('focus','focus'), ('lenslets','one-lenslet')])
def test_science_lo_scope_is_explicit(tmp_path, model, change, message):
    build(tmp_path, True, model=model)
    config=ConfigParser();config.optionxform=str;config.read(tmp_path/'optional.ini')
    if change=='tilt':config['sensor_HO']['TipTiltRemoval']="'spectral'"
    elif change=='focus':config['telescope']['glFocusOnNGS']='True'
    elif change=='lenslets':config['sensor_LO']['NumberLenslets']='[2, 2, 2]'
    with (tmp_path/'optional.ini').open('w') as stream:config.write(stream)
    with pytest.raises(ValueError,match=message):
        baseSimulation(str(tmp_path),'optional',str(tmp_path),'unused',
            fullFluxNormalization=change!='normalization',getHoErrorBreakDown=False)


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
