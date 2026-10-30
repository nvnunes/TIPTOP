"""Finite-aperture opt-in is explicit and rejects incompatible output modes."""

import pytest

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
