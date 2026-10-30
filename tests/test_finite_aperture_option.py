"""Finite-aperture opt-in is explicit and rejects incompatible output modes."""

import pytest

from test_optional_products import _simulation


@pytest.mark.parametrize('option', [None, False, True])
def test_finite_aperture_option_is_omitted_or_boolean(tmp_path, option):
    sensor = {'TipTiltRemoval': "'pupil_fit'"}
    if option is not None:
        sensor['FiniteApertureReconstruction'] = repr(option)
    simulation = _simulation(tmp_path, False, fullFluxNormalization=True,
        config_updates={'sensor_HO': sensor})
    assert simulation.finiteApertureReconstruction is (option is True)
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
                'TipTiltRemoval': repr(method), 'FiniteApertureReconstruction': 'True'}})


@pytest.mark.parametrize('value', ['1', "'True'"])
def test_finite_aperture_rejects_non_boolean_values(tmp_path, value):
    with pytest.raises(ValueError, match='must be boolean'):
        _simulation(tmp_path, False, fullFluxNormalization=True,
            config_updates={'sensor_HO': {
                'TipTiltRemoval': "'pupil_fit'", 'FiniteApertureReconstruction': value}})
