"""Skipping the NGS pixel floor is opt-in and preserves HO optics."""
import numpy as np
import pytest

from test_optional_products import _simulation


@pytest.mark.parametrize('value', ['1', "'False'", '[False]'])
def test_ngs_energy_policy_requires_boolean(tmp_path, value):
    with pytest.raises(ValueError, match='SkipEncircledEnergyPixelFloor must be a boolean'):
        _simulation(tmp_path, True, config_updates={
            'sensor_LO': {'SkipEncircledEnergyPixelFloor': value}})


@pytest.mark.parametrize('pupil_fit', [False, True])
def test_ngs_energy_policy_changes_only_lo_handoff(tmp_path, pupil_fit):
    simulations = []
    for name, option in [('omitted', None), ('floor', 'False'), ('skipped', 'True')]:
        sensor = {'PixelScale': '200.', 'NumberLenslets': '[1, 1, 1]'}
        if option is not None:
            sensor['SkipEncircledEnergyPixelFloor'] = option
        updates = {'sensor_LO': sensor, 'telescope': {'glFocusOnNGS': 'False'}}
        if pupil_fit:
            updates['sensor_HO'] = {'TipTiltRemoval': "'pupil_fit'"}
        simulation = _simulation(tmp_path / name, True, config_updates=updates,
                                 fullFluxNormalization=pupil_fit)
        simulation.doOverallSimulation(skipMerit=True, skipPSF1D=True)
        simulations.append(simulation)
    omitted, floor, skipped = simulations
    np.testing.assert_array_equal(omitted.cubeResultsArray, floor.cubeResultsArray)
    np.testing.assert_array_equal(floor.NGS_FWHM_mas_field, skipped.NGS_FWHM_mas_field)
    assert np.all(np.array(skipped.NGS_EE_field) < floor.NGS_EE_field)
    for a, b in zip(floor.psfLongExpPointingsArr, skipped.psfLongExpPointingsArr):
        np.testing.assert_array_equal(a.sampling, b.sampling)
    assert not np.array_equal(floor._Ctot_temp, skipped._Ctot_temp)
    if pupil_fit:
        for a, b in zip(floor.ngsPsfFields, skipped.ngsPsfFields):
            np.testing.assert_array_equal(a.sampling, b.sampling)
