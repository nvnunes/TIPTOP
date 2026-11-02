"""The existing reporting flag exposes finite budgets without changing optics."""
from configparser import ConfigParser

import numpy as np
import pytest

from tiptop.baseSimulation import baseSimulation
from test_two_stage_fourier import build


@pytest.mark.parametrize('science', [False, True])
def test_opt_in_finite_budget_closes_and_preserves_psfs(tmp_path, science):
    build(tmp_path, False, model='finite_aperture')
    if not science:
        config = ConfigParser(); config.optionxform = str
        config.read(tmp_path/'optional.ini')
        config.remove_section('DM_science'); config.remove_section('RTC_science')
        with (tmp_path/'optional.ini').open('w') as stream:
            config.write(stream)
    sim = baseSimulation(str(tmp_path), 'optional', str(tmp_path), 'unused',
        fullFluxNormalization=True, getHoErrorBreakDown=True)
    with pytest.raises(ValueError, match='evaluate science optics'):
        sim.errorBreakDown(verbose=False)
    sim.doOverallSimulation(skipMerit=True, skipPSF1D=True)
    budgets = sim.hoErrorBreakDown
    assert len(budgets) == sim.nPointings
    np.testing.assert_allclose([r.dynamic_ho_nm2 for r in budgets.values()],
                               np.asarray(sim.HO_res)**2, rtol=1e-8, atol=1e-6)
    for result in budgets.values():
        assert result.stage == ('moao' if science else 'ltao')
    before = np.array([field.sampling for field in sim.psfLongExpPointingsArr])
    sim.getHoErrorBreakDown = sim.fao.getErrorBreakDown = False
    after = sim._finite_aperture_psfs()
    np.testing.assert_array_equal(before, [field.sampling for field in after])
    assert sim.hoErrorBreakDown == {}


def test_budget_keeps_native_lo_reporting_separate(tmp_path, capsys):
    sim = build(tmp_path, True, model='finite_aperture')
    sim.getHoErrorBreakDown = True
    sim.verbose = True
    sim.doOverallSimulation(skipMerit=True, skipPSF1D=True)
    records = sim.errorBreakDown()
    for field, record in records.items():
        ho = sim.hoErrorBreakDown[field]
        assert record is ho
        np.testing.assert_allclose(record.total_nm2,
            sim.HO_res[field]**2+record.ncpa_nm2, rtol=1e-8)
    output = capsys.readouterr().out
    assert '.Aliasing error:' in output
    assert 'Tip-tilt error (MASTSEL)' not in output
    assert 'LO_res [nm]:' in output
    assert 'HO_res [nm]:' in output
    assert 'LTAO: first stage' in output
    assert 'MOAO: both stages' in output
    for record in records.values():
        assert record.first_stage.ncpa_nm2 == record.ncpa_nm2
    assert 'Fitting–aliasing covariance [nm²]:' in output

    # HO reporting does not depend on LO being available or on its magnitude.
    sim.LO_res = None
    assert sim.errorBreakDown(verbose=False) == records


def test_original_map_optics_attaches_static_variance_without_changing_psfs(tmp_path):
    from astropy.io import fits
    from tiptop.baseSimulation import baseSimulation
    build(tmp_path, False, model='finite_aperture')
    y, x = np.mgrid[:64, :64]
    path = tmp_path/'opd.fits'
    fits.writeto(path,20*np.sin(x/7)+10*np.cos(y/9))
    config = ConfigParser(); config.optionxform = str
    config.read(tmp_path/'optional.ini')
    config['telescope']['PathStaticOn'] = repr(str(path))
    with (tmp_path/'optional.ini').open('w') as stream:
        config.write(stream)
    sim = baseSimulation(str(tmp_path),'optional',str(tmp_path),'unused',
        fullFluxNormalization=True,getHoErrorBreakDown=True)
    sim.doOverallSimulation(skipMerit=True,skipPSF1D=True)
    before = np.array([field.sampling for field in sim.psfLongExpPointingsArr])
    assert all(record.ncpa_nm2 > 0 for record in sim.hoErrorBreakDown.values())
    assert all(record.first_stage.ncpa_nm2 == record.ncpa_nm2
               for record in sim.hoErrorBreakDown.values())
    sim.fao.getErrorBreakDown = False
    after = sim._finite_aperture_psfs()
    np.testing.assert_array_equal(before,[field.sampling for field in after])
