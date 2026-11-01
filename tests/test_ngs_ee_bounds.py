"""Finite NGS encircled energy when the detector pixel exceeds the PSF field."""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.interpolate import interp1d

from tiptop.baseSimulation import baseSimulation


def _ngs_state(detector_scale):
    n, pupil_size = 64, 32
    wavelength, freq_range = 1.65e-6, 4.0
    y, x = np.mgrid[:pupil_size, :pupil_size] - (pupil_size - 1) / 2
    pupil = (np.hypot(x, y) < 15).astype(float)
    pixel_scale = wavelength * freq_range / n * (180 / np.pi) * 3600 * 1000
    return SimpleNamespace(
        psInMas=pixel_scale, LO_wvl=wavelength, wvlMax=wavelength,
        LO_psInMas=detector_scale, overSamp_lo=1, overSamp=1,
        nPixPSF=32, multiGridPSD=False, PSD=np.zeros((1, n, n)),
        nNaturalGS_field=1, freq_range=freq_range, dk=1.0, sx=pupil_size,
        fao=SimpleNamespace(freq=SimpleNamespace(k2_=np.zeros((n, n))),
                            ao=SimpleNamespace(tel=SimpleNamespace(pupil=pupil))),
        my_data_map={'sensor_LO': {'NumberLenslets': [1]}},
        tel_radius=4.0, opdMap=None, verbose=False, tipTiltRemoval='spectral',
        addLoAlias=False, addFocusError=False, _independent_ngs_optics=lambda: False,
    )


@pytest.mark.parametrize('as_list', [False, True])
@pytest.mark.parametrize('detector_scale', [20.0, 1000.0])
def test_ngs_ee_stays_in_the_available_profile(monkeypatch, as_list, detector_scale):
    module = importlib.import_module('tiptop.baseSimulation')
    state = _ngs_state([detector_scale] if as_list else detector_scale)
    profiles = []
    get_energy = module.getEncircledEnergy

    def record_profile(*args, **kwargs):
        energy, radius = get_energy(*args, **kwargs)
        profiles.append((energy.copy(), radius.copy()))
        return energy, radius

    monkeypatch.setattr(module, 'getEncircledEnergy', record_profile)
    baseSimulation._compute_ngs_psf(state)

    energy, radius = profiles[0]
    energy /= energy.max()
    result = state.NGS_EE_field[0]
    assert np.isfinite(result)
    if detector_scale > radius[-1]:
        assert result == pytest.approx(1.0)
    else:
        requested = max(state.NGS_FWHM_mas_field[0], detector_scale)
        expected = float(interp1d(radius, energy, kind='cubic')(requested))
        assert result == pytest.approx(expected)
