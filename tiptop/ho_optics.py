"""Native HO optics shared by science and one-lenslet guide-star sensing.

The sensor uses the configured common-mirror reconstruction. Its atmosphere
is evaluated at the sensing wavelength and directions, with fitted pupil tilt
removed, and without science-arm static aberrations. No LO covariance is used
while forming these images. Directional science mirrors do not lie in the
guide-star sensing path. MASTSEL applies the resulting sensor noise and
anisoplanatic/temporal tilt covariance in the subsequent LO stage.
"""
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
from p3.aoSystem.fourierModel import fourierModel
from mastsel import psdSetToPsfSet, Field
from .tiptopUtils import arrayP3toMastsel, cpuArray


def finite_pupil_fields(native, pupil, opd, n, side, period, fields):
    """Evaluate the finite residual covariance without a diagonal PSD handoff."""
    from .finite_pupil import FinitePupilSampling
    from .static_phase import static_phase_rms_scale
    wavelength = float(native.freq.wvlRef)
    phase = None if opd is None else cpuArray(opd)*(2*np.pi*1e-9/wavelength)
    scale = 1. if phase is None else static_phase_rms_scale(pupil, phase, side)
    sampling = FinitePupilSampling(pupil, n, side, period,
        static_phase=None if phase is None else phase*scale)
    images, profiles = [], []
    for field in fields:
        started = time.perf_counter()
        residual, profile = native.finite_aperture_residual(sampling, field)
        image, checks = sampling.psf_blocks(residual.diagonal, residual.blocks())
        result = Field(wavelength, n, period)
        result.sampling = image
        result.residual_phase_variance = checks['residual_phase_variance']
        images.append(result)
        profiles.append(dict(seconds=time.perf_counter()-started, **profile, **checks))
    return images, profiles, scale


def guide_star_optics(simulation):
    """Build a normal native HO model at the NGS wavelength and directions."""
    started = time.perf_counter()
    config = deepcopy(simulation.my_data_map)
    # Guide stars see the common mirror, before any directional science arm.
    config.pop('DM_science', None)
    config.pop('RTC_science', None)
    wavelength = float(simulation.LO_wvl)
    if np.any(np.asarray(config['telescope'].get('extraErrorLoNm', 0.)) != 0):
        raise ValueError('Pupil-fit NGS optics has not validated sensor-specific extraErrorLoNm')
    config['sources_science'] = dict(Wavelength=[wavelength],
        Zenith=list(simulation.LO_zen_field), Azimuth=list(simulation.LO_az_field))
    config['sensor_science']['PixelScale'] = simulation.psInMas*wavelength/simulation.wvlRef
    config['telescope']['PathStaticOn'] = ''
    config['telescope']['PathStaticOff'] = ''
    config['telescope']['StaticOnRmsCompensation'] = False
    config['telescope']['extraErrorNm'] = 0.
    config['telescope']['extraErrorLoNm'] = 0.
    for section in ('sources_LO', 'sensor_LO', 'sources_Focus', 'sensor_Focus'):
        config.pop(section, None)
    config['telescope']['glFocusOnNGS'] = False
    native = fourierModel(simulation.fullPathFilename, calcPSF=False,
        verbose=False, display=False, getPSDatNGSpositions=False,
        computeFocalAnisoCov=False, TiltFilter=False, computeOpticalPSD=True,
        getErrorBreakDown=False, doComputations=False, psdExpansion=True,
        reduce_memory=not simulation.finiteApertureReconstruction,
        path_root=str(Path(__file__).resolve().parent.parent),
        config_dict=config)
    native.initComputations()
    n = int(native.freq.nOtf)
    freq_range = n*native.freq.PSDstep
    side = int(2*np.round(simulation.tel_radius*freq_range))
    period = 1/float(native.freq.kxAO_[1,0]-native.freq.kxAO_[0,0])
    pupil = cpuArray(native.ao.tel.pupil)
    profiles = None
    if simulation.finiteApertureReconstruction:
        images, profiles, _ = finite_pupil_fields(native, pupil, None, n, side,
            period, range(len(simulation.LO_zen_field)))
    else:
        images = psdSetToPsfSet(arrayP3toMastsel(native.opticalPSD.transpose()),
            arrayP3toMastsel(pupil), wavelength, nPixPup=side,
            freq_range=freq_range, dk=native.freq.dk_, nPixPsf=n,
            oversampling=1., full_flux=True, tip_tilt_removal='pupil_fit',
            tilt_fit_pupil=arrayP3toMastsel(pupil))
    return images, native, dict(seconds=time.perf_counter()-started,
        wavelength_m=wavelength, pixel_scale_mas=float(config['sensor_science']['PixelScale']),
        static_map_applied=False, full_flux=True, finite_profiles=profiles,
        correction='common', science_mirror_applied=False,
        image_size=n, pupil_size=side)
