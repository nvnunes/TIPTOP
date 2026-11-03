"""Native HO optics shared by science and one-lenslet guide-star sensing.

The sensor uses the configured common-mirror reconstruction. Its atmosphere
is evaluated at the sensing wavelength and directions, with fitted pupil tilt
removed, and without science-arm static aberrations. No LO covariance is used
while forming these images. Directional science mirrors do not lie in the
guide-star sensing path. MASTSEL applies the resulting sensor noise and
anisoplanatic/temporal tilt covariance in the subsequent LO stage.
"""
from copy import deepcopy
from contextlib import nullcontext
import hashlib
from pathlib import Path
import time
import numpy as np
from p3.aoSystem.fourierModel import fourierModel
from mastsel import psdSetToPsfSet, Field
from .tiptopUtils import arrayP3toMastsel, cpuArray


class _FinitePupilPreparation:
    """One native model's most recent optical preparation; no result cache.

    P3 validates the complete reconstruction signature before reuse. Its backend
    identity plus exact pupil/phase bytes and optical dimensions own this cache.
    A changed input replaces the single entry, releasing its static/device
    optical objects. There is no global, cross-model or disk cache. P3 continues
    to own and invalidate atmospheric operators and resident covariance storage.
    """
    def __init__(self,key,backend,pupil,phase,diameter,n,side,period):
        from .finite_pupil import FinitePupilSampling
        self.key=key;self.backend=backend
        self.static=None
        if phase is not None:
            from .finite_static import FiniteStaticOptics
            self.static=FiniteStaticOptics(pupil,phase,diameter,n,side,period)
            self.sampling=self.static.sampling
        else:self.sampling=FinitePupilSampling(pupil,n,side,period)
        self.static_prepared=False
        self.device_optics=None


def _finite_pupil_preparation(native,pupil,phase,n,side,period):
    backend=native._finite_aperture_backend()
    diameter=float(native.ao.tel.D)
    digest=hashlib.sha256(repr((n,side,period,diameter,float(native.freq.wvlRef))).encode())
    for array in (pupil,phase):
        if array is None:digest.update(b'none')
        else:
            array=np.ascontiguousarray(array)
            digest.update(str((array.shape,array.dtype.str)).encode())
            digest.update(array.view(np.uint8))
    key=digest.digest()
    cached=getattr(native,'_finitePupilPreparation',None)
    if cached is None or cached.key!=key or cached.backend is not backend:
        previous=cached
        cached=_FinitePupilPreparation(key,backend,pupil,phase,diameter,n,side,period)
        # A map-only change replaces static weights, not covariance sampling.
        # Preserve its identity when P3 is reusing the same optical geometry.
        if previous is not None:
            names=('n','native_n','native_side','period','signature')
            if all(getattr(previous.sampling,name)==getattr(cached.sampling,name) for name in names):
                cached.sampling=previous.sampling
                if cached.static is not None:cached.static.sampling=cached.sampling
        native._finitePupilPreparation=cached
    return cached


def finite_pupil_fields(native, pupil, opd, n, side, period, fields):
    """Evaluate finite covariance; reuse one model's exactly matching preparation.

    Field results are always recalculated. Callers may repeat a prediction or
    request another subset on the same native model without recreating optics.
    Different wavelengths/models do not share preparation through this path.
    """
    wavelength = float(native.freq.wvlRef)
    phase = None if opd is None else cpuArray(opd)*(2*np.pi*1e-9/wavelength)
    if phase is not None and not np.any(phase):phase=None
    prepared=_finite_pupil_preparation(native,pupil,phase,n,side,period)
    static=prepared.static;sampling=prepared.sampling
    device_optics=prepared.device_optics
    images, profiles = [], []
    fields=tuple(fields)
    backend=prepared.backend;device=None
    if getattr(native,'finiteApertureBackend',None) in ('mps', 'cuda'):
        backend.prepare_optics(sampling,dense_source=False)
        device=getattr(backend,'device',None)
    pipeline=(backend.directional_preparation(fields) if device is not None
              and getattr(native, 'twoStageEnabled', False)
              and len(fields)>1 and not getattr(native,'getErrorBreakDown',False) else nullcontext())
    with pipeline:
        for position,field in enumerate(fields):
            started = time.perf_counter()
            batch_seconds=0.
            if device is not None and position%device.direction_batch_size==0:
                tick=time.perf_counter()
                device.prefetch(fields[position:position+device.direction_batch_size])
                batch_seconds=time.perf_counter()-tick
            residual, profile = native.finite_aperture_residual(sampling, field)
            if batch_seconds:
                profile=dict(profile,direction_batch_query_seconds=batch_seconds,
                             direction_batch_size=min(device.direction_batch_size,len(fields)-position))
            backend=native._finite_aperture_backend() if getattr(native,'finiteApertureBackend',None) in ('mps', 'cuda') else None
            device=getattr(backend,'device',None)
            if device is not None:
                if static is not None and not prepared.static_prepared:
                    seconds=static.prepare_high(backend)
                    profile=dict(profile,static_optics_preparation_seconds=static.preparation_seconds,
                                 static_source_preparation_seconds=seconds)
                    prepared.static_prepared=True
                if device_optics is None or device_optics.owner is not device:
                    if native.finiteApertureBackend == "cuda":
                        from .finite_cuda_optics import CudaPupilIntegrator as DevicePupilIntegrator
                    else:
                        from .finite_device_optics import DevicePupilIntegrator
                    amplitude=sampling.amplitude if static is None else static.amplitude
                    phasor=sampling._static_phasor if static is None else static.phasor
                    device_optics=DevicePupilIntegrator(device,sampling,amplitude,phasor)
                    prepared.device_optics=device_optics
                resident=device.resident_covariance(residual)
                otf,checks=device_optics.integrate(resident)
                tick=time.perf_counter()
                if static is None:
                    from scipy import fft
                    image=fft.fftshift(fft.fft2(otf)).real/sampling.n**2
                    optical_checks=dict(full_flux=float(image.sum()),minimum_pixel=float(image.min()))
                    if optical_checks['minimum_pixel'] < -1e-11 or abs(optical_checks['full_flux']-1)>2e-12:
                        raise ValueError('Resident optics failed positivity or full-flux validation')
                else:
                    image,optical_checks=static.correction.from_otf(otf)
                checks=dict(checks,**optical_checks,static_transfer_seconds=time.perf_counter()-tick)
            elif static is None:
                image, checks = sampling.psf_blocks(residual.diagonal, residual.blocks())
            else:
                if not prepared.static_prepared:
                    seconds = static.prepare_high(native._finite_aperture_backend())
                    profile = dict(profile, static_optics_preparation_seconds=static.preparation_seconds,
                                   static_source_preparation_seconds=seconds)
                    prepared.static_prepared = True
                image, checks = static.psf_blocks(residual.diagonal, residual.blocks())
            if getattr(native, 'getErrorBreakDown', False):
                budget = native.finiteErrorBreakDown[field]
                if static is not None:
                    budget = budget.with_optics(ncpa_nm2=static.static_variance*(wavelength*1e9/(2*np.pi))**2)
                    native.finiteErrorBreakDown[field] = budget
                optical = checks['residual_phase_variance']*(wavelength*1e9/(2*np.pi))**2
                tolerance = 2e-5 if native.finiteApertureBackend == 'mps' else 1e-8
                if not np.isclose(budget.dynamic_ho_nm2, optical, rtol=tolerance, atol=1e-6):
                    raise ValueError('Finite error budget does not close against the optical residual')
                profile = dict(profile, error_budget_optical_closure_nm2=budget.dynamic_ho_nm2-optical)
            result = Field(wavelength, n, period)
            result.sampling = image
            result.residual_phase_variance = checks['residual_phase_variance']
            images.append(result)
            profiles.append(dict(seconds=time.perf_counter()-started, **profile, **checks))
    # No map-amplitude rescaling is applied by the original-static-OTF path.
    return images, profiles, 1.


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
