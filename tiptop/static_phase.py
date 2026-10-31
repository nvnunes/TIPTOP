"""Static-map resampling policy for TIPTOP's pupil-fit science optics.

The measured RMS ratio compensates amplitude lost in interpolation, not its
spatial spectrum. Input OPD maps and atmospheric spectra are never modified.
MASTSEL receives an ordinary map through its existing public optics API.
"""

import numpy as np
from mastsel import congrid, psdSetToPsfSet
from .tiptopUtils import cpuArray


def static_phase_rms_scale(pupil, phase, pupil_pixels):
    """Return original/resampled intensity-weighted, piston-removed RMS.

    Pupil is a nonempty square amplitude mask; phase is a matching real finite
    map in phase or OPD units. Zero/piston maps return one. A nonconstant map
    whose variance disappears on the target grid raises ValueError. The caller
    applies the factor before fitting static tilt on the original pupil.
    """
    pupil = np.asarray(cpuArray(pupil))
    phase = np.asarray(cpuArray(phase))
    if (pupil.ndim != 2 or pupil.shape[0] != pupil.shape[1] or
            not np.isrealobj(pupil) or not np.isfinite(pupil).all() or
            not np.any(pupil) or pupil.shape != phase.shape or
            not np.isrealobj(phase) or not np.isfinite(phase).all()):
        raise ValueError('Static phase must be finite and match a nonempty square fit pupil')

    def variance(values, amplitude):
        weights = amplitude**2
        mean = np.sum(weights*values)/np.sum(weights)
        return np.sum(weights*(values-mean)**2)/np.sum(weights)

    original = variance(phase, pupil)
    if original <= np.finfo(float).eps**2*max(1.,float(np.max(abs(phase)))**2):
        return 1.
    coarse = np.asarray(cpuArray(congrid(phase, [pupil_pixels]*2)))
    sampled = variance(coarse, np.asarray(cpuArray(congrid(pupil, [pupil_pixels]*2))))
    if sampled <= 0:
        raise ValueError('Static map loses all phase variance on the native optical pupil')
    return float(np.sqrt(original/sampled))


def rms_compensated_psfs(input_psds, pupil, wavelength, *, nPixPup, opdMap=None, **options):
    """Call native MASTSEL optics with measured static RMS compensation.

    This TIPTOP policy requires pupil-fit tilt removal and a common telescope
    pupil. Shared-grid wavelengths use one scaled map. Per-wavelength grids
    receive independently scaled maps and retain MASTSEL's exact decimation.
    Zero/omitted maps preserve native output. Remaining arguments use the
    existing psdSetToPsfSet contract. No atmospheric error power is added here.
    """
    if options.get('tip_tilt_removal') != 'pupil_fit' or isinstance(pupil, list):
        raise ValueError('Static RMS compensation requires pupil_fit and one common pupil mask')
    if opdMap is None or not np.any(cpuArray(opdMap)):
        return psdSetToPsfSet(input_psds, pupil, wavelength, nPixPup=nPixPup,
                             opdMap=opdMap, **options)
    fit_pupil = options.get('tilt_fit_pupil')
    if fit_pupil is None:
        fit_pupil = pupil
    sides = np.atleast_1d(nPixPup)
    if len(sides) == 1:
        scale = static_phase_rms_scale(fit_pupil, opdMap, sides[0])
        return psdSetToPsfSet(input_psds, pupil, wavelength, nPixPup=nPixPup,
                             opdMap=opdMap*scale, **options)

    wavelengths = np.atleast_1d(wavelength)
    first = input_psds[0]
    per_wavelength = isinstance(first, (list, tuple)) or np.ndim(first) >= 3
    if (not per_wavelength or len(sides) != len(wavelengths) or
            len(input_psds) != len(wavelengths)):
        raise ValueError('Per-wavelength pupil widths and spectra must match wavelengths')
    frequency_range = np.broadcast_to(np.atleast_1d(options['freq_range']), wavelengths.shape)
    spacing = np.broadcast_to(np.atleast_1d(options['dk']), wavelengths.shape)
    results = []
    for i, (wave, side) in enumerate(zip(wavelengths, sides)):
        scale = static_phase_rms_scale(fit_pupil, opdMap, side)
        local_options = dict(options, freq_range=[frequency_range[i]], dk=[spacing[i]])
        # Preserve the per-wavelength calling convention even for this one wave.
        results.append(psdSetToPsfSet([input_psds[i]], pupil, [wave], nPixPup=[side],
            opdMap=opdMap*scale, **local_options))
    return results if len(wavelengths) > 1 else results[0]
