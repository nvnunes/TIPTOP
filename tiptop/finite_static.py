"""Original static diffraction with coarse, nonstationary atmospheric optics."""
import time

import numpy as np
from scipy import fft

from .finite_pupil import FinitePupilSampling, _native_pairs
from .static_map import StaticOtfCorrection, averaged_static_pupil


class FiniteStaticOptics:
    """Integrate atmospheric coherence with complex-cell static weights.

    Atmospheric covariance is evaluated at the unchanged native optical pupil.
    Original-pupil deterministic phase and tilt fitting define complex cell
    averages. Their pair-weighted atmospheric transfer multiplies the original
    static OTF. This last separation is approximate for nonstationary coherence;
    it is exact for zero atmospheric covariance, not for arbitrary covariance.
    Invalid final intensities raise rather than being clipped.
    """

    def __init__(self, pupil, phase, diameter, n, side, period):
        started = time.perf_counter()
        self.sampling = FinitePupilSampling(pupil, n, side, period)
        s = self.sampling
        wave = averaged_static_pupil(pupil, phase, diameter, period, n)
        wave *= s.pupil != 0
        selected = wave[s.ys, s.xs]
        self.amplitude = abs(selected)
        self.phasor = np.divide(selected, self.amplitude, out=np.ones_like(selected),
                                where=self.amplitude > 0)
        self.power = float(np.sum(self.amplitude**2))
        self.correction = StaticOtfCorrection(pupil, phase, diameter, period, n, side,
                                               coherent_pupil=wave, output_n=n)
        original = self.correction.original
        fitted = original.static_phase
        self.static_variance = float(original.intensity @ (fitted-original.intensity @ fitted)**2)
        self.preparation_seconds = time.perf_counter() - started

    def prepare_high(self, backend):
        """Factor the known stationary source part without dropping its cross terms."""
        from p3.aoSystem.finiteAperture import _fold
        from p3.aoSystem.atmosphericTail import AtmosphericTail

        started = time.perf_counter()
        fine = self.correction.original.n
        power = backend.raw.sum(0) * backend.dk**2
        power *= np.hypot(backend.qx, backend.qy) > backend.native.freq.kcMax_
        original = fft.ifft2(_fold(power, fine)).real * fine**2
        original += AtmosphericTail(backend.native.ao.atm.r0, backend.native.ao.atm.L0,
            backend.dk, backend.count, fine, symmetric_source=True).covariance()
        if backend.native.ao.tel.extraErrorNm > 0:
            additional = backend.native.extraErrorPSD() * backend.aux_conversion
            original += fft.ifft2(_fold(additional, fine)).real * fine**2
        coarse = backend.stationary_high_lags[0]
        lags = self.correction.lags % fine
        self.correction.stationary_factor = (np.exp(coarse[0, 0] - coarse),
            np.exp(original[lags[:, None], lags[None]] - original[0, 0]))
        return time.perf_counter() - started

    def psf_blocks(self, diagonal, blocks, complete=False):
        """Consume the complete physical covariance, without a diagonal-PSD shortcut."""
        total_started = time.perf_counter()
        s = self.sampling
        diagonal = np.asarray(diagonal)
        if diagonal.shape != (len(s.xs),) or not np.isfinite(diagonal).all():
            raise ValueError("Residual diagonal must match finite optical pupil points")
        coherence = np.zeros(s._lag_count, complex)
        expected = 0
        minimum = 0.
        quadratic = 0.
        assembly = 0.
        integration = 0.
        iterator = iter(blocks)
        while True:
            started = time.perf_counter()
            try:
                first, covariance = next(iterator)
            except StopIteration:
                break
            assembly += time.perf_counter() - started
            covariance = np.asarray(covariance)
            if (first != expected or covariance.ndim != 2 or
                    covariance.shape[1] != len(diagonal) - first or
                    not 0 < len(covariance) <= len(diagonal) - first or
                    not np.isfinite(covariance).all()):
                raise ValueError("Covariance blocks must cover consecutive upper row strips")
            started = time.perf_counter()
            sf, value = _native_pairs(covariance, diagonal, first, s.ys, s.xs,
                self.amplitude, self.phasor, s.intensity,
                s._lag_y[0], s._lag_x[0], len(s._lag_x), coherence)
            integration += time.perf_counter() - started
            minimum = min(minimum, sf)
            quadratic += value
            expected += len(covariance)
        if expected != len(diagonal) or minimum < -2e-7:
            raise ValueError("Incomplete or nonphysical finite residual covariance")
        otf = np.zeros(s.n**2, complex)
        np.add.at(otf, s._lag_fft_indices, coherence)
        otf = otf.reshape(s.n, s.n) / self.power
        transfer_started = time.perf_counter()
        image, checks = self.correction.from_otf(otf, complete=complete)
        transfer_seconds = time.perf_counter() - transfer_started
        variance = float(s.intensity @ diagonal - quadratic)
        return image, dict(**checks, minimum_structure_function=minimum,
            residual_phase_variance=variance, covariance_block_seconds=assembly,
            coherence_seconds=integration, static_transfer_seconds=transfer_seconds,
            optics_seconds=time.perf_counter()-total_started)
