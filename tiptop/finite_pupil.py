"""Gaussian optics for a nonstationary, fitted-pupil phase covariance.

This sampling contract is independent of the reconstruction that supplies the
covariance. Optical downsampling changes quadrature, not the original pupil
used to fit tilt or the angular pixel scale. No stationarity is assumed.
"""

import hashlib
import math
import time
import numpy as np
from scipy import fft, ndimage

from mastsel import congrid, padOrCropCentered, pupilTiltFit

try:
    from numba import config as _numba_config, njit as _njit
except ImportError:
    _njit = None
else:
    if _numba_config.DISABLE_JIT:
        _njit = None


_PAIR_LAG_CACHE_BYTES = 64*1024**2


def _native_pairs_numpy(covariance, diagonal, first, ys, xs, amplitude, phasor, intensity,
                        origin_y, origin_x, span_x, coherence):
    """Accumulate one upper row strip and its Hermitian counterpart."""
    count = len(diagonal); last = first+len(covariance)
    rows = np.arange(first,last)[:,None]; columns = np.arange(first,count)[None]
    upper = columns >= rows
    sf = diagonal[first:last,None]+diagonal[None,first:]-2*covariance
    minimum = float(sf[upper].min())
    np.maximum(sf,0,out=sf)
    weights = np.exp(-.5*sf)*amplitude[first:last,None]*amplitude[None,first:]
    weights = weights*phasor[first:last,None]*phasor[None,first:].conj()
    weights[~upper] = 0
    codes = (ys[first:last,None]-ys[None,first:]-origin_y)*span_x
    codes += xs[first:last,None]-xs[None,first:]-origin_x
    direct = np.bincount(codes.ravel(),weights.real.ravel(),minlength=len(coherence))
    direct = direct+1j*np.bincount(codes.ravel(),weights.imag.ravel(),minlength=len(coherence))
    coherence += direct+direct[::-1].conj()
    coherence[-origin_y*span_x-origin_x] -= np.sum(amplitude[first:last]**2)
    weighted = covariance*intensity[first:last,None]*intensity[None,first:]
    quadratic = 2*np.sum(weighted[upper])-np.sum(diagonal[first:last]*intensity[first:last]**2)
    return minimum, float(quadratic)


def _native_pairs_loop(covariance, diagonal, first, ys, xs, amplitude, phasor, intensity,
                       origin_y, origin_x, span_x, coherence):
    """Fuse the same double-precision sum without array temporaries."""
    minimum = 0.; quadratic = 0.
    for local in range(len(covariance)):
        row = first+local
        for col in range(row,len(diagonal)):
            value = covariance[local,col-first]
            sf = diagonal[row]+diagonal[col]-2*value
            minimum = min(minimum,sf)
            if row == col:
                sf = 0.
            weight = math.exp(-.5*max(sf,0.))*amplitude[row]*amplitude[col]
            z = weight*phasor[row]*phasor[col].conjugate()
            dy = ys[row]-ys[col]; dx = xs[row]-xs[col]
            coherence[(dy-origin_y)*span_x+dx-origin_x] += z
            term = value*intensity[row]*intensity[col]
            quadratic += term
            if row != col:
                coherence[(-dy-origin_y)*span_x-dx-origin_x] += z.conjugate()
                quadratic += term
    return minimum,quadratic


_native_pairs_numba = _njit(cache=True,nogil=True)(_native_pairs_loop) if _njit is not None else None
_native_pairs = _native_pairs_numba if _native_pairs_numba is not None else _native_pairs_numpy





class FinitePupilSampling:
    """Pupil coordinates, original tilt fit and bounded Gaussian pair sum.

    Parameters are the real pupil amplitude, native optical FFT size, native
    pupil width in pixels and physical optical period in metres. ``optical_n``
    is a numerical quadrature size; omitted uses the native optical grid for
    every static-phase selection. Explicit reduced grids remain available for
    retained numerical studies without static phase. Returned
    images retain native angular pixels and complete represented-plane flux.
    ``static_phase`` is optional deterministic phase in radians on the original
    fit-pupil grid. Its fitted tilt is removed with the same public coefficients
    as the dynamic covariance. It affects optics only, not reconstruction.
    """

    def __init__(self, fit_pupil, native_n, pupil_pixels, period, optical_n=None, static_phase=None):
        phase = None
        if static_phase is not None:
            phase = np.asarray(static_phase, dtype=np.float64)
            if phase.shape != np.shape(fit_pupil) or not np.isfinite(phase).all():
                raise ValueError('Static phase must be finite radians on the original pupil grid')
            if not np.any(phase):
                phase = None
        self.native_n = int(native_n)
        self.native_side = int(pupil_pixels)
        self.n = self.native_n if optical_n is None else int(optical_n)
        if phase is not None and self.n != self.native_n:
            raise ValueError('Static phase requires native finite-pupil quadrature')
        self.side = round(self.native_side*self.n/self.native_n)
        if self.n < 2 or self.n % 2 or self.side*self.native_n != self.native_side*self.n:
            raise ValueError('Finite-pupil quadrature must preserve an integral pupil width')
        self.period = float(period)
        if not np.isfinite(self.period) or self.period <= 0:
            raise ValueError('Finite-pupil optical period must be positive metres')
        self.fit = pupilTiltFit(fit_pupil)
        self.signature = hashlib.sha256(np.ascontiguousarray(fit_pupil).view(np.uint8)).hexdigest()
        self.pupil = padOrCropCentered(congrid(np.asarray(fit_pupil), [self.side]*2), self.n, np)
        self.ys, self.xs = np.nonzero(self.pupil)
        self.amplitude = self.pupil[self.ys, self.xs]
        self.xy = np.column_stack((self.xs-self.n/2, self.ys-self.n/2))*self.period/self.n
        self.ramps = np.stack((2*(self.xs-self.n/2)/self.side+1/len(fit_pupil),
                               2*(self.ys-self.n/2)/self.side+1/len(fit_pupil)))
        self.intensity = self.amplitude**2
        self.intensity /= self.intensity.sum()
        self.native_pupil = padOrCropCentered(congrid(np.asarray(fit_pupil), [self.native_side]*2), self.native_n, np)
        self.pair_count = fft.ifft2(abs(fft.fft2(self.pupil))**2).real
        self.native_pair_count = fft.ifft2(abs(fft.fft2(self.native_pupil))**2).real
        # One compact signed-lag geometry serves covariance sampling and optics.
        # Cache at most 64 MiB; larger pupils compute the same codes in blocks.
        self._lag_y = np.arange(self.ys.min()-self.ys.max(), self.ys.max()-self.ys.min()+1)
        self._lag_x = np.arange(self.xs.min()-self.xs.max(), self.xs.max()-self.xs.min()+1)
        self._lag_count = len(self._lag_y)*len(self._lag_x)
        self._pair_lags = None
        dtype = np.min_scalar_type(self._lag_count-1)
        if len(self.xs)**2*np.dtype(dtype).itemsize <= _PAIR_LAG_CACHE_BYTES:
            codes = np.empty((len(self.xs), len(self.xs)), dtype=dtype)
            for first in range(0, len(self.xs), 128):
                codes[first:first+128] = self._pair_lag_codes(first, first+128)
            self._pair_lags = codes
        self._lag_fft_indices = ((self._lag_y[:, None] % self.n)*self.n+
            self._lag_x[None] % self.n).ravel()
        if self.n != self.native_n:
            valid = self.pair_count > self.pair_count[0, 0]*1e-9
            self._nearest_lags = ndimage.distance_transform_edt(
                ~valid, return_distances=False, return_indices=True)
            lag = fft.fftfreq(self.native_n)*self.n
            self._native_lag_coordinates = np.array(np.meshgrid(lag, lag, indexing='ij'))
            self._valid_lags = valid
        self._reverse = (-np.arange(self.native_n)) % self.native_n
        self._normalization = self.native_n**2*np.sum(self.native_pupil**2)
        self.static_phase = None
        self._static_phasor = np.ones(len(self.xs), dtype=complex)
        if phase is not None:
            coefficients = np.einsum('iyx,yx->i', self.fit, phase)
            def coherent(size, side):
                native_phase = padOrCropCentered(congrid(phase, [side]*2), size, np).copy()
                y, x = np.mgrid[:size, :size]
                native_phase -= coefficients[0]*(2*(x-size/2)/side+1/len(fit_pupil))
                native_phase -= coefficients[1]*(2*(y-size/2)/side+1/len(fit_pupil))
                return native_phase
            native_phase = coherent(self.n, self.side)
            self.static_phase = native_phase[self.ys, self.xs]
            self._static_phasor = np.exp(1j*self.static_phase)

    def _pair_lag_codes(self, first, last):
        if self._pair_lags is not None:
            return self._pair_lags[first:last]
        dy = self.ys[first:last, None]-self.ys[None]-self._lag_y[0]
        dx = self.xs[first:last, None]-self.xs[None]-self._lag_x[0]
        return dy*len(self._lag_x)+dx

    def covariance_from_lags(self, covariance, rows=None, columns=None):
        """Sample a square periodic lag covariance at every valid pupil pair.

        Values retain their input units and precision; no tilt is removed here.
        The immutable sampling geometry owns the bounded compact-index cache.
        Construct a new sampling object when pupil or optical sampling changes.
        """
        covariance = np.asarray(covariance)
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            raise ValueError('Periodic lag covariance must be a square grid')
        size = len(covariance)
        if rows is not None or columns is not None:
            rows = slice(None) if rows is None else rows
            columns = slice(None) if columns is None else columns
            return covariance[(self.ys[rows,None]-self.ys[None,columns])%size,
                              (self.xs[rows,None]-self.xs[None,columns])%size]
        values = covariance[(self._lag_y % size)[:, None], (self._lag_x % size)[None]].ravel()
        if self._pair_lags is not None:
            return values[self._pair_lags]
        result = np.empty((len(self.xs), len(self.xs)), dtype=values.dtype)
        for first in range(0, len(self.xs), 128):
            result[first:first+128] = values[self._pair_lag_codes(first, first+128)]
        return result

    def tilt_moments(self, count, period):
        """Original-pupil ramp-fit coefficients of centred physical Fourier waves."""
        fine = self.fit.shape[1]
        axis = (-self.native_side/2+np.arange(fine)*self.native_side/fine)*self.period/self.native_n
        q = (np.arange(count)-count//2)/period
        e = np.exp(2j*np.pi*np.outer(q, axis))
        return np.asarray([e@f.T@e.T for f in self.fit])

    def psf(self, covariance, stationary_lags=None):
        """Integrate exp(-SF/2) at every represented pupil pair, then diffract.

        ``covariance`` is radians squared after the original-pupil tilt fit.
        No average SF is exponentiated. At reduced quadrature, pair-averaged
        coherence is interpolated and multiplied by native pupil diffraction.
        Optional stationary lag covariances factor a known contribution before
        interpolation and restore it on the native grid afterwards. The final
        native image has unit full-plane flux. Accuracy remains a numerical
        validation concern.
        """
        covariance = np.asarray(covariance)
        if covariance.shape != (len(self.xs), len(self.xs)) or not np.isfinite(covariance).all():
            raise ValueError('Finite-pupil covariance must match the optical pupil points')
        diagonal = np.diag(covariance)
        lag_coherence = np.zeros(self._lag_count, dtype=complex if self.static_phase is not None else float)
        double = covariance.dtype == np.float64
        doubled = np.empty((min(128, len(diagonal)), len(diagonal))) if double else None
        minimum_sf = 0.
        for first in range(0, len(diagonal), 128):
            sl = slice(first, min(first+128, len(diagonal)))
            if double:
                sf = diagonal[sl, None]+diagonal[None]
                np.multiply(covariance[sl], 2, out=doubled[:len(sf)])
                sf -= doubled[:len(sf)]
            else:
                sf = diagonal[sl, None]+diagonal[None]-2*covariance[sl]
            minimum_sf = min(minimum_sf, float(sf.min()))
            if sf.min() < -2e-7:
                raise ValueError('Finite-pupil residual has a negative structure function')
            np.maximum(sf, 0, out=sf)
            if double:
                sf *= -.5
                np.exp(sf, out=sf)
                sf *= self.amplitude[sl, None]
                sf *= self.amplitude[None]
                weights = sf
            else:
                # Preserve the existing promotion/rounding of other inputs.
                weights = np.exp(-.5*sf)*self.amplitude[sl, None]*self.amplitude[None]
            codes = self._pair_lag_codes(first, first+128).ravel()
            if self.static_phase is None:
                lag_coherence += np.bincount(codes, weights=weights.ravel(), minlength=self._lag_count)
            else:
                weights = weights*self._static_phasor[sl, None]*self._static_phasor[None].conj()
                lag_coherence += (np.bincount(codes, weights=weights.real.ravel(), minlength=self._lag_count)
                    +1j*np.bincount(codes, weights=weights.imag.ravel(), minlength=self._lag_count))
        coherence = np.zeros(self.n**2, dtype=lag_coherence.dtype)
        np.add.at(coherence, self._lag_fft_indices, lag_coherence)
        coherence = coherence.reshape(self.n, self.n)
        if self.n != self.native_n:
            # Interpolate the pair-averaged atmospheric coherence after its
            # exponential, then use the exact native pupil autocorrelation.
            # This preserves native diffraction in the zero-turbulence limit.
            transfer = np.divide(coherence, self.pair_count, out=np.zeros_like(coherence), where=self._valid_lags)
            if stationary_lags is not None:
                coarse, native = stationary_lags
                if coarse.shape != (self.n, self.n) or native.shape != (self.native_n, self.native_n):
                    raise ValueError('Stationary lag covariances must share both optical grids')
                transfer *= np.exp(coarse[0, 0]-coarse)
            transfer = transfer[tuple(self._nearest_lags)]
            transfer = ndimage.map_coordinates(transfer, self._native_lag_coordinates, order=3, mode='grid-wrap')
            transfer = .5*(transfer+transfer[self._reverse][:, self._reverse])
            transfer[0, 0] = 1.
            if stationary_lags is not None:
                transfer *= np.exp(native-native[0, 0])
            coherence = transfer*self.native_pair_count
        image = fft.fftshift(fft.fft2(coherence)).real
        image /= self._normalization
        if image.min() < -1e-11 or abs(image.sum()-1) > 2e-12:
            raise ValueError(f'Finite-pupil optical calculation failed positivity or full-flux validation: minimum={image.min():.3g}, flux={image.sum():.16g}')
        variance = float(self.intensity@diagonal-self.intensity@covariance@self.intensity)
        return image, dict(
            full_flux=float(image.sum()), minimum_pixel=float(image.min()),
            minimum_structure_function=minimum_sf, residual_phase_variance=variance)

    def psf_blocks(self, diagonal, blocks):
        """Integrate native Gaussian optics from upper covariance row strips.

        ``diagonal`` contains every fitted-pupil variance in radians squared.
        ``blocks`` yields ``(first, covariance)`` with consecutive rows and
        columns ``first:``. All physical covariance terms must be included.
        Static phase is optional; its absence uses unit complex phasors in
        the same kernel. No coupling or phase information is discarded. Block storage belongs
        to the producer; it may be released after each yielded strip.
        """
        if self.n != self.native_n:
            raise ValueError('Blocked optics require native finite-pupil quadrature')
        diagonal = np.asarray(diagonal)
        if diagonal.shape != (len(self.xs),) or not np.isfinite(diagonal).all():
            raise ValueError('Residual diagonal must match finite optical pupil points')
        lag_coherence = np.zeros(self._lag_count,complex)
        minimum_sf = 0.; quadratic = 0.; expected = 0
        assembly_seconds = 0.; coherence_seconds = 0.
        iterator = iter(blocks)
        while True:
            started = time.perf_counter()
            try: first, covariance = next(iterator)
            except StopIteration: break
            assembly_seconds += time.perf_counter()-started
            covariance = np.asarray(covariance)
            if (first != expected or covariance.ndim != 2 or covariance.shape[1] != len(diagonal)-first
                    or not 0 < len(covariance) <= len(diagonal)-first or not np.isfinite(covariance).all()):
                raise ValueError('Covariance blocks must cover consecutive upper row strips')
            started = time.perf_counter()
            sf, value = _native_pairs(covariance,diagonal,first,self.ys,self.xs,
                self.amplitude,self._static_phasor,self.intensity,
                self._lag_y[0],self._lag_x[0],len(self._lag_x),lag_coherence)
            coherence_seconds += time.perf_counter()-started
            minimum_sf = min(minimum_sf,sf); quadratic += value
            expected += len(covariance)
        if expected != len(diagonal):
            raise ValueError('Covariance blocks must cover every optical pupil point')
        if minimum_sf < -2e-7:
            raise ValueError('Finite-pupil residual has a negative structure function')
        coherence = np.zeros(self.n**2,complex)
        np.add.at(coherence,self._lag_fft_indices,lag_coherence)
        image = fft.fftshift(fft.fft2(coherence.reshape(self.n,self.n))).real/self._normalization
        if image.min() < -1e-11 or abs(image.sum()-1)>2e-12:
            raise ValueError('Native finite-pupil optics failed positivity or full-flux validation')
        variance = float(self.intensity@diagonal-quadratic)
        return image,dict(full_flux=float(image.sum()),minimum_pixel=float(image.min()),
            minimum_structure_function=minimum_sf,residual_phase_variance=variance,
            covariance_block_seconds=assembly_seconds,coherence_seconds=coherence_seconds)
