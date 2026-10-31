"""Spectral pupil-fit moments for TIPTOP's static optical composition.

These helpers retain the original-pupil fit at every source frequency and
fold only the optical sampling. Coherence uses TIPTOP's existing bounded pair
sum. MASTSEL supplies only its existing public pupilTiltFit coefficients.
"""

import numpy as np
from numpy import fft
from mastsel import pupilTiltFit
from . import finite_pupil


class SpectralPupilFit:
    """Project Gaussian phase onto the complement of two original-pupil ramps.

    ``n`` is an even source FFT size. ``pupil_pixels`` is its positive physical
    pupil span, which may be fractional. An unshifted spectrum's inverse FFT
    has covariance units rad². Optical resampling preserves the same period,
    complete source power and original-frequency fit moments.
    """

    def __init__(self, pupil, n, pupil_pixels):
        if int(n) != n or n < 2 or n % 2 or not np.isfinite(pupil_pixels) or pupil_pixels <= 0:
            raise ValueError('Spectral pupil fit requires an even grid and positive pupil span')
        self.n = int(n)
        self.span = float(pupil_pixels)
        fit = pupilTiltFit(pupil)
        fine = fit.shape[1]
        positions = self.n/2-self.span/2+np.arange(fine)*self.span/fine
        transform = np.exp(2j*np.pi*fft.fftfreq(self.n)[:,None]*positions[None])
        self.kernels = np.asarray([transform@component@transform.T/self.n**2 for component in fit])
        y, x = np.mgrid[:self.n,:self.n]
        self.ramps = np.stack((2*(x-self.n/2)/self.span+1/fine,
                               2*(y-self.n/2)/self.span+1/fine))

    def covariance_moments(self, phase_spectrum, output_n=None):
        """Return lag covariance, phase–fit covariance and fit covariance in rad².

        Shapes are (M,M), (2,M,M), (2,2). Source modes fold without truncation;
        the original fit is evaluated before folding. ``output_n`` must be an
        even integral sampling of the same Fourier period.
        """
        power = np.asarray(phase_spectrum, dtype=np.float64)
        if power.shape != (self.n,self.n) or not np.isfinite(power).all():
            raise ValueError('Pupil-fit phase power must be finite on the native optical grid')
        if power.min() < -1e-12*max(float(power.max()),np.finfo(float).tiny):
            raise ValueError('Pupil-fit phase power must be nonnegative')
        power = np.maximum(power,0)
        reverse = (-np.arange(self.n))%self.n
        power = .5*(power+power[reverse][:,reverse])
        size = self.n if output_n is None else int(output_n)
        if size < 2 or size % 2 or (output_n is not None and size != output_n):
            raise ValueError('Covariance sampling requires a positive even integral grid')

        def fold(values):
            if size == self.n:
                return values
            index = np.rint(fft.fftfreq(self.n)*self.n).astype(np.intp)%size
            rows = np.zeros((size,self.n),dtype=values.dtype)
            np.add.at(rows,index,values)
            result = np.zeros((size,size),dtype=values.dtype)
            np.add.at(result.T,index,rows.T)
            return result

        covariance = fft.ifft2(fold(power)).real
        if size != self.n:
            covariance *= size**2/self.n**2
        cross = np.asarray([size**2*fft.ifft2(fold(power*k.conj())).real for k in self.kernels])
        tilt = self.n**2*np.einsum('yx,iyx,jyx->ij',power,self.kernels,self.kernels.conj()).real
        return covariance,cross,tilt

    def coherence_from_covariance(self, covariance, cross, tilt, coherent_pupil):
        """Integrate every Gaussian pupil pair with TIPTOP's native pair kernel.

        Moments must be on this optical grid and retain the original fit.
        Returns raw periodic coherence, pupil power and the intensity-weighted
        dynamic phase variance (rad²). No static OTF ratio or crop is applied.
        Row strips bound temporary covariance storage; no full pair matrix is
        retained. These are the same covariance terms used by finite optics.
        """
        wave = np.asarray(coherent_pupil,dtype=np.complex128)
        if (wave.shape != (self.n,self.n) or np.shape(covariance) != wave.shape or
                np.shape(cross) != (2,self.n,self.n) or np.shape(tilt) != (2,2)):
            raise ValueError('Pupil and covariance moments must match the optical grid')
        if not all(np.isfinite(v).all() for v in (wave,covariance,cross,tilt)):
            raise ValueError('Pupil-fit optics require finite pupil and covariance moments')
        ys,xs = np.nonzero(wave)
        if not len(xs):
            raise ValueError('Optical pupil has zero power')
        amplitudes = np.abs(wave[ys,xs])
        phasors = wave[ys,xs]/amplitudes
        intensity = amplitudes**2
        pupil_power = float(intensity.sum())
        intensity /= pupil_power
        ramps = self.ramps[:,ys,xs]
        moments = cross[:,ys,xs]
        diagonal = (covariance[0,0]-2*np.sum(ramps*moments,axis=0)
                    +np.einsum('iy,ij,jy->y',ramps,tilt,ramps))
        variance = float(np.average(diagonal,weights=amplitudes**2))
        if variance < -1e-10*max(float(covariance[0,0]),1.):
            raise ValueError('Projected phase variance is negative')
        origin_y,origin_x = ys.min()-ys.max(),xs.min()-xs.max()
        height,width = 1-2*origin_y,1-2*origin_x
        compact = np.zeros(height*width,complex)
        rows = max(1,min(128,(64*1024**2)//(8*len(xs))))
        for first in range(0,len(xs),rows):
            last = min(first+rows,len(xs)); r = slice(first,last); c = slice(first,None)
            block = covariance[(ys[r,None]-ys[None,c])%self.n,(xs[r,None]-xs[None,c])%self.n].copy()
            block -= ramps[:,r].T@moments[:,c]
            block -= moments[:,r].T@ramps[:,c]
            block += (ramps[:,r].T@tilt)@ramps[:,c]
            index = np.arange(last-first)
            block[index,index] = diagonal[r]
            minimum,_ = finite_pupil._native_pairs(block,diagonal,first,ys,xs,amplitudes,
                phasors,intensity,origin_y,origin_x,width,compact)
            if minimum < -2e-7:
                raise ValueError('Pupil-fit residual has a negative structure function')
        result = np.zeros((self.n,self.n),complex)
        lag_y = np.arange(origin_y,1-origin_y)
        lag_x = np.arange(origin_x,1-origin_x)
        np.add.at(result,(lag_y[:,None]%self.n,lag_x[None]%self.n),compact.reshape(height,width))
        return result,pupil_power,max(variance,0.)
