"""Original-pupil static optics and a bounded fast static-OTF replacement.

The original lattice keeps deterministic phase pixels unchanged. Full-plane
normalization precedes the requested detector crop, which is not renormalized.
"""

import numpy as np
from scipy import fft, ndimage, signal
from scipy.sparse import csr_matrix
from mastsel import congrid, padOrCropCentered, pupilTiltFit

from .finite_pupil import FinitePupilSampling
from .pupil_fit import SpectralPupilFit


def _lag_interpolator(lags,size):
    """Periodic cubic B-spline evaluation at fixed physical lag coordinates."""
    coordinate=np.remainder(lags,size);first=np.floor(coordinate).astype(np.intp)
    t=coordinate-first
    weights=np.stack(((1-t)**3,3*t**3-6*t*t+4,-3*t**3+3*t*t+3*t+1,t**3),axis=1)/6
    columns=(first[:,None]+np.arange(-1,3))%size
    return csr_matrix((weights.ravel(),columns.ravel(),np.arange(0,4*len(lags)+1,4)),shape=(len(lags),size))


def averaged_static_pupil(pupil,phase,diameter,period,n):
    """Area-average original complex amplitudes onto the atmospheric grid.

    This is an intermediate for dynamic pupil weighting, not a substitute for
    the original static OTF. Phase samples and original-pupil tilt fitting are
    retained before the complex averages are formed.
    """
    size=len(pupil);fine_step=diameter/size;step=period/n
    fine=(np.arange(size)-size/2)*fine_step
    coarse=(np.arange(n)-n/2)*step
    overlap=np.maximum(0,np.minimum(coarse[:,None]+step/2,fine[None]+fine_step/2)
        -np.maximum(coarse[:,None]-step/2,fine[None]-fine_step/2))/step
    coefficients=np.einsum('iyx,yx->i',pupilTiltFit(pupil),phase)
    ramps=2*fine/diameter+1/size
    fitted=phase-coefficients[0]*ramps[None]-coefficients[1]*ramps[:,None]
    return overlap@(pupil*np.exp(1j*fitted))@overlap.T


class OriginalPupilSampling(FinitePupilSampling):
    """Physical original-map lattice sharing the atmospheric Fourier period.

    ``diameter`` and ``period`` are metres; phase is original-pupil radians.
    The current lattice requires an even integral number of pupil pixels in
    the Fourier period. ``output_n`` identifies the requested detector crop.
    Only static diffraction uses this lattice; atmospheric covariance is
    supplied by the separate coarse-grid pupil-fit calculation.
    """

    def __init__(self, pupil, phase, diameter, period, output_n):
        side = len(pupil)
        size = round(period*side/diameter)
        if size % 2 or not np.isclose(size, period*side/diameter, rtol=1e-12):
            raise ValueError('Original-map pupil sampling must be commensurate with the atmospheric Fourier period')
        super().__init__(pupil,size,side,period,optical_n=size,static_phase=phase)
        self.output_n = int(output_n)
        if self.output_n%2 or self.output_n>self.n:
            raise ValueError('Original-map detector crop must be even and fit within the complete optical plane')
        if self.static_phase is None:
            self.static_phase=np.zeros(len(self.xs))
            self._static_phasor=np.ones(len(self.xs),complex)

    def detector_crop(self, image):
        first = (self.n-self.output_n)//2
        return image[first:first+self.output_n,first:first+self.output_n].copy()



class StaticOtfCorrection:
    """Replace resampled static optics, retaining its dynamic pupil-fit transfer.

    The inferred dynamic transfer is an approximation for nonstationary
    pupil-fit coherence. It is exact for zero atmospheric power. Small/zero
    denominators are excluded, extended from the nearest valid supported lag,
    and exposed in diagnostics. No PSF clipping or map-amplitude fit is used.
    """

    def __init__(self,pupil,phase,diameter,period,n,pupil_pixels,coherent_pupil=None,output_n=None):
        output_n=n if output_n is None else output_n
        self.original = OriginalPupilSampling(pupil,phase,diameter,period,output_n)
        self.n = n
        self.stationary_factor=None
        fit = pupilTiltFit(pupil)
        coefficients = np.einsum('iyx,yx->i',fit,phase)
        coarse_pupil = padOrCropCentered(congrid(pupil,[pupil_pixels]*2),n,np)
        coarse_phase = padOrCropCentered(congrid(phase,[pupil_pixels]*2),n,np).copy()
        y,x = np.mgrid[:n,:n]
        coarse_phase -= coefficients[0]*(2*(x-n/2)/pupil_pixels+1/len(pupil))
        coarse_phase -= coefficients[1]*(2*(y-n/2)/pupil_pixels+1/len(pupil))
        wave = coarse_pupil*np.exp(1j*coarse_phase)
        if coherent_pupil is not None:
            wave=np.asarray(coherent_pupil)
            coarse_pupil=abs(wave)
        self.denominator = fft.ifft2(abs(fft.fft2(wave))**2)/np.sum(coarse_pupil**2)
        self.valid = abs(self.denominator)>1e-8
        self.nearest = ndimage.distance_transform_edt(~fft.fftshift(self.valid),
            return_distances=False,return_indices=True)
        fine = self.original
        wave = np.zeros((fine.n,fine.n),complex)
        wave[fine.ys,fine.xs] = fine.amplitude*fine._static_phasor
        static = fft.ifft2(abs(fft.fft2(wave))**2)/np.sum(fine.amplitude**2)
        self.lags = np.arange(-len(pupil)+1,len(pupil))
        self.static = static[self.lags[:,None]%fine.n,self.lags[None]%fine.n]
        lag = self.lags*n/fine.n
        self._interpolator=_lag_interpolator(lag,n)
        self._detector_transform=signal.CZT(len(self.lags),m=output_n,
            w=np.exp(-2j*np.pi/fine.n),a=np.exp(-1j*np.pi*output_n/fine.n))
        self._detector_phase=np.exp(2j*np.pi*(np.arange(output_n)-output_n/2)*(len(pupil)-1)/fine.n)

    def apply(self,image,complete=True):
        """Return original-lattice diffraction, optionally only its detector crop.

        The CZT crop evaluates the same Fourier sum without allocating the
        complete image. Full-plane flux follows from the zero-lag OTF; the
        detector crop is not renormalized.
        """
        return self.from_otf(fft.ifft2(fft.ifftshift(image))*self.n**2,complete)

    def from_otf(self,otf,complete=False):
        """Compose an already available unit-DC dynamic/static input OTF."""
        dynamic=np.asarray(otf)
        if dynamic.shape!=(self.n,self.n) or not np.isfinite(dynamic).all() or abs(dynamic[0,0]-1)>1e-8:
            raise ValueError('Static-OTF composition requires a finite unit-DC OTF on the atmospheric grid')
        dynamic = np.divide(dynamic,self.denominator,out=np.zeros_like(dynamic),where=self.valid)
        dynamic = fft.ifftshift(fft.fftshift(dynamic)[tuple(self.nearest)])
        dynamic[0,0] = 1.
        if self.stationary_factor is not None:dynamic*=self.stationary_factor[0]
        coefficients=ndimage.spline_filter(dynamic,order=3,output=np.complex128,mode='grid-wrap')
        along=self._interpolator@coefficients
        transfer=(self._interpolator@np.ascontiguousarray(along.T)).T
        compact=self.static*transfer
        if self.stationary_factor is not None:compact*=self.stationary_factor[1]
        flux=float(compact[len(self.lags)//2,len(self.lags)//2].real)
        if complete:
            coherence = np.zeros((self.original.n,self.original.n),complex)
            coherence[self.lags[:,None]%self.original.n,self.lags[None]%self.original.n] = compact
            result = fft.fftshift(fft.fft2(coherence)).real/self.original.n**2
            flux=float(result.sum())
        else:
            result=self._detector_transform(self._detector_transform(compact,axis=0),axis=1)
            result=(result*self._detector_phase[:,None]*self._detector_phase[None]).real/self.original.n**2
        minimum = float(result.min())
        if minimum < -1e-11 or abs(flux-1)>2e-12:
            raise ValueError(f'Static-OTF replacement failed positivity/flux: minimum={minimum}, flux={flux}')
        return result,dict(full_flux=flux,minimum_pixel=minimum,captured_flux=float(result.sum()),
            excluded_denominators=int((~self.valid).sum()),denominator_threshold=1e-8)


class AliasStaticOptics:
    """Fast original-static-OTF optics with complex-cell pupil-fit weighting.

    Reconstruction and phase statistics are supplied unchanged. Complex-cell
    averages only estimate the dynamic transfer; the original static OTF sets
    deterministic diffraction and complete-plane normalization. This optical
    approximation has been checked for broad atmospheric spectra; it does not
    preserve positive definiteness for every possible input (a narrow spectrum
    is a retained counterexample). Nonphysical output raises ValueError rather
    than being clipped or silently replaced.
    """

    def __init__(self,pupil,phase,diameter,period,n,pupil_pixels,correction_cutoff):
        self.source_n=n;self.n=3*n//4
        if self.n%2:raise ValueError('Fast original-map quadrature requires an even three-quarter grid')
        self.wave=averaged_static_pupil(pupil,phase,diameter,period,self.n)
        self.source_projector=SpectralPupilFit(pupil,n,diameter*n/period)
        self.projector=SpectralPupilFit(pupil,self.n,diameter*self.n/period)
        self.correction=StaticOtfCorrection(pupil,phase,diameter,period,self.n,pupil_pixels,self.wave,output_n=n)
        q=fft.fftfreq(n)*n/period
        self._high_mask=np.hypot(q[:,None],q[None])>correction_cutoff
        self._high_power=None
        self.high_cache_builds=0

    def _prepare_high_factor(self,power):
        reverse=(-np.arange(self.source_n))%self.source_n
        high=.5*(power+power[reverse][:,reverse])*self._high_mask
        if self._high_power is not None and np.array_equal(high,self._high_power):return
        coarse=self.source_projector.covariance_moments(high,output_n=self.n)[0]
        fine=self.source_projector.covariance_moments(high,output_n=self.correction.original.n)[0]
        lags=self.correction.lags%len(fine)
        self.correction.stationary_factor=(np.exp(coarse[0,0]-coarse),
            np.exp(fine[lags[:,None],lags[None]]-fine[0,0]))
        self._high_power=high.copy();self.high_cache_builds+=1

    def psf(self,phase_spectrum,complete=False):
        """Use the unshifted phase-spectrum normalization and original pupil fit."""
        moments=self.source_projector.covariance_moments(phase_spectrum,output_n=self.n)
        self._prepare_high_factor(np.asarray(phase_spectrum))
        coherence,_,variance=self.projector.coherence_from_covariance(*moments,self.wave)
        image,checks=self.correction.from_otf(coherence/coherence[0,0],complete=complete)
        checks['residual_phase_variance']=variance
        return image,checks
