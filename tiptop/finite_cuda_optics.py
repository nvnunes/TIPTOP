"""Bounded double-precision pupil-pair integration on resident CUDA covariance."""
import time
import numpy as np

_SOURCE = r'''
extern "C" __global__ void pupil_pairs(
 const double* covariance, const double* diagonal,
 const long long* ys, const long long* xs, const double* real, const double* imag,
 double* output_r, double* output_i, unsigned long long* negative,
 int first, int count, long long total, int lag_y, int lag_x, int width) {
 long long index=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(index>=total)return;
 int columns=count-first;
 int i=first+index/columns, j=first+index%columns;
 if(j<i)return;
 double sf=(diagonal[i]+diagonal[j])-2.*covariance[index];
 if(sf<0.)atomicMax(negative,__double_as_longlong(-sf));
 if(j==i)return;
 double coherence=exp(-.5*fmax(sf,0.));
 double r=coherence*(real[i]*real[j]+imag[i]*imag[j]);
 double im=coherence*(imag[i]*real[j]-real[i]*imag[j]);
 long long code=(ys[i]-ys[j]-lag_y)*width+xs[i]-xs[j]-lag_x;
 atomicAdd(output_r+code,r);atomicAdd(output_i+code,im);
}
'''


class CudaPupilIntegrator:
    """Integrate every pair without downloading covariance or changing quadrature.

    P3 owns resident covariance. TIPTOP owns complex pupil weights and bounded
    per-tile lag buffers. Only the final coherence array returns to CPU optics.
    Double atomics require CUDA compute capability 6.0 or newer.
    """
    def __init__(self, owner, sampling, amplitude, phasor, tile_rows=128):
        self.owner=owner; self.sampling=sampling; self.rows=tile_rows
        self.amplitude=np.asarray(amplitude); self.power=float(np.sum(self.amplitude**2))
        self.ys=owner.upload(sampling.ys,True); self.xs=owner.upload(sampling.xs,True)
        self.real=owner.upload(self.amplitude*np.asarray(phasor).real)
        self.imag=owner.upload(self.amplitude*np.asarray(phasor).imag)
        properties=owner.cupy.cuda.runtime.getDeviceProperties(owner.cupy.cuda.Device().id)
        if properties['major']<6:
            raise RuntimeError('Resident float64 pupil optics requires CUDA compute capability >= 6.0')
        self.kernel=owner.cupy.RawKernel(_SOURCE,'pupil_pairs',options=('--fmad=false',))

    def integrate(self, resident):
        if resident.owner is not self.owner or resident.sampling is not self.sampling:
            raise ValueError('Optical covariance and weights have different owners or sampling')
        started=time.perf_counter(); s=self.sampling; count=len(self.amplitude)
        tiles=(count+self.rows-1)//self.rows
        partial_r=self.owner.zeros((tiles,s._lag_count))
        partial_i=self.owner.zeros((tiles,s._lag_count))
        negative=self.owner.cupy.zeros((),dtype=self.owner.cupy.uint64)
        for tile,first in enumerate(range(0,count,self.rows)):
            last=min(first+self.rows,count)
            covariance=resident.optical_block_array(first,last)
            self.kernel(((covariance.size+255)//256,), (256,),
                (covariance,resident.optical_diagonal,self.ys,self.xs,self.real,self.imag,
                 partial_r[tile],partial_i[tile],negative,np.int32(first),np.int32(count),
                 np.int64(covariance.size),np.int32(s._lag_y[0]),np.int32(s._lag_x[0]),np.int32(len(s._lag_x))))
        real=self.owner.to_host(partial_r.sum(axis=0)); imag=self.owner.to_host(partial_i.sum(axis=0))
        minimum=-float(self.owner.to_host(negative).view(np.float64))
        if minimum < -2e-7:
            raise ValueError('Resident finite-pupil covariance has a negative structure function')
        direct=real+1j*imag; coherence=direct+direct[::-1].conj()
        centre=-int(s._lag_y[0])*len(s._lag_x)-int(s._lag_x[0]); coherence[centre]=self.power
        otf=np.zeros(s.n*s.n,complex); np.add.at(otf,s._lag_fft_indices,coherence)
        otf=otf.reshape(s.n,s.n)/self.power
        return otf,dict(minimum_structure_function=minimum,residual_phase_variance=resident.variance(),
            device_covariance_coherence_seconds=time.perf_counter()-started,
            device_covariance_preparation_seconds=resident.preparation_seconds,
            device_background_prepared=resident.background_prepared,
            device_pair_tile_rows=self.rows,device_pair_implementation='cuda_float64',
            device_pair_fallback_reason=None)
