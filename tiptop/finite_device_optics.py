"""Pupil-pair optics on resident MPS covariance, with one coherence handoff."""
import time
import numpy as np


class DevicePupilIntegrator:
    """Stream upper covariance tiles into bounded, pairwise-reduced lag buffers.

    Device covariance remains owned by P3. This optical object owns pupil
    weights and accumulation storage. No full covariance tile is downloaded.
    The final transform stays with the existing TIPTOP optical implementation.
    """
    def __init__(self,owner,sampling,amplitude,phasor,tile_rows=128):
        self.owner=owner
        self.torch=owner.torch
        self.sampling=sampling
        self.rows=tile_rows
        self.amplitude=np.asarray(amplitude)
        self.power=float(np.sum(self.amplitude**2))
        self.ys=owner.upload(sampling.ys,integer=True)
        self.xs=owner.upload(sampling.xs,integer=True)
        self.real=owner.upload(self.amplitude*np.asarray(phasor).real)
        self.imag=owner.upload(self.amplitude*np.asarray(phasor).imag)
        self.index=owner.upload(np.arange(len(amplitude)),integer=True)
        self.fused=None
        self.fused_fallback_reason=None
        # Host tensor tests and older Torch builds retain the ordinary path.
        if self.real.device.type=='mps' and hasattr(self.torch.mps,'compile_shader'):
            from .finite_mps_optics import FusedPupilPairs
            try:self.fused=FusedPupilPairs(owner)
            except (RuntimeError,NotImplementedError) as exc:
                self.fused_fallback_reason=str(exc)

    def integrate(self,resident):
        if resident.owner is not self.owner or resident.sampling is not self.sampling:
            raise ValueError('Optical covariance and weights have different owners or sampling')
        started=time.perf_counter()
        torch=self.torch;s=self.sampling
        count=len(self.amplitude)
        tile_count=(count+self.rows-1)//self.rows
        partial_r=self.owner.zeros((tile_count,s._lag_count))
        partial_i=self.owner.zeros((tile_count,s._lag_count))
        minimum=self.owner.zeros(())
        for tile,first in enumerate(range(0,count,self.rows)):
            last=min(first+self.rows,count)
            covariance=resident.optical_block_array(first,last)
            diagonal=resident.optical_diagonal
            if self.fused is not None:
                self.fused.accumulate(covariance,diagonal,self,first,
                    partial_r[tile],partial_i[tile],minimum)
                continue
            sf=diagonal[first:last,None]+diagonal[None,first:]-2*covariance
            upper=self.index[None,first:]>=self.index[first:last,None]
            minimum=torch.minimum(minimum,torch.where(upper,sf,torch.full_like(sf,float('inf'))).amin())
            strict=self.index[None,first:]>self.index[first:last,None]
            coherence=torch.exp(-.5*torch.clamp(sf,min=0))*strict
            real=coherence*(self.real[first:last,None]*self.real[None,first:]
                            +self.imag[first:last,None]*self.imag[None,first:])
            imag=coherence*(self.imag[first:last,None]*self.real[None,first:]
                            -self.real[first:last,None]*self.imag[None,first:])
            codes=(self.ys[first:last,None]-self.ys[None,first:]-int(s._lag_y[0]))*len(s._lag_x)
            codes+=self.xs[first:last,None]-self.xs[None,first:]-int(s._lag_x[0])
            partial_r[tile].scatter_add_(0,codes.reshape(-1),real.reshape(-1))
            partial_i[tile].scatter_add_(0,codes.reshape(-1),imag.reshape(-1))
        real=self.owner.to_host(partial_r.sum(axis=0))
        imag=self.owner.to_host(partial_i.sum(axis=0))
        minimum=self.owner.to_host(minimum)
        if self.fused is not None:
            # Shader stores max(-negative SF) as positive float bits via uint atomics.
            minimum=-float(minimum)
        else:minimum=float(minimum)
        if minimum < -2e-7:
            raise ValueError('Resident finite-pupil covariance has a negative structure function')
        direct=real+1j*imag
        coherence=direct+direct[::-1].conj()
        # The diagonal pair sum is analytically sum(amplitude**2), not a fitted normalization.
        centre=-int(s._lag_y[0])*len(s._lag_x)-int(s._lag_x[0])
        coherence[centre]=self.power
        otf=np.zeros(s.n*s.n,complex)
        np.add.at(otf,s._lag_fft_indices,coherence)
        otf=otf.reshape(s.n,s.n)/self.power
        variance=resident.variance()
        return otf,dict(minimum_structure_function=minimum,residual_phase_variance=variance,
            device_covariance_coherence_seconds=time.perf_counter()-started,
            device_covariance_preparation_seconds=resident.preparation_seconds,
            device_background_prepared=resident.background_prepared,
            device_pair_tile_rows=self.rows,
            device_pair_implementation='metal_fused' if self.fused is not None else 'torch',
            device_pair_fallback_reason=self.fused_fallback_reason)
