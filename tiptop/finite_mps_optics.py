"""Optional fused Metal pupil-pair weighting and lag accumulation."""

_SOURCE=r'''
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

kernel void pupil_pairs(
    device const float* covariance [[buffer(0)]],
    device const float* diagonal [[buffer(1)]],
    device const long* ys [[buffer(2)]], device const long* xs [[buffer(3)]],
    device const float* real [[buffer(4)]], device const float* imag [[buffer(5)]],
    device atomic_float* output_r [[buffer(6)]],
    device atomic_float* output_i [[buffer(7)]],
    device atomic_uint* negative [[buffer(8)]],
    constant uint& first [[buffer(9)]], constant uint& count [[buffer(10)]],
    constant uint& total [[buffer(11)]],
    constant int& lag_y [[buffer(12)]], constant int& lag_x [[buffer(13)]],
    constant uint& width [[buffer(14)]],
    uint index [[thread_position_in_grid]]) {
    if(index>=total)return;
    uint columns=count-first;
    uint i=first+index/columns;
    uint j=first+index%columns;
    if(j<i)return;
    float sf=(diagonal[i]+diagonal[j])-2.f*covariance[index];
    if(sf<0.f)atomic_fetch_max_explicit(negative,as_type<uint>(-sf),memory_order_relaxed);
    if(j==i)return;
    float coherence=exp(-.5f*max(sf,0.f));
    float r=coherence*(real[i]*real[j]+imag[i]*imag[j]);
    float im=coherence*(imag[i]*real[j]-real[i]*imag[j]);
    uint code=uint((ys[i]-ys[j]-lag_y)*width+xs[i]-xs[j]-lag_x);
    atomic_fetch_add_explicit(output_r+code,r,memory_order_relaxed);
    atomic_fetch_add_explicit(output_i+code,im,memory_order_relaxed);
}
'''


class FusedPupilPairs:
    """Accumulate every strict upper pair into an owner's bounded lag buffer.

    P3 owns the covariance tensors. TIPTOP supplies the same float32 pupil
    weights and integer lag indices as its Torch path. Only temporary pair
    expressions are fused; neither covariance nor optical sampling is reduced.
    The caller allocates disjoint per-tile outputs, then reduces them normally.
    """
    def __init__(self,owner):
        self.library=owner.torch.mps.compile_shader(_SOURCE)

    def accumulate(self,covariance,diagonal,optics,first,real,imag,negative):
        if not covariance.is_contiguous():
            raise ValueError('Fused pupil covariance must be contiguous')
        s=optics.sampling
        self.library.pupil_pairs(covariance,diagonal,optics.ys,optics.xs,
            optics.real,optics.imag,real,imag,negative,first,len(optics.amplitude),
            covariance.numel(),int(s._lag_y[0]),int(s._lag_x[0]),len(s._lag_x),
            threads=covariance.numel(),group_size=min(256,covariance.numel()))
