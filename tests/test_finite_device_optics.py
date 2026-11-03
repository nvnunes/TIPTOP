"""Host tensor algebra checks; native MPS execution is validated separately."""
import numpy as np
import pytest
from scipy import fft

torch=pytest.importorskip('torch')
from p3.aoSystem.finiteTwoStageProducts import DeviceFiniteProducts
from p3.aoSystem.finiteTwoStageMps import MpsFiniteProducts
from tiptop.finite_device_optics import DevicePupilIntegrator
from test_finite_two_stage import coupled_model,sampling


class TensorProducts(MpsFiniteProducts):
    host_covariance_dtype=np.float64
    def __init__(self,backend,sample):
        self.torch=torch
        DeviceFiniteProducts.__init__(self,backend,sample)

    def upload(self,array,integer=False):
        dtype=torch.int64 if integer else torch.complex128 if np.iscomplexobj(array) else torch.float64
        return torch.tensor(np.asarray(array),dtype=dtype)

    def zeros(self,shape):return torch.zeros(shape,dtype=torch.float64)
    def empty(self,shape):return torch.empty(shape,dtype=torch.float64)
    def to_host(self,array):return array.numpy().copy()
    def copy(self,array):return array.clone()


@pytest.mark.parametrize('rows',[16,64])
@pytest.mark.parametrize('static',[False,True])
def test_resident_pair_integral_matches_complete_cpu_pairs(tmp_path,rows,static):
    native=coupled_model(tmp_path/'case')
    native.removeMeanSlopes=True
    sample=sampling(native)
    if static:sample._static_phasor=np.exp(.2j*np.sin(sample.xy[:,0]))
    backend=native._finite_aperture_backend()
    backend.residual_blocks(sample,0)
    backend.device=TensorProducts(backend,sample)
    residual,_=backend.residual_blocks(sample,0)
    expected,checks=sample.psf_blocks(residual.diagonal,residual.blocks())
    resident=backend.device.resident_covariance(residual)
    optics=DevicePupilIntegrator(backend.device,sample,sample.amplitude,sample._static_phasor,tile_rows=rows)
    otf,actual_checks=optics.integrate(resident)
    actual=fft.fftshift(fft.fft2(otf)).real/sample.n**2
    np.testing.assert_allclose(actual,expected,rtol=2e-10,atol=2e-12)
    np.testing.assert_allclose(actual_checks['residual_phase_variance'],checks['residual_phase_variance'],rtol=2e-10,atol=2e-10)
    assert abs(actual.sum()-1)<1e-12


@pytest.mark.parametrize('complex_feedback',[False,True])
def test_batched_queries_and_feedback_match_unbatched_complete_calculation(tmp_path,complex_feedback):
    native=coupled_model(tmp_path/'case')
    native.removeMeanSlopes=True
    sample=sampling(native)
    backend=native._finite_aperture_backend()
    expected,_=backend.residual_covariance(sample,0)
    owner=TensorProducts(backend,sample)
    owner.complex_feedback_supported=complex_feedback
    backend.device=owner
    origins=np.array([[-.15,.21],[.45,-.18]])
    kernel=owner.upload(backend.interpolation[0].host_phase())
    result=owner.empty((2,2,len(sample.xs),backend.nv))
    owner.evaluate_many(kernel,origins,backend.contraction[0],result)
    for i,origin in enumerate(origins):
        single=owner.empty((2,len(sample.xs),backend.nv))
        owner.evaluate(kernel,origin,backend.contraction[0],single)
        np.testing.assert_allclose(result[:,i].numpy(),single.numpy(),rtol=2e-11,atol=2e-11)
    direct=owner.crosses(0).numpy().copy()
    owner.prefetch([0])
    np.testing.assert_allclose(owner.crosses(0).numpy(),direct,rtol=2e-11,atol=2e-11)
    assert not owner._prefetched
    actual,_=backend.residual_covariance(sample,0)
    np.testing.assert_allclose(actual,expected,rtol=2e-10,atol=2e-10)


def test_direction_batches_preserve_order_and_field_owned_covariance(tmp_path):
    from configparser import ConfigParser
    from p3.aoSystem.fourierModel import fourierModel
    from test_p3_bugfixes_noise_tomo import _p3_path

    native=coupled_model(tmp_path/'case')
    config=ConfigParser();config.optionxform=str
    config.read(tmp_path/'case/system.ini')
    config['sources_science'].update(Zenith='[0.,10.,20.]',Azimuth='[0.,40.,135.]')
    with (tmp_path/'case/system.ini').open('w') as stream:config.write(stream)
    native=fourierModel(str(tmp_path/'case/system.ini'),path_root=_p3_path(),
        calcPSF=False,display=False,verbose=False,reduce_memory=False,
        computeOpticalPSD=True,computeFocalAnisoCov=False)
    native.removeMeanSlopes=True
    sample=sampling(native);backend=native._finite_aperture_backend()
    backend.residual_blocks(sample,0)
    owner=TensorProducts(backend,sample);backend.device=owner
    expected={field:owner.crosses(field).numpy().copy() for field in range(3)}
    owner.prefetch([2,0])
    with pytest.raises(RuntimeError,match='previous direction batch'):
        owner.prefetch([1])
    for field in (0,2):
        np.testing.assert_allclose(owner.crosses(field).numpy(),expected[field],rtol=2e-11,atol=2e-11)
    owner.prefetch([1])
    np.testing.assert_allclose(owner.crosses(1).numpy(),expected[1],rtol=2e-11,atol=2e-11)
    first,_=backend.residual_blocks(sample,0)
    resident=owner.resident_covariance(first)
    snapshot=resident.block_array(0,13).numpy().copy()
    second,_=backend.residual_blocks(sample,1)
    other=owner.resident_covariance(second)
    assert resident.background is other.background
    np.testing.assert_array_equal(resident.block_array(0,13).numpy(),snapshot)
    replacement=TensorProducts(backend,sample)
    assert replacement._resident_background is None
    assert replacement._kernel_cache is not owner._kernel_cache


@pytest.mark.skipif(not torch.backends.mps.is_available() or not hasattr(torch.mps,'compile_shader'),reason='Native Metal shader API required')
def test_fused_pupil_accumulation_matches_torch_and_rejects_negative_sf():
    from types import SimpleNamespace
    rng=np.random.default_rng(67)
    ys,xs=np.nonzero(np.ones((13,13)))
    lag=np.arange(-12,13);n=32
    sample=SimpleNamespace(xs=xs,ys=ys,_lag_count=len(lag)**2,
        _lag_x=lag,_lag_y=lag,n=n,
        _lag_fft_indices=((lag[:,None]%n)*n+lag[None,:]%n).ravel())
    class Owner:
        torch=torch
        def upload(self,a,integer=False):return torch.tensor(a,device='mps',dtype=torch.int64 if integer else torch.float32)
        def zeros(self,shape):return torch.zeros(shape,device='mps')
        def to_host(self,a):return a.cpu().numpy().astype(float)
    owner=Owner();amplitude=rng.uniform(.5,1,len(xs));phasor=np.exp(1j*rng.normal(size=len(xs)))
    optics=DevicePupilIntegrator(owner,sample,amplitude,phasor,tile_rows=32)
    assert optics.fused is not None
    factors=rng.normal(size=(len(xs),11))*.12
    covariance=owner.upload(factors@factors.T)
    class Resident:
        sampling=sample
        optical_diagonal=covariance.diag().contiguous()
        preparation_seconds=0.
        background_prepared=False
        def optical_block_array(self,first,last):return covariance[first:last,first:].contiguous()
        def variance(self):return 0.
    resident=Resident();resident.owner=owner
    fused,_=optics.integrate(resident)
    optics.fused=None
    plain,_=optics.integrate(resident)
    np.testing.assert_allclose(fused,plain,rtol=2e-6,atol=2e-7)
    assert fused[0,0]==1
    optics=DevicePupilIntegrator(owner,sample,amplitude,phasor,tile_rows=32)
    covariance[0,1]=covariance[1,0]=100
    with pytest.raises(ValueError,match='negative structure function'):
        optics.integrate(resident)


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='Native Metal required')
def test_native_optical_reuse_preserves_owner_on_static_map_change(tmp_path):
    from tiptop.ho_optics import finite_pupil_fields
    native=coupled_model(tmp_path/'case',covariance_backend='mps')
    native.removeMeanSlopes=True
    sample=sampling(native);pupil=native.ao.tel.pupil
    def run(opd):
        return finite_pupil_fields(native,pupil,opd,sample.n,sample.native_side,sample.period,[0])[0][0].sampling
    off=run(None);initial=native._finitePupilPreparation
    assert np.isfinite(off).all()
    y,x=np.indices(pupil.shape);opd=10*np.sin(x/3)+4*np.cos(y/4)
    mapped=run(opd);changed=native._finitePupilPreparation
    assert changed is not initial
    assert changed.sampling is initial.sampling
    assert changed.backend is initial.backend
    again=run(opd.copy());assert native._finitePupilPreparation is changed
    np.testing.assert_allclose(again,mapped,rtol=2e-6,atol=2e-9)
    final=run(None)
    np.testing.assert_allclose(final,off,rtol=2e-6,atol=2e-9)


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='Native Metal required')
def test_optional_error_budget_keeps_serial_preparation(tmp_path,monkeypatch):
    from configparser import ConfigParser
    from p3.aoSystem.fourierModel import fourierModel
    from tiptop.ho_optics import finite_pupil_fields
    from test_p3_bugfixes_noise_tomo import _p3_path
    case=tmp_path/'case';coupled_model(case)
    config=ConfigParser();config.optionxform=str;config.read(case/'system.ini')
    config['sources_science'].update(Zenith='[0.,10.]',Azimuth='[0.,45.]')
    with (case/'system.ini').open('w') as stream:config.write(stream)
    native=fourierModel(str(case/'system.ini'),path_root=_p3_path(),calcPSF=False,
        display=False,verbose=False,reduce_memory=False,computeOpticalPSD=True,
        computeFocalAnisoCov=False,finiteApertureBackend='mps')
    native.removeMeanSlopes=True;native.getErrorBreakDown=True
    b=native._finite_aperture_backend();sample=sampling(native)
    def unexpected(*args):raise AssertionError('Error-budget prediction must remain serial')
    monkeypatch.setattr(b,'directional_preparation',unexpected)
    images,profiles,_=finite_pupil_fields(native,native.ao.tel.pupil,None,
        sample.n,sample.native_side,sample.period,[0,1])
    assert len(images)==2 and len(native.finiteErrorBreakDown)==2
    assert all(not p['science_rows_pipelined'] for p in profiles)
    assert all(abs(image.sampling.sum()-1)<1e-12 for image in images)
