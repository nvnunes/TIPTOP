"""Native CUDA parity; skipped without the optional CUDA runtime."""
import numpy as np
import pytest
from scipy import fft
from test_finite_two_stage import coupled_model, sampling, require_device
from p3.aoSystem.finiteTwoStageCupy import CupyFiniteProducts
from tiptop.finite_cuda_optics import CudaPupilIntegrator


@pytest.mark.parametrize('single', [False, True])
@pytest.mark.parametrize('means', [False, True])
def test_cuda_resident_queries_covariance_and_pairs(tmp_path, single, means):
    require_device('cuda')
    native=coupled_model(tmp_path/'case',enabled=not single)
    native.removeMeanSlopes=means
    sample=sampling(native); b=native._finite_aperture_backend()
    expected_residual,_=b.residual_blocks(sample,0)
    expected,checks=sample.psf_blocks(expected_residual.diagonal,expected_residual.blocks())
    expected_blocks=list(expected_residual.blocks(rows=13))
    b.device=CupyFiniteProducts(b,sample)
    if single:
        # Single-stage native residual queries use the same device owner.
        b.device.prefetch([0])
    actual_residual,_=b.residual_blocks(sample,0)
    resident=b.device.resident_covariance(actual_residual)
    for first,block in expected_blocks:
        np.testing.assert_allclose(resident.block_array(first,first+len(block)).get(),block,rtol=2e-10,atol=2e-10)
    phasor=np.exp(.2j*np.sin(sample.xy[:,0])); sample._static_phasor=phasor
    expected,_=sample.psf_blocks(expected_residual.diagonal,expected_residual.blocks())
    optics=CudaPupilIntegrator(b.device,sample,sample.amplitude,phasor,tile_rows=16)
    otf,actual_checks=optics.integrate(resident)
    actual=fft.fftshift(fft.fft2(otf)).real/sample.n**2
    assert abs(actual-expected).sum()<1e-9
    assert abs(actual.sum()-1)<2e-12
    np.testing.assert_allclose(actual_checks['residual_phase_variance'],checks['residual_phase_variance'],rtol=2e-10,atol=2e-10)
    again=b.device.resident_covariance(actual_residual)
    assert again.background is resident.background
