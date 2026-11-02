"""Guide-star optics have their own wavelength, directions and static path."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
from tiptop import ho_optics


def test_guide_sensor_excludes_science_static_map_and_preserves_parent(monkeypatch):
    config=dict(telescope=dict(PathStaticOn='science.fits',extraErrorNm=120.,
            StaticOnRmsCompensation=True),sensor_science=dict(PixelScale=6.),
        sources_science=dict(Wavelength=[1.2e-6],Zenith=[0.],Azimuth=[0.]),
        DM_science=dict(DmPitchs=[1.]),RTC_science=dict(LoopGain_HO=1.),
        sensor_HO=dict(GridOffsets=[[.2,.2]],ReconstructionModel='alias_aware'),
        sources_LO={},sensor_LO={},RTC={})
    before=deepcopy(config);record={}
    native=SimpleNamespace(freq=SimpleNamespace(nOtf=16,PSDstep=.25,
        kxAO_=np.array([[0.,0.],[.25,.25]]),dk_=1.,wvlRef=.6e-6),
        ao=SimpleNamespace(tel=SimpleNamespace(pupil=np.ones((8,8)))),
        opticalPSD=np.zeros((16,16,1)),initComputations=lambda:None)
    def model(*args,**kwargs):
        record.update(kwargs['config_dict']);return native
    monkeypatch.setattr(ho_optics,'fourierModel',model)
    monkeypatch.setattr(ho_optics,'psdSetToPsfSet',lambda *a,**k:[])
    sim=SimpleNamespace(my_data_map=config,LO_wvl=.6e-6,LO_zen_field=[30.],
        LO_az_field=[120.],psInMas=6.,wvlRef=1.2e-6,
        fullPathFilename='unused',tel_radius=2.,finiteApertureReconstruction=False)
    _,_,provenance=ho_optics.guide_star_optics(sim)
    assert config==before
    assert record['sources_science']==dict(Wavelength=[.6e-6],Zenith=[30.],Azimuth=[120.])
    assert record['sensor_science']['PixelScale']==3.
    assert record['telescope']['PathStaticOn']==''
    assert record['telescope']['extraErrorNm']==0.
    assert 'sensor_LO' not in record and 'sources_LO' not in record
    assert 'DM_science' not in record and 'RTC_science' not in record
    assert record['sensor_HO']==before['sensor_HO']
    assert provenance['static_map_applied'] is False
    assert provenance['science_mirror_applied'] is False


def test_sensor_specific_extra_error_is_not_silently_discarded():
    import pytest
    sim=SimpleNamespace(my_data_map={'telescope': {'extraErrorLoNm': [1.,0.]}}, LO_wvl=.71e-6)
    with pytest.raises(ValueError, match='extraErrorLoNm'):
        ho_optics.guide_star_optics(sim)
