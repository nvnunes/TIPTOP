Parameter files explained
=========================

Introduction
------------

To run TIPTOP you need two things: to execute the function and to create a parameter file. This section explaines
what should your parameter file contain and the various parameters you can set. You can already find example parameter 
files in the `github <https://github.com/astro-tiptop/TIPTOP/tree/main/tiptop/perfTest>`_ .


The parameter files are divided in sections and they can contain multiple parameter. It is very important that each 
parameter be placed in the appropriate section. The section starts with its name between '[]' and ends either with 
the end of file or with the next section. The order they are placed in the file does not matter.

Overview
--------
The following table resume the what the parameter file should contain and what is mandatory.

+---------------+--------------------------------+
|section        | required?                      |
+===============+================================+
|telescope      | Yes                            |
+---------------+--------------------------------+
|atmosphere     | Yes                            |
+---------------+--------------------------------+
|sources_science| Yes                            |
+---------------+--------------------------------+
|source_HO      | Yes                            |
+---------------+--------------------------------+
|sources_LO     | No/Yes if sensor_LO defined    |
+---------------+--------------------------------+
|sources_Focus  | No/Yes if sensor_Focus defined |
+---------------+--------------------------------+
|sensor_science | Yes                            |
+---------------+--------------------------------+
|sensor_HO      | Yes                            |
+---------------+--------------------------------+
|sensor_LO      | No/Yes if sources_LO defined   |
+---------------+--------------------------------+
|sensor_Focus   | No/Yes if sources_Focus defined|
+---------------+--------------------------------+
|DM             | Yes                            |
+---------------+--------------------------------+
|RTC            | No/Yes if sensor_LO defined    |
+---------------+--------------------------------+
|Computation    | No                             |
+---------------+--------------------------------+


We now go more in detail for each section:

.. warning::

   In the current state, if the parameter file is not found, it raises ``NoSectionError: No section: 'telescope'``.

[telescope]
-----------

+--------------------------+----------+-------+--------------------------------------------------------------------------+
| Parameter                | Required | Type  | Description                                                              |
+==========================+==========+=======+==========================================================================+
|TelescopeDiameter         |Yes       |float  |Set the outer diameter of the telescope pupil in unit of meters.          |
+--------------------------+----------+-------+--------------------------------------------------------------------------+
|Resolution                |No        |integer|*Default : 256*, Number of pixels across the pupil diameter.              |
|                          |          |       |This value is used in computation of the telescope OTF.                   |
+--------------------------+----------+-------+--------------------------------------------------------------------------+
|ObscurationRatio          |No        |float  |*Default : 0.0*, Defines the central obstruction                          |
|                          |          |       |due to the secondary as a ratio of the ``TelescopeDiameter``.             |
+--------------------------+----------+-------+--------------------------------------------------------------------------+
|ZenithAngle               |No/Yes if |float  |*Default: 0.0*, Set the pointing direction of the telescope in degree     |
|                          |LO        |       |with respect to the zenith. Used to compute airmass, to scale atmospheric |
|                          |          |       |layers and stars altitude.                                                |
+--------------------------+----------+-------+--------------------------------------------------------------------------+
|PupilAngle                |No        |float  |*default : 0.0*, Rotation angle of the telescope pupil in degrees.        |
|                          |          |       |Applied to pupil mask and static aberration maps to match instrument      |
|                          |          |       |orientation.                                                              |
+--------------------------+----------+-------+--------------------------------------------------------------------------+
|PathPupil                 |No        |string |*default: ''*, Path to the pupil model in .fits file (if provided,        |
|                          |          |       |the pupil model is interpolated). if absent or '', not used.              |
+--------------------------+----------+-------+--------------------------------------------------------------------------+
|PathStaticOn              |No        |string |*default: None*, Path to a fits file containing an on-axis static         |
|                          |          |       |aberration map ([nm]). This parameter can be used to add any kind of      |
|                          |          |       |static aberrations. Example: the static aberration of the `ELT M1`_.      |
|                          |          |       |If absent or '', not used.                                                |
+--------------------------+----------+-------+--------------------------------------------------------------------------+
|StaticOnRmsCompensation   |No        |bool   |Default False. Opt in to measured RMS compensation for PathStaticOn.      |
|                          |          |       |Requires pupil_fit; restores amplitude lost during map resampling.        |
+--------------------------+----------+-------+--------------------------------------------------------------------------+
|zCoefStaticOn             |No        |list of|*default: None*, Combination of zernike modes that models an on-axis      |
|                          |          |float  |static aberration. Coefficients are in [nm RMS].                          |
|                          |          |       |Examples: Focus ``[0,0,100]``; Astigmatism ``[0,0,0,100]``;               |
|                          |          |       |Trefoil ``[0,0,0,0,0,0,0,100]``; Spherical ``[0,0,0,0,0,0,0,0,0,100]``.   |
|                          |          |       |If absent not used.                                                       |
+--------------------------+----------+-------+--------------------------------------------------------------------------+
|PathStaticOff             |No        |string |*default: None*, Path to a .fits file that contains field-dependent       |
|                          |          |       |(off-axis) static aberration maps. Must be provided together with         |
|                          |          |       |``PathStaticPos`` specifying the corresponding positions.                 |
|                          |          |       |If absent or '', not used.                                                |
+--------------------------+----------+-------+--------------------------------------------------------------------------+
|PathStaticPos             |No        |string |*default: None*, Required if ``PathStaticOff``. Path to a fits file that  |
|                          |          |       |contains the field positions [zenith in arcsec, azimuth in rad]           |
|                          |          |       |corresponding to each off-axis static aberration map in ``PathStaticOff``.|
+--------------------------+----------+-------+--------------------------------------------------------------------------+
|PathApodizer              |No        |string |*default: ''*, Path to a fits file that contains an amplitude apodizer    |
|                          |          |       |map. Used to apply pupil amplitude weighting (transmission mask) in the   |
|                          |          |       |simulation. if absent or '', not used.                                    |
+--------------------------+----------+-------+--------------------------------------------------------------------------+
|PathStatModes             |No        |string |*default: ''*, path to a .fits file that contains a cube of static        |
|                          |          |       |aberration modes. Each mode is normalized to have 1 nm RMS amplitude.     |
|                          |          |       |If absent or '', not used.                                                |
+--------------------------+----------+-------+--------------------------------------------------------------------------+
|windPsdFile               |No        |string |*default: ''*, file name of a fits file with a 2D array with a frequency  |
|                          |          |       |vector and PSD of tip and tilt windshake.                                 |
+--------------------------+----------+-------+--------------------------------------------------------------------------+
|extraErrorNm              |No        |float  |*default: 0*, nm RMS of the additional wavefront error to be added (an    |
|                          |          |       |error that is not otherwise considered). This parameter is used to define |
|                          |          |       |a PSD that is summed to the AO PSD. The default power law is f^(-2), but  |
|                          |          |       |is can be modified using the ``extraErrorExp`` parameter. It models a     |
|                          |          |       |generic static aberration.                                                |
+--------------------------+----------+-------+--------------------------------------------------------------------------+
|extraErrorExp             |No        |float  |*default: -2*, Exponent of the power of spatial frequencies used to       |
|                          |          |       |generate the PSD associated with extraErrorNm                             |
+--------------------------+----------+-------+--------------------------------------------------------------------------+
|extraErrorMin             |No        |float  |*default: 0*, Minimum spatial frequency ([m^(-1)]) for which PSD          |
|                          |          |       |associated with extraErrorNm is > 0                                       |
+--------------------------+----------+-------+--------------------------------------------------------------------------+
|extraErrorMax             |No        |float  |*default: 0*, Maximum spatial frequency ([m^(-1)]) for which PSD          |
|                          |          |       |associated with extraErrorNm is > 0                                       |
|                          |          |       |                                                                          |
|                          |          |       |Note: 0 means maximum frequency is the one present in the spatial         |
|                          |          |       |frequency array of the PSDs.                                              |
+--------------------------+----------+-------+--------------------------------------------------------------------------+
|extraErrorLoNm            |No        |float  |*default: 0*, nm RMS of the additional wavefront error to be added (an    |
|                          |          |       |error that is not otherwise considered).                                  |
|                          |          |       |                                                                          |
|                          |          |       |It can be a list of two values, the on-axis error and the error at the    |
|                          |          |       |edge of the technical field ([telescope]TechnicalFoV)                     |
|                          |          |       |                                                                          |
|                          |          |       |Note: (1) only makes sense if [sensor_LO] is present (2) if not present   |
|                          |          |       |extraErrorNm is used on LO directions.                                    |
+--------------------------+----------+-------+--------------------------------------------------------------------------+
|extraErrorLoExp           |No        |float  |*default: -2*, exponent of the power of spatial frequencies used to       |
|                          |          |       |generate the PSD associated with extraErrorLoNm                           |
+--------------------------+----------+-------+--------------------------------------------------------------------------+
|extraErrorLoMin           |No        |float  |*default: 0*, minimum spatial frequency for which PSD associated with     |
|                          |          |       |extraErrorLoNm is > 0                                                     |
+--------------------------+----------+-------+--------------------------------------------------------------------------+
|extraErrorLoMax           |No        |float  |*default: 0*, maximum spatial frequency for which PSD associated with     |
|                          |          |       |extraErrorLoNm is > 0                                                     |
|                          |          |       |                                                                          |
|                          |          |       |Note: 0 means maximum frequency is the one present in the spatial         |
|                          |          |       |frequency array of the PSDs.                                              |
+--------------------------+----------+-------+--------------------------------------------------------------------------+
|jitter_FWHM               |No        |float  |*default: None*, additional kernel to be convolved with PSF, it could be  |
|                          |          |       |a scalar (FWHM in mas) for a round kernel or a list of three values       |
|                          |          |       |[FWHM_mas_max, FWHM_mas_min, angle_rad]. It models an additional tip/tilt |
|                          |          |       |jitter (e.g. vibrations, PSF drifts, ...).                                |
+--------------------------+----------+-------+--------------------------------------------------------------------------+
|glFocusOnNGS              |No        |bool   |*default: False*, global focus control with natural guide stars.          |
|                          |          |       |Multi-conjugate systems only. Requires NumberLenslets >= 2 in sensor_LO or|
|                          |          |       |a specific global focus sensor (``[sources_Focus]`` and ``[sensor_Focus]``|
|                          |          |       |sections).                                                                |
+--------------------------+----------+-------+--------------------------------------------------------------------------+
|TechnicalFoV              |No/Yes if |float  |*default: 0.0*, Diameter of the technical field of view in [arcsec]. In   |
|                          |LO        |       |MCAO/LGS configurations, used when ``NumberActuators`` from ``[DM]``      |
|                          |          |       |section is not set: scales the projected DM size with altitude and        |
|                          |          |       |derives the actuator count from ``DmPitchs``. In LO, also sets the        |
|                          |          |       |angular range for interpolating additional low-order error terms          |
|                          |          |       |(``extraErrorLoNm``).                                                     |
|                          |          |       |*Warning*: Mandatory and no default if LO section is used.                |
+--------------------------+----------+-------+--------------------------------------------------------------------------+

.. _ELT M1: https://github.com/astro-tiptop/TIPTOP/blob/main/tiptop/data/ELT_M1_MORFEO_DMs_static_wfe_480px.fits


[atmosphere]
------------

+-------------------------+------------+-------+--------------------------------------------------------------------------+
| Parameter               |  Required  | Type  | Description                                                              |
+=========================+============+=======+==========================================================================+
|Seeing                   |Yes, unless |float  |Set the seeing at Zenith in [arcsec]. Used to compute ``r0`` as           |
|                         |``r0_value``|       |``r0 = 0.976 × λ / Seeing(rad)``. If not set, TipTop uses ``r0_value``.   |
|                         |given       |       |                                                                          |
+-------------------------+------------+-------+--------------------------------------------------------------------------+
|r0_value                 |Yes, unless |float  |Set the atmosphere Fried parameter ``r0`` in [meters]. Used directly      |
|                         |``Seeing``  |       |if ``Seeing`` is not provided.                                            |
|                         |given       |       |                                                                          |
+-------------------------+------------+-------+--------------------------------------------------------------------------+
|Wavelength               |No/Yes if LO|float  |*Default : 500e-9*, Wavelength at which the atmospheric statistics are    |
|                         |            |       |defined (in meters).                                                      |
|                         |            |       |                                                                          |
|                         |            |       |*Warning*: Mandatory and no default if LO section is used.                |
+-------------------------+------------+-------+--------------------------------------------------------------------------+
|L0                       |No/Yes if LO|float  |*Default : 25.0*, Outer Scale of the atmosphere  in meters                |
|                         |            |       |                                                                          |
|                         |            |       |*Warning*: not optional in MavisLO.py                                     |
+-------------------------+------------+-------+--------------------------------------------------------------------------+
|Cn2Weights               |No/Yes      |list of|*Default : [1.0]*, Relative contribution of each layer. The sum of all the|
|                         |if LO       |float  |list element must be 1. Must have the same length as ``Cn2Heights``,      |
|                         |            |       |``WindSpeed`` and ``WindDirection``.                                      |
|                         |            |       |                                                                          |
|                         |            |       |*Warning : required if ``Cn2Heights``, ``WindSpeed`` or ``WindDirection`` |
|                         |            |       |are defined                                                               |
|                         |            |       |*Warning* : extremely confusing error message if absent when it must be   |
|                         |            |       |defined                                                                   |
+-------------------------+------------+-------+--------------------------------------------------------------------------+
|Cn2Heights               |No/Yes      |list of|*Default : [0.0]*, altitude of layers in [meters].                        |
|                         |if LO       |float  |Must have the same length as ``Cn2Weights``, ``WindSpeed`` and            |
|                         |            |       |``WindDirection``.                                                        |
|                         |            |       |                                                                          |
|                         |            |       |*Warning* : required if ``Cn2Weights``, ``WindSpeed`` or ``WindDirection``|
|                         |            |       |are defined                                                               |
|                         |            |       |*Warning* : extremely confusing error message if absent when it must be   |
|                         |            |       |defined                                                                   |
+-------------------------+------------+-------+--------------------------------------------------------------------------+
|WindSpeed                |No/Yes      |list of|*Default : [10.0]*, Wind speed values for each layer in [m/s].            |
|                         |if LO       |float  |Must have the same length as ``Cn2Weights``, ``Cn2Heights`` and           |
|                         |            |       |``WindDirection``.                                                        |
|                         |            |       |                                                                          |
|                         |            |       |*Warning* : required if ``Cn2Weights``, ``Cn2Heights`` or                 |
|                         |            |       |``WindDirection`` are defined                                             |
|                         |            |       |*Warning* : extremely confusing error message if absent when it must be   |
|                         |            |       |defined                                                                   |
+-------------------------+------------+-------+--------------------------------------------------------------------------+
|WindDirection            |No          |list of|*Default : a list of 0 of the length of WindSpeed*, wind direction for    |
|                         |            |float  |each layer in [degrees]. 0 degree is alogn the x axis then anticlockwise. |
|                         |            |       |Must have the same length as ``Cn2Weights``, ``Cn2Heights`` and           |
|                         |            |       |``WindSpeed``.                                                            |
+-------------------------+------------+-------+--------------------------------------------------------------------------+
|testWindspeed            |No          |float  |Used only for tests                                                       |
+-------------------------+------------+-------+--------------------------------------------------------------------------+

[sources_science]
-----------------

+-------------------------+---------+--------+--------------------------------------------------------------------------+
| Parameter               | Required| Type   | Description                                                              |
+=========================+=========+========+==========================================================================+
|Wavelength               |Yes      |list of |list of wavelengths in meters.                                            |
|                         |         |float   |                                                                          |
|                         |         |or float|When more than one elements is present the output PSF saved in the fits   |
|                         |         |        |file is a 4D array with dimension (Nw, Ns, Npix, Npix), where Nw is the   |
|                         |         |        |number of wavelengths required ([sources_science] Wavelength), Ns is the  |
|                         |         |        |number of directions required ([sources_science] Zenith and Azimuth) and  |
|                         |         |        |Npix is the size required for the PSFs ([sensor_science] FieldOfView).    |
|                         |         |        |If a single elements is present the fits file is a 3D array with          |
|                         |         |        |dimension (Ns, Npix, Npix).                                               |
|                         |         |        |Instead the profiles will be a 3D array (fourth fits file extension) with |
|                         |         |        |dimensions (2*Nw, Ns, Npix/2). The first Nw elements contain the radius   |
|                         |         |        |and the second Nw elements the profile values (the first radius and       |
|                         |         |        |profile pair is radius=data[0,0,:] profile=data[Nw,0,:], the second is    |
|                         |         |        |radius=data[1,0,:] profile=data[Nw+1,0,:], ...)                           |
|                         |         |        |json file: two lists, radius and psf with dimensions (Nw, Ns, Npix/2).    |
|                         |         |        |                                                                          |
|                         |         |        |In this case more memory is required and small differences with respect   |
|                         |         |        |to monochromatic PSF will be present because: (1) errors Differential     | 
|                         |         |        |refractive anisoplanatism and Chromatism from P3 are computed for a       |
|                         |         |        |single wavelength (the shortest one) (2) effective field-of-view of the   |
|                         |         |        |PSF is typically larger to guarantee that the PSF at the shortest         |
|                         |         |        |wavelength has the required field-of-view (3) The PSF is typically        |
|                         |         |        |computed with a higher sampling to guarantee that the longest wavelength  |
|                         |         |        |has the required sampling and then the PSFs at the shorter wavelengths    |
|                         |         |        |are rebinned.                                                             |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|Zenith                   |Yes      |list of |Zenithal coordinate in arcsec (distance from axis) of science sources.    |
|                         |         |float   |Must be the same length as ``Azimuth``                                    |
+-------------------------+---------+--------+--------------------------------------------------------------------------+   
|Azimuth                  |Yes      |list of |Azimuthal coordinate in degree (angle from the ref. direction: polar axis |
|                         |         |float   |is x-axis) of science sources. Must be the same length as ``Zenith``      |
+-------------------------+---------+--------+--------------------------------------------------------------------------+

[sources_HO]
------------

+-------------------------+---------+--------+--------------------------------------------------------------------------+
| Parameter               | Required| Type   | Description                                                              |
+=========================+=========+========+==========================================================================+
|Wavelength               |Yes      |float   |Sensing wavelength for Hight Order modes in meters,                       |
|                         |         |        |*Warning* : gives a confusing error message if absent                     |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|Zenith                   |No       |list of |*Default : [0.0]*, Zenithal coordinate of each guide stars in arcsec      |
|                         |         |float   |(distance from axis). Must be the same length as ``Azimuth``, Even if     |
|                         |         |        |``Azimutal`` is defined, this is optional.                                |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|Azimuth                  |No       |list of |*Default : [0.0]*, Azimuthal coordinate in degree (angle from the ref.    |
|                         |         |float   |direction: polar axis is x-axis) of each guide stars.                     |
|                         |         |        |Must be the same length as ``Zenith``, even if ``Zenith`` is defined,     |
|                         |         |        |this is optional.                                                         |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|Height                   |No       |float   |*Default : 0.0*, altitude of the guide stars (0 if infinite). Consider    |
|                         |         |        |that all guide star are at the same height.                               |
+-------------------------+---------+--------+--------------------------------------------------------------------------+

[sources_LO]
------------
.. note::

   This section is completely optional (``[sensor_LO]`` section is required to have the LO part simulated)

+-------------------------+---------+--------+--------------------------------------------------------------------------+
| Parameter               | Required| Type   | Description                                                              |
+=========================+=========+========+==========================================================================+
|Wavelength               |Yes      |float   |Sensing wavelength for Low Order modes in meters.                         |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|Zenith                   |Yes      |list of |Zenithal coordinate of each guide stars in arcsec (distance from axis).   |
|                         |         |float   |Must be the same length as ``Azimuth``                                    |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|Azimuth                  |Yes      |list of |Azimuthal coordinate in degree (angle from the reference direction: polar |
|                         |         |float   |axis is x-axis) of each guide stars.                                      |
|                         |         |        |Must be the same length as ``Zenith``                                     |
+-------------------------+---------+--------+--------------------------------------------------------------------------+   

[sources_Focus]
---------------
.. note::

   This section is completely optional.
   The ``[sources_Focus]`` section is required to have the global focus part simulated considering specific focus sensors and not the LO sensors.
   This happens when the key ``glFocusOnNGS`` in the ``[telescope]`` section is True and multiple DMs are present.

   Note that the coordinates (``Zenith`` and ``Azimuth``) of the NGSs are the same of the ``[sources_LO]`` section.

+-------------------------+---------+--------+--------------------------------------------------------------------------+
| Parameter               | Required| Type   | Description                                                              |
+=========================+=========+========+==========================================================================+
|Wavelength               |Yes      |float   |Sensing wavelength for global focus modes in meters                       |
+-------------------------+---------+--------+--------------------------------------------------------------------------+

[sensor_science]
----------------

+-------------------------+---------+--------+--------------------------------------------------------------------------+
| Parameter               | Required| Type   | Description                                                              |
+=========================+=========+========+==========================================================================+
|PixelScale               |Yes      |float   |Pixel/spaxel scale in milliarcsec.                                        |
|                         |         |        |                                                                          |
|                         |         |        |*Warning*: confusing error message if missing                             |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|FieldOfView              |Yes      |integer |Field of view of the camera in pixel/spaxel.                              |
|                         |         |        |If -1 the automatic choice uses the maximum between twice the             |
|                         |         |        |AO-corrected size and twice the seeing.                                   |
|                         |         |        |                                                                          |
|                         |         |        |*Warning*: confusing error massage if missing                             |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|Super_Sampling           |No       |float   |Desired radial interpolation sampling step in milliarcsec.                |
|                         |         |        |                                                                          |
|                         |         |        |If provided, TipTop performs a 2D polar interpolation of the PSF to       |
|                         |         |        |produce a radial profile resampled at the requested scale.                |
+-------------------------+---------+--------+--------------------------------------------------------------------------+

.. note::

    Following parameters were added to uniformise all the sensor (HO and LO), but they are not used.

    Binning, NumberPhotons, SpotFWHM, SpectralBandwidth, Transmittance, Dispersion, SigmaRON, Dark, SkyBackground, Gain, ExcessNoiseFactor, Wavelength, FieldOfView

[sensor_HO]
-----------

``TipTiltRemoval`` selects the P3 backend's tilt treatment:

* ``'auto'`` (default): use ``'spectral'`` when ``[sensor_LO]`` is present,
  otherwise ``'none'``. This preserves existing behavior.
* ``'none'``: retain tilt in the residual spectrum and optical calculation.
* ``'spectral'``: apply P3's existing approximate PSD tilt filter, independently
  of whether LO processing is enabled.
* ``'pupil_fit'``: request optical residual power from P3 before the
  variance-only piston factors on the main residual and aliases, fit two ramps on the actual
  full telescope pupil, and average the projected Gaussian coherence through
  the native optics. Static OPD tilt is removed by the same fit, matching a fit
  after NCPA is added. The existing PSF resampling, flux and LO convolution
  contracts are retained. This optional path uses CPU pupil-pair integration.

Set, for example, ``TipTiltRemoval = 'pupil_fit'`` under ``[sensor_HO]`` for an
HO-only fitted-pupil prediction. Explicit choices are independent of LO
availability. The replaced ``FilterTipTilt`` name is rejected; it is not an
alias. P3's ``PSD`` and component error budget retain their variance-reporting
convention. Its separate ``opticalPSD`` supplies the covariance input to
MASTSEL only for ``'pupil_fit'``. Reconstruction weights and the other tilt
selections are unchanged. TIPTOP's ``HO_res`` and NGS/focus Strehl estimates use the
projected dynamic phase variance returned by the optical calculation.
That metadata is a pupil-weighted post-tilt phase variance; it is distinct
from P3's piston-filtered component budget. Do not interpret an integral of
the unfiltered optical power as a piston-removed wavefront variance.

``GridOffsets`` supplies one physical ``[x, y]`` displacement per HO WFS
in subaperture units. Omitted or all-zero offsets retain coincident grids
exactly. Nonzero offsets enter the scalar tomographic sensing operator and
require multiple Shack-Hartmann WFSs. The sign describes displacement of the
sampling cells, rather than interpolation of a query location. Shifted grids
can provide complementary measurements for correction beyond one grid's
sampling band; the configured reconstruction and DM determine that correction.

``ReconstructionModel`` selects the complete sensing/reconstruction model:

* ``'standard'`` (default when omitted): existing ordinary P3 reconstruction,
  combined slopes and its existing alias propagation/truncation.
* ``'alias_aware'``: frequency-local reconstruction with separate x/y slopes,
  full correlated alias covariance, symmetric alias cutoff and propagation
  through the actual reconstructor. Continuous square-cell sensing, configured
  shifts, main residual, alias and noise all use the same measurement basis.
* ``'finite_aperture'``: finite valid-lenslet sensing, physical layer-dependent
  LGS footprints, physical phase-prior reconstruction and Gaussian actuator
  fitting. Per-LGS mean-slope removal includes complete POLC feedback. TIPTOP
  forms PSFs from the full residual covariance; it does not build or add the
  ``alias_aware`` estimator or atmospheric PSD.

The selector never changes ``NumberLenslets``, ``GridOffsets``, pupil geometry,
``TipTiltRemoval``, ``TomographicRegularizationFloor`` or physical controller
parameters (loop gain, frame rate and delay). ``alias_aware`` currently supports
multiple equal-pitch Shack-Hartmann WFSs and one common ground-conjugate DM.
The finite model shares these restrictions. Both models can add a directional
science mirror through paired ``DM_science`` and ``RTC_science`` sections;
see :doc:`twoStageFourier` for the supported two-stage scope.

The retired keys ``AliasAwareReconstruction``, ``FiniteApertureReconstruction``,
``TomographicAliasing``, ``SymmetricAliasing`` and ``SuperResolution`` are
rejected, even if false. Remove them and choose a model explicitly. Earlier
experimental combinations are not automatically reinterpreted; reproduce them
with their recorded frozen sources and configurations. No separate slope-mode
selector is provided. Non-None ``PhaseSamplesPerSubaperture`` and ``AliasOrder``
remain unsupported.

``TomographicRegularizationFloor`` is a finite nonnegative fraction, default
``1e-12``, that sets the minimum diagonal regularization of P3's tomographic
measurement covariance at each spatial frequency. P3 multiplies it by that
frequency's largest covariance diagonal and uses the larger of this floor
and each measurement channel's noise variance. The largest diagonal is
evaluated in the active measurement basis; this is not a fixed absolute
regularization. It changes reconstruction
weights and can materially affect propagated alias power; it does not inject
measurement noise. Zero disables the relative floor. Zero-covariance bins
retain a positive numerical diagonal and zero reconstructed signal.
The option is per model; setting it does not change other predictions.

``ReconstructionRegularization`` applies only to ``finite_aperture`` and defaults
to ``0.0``. It adds that nonnegative fraction of the mean prior slope variance
to the estimator covariance's diagonal. It is separate from
``TomographicRegularizationFloor`` and does not inject measurement noise.

The selected model owns its temporal response and measurement averaging:

* ``standard``: existing integrator behaviour and the existing duration derived
  from each detector's clock rate and the loop frame rate.
* ``alias_aware`` and ``finite_aperture``: the accepted queued scalar POLC
  response and instantaneous measurements (duration ``0.0``).

``TemporalModel_HO`` and ``ExposureTime`` are retired keys. Supplying either
raises a migration error, even if it matches the selected model. They are not
read as overrides. Historical timing/exposure combinations require their
frozen sources and configurations. Gain, frame rate and command delay remain
explicit physical inputs under ``[RTC]``.

For ``standard`` with ``TipTiltRemoval='pupil_fit'``, setting
``[telescope] StaticOnRmsCompensation=True`` makes static-map resampling preserve the original intensity-weighted, piston-removed RMS using its measured
original/resampled ratio. The factor is recomputed on each actual wavelength's
pupil grid and applies to the phase and its original-pupil tilt fit together.
The supplied OPD array, reconstruction and atmospheric spectrum are unchanged.
This restores amplitude, not lost spatial structure. Other tilt-removal
selections and omitted/zero maps retain their existing behavior. TIPTOP applies the ratio before
calling MASTSEL's unchanged public optics API, separately for each actual
wavelength grid. The map supplied by the caller stays unchanged.

``StaticOnRmsCompensation`` defaults to False and must be a boolean. It applies
only to the common map supplied by ``PathStaticOn`` in standard pupil-fit
optics. Finite-aperture optics always uses RMS-preserving resampling, regardless
of this option. It does not compensate field-dependent
``PathStaticOff`` maps. Alias-aware original-static-OTF optics does not resample
that phase map and is unaffected by this option.

For ``alias_aware``, TIPTOP's own spectral moments and bounded pupil-pair
calculation supply the dynamic covariance/coherence. Only MASTSEL's existing
public pupil-fit coefficients are reused; its LO model and optics API are
unchanged. A nonzero ``PathStaticOn`` retains the original
static OTF and estimates its dynamic pupil-fit transfer using complex-cell
averages. It requires one wavelength, pupil-fit removal, full-flux output,
operation without additional jitter and native detector sampling. Its accuracy
has been checked for the retained broad atmospheric spectra. The OTF ratio is
an approximation and need not preserve positive definiteness for arbitrary
narrow spectra; nonpositive optical results raise an error rather than being
clipped. Zero/omitted maps retain ordinary native pupil-fit optics.

With ``alias_aware``, channels are interleaved by guide star:
``g0-x, g0-y, g1-x, g1-y, ...``. ``NoiseVariance`` remains one per-axis
variance per guide star in existing P3 units; both axes receive that guide's
variance. Independent measurement noise is diagonal, while alias covariance
is correlated. Automatic noise follows the same mapping. Symmetric alias sums
include ``-nTimes`` through ``+nTimes`` on both axes, excluding the target.
This model does not couple different output Fourier frequencies.

The finite-aperture HO contract requires one science wavelength, zero injected
HO noise, multiple equal-pitch Shack-Hartmann WFSs, equal-height LGSs above every
layer, all physical reconstruction layers, and one common Gaussian ground mirror.
It uses ``TipTiltRemoval='pupil_fit'`` and model-owned POLC/instantaneous
measurements, including the feedback introduced by LGS mean-slope projection.
Paired science mirror/controller sections add a directional Gaussian ground
mirror for MOAO. Reconstruction runs on CPU; optional CUDA/Metal covariance
products for active two-stage finite aperture are described in :doc:`twoStageFourier`.

``PathStaticOn`` supplies deterministic nm OPD on the telescope
pupil, with the same original-pupil tilt fit as the dynamic residual. Nonzero
maps and zero/omitted maps all use the native optical pupil-pair quadrature
and the same blocked covariance-to-PSF calculation. For static maps, the
original-to-resampled intensity-weighted, piston-removed phase RMS ratio
compensates the measured interpolation loss before the original-pupil tilt
fit. This is a coarse-map approximation, not preservation of its complete
wide-angle halo. This model-owned treatment is always applied and is independent of
``StaticOnRmsCompensation``. It introduces no adjustable strength.
The supplied map and reconstruction weights remain unchanged.
The existing ``extraErrorNm`` spectrum
is supported as an independent residual covariance, using the same
normalization and pupil-fit removal as ordinary P3 optics; it does not apply
a deterministic NCPA map. Set it to zero when the same aberration is supplied
through ``PathStaticOn``. TIPTOP also requires
``fullFluxNormalization=True``. Unsupported combinations raise an error;
nonzero HO noise and other controllers have not been validated.

For single-stage LTAO and two-stage MOAO, TIPTOP can compose this HO prediction with MASTSEL's residual pointing
covariance. With ``pupil_fit`` and full-flux output, TIPTOP forms separate
native HO guide-star images at ``sources_LO.Wavelength`` and its configured
directions. These images use only the common mirror and exclude the directional
science mirror, science-arm static map and extra error;
their spot widths and circular sensor-aperture energies supply MASTSEL's
centroid-noise model. Finite-aperture NGS images retain the finite residual
covariance. The science PSF retains that same covariance and any deterministic
static map, then receives the existing Gaussian pointing convolution.

This handoff assumes independence between the tilt-removed HO residual and
the remaining pointing error. Its tested scope is one-lenslet tilt sensing,
without a focus loop, sensor-specific ``extraErrorLoNm`` or the optional extra
LO alias-error contribution. Those unsupported inputs are rejected by the
full-flux pupil-fit NGS path. Other tilt/normalization selections retain their
existing NGS processing. Selecting LO does not add the removed HO tilt again
or turn the finite residual into a diagonal PSD.

P3's instance-owned reconstruction and optical caches are keyed by physical
configuration, realized atmospheric/guide inputs, numerical policy and the
original pupil fit. Before reusing a finite calculation, P3 revalidates and
refreshes ``DM_science``/``RTC_science`` settings, including science-on/off
transitions. Changed settings rebuild the calculation; unchanged settings reuse it. TIPTOP retains the native pupil diffraction and known
high-frequency incoming covariance on its native optical quadrature,
sharing MASTSEL's public pupil-tilt fit coefficients with the
existing optics. Accuracy and cost must be validated for a new configuration.

The finite residual has no equivalent diagonal PSD: ``PSD`` and ``opticalPSD``
are ``None`` in P3/TIPTOP for this model. Ordinary reconstruction, controller
power propagation and error decomposition are skipped. The shared native
frequency grid and atmospheric setup are retained, together with independent
chromatic/refraction contributions. Gaussian targets use the physical weighted
optimization directions directly, without the Fourier projector's scalar normalization. The finite path includes incoming turbulence and applied
correction once; independent chromatic/refraction terms are added once.
TIPTOP reports finite pupil-removed residual variance and full-flux PSFs;
PSD export is rejected for this option. Direct P3 callers use
``calcPSF=False``, ``reduce_memory=False`` and
``finite_aperture_covariance`` with an optics sampling contract.

The High Order WaveFront Sensor can be a pyramid WFS or a Shack-Hartmann. Regardless of the WFS, the following parameters can de defined.

+-------------------------+---------+--------+--------------------------------------------------------------------------+
| Parameter               | Required| Type   | Description                                                              |
+=========================+=========+========+==========================================================================+
|WfsType                  |No       |string  |*default : 'Shack-Hartmann'*, Type of wavefront sensor used for the High  |
|                         |         |        |Order sensing. Other available option: 'Pyramid'                          |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|NumberLenslets           |No       |list of |*Default : [20]*, Number of WFS lenslets.  Used for both                  |
|                         |         |int     |Shack-Hartmann and Pyramid sensors. Also used for noise computation if    |
|                         |         |        |``NoiseVariance`` is not set.                                             |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|SizeLenslets             |No       |list of |*Default: ``[Telescope] TelescopeDiameter/[sensor_HO] NumberLenslet``*    |
|                         |         |float   |Size of WFS lenslets in meters. Overrides the ratio between telescope     |
|                         |         |        |size and Number of lenslet used to compute the matrix size.               |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|PixelScale               |Yes      |integer |High Order WFS pixel scale in [mas],  Not used when a pyramid wavefront   |
|                         |         |        |sensor has been selected.                                                 |
|                         |         |        |                                                                          |
|                         |         |        |*Warning*: gives a confusing error message if missing                     |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|FieldOfView              |Yes      |integer |Number of pixels per subaperture. Not used when a pyramid wavefront sensor|
|                         |         |        |has been selected (4 pixels are used in this case).                       |
|                         |         |        |                                                                          |
|                         |         |        |*Warning*: gives a confusing error message if missing                     |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|NumberPhotons            |No       |list of |*default : [Inf]*, Flux return in [nph/frame/subaperture]                 |
|                         |         |integer |                                                                          |
|                         |         |        |It can be computed as:                                                    |
|                         |         |        |                                                                          |
|                         |         |        |``(0-magn-flux [ph/s/m2]) * (size of sub-aperture [m])^2                  |
|                         |         |        |* (1/SensorFrameRate_HO) * (total throughput)                             |
|                         |         |        |* (10^(-0.4*magn_source_HO))``                                            |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|SpotFWHM                 |No       |list of |*defaut: [[0.0, 0.0]]*, Represents the instrumental broadening of         |
|                         |         |list of |Shack–Hartmann spot size (FWHM) along x and y, in [milliarcseconds]       |
|                         |         |float   |without turbulence. If set to [[0.0, 0.0]], only atmospheric broadening is|
|                         |         |        |considered. Not used with a Pyramid WFS.                                  |
+-------------------------+---------+--------+--------------------------------------------------------------------------+    
|SpectralBandwidth        |No       |float   |*default: 0.0*, Spectral fullwidth around each central wavelength         |
|                         |         |        |(in [meters]). If 0, monochromatic simulation.                            |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|Transmittance            |No       |list of |*default: [1.0]*, Transmission factors at the WFS plane. Expected in the  |
|                         |         |float   |range [0,1].                                                              |
+-------------------------+---------+--------+--------------------------------------------------------------------------+    
|Dispersion               |No       |list of |*default: [[0.0],[0.0]]*, Chromatic shift of the image on the detector,   |
|                         |         |list of |in pixels. The first sub-list corresponds to x-offsets, the second to     |
|                         |         |float?  |y-offsets. Must have the same number of elements as ``Transmittance``.    |
|                         |         |        |Used only in PSF computation to account for wavelength-dependent shifts   |
|                         |         |        |(e.g. due to residual atmospheric dispersion).                            |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|Gain                     |No       |float   |*default : 1.0*, Detector pixel gain.                                     |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|ExcessNoiseFactor        |No       |float   |*default : 1.0*, excess noise factor.                                     |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|NoiseVariance            |No       |list of |*Default : [None]*, Noise Variance in rad2. If set, this value            |
|                         |         |float   |overrides the analytical noise variance calculation.                      |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|SigmaRON                 |No       |float   |*Default : 0.0*, read-out noise std in [e-], used only if the             |
|                         |         |        |`NoiseVariance` is not set.                                               |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|addMcaoWFsensConeError   |No       |bool    |*Default : False*, additional error to consider the reduced sensing volume|
|                         |         |        |due to the cone effect. Multi-conjugate systems only.                     |
+-------------------------+---------+--------+--------------------------------------------------------------------------+

In the two following section we list the parameters that are specific to each wavefront sensor. If you define a parameter 
for one WFS while another WFS is defined The parameter will be ignored. For example, if you define the parameter SigmaRON,
while WfsType is 'Pyramid', SigmaRON is ignored.

Shack-Hartmann requirement
^^^^^^^^^^^^^^^^^^^^^^^^^^

+-------------------------+---------+--------+--------------------------------------------------------------------------+
| Parameter               | Required| Type   | Description                                                              |
+=========================+=========+========+==========================================================================+
|Algorithm                |No       |string  |*defaut:'wcog'*, other options: 'cog' (simple center-of-gravity), 'tcog'  |
|                         |         |        |(center-of-gravity with threshold), 'qc' (quad-cell)                      |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|WindowRadiusWCoG         |No       |integer |*default: 5*, FWHM in pixel of the gaussian weighting function            |
+-------------------------+---------+--------+--------------------------------------------------------------------------+

Pyramid requirement
^^^^^^^^^^^^^^^^^^^

+-------------------------+---------+--------+--------------------------------------------------------------------------+
| Parameter               | Required| Type   | Description                                                              |
+=========================+=========+========+==========================================================================+
|Modulation               |Yes      |float   |*default : None*, If the chosen wavefront sensor is the ``'Pyramid'``,    |
|                         |         |        |Spot modulation radius in lambda/D units. This is ignored if the WFS is   |
|                         |         |        |`'Shack-Hartmann'`                                                        |
|                         |         |        |                                                                          |
|                         |         |        |*Warning* : gives a confusing message if missing when required            |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|Binning                  |No       |integer |*default: 1*, Binning factor of the detector, only used in the pyramid    |
|                         |         |        |case, optional for pyramid                                                |
+-------------------------+---------+--------+--------------------------------------------------------------------------+

Can be set but not used
^^^^^^^^^^^^^^^^^^^^^^^

+-------------------------+---------+--------+--------------------------------------------------------------------------+
| Parameter               | Required| Type   | Description                                                              |
+=========================+=========+========+==========================================================================+
|Dark                     |No       |float   |*default: 0.0*, dark current in [e-/s/pix]                                |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|SkyBackground            |No       |float   |*default: 0.0*, Sky background [e-/s/pix]                                 |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|ThresholdWCoG            |No       |float?  |*default: 0.0*, Threshold Number of pixels for windowing the low order WFS| 
|                         |         |        |pixels                                                                    |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|NewValueThrPix           |No       |float   |*default: 0.0*, New value for pixels lower than `ThresholdWCoG`. Is there |
|                         |         |        |a reason to want to force these values to something else?                 |
+-------------------------+---------+--------+--------------------------------------------------------------------------+

[sensor_LO]
-----------

.. note::

   This section is optional, if this section is not present only the HO part will be used (for ex. to simulate a SCAO NGS).

``SkipEncircledEnergyPixelFloor`` is an optional boolean, defaulting to ``False``.
It selects the radius used to measure NGS encircled energy for MASTSEL's
Gaussian spot amplitude: ``False`` preserves the existing
``max(FWHM, PixelScale)`` radius; ``True`` uses one FWHM radius. Detector pixel
integration, centroid windows and focus-sensor treatment are unchanged.
The option applies to both ordinary and full-flux pupil-fit NGS optics.

+-------------------------+---------+--------+--------------------------------------------------------------------------+
| Parameter               | Required| Type   | Description                                                              |
+=========================+=========+========+==========================================================================+
|PixelScale               |Yes      |float   |LO WFS pixel scale in [mas],                                              |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|FieldOfView              |Yes      |integer |Number of pixels per subaperture.  Not used when a Pyramid wavefront      |
|                         |         |        |sensor has been selected (4 pixels are used in this case).                |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|NumberPhotons            |Yes      |list of |Detected flux in [nph/frame/subaperture], Must be the same length as      |
|                         |         |integer |NumberLenslet                                                             |
|                         |         |        |                                                                          |
|                         |         |        |It can be computed as:                                                    |
|                         |         |        |                                                                          |
|                         |         |        |``(0-magn-flux [ph/s/m2]) * (size of subaperture [m])**2                  |
|                         |         |        |* (1/SensorFrameRate_LO) * (total throughput)                             |
|                         |         |        |* (10**(-0.4*magn_source_LO))``                                           |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|NumberLenslets           |Yes      |list of |*Default : [1]*, number of WFS lenslets, Must be the same length as       |
|                         |         |integer |NumberPhotons                                                             |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|SigmaRON                 |Yes      |float   |*default: 0.0*, read out noise in [e-]                                    |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|Dark                     |Yes      |float   |*default: 0.0*, dark current[e-/s/pix]                                    |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|SkyBackground            |Yes      |float   |*default: 0.0*, Sky background [e-/s/pix]                                 |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|ExcessNoiseFactor        |Yes      |float   |*default: 1.0*, excess noise factor                                       |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|WindowRadiusWCoG         |Yes      |integer |*default: 1*, Radius in pixel of the FWHM of the weights map of the       |
|                         |         |or      |weighted CoG the low order WFS pixels                                     |
|                         |         |string  |                                                                          |
|                         |         |        |*Warning* : if set to 'optimize', gain is automatically optimized by      |
|                         |         |        |TIPTOP (closest int to half of PSF FWHM), otherwise the float value set is|
|                         |         |        |used.                                                                     |
+-------------------------+---------+--------+--------------------------------------------------------------------------+    
|ThresholdWCoG            |Yes      |float   |*default: 0.0*, Threshold Number of pixels for windowing the low order WFS|
|                         |         |        |pixels                                                                    |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|NewValueThrPix           |Yes      |float   |*default: 0.0*, New value for pixels lower than threshold.                |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|filtZernikeCov           |No       |bool    |*Default : False*, Filter for the zernike covariance. The zernike cov. is |
|                         |         |        |used to quantify for the TT tomographic (anisoplanatic) error. This filter|
|                         |         |        |accounts for the HO correction of an MCAO system. Multi-conjugate systems |
|                         |         |        |only.                                                                     |
|                         |         |        |                                                                          |
|                         |         |        |*Warning*: Do not use in systems with a single DM.                        |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|addAliasError            |No       |bool    |*Default : False*, Add aliasing error as a function of the spot size.     |
+-------------------------+---------+--------+--------------------------------------------------------------------------+

Can be set but not used
^^^^^^^^^^^^^^^^^^^^^^^

+-------------------------+---------+--------+--------------------------------------------------------------------------+
| Parameter               | Required| Type   | Description                                                              |
+=========================+=========+========+==========================================================================+
|Binning                  |No       |integer |*default: 1*, binning factor of the detector                              |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|SpotFWHM                 |No       |list of |*default: [[0.0, 0.0]]*, Low Order spot scale in [mas]                    |
|                         |         |list of |                                                                          |
|                         |         |integer |                                                                          |
+-------------------------+---------+--------+--------------------------------------------------------------------------+   
|Gain                     |No       |float   |*default: 1.0*, Camera gain                                               |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|Algorithm                |No       |string  |*default: 'wcog'*, CoG computation algorithm                              |
+-------------------------+---------+--------+--------------------------------------------------------------------------+

[sensor_Focus]
--------------

.. note::

   This section is completely optional.
   The ``[sensor_Focus]`` section is required to have the global focus part simulated considering specific focus sensors and not the LO sensors.
   This happens when the key ``glFocusOnNGS`` in the ``[telescope]`` section is True and multiple DMs are present.

+-------------------------+---------+--------+--------------------------------------------------------------------------+
| Parameter               | Required| Type   | Description                                                              |
+=========================+=========+========+==========================================================================+
|PixelScale               |Yes      |float   |Focus WFS pixel scale in [mas],                                           |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|FieldOfView              |Yes      |integer |not used. Number of pixels per subaperture,                               |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|NumberPhotons            |Yes      |list of |Detected flux in [nph/frame/subaperture], Must be the same length as      |
|                         |         |integer |NumberLenslet                                                             |
|                         |         |        |                                                                          |
|                         |         |        |It can be computed as:                                                    |
|                         |         |        |                                                                          |
|                         |         |        |``(0-magn-flux [ph/s/m2]) * (size of subaperture [m])**2                  |
|                         |         |        |* (1/SensorFrameRate_Focus) * (total throughput)                          |
|                         |         |        |* (10**(-0.4*magn_source_Focus))``                                        |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|NumberLenslets           |Yes      |list of |*Default : [1]*, number of WFS lenslets, Must be the same length as       |
|                         |         |integer |NumberPhotons                                                             |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|SigmaRON                 |Yes      |float   |*default: 0.0*, read out noise in [e-]                                    |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|Dark                     |Yes      |float   |*default: 0.0*, dark current[e-/s/pix]                                    |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|SkyBackground            |Yes      |float   |*default: 0.0*, Sky background [e-/s/pix]                                 |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|ExcessNoiseFactor        |Yes      |float   |Excess noise factor                                                       |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|WindowRadiusWCoG         |Yes      |integer |*default: 1*, Radius in pixel of the HWHM of the weights map of the       |
|                         |         |or      |weighted CoG the global focus WFS pixels                                  |
|                         |         |string  |                                                                          |
|                         |         |        |*Warning* : if set to 'optimize', gain is automatically optimized by      |
|                         |         |        |TIPTOP (closest int to half of PSF FWHM), otherwise the float value set is|
|                         |         |        |used.                                                                     |
+-------------------------+---------+--------+--------------------------------------------------------------------------+    
|ThresholdWCoG            |Yes      |float   |*default: 0.0*, Threshold Number of pixels for windowing the low order WFS|
|                         |         |        |pixels                                                                    |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|NewValueThrPix           |Yes      |float   |*default: 0.0*, New value for pixels lower than threshold.                |
+-------------------------+---------+--------+--------------------------------------------------------------------------+


[DM]
----

+-------------------------+---------+--------+--------------------------------------------------------------------------+
| Parameter               | Required| Type   | Description                                                              |
+=========================+=========+========+==========================================================================+
|DmPitchs                 |Yes      |list of |DM actuators pitch in meters, on the meta pupil at the conjugation        |
|                         |         |float   |altitude, used for fitting error computation.                             |
|                         |         |        |                                                                          |
|                         |         |        |*Warning*: if it is smaller than [sensor_HO] SizeLenslets                 |
|                         |         |        |(= [Telescope] TelescopeDiameter/[sensor_HO] NumberLenslet ) aliasing     |
|                         |         |        |error will be significant.                                                |
|                         |         |        |                                                                          |
|                         |         |        |Must be the same length as NumberActuators                                |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|NumberActuators          |No       |list of |*default: computed from ``TelescopeDiameter``, ``TechnicalFoV``,          |
|                         |         |integer |``DMHeights`` and ``DMPitchs``.* Number of actuator on the pupil diameter.|
|                         |         |        |Must be the same length as ``DmPitchs``.                                  |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|InfModel                 |No       |string  |*default: 'gaussian'*, DM influence function model. Supported values:     |
|                         |         |        |``'gaussian'`` or ``'xinetics'``.                                         |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|InfCoupling              |No       |list of |*default: [0.2]*, Mechanical coupling coefficient (0–1) between DM        |
|                         |         |float   |actuators. Controls the width of the influence function. Must have the    |
|                         |         |        |same length as ``NumberActuators`` (one value per DM).                    |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|DmHeights                |No/Yes if|list of |*default: [0.0]*, DM altitude in meters, Must be the same length as       |
|                         |LO or    |float   |NumberActuators and DmPitchs                                              |
|                         |multi DMs|        |                                                                          |
+-------------------------+---------+--------+--------------------------------------------------------------------------+   
|OptimizationZenith       |No       |float   |*default: [0.0]*, Zenith position in arcsec (distance from axis) of the   |
|                         |         |        |direction in which the AO correction is optimized. Must be the same length|
|                         |         |        |as OptimisationAzimuth  and OptimizationWeight. These are for wide field  |
|                         |         |        |AO system, should be a requirement for MCAO and GLAO                      |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|OptimizationAzimuth      |No       |list of |*default: [0.0]*, Azimuth in degrees (angle from the ref. direction: polar|
|                         |         |float   |axis is x-axis) of the direction in which the AO correction is optimized. |
|                         |         |        |Must be the same length as OptimizationZenith and OptimizationWeight.     |
|                         |         |        |These are for wide field AO system, should be a requirement for MCAO and  |
|                         |         |        |GLAO                                                                      |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|OptimizationWeight       |No       |float   |*default: [1.0]*, Weights of the optimisation directions. Must be the same|
|                         |         |        |length as OptimizationZenith and OptimizationAzimuth. These are for wide  |
|                         |         |        |field AO system, should be a requirement for MCAO and GLAO.               |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|OptimizationConditioning |No       |float   |*default: 1.0e2*, Matrix Conditioning threshold in the truncated SVD      |
|                         |         |        |inversion.                                                                |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|NumberReconstructedLayers|No       |integer |*default: 10*, Number of reconstructed atmospheric layers for tomographic |
|                         |         |        |AO systems (multi-guide-star).                                            |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|AoArea                   |No       |string  |*default: 'circle'*, Shape of the AO-corrected area. Any other options are| 
|                         |         |        |not defined and will give a squarre correction area.                      |
+-------------------------+---------+--------+--------------------------------------------------------------------------+

For ``finite_aperture``, ``InfModel='gaussian'`` and ``InfCoupling`` define the
physical influence functions. Optional ``PathInfGeometry`` supplies a FITS
actuator layout and calibration coordinates. Without it, the mirror uses a
centred nominal count/pitch grid, with actuator centres selected on the sampled
illuminated telescope pupil. The telescope owns pupil weights and the INI owns
coupling. See :doc:`twoStageFourier` for the geometry override and science mirror.

[RTC]
-----

The reconstruction model determines the high-order temporal response.
``standard`` retains the existing integrator calculations, including the omitted
gain response in the main tomographic residual. ``alias_aware`` and
``finite_aperture`` apply a leaky pseudo-open-loop update followed by an
integer-frame command queue and instantaneous WFS measurements. Both use the
configured ``LoopGain_HO``, ``SensorFrameRate_HO`` and ``LoopDelaySteps_HO``;
these physical parameters are never changed by model selection.

POLC requires gain in [0, 1], a positive rate and a nonnegative integer delay.
Each atmospheric layer and alias uses its own frozen-flow temporal frequency
before calculating power. For ``alias_aware``, independent white measurement
noise uses power gain ``g/(2-g)``; its Fourier response assumes exact subtraction
of the applied common DM contribution in the POL measurement. The finite model
instead retains the physical common-mirror WFS response and the complete
feedback caused by per-LGS mean-slope projection. It currently requires zero
injected HO noise. The geometry restrictions of the selected model still apply.
``TemporalModel_HO`` and ``ExposureTime`` cannot be configured independently.

.. note::

   This section is optional, if this section is not present the defaul values are used.

+-------------------------+---------+--------+--------------------------------------------------------------------------+
| Parameter               | Required| Type   | Description                                                              |
+=========================+=========+========+==========================================================================+
|LoopGain_HO              |No       |float   |*Default : 0.5*, High Order Loop gain.                                    |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|SensorFrameRate_HO       |No       |float   |*Default : 500.0*, High Order loop frequency in [Hz]                      |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|LoopDelaySteps_HO        |No       |integer |*Default : 2*, High Order loop delay in [frame]                           |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|LoopGain_LO              |No/Yes if|float or|*default: None*, Low Order loop gain, *Warning*: if set to 'optimize',    |
|                         |LO       |string  |gain is automatically optimized by TIPTOP, otherwise the float value set  |
|                         |         |        |is used.                                                                  |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|SensorFrameRate_LO       |No/Yes if|float   |*default: None*, Loop frequency in [Hz]. If ``[sensor_LO]`` section is    |
|                         |LO       |        |present it must be set.                                                   |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|LoopDelaySteps_LO        |No/Yes if|integer |*default: None*, Low Order loop delays in [frames]. If ``[sensor_LO]``    |
|                         |LO       |        |section is present it must be set.                                        |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|LoopGain_Focus           |No/Yes if|float or|*default: None*, Global focus loop gain, *Warning*: if set to 'optimize', |
|                         |Focus    |string  |gain is automatically optimized by TIPTOP, otherwise the float value set  |
|                         |         |        |is used.                                                                  |
+-------------------------+---------+--------+--------------------------------------------------------------------------+   
|SensorFrameRate_Focus    |No/Yes if|float   |*default: None*, Global focus loop frequency in [Hz]. If                  |
|                         |Focus    |        |``[sensor_Focus]`` section is present it must be set.                     |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|LoopDelaySteps_Focus     |No/Yes if|integer |*default: None*, Global focus loop delays in [frames]. If                 |
|                         |Focus    |        |``[sensor_Focus]`` section is present it must be set.                     |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|MMSE_Rec_LO              |No       |bool    |*default: True*, Enables the *MMSE* reconstructor for the LO loop.        |
|                         |         |        |If ``False``, a *Tikhonov* reconstruction is used with λ = 0.05 (empirical|
|                         |         |        |value chosen to remove unseen modes when fewer than 3 stars or poor       |
|                         |         |        |poor asterism geometry are present). Ignored if only one guide star.      |
+-------------------------+---------+--------+--------------------------------------------------------------------------+

[DM_science] and [RTC_science]
------------------------------

These optional sections must appear together. ``DM_science`` uses
``NumberActuators``, ``DmPitchs``, ``DmHeights`` (currently ground only), and,
for finite aperture, ``InfModel``, ``InfCoupling`` and optional ``PathInfGeometry``.
``RTC_science`` requires explicit ``LoopGain_HO`` and ``LoopDelaySteps_HO``.
Its gain multiplies the residual estimate directly; its integer delay uses the
common ``RTC.SensorFrameRate_HO`` clock. It is not another integrator.

Omit both sections for single-stage LTAO. ``LoopGain_HO=0`` is the science-off
control and does not load science geometry. Active science correction supports
``alias_aware`` or ``finite_aperture`` with zero injected HO noise, an active
common POLC and science pitch at least the common pitch. HO+LO additionally
requires ``TipTiltRemoval='pupil_fit'``, ``fullFluxNormalization=True``,
one-lenslet NGS tilt sensing and no focus loop.
See :doc:`twoStageFourier` for the two-stage calculation and numerical backend.

[COMPUTATION]
-------------

.. note::

   This section is optional, if this section is not present the defaul values are used.

+-------------------------+---------+--------+--------------------------------------------------------------------------+
| Parameter               | Required| Type   | Description                                                              |
+=========================+=========+========+==========================================================================+
|platform                 |No       |string  |*default: 'GPU'* Set to it to 'CPU' to forcy the library to use numpy     |
|                         |         |        |instead of cupy.                                                          |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|integralDiscretization1  |No       |float   |*default: 1000.0*, Discretization used in the integrals                   |
|                         |         |        |(astro-tiptop/SEEING library).                                            |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|integralDiscretization2  |No       |float   |*default: 4000*, Discretization used in the integrals                     |
|                         |         |        |(astro-tiptop/SEEING library).                                            |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
|precision                |No       |string  |*default: 'double'*, precision used in computations, 'single' or 'double'.|
|                         |         |        |Using single precision improves speed and reduces memory usage, albeit at |
|                         |         |        |the cost of precision.                                                    |
+-------------------------+---------+--------+--------------------------------------------------------------------------+
