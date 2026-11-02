Residual-driven science correction
==================================

Paired ``DM_science`` and ``RTC_science`` sections select an additional ground
science mirror per direction with ``ReconstructionModel='alias_aware'`` or
``'finite_aperture'``. The common ``DM`` and ``RTC`` remain separate. The science
RTC requires explicit ``LoopGain_HO`` and ``LoopDelaySteps_HO``. These set a
direct residual-command multiplier and integer delay on the common HO clock;
it is not a second integrator. Omit both sections for single-stage LTAO.

P3 owns mirror fitting, command coupling and residual covariance. TIPTOP retains
its full-flux pupil-fit/static-map optics. Active science correction supports
HO-only predictions and MASTSEL HO+LO with ``TipTiltRemoval='pupil_fit'``,
``fullFluxNormalization=True``, one-lenslet NGS tilt sensing and no focus loop.
NGS images use only the common mirror, excluding the directional science
mirror and science static map. MASTSEL's existing pointing covariance is
then convolved with the two-stage science PSF. This retains the assumption
that residual pointing and the tilt-removed HO structure are independent.

Finite aperture uses physical Gaussian surfaces via existing ``InfModel`` and
``InfCoupling``. Optional ``PathInfGeometry`` FITS files override the
regular layouts derived from mirror settings and the illuminated telescope pupil.
Finite aperture owns physical phase reconstruction and LGS mean-slope removal with complete feedback. The only added
estimator parameter is ``ReconstructionRegularization`` (default zero), a fraction
of mean prior slope variance, separate from actual noise and the numerical floor.

``baseSimulation(..., finiteApertureBackend='numpy')`` defaults to CPU. Explicit
``'cuda'`` (CuPy float64) and ``'mps'`` (Torch Metal float32) accelerate only
active two-stage finite aperture. Reconstruction solves stay CPU float64. Set
``P3_DISABLE_GPU=1`` before importing TIPTOP/P3 to keep its global arrays on CPU;
this leaves explicit finite-aperture device products enabled. CuPy uses the
current CUDA device, which must remain current while using the model. Missing
dependencies or devices raise errors without CPU fallback. This execution choice
is not an INI physics option.

See P3's ``docs/two-stage-fourier.md`` for the science RTC and supported scope,
and ``docs/mirror-surfaces.md`` for the FITS geometry contract. Retired experimental
keys are rejected; historical experiments use their retained sources.

Finite-aperture error budget
---------------------------

``baseSimulation(..., getHoErrorBreakDown=True)`` enables TIPTOP-style
finite-aperture LTAO/MOAO reporting. After simulation,
``simulation.errorBreakDown()`` prints the evaluated science direction nearest
the axis and returns a field mapping. Each immutable result supports
``as_dict()``. ``simulation.hoErrorBreakDown`` holds the same HO records.
LO remains in TIPTOP's existing separate ``LO_res [nm]`` output and is excluded
from the breakdown rows and totals.

MOAO prints two columns: ``LTAO: first stage`` and ``MOAO: both stages``.
``result.first_stage`` exposes the former at the same science direction, with
the actual common correction retained and only the science-mirror correction
removed. No new common reconstructor or second simulation is needed. Both
columns receive the same science-plane NCPA, isolating the HO change. Standalone LTAO results have ``first_stage=None``.

Rows are NCPA, fitting, differential refraction, chromatism, aliasing, noise,
spatio-temporal error and extra error. Fitting
uses TIPTOP's existing correction-band cutoff. In-band limitations of the
coarser science mirror remain in spatio-temporal error. Aliasing is the joint
correction driven by above-band inputs, following the GIRMOS input-band
convention rather than Fourier replica bookkeeping. Both actual mirrors,
controllers and their correlations are retained.

The signed fitting–aliasing covariance is printed separately in nm². It is
included in the residual total but excluded from the displayed quadrature
total. ``dynamic_ho_nm2`` closes against ``HO_res**2``. ``total_nm2`` additionally
includes the applied NCPA variance; it excludes LO.
NCPA is measured after the same resampling, compensation and tilt removal used
by the optics; it is not injected into atmospheric covariance.

The flag changes no PSF and adds no INI setting. Nonzero physical HO noise
remains unsupported by the finite model. Conventional Fourier reporting stays
unchanged. ``finiteApertureTimings`` records ``error_budget_seconds`` and
``error_budget_optical_closure_nm2``. See P3's ``docs/finite-error-budget.md`` for
definitions, units, caching and limitations.
