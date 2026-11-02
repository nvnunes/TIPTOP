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
