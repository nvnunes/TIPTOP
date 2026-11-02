import numpy as np
from pathlib import Path
from matplotlib import rc
from scipy.interpolate import interp1d

# P3 Imports
from p3.aoSystem.fourierModel import fourierModel
from p3.aoSystem.reconstruction import reconstruction_model
from p3.aoSystem.FourierUtils import getStrehl, getFWHM, getEncircledEnergy, getEnsquaredEnergy, pistonFilter

# Mastsel Imports
from mastsel import MavisLO, psdSetToPsfSet, longExposurePsf, Field, convolve, residualToSpectrum, maskSA
from mastsel.mavisPsf import centeredPixelCoords, padOrCropCentered, mastselPsfPrecision
from mastsel.mavisUtilities import sigma_from_FWHM, tiledDisplay, plotEllipses, polarToCartesian, congrid

# Tiptop Imports
from .tiptopUtils import arrayP3toMastsel, arrayMastseltoP3, cpuArray
from .abstractSimulation import AbstractSimulation

rc("text", usetex=False)
rad2mas = 3600 * 180 * 1000 / np.pi

# TIPTOP repo root — used as path_root when calling fourierModel so that
# auxiliary files referenced as 'tiptop/data/...' in config files are found.
_TIPTOP_ROOT = str(Path(__file__).resolve().parent.parent)


class baseSimulation(AbstractSimulation):
    """
    P3-based Backend Implementation for TIPTOP Simulations.
    Inherits the strict lifecycle template from AbstractSimulation.

    fullFluxNormalization=True opts science/reference PSFs into complete-plane
    unit-flux normalization. Crops retain their enclosed fraction through LO
    and jitter convolution and optional profile computation. The default False
    preserves existing output behavior. This option requires MASTSEL's
    full_flux propagation and normalize_input convolution APIs.

    sensor_HO.TipTiltRemoval selects 'auto' (default), 'none', 'spectral', or
    'pupil_fit'. Auto preserves the LO-dependent default: spectral with LO on,
    none with LO off. Explicit methods are independent of LO availability.
    Pupil-fit projects Gaussian phase covariance and static phase onto the
    complement of the two actual-pupil tilt ramps in MASTSEL's native optics.
    ``finiteApertureBackend`` selects numerical products ("numpy", "cuda", "mps").
    GPU products require active finite-aperture science correction. CUDA uses
    CuPy float64 on the current device; Metal uses float32. Reconstruction remains
    CPU float64; set P3_DISABLE_GPU=1 before importing TIPTOP/P3.
    """

    def __init__(self, path, parametersFile, outputDir, outputFile, doConvolve=True,
                 doPlot=False, addSrAndFwhm=True, verbose=False, getHoErrorBreakDown=False,
                 savePSDs=False, ensquaredEnergy=False, eeRadiusInMas=50,
                 exactMultiWavelengthPSD=False, fullFluxNormalization=False, finiteApertureBackend="numpy"):

        # Superclass handles all configuration loading, validation, and standard properties
        super().__init__(path, parametersFile, outputDir, outputFile, doConvolve,
                         doPlot, addSrAndFwhm, verbose, getHoErrorBreakDown,
                         savePSDs, ensquaredEnergy, eeRadiusInMas)

        # P3-specific state variables
        self.fao = None
        self.mLO = None

        # When True and more than one [sources_science] Wavelength is
        # requested, P3 computes one exact PSD grid per wavelength
        # (psdPerWavelength=True) instead of one shared, approximate grid --
        # see fourierModel/frequencyDomain in P3 for why this matters
        # (equivalence with a standalone single-wavelength run) and its cost
        # (roughly proportional to nWvl, since the reconstructor/controller
        # are recomputed per wavelength too). Defaults to False: zero change
        # in behaviour or cost for existing configurations.
        self.exactMultiWavelengthPSD = exactMultiWavelengthPSD
        # True marks science outputs as fractions of complete propagation power.
        # Convolution and optional profiles retain this scale; crops may sum < 1.
        self.fullFluxNormalization = fullFluxNormalization
        self.skipEncircledEnergyPixelFloor = self.my_data_map.get('sensor_LO', {}).get('SkipEncircledEnergyPixelFloor', False)
        if not isinstance(self.skipEncircledEnergyPixelFloor, bool):
            raise ValueError('sensor_LO.SkipEncircledEnergyPixelFloor must be a boolean')
        if 'FilterTipTilt' in self.my_data_map['sensor_HO']:
            raise ValueError('FilterTipTilt is replaced by sensor_HO.TipTiltRemoval')
        method = self.my_data_map['sensor_HO'].get('TipTiltRemoval', 'auto')
        if not isinstance(method, str) or method not in ('auto', 'none', 'spectral', 'pupil_fit'):
            raise ValueError("sensor_HO.TipTiltRemoval must be 'auto', 'none', 'spectral' or 'pupil_fit'")
        self.tipTiltRemoval = ('spectral' if self.LOisOn else 'none') if method == 'auto' else method
        self.staticOnRmsCompensation = self.my_data_map['telescope'].get('StaticOnRmsCompensation', False)
        if not isinstance(self.staticOnRmsCompensation, bool):
            raise ValueError('telescope.StaticOnRmsCompensation must be a boolean')
        if self.staticOnRmsCompensation and self.tipTiltRemoval != 'pupil_fit':
            raise ValueError("StaticOnRmsCompensation=True currently requires TipTiltRemoval='pupil_fit'")
        self.reconstructionModel = reconstruction_model(
            self.my_data_map['sensor_HO'], self.my_data_map.get('RTC', {}))
        from p3.aoSystem.scienceCorrection import configuration, covariance_backend
        science, _ = configuration(self.my_data_map, self.reconstructionModel)
        self.twoStageEnabled = science is not None and science.active
        self.finiteApertureBackend = covariance_backend(finiteApertureBackend,
            self.reconstructionModel, science)
        if self.twoStageEnabled and self.LOisOn:
            if not self._independent_ngs_optics():
                raise ValueError('Science-correction LO integration requires pupil_fit and fullFluxNormalization=True')
            if self.addFocusError or any(n != 1 for n in self.my_data_map['sensor_LO']['NumberLenslets']):
                raise ValueError('Science-correction LO integration supports one-lenslet tilt sensing without a focus loop')
        self.finiteApertureReconstruction = self.reconstructionModel == 'finite_aperture'
        self.hoErrorBreakDown = {}
        if self.finiteApertureReconstruction:
            if self.tipTiltRemoval != 'pupil_fit' or not self.fullFluxNormalization:
                raise ValueError("ReconstructionModel=finite_aperture requires pupil_fit and fullFluxNormalization=True")
            if self.LOisOn and (self.addFocusError or any(n != 1 for n in self.my_data_map['sensor_LO']['NumberLenslets'])):
                raise ValueError('Finite-aperture LO integration currently supports one-lenslet tilt sensing without a focus loop')
            if self.savePSDs:
                raise ValueError('ReconstructionModel=finite_aperture has no equivalent diagonal PSD export')

    # ----------------------------------------------------------------------------
    # --- IMPLEMENTATION OF ABSTRACT METHODS ---
    # ----------------------------------------------------------------------------

    def _configure_LO_parameters(self, astIndex):
        """
        Configures the geometry and low order arrays based on the loaded configuration.
        """
        if not self.LOisOn:
            return

        self.cartSciencePointingCoords = np.dstack(
            (self.pointings[0, :], self.pointings[1, :])
        ).reshape(-1, 2)

        # Extract LO wavelength
        LO_wvl_temp = self.my_data_map['sources_LO']['Wavelength']
        self.LO_wvl = LO_wvl_temp[0] if isinstance(LO_wvl_temp, list) else LO_wvl_temp

        self.LO_zen_field = self.my_data_map['sources_LO']['Zenith']
        self.LO_az_field = self.my_data_map['sources_LO']['Azimuth']
        self.LO_fluxes_field = self.my_data_map['sensor_LO']['NumberPhotons']

        # Format Pixel Scale
        lo_ps = self.my_data_map['sensor_LO']['PixelScale']
        self.LO_psInMas = lo_ps if isinstance(lo_ps, list) else [lo_ps] * len(self.LO_zen_field)

        # Format Frame Rate
        lo_fr = self.my_data_map['RTC']['SensorFrameRate_LO']
        self.LO_freqs_field = lo_fr if isinstance(lo_fr, list) else [lo_fr] * len(self.LO_zen_field)

        self.addLoAlias = self.my_data_map['sensor_LO'].get('addAliasError', False)

        # Focus Sensor Configuration
        if 'sensor_Focus' in self.my_data_map:
            self.Focus_fluxes4s_field = self.my_data_map['sensor_Focus']['NumberPhotons']
            f_ps = self.my_data_map['sensor_Focus']['PixelScale']
            self.Focus_psInMas = f_ps if isinstance(f_ps, list) else [f_ps] * len(self.LO_zen_field)

            f_wvl = self.my_data_map.get('sources_Focus', {}).get('Wavelength', self.my_data_map['sources_LO']['Wavelength'])
            self.Focus_wvl = f_wvl[0] if isinstance(f_wvl, list) else f_wvl
        else:
            self.Focus_fluxes4s_field = self.LO_fluxes_field
            self.Focus_psInMas = self.LO_psInMas
            self.Focus_wvl = self.LO_wvl

        f_reqs = self.my_data_map['RTC'].get('SensorFrameRate_Focus', self.LO_freqs_field)
        self.Focus_freqs_field = f_reqs if isinstance(f_reqs, list) else [f_reqs] * len(self.LO_zen_field)

        self.NGS_fluxes_field = [f * fr for f, fr in zip(self.LO_fluxes_field, self.LO_freqs_field)]
        self.Focus_fluxes_field = [f * fr for f, fr in zip(self.Focus_fluxes4s_field, self.Focus_freqs_field)]

        # Coordinate arrays
        polarNGSCoords = np.column_stack((self.LO_zen_field, self.LO_az_field))
        self.nNaturalGS_field = len(self.LO_zen_field)
        self.cartNGSCoords_field = np.asarray([polarToCartesian(polarNGSCoords[i, :]) for i in range(self.nNaturalGS_field)])
        self.currentAsterismIndices = list(range(len(self.LO_zen_field)))

        self._set_asterism_data()

    def _set_asterism_data(self):
        """Helper to set current asterism properties based on indices."""
        self.LO_zen_asterism = [self.LO_zen_field[i] for i in self.currentAsterismIndices]
        self.LO_az_asterism = [self.LO_az_field[i] for i in self.currentAsterismIndices]
        self.LO_fluxes_asterism = [self.LO_fluxes_field[i] for i in self.currentAsterismIndices]
        self.LO_freqs_asterism = [self.LO_freqs_field[i] for i in self.currentAsterismIndices]
        self.NGS_fluxes_asterism = [self.NGS_fluxes_field[i] for i in self.currentAsterismIndices]
        self.Focus_fluxes_asterism = [self.Focus_fluxes_field[i] for i in self.currentAsterismIndices]
        self.cartNGSCoords_asterism = [self.cartNGSCoords_field[i] for i in self.currentAsterismIndices]

    def _prepare_static_PSF_state(self, astIndex):
        """
        Initializes the Fourier HO model and computes the static OL/DL representations.
        """
        if not (astIndex is None or self.firstSimCall):
            return

        if self.verbose:
            print('******** HO PSD science and NGSs directions')

        # Pass config_dict so P3 uses TIPTOP's already-modified my_data_map directly,
        # without re-reading the file. path_root lets P3 resolve 'tiptop/...' aux files.
        self.fao = fourierModel(self.fullPathFilename, calcPSF=False, verbose=self.verbose,
                                display=False, getPSDatNGSpositions=self.LOisOn and not self._independent_ngs_optics(),
                                computeFocalAnisoCov=False,
                                TiltFilter=self.tipTiltRemoval == 'spectral',
                                computeOpticalPSD=self.tipTiltRemoval == 'pupil_fit',
                                getErrorBreakDown=self.getHoErrorBreakDown, doComputations=False,
                                psdExpansion=True, psdPerWavelength=self.exactMultiWavelengthPSD,
                                reduce_memory=not (self.finiteApertureReconstruction or self.twoStageEnabled),
                                path_root=_TIPTOP_ROOT,
                                config_dict=self.my_data_map, finiteApertureBackend=self.finiteApertureBackend)

        if self.verbose:
            print('Setting MASTSEL PSF precision to:', self.fao.dtype)

        mastselPsfPrecision(dtype=self.fao.dtype)

        self.fao.initComputations()

        if self.finiteApertureReconstruction:
            # The finite residual is a covariance, not an alternative diagonal
            # atmospheric PSD. Its optics are selected by the model itself.
            self.multiGridPSD = False
            self.PSD = None
        else:
            # Fail here with a clear message: a non-finite HO PSD otherwise surfaces
            # much later as an obscure error inside the LO (MASTSEL) computations.
            for psd_i in (self.fao.PSD if isinstance(self.fao.PSD, list) else [self.fao.PSD]):
                if not bool(np.isfinite(psd_i).all()):
                    raise ValueError(
                        "The HO PSD computed by P3 contains non-finite values (NaN/Inf): "
                        "check the [atmosphere], [sources_HO] and [sensor_HO] parameters."
                    )

            # self.fao.PSD is a list (one exact-grid array per science wavelength)
            # only when exactMultiWavelengthPSD=True *and* more than one wavelength
            # was requested -- see frequencyDomain.wvl_grids in P3. Otherwise (the
            # default, and the only case before this feature existed) it stays a
            # single (nOtf,nOtf,nSrc) array, exactly as today.
            self.multiGridPSD = isinstance(self.fao.PSD, list)
            # Component variance reporting retains P3's convention. Finite-pupil
            # optics consume stationary main/alias power before variance filtering.
            optical_psd = self.fao.opticalPSD if self.tipTiltRemoval == 'pupil_fit' else self.fao.PSD
            if self.multiGridPSD:
                self.PSD = [p.transpose() for p in optical_psd]
            else:
                self.PSD = optical_psd.transpose()

        # N/PSDstep/freq_range/grid_diameter/dk/mask always describe P3's
        # *shared* grid (self.fao.freq, unaffected by wvl_grids -- see
        # frequencyDomain.py) and are what LO/Focus/open-loop/diffraction-
        # limited PSFs keep using regardless of multiGridPSD, since those are
        # single-wavelength constructs unrelated to the science wavelength
        # list. Only the HO/science PSF path (_generate_final_PSF) additionally
        # needs the per-wavelength freq_range/dk built below.
        self.N = self.fao.freq.nOtf
        self.nPointings = self.pointings.shape[1]
        self.nPixPSF = int(self.fao.ao.cam.fovInPix)
        self.overSamp = getattr(self.fao.freq, 'kRef_float', int(self.fao.freq.kRef_))
        self.PSDstep = self.fao.freq.PSDstep
        self.overSamp_lo = self.fao.freq.kGrid_
        self.freq_range = self.N * self.PSDstep
        self.grid_diameter = 1 / self.PSDstep
        self.sx = int(2 * np.round(self.tel_radius * self.freq_range))
        self.dk = self.fao.freq.dk_
        self.wvlRef = self.fao.freq.wvlRef

        if self.multiGridPSD:
            # One freq_range/dk per science wavelength, matching self.PSD[i].
            self.freq_range_per_wvl = [float(g.nOtf * g.PSDstep) for g in self.fao.freq.wvl_grids]
            self.dk_per_wvl = [float(g.dk_) for g in self.fao.freq.wvl_grids]
            # The pupil occupies a different fraction of each wavelength's own
            # grid (nOtf_i varies with k_i while the physical pupil diameter
            # does not), so nPixPup (self.sx) must vary per wavelength too --
            # same formula as self.sx above, evaluated at each grid's own
            # freq_range_i instead of the shared one.
            self.sx_per_wvl = [int(2 * np.round(self.tel_radius * fr))
                              for fr in self.freq_range_per_wvl]
            # Index into self.wvl/self.PSD whose grid matches self.wvlRef
            # (the minimum requested wavelength): used as "the" representative
            # grid wherever a single-wavelength PSD slice is still needed
            # (HO_res, NGS/Focus LO PSD) -- these quantities are physically
            # wavelength-independent (WFE in nm), so any grid gives the same
            # value up to the numerical-resolution differences between grids.
            self.wvlRef_grid_idx = int(np.argmin(self.wvl))

        # Setup Mask
        self.mask = Field(self.wvlRef, self.N, self.grid_diameter)
        self.mask.sampling = congrid(arrayP3toMastsel(self.fao.ao.tel.pupil), [self.sx, self.sx])
        self.mask.sampling = padOrCropCentered(self.mask.sampling, self.N, xp=self.mask.xp)

        if abs(float(self.psInMas) - float(cpuArray(self.fao.freq.psInMas[0]))) > 1e-6:
            raise ValueError(f"sensor_science.PixelScale '{self.psInMas}'"
                             f" differs from P3 '{float(cpuArray(self.fao.freq.psInMas[0]))}'")

        self.opdMap = arrayP3toMastsel(self.fao.ao.tel.opdMap_on) if self.fao.ao.tel.opdMap_on is not None else None

        if self.verbose:
            print('PSD step:', self.PSDstep)
            print('PSD freq range:', self.freq_range)
            print('oversampling:', self.overSamp)
            print('sensor_science.PixelScale:', self.psInMas)

        if self.LOisOn:
            if self.verbose:
                print('******** LO PART')
            self._compute_ngs_psf()
            self.mLO = MavisLO(verbose=self.verbose, config_dict=self.my_data_map)

    def _compute_LO_terms(self, astIndex) -> dict:
        """
        Computes Low Order terms and returns them as a dictionary.
        Note: when addFocusError is active, self.PSD is modified in-place to include
        the focus contribution before PSF generation.
        """
        if not self.LOisOn:
            return {}

        lo_data = {}

        if astIndex is None:
            # Full Field Processing
            Ctot = self.mLO.computeTotalResidualMatrix(
                np.array(self.cartSciencePointingCoords), self.cartNGSCoords_field,
                self.NGS_fluxes_field, self.LO_freqs_field, self.NGS_SR_field,
                self.NGS_EE_field, self.NGS_FWHM_mas_field,
                aNGS_FWHM_DL_mas=self.NGS_DL_FWHM_mas, doAll=True)

            if self.addFocusError:
                CtotFocus = self.mLO.computeFocusTotalResidualMatrix(
                    self.cartNGSCoords_field, self.Focus_fluxes_field,
                    self.Focus_freqs_field, self.Focus_SR_field,
                    self.Focus_EE_field, self.Focus_FWHM_mas_field)

                lo_data['GF_res'] = float(np.sqrt(np.maximum(cpuArray(CtotFocus).ravel()[0], 0.0)))

                # Apply Global Focus filtering to PSD. FocusFilter() reads
                # self.fao.freq.k2_, whose shape follows whichever grid
                # self.fao.freq currently points to -- with a per-wavelength
                # PSD list, self.PSD[i] has grid_i's own shape, so the filter
                # must be rebuilt once per wavelength grid too (same swap
                # pattern P3 itself uses internally for the PSD terms).
                if self.multiGridPSD:
                    saved_freq = self.fao.freq
                    for i_wvl, grid_ctx in enumerate(self.fao.freq.wvl_grids):
                        self.fao.freq = grid_ctx
                        FocusFilter = self.fao.FocusFilter()
                        FocusFilter *= 1 / FocusFilter.sum()
                        for PSDho in self.PSD[i_wvl]:
                            PSDho += (lo_data['GF_res']**2) * FocusFilter
                    self.fao.freq = saved_freq
                else:
                    FocusFilter = self.fao.FocusFilter()
                    FocusFilter *= 1 / FocusFilter.sum()
                    for PSDho in self.PSD:
                        PSDho += (lo_data['GF_res']**2) * FocusFilter
                lo_data['GFinPSD'] = True
        else:
            # Asterism Specific Processing
            if self.firstSimCall:
                self.mLO.computeTotalResidualMatrix(
                    np.array(self.cartSciencePointingCoords), self.cartNGSCoords_field,
                    self.NGS_fluxes_field, self.LO_freqs_field, self.NGS_SR_field,
                    self.NGS_EE_field, self.NGS_FWHM_mas_field,
                    aNGS_FWHM_DL_mas=self.NGS_DL_FWHM_mas, doAll=False)
                if self.addFocusError:
                    self.mLO.computeFocusTotalResidualMatrix(
                        self.cartNGSCoords_field, self.Focus_fluxes_field,
                        self.Focus_freqs_field, self.Focus_SR_field,
                        self.Focus_EE_field, self.Focus_FWHM_mas_field)

            # Clean faint guide stars
            if np.min(self.NGS_fluxes_asterism) < 1 and np.max(self.NGS_fluxes_asterism) > 1:
                valid_idx = np.where(np.array(self.NGS_fluxes_asterism) > 1)[0]
                self.NGS_fluxes_asterism = [elem for i, elem in enumerate(self.NGS_fluxes_asterism) if i in valid_idx]
                self.Focus_fluxes_asterism = [elem for i, elem in enumerate(self.Focus_fluxes_asterism) if i in valid_idx]
                self.cartNGSCoords_asterism = [elem for i, elem in enumerate(self.cartNGSCoords_asterism) if i in valid_idx]
                self.currentAsterismIndices = [elem for i, elem in enumerate(self.currentAsterismIndices) if i in valid_idx]

            Ctot = self.mLO.computeTotalResidualMatrixI(
                self.currentAsterismIndices, np.array(self.cartSciencePointingCoords),
                np.array(self.cartNGSCoords_asterism), self.NGS_fluxes_asterism)

            if self.addFocusError:
                CtotFocus = self.mLO.computeFocusTotalResidualMatrixI(
                    self.currentAsterismIndices, np.array(self.cartNGSCoords_asterism),
                    self.Focus_fluxes_asterism)
                lo_data['GF_res'] = float(np.sqrt(np.maximum(cpuArray(CtotFocus).ravel()[0], 0.0)))
                lo_data['GFinPSD'] = False

        # Export computed traces
        lo_data['LO_res'] = np.sqrt(np.trace(Ctot, axis1=1, axis2=2))
        if self.doConvolve:
            lo_data['cov_ellipses'] = self.mLO.ellipsesFromCovMats(Ctot)

        # Save reference for internal convolutions
        self._Ctot_temp = Ctot

        return lo_data

    def _generate_final_PSF(self, astIndex, lo_data):
        """
        Convolves PSDs and standardizes the output in self.cubeResultsArray.
        """
        psf_builder = psdSetToPsfSet
        psf_options = {'full_flux': True} if self.fullFluxNormalization else {}
        if self.tipTiltRemoval == 'pupil_fit':
            psf_options['tip_tilt_removal'] = 'pupil_fit'
            if self.staticOnRmsCompensation:
                from .static_phase import rms_compensated_psfs
                psf_builder = rms_compensated_psfs
        if astIndex is None or self.firstSimCall:
            mask = arrayP3toMastsel(self.fao.ao.tel.pupil)

            if self.verbose:
                print('******** HO PSF')

            if self.finiteApertureReconstruction:
                psfLongExpPointingsArr = self._finite_aperture_psfs()
                PSD_HO = None
            elif (self.reconstructionModel == 'alias_aware' and self.opdMap is not None
                    and bool(np.any(cpuArray(self.opdMap)))):
                psfLongExpPointingsArr = self._original_map_alias_psfs()
                PSD_HO = arrayP3toMastsel(self.PSD[0:self.nPointings])
            elif self.multiGridPSD:
                # One list of per-direction PSDs per science wavelength --
                # see mastsel.mavisPsf.psdSetToPsfSet's per-wavelength
                # calling convention.
                PSD_HO = [arrayP3toMastsel(self.PSD[i][0:self.nPointings])
                          for i in range(self.nWvl)]
                psfLongExpPointingsArr = psf_builder(
                    PSD_HO, mask, self.wvl, nPixPup=self.sx_per_wvl,
                    freq_range=self.freq_range_per_wvl, dk=self.dk_per_wvl,
                    nPixPsf=self.nPixPSF, opdMap=self.opdMap, **psf_options,
                )
                # HO_res (WFE in nm) is physically wavelength-independent;
                # any grid gives the same value up to the numerical-resolution
                # differences between grids, so the wvlRef-matching one is
                # used as "the" representative value (consistent with wvlRef
                # already being used this way elsewhere, e.g. computeMetrics).
                self.HO_res = np.sqrt(np.sum(
                    self.PSD[self.wvlRef_grid_idx][0:self.nPointings], axis=(1, 2)))
            else:
                PSD_HO = arrayP3toMastsel(self.PSD[0:self.nPointings])
                psfLongExpPointingsArr = psf_builder(PSD_HO, mask, self.wvl,
                                                        nPixPup=self.sx, freq_range=self.freq_range,
                                                        dk=self.dk, nPixPsf=self.nPixPSF,
                                                        oversampling=self.overSamp,
                                                        opdMap=self.opdMap, **psf_options)

                # Safely compute HO residuals
                self.HO_res = np.sqrt(np.sum(self.PSD[0:self.nPointings], axis=(1, 2)))

            if self.tipTiltRemoval == 'pupil_fit':
                reference_psfs = (psfLongExpPointingsArr[int(np.argmin(self.wvl))]
                                  if self.nWvl > 1 else psfLongExpPointingsArr)
                self.HO_res = np.sqrt([img.residual_phase_variance for img in reference_psfs]) * self.wvlRef * 1e9 / (2 * np.pi)

            if not self.skipMerit:
                self.pointings_FWHM_mas = []
                for i in range(self.nWvl):
                    psfList = psfLongExpPointingsArr[i] if self.nWvl > 1 else psfLongExpPointingsArr
                    psd_i = PSD_HO[i] if self.multiGridPSD else PSD_HO
                    wvl_c = self.wvl[i] if self.nWvl > 1 else self.wvl[0]
                    samp_i = wvl_c * rad2mas / (self.psInMas * 2 * self.tel_radius)
                    rebin_i = max(1, int(np.ceil(2.0 / samp_i))) if samp_i < 2.0 else 1
                    fwhmList = []
                    for idx, img in enumerate(psfList):
                        fwhmX, fwhmY = getFWHM(arrayMastseltoP3(img.sampling), self.psInMas, method='contour',
                                               rebin=rebin_i, nargout=2)
                        fwhm = np.sqrt(fwhmX * fwhmY)
                        fwhmList.append(fwhm)
                        if self.verbose:
                            s1 = 0. if self.finiteApertureReconstruction else cpuArray(psd_i[idx]).sum()
                            sr = (np.exp(-img.residual_phase_variance) if self.finiteApertureReconstruction
                                  else np.exp(-s1 * (2*np.pi*1e-9/wvl_c)**2))
                            print(f'SR(@{int(wvl_c*1e9)}nm)        : {sr:.5f}')
                            print(f'FWHM(@{int(wvl_c*1e9)}nm) [mas]: {fwhm:.3f}')

                    if self.nWvl > 1:
                        self.pointings_FWHM_mas.append(fwhmList)
                    else:
                        self.pointings_FWHM_mas = fwhmList

            self.psfLongExpPointingsArr = psfLongExpPointingsArr

        # ------------------------------------------------------------------------
        # --- Final Convolutions ---
        self.results = []
        if self.LOisOn:
            if self.doConvolve:
                if self.doConvolveAsterism:
                    self._apply_final_convolution(lo_data)
                else:
                    self.cov_ellipses = lo_data.get('cov_ellipses')
            else:
                self._apply_no_convolution()
        else:
            self._apply_jitter_convolution()

        # FORMAT TO STANDARD ARRAY (Expected by AbstractSimulation)
        cubeResultsArray = []
        for i in range(self.nWvl):
            resList = self.results[i] if self.nWvl > 1 else self.results
            cubeResultsArray.append(np.array([cpuArray(img.sampling) for img in resList]))

        if self.nWvl > 1:
            self.cubeResultsArray = np.array(cubeResultsArray)
        else:
            self.cubeResultsArray = np.array(cubeResultsArray[0])

    def _finite_aperture_psfs(self):
        """Compose P3 finite covariance with TIPTOP's nonstationary optics."""
        from .ho_optics import finite_pupil_fields
        if self.nWvl != 1:
            raise ValueError('ReconstructionModel=finite_aperture currently requires one wavelength')
        if self.opdMap is not None and np.any(cpuArray(self.opdMap)):
            self._validate_static_map_optics()
        period = 1/float(self.fao.freq.kxAO_[1,0]-self.fao.freq.kxAO_[0,0])
        images, self.finiteApertureTimings, self.finiteApertureStaticRmsScale = finite_pupil_fields(
            self.fao, cpuArray(self.fao.ao.tel.pupil), self.opdMap,
            self.N, self.sx, period, range(self.nPointings))
        self.hoErrorBreakDown = (self.fao.errorBreakDown(verbose=False)
                                 if self.getHoErrorBreakDown else {})
        if self.getHoErrorBreakDown and self.verbose:
            self.errorBreakDown()
        return images

    def errorBreakDown(self, verbose=True):
        """Return finite HO field budgets with applied NCPA.

        Requires completed science optics with getHoErrorBreakDown=True.
        Records are also available in hoErrorBreakDown. LO remains in the
        existing separate LO_res output and is excluded from these totals.
        Conventional models retain P3's existing reporting behavior.
        """
        if not self.finiteApertureReconstruction:
            return self.fao.errorBreakDown(verbose=verbose)
        records = dict(self.hoErrorBreakDown)
        if not records:
            raise ValueError('Enable getHoErrorBreakDown and evaluate science optics first')
        if verbose:
            field = min(records, key=lambda i: self.fao.ao.src.zenith[i])
            records[field].print(float(self.wvlRef))
        return records

    def _validate_static_map_optics(self):
        if self.nWvl != 1 or self.tipTiltRemoval != 'pupil_fit' or not self.fullFluxNormalization:
            raise ValueError('Static-map optics currently require one wavelength, pupil_fit and full-flux output')
        if self.jitter_FWHM is not None:
            raise ValueError('Static-map optics currently require no additional jitter')
        if self.nPixPSF != self.N or not np.isclose(self.overSamp,1.):
            raise ValueError('Static-map optics currently require the native detector size and pixel sampling')

    def _original_map_alias_psfs(self):
        """Preserve phase-spectrum physics and replace resampled static optics."""
        self._validate_static_map_optics()
        from .static_map import AliasStaticOptics
        import time
        phase=cpuArray(self.opdMap)*(2*np.pi*1e-9/self.wvlRef)
        period=1/float(self.fao.freq.kxAO_[1,0]-self.fao.freq.kxAO_[0,0])
        optics=AliasStaticOptics(cpuArray(self.fao.ao.tel.pupil),phase,
            float(self.fao.ao.tel.D),period,self.N,self.sx,float(self.fao.freq.kcMax_))
        self.staticMapTimings=[];results=[]
        for psd in self.PSD:
            started=time.perf_counter()
            spectrum=np.fft.ifftshift(cpuArray(psd)/self.dk**2)*(np.pi/self.wvlRef*self.freq_range)**2
            image,checks=optics.psf(spectrum)
            result=Field(self.wvlRef,self.N,self.wvlRef*self.freq_range,'rad')
            result.sampling=image;result.residual_phase_variance=checks['residual_phase_variance']
            results.append(result);self.staticMapTimings.append(dict(seconds=time.perf_counter()-started,**checks))
        self.staticMapHighCacheBuilds=optics.high_cache_builds
        return results

    def _finalize_full_field_results(self, astIndex):
        """
        Creates analytical Open-Loop and Diffraction-Limited PSFs.
        Populates standardized NumPy arrays.
        """
        if astIndex is not None:
            return
        flux_options = {'full_flux': True} if self.fullFluxNormalization else {}

        # OPEN-LOOP PSD
        spectrum = arrayP3toMastsel(self._open_loop_psd())
        psdOL = Field(self.wvlRef, self.N, self.freq_range, 'rad')
        psdOL.sampling = spectrum * (self.dk*self.wvlRef/np.pi)**2

        mask = arrayP3toMastsel(self.fao.ao.tel.pupil)

        psfOL = psdSetToPsfSet([psdOL.sampling], mask, self.wvlRef,
                               nPixPup=self.sx, freq_range=self.freq_range,
                               dk=self.dk, nPixPsf=self.nPixPSF, oversampling=self.overSamp,
                               **flux_options)
        self.psf_ol_array = cpuArray(psfOL[0].sampling)

        # DIFFRACTION LIMITED PSD
        psdDL = Field(self.wvlRef, self.N, self.freq_range, 'rad')
        psfDL = psdSetToPsfSet([psdDL.sampling], mask, self.wvlRef,
                               nPixPup=self.sx, freq_range=self.freq_range,
                               dk=self.dk, nPixPsf=self.nPixPSF, oversampling=self.overSamp,
                               **flux_options)
        self.psf_dl_array = cpuArray(psfDL[0].sampling)

        if self.savePSDs:
            self.psd_array = cpuArray(self.PSD)

        if self.skipPSF1D:
            # radial_profile normally applies this cube scaling in place.
            # Keep Field results untouched for conserved-flux consumers.
            if not self.fullFluxNormalization:
                self.cubeResultsArray /= self.cubeResultsArray.sum(axis=(-2, -1), keepdims=True)
        else:
            self.computePSF1D()

        if self.verbose:
            print('HO_res [nm]:', self.HO_res)
            if self.LOisOn:
                print('LO_res [nm]:', self.LO_res)
            if self.GF_res is not None:
                print('GF_res [nm]:', self.GF_res)

    def _open_loop_psd(self):
        """Open-loop phase PSD on the P3 grid.

        No piston filter: it would remove the low-order power (scales larger
        than the pupil) that broadens the seeing-limited PSF.
        """
        return self.fao.ao.atm.spectrum(np.sqrt(self.fao.freq.k2_))

    def computeMetrics(self):
        """
        Populates Strehl, FWHM, and EE metrics natively.
        """
        self.penalty, self.sr, self.fwhm, self.ee = [], [], [], []

        if len(self.results) == 0:
            # Metric fallback if convolution was skipped
            if self.LOisOn:
                base_pen = np.mean(cpuArray(self.LO_res)**2 + cpuArray(self.HO_res)**2)
                if self.addFocusError and not self.GFinPSD:
                    self.penalty.append(np.sqrt(base_pen + self.GF_res**2))
                else:
                    self.penalty.append(np.sqrt(base_pen))
            else:
                self.penalty.append(np.sqrt(np.mean(cpuArray(self.HO_res)**2)))

            self.sr.append(np.exp(-4*np.pi**2 * (self.penalty[-1]**2)/(self.wvlRef*1e9)**2))
            scale = (np.pi/(180*3600*1000) * 2 * self.tel_radius / (4*1e-9))
            fwhms_lo = 2.355 * self.LO_res/scale / np.sqrt(2) if self.LOisOn else 0.0

            p_fwhm_mas = np.asarray(self.pointings_FWHM_mas) if self.pointings_FWHM_mas is not None else 0.0
            self.fwhm.append(np.sqrt(fwhms_lo**2 + p_fwhm_mas**2))
            self.ee.append(0)
        else:
            for idx, HO_res in enumerate(cpuArray(self.HO_res)):
                if self.LOisOn:
                    if self.addFocusError and not self.GFinPSD:
                        self.penalty.append(np.sqrt(cpuArray(self.LO_res)[idx]**2 + HO_res**2 + self.GF_res**2))
                    else:
                        self.penalty.append(np.sqrt(cpuArray(self.LO_res)[idx]**2 + HO_res**2))
                else:
                    self.penalty.append(HO_res)

            if self.verbose:
                print(f'EE is computed for a radius of {self.eeRadiusInMas} mas')

            for i in range(self.nWvl):
                results_slice = self.results[i] if self.nWvl > 1 else self.results
                samp = self.wvl[i] * rad2mas / (self.psInMas * 2 * self.tel_radius)
                rebin_fwhm = max(1, int(np.ceil(2.0 / samp))) if samp < 2.0 else 1

                sr_l, fwhm_l, ee_l = [], [], []
                for img in results_slice:
                    sr_l.append(getStrehl(arrayMastseltoP3(img.sampling), self.fao.ao.tel.pupil, samp, method='max', psfInOnePix=True))
                    fwhm_l.append(getFWHM(arrayMastseltoP3(img.sampling), self.psInMas, method='contour',
                                          rebin=rebin_fwhm, nargout=1))

                    if self.ensquaredEnergy:
                        ee_ = cpuArray(getEnsquaredEnergy(arrayMastseltoP3(img.sampling)))
                        rr_ = np.arange(1, ee_.shape[0]*2, 2) * self.psInMas * 0.5
                    else:
                        ee_, rr_ = getEncircledEnergy(arrayMastseltoP3(img.sampling), pixelscale=self.psInMas,
                                                      center=centeredPixelCoords(self.nPixPSF), nargout=2)

                    ee_fn = interp1d(rr_, ee_, kind='cubic', bounds_error=False)
                    ee_val = ee_fn(self.eeRadiusInMas)
                    ee_l.append(float(ee_val.item() if hasattr(ee_val, 'item') else ee_val))

                if self.nWvl > 1:
                    self.sr.append(sr_l)
                    self.fwhm.append(fwhm_l)
                    self.ee.append(ee_l)
                else:
                    self.sr = sr_l
                    self.fwhm = fwhm_l
                    self.ee = ee_l

    def _plot_final_PSFs(self):
        if not self.doPlot:
            return
        res = self.results[0] if self.nWvl > 1 else self.results
        if self.LOisOn and self.doConvolve:
            tiledDisplay(res)
            plotEllipses(self.cartSciencePointingCoords, self.cov_ellipses, 0.4)
        else:
            res[0].standardPlot(True)

    # ----------------------------------------------------------------------------
    # --- P3-SPECIFIC HELPER METHODS ---
    # ----------------------------------------------------------------------------

    def _independent_ngs_optics(self):
        """Use the same full-flux pupil-fit HO optics as standalone NGS science.

        Science static aberrations belong to the science arm. The guide-star
        optics are formed at their own wavelength, without that static map.
        Legacy optical selections keep their existing NGS handoff.
        """
        return self.tipTiltRemoval == 'pupil_fit' and self.fullFluxNormalization

    def _compute_ngs_psf(self):
        """Generate the guide-star images and MASTSEL sensor-energy inputs."""
        if self._independent_ngs_optics():
            from .ho_optics import guide_star_optics
            if self.addFocusError or any(n != 1 for n in self.my_data_map['sensor_LO']['NumberLenslets']):
                raise ValueError('Full-flux pupil-fit NGS optics currently supports one-lenslet tilt sensing without a focus loop')
            images, native, provenance = guide_star_optics(self)
            self.ngsPsfFields = images
            self.ngsOpticsProvenance = provenance
            self.NGS_SR_field, self.NGS_FWHM_mas_field, self.NGS_EE_field = [], [], []
            scale = provenance['pixel_scale_mas']
            for img in images:
                self.NGS_SR_field.append(float(np.exp(-img.residual_phase_variance)))
                fx, fy = getFWHM(arrayMastseltoP3(img.sampling), scale, method='contour', nargout=2)
                width = float(np.sqrt(fx*fy)); self.NGS_FWHM_mas_field.append(width)
                # MASTSEL scales its Gaussian centroid model by the physical
                # flux fraction inside the selected NGS energy radius, in mas.
                # Do not renormalize a finite crop to its enclosed flux.
                energy, radius = getEncircledEnergy(arrayMastseltoP3(img.sampling),
                    pixelscale=scale, center=centeredPixelCoords(img.sampling.shape[0]), nargout=2)
                pixel = self.LO_psInMas[len(self.NGS_EE_field)]
                target = width if self.skipEncircledEnergyPixelFloor else max(width, pixel)
                target = min(float(radius[-1]), target)
                self.NGS_EE_field.append(float(interp1d(radius, energy, kind='cubic')(target)))
            self.NGS_DL_FWHM_mas = None
            if self.addLoAlias:
                raise ValueError('Full-flux pupil-fit NGS optics has not validated the optional LO alias-error model')
            return
        LO_PSFsInMas = self.psInMas * self.LO_wvl / self.wvlMax

        if LO_PSFsInMas / np.min(self.LO_psInMas) > 1 and self.overSamp_lo > 1:
            LO_PSFsInMas /= self.overSamp_lo
            nPixPSFLO = int(self.overSamp_lo * self.nPixPSF)
            lo_oversampling = 1.0
        else:
            nPixPSFLO = self.nPixPSF
            lo_oversampling = self.overSamp

        # HO_res/NGS/Focus PSDs are physically wavelength-independent (WFE in
        # nm), but when multiGridPSD is on there is no shared-grid PSD array
        # left (self.PSD is a per-science-wavelength list) -- so the NGS/Focus
        # PSD slices, and the grid geometry (k2_/freq_range/dk/nPixPup) used
        # to turn them into PSFs, must all come from the *same* single grid,
        # here the one matching self.wvlRef (see _prepare_static_PSF_state).
        if self.multiGridPSD:
            ref_grid = self.fao.freq.wvl_grids[self.wvlRef_grid_idx]
            k = np.sqrt(ref_grid.k2_)
            psd_source = self.PSD[self.wvlRef_grid_idx]
            ngs_freq_range = self.freq_range_per_wvl[self.wvlRef_grid_idx]
            ngs_dk = self.dk_per_wvl[self.wvlRef_grid_idx]
            ngs_nPixPup = self.sx_per_wvl[self.wvlRef_grid_idx]
        else:
            k = np.sqrt(self.fao.freq.k2_)
            psd_source = self.PSD
            ngs_freq_range = self.freq_range
            ngs_dk = self.dk
            ngs_nPixPup = self.sx

        psdNGS_view = arrayP3toMastsel(psd_source[-self.nNaturalGS_field:])
        ngs_optics = {}
        if self.tipTiltRemoval == 'pupil_fit':
            ngs_optics = {'tip_tilt_removal': 'pupil_fit',
                          'tilt_fit_pupil': arrayP3toMastsel(self.fao.ao.tel.pupil)}

        psdNGS = []

        nSA = self.my_data_map['sensor_LO']['NumberLenslets']
        maskLO = maskSA(nSA, self.nNaturalGS_field, arrayP3toMastsel(self.fao.ao.tel.pupil))

        for i in range(self.nNaturalGS_field):
            nSAi = nSA[i] if len(nSA) == self.nNaturalGS_field else nSA[0]
            if nSAi != 1:
                pf = pistonFilter(2*self.tel_radius/nSAi, k)
                psdNGS.append(psdNGS_view[i] * arrayP3toMastsel(pf))
            else:
                psdNGS.append(psdNGS_view[i])

        if self.verbose:
            print('******** LO PSF - NGS directions (1 sub-aperture)')

        psfLE_NGS = psdSetToPsfSet(psdNGS, maskLO, self.LO_wvl,
                                   nPixPup=ngs_nPixPup, freq_range=ngs_freq_range,
                                   dk=ngs_dk, nPixPsf=nPixPSFLO, oversampling=lo_oversampling,
                                   opdMap=self.opdMap, **ngs_optics)

        self.NGS_SR_field, self.NGS_FWHM_mas_field, self.NGS_EE_field = [], [], []
        for idx, img in enumerate(psfLE_NGS):
            s1 = cpuArray(psdNGS[idx]).sum()
            SR = (np.exp(-img.residual_phase_variance) if self.tipTiltRemoval == 'pupil_fit'
                  else np.exp(-s1 * (2*np.pi*1e-9/self.LO_wvl)**2))
            self.NGS_SR_field.append(SR)

            fwhmX, fwhmY = getFWHM(arrayMastseltoP3(img.sampling), LO_PSFsInMas, method='contour', nargout=2)
            FWHM = np.sqrt(fwhmX * fwhmY)
            self.NGS_FWHM_mas_field.append(FWHM)

            if 2 * FWHM >= nPixPSFLO * LO_PSFsInMas:
                ee_NGS = 1.0
            else:
                ee_, rr_ = getEncircledEnergy(arrayMastseltoP3(img.sampling), pixelscale=LO_PSFsInMas,
                                              center=centeredPixelCoords(nPixPSFLO), nargout=2)
                ee_ *= 1 / np.max(ee_)
                ee_fn = interp1d(rr_, ee_, kind='cubic', bounds_error=False)
                ps_mas_i = self.LO_psInMas[idx] if isinstance(self.LO_psInMas, list) else self.LO_psInMas
                radius = FWHM if self.skipEncircledEnergyPixelFloor else max(FWHM, ps_mas_i)
                ee_NGS = ee_fn(min(rr_[-1], radius))
                ee_NGS = float(ee_NGS.item() if hasattr(ee_NGS, 'item') else ee_NGS)
            self.NGS_EE_field.append(ee_NGS)

        if self.addLoAlias:
            self.NGS_DL_FWHM_mas = [] if isinstance(maskLO, list) else None
            len_nSA = len(nSA) if len(nSA) == self.nNaturalGS_field else 1
            for i in range(len_nSA):
                maskI = maskLO[i] if isinstance(maskLO, list) else maskLO
                psdDL = Field(self.LO_wvl, self.N, self.freq_range, 'rad')
                maskField = Field(self.LO_wvl, self.N, self.grid_diameter)
                maskField.sampling = congrid(maskI, [self.sx, self.sx])
                maskField.sampling = padOrCropCentered(maskField.sampling, self.N, xp=maskField.xp)
                psfNgsDL = longExposurePsf(maskField, psdDL)
                fx, fy = getFWHM(arrayMastseltoP3(psfNgsDL.sampling), LO_PSFsInMas, method='contour', nargout=2)
                f_val = np.sqrt(fx * fy)
                if self.NGS_DL_FWHM_mas is None:
                    self.NGS_DL_FWHM_mas = f_val
                else:
                    self.NGS_DL_FWHM_mas.append(f_val)
        else:
            self.NGS_DL_FWHM_mas = None

        if self.addFocusError:
            Focus_PSFsInMas = self.psInMas * self.Focus_wvl / self.wvlMax

            if Focus_PSFsInMas / np.min(self.Focus_psInMas) > 1 and self.overSamp_lo > 1:
                Focus_PSFsInMas /= self.overSamp_lo
                nPixPSFFocus = int(self.overSamp_lo * self.nPixPSF)
                focus_oversampling = 1.0
            else:
                nPixPSFFocus = self.nPixPSF
                focus_oversampling = self.overSamp

            if 'sensor_Focus' in self.my_data_map:
                nSAfocus = self.my_data_map['sensor_Focus']['NumberLenslets']
                psdFocus = arrayP3toMastsel(psd_source[-self.nNaturalGS_field:])
                maskFocus = maskSA(nSAfocus, self.nNaturalGS_field, arrayP3toMastsel(self.fao.ao.tel.pupil))

                for i in range(self.nNaturalGS_field):
                    nSAfocusI = nSAfocus[i] if len(nSAfocus) == self.nNaturalGS_field else nSAfocus[0]
                    if nSAfocusI != 1:
                        pf = pistonFilter(2*self.tel_radius/nSAfocusI, k)
                        psdFocus[i] = psdFocus[i] * pf

                psfLE_Focus = psdSetToPsfSet(psdFocus, maskFocus, self.Focus_wvl,
                                             nPixPup=ngs_nPixPup, freq_range=ngs_freq_range,
                                             dk=ngs_dk, nPixPsf=nPixPSFFocus,
                                             oversampling=focus_oversampling,
                                             opdMap=self.opdMap, **ngs_optics)

                self.Focus_SR_field, self.Focus_FWHM_mas_field, self.Focus_EE_field = [], [], []
                for idx, img in enumerate(psfLE_Focus):
                    s1 = cpuArray(psdFocus[idx]).sum()
                    self.Focus_SR_field.append(np.exp(-img.residual_phase_variance)
                                               if self.tipTiltRemoval == 'pupil_fit'
                                               else np.exp(-s1 * (2*np.pi*1e-9/self.Focus_wvl)**2))

                    fwhmX, fwhmY = getFWHM(arrayMastseltoP3(img.sampling), Focus_PSFsInMas, method='contour', nargout=2)
                    FWHM = np.sqrt(fwhmX * fwhmY)
                    self.Focus_FWHM_mas_field.append(FWHM)

                    if 2 * FWHM >= nPixPSFFocus * Focus_PSFsInMas:
                        ee_Focus = 1.0
                    else:
                        ee_, rr_ = getEncircledEnergy(arrayMastseltoP3(img.sampling), pixelscale=Focus_PSFsInMas,
                                                      center=centeredPixelCoords(nPixPSFFocus), nargout=2)
                        ee_ *= 1 / np.max(ee_)
                        ee_fn = interp1d(rr_, ee_, kind='cubic', bounds_error=False)
                        ps_mas_i = self.Focus_psInMas[idx] if isinstance(self.Focus_psInMas, list) else self.Focus_psInMas
                        ee_Focus = ee_fn(max([FWHM, ps_mas_i]))
                        ee_Focus = float(ee_Focus.item() if hasattr(ee_Focus, 'item') else ee_Focus)
                    self.Focus_EE_field.append(ee_Focus)
            else:
                self.Focus_SR_field = self.NGS_SR_field
                self.Focus_FWHM_mas_field = self.NGS_FWHM_mas_field
                self.Focus_EE_field = self.NGS_EE_field

    def _apply_final_convolution(self, lo_data):
        self.cov_ellipses = lo_data.get('cov_ellipses')
        resSpecList, resSpecListJ = [], []

        for ellp in self.cov_ellipses:
            ellp = ellp.astype(self.fao.dtype)
            resSpecList.append(residualToSpectrum(ellp, self.wvlRef, self.nPixPSF, 1/(self.nPixPSF * self.psInMas)))

            if self.jitter_FWHM is not None:
                ellpJ = [self.jitter_FWHM[2], sigma_from_FWHM(self.jitter_FWHM[0]), sigma_from_FWHM(self.jitter_FWHM[1])] \
                    if isinstance(self.jitter_FWHM, list) \
                        else [0, sigma_from_FWHM(self.jitter_FWHM), sigma_from_FWHM(self.jitter_FWHM)]
                ellpJ = np.array(ellpJ, dtype=self.fao.dtype)
                resSpecListJ.append(residualToSpectrum(ellpJ, self.wvlRef, self.nPixPSF, 1/(self.nPixPSF * self.psInMas)))
            else:
                resSpecListJ.append(0)

        for i in range(self.nWvl):
            psfList = self.psfLongExpPointingsArr[i] if self.nWvl > 1 else self.psfLongExpPointingsArr
            resultList = []
            for psfLongExp, resSpec, resSpecJ in zip(psfList, resSpecList, resSpecListJ):
                temp = self._convolve_science_psf(psfLongExp, resSpec)
                if self.jitter_FWHM is not None:
                    temp = self._convolve_science_psf(temp, resSpecJ)
                resultList.append(temp)

            if self.nWvl > 1:
                self.results.append(resultList)
            else:
                self.results = resultList

    def _apply_no_convolution(self):
        for i in range(self.nWvl):
            psfList = self.psfLongExpPointingsArr[i] if self.nWvl > 1 else self.psfLongExpPointingsArr
            if self.nWvl > 1:
                self.results.append(list(psfList))
            else:
                self.results = list(psfList)

    def _apply_jitter_convolution(self):
        if self.jitter_FWHM is not None:
            ellpJ = [self.jitter_FWHM[2], sigma_from_FWHM(self.jitter_FWHM[0]), sigma_from_FWHM(self.jitter_FWHM[1])] \
                if isinstance(self.jitter_FWHM, list) \
                    else [0, sigma_from_FWHM(self.jitter_FWHM), sigma_from_FWHM(self.jitter_FWHM)]
            resSpecJ = residualToSpectrum(ellpJ, self.wvlRef, self.nPixPSF, 1/(self.nPixPSF * self.psInMas))

        for i in range(self.nWvl):
            psfList = self.psfLongExpPointingsArr[i] if self.nWvl > 1 else self.psfLongExpPointingsArr
            resultList = []
            for psfLongExp in psfList:
                resultList.append(self._convolve_science_psf(psfLongExp, resSpecJ)
                                  if self.jitter_FWHM is not None else psfLongExp)

            if self.nWvl > 1:
                self.results.append(resultList)
            else:
                self.results = resultList

    def _convolve_science_psf(self, psf, kernel):
        """Select MASTSEL's explicit input-normalization contract."""
        if self.fullFluxNormalization:
            return convolve(psf, kernel, normalize_input=False)
        return convolve(psf, kernel)
