# Python API

- Build [PSF image interpolators](#hybrid_ao_psf.build_psf_interpolator) and
  [PSF metric interpolators](#hybrid_ao_psf.build_psf_metric_interpolator_from_psfs).
- Read existing [PSF files](#hybrid_ao_psf.PsfFileProvider) with a supplied reader.
- Build the [four fitted products](../calibration.md#calibration-workflow)
  from [saved HDF5 telemetry](#hybrid_ao_psf.load_telemetry_calibration) or from
  [live E2E measurements](#hybrid_ao_psf.E2ECalibration).
- Evaluate [calibrated low-order motion](#hybrid_ao_psf.E2ECalibratedModel)
  or [simulate the final science PSFs](#hybrid_ao_psf.simulate).
- Run [one configured simulation](#hybrid_ao_psf.run_configuration) and
  [load its HDF5 result](#hybrid_ao_psf.load_result).

::: hybrid_ao_psf
    options:
      preload_modules: [ao_core.models.control]
      inherited_members:
        - transition
        - filter
        - noise_covariance
        - configuration
        - build_reconstructor
        - save_reconstructor
        - build_covariance
        - save_covariance
        - build_coma
        - save_coma
        - simulate_sensing_noise
        - build_sensing
        - save_sensing
