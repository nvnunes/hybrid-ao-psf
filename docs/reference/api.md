# Python API

- Build [PSF image interpolators](#hybrid_ao_psf.build_psf_interpolator) and
  [PSF metric interpolators](#hybrid_ao_psf.build_psf_metric_interpolator_from_psfs).
- Read existing [PSF files](#hybrid_ao_psf.PsfFileProvider) with a supplied reader.
- Supply calibration data through [E2ECalibration](#hybrid_ao_psf.E2ECalibration)
  and build the [four fitted products](../calibration.md#calibration-workflow).
- Evaluate [calibrated low-order motion](#hybrid_ao_psf.E2ECalibratedModel)
  or [simulate the final science PSFs](#hybrid_ao_psf.simulate).
- Run [one configured simulation](#hybrid_ao_psf.run_configuration) and
  [load its HDF5 result](#hybrid_ao_psf.load_result).

::: hybrid_ao_psf
    options:
      preload_modules: [ao_models]
      inherited_members:
        - transition
        - filter
        - noise_covariance
