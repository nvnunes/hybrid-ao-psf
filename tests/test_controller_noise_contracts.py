"""Standalone controller covariance and noise-blur scientific contracts."""

import subprocess
import sys

import numpy as np
import pytest
from astropy import units as u

from hybrid_ao_psf import FixedGainIntegrator, NoiseBlurModel


@pytest.mark.parametrize("delay", [1, 2, 4])
def test_lyapunov_covariance_matches_direct_delayed_impulse_filtering(delay):
    response = np.array([[0.8, 0.15], [-0.05, 1.1]])
    noise = np.array([[4.0, 1.0], [1.0, 2.0]])
    gain = 0.17
    response_before, noise_before = response.copy(), noise.copy()
    # Propagate the stated command recurrence without using its state matrix.
    impulse = np.zeros((2048, 2, 2))
    for k in range(1, len(impulse)):
        impulse[k] = impulse[k - 1]
        if k >= delay:
            impulse[k] -= gain * response @ impulse[k - delay]
        if k == delay:
            impulse[k] += gain * np.eye(2)
    expected = np.sum(impulse @ noise @ impulse.swapaxes(-1, -2), axis=0)
    actual = (
        FixedGainIntegrator(gain=gain, delay=delay, frame_rate=500 * u.Hz)
        .noise_covariance(response, noise * u.mas**2)
        .value
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(response, response_before)
    np.testing.assert_array_equal(noise, noise_before)


def test_transition_matches_newest_first_command_history():
    response = np.array([[0.8, 0.15], [-0.05, 1.1]])
    history = np.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]])
    expected = np.vstack([history[0] - 0.1 * response @ history[-1], history[:-1]])
    transition = FixedGainIntegrator(
        gain=0.1, delay=3, frame_rate=500 * u.Hz
    ).transition(response)
    np.testing.assert_allclose(transition @ history.ravel(), expected.ravel())


@pytest.mark.parametrize("delay", [0, True, 1.5])
def test_invalid_frame_delays_are_rejected(delay):
    with pytest.raises(ValueError, match="delay"):
        FixedGainIntegrator(gain=0.3, delay=delay, frame_rate=500 * u.Hz).transition(
            np.eye(2)
        )


def test_unstable_loop_and_nonphysical_noise_are_rejected():
    with pytest.raises(ValueError, match="Unstable"):
        FixedGainIntegrator(gain=3.0, delay=1, frame_rate=500 * u.Hz).noise_covariance(
            np.eye(2), np.eye(2) * u.mas**2
        )
    with pytest.raises(ValueError, match="positive semidefinite"):
        FixedGainIntegrator(gain=0.3, delay=1, frame_rate=500 * u.Hz).noise_covariance(
            np.eye(2), np.diag([-1, 2]) * u.mas**2
        )
    with pytest.raises(ValueError, match="symmetric"):
        FixedGainIntegrator(gain=0.3, delay=1, frame_rate=500 * u.Hz).noise_covariance(
            np.eye(2), np.array([[1, 2], [0, 1]]) * u.mas**2
        )


@pytest.mark.parametrize("count", [1, 2, 3])
def test_blur_preserves_covariance_and_matches_gaussian_characteristic_function(count):
    model = NoiseBlurModel(0.72, 0.32, 0.63)
    covariance = np.array([[4.0, 1.0], [1.0, 2.0]])
    before = covariance.copy()
    narrow, broad = model.scales(count)
    assert model.mixture_weight * narrow + (
        1 - model.mixture_weight
    ) * broad == pytest.approx(1)
    frequency = np.fft.fftfreq(32, 5.0)
    expected = np.zeros((32, 32))
    # Independent per-frequency expression checks xy ordering and the cross term.
    for row, fy in enumerate(frequency):
        for column, fx in enumerate(frequency):
            vector = np.array([fx, fy])
            exponent = -2 * np.pi**2 * vector @ covariance @ vector
            expected[row, column] = model.mixture_weight * np.exp(narrow * exponent) + (
                1 - model.mixture_weight
            ) * np.exp(broad * exponent)
    np.testing.assert_allclose(
        model.otf(covariance * u.mas**2, count, 32, 5.0 * u.mas), expected, atol=1e-15
    )
    np.testing.assert_array_equal(covariance, before)
    assert model.otf(covariance * u.mas**2, count, 32, 5.0 * u.mas)[0, 0] == 1
    np.testing.assert_array_equal(
        model.otf(np.zeros((2, 2)) * u.mas**2, count, 8, 5.0 * u.mas), np.ones((8, 8))
    )


def test_gaussian_limit_has_chi_squared_radial_cdf():
    model = NoiseBlurModel(0.72, 1.0, 1.0)
    radius = np.array([0, 0.1, 2.0, 20.0, np.inf])
    for count in (1, 3):
        np.testing.assert_allclose(
            model.radial_cdf(radius, count), -np.expm1(-radius / 2)
        )


def test_otf_uses_same_coefficient_precision_as_saved_parameters():
    model = NoiseBlurModel(*map(np.float32, (0.72, 0.32, 0.63)))
    restored = NoiseBlurModel(**model.parameters())
    covariance = np.array([[4.0, 1.0], [1.0, 2.0]])
    for count in (1, 3):
        np.testing.assert_array_equal(
            model.otf(covariance * u.mas**2, count, 32, 5.0 * u.mas),
            restored.otf(covariance * u.mas**2, count, 32, 5.0 * u.mas),
        )


@pytest.mark.parametrize(
    "values", [(0, 0.5, 0.5), (0.5, 0, 1), (0.5, 1.1, 1), (0.5, np.nan, 1)]
)
def test_nonphysical_mixture_coefficients_are_rejected(values):
    with pytest.raises(ValueError):
        NoiseBlurModel(*values)


def test_invalid_blur_inputs_are_rejected():
    model = NoiseBlurModel(0.72, 0.32, 0.63)
    for count in (0, True, 1.5):
        with pytest.raises(ValueError, match="NGS count"):
            model.otf(np.eye(2) * u.mas**2, count, 8, 5.0 * u.mas)
    for side in (0, True, 1.5):
        with pytest.raises(ValueError, match="FFT side"):
            model.otf(np.eye(2) * u.mas**2, 1, side, 5.0 * u.mas)
    with pytest.raises(ValueError, match="positive semidefinite"):
        model.otf(np.diag([-1, 1]) * u.mas**2, 1, 8, 5.0 * u.mas)
    with pytest.raises(ValueError, match="Squared radius"):
        model.radial_cdf(-1, 1)


def test_kernels_import_without_corrections_or_calibration():
    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith(('ao_tools.hybrid.corrections', 'ao_tools.hybrid.sensing', 'girmos_oomao')):
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from hybrid_ao_psf import FixedGainIntegrator, NoiseBlurModel
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )


def test_mixture_coefficients_own_immutable_scalar_values():
    weight = np.array(0.72)
    model = NoiseBlurModel(weight, 0.32, 0.63)
    weight[...] = 0.1
    assert model.mixture_weight == 0.72
    assert isinstance(model.mixture_weight, float)
