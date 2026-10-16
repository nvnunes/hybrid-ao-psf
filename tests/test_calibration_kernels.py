"""Independent recurrence and covariance checks for shared calibration kernels."""

import numpy as np
import pytest
from astropy import units as u

from hybrid_ao_psf import FixedGainIntegrator, NoiseBlurModel
from hybrid_ao_psf._calibration_numerics import (
    lo_reconstructor,
    scalar_filter,
    settling_cut,
)


def direct_response(history, response, gain, delay, equilibrium=False):
    output = np.zeros_like(history)
    prehistory = np.linalg.solve(response, history[0]) if equilibrium else np.zeros(2)
    for frame in range(len(history)):
        previous = output[frame - 1] if frame else prehistory
        delayed = output[frame - delay] if frame >= delay else prehistory
        source = (
            history[frame - delay]
            if frame >= delay
            else history[0]
            if equilibrium
            else np.zeros(2)
        )
        output[frame] = previous + gain * (source - response @ delayed)
    return output


@pytest.mark.parametrize("delay", [1, 2, 4])
def test_matrix_response_matches_direct_independent_resets(delay):
    rng = np.random.default_rng(110)
    histories = rng.normal(size=(3, 120, 2)) + np.arange(3)[:, None, None] * 4
    response = np.array([[0.8, 0.16], [-0.12, 1.1]])
    controller = FixedGainIntegrator(gain=0.2, delay=delay, frame_rate=500 * u.Hz)
    expected = np.stack([direct_response(h, response, 0.2, delay) for h in histories])
    actual = controller.filter(histories * u.arcsec, response)
    np.testing.assert_allclose(
        actual.to_value(u.arcsec), expected, rtol=1e-11, atol=1e-11
    )
    assert not np.shares_memory(actual.value, histories)


@pytest.mark.parametrize("delay", [1, 2, 3])
def test_scalar_filter_matches_equilibrium_and_zero_recurrence(delay):
    histories = np.random.default_rng(42).normal(size=(2, 100)) + 3
    for equilibrium in (False, True):
        expected = direct_response(
            histories.T, 0.7 * np.eye(2), 0.2, delay, equilibrium
        ).T
        actual = scalar_filter(histories, 0.7, 0.2, delay, equilibrium=equilibrium)
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)


def test_accepted_settling_window_unchanged():
    cut, envelope = settling_cut(0.3, 2, np.arange(2, 21) / 10)
    assert cut == 285 and envelope <= 1e-8


def test_lyapunov_covariance_matches_independent_impulse_accumulation():
    response = np.array([[0.8, 0.1], [-0.2, 1.0]])
    noise = np.array([[3.0, 0.6], [0.6, 2.0]])
    controller = FixedGainIntegrator(gain=0.2, delay=2, frame_rate=500 * u.Hz)
    impulse = np.zeros((2, 1000, 2))
    impulse[:, 0, :] = np.eye(2)
    outputs = np.stack([direct_response(h, response, 0.2, 2) for h in impulse])
    kernels = outputs.transpose(1, 2, 0)
    expected = np.einsum("kab,bc,kdc->ad", kernels, noise, kernels)
    np.testing.assert_allclose(
        controller.noise_covariance(response, noise * u.mas**2).value,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        controller.noise_covariance(np.eye(2), np.eye(2) * u.mas**2).value,
        np.eye(2) * 0.2 * 1.2 / (2 - 0.2 - 0.2**2),
        rtol=1e-12,
    )


def test_weight_solver_preserves_absolute_response_and_one_ngs_identity():
    css = np.diag([3.0, 5.0, 7.0, 9.0])
    noise = np.diag([0.2, 0.5, 0.4, 0.3])
    cas = np.arange(8).reshape(2, 4) / 10
    expected = cas / np.diag(css + noise)
    np.testing.assert_allclose(lo_reconstructor(css, cas, noise), expected, rtol=1e-14)
    np.testing.assert_array_equal(
        lo_reconstructor(np.eye(2) * 3, np.eye(2), np.eye(2)), np.eye(2)
    )


@pytest.mark.parametrize("count", [1, 3])
def test_noise_blur_has_requested_covariance(count):
    model = NoiseBlurModel(0.7, 0.35, 0.65)
    narrow, broad = model.scales(count)
    assert model.mixture_weight * narrow + (
        1 - model.mixture_weight
    ) * broad == pytest.approx(1)
    covariance = np.array([[24.0, 7.0], [7.0, 16.0]])
    otf = model.otf(covariance * u.mas**2, count, 1024, 1 * u.mas)
    frequency = 1 / 1024
    recovered = -np.log(otf[0, 1]) / (2 * np.pi**2 * frequency**2)
    assert recovered == pytest.approx(covariance[0, 0], rel=3e-4)
    assert otf[0, 0] == 1


def test_kernel_public_units_and_unstable_loop_rejection():
    controller = FixedGainIntegrator(gain=0.3, delay=2, frame_rate=500 * u.Hz)
    with pytest.raises(ValueError, match="Unstable"):
        controller.transition(np.eye(2) * 10)
    with pytest.raises(TypeError):
        controller.noise_covariance(np.eye(2), np.eye(2))
    with pytest.raises(ValueError, match="positive semidefinite"):
        controller.noise_covariance(np.eye(2), np.diag([-1, 2]) * u.mas**2)


def test_matrix_first_input_equilibrium_matches_direct_recurrence():
    histories = np.random.default_rng(239).normal(size=(3, 120, 2)) + 4
    response = np.array([[0.8, 0.16], [-0.12, 1.1]])
    controller = FixedGainIntegrator(gain=0.2, delay=2, frame_rate=500 * u.Hz)
    expected = np.stack([direct_response(h, response, 0.2, 2, True) for h in histories])
    np.testing.assert_allclose(
        controller.filter(histories * u.mas, response, equilibrium=True).value,
        expected,
        rtol=1e-11,
        atol=1e-11,
    )
