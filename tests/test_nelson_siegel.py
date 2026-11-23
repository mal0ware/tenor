import numpy as np
import pytest

from tenor import nelson_siegel as ns
from tenor import published

TAU = np.array([3.0, 6.0, 12.0, 24.0, 36.0, 60.0, 84.0, 120.0])


def test_loadings_limits():
    np.testing.assert_allclose(ns.loadings(0.0)[0], [1.0, 1.0, 0.0])
    np.testing.assert_allclose(ns.loadings(1e-9)[0], [1.0, 1.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(ns.loadings(1e7)[0], [1.0, 0.0, 0.0], atol=1e-5)


def test_yield_limits_match_paper_statements():
    # Diebold and Li (2006), section 2.2: y(infinity) = b1 and y(0) = b1 + b2.
    beta = np.array([7.5, -2.1, 0.4])
    assert ns.ns_yield(beta, 1e9)[0] == pytest.approx(7.5, abs=1e-6)
    assert ns.ns_yield(beta, 0.0)[0] == pytest.approx(7.5 - 2.1)


def test_slope_loading_decreases_and_curvature_is_a_hump():
    grid = np.linspace(0.01, 400.0, 4000)
    load = ns.loadings(grid)
    assert (np.diff(load[:, 1]) < 0).all()
    peak = int(np.argmax(load[:, 2]))
    assert (np.diff(load[: peak + 1, 2]) > 0).all()
    assert (np.diff(load[peak:, 2]) < 0).all()
    assert (load[:, 2] >= 0).all()


def test_published_slope_and_curvature_coefficients():
    # Diebold and Li (2006), section 2.2, with lambda = 0.0609:
    #   y(120) - y(3)            = -0.78 b2 + 0.06 b3
    #   2 y(24) - y(3) - y(120)  = 0.00053 b2 + 0.37 b3
    load = ns.loadings([3.0, 24.0, 120.0], ns.DL_LAMBDA)
    slope = load[2] - load[0]
    curv = 2 * load[1] - load[0] - load[2]
    assert slope[0] == pytest.approx(0.0, abs=1e-15)
    assert curv[0] == pytest.approx(0.0, abs=1e-15)
    assert round(slope[1], 2) == published.DL_SLOPE_COEFS[0]
    assert round(slope[2], 2) == published.DL_SLOPE_COEFS[1]
    assert round(curv[2], 2) == published.DL_CURVATURE_COEFS[1]
    # The paper prints 0.00053 for the b2 coefficient of the curvature. With
    # lambda = 0.0609 it computes to 0.00038. Both are negligible next to the
    # 0.37 on b3; the difference is recorded in PAPER.md.
    assert round(curv[1], 5) == 0.00038
    assert published.DL_CURVATURE_COEFS[0] == 0.00053
    assert abs(curv[1] - published.DL_CURVATURE_COEFS[0]) < 2e-4


def test_hump_argmax_solves_first_order_condition():
    x = ns.HUMP_ARGMAX
    assert np.exp(-x) * (x * x + x + 1.0) == pytest.approx(1.0, abs=1e-12)
    grid = np.linspace(0.5, 5.0, 200001)
    hump = -np.expm1(-grid) / grid - np.exp(-grid)
    assert grid[np.argmax(hump)] == pytest.approx(x, abs=1e-4)
    assert x == pytest.approx(1.7933, abs=5e-5)


def test_paper_lambda_does_not_peak_at_exactly_thirty_months():
    # The paper says 0.0609 maximises the curvature loading "at exactly 30
    # months". The loading at 0.0609 peaks near 29.4 months; the decay that
    # peaks at 30 months is about 0.0598. Recorded in PAPER.md.
    assert ns.curvature_peak(ns.DL_LAMBDA) == pytest.approx(29.45, abs=0.01)
    assert ns.lambda_for_peak(30.0) == pytest.approx(0.05978, abs=1e-5)
    assert ns.curvature_peak(ns.lambda_for_peak(30.0)) == pytest.approx(30.0)


def test_fit_reproduces_its_own_curves():
    rng = np.random.default_rng(0)
    beta = rng.normal([6.0, -2.0, 0.5], [2.0, 1.5, 2.0], size=(200, 3))
    y = ns.ns_yield(beta, TAU)
    fitted, resid = ns.fit(y, TAU)
    np.testing.assert_allclose(fitted, beta, atol=1e-10)
    np.testing.assert_allclose(resid, 0.0, atol=1e-10)


def test_fit_single_curve_and_other_decay():
    beta = np.array([4.0, 1.0, -3.0])
    for lam in (0.02, 0.0609, 0.2):
        y = ns.ns_yield(beta, TAU, lam)
        fitted, resid = ns.fit(y, TAU, lam)
        assert fitted.shape == (3,)
        np.testing.assert_allclose(fitted, beta, atol=1e-9)
        np.testing.assert_allclose(resid, 0.0, atol=1e-9)


def test_fit_is_ordinary_least_squares():
    rng = np.random.default_rng(1)
    y = rng.normal(5.0, 1.0, size=(30, TAU.size))
    beta, resid = ns.fit(y, TAU)
    design = ns.loadings(TAU)
    for row in range(30):
        ref, *_ = np.linalg.lstsq(design, y[row], rcond=None)
        np.testing.assert_allclose(beta[row], ref, atol=1e-10)
    # residuals are orthogonal to every regressor
    np.testing.assert_allclose(resid @ design, 0.0, atol=1e-9)


def test_fit_is_unbiased_under_noise():
    rng = np.random.default_rng(2)
    beta = np.array([6.0, -2.0, 1.0])
    y = ns.ns_yield(beta, TAU) + 0.05 * rng.standard_normal((20000, TAU.size))
    fitted, _ = ns.fit(y, TAU)
    np.testing.assert_allclose(fitted.mean(axis=0), beta, atol=0.01)


def test_original_nelson_siegel_factorisation_is_equivalent():
    # Diebold and Li (2006), end of section 2.2: the original form
    # y = b1 + b2 (1-e^{-x})/x - b3 e^{-x} matches theirs with
    # b1 = beta1, b2 = beta2 + beta3, b3 = beta3.
    beta = np.array([6.5, -1.7, 2.3])
    x = ns.DL_LAMBDA * TAU
    original = beta[0] + (beta[1] + beta[2]) * (1 - np.exp(-x)) / x - beta[2] * np.exp(-x)
    np.testing.assert_allclose(ns.ns_yield(beta, TAU), original, atol=1e-12)


def test_yield_is_the_average_forward_rate():
    # Section 2.1: y(tau) = (1/tau) * integral_0^tau f(u) du.
    beta = np.array([6.0, -2.5, 1.5])
    for tau in (3.0, 24.0, 120.0):
        u = np.linspace(0.0, tau, 200001)
        f = ns.ns_forward(beta, u)
        integral = float(((f[1:] + f[:-1]) * 0.5 * np.diff(u)).sum())
        assert integral / tau == pytest.approx(float(ns.ns_yield(beta, tau)[0]), abs=1e-8)


def test_forward_curve_limits():
    beta = np.array([6.0, -2.5, 1.5])
    assert ns.ns_forward(beta, 0.0)[0] == pytest.approx(6.0 - 2.5)
    assert ns.ns_forward(beta, 1e6)[0] == pytest.approx(6.0)


def test_validation():
    with pytest.raises(ValueError):
        ns.loadings([1.0], lam=0.0)
    with pytest.raises(ValueError):
        ns.loadings([-1.0])
    with pytest.raises(ValueError):
        ns.fit(np.ones((2, 3)), TAU)
    with pytest.raises(ValueError):
        ns.fit(np.ones((2, 2)), [1.0, 2.0])
    with pytest.raises(ValueError):
        ns.fit(np.full((2, 8), np.nan), TAU)
    with pytest.raises(ValueError):
        ns.curvature_peak(0.0)
    with pytest.raises(ValueError):
        ns.lambda_for_peak(-1.0)
