import numpy as np
import pytest

from tenor import forecast as F
from tenor import nelson_siegel as ns

TAU = np.array([3.0, 6.0, 12.0, 24.0, 36.0, 60.0, 84.0, 120.0])


def _ar1_path(c: float, g: float, x0: float, n: int) -> np.ndarray:
    x = np.empty(n)
    x[0] = x0
    for t in range(1, n):
        x[t] = c + g * x[t - 1]
    return x


def _factor_panel(seed: int = 0, n: int = 240, noise: float = 0.02):
    rng = np.random.default_rng(seed)
    means = np.array([6.0, -2.0, 0.5])
    phi = np.array([0.98, 0.95, 0.85])
    beta = np.empty((n, 3))
    state = means.copy()
    for t in range(n):
        state = means + phi * (state - means) + np.array([0.2, 0.25, 0.4]) * rng.standard_normal(3)
        beta[t] = state
    y = ns.ns_yield(beta, TAU) + noise * rng.standard_normal((n, TAU.size))
    return y, beta


@pytest.mark.parametrize("h", [1, 6, 12])
def test_direct_ar1_recovers_a_noiseless_ar1(h):
    # For x_{t+1} = c + g x_t the h-step map is x_{t+h} = c (1-g^h)/(1-g) + g^h x_t.
    c, g = 0.3, 0.9
    x = _ar1_path(c, g, 10.0, 80)
    c_h, g_h = F.direct_ar1(x, h)
    assert g_h[0] == pytest.approx(g**h, rel=1e-8)
    assert c_h[0] == pytest.approx(c * (1 - g**h) / (1 - g), rel=1e-8)
    nxt = _ar1_path(c, g, x[-1], h + 1)[-1]
    assert F.direct_ar1_forecast(x, h)[0] == pytest.approx(nxt, rel=1e-9)


def test_direct_ar1_matches_least_squares_on_each_column():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((60, 3)).cumsum(axis=0)
    h = 4
    c, g = F.direct_ar1(x, h)
    for j in range(3):
        design = np.column_stack([np.ones(60 - h), x[: 60 - h, j]])
        ref, *_ = np.linalg.lstsq(design, x[h:, j], rcond=None)
        assert c[j] == pytest.approx(ref[0], abs=1e-10)
        assert g[j] == pytest.approx(ref[1], abs=1e-10)


def test_direct_var1_recovers_a_noiseless_var1():
    big_g = np.array([[0.9, 0.05, 0.0], [0.0, 0.8, 0.1], [0.02, 0.0, 0.7]])
    c = np.array([0.1, -0.2, 0.3])
    x = np.empty((60, 3))
    x[0] = [1.0, 2.0, -1.0]
    for t in range(1, 60):
        x[t] = c + big_g @ x[t - 1]
    c_hat, g_hat = F.direct_var1(x, 1)
    np.testing.assert_allclose(g_hat, big_g, atol=1e-7)
    np.testing.assert_allclose(c_hat, c, atol=1e-7)
    np.testing.assert_allclose(F.direct_var1_forecast(x, 1), c + big_g @ x[-1], atol=1e-8)
    # two steps ahead: x_{t+2} = c + G c + G^2 x_t
    c2, g2 = F.direct_var1(x, 2)
    np.testing.assert_allclose(g2, big_g @ big_g, atol=1e-6)
    np.testing.assert_allclose(c2, c + big_g @ c, atol=1e-6)


def test_var_with_diagonal_dynamics_agrees_with_ar_in_the_limit():
    x = np.column_stack([_ar1_path(0.2, 0.9, 5.0, 100), _ar1_path(-0.1, 0.7, -3.0, 100) + 0.0])
    # perturb so the regressors are not collinear
    rng = np.random.default_rng(3)
    x = x + 1e-3 * rng.standard_normal(x.shape)
    ar = F.direct_ar1_forecast(x, 1)
    var = F.direct_var1_forecast(x, 1)
    np.testing.assert_allclose(ar, var, atol=5e-3)


def test_random_walk_is_no_change_on_exactly_the_forecast_dates():
    y, _ = _factor_panel()
    for h in (1, 6, 12):
        fs = F.recursive_forecasts(y, TAU, h, first_target=120)
        np.testing.assert_array_equal(fs.targets - fs.origins, h)
        np.testing.assert_array_equal(fs.targets, np.arange(120, 240))
        np.testing.assert_array_equal(fs.forecasts["rw"], y[fs.origins])
        np.testing.assert_array_equal(fs.actual, y[fs.targets])
        np.testing.assert_array_equal(fs.errors("rw"), y[fs.targets] - y[fs.targets - h])


def test_all_models_share_dates_and_shapes():
    y, _ = _factor_panel()
    fs = F.recursive_forecasts(y, TAU, 6, first_target=120, last_target=200)
    assert set(fs.forecasts) == set(F.MODELS)
    for name in F.MODELS:
        assert fs.forecasts[name].shape == (81, TAU.size)
        assert fs.errors(name).shape == fs.actual.shape
        assert np.isfinite(fs.forecasts[name]).all()


def test_no_look_ahead():
    y, _ = _factor_panel(seed=5)
    h = 6
    base = F.recursive_forecasts(y, TAU, h, first_target=120, last_target=180)
    y2 = y.copy()
    cut = 150  # change everything after row 150
    y2[cut + 1 :] += 5.0
    alt = F.recursive_forecasts(y2, TAU, h, first_target=120, last_target=180)
    unaffected = base.origins <= cut
    assert unaffected.any() and (~unaffected).any()
    for name in F.MODELS:
        np.testing.assert_array_equal(
            base.forecasts[name][unaffected], alt.forecasts[name][unaffected]
        )
    assert not np.allclose(
        base.forecasts["ns_ar"][~unaffected], alt.forecasts["ns_ar"][~unaffected]
    )


def test_forecast_at_one_origin_matches_manual_computation():
    y, _ = _factor_panel(seed=6)
    h = 12
    fs = F.recursive_forecasts(y, TAU, h, first_target=150, last_target=150)
    origin = 150 - h
    beta, _ = ns.fit(y[: origin + 1], TAU)
    pred = np.empty(3)
    for i in range(3):
        design = np.column_stack([np.ones(origin + 1 - h), beta[: origin + 1 - h, i]])
        coef, *_ = np.linalg.lstsq(design, beta[h:, i], rcond=None)
        pred[i] = coef[0] + coef[1] * beta[-1, i]
    np.testing.assert_allclose(fs.forecasts["ns_ar"][0], ns.ns_yield(pred, TAU), atol=1e-9)


def test_start_row_excludes_earlier_data():
    y, _ = _factor_panel(seed=7)
    base = F.recursive_forecasts(y[60:], TAU, 1, first_target=60)
    y2 = y.copy()
    y2[:60] = 99.0
    alt = F.recursive_forecasts(y2, TAU, 1, first_target=120, start=60)
    for name in F.MODELS:
        np.testing.assert_allclose(base.forecasts[name], alt.forecasts[name])


def test_constant_factor_curve_is_forecast_almost_exactly():
    # If the curve is Nelson-Siegel and the factors follow an exact AR(1),
    # the Nelson-Siegel AR forecast has no error.
    n = 120
    beta = np.column_stack(
        [
            _ar1_path(0.06, 0.99, 7.0, n),
            _ar1_path(-0.1, 0.95, -1.0, n),
            _ar1_path(0.05, 0.9, 2.0, n),
        ]
    )
    y = ns.ns_yield(beta, TAU)
    fs = F.recursive_forecasts(y, TAU, 6, first_target=60)
    np.testing.assert_allclose(fs.errors("ns_ar"), 0.0, atol=1e-6)
    assert np.abs(fs.errors("rw")).max() > 1e-3


def test_ns_ar_beats_random_walk_at_long_horizon_when_factors_mean_revert():
    y, _ = _factor_panel(seed=11, n=600)
    fs = F.recursive_forecasts(y, TAU, 12, first_target=240)
    mse_model = (fs.errors("ns_ar") ** 2).mean()
    mse_rw = (fs.errors("rw") ** 2).mean()
    assert mse_model < mse_rw


def test_validation():
    y, _ = _factor_panel(n=60)
    with pytest.raises(ValueError):
        F.recursive_forecasts(y, TAU, 12, first_target=5)
    with pytest.raises(ValueError):
        F.recursive_forecasts(y, TAU, 1, first_target=70)
    with pytest.raises(ValueError):
        F.direct_ar1(np.arange(4.0), 2)
    with pytest.raises(ValueError):
        F.direct_ar1(np.ones(20), 1)
    with pytest.raises(ValueError):
        F.direct_ar1(np.arange(10.0), 0)
    with pytest.raises(ValueError):
        F.direct_var1(np.ones((5, 3)), 1)


def test_factor_random_walk_is_todays_fitted_curve():
    y, _ = _factor_panel(seed=12)
    fs = F.recursive_forecasts(y, TAU, 6, first_target=120)
    beta, resid = ns.fit(y, TAU)
    np.testing.assert_allclose(fs.forecasts["ns_rw"], ns.ns_yield(beta[fs.origins], TAU))
    # it differs from the random walk by the fit residual at the origin only
    np.testing.assert_allclose(
        fs.forecasts["rw"] - fs.forecasts["ns_rw"], resid[fs.origins], atol=1e-12
    )


def test_factor_random_walk_is_the_ar_model_with_unit_slope_imposed():
    # On an exactly constant factor path the AR(1) regression is undefined, so
    # check the restriction directly: c = 0, g = 1 maps beta_t to itself.
    y, _ = _factor_panel(seed=13)
    beta, _ = ns.fit(y, TAU)
    design = ns.loadings(TAU)
    fs = F.recursive_forecasts(y, TAU, 3, first_target=100, last_target=110)
    for i, origin in enumerate(fs.origins):
        np.testing.assert_allclose(fs.forecasts["ns_rw"][i], (0.0 + 1.0 * beta[origin]) @ design.T)


def test_rolling_window_uses_only_the_last_rows():
    y, _ = _factor_panel(seed=14)
    h, window = 6, 60
    rolled = F.recursive_forecasts(y, TAU, h, first_target=150, last_target=170, window=window)
    for i, origin in enumerate(rolled.origins):
        lo = origin + 1 - window
        alone = F.recursive_forecasts(
            y[lo:], TAU, h, first_target=origin + h - lo, last_target=origin + h - lo
        )
        for name in F.MODELS:
            np.testing.assert_allclose(rolled.forecasts[name][i], alone.forecasts[name][0])
    y2 = y.copy()
    y2[:80] += 3.0  # before every window
    again = F.recursive_forecasts(y2, TAU, h, first_target=150, last_target=170, window=window)
    for name in F.MODELS:
        np.testing.assert_array_equal(rolled.forecasts[name], again.forecasts[name])
    recursive = F.recursive_forecasts(y, TAU, h, first_target=150, last_target=170)
    assert not np.allclose(rolled.forecasts["ns_ar"], recursive.forecasts["ns_ar"])
    np.testing.assert_array_equal(rolled.forecasts["rw"], recursive.forecasts["rw"])
    with pytest.raises(ValueError):
        F.recursive_forecasts(y, TAU, 12, first_target=150, window=14)
