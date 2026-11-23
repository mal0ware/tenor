import math

import numpy as np
import pytest

from tenor import evaluate as E


def test_rmse_decomposition():
    rng = np.random.default_rng(0)
    e = rng.normal(0.3, 1.2, 500)
    s = E.error_stats(e)
    assert s.n == 500
    assert s.rmse**2 == pytest.approx(s.mean**2 + s.sd**2, rel=1e-12)
    assert E.rmse([3.0, 4.0]) == pytest.approx(math.sqrt(12.5))


def test_autocorr_known_values():
    # x = 1, 2, 3, 4: mean 2.5, deviations -1.5, -0.5, 0.5, 1.5, sum of squares 5.
    x = [1.0, 2.0, 3.0, 4.0]
    assert E.autocorr(x, 0) == 1.0
    assert E.autocorr(x, 1) == pytest.approx((0.75 + -0.25 + 0.75) / 5.0)
    assert E.autocorr(x, 2) == pytest.approx((-0.75 - 0.75) / 5.0)
    assert math.isnan(E.autocorr([1.0, 1.0, 1.0], 1))
    assert math.isnan(E.autocorr([1.0, 2.0], 5))


def test_autocorr_of_ar1_is_near_phi():
    rng = np.random.default_rng(1)
    x = np.empty(50000)
    x[0] = 0.0
    eps = rng.standard_normal(50000)
    for t in range(1, x.size):
        x[t] = 0.8 * x[t - 1] + eps[t]
    assert E.autocorr(x, 1) == pytest.approx(0.8, abs=0.02)
    assert E.autocorr(x, 3) == pytest.approx(0.8**3, abs=0.03)


def test_long_run_variance_by_hand():
    d = np.array([1.0, -1.0, 2.0, 0.0, -2.0])
    c = d - d.mean()
    g0 = (c @ c) / 5
    g1 = (c[1:] @ c[:-1]) / 5
    g2 = (c[2:] @ c[:-2]) / 5
    assert E.long_run_variance(d, 0) == pytest.approx(g0)
    assert E.long_run_variance(d, 2) == pytest.approx(g0 + 2 * g1 + 2 * g2)
    assert E.long_run_variance(d, 2, "bartlett") == pytest.approx(
        g0 + 2 * (2 / 3) * g1 + 2 * (1 / 3) * g2
    )


def test_dm_one_step_is_a_t_statistic_on_the_loss_differential():
    rng = np.random.default_rng(2)
    e1 = rng.standard_normal(300)
    e2 = 1.2 * rng.standard_normal(300)
    d = e1**2 - e2**2
    expected = d.mean() / math.sqrt(d.var() / d.size)
    res = E.diebold_mariano(e1, e2, h=1)
    assert res.stat == pytest.approx(expected)
    assert res.n == 300
    assert res.pvalue == pytest.approx(math.erfc(abs(expected) / math.sqrt(2)))
    # with no lags the two windows coincide
    assert E.diebold_mariano(e1, e2, h=1, window="bartlett").stat == pytest.approx(expected)


def test_dm_is_antisymmetric_and_signed_towards_the_better_forecast():
    rng = np.random.default_rng(3)
    good = 0.5 * rng.standard_normal(400)
    bad = 2.0 * rng.standard_normal(400)
    a = E.diebold_mariano(good, bad, h=3)
    b = E.diebold_mariano(bad, good, h=3)
    assert a.stat == pytest.approx(-b.stat)
    assert a.stat < -5
    assert a.pvalue < 1e-6


def test_dm_identical_forecasts_give_zero_and_no_rejection():
    e = np.linspace(-1, 1, 50)
    res = E.diebold_mariano(e, e, h=4)
    assert res.stat == 0.0
    assert res.pvalue == 1.0


def test_dm_negative_variance_estimate_is_a_rejection():
    # An alternating differential has a strongly negative first autocovariance,
    # so the uniform window gives a negative variance at h = 2.
    e1 = np.tile([2.0, 0.0], 20)
    e2 = np.tile([0.0, 1.0], 20)
    d = e1**2 - e2**2
    assert E.long_run_variance(d, 1) < 0
    res = E.diebold_mariano(e1, e2, h=2)
    assert math.isinf(res.stat) and res.stat > 0
    assert res.pvalue == 0.0
    bart = E.diebold_mariano(e1, e2, h=2, window="bartlett")
    assert math.isfinite(bart.stat)


def test_dm_size_under_the_null_one_step():
    rng = np.random.default_rng(4)
    reject = 0
    trials = 2000
    for _ in range(trials):
        e1 = rng.standard_normal(200)
        e2 = rng.standard_normal(200)
        if abs(E.diebold_mariano(e1, e2, h=1).stat) > 1.959964:
            reject += 1
    # nominal 5 percent; binomial s.e. is about 0.5 percent
    assert 0.03 < reject / trials < 0.07


def test_dm_validation():
    with pytest.raises(ValueError):
        E.diebold_mariano([1.0, 2.0], [1.0], h=1)
    with pytest.raises(ValueError):
        E.diebold_mariano([1.0, 2.0], [1.0, 2.0], h=0)
    with pytest.raises(ValueError):
        E.long_run_variance([1.0, 2.0], 1, window="parzen")
    with pytest.raises(ValueError):
        E.error_stats([])
