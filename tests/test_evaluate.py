import math

import numpy as np
import pytest
from scipy import stats

from tenor import evaluate as E
from tenor import size_sim as S


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


def test_hln_factor_by_hand():
    # T = 84, h = 12: (84 + 1 - 24 + 12*11/84) / 84
    assert E.hln_factor(84, 12) == pytest.approx(math.sqrt((61 + 132 / 84) / 84))
    # h = 1: (T - 1) / T, the usual small-sample variance correction
    assert E.hln_factor(50, 1) == pytest.approx(math.sqrt(49 / 50))
    assert E.hln_factor(394, 12) < E.hln_factor(394, 6) < E.hln_factor(394, 1) < 1.0
    with pytest.raises(ValueError):
        E.hln_factor(1, 1)
    with pytest.raises(ValueError):
        E.hln_factor(10, 10)  # (T - h)(T - h + 1) / T^2 is zero at h = T


def test_dm_hln_scales_the_statistic_and_uses_student_t():
    rng = np.random.default_rng(5)
    e1 = rng.standard_normal(84)
    e2 = 1.1 * rng.standard_normal(84)
    raw = E.diebold_mariano(e1, e2, h=6)
    hln = E.dm_hln(e1, e2, h=6)
    assert hln.stat == pytest.approx(raw.stat * E.hln_factor(84, 6))
    assert hln.pvalue == pytest.approx(2 * stats.t.sf(abs(hln.stat), 83))
    assert abs(hln.stat) < abs(raw.stat)
    assert hln.pvalue > raw.pvalue
    same = E.dm_hln(e1, e1, h=6)
    assert same.stat == 0.0 and same.pvalue == 1.0


def test_clark_west_by_hand():
    rng = np.random.default_rng(6)
    actual = rng.standard_normal(60)
    f_small = np.zeros(60)
    f_large = 0.3 * rng.standard_normal(60)
    e_s, e_l = actual - f_small, actual - f_large
    f = e_s**2 - (e_l**2 - (f_small - f_large) ** 2)
    expected = f.mean() / math.sqrt(f.var() / 60) * E.hln_factor(60, 1)
    res = E.clark_west(e_s, e_l, f_small, f_large, h=1)
    assert res.stat == pytest.approx(expected)
    assert res.pvalue == pytest.approx(stats.t.sf(expected, 59))
    assert res.mspe_small == pytest.approx((e_s**2).mean())
    assert res.mspe_adjusted == pytest.approx((e_l**2).mean() - (f_large**2).mean())
    assert res.n == 60


def test_clark_west_adjustment_removes_pure_noise():
    # The large model adds noise that is independent of the target. Its mean
    # squared error is higher by the variance of that noise, and the adjusted
    # differential has mean zero.
    rng = np.random.default_rng(7)
    actual = rng.standard_normal(20000)
    noise = 0.5 * rng.standard_normal(20000)
    res = E.clark_west(actual, actual - noise, np.zeros(20000), noise, h=1)
    assert res.mspe_large - res.mspe_small == pytest.approx(0.25, abs=0.02)
    assert res.mspe_adjusted == pytest.approx(res.mspe_small, abs=0.02)
    assert abs(res.stat) < 3.0


def test_clark_west_rejects_when_the_large_model_has_signal():
    rng = np.random.default_rng(8)
    signal = rng.standard_normal(400)
    actual = 0.5 * signal + rng.standard_normal(400)
    res = E.clark_west(actual, actual - 0.5 * signal, np.zeros(400), 0.5 * signal, h=1)
    assert res.stat > 4.0
    assert res.pvalue < 1e-4
    same = E.clark_west(actual, actual, np.zeros(400), np.zeros(400), h=3)
    assert same.stat == 0.0 and same.pvalue == 0.5
    with pytest.raises(ValueError):
        E.clark_west([1.0, 2.0], [1.0], [0.0, 0.0], [0.0, 0.0])


def test_row_wise_statistics_match_the_scalar_ones():
    rng = np.random.default_rng(9)
    d = rng.standard_normal((6, 90)).cumsum(axis=1) * 0.1 + rng.standard_normal((6, 90))
    for h in (1, 6, 12):
        for window in ("uniform", "bartlett"):
            rows = S.dm_stats_rows(d, h, window)
            for i in range(6):
                lrv = E.long_run_variance(d[i], h - 1, window)
                if lrv > 0:
                    assert rows[i] == pytest.approx(d[i].mean() / math.sqrt(lrv / 90))
                else:
                    assert math.isinf(rows[i])


def test_recursive_ar1_forecasts_match_the_loop():
    from tenor import forecast as F

    rng = np.random.default_rng(10)
    x = rng.standard_normal((3, 80)).cumsum(axis=1)
    for h in (1, 6):
        fast = S.recursive_ar1_forecasts(x, h, 40)
        assert fast.shape == (3, 40)
        for r in range(3):
            for j, target in enumerate(range(40, 80)):
                origin = target - h
                slow = F.direct_ar1_forecast(x[r, : origin + 1], h)[0]
                assert fast[r, j] == pytest.approx(slow, abs=1e-9)


@pytest.mark.parametrize("h", [6, 12])
def test_dm_size_at_longer_horizons(h):
    # 394 forecasts is the length of the long sample. With the Harvey-
    # Leybourne-Newbold correction the test is close to its nominal 5 percent;
    # the Bartlett window with h - 1 lags rejects too often.
    rates = S.dm_size(h, 394, reps=4000, seed=11 + h)
    assert 0.035 < rates["dm_hln_t"] < 0.075
    assert rates["dm_hln_t"] <= rates["dm_uniform_normal"]
    assert rates["dm_bartlett_normal"] > rates["dm_hln_t"] + 0.02


@pytest.mark.parametrize("h", [6, 12])
def test_dm_size_is_too_large_on_the_paper_window(h):
    # 84 forecasts is the paper's window. The correction helps and does not
    # cure the problem: this documents that the size is still above nominal.
    rates = S.dm_size(h, 84, reps=4000, seed=21 + h)
    assert rates["dm_hln_t"] < rates["dm_uniform_normal"]
    assert 0.05 < rates["dm_hln_t"] < 0.15
    assert rates["dm_uniform_normal"] > 0.08


def test_dm_misleads_for_nested_models_and_clark_west_does_not():
    # A random walk forecast by "no change" and by a recursively estimated
    # AR(1). Equal accuracy holds in population. One step ahead with 394
    # forecasts, Diebold-Mariano declares the random walk better far more often
    # than 5 percent, and Clark-West stays at or below its nominal size.
    rates = S.nested_size(1, 394, reps=4000, seed=31)
    assert rates["ar1_has_higher_mse"] > 0.8
    assert rates["dm_hln_t_favouring_random_walk"] > 0.12
    assert rates["cw_one_sided"] < 0.07


def test_size_table_is_seeded_and_complete():
    a = S.size_table(reps=200)
    b = S.size_table(reps=200)
    assert a.equals(b)
    assert set(a["horizon_months"]) == {1, 6, 12}
    assert set(a["forecasts"]) == {84, 394}
    assert a["rate_pct"].between(0, 100).all()
    assert len(a) == 2 * 3 * (3 + 4)
