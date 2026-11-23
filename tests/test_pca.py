import numpy as np
import pytest

from tenor import pca as P


def _three_factor_panel(seed: int = 0, t: int = 5000, n: int = 8):
    """Data built from three known orthonormal loadings and factor variances 9, 4, 1."""
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.standard_normal((n, n)))
    load = P.orient(basis[:, :3])
    f = rng.standard_normal((t, 3))
    # make the factors exactly uncorrelated with exact variances 9, 4, 1
    f -= f.mean(axis=0)
    u, _, _ = np.linalg.svd(f, full_matrices=False)
    f = u * np.sqrt(t - 1) * np.array([3.0, 2.0, 1.0])
    return f @ load.T, f, load


def test_known_three_factors_are_recovered():
    x, f, load = _three_factor_panel()
    res = P.pca(x)
    np.testing.assert_allclose(res.eigenvalues[:3], [9.0, 4.0, 1.0], rtol=1e-10)
    np.testing.assert_allclose(res.eigenvalues[3:], 0.0, atol=1e-10)
    np.testing.assert_allclose(res.explained[:3], np.array([9.0, 4.0, 1.0]) / 14.0, rtol=1e-10)
    for j in range(3):
        # eigenvectors are defined up to sign
        dot = float(res.loadings[:, j] @ load[:, j])
        assert abs(abs(dot) - 1.0) < 1e-10
        s = np.sign(dot)
        np.testing.assert_allclose(P.scores(res, x)[:, j], s * f[:, j], atol=1e-8)


def test_three_factors_explain_everything_per_variable():
    x, _, _ = _three_factor_panel(seed=3)
    res = P.pca(x)
    np.testing.assert_allclose(P.communality(res, 3), 1.0, atol=1e-10)
    assert (P.communality(res, 1) < 1.0).all()
    np.testing.assert_allclose(P.communality(res, 0), 0.0)
    np.testing.assert_allclose(P.communality(res, 8), 1.0)


def test_eigenvalues_sum_to_total_variance_and_loadings_are_orthonormal():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((400, 6)) @ rng.standard_normal((6, 6))
    res = P.pca(x)
    assert res.eigenvalues.sum() == pytest.approx(x.var(axis=0, ddof=1).sum(), rel=1e-12)
    np.testing.assert_allclose(res.loadings.T @ res.loadings, np.eye(6), atol=1e-12)
    assert (np.diff(res.eigenvalues) <= 1e-12).all()
    assert res.explained.sum() == pytest.approx(1.0)


def test_reconstruction_from_all_components_is_exact():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((50, 5))
    res = P.pca(x)
    rebuilt = P.scores(res, x) @ res.loadings.T + res.mean
    np.testing.assert_allclose(rebuilt, x, atol=1e-12)


def test_scores_are_uncorrelated_with_variance_equal_to_eigenvalues():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((300, 4)) @ rng.standard_normal((4, 4))
    res = P.pca(x)
    cov = np.cov(P.scores(res, x), rowvar=False)
    np.testing.assert_allclose(cov, np.diag(res.eigenvalues), atol=1e-10)


def test_correlation_pca_eigenvalues_sum_to_number_of_variables():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((200, 7)) * np.arange(1, 8)
    res = P.pca(x, standardize=True)
    assert res.eigenvalues.sum() == pytest.approx(7.0)
    # rescaling a column does not change a correlation decomposition
    x2 = x.copy()
    x2[:, 2] *= 50.0
    np.testing.assert_allclose(P.pca(x2, standardize=True).eigenvalues, res.eigenvalues)


def test_two_variable_closed_form():
    # For a 2x2 covariance [[a, c], [c, b]] the eigenvalues are
    # (a+b)/2 +- sqrt(((a-b)/2)^2 + c^2).
    rng = np.random.default_rng(6)
    x = rng.standard_normal((1000, 2)) @ np.array([[1.0, 0.7], [0.0, 0.4]])
    s = np.cov(x, rowvar=False)
    a, b, c = s[0, 0], s[1, 1], s[0, 1]
    root = np.sqrt(((a - b) / 2) ** 2 + c**2)
    np.testing.assert_allclose(P.pca(x).eigenvalues, [(a + b) / 2 + root, (a + b) / 2 - root])


def test_perfectly_parallel_moves_have_one_factor_with_equal_loadings():
    rng = np.random.default_rng(7)
    shift = rng.standard_normal(500)
    x = np.tile(shift[:, None], (1, 5))
    res = P.pca(x)
    assert res.explained[0] == pytest.approx(1.0)
    np.testing.assert_allclose(res.loadings[:, 0], np.full(5, 1 / np.sqrt(5)), atol=1e-12)


def test_shift_invariance_and_scale_equivariance():
    rng = np.random.default_rng(8)
    x = rng.standard_normal((120, 4))
    base = P.pca(x)
    np.testing.assert_allclose(P.pca(x + 100.0).eigenvalues, base.eigenvalues, atol=1e-10)
    np.testing.assert_allclose(P.pca(3.0 * x).eigenvalues, 9.0 * base.eigenvalues, atol=1e-10)
    np.testing.assert_allclose(P.pca(3.0 * x).explained, base.explained, atol=1e-12)


def test_orientation_rules():
    load = np.array(
        [
            [-0.5, 0.6, 0.4, 0.1],
            [-0.5, 0.2, -0.5, -0.9],
            [-0.5, -0.2, -0.5, 0.2],
            [-0.5, -0.6, 0.4, 0.1],
        ]
    )
    out = P.orient(load)
    assert out[:, 0].sum() > 0
    assert out[-1, 1] > out[0, 1]
    assert out[2, 2] > 0.5 * (out[0, 2] + out[-1, 2])
    assert out[1, 3] > 0
    np.testing.assert_allclose(np.abs(out), np.abs(load))
    np.testing.assert_allclose(P.orient(out), out)


def test_rolling_matches_direct_computation():
    rng = np.random.default_rng(9)
    x = rng.standard_normal((300, 5)) @ rng.standard_normal((5, 5))
    ends, shares = P.rolling_explained(x, window=100, step=50, k=3)
    np.testing.assert_array_equal(ends, [99, 149, 199, 249, 299])
    for end, row in zip(ends, shares, strict=True):
        direct = P.pca(x[end - 99 : end + 1]).explained[:3]
        np.testing.assert_allclose(row, direct)


def test_input_validation():
    with pytest.raises(ValueError):
        P.pca(np.ones(5))
    with pytest.raises(ValueError):
        P.pca(np.ones((1, 3)))
    with pytest.raises(ValueError):
        P.pca(np.array([[1.0, np.nan], [2.0, 3.0]]))
    with pytest.raises(ValueError):
        P.pca(np.ones((5, 2)), standardize=True)
    with pytest.raises(ValueError):
        P.rolling_explained(np.ones((5, 2)), window=10)
