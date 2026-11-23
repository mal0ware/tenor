"""Principal components of a panel of yields or yield changes.

The decomposition is the eigendecomposition of the sample covariance matrix,
S = Q L Q', with eigenvalues in descending order. Diebold and Li (2006,
section 3.4, competitor 10) state it in this form. The share of variance
carried by component j is L_j / trace(S).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class PCAResult:
    """Eigendecomposition of a sample covariance (or correlation) matrix.

    Attributes
    ----------
    eigenvalues:
        Variances of the components, descending. Shape (n,).
    loadings:
        Orthonormal eigenvectors, one per column, in the same order. Shape (n, n).
    explained:
        eigenvalues / sum(eigenvalues). Shape (n,).
    mean:
        Column means removed before the decomposition. Shape (n,).
    scale:
        Column scale divided out before the decomposition: ones for a
        covariance decomposition, sample standard deviations for a correlation
        decomposition. Shape (n,).

    """

    eigenvalues: FloatArray
    loadings: FloatArray
    explained: FloatArray
    mean: FloatArray
    scale: FloatArray


def orient(loadings: FloatArray) -> FloatArray:
    """Fix the sign of each eigenvector so results are reproducible.

    Eigenvectors are defined up to sign. The convention, for columns ordered
    by maturity:

    - component 1 (level): the loadings sum to a positive number;
    - component 2 (slope): the longest maturity loads higher than the shortest;
    - component 3 (curvature): the middle loads higher than the average of the ends;
    - any later component: its largest loading in absolute value is positive.
    """
    out = np.array(loadings, dtype=np.float64, copy=True)
    n = out.shape[0]
    for j in range(out.shape[1]):
        q = out[:, j]
        if j == 0:
            key = float(q.sum())
        elif j == 1:
            key = float(q[-1] - q[0])
        elif j == 2:
            key = float(q[n // 2] - 0.5 * (q[0] + q[-1]))
        else:
            key = float(q[int(np.argmax(np.abs(q)))])
        if key < 0.0:
            out[:, j] = -q
    return out


def pca(x: FloatArray, *, standardize: bool = False) -> PCAResult:
    """Principal components of the columns of ``x``.

    Parameters
    ----------
    x:
        Observations in rows, variables (maturities) in columns. Shape (T, n).
    standardize:
        If true, decompose the correlation matrix instead of the covariance
        matrix.

    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("x must be two-dimensional")
    t, n = x.shape
    if t < 2:
        raise ValueError("need at least two observations")
    if not np.isfinite(x).all():
        raise ValueError("x contains missing or infinite values")
    mean = x.mean(axis=0)
    centred = x - mean
    if standardize:
        scale = centred.std(axis=0, ddof=1)
        if (scale == 0.0).any():
            raise ValueError("a column has zero variance; cannot standardize")
    else:
        scale = np.ones(n, dtype=np.float64)
    z = centred / scale
    cov = (z.T @ z) / (t - 1)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    values = np.clip(values[order], 0.0, None)
    vectors = orient(vectors[:, order])
    total = float(values.sum())
    explained = values / total if total > 0.0 else np.zeros_like(values)
    return PCAResult(
        eigenvalues=values, loadings=vectors, explained=explained, mean=mean, scale=scale
    )


def scores(result: PCAResult, x: FloatArray) -> FloatArray:
    """Component scores x_it = q_i' (y_t - mean) for every observation."""
    z = (np.asarray(x, dtype=np.float64) - result.mean) / result.scale
    out: FloatArray = z @ result.loadings
    return out


def communality(result: PCAResult, k: int) -> FloatArray:
    """Share of each variable's variance explained by the first ``k`` components.

    For variable i this is sum_{j<k} L_j q_ij^2 / sum_j L_j q_ij^2. It is the
    per-maturity counterpart of the total share, comparable in spirit with the
    "total variance explained" column of Litterman and Scheinkman (1991,
    Table 2), which comes from a factor model rather than from principal
    components.
    """
    if not 0 <= k <= result.eigenvalues.size:
        raise ValueError("k out of range")
    weighted = result.loadings**2 * result.eigenvalues
    total = weighted.sum(axis=1)
    out: FloatArray = weighted[:, :k].sum(axis=1) / total
    return out


def rolling_explained(
    x: FloatArray, window: int, step: int = 1, k: int = 3, *, standardize: bool = False
) -> tuple[npt.NDArray[np.int64], FloatArray]:
    """Variance shares of the first ``k`` components over rolling windows.

    Returns
    -------
    ends:
        Row index of the last observation in each window.
    shares:
        Shape (number of windows, k). Row i is from rows ends[i]-window+1 .. ends[i].

    """
    x = np.asarray(x, dtype=np.float64)
    t = x.shape[0]
    if window < 2 or window > t:
        raise ValueError("window must be between 2 and the number of rows")
    if step < 1:
        raise ValueError("step must be positive")
    ends = np.arange(window - 1, t, step, dtype=np.int64)
    shares = np.empty((ends.size, k), dtype=np.float64)
    for i, end in enumerate(ends):
        res = pca(x[end - window + 1 : end + 1], standardize=standardize)
        shares[i] = res.explained[:k]
    return ends, shares
