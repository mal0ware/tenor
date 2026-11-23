"""Numbers printed in the source papers, kept for side-by-side comparison.

Nothing here is computed. Each block names the table it was read from.
"""

from __future__ import annotations

#: Maturities (months) at which Diebold and Li (2006) report forecast results.
DL_MATURITIES: tuple[int, ...] = (3, 12, 36, 60, 120)

#: Diebold and Li (2006), Tables 4, 5 and 6: out-of-sample RMSE in percentage
#: points, forecasts for 1994:01 to 2000:12, by model, horizon (months) and
#: maturity (order of DL_MATURITIES). The Nelson-Siegel VAR(1) model is
#: reported at the 12-month horizon only.
DL_RMSE: dict[str, dict[int, tuple[float, ...]]] = {
    "ns_ar": {
        1: (0.176, 0.236, 0.279, 0.292, 0.260),
        6: (0.517, 0.669, 0.750, 0.777, 0.721),
        12: (0.739, 0.841, 0.918, 0.978, 0.981),
    },
    "rw": {
        1: (0.179, 0.241, 0.279, 0.276, 0.254),
        6: (0.605, 0.779, 0.879, 0.861, 0.758),
        12: (1.019, 1.197, 1.237, 1.191, 1.052),
    },
    "ar_yield": {
        1: (0.182, 0.239, 0.276, 0.276, 0.258),
        6: (0.584, 0.725, 0.801, 0.802, 0.755),
        12: (0.845, 0.970, 1.002, 1.007, 1.030),
    },
    "ns_var": {
        12: (1.102, 1.293, 1.393, 1.385, 1.279),
    },
}

#: Diebold and Li (2006), Table 7: Diebold-Mariano statistics of the
#: Nelson-Siegel AR(1) forecasts against the random walk. Negative favours
#: Nelson-Siegel. Only the 1- and 12-month horizons are printed.
DL_DM_VS_RW: dict[int, tuple[float, ...]] = {
    1: (-0.27, -0.64, -0.02, 0.97, 0.49),
    12: (-1.65, -2.04, -2.11, -1.61, -0.63),
}

#: Diebold and Li (2006), section 3.2: correlations of the estimated factors
#: with the empirical level (10-year yield), slope (10-year minus 3-month) and
#: curvature (twice 2-year minus 3-month minus 10-year).
DL_FACTOR_CORRELATIONS: dict[str, float] = {"level": 0.97, "slope": -0.99, "curvature": 0.99}

#: Diebold and Li (2006), section 2.2: the empirical slope and curvature
#: written in terms of the factors, with lambda = 0.0609.
#: y(120) - y(3) = -0.78*b2 + 0.06*b3 ; 2*y(24) - y(3) - y(120) = 0.00053*b2 + 0.37*b3.
DL_SLOPE_COEFS: tuple[float, float] = (-0.78, 0.06)
DL_CURVATURE_COEFS: tuple[float, float] = (0.00053, 0.37)

#: Litterman and Scheinkman (1991), Table 2: implied zeros, weekly excess
#: returns, January 1984 to June 1988. Per maturity (years): total variance
#: explained by three factors, then the split of that explained variance
#: across factors 1, 2 and 3. All in percent.
LS_TABLE2: dict[float, tuple[float, float, float, float]] = {
    0.5: (99.5, 79.5, 17.2, 3.3),
    1.0: (99.4, 89.7, 10.1, 0.2),
    2.0: (98.2, 93.4, 2.4, 4.2),
    5.0: (98.8, 98.2, 1.1, 0.7),
    8.0: (98.7, 95.4, 4.6, 0.0),
    10.0: (98.8, 92.9, 6.9, 0.2),
    14.0: (98.4, 86.2, 11.5, 2.2),
    18.0: (95.3, 80.5, 14.3, 5.2),
}

#: Litterman and Scheinkman (1991), Table 2, "Average" row.
LS_TABLE2_AVERAGE: tuple[float, float, float, float] = (98.4, 89.5, 8.5, 2.0)
