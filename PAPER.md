# Sources, formulas and comparison

Every number below marked "here" was produced by `scripts/reproduce.py` and is in `results/`. Numbers marked "paper" were read from the printed tables and are stored in `src/tenor/published.py`.

## Sources read

1. Diebold, F. X. and Li, C. (2006). "Forecasting the term structure of government bond yields." Journal of Econometrics 130(2), 337-364. Read in full from the authors' copy of the journal version at `sas.upenn.edu/~fdiebold/papers/paper49/Diebold-Li.pdf`. The paper does not number its equations, so formulas are cited by section.
2. Litterman, R. and Scheinkman, J. (1991). "Common Factors Affecting Bond Returns." Journal of Fixed Income 1(1), 54-61. Read from a scanned copy at `math.nyu.edu/~avellane/Litterman1991.pdf`. It has no numbered equations and no formula for the estimator. Table 2 was read from the page image.
3. Diebold, F. X. and Mariano, R. S. (1995). "Comparing Predictive Accuracy." Journal of Business and Economic Statistics 13(3), 253-263 (reprint pages 134-144 in the 2002 anniversary issue). Section 1.1 read from the authors' copy at `sas.upenn.edu/~fdiebold/papers/paper68/pa.dm.pdf`. The text layer of that scan garbles the displayed formula, so the statistic below is written from the surrounding prose, which is legible.

Nelson and Siegel (1987) was not read. The curve is implemented as Diebold and Li state it.

## Formulas implemented

### Principal components (`src/tenor/pca.py`)

Diebold and Li, section 3.4, competitor 10: decompose the covariance matrix of the yields as

    S = Q L Q'

where L is diagonal with the eigenvalues and the columns of Q are the eigenvectors. The principal components are

    x_it = q_i' y_t,  i = 1, 2, 3.

The share of variance carried by component j is `L_j / sum(L)`. For one maturity i, the share explained by the first k components is `sum_{j<=k} L_j q_ij^2 / sum_j L_j q_ij^2`. That per-maturity share is a standard identity and is not taken from either paper.

Eigenvectors have no natural sign. This code orients them so that component 1 has a positive sum, component 2 rises from the shortest to the longest maturity, and component 3 is higher in the middle than at the ends.

### Nelson-Siegel curve (`src/tenor/nelson_siegel.py`)

Diebold and Li, section 2.2, with maturity `tau` in months:

    y_t(tau) = b1_t + b2_t * (1 - exp(-lam*tau)) / (lam*tau)
                    + b3_t * ( (1 - exp(-lam*tau)) / (lam*tau) - exp(-lam*tau) )

Forward curve, same section:

    f_t(tau) = b1_t + b2_t * exp(-lam*tau) + b3_t * lam*tau * exp(-lam*tau)

Section 2.1: `y_t(tau) = (1/tau) * integral_0^tau f_t(u) du`. A test checks this numerically.

Section 3.2 and footnote 5: `lam` is fixed at 0.0609 for every date and the three betas come from ordinary least squares on each date's cross-section.

Section 2.2 also gives `y(infinity) = b1`, `y(0) = b1 + b2`, and the mapping to the original Nelson-Siegel form (`b1`, `b2 + b3`, `b3`). All three are tested.

### Forecasts (`src/tenor/forecast.py`)

Section 3.3, Nelson-Siegel with AR(1) factors:

    yhat_{t+h|t}(tau) = b1hat_{t+h|t} + b2hat_{t+h|t} * slope(tau) + b3hat_{t+h|t} * curv(tau)
    bihat_{t+h|t}     = c_i + g_i * bihat_t

with `c_i`, `g_i` from regressing `bihat_t` on an intercept and `bihat_{t-h}`. Footnote 11 says this is a direct regression of the factor at t+h on the factor at t, and that every competitor is estimated the same way.

Section 3.3, VAR(1) factors: `bhat_{t+h|t} = c + G * bhat_t`.

Section 3.4, competitor 1, random walk: `yhat_{t+h|t}(tau) = y_t(tau)`.

Section 3.4, competitor 5, AR(1) on yield levels: `yhat_{t+h|t}(tau) = c(tau) + g(tau) * y_t(tau)`.

Section 3.4: forecast errors are `y_{t+h}(tau) - yhat_{t+h|t}(tau)`, and the object forecast is the observed yield, not the fitted one. Estimation is recursive, using data from the start of the sample to the date the forecast is made.

The paper's other competitors (slope regression, Fama-Bliss and Cochrane-Piazzesi forward regressions, VARs and error-correction models on yields, regression on principal components) are not implemented.

### Diebold-Mariano test (`src/tenor/evaluate.py`)

Diebold and Mariano, section 1.1, with squared-error loss, `d_t = e1_t^2 - e2_t^2`:

    S1 = dbar / sqrt( V / T )
    V  = sum over |k| <= h-1 of gamma_d(k)
    gamma_d(k) = (1/T) * sum_t (d_t - dbar)(d_{t-|k|} - dbar)

The lag window is uniform with truncation `h - 1`, which the paper motivates by the fact that optimal h-step errors are at most (h-1)-dependent. S1 is compared with the standard normal. If V comes out negative the paper treats it as zero and rejects; the code returns an infinite statistic in that case. It did not happen in any of the 360 comparisons here. A Bartlett window with weights `1 - k/h` is also reported, since the paper names it as the way to guarantee a positive variance.

Negative values favour the first forecast. In the tables the first forecast is the model and the second is the random walk, the same orientation as Diebold and Li's Table 7.

## Checks against printed values

| Statement in Diebold and Li | Paper | Here |
|---|---|---|
| `y(120) - y(3)` in terms of the factors (section 2.2) | -0.78 b2 + 0.06 b3 | -0.777 b2 + 0.055 b3 |
| `2 y(24) - y(3) - y(120)` (section 2.2) | 0.00053 b2 + 0.37 b3 | 0.00038 b2 + 0.370 b3 |
| Maturity at which lam = 0.0609 maximises the curvature loading (section 3.2) | "exactly 30 months" | 29.45 months |
| Correlation of b1 with the 10-year yield | 0.97 | 0.969 |
| Correlation of b2 with the 10-year minus 3-month slope | -0.99 | -0.995 |
| Correlation of b3 with the curvature measure | 0.99 | 0.998 |

Two of these do not match.

The curvature loading `(1 - e^-x)/x - e^-x` peaks at `x = 1.7933`, so `lam = 0.0609` puts the peak at 29.45 months, and the value that puts it at 30 months is 0.0598. The paper's 0.0609 is used throughout because it is the value the paper fixes.

The 0.00053 coefficient computes to 0.00038 at 0.0609. That coefficient is the difference of three numbers near 0.5 and moves fast with `lam`: it is 0.00281 at 0.0605 and -0.00023 at 0.0610. The printed value corresponds to a `lam` of about 0.06088, so the paper probably computed with an unrounded value. It changes nothing downstream.

The correlations are on different data (par yields 1990 to 2026 here, Fama-Bliss zero yields 1985 to 2000 in the paper) and still agree to two decimals.

## Claim 1: three factors explain almost all of the variation

Litterman and Scheinkman, Table 2, report that three factors explain 98.4% of the variance of returns on average across eight maturities from 6 months to 18 years, with a minimum of 95.3% at 18 years, and that the first factor accounts for 89.5% of the explained variance, the second 8.5% and the third 2.0%.

Here, from `results/pca_variance.csv`:

| Treatment | Observations | PC1 % | PC2 % | PC3 % | First three % |
|---|---|---|---|---|---|
| daily changes, covariance | 9,210 | 79.41 | 12.69 | 4.06 | 96.16 |
| daily changes, correlation | 9,210 | 74.34 | 16.72 | 4.45 | 95.50 |
| weekly changes, covariance | 1,921 | 80.97 | 12.82 | 3.60 | 97.38 |
| monthly changes, covariance | 441 | 81.17 | 14.67 | 2.66 | 98.50 |
| monthly levels, covariance | 442 | 95.43 | 4.27 | 0.26 | 99.95 |
| daily changes, with 20y and 30y, from 2006-02-10 | 5,182 | 75.57 | 13.68 | 6.27 | 95.53 |

Per maturity, daily changes (`results/pca_by_maturity.csv`): three components explain 95.8% at 3 months, 87.5% at 6 months, 91.4% at 1 year, and between 96.5% and 98.7% from 2 to 10 years.

Over rolling windows of 1,250 trading days the three-component share ranges from 94.1% (window ending 2018-12-06) to 97.4% (ending 2006-08-31). The first component alone ranges from 70.0% to 90.0%.

The headline in the design was "about 99%". On daily changes it is 96.2%. Reasons the numbers differ from Litterman and Scheinkman, most likely first:

1. Their figure is from a factor model with a specific variance for each maturity, fitted to weekly observations. Mine is plain principal components. The two are different estimators.
2. Frequency. The share rises from 96.2% daily to 97.4% weekly to 98.5% monthly on the same data. Their weekly figure of 98.4% sits one point above my weekly figure.
3. The short end. Bill yields at 6 months and 1 year have day-to-day moves the three components do not capture. Their panel is fitted zero-coupon prices, which are smoother than quoted constant-maturity bill yields.
4. Their sample is January 1984 to June 1988. Mine is 1990 to 2026 and includes the years after 2008 when bill yields sat near zero.
5. They analyse excess returns on zeros; I analyse changes in par yields.

Their first-factor share of 89.5% is also higher than the 79.4% here. The first component of daily par yield changes is not a parallel shift: a one-standard-deviation move is 2.0 basis points at 3 months and 5.3 to 6.0 basis points from 2 to 10 years.

## Claim 2: Nelson-Siegel forecasts beat the random walk at long horizons

Diebold and Li, section 3.4: at 1 month their forecasts are "only very slightly better" than the random walk and no Diebold-Mariano statistic is significant. At 12 months their forecasts "outperform those of all competitors at all maturities".

### Main sample: targets 1999-01 to 2026-10, 334 forecasts per horizon

The first 108 months (1990-01 to 1998-12) are the initial estimation window, the same length as the paper's (1985:01 to 1993:12). From `results/forecast_vs_random_walk.csv`, Nelson-Siegel AR(1) against the random walk, RMSE in percentage points:

| Horizon | Maturity | RMSE model | RMSE random walk | Ratio | DM (uniform) | DM (Bartlett) |
|---|---|---|---|---|---|---|
| 1 | 3m | 0.227 | 0.214 | 1.062 | 1.85 | 1.85 |
| 1 | 1y | 0.230 | 0.211 | 1.092 | 3.31 | 3.31 |
| 1 | 3y | 0.278 | 0.260 | 1.069 | 3.19 | 3.19 |
| 1 | 5y | 0.283 | 0.270 | 1.049 | 2.58 | 2.58 |
| 1 | 10y | 0.268 | 0.258 | 1.040 | 2.73 | 2.73 |
| 6 | 3m | 0.898 | 0.834 | 1.077 | 2.04 | 2.61 |
| 6 | 1y | 0.894 | 0.794 | 1.126 | 2.16 | 2.77 |
| 6 | 3y | 0.827 | 0.741 | 1.115 | 2.47 | 2.98 |
| 6 | 5y | 0.753 | 0.691 | 1.089 | 2.07 | 2.42 |
| 6 | 10y | 0.650 | 0.613 | 1.059 | 2.29 | 2.54 |
| 12 | 3m | 1.568 | 1.459 | 1.074 | 0.98 | 1.16 |
| 12 | 1y | 1.513 | 1.365 | 1.108 | 1.23 | 1.49 |
| 12 | 3y | 1.298 | 1.136 | 1.142 | 1.52 | 1.91 |
| 12 | 5y | 1.126 | 0.989 | 1.139 | 1.59 | 2.00 |
| 12 | 10y | 0.913 | 0.824 | 1.107 | 1.92 | 2.28 |

Across all eight maturities the ratio is above 1 in all 24 cells, from 1.028 to 1.142. With the uniform window the random walk is better at the 5% level in 7 of 8 maturities at 1 month, 7 of 8 at 6 months and none at 12 months. Positive statistics favour the random walk.

The claim does not replicate on this sample. The paper's 12-month ratios were 0.70 to 0.93.

The other models, full sample, range of RMSE ratios to the random walk over eight maturities:

| Model | 1 month | 6 months | 12 months |
|---|---|---|---|
| Nelson-Siegel, AR(1) factors | 1.028 to 1.106 | 1.059 to 1.126 | 1.074 to 1.142 |
| Nelson-Siegel, VAR(1) factors | 0.937 to 1.036 | 0.926 to 1.104 | 1.030 to 1.237 |
| AR(1) on yield levels | 1.004 to 1.009 | 1.022 to 1.061 | 1.069 to 1.152 |

The VAR(1) version has ratios below 1 at 3 and 6 months of maturity for the 1- and 6-month horizons (0.937, 0.975, 0.926, 0.962). None of those is significant; the largest statistic in absolute value is -1.46. The paper expected the VAR to be the worse of the two, and at 12 months in its own table it was.

### By period of the target month

12-month horizon, Nelson-Siegel AR(1), ratio to the random walk (`results/forecast_vs_random_walk.csv`, plotted in `results/fig_forecast_subperiods.png`):

| Targets | 3m | 1y | 3y | 5y | 10y |
|---|---|---|---|---|---|
| 1999 to 2008 | 0.994 | 1.036 | 1.067 | 1.070 | 1.091 |
| 2009 to 2017 | 2.063 | 1.933 | 2.171 | 1.698 | 1.282 |
| 2018 to 2026 | 1.045 | 1.067 | 1.008 | 0.995 | 1.016 |

Most of the loss is in 2009 to 2017. Short yields sat near zero, the random walk's 12-month RMSE at 3 months fell to 0.50, and the autoregression kept forecasting a return towards the historical mean: its mean error at 12 months was between -0.48 and -0.77 percentage points at every maturity. Outside that period the model is close to the random walk and still not ahead of it.

The three periods were fixed before the run as three roughly equal slices of the target dates.

### The paper's own window: targets 1994-01 to 2000-12, 84 forecasts

`results/forecast_paper_window.csv` sets my numbers beside the printed ones. The Treasury's files start in January 1990, so estimation here starts in 1990, five years later than the paper's 1985. For the first 12-month forecast the model has 37 months of data and 25 regression pairs.

| Horizon | Maturity | Random walk RMSE here | paper | NS AR(1) RMSE here | paper | Ratio here | paper |
|---|---|---|---|---|---|---|---|
| 1 | 3m | 0.192 | 0.179 | 0.187 | 0.176 | 0.970 | 0.983 |
| 1 | 1y | 0.247 | 0.241 | 0.263 | 0.236 | 1.065 | 0.979 |
| 1 | 3y | 0.285 | 0.279 | 0.293 | 0.279 | 1.027 | 1.000 |
| 1 | 5y | 0.288 | 0.276 | 0.287 | 0.292 | 0.997 | 1.058 |
| 1 | 10y | 0.262 | 0.254 | 0.264 | 0.260 | 1.011 | 1.024 |
| 12 | 3m | 1.025 | 1.019 | 1.894 | 0.739 | 1.848 | 0.725 |
| 12 | 1y | 1.224 | 1.197 | 2.002 | 0.841 | 1.636 | 0.703 |
| 12 | 3y | 1.285 | 1.237 | 1.803 | 0.918 | 1.402 | 0.742 |
| 12 | 5y | 1.226 | 1.191 | 1.621 | 0.978 | 1.322 | 0.821 |
| 12 | 10y | 1.098 | 1.052 | 1.425 | 0.981 | 1.298 | 0.933 |

The random-walk RMSE is within 0.05 of the paper's at every maturity and horizon, including the 6-month rows not shown here. That says par yields and Fama-Bliss zero yields moved almost identically over these seven years, and that the dates and error definitions line up.

The model rows are close at 1 month (within 0.03) and far off at 6 and 12 months. I read this as a consequence of the short estimation history and not as evidence about the paper. Yields fell through most of 1990 to 1993, and an autoregression fitted to three or four years of a falling series extrapolates badly into 1994, when yields rose. The paper's regressions had five more years, including the 1985 to 1989 cycle. This run cannot confirm or contradict Table 6.

A faithful rerun of the paper's window needs par yields for 1985 to 1989. The Treasury's download starts in 1990. The Federal Reserve's H.15 release, also served by FRED, publishes constant-maturity series for earlier years. It was not used because the design named the Treasury as the only source, and I did not check its coverage for all eight maturities.

## Differences from the papers

| Item | Paper | Here | Why |
|---|---|---|---|
| Yields | Unsmoothed Fama-Bliss zero-coupon yields from CRSP | Treasury constant-maturity par yields, bond-equivalent basis | CRSP is licensed. The par curve is free. |
| Maturities | 17, from 3 to 120 months | 8: 3, 6, 12, 24, 36, 60, 84, 120 months | What the Treasury has published every day since 1990. |
| Sample | 1985:01 to 2000:12 | 1990-01 to 2026-10 | The Treasury files start in 1990. |
| Forecast targets | 1994:01 to 2000:12 | 1999-01 to 2026-10 for the main results | 108 months of initial estimation, as in the paper. |
| Month end | Last trading day's bid-ask average price | Last day in the month with a published curve | Same intent. |
| Factor method (Litterman-Scheinkman) | Factor model with specific variances on weekly excess returns of fitted zeros | Principal components of daily par yield changes | Par yield changes are what the data gives. The paper states no estimator formula to copy. |
| Which targets count | "beginning in 1994:1 and extending through 2000:12" | Every horizon is scored on the same target months | The paper does not say whether the dates are origins or targets. Reading them as targets gives 84 forecasts per horizon. |
| DM variance | Not stated in Diebold-Li | Uniform window, h-1 lags, per Diebold-Mariano; Bartlett also reported | The 1995 paper's recommendation. |

Four days were dropped for missing values: 2008-12-10, 2008-12-18 and 2008-12-24, when no 3-month yield was published, and 2010-10-11, a row with no values.

## Variants tried

Everything that was run is in the committed tables. Nothing was dropped.

- Principal components: six treatments, all listed under Claim 1. The first, daily changes with the covariance matrix, was chosen as the headline before the run because the design says "daily".
- Forecast models: four (Nelson-Siegel AR, Nelson-Siegel VAR, AR on yield levels, random walk).
- Horizons: three (1, 6, 12 months).
- Evaluation samples: five (full, three sub-periods, the paper's window).
- Diebold-Mariano windows: two.
- Decay `lam`: one value, 0.0609. No other value was tried, including the 0.0598 that would put the curvature peak at 30 months.
- Initial window: one length, 108 months.
- Rolling window: one length, 1,250 trading days.

The settings in `tenor.pipeline.Config` were written before the first run on the Treasury data and were not changed after it. The only edit after seeing results was the position of a caption on one chart.

That is 360 model-against-random-walk comparisons in one table (3 models, 3 horizons, 8 maturities, 5 samples). With that many, a few cells will look significant by chance. The one cell where the Nelson-Siegel AR model beats the random walk with a statistic beyond -2 is the 3-month yield at the 1-month horizon in 2018 to 2026 (ratio 0.876, statistic -2.91). I would not read anything into one cell out of 120 for that model.

## Not verified

- Nelson and Siegel (1987) was not read.
- The displayed Diebold-Mariano formula could not be read from the scan's text layer. The implementation follows the prose of section 1.1.
- Litterman and Scheinkman's Table 2 was read from an 80 dpi page image. A test confirms that each row's three factor shares sum to 100 and that the printed averages equal the column means, which makes a misread digit unlikely.
- Whether Diebold and Li's dates are forecast origins or targets.
- The Treasury's par yields are read off a curve fitted to indicative bid-side quotes. Its methodology page says the fit changed on 6 December 2021 from a quasi-cubic Hermite spline to a monotone convex method. I did not test whether that change is visible in the results.
- The CI workflow has not been seen running on GitHub.
