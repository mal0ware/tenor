# Sources, formulas and comparison

Every number below marked "here" was produced by `scripts/reproduce.py` and is in `results/`. Numbers marked "paper" were read from the printed tables and are stored in `src/tenor/published.py`.

## Sources

Read in full or in the cited section:

1. Diebold, F. X. and Li, C. (2006). "Forecasting the term structure of government bond yields." Journal of Econometrics 130(2), 337-364. Read in full from the authors' copy of the journal version at `sas.upenn.edu/~fdiebold/papers/paper49/Diebold-Li.pdf`. The paper does not number its equations, so formulas are cited by section.
2. Litterman, R. and Scheinkman, J. (1991). "Common Factors Affecting Bond Returns." Journal of Fixed Income 1(1), 54-61. Read from a scanned copy at `math.nyu.edu/~avellane/Litterman1991.pdf`. It has no numbered equations and no formula for the estimator. Table 2 was read from the page image.
3. Diebold, F. X. and Mariano, R. S. (1995). "Comparing Predictive Accuracy." Journal of Business and Economic Statistics 13(3), 253-263. Section 1.1 read from the authors' copy at `sas.upenn.edu/~fdiebold/papers/paper68/pa.dm.pdf`. The text layer of that scan garbles the displayed formula, so the statistic below is written from the surrounding prose, which is legible.

Used without re-reading the paper. The formulas are the standard statements, and each has a numerical check listed under "Not verified":

4. Gürkaynak, R. S., Sack, B. and Wright, J. H. (2007). "The U.S. Treasury yield curve: 1961 to the present." Journal of Monetary Economics 54(8), 2291-2304. The data file is `federalreserve.gov/data/yield-curve-tables/feds200628.csv`.
5. Harvey, D., Leybourne, S. and Newbold, P. (1997). "Testing the equality of prediction mean squared errors." International Journal of Forecasting 13(2), 281-291.
6. Clark, T. E. and West, K. D. (2007). "Approximately normal tests for equal predictive accuracy in nested models." Journal of Econometrics 138(1), 291-311.
7. Giacomini, R. and White, H. (2006). "Tests of Conditional Predictive Ability." Econometrica 74(6), 1545-1578.

Nelson and Siegel (1987) was not read. The curve is implemented as Diebold and Li state it.

## Formulas implemented

### Zero-coupon yields (`src/tenor/gsw.py`)

The Fed's file gives six parameters per day. With maturity `n` in years, the continuously compounded zero-coupon yield is

    y(n) = b0 + b1 * (1 - exp(-n/t1)) / (n/t1)
              + b2 * ( (1 - exp(-n/t1)) / (n/t1) - exp(-n/t1) )
              + b3 * ( (1 - exp(-n/t2)) / (n/t2) - exp(-n/t2) )

Rows before 1980 have `b3 = 0`. The panel is this formula at 3, 6, 9, 12, 15, 18, 21, 24, 30, 36, 48, 60, 72, 84, 96, 108 and 120 months, the maturities of Diebold and Li's section 3.1, rounded to four decimals. The file lists every weekday; 471 of them from 1985 to October 2026 are holidays with no curve and are dropped, leaving 10,443 days.

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

Not in the paper: Nelson-Siegel with random-walk factors, `bihat_{t+h|t} = bihat_t`. The forecast is today's fitted curve. It is the AR(1) factor model with `c_i = 0` and `g_i = 1` imposed, and it differs from the random walk on yields only by today's fit residual.

Also not in the paper: a rolling scheme in which each forecast is estimated on the last 108 months only.

The paper's other competitors (slope regression, Fama-Bliss and Cochrane-Piazzesi forward regressions, VARs and error-correction models on yields, regression on principal components) are not implemented.

### Accuracy tests (`src/tenor/evaluate.py`)

Diebold and Mariano, section 1.1, with squared-error loss, `d_t = e1_t^2 - e2_t^2`:

    S1 = dbar / sqrt( V / T )
    V  = sum over |k| <= h-1 of gamma_d(k)
    gamma_d(k) = (1/T) * sum_t (d_t - dbar)(d_{t-|k|} - dbar)

The lag window is uniform with truncation `h - 1`. If V comes out negative the paper treats it as zero and rejects; the code returns an infinite statistic in that case. A Bartlett window with weights `1 - k/h` is also reported.

Harvey, Leybourne and Newbold:

    S1* = S1 * sqrt( (T + 1 - 2h + h(h-1)/T) / T )

compared with a Student t with T - 1 degrees of freedom. Every p-value in the tables uses this.

Clark and West, for a small model nested in a large one, with forecasts `yhat_s`, `yhat_l`:

    f_t = e_s,t^2 - ( e_l,t^2 - (yhat_s,t - yhat_l,t)^2 )

The statistic is the t-ratio of the mean of `f`, tested one-sided. Clark and West leave the variance estimator for h > 1 open. I use the uniform window with `h - 1` lags and the Harvey-Leybourne-Newbold factor, because in the size simulation it rejected a true null less often than a Bartlett window with `h - 1`, `1.5h` or `2h` lags. That choice was made on simulated data, before looking at any Clark-West result on the yields.

Negative Diebold-Mariano values favour the first forecast. In the tables the first forecast is the model and the second is the random walk, the same orientation as Diebold and Li's Table 7. Positive Clark-West values favour the large model.

Which test applies where:

| Comparison | Nested? | Reported |
|---|---|---|
| Nelson-Siegel AR(1) or VAR(1) against the random walk | No. With unit-root factors the model gives today's fitted curve, which is not today's yields. | Diebold-Mariano with the correction |
| Nelson-Siegel with unchanged factors against the random walk | No, and neither has estimated dynamics | Diebold-Mariano with the correction |
| AR(1) on yields against the random walk | Yes | Clark-West. The Diebold-Mariano columns are still written to the table and flagged `nests_random_walk`. |
| Nelson-Siegel AR(1) or VAR(1) against unchanged factors | Yes | Clark-West |
| Any pair under the rolling scheme | Does not matter | Diebold-Mariano with the correction, read as the Giacomini-White unconditional test |

The first row is close to nested, because the fit residual is small (1 to 9 basis points on the zero-coupon panel). That is the reason for also running the rolling scheme, where the Giacomini-White result does not depend on nesting.

### Size of the tests (`src/tenor/size_sim.py`, `results/test_size_simulation.csv`)

Rejection rates in percent at a nominal 5%, 5,000 simulations per cell. 84 forecasts is the paper's window and 394 is the long sample.

Two independent error series of equal variance, each a moving sum of h normals:

| Statistic | 84, h=1 | 84, h=6 | 84, h=12 | 394, h=1 | 394, h=6 | 394, h=12 |
|---|---|---|---|---|---|---|
| Uniform window, normal | 5.5 | 9.9 | 16.6 | 4.7 | 5.6 | 7.1 |
| With the correction, Student t | 5.0 | 7.8 | 11.3 | 4.6 | 5.2 | 6.2 |
| Bartlett window, normal | 5.5 | 12.0 | 15.2 | 4.7 | 8.8 | 10.6 |

A random walk forecast by "no change" and by a recursively estimated direct AR(1), 108 observations before the first forecast:

| Statistic | 84, h=1 | 84, h=6 | 84, h=12 | 394, h=1 | 394, h=6 | 394, h=12 |
|---|---|---|---|---|---|---|
| Clark-West, one-sided | 5.2 | 10.5 | 15.1 | 3.4 | 5.3 | 7.1 |
| Diebold-Mariano rejecting in favour of the random walk | 9.7 | 5.4 | 6.2 | 18.6 | 10.6 | 8.1 |
| Share of simulations where the AR(1) has the higher RMSE | 74.3 | 73.8 | 73.7 | 90.9 | 91.4 | 91.9 |

Three things follow. The correction is needed and is not enough on 84 forecasts at 6 and 12 months. The Bartlett window with `h - 1` lags underestimates the variance, which is why it gave larger statistics than the uniform window in the first version of the tables. And for a nested pair the Diebold-Mariano test rejects in favour of the small model far more than 5% of the time when the two are equally good, so "the random walk is better at the 5% level" was not a valid reading of those statistics for the AR(1) on yields.

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

The correlations are computed on the Treasury par yields for 1990 to 2026, different data from the paper's, and still agree to two decimals.

## Claim 1: three factors explain almost all of the variation

Litterman and Scheinkman, Table 2, report that three factors explain 98.4% of the variance of returns on average across eight maturities from 6 months to 18 years, with a minimum of 95.3% at 18 years, and that the first factor accounts for 89.5% of the explained variance, the second 8.5% and the third 2.0%.

Here, on the Treasury par yields, from `results/pca_variance.csv`:

| Treatment | Observations | PC1 % | PC2 % | PC3 % | First three % |
|---|---|---|---|---|---|
| daily changes, covariance | 9,210 | 79.41 | 12.69 | 4.06 | 96.16 |
| daily changes, correlation | 9,210 | 74.34 | 16.72 | 4.45 | 95.50 |
| weekly changes, covariance | 1,921 | 80.96 | 12.82 | 3.60 | 97.38 |
| monthly changes, covariance | 441 | 81.17 | 14.67 | 2.66 | 98.50 |
| monthly levels, covariance | 442 | 95.43 | 4.27 | 0.26 | 99.95 |
| daily changes, with 20y and 30y, from 2006-02-10 | 5,182 | 75.57 | 13.68 | 6.27 | 95.53 |

Per maturity, daily changes (`results/pca_by_maturity.csv`): three components explain 95.8% at 3 months, 87.5% at 6 months, 91.4% at 1 year, and between 96.5% and 98.7% from 2 to 10 years.

Over rolling windows of 1,250 trading days the three-component share ranges from 94.1% to 97.4%. The first component alone ranges from 70.0% (window ending 2012-07-18) to 89.96% (window ending 2014-02-25). `results/pca_bill_volatility.csv` holds the 3-month yield's daily volatility: 19.1 basis points over the 80 trading days from September to December 2008 and 4.6 over the whole sample.

Daily changes give 96.2%, below the figure usually quoted from Litterman and Scheinkman. Reasons the numbers differ, most likely first:

1. Their figure is from a factor model with a specific variance for each maturity, fitted to weekly observations. Mine is plain principal components. The two are different estimators.
2. Frequency. The share rises from 96.2% daily to 97.4% weekly to 98.5% monthly on the same data. Their weekly figure of 98.4% sits one point above my weekly figure.
3. The short end. Bill yields at 6 months and 1 year have day-to-day moves the three components do not capture. Their panel is fitted zero-coupon prices, which are smoother than quoted constant-maturity bill yields.
4. Their sample is January 1984 to June 1988. Mine is 1990 to 2026 and includes the years after 2008 when bill yields sat near zero.
5. They analyse excess returns on zeros; I analyse changes in par yields.

Their first-factor share of 89.5% is also higher than the 79.4% here. The first component of daily par yield changes is not a parallel shift: a one-standard-deviation move is 2.0 basis points at 3 months and 5.3 to 6.0 basis points from 2 to 10 years.

The principal components are not run on the Fed's zero-coupon yields. Those come from a six-parameter curve, so six components would explain all of their variance by construction.

## Claim 2: Nelson-Siegel forecasts beat the random walk at long horizons

Diebold and Li, section 3.4: at 1 month their forecasts are "only very slightly better" than the random walk and no Diebold-Mariano statistic is significant. At 12 months their forecasts "outperform those of all competitors at all maturities".

### The paper's window: targets 1994-01 to 2000-12, 84 forecasts

Zero-coupon yields, estimation from 1985-01, so the first forecast uses the same 108 months as the paper's. The random walk and Nelson-Siegel AR(1) rows are in the README. The other two models the paper prints, from `results/forecast_paper_window.csv`:

| Model | Horizon | Maturity (months) | RMSE here | paper | Ratio to random walk here | paper |
|---|---|---|---|---|---|---|
| AR(1) on yields | 1 | 3 | 0.188 | 0.182 | 1.010 | 1.017 |
| AR(1) on yields | 1 | 12 | 0.234 | 0.239 | 0.997 | 0.992 |
| AR(1) on yields | 1 | 36 | 0.278 | 0.276 | 0.991 | 0.989 |
| AR(1) on yields | 1 | 60 | 0.276 | 0.276 | 0.995 | 1.000 |
| AR(1) on yields | 1 | 120 | 0.247 | 0.258 | 1.007 | 1.016 |
| AR(1) on yields | 6 | 3 | 0.620 | 0.584 | 1.016 | 0.965 |
| AR(1) on yields | 6 | 12 | 0.757 | 0.725 | 0.972 | 0.931 |
| AR(1) on yields | 6 | 36 | 0.809 | 0.801 | 0.923 | 0.911 |
| AR(1) on yields | 6 | 60 | 0.786 | 0.802 | 0.932 | 0.931 |
| AR(1) on yields | 6 | 120 | 0.737 | 0.755 | 0.995 | 0.996 |
| AR(1) on yields | 12 | 3 | 0.934 | 0.845 | 0.912 | 0.829 |
| AR(1) on yields | 12 | 12 | 1.070 | 0.970 | 0.899 | 0.810 |
| AR(1) on yields | 12 | 36 | 1.011 | 1.002 | 0.821 | 0.810 |
| AR(1) on yields | 12 | 60 | 1.005 | 1.007 | 0.864 | 0.846 |
| AR(1) on yields | 12 | 120 | 1.090 | 1.030 | 1.073 | 0.979 |
| Nelson-Siegel VAR(1) | 12 | 3 | 1.137 | 1.102 | 1.111 | 1.081 |
| Nelson-Siegel VAR(1) | 12 | 12 | 1.325 | 1.293 | 1.114 | 1.080 |
| Nelson-Siegel VAR(1) | 12 | 36 | 1.462 | 1.393 | 1.188 | 1.126 |
| Nelson-Siegel VAR(1) | 12 | 60 | 1.493 | 1.385 | 1.284 | 1.163 |
| Nelson-Siegel VAR(1) | 12 | 120 | 1.397 | 1.279 | 1.376 | 1.216 |

Largest absolute gap between my RMSE and the paper's, over the five printed maturities (`results/headline.json`):

| Model | 1 month | 6 months | 12 months |
|---|---|---|---|
| Random walk | 0.009 | 0.018 | 0.036 |
| Nelson-Siegel AR(1) | 0.012 | 0.027 | 0.130 |
| AR(1) on yields | 0.011 | 0.036 | 0.100 |
| Nelson-Siegel VAR(1) | | | 0.119 |

What replicates: the random walk at every horizon; all three models at 1 month; the Nelson-Siegel AR(1) at 6 months, where four of five ratios are within 0.01 of the paper's; the ranking at 12 months, with the AR(1) factor model best up to 3 years, the VAR(1) worst, and both mean-reverting models ahead of the random walk from 3 months to 5 years.

What does not: the size of the 12-month gain at 5 and 10 years. The paper's model has ratios of 0.82 and 0.93 there. Mine has 0.94 and 1.09. The paper's sentence that its forecasts beat all competitors at all maturities at 12 months holds here at 13 of 17 maturities against the random walk, and fails from 7 to 10 years.

Possible reasons, none of them tested: the paper's unsmoothed Fama-Bliss yields and the Fed's smoothed curve are different estimates of the same yields; the fit residual here is smaller and smoother than the paper's would be; and 12-month errors over seven years amount to about seven independent observations.

### The long sample: targets 1994-01 to 2026-10, 394 forecasts

Nelson-Siegel AR(1) against the random walk, recursive estimation from 1985-01 (`results/forecast_vs_random_walk.csv`, `panel = zero`, `sample = full`):

| Horizon | Maturity (months) | RMSE model | RMSE random walk | Ratio | DM corrected | p | DM Bartlett |
|---|---|---|---|---|---|---|---|
| 1 | 3 | 0.243 | 0.228 | 1.064 | 1.94 | 0.053 | 1.94 |
| 1 | 12 | 0.236 | 0.221 | 1.067 | 3.73 | 0.000 | 3.74 |
| 1 | 36 | 0.284 | 0.265 | 1.072 | 3.68 | 0.000 | 3.69 |
| 1 | 60 | 0.300 | 0.271 | 1.108 | 4.00 | 0.000 | 4.00 |
| 1 | 120 | 0.275 | 0.264 | 1.042 | 2.94 | 0.003 | 2.95 |
| 6 | 3 | 0.910 | 0.822 | 1.107 | 1.87 | 0.062 | 2.46 |
| 6 | 12 | 0.903 | 0.802 | 1.127 | 1.85 | 0.065 | 2.40 |
| 6 | 36 | 0.855 | 0.758 | 1.127 | 1.83 | 0.068 | 2.29 |
| 6 | 60 | 0.806 | 0.709 | 1.138 | 1.96 | 0.051 | 2.40 |
| 6 | 120 | 0.666 | 0.633 | 1.052 | 1.20 | 0.230 | 1.40 |
| 12 | 3 | 1.550 | 1.405 | 1.103 | 1.04 | 0.300 | 1.32 |
| 12 | 12 | 1.490 | 1.349 | 1.105 | 0.89 | 0.375 | 1.15 |
| 12 | 36 | 1.325 | 1.144 | 1.158 | 1.22 | 0.222 | 1.60 |
| 12 | 60 | 1.216 | 1.006 | 1.209 | 1.69 | 0.092 | 2.17 |
| 12 | 120 | 0.998 | 0.852 | 1.171 | 2.21 | 0.028 | 2.56 |

Positive statistics favour the random walk. Across all seventeen maturities the ratio is above 1 in all 51 cells.

The same comparison with a rolling 108-month window (`results/forecast_rolling_scheme.csv`):

| Horizon | Maturity (months) | Ratio | DM corrected | p |
|---|---|---|---|---|
| 1 | 3 | 1.116 | 3.37 | 0.001 |
| 1 | 12 | 1.073 | 3.49 | 0.001 |
| 1 | 36 | 1.074 | 3.86 | 0.000 |
| 1 | 60 | 1.119 | 4.19 | 0.000 |
| 1 | 120 | 1.035 | 1.96 | 0.050 |
| 6 | 3 | 1.201 | 2.60 | 0.010 |
| 6 | 12 | 1.210 | 2.35 | 0.019 |
| 6 | 36 | 1.181 | 2.05 | 0.041 |
| 6 | 60 | 1.184 | 1.89 | 0.059 |
| 6 | 120 | 1.114 | 1.36 | 0.176 |
| 12 | 3 | 1.232 | 1.51 | 0.132 |
| 12 | 12 | 1.242 | 1.39 | 0.166 |
| 12 | 36 | 1.260 | 1.52 | 0.129 |
| 12 | 60 | 1.272 | 1.81 | 0.071 |
| 12 | 120 | 1.211 | 2.07 | 0.040 |

The rolling window makes the model worse, not better, so the loss is not a matter of old data weighing on the estimates.

The claim holds on 1994 to 2000 and fails on 1994 to 2026.

### Periods, variants and nested tests

The README has the 12-month ratios by period and for the three variants. Ranges of the Nelson-Siegel AR(1) ratio over the seventeen maturities:

| Targets | Forecasts | 1 month | 6 months | 12 months |
|---|---|---|---|---|
| 1994 to 2000 | 84 | 0.972 to 1.024 | 0.842 to 0.946 | 0.696 to 1.094 |
| 2001 to 2008 | 96 | 1.005 to 1.147 | 1.114 to 1.271 | 1.053 to 1.414 |
| 2009 to 2017 | 108 | 1.027 to 2.042 | 1.119 to 1.985 | 1.340 to 2.570 |
| 2018 to 2026 | 106 | 0.901 to 1.052 | 0.995 to 1.052 | 0.968 to 1.026 |
| All but 2009 to 2017 | 286 | 1.003 to 1.082 | 1.027 to 1.079 | 1.010 to 1.124 |
| All | 394 | 1.011 to 1.108 | 1.052 to 1.138 | 1.103 to 1.215 |

The "all but 2009 to 2017" sample joins two stretches of forecasts end to end. Its autocovariances cross one seam, which matters little at 286 forecasts but is not exact.

Clark-West rejections at the 5% level, number of maturities out of 17 (`results/forecast_nested_tests.csv`):

| Pair | Targets | 1 month | 6 months | 12 months |
|---|---|---|---|---|
| AR(1) factors against unchanged factors | 1994 to 2000 | 1 | 0 | 0 |
| AR(1) factors against unchanged factors | All | 2 | 0 | 0 |
| AR(1) on yields against the random walk | 1994 to 2000 | 0 | 6 | 10 |
| AR(1) on yields against the random walk | All | 0 | 0 | 0 |

### Par yields

The Treasury par panel has eight maturities from 1990. With 108 months before the first target, the targets run from 1999-01 to 2026-10 (334 forecasts). Nelson-Siegel AR(1) ratios to the random walk over the eight maturities: 1.028 to 1.106 at 1 month, 1.059 to 1.126 at 6 months and 1.074 to 1.142 at 12 months. These were the headline of the first version of this repository and are unchanged.

That version also ran the paper's 1994 to 2000 window on the par yields, with estimation starting in 1990. The first 12-month forecast there rested on 25 regression pairs and the model's RMSE was 1.4 to 2.0. That run has been removed because the zero-coupon panel does the job properly. Its table is in the git history at commit `d724795`.

## Differences from the papers

| Item | Paper | Here | Why |
|---|---|---|---|
| Yields for the forecasts | Unsmoothed Fama-Bliss zero-coupon yields from CRSP | Zero-coupon yields from the Fed's fitted Svensson curve | CRSP is licensed. The Fed's file is free. |
| Maturities | 17, from 3 to 120 months | The same 17 | |
| Estimation sample | From 1985:01 | From 1985-01 | |
| Forecast targets | 1994:01 to 2000:12 | The same, then on to 2026-10 | |
| Month end | Last trading day's bid-ask average price | Last day in the month with a published curve | Same intent. |
| Short end | Bills included | Bills excluded by the Fed's fit; maturities under a year are extrapolated | A property of the source. |
| Factor method (Litterman-Scheinkman) | Factor model with specific variances on weekly excess returns of fitted zeros | Principal components of daily Treasury par yield changes | The paper states no estimator formula to copy. |
| Which targets count | "beginning in 1994:1 and extending through 2000:12" | Every horizon is scored on the same target months | The paper does not say whether the dates are origins or targets. Reading them as targets gives 84 forecasts per horizon. |
| DM variance | Not stated in Diebold-Li | Uniform window, h-1 lags, with the Harvey-Leybourne-Newbold correction | See the size simulation. |

Four days were dropped from the par panel for missing values: 2008-12-10, 2008-12-18 and 2008-12-24, when no 3-month yield was published, and 2010-10-11, a row with no values.

## Everything that was run

All of it is in the committed tables.

Chosen before any forecast result on the zero-coupon data was seen, because they are the paper's: the seventeen maturities, estimation from 1985-01, the first target 1994-01, the 1994 to 2000 window, the decay 0.0609, the three horizons, and the four models in the first version.

Post hoc, chosen after seeing the par-yield results of the first version:

- The split of the long sample into 1994-2000, 2001-2008, 2009-2017 and 2018-2026. The first is the paper's window. The third was already known to be the worst stretch on par yields.
- The sample without 2009 to 2017.
- The Nelson-Siegel forecast with unchanged factors.
- The rolling 108-month estimation scheme. 108 is the length of the paper's first estimation window and no other length was tried.
- The Clark-West tests and the Harvey-Leybourne-Newbold correction, added after a review pointed out that the original tests did not apply.

Counts: principal components, six treatments. Forecast models, five. Horizons, three. Evaluation samples, six on the zero-coupon panel and five on the par panel. Estimation schemes, two on the zero-coupon panel. Decay, one value. Rolling principal-components window, one length of 1,250 trading days.

`forecast_vs_random_walk.csv` holds 1,704 model-against-random-walk comparisons. With that many, some cells will pass 5% by chance. The only cell on either panel where the Nelson-Siegel AR(1) beats the random walk with a corrected statistic beyond -1.97 is the 3-month par yield at the 1-month horizon in 2018 to 2026 (ratio 0.876, statistic -2.89). I would not read anything into one cell.

## Not verified

- Nelson and Siegel (1987) was not read.
- The displayed Diebold-Mariano formula could not be read from the scan's text layer. The implementation follows the prose of section 1.1.
- Gürkaynak, Sack and Wright (2007) was not re-read. The check on the yield formula is numerical: the yields it gives at whole years match the Fed's own tabulated `SVENY` columns on all 10,443 days to within 0.0004 percentage points (`results/data_vintage.json`). I did not check from the paper which securities the fit excludes in each year; the statement that bills are excluded is from memory of the paper.
- Harvey, Leybourne and Newbold (1997), Clark and West (2007) and Giacomini and White (2006) were not re-read. The first two formulas are checked only by hand calculations in the tests and by the size simulation. The reading of the rolling-scheme statistic as a Giacomini-White test rests on my understanding of their result and has no numerical check here.
- Litterman and Scheinkman's Table 2 was read from an 80 dpi page image. A test confirms that each row's three factor shares sum to 100 and that the printed averages equal the column means, which makes a misread digit unlikely.
- Whether Diebold and Li's dates are forecast origins or targets.
- Which variance estimator Diebold and Li used for their Table 7.
- The Treasury's par yields are read off a curve fitted to indicative bid-side quotes. Its methodology page says the fit changed on 6 December 2021 from a quasi-cubic Hermite spline to a monotone convex method. I did not test whether that change is visible in the results.
- The Fed's file can be revised without notice. The committed results are from the file downloaded on 23 November 2026.
