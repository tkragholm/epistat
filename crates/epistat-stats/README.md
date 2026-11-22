# epistat-stats

Sample quantiles (R's types 1 and 7, and a weighted type 1), the normal distribution function and
its inverse with truncated and folded quantiles, the log-gamma, digamma, trigamma and
regularised incomplete gamma functions with the chi-square distribution, the
Student t distribution and the incomplete beta function,
floating-point sums in the order Python computes them, and log-sum-exp, softmax, information criteria and covariate balance statistics
(standardised mean differences, variance ratios, eCDF and eQQ distances,
Cramér's V), and MCMC diagnostics (autocorrelation, effective sample size,
split R-hat).

Also weighted means and variances with reliability weights, direct
standardisation and the Kitagawa decomposition, Poisson confidence intervals,
E-values, Rosenbaum bounds with the signed rank statistic and its ranks, k-means
clustering from given centres, and sums and dot products over eight fixed lanes.

The quantiles and the normal distribution function give the same doubles as
R; the normal quantile is within two units in the last place of R's `qnorm`.
Nothing uses fused multiply-add, so results are the same on every platform.
