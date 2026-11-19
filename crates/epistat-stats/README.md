# epistat-stats

Sample quantiles (R's types 1 and 7), the normal distribution function and
its inverse with truncated and folded quantiles, floating-point sums in the order Python computes them, and
log-sum-exp, softmax, information criteria and covariate balance statistics
(standardised mean differences, variance ratios, eCDF and eQQ distances,
Cramér's V), and MCMC diagnostics (autocorrelation, effective sample size,
split R-hat).

The quantiles and the normal distribution function give the same doubles as
R; the normal quantile is within two units in the last place of R's `qnorm`.
Nothing uses fused multiply-add, so results are the same on every platform.
