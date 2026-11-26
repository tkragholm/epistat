# epistat-stats

Distribution functions (normal, t, chi-square, F, beta, gamma, Poisson and
binomial, with R's names and NaN outside the domain), sample quantiles (R's
types 1 and 7, and a weighted type 1), floating-point sums in the order
Python and polars compute them and over eight fixed lanes, weighted means and
variances with reliability weights, Kish's effective sample size and weight
limits, plain moments, the logistic function, Wald estimates, log-sum-exp,
softmax and information criteria, MCMC diagnostics (autocorrelation,
effective sample size, split R-hat) and k-means clustering from given
centres. The approximations that published analyses reproduce are in
`compat`.

The epidemiological measures that were here until 0.2 are in `epistat-epi`.
The modules `balance`, `decomposition`, `evalue`, `rosenbaum`, `signed_rank`
and `standardisation`, and the 0.2 paths of the distribution functions, stay
for 0.3 as deprecated copies and forwarders, and are removed in 0.4.

The quantiles and the normal distribution function give the same doubles as
R; the normal quantile is within two units in the last place of R's `qnorm`.
Nothing uses fused multiply-add, so results are the same on every platform.
