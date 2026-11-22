# epistat-glm

Generalised linear models (Gaussian, binomial, Poisson, gamma, Tweedie, negative
binomial with a given shape, and quasi-Poisson and quasi-binomial,
with identity, log, logit, probit and complementary log-log links) fitted by IRLS as
R's `glm.fit` does: the same starting values, step halving and convergence rule, with prior
weights, offsets, starting coefficients and an optional ridge. Coefficients,
deviances, iteration counts and standard errors agree with R's `glm()` on
the test fits, and results do not depend on the number of threads.

The `multinom` module fits the multinomial logit, the model of the generalised
propensity score, as `nnet::multinom()` does. The `elastic_net` module fits Gaussian and binomial elastic nets by coordinate
descent as glmnet does, with weights, penalty factors and warm starts, and the
`park` module has the modified Park test of Manning and Mullahy and the classic
Park test. Both are checked against R.

The `negbin` module fits the negative binomial model with its shape estimated by
maximum likelihood, either by the loops of `MASS::glm.nb` or to the maximum, and
reports a Poisson limit where the data are not overdispersed. The `robust` module
gives sandwich covariances of a fit (HC0, HC1, HC3, clustered) as
`sandwich::vcovHC` and `vcovCL`, and risk ratios and risk differences of a binary
outcome by modified Poisson and binomial identity regression. The `within`
module is the linear within-family estimator with clustered variance as
`fixest::feols`. Each is checked against R.
