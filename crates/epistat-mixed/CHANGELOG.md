# Changelog

## [0.1.0] - unreleased

First release of `epistat-mixed`: linear and generalised linear mixed models with one grouping factor, fitted as `lme4` fits them, in time linear in rows and groups.

### Added

- `Lmm`: linear mixed models with a random intercept and optionally one random slope with an unstructured covariance, by REML or ML, as `lme4::lmer`. After one pass over the rows each evaluation of the profiled criterion costs `O(G q p^2)` for `G` groups, and its gradient is in closed form.
- `Glmm`: binomial (logit) and Poisson (log) models with a random intercept by the Laplace approximation or adaptive Gauss-Hermite quadrature, as `lme4::glmer` with `nAGQ` of 1 or more, with a closed-form gradient of the approximated likelihood.
- Fixed effects with their covariance, variance components, the log likelihood with the family's constants, conditional modes with their conditional variances, and the intraclass correlation where it is defined. Singular fits are reported with the variance at exactly 0.
- `evaluate` on both models, for evaluation at given variance parameters.
- `quadrature`: Gauss-Hermite rules for the standard normal density, as `lme4::GHrule`.
- Results do not depend on the number of threads.
- Not included: several grouping factors, nested or crossed random effects, random slopes in generalised models, other families and links, and standard errors of the variance components of linear models.

### Results

No released result changes.
