# epistat-mixed

Linear and generalised linear mixed models with one grouping factor, as R's
`lme4` fits them, at the scale of national registers: the cost is linear in
rows and in clusters.

- Linear mixed models with a random intercept and optionally one random
  slope with an unstructured covariance, by REML or ML, as `lme4::lmer`. One
  grouping factor makes every matrix of `lme4`'s formulation block diagonal,
  so after one pass over the rows each evaluation of the profiled criterion
  costs `O(G q p²)` for `G` clusters, and its gradient is closed form.
- Binomial (logit) and Poisson (log) models with a random intercept by the
  Laplace approximation or adaptive Gauss-Hermite quadrature, as
  `lme4::glmer` with `nAGQ = 1` or more, with a closed-form gradient of the
  approximated likelihood that follows the moving nodes.
- Fixed effects with their covariance, variance components, the log
  likelihood with the family's constants (comparable across numbers of
  nodes), conditional modes (BLUPs) with their conditional variances, and the
  intraclass correlation where it is defined. Singular fits are reported with
  the variance at exactly 0, as `lme4` reports them.
- Evaluation at given variance parameters, for profiles and for checking the
  quadrature against more nodes.
- Gauss-Hermite rules for the standard normal density, as `lme4::GHrule`.

The fits are checked against `lme4` at fixed parameters and at the optimum,
and the quadrature against the marginal likelihood integrated by R's
`integrate()`. Results do not depend on the number of threads.

Not included: several grouping factors, nested or crossed random effects,
random slopes in generalised models, other families and links, and standard
errors of the variance components of linear models, which `lme4` does not
give either.
