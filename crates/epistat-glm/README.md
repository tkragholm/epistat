# epistat-glm

Generalised linear models (Gaussian, binomial, Poisson, gamma, Tweedie, negative
binomial with a given shape, and quasi-Poisson and quasi-binomial,
with identity, log, logit, probit and complementary log-log links) fitted by IRLS as
R's `glm.fit` does: the same starting values, step halving and convergence rule, with prior
weights, offsets, starting coefficients and an optional ridge. Each model takes
its data as one checked view (`GlmData`, `GeeData`, `OrdinalData`,
`MultinomData`), every covariance knows its dimension, and every error is one
`Error` with a stable code. `Fit::full_rank` gives the coefficients of a fit
that converged with no aliased column and refuses any other, and
`Glm::predict` gives the means of a fit on new rows. Coefficients,
deviances, iteration counts and standard errors agree with R's `glm()` on
the test fits, and results do not depend on the number of threads.

The `multinom` module fits the multinomial logit, the model of the generalised
propensity score, as `nnet::multinom()` does. The `elastic_net` module fits Gaussian and binomial elastic nets by coordinate
descent as glmnet does, with weights, penalty factors and warm starts, and the
`park` module has the modified Park test of Manning and Mullahy and the classic
Park test. Both are checked against R.

The `negbin` module fits the negative binomial model with its shape estimated by
maximum likelihood, either by the loops of `MASS::glm.nb` or to the maximum, and
reports a Poisson limit where the data are not overdispersed, with a log or an
identity link. The `robust` module
gives sandwich covariances of a fit (HC0, HC1, HC3, clustered) as
`sandwich::vcovHC` and `vcovCL`, and risk ratios and risk differences of a binary
outcome by modified Poisson and binomial identity regression. Each is checked
against R.

The `absorb` module removes any number of fixed-effect factors (unit, time,
firm and more) from columns, with prior weights, the repeated removal of
singletons and a stopping rule on the weighted level means of the residual
that bounds the distance to the exact projection. It solves by
preconditioned conjugate gradients on the equations left after the largest
factor is eliminated, as fast as alternating projections on well connected
panels and far faster on poorly connected designs, such as workers who
seldom move between firms, where the projections can fail to converge. It
returns the fixed effects with one reference per connected component (two
factors) or `fixest`'s count of references (three or more), and predicts their
sums for new rows, so that a model can be fitted on a subset of the rows (0/1
weights) and the fixed effects carried to the others. The `feols` module fits
linear models on top of it with the iid, HC1 and one-way and multi-way
clustered covariances of `fixest::feols`, including its small-sample
corrections and its counting of nested fixed effects, and the influence of
each row. Both are checked against `fixest` 0.14.2 and against the exact
projections on the dummies, and they scale linearly in the rows without a
dummy design. The within-family (sibling) estimator is `Feols` with the family
as its one factor, and `between_within` splits columns into family means and
deviations.

The `gee` module fits generalised estimating equations with independence,
exchangeable, AR(1) and unstructured working correlations, with the robust,
Mancl and DeRouen and naive covariances. The association parameters are
estimated either by the equations of `geepack::geeglm` or by the moment
estimators of the `gee` package, and both are checked against those packages.
The `ordinal` module fits cumulative link models (logit, probit, complementary
log-log and log-log) as `MASS::polr` and `ordinal::clm` do, with case weights,
an offset, cluster-robust covariances and category-specific coefficients for the
partial proportional odds model, and has the Brant test of proportional odds.

The `attributable` module gives the attributable fraction of an
exposure from a logistic model, cross-sectional or case-control, with a sandwich
standard error, weights and clusters, as `AFglm` of the R package AF. It is checked against R.
