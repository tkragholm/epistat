# epistat-causal

Causal effect estimators for cohort data.

- Augmented inverse probability weighting (AIPW) for binary and continuous
  outcomes: the average effect in the population, on the treated or on the
  controls, as a difference and a ratio of treatment-specific means. The core
  takes the predictions of the outcome and propensity models, and its variance
  is the one of the efficient influence function, which suits predictions from
  cross-fitting. A wrapper fits logistic or generalised linear models and
  gives the variance that accounts for their estimation, valid when only one of
  the two models is right.

- Entropy balancing and the covariate balancing propensity score (CBPS) for a
  binary treatment, for the effect in the population, on the treated or on the
  controls. Entropy balancing matches the means, and with `balance_columns` the
  higher moments and products, of the target group exactly, from base weights
  and sample weights, and refuses a target that the group cannot reach. CBPS is
  just or over identified, with a fixed or continuously updated weighting
  matrix. Both give an outcome mean in each group, their difference, or a
  weighted regression, with the variance of the stacked estimating equations of
  the weights and the outcome model, valid with clusters. Continuous and
  multi-valued treatments and non-parametric CBPS are not implemented.

- The parametric g-formula for time-varying treatments and confounders on
  person-period data, as `gfoRmula::gformula`: logistic and linear models for
  binary and normal covariates with lags, cumulative averages, lagged
  cumulative averages, interactions and restrictions (absorbing covariates
  among them), a survival outcome with a competing event or a binary or
  continuous outcome at the end of follow-up, and the natural course with
  static, scheduled, threshold and dynamic strategies at all or some periods.
  It gives the risks or means, their ratios and differences, the Monte Carlo
  standard error of each, and a nonparametric bootstrap over persons with
  percentile or normal limits. Every strategy is simulated from the same
  draws, seeded per person, so the results do not depend on the number of
  threads. Categorical and truncated covariates, restrictions on the outcome
  model, censoring weights and hazard ratios are not implemented.

The risk by a time with right censoring is in `epistat-survival::aipw`, since it
needs the Cox and censoring machinery of that crate.

Case weights and clusters are supported. Results do not depend on the number
of threads, and are checked against `riskRegression::ate` and against
estimators written out by hand in R. Entropy balancing is also checked against
`WeightIt` and `ebal`, and CBPS against `CBPS`, with the variance checked against
the infinitesimal jackknife of the whole procedure and a bootstrap. The
g-formula is checked against `gfoRmula`: exactly where every simulated variable
is set by the strategy, within the Monte Carlo error elsewhere, and against the
exact sum over two periods and the truth of a simulated data-generating
process.
