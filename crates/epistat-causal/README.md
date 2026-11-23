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

The risk by a time with right censoring is in `epistat-survival::aipw`, since it
needs the Cox and censoring machinery of that crate.

Case weights and clusters are supported. Results do not depend on the number
of threads, and are checked against `riskRegression::ate` and against
estimators written out by hand in R.
