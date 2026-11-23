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

- Targeted maximum likelihood estimation (TMLE) for a binary or bounded
  continuous outcome, as the R package `tmle` does it: the effect in the
  population by logistic fluctuation, on the treated and on the controls by the
  one-step estimator with `tmle`'s support rule, with the outcome mapped to
  `[0, 1]` and the propensity bounded as `tmle` does. It takes the predictions of
  the two models, or fits a main-terms or per-arm generalised linear model of the
  outcome and a logistic model of the treatment, and gives the variance of the
  efficient influence function. The estimates stay inside the range of the
  outcome. The risk by a time with right censoring is estimated in discrete
  time, on given predictions of the event and censoring hazards, with iterated
  logistic fluctuation of the hazards of each arm.

The risk by a time with right censoring is in `epistat-survival::aipw`, since it
needs the Cox and censoring machinery of that crate.

Case weights and clusters are supported. Results do not depend on the number
of threads, and are checked against `riskRegression::ate` and against
estimators written out by hand in R. TMLE is checked against `tmle` with its
learners replaced by `glm`, and its survival form against the estimator written
out in R. Entropy balancing is also checked against `WeightIt` and `ebal`, and
CBPS against `CBPS`, with the variance checked against the infinitesimal
jackknife of the whole procedure and a bootstrap.
