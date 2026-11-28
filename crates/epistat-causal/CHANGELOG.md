# Changelog

## [0.1.0] - unreleased

First release of `epistat-causal`: augmented inverse probability weighting, targeted maximum likelihood estimation, entropy balancing, the covariate balancing propensity score and the parametric g-formula, with variances that do not depend on the number of threads.

### Added

- `aipw`: augmented inverse probability weighting for binary and continuous outcomes, for the ATE, ATT and ATC as a difference and a ratio of treatment-specific means. `AipwOptions` takes the estimand, the level and a propensity truncation. The core takes predictions of the outcome and propensity models, and a wrapper fits logistic or generalised linear models and gives the variance that accounts for their estimation.
- `tmle`: targeted maximum likelihood estimation for binary and bounded continuous outcomes as the `tmle` package does it, with logistic fluctuation for the ATE and the one-step estimator for the ATT and ATC. `tmle::survival` estimates the risk by a time with right censoring in discrete time, with iterated logistic fluctuation of the hazards of each arm.
- `balance`: entropy balancing, which matches the means (and with `balance_columns` the higher moments and products) of the target group exactly, and CBPS, just or over identified with a fixed or continuously updated weighting matrix. Both give outcome means, their difference or a weighted regression, with the variance of the stacked estimating equations, valid with clusters.
- `gformula`: the parametric g-formula for time-varying treatments and confounders on person-period data, as `gfoRmula::gformula`, with logistic and linear covariate models, lags, cumulative averages, interactions and restrictions, survival outcomes with a competing event or a binary or continuous outcome at the end of follow-up, and static, scheduled, threshold and dynamic strategies. It gives risks or means, their ratios and differences, Monte Carlo standard errors and a bootstrap over persons. Every strategy is simulated from the same per-person draws. `PersonPeriods` takes the data keyed by `Panel`, and `GFormulaOptions::new(seed)` has no default seed.
- Estimators of a contrast return the two means, their difference and ratio as `Contrast`, and the influence functions of both means on the per-unit scale as `Influence`. Standard errors are those of `Influence::variance` under the cluster rule of each estimator's R reference: AIPW, the balancing weights and the survival TMLE use the `n` divisor without clusters and `G / (G - 1)` with `G` clusters, and TMLE uses `tmle`'s `n - 1` divisor. With fewer than two clusters the standard error, interval and p-value are `None`.
- `Error` with `Classify` and stable `causal.*` codes, and a `Model` enum that names the model that failed. Weights, clusters, `Level`, `Estimand` and `Bootstrap` are the shared types of the epistat crates. Results do not depend on the number of threads.
- Not included: continuous and multi-valued treatments, non-parametric CBPS, categorical and truncated covariates in the g-formula, restrictions on its outcome model and hazard ratios.

### Results

No released result changes.
