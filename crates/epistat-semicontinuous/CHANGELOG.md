# Changelog

## [0.1.0] - unreleased

First release of `epistat-semicontinuous`, which continues `semicontinuous_models` (last release 0.9.0 on crates.io). Replace `semicontinuous_models` with `epistat-semicontinuous` in `Cargo.toml`. The entry points, options and error type are new.

### Added

- `two_part::fit`, `two_part::bootstrap`, `percentile_intervals`, `summary` and `wald_intervals`: the two-part model, a logit for any positive outcome and a gamma regression with log link for the positive ones, with prior weights, ridge and elastic-net penalties, warm starts, and model-based, robust or cluster-robust standard errors. Compact designs refit one design under many weight vectors for a fractional-random-weight bootstrap.
- `tweedie::fit`: the Tweedie GLM with log link by quasi-likelihood, in agreement with R's `glm` and `statmod::tweedie`.
- `lognormal::fit`: log-normal regression with Duan's smearing retransformation.
- `mtp::fit`: the marginalised two-part model for panels, fitted by MCMC with correlated subject effects.
- `compare::compare`, `cross_validate` and `select`: in-sample metrics, AIC and BIC, K-fold cross-validation and the Park test, over a `Candidate`. A Tweedie power that is not finite is refused with `Error::Input`.
- `ModelFitMetrics::r2` is `Option<f64>`, `None` where the outcome does not vary. The cross-validated `r2` is the mean over the folds where it is defined, as caret's `na.rm = TRUE`, and `None` when no fold has one; the other metrics average over every fold. `CrossValidation::constant_folds` and `ModelComparison::constant_folds` count the folds whose held-out outcome is constant. The ranking guard judges a score without `r2` by its RMSE being finite.
- `CostData` and `LongitudinalData`, checked views of the columns built with `CostData::builder(x, y)`. Bad input is `Error::Input(InputError)`.
- Each fit returns the model with its report. A relaxed ridge ladder for fits that fail is opt-in and reported, and an intercept-only logit is reported as `AllPositive`, `AllZero` or `Separable`. Undefined quantities (AIC, BIC, outcome shares, the Park test, MTP summaries) are `Option`.
- `Error` with `Classify` and stable `semicontinuous.*` codes. Intervals are `epistat_core::Interval` and levels are `Level`. `MtpOptions`, `CompareOptions` and `CrossValidationOptions` are built with `new(seed)`, with no default seed.
- Results do not depend on the number of threads. The fits use fused multiply-add, so they can differ between platforms in the last bits.

### Results

No released result changes, apart from the R-squared of an outcome that does not vary. `semicontinuous_models` reported it as 0 and averaged that 0 into the cross-validated R-squared, which pulled the mean toward 0 and could carry a fit that failed on the other folds past the ranking guard. A cross-validation with a constant held-out fold now has a different `r2` and can rank differently; one without such a fold gives the same bits.
