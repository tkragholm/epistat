# Changelog

## [0.3.0] - unreleased

Models take their data as one checked value instead of five loose arguments, errors are one type with stable codes, and covariances are `Covariance` values. The crate adds negative binomial and quasi families, sandwich covariances, GEE, ordinal regression, attributable fractions, models fitted to rows formed block by block, and fixed-effects least squares.

### Breaking changes

- `GlmError`, `ElasticNetError` and `MultinomError` are replaced by `epistat_glm::Error`, a non-exhaustive enum. Its variants include `Input` (wrapping `epistat_core::InputError`), `Response { row, value }`, `NotConverged`, `Aliased { column }`, `Singular`, `Separation`, `StepHalving`, `NoValidStart`, `ConstantResponse` and `Solve`. Match on `Error` or read `code()` through `epistat_core::Classify`.
- `Glm::fit(&data)`, `ElasticNet::fit(&data)` and `Multinom::fit(&data)` take a `GlmData` or `MultinomData`, built with `GlmData::builder(x, y).weights(w).offset(o).start(s).build()?`. The five-argument forms are `fit_slices`, deprecated.
- Penalty factors of the elastic net are the field `ElasticNet::penalty_factor` and no longer an argument of `fit`.
- `Fit::covariance` and `MultinomFit::covariance` are `epistat_core::Covariance`, in place of `Vec<Option<f64>>` and `Vec<f64>`. Use `covariance.get(i, j)` or `standard_errors()`.
- `MultinomFit::coefficient`, `row_probabilities` and `linear_predictor` return `Option`, and `standard_errors` returns `Vec<Option<f64>>`, with `None` where the value is undefined.
- `Link`, `Family`, `ResponseScaling`, `Fit`, `ElasticNetFit`, `MultinomFit`, `ParkTest` and `ClassicParkTest` are `#[non_exhaustive]`. A `match` on `Family` or `Link` needs a wildcard arm, and results are read, not built with a struct literal.
- `Family` has three new variants, `NegativeBinomial`, `QuasiPoisson` and `QuasiBinomial`.
- `Fit` has a new field `residual_df`.
- The crate depends on `epistat-core` and `epistat-inference`, and its minimum supported Rust version is 1.89, up from 1.85.

### Added

- `GlmData`, `GeeData`, `OrdinalData` and `MultinomData`, with builders, which check lengths, finiteness and weights once.
- `Glm::linear_predictor` and `Glm::predict`, `Fit::full_rank`, which returns the coefficients or an error when a column is aliased or the fit did not converge, and `Link::inverse_and_derivative`.
- `NegBin` and `NegBinFit`, the negative binomial model with an estimated or fixed `theta` (`ThetaRule`), and the quasi-Poisson and quasi-binomial families.
- `robust_covariance` and `SandwichType`, sandwich covariances of a fit, and `risk_regression` with `RiskScale`, risk ratios by a Poisson model with a log link and risk differences by a binomial model with an identity link.
- `between_within`, the within-family estimator.
- `Gee`, `GeeFit`, `Correlation`, `Association` and `RobustVariance`, generalised estimating equations with four working correlations.
- `Ordinal`, `OrdinalFit` and `OrdinalLink`, ordinal regression with partial proportional odds, and `brant`, the Brant test.
- `attributable::attributable_logistic`, attributable fractions from logistic models, with `Design`, `IntervalScale` and `AfOptions`.
- `Glm::fit_rows`, `Rows`, `RowsFit` and `Block`, models fitted to rows formed block by block, with the same bits as `Glm::fit` on the matrix those rows make.
- `Feols`, `FeolsFit`, `FixedEffects`, `Absorb`, `Demeaned` and `FixedEffectValues`, least squares with absorbed fixed effects solved by conjugate gradients, with the covariances of `Vcov`, `Ssc`, `KFixef` and `GDf`.
- `expit`, re-exported from `epistat_stats::logistic`.
- `Error`, which implements `Classify`.

### Deprecated

Deprecated items are removed in the next release that breaks the API.

- `GlmError`, `ElasticNetError` and `MultinomError`, use `Error`.
- `Glm::fit_slices`, use `GlmData::builder` and `Glm::fit`.
- `ElasticNet::fit_slices`, use `GlmData::builder`, set `ElasticNet::penalty_factor` and call `ElasticNet::fit`.
- `Multinom::fit_slices`, use `MultinomData::builder` and `Multinom::fit`.

### Results

No released result changes. The test for aliased columns in `Glm::fit` is now settled from the Cholesky factor of the first information matrix where a derived bound allows it, and falls back to the earlier Gram-Schmidt test near the tolerance. It makes the same decisions, and the coefficients do not change.

## [0.2.0] - 2026-11-22

- Added the complementary log-log and probit links, checked against R's `glm`.
- Added the multinomial logit, which agrees with `nnet::multinom`.
- Added weights, links and refitting for propensity scores.
- The elastic net is fitted from weighted moments of the design when there are few columns, and the folds of the cross-validated fit run at the same time.

## [0.1.1] - 2026-11-19

- Added the elastic net, checked against `glmnet`, and the modified and classic Park tests.

## [0.1.0] - 2026-11-19

- First release: generalised linear models fitted by iteratively reweighted least squares as R's `glm.fit` does, with Tweedie families checked against `statmod`, and aliased columns reported as R does.
