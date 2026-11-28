# Changelog

## [0.4.0] - unreleased

Confidence levels, bootstrap settings and errors now use the shared types of `epistat-core`. The crate adds influence functions with clustered variances, p-value adjustment, a contrast curve with simultaneous bands, and pooling of vectors and transformed estimates.

### Breaking changes

- `InferenceError` is replaced by `epistat_inference::Error`, a non-exhaustive enum with the variants `Input` (wrapping `epistat_core::InputError`), `NotFinite`, `NotSymmetric`, `Imputations`, `KeyMismatch` and `Solve`. The old `Empty`, `Length` and `NanReplicate` are now `Error::Input`, and `Alpha` is gone because `Level` checks its value. Match on `Error` or read `code()` through `epistat_core::Classify`.
- `bootstrap::multiplier(estimate, psi, bootstrap)` and `bootstrap::cluster_totals(columns, bootstrap)` take an `epistat_core::Bootstrap` in place of `replicates` and `seed`: write `Bootstrap::new(replicates, Seed(seed))`. `multiplier` refuses a non-finite `psi`.
- `interval::normal(estimate, se, level)` takes a `Level` and returns an `Interval`, not a `Result` of a pair. `percentile`, `trimmed_percentile` and `bias_corrected` take a `Level` where they took `alpha`, and return `Result<Interval, Error>`. Write `Level::alpha(0.05)?` for the old `0.05`.
- `Percentile` is `#[non_exhaustive]`.
- `simultaneous::sup_t_from_covariance(covariance, level, draws)` and `multiplier_bands(influence, clusters, level, bootstrap)` take a `Level` and a `Bootstrap`. `multiplier_bands` takes the influence functions as a faer `MatRef` with one row per cluster and one column per estimate, in place of `&[&[f64]]`. `MultiplierBands::at_least_pointwise` takes no argument and `MultiplierBands` has a new field `level`. `MultiplierBands` is `#[non_exhaustive]`. Both functions refuse non-finite input.
- `pooling::Pooled::interval(level)` takes a `Level` and returns an `Interval`. `Pooled` is `#[non_exhaustive]`.
- `wald::Wald` is `#[non_exhaustive]`.
- `Clusters` numbers clusters as `u32`, like the codes they come from: `Clusters::of` returns `&[u32]` in place of `&[usize]`. `from_labels_first_seen`, `from_sorted_labels`, `singletons` and the deprecated `from_labels` return `Result<Clusters, Error>` and refuse more than `u32::MAX` clusters with `InputError::TooLarge` (code `input.too_large`); add `?`. `from_codes`, `per_unit`, `per_row` and `for_keys` cannot exceed the limit and are unchanged.
- `epistat_linalg::SolveError` inside `Error::Solve` is now `epistat_linalg::Error`.
- The minimum supported Rust version is 1.89, up from 1.85. The crate depends on `epistat-core`, and on `epistat-linalg` 0.4, `epistat-rng` 0.2 and `epistat-stats` 0.3.
- `wald::wald` returns the shared `epistat_core::Test`, a chi-square test whose `df` is `Some(k as f64)` and whose `statistic` and `p_value` are `Option`s, in place of `Wald`. Match on the `Option` where 0.3 read a number. `Wald` is a deprecated alias of `Test`.
- `wald::wald` refuses a covariance whose estimated reciprocal condition number is below the machine epsilon with `Error::Solve` holding `epistat_linalg::Error::IllConditioned`, the rule of R's `solve`, where it returned a statistic with no correct digits: `[[1, 1], [1, 1 + 2^-52]]` gave a statistic near `10^16` and a p-value of 0. It refuses a NaN or infinite estimate or covariance entry with `InputError::NotFinite`. A covariance R accepts gives the same bits as before.
- `interval::trimmed_percentile` refuses a trim outside `[0, 0.5)` and one that leaves fewer than two replicates with `InputError::Domain`, where it capped the trim at 0.49 and returned the untrimmed interval when the trim left nothing. A valid trim gives the same interval.
- `interval::Percentile::OutwardFromLast` and `OutwardFromCount` are renamed `OutwardRank` and `OutwardCount`, after where they count positions. The old names are deprecated constants of the same values, so code that names them still compiles and matches.

### Added

- `Influence`, the influence function of one or more estimates, with `ClusterRule` for the small-sample rule, `variance` and `variance_with` for the clustered covariance, and `combine`, `select` and `with_values`. It is keyed either by unit codes (`Influence::new`, `from_dfbeta`) or by rows (`over_rows`, `from_dfbeta_over_rows`), and `KeyedBy` names which.
- `Clusters::per_unit` and `Clusters::per_row`, which key clusters by unit codes or by rows, `Clusters::from_codes`, `for_keys`, `keys` and `keyed_by`. `Influence::variance` refuses clusters keyed the other way with `Error::KeyMismatch`. Clusters given by position work with either.
- `Clusters::from_labels_first_seen`, the numbering of `from_labels` under a name that states its order.
- `sandwich::SandwichAdjustment` (`None`, `ClusterCount`, `Hc1`), with `factor`, and `sandwich::scaled`, for the small-sample factors of a sandwich covariance, each listed in the module against the call of R's `sandwich` package and of Stata that applies it, and `sandwich::meat_from`, the meat from a vector of residuals.
- `curve::contrast_curve` and `curve::ContrastCurve`, the contrast of a spline term against a reference value, with pointwise or simultaneous (sup-t) bands, chosen by `curve::Band`.
- `multiplicity::adjust` with `Adjustment`, the p-value adjustments of Bonferroni, Holm, Hochberg, Hommel, Benjamini-Hochberg and Benjamini-Yekutieli.
- `pooling::rubin_vector` and `PooledVector`, `pooling::d1`, `d2` and `d3` with `MultiMethod`, `NegativeRiv` and `MultiTest`, `pooling::rubin_transformed` with `Transform` and `PooledOnScale`, `dfcom_residual` and `dfcom_events`, and `Pooled::p_value`.
- `interval::percentile_sorted`, the percentile interval of replicates that are already sorted.
- `bootstrap::unit_counts_of`, the resampling counts of one replicate.
- `Error`, which implements `Classify`.
- `CovarianceInput`, the covariance forms `wald` takes: a faer `MatRef` or `&Mat`, an `epistat_core::Covariance` such as `Influence::variance` returns, or an `epistat_core::Matrix`. A `Covariance` with an undefined entry is refused with `InputError::NotFinite`.
- `interval::try_normal`, the normal interval with its standard error checked: a negative standard error is `InputError::Domain` and a NaN estimate or standard error `InputError::NotFinite`, where `interval::normal` returns reversed or undefined limits.
- `sandwich::covariance`, the sandwich as an `epistat_core::Covariance`, and `sandwich::defined_standard_errors`, the square roots of the diagonal with `None` for a negative or NaN variance, as `Covariance::standard_errors` gives them. `sandwich::standard_errors` still reads a negative variance as zero.
- `simultaneous::sup_t_from_covariance` takes any `CovarianceInput`, so an `epistat_core::Covariance` goes in as it is; a faer `MatRef` still does.
- `Influence::from_values`, one constructor that takes the keys as `Keys::Units` or `Keys::Rows` and the scale of the values as a required `InfluenceScale::PerUnit` or `InfluenceScale::Dfbeta`, so values on the dfbeta scale cannot be read per unit by default, which gave a variance `n^2` times too small. `Influence::new`, `over_rows`, `from_dfbeta` and `from_dfbeta_over_rows` are each one of its four forms.

### Deprecated

Deprecated items are removed in the next release that breaks the API.

- `InferenceError`, use `Error`.
- `Clusters::from_labels`, use `Clusters::from_labels_first_seen`, which gives the same numbering, or `Clusters::from_sorted_labels` for the order of `Codes::from_labels`.
- `influence::standard_error`, use `Influence::variance_with` with `ClusterRule::NoSmallSample`.
- `influence::clustered_variance`, `clustered_standard_error` and `clustered_covariance`, use `Influence::variance`, which centres the influence function.
- `wald::Wald`, use `epistat_core::Test`.
- `interval::Percentile::OutwardFromLast` and `OutwardFromCount`, use `Percentile::OutwardRank` and `Percentile::OutwardCount`.

### Results

- `influence::standard_error`, `clustered_variance`, `clustered_standard_error`, `clustered_covariance` and `bootstrap::cluster_totals`: in the last bits, because sums and dot products run over eight fixed lanes. Inputs shorter than 8 terms are unchanged.
- `simultaneous::multiplier_bands` and `sup_t_from_covariance`: in the last bits, for the same reason, where a sum has 8 or more terms.
- `Pooled::interval`, `Pooled::p_value`, `pooling::d1` and the other t and F tail results: in the last bits, at most about 1e-15 relative, because the incomplete beta function in `epistat-stats` is evaluated differently. The change is larger for 2,000 or more degrees of freedom, where the old tail lost accuracy.

## [0.3.2] - 2026-11-22

- Added Rubin's rules with Barnard-Rubin degrees of freedom.
- Added `sandwich_with_inverse`, and the sandwich is formed correctly for a bread that is not symmetric.
- faer types are taken from `epistat-linalg`.

## [0.3.1] - 2026-11-19

- Added simultaneous inference: sup-t critical values and multiplier-bootstrap bands.

## [0.3.0] - 2026-11-19

- Undefined results are returned as `None`, and invalid weights are refused.

## [0.2.0] - 2026-11-19

- Follows `epistat-linalg`, whose factorisation attempts can refuse small pivots and are non-exhaustive.

## [0.1.1] - 2026-11-19

- Added the Wald test.

## [0.1.0] - 2026-11-19

- First release: influence-function and cluster-robust standard errors, the multiplier and cluster bootstrap, normal, percentile and bias-corrected intervals, and sandwich covariances.
