# Changelog

## [0.1.0] - unreleased

First release. The crate holds the types the other epistat crates share, so that results pass between them without conversion.

### Added

- `Level`, a confidence level kept as the confidence or the alpha the caller gave, built with `Level::confidence` or `Level::alpha`. `lower_probability` and `upper_probability` give the tail probabilities.
- `Level::P90`, `Level::P95` and `Level::P99`, and `Level::percent(k)` for `k` percent, so a usual level needs no `unwrap`. `same_coverage` compares two levels whichever form each was given in. A level prints as `95%`.
- `InputError::Percentage`: `Level::confidence(95.0)` is refused with a message that says 95% is 0.95. The documentation of each `InputError` variant names its code.
- `Interval`, `Estimate`, `Scale` and `Contrast`, the way an estimator reports a number. Undefined values are `None`, never NaN.
- `Coverage` on `Interval`: pointwise, as `Interval::new` builds it, or simultaneous with the family's critical value, as `Interval::simultaneous` builds it for a uniform band. An estimate with a simultaneous interval carries no p-value.
- `Estimate::null`, the value the p-value tests (0, or 1 for a ratio on the log scale), and `Estimate::se_log`, the standard error of the log, which is `None` on the identity scale.
- `Display` for `Interval` and `Estimate`: `0.510 (95% CI 0.428 to 0.608), p = 4.9e-14`.
- `Coefficients`, a fit's coefficient table as R's `summary` and broom's `tidy` give it: named `Term`s with estimates, standard errors, z or t statistics, p-values and intervals at one `Level`, the covariance they came from, its `VarianceKind` (model-based or robust) and the `EffectScale` (identity, or log hazard, subdistribution hazard, odds, risk, rate, mean or time ratios). `exp` gives the ratios with the standard errors kept on the log scale and is `None` on the identity scale. `get` reads a term by name, `tidy` gives broom's rows, `sub_covariance` the block for a joint test, and the table prints as `printCoefmat` does. `Glance` is the one-row summary of a fit (rows, rows used, events, log likelihoods, AIC, BIC, deviance, iterations, convergence), and `Tidy` is the trait every fit implements to give both.
- `Covariance`, a covariance matrix that knows its dimension, with `variances`, `standard_errors` and `quadratic`. It prints as R prints a matrix.
- `Matrix`, a rectangular matrix for results, stored row by row, of finite `f64` or of `Option<f64>` with `None` for an undefined entry. It has its shape, `get`, `row`, `rows`, `column`, `columns`, `transpose`, conversions from and to column-major order, rows as vectors and `Covariance`, prints as R prints a matrix and serialises as its shape and row-major values. epistat-linalg's `convert` copies between it and faer's `Mat`.
- `EventStudy`, event-study coefficients with their covariance and reference period. `is_consecutive` says whether the relative periods and the reference are consecutive integers, as HonestDiD's restrictions assume.
- `Bootstrap`, a number of replicates and a root seed, and `Seed`, the root seed as a type of its own, so that `Bootstrap::new(999, Seed(20_381))` cannot be written with the two numbers swapped. A seed's value and the streams derived from it are those of the integer. There is no default seed. A bootstrap prints as `999 replicates from seed 20381`.
- `Test` and `Reference`, the statistic, degrees of freedom and p-value of a hypothesis test and the distribution the p-value is read from (normal, t, chi-square, F, the chi-bar-square of a boundary test, or exact). A test prints as `chi-square = 3.21 on 2 df, p = 0.20`.
- `Estimand`, the population an average treatment effect is averaged over.
- `Codes`, labels as dense `u32` codes in sorted order, with the way back to the labels.
- `Panel`, the validated unit-and-period key of person-period data. `Panel::sort_order` returns a `Permutation`, whose `apply` puts a column in the sorted order and refuses a column of another length, and whose `inverse` puts per-row results back. It derefs to the row numbers.
- `day`, civil dates as `i64` days since 1970-01-01, with `from_civil`, `to_civil`, `add_years`, `add_months` and the `LeapBirthday` rule for 29 February. `from_civil` does not check its date, so `from_civil(2020, 13, 45)` is a day in 2021; `checked_from_civil` returns `None` for a date that does not exist.
- `Day`, one civil date. It is made from a date that exists (`Day::from_civil` returns `None` otherwise) or from a day count whose epoch the call names, `Day::from_unix_days` or `Day::from_sas_days` (SAS counts from 1960-01-01, `SAS_OFFSET_DAYS` = 3653 days earlier), and reads back as either. It prints as an ISO date, `2024-02-29`, and its serde form is the Unix day count. `day - earlier` and `days_since` give the days between two days, `checked_add_days`, `checked_add_years` and `checked_add_months` move a day within its range, and `from_unix_column`, `from_sas_column`, `from_unix_column_i32` (Arrow's `Date32`) and `to_unix_column` convert whole columns with the epoch named.
- `CaseWeights`, a borrowed column of case weights checked once where it enters: finite, not negative, not all zero, with a `WeightKind` of `Prior` (R's `weights`), `Frequency` (whole numbers of occurrences) or `Sampling` (inverse selection probabilities). A weight of 0 means the row leaves the fit. Views and estimators take it in place of `&[f64]`. `InputError::AllZero` refuses weights that are zero in every row.
- `check` and `InputError`, the input checks the estimators share: `same_length`, `non_empty`, `finite`, `finite_column`, `weights`, `positive_weights`, `positive`, `below`, `probability`, `sorted_by`, `strictly_sorted_by`, `csr_offsets`, `constant_within` and `fits_u32`. `InputError` reports the position of the offending row or column where there is one.
- `ErrorKind` (`Input`, `Numerical`, `Unsupported`) and the `Classify` trait, which give every error a kind and a stable code such as `input.length`. A `Classify` type is an `Error` that is `Send`, `Sync` and `'static`, so a binding can take `&dyn Classify`, print it and walk its sources, and an error can cross threads. `Tidy`'s error type implements `Classify`.
- The `serde` feature, which derives serialisation for the shared types. Types whose fields must agree with each other are validated when deserialised, and `Bootstrap` refuses unknown fields.

### Results

No released result changes.
