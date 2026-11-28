# Changelog

## [0.1.0] - unreleased

First release. The crate holds the types the other epistat crates share, so that results pass between them without conversion.

### Added

- `Level`, a confidence level kept as the confidence or the alpha the caller gave, built with `Level::confidence` or `Level::alpha`. `lower_probability` and `upper_probability` give the tail probabilities.
- `Level::P90`, `Level::P95` and `Level::P99`, and `Level::percent(k)` for `k` percent, so a usual level needs no `unwrap`. `same_coverage` compares two levels whichever form each was given in. A level prints as `95%`.
- `InputError::Percentage`: `Level::confidence(95.0)` is refused with a message that says 95% is 0.95. The documentation of each `InputError` variant names its code.
- `Interval`, `Estimate`, `Scale` and `Contrast`, the way an estimator reports a number. Undefined values are `None`, never NaN.
- `Covariance`, a covariance matrix that knows its dimension, with `variances`, `standard_errors` and `quadratic`.
- `EventStudy`, event-study coefficients with their covariance and reference period.
- `Bootstrap`, a number of replicates and a root seed. There is no default seed.
- `Estimand`, the population an average treatment effect is averaged over.
- `Codes`, labels as dense `u32` codes in sorted order, with the way back to the labels.
- `Panel`, the validated unit-and-period key of person-period data.
- `day`, civil dates as `i64` days since 1970-01-01, with `from_civil`, `to_civil`, `add_years`, `add_months` and the `LeapBirthday` rule for 29 February. `from_civil` does not check its date, so `from_civil(2020, 13, 45)` is a day in 2021; `checked_from_civil` returns `None` for a date that does not exist.
- `Day`, one civil date. It is made from a date that exists (`Day::from_civil` returns `None` otherwise) or from a day count whose epoch the call names, `Day::from_unix_days` or `Day::from_sas_days` (SAS counts from 1960-01-01, `SAS_OFFSET_DAYS` = 3653 days earlier), and reads back as either. It prints as an ISO date, `2024-02-29`, and its serde form is the Unix day count.
- `check` and `InputError`, the input checks the estimators share: `same_length`, `non_empty`, `finite`, `finite_column`, `weights`, `positive_weights`, `positive`, `below`, `probability`, `sorted_by`, `strictly_sorted_by`, `csr_offsets`, `constant_within` and `fits_u32`. `InputError` reports the position of the offending row or column where there is one.
- `ErrorKind` (`Input`, `Numerical`, `Unsupported`) and the `Classify` trait, which give every error a kind and a stable code such as `input.length`.
- The `serde` feature, which derives serialisation for the shared types. Types whose fields must agree with each other are validated when deserialised, and `Bootstrap` refuses unknown fields.

### Results

No released result changes.
