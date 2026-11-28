# Changelog

## [0.1.0] - unreleased

First release. The crate holds the types the other epistat crates share, so that results pass between them without conversion.

### Added

- `Level`, a confidence level kept as the confidence or the alpha the caller gave, built with `Level::confidence` or `Level::alpha`. `lower_probability` and `upper_probability` give the tail probabilities.
- `Interval`, `Estimate`, `Scale` and `Contrast`, the way an estimator reports a number. Undefined values are `None`, never NaN.
- `Covariance`, a covariance matrix that knows its dimension, with `variances`, `standard_errors` and `quadratic`.
- `EventStudy`, event-study coefficients with their covariance and reference period.
- `Bootstrap`, a number of replicates and a root seed. There is no default seed.
- `Estimand`, the population an average treatment effect is averaged over.
- `Codes`, labels as dense `u32` codes in sorted order, with the way back to the labels.
- `Panel`, the validated unit-and-period key of person-period data.
- `day`, civil dates as `i64` days since 1970-01-01, with `from_civil`, `to_civil`, `add_years`, `add_months` and the `LeapBirthday` rule for 29 February.
- `check` and `InputError`, the input checks the estimators share: `same_length`, `non_empty`, `finite`, `finite_column`, `weights`, `positive_weights`, `positive`, `below`, `probability`, `sorted_by`, `strictly_sorted_by`, `csr_offsets` and `constant_within`. `InputError` reports the position of the offending row or column where there is one.
- `ErrorKind` (`Input`, `Numerical`, `Unsupported`) and the `Classify` trait, which give every error a kind and a stable code such as `input.length`.
- The `serde` feature, which derives serialisation for the shared types. Types whose fields must agree with each other are validated when deserialised, and `Bootstrap` refuses unknown fields.

### Results

No released result changes.
