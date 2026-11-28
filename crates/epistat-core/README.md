# epistat-core

The types every epistat crate shares, so that results pass from crate to
crate without conversion and a binding converts one set of types:

- `Level`, a confidence level kept as the confidence or the alpha the caller
  gave, so that no conversion moves a bit, with `Level::P95` and the other
  usual levels;
- `Interval`, `Coverage`, `Estimate`, `Scale` and `Contrast`, how an
  estimator reports a number, with `None` where it is undefined;
- `Coefficients`, `Glance` and the `Tidy` trait, a fit's coefficient table and
  summary as R's `summary` and broom give them;
- `Test`, the statistic, degrees of freedom and p-value of a hypothesis test;
- `Covariance`, a covariance matrix that knows its dimension, and `Matrix`, a
  rectangular matrix for results;
- `EventStudy`, the input of HonestDiD and the pre-trend power analysis;
- `Bootstrap` and `Seed`, replicates and a root seed, with no default seed;
- `Estimand`, the target population of an average treatment effect;
- `CaseWeights`, a checked column of case weights and what they mean;
- `Codes`, labels as dense `u32` codes in sorted order and back;
- `Panel`, the validated unit-and-period key of person-period data, and
  `Permutation`, the order that sorts it, applied column by column;
- `day` and `Day`, civil dates as `i64` days since 1970-01-01;
- `check` and `InputError`, the input checks every estimator shares;
- `ErrorKind` and `Classify`, the kind and stable code of every error.

The types print as R prints its results:

```rust
use epistat_core::{Estimate, Interval, Level, Scale};

let hr = Estimate::point(0.510)
    .with_interval(Some(Interval::new(Some(0.428), Some(0.608), Level::P95)))
    .with_p_value(Some(4.9e-14))
    .with_scale(Scale::Log);
assert_eq!(hr.to_string(), "0.510 (95% CI 0.428 to 0.608), p = 4.9e-14");
```

The crate has no dependencies. The `serde` feature derives serialisation for
the shared types, with validation on deserialisation.
