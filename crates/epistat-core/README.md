# epistat-core

The types every epistat crate shares, so that results pass from crate to
crate without conversion and a binding converts one set of types:

- `Level`, a confidence level kept as the confidence or the alpha the caller
  gave, so that no conversion moves a bit;
- `Interval`, `Estimate`, `Scale` and `Contrast`, how an estimator reports a
  number, with `None` where it is undefined;
- `Covariance`, a covariance matrix that knows its dimension;
- `EventStudy`, the input of HonestDiD and the pre-trend power analysis;
- `Bootstrap`, replicates and a root seed, with no default seed;
- `Estimand`, the target population of an average treatment effect;
- `Codes`, labels as dense `u32` codes in sorted order and back;
- `Panel`, the validated unit-and-period key of person-period data;
- `day`, civil dates as `i64` days since 1970-01-01;
- `check` and `InputError`, the input checks every estimator shares;
- `ErrorKind` and `Classify`, the kind and stable code of every error.

The crate has no dependencies. The `serde` feature derives serialisation for
the shared types, with validation on deserialisation.
