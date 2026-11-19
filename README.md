# epistat

Rust crates for statistics in epidemiology and register research.

| Crate | Contents |
|---|---|
| [`epistat-optim`](crates/epistat-optim) | BFGS with a Moré-Thuente line search, Nelder-Mead and a dense simplex method |
| [`epistat-rng`](crates/epistat-rng) | `SplitMix64` streams and bootstrap draws that reproduce across threads, platforms and languages |
| [`epistat-stats`](crates/epistat-stats) | Sample quantiles, the normal distribution, sums, log-likelihood helpers and balance statistics, checked against R and Python |
| [`epistat-inference`](crates/epistat-inference) | Influence-function and sandwich variances, cluster-robust and bootstrap intervals |
| [`epistat-linalg`](crates/epistat-linalg) | Dense solves with fallbacks and row-blocked regression products on faer, independent of the thread count |
| [`epistat-testkit`](crates/epistat-testkit) | Tolerances, bit fingerprints, thread-count checks and R reference files, for tests |

Licensed under either of MIT or Apache-2.0, at your option.
