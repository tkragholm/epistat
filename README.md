# epistat

Rust crates for statistics in epidemiology and register research.

| Crate | Contents |
|---|---|
| [`epistat-causal`](crates/epistat-causal) | Augmented inverse probability weighting for binary and continuous outcomes, with efficient and stacked variances |
| [`epistat-did`](crates/epistat-did) | Difference-in-differences: staggered ATT(g,t) and its aggregations, doubly robust two-period estimators and HonestDiD sensitivity analysis |
| [`epistat-glm`](crates/epistat-glm) | Generalised linear models fitted by IRLS as R's `glm.fit` does |
| [`epistat-impute`](crates/epistat-impute) | Multiple imputation by chained equations: normal, predictive mean matching, logistic and multinomial methods, passive columns and survival predictors |
| [`epistat-inference`](crates/epistat-inference) | Influence-function and sandwich variances, cluster-robust and bootstrap intervals |
| [`epistat-linalg`](crates/epistat-linalg) | Dense solves with fallbacks and row-blocked regression products on faer, independent of the thread count |
| [`epistat-matching`](crates/epistat-matching) | Nearest-neighbour and risk-set matching under calipers and register constraints, with weights, exclusion reasons and balance statistics |
| [`epistat-mixed`](crates/epistat-mixed) | Linear and generalised linear mixed models with one grouping factor, by REML, ML, the Laplace approximation and adaptive Gauss-Hermite quadrature, as `lme4` fits them |
| [`epistat-optim`](crates/epistat-optim) | BFGS with a Moré-Thuente line search, Nelder-Mead and a dense simplex method |
| [`epistat-register`](crates/epistat-register) | Calendar-year person-time, interval merging, connected components over shared keys and constant-price deflation |
| [`epistat-rng`](crates/epistat-rng) | `SplitMix64` streams and bootstrap draws that reproduce across threads, platforms and languages |
| [`epistat-semicontinuous`](crates/epistat-semicontinuous) | Two-part, Tweedie, log-normal and marginalised two-part models for semicontinuous outcomes such as costs |
| [`epistat-sequences`](crates/epistat-sequences) | Optimal-matching and Hamming distances, k-medoids, silhouettes and typology validation for state sequences |
| [`epistat-stats`](crates/epistat-stats) | Sample quantiles, the normal distribution, sums, log-likelihood helpers and balance statistics, checked against R and Python |
| [`epistat-survival`](crates/epistat-survival) | Cox, Fine-Gray and multi-state regression, Aalen-Johansen and multi-state occupation with cluster bootstraps, Kaplan-Meier, pseudo-observations and prediction metrics |
| [`epistat-testkit`](crates/epistat-testkit) | Tolerances, bit fingerprints, thread-count checks and R reference files, for tests |

Licensed under either of MIT or Apache-2.0, at your option.
