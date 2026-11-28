# Changelog

## [0.1.0] - unreleased

First release of `epistat-did`, which continues `did_methods` (last release 0.2.0 on crates.io). Replace `did_methods` with `epistat-did` in `Cargo.toml`. The HonestDiD sensitivity analysis, which `did_methods` held in its `honest` module, is the separate crate `epistat-honestdid`.

### Added

- `att_gt`: group-time average treatment effects ATT(g,t) for staggered adoption (Callaway and Sant'Anna) by outcome regression, inverse probability weighting or the doubly robust estimator, on panels and repeated cross-sections, with never-treated or not-yet-treated comparison groups, varying or universal base periods, anticipation and sampling weights, and aggregation by event time, cohort, calendar time or overall as `aggte` does. `AttGt::event_study` returns the event study that `epistat-honestdid` takes.
- `two_period::estimate`: the two-period doubly robust estimators of `DRDID` on a `TwoPeriodData` (panel or repeated cross-section), with `Estimator::{Improved, DoublyRobust, TraditionalDoublyRobust, InverseProbabilityWeighting, OutcomeRegression}`. `Improved` is the default and reports the estimate plus or minus 1.96 standard errors as a 95 per cent interval whatever `level` is set.
- `did_cc::estimate`: compositional changes in repeated cross-sections (Sant'Anna and Xu), with cross-fitting. A propensity model that fails to fit is an error unless `DidCcOptions::propensity_fallback` is set, in which case the fallbacks used are listed in `DidCc::propensity_fallbacks`.
- `triple_did::estimate` for doubly robust triple differences, and `continuous::estimate` for the average causal response to a continuous treatment by a sieve estimator, with a polynomial or B-spline basis.
- The efficient ATT(g,t) that combines all baseline periods (Chen, Sant'Anna and Xie), with aggregations whose standard errors use the combined influence function.
- `dcdh::estimate`: the event-study estimator of de Chaisemartin and D'Haultfoeuille for treatments that switch on and off or take several values, as `did_multiplegt_dyn` of `DIDmultiplegtDYN`, with placebos, normalised effects, controls, strata, weights, clusters, joint tests, heterogeneity regressions and a bootstrap.
- `twfe` and `bacon`: static and event-study two-way fixed-effects regressions as `fixest` fits them, and the Goodman-Bacon decomposition as `bacondecomp::bacon`.
- `sunab`, `etwfe`, `imputation`, `two_stage` and `synthdid`: the interaction-weighted estimator of Sun and Abraham, Wooldridge's extended two-way fixed effects (linear, Poisson and logit), the imputation estimator of Borusyak, Jaravel and Spiess, the two-stage estimator of Gardner, and synthetic difference-in-differences with synthetic control.
- `missing::att_missing_pre` and `att_missing_pre_from_predictions`: the two-period ATT with a missing pre-treatment outcome (Testa, Kennedy and Reimherr), with options for the propensity ceiling and the missingness floor.
- Inference by influence functions: analytic and cluster-robust standard errors, and multiplier-bootstrap simultaneous bands as `did::mboot`. `did_cluster_sums`, `did_standard_error` and `did_covariance` expose the cluster sums.
- `Error` with `Classify` and stable `did.*` codes. Confidence levels are `epistat_core::Level` and bootstraps are `epistat_core::Bootstrap`, with no default seed. Results do not depend on the number of threads.

### Results

No released result changes.
