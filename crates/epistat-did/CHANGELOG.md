# Changelog

## [0.1.0] - unreleased

First release of `epistat-did`, which continues `did_methods` (last release 0.2.0 on crates.io). Replace `did_methods` with `epistat-did` in `Cargo.toml`. The HonestDiD sensitivity analysis, which `did_methods` held in its `honest` module, is the separate crate `epistat-honestdid`.

### Added

- `att_gt`: group-time average treatment effects ATT(g,t) for staggered adoption (Callaway and Sant'Anna) by outcome regression, inverse probability weighting or the doubly robust estimator, on panels and repeated cross-sections, with never-treated or not-yet-treated comparison groups, varying or universal base periods, anticipation and sampling weights, and aggregation by event time, cohort, calendar time or overall as `aggte` does. `AttGt::event_study` returns the event study that `epistat-honestdid` takes.
- `two_period::estimate`: the two-period doubly robust estimators of `DRDID` on a `TwoPeriodData` (panel or repeated cross-section), with `Estimator::{Improved, DoublyRobust, TraditionalDoublyRobust, InverseProbabilityWeighting, OutcomeRegression}`. `Improved` is the default. Every estimator's normal interval is at `level`.
- `did_cc::estimate`: compositional changes in repeated cross-sections (Sant'Anna and Xu), with cross-fitting. A propensity model that fails to fit is an error unless `DidCcOptions::propensity_fallback` is set, in which case the fallbacks used are listed in `DidCc::propensity_fallbacks`.
- `triple_did::estimate` for doubly robust triple differences, and `continuous::estimate` for the average causal response to a continuous treatment by a sieve estimator, with a polynomial or B-spline basis.
- The efficient ATT(g,t) that combines all baseline periods (Chen, Sant'Anna and Xie), with aggregations whose standard errors use the combined influence function.
- `dcdh::estimate`: the event-study estimator of de Chaisemartin and D'Haultfoeuille for treatments that switch on and off or take several values, as `did_multiplegt_dyn` of `DIDmultiplegtDYN`, with placebos, normalised effects, controls, strata, weights, clusters, joint tests, heterogeneity regressions and a bootstrap.
- `twfe` and `bacon`: static and event-study two-way fixed-effects regressions as `fixest` fits them, and the Goodman-Bacon decomposition as `bacondecomp::bacon`.
- `sunab`, `etwfe`, `imputation`, `two_stage` and `synthdid`: the interaction-weighted estimator of Sun and Abraham, Wooldridge's extended two-way fixed effects (linear, Poisson and logit), the imputation estimator of Borusyak, Jaravel and Spiess, the two-stage estimator of Gardner, and synthetic difference-in-differences with synthetic control.
- `missing::estimate` and `missing::from_predictions`: the two-period ATT with a missing pre-treatment outcome (Testa, Kennedy and Reimherr) on a validated `MissingData`, which holds the covariates and cluster codes. `MissingOptions` holds the missing-at-random assumption, the covariates of each nuisance model (`NuisanceModels`, every covariate by default), the cross-fitting folds, the propensity ceiling and the missingness floor. `missing::cross_fit_folds` gives the folds. `MissingAtt::influence` is an `Influence` keyed by row under `ClusterRule::NoSmallSample`.
- Inference by influence functions: analytic and cluster-robust standard errors, and multiplier-bootstrap simultaneous bands as `did::mboot`. `did_cluster_sums`, `did_standard_error` and `did_covariance` expose the cluster sums.
- `Error` with `Classify` and stable `did.*` codes. Confidence levels are `epistat_core::Level` and bootstraps are `epistat_core::Bootstrap`, with no default seed. Results do not depend on the number of threads.

### Deprecated

Deprecated items are removed in the next release that breaks the API.

- `att_missing_pre`, `att_missing_pre_from_predictions` and `missing_cross_fit_folds`, use `missing::estimate`, `missing::from_predictions` and `missing::cross_fit_folds`. They take the earlier `MissingPreData` and `MissingModels`, also deprecated, and give the same bits.

### Results

- The improved two-period estimators (`Estimator::Improved`) form their interval with the normal quantile of the level asked for, as the other two-period estimators do, where they reported `att +- 1.96 se` labelled 95 per cent whatever the level, as `DRDID`'s `drdid_imp_panel` and `drdid_imp_rc` do. At 95 per cent each limit moves by `(1.96 - 1.959964) se`, about `3.6e-5 se`; at other levels the interval is the one asked for. Estimates and standard errors do not change.
- The Wald p-value of the pre-test of `att_gt` and the Hausman p-value of `did_cc` change by at most 9e-16 relative (8 units in the last place), because the chi-square tail in `epistat-stats` is computed by a new method.
