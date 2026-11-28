# Changelog

## [0.4.0] - unreleased

This release moves the crate onto the shared types of the epistat crates (`Level`, `Bootstrap`, `Covariance`, `Influence`, `Estimand` and one typed `Error`), removes the entry points that 0.3.0 deprecated, and adds frailty, flexible parametric, relative survival, case-cohort, nested case-control, self-controlled case series, crossover and target trial emulation methods.

### Breaking changes

- `Error` has typed variants. `Error::Input(String)` is now `Error::Input(InputError)` for the shared checks, plus `Mismatch`, `OutOfRange`, `Missing`, `Options`, `Value`, `Order`, `Inconsistent`, `NoInformation`, `Fit` and `NotApplicable` for the other refusals, and `Internal`, `Unsupported`, `Undefined`, `Aliased`, `NotConverged` and `Singular` for failures of the computation. Match on the variants, or use `Classify::kind() == ErrorKind::Input` to test for bad input. The text of many messages changed where a check moved to `InputError`.
- Error kinds changed for some refusals. A risk score or inverse link that is not finite, and the three SCCS "information not positive definite" cases, are now `Undefined`. The residuals refusal for ties is `Unsupported`. A covariance entry that cannot be formed in the trial outcome models is `Undefined`, not an input error.
- `Lexis::n_intervals`, `Counting::complete_rows` and `RowWindows::get` return `Result`. Write `lexis.n_intervals(scale)?`.
- Result structs and public enums are `#[non_exhaustive]` (210 types). Literals of a result and matches without a wildcard arm no longer compile. Option, data-view and input types stay exhaustive.
- Seeds are given as `Bootstrap`. `aalen_johansen::bootstrap`, `multistate::bootstrap` and `markov::test` take `bootstrap: Bootstrap` instead of `&[u64]` seeds. Write `Bootstrap::new(replicates, seed)`; replicate `r` draws from `replicate_seed(seed, r)`. The result type `multistate::Bootstrap` is now `Replicates`.
- Confidence levels are `Level`. Fields `conf_level: f64`, `level: f64` and the `alpha` of the expected-hazard and flexible parametric predictions are now `level: Level`, for example `level: Level::confidence(0.95)`.
- Standard errors are named `se`. Fields `std_err`, `standard_error`, `standard_errors`, `robust_standard_errors` and `diagonal_standard_errors` are now `se`, `robust_se` and `diagonal_se`. The option flag `std_err` is `se`. The R names `se_fit`, `se_cumhaz` and `theta_se` are unchanged.
- Covariance matrices in results are `Covariance`, read with `get(i, j)` or `as_slice()` instead of index arithmetic. This covers the results of the Cox, Fine-Gray, case-cohort, flexible parametric, SCCS, relative survival, RMST, pseudo-value, recalibration, Yates and weighting modules, `SurvregFit::variance`, `SurvregFit::robust_variance`, `CoxZph::variance` and `Survdiff::variance`. Values that are not finite read as `None`. Per-time blocks (`WeightedKm`, `Standardised`, `ProbTrans`, the `msreg` cumulative hazards) hold one `Covariance` per time, so `covariance[t].get(a, b)` replaces the pair layout.
- Iteration controls are `max_iterations: usize` and `tolerance: f64` in the options of `cox`, `fine_gray`, `clogit`, `msreg`, `casecohort` and `survreg`. `rel_tolerance` is `tolerance`, and iteration counts in results are `usize`.
- The offset is a column of `SurvData`, set with `SurvDataBuilder::offset` and read with `SurvData::offset`. The `offset` field of the Cox, residual, prediction, `tt`, cause-specific, standardisation, `msreg`, `survreg` and flexible parametric options is gone, and `init` is `start: Option<Vec<f64>>`.
- Options that borrowed per-row data own it. `ZphOptions`, `ExcessOptions`, `CaseCohortOptions`, `CalibrationOptions`, `YatesOptions`, `MsOptions`, `MsTerms`, `MsBaseline`, `FlexParOptions` and `Centring::Given` hold `Vec`s instead of slices, lose their lifetime and are `Clone` but no longer `Copy`.
- `clogit(&MatchedSets, x, &ClogitOptions)` takes the per-row columns in `MatchedSets`. `Collapse::Codes` is removed: collapse by the data's cluster or id. `reference_means(model, &Centring)` takes a `Centring`. `brier`, `null_brier` and `auc` take `&ScoreCensoring`.
- `weighting::km::estimate` and `weighting::cox::fit` take a `Weights` as their last argument, in place of an `Iptw` and subject. Use `Weights::known()`, `Weights::iptw(&iptw, subject)` and `Weights::with_censoring`.
- `weighting::iptw::Estimand` is removed and `IptwOptions::estimand` is `epistat_core::Estimand`. The enum is non-exhaustive, so a match needs a wildcard arm. `Ato` and `Atm` are refused.
- `fine_gray::Rows::event` is `FineGrayRows::status`. `Tmerge::tstart` and `Tmerge::tstop` are `start` and `stop`.
- The `influence` fields of the concordance, the AIPW and the prediction scores are `epistat_inference::Influence` on the per-unit scale, not `Vec<f64>` on the dfbeta scale. The values are n times the old ones for the concordance. Cluster them with `Clusters::per_row` or by position, since a person-coded `Clusters` is refused with `KeyMismatch`.
- `Error::Unsupported` and `Error::Undefined` are struct variants with `what` (the option, column or quantity) and `message`. An information or covariance that is not positive definite is `Error::NotPositiveDefinite { matrix, .. }` (code `survival.not_positive_definite`), and a parameter the data do not identify, such as a frailty variance with singleton clusters, is `Error::NotIdentified { what, .. }` (code `survival.not_identified`). Kinds and messages are unchanged.
- `weighting::trials::PersonPeriod`, `weighting::trials::ProtocolPanel` and `weighting::clone::CloneInput` take a `panel: epistat_core::Panel` in place of their `id: &[u64]` and `period: &[i64]` fields: write `panel: Panel::new(&unit, &period)?` with `u32` unit codes (`epistat_core::Codes::from_labels` turns ids into codes in sorted order). The order of the rows is checked by `Panel::new`. `TrialRows::id` is `unit: &[u32]`, `MsmInput::person` is `unit: &[u32]`, and `PersonCounts`, `bootstrap_trials` and `Cloned::ids` take and give `u32` units. Results do not change for codes in the order of the ids.
- `AipwSurvOptions::truncation` and `PooledOptions::truncation` are `Option<epistat_stats::weighted::ProbabilityBounds>`, built with `ProbabilityBounds::new(lower, upper)?`, in place of `Option<(f64, f64)>`.
- `lexis::rates::LinCi` and `ExpCi` have an `unestimable` field, and `Rates` has `separated` and `n_separated`; `n_used` no longer counts the pieces a separated column leaves out.
- `lowlevel::advance` and `lowlevel::integrate` refuse slices of different lengths, an initial distribution of the wrong length and a horizon before the origin, where the 0.3.0 functions advanced the shorter slice, panicked or returned zeros.

### Added

- `frailty`: shared gamma frailty Cox models as `coxph(frailty(cluster, distribution = "gamma"))`, with the variance of theta from the profile score under Breslow ties, a confidence interval for theta, and `Theta::{Fixed, Profile}`.
- `flexpar`: flexible parametric survival models of Royston and Parmar on the log cumulative hazard, odds or normal scale, with an expected hazard and predictions.
- `relsurv` and `ratetable`: expected survival (Ederer I, Hakulinen), crude mortality, net survival, excess hazard models, and rate tables with expected counts.
- `casecohort`: the five case-cohort methods of `survival::cch`. `ncc`: nested case-control weights, the weighted Cox fit as `multipleNCC`, and absolute risk.
- `sccs`: the self-controlled case series with several exposures, seasons and calendar dates, event-dependent exposures, a sample size calculation and the semiparametric case series. `crossover`: case-crossover and case-time-control windows, stratified referents and control sampling.
- `aipw`: augmented inverse probability weighting of risk with censoring, as riskRegression's `ate`. Attributable fractions from logistic models and from standardised Cox risks in `standardise`.
- `weighting::trials` and `weighting::clone`: per-protocol weights, the marginal structural model of sequential trials with a competing event, weight limits, clone-censor-weight analysis with start windows, durations and dynamic starts, and a bootstrap over persons through `trials::bootstrap_trials`. `trials::fit_msm` takes `MsmRows::{Design, Trials, Clones}`.
- Weights for overlap, matching and optimal-subset estimands in `weighting::iptw`, and inverse probability weights for time-varying treatment.
- `WeightedKm::at`, `survival`, `risk`, `position_at`, `covariance_known_between` and `contrast`, which gives the risk difference and ratio with intervals. `NetSurvival::at` and `position_at`.
- `lowlevel`: `advance`, `integrate`, `float_codes` and checked helpers for callers who form their own sums.
- `gather` on the outputs that expand their input (`split`, `condense`, `lexis`, `fine_gray::rows`, `surv2`, `tmerge`, `clone_censor`, `crossover`, `sccs`), and `to_surv_buf` on those with start, stop and status.
- `clogit` and stratified Cox fits report the number of informative sets.
- `Error::Counted`, `Classify` for `Error` with stable `survival.*` codes, and re-exports of `Bootstrap`, `Covariance` and `Level` at the crate root.

### Changed

- `lexis::rates::fit_rates` finds the columns that are 0 on every piece with events and of one sign on the others, such as the indicator of a group without events. Their coefficients and every contrast with weight on them are `None`, with the reason in `Unestimable`, and the other coefficients are fitted on the pieces where those columns are 0, the limit `glm` approaches. Data without any event are an error. Fits without such a column are unchanged.
- `relsurv::excess_hazard` fixes an interval at the boundary of an excess hazard of 0 when the fit converges with the derivative of the log likelihood in that excess hazard at 0 not positive, and reports it in `at_bound`. Fits that collapsed towards 0 without reaching the old share-of-events rule were reported inside the boundary, with a coefficient near -26 to -34. Fits that do not reach the boundary are unchanged.

- Fine-Gray is fitted by a sweep over the event times, not on the expanded rows, so memory no longer grows with the weighted-row expansion. `fine_gray::fit`, `rows` and `risk_rows` use the stratum of the data as the censoring stratum (`crr`'s `cengroup`). `n_rows` and `n_used` count the rows of the data, as `cox::fit` does.
- Large Cox strata (2^14 rows or more) are swept in parallel blocks of event times, and rows are prepared in parallel. Results are the same on 1, 3 and 8 threads.
- The covariance of a pooled weighted Cox fit with truncated weights includes the derivative of the truncation limits.
- The logistic nuisance models (pooled censoring and treatment weights, AIPW propensity, trial and clone models) use `epistat_glm`'s inverse link, so a fitted probability of exactly 0 or 1 that was refused can now be bounded.
- The trial outcome models use `SandwichType::Cr1` from `epistat-glm`. Recalibration is fitted through `pseudo::regression::fit`, with `epistat_glm::Link` and `SandwichType` in place of the module's own `Link` and `Covariance`.
- `ProbTrans::probability_of` and `se_of` return `None` for an index out of range. Entry points that panicked on caller input (landmark grids, horizons, times and exposures that are not finite, short columns) return an error.
- Types that shared a name across modules carry the method as a prefix, for example `CoxPrediction`, `SurvregPrediction`, `MarkovOptions`, `MultistateOptions`, `SurvdiffOptions`, `OccupationEstimate`, `FineGrayRows`, `ExposureLevel`, `RoystonAdjustment`, `PseudoRegression` and `PseudoRegressionOptions`. The 0.3.0 names remain as deprecated aliases.
- The minimum supported Rust version is 1.89.
- `weighting::km::estimate` computes IPTW-only weights and general weights through separate cores. Where both apply they agree to rounding, at most 4e-16 relative in nine covariance entries of the test case.

### Deprecated

All of these are removed in the next release that breaks the API.

- `aalen_johansen`: `VarianceOptions` (use `AalenJohansenVarianceOptions`), `Variance` (use `AalenJohansenVariance`), `advance` (use `lowlevel::advance`).
- `multistate`: `Options` (use `MultistateOptions`), `Estimate` (use `OccupationEstimate`), `Bootstrap` (use `Replicates`), `VarianceOptions` (use `OccupationVarianceOptions`), `Variance` (use `OccupationVariance`), `integrate` (use `lowlevel::integrate`).
- `markov`: `Options` (use `MarkovOptions`), `Test` (use `MarkovTest`).
- `cox::predict` and `cox::residuals`: `HazardType` (use `CoxHazardType`), `Variance` (use `CoxVariance`), `PredictType` (use `CoxPredictType`), `PredictOptions` (use `CoxPredictOptions`), `SurvfitOptions` (use `CoxSurvfitOptions`), `Prediction` (use `CoxPrediction`), `ResidualKind` (use `CoxResidualKind`).
- `survreg::predict`: `PredictType` (use `SurvregPredictType`), `PredictOptions` (use `SurvregPredictOptions`), `Prediction` (use `SurvregPrediction`), `ResidualKind` (use `SurvregResidualKind`).
- `fine_gray`: `Rows` (use `FineGrayRows`), `float_codes` (use `lowlevel::float_codes`).
- `pseudo` and `pseudo::regression`: `Link` (use `epistat_glm::Link`), `Method` (use `PseudoMethod`), `Covariance` (use `epistat_glm::robust::SandwichType`), `GeeOptions` (use `PseudoRegressionOptions`), `GeeFit` (use `PseudoRegression`), `PseudoRegression::standard_errors` (use `se`), `pseudo::ipcw::IpcwOptions` (use `PseudoIpcwOptions`).
- `metrics`: `score::Censoring` (use `ScoreCensoring`), `contrast::Contrast` (use `ScoreContrast`), `royston::Adjustment` (use `RoystonAdjustment`), `Recalibration::std_err_intercept` and `std_err_slope` (use `se_intercept` and `se_slope`).
- `standardise`: `Level` (use `ExposureLevel`), `Contrast` (use `StandardiseContrast`), `yates::Test` (use `YatesTest`). `rmst::Contrast` (use `RmstContrast`). `survdiff::Options` (use `SurvdiffOptions`).
- `weighting`: `estimated::Censoring` (use `OutcomeCensoring`), `estimated::Estimated` (use `Weights`), `cox::WeightedCox` (use `EstimatedCox`), `cox::fit_estimated` (use `cox::fit`), `km::estimate_estimated` and `km::estimate_known` (use `km::estimate`, with `Weights::known()` for known weights), `WeightedKm::standard_error` and `standard_error_known` (use `se` and `se_known`).
- `Tmerge::tstart` and `tstop` (use `start` and `stop`), `ProbTrans::std_err_of` (use `se_of`), the pair-index layout of the `msreg` cumulative hazard covariance (use `covariance[t].get(a, b)` or `covariance_of`).

### Removed

- The entry points that 0.3.0 deprecated. `cox::fit_cox`, `CoxDesign` and `cox_score_residuals`: use `cox::fit` and `cox::score_residuals` with a `SurvData` and `CoxOptions`. `cox::fit_cox_single_covariate`, `CoxData` and `CoxFit`: fit a one-column matrix with `cox::fit`. `fine_gray::fine_gray_fit`, `FineGrayData` and `FineGrayData::expand`: use `fine_gray::rows`, `risk_rows` and `fit`. `aalen_johansen::curves`, `ArmColumns` and `bootstrap_arm_incidence`: use `estimate` and `bootstrap` with a `SurvData`. `markov::markov_test` and `Stays`: use `markov::test`. `multistate::state_occupation_curve`, `bootstrap_state_occupation`, `Intervals` and `Occupation`: use `multistate::estimate` and `bootstrap`.
- `weighting::iptw::Estimand`, replaced by `epistat_core::Estimand`.
- The per-row `Collapse::Codes`, and the `offset` and `init` fields of the options that now read them from `SurvData`.

### Fixed

- `fine_gray` ignored the stratum of the data. It is now the censoring stratum.

### Results

- `cox::fit` and the fits built on it (`clogit`, stratified fits, `weighting::cox::fit`): a stratum of 2^14 rows or more is summed in blocks, so coefficients, covariances and log-likelihoods of fits of 2e4 to 2e5 rows change by 4 to 214 machine epsilons relative, at most about 5e-14. Fits with fewer rows per stratum are unchanged.
- `cox::fit` with many columns: the LDL' factorisation is a left-looking loop. In a test design with 18 columns the worst change is 1.4e-15 relative for the plain fit and 1.9e-13 for a fit with every option. Other pinned Cox, Fine-Gray and `survreg` fits do not change.
- `fine_gray::fit`: the sweep over event times changes coefficients and standard errors by at most 1.3e-14 relative. A fit on data with strata now uses them as censoring strata, which changes the result by more than rounding.
- `weighting::pooled::fit`: the covariance of the weighted Cox coefficients with truncated weights changes by up to 1.5 per cent on the test data, because the limits' derivative is included. Weights and coefficients are bit-identical on that path. The treatment weights' coefficients change by up to 9.4e-14 relative because of glm's inverse link.
- `concordance`: the returned influence values are n times the old values, since they are now on the per-unit scale. The variance and standard error are unchanged.

## [0.3.0] - 2026-11-22

- Pseudo-observations by inverse probability of censoring weighting, pseudo-values of state occupation, time in state and transition hazards, and the regression of the restricted mean on covariates.
- Pooled censoring weights, weighted Cox and Kaplan-Meier fits with estimated weights, and prediction metrics with a Cox censoring model, calibration and contrasts.
- Greenwood variances, fixed horizons and several profiles in `msreg`.
- Cox predictions with each person's own offset, an offset in the cause-specific hazard, and Cox curve limits that follow the `survfit` formula.
- Unreleased entry points that took a `SurvData` replaced the deprecated slice-based ones for Cox, Fine-Gray, Aalen-Johansen, multistate and Markov.

## [0.2.3] - 2026-11-20

- The state occupation integral is public, and a decomposition by part.

## [0.2.2] - 2026-11-19

- Dependency update for `epistat-rng`.

## [0.2.1] - 2026-11-19

- Cox models are fitted on a design matrix, and the single-covariate fit is deprecated.

## [0.2.0] - 2026-11-19

- The caller gives restart seeds, and degenerate survival fits return `None`.

## [0.1.0] - 2026-11-19

- First release: Cox, Fine-Gray, Aalen-Johansen, multi-state models, Kaplan-Meier, survreg, `msreg`, concordance, `pyears`, `survdiff`, Lexis splitting and prediction metrics.
