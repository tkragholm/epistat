# Changelog

## [0.3.0] - unreleased

This release gives the crate one `Error`, a smaller public surface, case and control naming throughout, and the shared `Estimand`, `Bootstrap` and `Level` types. Nothing that 0.2.0 computed changes in value, apart from the last bits of the effect bootstrap.

### Breaking changes

- One `Error` replaces the crate's 27 error types and `TrialsError<E>`. Input refusals are `Error::Input(InputError)`, and the errors of `epistat-glm`, `epistat-optim`, `epistat-inference`, `epistat-stats` and `epistat-epi` are wrapped. `Classify` gives a kind and a `matching.*` code. Match on `Error` variants where you matched on the old types.
- `MatchJob::run` returns `Result<MatchOutcome, Error>`. A duplicate id, an event before birth, invalid criteria or common support that cannot be set were an outcome with no pairs and one exclusion reason, and are now errors. Write `job.run()?`. A case that finds no control is still counted, not an error.
- `Estimand` is `epistat_core::Estimand`, which is `#[non_exhaustive]`, has no `Default` and has an `Ato` variant. Matchers refuse an estimand they do not target with `Error::UnsupportedEstimand`, so a function that took the crate's own enum now takes the core one. The ATO is refused everywhere, and `AbadieImbens` refuses the ATM, which 0.2.0 estimated as the ATT. `MatchWeightMethod::for_estimand`, `BalanceWeights::from_outcome` and `balance_report` return `Result`, and the subclass weight accessors return `None` outside the ATT, ATC and ATE.
- The serde form of `Estimand` is snake_case, so `"att"` is read and `"Att"` is no longer read.
- Seeds have no default. `GeneticSearch` has no `Default`: write `GeneticSearch::new(seed)`. `MatchSpec::seed` is an `Option<u64>`, and a spec without a seed is refused with `Error::Option { option: "seed" }` for every method. `ElasticNetCvConfig::new(seed)` replaces a default seed.
- The effect bootstrap takes `Bootstrap` and `Level`. `BootstrapSpec { bootstrap: Bootstrap, level: Level }` replaces the replicate count, seed and `alpha`, `BootstrapSpec::new(Bootstrap::new(replicates, seed))` gives intervals at `Level::alpha(0.05)`, and `cluster_resamples(n, Bootstrap)` takes the same type. The summary intervals are `epistat_core::Interval`, so `summary.normal.lower` is an `Option<f64>` where it was NaN when undefined.
- The names say case and control. Items called anchor, candidate, comparator or treated are renamed, for example `MatchedPair::case_id` and `control_id`, `UnitRole::Case` and `Control`, and `anchor_ids` is `case_ids`. The neutral accessors of `MatchedPair` and `MatchOutcome` are removed. `UnitWeightSet` has private ordered maps read through `cases()`, `controls()`, `case_weight()` and `control_weight()`.
- Duplicated entry points are merged. `balance_report` takes its options (`balance_report_on_weights` is `balance_summary(..).report`, and `run_with_balance`, `balance_report_with_options` and `match_exposure_density` are removed). `match_trials` takes the `TrialMatching` of `match_trials_with`. `prepare_propensity_distance_config` takes the discard policy. The request structs, `match_standard` and `match_transition` are crate-private: use the `MatchJob` builder or `run_spec`. The crate root no longer re-exports every item, and the prelude holds fourteen names.
- `MatchingRecord::stratum` is required, and `strata` defaults to an empty map. Use `delegate_matching_record!` in a wrapper type to forward every method.
- Options are plain structs with public fields and `Default`. The `bon` builders of the estimation configs, `MatchingCriteria` and `BalanceReportOptions` are removed: use struct update syntax, `BalanceReportOptions { .. Default::default() }`.
- Sequential trials take a `TrialPanel` keyed by `epistat_core::Panel`. `expand_trial_columns`, `matched_trial_columns` and `match_trials` take it, and `ExpandedRow`, `expand_trials`, `PanelColumns`, `TrialPerson` and `TrialPeriod` are removed.
- `SpecInput` is `#[non_exhaustive]`, so a `match` on it outside the crate needs a wildcard arm. Its variants are still built as before. `MatchSpec` stays exhaustive.
- 72 result and reason types are `#[non_exhaustive]`, among them `MatchDiagnostics`, the outcome, balance, subclass, weight, genetic, cardinality, propensity and trial types. Build a `MatchOutcome` from pairs made elsewhere with `MatchOutcome::from_pairs(pairs, estimand)`.
- The unused `reporting` module and feature, `KsTest` and the two-sample Kolmogorov-Smirnov test (it is in `epistat-epi`), and the projected balance rows are removed.
- The minimum supported Rust version is 1.89. The `nightly` feature is removed.
- The constraint rules are in one `constraints` module. `ExactMatchConstraint` is no longer a unit struct: write `ExactMatchConstraint::REQUIRED_STRATA` for the rule on the criteria's required strata, and `ExactMatchConstraint::on_key(key)` for one key. `MatchJob::with_gender_match` adds `ExactMatchConstraint::on_key("sex")`, so its type parameter names that type instead of `GenderMatch`.

### Added

- `MatchSpec.discard` and `reestimate` apply to subclass, full, optimal, quick and genetic matching as they do to nearest matching. Exact, CEM and cardinality matching still refuse `discard`.
- Diagnostics and the subclass, full and quick outcomes list the ids of the units that common-support trimming leaves out (`TrimmedUnits`).
- `match_trials_with` takes `TrialMatching` with `PeriodDates::PerPerson` for each person's own period dates and `PeriodStrata` for strata that change between trials.
- One-call sensitivity values for matched sets: the signed rank test, the m-statistic and its interval, the Mantel-Haenszel test for binary outcomes, Gehan's test for censored times, and the gamma at which the Hodges-Lehmann bound reaches a value.
- `ExactMatchConstraint::on_key`, `allowing_unknown` and `with_reason`, the rule `GenderMatch` was, for any strata key.
- An `Option` of a constraint is a constraint group; `None` allows every pair.
- `ResidenceSpell::from_register` turns `epistat_register`'s residence spells into the periods `MustBeResidentInSpells` reads, with `None` for a person without spells.
- `CompleteDataDf` lets the caller choose the complete-data degrees of freedom in `pool_effects`.
- `Classify` and stable `matching.*` error codes. `MatchJob`'s constraints have a `check()` that refuses a negative or non-finite caliper or date window before matching.
- A field caliper in standard deviations works with `reestimate`.
- An undefined average number of controls per case is reported as `None`.

### Changed

- Quick matching drops a stratum with too few units instead of failing the run.
- The balance statistics, the type 7 quantile of CEM and subclassification, and the Brent minimiser of CBPS are the shared ones from `epistat-stats` and `epistat-optim`. Results are unchanged.

### Deprecated

All of these are removed in the next release that breaks the API.

- `Estimand` at the crate root (use `epistat_core::Estimand`).
- `UnitRole::Anchor` and `UnitRole::Candidate` (use `UnitRole::Case` and `UnitRole::Control`).
- `MatchWeightMethod::AnchorUnitCandidateFractional` and `CandidateUnitAnchorFractional` (use `CaseUnitControlFractional` and `ControlUnitCaseFractional`).
- `effective_sample_size` (use `epistat_stats::weighted::effective_sample_size` on a slice).
- `constraints::GenderMatch` (use `ExactMatchConstraint::on_key(key)`, with `.allowing_unknown()` for `allow_unknown` and `.with_reason("gender_mismatch")` to keep the reason it is counted under).
- `constraints::MissingPolicy` (use `CaliperMissing`, the same enum).
- `effects::cluster_resamples` (use `epistat_rng::resample_counts(n, epistat_rng::replicate_seed(seed, r))`, the same draws as counts).
- The spec keys `min_anchors` and `min_candidates` are still read, as aliases of the case and control names.

### Removed

- The crate's own `Estimand` enum, `CbpsEstimand` and `effects::Target`, in favour of `epistat_core::Estimand`.
- `AnchorRecord`, `CandidateRecord`, `StrataExactConstraint`, `UsedControlsVec`, and the aliases of the exact, quick and full outcome types.
- The `bon` dependency.

### Fixed

- A subclass or optimal run with `reestimate` keeps the first-fit scores of the trimmed units, as `matchit()` reports them.

### Results

The effect bootstrap draws its replicates as counts with `epistat_rng::resample_counts`. The pairs drawn are those of 0.2.0, but a replicate's rows are grouped by pair rather than in the order of the draws, so the refitted model sums them in another order and the last bits of the replicates, their standard error and the intervals can change. No other released result changes.

## [0.2.0] - 2026-11-22

- Optimal and full matching by minimum-cost flow, fine, near-fine and refined balance, coarsened exact matching, generalised full matching and subclassification as in MatchIt.
- Genetic matching, cardinality and profile matching on the crate's own solver, the Abadie-Imbens estimator, vector matching for more than two groups, and matching after multiple imputation.
- Effect estimates with cluster-robust variances, the bootstrap of matched sets, sensitivity analysis for matched samples, and sequential trial emulation.
- The covariate balancing propensity score, and a propensity fit through `epistat-glm` with links, weights and refit.
- k-d tree search, sweeps for risk-set matching, and a columnar cohort that can be built from Arrow arrays.

## [0.1.0] - 2026-11-20

- First release as `epistat-matching`. It continues `cohort_matching`, which was published separately on crates.io up to 0.8. Replace `cohort_matching` with `epistat-matching` in `Cargo.toml`.
- Nearest-neighbour, exact and risk-set matching with propensity scores, balance reports and effect estimates.
