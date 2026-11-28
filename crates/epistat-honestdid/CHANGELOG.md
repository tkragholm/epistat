# Changelog

## [0.1.0] - unreleased

First release of `epistat-honestdid`. It holds the HonestDiD sensitivity analysis that `did_methods` (last release 0.2.0 on crates.io) kept in its `honest` module, which `epistat-did` continues without. Take an `epistat_core::EventStudy` from `epistat-did` or any other estimator.

### Added

- `honest::original`, `honest::summarize` and `honest::joint_path`: the sensitivity analysis of Rambachan and Roth, with smoothness and relative-magnitude restrictions and their sign and monotonicity variants, the identified set, fixed-length and conditional confidence sets and their hybrids for any linear combination of the post-treatment coefficients, the breakdown value, and joint confidence regions for the post-treatment path.
- `honest::smoothness_bound_from_pretrends`.
- `pretrends::power`: the power of the pre-trends test and the bias after passing it (Roth, 2022), with the probability that no pre-period coefficient is significant, the conditional means, the Bayes factor, the likelihood ratio and the slope of a linear violation detected with a given power. Every probability comes with the standard error of its evaluation.
- `Error` with `Classify` and stable codes. A failed solver is `Error::Solver { program, status }` with a `SolverStatus` of `Setup`, `Infeasible`, `Unbounded` or `NotConverged`.
- Simulations take a `Bootstrap` or simulation setting from the caller, and there is no default seed. Only the methods that draw need one: `HonestOptions::least_favorable` and `JointPathOptions::least_favorable` are `Option<Bootstrap>`, read by the `C-LF` hybrid alone, and both options have a `Default`. Pointwise levels are `epistat_core::Level`. Results do not depend on the number of threads.
- `Restriction` takes its values as a `Grid`, `Grid::Default` for `HonestDiD`'s default grid or `Grid::Values`, which must not decrease. `SensitivitySummary::breakdown` returns a `Breakdown { value, exact }`, where `exact` is false when every row excludes zero.
- An event study whose relative periods, with the reference among them, are not consecutive is refused, since the restrictions compare neighbouring periods.
- Rows of the identified set have no confidence interval: `SensitivityRow::interval` is `None` for them and the set is in `identified`.
- `smoothness_bound_from_pretrends` takes its level as a `OneSided` level, and `PretrendOptions::critical` is a `Critical::Level` or `Critical::Value` in place of a level and a threshold that overrode it. `Surface::new` takes a `Seed`.

### Results

No released result changes.
