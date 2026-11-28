# Changelog

## [0.1.0] - unreleased

First release of `epistat-honestdid`. It holds the HonestDiD sensitivity analysis that `did_methods` (last release 0.2.0 on crates.io) kept in its `honest` module, which `epistat-did` continues without. Take an `epistat_core::EventStudy` from `epistat-did` or any other estimator.

### Added

- `honest::original`, `honest::summarize` and `honest::joint_path`: the sensitivity analysis of Rambachan and Roth, with smoothness and relative-magnitude restrictions and their sign and monotonicity variants, the identified set, fixed-length and conditional confidence sets and their hybrids for any linear combination of the post-treatment coefficients, the breakdown value, and joint confidence regions for the post-treatment path.
- `honest::smoothness_bound_from_pretrends`.
- `pretrends::power`: the power of the pre-trends test and the bias after passing it (Roth, 2022), with the probability that no pre-period coefficient is significant, the conditional means, the Bayes factor, the likelihood ratio and the slope of a linear violation detected with a given power. Every probability comes with the standard error of its evaluation.
- `Error` with `Classify` and stable codes. A failed solver is `Error::Solver { program, status }` with a `SolverStatus` of `Setup`, `Infeasible`, `Unbounded` or `NotConverged`.
- Simulations take a `Bootstrap` or simulation setting from the caller, and there is no default seed. Pointwise levels are `epistat_core::Level`. Results do not depend on the number of threads.

### Results

No released result changes.
