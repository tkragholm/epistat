# Changelog

## [0.3.0] - unreleased

The distribution functions move into a `dist` module, the epidemiological methods move to the new crate `epistat-epi`, and the crate has one error type. Every path of 0.2 still compiles and is deprecated.

### Breaking changes

- `cluster::KMeansError` is replaced by `epistat_stats::Error`, a non-exhaustive enum with the variants `Input` (wrapping `epistat_core::InputError`) and `EmptyCluster`. `kmeans` returns `Result<KMeans, Error>`. The old variants `Empty`, `Shape`, `NotFinite`, `Centres` and `NoIterations` are now `Error::Input`. Match on `Error` or read `code()` through `epistat_core::Classify`.
- `likelihood::InformationCriteria` is `#[non_exhaustive]`. Read its fields and do not build it with a struct literal.
- The minimum supported Rust version is 1.89, up from 1.85.
- The crate depends on `epistat-core` and `epistat-rng`.

### Added

- `dist`, one module per distribution: `normal`, `t`, `chi_square`, `f`, `beta`, `gamma`, `poisson` and `binomial`, with density or mass, cumulative and upper tail probabilities and quantiles, `gamma::digamma` and `gamma::trigamma`, `beta::regularised` and `poisson::cipoisson`, which takes a `Level` and returns an `epistat_core::Interval`. These functions return NaN for a parameter outside the domain, as R does.
- `compat`, the older approximations kept for callers that need their numbers: `cdf_as26`, `erf_as26`, `quantile_acklam` and `rounded_rank`.
- `sum::lanes`, sums, sums of squares, centred squares and dot products over eight fixed lanes, so the result does not depend on the machine.
- `moments`, `mean`, `variance` and `mean_sd` over those lanes, with `Divisor`.
- `logistic`, `expit`, `logit`, `log1p_exp` and `log_expit`.
- `wald`, an estimate with its Wald interval and p-value for a given `Level` and `Scale`.
- `weighted::effective_sample_size`, `weighted::WeightLimits` (clip weights at quantiles or values, with the preset `P99`) and `weighted::ProbabilityBounds`.
- `Error`, which implements `Classify`.
- `weighted::try_mean`, the weighted mean with the reason when there is none, and `weighted::ROUNDING_LIMIT`. `Error` has the variants `FixedPointRounding` and `FixedPointOverflow` for it.

### Changed

- `mcmc::autocorrelation`, `effective_sample_size` and `split_rhat`, `quantile::weighted_type1`, and the weighted balance statistics sum over eight fixed lanes. See Results.

### Deprecated

Deprecated items are removed in the next release that breaks the API.

- `normal::pdf`, `cdf`, `quantile`, `truncated_quantile`, `two_sided_z` and `folded_quantile`, use the functions of the same name in `dist::normal`. The new `truncated_quantile` and `folded_quantile` return NaN where the old ones returned `None`, and `two_sided_z` takes a `Level`.
- `normal::cdf_as26`, `erf_as26` and `quantile_acklam`, use `compat`.
- `gamma::ln_gamma`, `lower_regularised`, `upper_regularised`, `chi_square_cdf` and `chi_square_sf`, use `dist::gamma::ln_gamma`, `dist::gamma::cdf(x, a)`, `dist::gamma::sf(x, a)`, `dist::chi_square::cdf` and `dist::chi_square::sf`.
- `student::t_ln_density`, `t_cdf_sf`, `t_quantile` and `regularised_beta`, use `dist::t::ln_pdf`, `dist::t::cdf` and `dist::t::sf`, `dist::t::quantile` (NaN instead of `None`) and `dist::beta::regularised`.
- `poisson::cipoisson`, `Method`, `Interval`, `PoissonError` and `gamma_quantile`, use `dist::poisson::cipoisson` with a `Level`, `dist::poisson::Method`, `epistat_core::Interval`, `epistat_stats::Error` and `dist::gamma::quantile`.
- `quantile::rounded_rank`, use `compat::rounded_rank`.
- `cluster::KMeansError`, use `epistat_stats::Error`.
- Everything in `balance`, `decomposition`, `evalue`, `rosenbaum`, `signed_rank` and `standardisation` has moved to the same module name in `epistat-epi`. The deprecation note on each item gives its new path.

### Fixed

- `weighted::mean` returned 0 or a value with few correct digits when its terms `x w` or its weights were near `1e-15`, since each is rounded to 15 decimals: `mean(&[1e-9, 2e-9, 3e-9], Some(&[1e-7; 3]))` was `Some(0.0)`. It now returns `None` when the rounding bound of a sum, half a unit in the 15th decimal per term, exceeds `1e-12` of the sum of the magnitudes of its terms. Values that pass the check are the same bits as before.
- `weighted::mean` and `weighted::variance` dropped a row with a negative weight as if its weight were 0. They now return `None`, and `try_mean` returns `Error::Input` with `InputError::Domain`, as every other weight check of the workspace does. A weight of 0 still drops its row.
- `quantile::type7`, `type7_ratio`, `type1`, `weighted_type1` and `compat::rounded_rank` read order statistics from the sample as given, so an unsorted sample gave a wrong number: `type7(&[5.0, 1.0, 2.0, 9.0, 3.0], 0.25)` was 1. They now return `None` for a sample with a value below the one before it, checked in the same pass as the check for NaN. `quantile::type7_unsorted` takes a sample in any order.
- The regularised incomplete gamma functions `dist::gamma::cdf` and `sf`, and with them the chi-square and Poisson tails, the F tail at large degrees of freedom, the gamma and chi-square quantiles and the exact Poisson limits, are written again from Gil, Segura and Temme (2012): the uniform asymptotic expansion near `x = a` from a shape of 12, the series for `P`, a series for `Q` at small shapes and the continued fraction for `Q`, with the prefactor `x^a e^-x / Γ(a + 1)` formed as a product below a shape of 10 and about `x = a` above. Against mpmath at 60 digits the relative error of the smaller tail is below 6e-15 for shapes below 12 and 2e-13 for shapes to 1e7, where it was up to 2e-14, 1e-11 at 1e4 and 1.7e-8 at 1e7. `P(1e15, 1e15)` is 0.5000000042 where it was 0.0485, `P(1e300, 1e300)` is 0.5 where it was NaN, `Q(1e-300, 1e-300)` is 6.9e-298 where it was -2.35e-14, and a subnormal shape gives `Q = a E1(x)` where it gave NaN. An infinite shape gives `P = 0` and `Q = 1` at finite `x`, as R does.
- The Poisson and binomial point probabilities take their Stirling correction and deviance term from the same helper, with the Stirling series from 10 on.
- The incomplete beta function, and with it the t, F and binomial tails, lost accuracy for large shapes. At a shape of 1000 the largest error against a 50-digit reference falls from about 2,000 units in the last place to about 20. For a and b of 1e7 it falls from about 1.6e-8 relative to below 2e-12. The t tail at 1e5 degrees of freedom was off by about 5e-10 relative to R.

### Results

- `dist::gamma::cdf`, `sf` and `quantile`, `dist::chi_square`, the `dist::poisson` tails and `cipoisson`, `dist::f` with a degree of freedom above 4e5 (quantile) or 1e7 (tail), and the point probabilities of `dist::poisson` and `dist::binomial` (and the deprecated `gamma` and `poisson` paths), because the incomplete gamma function and its quantile are computed by the new method described under Fixed. For shapes below 12 the values change by up to about 1e-14 relative, mostly towards the exact value: on the chi-square tails with 1 to 12 degrees of freedom and `x` from 0 to 40 the largest change is 7.8e-15 (66 units in the last place), where the old values were up to 64 units and the new are up to 27 units from mpmath at 60 digits. Exact Poisson p-values far in the tail change by up to 1.7e-13 relative (a p-value of 1e-73, now 1.3e-14 from the exact value, before 1.8e-13). For larger shapes the change is up to the old error described under Fixed.
- `dist::beta::regularised`, `cdf` and `quantile`, `dist::t`, `dist::f` and `dist::binomial` tails and quantiles (and the deprecated `student::regularised_beta`, `t_cdf_sf` and `t_quantile`): in the last bits, typically 1e-15 to 5e-15 relative, at most 1.4e-14 on a derived quantile, because the incomplete beta function is evaluated by a new continued fraction and prefactor. For a shape of 1000 or more the change is larger, as described under Fixed.
- `mcmc::autocorrelation`, `mcmc::effective_sample_size` and `mcmc::split_rhat`: in the last bits, because the sums run over eight fixed lanes. Chains shorter than 8 draws are unchanged.
- `quantile::weighted_type1`: the total weight is summed over eight lanes, so it can differ in the last bit. The selected observation changes only if the target falls within that rounding of a cumulative weight.
- Balance statistics that sum weights or gaps (`balance::eqq_distance`, `ecdf_distance`, `eqq_distance_weighted`, `cramers_v_weighted`) and `signed_rank::normal_upper_tail`: in the last bits, for the same reason.

## [0.2.5] - 2026-11-22

- Added Poisson confidence intervals, the gamma quantile and `normal::two_sided_z`.
- Added the t distribution and the incomplete beta function, E-values, the signed rank distribution with Rosenbaum bounds, k-means from given starts, and covariate balance on unit weights with a weighted eQQ distance.
- Added a streaming compensated sum.
- Fixed a Cramer's V that depended on the scale of the weights, an overflow in an exact tail, and k-means with zero passes.

## [0.2.4] - 2026-11-20

- Kitagawa decompositions by part, and the occupation integral made public.

## [0.2.3] - 2026-11-19

- Added weighted moments, standardisation, decompositions and SAS dates.

## [0.2.2] - 2026-11-19

- Added a weighted quantile.

## [0.2.1] - 2026-11-19

- A constant MCMC chain is now recognised by comparing its draws.

## [0.2.0] - 2026-11-19

- Undefined results are `None` instead of NaN, and invalid weights are refused.
- The truncated normal quantile returns `None` for a NaN bound.

## [0.1.1] - 2026-11-19

- Added the gamma and chi-square functions and the Wald test.

## [0.1.0] - 2026-11-19

- First release: sample quantiles, the normal distribution, sums, truncated and folded normal quantiles, log-sum-exp, softmax and information criteria, covariate balance and MCMC diagnostics.
