# Changelog

## [0.1.0] - unreleased

First release. The crate holds the epidemiological measures and sensitivity analyses. Some of them were in `epistat-stats` 0.2.5 under the same module names and are now deprecated there.

### Added

- `rates`, rates with exact intervals, rate ratios (with an exact interval and an exact test) and differences, directly standardised rates with Fay-Feuer intervals, and the ratio and difference of two standardised rates.
- `mantel_haenszel`, Mantel-Haenszel estimators of the odds ratio, risk ratio, risk difference, rate ratio and rate difference, with the Breslow-Day test and tests of a common rate ratio over strata.
- `sir`, standardised incidence ratios with exact, Byar, Wald and profile intervals, and tests of homogeneity and trend.
- `standardisation`, the weights of direct standardisation onto the exposed.
- `decomposition`, the Kitagawa decomposition of a difference in rates with `kitagawa_variance` for the variances of its parts and `kitagawa_poisson` for Poisson rates.
- `evalue`, E-values for risk, odds and hazard ratios and for mean differences, plus `evalues_rd` for a risk difference and `selection_evalues_rr` for selection bias.
- `bias`, quantitative bias analysis of misclassification, selection, an unmeasured confounder and misclassified person-time, simple and probabilistic. The probabilistic analyses take `bias::ProbOptions`, which holds a `Bootstrap`, the random-error setting and a `Level`, and return the draws and the risk difference.
- `rosenbaum`, Rosenbaum bounds for matched designs (Wilcoxon, Hodges-Lehmann and M-statistic scores, with sets of different sizes) and the sensitivity values `wilcoxon_sensitivity_value`, `hodges_lehmann_gamma` and `scores_sensitivity_value`.
- `signed_rank`, the signed rank statistic with its exact and normal upper tails.
- `balance`, standardised mean differences, variance ratios, eCDF and eQQ distances, and Cramer's V, with weighted forms.
- `tables`, the `Counts2x2` and `PersonTime2` tables every measure starts from.
- `Error`, one error type with stable codes that implements `epistat_core::Classify`. Bad input is reported as `Error::Input` with the row where there is one.

### Changed

Compared with the same items in `epistat-stats` 0.2.5:

- The errors `EValueError`, `SensitivityError` and `SignedRankError` are replaced by `Error`.
- Intervals take an `epistat_core::Level` and results are `epistat_core::Estimate` values, with `None` for an undefined number.
- Result structs of `rosenbaum`, `decomposition` and `evalue` are `#[non_exhaustive]`.

### Results

No released result changes. The functions that were also in `epistat-stats` 0.2.5 use the distribution functions and sums of `epistat-stats` 0.3, and differ from the 0.2.5 values in the last bits for the reasons given in that crate's changelog.
