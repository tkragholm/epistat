# Changelog

## [0.1.0] - unreleased

First release. The crate holds the epidemiological measures and sensitivity analyses. Some of them were in `epistat-stats` 0.2.5 under the same module names and are now deprecated there.

### Added

- `rates`, rates with exact intervals, rate ratios (with an exact interval and an exact test) and differences, directly standardised rates with Fay-Feuer intervals, and the ratio and difference of two standardised rates. The two-group functions `rate_ratio`, `rate_difference`, `rate_ratio_exact` and `rate_ratio_exact_test` take a `PersonTime2` and `crude_rate` a `PersonTime`, so events and person-time are named and cannot be swapped as four positional numbers could. Every rate and difference of rates takes `per`, the person-time it is expressed per, which multiplies; `direct_standardised` takes the group as `Strata` and `per`, so a standardised and a crude rate print on one scale. `decomposition::kitagawa_poisson` calls its multiplier `per` as well.
- `mantel_haenszel`, Mantel-Haenszel estimators of the odds ratio, risk ratio, risk difference, rate ratio and rate difference, with the Breslow-Day test and tests of a common rate ratio over strata.
- `sir`, standardised incidence ratios with exact, Byar, Wald and profile intervals, and tests of homogeneity and trend.
- `standardisation`, the weights of direct standardisation onto the exposed. `stratum_counts` and `standardisation_weights` take the weights as `Option<epistat_core::CaseWeights>`, `None` for a weight of 1 for every person, and return `Result`: columns of different lengths are `InputError::Length` and a stratum code of `n_strata` or more `InputError::Domain` at its row, where `epistat-stats` 0.2.5 returned `None` for both.
- `decomposition`, the Kitagawa decomposition of a difference in rates with `kitagawa_variance` for the variances of its parts and `kitagawa_poisson` for Poisson rates.
- `evalue`, E-values for risk, odds and hazard ratios and for mean differences, plus `evalues_rd` for a risk difference and `selection_evalues_rr` for selection bias. `evalues` takes an `epistat_core::Estimate` and an `EffectMeasure`: it reads a ratio from an estimate on `Scale::Log` and refuses one on `Scale::Identity`, where a Cox or logistic coefficient is a log ratio, and `EValues.level` carries the level of the interval it read. The functions on numbers (`evalues_rr`, `evalues_or`, `evalues_hr`, `evalues_md`) take ratios; `evalues_md`'s limits are those of a 95% interval, as in R.
- `bias`, quantitative bias analysis of misclassification, selection, an unmeasured confounder and misclassified person-time, simple and probabilistic. The probabilistic analyses take `bias::ProbOptions`, which holds a `Bootstrap`, the random-error setting and a `Level`, and return the draws and the risk difference. The parameters of the simple corrections are named by their stratum: `ExposureMisclass { cases, noncases }`, `OutcomeMisclass { exposed, unexposed }` and `RateMisclass { events, person_time }` of a checked `SeSp`, predictive values in `ByOutcome<Predictive>` or `ByExposure<Predictive>`, a validation study's `PpvSe`, and the selection probabilities in `Selection`, a `Cells` named as the cells of `Counts2x2`, which `probabilistic_selection` and `BiasStep::Selection` take as `Cells<Dist>`. A sensitivity and specificity no better than chance, or a value outside its range, is refused when the parameter is built. The module documentation names the cells as `Counts2x2` does.
- `rosenbaum`, Rosenbaum bounds for matched designs (Wilcoxon, Hodges-Lehmann and M-statistic scores, with sets of different sizes) and the sensitivity values `wilcoxon_sensitivity_value`, `hodges_lehmann_gamma` and `scores_sensitivity_value`.
- `signed_rank`, the signed rank statistic with its exact and normal upper tails.
- `balance`, standardised mean differences, variance ratios, eCDF and eQQ distances, and Cramer's V, with weighted forms. `smd` takes each group as `Moments { mean, variance }`, with `Moments::from_sd` for a standard deviation, so a standard deviation cannot be passed where a variance is read.
- `tables`, the `Counts2x2`, `PersonTime` and `PersonTime2` tables every measure starts from.
- `Error`, one error type with stable codes that implements `epistat_core::Classify`. Bad input is reported as `Error::Input` with the row where there is one.

### Changed

Compared with the same items in `epistat-stats` 0.2.5:

- The errors `EValueError`, `SensitivityError` and `SignedRankError` are replaced by `Error`.
- Intervals take an `epistat_core::Level` and results are `epistat_core::Estimate` values, with `None` for an undefined number.
- Result structs of `rosenbaum`, `decomposition` and `evalue` are `#[non_exhaustive]`.

### Results

No released result changes. The functions that were also in `epistat-stats` 0.2.5 use the distribution functions and sums of `epistat-stats` 0.3, and differ from the 0.2.5 values in the last bits for the reasons given in that crate's changelog.

- The exact and gamma limits of `crude_rate`, `direct_standardised` and `sir`, and the exact p-values of `sir` and `poisson_two_sided`, change by up to 1e-14 relative, and up to 1.7e-13 for a p-value of 1e-73, because the incomplete gamma function and its quantile in `epistat-stats` are computed by a new method. The checked values moved towards the exact ones.
