//! Epidemiological measures and sensitivity analyses, checked against R.
//!
//! | module | what it holds | R |
//! |---|---|---|
//! | [`rates`] | rates with exact intervals, rate ratios and differences, directly standardised rates and their ratio and difference | `epitools`, `epiR`, `dsr` |
//! | [`mantel_haenszel`] | Mantel-Haenszel odds ratio, risk ratio, risk difference, rate ratio and rate difference, Breslow-Day and homogeneity tests | `epiR`, `metafor` |
//! | [`sir`] | standardised incidence ratios with exact, Byar, Wald and profile intervals, tests of homogeneity and trend | `popEpi`, `epitools` |
//! | [`standardisation`] | the weights of direct standardisation onto the exposed | `MatchIt` |
//! | [`decomposition`] | the Kitagawa decomposition of a difference in rates, with the variances of its parts | `DasGuptR` |
//! | [`evalue`] | E-values for risk, odds and hazard ratios and differences, and for selection | `EValue` |
//! | [`bias`] | quantitative bias analysis, simple and probabilistic, of misclassification, selection and confounding | `episensr` |
//! | [`rosenbaum`] | Rosenbaum bounds for matched designs | `rbounds`, `sensitivitymv`, `sensitivitymw`, `sensitivity2x2xk` |
//! | [`signed_rank`] | the signed rank statistic, its exact and normal tails | `stats::wilcox.test` |
//! | [`balance`] | standardised mean differences, variance ratios, eCDF and eQQ distances, Cramér's V | `cobalt`, `MatchIt` |
//!
//! The distribution functions these rest on are in `epistat_stats::dist`.
//! Nothing uses fused multiply-add, so results are the same on every
//! platform.
#![warn(missing_docs)]

pub mod balance;
pub mod bias;
pub mod decomposition;
pub mod evalue;
mod exact;
pub mod mantel_haenszel;
pub mod rates;
pub mod rosenbaum;
pub mod signed_rank;
pub mod sir;
pub mod standardisation;
