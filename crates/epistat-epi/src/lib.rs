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
//! | [`tables`] | the 2 x 2 table and the person-time table every measure starts from | |
//! | [`balance`] | standardised mean differences, variance ratios, eCDF and eQQ distances, Cramér's V | `cobalt`, `MatchIt` |
//!
//! # Results and levels
//!
//! An estimate with its uncertainty is an [`epistat_core::Estimate`]: the
//! estimate, its standard error, an [`epistat_core::Interval`] at the level
//! asked for and a p-value, each `None` where the method does not define
//! it, and the [`epistat_core::Scale`] on which the standard error and the
//! interval were formed. A result whose estimate is undefined, such as a
//! rate over no person-time, is `None` as a whole. Every interval takes an
//! [`epistat_core::Level`], whose default is a confidence of 0.95 and which
//! keeps the number the caller gave, so `Level::confidence(0.95)` and
//! `Level::alpha(0.05)` give the limits R gives for `conf.level = 0.95` and
//! `alpha = 0.05`. The sensitivity values of [`rosenbaum`] take the `alpha`
//! of a one-sided test as a number, since they form no interval.
//!
//! # Names shared between modules
//!
//! [`rates::rate_ratio`] and [`mantel_haenszel::rate_ratio`] are the crude
//! and the pooled rate ratio, and the same holds for `rate_difference`. They
//! keep their names and are called through their modules: with both modules
//! imported by glob, the bare name is ambiguous.
//!
//! The distribution functions these rest on are in `epistat_stats::dist`.
//! Nothing uses fused multiply-add, so results are the same on every
//! platform.
#![warn(missing_docs)]

pub mod balance;
pub mod bias;
pub mod decomposition;
mod error;
pub mod evalue;
mod exact;
mod limits;
pub mod mantel_haenszel;
pub mod rates;
pub mod rosenbaum;
pub mod signed_rank;
pub mod sir;
pub mod standardisation;
pub mod tables;

pub use error::Error;
pub use tables::{Counts2x2, PersonTime2};
