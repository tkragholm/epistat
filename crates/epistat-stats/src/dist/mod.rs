//! Distribution functions, one module per distribution.
//!
//! | module | functions | R |
//! |---|---|---|
//! | [`normal`] | `pdf`, `cdf`, `sf`, `log_cdf`, `quantile`, `truncated_quantile`, `folded_quantile`, `two_sided_z` | `dnorm`, `pnorm`, `qnorm` |
//! | [`t`] | `ln_pdf`, `cdf`, `sf`, `quantile` | `dt`, `pt`, `qt` |
//! | [`chi_square`] | `cdf`, `sf`, `quantile` | `pchisq`, `qchisq` |
//! | [`f`] | `sf`, `quantile` | `pf`, `qf` |
//! | [`beta`] | `cdf`, `regularised`, `quantile` | `pbeta`, `qbeta` |
//! | [`gamma`] | `cdf`, `sf`, `quantile`, `ln_gamma`, `digamma`, `trigamma` | `pgamma`, `qgamma`, `lgamma`, `digamma`, `trigamma` |
//! | [`poisson`] | `pmf`, `cdf`, `sf`, `cipoisson` | `dpois`, `ppois`, `survival::cipoisson` |
//! | [`binomial`] | `pmf`, `cdf`, `sf` | `dbinom`, `pbinom` |
//!
//! # Conventions
//!
//! The names follow one rule: `pdf` or `pmf` for the density or point
//! probability (`ln_pdf` on the log scale), `cdf` for `P(X <= x)`, `sf` for
//! `P(X > x)` computed directly so that a small upper tail keeps its relative
//! precision, and `quantile` for the inverse of `cdf`. The variable comes
//! first and the parameters follow in R's order.
//!
//! The scalar functions return `f64`, and NaN for a parameter or a
//! probability outside its domain, as R's `p` and `q` functions do. They sit
//! in inner loops, where an `Option` would be unwrapped at every call; the
//! estimators that call them report an undefined result as `None`. A
//! quantile is `-inf` or `inf` where the distribution's support ends there,
//! as in R.
//!
//! Every function uses plain operations in a fixed order and no fused
//! multiply-add, so its results are the same on every platform.

pub mod beta;
pub mod binomial;
pub mod chi_square;
pub(crate) mod discrete;
pub mod f;
pub mod gamma;
pub mod normal;
pub mod poisson;
pub mod t;
