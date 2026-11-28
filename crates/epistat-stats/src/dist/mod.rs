//! Distribution functions, one module per distribution, and the
//! distributions as validated values.
//!
//! | module | functions | R |
//! |---|---|---|
//! | [`normal`] | `pdf`, `ln_pdf`, `cdf`, `sf`, `log_cdf`, `ln_sf`, `quantile`, `isf`, `two_sided_p`, `truncated_quantile`, `folded_quantile`, `two_sided_z` | `dnorm`, `pnorm`, `qnorm` |
//! | [`t`] | `pdf`, `ln_pdf`, `cdf`, `sf`, `quantile`, `isf`, `two_sided_p` | `dt`, `pt`, `qt` |
//! | [`chi_square`] | `pdf`, `ln_pdf`, `cdf`, `sf`, `quantile`, `isf` | `dchisq`, `pchisq`, `qchisq` |
//! | [`f`] | `pdf`, `ln_pdf`, `cdf`, `sf`, `quantile`, `isf` | `df`, `pf`, `qf` |
//! | [`beta`] | `pdf`, `ln_pdf`, `cdf`, `sf`, `regularised`, `quantile`, `isf` | `dbeta`, `pbeta`, `qbeta` |
//! | [`gamma`] | `pdf`, `ln_pdf`, `cdf`, `sf`, `quantile`, `isf`, `ln_gamma`, `digamma`, `trigamma` | `dgamma`, `pgamma`, `qgamma`, `lgamma`, `digamma`, `trigamma` |
//! | [`poisson`] | `pmf`, `ln_pmf`, `cdf`, `sf`, `quantile`, `isf`, `cipoisson` | `dpois`, `ppois`, `qpois`, `survival::cipoisson` |
//! | [`binomial`] | `pmf`, `ln_pmf`, `cdf`, `sf`, `quantile`, `isf` | `dbinom`, `pbinom`, `qbinom` |
//!
//! [`Normal`], [`StudentT`], [`ChiSquared`], [`FisherF`], [`Gamma`] (with a
//! rate or a scale), [`Beta`], [`Poisson`] and [`Binomial`] check their
//! parameters once and refuse a value outside the domain with an error,
//! where the free functions return NaN; their methods call the free
//! functions, so they give the same bits. Use them for parameters that come
//! from data, such as degrees of freedom from a sample size.
//!
//! ```
//! use epistat_stats::dist::{Gamma, StudentT};
//!
//! let t = StudentT::new(24.0)?;
//! let p = t.two_sided_p(2.1);
//! assert!((p - 0.046_421_502_183_235).abs() < 1e-14);
//! // R's pgamma(3, shape = 2, rate = 1.5).
//! let g = Gamma::with_rate(2.0, 1.5)?;
//! assert!((g.cdf(3.0) - 0.938_900_519_039_667_3).abs() < 1e-14);
//! # Ok::<(), epistat_stats::Error>(())
//! ```
//!
//! # Conventions
//!
//! The names follow one rule: `pdf` or `pmf` for the density or point
//! probability (`ln_pdf` and `ln_pmf` on the log scale), `cdf` for
//! `P(X <= x)`, `sf` for `P(X > x)` computed directly so that a small upper
//! tail keeps its relative precision, `quantile` for the inverse of `cdf` and
//! `isf` for the inverse of `sf`, R's `lower.tail = FALSE`, which takes the
//! upper-tail probability itself so that one below `1e-16` keeps its
//! quantile. The variable comes
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
mod objects;
pub mod poisson;
pub(crate) mod stirling;
pub mod t;

pub use objects::{Beta, Binomial, ChiSquared, FisherF, Gamma, Normal, Poisson, StudentT};
