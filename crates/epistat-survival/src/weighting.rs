//! Inverse probability weights for treatment and for censoring, and weighted
//! outcome models whose variance accounts for the estimated weights.
//!
//! - [`iptw`] fits a logistic propensity model with `epistat-glm` and returns
//!   the weights of the average treatment effect, of the treated or of the
//!   controls, unstabilised or stabilised, as `WeightIt::weightit(method =
//!   "glm")`, with the summary of the weights that `summary.weightit` gives.
//! - [`cox`] fits a weighted Cox model on estimated treatment weights and
//!   gives the variance that accounts for the estimation of the weights, with
//!   the variance that ignores it for comparison.
//! - [`km`] gives the weighted Kaplan-Meier curves at chosen times, with the
//!   same two variances.
//!
//! # Variance
//!
//! The treatment weights come from a propensity model whose parameters
//! `alpha` solve `sum_i psi_i(alpha) = 0`, and the outcome estimate `beta`
//! from weights `w_i(alpha)` solves `sum_i U_i(beta; w(alpha)) = 0`. Treating
//! the weights as known gives the robust variance of `coxph(weights =,
//! robust = TRUE)`. Stacking the two sets of equations gives the M-estimation
//! variance of Shu, Young, Toh and Wang (2021) and of `WeightIt::coxph_weightit`,
//! `A^-1 B A^-T` with
//!
//! ```text
//! A = [ d psi / d alpha        0            ]      B = sum over units of
//!     [ d U / d alpha     d U / d beta      ]          (psi_u, U_u)(psi_u, U_u)'
//! ```
//!
//! The off-diagonal block is the dependence of the outcome equations on the
//! weights, `d U / d alpha = sum_i (d U / d w_i) (d w_i / d alpha)`. For the
//! Cox score `d U / d w_i` is the score residual of subject `i` divided by its
//! weight, and for the Kaplan-Meier curve it is the infinitesimal jackknife
//! derivative. The stabilised weights add the marginal probability of
//! treatment as a parameter, fitted by the equation `sum_i (a_i - pi) = 0`.
//!
//! The estimating functions of a unit (a subject, or a cluster of subjects)
//! are summed before the outer product is formed, so subjects with several
//! rows and clustered data are handled by the units. The stacked matrix is
//! held in memory as one row per unit and one column per parameter.
//!
//! # What is not here
//!
//! The variance of an estimate that also uses estimated censoring weights, the
//! pooled logistic censoring model of `ipw::ipwtm`, sampling weights in the
//! propensity model, and the propensity models of `WeightIt` other than the
//! logistic one are not implemented.
//!
//! # References
//!
//! 1. Robins JM, Hernan MA, Brumback B. Marginal structural models and
//!    causal inference in epidemiology. Epidemiology. 2000;11(5):550-60.
//! 2. Hernan MA, Robins JM. Estimating causal effects from epidemiological
//!    data. J Epidemiol Community Health. 2006;60(7):578-86.
//! 3. Shu D, Young JG, Toh S, Wang R. Variance estimation in inverse
//!    probability weighted Cox models. Biometrics. 2021;77(3):1101-12.
//! 4. Stefanski LA, Boos DD. The calculus of M-estimation. Am Stat.
//!    2002;56(1):29-38.
//! 5. Robins JM, Finkelstein DM. Correcting for noncompliance and dependent
//!    censoring in an AIDS clinical trial with inverse probability of
//!    censoring weighted (IPCW) log-rank tests. Biometrics. 2000;56(3):779-88.
//! 6. Greifer N. WeightIt: weighting for covariate balance in observational
//!    studies. R package.

// Rows and subjects are counted in `u32`: `SurvData` refuses more rows, and the
// subject map is `u32`.
#![expect(
    clippy::cast_possible_truncation,
    reason = "the rows and subjects are counted in u32, which the data type checks"
)]

pub mod cox;
pub mod iptw;
pub mod km;
mod stack;
mod summary;

pub use iptw::{Estimand, Iptw, IptwOptions, IptwSummary};
pub use summary::WeightSummary;
