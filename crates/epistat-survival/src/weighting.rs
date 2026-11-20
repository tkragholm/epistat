//! Inverse probability weights for treatment and for censoring, and weighted
//! outcome models whose variance accounts for the estimated weights.
//!
//! - [`iptw`] fits a logistic propensity model with `epistat-glm` and returns
//!   the weights of the average treatment effect, of the treated or of the
//!   controls, unstabilised or stabilised, as `WeightIt::weightit(method =
//!   "glm")`, with the summary of the weights that `summary.weightit` gives.
//! - [`ipcw`] gives inverse probability of censoring weights for time-to-event
//!   data from a Kaplan-Meier or Cox model of the censoring hazard, with
//!   counting-process rows and covariates that change over time.
//! - [`cox`] fits a weighted Cox model on estimated treatment weights and
//!   gives the variance that accounts for the estimation of the weights, with
//!   the variance that ignores it for comparison.
//! - [`km`] gives the weighted Kaplan-Meier curves at chosen times, with the
//!   same two variances.
//! - [`pooled`] gives inverse probability of censoring weights from a pooled
//!   logistic model of censoring on person-period data, as
//!   `ipw::ipwtm(type = "cens", family = "binomial")`, stabilised by a numerator
//!   model.
//! - [`estimated`] names the estimated weights of an outcome model, of treatment,
//!   of censoring or both, and [`cox::fit_estimated`] and
//!   [`km::estimate_estimated`] give the variance that accounts for every model
//!   being estimated.
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
//! rows and clustered data are handled by the units. The stacked matrix of
//! [`cox::fit`] and [`km::estimate`] is held in memory as one row per unit and
//! one column per parameter.
//!
//! # Estimated censoring weights, and both kinds together
//!
//! [`cox::fit_estimated`] and [`km::estimate_estimated`] take the weights as the
//! product of those of the models named in an [`Estimated`]: the propensity model
//! of treatment, and the pooled logistic, Kaplan-Meier or Cox model of
//! censoring. Write the stacked equations for one model of weights with parameters
//! `theta` as above. The influence of a unit `u` on the outcome estimate is
//! `V (U_u + sum over rows k of g_k IF_u(w_k))` with `V` the inverse of minus the
//! derivative of `U`, `g_k = d U / d w_k` and `IF_u(w_k)` the change of the weight
//! of row `k` when the case weight of unit `u` is perturbed; for a model with
//! parameters this is `-A^-1 psi_u` times `d w_k / d theta`, which is what stacking
//! gives. Writing it as an influence applies to every model of weights, also the
//! Kaplan-Meier and Cox models of the censoring hazard, whose parameters are
//! the jumps of a baseline hazard. The weights are positive, so `IF_u(w_k) = w_k
//! IF_u(log w_k)`, and the log of a product is a sum: the term of each model is
//! added to the influence, and the models, fitted separately, do not affect each
//! other's parameters. The variance is the sum over units of the outer products of
//! the total influences, `V (sum (U_u + terms)(U_u + terms)') V`. With treatment
//! weights alone it is that of [`cox::fit`] and [`km::estimate`].
//!
//! The terms for a logistic model (propensity, pooled censoring) need the
//! derivative `d log w_k / d theta` and `psi_u` of its items, which are added into
//! one matrix of units by outcome parameters, without the matrix of units by
//! parameters of the weights that stacking would form. The Kaplan-Meier censoring
//! model has the jumps `log Y_l - log (Y_l - d_l)` of its cumulative hazard and
//! the Cox model the increments of its baseline hazard, together with the
//! coefficients, whose influence is the covariance times the score residual of the
//! unit; see `censoring` for the derivatives. All are computed from sums over the
//! risk sets.
//!
//! # Memory and cost
//!
//! [`cox::fit_estimated`] holds the score residuals of the rows (`rows x p`, in
//! two copies), the sums of the residuals by unit (`units x p`), and again the
//! sums with the terms of the weights added. No matrix of units by parameters of
//! the weights is formed, as stacking would: a term is added into the same
//! `units x p` matrix, so the memory does not grow with the number of parameters of
//! the weights. [`km::estimate_estimated`] holds `rows x curves` and `units x
//! curves` for one time at a time. At register scale, with `10^7` units and `p =
//! 10`, a matrix of units is `10^7 x 10 x 8` bytes, 0.8 GB, and there are two of
//! them besides the residuals. The matrices of units could be streamed by
//! forming the terms unit by unit, but the rows of a unit are scattered in the
//! data and the residuals are needed for every unit, so that is not done.
//!
//! The time is `O((n + L) log n)` per outcome parameter for the Kaplan-Meier and
//! Cox models of censoring, with `L` the number of censoring times, and `O(n q)`
//! per outcome parameter for logistic models with `q` parameters. [`cox::fit`]
//! and [`km::estimate`], which stack the propensity model, hold `units x (q + p)`
//! and `units x q`.
//!
//! # What is not here
//!
//! Sampling weights in the propensity model, and the propensity models of
//! `WeightIt` other than the logistic one, are not implemented. The censoring
//! models are fitted without treatment weights and the propensity model without
//! censoring weights, and the variance treats their parameters as not affecting
//! each other. The variance of weights from a Cox model of censoring is not
//! available for exact ties, and rows of the outcome data are those of the
//! censoring model.
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

mod adjust;
mod censoring;
pub mod cox;
pub mod estimated;
pub mod ipcw;
pub mod iptw;
pub mod km;
pub mod pooled;
mod stack;
mod summary;

pub use adjust::WeightAt;
pub use estimated::{Censoring, Estimated};
pub use iptw::{Estimand, Iptw, IptwOptions, IptwSummary};
pub use summary::WeightSummary;
