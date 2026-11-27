//! Conditional logistic regression for matched sets, as `survival::clogit`.
//!
//! [`clogit`] takes the case indicator, the matched set of each row and a design
//! matrix, and fits the conditional likelihood of the cases given the sets.
//! Like `clogit`, it does so as a stratified Cox model: every row has the time
//! 1, a case is an event and a control is censored, the matched sets are the
//! strata, and the cases of a set are tied events at one time. The
//! entry point builds that data itself, so no one has to invent the times.
//!
//! # Likelihood
//!
//! A set with `n` rows of which `d` are cases contributes the probability
//! that the cases are the `d` that they are, out of all the sets of `d` rows
//! from the `n`, each set weighted by the product of its risk scores
//! `exp(beta' x_j + o_j)`. That is the exact partial likelihood of
//! [`Ties::Exact`], whose algorithm, cost and references are in the
//! [`cox::design`](crate::cox::design) module. With one case and `n - 1`
//! controls it is the multinomial logit of the case among the `n`, and with
//! one control and one case the logistic regression of the difference of
//! their covariates. A set without a case or without a control adds nothing to
//! the likelihood, the score or the information.
//!
//! The other methods replace the exact likelihood by an approximation for tied
//! events. `Breslow`, which `clogit` also calls `approximate`, gives every case
//! the whole set as its denominator, and `Efron` removes `k / d` of the cases
//! from the denominator of the `k`-th. They are faster for sets with many
//! cases and give a different estimate.
//!
//! # What is returned
//!
//! The fit is the [`CoxModelFit`] of the Cox fit, with the conventions of
//! `clogit`: `n_rows` counts every row, as `n` of a `clogit` fit does, whether
//! its set has a case or not, `n_events` is the number of cases (or their
//! weight), `log_partial_likelihood` is the conditional log likelihood at the
//! estimate and `null_log_partial_likelihood` the same at zero, which are the
//! two numbers of `loglik` of the fit. The tests, aliased columns and the
//! meaning of `None` are those of [`cox::fit`](crate::cox::fit).
//!
//! # Where it differs from clogit
//!
//! With `Exact`, `clogit` ignores case weights with a warning and refuses a
//! cluster-robust variance. Both are an error here. With another method the
//! weights and clusters are passed on to the Cox fit as in `coxph`. The
//! iteration is that of `coxph`: of `coxexact.fit` for the exact method and of
//! `coxph.fit` for the others, since the data have no entry times.
//!
//! # Agreement with R
//!
//! The tests compare coefficients, standard errors, both log likelihoods and
//! the iteration count with `survival::clogit` on the `infert` data and on a
//! simulated set of matched case-control data, for the four methods.

use epistat_linalg::MatRef;

use crate::check::check_lengths;
use crate::cox::{CoxModelFit, CoxOptions, Ties, fit};
use crate::error::Result;
use crate::surv::SurvData;

/// How the cases of a matched set enter the likelihood.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ClogitMethod {
    /// The exact conditional likelihood, `method = "exact"`.
    Exact,
    /// Breslow's approximation, `method = "approximate"`.
    Approximate,
    /// Efron's approximation, `method = "efron"`.
    Efron,
    /// Breslow's approximation, `method = "breslow"`.
    Breslow,
}

/// What [`clogit`] takes besides the data.
#[derive(Debug, Clone, PartialEq)]
pub struct ClogitOptions<'a> {
    /// The treatment of sets with more than one case. `Exact`, as in `clogit`.
    pub method: ClogitMethod,
    /// A term added to the linear predictor of every row, one value per row.
    pub offset: Option<&'a [f64]>,
    /// Case weights, one per row. An error with `Exact`, which has no weights.
    pub weight: Option<&'a [f64]>,
    /// Cluster codes for a robust variance. An error with `Exact`.
    pub cluster: Option<&'a [u32]>,
    /// Starting coefficients, as `init` of `coxph`.
    pub start: Option<Vec<f64>>,
    /// The largest number of trial points, `iter.max` of `coxph.control`.
    pub max_iterations: usize,
    /// The relative change of the log likelihood at convergence, `eps` of
    /// `coxph.control`.
    pub tolerance: f64,
}

impl Default for ClogitOptions<'_> {
    fn default() -> Self {
        let base = CoxOptions::default();
        Self {
            method: ClogitMethod::Exact,
            offset: None,
            weight: None,
            cluster: None,
            start: None,
            max_iterations: base.max_iterations,
            tolerance: base.tolerance,
        }
    }
}

/// Conditional logistic regression of `case` on the columns of `x` within the
/// matched sets `set`, as `survival::clogit`.
///
/// `case`, `set` and the rows of `x` have one entry per row of the data. The
/// codes of `set` may be any `u32` values. Rows with a weight of 0 are left
/// out of the fit, and counted in `n_rows`. See the module documentation for
/// the likelihood and for what the result holds.
///
/// # Errors
///
/// An input error when the columns differ in length,
/// when `x` has no column or a value that is not finite, for the invalid
/// weights, offset, starting values and `tolerance` that [`cox::fit`](crate::cox::fit)
/// refuses, and with the exact method when case weights or clusters are given.
pub fn clogit(
    case: &[bool],
    set: &[u32],
    x: MatRef<'_, f64>,
    opts: &ClogitOptions<'_>,
) -> Result<CoxModelFit> {
    check_lengths(&[("case", case.len()), ("set", set.len()), ("x", x.nrows())])?;
    let time = vec![1.0f64; case.len()];
    let status: Vec<i32> = case.iter().map(|&c| i32::from(c)).collect();
    let mut builder = SurvData::builder(&time, &status).stratum(set);
    if let Some(weight) = opts.weight {
        builder = builder.weight(weight);
    }
    if let Some(cluster) = opts.cluster {
        builder = builder.cluster(cluster);
    }
    if let Some(offset) = opts.offset {
        builder = builder.offset(offset);
    }
    let data = builder.build()?;
    let cox = CoxOptions {
        ties: match opts.method {
            ClogitMethod::Exact => Ties::Exact,
            ClogitMethod::Approximate | ClogitMethod::Breslow => Ties::Breslow,
            ClogitMethod::Efron => Ties::Efron,
        },
        start: opts.start.clone(),
        max_iterations: opts.max_iterations,
        tolerance: opts.tolerance,
        ..CoxOptions::default()
    };
    fit(&data, x, &cox)
}
