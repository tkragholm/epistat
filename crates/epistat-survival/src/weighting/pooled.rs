//! Inverse probability of censoring weights from a pooled logistic model of
//! censoring on person-period data.
//!
//! # Algorithm
//!
//! The data has one row per subject and interval, as follow-up split at fixed
//! cut points with [`crate::split::split`] or written as discrete time. Each row
//! is an observation of a binary variable, censored in this interval or not.
//! A row is a censoring event when its status is 0 and it is the last row of
//! its subject, the rule of [`ipcw`](super::ipcw), or where the caller says so.
//!
//! The probability of censoring `d_i = P(C_i = 1 | z_i)` is the fit of a logistic
//! regression of the censoring indicator on the design `z`, with `glm(family =
//! binomial)`, on the rows from the subject's first up to and including its
//! first censoring event. The probability of remaining uncensored in row `i` is
//! `1 - d_i`. The weight of row `i` is the product, over the rows of its subject
//! up to and including `i`, of the probability of remaining uncensored under the
//! numerator model over that under the denominator model:
//!
//! ```text
//! w_i = prod over j <= i of (1 - n_j) / (1 - d_j)
//! ```
//!
//! The numerator model is a second logistic regression, on its own design, which
//! stabilises the weights (Robins, Hernan and Brumback 2000); without it the
//! numerator is 1. This is `ipw::ipwtm(type = "cens", family = "binomial")`, which
//! fits both models on the same rows and takes the product through the censoring
//! row. A row after the first censoring event of its subject, which the weights
//! of `ipwtm` also allow, carries the weight of the censoring row.
//!
//! Two weights are returned for each row: `weight`, through the row as above,
//! and `weight_before`, the product over the rows before it. Which one an outcome
//! model takes depends on whether the outcome of an interval is observed given
//! that the subject stays uncensored in the interval (`weight`, as in target
//! trial emulation) or at its start.
//!
//! # Edge cases
//!
//! A design column that is a combination of the earlier ones, a model that does
//! not converge, and a fitted probability of censoring of exactly 0 or 1 are
//! errors, since a weight would be undefined or infinite. Two rows of a subject
//! with the same stop are an error: their order is not known. Without an `id` in
//! the data every row is a subject of its own.
//!
//! # References
//!
//! 1. Robins JM, Hernan MA, Brumback B. Marginal structural models and causal
//!    inference in epidemiology. Epidemiology 2000;11(5):550-60.
//! 2. Hernan MA, Robins JM. Using big data to emulate a target trial when a
//!    randomized trial is not available. Am J Epidemiol 2016;183(8):758-64.
//! 3. van der Wal WM, Geskus RB. ipw: an R package for inverse probability
//!    weighting. J Stat Softw 2011;43(13):1-23.

use epistat_glm::{Family, Glm, Link};
use epistat_linalg::{Mat, MatRef};

use super::summary::WeightSummary;
use crate::error::{Error, Result};
use crate::surv::SurvData;

/// The weights of a pooled logistic model of censoring, and what the variance of
/// an outcome fitted with them needs.
///
/// It borrows the designs, which the variance reads again.
#[derive(Debug, Clone)]
pub struct PooledIpcw<'a> {
    /// The weight of each row through the row: the product over the rows of its
    /// subject up to and including it.
    pub weight: Vec<f64>,
    /// The weight of each row before it: the product over the earlier rows of
    /// its subject, 1 for the first.
    pub weight_before: Vec<f64>,
    /// Whether each row is a censoring event.
    pub censored: Vec<bool>,
    /// Whether each row is in the fit: not after the first censoring event of its
    /// subject.
    pub active: Vec<bool>,
    /// The fitted probability of censoring in each row under the denominator
    /// model.
    pub probability: Vec<f64>,
    /// The coefficients of the denominator model.
    pub coefficients: Vec<f64>,
    /// The coefficients of the numerator model, `None` without one.
    pub numerator_coefficients: Option<Vec<f64>>,
    /// The summary of `weight` over the active rows.
    pub summary: Option<WeightSummary>,
    pub(super) numerator_probability: Option<Vec<f64>>,
    pub(super) subject: Vec<u32>,
    /// The rows in the order of their subject and stop.
    pub(super) order: Vec<u32>,
    pub(super) design: MatRef<'a, f64>,
    pub(super) numerator_design: Option<MatRef<'a, f64>>,
}

impl PooledIpcw<'_> {
    /// The number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.weight.len()
    }

    /// Whether there are no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.weight.is_empty()
    }

    /// The number of parameters of the models, the denominator's then the
    /// numerator's.
    pub(super) fn n_params(&self) -> usize {
        self.design.ncols() + self.numerator_design.map_or(0, |m| m.ncols())
    }
}

fn expit(eta: f64) -> f64 {
    1.0 / (1.0 + (-eta).exp())
}

/// Fit the model of one design on the active rows, and give its coefficients and
/// the probability for every row.
fn fit_model(
    design: MatRef<'_, f64>,
    y: &[f64],
    active: &[bool],
    what: &str,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let kept: Vec<usize> = (0..design.nrows()).filter(|&i| active[i]).collect();
    let subset;
    let x = if kept.len() == design.nrows() {
        design
    } else {
        subset = Mat::from_fn(kept.len(), design.ncols(), |i, j| design[(kept[i], j)]);
        subset.as_ref()
    };
    let ys: Vec<f64> = kept.iter().map(|&i| y[i]).collect();
    let model = Glm::new(Family::Binomial, Link::Logit)
        .fit(x, &ys, None, None, None)
        .map_err(|e| Error::input(format!("{what} model: {e}")))?;
    if !model.converged {
        return Err(Error::input(format!("{what} model: no convergence")));
    }
    let mut coefficients = Vec::with_capacity(model.coefficients.len());
    for (j, c) in model.coefficients.iter().enumerate() {
        coefficients.push(
            c.ok_or_else(|| Error::input(format!("{what} model: design column {j} is aliased")))?,
        );
    }
    let probability: Vec<f64> = (0..design.nrows())
        .map(|i| {
            let eta: f64 = (0..design.ncols())
                .map(|j| design[(i, j)] * coefficients[j])
                .sum();
            expit(eta)
        })
        .collect();
    if let Some(i) = probability.iter().position(|&p| !(p > 0.0 && p < 1.0)) {
        return Err(Error::input(format!(
            "{what} model: row {i} has fitted probability {}",
            probability[i]
        )));
    }
    Ok((coefficients, probability))
}

/// Fit the pooled logistic model of censoring on `design` and weight the rows
/// of `data`.
///
/// `design` has one row per row of `data` and includes the intercept column, as
/// the design of `glm`; it may hold the interval, the covariates at the start of
/// the interval and anything else the model uses. `numerator` is the design of
/// the numerator model of stabilised weights, `None` for unstabilised weights.
/// `censored` marks the censoring events; `None` takes the last row of each
/// subject when its status is 0. The data needs an `id` for subjects with several
/// rows, and carries the stop that orders them.
///
/// # Errors
///
/// [`Error::Input`] for interval-censored data, for a design or `censored` of
/// another length than the data, a design without columns or with a value that is
/// not finite, when two rows of a subject have the same stop, when no row or
/// every row of the fit is a censoring event, when a model has an aliased
/// column or does not converge, and when a fitted probability is exactly 0 or 1.
pub fn fit<'a>(
    data: &SurvData<'_>,
    censored: Option<&[bool]>,
    design: MatRef<'a, f64>,
    numerator: Option<MatRef<'a, f64>>,
) -> Result<PooledIpcw<'a>> {
    let surv = data.followup()?;
    let n = data.n_rows();
    for (name, m) in [("design", Some(design)), ("numerator", numerator)] {
        let Some(m) = m else { continue };
        if m.nrows() != n {
            return Err(Error::input(format!(
                "{name}: {} rows for {n} rows of data",
                m.nrows()
            )));
        }
        if m.ncols() == 0 {
            return Err(Error::input(format!("{name}: no columns")));
        }
        if !epistat_linalg::is_finite(m) {
            return Err(Error::input(format!("{name}: a value is not finite")));
        }
    }
    if let Some(c) = censored {
        if c.len() != n {
            return Err(Error::input(format!(
                "censored: {} values for {n} rows",
                c.len()
            )));
        }
    }
    let stop = surv.stop();
    let subject: Vec<u32> = data
        .id()
        .map_or_else(|| (0..n as u32).collect(), <[u32]>::to_vec);
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by(|&a, &b| {
        subject[a as usize]
            .cmp(&subject[b as usize])
            .then(stop[a as usize].total_cmp(&stop[b as usize]))
    });
    for w in order.windows(2) {
        let (a, b) = (w[0] as usize, w[1] as usize);
        if subject[a] == subject[b] && stop[a].total_cmp(&stop[b]).is_eq() {
            return Err(Error::input(format!(
                "id: rows {a} and {b} of one subject have the same stop"
            )));
        }
    }
    // The censoring events, and the rows up to the first of each subject.
    let mut event = vec![false; n];
    match censored {
        Some(c) => event.copy_from_slice(c),
        None => {
            for (k, &r) in order.iter().enumerate() {
                let r = r as usize;
                let last = order
                    .get(k + 1)
                    .is_none_or(|&s| subject[s as usize] != subject[r]);
                event[r] = last && surv.status()[r] == 0;
            }
        }
    }
    let mut active = vec![true; n];
    let mut weight = vec![0.0; n];
    let mut weight_before = vec![0.0; n];
    let mut seen = false;
    for (k, &r) in order.iter().enumerate() {
        let r = r as usize;
        if k == 0 || subject[order[k - 1] as usize] != subject[r] {
            seen = false;
        }
        active[r] = !seen;
        seen |= event[r];
    }
    let y: Vec<f64> = event.iter().map(|&e| f64::from(u8::from(e))).collect();
    let n_events = (0..n).filter(|&r| active[r] && event[r]).count();
    let n_active = active.iter().filter(|&&a| a).count();
    if n_events == 0 || n_events == n_active {
        return Err(Error::input(
            "censored: the rows of the fit need both censoring events and rows that are not",
        ));
    }
    let (coefficients, probability) = fit_model(design, &y, &active, "censoring")?;
    let numerator_fit = numerator
        .map(|m| fit_model(m, &y, &active, "numerator"))
        .transpose()?;
    let mut cum = 1.0;
    for (k, &r) in order.iter().enumerate() {
        let r = r as usize;
        if k == 0 || subject[order[k - 1] as usize] != subject[r] {
            cum = 1.0;
        }
        weight_before[r] = cum;
        if active[r] {
            let num = numerator_fit.as_ref().map_or(1.0, |(_, pn)| 1.0 - pn[r]);
            cum *= num / (1.0 - probability[r]);
        }
        weight[r] = cum;
    }
    let summary = WeightSummary::new((0..n).filter(|&r| active[r]).map(|r| weight[r]));
    let (numerator_coefficients, numerator_probability) = match numerator_fit {
        Some((c, p)) => (Some(c), Some(p)),
        None => (None, None),
    };
    Ok(PooledIpcw {
        weight,
        weight_before,
        censored: event,
        active,
        probability,
        coefficients,
        numerator_coefficients,
        summary,
        numerator_probability,
        subject,
        order,
        design,
        numerator_design: numerator,
    })
}
