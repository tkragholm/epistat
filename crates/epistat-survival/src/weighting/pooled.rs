//! Inverse probability weights from a pooled logistic model of censoring, or of
//! treatment, on person-period data.
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
//! # Treatment weights
//!
//! For a treatment that changes over time (Robins, Hernan and Brumback 2000) the
//! binary variable of each row is the treatment `a_j` of the interval. The factor
//! of a row is the probability of the value that was observed, `p_j` if `a_j = 1`
//! and `1 - p_j` otherwise, under the denominator model, which holds the
//! covariates that change over time, and under the numerator model of
//! stabilised weights:
//!
//! ```text
//! w_i = prod over j <= i of f_n(a_j) / f_d(a_j)
//! ```
//!
//! The probability of censoring above is the case where the observed value is
//! always "not censored". [`TreatmentHistory::First`] is `ipw::ipwtm(type =
//! "first", family = "binomial")`, for a treatment that is absorbing: the fit and
//! the product run from the first row of a subject up to and including its first
//! treated row, and the later rows carry the weight of that row.
//! [`TreatmentHistory::All`] is `type = "all"`, where every row is in the fit and
//! the product.
//!
//! Truncation, `trunc` of `ipwtm`, sets the weights of all rows below the
//! `lower` quantile and above the `upper` quantile (type 7, over every row) to
//! those quantiles. It is applied to the final weights, after the cumulative
//! product, and the weights before it are in [`PooledIpcw::untruncated`]. The
//! weight of a row before its own factor is truncated with the same limits.
//!
//! A truncated weight is its limit, and a limit moves with the parameters of
//! the models through the two order statistics it interpolates: with the type 7
//! position `1 + (N - 1) p = j + h` over the `N` sorted weights,
//!
//! ```text
//! q = (1 - h) w_(j) + h w_(j+1),     dq / dtheta = (1 - h) dw_(j) / dtheta + h dw_(j+1) / dtheta,
//! ```
//!
//! and the variance that accounts for the estimated models takes this
//! derivative for every truncated row. The ranks themselves are held fixed, as
//! they are under the infinitesimal jackknife with case weights, so the
//! sampling variation of the quantile from the ranks is not in the variance.
//! Where rows tie at an order statistic, the first in the sort of the weights
//! is taken.
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

#![expect(
    deprecated,
    reason = "the fits take their data as separate arguments, the epistat-glm 0.2 form"
)]

use epistat_core::InputError;
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
    /// Whether each row is a censoring event. All false for treatment weights.
    pub censored: Vec<bool>,
    /// The treatment of each row, for treatment weights, otherwise `None`.
    pub treated: Option<Vec<bool>>,
    /// The weight through the row before truncation, `None` without truncation.
    pub untruncated: Option<Vec<f64>>,
    /// Whether each row is in the fit: not after the first censoring event of its
    /// subject.
    pub active: Vec<bool>,
    /// The fitted probability of censoring, or of treatment, in each row under the
    /// denominator model.
    pub probability: Vec<f64>,
    /// The coefficients of the denominator model.
    pub coefficients: Vec<f64>,
    /// The coefficients of the numerator model, `None` without one.
    pub numerator_coefficients: Option<Vec<f64>>,
    /// The summary of `weight` over the active rows.
    pub summary: Option<WeightSummary>,
    pub(super) numerator_probability: Option<Vec<f64>>,
    /// Whether the weight through each row, and before it, was set to a limit of
    /// the truncation.
    pub(super) clamped: Option<(Vec<bool>, Vec<bool>)>,
    /// The lower and upper limit of the truncation.
    pub(super) limits: Option<[Limit; 2]>,
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

    /// The binary variable the models are fitted to in a row.
    pub(super) fn response(&self, row: usize) -> bool {
        self.treated.as_ref().map_or(self.censored[row], |t| t[row])
    }

    /// The value whose probability is the factor of a row: treatment as
    /// observed, and not censored for the weights of censoring.
    pub(super) fn observed(&self, row: usize) -> bool {
        self.treated.as_ref().is_some_and(|t| t[row])
    }

    /// The number of parameters of the models, the denominator's then the
    /// numerator's.
    pub(super) fn n_params(&self) -> usize {
        self.design.ncols() + self.numerator_design.map_or(0, |m| m.ncols())
    }

    /// The derivative of the log of the weight through `row`, before truncation,
    /// with respect to the parameters of the models: the sum over the active rows
    /// of its subject up to it of `(p - o) z` for the denominator and `(o - p) z`
    /// for the numerator, with `o` the value observed.
    pub(super) fn log_weight_gradient_into(&self, row: usize, out: &mut [f64]) {
        out.fill(0.0);
        let qd = self.design.ncols();
        let id = self.subject[row];
        let Some(pos) = self.order.iter().position(|&r| r as usize == row) else {
            return;
        };
        for &r in self.order[..=pos].iter().rev() {
            let r = r as usize;
            if self.subject[r] != id {
                break;
            }
            if !self.active[r] {
                continue;
            }
            let o = f64::from(u8::from(self.observed(r)));
            let pd = self.probability[r] - o;
            for (j, g) in out.iter_mut().enumerate().take(qd) {
                *g += pd * self.design[(r, j)];
            }
            if let (Some(zn), Some(pn)) = (self.numerator_design, &self.numerator_probability) {
                let v = o - pn[r];
                for j in 0..zn.ncols() {
                    out[qd + j] += v * zn[(r, j)];
                }
            }
        }
    }
}

/// A limit of the truncation: a type 7 quantile of the weights through the rows,
/// `(1 - h) w[rows.0] + h w[rows.1]`, with the rows of the two order statistics.
#[derive(Debug, Clone, Copy)]
pub(super) struct Limit {
    pub value: f64,
    pub rows: (usize, usize),
    pub h: f64,
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
        .fit_slices(x, &ys, None, None, None)
        .map_err(Error::model(format!("{what} model")))?;
    if !model.converged {
        return Err(Error::invalid(format!("{what} model: no convergence")));
    }
    let mut coefficients = Vec::with_capacity(model.coefficients.len());
    for (j, c) in model.coefficients.iter().enumerate() {
        coefficients.push(c.ok_or_else(|| {
            Error::invalid(format!("{what} model: design column {j} is aliased"))
        })?);
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
        return Err(Error::invalid(format!(
            "{what} model: row {i} has fitted probability {}",
            probability[i]
        )));
    }
    Ok((coefficients, probability))
}

/// What the rows of a pooled logistic model are events of.
#[derive(Debug, Clone, Copy)]
pub enum PooledEvent<'a> {
    /// The rows are intervals at risk of censoring, as in [`fit`]. `censored`
    /// marks the censoring events, `None` takes the last row of each subject
    /// when its status is 0.
    Censoring {
        /// The censoring events.
        censored: Option<&'a [bool]>,
    },
    /// The binary treatment of each row. The weight uses the probability of the
    /// value observed.
    Treatment {
        /// The treatment in each row.
        treated: &'a [bool],
        /// Which rows are in the fit and the product.
        history: TreatmentHistory,
    },
}

/// The rows of a subject that treatment weights use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreatmentHistory {
    /// From the first row up to and including the first treated one:
    /// `type = "first"`, for an absorbing treatment.
    First,
    /// Every row: `type = "all"`.
    All,
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
/// An input error for interval-censored data, for a design or `censored` of
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
    fit_pooled(
        data,
        PooledEvent::Censoring { censored },
        design,
        numerator,
        None,
    )
}

/// Fit the pooled logistic model of `event` on `design` and weight the rows of
/// `data`.
///
/// `design`, `numerator` and the rules of the rows are those of [`fit`] for
/// censoring. For treatment, `design` holds the covariates that change over
/// time and `numerator` those of the stabilising model. `trunc` is `(lower,
/// upper)`, two probabilities with `0 <= lower < upper <= 1`, at whose quantiles
/// the weights are truncated; `ipwtm(trunc = t)` is `(t, 1 - t)`.
///
/// # Errors
///
/// The errors of [`fit`], an input error for `treated` of another length than
/// the data, when the rows of the fit are all treated or all untreated, and for
/// limits of truncation that are not ordered probabilities.
pub fn fit_pooled<'a>(
    data: &SurvData<'_>,
    event: PooledEvent<'_>,
    design: MatRef<'a, f64>,
    numerator: Option<MatRef<'a, f64>>,
    trunc: Option<(f64, f64)>,
) -> Result<PooledIpcw<'a>> {
    let surv = data.followup()?;
    let n = data.n_rows();
    if let Some((lo, hi)) = trunc
        && !(0.0 <= lo && lo < hi && hi <= 1.0)
    {
        return Err(Error::invalid(format!(
            "trunc: ({lo}, {hi}) are not probabilities in increasing order"
        )));
    }
    let censored = match event {
        PooledEvent::Censoring { censored } => censored,
        PooledEvent::Treatment { .. } => None,
    };
    let treated = match event {
        PooledEvent::Treatment { treated, .. } => Some(treated),
        PooledEvent::Censoring { .. } => None,
    };
    if let Some(t) = treated
        && t.len() != n
    {
        return Err(Error::from(InputError::Length {
            what: "treated",
            expected: n,
            found: t.len(),
        }));
    }
    for (name, m) in [("design", Some(design)), ("numerator", numerator)] {
        let Some(m) = m else { continue };
        if m.nrows() != n {
            return Err(Error::invalid(format!(
                "{name}: {} rows for {n} rows of data",
                m.nrows()
            )));
        }
        if m.ncols() == 0 {
            return Err(Error::invalid(format!("{name}: no columns")));
        }
        if !epistat_linalg::is_finite(m) {
            return Err(Error::invalid(format!("{name}: a value is not finite")));
        }
    }
    if let Some(c) = censored
        && c.len() != n
    {
        return Err(Error::from(InputError::Length {
            what: "censored",
            expected: n,
            found: c.len(),
        }));
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
            return Err(Error::invalid(format!(
                "id: rows {a} and {b} of one subject have the same stop"
            )));
        }
    }
    // The censoring events, and the rows up to the first of each subject.
    let mut event_rows = vec![false; n];
    match censored {
        _ if treated.is_some() => {}
        Some(c) => event_rows.copy_from_slice(c),
        None => {
            for (k, &r) in order.iter().enumerate() {
                let r = r as usize;
                let last = order
                    .get(k + 1)
                    .is_none_or(|&s| subject[s as usize] != subject[r]);
                event_rows[r] = last && surv.status()[r] == 0;
            }
        }
    }
    // The rows after which a subject leaves the fit.
    let ends: Vec<bool> = match event {
        PooledEvent::Censoring { .. } => event_rows.clone(),
        PooledEvent::Treatment {
            treated,
            history: TreatmentHistory::First,
        } => treated.to_vec(),
        PooledEvent::Treatment {
            history: TreatmentHistory::All,
            ..
        } => vec![false; n],
    };
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
        seen |= ends[r];
    }
    let response: &[bool] = treated.unwrap_or(&event_rows);
    let y: Vec<f64> = response.iter().map(|&e| f64::from(u8::from(e))).collect();
    let n_positive = (0..n).filter(|&r| active[r] && response[r]).count();
    let n_active = active.iter().filter(|&&a| a).count();
    if n_positive == 0 || n_positive == n_active {
        return Err(Error::invalid(if treated.is_some() {
            "treated: the rows of the fit need both treated and untreated rows"
        } else {
            "censored: the rows of the fit need both censoring events and rows that are not"
        }));
    }
    let what = if treated.is_some() {
        "treatment"
    } else {
        "censoring"
    };
    let (coefficients, probability) = fit_model(design, &y, &active, what)?;
    let numerator_fit = numerator
        .map(|m| fit_model(m, &y, &active, "numerator"))
        .transpose()?;
    let factor = |p: f64, row: usize| {
        if treated.is_some_and(|t| t[row]) {
            p
        } else {
            1.0 - p
        }
    };
    let mut cum = 1.0;
    for (k, &r) in order.iter().enumerate() {
        let r = r as usize;
        if k == 0 || subject[order[k - 1] as usize] != subject[r] {
            cum = 1.0;
        }
        weight_before[r] = cum;
        if active[r] {
            let num = numerator_fit
                .as_ref()
                .map_or(1.0, |(_, pn)| factor(pn[r], r));
            cum *= num / factor(probability[r], r);
        }
        weight[r] = cum;
    }
    let mut untruncated = None;
    let mut clamped = None;
    let mut limits = None;
    if let Some((lower, upper)) = trunc {
        let mut by_weight: Vec<usize> = (0..n).collect();
        by_weight.sort_by(|&a, &b| weight[a].total_cmp(&weight[b]));
        let sorted: Vec<f64> = by_weight.iter().map(|&r| weight[r]).collect();
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "the position is between 1 and the number of rows"
        )]
        let limit = |p: f64| -> Result<Limit> {
            let value = epistat_stats::quantile::type7(&sorted, p)
                .ok_or_else(|| Error::invalid("trunc: there are no weights"))?;
            // The position of R's type 7, as epistat_stats::quantile computes it.
            let index = 1.0 + (n - 1) as f64 * p;
            let (j, k) = (index.floor() as usize - 1, index.ceil() as usize - 1);
            Ok(Limit {
                value,
                rows: (by_weight[j], by_weight[k]),
                h: index - index.floor(),
            })
        };
        let (lo, hi) = (limit(lower)?, limit(upper)?);
        let outside = |w: &[f64]| {
            w.iter()
                .map(|&v| v < lo.value || v > hi.value)
                .collect::<Vec<_>>()
        };
        clamped = Some((outside(&weight), outside(&weight_before)));
        untruncated = Some(weight.clone());
        for w in weight.iter_mut().chain(weight_before.iter_mut()) {
            *w = w.clamp(lo.value, hi.value);
        }
        limits = Some([lo, hi]);
    }
    let summary = WeightSummary::new((0..n).filter(|&r| active[r]).map(|r| weight[r]));
    let (numerator_coefficients, numerator_probability) = match numerator_fit {
        Some((c, p)) => (Some(c), Some(p)),
        None => (None, None),
    };
    Ok(PooledIpcw {
        weight,
        weight_before,
        censored: event_rows,
        treated: treated.map(<[bool]>::to_vec),
        untruncated,
        clamped,
        limits,
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
