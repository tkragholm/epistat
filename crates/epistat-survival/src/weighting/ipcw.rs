//! Inverse probability of censoring weights from a Kaplan-Meier or Cox model of
//! the censoring hazard.
//!
//! # Algorithm
//!
//! Censoring is treated as the event of a second process. A row is a censoring
//! event when its status is 0 and, with an `id` column, it is the last row of its
//! subject, so that a row that ends at a change of covariates is not one. The
//! censoring survival of a subject at time `tau`, given entry at `s_0`, is
//!
//! ```text
//! G(tau) = exp(-Lambda(tau)),   Lambda(tau) = sum over the rows k of the subject of
//!          exp(x_k' b) (H_0(min(u_k, tau)) - H_0(s_k))      (Cox)
//! G(tau) = S_KM(tau) / S_KM(s_0)                            (Kaplan-Meier)
//! ```
//!
//! with `H_0` the Breslow or Efron baseline cumulative hazard of the censoring
//! process, from [`crate::cox::basehaz`], `S_KM` the Kaplan-Meier curve of
//! censoring from [`crate::survfit::estimate`], and `(s_k, u_k]` the rows. With
//! the Kaplan-Meier model the groups are the strata of the data. The Cox model
//! has the strata of the data and covariates `x` that may change between rows,
//! which is the time-varying censoring model of Robins and Finkelstein (2000).
//!
//! Two weights are returned for each row. At the stop of the row, `1 / G(u-)`
//! with `G(u-)` the survival just before the stop: the weight of an event or
//! of an outcome observed at the stop, which requires being uncensored before it.
//! At the start, `1 / G(s)`: the weight that a row carries over its interval
//! when the follow-up is split at cut points with [`crate::split::split`], so that
//! the weight changes at each cut. It is the piecewise-constant form of the
//! time-varying weight and approaches it as the cut points become dense.
//!
//! Stabilised weights multiply by the Kaplan-Meier censoring survival of the
//! stratum in the same form, `G_0(u-) / G_0(s_0)`.
//!
//! # Edge cases
//!
//! A censoring survival of 0 before a row's time gives an infinite weight and
//! is an error. A coefficient of the censoring model that is `None` because
//! its column is aliased counts as 0, as in `predict.coxph`. Rows of one
//! subject are taken not to overlap and are ordered by their stops.
//!
//! # References
//!
//! Robins JM, Finkelstein DM. Correcting for noncompliance and dependent
//! censoring in an AIDS clinical trial with inverse probability of censoring
//! weighted (IPCW) log-rank tests. Biometrics 2000;56(3):779-88.
//! Robins JM, Rotnitzky A. Recovery of information and adjustment for
//! dependent censoring using surrogate markers. In: AIDS Epidemiology,
//! Birkhauser 1992:297-331.

use epistat_core::{Covariance, InputError};
use epistat_linalg::MatRef;

use crate::cox::{
    BaselineOptions, Centring, CoxOptions, FittedModel, Ties, Variance, basehaz, fit as cox_fit,
};
use crate::error::{Error, Result};
use crate::step::Steps;
use crate::surv::{SurvBuf, SurvData};
use crate::survfit::{self, SurvfitOptions};
use crate::timefix::TimeFix;

/// The model of the censoring hazard.
#[derive(Debug, Clone, Copy)]
pub enum CensoringModel<'a> {
    /// The Kaplan-Meier curve of censoring, by the strata of the data.
    KaplanMeier,
    /// A Cox model of the censoring hazard on the columns of `x`, one row per
    /// row of the data, stratified by the strata of the data.
    Cox {
        /// The design of the censoring model, which may change between rows of
        /// a subject.
        x: MatRef<'a, f64>,
    },
}

/// What [`fit`] does besides the data and the model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IpcwOptions {
    /// Multiply by the Kaplan-Meier censoring survival of the stratum,
    /// stabilising the weights. The default is unstabilised weights.
    pub stabilise: bool,
    /// The ties of the Cox model of censoring, `Efron` as in `coxph`.
    pub ties: Ties,
    /// The rule that makes times equal when they differ by rounding error,
    /// applied to the times before anything is estimated.
    pub timefix: TimeFix,
}

impl Default for IpcwOptions {
    fn default() -> Self {
        Self {
            stabilise: false,
            ties: Ties::Efron,
            timefix: TimeFix::default(),
        }
    }
}

/// The censoring weights of every row.
#[derive(Debug, Clone, PartialEq)]
pub struct Ipcw {
    /// `1 / G(stop-)` of each row, stabilised when asked.
    pub weight_at_stop: Vec<f64>,
    /// `1 / G(start)` of each row, stabilised when asked.
    pub weight_at_start: Vec<f64>,
    /// Whether each row is a censoring event.
    pub censored: Vec<bool>,
    /// The coefficients of the Cox model of censoring, `None` for the
    /// Kaplan-Meier model and for an aliased column.
    pub coefficients: Option<Vec<Option<f64>>>,
}

/// The censoring weights of the rows of `data` from `model`.
///
/// # Errors
///
/// An input error for interval-censored data, for a design with another
/// number of rows than the data or a value that is not finite, when a subject
/// has two last rows at the same stop, when the Cox model of censoring does
/// not converge, and when the censoring survival is 0 before a row.
pub fn fit(data: &SurvData<'_>, model: CensoringModel<'_>, opts: &IpcwOptions) -> Result<Ipcw> {
    fit_parts(data, model, opts).map(|(weights, _)| weights)
}

/// What the variance of an outcome that uses the weights needs besides the
/// weights.
pub(super) struct Parts<'a> {
    /// The stops and starts of the censoring process, with the tie rule applied.
    pub stop: Vec<f64>,
    pub start: Option<Vec<f64>>,
    pub censored: Vec<bool>,
    /// The dense stratum of each row, and the number of strata.
    pub group: Vec<usize>,
    pub n_groups: usize,
    /// The design of the Cox model, its coefficients and covariance, `None` for
    /// the Kaplan-Meier model.
    pub x: Option<MatRef<'a, f64>>,
    pub coefficients: Option<Vec<Option<f64>>>,
    pub covariance: Option<Covariance>,
    pub ties: Ties,
    pub stabilise: bool,
    /// The first start of the subject of each row.
    pub entry: Vec<f64>,
    /// The rows in the order of their subject and stop, the rows of a subject
    /// next to each other, and the subject of each row.
    pub order: Vec<usize>,
    pub subject: Vec<usize>,
}

pub(super) fn fit_parts<'a>(
    data: &SurvData<'_>,
    model: CensoringModel<'a>,
    opts: &IpcwOptions,
) -> Result<(Ipcw, Parts<'a>)> {
    let surv = data.followup()?;
    let n = data.n_rows();
    let fixed = {
        let mut builder =
            SurvBuf::builder(surv.stop().to_vec(), surv.status().to_vec()).n_codes(surv.n_codes());
        if let Some(s) = surv.start() {
            builder = builder.start(s.to_vec());
        }
        builder.build()?
    };
    let fixed = fixed.as_data().with_timefix(opts.timefix)?;
    let (start, stop) = (
        fixed.followup()?.start().map(<[f64]>::to_vec),
        fixed.followup()?.stop().to_vec(),
    );
    let status = surv.status();

    // The censoring events.
    let mut censored: Vec<bool> = status.iter().map(|&s| s == 0).collect();
    let id = data.id();
    if let Some(id) = id {
        let mut last = vec![(f64::NEG_INFINITY, usize::MAX, 0u32); data.n_ids()];
        for r in 0..n {
            let slot = &mut last[id[r] as usize];
            match stop[r].total_cmp(&slot.0) {
                std::cmp::Ordering::Greater => *slot = (stop[r], r, 1),
                std::cmp::Ordering::Equal => slot.2 += 1,
                std::cmp::Ordering::Less => {}
            }
        }
        if let Some(bad) = last.iter().find(|l| l.2 > 1) {
            return Err(Error::invalid(format!(
                "id: row {} has the same stop as another row of its subject",
                bad.1
            )));
        }
        for r in 0..n {
            censored[r] = censored[r] && last[id[r] as usize].1 == r;
        }
    }

    // The data of the censoring process, with the times as fixed.
    let flipped: Vec<i32> = censored.iter().map(|&c| i32::from(c)).collect();
    let mut builder = SurvBuf::builder(stop.clone(), flipped);
    if let Some(s) = &start {
        builder = builder.start(s.clone());
    }
    if let Some(w) = data.weight() {
        builder = builder.weight(w.to_vec());
    }
    if let Some(s) = data.stratum() {
        builder = builder.stratum(s.to_vec());
    }
    let buf = builder.build()?;
    let process = buf.as_data();
    let n_groups = data.n_strata().max(1);
    let group_of = |r: usize| data.stratum().map_or(0, |s| s[r] as usize);
    let label = |g: usize| data.stratum_levels().map_or(0, |l| l.code(g).unwrap_or(0));

    // The Kaplan-Meier censoring survival of each stratum, on the log scale.
    let km = survfit::estimate(
        &process,
        &SurvfitOptions {
            timefix: TimeFix::Exact,
            conf_type: survfit::ConfType::None,
            ..SurvfitOptions::default()
        },
    )?;
    let km_steps: Vec<Steps> = (0..n_groups)
        .map(|g| {
            km.group(label(g)).map_or(
                Steps {
                    time: Vec::new(),
                    first: 0.0,
                    value: Vec::new(),
                },
                |c| Steps {
                    time: c.time.clone(),
                    first: 0.0,
                    value: c.surv.iter().map(|&s| s.ln()).collect(),
                },
            )
        })
        .collect();

    // The cumulative hazard of the censoring model along each row, the part
    // of the earlier rows of the subject and the part of the row up to a time.
    let mut coefficients = None;
    let mut covariance = None;
    let mut risk = vec![1.0; n];
    let mut baseline: Vec<Steps> = Vec::new();
    if let CensoringModel::Cox { x } = model {
        if x.nrows() != n {
            return Err(Error::from(InputError::Length {
                what: "x",
                expected: n,
                found: x.nrows(),
            }));
        }
        let cox_opts = CoxOptions {
            ties: opts.ties,
            timefix: TimeFix::Exact,
            ..CoxOptions::default()
        };
        let model_fit = cox_fit(&process, x, &cox_opts)?;
        if !model_fit.converged {
            return Err(Error::NotConverged {
                iterations: model_fit.iterations as usize,
            });
        }
        let zero = vec![0.0; x.ncols()];
        let base = basehaz(
            &FittedModel {
                data: &process,
                x,
                fit: &model_fit,
                options: &cox_opts,
            },
            &BaselineOptions {
                centring: Centring::Given(&zero),
                hazard: None,
                variance: Variance::Model,
            },
        )?;
        for r in 0..n {
            let eta: f64 = (0..x.ncols())
                .map(|j| x[(r, j)] * model_fit.coefficients[j].unwrap_or(0.0))
                .sum();
            risk[r] = eta.exp();
        }
        baseline = (0..n_groups)
            .map(|g| {
                base.strata.get(g).map_or(
                    Steps {
                        time: Vec::new(),
                        first: 0.0,
                        value: Vec::new(),
                    },
                    |s| Steps {
                        time: s.time.clone(),
                        first: 0.0,
                        value: s.cumulative_hazard.clone(),
                    },
                )
            })
            .collect();
        coefficients = Some(model_fit.coefficients.clone());
        covariance = Some(model_fit.covariance.clone());
    }
    let cox_model = matches!(model, CensoringModel::Cox { .. });

    // Entry of each subject and the hazard accumulated over earlier rows.
    let mut order: Vec<usize> = (0..n).collect();
    if let Some(id) = id {
        order.sort_by(|&a, &b| id[a].cmp(&id[b]).then(stop[a].total_cmp(&stop[b])));
    }
    let mut prior = vec![0.0; n];
    let mut entry = vec![0.0; n];
    {
        let mut previous: Option<usize> = None;
        let mut carried = 0.0;
        let mut first_start = 0.0;
        for &r in &order {
            let g = group_of(r);
            let same = id.is_some_and(|id| previous.is_some_and(|q| id[q] == id[r]));
            if same && cox_model {
                let q = previous.unwrap_or(r);
                let h = &baseline[g];
                carried += risk[q] * (h.at(stop[q]) - h.at(start.as_ref().map_or(0.0, |s| s[q])));
            } else if !same {
                carried = 0.0;
                first_start = start.as_ref().map_or(0.0, |s| s[r]);
            }
            if cox_model {
                prior[r] = carried;
            }
            entry[r] = first_start;
            previous = Some(r);
        }
    }

    let mut weight_at_stop = vec![0.0; n];
    let mut weight_at_start = vec![0.0; n];
    for r in 0..n {
        let g = group_of(r);
        let s = start.as_ref().map_or(0.0, |s| s[r]);
        let (lambda_stop, lambda_start) = if cox_model {
            let h = &baseline[g];
            (prior[r] + risk[r] * (h.before(stop[r]) - h.at(s)), prior[r])
        } else {
            let k = &km_steps[g];
            (
                -(k.before(stop[r]) - k.at(entry[r])),
                -(k.at(s) - k.at(entry[r])),
            )
        };
        let (num_stop, num_start) = if opts.stabilise {
            let k = &km_steps[g];
            (
                -(k.before(stop[r]) - k.at(entry[r])),
                -(k.at(s) - k.at(entry[r])),
            )
        } else {
            (0.0, 0.0)
        };
        let (ws, w0) = (
            (lambda_stop - num_stop).exp(),
            (lambda_start - num_start).exp(),
        );
        if !(ws.is_finite() && w0.is_finite()) {
            return Err(Error::invalid(format!(
                "the censoring survival is 0 before row {r}, so its weight is infinite"
            )));
        }
        weight_at_stop[r] = ws;
        weight_at_start[r] = w0;
    }
    let cox_x = match model {
        CensoringModel::Cox { x } => Some(x),
        CensoringModel::KaplanMeier => None,
    };
    let parts = Parts {
        stop,
        start,
        censored: censored.clone(),
        group: (0..n).map(group_of).collect(),
        n_groups,
        x: cox_x,
        coefficients: coefficients.clone(),
        covariance,
        ties: opts.ties,
        stabilise: opts.stabilise,
        entry,
        subject: (0..n).map(|r| id.map_or(r, |id| id[r] as usize)).collect(),
        order,
    };
    Ok((
        Ipcw {
            weight_at_stop,
            weight_at_start,
            censored,
            coefficients,
        },
        parts,
    ))
}
