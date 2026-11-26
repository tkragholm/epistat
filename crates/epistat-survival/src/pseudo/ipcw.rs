//! Pseudo-observations made by inverse probability of censoring weighting, with
//! a Cox model of the censoring, as `eventglm` makes them with
//! `model.censoring = "coxph"`.
//!
//! # Algorithm
//!
//! For a subject with follow-up time `T_i`, status `D_i` (0 for censored) and the
//! covariates `z_i` of the censoring model, and a time `t`, the outcome of the
//! target is
//!
//! ```text
//! Survival                  V_i = 1 - 1(T_i < t, D_i = 1)
//! CumulativeIncidence(k)    V_i = 1(T_i < t, D_i = k)
//! RestrictedMean            V_i = min(T_i, t)
//! RestrictedTimeLost(k)     V_i = (t - min(T_i, t)) 1(D_i = k)
//! ```
//!
//! and it is observed when `I_i = 1(T_i >= t or D_i > 0)`. The probability of not
//! being censored, `G_i = G(min(T_i, t) | z_i)`, comes from the Cox model of the
//! censoring, fitted to `(T_i, 1(D_i = 0))` on `z`, whose survival function at the
//! subject's covariates is `exp(-theta_i H_0)` with the Efron or Breslow
//! baseline of [`cox::predict`](crate::cox::predict), as `survfit.coxph` gives it
//! \[1\]\[2\]. A time before the first exit time has `G = 1`. With
//! `X_i = I_i V_i / G_i` and `n` subjects, the pseudo-observation is
//!
//! ```text
//! Binder:   theta_hat + (n - 1)(theta_hat - mean over j != i of X_j) = X_i,   theta_hat = mean X,
//! Hajek:    n theta_hat - (n - 1) theta_hat(-i),   theta_hat = sum X / sum (I / G),
//! ```
//!
//! where `theta_hat(-i)` leaves out subject `i` from both sums. The Binder form
//! reduces to the weighted outcome itself. It is `eventglm`'s default. The
//! regression on the pseudo-observations is [`Pseudo::regress`].
//!
//! # The other censoring models of `eventglm`
//!
//! `model.censoring = "independent"`, the default of `eventglm`, is not weighted.
//! It is the exact jackknife of the Kaplan-Meier or Aalen-Johansen estimate of
//! the whole sample, which is [`Method::Jackknife`](super::Method) without a
//! stratum column. `"stratified"` is the exact jackknife within the strata of the
//! interaction of the censoring covariates, which is [`Method::Jackknife`](super::Method)
//! with that interaction as the stratum column. Both are checked against
//! `eventglm`. `"aareg"`, a censoring model of Aalen's additive regression, is
//! not provided.
//!
//! # Differences from eventglm
//!
//! `eventglm` reads `coxsurv$time <= t` and falls back to the first time of the
//! curve when no time is that early, so for a time before every exit time it takes
//! `G` at the first exit time, which is below 1 when someone is censored there. The
//! probability of not being censored is 1 before any censoring, and that is what is
//! used here. The difference concerns only times before the first exit time.
//!
//! # References
//!
//! 1. Binder D. A. (1992) Fitting Cox's proportional hazards models from survey
//!    data, Biometrika 79: 139-147.
//! 2. Sachs M. C., Gabriel E. E. (2022) Event history regression with
//!    pseudo-observations: computation and an implementation in R, J Stat
//!    Softw 102(9).

use epistat_core::InputError;
use epistat_linalg::MatRef;

use crate::check::check_ascending;
use crate::cox::predict::{FittedModel, NewData, PredictOptions, PredictType, predict};
use crate::cox::{CoxOptions, fit};
use crate::error::{Error, Result};
use crate::surv::SurvData;
use crate::timefix::TimeFix;

use super::{GroupEstimate, Pseudo, Target};

/// How the pseudo-observation is formed from the weighted outcome, `ipcw.method`
/// of `eventglm`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IpcwMethod {
    /// The mean of the weighted outcomes, `"binder"`.
    #[default]
    Binder,
    /// The ratio of the sums of the weighted outcome and of the weights,
    /// `"hajek"`.
    Hajek,
}

/// What [`pseudo_ipcw`] does besides the data, the censoring covariates and the
/// times.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IpcwOptions {
    /// What is estimated. It has no default. [`Target::CumulativeHazard`] and
    /// [`Target::CauseHazard`] are not outcomes of a subject and are refused.
    pub target: Target,
    /// How the weighted outcome becomes a pseudo-observation. `Binder`, as
    /// `eventglm`.
    pub method: IpcwMethod,
    /// How near-equal times are treated, `survival`'s rule by default.
    pub timefix: TimeFix,
}

impl IpcwOptions {
    /// Options for a target with `eventglm`'s Binder form and `survival`'s tie
    /// rule.
    #[must_use]
    pub fn new(target: Target) -> Self {
        Self {
            target,
            method: IpcwMethod::default(),
            timefix: TimeFix::default(),
        }
    }
}

/// The pseudo-observations of every row of `data` at `times`, by inverse
/// probability of censoring weighting with a Cox model of the censoring on the
/// columns of `censoring`.
///
/// Every row is a subject, so the data have no delayed entry, case weights,
/// strata, clusters or ids. `censoring` has one row per row of `data` and may
/// have no column, which models the censoring without covariates. The result is
/// in the form of [`pseudo`](super::pseudo()) with one group, and
/// [`Pseudo::subject`] the row numbers. See [inverse probability of censoring
/// weighting](crate::pseudo#inverse-probability-of-censoring-weighting).
///
/// # Errors
///
/// An input error when `times` is empty, not finite or not strictly
/// ascending, the data have a start, weights, a stratum, cluster or id column, or
/// no row, `censoring` has another number of rows than the data or a value that
/// is not finite, the target is a hazard or needs a single event code and the data
/// have several, the cause is outside `1..=n_codes`, the tie rule fails, the
/// Cox model of the censoring cannot be fitted, has an aliased coefficient or did
/// not converge, or a subject that is observed has a probability of not being
/// censored of 0.
pub fn pseudo_ipcw(
    data: &SurvData<'_>,
    censoring: MatRef<'_, f64>,
    times: &[f64],
    opts: &IpcwOptions,
) -> Result<Pseudo> {
    if times.is_empty() {
        return Err(Error::from(InputError::Empty { what: "times" }));
    }
    check_ascending(times, "times", true)?;
    if data.weight().is_some()
        || data.stratum().is_some()
        || data.cluster().is_some()
        || data.id().is_some()
        || data.followup()?.start().is_some()
    {
        return Err(Error::invalid(
            "inverse probability of censoring weighting takes each row as a subject, \
             without delayed entry, weights, strata, clusters or ids",
        ));
    }
    let n = data.n_rows();
    if n == 0 {
        return Err(Error::invalid("no row has positive weight"));
    }
    if censoring.nrows() != n {
        return Err(Error::invalid(format!(
            "the censoring covariates have {} rows and the data {n}",
            censoring.nrows()
        )));
    }
    let data = data.clone().with_timefix(opts.timefix)?;
    let surv = data.followup()?;
    let (stop, status) = (surv.stop(), surv.status());
    let n_codes = surv.n_codes();
    match opts.target {
        Target::CumulativeHazard | Target::CauseHazard { .. } => {
            return Err(Error::invalid(
                "a cumulative hazard is not the outcome of a subject",
            ));
        }
        Target::Survival | Target::RestrictedMean if n_codes != 1 => {
            return Err(Error::invalid(
                "the survival and the restricted mean need a single event code",
            ));
        }
        _ => {}
    }
    let cause = opts
        .target
        .cause()
        .map_or(1, |c| i32::try_from(c).unwrap_or(0));
    if let Some(c) = opts.target.cause() {
        data.cause(c)?;
    }

    // The model of the censoring: the censoring is the event.
    let censored: Vec<i32> = status.iter().map(|&s| i32::from(s == 0)).collect();
    let cdata = SurvData::builder(stop, &censored).build()?;
    let copts = CoxOptions {
        timefix: TimeFix::Exact,
        ..CoxOptions::default()
    };
    let model = fit(&cdata, censoring, &copts)?;
    if model.coefficients.iter().any(Option::is_none) {
        return Err(Error::invalid(
            "the model of the censoring has a column that is a combination of the others",
        ));
    }
    if !model.converged {
        return Err(Error::NotConverged {
            iterations: model.iterations as usize,
        });
    }
    let fitted = FittedModel {
        data: &cdata,
        x: censoring,
        fit: &model,
        options: &copts,
    };
    let last_time = stop.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let k = times.len();
    let mut values = vec![0.0f64; n * k];
    let mut estimate = vec![0.0f64; k];
    let mut at = vec![0.0f64; n];
    #[expect(clippy::cast_precision_loss, reason = "a row count is far below 2^53")]
    let (n_f, n_less) = (n as f64, (n - 1) as f64);
    for (slot, &t) in times.iter().enumerate() {
        for (a, &s) in at.iter_mut().zip(stop) {
            *a = s.min(t);
        }
        let prediction = predict(
            &fitted,
            PredictType::Survival,
            Some(&NewData {
                x: censoring,
                stratum: None,
                start: None,
                stop: Some(&at),
                offset: None,
            }),
            &PredictOptions::default(),
        )?;
        let g = prediction.fit;
        // The weighted outcome, and whether the row is observed.
        let mut weighted = vec![0.0f64; n];
        let mut observed = vec![0.0f64; n];
        for i in 0..n {
            let (ti, di) = (stop[i], status[i]);
            let v = match opts.target {
                Target::Survival => 1.0 - f64::from(u8::from(ti < t && di > 0)),
                Target::CumulativeIncidence { .. } => f64::from(u8::from(ti < t && di == cause)),
                Target::RestrictedMean => ti.min(t),
                Target::RestrictedTimeLost { .. } => {
                    (t - ti.min(t)) * f64::from(u8::from(di == cause))
                }
                _ => unreachable!("hazards were refused"),
            };
            if ti >= t || di > 0 {
                if g[i].is_nan() || g[i] <= 0.0 {
                    return Err(Error::invalid(format!(
                        "row {i} is observed and has a probability of 0 of not being censored"
                    )));
                }
                weighted[i] = v / g[i];
                observed[i] = 1.0 / g[i];
            }
        }
        match opts.method {
            IpcwMethod::Binder => {
                estimate[slot] = weighted.iter().sum::<f64>() / n_f;
                for i in 0..n {
                    values[i * k + slot] = weighted[i];
                }
            }
            IpcwMethod::Hajek => {
                let (sum_x, sum_i) = (weighted.iter().sum::<f64>(), observed.iter().sum::<f64>());
                let theta = sum_x / sum_i;
                estimate[slot] = theta;
                for i in 0..n {
                    let rest = (sum_x - weighted[i]) / (sum_i - observed[i]);
                    values[i * k + slot] = n_f * theta - n_less * rest;
                }
            }
        }
    }
    Ok(Pseudo {
        times: times.to_vec(),
        group: vec![0; n],
        subject: (0..n)
            .map(|i| u32::try_from(i).map_err(|_| Error::invalid("more rows than u32::MAX")))
            .collect::<Result<_>>()?,
        values,
        groups: vec![GroupEstimate {
            group: 0,
            n_subjects: n,
            estimate,
            beyond: times.iter().map(|&t| t > last_time).collect(),
        }],
        n_rows: n,
        n_used: n,
    })
}
