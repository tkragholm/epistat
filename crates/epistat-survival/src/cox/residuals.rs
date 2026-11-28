//! Residuals of a Cox fit, as `residuals.coxph` of `survival`: martingale,
//! deviance, score, Schoenfeld, scaled Schoenfeld, `dfbeta` and `dfbetas`.
//!
//! [`residuals`] takes the data, the design matrix and the options the model
//! was fitted with, the [`CoxModelFit`] and a [`ResidualOptions`] that names the
//! kind. One function with a kind, and not one function per kind, because the
//! kinds share the preparation of the data (the tie rule, the rows of positive
//! weight, the strata, the risk scores), the arguments, the `weighted` and
//! `collapse` options, and the result, and a caller that wants two kinds writes
//! two calls of the same form. The score residuals are those of
//! [`score_residuals`].
//!
//! # The kinds
//!
//! Row `i` has weight `w_i`, risk score `theta_i = exp(beta' x_i + o_i)`, an
//! interval `(a_i, t_i]` and an event indicator `delta_i`. At an event time `t`
//! of a stratum, the risk set `R(t)` is the rows of the stratum with
//! `a_j < t <= t_j`, and the terms of the partial likelihood are those of
//! [`fit`](super::fit): Breslow's \[1\] one term with the weight sum of the
//! events, Efron's \[2\] `d` terms with the mean weight `u` of the events and
//! the risk set less `k / d` of the events, `k = 0, ..., d - 1`. A term has the
//! denominators `D_0 = sum w_j theta_j` and `D_1 = sum w_j theta_j x_j` over its
//! risk set.
//!
//! - Martingale: `delta_i - theta_i (H(t_i) - H(a_i))`, with `H` the Breslow or
//!   Efron cumulative hazard, which at an event time is the sum of `u / D_0`
//!   over the terms. A row that has its event at `t` is in the risk set of the
//!   `k`-th term of that time with the fraction `1 - k / d` under Efron's
//!   approximation, so its hazard at `t` is the sum of `u (1 - k / d) / D_0`.
//!   The case weight enters the hazard through the risk set and not the factor
//!   `theta_i`, as in `coxph`.
//! - Deviance: from the martingale residual `m`, with `s` the status,
//!   `sign(m) sqrt(-2 (m + s log(1 - m / s)))`, and `sign(m) sqrt(-2 m)` where
//!   `s` is 0. With `collapse` the martingale residuals and the statuses are
//!   summed within the group first, so the deviance residual is that of the
//!   group. `residuals.coxph` has `s log(s - m)` in place of `s log(1 - m / s)`.
//!   The two are equal for `s` of 0 and 1, which is every row without
//!   `collapse`. For a group with `s` of 2 or more R's value is not the
//!   deviance, and is `NaN` with a warning when `m` is small, because
//!   `m + s log(s - m)` is then positive. This function gives the deviance of
//!   the group, `2 (s log(s / (s - m)) - m)` with the sign of `m`, which is
//!   never negative under the root.
//! - Score: the derivative of the log partial likelihood with respect to each
//!   coefficient through the weight of the row, as described for
//!   [`score_residuals`].
//! - Schoenfeld: one row per event, `x_i - xbar(t)`, with `xbar(t)` the mean of
//!   `x` over the risk set weighted by `w theta`, and under Efron's
//!   approximation the mean over the terms of `D_1 / D_0`. Rows of one time have
//!   the same `xbar`. The rows are in the order of stratum, event time and then
//!   the row of the data.
//! - Scaled Schoenfeld: `d V r + beta`, with `d` the number of events used, `V`
//!   the model-based covariance matrix of the fit and `r` the Schoenfeld
//!   residual \[3\].
//! - `dfbeta`: the score residual times `V`, the approximate change of the
//!   coefficients on leaving the row out, and `dfbetas` the same divided by the
//!   standard errors `sqrt(V_jj)`. `V` is the model-based covariance matrix also
//!   for a fit with clusters, as in `residuals.coxph`.
//!
//! # Weights and collapsing
//!
//! `weighted` multiplies a residual by the case weight of its row. `None`
//! takes the default of `residuals.coxph`, `false` for every kind except
//! `dfbeta` and `dfbetas`, where it is `true`. The score residual that
//! [`score_residuals`] returns contains the weight, so
//! unweighted it is divided by the weight. The deviance residual is not defined
//! with `weighted` because the status is not weighted, and it is refused.
//!
//! `collapse` sums the residuals of the rows of a group, as the `collapse`
//! argument of `residuals.coxph`: by the cluster of the data, by its id, or by
//! codes the caller gives, one per row. It applies to the martingale,
//! deviance, score, `dfbeta` and `dfbetas` kinds, which have one row per row of
//! the data. The Schoenfeld kinds have one row per event, and `collapse` is
//! refused for them. The groups come out in ascending order of their codes.
//!
//! # Rows of weight zero and aliased columns
//!
//! Rows of weight 0 are dropped, as in the fit, and `n_used` counts the rows
//! kept. The result has one entry for each row kept, with the row of the data in
//! `rows`, so a residual that `residuals.coxph` gives to a row of weight zero
//! is not given. A group that has no row of positive weight has no entry.
//! A column whose coefficient is `None` has `None` for every kind that has a
//! column per coefficient.
//!
//! # Not covered
//!
//! The kinds `partial` and `terms` of `residuals.coxph` are not computed, and
//! `ties` of the options must be Breslow or Efron.
//!
//! # Agreement with R
//!
//! The tests in `tests/r_cox_residuals.rs` compare every kind with
//! `residuals.coxph` on the `lung` data and on counting-process data with
//! strata, weights and a cluster, for Breslow and Efron ties. The residuals are
//! smooth functions of the coefficients and sums over at most `n` event times,
//! so the tolerance is stated there from the condition number of the
//! information matrix and `n` times the machine epsilon.
//!
//! # References
//!
//! 1. Breslow N. Covariance analysis of censored survival data. Biometrics
//!    1974;30(1):89-99. doi:10.2307/2529620
//! 2. Efron B. The efficiency of Cox's likelihood function for censored data.
//!    J Am Stat Assoc 1977;72(359):557-65. doi:10.1080/01621459.1977.10480613
//! 3. Therneau TM, Grambsch PM. Modeling survival data: extending the Cox
//!    model. New York: Springer; 2000. Chapters 4 and 6.
//! 4. Schoenfeld D. Partial residuals for the proportional hazards regression
//!    model. Biometrika 1982;69(1):239-41. doi:10.1093/biomet/69.1.239
//! 5. Therneau TM, Grambsch PM, Fleming TR. Martingale-based residuals for
//!    survival models. Biometrika 1990;77(1):147-60. doi:10.1093/biomet/77.1.147

#![expect(
    clippy::cast_possible_truncation,
    reason = "the data type checks that the row count fits in u32"
)]

use std::borrow::Cow;
use std::cmp::Ordering;

use epistat_core::InputError;
use epistat_linalg::MatRef;
use epistat_stats::sum::compensated;

use super::Ties;
use super::design::{CoxModelFit, CoxOptions, score_residuals};
pub(crate) use super::sweep::Sums;
use super::sweep::{RiskSets, terms};
use crate::check::{check_lengths, refuse_non_finite_matrix};
use crate::error::{Error, Result};
use crate::surv::SurvData;

/// The kinds of residual of [`residuals`], named as in `residuals.coxph`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CoxResidualKind {
    /// `type = "martingale"`, one value per row.
    Martingale,
    /// `type = "deviance"`, one value per row.
    Deviance,
    /// `type = "score"`, one value per row and column.
    Score,
    /// `type = "schoenfeld"`, one row per event and one value per column.
    Schoenfeld,
    /// `type = "scaledsch"`, the Schoenfeld residuals scaled by the covariance
    /// matrix and shifted by the coefficients.
    ScaledSchoenfeld,
    /// `type = "dfbeta"`, the score residuals times the covariance matrix.
    Dfbeta,
    /// `type = "dfbetas"`, `dfbeta` over the standard errors.
    Dfbetas,
}

/// `CoxResidualKind`, under its name in 0.3.
#[deprecated(since = "0.4.0", note = "renamed `CoxResidualKind`")]
pub type ResidualKind = CoxResidualKind;

/// The groups whose residuals are summed, as `collapse` of `residuals.coxph`.
///
/// The groups are a column of the data: to sum within other groups, give
/// their codes as the cluster or the id column of the data the residuals are
/// computed on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Collapse {
    /// One residual per row.
    #[default]
    None,
    /// Sum within the clusters of the data, which are required.
    Cluster,
    /// Sum within the ids of the data, which are required.
    Id,
}

/// What [`residuals`] takes besides the data, the design and the fit.
///
/// The kind has no default, so the options are made with [`new`](Self::new).
#[derive(Debug, Clone, PartialEq)]
pub struct ResidualOptions {
    /// Which residuals to compute.
    pub kind: CoxResidualKind,
    /// Multiply each residual by the case weight of its row. `None` takes the
    /// default of `residuals.coxph`: `true` for `dfbeta` and `dfbetas`, `false`
    /// for the other kinds.
    pub weighted: Option<bool>,
    /// The groups to sum the residuals over. The default is none.
    pub collapse: Collapse,
    /// The options of the fit, which must be those the fit was made with: the
    /// tie rule, the rule for the times and the event code. The offset is that
    /// of the data.
    pub cox: CoxOptions,
    #[doc(hidden)]
    pub _extensible: epistat_core::Extensible,
}

impl ResidualOptions {
    /// The residuals of `kind` for a fit made with `cox`, with the defaults of
    /// `residuals.coxph` for `weighted` and `collapse`.
    #[must_use]
    pub fn new(kind: CoxResidualKind, cox: CoxOptions) -> Self {
        Self {
            kind,
            weighted: None,
            collapse: Collapse::None,
            cox,
            _extensible: epistat_core::Extensible::NEW,
        }
    }
}

/// The residuals of a Cox fit.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct CoxResiduals {
    /// The kind that was computed.
    pub kind: CoxResidualKind,
    /// One entry per coefficient for the score, Schoenfeld, scaled Schoenfeld,
    /// `dfbeta` and `dfbetas` kinds and one entry for the martingale and
    /// deviance kinds. A column is `None` where the coefficient is `None`.
    /// Every column has one value per row of `rows`, per group of `groups` or
    /// per event of `time`.
    pub columns: Vec<Option<Vec<f64>>>,
    /// The row of the data of each value, in ascending order, for the kinds
    /// with one value per row without `collapse`; empty otherwise. For the
    /// Schoenfeld kinds it is the row of each event.
    pub rows: Vec<u32>,
    /// With `collapse`, the code of each group in ascending order: the cluster
    /// or id the caller gave, or the codes passed. Empty otherwise.
    pub groups: Vec<u32>,
    /// For the Schoenfeld kinds, the time of each event, rows ordered by
    /// stratum and time. Empty otherwise.
    pub time: Vec<f64>,
    /// For the Schoenfeld kinds with strata, the stratum code of each event as
    /// the caller gave it. Empty otherwise.
    pub stratum: Vec<u32>,
    /// The rows of the data.
    pub n_rows: usize,
    /// The rows with a weight above zero.
    pub n_used: usize,
}

/// One stratum's rows of positive weight, in descending exit order, ties in
/// ascending row order.
pub(crate) struct Stratum {
    /// The dense code of the stratum.
    pub(crate) code: u32,
    /// Centred covariates of the kept columns, `p` per row.
    pub(crate) x: Vec<f64>,
    /// `exp(eta)` of each row, with the linear predictor centred.
    pub(crate) theta: Vec<f64>,
    pub(crate) weight: Vec<f64>,
    /// `weight * theta`.
    pub(crate) risk: Vec<f64>,
    pub(crate) stop: Vec<f64>,
    pub(crate) start: Vec<f64>,
    pub(crate) is_event: Vec<bool>,
    /// The row of the data of each position.
    pub(crate) row: Vec<u32>,
    /// Positions in descending entry order and the entry times in that order,
    /// empty without delayed entry.
    by_start: Vec<u32>,
    start_sorted: Vec<f64>,
    /// Distinct event times, descending.
    pub(crate) event_times: Vec<f64>,
    /// The positions of the events at each time:
    /// `dying[dying_offset[k]..dying_offset[k + 1]]`.
    pub(crate) dying: Vec<u32>,
    pub(crate) dying_offset: Vec<usize>,
    /// For each kept column, whether it takes more than one value among the
    /// rows of the stratum.
    pub(crate) varies: Vec<bool>,
}

/// The rows, centred covariates and risk scores of a fit, by stratum.
pub(crate) struct Prepared {
    /// The number of kept columns.
    pub(crate) p: usize,
    /// The column of the design of each kept column.
    pub(crate) keep: Vec<usize>,
    pub(crate) efron: bool,
    pub(crate) strata: Vec<Stratum>,
    /// The rows of positive weight with an event.
    pub(crate) n_events: usize,
    pub(crate) n_rows: usize,
    pub(crate) n_used: usize,
}

impl Stratum {
    fn risk_sets(&self) -> RiskSets<'_> {
        RiskSets {
            x: &self.x,
            stop: &self.stop,
            by_start: &self.by_start,
            start_sorted: &self.start_sorted,
            event_times: &self.event_times,
            dying: &self.dying,
            dying_offset: &self.dying_offset,
        }
    }

    /// Visit every event time in descending order with the risk-set sums and
    /// the sums over the events at that time.
    pub(crate) fn sweep(
        &self,
        p: usize,
        active: &mut Vec<bool>,
        sums: &mut Sums,
        visit: impl FnMut(usize, &Sums),
    ) {
        self.risk_sets().sweep(p, &self.risk, active, sums, visit);
    }

    /// The terms at an event time as `(weight, fraction)`: the denominator is
    /// the risk-set sum less `fraction` of the sum over the events, and the term
    /// is multiplied by `weight`.
    pub(crate) fn terms(
        &self,
        position: usize,
        efron: bool,
    ) -> impl Iterator<Item = (f64, f64)> + use<> {
        let range = self.dying_offset[position]..self.dying_offset[position + 1];
        terms(&self.weight, &self.dying[range], efron)
    }

    /// The mean of `D_1 / D_0` over the terms, and the sums over the terms of
    /// `u / D_0` and of `u * fraction / D_0`, at an event time. `None` when a
    /// denominator is not positive and finite.
    pub(crate) fn at_time(
        &self,
        position: usize,
        sums: &Sums,
        efron: bool,
        mean: &mut [f64],
    ) -> Option<(f64, f64)> {
        mean.fill(0.0);
        let (mut hazard, mut late, mut count) = (0.0f64, 0.0f64, 0.0f64);
        for (u, fraction) in self.terms(position, efron) {
            let d0 = sums.risk.m0 - fraction * sums.dying.m0;
            if d0 <= 0.0 || !d0.is_finite() {
                return None;
            }
            hazard += u / d0;
            late += u * fraction / d0;
            for (a, m) in mean.iter_mut().enumerate() {
                let d1 = sums.risk.m1[a] - fraction * sums.dying.m1[a];
                *m += d1 / d0;
            }
            count += 1.0;
        }
        for m in mean.iter_mut() {
            *m /= count;
        }
        Some((hazard, late))
    }
}

fn not_finite_risk() -> Error {
    Error::undefined(
        "risk score",
        "the risk score or the denominator of a partial-likelihood term is not finite and positive at the coefficients",
    )
}

/// Whether the tie rule is Efron's, and an error for a rule that has no
/// residuals of this kind.
fn efron(ties: Ties) -> Result<bool> {
    if ties == Ties::Efron {
        Ok(true)
    } else if ties == Ties::Breslow {
        Ok(false)
    } else {
        Err(Error::unsupported(
            "ties",
            "ties: residuals are defined for Breslow and Efron ties only",
        ))
    }
}

/// The rows of positive weight, grouped by stratum, for the columns of `x`
/// whose coefficient is not `None`.
#[expect(
    clippy::too_many_lines,
    reason = "one pass that builds every per-stratum array"
)]
pub(crate) fn prepare(
    data: &SurvData<'_>,
    x: MatRef<'_, f64>,
    opts: &CoxOptions,
    coefficients: &[Option<f64>],
) -> Result<Prepared> {
    let efron = efron(opts.ties)?;
    check_lengths(&[("data", data.n_rows()), ("x", x.nrows())])?;
    check_lengths(&[
        ("x columns", x.ncols()),
        ("coefficients", coefficients.len()),
    ])?;
    if x.ncols() == 0 {
        return Err(Error::missing("x has no columns"));
    }
    refuse_non_finite_matrix(x, "x")?;
    if let Some(row) = coefficients
        .iter()
        .position(|c| c.is_some_and(|c| !c.is_finite()))
    {
        return Err(InputError::NotFinite {
            what: "coefficients",
            row,
            column: None,
        }
        .into());
    }
    let keep: Vec<usize> = (0..x.ncols())
        .filter(|&j| coefficients[j].is_some())
        .collect();
    if keep.is_empty() {
        return Err(Error::fit(
            "the fit has no coefficient: there are no residuals of a fit without an estimate",
        ));
    }
    let beta: Vec<f64> = keep
        .iter()
        .map(|&j| coefficients[j].unwrap_or(0.0))
        .collect();
    let p = keep.len();

    let data = data.clone().with_timefix(opts.timefix)?;
    let surv = data.followup();
    let code = match (opts.cause, surv.n_codes()) {
        (None, 1) => 1,
        (Some(code), _) => code,
        (None, _) => {
            return Err(Error::missing(
                "cause: the status has several codes, so the event code is required",
            ));
        }
    };
    let is_event: Vec<bool> = data.cause(code)?.iter().collect();
    let rows = data.n_rows();
    let stop = surv.stop();
    let start = surv.start();
    let weight = data
        .weight()
        .map_or_else(|| Cow::Owned(vec![1.0; rows]), Cow::Borrowed);
    let stratum = data
        .stratum()
        .map_or_else(|| Cow::Owned(vec![0u32; rows]), Cow::Borrowed);

    let used: Vec<u32> = (0..rows as u32)
        .filter(|&i| weight[i as usize] > 0.0)
        .collect();
    let n_used = used.len();
    let n_events = used.iter().filter(|&&i| is_event[i as usize]).count();

    // The linear predictor and the covariates are centred over the rows used,
    // which changes no residual and keeps the exponential in range.
    let eta_of = |i: usize| -> f64 {
        let mut value = 0.0f64;
        for (b, &j) in beta.iter().zip(&keep) {
            value += x[(i, j)] * b;
        }
        if let Some(offset) = data.offset() {
            value += offset[i];
        }
        value
    };
    let eta_mean = compensated(used.iter().map(|&i| eta_of(i as usize))) / n_used.max(1) as f64;
    let x_mean: Vec<f64> = keep
        .iter()
        .map(|&j| compensated(used.iter().map(|&i| x[(i as usize, j)])) / n_used.max(1) as f64)
        .collect();

    let mut order = used;
    order.sort_unstable_by(|&a, &b| {
        stratum[a as usize]
            .cmp(&stratum[b as usize])
            .then(stop[b as usize].total_cmp(&stop[a as usize]))
            .then(a.cmp(&b))
    });

    let mut strata = Vec::new();
    let mut from = 0usize;
    while from < order.len() {
        let code = stratum[order[from] as usize];
        let mut to = from;
        while to < order.len() && stratum[order[to] as usize] == code {
            to += 1;
        }
        let run = &order[from..to];
        let size = run.len();
        let mut s = Stratum {
            code,
            x: Vec::with_capacity(size * p),
            theta: Vec::with_capacity(size),
            weight: Vec::with_capacity(size),
            risk: Vec::with_capacity(size),
            stop: Vec::with_capacity(size),
            start: Vec::with_capacity(size),
            is_event: Vec::with_capacity(size),
            row: run.to_vec(),
            by_start: Vec::new(),
            start_sorted: Vec::new(),
            event_times: Vec::new(),
            dying: Vec::new(),
            dying_offset: Vec::new(),
            varies: vec![false; p],
        };
        let first = run[0] as usize;
        for &row in run {
            let i = row as usize;
            for (c, &j) in keep.iter().enumerate() {
                s.x.push(x[(i, j)] - x_mean[c]);
                if x[(i, j)]
                    .partial_cmp(&x[(first, j)])
                    .is_some_and(Ordering::is_ne)
                {
                    s.varies[c] = true;
                }
            }
            let theta = (eta_of(i) - eta_mean).exp();
            let risk = weight[i] * theta;
            if !risk.is_finite() {
                return Err(not_finite_risk());
            }
            s.theta.push(theta);
            s.weight.push(weight[i]);
            s.risk.push(risk);
            s.stop.push(stop[i]);
            s.start.push(start.map_or(f64::NEG_INFINITY, |v| v[i]));
            s.is_event.push(is_event[i]);
        }
        for position in 0..size {
            if s.is_event[position] {
                if s.event_times.last() != Some(&s.stop[position]) {
                    s.event_times.push(s.stop[position]);
                    s.dying_offset.push(s.dying.len());
                }
                s.dying.push(position as u32);
            }
        }
        s.dying_offset.push(s.dying.len());
        if start.is_some() {
            let mut by_start: Vec<u32> = (0..size as u32).collect();
            by_start.sort_unstable_by(|&a, &b| {
                s.start[b as usize]
                    .total_cmp(&s.start[a as usize])
                    .then(a.cmp(&b))
            });
            s.start_sorted = by_start.iter().map(|&k| s.start[k as usize]).collect();
            s.by_start = by_start;
        }
        strata.push(s);
        from = to;
    }
    Ok(Prepared {
        p,
        keep,
        efron,
        strata,
        n_events,
        n_rows: rows,
        n_used,
    })
}

/// The residuals of a Cox fit.
///
/// `data`, `x` and `opts.cox` are those the fit was made with, and `fit` is its
/// result. The residuals are those of the coefficients in `fit`. See the
/// module documentation for the kinds, `weighted` and `collapse`.
///
/// # Errors
///
/// An input error for the invalid input of [`fit`](super::fit), for a fit
/// whose coefficients or covariance matrix do not have the shape of `x`, for a
/// fit without a coefficient, for ties other than Breslow and Efron, for
/// `collapse` of the Schoenfeld kinds, for `collapse` by cluster or id on data
/// without them, for codes of another length than the data, for `weighted` with
/// the deviance kind, and for coefficients at which a risk score or the
/// denominator of a term of the partial likelihood is not finite and positive.
pub fn residuals(
    data: &SurvData<'_>,
    x: MatRef<'_, f64>,
    fit: &CoxModelFit,
    opts: &ResidualOptions,
) -> Result<CoxResiduals> {
    let p = x.ncols();
    check_lengths(&[
        ("x columns", p),
        ("fit coefficients", fit.coefficients.len()),
    ])?;
    if fit.covariance.dim() != p {
        return Err(Error::mismatch(
            "fit: the covariance matrix does not have one row and column per column of x",
        ));
    }
    if fit.n_used != data.n_used() {
        return Err(Error::fit(
            "fit: the number of rows used differs from the data's, so it is not a fit of these data",
        ));
    }
    let kind = opts.kind;
    let schoenfeld = matches!(
        kind,
        CoxResidualKind::Schoenfeld | CoxResidualKind::ScaledSchoenfeld
    );
    if schoenfeld && opts.collapse != Collapse::None {
        return Err(Error::options(
            "collapse: the Schoenfeld residuals have one row per event and cannot be collapsed",
        ));
    }
    if kind == CoxResidualKind::Deviance && opts.weighted == Some(true) {
        return Err(Error::options(
            "weighted: deviance residuals are defined for the unweighted martingale residuals",
        ));
    }
    let weighted = opts.weighted.unwrap_or(matches!(
        kind,
        CoxResidualKind::Dfbeta | CoxResidualKind::Dfbetas
    ));
    let prep = prepare(data, x, &opts.cox, &fit.coefficients)?;
    let rows = data.n_rows();
    let weight = data
        .weight()
        .map_or_else(|| Cow::Owned(vec![1.0; rows]), Cow::Borrowed);

    if schoenfeld {
        return schoenfeld_residuals(data, fit, &prep, kind, weighted);
    }

    // One value per row of the data for each column, zero for rows not used.
    let mut status_sum: Option<Vec<f64>> = None;
    let columns: Vec<Option<Vec<f64>>> = match kind {
        CoxResidualKind::Martingale | CoxResidualKind::Deviance => {
            let mut m = martingale(&prep)?;
            if kind == CoxResidualKind::Deviance {
                // The status of every row, which `collapse` sums with the
                // residuals.
                let event = opts.cox.cause.unwrap_or(1);
                let status: Vec<f64> = data
                    .cause(event)?
                    .iter()
                    .map(|e| if e { 1.0 } else { 0.0 })
                    .collect();
                status_sum = Some(status);
            } else if weighted {
                for (v, w) in m.iter_mut().zip(weight.iter()) {
                    *v *= w;
                }
            }
            vec![Some(m)]
        }
        CoxResidualKind::Score | CoxResidualKind::Dfbeta | CoxResidualKind::Dfbetas => {
            let mut columns = score_residuals(data, x, &fit.coefficients, &opts.cox)?;
            // The function returns the weighted residual, and zero for rows
            // not used.
            if kind == CoxResidualKind::Score {
                if !weighted {
                    for column in columns.iter_mut().flatten() {
                        for (v, w) in column.iter_mut().zip(weight.iter()) {
                            if *w > 0.0 {
                                *v /= w;
                            }
                        }
                    }
                }
                columns
            } else {
                dfbeta(
                    &columns,
                    fit,
                    p,
                    kind == CoxResidualKind::Dfbetas,
                    !weighted,
                    &weight,
                )
            }
        }
        CoxResidualKind::Schoenfeld | CoxResidualKind::ScaledSchoenfeld => unreachable!(),
    };

    let used: Vec<u32> = (0..rows as u32)
        .filter(|&i| weight[i as usize] > 0.0)
        .collect();
    let per_row_codes: Option<&[u32]> = match opts.collapse {
        Collapse::None => None,
        Collapse::Cluster => Some(
            data.cluster()
                .ok_or_else(|| Error::missing("collapse: the data have no clusters"))?,
        ),
        Collapse::Id => Some(
            data.id()
                .ok_or_else(|| Error::missing("collapse: the data have no ids"))?,
        ),
    };
    let keys = per_row_codes.map(|codes| {
        let mut unique: Vec<u32> = used.iter().map(|&i| codes[i as usize]).collect();
        unique.sort_unstable();
        unique.dedup();
        (codes, unique)
    });
    let Some((per_row, labels)) = keys else {
        let columns = columns
            .into_iter()
            .map(|c| c.map(|c| used.iter().map(|&i| c[i as usize]).collect()))
            .collect();
        let mut result = CoxResiduals {
            kind,
            columns,
            rows: used,
            groups: Vec::new(),
            time: Vec::new(),
            stratum: Vec::new(),
            n_rows: rows,
            n_used: prep.n_used,
        };
        if kind == CoxResidualKind::Deviance {
            let status = status_sum.unwrap_or_default();
            if let Some(Some(m)) = result.columns.first_mut() {
                for (v, &i) in m.iter_mut().zip(&result.rows) {
                    *v = deviance(*v, status[i as usize])?;
                }
            }
        }
        return Ok(result);
    };
    // Sum within the groups, in the order of the rows.
    let group_of = |row: u32| -> usize {
        labels
            .binary_search(&per_row[row as usize])
            .unwrap_or_else(|_| unreachable!("every used row's code is among the labels"))
    };
    let sum_by_group = |values: &[f64]| -> Vec<f64> {
        let mut totals = vec![0.0f64; labels.len()];
        for &i in &used {
            totals[group_of(i)] += values[i as usize];
        }
        totals
    };
    let mut collapsed: Vec<Option<Vec<f64>>> = columns
        .iter()
        .map(|c| c.as_ref().map(|c| sum_by_group(c)))
        .collect();
    if kind == CoxResidualKind::Deviance {
        let status = sum_by_group(&status_sum.unwrap_or_default());
        if let Some(Some(m)) = collapsed.first_mut() {
            for (v, s) in m.iter_mut().zip(&status) {
                *v = deviance(*v, *s)?;
            }
        }
    }
    Ok(CoxResiduals {
        kind,
        columns: collapsed,
        rows: Vec::new(),
        groups: labels_to_codes(opts.collapse, data, &labels),
        time: Vec::new(),
        stratum: Vec::new(),
        n_rows: rows,
        n_used: prep.n_used,
    })
}

/// The caller's codes of the groups, which the levels of the dense cluster and
/// id codes map back to.
fn labels_to_codes(collapse: Collapse, data: &SurvData<'_>, labels: &[u32]) -> Vec<u32> {
    let levels = match collapse {
        Collapse::Cluster => data.cluster_levels(),
        Collapse::Id => data.id_levels(),
        Collapse::None => None,
    };
    levels.map_or_else(
        || labels.to_vec(),
        |levels| {
            labels
                .iter()
                .map(|&l| levels.code(l as usize).unwrap_or(l))
                .collect()
        },
    )
}

/// `sign(m) sqrt(-2 (m + s log(1 - m / s)))` for the martingale residual `m` of
/// a row or group with status `s`, and `sign(m) sqrt(-2 m)` for `s` of 0.
fn deviance(m: f64, status: f64) -> Result<f64> {
    let inner = if status == 0.0 {
        m
    } else {
        let argument = status - m;
        if argument <= 0.0 || !argument.is_finite() {
            return Err(Error::undefined(
                "deviance residual",
                "the martingale residual is not below the status, so its deviance residual is not defined",
            ));
        }
        m + status * (-m / status).ln_1p()
    };
    let magnitude = (-2.0 * inner).max(0.0).sqrt();
    Ok(if m > 0.0 {
        magnitude
    } else if m < 0.0 {
        -magnitude
    } else {
        0.0
    })
}

/// The martingale residual of every row of the data, zero for rows not used.
fn martingale(prep: &Prepared) -> Result<Vec<f64>> {
    let mut out = vec![0.0f64; prep.n_rows];
    let p = prep.p;
    let mut active = Vec::new();
    let mut sums = Sums::new(p, false);
    let mut mean = vec![0.0f64; p];
    for stratum in &prep.strata {
        let count = stratum.event_times.len();
        let mut hazard = vec![0.0f64; count];
        let mut late = vec![0.0f64; count];
        let mut failed = false;
        stratum.sweep(p, &mut active, &mut sums, |position, at| {
            match stratum.at_time(position, at, prep.efron, &mut mean) {
                Some((h, l)) => {
                    hazard[position] = h;
                    late[position] = l;
                }
                None => failed = true,
            }
        });
        if failed {
            return Err(not_finite_risk());
        }
        let mut cumulative = vec![0.0f64; count + 1];
        for k in 0..count {
            cumulative[k + 1] = cumulative[k] + hazard[k];
        }
        for j in 0..stratum.stop.len() {
            let first = stratum
                .event_times
                .partition_point(|&t| t > stratum.stop[j]);
            let last = stratum
                .event_times
                .partition_point(|&t| t > stratum.start[j]);
            let mut expected = cumulative[last] - cumulative[first];
            let mut status = 0.0;
            if stratum.is_event[j] {
                expected -= late[first];
                status = 1.0;
            }
            out[stratum.row[j] as usize] = status - stratum.theta[j] * expected;
        }
    }
    Ok(out)
}

/// `dfbeta` or `dfbetas` from the weighted score residuals, one column per
/// coefficient of the design.
fn dfbeta(
    score: &[Option<Vec<f64>>],
    fit: &CoxModelFit,
    p: usize,
    scaled: bool,
    unweighted: bool,
    weight: &[f64],
) -> Vec<Option<Vec<f64>>> {
    let rows = weight.len();
    (0..p)
        .map(|j| {
            fit.coefficients[j]?;
            let mut column = vec![0.0f64; rows];
            for (i, out) in column.iter_mut().enumerate() {
                if weight[i] <= 0.0 {
                    continue;
                }
                let mut total = 0.0f64;
                for (k, s) in score.iter().enumerate() {
                    if let (Some(s), Some(v)) = (s, fit.covariance.get(k, j)) {
                        let r = if unweighted { s[i] / weight[i] } else { s[i] };
                        total += r * v;
                    }
                }
                *out = total;
            }
            if scaled {
                let se = fit.covariance.get(j, j)?.sqrt();
                for v in &mut column {
                    *v /= se;
                }
            }
            Some(column)
        })
        .collect()
}

/// The Schoenfeld rows of every event, in the order of stratum, time and row.
pub(crate) struct SchoenfeldRows {
    /// One vector per kept column.
    pub(crate) values: Vec<Vec<f64>>,
    pub(crate) time: Vec<f64>,
    pub(crate) row: Vec<u32>,
    /// The dense stratum code of each event.
    pub(crate) stratum: Vec<u32>,
    pub(crate) weight: Vec<f64>,
}

pub(crate) fn schoenfeld_rows(prep: &Prepared) -> Result<SchoenfeldRows> {
    let p = prep.p;
    let mut out = SchoenfeldRows {
        values: vec![Vec::with_capacity(prep.n_events); p],
        time: Vec::with_capacity(prep.n_events),
        row: Vec::with_capacity(prep.n_events),
        stratum: Vec::with_capacity(prep.n_events),
        weight: Vec::with_capacity(prep.n_events),
    };
    let mut active = Vec::new();
    let mut sums = Sums::new(p, false);
    for stratum in &prep.strata {
        let count = stratum.event_times.len();
        let mut means = vec![0.0f64; count * p];
        let mut mean = vec![0.0f64; p];
        let mut failed = false;
        stratum.sweep(p, &mut active, &mut sums, |position, at| {
            if stratum
                .at_time(position, at, prep.efron, &mut mean)
                .is_none()
            {
                failed = true;
            } else {
                means[position * p..(position + 1) * p].copy_from_slice(&mean);
            }
        });
        if failed {
            return Err(not_finite_risk());
        }
        for position in (0..count).rev() {
            for &k in
                &stratum.dying[stratum.dying_offset[position]..stratum.dying_offset[position + 1]]
            {
                let k = k as usize;
                for (a, column) in out.values.iter_mut().enumerate() {
                    column.push(stratum.x[k * p + a] - means[position * p + a]);
                }
                out.time.push(stratum.event_times[position]);
                out.row.push(stratum.row[k]);
                out.stratum.push(stratum.code);
                out.weight.push(stratum.weight[k]);
            }
        }
    }
    Ok(out)
}

fn schoenfeld_residuals(
    data: &SurvData<'_>,
    fit: &CoxModelFit,
    prep: &Prepared,
    kind: CoxResidualKind,
    weighted: bool,
) -> Result<CoxResiduals> {
    let mut rows = schoenfeld_rows(prep)?;
    let events = rows.time.len();
    if weighted {
        for column in &mut rows.values {
            for (v, w) in column.iter_mut().zip(&rows.weight) {
                *v *= w;
            }
        }
    }
    let p = fit.coefficients.len();
    let mut columns: Vec<Option<Vec<f64>>> = vec![None; p];
    if kind == CoxResidualKind::ScaledSchoenfeld {
        let ndead = events as f64;
        for &j in &prep.keep {
            let beta = fit.coefficients[j].unwrap_or(0.0);
            let column: Vec<f64> = (0..events)
                .map(|e| {
                    let r = compensated(prep.keep.iter().enumerate().map(|(d, &k)| {
                        rows.values[d][e] * fit.covariance.get(k, j).unwrap_or(0.0)
                    }));
                    r * ndead + beta
                })
                .collect();
            columns[j] = Some(column);
        }
    } else {
        for (values, &j) in rows.values.into_iter().zip(&prep.keep) {
            columns[j] = Some(values);
        }
    }
    let stratum = data.stratum_levels().map_or_else(Vec::new, |levels| {
        rows.stratum
            .iter()
            .map(|&s| levels.code(s as usize).unwrap_or(s))
            .collect()
    });
    Ok(CoxResiduals {
        kind,
        columns,
        rows: rows.row,
        groups: Vec::new(),
        time: rows.time,
        stratum,
        n_rows: prep.n_rows,
        n_used: prep.n_used,
    })
}
