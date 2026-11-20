//! Cox regression on one covariate, with Breslow or Efron ties, delayed entry,
//! strata, case weights and a cluster-robust variance.
//!
//! This module fits one covariate with [`fit_cox_single_covariate`] and
//! [`CoxData`]. With one covariate the score and the information are scalars,
//! which keeps the Newton step and the sweep over the risk sets simple. A design
//! matrix is fitted by [`fit`] on a [`SurvData`](crate::SurvData), which
//! follows `survival::coxph` in its iteration, applies its tie rule to the
//! times and is documented in the [`design`] module. The two stop by different
//! rules, so a fit of one column by either can differ in the last digits.
//! [`fit_cox_single_covariate`] and [`CoxData`] keep
//! their form for now. [`fit_cox`] and [`CoxDesign`] are deprecated.
//!
//! # Algorithm
//!
//! Each row `i` has a covariate `x_i`, a case weight `w_i > 0`, an entry time
//! `a_i`, an exit time `t_i` and an event indicator `delta_i`. Rows are in
//! strata, and a stratum has its own baseline hazard. At an event time `t` the
//! risk set `R(t)` of a stratum is the rows with `a_j < t <= t_j`. With
//! `theta_j = exp(beta x_j)` the weighted sums over the risk set are
//!
//! ```text
//! S_r(t) = sum over R(t) of w_j theta_j x_j^r,    r = 0, 1, 2.
//! ```
//!
//! Let `D(t)` be the events at `t`, `d` their number, `w_bar` the mean of their
//! weights, and `E_r(t)` the sums above taken over `D(t)` only. Breslow's
//! approximation \[1\] gives every tied event the whole risk set as its
//! denominator. Efron's approximation \[2\] gives the `k`-th tied event the risk
//! set less `k / d` of the tied events, `k = 0, ..., d - 1`:
//!
//! ```text
//! Breslow:  one term,   weight sum of w over D(t),  denominators S_r
//! Efron:    d terms,    weight w_bar each,          denominators S_r - (k / d) E_r
//! ```
//!
//! For a term with denominators `D_0, D_1, D_2` and weight `u`, the log partial
//! likelihood gains `u (beta x - log D_0)` over the covariates of the events,
//! so the score and the information are
//!
//! ```text
//! U(beta) = sum over t of [ sum over D(t) of w x  -  sum over terms of u D_1 / D_0 ]
//! I(beta) = sum over t, terms of u [ D_2 / D_0  -  (D_1 / D_0)^2 ].
//! ```
//!
//! Newton-Raphson starts at `beta = 0` and steps by `U / I`. It stops when the
//! step is below `1e-10` in absolute value, or after 50 steps, and then
//! `converged` is false. The model-based standard error is `1 / sqrt(I)` with
//! `I` at the iterate before the last step, so it differs from the one at the
//! estimate by at most half the range of the covariate times the last step,
//! relative to itself. The robust standard error uses the information at the
//! estimate.
//! Strata add their scores and informations. A stratum with no event adds
//! nothing and is skipped.
//!
//! The sweep visits event times in descending order within a stratum, so a row
//! joins the risk set when the sweep passes its exit and leaves when it passes
//! its entry. Both are monotone, so after one sort per stratum each Newton step
//! touches every row twice. The covariate is mapped to its distinct values, so
//! `exp(beta x)` is evaluated once per value, not once per row.
//!
//! # Cluster-robust variance
//!
//! With cluster codes the robust standard error is the sandwich estimate for
//! one parameter \[3\]:
//!
//! ```text
//! se_robust = sqrt( sum over clusters g of U_g^2 ) / I(beta_hat)
//! ```
//!
//! where `U_g` is the sum within cluster `g` of the score residuals. The score
//! residual of row `j` is the derivative of the log partial likelihood with
//! respect to `beta` through the weight of `j`. A row with an event contributes
//! `w_j (x_j - m)` with `m` the mean over the tied events of `D_1 / D_0`. Every
//! row contributes `-w_j theta_j sum_t (x_j h_t - c_t)` over the event times it
//! is at risk at, where `h_t` is the sum of `u / D_0` over the terms at `t` and
//! `c_t` the sum of `u D_1 / D_0^2`. Under Efron's approximation a tied event
//! is at risk in only part of the terms of its own time, and a final correction
//! adds that back. The residuals add up to the score. The sum of squares is
//! compensated.
//!
//! # Edge cases
//!
//! Rows with a weight that is not positive are dropped before anything else, so
//! `n_rows` counts the rows fitted. The fit is degenerate, with no estimate,
//! zero iterations and `converged` false, when no row is left, there is no
//! event, the covariate takes one value among the rows kept, `exp(beta x)`
//! overflows because the covariate separates the events, or the information is
//! not positive. The estimate and both standard errors are then `None`. Without
//! delayed entry, pass negative infinity as every entry time.
//! Times are compared with `total_cmp`, so a NaN time sorts after every number.
//!
//! # References
//!
//! 1. Breslow N. Covariance analysis of censored survival data. Biometrics
//!    1974;30(1):89-99. doi:10.2307/2529620
//! 2. Efron B. The efficiency of Cox's likelihood function for censored data.
//!    J Am Stat Assoc 1977;72(359):557-65. doi:10.1080/01621459.1977.10480613
//! 3. Lin DY, Wei LJ. The robust inference for the Cox proportional hazards
//!    model. J Am Stat Assoc 1989;84(408):1074-8.
//!    doi:10.1080/01621459.1989.10478874
//! 4. Therneau TM, Grambsch PM. Modeling survival data: extending the Cox
//!    model. New York: Springer; 2000.

// Rows, levels and strata are counted in `u32`; the entry point refuses more
// rows than that.
#![expect(
    clippy::cast_possible_truncation,
    reason = "the entry point checks that the row count fits in u32"
)]

use std::cmp::Ordering;

use epistat_stats::sum::compensated;

use crate::check::{check_lengths, check_row_count};
use crate::error::Result;

pub mod design;
pub use design::{ChiSquareTest, CoxModelFit, CoxOptions, CoxTests, fit, score_residuals};
#[expect(
    deprecated,
    reason = "the deprecated names stay exported for a release"
)]
pub use design::{CoxDesign, cox_score_residuals, fit_cox};

/// Largest number of Newton-Raphson steps.
const MAX_ITERATIONS: u32 = 50;
/// Convergence threshold on the absolute Newton step in beta.
const TOLERANCE: f64 = 1e-10;

/// How events that share a time are placed in the partial likelihood.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ties {
    /// Every event at a time has the whole risk set as its denominator.
    Breslow,
    /// Efron's approximation, with the weights of tied events averaged as in
    /// `survival::coxph`: the `k`-th of `d` tied events has the risk set less
    /// `k / d` of the weight of the tied events.
    Efron,
}

/// One Cox fit. See [`fit_cox_single_covariate`] for when it is degenerate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoxFit {
    /// The estimate of `beta`, `None` when the fit is degenerate.
    pub log_hazard_ratio: Option<f64>,
    /// The model-based standard error, `1 / sqrt(I)`, `None` when the fit is
    /// degenerate.
    pub standard_error: Option<f64>,
    /// The summed weight of the events among the rows fitted.
    pub n_events: f64,
    /// The rows fitted, after dropping rows whose weight is not positive.
    pub n_rows: usize,
    /// The Newton steps taken.
    pub iterations: u32,
    /// Whether the last step was below the tolerance.
    pub converged: bool,
    /// Sandwich standard error with the scores summed within cluster. `None`
    /// when no cluster was given or the fit is degenerate.
    pub robust_standard_error: Option<f64>,
}

impl CoxFit {
    fn degenerate(n_events: f64, n_rows: usize) -> Self {
        Self {
            log_hazard_ratio: None,
            standard_error: None,
            n_events,
            n_rows,
            iterations: 0,
            converged: false,
            robust_standard_error: None,
        }
    }
}

/// The rows of a Cox fit on one covariate, one entry per row, all the same
/// length.
#[derive(Debug, Clone, Copy)]
pub struct CoxData<'a> {
    /// The covariate.
    pub x: &'a [f64],
    /// Whether the row ends in the event of interest.
    pub is_event: &'a [bool],
    /// Entry time. Minus infinity when there is no delayed entry.
    pub start: &'a [f64],
    /// Exit time.
    pub stop: &'a [f64],
    /// Case weight. Rows with a weight that is not positive are dropped.
    pub weight: &'a [f64],
    /// Stratum codes. Strata are summed in code order.
    pub stratum: &'a [u32],
    /// How tied events enter the partial likelihood.
    pub ties: Ties,
    /// Cluster codes for the robust variance, or `None` for the model-based
    /// variance alone.
    pub cluster: Option<&'a [u32]>,
}

/// One stratum's rows in descending exit order, with what the sweep needs.
struct Stratum {
    /// Covariate level of each row, in descending exit order.
    level: Vec<u32>,
    weight: Vec<f64>,
    stop: Vec<f64>,
    /// Entry times in the same order, and whether each row ends in an event.
    start: Vec<f64>,
    is_event: Vec<bool>,
    /// Cluster code of each row, empty without clusters.
    cluster: Vec<u32>,
    /// Positions into the arrays above, in descending entry order.
    by_start: Vec<u32>,
    /// Entry times in the order of `by_start`.
    start_sorted: Vec<f64>,
    /// Distinct event times, descending, with the weight dying at each, the
    /// weight times covariate and the number of rows dying. None depends on
    /// beta.
    event_times: Vec<f64>,
    dying_w: Vec<f64>,
    dying_wx: Vec<f64>,
    dying_n: Vec<u32>,
    /// Weight dying at each event time by covariate level, `levels` entries
    /// per time. Efron's denominators need it, since the dying weight times
    /// `exp(beta * x)` depends on beta.
    dying_by_level: Vec<f64>,
    levels: usize,
}

/// Distinct covariate values, sorted, and each row's index into them.
///
/// `exp(beta * x)` is the costly call in a Newton step and a binary covariate
/// takes two values, so it is evaluated per value rather than per row.
fn covariate_levels(x: &[f64], rows: &[usize]) -> (Vec<f64>, Vec<u32>) {
    let mut distinct: Vec<f64> = rows.iter().map(|&i| x[i]).collect();
    distinct.sort_by(f64::total_cmp);
    distinct.dedup_by(|a, b| a.total_cmp(b) == Ordering::Equal);
    let mut level = vec![0u32; x.len()];
    for &i in rows {
        let position = distinct
            .binary_search_by(|probe| probe.total_cmp(&x[i]))
            .unwrap_or_default();
        level[i] = position as u32;
    }
    (distinct, level)
}

fn build_strata(
    input: &CoxData<'_>,
    rows: &[usize],
    level: &[u32],
    distinct: &[f64],
) -> Vec<Stratum> {
    let n_strata = rows
        .iter()
        .map(|&i| input.stratum[i] as usize + 1)
        .max()
        .unwrap_or(0);
    // Counting sort by stratum, stable in input order.
    let mut counts = vec![0usize; n_strata + 1];
    for &i in rows {
        counts[input.stratum[i] as usize + 1] += 1;
    }
    for s in 0..n_strata {
        counts[s + 1] += counts[s];
    }
    let mut grouped = vec![0usize; rows.len()];
    let mut next = counts.clone();
    for &i in rows {
        let s = input.stratum[i] as usize;
        grouped[next[s]] = i;
        next[s] += 1;
    }

    let levels = distinct.len();
    let mut strata = Vec::new();
    for s in 0..n_strata {
        let members = &mut grouped[counts[s]..counts[s + 1]];
        if !members.iter().any(|&i| input.is_event[i]) {
            // No events: the stratum adds nothing to the score or the
            // information, because its baseline hazard is unidentified.
            continue;
        }
        // Stable, so rows with the same exit keep their input order.
        members.sort_by(|&a, &b| input.stop[b].total_cmp(&input.stop[a]));

        let mut by_start: Vec<u32> = (0..members.len() as u32).collect();
        by_start.sort_by(|&a, &b| {
            input.start[members[b as usize]].total_cmp(&input.start[members[a as usize]])
        });
        let start_sorted = by_start
            .iter()
            .map(|&p| input.start[members[p as usize]])
            .collect();

        let mut event_times = Vec::new();
        let mut dying_w = Vec::new();
        let mut dying_wx = Vec::new();
        let mut dying_n = Vec::new();
        let mut dying_by_level: Vec<f64> = Vec::new();
        for &i in members.iter() {
            if !input.is_event[i] {
                continue;
            }
            let w = input.weight[i];
            let wx = w * distinct[level[i] as usize];
            if event_times.last() != Some(&input.stop[i]) {
                event_times.push(input.stop[i]);
                dying_w.push(0.0);
                dying_wx.push(0.0);
                dying_n.push(0u32);
                dying_by_level.extend(std::iter::repeat_n(0.0, levels));
            }
            let last = event_times.len() - 1;
            dying_w[last] += w;
            dying_wx[last] += wx;
            dying_n[last] += 1;
            dying_by_level[last * levels + level[i] as usize] += w;
        }

        strata.push(Stratum {
            level: members.iter().map(|&i| level[i]).collect(),
            weight: members.iter().map(|&i| input.weight[i]).collect(),
            stop: members.iter().map(|&i| input.stop[i]).collect(),
            start: members.iter().map(|&i| input.start[i]).collect(),
            is_event: members.iter().map(|&i| input.is_event[i]).collect(),
            cluster: input
                .cluster
                .map(|cluster| members.iter().map(|&i| cluster[i]).collect())
                .unwrap_or_default(),
            by_start,
            start_sorted,
            event_times,
            dying_w,
            dying_wx,
            dying_n,
            dying_by_level,
            levels,
        });
    }
    strata
}

/// The risk-set sums at one event time, and the sums over the rows that have
/// their event there, both at the current beta.
struct EventTime {
    /// `sum w exp(beta x) x^r` over the risk set, for `r = 0, 1, 2`.
    s0: f64,
    s1: f64,
    s2: f64,
    /// The same sums over the tied events.
    e0: f64,
    e1: f64,
    e2: f64,
}

impl Stratum {
    /// Visit every event time with a positive risk-set weight, in descending
    /// order, with the sums at the current beta.
    ///
    /// At event time `t` the risk set is every row that has joined
    /// (`stop >= t`) and not yet left (`start < t`).
    fn sweep(
        &self,
        exponential: &[f64],
        distinct: &[f64],
        active: &mut Vec<bool>,
        mut visit: impl FnMut(usize, &EventTime),
    ) {
        let size = self.level.len();
        active.clear();
        active.resize(size, false);
        let (mut s0, mut s1, mut s2) = (0.0f64, 0.0f64, 0.0f64);
        let (mut entering, mut leaving) = (0usize, 0usize);

        for (position, &time) in self.event_times.iter().enumerate() {
            while entering < size && self.stop[entering] >= time {
                let x = distinct[self.level[entering] as usize];
                let contribution =
                    self.weight[entering] * exponential[self.level[entering] as usize];
                s0 += contribution;
                s1 += contribution * x;
                s2 += contribution * x * x;
                active[entering] = true;
                entering += 1;
            }
            while leaving < size && self.start_sorted[leaving] >= time {
                let p = self.by_start[leaving] as usize;
                if active[p] {
                    let x = distinct[self.level[p] as usize];
                    let contribution = self.weight[p] * exponential[self.level[p] as usize];
                    s0 -= contribution;
                    s1 -= contribution * x;
                    s2 -= contribution * x * x;
                    active[p] = false;
                }
                leaving += 1;
            }
            if s0 <= 0.0 {
                continue;
            }
            let (mut e0, mut e1, mut e2) = (0.0f64, 0.0f64, 0.0f64);
            for (l, &x) in distinct.iter().enumerate() {
                let dying = self.dying_by_level[position * self.levels + l] * exponential[l];
                e0 += dying;
                e1 += dying * x;
                e2 += dying * x * x;
            }
            visit(
                position,
                &EventTime {
                    s0,
                    s1,
                    s2,
                    e0,
                    e1,
                    e2,
                },
            );
        }
    }

    /// The terms of the log partial likelihood at event time `position`, as
    /// `(weight, fraction)`: the term has denominator `S - fraction * E` and
    /// is multiplied by `weight`. Breslow has one term with the whole dying
    /// weight and fraction 0. Efron has one per tied event, each with the mean
    /// weight and the fraction `k / d`, for `k = 0, ..., d - 1`.
    fn terms(&self, position: usize, ties: Ties) -> impl Iterator<Item = (f64, f64)> {
        let (count, weight) = match ties {
            Ties::Breslow => (1, self.dying_w[position]),
            Ties::Efron => (
                self.dying_n[position],
                self.dying_w[position] / f64::from(self.dying_n[position]),
            ),
        };
        let spread = match ties {
            Ties::Breslow => 0.0,
            Ties::Efron => 1.0 / f64::from(count),
        };
        (0..count).map(move |k| (weight, f64::from(k) * spread))
    }

    /// This stratum's contribution to the score and the information.
    fn accumulate(
        &self,
        exponential: &[f64],
        distinct: &[f64],
        active: &mut Vec<bool>,
        ties: Ties,
    ) -> (f64, f64) {
        let (mut score, mut information) = (0.0f64, 0.0f64);
        self.sweep(exponential, distinct, active, |position, at| {
            let (mut taken, mut curvature) = (0.0f64, 0.0f64);
            for (weight, fraction) in self.terms(position, ties) {
                let d0 = at.s0 - fraction * at.e0;
                let d1 = at.s1 - fraction * at.e1;
                let d2 = at.s2 - fraction * at.e2;
                let ratio = d1 / d0;
                taken += weight * ratio;
                curvature += weight * ((d2 / d0) - ratio * ratio);
            }
            score += self.dying_wx[position] - taken;
            information += curvature;
        });
        (score, information)
    }

    /// Add each row's score residual to its cluster's total in `totals`.
    ///
    /// The residual of row `j` is the derivative of the log partial
    /// likelihood with respect to beta through the weight of `j`. A row that
    /// has its event contributes `w_j (x_j - m)` with `m` the mean of the
    /// weighted covariate averages `d1 / d0` over the tied events, and every
    /// row contributes `-w_j exp(beta x_j) sum_t (x_j hazard_t - centre_t)`
    /// over the event times `t` it is at risk at. Under Efron's approximation a tied
    /// event is at risk in only part of the terms of its own time, which the
    /// last correction adds back. Residuals sum to the score.
    fn add_residuals(
        &self,
        exponential: &[f64],
        distinct: &[f64],
        active: &mut Vec<bool>,
        ties: Ties,
        totals: &mut [f64],
    ) {
        let count = self.event_times.len();
        // Per event time: `hazard` is the sum of weight / d0 over the terms,
        // `centre` the sum of weight * d1 / d0^2, `late_hazard` and
        // `late_centre` the same two sums with the fraction as a factor, and
        // `mean` the mean of d1 / d0.
        let mut hazard = vec![0.0f64; count];
        let mut centre = vec![0.0f64; count];
        let mut late_hazard = vec![0.0f64; count];
        let mut late_centre = vec![0.0f64; count];
        let mut mean = vec![0.0f64; count];
        let mut valid = vec![false; count];
        self.sweep(exponential, distinct, active, |position, at| {
            valid[position] = true;
            let mut terms = 0.0f64;
            for (weight, fraction) in self.terms(position, ties) {
                let d0 = at.s0 - fraction * at.e0;
                let d1 = at.s1 - fraction * at.e1;
                hazard[position] += weight / d0;
                centre[position] += weight * d1 / (d0 * d0);
                late_hazard[position] += weight * fraction / d0;
                late_centre[position] += weight * fraction * d1 / (d0 * d0);
                mean[position] += d1 / d0;
                terms += 1.0;
            }
            mean[position] /= terms;
        });
        // Prefix sums over the descending event times.
        let prefix = |values: &[f64]| -> Vec<f64> {
            let mut sums = Vec::with_capacity(count + 1);
            sums.push(0.0);
            for &value in values {
                sums.push(sums[sums.len() - 1] + value);
            }
            sums
        };
        let (hazard_sum, mean_sum) = (prefix(&hazard), prefix(&centre));

        for j in 0..self.level.len() {
            let x = distinct[self.level[j] as usize];
            let risk = self.weight[j] * exponential[self.level[j] as usize];
            // Event times t with start < t <= stop.
            let first = self.event_times.partition_point(|&t| t > self.stop[j]);
            let last = self.event_times.partition_point(|&t| t > self.start[j]);
            let mut residual = -risk
                * (x * (hazard_sum[last] - hazard_sum[first]) - (mean_sum[last] - mean_sum[first]));
            if self.is_event[j] {
                // The row has its event at `stop`, an event time of the stratum.
                let at = first;
                if valid[at] {
                    residual += self.weight[j] * (x - mean[at]);
                    residual += risk * (x * late_hazard[at] - late_centre[at]);
                }
            }
            totals[self.cluster[j] as usize] += residual;
        }
    }
}

/// Newton-Raphson on the weighted partial likelihood of one covariate, Breslow
/// or Efron.
///
/// See the module documentation for the method, the robust variance and when
/// the fit is degenerate. [`fit_cox`] takes a design matrix.
///
/// # Errors
///
/// [`Error::Input`](crate::Error::Input) when the slices differ in length.
pub fn fit_cox_single_covariate(data: &CoxData<'_>) -> Result<CoxFit> {
    check_lengths(&[
        ("x", data.x.len()),
        ("is_event", data.is_event.len()),
        ("start", data.start.len()),
        ("stop", data.stop.len()),
        ("weight", data.weight.len()),
        ("stratum", data.stratum.len()),
        ("cluster", data.cluster.map_or(data.x.len(), <[u32]>::len)),
    ])?;
    check_row_count(data.x.len())?;
    Ok(fit_unchecked(data))
}

#[must_use]
fn fit_unchecked(input: &CoxData<'_>) -> CoxFit {
    let rows: Vec<usize> = (0..input.x.len())
        .filter(|&i| input.weight[i] > 0.0)
        .collect();
    if rows.is_empty() {
        return CoxFit::degenerate(0.0, 0);
    }
    let n_rows = rows.len();
    let total_events = compensated(
        rows.iter()
            .filter(|&&i| input.is_event[i])
            .map(|&i| input.weight[i]),
    );
    let any_event = rows.iter().any(|&i| input.is_event[i]);
    let (distinct, level) = covariate_levels(input.x, &rows);
    if !any_event || distinct.len() < 2 {
        return CoxFit::degenerate(total_events, n_rows);
    }

    let strata = build_strata(input, &rows, &level, &distinct);
    let mut active = Vec::new();
    let mut exponential = vec![0.0f64; distinct.len()];
    let mut beta = 0.0f64;
    let mut iterations = 0u32;
    let mut converged = false;
    let mut information = f64::NAN;

    for step_count in 1..=MAX_ITERATIONS {
        iterations = step_count;
        for (slot, &x) in exponential.iter_mut().zip(&distinct) {
            *slot = (beta * x).exp();
        }
        if exponential.contains(&f64::INFINITY) {
            // The likelihood is monotone: the covariate separates the events
            // and there is no finite maximum.
            return CoxFit::degenerate(total_events, n_rows);
        }
        let mut score = 0.0f64;
        information = 0.0;
        for stratum in &strata {
            let (stratum_score, stratum_information) =
                stratum.accumulate(&exponential, &distinct, &mut active, input.ties);
            score += stratum_score;
            information += stratum_information;
        }
        if information <= 0.0 || information.is_nan() {
            return CoxFit::degenerate(total_events, n_rows);
        }
        let step = score / information;
        beta += step;
        if step.abs() < TOLERANCE {
            converged = true;
            break;
        }
    }

    let standard_error = (1.0 / information).sqrt();
    let robust_standard_error = input.cluster.map(|cluster| {
        let n_clusters = rows
            .iter()
            .map(|&i| cluster[i] as usize + 1)
            .max()
            .unwrap_or(0);
        let mut totals = vec![0.0f64; n_clusters];
        for (slot, &x) in exponential.iter_mut().zip(&distinct) {
            *slot = (beta * x).exp();
        }
        let mut final_information = 0.0f64;
        for stratum in &strata {
            final_information += stratum
                .accumulate(&exponential, &distinct, &mut active, input.ties)
                .1;
            stratum.add_residuals(
                &exponential,
                &distinct,
                &mut active,
                input.ties,
                &mut totals,
            );
        }
        let squares = compensated(totals.iter().map(|total| total * total));
        squares.sqrt() / final_information
    });
    CoxFit {
        log_hazard_ratio: Some(beta),
        standard_error: Some(standard_error),
        n_events: total_events,
        n_rows,
        iterations,
        converged,
        robust_standard_error,
    }
}

#[cfg(test)]
mod tests {
    use super::{CoxData, CoxFit, Ties, fit_cox_single_covariate};

    fn fit_cox(data: &CoxData<'_>) -> CoxFit {
        fit_cox_single_covariate(data).expect("consistent lengths")
    }

    // Two groups, two causes, with ties within a cause, across causes, and
    // between a censoring time and an event time.
    const TIME: [f64; 20] = [
        2., 3., 3., 5., 6., 8., 9., 9., 11., 12., 1., 4., 4., 5., 7., 7., 10., 10., 13., 14.,
    ];
    const EVENT: [i32; 20] = [1, 2, 1, 0, 1, 2, 1, 0, 2, 1, 2, 1, 1, 0, 2, 1, 1, 0, 2, 1];

    fn group() -> Vec<f64> {
        (0..20).map(|i| if i < 10 { 1.0 } else { 0.0 }).collect()
    }

    fn cause_specific(cause: i32, weight: f64) -> CoxFit {
        let x = group();
        let is_event: Vec<bool> = EVENT.iter().map(|&e| e == cause).collect();
        fit_cox(&CoxData {
            x: &x,
            is_event: &is_event,
            start: &[f64::NEG_INFINITY; 20],
            stop: &TIME,
            weight: &[weight; 20],
            stratum: &[0; 20],
            ties: Ties::Breslow,
            cluster: None,
        })
    }

    fn close(a: f64, b: f64, tolerance: f64) -> bool {
        (a - b).abs() < tolerance
    }

    /// `survival::coxph(Surv(time, event == k) ~ grp, ties = "breslow")`,
    /// R 4.6.1, survival 3.8.6.
    #[test]
    fn cause_specific_fits_match_r() {
        for (cause, coef, se, events) in [
            (1, 0.444_557_921_663_022, 0.677_346_033_252_406, 10.0),
            (2, 0.398_994_394_444_171, 0.914_663_190_364_912, 6.0),
        ] {
            let fit = cause_specific(cause, 1.0);
            assert!(fit.converged);
            assert!(close(fit.log_hazard_ratio.unwrap(), coef, 1e-10), "{fit:?}");
            assert!(close(fit.standard_error.unwrap(), se, 1e-10), "{fit:?}");
            assert!((fit.n_events - events).abs() < f64::EPSILON);
        }
    }

    /// `survival::coxph(Surv(time, ev) ~ grp + strata(fam), ties = "breslow")`
    /// on six sibling pairs, R 4.6.1, survival 3.8.6.
    #[test]
    fn family_stratified_fit_matches_r() {
        let family = [0u32, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5];
        let time = [5., 8., 3., 9., 6., 6., 4., 7., 2., 10., 5., 5.];
        let event = [1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0];
        let x = [1., 0., 1., 0., 1., 0., 1., 0., 1., 1., 0., 0.];
        let is_event: Vec<bool> = event.iter().map(|&e| e == 1).collect();
        let fit = |stratum: &[u32]| {
            fit_cox(&CoxData {
                x: &x,
                is_event: &is_event,
                start: &[f64::NEG_INFINITY; 12],
                stop: &time,
                weight: &[1.0; 12],
                stratum,
                ties: Ties::Breslow,
                cluster: None,
            })
        };
        let stratified = fit(&family);
        assert!(close(
            stratified.log_hazard_ratio.unwrap(),
            1.098_612_288_667_694,
            1e-9
        ));
        assert!(close(
            stratified.standard_error.unwrap(),
            1.154_700_538_379_131,
            1e-9
        ));
        let plain = fit(&[0; 12]);
        assert!(close(
            plain.log_hazard_ratio.unwrap(),
            0.594_821_561_034_560,
            1e-9
        ));
    }

    /// `survival::coxph(Surv(time, event == k) ~ grp, ties = "efron",
    /// cluster = fam)` with `fam` the ten pairs `i %% 10`, R 4.6.1, survival
    /// 3.8.6. Coefficient, model-based and robust standard error. Cause 2 has
    /// no tied events of its own at a shared time, so it equals Breslow.
    #[test]
    fn efron_and_robust_fits_match_r() {
        let x = group();
        let family: Vec<u32> = (0..20).map(|i| i % 10).collect();
        for (cause, coef, se, robust) in [
            (
                1,
                0.431_252_248_794_995,
                0.677_404_606_310_407,
                0.470_347_816_743_448,
            ),
            (
                2,
                0.398_994_394_444_171,
                0.914_663_190_364_912,
                0.850_506_298_405_754,
            ),
        ] {
            let is_event: Vec<bool> = EVENT.iter().map(|&e| e == cause).collect();
            let fit = fit_cox(&CoxData {
                x: &x,
                is_event: &is_event,
                start: &[f64::NEG_INFINITY; 20],
                stop: &TIME,
                weight: &[1.0; 20],
                stratum: &[0; 20],
                ties: Ties::Efron,
                cluster: Some(&family),
            });
            assert!(fit.converged);
            assert!(close(fit.log_hazard_ratio.unwrap(), coef, 1e-10), "{fit:?}");
            assert!(close(fit.standard_error.unwrap(), se, 1e-10), "{fit:?}");
            let got = fit.robust_standard_error.expect("clustered");
            assert!(close(got, robust, 1e-9), "{fit:?}");
        }
    }

    #[test]
    fn doubling_the_weights_moves_the_error_not_the_estimate() {
        let single = cause_specific(1, 1.0);
        let doubled = cause_specific(1, 2.0);
        assert!(close(
            single.log_hazard_ratio.unwrap(),
            doubled.log_hazard_ratio.unwrap(),
            1e-10
        ));
        assert!(doubled.standard_error.unwrap() < single.standard_error.unwrap());
    }

    #[test]
    fn no_events_or_no_contrast_gives_no_estimate() {
        let none = fit_cox(&CoxData {
            x: &[1.0, 0.0],
            is_event: &[false, false],
            start: &[f64::NEG_INFINITY; 2],
            stop: &[1.0, 2.0],
            weight: &[1.0; 2],
            stratum: &[0; 2],
            ties: Ties::Breslow,
            cluster: None,
        });
        assert!(none.log_hazard_ratio.is_none() && !none.converged && none.iterations == 0);
        let constant = fit_cox(&CoxData {
            x: &[1.0, 1.0, 1.0],
            is_event: &[true, true, false],
            start: &[f64::NEG_INFINITY; 3],
            stop: &[1.0, 2.0, 3.0],
            weight: &[1.0; 3],
            stratum: &[0; 3],
            ties: Ties::Breslow,
            cluster: None,
        });
        assert!(constant.log_hazard_ratio.is_none() && !constant.converged);
        assert!((constant.n_events - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn a_covariate_that_separates_the_events_gives_no_estimate() {
        // Every exposed row has the event before any unexposed row: the
        // likelihood increases without bound.
        let fit = fit_cox(&CoxData {
            x: &[1.0, 1.0, 0.0, 0.0],
            is_event: &[true, true, false, false],
            start: &[f64::NEG_INFINITY; 4],
            stop: &[1.0, 2.0, 3.0, 4.0],
            weight: &[1.0; 4],
            stratum: &[0; 4],
            ties: Ties::Breslow,
            cluster: None,
        });
        assert!(fit.log_hazard_ratio.is_none() || !fit.converged);
    }

    #[test]
    fn slices_of_different_lengths_are_refused() {
        let error = fit_cox_single_covariate(&CoxData {
            x: &[1.0, 0.0],
            is_event: &[true],
            start: &[f64::NEG_INFINITY; 2],
            stop: &[1.0, 2.0],
            weight: &[1.0; 2],
            stratum: &[0; 2],
            ties: Ties::Breslow,
            cluster: None,
        })
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "columns differ in length: is_event has 1 values against 2 in x"
        );
    }
}
