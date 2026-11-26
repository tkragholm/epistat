//! The self-controlled case series for exposures that the event censors,
//! delays or stops, by the pseudo-likelihood of Farrington, Whitaker and
//! Hocine \[1\], as `SCCS::eventdepenexp`.
//!
//! # The problem
//!
//! The standard case series assumes that the event does not change later
//! exposure. When a vaccine is not given after the event, as rotavirus vaccine
//! after intussusception, the doses that would have come after the event are
//! missing, and the time after the event looks unexposed for a reason that
//! has to do with the event. A pre-exposure window does not repair it.
//!
//! # Algorithm
//!
//! Only the first event of each case is used, and doses after it are
//! dropped. Each case is cut into pieces at the age cuts and at the dose
//! dates and window boundaries; a dose opens contiguous risk windows, after a
//! gap from the dose date when the first window does not start at it, and
//! where the periods of two doses overlap the later dose takes the time. The
//! data are then stacked \[1\]:
//!
//! - level 0 holds the whole observation of a case whose observation starts
//!   outside every dose period, with no piece exposed;
//! - level `k` holds the observation from the start of dose `k` to the end,
//!   with only the risk windows of dose `k` exposed.
//!
//! Each level of a case is a stratum of the conditional Poisson likelihood of
//! [`super::fit`]. An event that falls in a risk window of a dose later than
//! the level, which that level counts as unexposed, enters with the weight
//! `exp(-beta_j)` of its window `j` instead of 1. The weights depend on
//! `beta`, so the fit is repeated with the weights of the last estimate until
//! `|beta_new - beta_old|` is at most `tolerance` times the number of exposure
//! columns.
//!
//! The variance is the sandwich of the estimating equations \[1\]: with
//! `r_l` the residual of piece `l`, its weighted event minus `n_s p_l` in its
//! stratum `s`, and `x_l` its design row,
//!
//! ```text
//! U_i     = sum_(l in case i) r_l x_l                         (all levels)
//! A[j, m] = - sum_(l: x_lj = 1, event of the stratum in window m of a later dose) r_l
//! B       = A - I,   I the information of the fit
//! V       = B^-1 (sum_i U_i U_i') B^-T
//! ```
//!
//! where `A` is the derivative of the estimating equations through the
//! weights, zero in the columns of the age effects.
//!
//! The windows of every dose share their coefficients when
//! `same_dose_parameters` is set; otherwise dose `k` has its own, and column
//! `(k - 1) w + j` is window `j` of dose `k` for `w` windows. The age bands
//! after the first follow.
//!
//! # Agreement with R
//!
//! The tests compare the coefficients and the sandwich with
//! `SCCS::eventdepenexp` (SCCS 1.8) on the rotavirus and intussusception data
//! of the package, `rotdat`, with one and two risk windows and with shared and
//! separate dose parameters. `eventdepenexp` fits each step with
//! `gnm::gnm(eliminate = )`, whose own convergence is looser, so the two agree
//! to that tolerance and not to the last bits. Times map as for
//! `standardsccs`: `end` is `aend + 1`, and the windows of `expogrp = c(1, 8)`
//! with `aedrug = adrug + 21` are `[1, 8)` and `[8, 22)` after the dose.
//!
//! # References
//!
//! 1. Farrington CP, Whitaker HJ, Hocine MN. Case series analysis for
//!    censored, perturbed, or curtailed post-event exposures. Biostatistics
//!    2009;10(1):3-16.

use epistat_linalg::{Attempt, Mat, Method, inverse};

use super::{SccsData, SccsOptions, WindowSpec, fit};
use crate::error::{Error, Result};

/// The cases, their first events and doses for [`fit_event_dependent`].
///
/// Doses are ragged: case `i` has `doses[dose_offsets[i]..dose_offsets[i +
/// 1]]`, in increasing order.
#[derive(Debug, Clone, Copy)]
pub struct EventDependentSpec<'a> {
    /// The start of observation of each case.
    pub start: &'a [f64],
    /// The end of observation of each case, exclusive.
    pub end: &'a [f64],
    /// The time of the first event of each case, in `[start, end)`.
    pub event: &'a [f64],
    /// The offsets of each case's doses, one more entry than cases.
    pub dose_offsets: &'a [usize],
    /// The dose dates, increasing within a case.
    pub doses: &'a [f64],
    /// The risk windows of a dose, contiguous, the first starting at or after
    /// the dose date and the last ending at a finite time.
    pub windows: &'a [WindowSpec],
    /// The strictly increasing age cut points.
    pub age_cuts: &'a [f64],
    /// Whether all doses share the coefficients of the windows.
    pub same_dose_parameters: bool,
}

/// What [`fit_event_dependent`] takes besides the data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EventDependentOptions {
    /// The outer iteration stops when the Euclidean change of the exposure
    /// coefficients is at most this times their number, as `tolerance` of
    /// `eventdepenexp`.
    pub tolerance: f64,
    /// The most outer iterations.
    pub max_iterations: usize,
    /// The options of each fit.
    pub fit: SccsOptions,
}

impl Default for EventDependentOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-8,
            max_iterations: 100,
            fit: SccsOptions {
                tolerance: 1e-12,
                ..SccsOptions::default()
            },
        }
    }
}

/// A fitted event-dependent exposure model.
#[derive(Debug, Clone)]
pub struct EventDependentFit {
    /// The coefficients: the exposure columns, then the age bands after the
    /// first. `None` for all when the last fit has no estimate or a column is
    /// aliased.
    pub coefficients: Vec<Option<f64>>,
    /// The sandwich covariance, `p * p` row-major, `None` without estimate.
    pub covariance: Vec<Option<f64>>,
    /// The square roots of its diagonal.
    pub standard_errors: Vec<Option<f64>>,
    /// The number of exposure columns, before the age columns.
    pub n_exposure_columns: usize,
    /// The cases.
    pub n_cases: usize,
    /// The doses dropped because they come after the case's event.
    pub n_doses_after_event: usize,
    /// The strata at each level of the stack, level 0 first.
    pub n_strata_by_level: Vec<usize>,
    /// The outer iterations.
    pub iterations: usize,
    /// Whether the outer iteration met its tolerance within the limit.
    pub converged: bool,
}

/// One piece of a case.
struct Piece {
    length: f64,
    event: bool,
    age: u32,
    /// The dose, from 1, 0 outside every dose period.
    dose: u32,
    /// The window, from 1, 0 in the gap before the first window or outside.
    window: u32,
}

/// One row of the stack.
struct Row {
    stratum: u32,
    length: f64,
    event: bool,
    age: u32,
    /// The exposure column, from 1, 0 unexposed.
    expo: u32,
    /// The exposure column whose `exp(-beta)` weights this row's event, 0
    /// when it is not weighted.
    weight: u32,
    /// The weight column of the event of the row's stratum, 0 when none.
    stratum_weight: u32,
    case: u32,
}

fn invalid(message: impl Into<String>) -> Error {
    Error::input(message)
}

fn check(spec: &EventDependentSpec<'_>) -> Result<usize> {
    let n = spec.start.len();
    if spec.end.len() != n || spec.event.len() != n || spec.dose_offsets.len() != n + 1 {
        return Err(invalid(
            "start, end and event need one entry per case and dose_offsets one more",
        ));
    }
    if u32::try_from(n).is_err() {
        return Err(invalid("more cases than a u32 addresses"));
    }
    let o = spec.dose_offsets;
    if o[0] != 0 || o[n] != spec.doses.len() || o.windows(2).any(|w| w[0] > w[1]) {
        return Err(invalid(
            "the dose offsets must start at 0, not decrease and end at the number of doses",
        ));
    }
    for i in 0..n {
        let (a, b, t) = (spec.start[i], spec.end[i], spec.event[i]);
        if !(a.is_finite() && b.is_finite() && a < b) {
            return Err(invalid(format!(
                "case {i} has the observation period [{a}, {b}), which is not a finite, non-empty interval"
            )));
        }
        if !(t >= a && t < b) {
            return Err(invalid(format!(
                "case {i} has its event at {t}, outside the observation period [{a}, {b})"
            )));
        }
        let d = &spec.doses[o[i]..o[i + 1]];
        if d.iter().any(|x| !x.is_finite()) || d.windows(2).any(|w| w[0] > w[1]) {
            return Err(invalid(format!(
                "the doses of case {i} are not finite and increasing"
            )));
        }
    }
    let w = spec.windows;
    if w.is_empty() {
        return Err(invalid("a dose needs at least one risk window"));
    }
    if !(w[0].from >= 0.0 && w[0].from.is_finite())
        || w.iter().any(|x| !(x.from < x.to && x.to.is_finite()))
        || w.windows(2).any(|p| p[0].to != p[1].from)
    {
        return Err(invalid(
            "the risk windows must be finite, contiguous and start at or after the dose date; a window before the dose is not allowed",
        ));
    }
    if !spec.age_cuts.iter().all(|c| c.is_finite())
        || spec.age_cuts.windows(2).any(|p| p[0] >= p[1])
    {
        return Err(invalid(
            "the age cut points must be finite and strictly increasing",
        ));
    }
    Ok(n)
}

/// The pieces of case `i` with the doses it keeps, and the number dropped.
#[expect(
    clippy::cast_possible_truncation,
    reason = "doses, windows and age bands are few"
)]
fn cut(spec: &EventDependentSpec<'_>, i: usize) -> (Vec<Piece>, usize) {
    let (start, end, event) = (spec.start[i], spec.end[i], spec.event[i]);
    let all = &spec.doses[spec.dose_offsets[i]..spec.dose_offsets[i + 1]];
    let doses: Vec<f64> = all.iter().copied().filter(|&d| d <= event).collect();
    let dropped = all.len() - doses.len();
    let last = spec.windows[spec.windows.len() - 1].to;
    let mut cuts = vec![start, end];
    let mut push = |c: f64| {
        if c > start && c < end {
            cuts.push(c);
        }
    };
    for &c in spec.age_cuts {
        push(c);
    }
    for &d in &doses {
        push(d);
        for w in spec.windows {
            push(d + w.from);
            push(d + w.to);
        }
    }
    cuts.sort_by(f64::total_cmp);
    cuts.dedup();
    let mut out = Vec::new();
    for pair in cuts.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let age = spec.age_cuts.partition_point(|&c| c <= a) as u32;
        let (mut dose, mut window) = (0, 0);
        for (k, &d) in doses.iter().enumerate() {
            if d <= a && a < d + last {
                dose = k as u32 + 1;
                window = spec
                    .windows
                    .iter()
                    .position(|w| d + w.from <= a && a < d + w.to)
                    .map_or(0, |j| j as u32 + 1);
            }
        }
        out.push(Piece {
            length: b - a,
            event: event >= a && event < b,
            age,
            dose,
            window,
        });
    }
    (out, dropped)
}

/// Fits the self-controlled case series for exposures that the event
/// censors or stops. See the module documentation for the stacked
/// pseudo-likelihood and its sandwich variance.
///
/// # Errors
///
/// [`Error::Input`] when the columns and offsets disagree in length, an
/// observation period is empty, an event lies outside it, the doses of a case
/// are not increasing, the windows are not contiguous, start before the dose
/// or end at an infinite time, the age cuts are not strictly increasing, the
/// tolerance is not positive, or a fit fails.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "strata and columns are far below u32 and 2^53"
)]
pub fn fit_event_dependent(
    spec: &EventDependentSpec<'_>,
    opts: &EventDependentOptions,
) -> Result<EventDependentFit> {
    let n = check(spec)?;
    if !(opts.tolerance > 0.0 && opts.tolerance.is_finite()) || opts.max_iterations == 0 {
        return Err(invalid(
            "the tolerance must be positive and finite and max_iterations at least 1",
        ));
    }
    let n_windows = spec.windows.len() as u32;
    let mut cases = Vec::with_capacity(n);
    let mut n_dropped = 0;
    for i in 0..n {
        let (pieces, dropped) = cut(spec, i);
        n_dropped += dropped;
        cases.push(pieces);
    }
    let max_dose = cases
        .iter()
        .flat_map(|c| c.iter().map(|p| p.dose))
        .max()
        .unwrap_or(0);
    let column = |dose: u32, window: u32| -> u32 {
        if spec.same_dose_parameters {
            window
        } else {
            (dose - 1) * n_windows + window
        }
    };
    let n_expo = if spec.same_dose_parameters {
        n_windows
    } else {
        max_dose * n_windows
    } as usize;
    let mut rows: Vec<Row> = Vec::new();
    let mut n_strata_by_level = vec![0_usize; max_dose as usize + 1];
    let mut stratum = 0_u32;
    for (i, pieces) in cases.iter().enumerate() {
        let event_piece = pieces.iter().find(|p| p.event);
        for level in 0..=max_dose {
            let from = if level == 0 {
                if pieces[0].dose != 0 {
                    continue;
                }
                0
            } else {
                match pieces.iter().position(|p| p.dose == level) {
                    Some(k) => k,
                    None => continue,
                }
            };
            n_strata_by_level[level as usize] += 1;
            let weight_of = |p: &Piece| {
                if p.event && p.dose > level && p.window > 0 {
                    column(p.dose, p.window)
                } else {
                    0
                }
            };
            let stratum_weight = event_piece
                .filter(|_| pieces[from..].iter().any(|p| p.event))
                .map_or(0, weight_of);
            for p in &pieces[from..] {
                rows.push(Row {
                    stratum,
                    length: p.length,
                    event: p.event,
                    age: p.age,
                    expo: if level > 0 && p.dose == level && p.window > 0 {
                        column(p.dose, p.window)
                    } else {
                        0
                    },
                    weight: weight_of(p),
                    stratum_weight,
                    case: i as u32,
                });
            }
            stratum += 1;
        }
    }
    let n_age = spec.age_cuts.len();
    let p = n_expo + n_age;
    let x = Mat::from_fn(rows.len(), p, |r, j| {
        let on = if j < n_expo {
            rows[r].expo as usize == j + 1
        } else {
            rows[r].age as usize == j - n_expo + 1
        };
        f64::from(u8::from(on))
    });
    let person: Vec<u32> = rows.iter().map(|r| r.stratum).collect();
    let length: Vec<f64> = rows.iter().map(|r| r.length).collect();
    let weighted = |beta: &[f64]| -> Vec<f64> {
        rows.iter()
            .map(|r| {
                if r.weight > 0 {
                    (-beta[r.weight as usize - 1]).exp()
                } else {
                    f64::from(u8::from(r.event))
                }
            })
            .collect()
    };
    let mut result = EventDependentFit {
        coefficients: vec![None; p],
        covariance: vec![None; p * p],
        standard_errors: vec![None; p],
        n_exposure_columns: n_expo,
        n_cases: n,
        n_doses_after_event: n_dropped,
        n_strata_by_level,
        iterations: 0,
        converged: false,
    };
    let mut beta = vec![0.0; n_expo];
    let mut events = weighted(&beta);
    let mut estimate: Vec<f64>;
    loop {
        let f = fit(
            &SccsData {
                person: &person,
                length: &length,
                events: &events,
            },
            x.as_ref(),
            &opts.fit,
        )?;
        result.iterations += 1;
        if f.coefficients.iter().any(Option::is_none) {
            return Ok(result);
        }
        estimate = f.coefficients.iter().map(|c| c.unwrap_or(0.0)).collect();
        let change: f64 = estimate[..n_expo]
            .iter()
            .zip(&beta)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        beta.copy_from_slice(&estimate[..n_expo]);
        if change <= opts.tolerance * n_expo as f64 {
            result.converged = true;
            break;
        }
        if result.iterations == opts.max_iterations {
            break;
        }
        events = weighted(&beta);
    }
    // The residuals, scores and derivatives at the last fit, whose weights
    // are those of the estimate before it.
    let mut info = vec![0.0; p * p];
    let mut scores = vec![0.0; n * p];
    let mut a = vec![0.0; p * p];
    let mut k = 0;
    while k < rows.len() {
        let s = rows[k].stratum;
        let mut e = k;
        while e < rows.len() && rows[e].stratum == s {
            e += 1;
        }
        let total: f64 = events[k..e].iter().sum();
        if total > 0.0 && e - k > 1 {
            let eta: Vec<f64> = (k..e)
                .map(|r| length[r].ln() + (0..p).map(|j| x[(r, j)] * estimate[j]).sum::<f64>())
                .collect();
            let top = eta.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let w: Vec<f64> = eta.iter().map(|v| (v - top).exp()).collect();
            let sum: f64 = w.iter().sum();
            let mut xbar = vec![0.0; p];
            for (r, wr) in (k..e).zip(&w) {
                for (j, xb) in xbar.iter_mut().enumerate() {
                    *xb += wr / sum * x[(r, j)];
                }
            }
            for (r, wr) in (k..e).zip(&w) {
                let pr = wr / sum;
                let residual = events[r] - total * pr;
                let case = rows[r].case as usize;
                let m = rows[r].stratum_weight as usize;
                for j in 0..p {
                    let cj = x[(r, j)] - xbar[j];
                    scores[case * p + j] += residual * x[(r, j)];
                    for l in 0..p {
                        info[j * p + l] += total * pr * cj * (x[(r, l)] - xbar[l]);
                    }
                    if m > 0 && x[(r, j)] != 0.0 {
                        a[j * p + m - 1] -= residual;
                    }
                }
            }
        }
        k = e;
    }
    let bread = Mat::from_fn(p, p, |j, l| a[j * p + l] - info[j * p + l]);
    let inv = inverse(bread.as_ref(), &[Attempt::plain(Method::PartialPivLu)])
        .map_err(|e| invalid(format!("the derivative of the estimating equations: {e}")))?
        .x;
    let mut meat = vec![0.0; p * p];
    for i in 0..n {
        for j in 0..p {
            for l in 0..p {
                meat[j * p + l] += scores[i * p + j] * scores[i * p + l];
            }
        }
    }
    for j in 0..p {
        for l in 0..p {
            let mut v = 0.0;
            for u in 0..p {
                let mut t = 0.0;
                for q in 0..p {
                    t += meat[u * p + q] * inv[(l, q)];
                }
                v += inv[(j, u)] * t;
            }
            result.covariance[j * p + l] = Some(v);
        }
        result.coefficients[j] = Some(estimate[j]);
    }
    result.standard_errors = (0..p)
        .map(|j| result.covariance[j * p + j].map(f64::sqrt))
        .collect();
    Ok(result)
}
