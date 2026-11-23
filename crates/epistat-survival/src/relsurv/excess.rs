//! The piecewise-constant excess hazard model on split follow-up.

#![expect(
    clippy::float_cmp,
    reason = "a weight of 0 or 1 and the end of a follow-up are compared with the values they were copied from"
)]

use super::ExpectedHazard;
use crate::error::{Error, Result};
use crate::ldl;
use crate::par;
use crate::surv::SurvData;

const BLOCK: usize = 1024;
const WAVE: usize = 64;
/// An interval whose fitted excess events fall below this share of its
/// observed events is at the boundary, an excess hazard of 0.
const BOUNDARY: f64 = 1e-10;

/// The settings of [`excess_hazard`].
#[derive(Clone, Copy, Debug)]
pub struct ExcessOptions<'a> {
    /// The cut points of the follow-up intervals, finite and strictly
    /// increasing from 0, at least two. Interval `k` is `(cuts[k], cuts[k + 1]]`
    /// and follow-up beyond the last cut is not used.
    pub cuts: &'a [f64],
    /// Whether the records are summed into cells of the same interval and
    /// covariate values before the fit, as Dickman et al. (2004) fit grouped
    /// data. Without it each record, a stretch of follow-up in one interval and
    /// one cell of the rate table, is a cell of its own, as `popEpi::relpois`
    /// fits split data. The two likelihoods differ, since
    /// `ln(sum d* + sum y e^eta)` is not a sum over the records.
    pub aggregate: bool,
    /// Convergence is declared when the decrement
    /// `score' I^-1 score` is at most `tolerance (1 + |log likelihood|)`.
    pub tolerance: f64,
    /// The most iterations.
    pub max_iterations: usize,
}

impl<'a> ExcessOptions<'a> {
    /// The cut points given, records not aggregated, a tolerance of `1e-12`
    /// and at most 100 iterations.
    #[must_use]
    pub fn new(cuts: &'a [f64]) -> Self {
        Self {
            cuts,
            aggregate: false,
            tolerance: 1e-12,
            max_iterations: 100,
        }
    }
}

/// A fitted excess hazard model.
///
/// The parameters are the log excess hazard of each follow-up interval, per
/// unit of the follow-up time, then the coefficients of the columns of `x`.
/// Matrices are square of that dimension in row-major order.
#[derive(Clone, Debug, PartialEq)]
pub struct ExcessFit {
    /// The estimate of each parameter: `None` for an interval in which no one
    /// is at risk, and minus infinity for an interval at the boundary.
    pub coefficients: Vec<Option<f64>>,
    /// Whether the estimate of the parameter is at the boundary of an excess
    /// hazard of 0, where the log likelihood increases without bound as the
    /// log excess hazard goes to minus infinity: an interval without events, or
    /// one whose fitted excess events fell below `1e-10` of its events. The
    /// other parameters are fitted with these fixed at 0 excess hazard.
    pub at_bound: Vec<bool>,
    /// The inverse of the Fisher information at the estimate, `None` in the
    /// rows and columns of an interval without person-time or at the boundary,
    /// and `None` as a whole when the fit did not converge.
    pub covariance: Option<Vec<Option<f64>>>,
    /// The log likelihood `sum d ln(mu) - mu` over the cells, without the
    /// constant of the Poisson mass function.
    pub log_likelihood: f64,
    /// The deviance of the cells, as `glm` gives it for the same cells.
    pub deviance: f64,
    /// The iterations used.
    pub iterations: usize,
    /// Whether the decrement fell below the tolerance.
    pub converged: bool,
    /// The number of cells: records, or with
    /// [`ExcessOptions::aggregate`] distinct pairs of an interval and covariate
    /// values.
    pub n_cells: usize,
    /// The events in each interval.
    pub events: Vec<f64>,
    /// The expected number of events in each interval.
    pub expected_events: Vec<f64>,
    /// The person-time in each interval.
    pub person_time: Vec<f64>,
}

#[derive(Clone, Copy)]
struct Cell {
    interval: usize,
    row: usize,
    y: f64,
    d: f64,
    dstar: f64,
}

enum Units {
    Rows(Vec<usize>),
    Cells(Vec<Cell>),
}

struct Pass {
    ll: f64,
    deviance: f64,
    score: Vec<f64>,
    /// The Fisher information.
    info: Vec<f64>,
    /// The observed information, minus the Hessian of the log likelihood.
    observed: Vec<f64>,
    /// The fitted excess events of each interval parameter.
    excess: Vec<f64>,
}

impl Pass {
    fn zero(q: usize, n_int: usize) -> Self {
        Self {
            ll: 0.0,
            deviance: 0.0,
            score: vec![0.0; q],
            info: vec![0.0; q * q],
            observed: vec![0.0; q * q],
            excess: vec![0.0; n_int],
        }
    }

    fn add(&mut self, b: &Self) {
        self.ll += b.ll;
        self.deviance += b.deviance;
        for (x, y) in self.score.iter_mut().zip(&b.score) {
            *x += y;
        }
        for (x, y) in self.info.iter_mut().zip(&b.info) {
            *x += y;
        }
        for (x, y) in self.observed.iter_mut().zip(&b.observed) {
            *x += y;
        }
        for (x, y) in self.excess.iter_mut().zip(&b.excess) {
            *x += y;
        }
    }
}

struct Setup<'a> {
    start: Option<&'a [f64]>,
    stop: &'a [f64],
    status: &'a [i32],
    expected: &'a ExpectedHazard,
    cuts: &'a [f64],
    x: &'a [&'a [f64]],
    /// The parameter of each interval, `usize::MAX` for one without person-time.
    slot: Vec<usize>,
    n_int: usize,
    q: usize,
}

impl Setup<'_> {
    /// Calls `f(interval, y, d, expected)` for each record of row `i`: its
    /// follow-up cut at the interval cuts and at the cuts of the rate table,
    /// so that the reference hazard is constant over each record.
    fn records(&self, i: usize, mut f: impl FnMut(usize, f64, f64, f64)) {
        let begin = self.start.map_or(0.0, |s| s[i]);
        let end = self.stop[i];
        let dead = self.status[i] != 0;
        let n = self.cuts.len() - 1;
        let mut k = self.cuts[1..].partition_point(|&c| c <= begin);
        while k < n {
            let lo = begin.max(self.cuts[k]);
            let hi = end.min(self.cuts[k + 1]);
            if hi > lo {
                self.expected.pieces_within(i, lo, hi, |a, b, hazard| {
                    let d = f64::from(u8::from(dead && b == end));
                    f(k, b - a, d, (b - a) * hazard);
                });
            }
            if hi >= end {
                break;
            }
            k += 1;
        }
    }

    fn pass(&self, units: &Units, theta: &[f64], derivatives: bool) -> Option<Pass> {
        let (q, n_int) = (self.q, self.n_int);
        let add =
            |t: &mut Pass, z: &mut [f64], row: usize, k: usize, y: f64, d: f64, dstar: f64| {
                let s = self.slot[k];
                let mut eta = theta[s];
                for (j, col) in self.x.iter().enumerate() {
                    eta += theta[n_int + j] * col[row];
                }
                let rate = y * eta.exp();
                let mu = dstar + rate;
                if !(mu.is_finite() && (mu > 0.0 || (mu == 0.0 && d == 0.0))) {
                    return false;
                }
                t.ll += if d > 0.0 { d * mu.ln() } else { 0.0 } - mu;
                t.deviance += if d > 0.0 {
                    2.0 * (d * (d / mu).ln() - (d - mu))
                } else {
                    2.0 * mu
                };
                t.excess[s] += rate;
                if derivatives && rate > 0.0 {
                    for (j, col) in self.x.iter().enumerate() {
                        z[n_int + j] = col[row];
                    }
                    z[s] = 1.0;
                    let g = (d / mu - 1.0) * rate;
                    let a = rate * rate / mu;
                    let o = d * (rate / mu) * (rate / mu) - g;
                    for r in 0..q {
                        if z[r] == 0.0 {
                            continue;
                        }
                        t.score[r] += g * z[r];
                        for c in 0..=r {
                            t.info[r * q + c] += a * z[r] * z[c];
                            t.observed[r * q + c] += o * z[r] * z[c];
                        }
                    }
                    z[s] = 0.0;
                }
                true
            };
        let make = |range: std::ops::Range<usize>| -> std::result::Result<Pass, ()> {
            let mut t = Pass::zero(q, n_int);
            let mut z = vec![0.0; q];
            match units {
                Units::Rows(rows) => {
                    for &i in &rows[range] {
                        let mut good = true;
                        self.records(i, |k, y, d, dstar| {
                            good &= add(&mut t, &mut z, i, k, y, d, dstar);
                        });
                        if !good {
                            return Err(());
                        }
                    }
                }
                Units::Cells(cells) => {
                    for c in &cells[range] {
                        if !add(&mut t, &mut z, c.row, c.interval, c.y, c.d, c.dstar) {
                            return Err(());
                        }
                    }
                }
            }
            Ok(t)
        };
        let n = match units {
            Units::Rows(r) => r.len(),
            Units::Cells(c) => c.len(),
        };
        let mut total =
            par::try_ordered_reduce(n, BLOCK, WAVE, make, |a: &mut Pass, b| a.add(&b)).ok()?;
        for r in 0..q {
            for c in 0..r {
                total.info[c * q + r] = total.info[r * q + c];
                total.observed[c * q + r] = total.observed[r * q + c];
            }
        }
        Some(total)
    }
}

/// The records of the rows summed into cells of one interval and one set of
/// covariate values. The sets of values are numbered in the order of their
/// first row, the cells come by set and then by interval, and each cell is
/// summed over its rows in row order.
fn aggregate(setup: &Setup<'_>, rows: &[usize]) -> Vec<Cell> {
    let x = setup.x;
    let mut order: Vec<usize> = rows.to_vec();
    order.sort_by(|&a, &b| {
        x.iter()
            .map(|c| c[a].total_cmp(&c[b]))
            .find(|o| o.is_ne())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    // The first row of each set of values, and the set of each row.
    let mut group = vec![0_usize; setup.stop.len()];
    let mut first: Vec<usize> = Vec::new();
    for (pos, &i) in order.iter().enumerate() {
        let same = pos > 0 && {
            let p = order[pos - 1];
            x.iter().all(|c| c[p].total_cmp(&c[i]).is_eq())
        };
        if !same {
            first.push(i);
        }
        group[i] = first.len() - 1;
    }
    let mut by_first: Vec<usize> = (0..first.len()).collect();
    by_first.sort_by_key(|&g| first[g]);
    let mut number = vec![0_usize; first.len()];
    for (new, &g) in by_first.iter().enumerate() {
        number[g] = new;
    }
    let n_int = setup.cuts.len() - 1;
    let mut keyed: Vec<(usize, Cell)> = Vec::new();
    for &i in rows {
        let g = group[i];
        setup.records(i, |k, y, d, dstar| {
            keyed.push((
                number[g] * n_int + k,
                Cell {
                    interval: k,
                    row: first[g],
                    y,
                    d,
                    dstar,
                },
            ));
        });
    }
    keyed.sort_by_key(|(key, _)| *key);
    let mut cells: Vec<Cell> = Vec::new();
    let mut last_key = usize::MAX;
    for (key, c) in keyed {
        match cells.last_mut() {
            Some(cell) if key == last_key => {
                cell.y += c.y;
                cell.d += c.d;
                cell.dstar += c.dstar;
            }
            _ => {
                cells.push(c);
                last_key = key;
            }
        }
    }
    cells
}

/// The information with the rows and columns of the fixed parameters replaced
/// by those of the identity.
fn reduced(info: &[f64], fixed: &[bool], q: usize) -> Vec<f64> {
    let mut a = info.to_vec();
    for s in (0..q).filter(|&s| fixed[s]) {
        for j in 0..q {
            a[s * q + j] = 0.0;
            a[j * q + s] = 0.0;
        }
        a[s * q + s] = 1.0;
    }
    a
}

/// Fits the piecewise-constant excess hazard model of Dickman et al. (2004)
/// by Newton's method.
///
/// Each row of `data` is cut at the cut points of `opts` and at the cuts of
/// the rate table, and a record has person-time `y`, an event indicator `d`
/// and the expected events `d* = y lambda*` of its reference hazard. The
/// events of a cell are Poisson with mean
/// `mu = d* + y exp(eta)` with `eta = alpha_k + x'beta`, the log excess hazard
/// of the interval plus the covariates. `x` has one slice for each covariate,
/// with one value for each row of `data`, and no intercept, since each
/// interval has its own.
///
/// # Errors
///
/// [`Error::Input`] for cut points that are not finite and strictly
/// increasing from 0, columns of other lengths than the data, covariates that
/// are not finite, expected hazards for another number of rows, follow-up
/// beyond what `expected` covers, weights other than 0 and 1, a status with
/// several event codes, no person-time, and an information matrix that is
/// singular, as with a constant or collinear covariate.
pub fn excess_hazard(
    data: &SurvData<'_>,
    x: &[&[f64]],
    expected: &ExpectedHazard,
    opts: &ExcessOptions<'_>,
) -> Result<ExcessFit> {
    let cuts = opts.cuts;
    if cuts.len() < 2
        || cuts[0] != 0.0
        || cuts.iter().any(|c| !c.is_finite())
        || cuts.windows(2).any(|w| w[1] <= w[0])
    {
        return Err(Error::input(
            "the cut points must be at least two, finite and strictly increasing from 0",
        ));
    }
    if opts.tolerance.is_nan() || opts.tolerance <= 0.0 || opts.max_iterations == 0 {
        return Err(Error::input(
            "tolerance and max_iterations must be positive",
        ));
    }
    let n = data.n_rows();
    if expected.n_rows() != n {
        return Err(Error::input(format!(
            "the expected hazards are for {} rows and the data has {n}",
            expected.n_rows()
        )));
    }
    if x.iter()
        .any(|c| c.len() != n || c.iter().any(|v| !v.is_finite()))
    {
        return Err(Error::input(
            "each covariate must be finite and have one value per row",
        ));
    }
    let surv = data.followup()?;
    if surv.n_codes() != 1 {
        return Err(Error::input(
            "the excess hazard model needs a binary status: use SurvData::cause",
        ));
    }
    if data
        .weight()
        .is_some_and(|w| w.iter().any(|&v| v != 0.0 && v != 1.0))
    {
        return Err(Error::input(
            "the excess hazard model takes weights of 0 and 1 only, which select rows",
        ));
    }
    let rows: Vec<usize> = data.used_rows().map(|r| r as usize).collect();
    let (stop, start) = (surv.stop(), surv.start());
    for &i in &rows {
        if start.is_some_and(|s| s[i] < 0.0) {
            return Err(Error::input(format!("row {i} has a negative entry time")));
        }
        if expected.horizon(i) < stop[i] {
            return Err(Error::input(format!(
                "the expected hazard of row {i} ends at {} before its follow-up at {}",
                expected.horizon(i),
                stop[i]
            )));
        }
    }
    let n_all = cuts.len() - 1;
    let mut setup = Setup {
        start,
        stop,
        status: surv.status(),
        expected,
        cuts,
        x,
        slot: vec![usize::MAX; n_all],
        n_int: 0,
        q: 0,
    };
    let (mut events, mut dstar, mut years) = (vec![0.0; n_all], vec![0.0; n_all], vec![0.0; n_all]);
    let mut n_records = 0;
    for &i in &rows {
        setup.records(i, |k, y, d, e| {
            events[k] += d;
            dstar[k] += e;
            years[k] += y;
            n_records += 1;
        });
    }
    let kept: Vec<usize> = (0..n_all).filter(|&k| years[k] > 0.0).collect();
    if kept.is_empty() {
        return Err(Error::input("there is no person-time"));
    }
    for (s, &k) in kept.iter().enumerate() {
        setup.slot[k] = s;
    }
    let n_int = kept.len();
    let q = n_int + x.len();
    setup.n_int = n_int;
    setup.q = q;
    let units = if opts.aggregate {
        Units::Cells(aggregate(&setup, &rows))
    } else {
        Units::Rows(rows)
    };
    let n_cells = match &units {
        Units::Rows(_) => n_records,
        Units::Cells(c) => c.len(),
    };

    // An interval without events has its maximum at an excess hazard of 0,
    // whatever the covariates, since its score is minus its excess events.
    let mut fixed = vec![false; q];
    let mut theta = vec![0.0; q];
    for (s, &k) in kept.iter().enumerate() {
        if events[k] == 0.0 {
            fixed[s] = true;
            theta[s] = f64::NEG_INFINITY;
        } else {
            theta[s] = ((events[k] - dstar[k]).max(0.5) / years[k]).ln();
        }
    }
    let mut current = setup
        .pass(&units, &theta, true)
        .ok_or_else(|| Error::input("the model is undefined at the starting values"))?;
    let mut converged = false;
    let mut iterations = 0;
    for _ in 0..opts.max_iterations {
        iterations += 1;
        let mut moved = false;
        for (s, &k) in kept.iter().enumerate() {
            if !fixed[s] && current.excess[s] < BOUNDARY * events[k] {
                fixed[s] = true;
                theta[s] = f64::NEG_INFINITY;
                moved = true;
            }
        }
        if moved {
            current = setup
                .pass(&units, &theta, true)
                .ok_or_else(|| Error::input("the model is undefined at the boundary"))?;
        }
        // Newton's step where the observed information is positive definite,
        // which converges quadratically near the maximum, and Fisher scoring
        // where it is not.
        let mut a = reduced(&current.observed, &fixed, q);
        let pivots = ldl::factorise(&mut a, q, 1e-13);
        if !(pivots.clean && pivots.rank == q) {
            a = reduced(&current.info, &fixed, q);
            let pivots = ldl::factorise(&mut a, q, 1e-13);
            if !(pivots.clean && pivots.rank == q) {
                return Err(Error::input(
                    "the information matrix is singular: a covariate is constant or collinear",
                ));
            }
        }
        let mut delta: Vec<f64> = current
            .score
            .iter()
            .zip(&fixed)
            .map(|(&s, &f)| if f { 0.0 } else { s })
            .collect();
        ldl::solve(&a, q, &mut delta);
        let decrement: f64 = delta.iter().zip(&current.score).map(|(d, s)| d * s).sum();
        let scale = 1.0 + current.ll.abs();
        let finishing = decrement <= opts.tolerance * scale;
        let mut alpha = 1.0;
        let mut accepted = false;
        for _ in 0..40 {
            let cand: Vec<f64> = theta
                .iter()
                .zip(&delta)
                .zip(&fixed)
                .map(|((t, d), &f)| if f { *t } else { t + alpha * d })
                .collect();
            if let Some(p) = setup.pass(&units, &cand, false)
                && (p.ll >= current.ll || (finishing && p.ll >= current.ll - 1e-12 * scale))
            {
                theta = cand;
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if accepted {
            current = setup
                .pass(&units, &theta, true)
                .ok_or_else(|| Error::input("the model is undefined at the estimate"))?;
        }
        if finishing {
            converged = true;
            break;
        }
        if !accepted {
            break;
        }
    }

    let d = n_all + x.len();
    let full = |s: usize| {
        if s < n_int {
            kept[s]
        } else {
            n_all + s - n_int
        }
    };
    let mut coefficients: Vec<Option<f64>> = vec![None; d];
    let mut at_bound = vec![false; d];
    for s in 0..q {
        coefficients[full(s)] = Some(theta[s]);
        at_bound[full(s)] = fixed[s];
    }
    let covariance = if converged {
        let mut a = reduced(&current.info, &fixed, q);
        let pivots = ldl::factorise(&mut a, q, 1e-13);
        (pivots.clean && pivots.rank == q).then(|| {
            let inv = ldl::inverse(&a, q);
            let mut out = vec![None; d * d];
            for r in (0..q).filter(|&r| !fixed[r]) {
                for c in (0..q).filter(|&c| !fixed[c]) {
                    out[full(r) * d + full(c)] = Some(inv[r * q + c]);
                }
            }
            out
        })
    } else {
        None
    };
    Ok(ExcessFit {
        coefficients,
        at_bound,
        covariance,
        log_likelihood: current.ll,
        deviance: current.deviance,
        iterations,
        converged,
        n_cells,
        events,
        expected_events: dstar,
        person_time: years,
    })
}
