//! The transition probabilities of a multi-state model, `mstate::probtrans`.

use epistat_core::{Covariance, InputError, check};
use rayon::prelude::*;

use super::CumulativeHazards;
use crate::covariance;
use crate::error::{Error, Result};

/// The form of the variance of the transition probabilities, `method` of
/// `probtrans`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ProbTransMethod {
    /// The recursion of Andersen et al., `method = "aalen"`. It evaluates the
    /// derivative of the product at the matrix after the step.
    #[default]
    Aalen,
    /// The Greenwood form, `method = "greenwood"`, which evaluates the
    /// derivative at the matrix before the step.
    Greenwood,
}

/// The direction of the product, `direction` of `probtrans`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ProbTransDirection {
    /// `P(predt, t)` for the times `t` after `predt`: the probabilities of
    /// being in each state at `t` for a subject in each state at `predt`.
    #[default]
    Forward,
    /// `P(t, predt)` for the times `t` up to `predt`: the probabilities of
    /// being in each state at the fixed horizon `predt` for a subject in each
    /// state at `t`.
    FixedHorizon,
}

/// The options of [`probtrans`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProbTransOptions {
    /// Whether to give the standard errors, `variance` of `probtrans`. They
    /// need the covariances of the cumulative hazards.
    pub se: bool,
    /// The form of the variance.
    pub method: ProbTransMethod,
    /// The direction of the product.
    pub direction: ProbTransDirection,
    /// Whether to give the covariance matrix of all the transition
    /// probabilities at each time, `covariance` of `probtrans`. As there, it
    /// implies the standard errors.
    pub covariance: bool,
    #[doc(hidden)]
    pub _extensible: epistat_core::Extensible,
}

impl Default for ProbTransOptions {
    fn default() -> Self {
        Self {
            se: true,
            method: ProbTransMethod::Aalen,
            direction: ProbTransDirection::Forward,
            covariance: false,
            _extensible: epistat_core::Extensible::NEW,
        }
    }
}

/// The transition probabilities of every pair of states at every event time
/// on one side of `predt`.
///
/// For [`ProbTransDirection::Forward`] they are `P(predt, t)` at the time
/// `predt` and the event times after it. For [`ProbTransDirection::FixedHorizon`]
/// they are `P(t, predt)`, laid out as `mstate` does: row `k` is `P(t_k,
/// predt)` for `t_k` the time of the row, the time 0 for the first row and the
/// event time before the one at which the product starts for the others, and
/// the rows after the last event time up to `predt` are the identity.
///
/// States are numbered from 0 here: state index `s` is the state code `s + 1`.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ProbTrans {
    /// The number of states.
    pub n_states: usize,
    /// The direction of the product.
    pub direction: ProbTransDirection,
    /// The time `predt`.
    pub predt: f64,
    /// The times of the rows: for a forward product `predt`, then the event
    /// times after it.
    pub time: Vec<f64>,
    /// The probability at each time, row-major by state left and state entered:
    /// entry `t * n_states^2 + from * n_states + to`. For a forward product the
    /// first time has the identity.
    pub probability: Vec<f64>,
    /// The standard error of each entry of `probability`, laid out as it is,
    /// `None` where the variance is below zero, which only rounding can cause.
    /// The outer `None` is without `se`.
    pub se: Option<Vec<Option<f64>>>,
    /// The covariance of the entries of `probability` at each time, for
    /// `covariance` in the options: one matrix of `n_states^2` by
    /// `n_states^2` for each time, whose entry `(r, c)` is the covariance of
    /// the entries `r` and `c` of that time's matrix of probabilities, each
    /// numbered in the order of the columns, `from + n_states * to`. The
    /// matrices are aligned to the rows of `time`. `mstate`'s `varMatrix` of a
    /// fixed horizon holds the matrix of row `k` in slice `k + 1`, and repeats
    /// the first one in the first slice.
    pub covariance: Option<Vec<Covariance>>,
}

impl ProbTrans {
    /// The position of `(t, from, to)` in the row-major arrays, `None` for an
    /// index out of range.
    fn position(&self, t: usize, from: usize, to: usize) -> Option<usize> {
        let s = self.n_states;
        (from < s && to < s && t < self.time.len()).then(|| t * s * s + from * s + to)
    }

    /// The probability of being in state index `to` at time index `t` for a
    /// subject in state index `from` at `predt`, `None` for an index out of
    /// range.
    #[must_use]
    pub fn probability_of(&self, t: usize, from: usize, to: usize) -> Option<f64> {
        self.position(t, from, to).map(|k| self.probability[k])
    }

    /// The standard error of [`probability_of`](Self::probability_of),
    /// `None` for an index out of range or without standard errors.
    #[must_use]
    pub fn se_of(&self, t: usize, from: usize, to: usize) -> Option<f64> {
        self.se.as_ref()?[self.position(t, from, to)?]
    }

    /// The standard error, under its name in 0.3.
    #[deprecated(since = "0.4.0", note = "use `se_of`")]
    #[must_use]
    pub fn std_err_of(&self, t: usize, from: usize, to: usize) -> Option<f64> {
        self.se_of(t, from, to)
    }

    /// The covariance at time index `t` of the probabilities `(from1, to1)` and
    /// `(from2, to2)`, `None` without `covariance` and for an index out of
    /// range.
    #[must_use]
    pub fn covariance_of(
        &self,
        t: usize,
        (from1, to1): (usize, usize),
        (from2, to2): (usize, usize),
    ) -> Option<f64> {
        let s = self.n_states;
        if from1 >= s || to1 >= s || from2 >= s || to2 >= s {
            return None;
        }
        self.covariance
            .as_ref()?
            .get(t)?
            .get(from1 + s * to1, from2 + s * to2)
    }
}

/// A dense square matrix of side `n`, row-major.
fn matmul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        for l in 0..n {
            let x = a[i * n + l];
            if x != 0.0 {
                for j in 0..n {
                    out[i * n + j] += x * b[l * n + j];
                }
            }
        }
    }
    out
}

fn transpose(a: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            out[j * n + i] = a[i * n + j];
        }
    }
    out
}

/// The Kronecker product `x (x) y` of two square matrices of side `s`, as R's
/// `kronecker`.
fn kron(x: &[f64], y: &[f64], s: usize) -> Vec<f64> {
    let n = s * s;
    let mut out = vec![0.0; n * n];
    for a in 0..s {
        for c in 0..s {
            let u = x[a * s + c];
            for i in 0..s {
                for j in 0..s {
                    out[(a * s + i) * n + (c * s + j)] = u * y[i * s + j];
                }
            }
        }
    }
    out
}

fn identity(s: usize) -> Vec<f64> {
    (0..s * s)
        .map(|i| f64::from(u8::from(i / s == i % s)))
        .collect()
}

/// `K V K'`.
fn sandwich(k: &[f64], v: &[f64], n: usize) -> Vec<f64> {
    matmul(&matmul(k, v, n), &transpose(k, n), n)
}

/// The covariance matrix of the cumulative hazards at time index `j`, as a
/// matrix over the entries of the transition matrix: the entry
/// `(from1 + S to1, from2 + S to2)` is the covariance of those two hazards.
fn cov_matrix(h: &CumulativeHazards, j: usize, s: usize) -> Vec<f64> {
    let n = s * s;
    let mut m = vec![0.0; n * n];
    let Some(cov) = &h.covariance else {
        return m;
    };
    let k_n = h.transitions.len();
    for a in 0..k_n {
        for b in a..k_n {
            let value = cov[j].get(a, b).unwrap_or(f64::NAN);
            let (f1, t1) = (
                h.transitions[a].0 as usize - 1,
                h.transitions[a].1 as usize - 1,
            );
            let (f2, t2) = (
                h.transitions[b].0 as usize - 1,
                h.transitions[b].1 as usize - 1,
            );
            let (r, c) = (f1 + s * t1, f2 + s * t2);
            m[r * n + c] = value;
            m[c * n + r] = value;
        }
    }
    m
}

/// Complete an increment of the covariance matrix of the cumulative hazards
/// with the entries of the diagonal of the transition matrix, which are minus
/// the sum of the rest of their row.
fn complete_diagonal(m: &mut [f64], s: usize) {
    let n = s * s;
    let at = |i: usize, j: usize| i + s * j;
    for from in 0..s {
        for from2 in 0..s {
            for to2 in 0..s {
                if to2 == from2 {
                    continue;
                }
                let mut sum = 0.0;
                for j in (0..s).filter(|&j| j != from) {
                    sum += m[at(from, j) * n + at(from2, to2)];
                }
                m[at(from, from) * n + at(from2, to2)] = -sum;
                m[at(from2, to2) * n + at(from, from)] = -sum;
            }
        }
    }
    for from in 0..s {
        for from2 in 0..s {
            let mut sum = 0.0;
            for j in (0..s).filter(|&j| j != from2) {
                sum += m[at(from, from) * n + at(from2, j)];
            }
            m[at(from, from) * n + at(from2, from2)] = -sum;
            m[at(from2, from2) * n + at(from, from)] = -sum;
        }
    }
}

/// `I + dA` at time index `j` of the cumulative hazards.
fn step_matrix(h: &CumulativeHazards, j: usize, s: usize) -> Result<Vec<f64>> {
    let mut b = identity(s);
    for k in 0..h.transitions.len() {
        let before = if j == 0 {
            0.0
        } else {
            h.cumulative_hazard[k][j - 1]
        };
        let d = h.cumulative_hazard[k][j] - before;
        let (from, to) = (
            h.transitions[k].0 as usize - 1,
            h.transitions[k].1 as usize - 1,
        );
        b[from * s + to] = d;
        b[from * s + from] -= d;
    }
    if let Some(from) = (0..s).find(|&i| b[i * s + i] < 0.0) {
        return Err(Error::value(format!(
            "the diagonal of I + dA for state {} is negative at time {}: the hazards are too large \
             for the product integral",
            from + 1,
            h.time[j]
        )));
    }
    Ok(b)
}

fn std_errs(var_p: &[f64], s: usize) -> impl Iterator<Item = Option<f64>> + '_ {
    let n = s * s;
    (0..n).map(move |i| {
        // The entry (from, to) sits at from + s * to, and the layout of the
        // output is by from and then to.
        let (from, to) = (i / s, i % s);
        let k = from + s * to;
        let v = var_p[k * n + k];
        (v >= 0.0).then(|| v.sqrt())
    })
}

/// The transition probabilities of the cumulative hazards by the product
/// integral, with the standard errors of `mstate`.
///
/// The probabilities are those of `probtrans(msfit, predt)` of `mstate`, for
/// every state at `predt` for the forward direction and at the earlier times
/// for the fixed horizon. See the [module documentation](super) for the
/// formulas.
///
/// # Errors
///
/// An input error when `predt` is not finite, when the standard errors or
/// the covariance are asked for and the cumulative hazards have no covariances,
/// when a transition of `h` is not between its states or its hazards and
/// covariances are not one for each transition and time,
/// when no time is after `predt` (forward) or at or before it (fixed horizon),
/// and when the diagonal of `I + dA` is negative at an event time, which is where
/// `probtrans` warns that the estimate may not be meaningful: the hazards of
/// the profile are too large for one step of the product integral.
pub fn probtrans(h: &CumulativeHazards, predt: f64, opts: &ProbTransOptions) -> Result<ProbTrans> {
    if !predt.is_finite() {
        return Err(Error::from(InputError::Domain {
            what: "predt",
            value: predt,
            domain: "(-inf, inf)",
            row: None,
        }));
    }
    let variance = opts.se || opts.covariance;
    if variance && h.covariance.is_none() {
        return Err(Error::fit(
            "se: the cumulative hazards have no covariances; ask `msfit` for them or set `se` false",
        ));
    }
    check_hazards(h)?;
    let s = h.n_states as usize;
    match opts.direction {
        ProbTransDirection::Forward => forward(h, predt, *opts, s, variance),
        ProbTransDirection::FixedHorizon => fixed_horizon(h, predt, *opts, s, variance),
    }
}

/// Refuse cumulative hazards whose parts do not agree: a transition from or
/// to a state outside `1..=n_states`, or hazards and covariances that are not
/// one for each transition and time.
fn check_hazards(h: &CumulativeHazards) -> Result<()> {
    let outside = |state: u32| state == 0 || state > h.n_states;
    if let Some(&(from, to)) = h
        .transitions
        .iter()
        .find(|&&(from, to)| outside(from) || outside(to))
    {
        return Err(Error::out_of_range(format!(
            "transitions: ({from}, {to}) is not a pair of the states 1..={}",
            h.n_states
        )));
    }
    check::same_length(
        "cumulative_hazard",
        h.transitions.len(),
        h.cumulative_hazard.len(),
    )?;
    for hazard in &h.cumulative_hazard {
        check::same_length("cumulative_hazard", h.time.len(), hazard.len())?;
    }
    if let Some(covariance) = &h.covariance {
        check::same_length("covariance", h.time.len(), covariance.len())?;
    }
    Ok(())
}

/// [`probtrans`] for each of several sets of cumulative hazards, in the order
/// given, evaluated in parallel with a result that does not depend on the
/// number of threads.
///
/// # Errors
///
/// As [`probtrans`], for the first set in the order given that fails.
pub fn probtrans_many(
    hazards: &[CumulativeHazards],
    predt: f64,
    opts: &ProbTransOptions,
) -> Result<Vec<ProbTrans>> {
    let results: Vec<Result<ProbTrans>> = hazards
        .par_iter()
        .map(|h| probtrans(h, predt, opts))
        .collect();
    results.into_iter().collect()
}

fn forward(
    h: &CumulativeHazards,
    predt: f64,
    opts: ProbTransOptions,
    s: usize,
    variance: bool,
) -> Result<ProbTrans> {
    let n = s * s;
    let first = h.time.partition_point(|&t| t <= predt);
    if first == h.time.len() {
        return Err(Error::no_information("predt: no event time is after it"));
    }
    let mut time = vec![predt];
    let mut probability = identity(s);
    let mut std_err: Option<Vec<Option<f64>>> = variance.then(|| vec![Some(0.0); n]);
    let mut covariance: Option<Vec<f64>> = opts.covariance.then(|| vec![0.0; n * n]);
    let mut p = probability.clone();
    let mut var_p = vec![0.0; n * n];
    let mut cov_before = if variance && first > 0 {
        cov_matrix(h, first - 1, s)
    } else {
        vec![0.0; n * n]
    };
    let eye = identity(s);

    for j in first..h.time.len() {
        let b = step_matrix(h, j, s)?;
        let p_before = p.clone();
        p = matmul(&p, &b, s);
        if let Some(se) = std_err.as_mut() {
            let cov_now = cov_matrix(h, j, s);
            let mut cov_step: Vec<f64> = cov_now
                .iter()
                .zip(&cov_before)
                .map(|(new, old)| new - old)
                .collect();
            cov_before = cov_now;
            complete_diagonal(&mut cov_step, s);
            // vec(P B) = (B' x I) vec(P) = (I x P) vec(B), the vec in the order
            // of the columns. The first form carries the variance of the matrix
            // before the step, the second the increment of the hazards, with the
            // matrix after the step for the Aalen form and before it for the
            // Greenwood form.
            let k1 = kron(&transpose(&b, s), &eye, s);
            let at = if opts.method == ProbTransMethod::Aalen {
                &p
            } else {
                &p_before
            };
            let k2 = kron(&eye, at, s);
            let first_part = sandwich(&k1, &var_p, n);
            let second_part = sandwich(&k2, &cov_step, n);
            for (v, (x, y)) in var_p.iter_mut().zip(first_part.iter().zip(&second_part)) {
                *v = x + y;
            }
            se.extend(std_errs(&var_p, s));
            if let Some(c) = covariance.as_mut() {
                c.extend_from_slice(&var_p);
            }
        }
        time.push(h.time[j]);
        probability.extend_from_slice(&p);
    }
    Ok(ProbTrans {
        n_states: s,
        direction: ProbTransDirection::Forward,
        predt,
        time,
        probability,
        se: std_err,
        covariance: covariance
            .map(|c| covariance::blocks(c.len() / (n * n), n, &c))
            .transpose()?,
    })
}

fn fixed_horizon(
    h: &CumulativeHazards,
    predt: f64,
    opts: ProbTransOptions,
    s: usize,
    variance: bool,
) -> Result<ProbTrans> {
    let n = s * s;
    let last = h.time.partition_point(|&t| t <= predt);
    if last == 0 {
        return Err(Error::no_information(
            "predt: no event time is at or before it",
        ));
    }
    let eye = identity(s);
    let mut p = identity(s);
    let mut var_p = vec![0.0; n * n];
    // The rows from the last event time back to the first.
    let mut probability_rows: Vec<Vec<f64>> = Vec::with_capacity(last);
    let mut var_rows: Vec<Vec<f64>> = Vec::with_capacity(if variance { last } else { 0 });
    let mut cov_now = if variance {
        cov_matrix(h, last - 1, s)
    } else {
        Vec::new()
    };
    for j in (0..last).rev() {
        let b = step_matrix(h, j, s)?;
        if variance {
            let cov_before = if j > 0 {
                cov_matrix(h, j - 1, s)
            } else {
                vec![0.0; n * n]
            };
            let mut cov_step: Vec<f64> = cov_now
                .iter()
                .zip(&cov_before)
                .map(|(new, old)| new - old)
                .collect();
            cov_now = cov_before;
            complete_diagonal(&mut cov_step, s);
            let k1 = kron(&eye, &b, s);
            let first_part = sandwich(&k1, &var_p, n);
            let pt = transpose(&p, s);
            let k2 = if opts.method == ProbTransMethod::Aalen {
                kron(&pt, &b, s)
            } else {
                kron(&pt, &eye, s)
            };
            let second_part = sandwich(&k2, &cov_step, n);
            for (v, (x, y)) in var_p.iter_mut().zip(first_part.iter().zip(&second_part)) {
                *v = x + y;
            }
            var_rows.push(var_p.clone());
        }
        p = matmul(&b, &p, s);
        probability_rows.push(p.clone());
    }
    probability_rows.reverse();
    var_rows.reverse();

    let mut time = Vec::with_capacity(last + 2);
    let mut probability = Vec::with_capacity((last + 2) * n);
    let mut std_err: Option<Vec<Option<f64>>> = variance.then(Vec::new);
    let mut covariance: Option<Vec<f64>> = opts.covariance.then(Vec::new);
    for k in 0..last {
        time.push(if k == 0 { 0.0 } else { h.time[k - 1] });
        probability.extend_from_slice(&probability_rows[k]);
        if let Some(se) = std_err.as_mut() {
            se.extend(std_errs(&var_rows[k], s));
        }
        if let Some(c) = covariance.as_mut() {
            c.extend_from_slice(&var_rows[k]);
        }
    }
    // The identity at the end: at the last event time too when `predt` lies
    // after it.
    let mut tail = vec![predt];
    // `predt` is an event time when it equals one exactly, as in `mstate`.
    if h.time[last - 1].total_cmp(&predt).is_ne() {
        tail.insert(0, h.time[last - 1]);
    }
    for t in tail {
        time.push(t);
        probability.extend_from_slice(&eye);
        if let Some(se) = std_err.as_mut() {
            se.extend(std::iter::repeat_n(Some(0.0), n));
        }
        if let Some(c) = covariance.as_mut() {
            c.extend(std::iter::repeat_n(0.0, n * n));
        }
    }
    Ok(ProbTrans {
        n_states: s,
        direction: ProbTransDirection::FixedHorizon,
        predt,
        time,
        probability,
        se: std_err,
        covariance: covariance
            .map(|c| covariance::blocks(c.len() / (n * n), n, &c))
            .transpose()?,
    })
}
