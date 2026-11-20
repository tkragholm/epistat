//! The transition probabilities of a multi-state model, `mstate::probtrans`.

use super::CumulativeHazards;
use crate::error::{Error, Result};

/// The options of [`probtrans`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProbTransOptions {
    /// Whether to give the standard errors, `variance` of `probtrans`. They
    /// need the covariances of the cumulative hazards.
    pub se: bool,
}

impl Default for ProbTransOptions {
    fn default() -> Self {
        Self { se: true }
    }
}

/// The transition probabilities `P(predt, t)` of every pair of states at every
/// event time after `predt`.
///
/// States are numbered from 0 here: state index `s` is the state code `s + 1`.
#[derive(Debug, Clone, PartialEq)]
pub struct ProbTrans {
    /// The number of states.
    pub n_states: usize,
    /// The time `predt`, then the event times after it.
    pub time: Vec<f64>,
    /// `P(predt, t)` at each time, row-major by state left and state entered:
    /// entry `t * n_states^2 + from * n_states + to`. The first time has the
    /// identity.
    pub probability: Vec<f64>,
    /// The standard error of each entry of `probability`, laid out as it is,
    /// `None` where the variance is below zero, which only rounding can cause.
    /// The outer `None` is without `se`.
    pub std_err: Option<Vec<Option<f64>>>,
}

impl ProbTrans {
    /// The probability of being in state index `to` at time index `t` for a
    /// subject in state index `from` at `predt`.
    ///
    /// # Panics
    ///
    /// When an index is out of range.
    #[must_use]
    pub fn probability_of(&self, t: usize, from: usize, to: usize) -> f64 {
        let s = self.n_states;
        self.probability[t * s * s + from * s + to]
    }

    /// The standard error of [`probability_of`](Self::probability_of).
    ///
    /// # Panics
    ///
    /// When an index is out of range.
    #[must_use]
    pub fn std_err_of(&self, t: usize, from: usize, to: usize) -> Option<f64> {
        let s = self.n_states;
        self.std_err.as_ref()?[t * s * s + from * s + to]
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
            let value = cov[h.pair_index(a, b)][j];
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

/// The transition probabilities from `predt` on, by the product integral of
/// the cumulative hazards, with the standard errors of `mstate`'s default.
///
/// The probabilities are those of `probtrans(msfit, predt)` of `mstate`
/// with `direction = "forward"`, for every state at `predt`. See the
/// [module documentation](super) for the formulas.
///
/// # Errors
///
/// [`Error::Input`] when `predt` is not finite, when `opts.se` is set and the
/// cumulative hazards have no covariances, when no time is after `predt`, and
/// when the diagonal of `I + dA` is negative at an event time, which is where
/// `probtrans` warns that the estimate may not be meaningful: the hazards of
/// the profile are too large for one step of the product integral.
pub fn probtrans(h: &CumulativeHazards, predt: f64, opts: &ProbTransOptions) -> Result<ProbTrans> {
    if !predt.is_finite() {
        return Err(Error::input("predt: must be finite"));
    }
    if opts.se && h.covariance.is_none() {
        return Err(Error::input(
            "se: the cumulative hazards have no covariances; ask `msfit` for them or set `se` false",
        ));
    }
    let s = h.n_states as usize;
    let n = s * s;
    let first = h.time.partition_point(|&t| t <= predt);
    if first == h.time.len() {
        return Err(Error::input("predt: no event time is after it"));
    }
    let k_n = h.transitions.len();

    let mut time = vec![predt];
    let mut probability: Vec<f64> = (0..n)
        .map(|i| f64::from(u8::from(i / s == i % s)))
        .collect();
    let mut std_err: Option<Vec<Option<f64>>> = opts.se.then(|| vec![Some(0.0); n]);
    let mut p: Vec<f64> = probability.clone();
    let mut var_p = vec![0.0; n * n];
    let mut cov_before = if opts.se && first > 0 {
        cov_matrix(h, first - 1, s)
    } else {
        vec![0.0; n * n]
    };

    for j in first..h.time.len() {
        // I + dA.
        let mut b: Vec<f64> = (0..n)
            .map(|i| f64::from(u8::from(i / s == i % s)))
            .collect();
        for k in 0..k_n {
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
            return Err(Error::input(format!(
                "the diagonal of I + dA for state {} is negative at time {}: the hazards are too large \
                 for the product integral",
                from + 1,
                h.time[j]
            )));
        }
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
            // of the columns.
            let mut k1 = vec![0.0; n * n];
            let mut k2 = vec![0.0; n * n];
            for a in 0..s {
                for c in 0..s {
                    for i in 0..s {
                        k1[(a * s + i) * n + (c * s + i)] = b[c * s + a];
                        k2[(a * s + i) * n + (a * s + c)] = p[i * s + c];
                    }
                }
            }
            let first_part = sandwich(&k1, &var_p, n);
            let second_part = sandwich(&k2, &cov_step, n);
            for (v, (x, y)) in var_p.iter_mut().zip(first_part.iter().zip(&second_part)) {
                *v = x + y;
            }
            for from in 0..s {
                for to in 0..s {
                    let v = var_p[(from + s * to) * n + (from + s * to)];
                    se.push((v >= 0.0).then(|| v.sqrt()));
                }
            }
        }
        time.push(h.time[j]);
        probability.extend_from_slice(&p);
    }
    Ok(ProbTrans {
        n_states: s,
        time,
        probability,
        std_err,
    })
}
