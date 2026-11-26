//! The terms of the variance that come from estimating the weights.
//!
//! An outcome estimate `beta` solves `sum_u U_u(beta; w) = 0` with row weights
//! `w_k(theta)` that depend on estimated parameters `theta` of a nuisance model,
//! which solve `sum_u psi_u(theta) = 0`. Perturbing the case weight of unit `u`
//! moves `beta` by `V (U_u + sum_k g_k IF_u(w_k))`, with `V` the inverse of minus
//! the derivative of `U`, `g_k` the derivative of the outcome equations with
//! respect to the weight of row `k` and `IF_u(w_k)` the change of that weight.
//! Weights are positive, so `IF_u(w_k) = w_k IF_u(log w_k)` and, with `a_k = g_k
//! w_k`, the term to add to the influence of unit `u` is
//!
//! ```text
//! sum over rows k of a_k IF_u(log w_k).
//! ```
//!
//! A [`Nuisance`] forms this for every unit and for several columns `a` at once.
//! Weights that are products of the weights of several models have the sum of the
//! terms of the models, because the log of a product is a sum, and because the
//! models are fitted separately, so that the influence of the parameters of one
//! does not depend on the others.
//!
//! For a logistic model with `q` parameters, `IF_u(theta) = -A^-1 psi_u` with `A`
//! the derivative of the summed estimating functions, so with `K = sum_k a_k d log
//! w_k / d theta` (`q` values per column) the term is `psi_u' N` where `A N = -K`.
//! The estimating functions of a unit are summed from its items, so no matrix of
//! units by parameters is formed: the terms are added into one matrix of units by
//! columns of `a`.

use epistat_linalg::{Factor, Mat, MatRef, weighted_gram};

use super::iptw::Iptw;
use super::pooled::PooledIpcw;
use super::stack::GENERAL;
use crate::error::{Error, Result};

/// Which weight of a row an outcome model uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WeightAt {
    /// The weight of an outcome observed at the end of the row: `1 / G(stop-)`
    /// of [`ipcw`](super::ipcw), and `weight` of [`pooled`](super::pooled), the
    /// product through the row.
    #[default]
    Stop,
    /// The weight at the start of the row, `1 / G(start)` and `weight_before`:
    /// the weight a row carries over its interval in split follow-up.
    Start,
}

/// One set of estimated weights, with its contribution to the influence of
/// every unit.
pub(super) trait Nuisance {
    /// Add `sum over rows k of a[k, m] IF_u(log w_k)` to `total[(u, m)]` for every
    /// unit `u` and column `m`. `a` has one row per row of the outcome data and
    /// `total` one row per unit.
    fn add(&self, a: MatRef<'_, f64>, total: &mut Mat<f64>) -> Result<()>;
}

/// Solve `A N = -K` for the symmetric derivative `A` of a logistic model.
fn influence_matrix(factor: &Factor, k: &Mat<f64>) -> Result<Mat<f64>> {
    let minus = Mat::from_fn(k.nrows(), k.ncols(), |i, j| -k[(i, j)]);
    factor.solve(minus.as_ref()).map_err(Error::from)
}

/// The propensity model of treatment weights.
pub(super) struct IptwNuisance<'a> {
    pub iptw: &'a Iptw<'a>,
    pub subject_of_row: Vec<u32>,
    pub unit_of_subject: Vec<usize>,
    factor: Factor,
}

impl<'a> IptwNuisance<'a> {
    pub fn new(
        iptw: &'a Iptw<'a>,
        subject_of_row: Vec<u32>,
        unit_of_subject: Vec<usize>,
    ) -> Result<Self> {
        let bread = iptw.bread()?;
        let factor = Factor::new(bread.as_ref(), &GENERAL)?;
        Ok(Self {
            iptw,
            subject_of_row,
            unit_of_subject,
            factor,
        })
    }
}

impl Nuisance for IptwNuisance<'_> {
    fn add(&self, a: MatRef<'_, f64>, total: &mut Mat<f64>) -> Result<()> {
        let (k, m) = (self.iptw.n_params(), a.ncols());
        let mut big_k = Mat::<f64>::zeros(k, m);
        let mut g = vec![0.0; k];
        for row in 0..a.nrows() {
            self.iptw
                .log_weight_gradient_into(self.subject_of_row[row] as usize, &mut g);
            for c in 0..m {
                let v = a[(row, c)];
                for (q, &gq) in g.iter().enumerate() {
                    big_k[(q, c)] += v * gq;
                }
            }
        }
        let n = influence_matrix(&self.factor, &big_k)?;
        let mut psi = vec![0.0; k];
        for (i, &unit) in self.unit_of_subject.iter().enumerate() {
            self.iptw.score_into(i, &mut psi);
            for c in 0..m {
                let mut s = 0.0;
                for (q, &p) in psi.iter().enumerate() {
                    s += p * n[(q, c)];
                }
                total[(unit, c)] += s;
            }
        }
        Ok(())
    }
}

/// The pooled logistic models of censoring weights.
pub(super) struct PooledNuisance<'a> {
    pub pooled: &'a PooledIpcw<'a>,
    pub at: WeightAt,
    pub unit_of_row: Vec<usize>,
    factor: Factor,
}

impl<'a> PooledNuisance<'a> {
    pub fn new(pooled: &'a PooledIpcw<'a>, at: WeightAt, unit_of_row: Vec<usize>) -> Result<Self> {
        let (qd, q) = (pooled.design.ncols(), pooled.n_params());
        let fisher = |design: MatRef<'_, f64>, p: &[f64]| -> Result<Mat<f64>> {
            let var: Vec<f64> = p
                .iter()
                .zip(&pooled.active)
                .map(|(&p, &on)| if on { p * (1.0 - p) } else { 0.0 })
                .collect();
            weighted_gram(design, &var).map_err(Error::from)
        };
        let gd = fisher(pooled.design, &pooled.probability)?;
        let gn = match (pooled.numerator_design, &pooled.numerator_probability) {
            (Some(d), Some(p)) => Some(fisher(d, p)?),
            _ => None,
        };
        let bread = Mat::from_fn(q, q, |i, j| {
            if i < qd && j < qd {
                -gd[(i, j)]
            } else if i >= qd && j >= qd {
                gn.as_ref().map_or(0.0, |g| -g[(i - qd, j - qd)])
            } else {
                0.0
            }
        });
        let factor = Factor::new(bread.as_ref(), &GENERAL)?;
        Ok(Self {
            pooled,
            at,
            unit_of_row,
            factor,
        })
    }
}

impl PooledNuisance<'_> {
    /// The part of `K` from the truncated rows: for each limit `L`, the sum of `a`
    /// over the rows set to it times `d log L / d theta`, with `dL / dtheta` the
    /// interpolation of the derivatives of the weights of its two order
    /// statistics. As `(parameter, column, value)` entries, `None` without
    /// truncation.
    fn limit_terms(&self, a: MatRef<'_, f64>) -> Option<Vec<(usize, usize, f64)>> {
        let p = self.pooled;
        let (Some(limits), Some((stop, before)), Some(untruncated)) =
            (&p.limits, &p.clamped, &p.untruncated)
        else {
            return None;
        };
        let (flags, values) = if self.at == WeightAt::Stop {
            (stop, &p.weight)
        } else {
            (before, &p.weight_before)
        };
        let (q, m) = (p.n_params(), a.ncols());
        let mut out = Vec::with_capacity(2 * q * m);
        let mut g = vec![0.0; q];
        let mut slope = vec![0.0; q];
        for (which, limit) in limits.iter().enumerate() {
            // A clamped weight equals one of the two limits exactly: the lower
            // one when it equals it, else the upper.
            let at_limit = |r: usize| {
                flags[r] && (values[r].to_bits() == limits[0].value.to_bits()) == (which == 0)
            };
            let sums: Vec<f64> = (0..m)
                .map(|c| {
                    (0..a.nrows())
                        .filter(|&r| at_limit(r))
                        .map(|r| a[(r, c)])
                        .sum()
                })
                .collect();
            if sums.iter().all(|&v| v == 0.0) {
                continue;
            }
            slope.fill(0.0);
            let (j, k) = limit.rows;
            for (row, share) in [(j, 1.0 - limit.h), (k, limit.h)] {
                if share == 0.0 {
                    continue;
                }
                p.log_weight_gradient_into(row, &mut g);
                for (s, &gq) in slope.iter_mut().zip(&g) {
                    *s += share * untruncated[row] * gq;
                }
            }
            for (jq, &s) in slope.iter().enumerate() {
                for (c, &sum) in sums.iter().enumerate() {
                    out.push((jq, c, sum * s / limit.value));
                }
            }
        }
        Some(out)
    }
}

impl Nuisance for PooledNuisance<'_> {
    fn add(&self, a: MatRef<'_, f64>, total: &mut Mat<f64>) -> Result<()> {
        let p = self.pooled;
        let (qd, q, m) = (p.design.ncols(), p.n_params(), a.ncols());
        let n = a.nrows();
        let limit_terms = self.limit_terms(a);
        // A weight set to a limit of the truncation moves with the limit, whose
        // term is added below, not with its own factors.
        let masked;
        let a = match &p.clamped {
            Some((stop, before)) => {
                let flags = if self.at == WeightAt::Stop {
                    stop
                } else {
                    before
                };
                masked = Mat::from_fn(n, m, |r, c| if flags[r] { 0.0 } else { a[(r, c)] });
                masked.as_ref()
            }
            None => a,
        };
        // K: the rows after a row, and the row itself for the weight through
        // the row, have the weight of that row in their product.
        let mut big_k = Mat::<f64>::zeros(q, m);
        let mut s = vec![0.0; m];
        let mut end = n;
        while end > 0 {
            let id = p.subject[p.order[end - 1] as usize];
            let mut start = end - 1;
            while start > 0 && p.subject[p.order[start - 1] as usize] == id {
                start -= 1;
            }
            s.fill(0.0);
            for pos in (start..end).rev() {
                let r = p.order[pos] as usize;
                if self.at == WeightAt::Stop {
                    for (c, sc) in s.iter_mut().enumerate() {
                        *sc += a[(r, c)];
                    }
                }
                if p.active[r] {
                    // The factor is the probability of the value observed, `o`:
                    // d log f / d theta is `(o - p) z`, and the weight divides by
                    // the denominator's and multiplies by the numerator's.
                    let o = f64::from(u8::from(p.observed(r)));
                    let pd = p.probability[r] - o;
                    for (c, &sc) in s.iter().enumerate() {
                        for j in 0..qd {
                            big_k[(j, c)] += sc * pd * p.design[(r, j)];
                        }
                    }
                    if let (Some(zn), Some(pn)) = (p.numerator_design, &p.numerator_probability) {
                        let pn = o - pn[r];
                        for (c, &sc) in s.iter().enumerate() {
                            for j in 0..zn.ncols() {
                                big_k[(qd + j, c)] += sc * pn * zn[(r, j)];
                            }
                        }
                    }
                }
                if self.at == WeightAt::Start {
                    for (c, sc) in s.iter_mut().enumerate() {
                        *sc += a[(r, c)];
                    }
                }
            }
            end = start;
        }
        if let Some(terms) = limit_terms {
            for (j, c, v) in terms {
                big_k[(j, c)] += v;
            }
        }
        let nn = influence_matrix(&self.factor, &big_k)?;
        let mut psi = vec![0.0; q];
        for r in 0..n {
            if !p.active[r] {
                continue;
            }
            let y = f64::from(u8::from(p.response(r)));
            let res = y - p.probability[r];
            for (j, v) in psi.iter_mut().enumerate().take(qd) {
                *v = p.design[(r, j)] * res;
            }
            if let (Some(zn), Some(pn)) = (p.numerator_design, &p.numerator_probability) {
                let res = y - pn[r];
                for j in 0..zn.ncols() {
                    psi[qd + j] = zn[(r, j)] * res;
                }
            }
            let unit = self.unit_of_row[r];
            for c in 0..m {
                let mut sum = 0.0;
                for (j, &v) in psi.iter().enumerate() {
                    sum += v * nn[(j, c)];
                }
                total[(unit, c)] += sum;
            }
        }
        Ok(())
    }
}
