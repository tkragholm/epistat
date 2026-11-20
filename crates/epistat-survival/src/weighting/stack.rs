//! The estimating equations of the propensity model, stacked under those of an
//! outcome model.
//!
//! The parameters are the coefficients of the propensity model and, for
//! stabilised weights, the proportion treated. With `w_i` the weight,
//! `eta_i` the linear predictor and `c_i = d log w_i / d eta_i`,
//!
//! ```text
//! psi_i(alpha) = z_i (a_i - p_i)         d psi_i / d alpha = - p_i (1 - p_i) z_i z_i'
//! psi_i(pi)    = a_i - pi                d psi_i / d pi    = -1
//! d w_i / d alpha = w_i c_i z_i          d w_i / d pi = w_i (a_i / pi - (1 - a_i) / (1 - pi))
//! ```
//!
//! and `c_i` is `-(1 - p_i)` for a treated and `p_i` for a control subject
//! under the average treatment effect, 1 for a control subject under the
//! effect on the treated, `-1` for a treated subject under the effect on the
//! controls, and 0 otherwise.

use epistat_linalg::{Attempt, Factor, Method, weighted_gram};
use faer::Mat;

use super::iptw::{Estimand, Iptw};
use crate::error::{Error, Result};
use crate::surv::{SurvBuf, SurvData};

/// The attempts for the stacked matrices, which are not symmetric.
pub(super) const GENERAL: [Attempt; 2] = [
    Attempt::plain(Method::PartialPivLu),
    Attempt::plain(Method::FullPivLu),
];

impl Iptw<'_> {
    /// The number of parameters of the weights.
    pub(super) fn n_params(&self) -> usize {
        self.design.ncols() + usize::from(self.stabilised)
    }

    /// `c_i`, the derivative of the log weight with respect to the linear
    /// predictor.
    fn log_slope(&self, i: usize) -> f64 {
        let (a, p) = (self.treatment[i], self.propensity[i]);
        match (self.estimand, a) {
            (Estimand::Ate, true) => -(1.0 - p),
            (Estimand::Ate, false) => p,
            (Estimand::Att, false) => 1.0,
            (Estimand::Atc, true) => -1.0,
            _ => 0.0,
        }
    }

    /// The estimating function of subject `i`, `n_params` values.
    pub(super) fn score_into(&self, i: usize, out: &mut [f64]) {
        let p = self.design.ncols();
        let r = f64::from(u8::from(self.treatment[i])) - self.propensity[i];
        for (j, o) in out.iter_mut().enumerate().take(p) {
            *o = self.design[(i, j)] * r;
        }
        if let Some(pi) = self.marginal {
            out[p] = f64::from(u8::from(self.treatment[i])) - pi;
        }
    }

    /// The derivative of the weight of subject `i` with respect to the
    /// parameters, `n_params` values.
    pub(super) fn weight_gradient_into(&self, i: usize, out: &mut [f64]) {
        let p = self.design.ncols();
        let w = self.weights[i];
        let f = w * self.log_slope(i);
        for (j, o) in out.iter_mut().enumerate().take(p) {
            *o = f * self.design[(i, j)];
        }
        if let Some(pi) = self.marginal {
            out[p] = if self.treatment[i] {
                w / pi
            } else {
                -w / (1.0 - pi)
            };
        }
    }

    /// The derivative of the summed estimating functions, block diagonal.
    pub(super) fn bread(&self) -> Result<Mat<f64>> {
        let (n, p) = (self.len(), self.design.ncols());
        let var: Vec<f64> = self.propensity.iter().map(|&q| q * (1.0 - q)).collect();
        let gram = weighted_gram(self.design, &var)
            .map_err(|e| Error::input(format!("propensity model: {e}")))?;
        let k = self.n_params();
        Ok(Mat::from_fn(k, k, |i, j| {
            if i < p && j < p {
                -gram[(i, j)]
            } else if i == p && j == p {
                -(n as f64)
            } else {
                0.0
            }
        }))
    }
}

/// The units that estimating functions are summed over: the cluster of the
/// rows of each subject when there are clusters, otherwise the subject.
///
/// Returns the unit of each subject and the number of units. A subject
/// without rows is a unit of its own when there are clusters.
pub(super) fn units(
    n_subjects: usize,
    subject_of_row: &dyn Fn(usize) -> usize,
    cluster: Option<&[u32]>,
    n_clusters: usize,
) -> Result<(Vec<usize>, usize)> {
    let Some(cluster) = cluster else {
        return Ok(((0..n_subjects).collect(), n_subjects));
    };
    let mut unit = vec![usize::MAX; n_subjects];
    for (row, &c) in cluster.iter().enumerate() {
        let s = subject_of_row(row);
        let c = c as usize;
        if unit[s] == usize::MAX {
            unit[s] = c;
        } else if unit[s] != c {
            return Err(Error::input(format!(
                "cluster: the rows of subject {s} are in more than one cluster"
            )));
        }
    }
    let mut next = n_clusters;
    for u in &mut unit {
        if *u == usize::MAX {
            *u = next;
            next += 1;
        }
    }
    Ok((unit, next))
}

/// Check that every row names a subject of the weights, and that without a
/// map the rows are the subjects.
pub(super) fn check_subjects(
    n_rows: usize,
    n_subjects: usize,
    subject: Option<&[u32]>,
) -> Result<()> {
    match subject {
        None if n_rows != n_subjects => Err(Error::input(format!(
            "subject: the data has {n_rows} rows and the weights {n_subjects} subjects, \
             so a subject for each row is needed"
        ))),
        None => Ok(()),
        Some(s) if s.len() != n_rows => Err(Error::input(format!(
            "subject: {} values for {n_rows} rows",
            s.len()
        ))),
        Some(s) => match s.iter().position(|&v| v as usize >= n_subjects) {
            Some(row) => Err(Error::input(format!(
                "subject: row {row} names subject {} of {n_subjects}",
                s[row]
            ))),
            None => Ok(()),
        },
    }
}

/// A copy of the follow-up columns of `data` with the case weights `weight`,
/// refusing data that already has weights.
pub(super) fn with_weights(data: &SurvData<'_>, weight: Vec<f64>) -> Result<SurvBuf> {
    if data.weight().is_some() {
        return Err(Error::input(
            "data: carries case weights, but the weights come from the weighting",
        ));
    }
    let surv = data.followup()?;
    let mut builder = SurvBuf::builder(surv.stop().to_vec(), surv.status().to_vec())
        .n_codes(surv.n_codes())
        .weight(weight);
    if let Some(start) = surv.start() {
        builder = builder.start(start.to_vec());
    }
    if let Some(s) = data.stratum() {
        builder = builder.stratum(s.to_vec());
    }
    if let Some(s) = data.cluster() {
        builder = builder.cluster(s.to_vec());
    }
    if let Some(s) = data.id() {
        builder = builder.id(s.to_vec());
    }
    if let Some(s) = data.istate() {
        builder = builder.istate(s.to_vec());
    }
    builder.build()
}

/// `A^-1 M A^-T` for a square `A` that need not be symmetric, by two solves,
/// averaged with its transpose so that the result is exactly symmetric.
///
/// `epistat_inference::sandwich::sandwich` computes `A^-1 M A^-1`, which is
/// the same only for a symmetric `A`, and the stacked matrix is block lower
/// triangular.
pub(super) fn sandwich_general(
    a: faer::MatRef<'_, f64>,
    m: faer::MatRef<'_, f64>,
) -> Result<Mat<f64>> {
    let solve = |e: epistat_linalg::SolveError| Error::input(format!("the stacked system: {e}"));
    let left = Factor::new(a, &GENERAL)
        .map_err(solve)?
        .solve(m)
        .map_err(solve)?;
    // V' = A^-1 L', with L = A^-1 M.
    let right = Factor::new(a, &GENERAL)
        .map_err(solve)?
        .solve(left.transpose())
        .map_err(solve)?;
    let n = right.nrows();
    Ok(Mat::from_fn(n, n, |i, j| {
        right[(j, i)].midpoint(right[(i, j)])
    }))
}
