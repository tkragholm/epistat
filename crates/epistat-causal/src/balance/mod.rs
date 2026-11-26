//! Weights that balance the covariates directly: entropy balancing and the
//! covariate balancing propensity score, with outcome estimates whose variance
//! accounts for the estimation of the weights.
//!
//! - [`entropy`]: entropy balancing (Hainmueller 2012). Weights closest to the
//!   base weights in Kullback-Leibler distance among those that give the
//!   weighted group exactly the moments of the target group.
//! - [`mod@cbps`]: the covariate balancing propensity score (Imai and Ratkovic
//!   2014) for a binary treatment. The propensity coefficients are chosen so
//!   that the weighted covariate means balance, just identified or
//!   over-identified with the score equations.
//!
//! Both return [`BalanceWeights`]. Its methods estimate an outcome mean in each
//! group, their difference or a weighted linear regression, with the variance of
//! the stacked estimating equations of the weights and the outcome model.
//!
//! # Algorithm
//!
//! Write `s_i` for the sample weight of person `i` and `w_i` for the weight the
//! method gives. The person counts as `s_i w_i`. The weights depend on a
//! parameter `phi` (the multipliers of entropy balancing, the propensity
//! coefficients of CBPS) that solves estimating equations `sum psi_i(phi) = 0` or,
//! for the over-identified CBPS, the first-order condition of a quadratic form in
//! them. In each case `phi` has an influence function `phi_i` with `phi_hat - phi
//! = sum phi_i`, given in the module of the method.
//!
//! The outcome stage is a weighted least squares fit of the outcome `y` on a
//! design `z`, `sum s_i w_i z_i (y_i - z_i' theta) = 0`. Differentiating the
//! equation, with `H = sum s_i w_i z_i z_i'`, `r_i = y_i - z_i' theta` and
//! `dlog_i = d log w_i / d phi`, gives the influence function of `theta`
//!
//! ```text
//! IF_i = H^-1 ( s_i w_i r_i z_i + B phi_i ),    B = sum_j s_j w_j r_j z_j dlog_j'
//! ```
//!
//! and the variance `sum IF_i IF_i'`, with the influences of a cluster summed
//! first and the factor `G / (G - 1)` for `G` clusters. The term `B phi_i` is
//! what the estimation of the weights adds. [`WeightVariance::Fixed`] leaves it
//! out, which treats the weights as given and, for balancing weights, gives a
//! variance that is too large when the model is right, since the weights remove
//! part of the variation of the outcome that they balance.
//!
//! Sample weights are probability weights: the score of person `i` carries the
//! factor `s_i`, so the variance has `s_i^2` where a frequency weight would have
//! `s_i`. Repeated rows are the same estimate as a case weight, but not the same
//! variance.
//!
//! The design `[1 - A, A]` gives the mean outcome in each group, whose
//! difference is the effect of the estimand: the weighted regression with it is
//! the Hajek estimator of each mean. A design that has more columns than the group
//! indicators is also valid. For CBPS with standardised weights, the sum of the weights
//! of a group, which the weights are divided by, is a parameter of the stack.
//!
//! # Memory
//!
//! A [`BalanceWeights`] stores `phi_i` and `dlog_i` for every person, two
//! matrices of `n` rows and as many columns as the weights have parameters (`p +
//! 1` per weighted group for entropy balancing, `p` for CBPS).
//!
//! # References
//!
//! 1. Hainmueller J. Entropy balancing for causal effects: a multivariate
//!    reweighting method to produce balanced samples in observational studies.
//!    Polit Anal. 2012;20(1):25-46.
//! 2. Imai K, Ratkovic M. Covariate balancing propensity score. J R Stat Soc B.
//!    2014;76(1):243-63.
//! 3. Zhao Q, Percival D. Entropy balancing is doubly robust. J Causal Inference.
//!    2017;5(1).
//! 4. Newey WK, McFadden D. Large sample estimation and hypothesis testing. In:
//!    Handbook of Econometrics, vol. 4. 1994. p. 2111-245.

pub mod cbps;
pub mod entropy;

use epistat_linalg::design::independent_columns;
use epistat_linalg::{Attempt, Mat, MatRef, inverse, weighted_gram, weighted_xtz};

use crate::aipw::Target;
use crate::error::{Error, Result};

pub use cbps::{Cbps, CbpsInput, CbpsOptions, Identification, cbps};
pub use entropy::{EntropyInput, EntropyOptions, balance_columns, entropy_balance};

/// What the variance of an outcome estimate takes into account.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WeightVariance {
    /// The weights were estimated: the stacked estimating equations of the
    /// weights and the outcome model.
    #[default]
    Estimated,
    /// The weights are taken as given.
    Fixed,
}

/// The weights of one group of persons.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroupSummary {
    /// Whether the method changed the weights of the group. The weights of the
    /// target group of an ATT or ATC estimate are fixed.
    pub weighted: bool,
    /// The number of persons with positive sample weight.
    pub n: usize,
    /// The sum of `s w`.
    pub sum: f64,
    /// The smallest weight.
    pub min: f64,
    /// The largest weight.
    pub max: f64,
    /// The effective sample size `(sum s w)^2 / sum s w^2`.
    pub effective_sample_size: f64,
}

/// Weights for a binary treatment and what is needed for the variance of the
/// estimates that use them.
#[derive(Debug, Clone)]
pub struct BalanceWeights {
    /// The population the weights refer to.
    pub target: Target,
    /// The weight of each person, to be multiplied with the sample weight. For
    /// entropy balancing the sum of `s w` in a weighted group is the sum of `s`
    /// in the target group. For CBPS with standardised weights it is 1 in each
    /// group, as in the `CBPS` package.
    pub weights: Vec<f64>,
    /// The sample weight of each person, 1 when none was given.
    pub sample_weight: Vec<f64>,
    /// Whether each person is treated.
    pub treatment: Vec<bool>,
    /// The summary of the weights of the controls.
    pub controls: GroupSummary,
    /// The summary of the weights of the treated.
    pub treated: GroupSummary,
    /// The iterations the solver used.
    pub iterations: usize,
    pub(crate) nuisance: Nuisance,
}

/// The influence of the parameters of the weights and the derivative of the
/// log weights with respect to them, one row per person.
#[derive(Debug, Clone)]
pub(crate) struct Nuisance {
    pub influence: Mat<f64>,
    pub dlog: Mat<f64>,
}

/// A weighted linear regression and its variance.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedRegression {
    /// The coefficients, one per design column.
    pub coefficients: Vec<f64>,
    /// The covariance matrix of the coefficients, row-major.
    pub covariance: Vec<f64>,
    /// Which variance it is.
    pub variance: WeightVariance,
}

impl WeightedRegression {
    /// The standard error of coefficient `j`.
    ///
    /// # Panics
    ///
    /// If `j` is not a column of the design.
    #[must_use]
    pub fn se(&self, j: usize) -> f64 {
        let k = self.coefficients.len();
        self.covariance[j * k + j].sqrt()
    }
}

pub use crate::aipw::Estimate;

/// The mean outcome in each group and their difference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeanDifference {
    /// The mean outcome of the treated, weighted to the target.
    pub mean1: Estimate,
    /// The mean outcome of the controls, weighted to the target.
    pub mean0: Estimate,
    /// The difference `mean1 - mean0`.
    pub difference: Estimate,
    /// Which variance it is.
    pub variance: WeightVariance,
}

pub(crate) fn compensated(values: impl IntoIterator<Item = f64>) -> f64 {
    epistat_stats::sum::compensated(values)
}

pub(crate) fn check_len(name: &str, found: usize, n: usize) -> Result<()> {
    if found == n {
        Ok(())
    } else {
        Err(Error::input(format!(
            "{name}: {found} values for {n} persons"
        )))
    }
}

/// The sample weights or ones, checked.
pub(crate) fn resolve_weights(name: &str, given: Option<&[f64]>, n: usize) -> Result<Vec<f64>> {
    match given {
        None => Ok(vec![1.0; n]),
        Some(w) => {
            check_len(name, w.len(), n)?;
            if let Some(i) = w.iter().position(|&x| !(x.is_finite() && x >= 0.0)) {
                return Err(Error::input(format!(
                    "{name}: value {i} is negative or not finite"
                )));
            }
            Ok(w.to_vec())
        }
    }
}

pub(crate) fn summarise(
    weighted: bool,
    arm: bool,
    treatment: &[bool],
    s: &[f64],
    w: &[f64],
) -> GroupSummary {
    let idx = || (0..s.len()).filter(|&i| treatment[i] == arm && s[i] > 0.0);
    let sum = compensated(idx().map(|i| s[i] * w[i]));
    let sq = compensated(idx().map(|i| s[i] * w[i] * w[i]));
    GroupSummary {
        weighted,
        n: idx().count(),
        sum,
        min: idx().map(|i| w[i]).fold(f64::INFINITY, f64::min),
        max: idx().map(|i| w[i]).fold(f64::NEG_INFINITY, f64::max),
        effective_sample_size: if sq > 0.0 { sum * sum / sq } else { 0.0 },
    }
}

/// The covariance of the estimators whose influences are `columns[k][i]`: the
/// sum of products, with the influences of a cluster summed first and the factor
/// `G / (G - 1)`.
fn covariance(columns: &[Vec<f64>], cluster: Option<&[u32]>) -> Result<Vec<f64>> {
    let k = columns.len();
    let n = columns.first().map_or(0, Vec::len);
    let mut out = vec![0.0; k * k];
    let summed: Vec<Vec<f64>>;
    let (data, factor): (&[Vec<f64>], f64) = match cluster {
        None => (columns, 1.0),
        Some(c) => {
            let mut labels: Vec<u32> = c.to_vec();
            labels.sort_unstable();
            labels.dedup();
            let g = labels.len();
            if g < 2 {
                return Err(Error::input("cluster: the variance needs two clusters"));
            }
            let pos = |x: u32| labels.binary_search(&x).unwrap_or(0);
            summed = columns
                .iter()
                .map(|col| {
                    let mut parts: Vec<Vec<f64>> = vec![Vec::new(); g];
                    for (i, &v) in col.iter().enumerate() {
                        parts[pos(c[i])].push(v);
                    }
                    parts
                        .iter()
                        .map(|p| compensated(p.iter().copied()))
                        .collect()
                })
                .collect();
            (&summed, g as f64 / (g as f64 - 1.0))
        }
    };
    let m = data.first().map_or(n, Vec::len);
    for a in 0..k {
        for b in a..k {
            let v = factor * compensated((0..m).map(|i| data[a][i] * data[b][i]));
            if !v.is_finite() {
                return Err(Error::input("the variance is not finite"));
            }
            out[a * k + b] = v;
            out[b * k + a] = v;
        }
    }
    Ok(out)
}

fn make_estimate(value: f64, variance: f64, alpha: f64) -> Result<Estimate> {
    let se = variance.max(0.0).sqrt();
    let level =
        epistat_core::Level::alpha(alpha).map_err(|e| Error::input(format!("alpha: {e}")))?;
    let interval = epistat_inference::interval::normal(value, se, level);
    let (lower, upper) = (
        interval.lower.unwrap_or(f64::NAN),
        interval.upper.unwrap_or(f64::NAN),
    );
    let p_value = (se > 0.0).then(|| 2.0 * epistat_stats::dist::normal::cdf(-(value / se).abs()));
    Ok(Estimate {
        estimate: value,
        se,
        lower,
        upper,
        p_value,
    })
}

struct Stage {
    theta: Vec<f64>,
    /// `influence[j][i]`: influence of coefficient `j` on person `i`.
    influence: Vec<Vec<f64>>,
}

impl BalanceWeights {
    fn n(&self) -> usize {
        self.weights.len()
    }

    fn stage(
        &self,
        outcome: &[f64],
        design: MatRef<'_, f64>,
        variance: WeightVariance,
    ) -> Result<Stage> {
        let n = self.n();
        check_len("outcome", outcome.len(), n)?;
        check_len("design", design.nrows(), n)?;
        if let Some(i) = outcome.iter().position(|y| !y.is_finite()) {
            return Err(Error::input(format!("outcome: value {i} is not finite")));
        }
        let k = design.ncols();
        if k == 0 {
            return Err(Error::input("design: no columns"));
        }
        if (0..n).any(|i| (0..k).any(|j| !design[(i, j)].is_finite())) {
            return Err(Error::input("design: a value is not finite"));
        }
        let total: Vec<f64> = (0..n)
            .map(|i| self.sample_weight[i] * self.weights[i])
            .collect();
        let keep = independent_columns(design, &total, None, 1e-8)
            .map_err(|e| Error::input(format!("design: {e}")))?;
        if let Some(j) = keep.iter().position(|&x| !x) {
            return Err(Error::input(format!(
                "design: column {j} is aliased among the weighted persons"
            )));
        }
        let h = weighted_gram(design, &total).map_err(|e| Error::input(format!("weights: {e}")))?;
        let rhs = weighted_xtz(design, &total, outcome)
            .map_err(|e| Error::input(format!("weights: {e}")))?;
        let hinv = inverse(h.as_ref(), &Attempt::SYMMETRIC)
            .map_err(|e| Error::input(format!("the weighted design is singular: {e}")))?
            .x;
        let theta: Vec<f64> = (0..k)
            .map(|a| compensated((0..k).map(|b| hinv[(a, b)] * rhs[(b, 0)])))
            .collect();
        let resid: Vec<f64> = (0..n)
            .map(|i| outcome[i] - (0..k).map(|j| design[(i, j)] * theta[j]).sum::<f64>())
            .collect();
        let d = self.nuisance.influence.ncols();
        // B = sum_j total_j r_j z_j dlog_j', k x d
        let mut b = Mat::<f64>::zeros(k, d);
        if variance == WeightVariance::Estimated {
            for i in 0..n {
                let c = total[i] * resid[i];
                if c == 0.0 {
                    continue;
                }
                for a in 0..k {
                    let za = c * design[(i, a)];
                    for e in 0..d {
                        b[(a, e)] += za * self.nuisance.dlog[(i, e)];
                    }
                }
            }
        }
        let mut influence = vec![vec![0.0; n]; k];
        let mut v = vec![0.0; k];
        for i in 0..n {
            for (a, va) in v.iter_mut().enumerate() {
                let mut x = total[i] * resid[i] * design[(i, a)];
                if variance == WeightVariance::Estimated {
                    for e in 0..d {
                        x += b[(a, e)] * self.nuisance.influence[(i, e)];
                    }
                }
                *va = x;
            }
            for a in 0..k {
                influence[a][i] = (0..k).map(|c| hinv[(a, c)] * v[c]).sum();
            }
        }
        Ok(Stage { theta, influence })
    }

    /// A weighted least squares regression of `outcome` on `design`, with the
    /// weights `s w`, and its covariance matrix.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] for values of different lengths, a value that is not
    /// finite, a design column that is aliased among the persons with positive
    /// weight, and fewer than two clusters.
    pub fn weighted_regression(
        &self,
        outcome: &[f64],
        design: MatRef<'_, f64>,
        cluster: Option<&[u32]>,
        variance: WeightVariance,
    ) -> Result<WeightedRegression> {
        if let Some(c) = cluster {
            check_len("cluster", c.len(), self.n())?;
        }
        let st = self.stage(outcome, design, variance)?;
        Ok(WeightedRegression {
            covariance: covariance(&st.influence, cluster)?,
            coefficients: st.theta,
            variance,
        })
    }

    /// The mean outcome of the treated and of the controls with the weights, and
    /// their difference, which estimates the effect for the target.
    ///
    /// # Errors
    ///
    /// As [`BalanceWeights::weighted_regression`], and `alpha` outside `(0, 1)`.
    pub fn mean_difference(
        &self,
        outcome: &[f64],
        cluster: Option<&[u32]>,
        variance: WeightVariance,
        alpha: f64,
    ) -> Result<MeanDifference> {
        let n = self.n();
        if !(alpha > 0.0 && alpha < 1.0) {
            return Err(Error::input(format!("alpha: {alpha} is not in (0, 1)")));
        }
        if let Some(c) = cluster {
            check_len("cluster", c.len(), n)?;
        }
        let z = Mat::from_fn(n, 2, |i, j| {
            f64::from(u8::from(self.treatment[i] == (j == 1)))
        });
        let st = self.stage(outcome, z.as_ref(), variance)?;
        let diff: Vec<f64> = (0..n)
            .map(|i| st.influence[1][i] - st.influence[0][i])
            .collect();
        let cols = [st.influence[0].clone(), st.influence[1].clone(), diff];
        let cov = covariance(&cols, cluster)?;
        Ok(MeanDifference {
            mean0: make_estimate(st.theta[0], cov[0], alpha)?,
            mean1: make_estimate(st.theta[1], cov[4], alpha)?,
            difference: make_estimate(st.theta[1] - st.theta[0], cov[8], alpha)?,
            variance,
        })
    }
}
