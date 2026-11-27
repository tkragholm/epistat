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
//! on the dfbeta scale. The variance is `Influence::variance` of `n IF_i`
//! under `ClusterRule::SmallSample`: the influences of a cluster summed first,
//! centred, with the factor `G / (G - 1)` for `G` clusters, every person its
//! own cluster when there are none, which is `sum IF_i IF_i'` times
//! `n / (n - 1)` up to the centring. Fewer than two clusters leave the variances `None`.
//! The term `B phi_i` is
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

use epistat_core::{Contrast, Covariance, Estimand, Level};
use epistat_inference::{ClusterRule, Influence};
use epistat_linalg::design::independent_columns;
use epistat_linalg::{Attempt, Mat, MatRef, inverse, weighted_gram, weighted_xtz};

use crate::error::{Error, Result};
pub(crate) use crate::shared::{case_weights as resolve_weights, check_len, compensated};
use crate::shared::{check_finite, clusters, contrast, unit_codes};

pub use cbps::{Cbps, CbpsInput, CbpsOptions, Identification, cbps};
pub use entropy::{EntropyInput, EntropyOptions, balance_columns, entropy_balance};

/// What the variance of an outcome estimate takes into account.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WeightVariance {
    /// The weights were estimated: the stacked estimating equations of the
    /// weights and the outcome model.
    #[default]
    Estimated,
    /// The weights are taken as given.
    Fixed,
}

/// What an outcome estimate does besides the data.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct OutcomeOptions {
    /// What the variance takes into account. The default is
    /// [`WeightVariance::Estimated`].
    pub variance: WeightVariance,
    /// The level of the confidence limits, by default 95%.
    pub level: Level,
}

/// The weights of one group of persons.
#[non_exhaustive]
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
    /// The effective sample size `(sum s w)^2 / sum s w^2`, `None` when the
    /// group has no weight.
    pub effective_sample_size: Option<f64>,
}

/// Weights for a binary treatment and what is needed for the variance of the
/// estimates that use them.
///
/// The fields are private, since the variance of the outcome estimates relies
/// on the weights being those the method found for the treatment and sample
/// weights it was given.
#[derive(Debug, Clone)]
pub struct BalanceWeights {
    estimand: Estimand,
    weights: Vec<f64>,
    sample_weight: Vec<f64>,
    treatment: Vec<bool>,
    controls: GroupSummary,
    treated: GroupSummary,
    iterations: usize,
    nuisance: Nuisance,
}

/// The influence of the parameters of the weights and the derivative of the
/// log weights with respect to them, one row per person.
#[derive(Debug, Clone)]
pub(crate) struct Nuisance {
    pub influence: Mat<f64>,
    pub dlog: Mat<f64>,
}

/// A weighted linear regression and its variance.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedRegression {
    /// The coefficients, one per design column.
    pub coefficients: Vec<f64>,
    /// The covariance matrix of the coefficients.
    pub covariance: Covariance,
    /// The influence functions of the coefficients, one column each, per
    /// person on the per-unit scale, the units the row numbers.
    pub influence: Influence,
    /// Which variance it is.
    pub variance: WeightVariance,
}

impl WeightedRegression {
    /// The standard error of coefficient `j`, `None` for a column that is not
    /// in the design or an undefined variance.
    #[must_use]
    pub fn se(&self, j: usize) -> Option<f64> {
        if j >= self.coefficients.len() {
            return None;
        }
        self.covariance.get(j, j).map(|v| v.max(0.0).sqrt())
    }
}

/// The mean outcome in each group and their contrast.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct MeanDifference {
    /// The mean outcome of the treated weighted to the target (`mean1`), of
    /// the controls (`mean0`), their difference, which estimates the effect
    /// for the target, and their ratio with the delta-method standard error,
    /// its interval on its own scale and the p-value of a ratio of 1; `None`
    /// when `mean0` is 0.
    pub contrast: Contrast,
    /// The influence functions of `mean1` (column 0) and `mean0` (column 1),
    /// per person on the per-unit scale, the units the row numbers.
    pub influence: Influence,
    /// Which variance it is.
    pub variance: WeightVariance,
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
        effective_sample_size: (sq > 0.0).then(|| sum * sum / sq),
    }
}

struct Stage {
    theta: Vec<f64>,
    /// `influence[j][i]`: influence of coefficient `j` on person `i`, on the
    /// dfbeta scale.
    influence: Vec<Vec<f64>>,
}

impl BalanceWeights {
    #[expect(
        clippy::too_many_arguments,
        reason = "the parts the two methods compute"
    )]
    pub(crate) const fn new(
        estimand: Estimand,
        weights: Vec<f64>,
        sample_weight: Vec<f64>,
        treatment: Vec<bool>,
        controls: GroupSummary,
        treated: GroupSummary,
        iterations: usize,
        nuisance: Nuisance,
    ) -> Self {
        Self {
            estimand,
            weights,
            sample_weight,
            treatment,
            controls,
            treated,
            iterations,
            nuisance,
        }
    }

    /// The population the weights refer to.
    #[must_use]
    pub const fn estimand(&self) -> Estimand {
        self.estimand
    }

    /// The weight of each person, to be multiplied with the sample weight. For
    /// entropy balancing the sum of `s w` in a weighted group is the sum of `s`
    /// in the target group. For CBPS with standardised weights it is 1 in each
    /// group, as in the `CBPS` package.
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// The sample weight of each person, 1 when none was given.
    #[must_use]
    pub fn sample_weight(&self) -> &[f64] {
        &self.sample_weight
    }

    /// Whether each person is treated.
    #[must_use]
    pub fn treatment(&self) -> &[bool] {
        &self.treatment
    }

    /// The summary of the weights of the controls.
    #[must_use]
    pub const fn controls(&self) -> GroupSummary {
        self.controls
    }

    /// The summary of the weights of the treated.
    #[must_use]
    pub const fn treated(&self) -> GroupSummary {
        self.treated
    }

    /// The iterations the solver used.
    #[must_use]
    pub const fn iterations(&self) -> usize {
        self.iterations
    }

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
        check_finite("outcome", outcome)?;
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

    /// The influence function of the columns `columns` of a stage, on the
    /// per-unit scale.
    fn influence(&self, columns: &[&[f64]]) -> Result<Influence> {
        let values: Vec<f64> = columns.iter().flat_map(|c| c.iter().copied()).collect();
        Influence::from_dfbeta(
            values,
            unit_codes(0..self.n())?,
            columns.len(),
            ClusterRule::SmallSample,
        )
        .map_err(|e| Error::input(format!("influence function: {e}")))
    }

    /// A weighted least squares regression of `outcome` on `design`, with the
    /// weights `s w`, and its covariance matrix.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] for values of different lengths, a value that is not
    /// finite and a design column that is aliased among the persons with
    /// positive weight.
    pub fn weighted_regression(
        &self,
        outcome: &[f64],
        design: MatRef<'_, f64>,
        cluster: Option<&[u32]>,
        variance: WeightVariance,
    ) -> Result<WeightedRegression> {
        let clusters = clusters(cluster, self.n())?;
        let st = self.stage(outcome, design, variance)?;
        let columns: Vec<&[f64]> = st.influence.iter().map(Vec::as_slice).collect();
        let influence = self.influence(&columns)?;
        let covariance = influence
            .variance(clusters.as_ref())
            .map_err(|e| Error::input(format!("influence function: {e}")))?;
        Ok(WeightedRegression {
            coefficients: st.theta,
            covariance,
            influence,
            variance,
        })
    }

    /// The mean outcome of the treated and of the controls with the weights,
    /// their difference, which estimates the effect for the target, and their
    /// ratio.
    ///
    /// # Errors
    ///
    /// As [`BalanceWeights::weighted_regression`].
    pub fn mean_difference(
        &self,
        outcome: &[f64],
        cluster: Option<&[u32]>,
        options: &OutcomeOptions,
    ) -> Result<MeanDifference> {
        let n = self.n();
        let clusters = clusters(cluster, n)?;
        // Column 0 of the design is the controls' indicator, column 1 the
        // treated's.
        let z = Mat::from_fn(n, 2, |i, j| {
            f64::from(u8::from(self.treatment[i] == (j == 1)))
        });
        let st = self.stage(outcome, z.as_ref(), options.variance)?;
        let influence = self.influence(&[&st.influence[1], &st.influence[0]])?;
        let contrast = contrast(
            [st.theta[1], st.theta[0]],
            &influence,
            clusters.as_ref(),
            options.level,
        )?;
        Ok(MeanDifference {
            contrast,
            influence,
            variance: options.variance,
        })
    }
}
