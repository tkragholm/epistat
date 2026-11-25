//! Linear models with absorbed fixed effects and their covariances, as
//! `fixest::feols` fits them.
//!
//! # Method
//!
//! [`Feols::fit`] demeans the outcome and the covariates by
//! [`FixedEffects::demean`] and solves the weighted normal equations of the
//! demeaned columns (the Frisch-Waugh-Lovell theorem \[1\]):
//!
//! ```text
//! beta = (X~' W X~)^-1 X~' W y~,      e = y~ - X~ beta
//! ```
//!
//! A covariate is aliased when its demeaned column, after the columns before
//! it are projected out, keeps less than `1e-7` of the weighted norm of the
//! covariate centred at its mean, which is the rule of `Glm::fit` applied to
//! the norms before demeaning; its coefficient is `None`, as `fixest` drops
//! it. The fixed effects of the fit are those of `y - X beta`, which by
//! linearity are those of `y` minus `beta` times those of each covariate.
//!
//! # Covariances
//!
//! With `B = (X~' W X~)^-1`, the scores `s_i = w_i e_i x~_i`, `n` rows and
//! `K` parameters, the covariances of [`Vcov`] are those of
//! `fixest::vcov` 0.14.2 \[2\]:
//!
//! ```text
//! iid        (sum w e^2) / (n - K) B
//! hetero     n / (n - K) B (sum_i s_i s_i') B
//! cluster    a (n - 1) / (n - K) sum_c (-1)^(|c| + 1) b_c B (sum_g s_g s_g') B
//! ```
//!
//! For clusters, `c` runs over the non-empty sets of the cluster variables,
//! clustered by the intersection of their values (one way: the variable
//! itself; two ways: the two and their intersection, subtracted), and `s_g`
//! sums the scores of a cluster. With [`GDf::Min`] `a = G/(G - 1)` for the
//! smallest number of clusters `G` of a single variable and `b_c = 1`; with
//! [`GDf::Conventional`] `a = 1` and `b_c = G_c/(G_c - 1)`. Without
//! [`Ssc::k_adj`] the factors in `n` and `K` are 1, except that the iid
//! covariance divides by `n - 1`; without [`Ssc::g_adj`] `a = b_c = 1`.
//!
//! `K` counts the covariates that are not aliased and the fixed effects. With
//! `G_q` levels of factor `q` and `Q` factors, `fixest` counts `sum G_q - (Q -
//! 1)` fixed effects, and [`KFixef`] and [`Ssc::k_exact`] change that as in
//! `fixest`'s `ssc_compute_K`:
//!
//! - [`KFixef::None`]: no fixed effects.
//! - [`KFixef::Full`], and [`KFixef::Nonnested`] without clusters: all of
//!   them, or with `k_exact` the levels minus the references of
//!   [`FixedEffectValues::references`].
//! - [`KFixef::Nonnested`] with clusters: a factor is nested when each of its
//!   levels lies in one cluster of some cluster variable. When all are,
//!   `K` is the covariates plus 1. Otherwise each nested factor counts
//!   nothing (`G_q - 1` fewer); with `k_exact`, `fixest` subtracts the
//!   references from the full count instead, with the references of the
//!   nested factors set to their levels when more than one is nested. `K` is
//!   at least the covariates plus 1.
//!
//! `fixest` makes a covariance that is not positive definite positive
//! definite by raising its eigenvalues to `1e-16`; this module reports the
//! covariance as computed, with its smallest eigenvalue, and
//! [`Covariance::positive_definite`] gives `fixest`'s repair.
//!
//! # Influence
//!
//! [`FeolsFit::influence`] is `B s_i` for each row, the first-order
//! contribution of the row to `beta - beta_0`. Its sums within clusters give
//! the unadjusted clustered covariance `B (sum_g s_g s_g') B`.
//!
//! # References
//!
//! 1. Lovell MC. A simple proof of the FWL theorem. J Econ Educ.
//!    2008;39(1):88-91. doi:10.3200/JECE.39.1.88-91
//! 2. Bergé L. The R package `fixest` 0.14.2: `feols`, `vcov.fixest`,
//!    `ssc` and `ssc_compute_K`.
//! 3. Cameron AC, Gelbach JB, Miller DL. Robust inference with multiway
//!    clustering. J Bus Econ Stat. 2011;29(2):238-49.
//!    doi:10.1198/jbes.2010.07136

use std::collections::HashMap;

use epistat_inference::Clusters;
use epistat_inference::sandwich::{meat, sandwich_with_inverse};
use epistat_linalg::design::select_columns;
use epistat_linalg::{Attempt, Factor, Mat, MatRef, ROW_BLOCK, matvec_into, row_scaled_gram, xtv};
use rayon::prelude::*;

use crate::absorb::{
    Absorb, AbsorbError, Demeaned, FixedEffectValues, FixedEffects, blocked_total,
};

/// The share of a covariate's centred norm below which it is aliased.
const ALIAS_TOLERANCE: f64 = 1e-7;

/// The controls of a fit.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Feols {
    /// The controls of the demeaning.
    pub absorb: Absorb,
}

/// A linear fit with absorbed fixed effects.
#[derive(Clone, Debug)]
pub struct FeolsFit<'a> {
    fe: &'a FixedEffects,
    /// The coefficients, `None` for an aliased covariate.
    pub coefficients: Vec<Option<f64>>,
    /// The aliased covariates.
    pub dropped_columns: Vec<usize>,
    /// The covariates that are not aliased.
    pub rank: usize,
    /// The residuals, one per row that enters ([`FixedEffects::rows`]).
    pub residuals: Vec<f64>,
    /// `sum w e^2`.
    pub rss: f64,
    /// `rss / (n - K)` with `K` the parameters of [`FeolsFit::n_parameters`],
    /// `None` without residual degrees of freedom.
    pub sigma2: Option<f64>,
    /// `1 - rss / sum w y~^2`, `None` when the outcome does not vary within
    /// the fixed effects.
    pub r2_within: Option<f64>,
    /// The iterations of the demeaning of each covariate and of the outcome,
    /// last.
    pub iterations: Vec<usize>,
    /// The kept covariates.
    columns: Vec<usize>,
    /// The demeaned kept covariates.
    design: Mat<f64>,
    /// `(X~' W X~)^-1`.
    bread: Mat<f64>,
    /// The coefficients of the fixed effects of `y - X beta`, before
    /// normalisation.
    alpha: Vec<Vec<f64>>,
}

/// The covariance to compute.
#[derive(Clone, Copy, Debug)]
pub enum Vcov<'c> {
    /// Homoskedastic.
    Iid,
    /// Heteroskedasticity-robust (HC1).
    Hetero,
    /// Clustered by one or more variables, each with one code per input row.
    Cluster(&'c [&'c [u32]]),
}

/// How the fixed effects enter `K`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KFixef {
    /// Not at all.
    None,
    /// All of them.
    Full,
    /// Those not nested in a cluster variable.
    Nonnested,
}

/// The number of clusters in the factor `G/(G - 1)` with several cluster
/// variables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GDf {
    /// The smallest number of clusters, for the whole covariance.
    Min,
    /// Each term's own.
    Conventional,
}

/// The small-sample correction, as `fixest::ssc`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ssc {
    /// The factor `(n - 1)/(n - K)`.
    pub k_adj: bool,
    /// How the fixed effects enter `K`.
    pub k_fixef: KFixef,
    /// Whether redundant fixed effects are found from the references.
    pub k_exact: bool,
    /// The factor `G/(G - 1)`.
    pub g_adj: bool,
    /// Which `G`.
    pub g_df: GDf,
}

impl Ssc {
    /// The default of `fixest::ssc()`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            k_adj: true,
            k_fixef: KFixef::Nonnested,
            k_exact: false,
            g_adj: true,
            g_df: GDf::Min,
        }
    }
}

impl Default for Ssc {
    fn default() -> Self {
        Self::new()
    }
}

/// A covariance of the coefficients.
#[derive(Clone, Debug)]
pub struct Covariance {
    /// Row-major `p x p`, `None` for aliased covariates and everywhere when a
    /// factor of the correction does not exist.
    pub matrix: Vec<Option<f64>>,
    /// The `K` of the correction.
    pub df_k: usize,
    /// The smallest number of clusters of a single cluster variable.
    pub min_clusters: Option<usize>,
    /// The smallest eigenvalue of the matrix over the covariates that are not
    /// aliased.
    pub min_eigenvalue: Option<f64>,
    dimension: usize,
}

impl Covariance {
    /// The matrix with eigenvalues below `1e-16` raised to `1e-16`, as
    /// `fixest` repairs a covariance that is not positive definite; the
    /// matrix itself when it is.
    #[must_use]
    pub fn positive_definite(&self) -> Vec<Option<f64>> {
        let p = self.dimension;
        let kept: Vec<usize> = (0..p)
            .filter(|&j| self.matrix[j * p + j].is_some())
            .collect();
        let k = kept.len();
        if k == 0 || self.min_eigenvalue.is_none_or(|m| m > 0.0) {
            return self.matrix.clone();
        }
        let a: Vec<f64> = (0..k * k)
            .map(|ab| self.matrix[kept[ab / k] * p + kept[ab % k]].unwrap_or(0.0))
            .collect();
        let (values, vectors) = epistat_linalg::eigen::symmetric_eigen_jacobi(&a, k);
        let mut out = self.matrix.clone();
        for a in 0..k {
            for b in 0..k {
                let mut total = 0.0;
                for (m, &v) in values.iter().enumerate() {
                    total += vectors[a * k + m] * v.max(1e-16) * vectors[b * k + m];
                }
                out[kept[a] * p + kept[b]] = Some(total);
            }
        }
        out
    }
}

/// The columns of `sqrt(W) X~` that are not aliased: Gram-Schmidt with a
/// second pass in column order, aliased where the norm left is below
/// [`ALIAS_TOLERANCE`] times `reference[j]`. Sums run over blocks of rows
/// added in block order.
fn independent_columns(x: MatRef<'_, f64>, w: Option<&[f64]>, reference: &[f64]) -> Vec<bool> {
    let n = x.nrows();
    let dot = |a: &[f64], b: &[f64]| blocked_total(n, |i| a[i] * b[i]);
    let mut basis: Vec<Vec<f64>> = Vec::new();
    (0..x.ncols())
        .map(|j| {
            let mut u: Vec<f64> = (0..n)
                .into_par_iter()
                .map(|i| w.map_or(1.0, |w| w[i].sqrt()) * x[(i, j)])
                .collect();
            for _ in 0..2 {
                for q in &basis {
                    let d = dot(q, &u);
                    u.par_iter_mut()
                        .zip(q.par_iter())
                        .for_each(|(v, a)| *v -= d * a);
                }
            }
            let left = dot(&u, &u).sqrt();
            if reference[j] == 0.0 || left < ALIAS_TOLERANCE * reference[j] {
                return false;
            }
            u.par_iter_mut().for_each(|v| *v /= left);
            basis.push(u);
            true
        })
        .collect()
}

impl Feols {
    /// Fits `y` on the columns of `x` with the fixed effects `fe`. `y` and
    /// `x` have one row per input row of `fe`; only the rows that enter are
    /// read.
    ///
    /// # Errors
    ///
    /// [`AbsorbError::Length`] for inputs of the wrong length, the errors of
    /// [`FixedEffects::demean`], and [`AbsorbError::Glm`] when the normal
    /// equations cannot be solved.
    pub fn fit<'a>(
        &self,
        y: &[f64],
        x: MatRef<'_, f64>,
        fe: &'a FixedEffects,
    ) -> Result<FeolsFit<'a>, AbsorbError> {
        let n_input = fe.n_input();
        for (what, found) in [("y", y.len()), ("x", x.nrows())] {
            if found != n_input {
                return Err(AbsorbError::Length {
                    what,
                    expected: n_input,
                    found,
                });
            }
        }
        let p = x.ncols();
        // Columns of a column-major matrix are read in place; others are
        // copied.
        let copies: Vec<Option<Vec<f64>>> = (0..p)
            .map(|j| {
                x.col(j)
                    .try_as_col_major()
                    .is_none()
                    .then(|| (0..n_input).map(|i| x[(i, j)]).collect())
            })
            .collect();
        let mut columns: Vec<&[f64]> = (0..p)
            .map(|j| {
                copies[j].as_deref().unwrap_or_else(|| {
                    x.col(j)
                        .try_as_col_major()
                        .map_or(&[][..], |column| column.as_slice())
                })
            })
            .collect();
        columns.push(y);
        let demeaned = fe.demean(&columns, &self.absorb)?;
        drop(copies);
        Self::solve(fe, demeaned, p)
    }

    /// The fit from the demeaned columns: covariates first, outcome last.
    fn solve(fe: &FixedEffects, demeaned: Demeaned, p: usize) -> Result<FeolsFit<'_>, AbsorbError> {
        let Demeaned {
            columns: mut matrix,
            scale,
            iterations,
            coefficients: column_coefficients,
        } = demeaned;
        let n = fe.n_rows();
        let w = fe.weights();
        let total_weight = w.map_or(n as f64, |w| blocked_total(w.len(), |k| w[k]));
        let y_tilde = matrix.col_as_slice(p).to_vec();
        let x_tilde = matrix.as_ref().subcols(0, p);
        let reference: Vec<f64> = scale[..p].iter().map(|s| s * total_weight.sqrt()).collect();
        let keep = independent_columns(x_tilde, w, &reference);
        let columns: Vec<usize> = (0..p).filter(|&j| keep[j]).collect();
        let rank = columns.len();
        let dropped_columns: Vec<usize> = (0..p).filter(|&j| !keep[j]).collect();
        let design = if rank == p {
            // The demeaned covariates are the design: drop the outcome.
            matrix.truncate(n, p);
            matrix
        } else {
            select_columns(x_tilde, &keep)
        };
        let root: Vec<f64> = w.map_or_else(Vec::new, |w| w.iter().map(|v| v.sqrt()).collect());
        let gram = row_scaled_gram(design.as_ref(), |k| root.get(k).copied().unwrap_or(1.0));
        let weighted_y: Vec<f64> = (0..n)
            .map(|k| w.map_or(1.0, |w| w[k]) * y_tilde[k])
            .collect();
        let rhs = xtv(design.as_ref(), &weighted_y);
        let (bread, beta) = if rank == 0 {
            (Mat::<f64>::zeros(0, 0), Mat::<f64>::zeros(0, 1))
        } else {
            let factor =
                Factor::new(gram.as_ref(), &Attempt::SYMMETRIC).map_err(crate::GlmError::from)?;
            let bread = factor
                .solve(Mat::<f64>::identity(rank, rank).as_ref())
                .map_err(crate::GlmError::from)?;
            let beta = factor.solve(rhs.as_ref()).map_err(crate::GlmError::from)?;
            (bread, beta)
        };
        let mut fitted = vec![0.0; n];
        if rank > 0 {
            matvec_into(&mut fitted, design.as_ref(), beta.as_ref());
        }
        let residuals: Vec<f64> = y_tilde.iter().zip(&fitted).map(|(y, f)| y - f).collect();
        let mut coefficients = vec![None; p];
        for (a, &j) in columns.iter().enumerate() {
            coefficients[j] = Some(beta[(a, 0)]);
        }
        let weight = |k: usize| w.map_or(1.0, |w| w[k]);
        let rss = blocked_total(n, |k| weight(k) * residuals[k] * residuals[k]);
        let tss = blocked_total(n, |k| weight(k) * y_tilde[k] * y_tilde[k]);
        let mut alpha = column_coefficients[p].clone();
        for (a, &j) in columns.iter().enumerate() {
            let b = beta[(a, 0)];
            for (target, source) in alpha.iter_mut().zip(&column_coefficients[j]) {
                for (t, s) in target.iter_mut().zip(source) {
                    *t -= b * s;
                }
            }
        }
        let mut fit = FeolsFit {
            fe,
            coefficients,
            dropped_columns,
            rank,
            residuals,
            rss,
            sigma2: None,
            r2_within: (tss > 0.0).then(|| 1.0 - rss / tss),
            iterations,
            columns,
            design,
            bread,
            alpha,
        };
        let k = fit.n_parameters();
        fit.sigma2 = (n > k).then(|| rss / (n - k) as f64);
        Ok(fit)
    }
}

/// The scores `w e x~` summed within the clusters of `of`.
fn cluster_meat(
    design: MatRef<'_, f64>,
    score: impl Fn(usize) -> f64 + Sync,
    of: &Clusters,
) -> Result<Mat<f64>, AbsorbError> {
    meat(design, score, Some(of)).map_err(|e| AbsorbError::Glm(crate::GlmError::from(e)))
}

impl FeolsFit<'_> {
    /// The number of parameters `fixest` reports as `nparams`: the covariates
    /// that are not aliased and `sum G_q - (Q - 1)` fixed effects.
    #[must_use]
    pub fn n_parameters(&self) -> usize {
        let levels: usize = self.fe.levels().iter().sum();
        self.rank + levels + 1 - self.fe.n_factors()
    }

    /// The demeaned covariates that are not aliased, one row per row that
    /// enters.
    #[must_use]
    pub fn design(&self) -> MatRef<'_, f64> {
        self.design.as_ref()
    }

    /// `(X~' W X~)^-1` over the covariates that are not aliased.
    #[must_use]
    pub fn bread(&self) -> MatRef<'_, f64> {
        self.bread.as_ref()
    }

    /// The covariates that are not aliased, in order.
    #[must_use]
    pub fn kept_columns(&self) -> &[usize] {
        &self.columns
    }

    /// The fixed effects of the fit, normalised as `fixest::fixef` does.
    #[must_use]
    pub fn fixed_effects(&self) -> FixedEffectValues {
        self.fe.normalise(self.alpha.clone())
    }

    /// `B w_i e_i x~_i` for each row that enters: one row per row, one column
    /// per covariate that is not aliased.
    #[must_use]
    pub fn influence(&self) -> Mat<f64> {
        let n = self.residuals.len();
        let k = self.rank;
        let w = self.fe.weights();
        let mut out = Mat::<f64>::zeros(n, k);
        for a in 0..k {
            let column = out.col_as_slice_mut(a);
            column
                .par_chunks_mut(ROW_BLOCK)
                .enumerate()
                .for_each(|(block, chunk)| {
                    for (r, o) in chunk.iter_mut().enumerate() {
                        let i = block * ROW_BLOCK + r;
                        let s = w.map_or(1.0, |w| w[i]) * self.residuals[i];
                        let mut total = 0.0;
                        for b in 0..k {
                            total += self.bread[(a, b)] * self.design[(i, b)];
                        }
                        *o = total * s;
                    }
                });
        }
        out
    }

    /// The influence summed within groups: `B sum_{i in g} w_i e_i x~_i` for
    /// each group `g` of `groups` (one code below `count` per row that enters),
    /// one row per group, one column per covariate that is not aliased. The
    /// scores are added in row order and then multiplied by `B`, so no
    /// matrix of one row per row is formed.
    ///
    /// # Errors
    ///
    /// [`AbsorbError::Length`] when `groups` does not have one code per row
    /// that enters, and [`AbsorbError::Invalid`] for a code not below `count`.
    pub fn influence_sums(&self, groups: &[u32], count: usize) -> Result<Mat<f64>, AbsorbError> {
        let n = self.residuals.len();
        if groups.len() != n {
            return Err(AbsorbError::Length {
                what: "groups",
                expected: n,
                found: groups.len(),
            });
        }
        if groups.iter().any(|&g| g as usize >= count) {
            return Err(AbsorbError::Invalid("a group code is not below the count"));
        }
        let k = self.rank;
        let w = self.fe.weights();
        let mut scores = vec![0.0; count * k];
        for (i, &g) in groups.iter().enumerate() {
            let s = w.map_or(1.0, |w| w[i]) * self.residuals[i];
            let row = &mut scores[g as usize * k..(g as usize + 1) * k];
            for (a, r) in row.iter_mut().enumerate() {
                *r += s * self.design[(i, a)];
            }
        }
        Ok(Mat::from_fn(count, k, |g, a| {
            (0..k)
                .map(|b| self.bread[(a, b)] * scores[g * k + b])
                .sum::<f64>()
        }))
    }

    /// The codes of `codes` on the rows that enter, as clusters.
    fn clusters_of(&self, codes: &[u32]) -> Clusters {
        let kept: Vec<u32> = self.fe.rows().iter().map(|&i| codes[i]).collect();
        Clusters::from_labels(&kept)
    }

    /// Whether each factor is nested in one of the cluster variables.
    fn nested(&self, ways: &[&[u32]]) -> Vec<bool> {
        let rows = self.fe.rows();
        (0..self.fe.n_factors())
            .map(|q| {
                let codes = self.fe.codes(q);
                let levels = self.fe.levels()[q];
                ways.iter().any(|way| {
                    let mut first = vec![u32::MAX; levels];
                    let mut seen = vec![false; levels];
                    rows.iter().zip(codes).all(|(&i, &g)| {
                        let g = g as usize;
                        if seen[g] {
                            first[g] == way[i]
                        } else {
                            seen[g] = true;
                            first[g] = way[i];
                            true
                        }
                    })
                })
            })
            .collect()
    }

    /// The `K` of the correction, as `fixest`'s `ssc_compute_K`.
    fn df_k(&self, ssc: Ssc, ways: Option<&[&[u32]]>) -> usize {
        let levels = self.fe.levels();
        let q = levels.len();
        let total: usize = levels.iter().sum();
        let nparams = self.n_parameters();
        let references = || {
            if q >= 2 {
                self.fixed_effects().references
            } else {
                vec![0; q]
            }
        };
        match (ssc.k_fixef, ways) {
            (KFixef::None, _) => nparams - (total + 1 - q),
            (KFixef::Full, _) | (KFixef::Nonnested, None) => {
                if ssc.k_exact && q >= 2 {
                    nparams + (q - 1) - references().iter().sum::<usize>()
                } else {
                    nparams
                }
            }
            (KFixef::Nonnested, Some(ways)) => {
                let nested = self.nested(ways);
                let k = if nested.iter().all(|&b| b) {
                    nparams + 1 - (total + 1 - q)
                } else if ssc.k_exact && q >= 2 {
                    let mut refs = references();
                    if nested.iter().filter(|&&b| b).count() > 1 {
                        for (r, (&b, &g)) in refs.iter_mut().zip(nested.iter().zip(&levels)) {
                            if b {
                                *r = g;
                            }
                        }
                    }
                    nparams.saturating_sub(refs.iter().sum::<usize>())
                } else {
                    let nested_levels: usize = nested
                        .iter()
                        .zip(&levels)
                        .filter(|&(&b, _)| b)
                        .map(|(_, &g)| g - 1)
                        .sum();
                    nparams - nested_levels
                };
                k.max(self.rank + 1)
            }
        }
    }

    /// The covariance `kind` with the correction `ssc`.
    ///
    /// # Errors
    ///
    /// [`AbsorbError::Length`] for cluster codes of the wrong length,
    /// [`AbsorbError::Invalid`] for no cluster variables, and
    /// [`AbsorbError::Glm`] when a cluster sum cannot be formed.
    pub fn vcov(&self, kind: Vcov<'_>, ssc: Ssc) -> Result<Covariance, AbsorbError> {
        let n = self.residuals.len();
        let p = self.coefficients.len();
        let w = self.fe.weights();
        let score = |k: usize| w.map_or(1.0, |w| w[k]) * self.residuals[k];
        let ways = match kind {
            Vcov::Cluster(ways) => {
                if ways.is_empty() {
                    return Err(AbsorbError::Invalid("no cluster variables"));
                }
                for way in ways {
                    if way.len() != self.fe.n_input() {
                        return Err(AbsorbError::Length {
                            what: "cluster codes",
                            expected: self.fe.n_input(),
                            found: way.len(),
                        });
                    }
                }
                Some(ways)
            }
            Vcov::Iid | Vcov::Hetero => None,
        };
        let k = self.df_k(ssc, ways);
        let design = self.design.as_ref();
        let mut min_clusters = None;
        let matrix: Option<Mat<f64>> = match kind {
            Vcov::Iid => {
                let denominator = if ssc.k_adj {
                    n.checked_sub(k)
                } else {
                    n.checked_sub(1)
                };
                denominator.filter(|&d| d > 0).map(|d| {
                    let s = self.rss / d as f64;
                    Mat::from_fn(self.rank, self.rank, |a, b| s * self.bread[(a, b)])
                })
            }
            Vcov::Hetero => {
                let middle = meat(design, score, None)
                    .map_err(|e| AbsorbError::Glm(crate::GlmError::from(e)))?;
                let v = sandwich_with_inverse(self.bread.as_ref(), middle.as_ref());
                let factor = if ssc.k_adj {
                    (n > k).then(|| n as f64 / (n - k) as f64)
                } else {
                    Some(1.0)
                };
                factor.map(|f| Mat::from_fn(self.rank, self.rank, |a, b| f * v[(a, b)]))
            }
            Vcov::Cluster(ways) => {
                let singles: Vec<Clusters> = ways.iter().map(|way| self.clusters_of(way)).collect();
                let g_min = singles.iter().map(Clusters::count).min().unwrap_or(0);
                min_clusters = Some(g_min);
                let mut total = Mat::<f64>::zeros(self.rank, self.rank);
                let mut defined = true;
                for subset in 1..(1usize << ways.len()) {
                    let members: Vec<usize> =
                        (0..ways.len()).filter(|&j| subset >> j & 1 == 1).collect();
                    let clusters = if members.len() == 1 {
                        singles[members[0]].clone()
                    } else {
                        intersection(self.fe.rows(), ways, &members)
                    };
                    let middle = cluster_meat(design, score, &clusters)?;
                    let v = sandwich_with_inverse(self.bread.as_ref(), middle.as_ref());
                    let g = clusters.count();
                    let factor = if ssc.g_adj && ssc.g_df == GDf::Conventional {
                        if g < 2 {
                            defined = false;
                            1.0
                        } else {
                            g as f64 / (g - 1) as f64
                        }
                    } else {
                        1.0
                    };
                    let sign = if members.len() % 2 == 1 { 1.0 } else { -1.0 };
                    for a in 0..self.rank {
                        for b in 0..self.rank {
                            total[(a, b)] += sign * factor * v[(a, b)];
                        }
                    }
                }
                let mut factor = 1.0;
                if ssc.g_adj && ssc.g_df == GDf::Min {
                    if g_min < 2 {
                        defined = false;
                    } else {
                        factor *= g_min as f64 / (g_min - 1) as f64;
                    }
                }
                if ssc.k_adj {
                    if n > k {
                        factor *= (n - 1) as f64 / (n - k) as f64;
                    } else {
                        defined = false;
                    }
                }
                defined.then(|| Mat::from_fn(self.rank, self.rank, |a, b| factor * total[(a, b)]))
            }
        };
        let mut out = vec![None; p * p];
        let mut min_eigenvalue = None;
        if let Some(v) = matrix {
            for (a, &i) in self.columns.iter().enumerate() {
                for (b, &j) in self.columns.iter().enumerate() {
                    out[i * p + j] = Some(v[(a, b)]);
                }
            }
            if self.rank > 0 {
                let flat: Vec<f64> = (0..self.rank * self.rank)
                    .map(|ab| v[(ab / self.rank, ab % self.rank)])
                    .collect();
                let (values, _) = epistat_linalg::eigen::symmetric_eigen_jacobi(&flat, self.rank);
                min_eigenvalue = values.iter().copied().reduce(f64::min);
            }
        }
        Ok(Covariance {
            matrix: out,
            df_k: k,
            min_clusters,
            min_eigenvalue,
            dimension: p,
        })
    }
}

/// Clusters by the intersection of the values of the cluster variables
/// `members`, on the rows `rows`.
fn intersection(rows: &[usize], ways: &[&[u32]], members: &[usize]) -> Clusters {
    if members.len() == 2 {
        let (a, b) = (ways[members[0]], ways[members[1]]);
        let labels: Vec<u64> = rows
            .iter()
            .map(|&i| u64::from(a[i]) << 32 | u64::from(b[i]))
            .collect();
        return Clusters::from_labels(&labels);
    }
    let mut number: HashMap<Vec<u32>, usize> = HashMap::new();
    let labels: Vec<usize> = rows
        .iter()
        .map(|&i| {
            let key: Vec<u32> = members.iter().map(|&j| ways[j][i]).collect();
            let next = number.len();
            *number.entry(key).or_insert(next)
        })
        .collect();
    Clusters::from_labels(&labels)
}
