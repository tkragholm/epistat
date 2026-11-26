//! Generalised estimating equations for clustered outcomes, as
//! `geepack::geeglm` and `gee::gee` fit them.
//!
//! The mean model is `g(E y_ij) = x_ij' β` for row `j` of cluster `i`, with
//! the variance function `v` of a [`Family`] and prior weights `w`. The
//! working covariance of cluster `i` is
//!
//! ```text
//! V_i = φ A_i^{1/2} R(α) A_i^{1/2}       A_i = diag(v(μ_ij) / w_ij)
//! ```
//!
//! with a working correlation `R(α)` of one of four forms \[1, 2\]: independence,
//! exchangeable (`R_jk = α`), AR(1) (`R_jk = α^{|t_j - t_k|}` at the waves
//! `t`) and unstructured (a free correlation for each pair of waves). The
//! estimate solves `Σ_i D_i' V_i^{-1} (y_i - μ_i) = 0` with `D_i = ∂μ_i/∂β`,
//! and its robust covariance is the sandwich `B^{-1} M B^{-1}` with
//! `B = Σ_i D_i' V_i^{-1} D_i` and `M = Σ_i s_i s_i'`, `s_i = D_i' V_i^{-1}
//! (y_i - μ_i)`, which is valid when `R` is wrong. The naive covariance is
//! `B^{-1}`. The Mancl and DeRouen covariance \[5\] replaces the residual of
//! cluster `i` by `(I - H_i)^{-1} (y_i - μ_i)` with `H_i = D_i B^{-1} D_i'
//! V_i^{-1}`, which removes most of the bias of the sandwich when there are few
//! clusters.
//!
//! # Algorithm
//!
//! With the Pearson residual `s_ij = (y_ij - μ_ij) / sqrt(v(μ_ij))`, the
//! iteration starts from the independence fit of [`Glm`] and repeats
//!
//! 1. estimate `φ` and `α` from the residuals at the current `β`,
//! 2. take the Fisher scoring step `β += B^{-1} Σ_i D_i' V_i^{-1} e_i`,
//!
//! until the largest of `|Δβ_j| / (1 + |β_j|)`, `|Δα_k| / (1 + |α_k|)` and
//! `|Δφ| / (1 + φ)` between two iterations is at most
//! [`Gee::tolerance`]. `φ` cancels in `β` and in the sandwich and enters only
//! the naive covariance and the scale. The estimate is not converged if this
//! has not happened after [`Gee::max_iterations`] iterations. `geeglm` stops
//! at a change of `1e-4` by default, so its output at the default settings
//! differs from the fixed point by about that much.
//!
//! ## Association parameters, two estimators
//!
//! [`Association::YanFine`] solves the equations of `geepack` (Yan and Fine
//! \[4\] with the identity working covariance for the products of residuals).
//! Writing `c_jk = w_j w_k` for a pair of rows of one cluster, `N` for the
//! number of rows and `p` for the rank of the design:
//!
//! ```text
//! φ = Σ w s² / Σ w
//! exchangeable   α = Σ_pairs c s_j s_k / (φ Σ_pairs c)
//! AR(1)          Σ_pairs c d α^{d-1} (s_j s_k / φ - α^d) = 0,   d = |t_j - t_k|
//! unstructured   R_jk = Σ_{pairs of waves j, k} c s_j s_k / (φ Σ c)
//! ```
//!
//! The AR(1) equation uses all pairs, not only adjacent waves, and is solved
//! for the root in `(-1, 1)` nearest the adjacent-wave estimate. The pair
//! weight `w_j w_k` is what `geeglm` does with prior weights, measured on
//! unbalanced data with `w` away from 1.
//!
//! [`Association::Moment`] is the estimator of the `gee` package, the moment
//! estimator of Liang and Zeger \[1\] with the degrees of freedom corrections of
//! that package. It does not accept prior weights. For the exchangeable
//! structure `φ = Σ s² / (N - p)` and `α = Σ_pairs s_j s_k / (φ (n_pairs - p))`.
//! For AR(1) it is the adjacent-wave estimate `α = Σ_pairs s_j s_k / (φ_N
//! n_1)`, with `φ_N = Σ s² / N`, and the unstructured correlations are
//! `Σ s_j s_k / (φ_N N_jk)`, where `gee` applies no correction. The reported
//! scale is `Σ s² / (N - p)`. These asymmetries are those of the package and
//! were measured on `ohio`.
//!
//! The two estimators differ by about `p / n_pairs` times `α`, which is
//! `1.6e-4` in `α` and `5e-6` in `β` on `ohio` with exchangeable correlation.
//!
//! ## Working correlation inverses
//!
//! `R^{-1}` is applied in `O(m p)` per cluster of `m` rows without forming
//! `R`: for the exchangeable structure by Sherman and Morrison,
//! `R^{-1} = (I - α / (1 + (m - 1) α) 11') / (1 - α)`, and for AR(1) by its
//! tridiagonal inverse, which holds for rows at any waves because the
//! correlation is Markov: with `ρ_j = α^{t_{j+1} - t_j}` the diagonal is
//! `[j = 1] + [j > 1] / (1 - ρ_{j-1}²) + [j < m] ρ_j² / (1 - ρ_j²)`, and the
//! off-diagonal is `-ρ_j / (1 - ρ_j²)`. For the unstructured structure the
//! inverse of the submatrix of `R` at the waves of the cluster is formed once
//! for each distinct set of waves.
//!
//! # Errors and edge cases
//!
//! A working correlation that is not positive definite is an error and not a
//! repair: the exchangeable `α` must lie in `(-1 / (m_max - 1), 1)`, the AR(1)
//! `α` in `(-1, 1)`, and the unstructured matrix must have a Cholesky factor.
//! A structure with no pair of rows to estimate from, an unstructured pair of
//! waves that never meet in a cluster, and an AR(1) equation with no root are
//! errors. Duplicate waves in a cluster and an unstructured fit without waves
//! are errors. The covariance of an aliased column is `None`, as in [`Glm`],
//! and the Mancl and DeRouen covariance is `None` when some `I - H_i` is
//! singular. For the unstructured structure the waves are ranked among the
//! distinct waves present.
//!
//! # Determinism
//!
//! Rows are sorted by cluster and wave. Clusters are summed in blocks of
//! 128 clusters, each block in order, and the blocks in order, so results
//! do not depend on the number of threads.
//!
//! # Cost
//!
//! `O(N p²)` per iteration for the independence, exchangeable and AR(1)
//! structures with waves that are adjacent, and `O(Σ m_i² )` for the
//! association of AR(1) and unstructured structures, `O(Σ m_i³)` for the
//! Mancl and DeRouen covariance.
//!
//! The fit is checked against `geepack::geeglm` 1.3.13 and `gee::gee` 4.13-30.
//! `geeglm` stops with a segmentation fault on the unstructured structure
//! when the waves of the clusters have gaps, so that case is checked against
//! an R transcription of the equations that agrees with `geeglm` where both
//! run.
//!
//! # References
//!
//! 1. Liang KY, Zeger SL. Longitudinal data analysis using generalized linear
//!    models. Biometrika. 1986;73(1):13-22.
//! 2. Zeger SL, Liang KY. Longitudinal data analysis for discrete and
//!    continuous outcomes. Biometrics. 1986;42(1):121-30.
//! 3. Halekoh U, Højsgaard S, Yan J. The R package geepack for generalized
//!    estimating equations. J Stat Softw. 2006;15(2):1-11.
//! 4. Yan J, Fine J. Estimating equations for association structures. Stat
//!    Med. 2004;23(6):859-74.
//! 5. Mancl LA, DeRouen TA. A covariance estimator for GEE with improved
//!    small-sample properties. Biometrics. 2001;57(1):126-34.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "waves, cluster sizes and lags are small whole numbers"
)]

use epistat_core::Covariance;
use epistat_inference::sandwich::sandwich_with_inverse;
use epistat_linalg::{Attempt, Factor, Mat, MatRef, Method, Regularisation};
use rayon::prelude::*;

use crate::columns::linear_predictor;
use crate::fit::covariance;
use crate::{Error, Family, GeeData, Glm, GlmData, Link};

/// The number of clusters summed together before the blocks are added in
/// order.
pub(crate) const BLOCK: usize = 128;

/// The working correlation.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Correlation {
    /// Uncorrelated rows: the estimate is that of [`Glm`] with a clustered
    /// sandwich covariance.
    Independence,
    /// One correlation `α` between any two rows of a cluster.
    Exchangeable,
    /// `α^{|t_j - t_k|}` at the waves `t`.
    Ar1,
    /// A free correlation for each pair of waves.
    Unstructured,
}

/// How the association parameters and the scale are estimated.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Association {
    /// The equations of `geepack::geeglm`.
    YanFine,
    /// The moment estimators of the `gee` package.
    Moment,
}

/// The robust covariance of the coefficients.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RobustVariance {
    /// The sandwich of Liang and Zeger.
    Sandwich,
    /// The sandwich with the bias correction of Mancl and DeRouen.
    ManclDeRouen,
}

/// A GEE model and its fitting controls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gee {
    /// The distribution of the response.
    pub family: Family,
    /// The link between the mean and the linear predictor.
    pub link: Link,
    /// The working correlation.
    pub correlation: Correlation,
    /// The estimator of the association parameters.
    pub association: Association,
    /// The robust covariance reported in [`GeeFit::covariance`].
    pub variance: RobustVariance,
    /// The largest relative change of a parameter at which the fit has
    /// converged.
    pub tolerance: f64,
    /// The most iterations.
    pub max_iterations: usize,
    /// A scale held fixed instead of estimated, such as 1 for the Poisson
    /// and binomial families.
    pub fix_scale: Option<f64>,
}

impl Gee {
    /// A model with the estimators of `geepack`, the sandwich covariance,
    /// tolerance `1e-8`, 100 iterations and the scale estimated.
    #[must_use]
    pub const fn new(family: Family, link: Link, correlation: Correlation) -> Self {
        Self {
            family,
            link,
            correlation,
            association: Association::YanFine,
            variance: RobustVariance::Sandwich,
            tolerance: 1e-8,
            max_iterations: 100,
            fix_scale: None,
        }
    }
}

/// A fitted GEE.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct GeeFit {
    /// The coefficients, `None` for an aliased column.
    pub coefficients: Vec<Option<f64>>,
    /// The robust covariance of the coefficients, `p` square, `None` for an
    /// aliased column. All `None` when the Mancl and DeRouen correction has a
    /// singular cluster.
    pub covariance: Covariance,
    /// The model-based covariance `φ B^{-1}`, `p` square.
    pub naive_covariance: Covariance,
    /// The association parameters: none for independence, one for exchangeable
    /// and AR(1), and for unstructured the correlations of the pairs of wave
    /// ranks `(0, 1), (0, 2), ..., (1, 2), ...`.
    pub alpha: Vec<f64>,
    /// The scale `φ`.
    pub scale: f64,
    /// The fitted means, in the order of the input rows.
    pub fitted: Vec<f64>,
    /// Whether the change fell below the tolerance.
    pub converged: bool,
    /// The number of iterations.
    pub iterations: usize,
    /// The number of clusters.
    pub n_clusters: usize,
    /// The number of identifiable coefficients.
    pub rank: usize,
}

impl GeeFit {
    /// The robust standard errors, `None` for an aliased coefficient or where
    /// the covariance is undefined.
    #[must_use]
    pub fn standard_errors(&self) -> Vec<Option<f64>> {
        self.covariance.standard_errors()
    }
}

/// Rows sorted by cluster and wave, with the identifiable columns.
struct Panel {
    n: usize,
    /// The input row at each sorted position.
    order: Vec<usize>,
    q: usize,
    /// Column-major, `n * q`.
    x: Vec<f64>,
    y: Vec<f64>,
    w: Vec<f64>,
    offset: Vec<f64>,
    /// The wave of each row, ranks for the unstructured structure.
    wave: Vec<u32>,
    /// Cluster `i` holds rows `starts[i]..starts[i + 1]`.
    starts: Vec<usize>,
    /// The number of distinct waves, for the unstructured structure.
    wave_count: usize,
    /// The widest span of waves within a cluster, plus one.
    lags: usize,
    /// For the unstructured structure, the id of each cluster's set of waves
    /// and the sets.
    pattern_of: Vec<usize>,
    patterns: Vec<Vec<u32>>,
}

/// Quantities at one coefficient vector, in the order of the sorted rows.
struct Rows {
    mu: Vec<f64>,
    /// `s = (y - μ) / sqrt(v)`.
    s: Vec<f64>,
    /// `(y - μ) sqrt(w / v)`.
    r: Vec<f64>,
    /// `μ'(η) sqrt(w / v)`.
    g: Vec<f64>,
}

/// The working correlation at an estimate.
enum Working {
    Independence,
    Exchangeable(f64),
    Ar1(f64),
    /// The inverse of the correlation of each pattern of waves.
    Unstructured(Vec<Mat<f64>>),
}

/// Sums over the clusters of a block.
struct Moments {
    information: Vec<f64>,
    score: Vec<f64>,
    meat: Vec<f64>,
}

impl Moments {
    fn new(q: usize) -> Self {
        Self {
            information: vec![0.0; q * q],
            score: vec![0.0; q],
            meat: vec![0.0; q * q],
        }
    }

    fn add(&mut self, other: &Self) {
        for (a, b) in self.information.iter_mut().zip(&other.information) {
            *a += b;
        }
        for (a, b) in self.score.iter_mut().zip(&other.score) {
            *a += b;
        }
        for (a, b) in self.meat.iter_mut().zip(&other.meat) {
            *a += b;
        }
    }
}

impl Gee {
    /// Fits the model to the response of `data` on the columns of its
    /// design, which holds the intercept if the model has one. The clusters
    /// of the data label the cluster of each row and the waves the time or
    /// position of the row within its cluster. Without waves the rows of a
    /// cluster are ordered as given. Prior weights scale the variance of a
    /// row by `1 / w`, and the offset is added to the linear predictor.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] for a design without columns, a tolerance or fixed
    /// scale that is not positive and finite, [`Error::Unsupported`] for the
    /// unstructured correlation without waves or prior weights with the
    /// moment estimators, [`Error::DuplicateWave`] for two rows of a cluster
    /// at one wave, the errors of the independence fit that starts the
    /// iteration, and [`Error::Singular`], [`Error::NotPositiveDefinite`],
    /// [`Error::CorrelationRange`] and [`Error::Undefined`] when the
    /// iteration cannot go on; see the module documentation.
    pub fn fit(&self, data: &GeeData<'_>) -> Result<GeeFit, Error> {
        let (x, y, cluster, wave) = (data.x(), data.y(), data.clusters(), data.waves());
        let (weights, offset) = (data.weights(), data.offset());
        let (n, p) = (x.nrows(), x.ncols());
        self.check(p, wave, weights)?;
        let ones = vec![1.0; n];
        let zeros = vec![0.0; n];
        let w_in = weights.unwrap_or(&ones);
        let o_in = offset.unwrap_or(&zeros);

        let start = Glm::new(self.family, self.link).fit(&GlmData::from_parts(
            x, y, weights, offset, None,
        )?)?;
        let kept: Vec<usize> = (0..p)
            .filter(|&j| start.coefficients[j].is_some())
            .collect();
        let mut beta: Vec<f64> = kept
            .iter()
            .map(|&j| start.coefficients[j].unwrap_or(0.0))
            .collect();
        let q = kept.len();

        let panel = Panel::new(self.correlation, x, &kept, y, cluster, wave, w_in, o_in)?;
        let clusters = panel.starts.len() - 1;
        let sum_w: f64 = panel.w.iter().sum();
        let ones_s = vec![1.0; n];
        let counts = panel.association_sums(self.correlation, &ones_s, &panel.w);
        let max_cluster = (0..clusters)
            .map(|i| panel.starts[i + 1] - panel.starts[i])
            .max()
            .unwrap_or(0);

        let mut rows = self.rows_at(&panel, &beta)?;
        let mut estimate = self.estimate(&panel, &rows, &counts, sum_w, q, max_cluster)?;
        let mut converged = false;
        let mut iterations = 0;
        for iteration in 1..=self.max_iterations {
            iterations = iteration;
            let working = self.working(&panel, &estimate.alpha)?;
            let total = panel.moments(&rows, &working, false);
            let step = solve_information(&total, q)?;
            let mut change = 0.0_f64;
            for (b, d) in beta.iter_mut().zip(&step) {
                *b += d;
            }
            for (d, b) in step.iter().zip(&beta) {
                change = change.max(d.abs() / (1.0 + b.abs()));
            }
            rows = self.rows_at(&panel, &beta)?;
            let next = self.estimate(&panel, &rows, &counts, sum_w, q, max_cluster)?;
            for (a, b) in estimate.alpha.iter().zip(&next.alpha) {
                change = change.max((a - b).abs() / (1.0 + a.abs()));
            }
            change = change.max((estimate.scale - next.scale).abs() / (1.0 + estimate.scale));
            estimate = next;
            if change <= self.tolerance {
                converged = true;
                break;
            }
        }

        let working = self.working(&panel, &estimate.alpha)?;
        let total = panel.moments(&rows, &working, true);
        let information = symmetric(&total.information, q);
        let inverse = invert(&information)?;
        let robust = match self.variance {
            RobustVariance::Sandwich => Some(sandwich_with_inverse(
                inverse.as_ref(),
                symmetric(&total.meat, q).as_ref(),
            )),
            RobustVariance::ManclDeRouen => panel.mancl_de_rouen(&rows, &working, &inverse),
        };

        let expand = |m: Option<&Mat<f64>>, factor: f64| -> Result<Covariance, Error> {
            let mut out = vec![None; p * p];
            if let Some(m) = m {
                for (a, &ja) in kept.iter().enumerate() {
                    for (b, &jb) in kept.iter().enumerate() {
                        out[ja * p + jb] = Some(factor * m[(a, b)]);
                    }
                }
            }
            covariance(p, out)
        };
        let mut coefficients = vec![None; p];
        for (&j, &b) in kept.iter().zip(&beta) {
            coefficients[j] = Some(b);
        }
        let mut fitted = vec![0.0; n];
        for (i, &o) in panel.order.iter().enumerate() {
            fitted[o] = rows.mu[i];
        }
        Ok(GeeFit {
            coefficients,
            covariance: expand(robust.as_ref(), 1.0)?,
            naive_covariance: expand(Some(&inverse), estimate.scale)?,
            alpha: estimate.alpha,
            scale: estimate.scale,
            fitted,
            converged,
            iterations,
            n_clusters: clusters,
            rank: q,
        })
    }
}

impl Gee {
    fn check(&self, p: usize, wave: Option<&[u32]>, weights: Option<&[f64]>) -> Result<(), Error> {
        if weights.is_some() && self.association == Association::Moment {
            return Err(Error::Unsupported {
                what: "prior weights with the moment estimators",
            });
        }
        crate::error::tolerance("tolerance", self.tolerance)?;
        if let Some(scale) = self.fix_scale {
            crate::error::tolerance("fix_scale", scale)?;
        }
        if self.correlation == Correlation::Unstructured && wave.is_none() {
            return Err(Error::Unsupported {
                what: "the unstructured correlation without waves",
            });
        }
        epistat_core::check::non_empty("x columns", p)?;
        Ok(())
    }

    /// The means, residuals and derivatives at `beta`.
    fn rows_at(&self, panel: &Panel, beta: &[f64]) -> Result<Rows, Error> {
        let n = panel.n;
        let columns: Vec<&[f64]> = panel.x.chunks_exact(n).collect();
        let eta = linear_predictor(&columns, beta, &panel.offset);
        let mut rows = Rows {
            mu: Vec::with_capacity(n),
            s: Vec::with_capacity(n),
            r: Vec::with_capacity(n),
            g: Vec::with_capacity(n),
        };
        for (i, &e) in eta.iter().enumerate() {
            let (mu, slope) = self.link.inverse_and_derivative(e);
            let v = self.family.variance(mu);
            if !(self.family.valid_mean(mu) && v.is_finite() && v > 0.0 && slope.is_finite()) {
                return Err(Error::Undefined(
                    "a variance at a fitted mean outside the family's domain",
                ));
            }
            let e = panel.y[i] - mu;
            let root = (panel.w[i] / v).sqrt();
            rows.mu.push(mu);
            rows.s.push(e / v.sqrt());
            rows.r.push(e * root);
            rows.g.push(slope * root);
        }
        Ok(rows)
    }

    /// The scale and the association parameters from the residuals in `rows`.
    fn estimate(
        &self,
        panel: &Panel,
        rows: &Rows,
        counts: &[f64],
        sum_w: f64,
        q: usize,
        max_cluster: usize,
    ) -> Result<Estimate, Error> {
        let mut ssq = 0.0;
        for (s, w) in rows.s.iter().zip(&panel.w) {
            ssq += w * s * s;
        }
        let moment = self.association == Association::Moment;
        if not_positive(sum_w) || (moment && panel.n <= q) {
            return Err(Error::Undefined(
                "the scale, with no more rows than coefficients",
            ));
        }
        let phi_n = ssq / sum_w;
        let phi_np = ssq / (panel.n - q.min(panel.n)) as f64;
        let scale = self
            .fix_scale
            .unwrap_or(if moment { phi_np } else { phi_n });
        let phi =
            self.fix_scale
                .unwrap_or(if moment && self.correlation == Correlation::Exchangeable {
                    phi_np
                } else {
                    phi_n
                });
        let sums = panel.association_sums(self.correlation, &rows.s, &panel.w);
        let alpha = match self.correlation {
            Correlation::Independence => Vec::new(),
            Correlation::Exchangeable => {
                let pairs = counts[0];
                let denominator = if moment { pairs - q as f64 } else { pairs };
                if not_positive(denominator) {
                    return Err(Error::Undefined(
                        "the correlation, with too few pairs of rows",
                    ));
                }
                let alpha = sums[0] / (phi * denominator);
                if alpha.is_nan() || alpha >= 1.0 || alpha <= -1.0 / (max_cluster as f64 - 1.0) {
                    return Err(Error::CorrelationRange(alpha));
                }
                vec![alpha]
            }
            Correlation::Ar1 => {
                if counts.len() < 2 || counts[1..].iter().all(|&c| c <= 0.0) {
                    return Err(Error::Undefined(
                        "the correlation, with too few pairs of rows",
                    ));
                }
                let alpha = if moment {
                    if not_positive(counts[1]) {
                        return Err(Error::Undefined(
                        "the correlation, with too few pairs of rows",
                    ));
                    }
                    sums[1] / (phi * counts[1])
                } else {
                    let z: Vec<f64> = sums.iter().map(|s| s / phi).collect();
                    ar1_root(&z, counts)?
                };
                if alpha.is_nan() || alpha.abs() >= 1.0 {
                    return Err(Error::CorrelationRange(alpha));
                }
                vec![alpha]
            }
            Correlation::Unstructured => {
                let m = panel.wave_count;
                let mut alpha = Vec::new();
                for j in 0..m {
                    for k in j + 1..m {
                        let c = counts[j * m + k];
                        if not_positive(c) {
                            return Err(Error::Undefined(
                                "the correlation of two waves that never occur in one cluster",
                            ));
                        }
                        alpha.push(sums[j * m + k] / (phi * c));
                    }
                }
                if epistat_linalg::cholesky_factor(
                    correlation_matrix(&alpha, m).as_ref(),
                    Regularisation::default(),
                )
                .is_none()
                {
                    return Err(Error::NotPositiveDefinite);
                }
                alpha
            }
        };
        Ok(Estimate { alpha, scale })
    }

    /// The working correlation for the association parameters `alpha`.
    fn working(&self, panel: &Panel, alpha: &[f64]) -> Result<Working, Error> {
        Ok(match self.correlation {
            Correlation::Independence => Working::Independence,
            Correlation::Exchangeable => Working::Exchangeable(alpha[0]),
            Correlation::Ar1 => Working::Ar1(alpha[0]),
            Correlation::Unstructured => {
                let full = correlation_matrix(alpha, panel.wave_count);
                let mut inverses = Vec::with_capacity(panel.patterns.len());
                for pattern in &panel.patterns {
                    let m = pattern.len();
                    let sub = Mat::from_fn(m, m, |a, b| {
                        full[(pattern[a] as usize, pattern[b] as usize)]
                    });
                    let factor = Factor::new(sub.as_ref(), &[Attempt::plain(Method::Cholesky)])
                        .map_err(|_| Error::NotPositiveDefinite)?;
                    inverses.push(factor.solve(Mat::<f64>::identity(m, m).as_ref())?);
                }
                Working::Unstructured(inverses)
            }
        })
    }
}

struct Estimate {
    alpha: Vec<f64>,
    scale: f64,
}

/// The unstructured correlation matrix from its upper triangle, row by row.
fn correlation_matrix(alpha: &[f64], m: usize) -> Mat<f64> {
    let mut full = Mat::<f64>::identity(m, m);
    let mut next = 0;
    for j in 0..m {
        for k in j + 1..m {
            full[(j, k)] = alpha[next];
            full[(k, j)] = alpha[next];
            next += 1;
        }
    }
    full
}

/// The root of `Σ_d d α^{d-1} (z_d - C_d α^d)` in `(-1, 1)` nearest the
/// adjacent-wave estimate `z_1 / C_1`.
/// Grid points and the largest `|α|` searched for the AR(1) root.
const GRID: usize = 2000;
const EDGE: f64 = 0.9999;

fn ar1_root(z: &[f64], counts: &[f64]) -> Result<f64, Error> {
    let g = |a: f64| -> f64 {
        let mut total = 0.0;
        for d in 1..z.len() {
            if counts[d] > 0.0 {
                let d_i = d as i32;
                total += f64::from(d_i) * a.powi(d_i - 1) * (z[d] - counts[d] * a.powi(d_i));
            }
        }
        total
    };
    let target = if counts[1] > 0.0 {
        z[1] / counts[1]
    } else {
        0.0
    };
    let point = |i: usize| -EDGE + 2.0 * EDGE * i as f64 / GRID as f64;
    let mut best: Option<(f64, f64, f64)> = None;
    let mut previous = g(point(0));
    for i in 1..=GRID {
        let (lo, hi) = (point(i - 1), point(i));
        let current = g(hi);
        if previous == 0.0 || (previous < 0.0) != (current < 0.0) {
            let distance = (lo.midpoint(hi) - target).abs();
            if best.is_none_or(|(d, _, _)| distance < d) {
                best = Some((distance, lo, hi));
            }
        }
        previous = current;
    }
    let (_, mut lo, mut hi) = best.ok_or(Error::Undefined(
        "the AR(1) correlation, whose equation has no root in (-1, 1)",
    ))?;
    let mut f_lo = g(lo);
    for _ in 0..200 {
        let mid = lo.midpoint(hi);
        if mid <= lo || mid >= hi {
            break;
        }
        let f_mid = g(mid);
        if (f_mid < 0.0) == (f_lo < 0.0) && f_mid != 0.0 {
            lo = mid;
            f_lo = f_mid;
        } else {
            hi = mid;
        }
    }
    Ok(lo.midpoint(hi))
}

fn not_positive(v: f64) -> bool {
    v.is_nan() || v <= 0.0
}

fn symmetric(v: &[f64], q: usize) -> Mat<f64> {
    Mat::from_fn(q, q, |a, b| v[a * q + b].midpoint(v[b * q + a]))
}

fn invert(information: &Mat<f64>) -> Result<Mat<f64>, Error> {
    epistat_linalg::inverse(information.as_ref(), &[Attempt::plain(Method::Cholesky)])
        .map(|solution| solution.x)
        .map_err(|_| Error::Singular)
}

fn solve_information(total: &Moments, q: usize) -> Result<Vec<f64>, Error> {
    let information = symmetric(&total.information, q);
    let factor = Factor::new(information.as_ref(), &[Attempt::plain(Method::Cholesky)])
        .map_err(|_| Error::Singular)?;
    let rhs = Mat::from_fn(q, 1, |i, _| total.score[i]);
    let step = factor.solve(rhs.as_ref())?;
    Ok((0..q).map(|i| step[(i, 0)]).collect())
}

impl Panel {
    fn new(
        correlation: Correlation,
        x: MatRef<'_, f64>,
        kept: &[usize],
        y: &[f64],
        cluster: &[u32],
        wave: Option<&[u32]>,
        w: &[f64],
        offset: &[f64],
    ) -> Result<Self, Error> {
        let n = x.nrows();
        let q = kept.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| (cluster[i], wave.map_or(0, |w| w[i])));
        let mut starts = vec![0];
        for i in 1..n {
            if cluster[order[i]] != cluster[order[i - 1]] {
                starts.push(i);
            }
        }
        starts.push(n);
        let mut waves = Vec::with_capacity(n);
        for c in 0..starts.len() - 1 {
            for (position, i) in (starts[c]..starts[c + 1]).enumerate() {
                let value = wave.map_or(position as u32, |w| w[order[i]]);
                if position > 0 && waves[i - 1] == value && wave.is_some() {
                    return Err(Error::DuplicateWave {
                        cluster: cluster[order[i]],
                        wave: value,
                    });
                }
                waves.push(value);
            }
        }
        let mut wave_count = 0;
        if correlation == Correlation::Unstructured {
            let mut distinct = waves.clone();
            distinct.sort_unstable();
            distinct.dedup();
            wave_count = distinct.len();
            for v in &mut waves {
                *v = distinct.binary_search(v).unwrap_or(0) as u32;
            }
        }
        let mut lags = 0;
        if correlation == Correlation::Ar1 {
            for c in 0..starts.len() - 1 {
                let span = waves[starts[c + 1] - 1] - waves[starts[c]];
                lags = lags.max(span as usize + 1);
            }
        }
        let mut pattern_of = Vec::new();
        let mut patterns: Vec<Vec<u32>> = Vec::new();
        if correlation == Correlation::Unstructured {
            let mut seen = std::collections::BTreeMap::new();
            for c in 0..starts.len() - 1 {
                let set = waves[starts[c]..starts[c + 1]].to_vec();
                let next = patterns.len();
                let id = *seen.entry(set.clone()).or_insert(next);
                if id == next {
                    patterns.push(set);
                }
                pattern_of.push(id);
            }
        }
        let mut xs = Vec::with_capacity(n * q);
        for &j in kept {
            xs.extend(order.iter().map(|&i| x[(i, j)]));
        }
        Ok(Self {
            n,
            q,
            x: xs,
            y: order.iter().map(|&i| y[i]).collect(),
            w: order.iter().map(|&i| w[i]).collect(),
            offset: order.iter().map(|&i| offset[i]).collect(),
            wave: waves,
            order,
            starts,
            wave_count,
            lags,
            pattern_of,
            patterns,
        })
    }

    fn clusters(&self) -> usize {
        self.starts.len() - 1
    }

    /// The sums over the pairs of rows of a cluster that the association
    /// estimators need, each pair `(j, k)` weighted by `w_j w_k s_j s_k`.
    fn association_sums(&self, correlation: Correlation, s: &[f64], w: &[f64]) -> Vec<f64> {
        let len = match correlation {
            Correlation::Independence => 0,
            Correlation::Exchangeable => 1,
            Correlation::Ar1 => self.lags,
            Correlation::Unstructured => self.wave_count * self.wave_count,
        };
        if len == 0 {
            return Vec::new();
        }
        let blocks = self.clusters().div_ceil(BLOCK);
        let parts: Vec<Vec<f64>> = (0..blocks)
            .into_par_iter()
            .map(|b| {
                let mut acc = vec![0.0; len];
                let last = ((b + 1) * BLOCK).min(self.clusters());
                for c in b * BLOCK..last {
                    let (lo, hi) = (self.starts[c], self.starts[c + 1]);
                    match correlation {
                        Correlation::Exchangeable => {
                            let (mut sum, mut squares) = (0.0, 0.0);
                            for i in lo..hi {
                                let q = w[i] * s[i];
                                sum += q;
                                squares += q * q;
                            }
                            acc[0] += (sum * sum - squares) / 2.0;
                        }
                        Correlation::Ar1 => {
                            for j in lo..hi {
                                for k in lo..j {
                                    let d = (self.wave[j] - self.wave[k]) as usize;
                                    acc[d] += w[j] * w[k] * s[j] * s[k];
                                }
                            }
                        }
                        _ => {
                            let m = self.wave_count;
                            for j in lo..hi {
                                for k in lo..j {
                                    let at = self.wave[k] as usize * m + self.wave[j] as usize;
                                    acc[at] += w[j] * w[k] * s[j] * s[k];
                                }
                            }
                        }
                    }
                }
                acc
            })
            .collect();
        let mut total = vec![0.0; len];
        for part in &parts {
            for (t, v) in total.iter_mut().zip(part) {
                *t += v;
            }
        }
        total
    }

    /// `R^{-1}` applied to the `c` columns of the row-major `m` by `c` buffer
    /// of cluster `cluster`, in place.
    fn apply_inverse(
        &self,
        working: &Working,
        cluster: usize,
        buffer: &mut [f64],
        c: usize,
        scratch: &mut Vec<f64>,
    ) {
        let (lo, hi) = (self.starts[cluster], self.starts[cluster + 1]);
        let m = hi - lo;
        match working {
            Working::Independence => {}
            Working::Exchangeable(alpha) => {
                let scale = 1.0 / (1.0 - alpha);
                let shrink = alpha / (1.0 + (m as f64 - 1.0) * alpha);
                for col in 0..c {
                    let mut sum = 0.0;
                    for j in 0..m {
                        sum += buffer[j * c + col];
                    }
                    for j in 0..m {
                        buffer[j * c + col] = scale * (buffer[j * c + col] - shrink * sum);
                    }
                }
            }
            Working::Ar1(alpha) => {
                if m < 2 {
                    return;
                }
                scratch.clear();
                scratch.extend_from_slice(&buffer[..m * c]);
                let rho: Vec<f64> = (0..m - 1)
                    .map(|j| alpha.powi((self.wave[lo + j + 1] - self.wave[lo + j]) as i32))
                    .collect();
                for j in 0..m {
                    let mut diagonal = if j == 0 {
                        1.0
                    } else {
                        1.0 / (1.0 - rho[j - 1] * rho[j - 1])
                    };
                    if j + 1 < m {
                        diagonal += rho[j] * rho[j] / (1.0 - rho[j] * rho[j]);
                    }
                    for col in 0..c {
                        let mut value = diagonal * scratch[j * c + col];
                        if j > 0 {
                            value -= rho[j - 1] / (1.0 - rho[j - 1] * rho[j - 1])
                                * scratch[(j - 1) * c + col];
                        }
                        if j + 1 < m {
                            value -= rho[j] / (1.0 - rho[j] * rho[j]) * scratch[(j + 1) * c + col];
                        }
                        buffer[j * c + col] = value;
                    }
                }
            }
            Working::Unstructured(inverses) => {
                let inverse = &inverses[self.pattern_of[cluster]];
                scratch.clear();
                scratch.extend_from_slice(&buffer[..m * c]);
                for j in 0..m {
                    for col in 0..c {
                        let mut value = 0.0;
                        for k in 0..m {
                            value += inverse[(j, k)] * scratch[k * c + col];
                        }
                        buffer[j * c + col] = value;
                    }
                }
            }
        }
    }

    /// `D̃` of a cluster, row-major `m` by `q`, with `D̃_j = g_j x_j`.
    fn derivatives(&self, rows: &Rows, cluster: usize, out: &mut [f64]) {
        let (lo, hi) = (self.starts[cluster], self.starts[cluster + 1]);
        for (j, i) in (lo..hi).enumerate() {
            for k in 0..self.q {
                out[j * self.q + k] = rows.g[i] * self.x[k * self.n + i];
            }
        }
    }

    /// The sums of `D̃' R^{-1} D̃`, `D̃' R^{-1} r` and `s s'`, with
    /// `s = D̃' R^{-1} r`, over the clusters.
    fn moments(&self, rows: &Rows, working: &Working, meat: bool) -> Moments {
        let q = self.q;
        let blocks = self.clusters().div_ceil(BLOCK);
        let parts: Vec<Moments> = (0..blocks)
            .into_par_iter()
            .map(|b| {
                let mut acc = Moments::new(q);
                let mut dt = Vec::new();
                let mut rd = Vec::new();
                let mut scratch = Vec::new();
                let last = ((b + 1) * BLOCK).min(self.clusters());
                for c in b * BLOCK..last {
                    let (lo, hi) = (self.starts[c], self.starts[c + 1]);
                    let m = hi - lo;
                    dt.resize(m * q, 0.0);
                    rd.resize(m * (q + 1), 0.0);
                    self.derivatives(rows, c, &mut dt);
                    for j in 0..m {
                        rd[j * (q + 1)..j * (q + 1) + q].copy_from_slice(&dt[j * q..(j + 1) * q]);
                        rd[j * (q + 1) + q] = rows.r[lo + j];
                    }
                    self.apply_inverse(working, c, &mut rd, q + 1, &mut scratch);
                    for a in 0..q {
                        for k in 0..q {
                            let mut sum = 0.0;
                            for j in 0..m {
                                sum += dt[j * q + a] * rd[j * (q + 1) + k];
                            }
                            acc.information[a * q + k] += sum;
                        }
                    }
                    let mut s = vec![0.0; q];
                    for (a, value) in s.iter_mut().enumerate() {
                        let mut sum = 0.0;
                        for j in 0..m {
                            sum += dt[j * q + a] * rd[j * (q + 1) + q];
                        }
                        *value = sum;
                        acc.score[a] += sum;
                    }
                    if meat {
                        for a in 0..q {
                            for k in 0..q {
                                acc.meat[a * q + k] += s[a] * s[k];
                            }
                        }
                    }
                }
                acc
            })
            .collect();
        let mut total = Moments::new(q);
        for part in &parts {
            total.add(part);
        }
        total
    }

    /// The covariance with the Mancl and DeRouen correction, `None` when a
    /// cluster has a singular `I - H`.
    fn mancl_de_rouen(
        &self,
        rows: &Rows,
        working: &Working,
        inverse: &Mat<f64>,
    ) -> Option<Mat<f64>> {
        let q = self.q;
        let blocks = self.clusters().div_ceil(BLOCK);
        let parts: Vec<Option<Vec<f64>>> = (0..blocks)
            .into_par_iter()
            .map(|b| {
                let mut meat = vec![0.0; q * q];
                let mut scratch = Vec::new();
                let last = ((b + 1) * BLOCK).min(self.clusters());
                for c in b * BLOCK..last {
                    let (lo, hi) = (self.starts[c], self.starts[c + 1]);
                    let m = hi - lo;
                    let mut dt = vec![0.0; m * q];
                    self.derivatives(rows, c, &mut dt);
                    let mut rinv = vec![0.0; m * m];
                    for j in 0..m {
                        rinv[j * m + j] = 1.0;
                    }
                    self.apply_inverse(working, c, &mut rinv, m, &mut scratch);
                    let d = Mat::from_fn(m, q, |j, k| dt[j * q + k]);
                    let p = Mat::from_fn(m, m, |j, k| {
                        let mut sum = 0.0;
                        for a in 0..q {
                            for bb in 0..q {
                                sum += d[(j, a)] * inverse[(a, bb)] * d[(k, bb)];
                            }
                        }
                        sum
                    });
                    let lhs = Mat::from_fn(m, m, |j, k| {
                        let mut g = 0.0;
                        for l in 0..m {
                            g += p[(j, l)] * rinv[l * m + k];
                        }
                        f64::from(u8::from(j == k)) - g
                    });
                    let r = Mat::from_fn(m, 1, |j, _| rows.r[lo + j]);
                    let corrected = epistat_linalg::solve(
                        lhs.as_ref(),
                        r.as_ref(),
                        &[Attempt::plain(Method::PartialPivLu)],
                    )
                    .ok()?
                    .x;
                    let mut s = vec![0.0; q];
                    for (a, value) in s.iter_mut().enumerate() {
                        let mut sum = 0.0;
                        for j in 0..m {
                            let mut rr = 0.0;
                            for l in 0..m {
                                rr += rinv[j * m + l] * corrected[(l, 0)];
                            }
                            sum += dt[j * q + a] * rr;
                        }
                        *value = sum;
                    }
                    for a in 0..q {
                        for k in 0..q {
                            meat[a * q + k] += s[a] * s[k];
                        }
                    }
                }
                Some(meat)
            })
            .collect();
        let mut total = vec![0.0; q * q];
        for part in parts {
            for (t, v) in total.iter_mut().zip(part?) {
                *t += v;
            }
        }
        Some(sandwich_with_inverse(
            inverse.as_ref(),
            symmetric(&total, q).as_ref(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn product_with_inverse(correlation: Correlation, alpha: f64, waves: &[u32]) {
        let m = waves.len();
        let x = Mat::from_fn(m, 1, |_, _| 1.0);
        let panel = Panel::new(
            correlation,
            x.as_ref(),
            &[0],
            &vec![0.0; m],
            &vec![0; m],
            Some(waves),
            &vec![1.0; m],
            &vec![0.0; m],
        )
        .unwrap();
        let working = match correlation {
            Correlation::Exchangeable => Working::Exchangeable(alpha),
            _ => Working::Ar1(alpha),
        };
        let mut inverse = vec![0.0; m * m];
        for j in 0..m {
            inverse[j * m + j] = 1.0;
        }
        panel.apply_inverse(&working, 0, &mut inverse, m, &mut Vec::new());
        for j in 0..m {
            for k in 0..m {
                let r = |a: usize, b: usize| match correlation {
                    Correlation::Exchangeable => {
                        if a == b {
                            1.0
                        } else {
                            alpha
                        }
                    }
                    _ => alpha.powi((waves[a] as i32 - waves[b] as i32).abs()),
                };
                let product: f64 = (0..m).map(|l| r(j, l) * inverse[l * m + k]).sum();
                let want = f64::from(u8::from(j == k));
                assert!((product - want).abs() < 1e-12, "{correlation:?} {j} {k}");
            }
        }
    }

    #[test]
    fn the_closed_form_inverses_are_inverses_of_the_correlation() {
        for alpha in [-0.4, 0.0, 0.3, 0.85] {
            product_with_inverse(Correlation::Exchangeable, alpha, &[0, 1, 2, 3, 4]);
            product_with_inverse(Correlation::Ar1, alpha, &[0, 1, 2, 3, 4]);
            product_with_inverse(Correlation::Ar1, alpha, &[1, 2, 5, 6, 12]);
            product_with_inverse(Correlation::Ar1, alpha, &[3]);
            product_with_inverse(Correlation::Ar1, alpha, &[3, 7]);
        }
    }
}
