//! The linear within-family (fixed effects) estimator with a clustered
//! variance, as `fixest::feols(y ~ x | family, cluster = ~family)`.
//!
//! # Method
//!
//! Sibling comparisons estimate the effect of a covariate from differences
//! between members of the same family, which removes every confounder the
//! family shares \[1\]. For a continuous outcome this is the within estimator:
//! subtract the family mean from the outcome and the covariates and regress
//! (the Frisch-Waugh theorem) \[2\]. With prior weights `w` the means are
//! weighted, `ȳ_g = Σ w y / Σ w`, and with `ỹ = y - ȳ_g`, `x̃ = x - x̄_g`,
//!
//! ```text
//! β = (Σ w x̃ x̃')^-1 Σ w x̃ ỹ
//! ```
//!
//! which equals the coefficients of the regression with one dummy variable per
//! family, without forming the dummies. A family's values of a covariate that
//! are all equal give a deviation of exactly 0, not of the order of rounding,
//! and a column that is constant within every family is aliased and reported in
//! [`WithinFit::dropped_columns`], as R reports `NA`. Equality is of the doubles.
//!
//! The classical covariance is `σ² (Σ w x̃ x̃')^-1` with
//! `σ² = Σ w e² / (N - G - K)`, the residual degrees of freedom of the model
//! with `G` family effects and `K` slopes. The clustered covariance sums the
//! scores `w e x̃` within families,
//!
//! ```text
//! V = c (X̃'WX̃)^-1 [ Σ_g s_g s_g' ] (X̃'WX̃)^-1,        s_g = Σ_{i in g} w_i e_i x̃_i
//! ```
//!
//! with [`DfRule::Nested`] the small-sample factor of `fixest`
//!
//! ```text
//! c = G/(G - 1) (N - 1)/(N - K - 1)
//! ```
//!
//! for effects nested in the clusters. `fixest` removes families with one row
//! before it counts, because their effect fits them exactly, so `N` and `G`
//! here are the rows and families that remain. The extra 1 in `N - K - 1` is
//! the family effect `fixest` counts once when all effects are nested in the
//! cluster. The factor for `lm` with dummies and `sandwich::vcovCL` differs,
//! and is not offered. [`DfRule::Unadjusted`] has `c = 1`. Removing the
//! singleton families does not change `σ²`, where `N` and `G` fall by the same
//! amount.
//!
//! # Informative families
//!
//! Only families with at least two rows and a covariate that varies inside them
//! contribute to the estimate. [`WithinFit`] reports how many there are, how
//! many have one row, and how many in all, so that an estimate resting on few
//! families is seen as such. For a binary exposure the informative families are
//! the discordant ones.
//!
//! # Bias of the design
//!
//! The within estimator is valid for covariates that vary within the family.
//! A confounder that varies within the family is not removed and its bias
//! `β_C cov_W(X, C) / var_W(X)` can exceed that of the population estimate when
//! `var_W(X)` is small \[3\]. Classical measurement error of variance `σ_e²`
//! attenuates the estimate by `var_W(X) / (var_W(X) + σ_e²)`, which is stronger
//! than in the population when the exposure is more alike within families than
//! between them \[3\]. If one sibling's exposure affects another's outcome the
//! estimate is the direct effect minus that carryover \[4\]. Birth order and
//! birth year differ between siblings by construction. Only informative
//! families contribute, so their covariates should be compared with the whole
//! sample.
//!
//! # Between-within decomposition
//!
//! [`between_within`] returns the family means and the deviations. Entering
//! both as covariates in a model separates the within-family effect, the
//! coefficient of the deviation, from the between-family contextual effect, that
//! of the mean \[5\].
//!
//! # References
//!
//! 1. Frisell T, Öberg S, Kuja-Halkola R, Sjölander A. Sibling comparison
//!    designs: bias from non-shared confounders and measurement error.
//!    Epidemiology. 2012;23(5):713-20.
//! 2. Wooldridge JM. Econometric Analysis of Cross Section and Panel Data. 2nd
//!    ed. Cambridge, MA: MIT Press; 2010. Chapter 10.
//! 3. Frisell T et al., as above, and the same article's appendix.
//! 4. Sjölander A, Frisell T, Kuja-Halkola R, Öberg S, Zetterqvist J.
//!    Carryover effects in sibling comparison designs. Epidemiology.
//!    2016;27(6):852-8.
//! 5. Neuhaus JM, Kalbfleisch JD. Between- and within-cluster covariate
//!    effects in the analysis of clustered data. Biometrics. 1998;54(2):638-45.
//! 6. Bergé L. Efficient estimation of maximum likelihood models with multiple
//!    fixed-effects: the R package FENmlm. CREA Discussion Paper 2018-13. The
//!    `fixest` package.

use epistat_inference::Clusters;
use epistat_inference::sandwich::{meat, sandwich_with_inverse};
use epistat_linalg::design::select_columns;
use epistat_linalg::{Attempt, Factor, Mat, MatRef, weighted_gram};

use crate::robust::kept_columns;
use crate::{Family, Glm, GlmError, Link};

/// The small-sample factor of the clustered covariance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DfRule {
    /// `G/(G-1) (N-1)/(N-K-1)` with singleton families removed, as `fixest`
    /// with effects nested in the clusters.
    Nested,
    /// No factor.
    Unadjusted,
}

/// The controls of a within-family fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Within {
    /// The small-sample factor of the clustered covariance.
    pub df_rule: DfRule,
}

impl Within {
    /// The nested rule of `fixest`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            df_rule: DfRule::Nested,
        }
    }
}

impl Default for Within {
    fn default() -> Self {
        Self::new()
    }
}

/// A within-family fit.
#[derive(Clone, Debug)]
pub struct WithinFit {
    /// The slopes, `None` for a column that is constant within every family.
    pub coefficients: Vec<Option<f64>>,
    /// The classical covariance `σ² (X̃'WX̃)^-1`, row-major `p x p`, `None`
    /// where `σ²` is undefined and for aliased columns.
    pub covariance: Vec<Option<f64>>,
    /// The covariance with scores summed within families, row-major `p x p`,
    /// `None` when the factor of the rule does not exist and for aliased
    /// columns.
    pub robust_covariance: Vec<Option<f64>>,
    /// `Σ w e² / (N - G - K)`, `None` without residual degrees of freedom.
    pub sigma2: Option<f64>,
    /// `1 - Σ w e² / Σ w ỹ²`, `None` when the outcome does not vary within
    /// families.
    pub r2_within: Option<f64>,
    /// The number of slopes that are not aliased.
    pub rank: usize,
    /// The number of rows with positive weight.
    pub n_rows: usize,
    /// The number of families with a row of positive weight.
    pub n_clusters: usize,
    /// The number of families with at least two rows in which some covariate
    /// varies.
    pub n_informative_clusters: usize,
    /// The number of families with one row.
    pub n_singleton_clusters: usize,
    /// The columns that are constant within every family, or collinear with
    /// the columns before them after the family means are removed.
    pub dropped_columns: Vec<usize>,
}

/// Family means and deviations, for the columns of `value(row, column)`.
struct Demeaned {
    deviations: Mat<f64>,
    /// Rows with positive weight in each cluster.
    rows: Vec<usize>,
    /// Whether any of the first `covariates` columns varies within each
    /// cluster.
    varies: Vec<bool>,
    means: Mat<f64>,
}

fn demean(
    columns: usize,
    covariates: usize,
    value: impl Fn(usize, usize) -> f64,
    weights: &[f64],
    clusters: &Clusters,
) -> Demeaned {
    let (n, g) = (clusters.len(), clusters.count());
    let of = clusters.of();
    let mut total = vec![0.0; g];
    let mut rows = vec![0usize; g];
    let mut sums = Mat::<f64>::zeros(g, columns);
    let mut first = Mat::<f64>::zeros(g, columns);
    let mut constant = vec![true; g * columns];
    for i in 0..n {
        if weights[i] > 0.0 {
            let c = of[i];
            for j in 0..columns {
                let v = value(i, j);
                if rows[c] == 0 {
                    first[(c, j)] = v;
                } else if v.to_bits() != first[(c, j)].to_bits() {
                    constant[c * columns + j] = false;
                }
                sums[(c, j)] += weights[i] * v;
            }
            total[c] += weights[i];
            rows[c] += 1;
        }
    }
    let means = Mat::from_fn(g, columns, |c, j| {
        if rows[c] == 0 {
            0.0
        } else if constant[c * columns + j] {
            first[(c, j)]
        } else {
            sums[(c, j)] / total[c]
        }
    });
    let deviations = Mat::from_fn(n, columns, |i, j| {
        let c = of[i];
        if weights[i] > 0.0 && !constant[c * columns + j] {
            value(i, j) - means[(c, j)]
        } else {
            0.0
        }
    });
    let varies = (0..g)
        .map(|c| (0..covariates).any(|j| !constant[c * columns + j]))
        .collect();
    Demeaned {
        deviations,
        rows,
        varies,
        means,
    }
}

/// The family means of the columns of `x` (one row per family, in the order of
/// the clusters) and the deviations from them (one row per row of `x`), for a
/// between-within model. A family whose values of a column are all equal has
/// deviation 0 in it.
///
/// # Errors
///
/// [`GlmError::Length`] when `clusters` covers another number of rows.
pub fn between_within(
    x: MatRef<'_, f64>,
    clusters: &Clusters,
) -> Result<(Mat<f64>, Mat<f64>), GlmError> {
    if clusters.len() != x.nrows() {
        return Err(GlmError::Length {
            what: "clusters",
            expected: x.nrows(),
            found: clusters.len(),
        });
    }
    let ones = vec![1.0; x.nrows()];
    let demeaned = demean(x.ncols(), x.ncols(), |i, j| x[(i, j)], &ones, clusters);
    Ok((demeaned.means, demeaned.deviations))
}

impl Within {
    /// Fits `y` on the columns of `x` within the families of `clusters`.
    /// `weights` are prior weights, 1 when `None`; rows with weight 0 do not
    /// enter.
    ///
    /// # Errors
    ///
    /// [`GlmError::Length`] for inputs of the wrong length,
    /// [`GlmError::Weight`] for a weight that is negative or not finite,
    /// [`GlmError::Response`] for an outcome or [`GlmError::Clusters`] for a
    /// covariate that is not finite, and the errors of the least squares fit.
    pub fn fit(
        &self,
        x: MatRef<'_, f64>,
        y: &[f64],
        clusters: &Clusters,
        weights: Option<&[f64]>,
    ) -> Result<WithinFit, GlmError> {
        let (n, p) = (x.nrows(), x.ncols());
        let ones = vec![1.0; n];
        let w = weights.unwrap_or(&ones);
        for (what, found) in [
            ("y", y.len()),
            ("weights", w.len()),
            ("clusters", clusters.len()),
        ] {
            if found != n {
                return Err(GlmError::Length {
                    what,
                    expected: n,
                    found,
                });
            }
        }
        if let Some((index, &value)) = w
            .iter()
            .enumerate()
            .find(|&(_, &v)| !(v.is_finite() && v >= 0.0))
        {
            return Err(GlmError::Weight { index, value });
        }
        if let Some((index, &value)) = y
            .iter()
            .enumerate()
            .find(|&(i, &v)| w[i] > 0.0 && !v.is_finite())
        {
            return Err(GlmError::Response { index, value });
        }
        if (0..n).any(|i| w[i] > 0.0 && (0..p).any(|j| !x[(i, j)].is_finite())) {
            return Err(GlmError::Clusters("a covariate is not finite"));
        }
        let demeaned = demean(
            p + 1,
            p,
            |i, j| if j < p { x[(i, j)] } else { y[i] },
            w,
            clusters,
        );
        let x_tilde = demeaned.deviations.as_ref().subcols(0, p);
        let y_tilde: Vec<f64> = (0..n).map(|i| demeaned.deviations[(i, p)]).collect();
        let n_rows = w.iter().filter(|&&v| v > 0.0).count();
        let n_clusters = demeaned.rows.iter().filter(|&&r| r > 0).count();
        let n_singleton_clusters = demeaned.rows.iter().filter(|&&r| r == 1).count();
        let n_informative_clusters = (0..clusters.count())
            .filter(|&c| demeaned.rows[c] >= 2 && demeaned.varies[c])
            .count();

        let fit = Glm::new(Family::Gaussian, Link::Identity)
            .fit(x_tilde, &y_tilde, weights, None, None)?;
        let columns = kept_columns(&fit);
        let rank = columns.len();
        let dropped_columns: Vec<usize> = (0..p).filter(|j| !columns.contains(j)).collect();
        let keep: Vec<bool> = (0..p).map(|j| columns.contains(&j)).collect();
        let reduced = select_columns(x_tilde, &keep);

        let gram = weighted_gram(reduced.as_ref(), w)?;
        let inverse = Factor::new(gram.as_ref(), &Attempt::SYMMETRIC)?
            .solve(Mat::<f64>::identity(rank, rank).as_ref())?;
        let residual: Vec<f64> = (0..n)
            .map(|i| y_tilde[i] - fit.linear_predictor[i])
            .collect();
        let rss: f64 = (0..n).map(|i| w[i] * residual[i] * residual[i]).sum();
        let tss: f64 = (0..n).map(|i| w[i] * y_tilde[i] * y_tilde[i]).sum();

        // The rows and families that remain when families with one row are
        // removed.
        let kept_rows: usize = demeaned.rows.iter().filter(|&&r| r >= 2).sum();
        let kept_clusters = demeaned.rows.iter().filter(|&&r| r >= 2).count();
        let sigma2 = (kept_rows > kept_clusters + rank)
            .then(|| rss / (kept_rows - kept_clusters - rank) as f64);
        let r2_within = (tss > 0.0).then(|| 1.0 - rss / tss);

        let expand = |matrix: &Mat<f64>, scale: Option<f64>| -> Vec<Option<f64>> {
            let mut out = vec![None; p * p];
            if let Some(scale) = scale {
                for (a, &i) in columns.iter().enumerate() {
                    for (b, &j) in columns.iter().enumerate() {
                        out[i * p + j] = Some(scale * matrix[(a, b)]);
                    }
                }
            }
            out
        };
        let covariance = expand(&inverse, sigma2);

        let scores = |i: usize| w[i] * residual[i];
        let middle = meat(reduced.as_ref(), scores, Some(clusters))?;
        let sandwich = sandwich_with_inverse(inverse.as_ref(), middle.as_ref());
        let factor = match self.df_rule {
            DfRule::Unadjusted => Some(1.0),
            DfRule::Nested => (kept_clusters >= 2 && kept_rows > rank + 1).then(|| {
                kept_clusters as f64 / (kept_clusters - 1) as f64 * (kept_rows - 1) as f64
                    / (kept_rows - rank - 1) as f64
            }),
        };
        let robust_covariance = expand(&sandwich, factor);

        Ok(WithinFit {
            coefficients: fit.coefficients,
            covariance,
            robust_covariance,
            sigma2,
            r2_within,
            rank,
            n_rows,
            n_clusters,
            n_informative_clusters,
            n_singleton_clusters,
            dropped_columns,
        })
    }
}
