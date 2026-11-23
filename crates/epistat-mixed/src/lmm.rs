//! Linear mixed models with one grouping factor, as `lme4::lmer` fits them.
//!
//! The model for row `j` of cluster `i` is
//!
//! ```text
//! y_ij = x_ij'β + z_ij'b_i + e_ij,     e_ij ~ N(0, σ² / w_ij),     b_i ~ N(0, σ² Λ Λ')
//! ```
//!
//! with `z_ij = 1` (a random intercept) or `z_ij = (1, t_ij)` (a random
//! intercept and one random slope on `t`), prior weights `w` and an offset
//! subtracted from `y`. `Λ` is the lower triangular relative covariance
//! factor of `lme4`, with `θ = (Λ_11)` or `θ = (Λ_11, Λ_21, Λ_22)` in `lme4`'s
//! order. The random effects are `b_i = Λ u_i` with spherical `u_i`.
//!
//! # Algorithm
//!
//! With one grouping factor the clusters are independent, and every matrix
//! of `lme4`'s sparse formulation \[2\] is block diagonal. For cluster `i`,
//! with `S_i = Z_i'W_i Z_i` (`q x q`), `F_i = Z_i'W_i X_i` (`q x p`) and
//! `g_i = Z_i'W_i y_i`, formed once, and for a given `θ`,
//!
//! ```text
//! A_i = Λ'S_i Λ + I = L_i L_i'         c_i = L_i^{-1} Λ'g_i,   C_i = L_i^{-1} Λ'F_i
//! M = X'WX - Σ C_i'C_i = R_X R_X'      m = X'Wy - Σ C_i'c_i
//! β(θ) = M^{-1} m                      u_i = L_i^{-T} (c_i - C_i β)
//! r²(θ) = y'Wy - Σ c_i'c_i - m'β      the penalised weighted residual sum of squares
//! ```
//!
//! and the profiled criteria of `lme4` are
//!
//! ```text
//! ML:    d(θ) = Σ ln|A_i| + n (1 + ln(2π r² / n)) - Σ ln w
//! REML:  d(θ) = Σ ln|A_i| + ln|M| + (n - p)(1 + ln(2π r² / (n - p))) - Σ ln w
//! ```
//!
//! with `σ² = r² / n` or `r² / (n - p)`. Each evaluation costs
//! `O(G q p²)` for `G` clusters, after the sums, which cost `O(N (p + q)²)`
//! once: the fit is linear in rows and in clusters.
//!
//! The gradient is closed form. Writing `E` for the matrix with a single 1 at
//! the position `(a, b)` of `Λ` that `θ_k` fills, `B_i = A_i^{-1} Λ'`,
//! `v_i = g_i - F_i β - S_i Λ u_i` and `Q_i = F_i M^{-1} F_i'`,
//!
//! ```text
//! ∂ Σ ln|A_i| / ∂θ_k = Σ 2 (B_i S_i)_ba
//! ∂ r² / ∂θ_k        = -2 Σ v_ia u_ib                         (β and u are at their optimum)
//! ∂ ln|M| / ∂θ_k     = -Σ [ 2 (B_i Q_i)_ba - 2 (S_i Λ B_i Q_i B_i')_ab ]
//! ```
//!
//! `d(θ) / ν`, with `ν = n` or `n - p`, is minimised by BFGS (`epistat-optim`)
//! from `Λ = I`, as `lme4` starts, over all real `θ`: `d` depends on `Λ`
//! only through `ΛΛ'`, so a change of sign of a column of `Λ` leaves it
//! unchanged and the boundary `Λ_jj = 0` of `lme4`'s bounded optimisation is an
//! interior point. Newton steps follow, with the Hessian from central
//! differences of the closed-form gradient at steps `1e-5 max(1, |θ_j|)`,
//! until the step is below [`Lmm::tolerance`] times `max(1, |θ|)`; that last
//! step is not taken, and the fit has converged. A Newton step is taken only
//! when it lowers the norm of the gradient. The signs are then made
//! non-negative on the diagonal. A diagonal entry below `1e-4` in absolute
//! value, `lme4`'s tolerance for a singular fit, is set to 0 and the other
//! entries refitted; the boundary fit is kept when its criterion is not above
//! the interior one by more than `1e-10` of the criterion, far below what a
//! likelihood ratio resolves.
//!
//! At the estimate, `r²` is recomputed from the residuals, as `lme4` does,
//! the covariance of `β` is `σ² M^{-1}` (`vcov`), the random-effect
//! covariance is `σ² ΛΛ'` (`VarCorr`), the conditional modes are `b_i = Λ u_i`
//! (`ranef`) and their conditional covariances `σ² Λ A_i^{-1} Λ'`
//! (`ranef(condVar = TRUE)`), which hold `θ`, `β` and `σ` fixed.
//!
//! The fit is checked against `lme4` 2.0.6 at fixed `θ` and at the optimum.
//!
//! # Scope
//!
//! One grouping factor, a random intercept and at most one random slope with
//! an unstructured `2 x 2` covariance. Several grouping factors, nested or
//! crossed, need the sparse Cholesky factor of the whole random-effects
//! system that `lme4` uses, and are not fitted. A design without full column
//! rank is an error, where `lme4` drops columns.
//!
//! # Determinism
//!
//! Clusters are summed in blocks of [`BLOCK`] clusters, each block in order,
//! and the blocks in order, so results do not depend on the number of threads.
//!
//! # References
//!
//! 1. Pinheiro JC, Bates DM. Mixed-Effects Models in S and S-PLUS. New York:
//!    Springer; 2000.
//! 2. Bates D, Mächler M, Bolker B, Walker S. Fitting linear mixed-effects
//!    models using lme4. J Stat Softw. 2015;67(1):1-48.

use epistat_linalg::{MatRef, matvec_into, weighted_gram, weighted_xtz};
use epistat_optim::bfgs::Bfgs;
use rayon::prelude::*;

use crate::MixedError;
use crate::groups::Groups;
use crate::small;

/// Clusters per block of a parallel sum.
pub const BLOCK: usize = 512;

/// `lme4`'s tolerance below which a diagonal entry of `Λ` counts as zero.
const SINGULAR: f64 = 1e-4;

/// The criterion a linear mixed model is fitted by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Criterion {
    /// Restricted maximum likelihood, `lmer`'s default.
    Reml,
    /// Maximum likelihood, `REML = FALSE`.
    Ml,
}

/// A linear mixed model and its fitting controls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lmm {
    /// The criterion.
    pub criterion: Criterion,
    /// The fit has converged when the Newton step from the returned `θ` is
    /// below this times `max(1, |θ|)`, Euclidean norms. BFGS also stops when
    /// the norm of the gradient of the criterion divided by `n` (ML) or
    /// `n - p` (REML) is below it.
    pub tolerance: f64,
    /// The most BFGS iterations.
    pub max_iterations: usize,
}

impl Lmm {
    /// A model fitted by `criterion`, with tolerance `1e-8` and 200
    /// iterations.
    #[must_use]
    pub const fn new(criterion: Criterion) -> Self {
        Self {
            criterion,
            tolerance: 1e-8,
            max_iterations: 200,
        }
    }
}

/// How the optimisation ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Convergence {
    /// BFGS iterations, summed over the interior and boundary fits.
    pub iterations: usize,
    /// Whether BFGS met its tolerance.
    pub converged: bool,
}

/// A fitted linear mixed model.
#[derive(Clone, Debug)]
pub struct LmmFit {
    /// The fixed effects.
    pub beta: Vec<f64>,
    /// Their covariance `σ² M^{-1}`, row-major `p x p`.
    pub beta_covariance: Vec<f64>,
    /// The residual standard deviation `σ`.
    pub sigma: f64,
    /// `lme4`'s `θ`, the lower triangle of `Λ` by columns, with a
    /// non-negative diagonal.
    pub theta: Vec<f64>,
    /// The covariance `σ² ΛΛ'` of the random effects, row-major `q x q`, the
    /// intercept first.
    pub random_covariance: Vec<f64>,
    /// The criterion: `-2` times the REML log likelihood (`REMLcrit`) or
    /// `-2` times the log likelihood (`deviance`).
    pub criterion: f64,
    /// `-criterion / 2`, which is what `logLik` reports.
    pub log_likelihood: f64,
    /// The gradient of the criterion with respect to `θ`.
    pub gradient: Vec<f64>,
    /// The cluster labels, increasing; the order of the modes.
    pub groups: Vec<u32>,
    /// The conditional modes `b_i`, row-major `G x q`.
    pub modes: Vec<f64>,
    /// Their conditional covariances `σ² Λ A_i^{-1} Λ'`, `G` row-major
    /// `q x q` blocks.
    pub mode_covariances: Vec<f64>,
    /// The intraclass correlation `τ² / (τ² + σ²)` of a random-intercept
    /// model; `None` with a random slope, where it depends on the slope's
    /// variable.
    pub icc: Option<f64>,
    /// Whether a diagonal entry of `Λ` is 0, so that the random-effect
    /// covariance is singular.
    pub singular: bool,
    /// How the optimisation ended, `None` for [`Lmm::evaluate`].
    pub convergence: Option<Convergence>,
}

impl Lmm {
    /// Fits the model.
    ///
    /// `x` is the `n x p` design of the fixed effects, `cluster` the
    /// cluster label of each row, `slope` the variable of a random slope,
    /// `weights` positive prior weights (1 when `None`) and `offset` a known
    /// part of the mean.
    ///
    /// # Errors
    ///
    /// See [`MixedError`].
    pub fn fit(
        &self,
        x: MatRef<'_, f64>,
        y: &[f64],
        cluster: &[u32],
        slope: Option<&[f64]>,
        weights: Option<&[f64]>,
        offset: Option<&[f64]>,
    ) -> Result<LmmFit, MixedError> {
        if !(self.tolerance.is_finite() && self.tolerance > 0.0) {
            return Err(MixedError::Control("the tolerance"));
        }
        if self.max_iterations == 0 {
            return Err(MixedError::Control("the iteration limit"));
        }
        let problem = Problem::new(x, y, cluster, slope, weights, offset, self.criterion)?;
        let q = problem.q;
        let k = q * (q + 1) / 2;
        let diagonal: &[usize] = if q == 1 { &[0] } else { &[0, 2] };
        let mut theta = if q == 1 {
            vec![1.0]
        } else {
            vec![1.0, 0.0, 1.0]
        };
        let mut free = vec![true; k];
        let (mut value, mut iterations, mut converged) =
            self.optimise(&problem, &mut theta, &free)?;
        canonical_signs(&mut theta, q);
        for &j in diagonal {
            if theta[j].abs() >= SINGULAR {
                continue;
            }
            let mut candidate = theta.clone();
            candidate[j] = 0.0;
            let mut candidate_free = free.clone();
            candidate_free[j] = false;
            let (v, it, c) = self.optimise(&problem, &mut candidate, &candidate_free)?;
            iterations += it;
            canonical_signs(&mut candidate, q);
            if v <= value + 1e-10 * value.abs().max(1.0) {
                theta = candidate;
                free = candidate_free;
                value = v;
                converged = c;
            }
        }
        let mut fit = problem.at(&theta)?;
        fit.convergence = Some(Convergence {
            iterations,
            converged,
        });
        Ok(fit)
    }

    /// The fit at a given `θ`, with `β`, `σ` and everything else at their
    /// profiled values, as `lme4` reports them for a model whose optimiser
    /// returned `θ`.
    ///
    /// # Errors
    ///
    /// See [`MixedError`]; [`MixedError::Parameters`] when `theta` does not
    /// have 1 entry (no slope) or 3 (a slope), all finite.
    pub fn evaluate(
        &self,
        x: MatRef<'_, f64>,
        y: &[f64],
        cluster: &[u32],
        slope: Option<&[f64]>,
        weights: Option<&[f64]>,
        offset: Option<&[f64]>,
        theta: &[f64],
    ) -> Result<LmmFit, MixedError> {
        let problem = Problem::new(x, y, cluster, slope, weights, offset, self.criterion)?;
        let k = problem.q * (problem.q + 1) / 2;
        if theta.len() != k || theta.iter().any(|t| !t.is_finite()) {
            return Err(MixedError::Parameters(k));
        }
        problem.at(theta)
    }

    /// Minimises the criterion over the free entries of `theta`, in place,
    /// returning the criterion, the iterations and whether BFGS converged.
    fn optimise(
        &self,
        problem: &Problem<'_>,
        theta: &mut [f64],
        free: &[bool],
    ) -> Result<(f64, usize, bool), MixedError> {
        let index: Vec<usize> = (0..theta.len()).filter(|&j| free[j]).collect();
        let nu = problem.nu();
        if index.is_empty() {
            let e = problem.evaluate(theta, false)?;
            return Ok((e.criterion, 0, true));
        }
        let fixed = theta.to_vec();
        let full = |point: &[f64]| {
            let mut t = fixed.clone();
            for (&j, &v) in index.iter().zip(point) {
                t[j] = v;
            }
            t
        };
        let bfgs = Bfgs {
            max_iterations: self.max_iterations,
            gradient_tolerance: self.tolerance,
            ..Bfgs::default()
        };
        let start: Vec<f64> = index.iter().map(|&j| theta[j]).collect();
        let result = bfgs.minimize(
            &start,
            |point| match problem.evaluate(&full(point), true) {
                Ok(e) => (
                    e.criterion / nu,
                    index.iter().map(|&j| e.gradient[j] / nu).collect(),
                ),
                Err(_) => (f64::INFINITY, vec![0.0; index.len()]),
            },
            None,
        );
        let minimum = result.map_err(|_| MixedError::NotFiniteStart)?;
        for (&j, &v) in index.iter().zip(&minimum.x) {
            theta[j] = v;
        }
        let converged = self.polish(problem, theta, &index)?;
        let value = problem.evaluate(theta, false)?.criterion;
        Ok((value, minimum.iterations, converged))
    }

    /// Newton steps on the free entries `index` of `theta`, with the Hessian
    /// from central differences of the gradient, until the step is below the
    /// tolerance, which is then not taken. A step is taken only when it
    /// lowers the norm of the gradient. Returns whether the step fell below
    /// the tolerance.
    fn polish(
        &self,
        problem: &Problem<'_>,
        theta: &mut [f64],
        index: &[usize],
    ) -> Result<bool, MixedError> {
        let k = index.len();
        let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let mut gradient: Vec<f64> = {
            let e = problem.evaluate(theta, true)?;
            index.iter().map(|&j| e.gradient[j]).collect()
        };
        for _ in 0..8 {
            let mut hessian = vec![0.0; k * k];
            for (a, &j) in index.iter().enumerate() {
                let h = 1e-5 * theta[j].abs().max(1.0);
                let mut plus = theta.to_vec();
                let mut minus = theta.to_vec();
                plus[j] += h;
                minus[j] -= h;
                let gp = problem.evaluate(&plus, true)?.gradient;
                let gm = problem.evaluate(&minus, true)?.gradient;
                for (b, &l) in index.iter().enumerate() {
                    hessian[a * k + b] = (gp[l] - gm[l]) / (2.0 * h);
                }
            }
            let Some(step) = small::solve_symmetric(&hessian, k, &gradient) else {
                return Ok(false);
            };
            let scale = norm(&index.iter().map(|&j| theta[j]).collect::<Vec<_>>()).max(1.0);
            if norm(&step) <= self.tolerance * scale {
                return Ok(true);
            }
            let mut trial = theta.to_vec();
            for (&j, s) in index.iter().zip(&step) {
                trial[j] -= s;
            }
            let e = problem.evaluate(&trial, true)?;
            let next: Vec<f64> = index.iter().map(|&j| e.gradient[j]).collect();
            if norm(&next).partial_cmp(&norm(&gradient)) != Some(std::cmp::Ordering::Less) {
                return Ok(false);
            }
            theta.copy_from_slice(&trial);
            gradient = next;
        }
        Ok(false)
    }
}

/// Makes the diagonal of `Λ` non-negative by changing the signs of its
/// columns, which leaves `ΛΛ'` unchanged.
fn canonical_signs(theta: &mut [f64], q: usize) {
    if theta[0] < 0.0 {
        theta[0] = -theta[0];
        if q == 2 {
            theta[1] = -theta[1];
        }
    }
    if q == 2 && theta[2] < 0.0 {
        theta[2] = -theta[2];
    }
}

/// `Λ`, row-major `q x q` in the first `q²` entries.
fn lambda(theta: &[f64], q: usize) -> [f64; 4] {
    if q == 1 {
        [theta[0], 0.0, 0.0, 0.0]
    } else {
        [theta[0], 0.0, theta[1], theta[2]]
    }
}

/// The positions `(a, b)` in `Λ` of the entries of `θ`.
const POSITIONS: [(usize, usize); 3] = [(0, 0), (1, 0), (1, 1)];

/// The criterion and the profiled quantities at one `θ`.
struct Evaluation {
    criterion: f64,
    gradient: Vec<f64>,
    beta: Vec<f64>,
    /// The lower factor of `M`.
    r: Vec<f64>,
    /// `Σ ln|A_i|`.
    log_det: f64,
}

/// The data of a fit reduced to the sums it needs.
struct Problem<'a> {
    n: usize,
    p: usize,
    q: usize,
    reml: bool,
    x: MatRef<'a, f64>,
    /// `y` less the offset.
    y: Vec<f64>,
    slope: Option<&'a [f64]>,
    w: Vec<f64>,
    groups: Groups,
    /// `S_i`, `G` blocks of `q x q`.
    s: Vec<f64>,
    /// `F_i`, `G` blocks of `q x p`.
    f: Vec<f64>,
    /// `g_i`, `G` blocks of `q`.
    g: Vec<f64>,
    sxx: Vec<f64>,
    sxy: Vec<f64>,
    yy: f64,
    log_w: f64,
}

/// The Cholesky factor of a `q x q` matrix, `q` at most 2, row-major with
/// stride `q`.
fn cholesky_q(t: &[f64; 4], q: usize) -> [f64; 4] {
    if q == 1 {
        return [t[0].sqrt(), 0.0, 0.0, 0.0];
    }
    let l00 = t[0].sqrt();
    let l10 = t[2] / l00;
    let l11 = (t[3] - l10 * l10).sqrt();
    [l00, 0.0, l10, l11]
}

/// `L^{-1} b` for a `q x q` lower factor, in place.
fn forward_q(l: &[f64; 4], q: usize, b: &mut [f64]) {
    b[0] /= l[0];
    if q == 2 {
        b[1] = (b[1] - l[2] * b[0]) / l[3];
    }
}

/// `L^{-T} b` for a `q x q` lower factor, in place.
fn backward_q(l: &[f64; 4], q: usize, b: &mut [f64]) {
    if q == 2 {
        b[1] /= l[3];
        b[0] = (b[0] - l[2] * b[1]) / l[0];
    } else {
        b[0] /= l[0];
    }
}

/// Per-cluster quantities at one `θ`.
struct Cluster {
    /// The factor of `A_i`.
    l: [f64; 4],
    /// `c_i`.
    c: [f64; 2],
}

impl<'a> Problem<'a> {
    fn new(
        x: MatRef<'a, f64>,
        y: &[f64],
        cluster: &[u32],
        slope: Option<&'a [f64]>,
        weights: Option<&[f64]>,
        offset: Option<&[f64]>,
        criterion: Criterion,
    ) -> Result<Self, MixedError> {
        let (n, p) = (x.nrows(), x.ncols());
        check("y", n, y.len())?;
        check("cluster", n, cluster.len())?;
        if let Some(s) = slope {
            check("slope", n, s.len())?;
        }
        if let Some(w) = weights {
            check("weights", n, w.len())?;
        }
        if let Some(o) = offset {
            check("offset", n, o.len())?;
        }
        if p == 0 {
            return Err(MixedError::Size("the design has no columns"));
        }
        if n <= p {
            return Err(MixedError::Size(
                "there must be more rows than coefficients",
            ));
        }
        for j in 0..p {
            if let Some(row) = (0..n).find(|&i| !x[(i, j)].is_finite()) {
                return Err(MixedError::NotFinite { what: "x", row });
            }
        }
        finite("y", y)?;
        if let Some(s) = slope {
            finite("slope", s)?;
        }
        if let Some(o) = offset {
            finite("offset", o)?;
        }
        let w: Vec<f64> = weights.map_or_else(|| vec![1.0; n], <[f64]>::to_vec);
        if let Some((row, &value)) = w
            .iter()
            .enumerate()
            .find(|&(_, &v)| !(v.is_finite() && v > 0.0))
        {
            return Err(MixedError::Weight { row, value });
        }
        let y: Vec<f64> = match offset {
            Some(o) => y.iter().zip(o).map(|(a, b)| a - b).collect(),
            None => y.to_vec(),
        };
        let q = if slope.is_some() { 2 } else { 1 };
        let groups = Groups::new(cluster);
        let count = groups.count();

        let gram = weighted_gram(x, &w).map_err(|_| MixedError::RankDeficient)?;
        let sxx: Vec<f64> = (0..p * p).map(|k| gram[(k / p, k % p)]).collect();
        let xty = weighted_xtz(x, &w, &y).map_err(|_| MixedError::RankDeficient)?;
        let sxy: Vec<f64> = (0..p).map(|k| xty[(k, 0)]).collect();
        // R's QR rank test, a pivot below 1e-7 of its column's norm, on the
        // Cholesky factor of X'WX.
        let factor = small::cholesky(&sxx, p).ok_or(MixedError::RankDeficient)?;
        for k in 0..p {
            let pivot = factor[k * p + k];
            if pivot * pivot < 1e-14 * sxx[k * p + k] {
                return Err(MixedError::RankDeficient);
            }
        }
        let mut yy = 0.0;
        let mut log_w = 0.0;
        for i in 0..n {
            yy += w[i] * y[i] * y[i];
            log_w += w[i].ln();
        }

        let mut s = vec![0.0; count * q * q];
        let mut f = vec![0.0; count * q * p];
        let mut g = vec![0.0; count * q];
        s.par_chunks_mut(BLOCK * q * q)
            .zip(f.par_chunks_mut(BLOCK * q * p))
            .zip(g.par_chunks_mut(BLOCK * q))
            .enumerate()
            .for_each(|(block, ((s, f), g))| {
                let first = block * BLOCK;
                for local in 0..s.len() / (q * q) {
                    let c = first + local;
                    let (s, f, g) = (
                        &mut s[local * q * q..(local + 1) * q * q],
                        &mut f[local * q * p..(local + 1) * q * p],
                        &mut g[local * q..(local + 1) * q],
                    );
                    for &row in &groups.rows[groups.starts[c]..groups.starts[c + 1]] {
                        let wr = w[row];
                        let z = [1.0, slope.map_or(0.0, |t| t[row])];
                        for a in 0..q {
                            let wz = wr * z[a];
                            for b in 0..q {
                                s[a * q + b] += wz * z[b];
                            }
                            g[a] += wz * y[row];
                            for col in 0..p {
                                f[a * p + col] += wz * x[(row, col)];
                            }
                        }
                    }
                }
            });
        Ok(Self {
            n,
            p,
            q,
            reml: criterion == Criterion::Reml,
            x,
            y,
            slope,
            w,
            groups,
            s,
            f,
            g,
            sxx,
            sxy,
            yy,
            log_w,
        })
    }

    /// `n` for ML, `n - p` for REML.
    fn nu(&self) -> f64 {
        if self.reml {
            (self.n - self.p) as f64
        } else {
            self.n as f64
        }
    }

    /// `A_i`'s factor and `c_i`, with `C_i` written to `cmat` (`q x p`).
    fn cluster(&self, i: usize, lam: &[f64; 4], cmat: &mut [f64]) -> Cluster {
        let (q, p) = (self.q, self.p);
        let s = &self.s[i * q * q..(i + 1) * q * q];
        let f = &self.f[i * q * p..(i + 1) * q * p];
        let g = &self.g[i * q..(i + 1) * q];
        let mut t = [0.0; 4];
        for a in 0..q {
            for b in 0..q {
                let mut v = 0.0;
                for c in 0..q {
                    for d in 0..q {
                        v += lam[c * q + a] * s[c * q + d] * lam[d * q + b];
                    }
                }
                t[a * q + b] = v + if a == b { 1.0 } else { 0.0 };
            }
        }
        let l = cholesky_q(&t, q);
        let mut c = [0.0; 2];
        for a in 0..q {
            let mut v = 0.0;
            for d in 0..q {
                v += lam[d * q + a] * g[d];
            }
            c[a] = v;
        }
        forward_q(&l, q, &mut c);
        let mut column = [0.0; 2];
        for col in 0..p {
            for a in 0..q {
                let mut v = 0.0;
                for d in 0..q {
                    v += lam[d * q + a] * f[d * p + col];
                }
                column[a] = v;
            }
            forward_q(&l, q, &mut column);
            for a in 0..q {
                cmat[a * p + col] = column[a];
            }
        }
        Cluster { l, c }
    }

    /// The criterion at `theta`, with its gradient when `gradient` is set.
    fn evaluate(&self, theta: &[f64], gradient: bool) -> Result<Evaluation, MixedError> {
        let (q, p, count) = (self.q, self.p, self.groups.count());
        let lam = lambda(theta, q);
        let blocks = count.div_ceil(BLOCK);
        // Σ ln|A_i|, Σ C_i'C_i (lower triangle), Σ C_i'c_i, Σ c_i'c_i.
        let parts: Vec<(f64, Vec<f64>, Vec<f64>, f64)> = (0..blocks)
            .into_par_iter()
            .map(|b| {
                let mut log_det = 0.0;
                let mut ctc = vec![0.0; p * p];
                let mut ctv = vec![0.0; p];
                let mut cc = 0.0;
                let mut cmat = vec![0.0; q * p];
                for i in b * BLOCK..((b + 1) * BLOCK).min(count) {
                    let cl = self.cluster(i, &lam, &mut cmat);
                    for a in 0..q {
                        log_det += 2.0 * cl.l[a * q + a].ln();
                        cc += cl.c[a] * cl.c[a];
                    }
                    for j in 0..p {
                        let mut v = 0.0;
                        for a in 0..q {
                            v += cmat[a * p + j] * cl.c[a];
                        }
                        ctv[j] += v;
                        for k in 0..=j {
                            let mut v = 0.0;
                            for a in 0..q {
                                v += cmat[a * p + j] * cmat[a * p + k];
                            }
                            ctc[j * p + k] += v;
                        }
                    }
                }
                (log_det, ctc, ctv, cc)
            })
            .collect();
        let mut log_det = 0.0;
        let mut m_mat = self.sxx.clone();
        let mut m = self.sxy.clone();
        let mut r2 = self.yy;
        for (ld, ctc, ctv, cc) in &parts {
            log_det += ld;
            for j in 0..p {
                m[j] -= ctv[j];
                for k in 0..=j {
                    m_mat[j * p + k] -= ctc[j * p + k];
                }
            }
            r2 -= cc;
        }
        for j in 0..p {
            for k in 0..j {
                m_mat[k * p + j] = m_mat[j * p + k];
            }
        }
        let r = small::cholesky(&m_mat, p).ok_or(MixedError::RankDeficient)?;
        let mut beta = m.clone();
        small::forward(&r, p, &mut beta);
        small::backward(&r, p, &mut beta);
        for j in 0..p {
            r2 -= m[j] * beta[j];
        }
        let criterion = self.criterion(log_det, &r, r2);
        let k = q * (q + 1) / 2;
        let mut grad = vec![0.0; k];
        if gradient {
            let parts: Vec<[f64; 9]> = (0..blocks)
                .into_par_iter()
                .map(|b| {
                    let mut acc = [0.0; 9];
                    let mut cmat = vec![0.0; q * p];
                    let mut y_col = vec![0.0; p * q];
                    for i in b * BLOCK..((b + 1) * BLOCK).min(count) {
                        self.gradient_terms(i, &lam, &beta, &r, &mut cmat, &mut y_col, &mut acc);
                    }
                    acc
                })
                .collect();
            let mut acc = [0.0; 9];
            for part in &parts {
                for (a, v) in acc.iter_mut().zip(part) {
                    *a += v;
                }
            }
            let nu = self.nu();
            for kk in 0..k {
                grad[kk] = acc[kk] + nu / r2 * acc[3 + kk];
                if self.reml {
                    grad[kk] += acc[6 + kk];
                }
            }
        }
        Ok(Evaluation {
            criterion,
            gradient: grad,
            beta,
            r,
            log_det,
        })
    }

    /// The criterion from `Σ ln|A_i|`, the factor of `M` and `r²`.
    fn criterion(&self, log_det: f64, r: &[f64], r2: f64) -> f64 {
        let nu = self.nu();
        let mut d = log_det + nu * (1.0 + (2.0 * std::f64::consts::PI * r2 / nu).ln()) - self.log_w;
        if self.reml {
            d += small::log_det(r, self.p);
        }
        d
    }

    /// Adds cluster `i`'s terms of the three gradient sums to `acc`: entries
    /// `0..3` for `Σ ln|A_i|`, `3..6` for `r²` and `6..9` for `ln|M|`.
    fn gradient_terms(
        &self,
        i: usize,
        lam: &[f64; 4],
        beta: &[f64],
        r: &[f64],
        cmat: &mut [f64],
        y_col: &mut [f64],
        acc: &mut [f64; 9],
    ) {
        let (q, p) = (self.q, self.p);
        let cl = self.cluster(i, lam, cmat);
        let s = &self.s[i * q * q..(i + 1) * q * q];
        let f = &self.f[i * q * p..(i + 1) * q * p];
        let g = &self.g[i * q..(i + 1) * q];
        // u = L^{-T} (c - C β), e = g - F β, v = e - S Λ u.
        let mut u = [0.0; 2];
        let mut e = [0.0; 2];
        for a in 0..q {
            let (mut cb, mut fb) = (0.0, 0.0);
            for j in 0..p {
                cb += cmat[a * p + j] * beta[j];
                fb += f[a * p + j] * beta[j];
            }
            u[a] = cl.c[a] - cb;
            e[a] = g[a] - fb;
        }
        backward_q(&cl.l, q, &mut u);
        let mut lu = [0.0; 2];
        for a in 0..q {
            for b in 0..q {
                lu[a] += lam[a * q + b] * u[b];
            }
        }
        let mut v = [0.0; 2];
        for a in 0..q {
            let mut slu = 0.0;
            for b in 0..q {
                slu += s[a * q + b] * lu[b];
            }
            v[a] = e[a] - slu;
        }
        // B = A^{-1} Λ', column by column: B e_b = A^{-1} (row b of Λ).
        let mut bm = [0.0; 4];
        for col in 0..q {
            let mut rhs = [0.0; 2];
            for a in 0..q {
                rhs[a] = lam[col * q + a];
            }
            forward_q(&cl.l, q, &mut rhs);
            backward_q(&cl.l, q, &mut rhs);
            for a in 0..q {
                bm[a * q + col] = rhs[a];
            }
        }
        let bs = mul(&bm, s, q);
        let k = q * (q + 1) / 2;
        for (kk, &(a, b)) in POSITIONS.iter().take(k).enumerate() {
            acc[kk] += 2.0 * bs[b * q + a];
            acc[3 + kk] -= 2.0 * v[a] * u[b];
        }
        if self.reml {
            // Q = F M^{-1} F' = Y'Y with Y = R^{-1} F'.
            for a in 0..q {
                let column = &mut y_col[a * p..(a + 1) * p];
                column.copy_from_slice(&f[a * p..(a + 1) * p]);
                small::forward(r, p, column);
            }
            let mut qm = [0.0; 4];
            for a in 0..q {
                for b in 0..q {
                    let mut v = 0.0;
                    for j in 0..p {
                        v += y_col[a * p + j] * y_col[b * p + j];
                    }
                    qm[a * q + b] = v;
                }
            }
            let bq = mul(&bm, &qm, q);
            let mut bt = [0.0; 4];
            for a in 0..q {
                for b in 0..q {
                    bt[a * q + b] = bm[b * q + a];
                }
            }
            let kmat = mul(&bq, &bt, q);
            let slam = mul(s, lam, q);
            let sk = mul(&slam, &kmat, q);
            for (kk, &(a, b)) in POSITIONS.iter().take(k).enumerate() {
                acc[6 + kk] += -2.0 * bq[b * q + a] + 2.0 * sk[a * q + b];
            }
        }
    }

    /// Everything a fit reports, at `theta`.
    fn at(&self, theta: &[f64]) -> Result<LmmFit, MixedError> {
        let (n, p, q, count) = (self.n, self.p, self.q, self.groups.count());
        let e = self.evaluate(theta, true)?;
        let lam = lambda(theta, q);
        // Modes, their unscaled conditional covariances, and ||u||².
        let mut modes = vec![0.0; count * q];
        let mut covariances = vec![0.0; count * q * q];
        let uu_parts: Vec<f64> = modes
            .par_chunks_mut(BLOCK * q)
            .zip(covariances.par_chunks_mut(BLOCK * q * q))
            .enumerate()
            .map(|(block, (modes, covs))| {
                let mut cmat = vec![0.0; q * p];
                let mut uu = 0.0;
                for local in 0..modes.len() / q {
                    let i = block * BLOCK + local;
                    let cl = self.cluster(i, &lam, &mut cmat);
                    let mut u = [0.0; 2];
                    for a in 0..q {
                        let mut cb = 0.0;
                        for j in 0..p {
                            cb += cmat[a * p + j] * e.beta[j];
                        }
                        u[a] = cl.c[a] - cb;
                    }
                    backward_q(&cl.l, q, &mut u);
                    for a in 0..q {
                        uu += u[a] * u[a];
                        let mut b = 0.0;
                        for c in 0..q {
                            b += lam[a * q + c] * u[c];
                        }
                        modes[local * q + a] = b;
                    }
                    // Λ A^{-1} Λ' = (L^{-1} Λ')' (L^{-1} Λ').
                    let mut h = [0.0; 4];
                    for col in 0..q {
                        let mut rhs = [0.0; 2];
                        for a in 0..q {
                            rhs[a] = lam[col * q + a];
                        }
                        forward_q(&cl.l, q, &mut rhs);
                        for a in 0..q {
                            h[a * q + col] = rhs[a];
                        }
                    }
                    for a in 0..q {
                        for b in 0..q {
                            let mut v = 0.0;
                            for c in 0..q {
                                v += h[c * q + a] * h[c * q + b];
                            }
                            covs[local * q * q + a * q + b] = v;
                        }
                    }
                }
                uu
            })
            .collect();
        let mut r2 = 0.0;
        for v in &uu_parts {
            r2 += v;
        }
        // The weighted residual sum of squares, cluster by cluster.
        let mut fitted = vec![0.0; n];
        let beta_col = epistat_linalg::Mat::from_fn(p, 1, |j, _| e.beta[j]);
        matvec_into(&mut fitted, self.x, beta_col.as_ref());
        let blocks = count.div_ceil(BLOCK);
        let rss_parts: Vec<f64> = (0..blocks)
            .into_par_iter()
            .map(|b| {
                let mut rss = 0.0;
                for i in b * BLOCK..((b + 1) * BLOCK).min(count) {
                    for &row in &self.groups.rows[self.groups.starts[i]..self.groups.starts[i + 1]]
                    {
                        let mut mean = fitted[row] + modes[i * q];
                        if let Some(t) = self.slope {
                            mean += modes[i * q + 1] * t[row];
                        }
                        let res = self.y[row] - mean;
                        rss += self.w[row] * res * res;
                    }
                }
                rss
            })
            .collect();
        for v in &rss_parts {
            r2 += v;
        }
        let criterion = self.criterion(e.log_det, &e.r, r2);
        let sigma2 = r2 / self.nu();
        let inverse = small::inverse(&e.r, p);
        let beta_covariance = inverse.iter().map(|v| sigma2 * v).collect();
        for v in &mut covariances {
            *v *= sigma2;
        }
        let mut random_covariance = vec![0.0; q * q];
        for a in 0..q {
            for b in 0..q {
                let mut v = 0.0;
                for c in 0..q {
                    v += lam[a * q + c] * lam[b * q + c];
                }
                random_covariance[a * q + b] = sigma2 * v;
            }
        }
        let icc = (q == 1).then(|| {
            let t2 = theta[0] * theta[0];
            t2 / (1.0 + t2)
        });
        let singular = theta[0] == 0.0 || (q == 2 && theta[2] == 0.0);
        Ok(LmmFit {
            beta: e.beta,
            beta_covariance,
            sigma: sigma2.sqrt(),
            theta: theta.to_vec(),
            random_covariance,
            criterion,
            log_likelihood: -criterion / 2.0,
            gradient: e.gradient,
            groups: self.groups.labels.clone(),
            modes,
            mode_covariances: covariances,
            icc,
            singular,
            convergence: None,
        })
    }
}

/// The product of two `q x q` row-major matrices.
fn mul(a: &[f64], b: &[f64], q: usize) -> [f64; 4] {
    let mut out = [0.0; 4];
    for i in 0..q {
        for j in 0..q {
            let mut v = 0.0;
            for k in 0..q {
                v += a[i * q + k] * b[k * q + j];
            }
            out[i * q + j] = v;
        }
    }
    out
}

pub(crate) fn check(what: &'static str, expected: usize, found: usize) -> Result<(), MixedError> {
    if expected == found {
        Ok(())
    } else {
        Err(MixedError::Length {
            what,
            expected,
            found,
        })
    }
}

pub(crate) fn finite(what: &'static str, values: &[f64]) -> Result<(), MixedError> {
    match values.iter().position(|v| !v.is_finite()) {
        Some(row) => Err(MixedError::NotFinite { what, row }),
        None => Ok(()),
    }
}
