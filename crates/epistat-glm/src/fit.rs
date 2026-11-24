//! Iteratively reweighted least squares for generalised linear models.
//!
//! [`Glm::fit`] follows R's `glm.fit`: the same starting values, working response
//! and weights, step halving, convergence rule, dispersion and covariance. The
//! families and links, with their formulas, are described in the documentation
//! of [`Family`] and [`Link`].
//!
//! # Algorithm
//!
//! For observation `i` let `y` be the response, `w` the prior weight, `o` the
//! offset, `x` the row of the design matrix and `β` the coefficients. The linear
//! predictor is `η = x'β + o`, the mean is `μ = g⁻¹(η)` for the link `g`, `V` is
//! the variance function of the family, and `μ'(η) = dμ/dη`.
//!
//! Start. Unless starting coefficients are given, the mean starts at the
//! family's starting value `μ0(y, w)` and `η = g(μ0)`, so `μ = g⁻¹(η)` is `μ0`
//! up to the bounds of the inverse link. Given coefficients, `η` and `μ` are
//! computed from them. The deviance `D_old` is computed at this `μ`.
//!
//! Each iteration forms the working response and the working weight
//!
//! ```text
//! z = η - o + (y - μ) / μ'(η)
//! W = w μ'(η)² / V(μ)
//! ```
//!
//! and solves the weighted least-squares problem for the new coefficients by the
//! normal equations
//!
//! ```text
//! (X'WX + R) β = X'W z
//! ```
//!
//! where `R` is the ridge, below. A row whose prior weight is not positive, or
//! whose `μ'(η)` is 0, gets `z = 0` and `W = 0` and does not enter. The system is
//! solved by `epistat_linalg`, which tries a Cholesky factorisation, then
//! Bunch-Kaufman, then LU with partial and with full pivoting. Its weighted
//! products do not depend on the number of threads. The working responses and
//! weights, the linear predictor, the means and the deviance terms are formed
//! row by row in parallel chunks, and the deviance and the Pearson sum add
//! their terms in row order on one thread, so the fit does not depend on the
//! number of threads either.
//! R solves the same problem by a pivoted QR of `sqrt(W) X`, which agrees to
//! about 1e-11 on well-conditioned designs. The normal equations square the
//! condition number, so a design with nearly collinear columns loses about twice
//! as many digits as in R.
//!
//! Then `η`, `μ` and the deviance `D = Σ w d(y, μ)` are computed at the new `β`,
//! with `d` the unit deviance of the family.
//!
//! # Step halving
//!
//! If `D` is not finite, or some `μ` is outside the range of the family, `β` is
//! replaced by the mean of itself and the coefficients of the previous iteration,
//! and `η`, `μ` and `D` are recomputed. This repeats until both are valid, at
//! most `max_iterations` times, after which the fit fails with
//! [`GlmError::StepHalving`]. If there are no previous coefficients, because the
//! first step from the family's starting means is invalid, the fit fails with
//! [`GlmError::NoValidStart`]. [`Fit::boundary`] is true when the last iteration
//! halved. R's `glm.fit` also halves on an invalid `η`, which is not checked here.
//!
//! # Convergence
//!
//! The iteration stops, converged, when
//!
//! ```text
//! |D - D_old| / (|D| + 0.1) < epsilon
//! ```
//!
//! with `epsilon = tolerance`, by default `1e-8`. Otherwise `D_old = D` and the
//! iteration continues, up to `max_iterations`, by default 25. The result then
//! has `converged = false`, and the coefficients are those of the last
//! iteration. The rule is that of R's `glm.control` \[1\].
//!
//! # Aliased columns
//!
//! R's pivoted QR (LINPACK `dqrdc2`) tests the columns of `U = sqrt(W) X` in
//! order. Column `j` is aliased when the part of it left after projecting out the
//! columns kept before it has a norm below `tol` times its own norm,
//!
//! ```text
//! ‖u_j - P_kept u_j‖ < tol ‖u_j‖,        tol = min(1e-7, epsilon / 1000)
//! ```
//!
//! and a column of zeros is aliased. With the default `epsilon = 1e-8` that is a
//! relative norm of 1e-11. The first of a dependent set is kept, so the last
//! dummy of a set that adds up to the intercept is dropped. R reports `NA` for the
//! coefficient and the standard error of an aliased column and fits the others,
//! with `n_w - rank` degrees of freedom. [`Glm::fit`] does the same: the
//! coefficient, the standard error and the covariance entries of an aliased
//! column are `None`, and [`Fit::rank`] counts the others.
//!
//! The decisions are those of modified Gram-Schmidt with a second pass of
//! orthogonalisation on `U` at the working weights of the first iteration, in
//! order, which keeps the residual norm accurate to the rounding of `U`
//! itself, far below `tol`, so exactly aliased columns are found. The set is
//! fixed from then on, where R tests again in each iteration, so a column
//! whose aliasing depends on the weights as they change between iterations is
//! not treated as R does. The test is skipped when `ridge` is positive, since
//! the penalised system is not singular.
//!
//! Gram-Schmidt holds the `n` values of every column it tests. A column whose
//! decision is clear from the weighted Gram matrix `G = U'U`, which the first
//! iteration forms anyway, is settled from `G` without it; Gram-Schmidt runs
//! only up to a column the bound below does not settle. The decisions are the
//! same either way. The derivation follows.
//!
//! Let `u` be the unit roundoff, `γ_m = m u / (1 - m u)`, `K` the `k` columns
//! kept before column `j`, and `ρ` the exact relative distance of `u_j` from
//! the span of `U_K`. With `C` the Gram matrix of the columns of `K` and `j`
//! scaled to unit norm, `ρ²` is the Schur complement of `C_KK` in `C`.
//!
//! *Gram-Schmidt keeps a column with `ρ` large enough.* Each update
//! `v ← fl(v - d q)` of the two passes errs by at most `7 u |u_j|` when the
//! kept columns have norm at most `1 + γ_n` and `|v| <= 2 |u_j|`, which holds
//! while `4 k γ_n <= 1`, since an inexact projection on a vector of norm
//! below `√2` does not increase the norm by more than the error of its
//! coefficient. The `2k` updates err by `14 k u |u_j|` in all. The computed
//! columns `q_a` span the columns `u_a + e_a` with `|e_a| <= c |u_a|`,
//! `c = (14 k + 2) u`, the two counting the normalisation. A vector of the form
//! `u_j - Q c` is then at least `dist(u_j, span(U_K + E)) >= ρ |u_j| - c √k
//! |u_j| / σ` from zero, with `σ = sqrt(λ_min(C_KK)) - c √k`, by comparing the
//! two spans at the minimiser over the perturbed one. The computed norm of
//! the residual is at least `1 - γ_n - u` times its true norm, and the
//! column's own norm at most `1 + γ_n + u` times its true one. Gram-Schmidt
//! therefore keeps column `j` when
//!
//! ```text
//! ρ - c √k / σ - 14 k u  >=  tol (1 + γ_n + u) / (1 - γ_n - u).
//! ```
//!
//! *The Gram matrix bounds `ρ` from below.* Each entry of the computed `G` errs
//! by at most `γ_n |u_a| |u_b|`, in any order of summation, so the computed
//! normalised entries err by at most `η = 2.01 γ_n + 5 u`. The Cholesky factor
//! `L` computed from them is the exact factor of a matrix within
//! `γ_{k+2} |L| |L'|` of them, and the rows of `L` have norm near one. The
//! computed factor is thus the exact factor of `M = C + F` with
//! `|F|_2 <= φ = (k + 1) (η + 2 γ_{k+2})`. As `C >= M - φ I` and Schur
//! complements keep the order of positive definite matrices,
//! `ρ² >= s(M - φ I) >= s(M) - φ (1 + 2 |y|²)` when `λ_min(M_KK) >= 2 φ`, with
//! `s(M) = 1 - |l|²` the square of the last pivot, `l` the solution of
//! `L_KK l = c_Kj` and `y = L_KK^{-T} l` the minimiser of the Schur form. The
//! inverse `Z` of `L_KK` is built row by row as columns are kept; when
//! `k u |L|_F |Z|_F <= 1/4` the inverse of `L_KK` has a norm at most twice
//! `|Z|_F` (Higham 2002, section 8.2), which bounds `|y| <= 2 |l| |Z|_F` and
//! `λ_min(M_KK) >= 1 / (2 |Z|_F)²`, and also `λ_min(C_KK) >= λ_min(M_KK) - φ`
//! for `σ` above. The rounding of `1 - |l|²` adds `(k + 2) u (1 + |l|²)`.
//!
//! A column is kept without Gram-Schmidt when the lower bound on `ρ` from the
//! second part meets the condition of the first, with a further `4 u` of
//! margin. Otherwise, at the first such column, Gram-Schmidt decides the
//! columns up to it and the screen resumes on the columns it kept; at a second
//! such column, or when the conditions on `Z` and `λ_min` fail, Gram-Schmidt
//! decides all columns. An aliased column is never settled by the screen: at
//! `n = 10^7` rows the bounds settle only columns with `ρ` above about
//! `10^-4`, and an aliased column has `ρ` at the rounding level, so it always
//! goes to Gram-Schmidt, which holds the columns up to it.
//!
//! In Gram-Schmidt each projection is a sum over the rows in row order. The
//! first-pass projections on a kept column are made on all later columns at
//! once, one thread per column, which changes no operation of any column.
//!
//! # Dispersion and covariance
//!
//! The dispersion is 1 for the binomial and Poisson families, and for the others
//! the Pearson estimate
//!
//! ```text
//! φ = Σ W r²  /  (n_w - k),        r = (y - μ) / μ'(η)
//! ```
//!
//! summed over rows with positive working weight `W`, those of the last
//! iteration, with `r` the working residual at the final `μ` and `η`, as R's
//! `summary.glm` computes it. `n_w` is the number of rows with positive prior
//! weight and `k` the rank, the number of coefficients that are not
//! aliased. It is
//! `None` when `n_w <= k`, and the covariance is then `None` too. The covariance
//! matrix is
//!
//! ```text
//! φ (X'WX + R)⁻¹
//! ```
//!
//! over the columns that are not aliased, with `None` in the rows and columns of
//! the others,
//!
//! with `W` the working weights of the last iteration, which were computed from
//! the `μ` before the last update, as R's `summary.glm` does. The working weights
//! are reported as [`Fit::working_weights`]. The deviance, `η` and `μ` are those
//! after the last update.
//!
//! # Ridge
//!
//! `ridge` is added to the diagonal of `X'WX` for every coefficient but the first.
//! The first column of `X` is taken to be the intercept, which is not penalised.
//! The default is 0, which is R's fit. The covariance above includes the ridge,
//! so with a nonzero ridge it is the covariance of the penalised normal
//! equations and not `φ (X'WX)⁻¹`.
//!
//! # Arguments
//!
//! Prior weights default to 1, the offset to 0. A response outside the support of
//! the family is rejected with [`GlmError::Response`] before any iteration, a
//! prior weight that is negative or not finite with [`GlmError::Weight`], a
//! tolerance that is not positive and finite with [`GlmError::Tolerance`], and
//! inputs of the wrong length with [`GlmError::Length`].
//!
//! # References
//!
//! 1. R Core Team. Fitting generalized linear models: `glm`, `glm.fit` and
//!    `glm.control`. Help pages of the stats package of R.
//! 2. Nelder JA, Wedderburn RWM. Generalized linear models. J R Stat Soc Ser A.
//!    1972;135(3):370-84. doi:10.2307/2344614
//! 3. McCullagh P, Nelder JA. Generalized Linear Models. 2nd ed. London: Chapman
//!    and Hall; 1989.
//! 4. Green PJ. Iteratively reweighted least squares for maximum likelihood
//!    estimation, and some robust and resistant alternatives. J R Stat Soc Ser B.
//!    1984;46(2):149-92.
//! 5. Dongarra JJ, Bunch JR, Moler CB, Stewart GW. LINPACK Users' Guide.
//!    Philadelphia: SIAM; 1979. R's `dqrdc2` modifies its `dqrdc` to pivot
//!    only the columns that fall below the tolerance.
//! 6. Higham NJ. Accuracy and Stability of Numerical Algorithms. 2nd ed.
//!    Philadelphia: SIAM; 2002. Chapters 3, 8 and 10: the error bounds of
//!    inner products, triangular inversion and the Cholesky factorisation.

use crate::columns::{column_slices, contiguous_copy, linear_predictor_into, rows};
use epistat_linalg::design::select_columns;
use epistat_linalg::{Attempt, DesignError, Factor, SolveError, weighted_gram, xtv};
use epistat_linalg::{Mat, MatRef};
use rayon::prelude::*;

use crate::{Family, Link};

/// A model and its fitting controls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Glm {
    /// The distribution of the response.
    pub family: Family,
    /// The link between the mean and the linear predictor.
    pub link: Link,
    /// R's `epsilon`.
    pub tolerance: f64,
    /// R's `maxit`.
    pub max_iterations: usize,
    /// Added to the diagonal of `X'WX` for every coefficient but the first,
    /// which is 0 in R.
    pub ridge: f64,
}

impl Glm {
    /// A model with R's defaults: tolerance `1e-8`, 25 iterations, no ridge.
    #[must_use]
    pub const fn new(family: Family, link: Link) -> Self {
        Self {
            family,
            link,
            tolerance: 1e-8,
            max_iterations: 25,
            ridge: 0.0,
        }
    }
}

/// Why a model could not be fitted.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GlmError {
    /// An input has the wrong length.
    #[error("{what} has {found} entries where {expected} were expected")]
    Length {
        /// The name of the input.
        what: &'static str,
        /// The length it needed.
        expected: usize,
        /// The length it had.
        found: usize,
    },
    /// The Tweedie power lies in `(0, 1)` or is not finite, or the negative
    /// binomial `theta` is not finite and positive.
    #[error("the family's parameter is outside its range")]
    Family,
    /// A prior weight is negative or not finite.
    #[error("prior weight {index} is {value}, which is negative or not finite")]
    Weight {
        /// The position of the weight.
        index: usize,
        /// Its value.
        value: f64,
    },
    /// The tolerance is not positive and finite.
    #[error("the tolerance must be positive and finite")]
    Tolerance,
    /// A response lies outside the support of the family.
    #[error("response {index} is {value}, outside the family's support")]
    Response {
        /// The position of the response.
        index: usize,
        /// Its value.
        value: f64,
    },
    /// The first iterate left the model's domain, with no earlier one to halve
    /// toward.
    #[error("no valid coefficients were found; supply starting values")]
    NoValidStart,
    /// Halving a step did not bring the fit back into the domain.
    #[error("the step could not be corrected by halving")]
    StepHalving,
    /// The design could not be reduced to its identifiable columns.
    #[error(transparent)]
    Design(#[from] DesignError),
    /// The normal equations could not be solved.
    #[error(transparent)]
    Solve(#[from] SolveError),
    /// A sandwich variance or cluster sum could not be formed.
    #[error(transparent)]
    Inference(#[from] epistat_inference::InferenceError),
    /// The link is not one the model is defined for.
    #[error("the link is not available for this model")]
    Link,
    /// The model needs a quantity the fit does not have, such as a dispersion
    /// that is undefined for lack of residual degrees of freedom.
    #[error("the fit has no {0}")]
    Missing(&'static str),
    /// The cluster labels or counts are not usable.
    #[error("{0}")]
    Clusters(&'static str),
}

/// A fitted model.
#[derive(Clone, Debug)]
pub struct Fit {
    /// The coefficients, `None` for an aliased column, as R reports `NA`.
    pub coefficients: Vec<Option<f64>>,
    /// The number of coefficients that are not aliased.
    pub rank: usize,
    /// The number of rows with positive prior weight less the rank, the
    /// degrees of freedom of the t distribution of a coefficient when the
    /// dispersion is estimated.
    pub residual_df: usize,
    /// The fitted means.
    pub fitted: Vec<f64>,
    /// The linear predictor, offset included.
    pub linear_predictor: Vec<f64>,
    /// The residual deviance.
    pub deviance: f64,
    /// The number of IRLS iterations.
    pub iterations: usize,
    /// Whether the deviance converged within the iteration limit.
    pub converged: bool,
    /// Whether a step was halved in the last iteration.
    pub boundary: bool,
    /// The IRLS weights `w mu'(eta)^2 / V(mu)` of the last iteration, which
    /// R reports as the fit's weights.
    pub working_weights: Vec<f64>,
    /// The Pearson dispersion, or 1 for the binomial and Poisson families.
    /// `None` when there are no residual degrees of freedom.
    pub dispersion: Option<f64>,
    /// `dispersion * (X'WX + ridge)^-1`, the ridge on every diagonal entry
    /// but the first, with the last iteration's weights, as R's
    /// `summary.glm` computes it, row-major and `p x p`. The rows and columns
    /// of aliased coefficients are `None`, and so is every entry when the
    /// dispersion is.
    pub covariance: Vec<Option<f64>>,
}

impl Fit {
    /// The square roots of the covariance diagonal, `None` for an aliased
    /// coefficient or an undefined dispersion.
    #[must_use]
    pub fn standard_errors(&self) -> Vec<Option<f64>> {
        let p = self.coefficients.len();
        (0..p)
            .map(|i| self.covariance[i * p + i].map(|v| v.max(0.0).sqrt()))
            .collect()
    }
}

impl Glm {
    /// Fits the model to `y` on the columns of `x`.
    ///
    /// `weights` are prior weights (1 when `None`), rows with weight 0 do not
    /// enter the fit. `offset` is added to the linear predictor. `start`
    /// replaces the family's starting means with starting coefficients.
    ///
    /// Columns of `x` that are linear combinations of the columns before them
    /// are aliased: their coefficients and standard errors are `None`, as R's
    /// `NA`, and the others are fitted without them.
    ///
    /// # Errors
    ///
    /// See [`GlmError`].
    pub fn fit(
        &self,
        x: MatRef<'_, f64>,
        y: &[f64],
        weights: Option<&[f64]>,
        offset: Option<&[f64]>,
        start: Option<&[f64]>,
    ) -> Result<Fit, GlmError> {
        let (n, p) = (x.nrows(), x.ncols());
        check("y", n, y.len())?;
        check("weights", n, weights.map_or(n, <[f64]>::len))?;
        check("offset", n, offset.map_or(n, <[f64]>::len))?;
        if !(self.tolerance.is_finite() && self.tolerance > 0.0) {
            return Err(GlmError::Tolerance);
        }
        if let Some((index, &value)) = weights
            .unwrap_or(&[])
            .iter()
            .enumerate()
            .find(|&(_, &w)| !(w.is_finite() && w >= 0.0))
        {
            return Err(GlmError::Weight { index, value });
        }
        if let Some(start) = start {
            check("start", p, start.len())?;
        }
        if !self.family.valid() {
            return Err(GlmError::Family);
        }
        if let Some((index, &value)) = y
            .iter()
            .enumerate()
            .find(|&(_, &v)| !self.family.valid_response(v))
        {
            return Err(GlmError::Response { index, value });
        }
        let (family, link) = (self.family, self.link);
        let weight = |i: usize| weights.map_or(1.0, |w| w[i]);
        let offset_at = |i: usize| offset.map_or(0.0, |o| o[i]);
        let copy = contiguous_copy(x);
        let columns = column_slices(x, &copy);
        // The linear predictor, the mean and the derivative of the mean, and a
        // vector of one value per row that holds the working response while
        // the normal equations are formed and is scratch space otherwise.
        let mut eta = vec![0.0; n];
        let mut mu = vec![0.0; n];
        let mut derivative = vec![0.0; n];
        let mut z = vec![0.0; n];
        let predict = |beta: &[f64], eta: &mut [f64], mu: &mut [f64], derivative: &mut [f64]| {
            linear_predictor_into(&columns, beta, offset, eta);
            rows::each2(mu, derivative, |i, m, d| {
                (*m, *d) = link.inverse_and_derivative(eta[i]);
            });
        };
        // The deviance, its terms formed in parallel in `scratch` and added in
        // row order.
        let deviance = |mu: &[f64], scratch: &mut [f64]| -> f64 {
            rows::each(scratch, |i, d| *d = family.deviance(y[i], mu[i], weight(i)));
            scratch.iter().copied().sum()
        };
        let valid = |mu: &[f64]| rows::all(mu, |m| family.valid_mean(m));

        // R starts from the family's means, mapped to the linear predictor and
        // back, unless coefficients are given.
        if let Some(start) = start {
            predict(start, &mut eta, &mut mu, &mut derivative);
        } else {
            rows::each(&mut eta, |i, e| {
                *e = link.link(family.start(y[i], weight(i)));
            });
            rows::each2(&mut mu, &mut derivative, |i, m, d| {
                (*m, *d) = link.inverse_and_derivative(eta[i]);
            });
        }
        let mut old_deviance = deviance(&mu, &mut z);
        let mut previous: Option<Vec<f64>> = start.map(<[f64]>::to_vec);
        let mut beta = previous.clone().unwrap_or_else(|| vec![0.0; p]);
        let (mut converged, mut boundary, mut iterations) = (false, false, 0);
        let mut current_deviance = old_deviance;
        let mut w2 = vec![0.0; n];
        let mut last_weights = vec![0.0; n];
        let mut last_gram = Mat::<f64>::zeros(p, p);
        // The columns kept when some are aliased, and the design without the
        // others.
        let mut active: Option<(Vec<usize>, Mat<f64>)> = None;

        for iteration in 1..=self.max_iterations {
            iterations = iteration;
            rows::each2(&mut z, &mut w2, |i, z, w2| {
                let slope = derivative[i];
                let prior = weight(i);
                (*z, *w2) = if prior > 0.0 && slope != 0.0 {
                    (
                        eta[i] - offset_at(i) + (y[i] - mu[i]) / slope,
                        prior * slope * slope / family.variance(mu[i]),
                    )
                } else {
                    (0.0, 0.0)
                };
            });
            // The Gram matrix of the first iteration settles the aliased
            // columns, and is that of the normal equations when none is.
            let mut first_gram = None;
            if iteration == 1 && self.ridge == 0.0 {
                let gram = self.information(x, &w2)?;
                let keep = independent_columns(&columns, &w2, self.tolerance, &gram);
                if keep.iter().any(|&k| !k) {
                    let columns = (0..p).filter(|&j| keep[j]).collect();
                    active = Some((columns, select_columns(x, &keep)));
                } else {
                    first_gram = Some(gram);
                }
            }
            let design = active.as_ref().map_or(x, |(_, reduced)| reduced.as_ref());
            let gram = match first_gram {
                Some(gram) => gram,
                None => self.information(design, &w2)?,
            };
            let (solution, gram) = Self::least_squares(design, gram, &w2, &mut z)?;
            last_gram = gram;
            beta = vec![0.0; p];
            match &active {
                Some((columns, _)) => {
                    for (&j, &b) in columns.iter().zip(&solution) {
                        beta[j] = b;
                    }
                }
                None => beta = solution,
            }
            // The weights of this iteration become the last ones; `w2` is
            // written in full by the next.
            std::mem::swap(&mut last_weights, &mut w2);
            predict(&beta, &mut eta, &mut mu, &mut derivative);
            current_deviance = deviance(&mu, &mut z);
            boundary = false;
            if !current_deviance.is_finite() || !valid(&mu) {
                let Some(old) = previous.as_ref() else {
                    return Err(GlmError::NoValidStart);
                };
                let mut halvings = 0;
                while !current_deviance.is_finite() || !valid(&mu) {
                    halvings += 1;
                    if halvings > self.max_iterations {
                        return Err(GlmError::StepHalving);
                    }
                    for (b, o) in beta.iter_mut().zip(old) {
                        *b = (*b + o) / 2.0;
                    }
                    predict(&beta, &mut eta, &mut mu, &mut derivative);
                    current_deviance = deviance(&mu, &mut z);
                }
                boundary = true;
            }
            if (current_deviance - old_deviance).abs() / (0.1 + current_deviance.abs())
                < self.tolerance
            {
                converged = true;
                break;
            }
            old_deviance = current_deviance;
            previous = Some(beta.clone());
        }

        let used = (0..n).filter(|&i| weight(i) > 0.0).count();
        let columns: Vec<usize> = active
            .as_ref()
            .map_or_else(|| (0..p).collect(), |(columns, _)| columns.clone());
        let rank = columns.len();
        let dispersion = if family.fixed_dispersion() {
            Some(1.0)
        } else if used > rank {
            // As summary.glm: the working weights of the last iteration times
            // the working residuals at the final mean, formed in parallel and
            // added in row order over the rows with a positive weight.
            rows::each(&mut z, |i, term| {
                if last_weights[i] > 0.0 {
                    let residual = (y[i] - mu[i]) / derivative[i];
                    *term = last_weights[i] * residual * residual;
                }
            });
            let pearson: f64 = (0..n)
                .filter(|&i| last_weights[i] > 0.0)
                .map(|i| z[i])
                .sum();
            Some(pearson / (used as f64 - rank as f64))
        } else {
            None
        };
        let inverse = Factor::new(last_gram.as_ref(), &Attempt::SYMMETRIC)?
            .solve(Mat::<f64>::identity(rank, rank).as_ref())?;
        let mut covariance = vec![None; p * p];
        if let Some(dispersion) = dispersion {
            for (a, &i) in columns.iter().enumerate() {
                for (b, &j) in columns.iter().enumerate() {
                    covariance[i * p + j] = Some(dispersion * inverse[(a, b)]);
                }
            }
        }
        let coefficients = (0..p)
            .map(|j| columns.contains(&j).then_some(beta[j]))
            .collect();
        Ok(Fit {
            coefficients,
            rank,
            residual_df: used.saturating_sub(rank),
            fitted: mu,
            linear_predictor: eta,
            deviance: current_deviance,
            iterations,
            converged,
            boundary,
            working_weights: last_weights,
            dispersion,
            covariance,
        })
    }

    fn information(&self, x: MatRef<'_, f64>, w2: &[f64]) -> Result<Mat<f64>, GlmError> {
        let mut gram = weighted_gram(x, w2)?;
        for j in 1..gram.ncols() {
            gram[(j, j)] += self.ridge;
        }
        Ok(gram)
    }

    /// The solution of the normal equations with `gram`, `X'WX` with the
    /// ridge, and `gram`. `z` holds the working response and is left holding
    /// `w z`.
    fn least_squares(
        x: MatRef<'_, f64>,
        gram: Mat<f64>,
        w2: &[f64],
        z: &mut [f64],
    ) -> Result<(Vec<f64>, Mat<f64>), GlmError> {
        // `X'Wz` as `weighted_xtz` forms it, from the products `w z`, which
        // are formed in place. `weighted_gram` has checked the weights.
        rows::each(z, |i, v| *v *= w2[i]);
        let rhs = xtv(x, z);
        let solution = epistat_linalg::solve(gram.as_ref(), rhs.as_ref(), &Attempt::SYMMETRIC)?;
        Ok(((0..x.ncols()).map(|j| solution.x[(j, 0)]).collect(), gram))
    }
}

/// The columns of `sqrt(W) X` that R's `dqrdc2` keeps, as
/// [`exact_independent_columns`] decides them, settled from the weighted Gram
/// matrix `gram` where its error bound allows and by Gram-Schmidt otherwise.
///
/// The columns are screened in order, as the module documentation describes.
/// At the first column the screen cannot settle, Gram-Schmidt decides the
/// columns up to it, holding the basis of only those, and the screen resumes
/// on the columns it kept. At a second such column, or when the screen cannot
/// resume, Gram-Schmidt decides every column.
fn independent_columns(columns: &[&[f64]], w2: &[f64], epsilon: f64, gram: &Mat<f64>) -> Vec<bool> {
    screened_columns(gram, w2.len(), epsilon, |m| {
        exact_independent_columns(&columns[..m], w2, epsilon)
    })
}

/// The aliased columns of a design of `n` rows whose weighted Gram matrix is
/// `gram`, settled by the screen of the module documentation where its bound
/// allows. `exact(m)` gives the decisions of modified Gram-Schmidt for the
/// first `m` columns, which are those of a test of all columns, since each
/// column's decision depends on the columns before it only.
pub(crate) fn screened_columns(
    gram: &Mat<f64>,
    n: usize,
    epsilon: f64,
    mut exact: impl FnMut(usize) -> Vec<bool>,
) -> Vec<bool> {
    let p = gram.ncols();
    let tol = 1e-7_f64.min(epsilon / 1000.0);
    let Some(bounds) = Bounds::new(n, tol) else {
        return exact(p);
    };
    let mut keep = vec![false; p];
    let mut screen = Screen::default();
    let mut fallen_back = false;
    for j in 0..p {
        if screen.certifies(gram, j, &bounds) {
            keep[j] = true;
            screen.push(gram, j);
            continue;
        }
        if fallen_back {
            return exact(p);
        }
        fallen_back = true;
        keep[..=j].copy_from_slice(&exact(j + 1));
        screen = Screen::default();
        for k in (0..=j).filter(|&k| keep[k]) {
            if !screen.usable(gram, k, &bounds) {
                return exact(p);
            }
            screen.push(gram, k);
        }
    }
    keep
}

/// The constants of the error bounds of the screen.
struct Bounds {
    /// The unit roundoff `u`.
    u: f64,
    /// The bound `eta` on an entry of the normalised Gram matrix.
    eta: f64,
    /// The relative norm a column must keep for Gram-Schmidt to keep it, with
    /// the rounding of its two norms.
    need: f64,
}

impl Bounds {
    /// `None` when `n u` is too large for the bounds, `gamma_n > 1 / 64`.
    fn new(n: usize, tol: f64) -> Option<Self> {
        let u = f64::EPSILON / 2.0;
        let gamma = gamma(n as f64, u)?;
        (gamma <= 1.0 / 64.0).then(|| Self {
            u,
            eta: 2.01 * gamma + 5.0 * u,
            need: tol * (1.0 + gamma + u) / (1.0 - gamma - u),
        })
    }

    /// `phi`, the bound on the norm of the error of the normalised Gram
    /// matrix and of its Cholesky factor over `k + 1` columns.
    fn phi(&self, k: usize) -> f64 {
        let size = k as f64 + 1.0;
        size * (self.eta + 2.0 * gamma(size + 1.0, self.u).unwrap_or(f64::INFINITY))
    }
}

/// `gamma_m = m u / (1 - m u)`, `None` when `m u >= 1`.
fn gamma(m: f64, u: f64) -> Option<f64> {
    (m * u < 1.0).then(|| m * u / (1.0 - m * u))
}

/// The Cholesky factor `L` of the normalised Gram matrix over the kept
/// columns, its inverse `Z`, and the Frobenius norms of both.
#[derive(Default)]
struct Screen {
    kept: Vec<usize>,
    /// Row-major lower triangles, `k` by `k`.
    l: Vec<f64>,
    z: Vec<f64>,
    l_norm2: f64,
    z_norm2: f64,
}

/// The row of a column against the kept ones, `l = L^{-1} c`, and the square
/// of its last pivot, `1 - |l|^2` on the normalised scale.
struct Row {
    l: Vec<f64>,
    pivot2: f64,
}

impl Screen {
    fn k(&self) -> usize {
        self.kept.len()
    }

    /// The normalised Gram entry of columns `a` and `b`.
    fn entry(gram: &Mat<f64>, a: usize, b: usize) -> f64 {
        gram[(a, b)] / (gram[(a, a)].sqrt() * gram[(b, b)].sqrt())
    }

    /// The row of column `j`, `None` when its squared norm is not positive
    /// and finite.
    fn row(&self, gram: &Mat<f64>, j: usize) -> Option<Row> {
        let g = gram[(j, j)];
        if !(g > 0.0 && g.is_finite()) {
            return None;
        }
        let k = self.k();
        let mut l = vec![0.0; k];
        for a in 0..k {
            let mut value = Self::entry(gram, self.kept[a], j);
            for (factor, solved) in self.l[a * k..a * k + a].iter().zip(&l) {
                value -= factor * solved;
            }
            l[a] = value / self.l[a * k + a];
        }
        let pivot2 = Self::entry(gram, j, j) - l.iter().map(|v| v * v).sum::<f64>();
        Some(Row { l, pivot2 })
    }

    /// Whether the bounds that the screen rests on hold for the kept columns:
    /// `k u |L|_F |Z|_F <= 1/4`, so that the computed `Z` bounds `L^{-1}`, and
    /// `lambda_low >= 2 phi`.
    fn holds(&self, bounds: &Bounds) -> Option<(f64, f64)> {
        let k = self.k();
        if k == 0 {
            return Some((f64::INFINITY, 0.0));
        }
        let (l_norm, z_norm) = (self.l_norm2.sqrt(), self.z_norm2.sqrt());
        let phi = bounds.phi(k);
        // |L^{-1}|_2 <= 2 |Z|_F, so lambda_min(L L') >= 1 / (2 |Z|_F)^2.
        let lambda = 1.0 / (4.0 * self.z_norm2);
        let sound = k as f64 * bounds.u * l_norm * z_norm <= 0.25
            && lambda.is_finite()
            && lambda >= 2.0 * phi;
        sound.then_some((lambda, 2.0 * z_norm))
    }

    /// Whether column `j` can be added to the kept columns with the bounds
    /// still holding.
    fn usable(&self, gram: &Mat<f64>, j: usize, bounds: &Bounds) -> bool {
        self.holds(bounds).is_some() && self.row(gram, j).is_some_and(|row| row.pivot2 > 0.0)
    }

    /// Whether the bound shows that Gram-Schmidt keeps column `j`, as the
    /// module documentation derives it.
    fn certifies(&self, gram: &Mat<f64>, j: usize, bounds: &Bounds) -> bool {
        let Some((lambda, inverse_norm)) = self.holds(bounds) else {
            return false;
        };
        let Some(row) = self.row(gram, j) else {
            return false;
        };
        let k = self.k() as f64;
        let u = bounds.u;
        let phi = bounds.phi(self.k());
        let l2: f64 = row.l.iter().map(|v| v * v).sum();
        // |y*| <= |l| |L^{-1}|_2.
        let y2 = l2 * inverse_norm * inverse_norm;
        let s_low = row.pivot2 - phi * (1.0 + 2.0 * y2) - (k + 2.0) * u * (1.0 + l2);
        if s_low.is_nan() || s_low <= 0.0 {
            return false;
        }
        let rho_low = s_low.sqrt() * (1.0 - u);
        let mgs = if self.k() == 0 {
            0.0
        } else {
            let c = (14.0 * k + 2.0) * u;
            let sigma = (lambda - phi).sqrt() - c * k.sqrt();
            if sigma.is_nan() || sigma <= 0.0 {
                return false;
            }
            c * k.sqrt() / sigma + 14.0 * k * u
        };
        rho_low - mgs >= bounds.need * (1.0 + 4.0 * u)
    }

    /// Adds column `j`, whose pivot is positive, to the kept columns.
    fn push(&mut self, gram: &Mat<f64>, j: usize) {
        let row = self.row(gram, j).expect("a column the screen has accepted");
        let k = self.k();
        let pivot = row.pivot2.sqrt();
        let grow = |old: &[f64], fill: &dyn Fn(usize, usize) -> f64| {
            let mut new = vec![0.0; (k + 1) * (k + 1)];
            for a in 0..k {
                new[a * (k + 1)..=a * (k + 1) + a].copy_from_slice(&old[a * k..=a * k + a]);
            }
            for b in 0..=k {
                new[k * (k + 1) + b] = fill(k, b);
            }
            new
        };
        let l = grow(&self.l, &|_, b| if b < k { row.l[b] } else { pivot });
        // The new row of L^{-1}: z_jj = 1 / pivot and, for b < k,
        // z_jb = -(sum over a >= b of l_a z_ab) / pivot.
        let z_old = &self.z;
        let z = grow(z_old, &|_, b| {
            if b == k {
                1.0 / pivot
            } else {
                -(b..k).map(|a| row.l[a] * z_old[a * k + b]).sum::<f64>() / pivot
            }
        });
        let last = |m: &[f64]| m[k * (k + 1)..].iter().map(|v| v * v).sum::<f64>();
        self.l_norm2 += last(&l);
        self.z_norm2 += last(&z);
        self.l = l;
        self.z = z;
        self.kept.push(j);
    }
}

/// The columns of `sqrt(W) X` that R's `dqrdc2` keeps: true where the norm left
/// after projecting out the kept columns before it is at least `tol` times the
/// norm of the column. Modified Gram-Schmidt with a second pass, in column
/// order. It holds the columns, `n` values each, while it runs.
///
/// Column `j` is projected on the kept columns before it in their order, twice
/// over, and each projection is the sum over the rows in row order. The
/// projections of the first pass are made as each kept column is finished, on
/// every later column at once and in parallel, which is the same sequence of
/// operations on each column as projecting column `j` on all of them when its
/// turn comes; the second pass runs at its turn. The update `u -= d q` of one
/// projection is made in the same pass over the rows as the sum of the next,
/// which reads the updated value, so every value has the bits of the plain
/// sequence.
fn exact_independent_columns(columns: &[&[f64]], w2: &[f64], epsilon: f64) -> Vec<bool> {
    let tol = 1e-7_f64.min(epsilon / 1000.0);
    let p = columns.len();
    // Each column scaled, its norm, and the projection last summed on it
    // whose update is not yet made: the coefficient and the kept column.
    let mut pending: Vec<(Vec<f64>, Deferred)> = columns
        .par_iter()
        .map(|column| {
            let u = column.iter().zip(w2).map(|(x, w)| w.sqrt() * x).collect();
            (u, None)
        })
        .collect();
    let original: Vec<f64> = pending.par_iter().map(|(u, _)| norm(u)).collect();
    let mut basis: Vec<Vec<f64>> = Vec::new();
    let mut keep = vec![false; p];
    for j in 0..p {
        let (mut u, mut deferred) = (std::mem::take(&mut pending[j].0), pending[j].1);
        for (k, q) in basis.iter().enumerate() {
            let dot = update_and_dot(&mut u, deferred.map(|(d, m)| (d, &basis[m][..])), Some(q));
            deferred = Some((dot, k));
        }
        let left = update_and_dot(&mut u, deferred.map(|(d, m)| (d, &basis[m][..])), None).sqrt();
        // A column of zeros is aliased, as in dqrdc2, which compares with 1.
        let original = original[j];
        if left < tol * if original > 0.0 { original } else { 1.0 } {
            continue;
        }
        keep[j] = true;
        for v in &mut u {
            *v /= left;
        }
        let k = basis.len();
        pending[j + 1..]
            .par_iter_mut()
            .for_each(|(later, deferred)| {
                let dot =
                    update_and_dot(later, deferred.map(|(d, m)| (d, &basis[m][..])), Some(&u));
                *deferred = Some((dot, k));
            });
        basis.push(u);
    }
    keep
}

/// A projection summed on a column whose update `u -= d q` is not yet made:
/// the coefficient `d` and the position of `q` among the kept columns.
type Deferred = Option<(f64, usize)>;

/// Makes the update `u -= d q` of `deferred` and returns `sum_i q_i u_i` over
/// the updated `u` for the column `q`, or `sum_i u_i^2` without one, in one
/// pass over the rows.
fn update_and_dot(u: &mut [f64], deferred: Option<(f64, &[f64])>, q: Option<&[f64]>) -> f64 {
    match (deferred, q) {
        (None, None) => u.iter().map(|a| a * a).sum(),
        (None, Some(q)) => q.iter().zip(&*u).map(|(a, b)| a * b).sum(),
        (Some((d, previous)), None) => u
            .iter_mut()
            .zip(previous)
            .map(|(v, a)| {
                *v -= d * a;
                *v * *v
            })
            .sum(),
        (Some((d, previous)), Some(q)) => u
            .iter_mut()
            .zip(previous)
            .zip(q)
            .map(|((v, a), b)| {
                *v -= d * a;
                b * *v
            })
            .sum(),
    }
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|a| a * a).sum::<f64>().sqrt()
}

fn check(what: &'static str, expected: usize, found: usize) -> Result<(), GlmError> {
    if expected == found {
        Ok(())
    } else {
        Err(GlmError::Length {
            what,
            expected,
            found,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stream of standard normal draws from a fixed seed.
    struct Normals(u64);

    impl Normals {
        fn next(&mut self) -> f64 {
            let mut unit = || {
                self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = self.0;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                ((z ^ (z >> 31)) >> 11) as f64 / (1_u64 << 53) as f64
            };
            let (a, b) = (1.0 - unit(), unit());
            (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
        }
    }

    /// The decisions of the screen and of Gram-Schmidt alone, and whether the
    /// screen settled every column without Gram-Schmidt.
    fn decisions(x: &Mat<f64>, w: &[f64], epsilon: f64) -> (Vec<bool>, Vec<bool>, bool) {
        let copy = contiguous_copy(x.as_ref());
        let columns = column_slices(x.as_ref(), &copy);
        let gram = weighted_gram(x.as_ref(), w).unwrap();
        let screened = independent_columns(&columns, w, epsilon, &gram);
        let exact = exact_independent_columns(&columns, w, epsilon);
        let bounds = Bounds::new(w.len(), 1e-7_f64.min(epsilon / 1000.0)).unwrap();
        let mut screen = Screen::default();
        let mut settled = true;
        for j in 0..x.ncols() {
            if screen.certifies(&gram, j, &bounds) {
                screen.push(&gram, j);
            } else {
                settled = false;
                break;
            }
        }
        (screened, exact, settled)
    }

    fn design(n: usize, p: usize, seed: u64) -> Mat<f64> {
        let mut rng = Normals(seed);
        Mat::from_fn(n, p, |_, j| if j == 0 { 1.0 } else { rng.next() })
    }

    fn weights(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| match i % 9 {
                0 => 0.0,
                1 => 3.5,
                _ => 0.25 + (i % 5) as f64 / 4.0,
            })
            .collect()
    }

    #[test]
    fn a_well_conditioned_design_is_settled_without_gram_schmidt() {
        for (n, p) in [(2_000, 3), (5_000, 8), (20_000, 25)] {
            let x = design(n, p, 7);
            let (screened, exact, settled) = decisions(&x, &weights(n), 1e-8);
            assert_eq!(screened, exact);
            assert!(settled, "n {n}, p {p}");
            assert!(screened.iter().all(|&k| k));
        }
    }

    #[test]
    fn near_collinear_columns_get_the_decisions_of_gram_schmidt() {
        // Column 3 is column 1 plus delta times an independent column, so
        // its relative distance from the span of the others is about delta;
        // the tolerance is 1e-11, and a further column follows it.
        let n = 4_000;
        let w = weights(n);
        let mut aliased = 0;
        for delta in [
            0.0, 1e-16, 1e-14, 1e-13, 5e-13, 1e-12, 3e-12, 6e-12, 9e-12, 1e-11, 1.1e-11, 1.5e-11,
            2e-11, 5e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2,
        ] {
            let base = design(n, 6, 11);
            let x = Mat::from_fn(n, 6, |i, j| {
                if j == 3 {
                    base[(i, 1)] + delta * base[(i, 3)]
                } else {
                    base[(i, j)]
                }
            });
            for epsilon in [1e-8, 1e-6, 1e-10] {
                let (screened, exact, _) = decisions(&x, &w, epsilon);
                assert_eq!(screened, exact, "delta {delta}, epsilon {epsilon}");
                aliased += usize::from(!exact[3]);
            }
        }
        // Both sides of the tolerance are reached.
        assert!(aliased > 0 && aliased < 69);
    }

    #[test]
    fn scaled_zero_and_dependent_columns_get_the_decisions_of_gram_schmidt() {
        let n = 3_000;
        let w = weights(n);
        let base = design(n, 8, 13);
        let mut groups = Normals(17);
        // A dummy set of three that adds up to the intercept, columns scaled
        // far from one, a column of zeros and a scaled copy.
        let group: Vec<usize> = (0..n)
            .map(|_| {
                let draw = groups.next();
                usize::from(draw > -0.43) + usize::from(draw > 0.43)
            })
            .collect();
        let x = Mat::from_fn(n, 12, |i, j| match j {
            0..=2 => base[(i, j)],
            3..=5 => f64::from(u8::from(group[i] == j - 3)),
            6 => base[(i, 3)] * 1e8,
            7 => base[(i, 4)] * 1e-8,
            8 => 0.0,
            9 => base[(i, 6)] * 1e-150,
            10 => base[(i, 1)] * -3e5,
            _ => base[(i, 7)],
        });
        let (screened, exact, _) = decisions(&x, &w, 1e-8);
        assert_eq!(screened, exact);
        assert_eq!(
            exact,
            [
                true, true, true, true, true, false, true, true, false, true, false, true
            ]
        );
    }

    #[test]
    fn many_columns_with_dependencies_get_the_decisions_of_gram_schmidt() {
        let n = 6_000;
        let w = weights(n);
        let base = design(n, 60, 19);
        // Every seventh column a combination of two earlier ones, with a
        // perturbation that ranges over both sides of the tolerance.
        let x = Mat::from_fn(n, 60, |i, j| {
            if j > 7 && j % 7 == 0 {
                let delta = [0.0, 1e-13, 8e-12, 1.2e-11, 1e-9, 1e-6, 1e-3][j / 7 % 7];
                base[(i, j - 3)] - 2.0 * base[(i, j - 5)] + delta * base[(i, j)]
            } else {
                base[(i, j)]
            }
        });
        let (screened, exact, _) = decisions(&x, &w, 1e-8);
        assert_eq!(screened, exact);
        assert!(exact.iter().any(|&k| !k));
    }

    fn data() -> (Mat<f64>, Vec<f64>) {
        let x = Mat::from_fn(40, 2, |i, j| {
            if j == 0 {
                1.0
            } else {
                f64::from(u32::try_from(i).unwrap()) / 10.0
            }
        });
        let y = (0..40)
            .map(|i| {
                0.5 + f64::from(u32::try_from(i % 7).unwrap()) * 0.3
                    + f64::from(u32::try_from(i).unwrap()) / 20.0
            })
            .collect();
        (x, y)
    }

    #[test]
    fn tweedie_at_power_two_is_the_gamma_fit() {
        let (x, y) = data();
        let gamma = Glm::new(Family::Gamma, Link::Log)
            .fit(x.as_ref(), &y, None, None, None)
            .unwrap();
        let tweedie = Glm::new(Family::Tweedie { power: 2.0 }, Link::Log)
            .fit(x.as_ref(), &y, None, None, None)
            .unwrap();
        assert_eq!(gamma.coefficients, tweedie.coefficients);
        assert_eq!(gamma.rank, 2);
    }

    #[test]
    fn tweedie_between_one_and_two_fits() {
        let (x, mut y) = data();
        y[3] = 0.0;
        let fit = Glm::new(Family::Tweedie { power: 1.5 }, Link::Log)
            .fit(x.as_ref(), &y, None, None, None)
            .unwrap();
        assert!(fit.converged && fit.deviance.is_finite() && fit.dispersion.unwrap() > 0.0);
    }

    #[test]
    fn a_ridge_shrinks_the_slope() {
        let (x, y) = data();
        let plain = Glm::new(Family::Gaussian, Link::Identity)
            .fit(x.as_ref(), &y, None, None, None)
            .unwrap();
        let ridged = Glm {
            ridge: 50.0,
            ..Glm::new(Family::Gaussian, Link::Identity)
        }
        .fit(x.as_ref(), &y, None, None, None)
        .unwrap();
        assert!(ridged.coefficients[1].unwrap().abs() < plain.coefficients[1].unwrap().abs());
    }

    #[test]
    fn tweedie_parameters_and_zeros_are_checked() {
        let (x, y) = data();
        let between = Glm::new(Family::Tweedie { power: 0.5 }, Link::Log);
        assert_eq!(
            between.fit(x.as_ref(), &y, None, None, None).unwrap_err(),
            GlmError::Family
        );
        let mut zero = y.clone();
        zero[0] = 0.0;
        let above_two = Glm::new(Family::Tweedie { power: 2.5 }, Link::Log);
        assert!(matches!(
            above_two.fit(x.as_ref(), &zero, None, None, None),
            Err(GlmError::Response { index: 0, .. })
        ));
        let compound = Glm::new(Family::Tweedie { power: 1.5 }, Link::Log);
        assert!(compound.fit(x.as_ref(), &zero, None, None, None).is_ok());
    }

    #[test]
    fn an_aliased_column_has_no_coefficient() {
        let (x, y) = data();
        let wide = Mat::from_fn(
            40,
            3,
            |i, j| if j < 2 { x[(i, j)] } else { 2.0 * x[(i, 1)] },
        );
        let fit = Glm::new(Family::Gaussian, Link::Identity)
            .fit(wide.as_ref(), &y, None, None, None)
            .unwrap();
        let plain = Glm::new(Family::Gaussian, Link::Identity)
            .fit(x.as_ref(), &y, None, None, None)
            .unwrap();
        assert_eq!(fit.rank, 2);
        assert_eq!(fit.coefficients[2], None);
        assert_eq!(fit.standard_errors()[2], None);
        assert_eq!(fit.covariance[2 * 3 + 1], None);
        assert_eq!(fit.coefficients[..2], plain.coefficients[..]);
        assert_eq!(fit.standard_errors()[..2], plain.standard_errors()[..]);
    }

    #[test]
    fn no_residual_degrees_of_freedom_leaves_the_dispersion_undefined() {
        let x = Mat::from_fn(2, 2, |i, j| if j == 0 { 1.0 } else { i as f64 });
        let fit = Glm::new(Family::Gaussian, Link::Identity)
            .fit(x.as_ref(), &[1.0, 2.0], None, None, None)
            .unwrap();
        assert_eq!(fit.dispersion, None);
        assert_eq!(fit.standard_errors(), [None, None]);
    }

    #[test]
    fn errors() {
        let (x, y) = data();
        let glm = Glm::new(Family::Binomial, Link::Logit);
        assert!(matches!(
            glm.fit(x.as_ref(), &y, None, None, None),
            Err(GlmError::Response { .. })
        ));
        assert!(matches!(
            glm.fit(x.as_ref(), &y[..3], None, None, None),
            Err(GlmError::Length { .. })
        ));
        let gaussian = Glm::new(Family::Gaussian, Link::Identity);
        let mut weights = vec![1.0; 40];
        weights[5] = -1.0;
        assert!(matches!(
            gaussian.fit(x.as_ref(), &y, Some(&weights), None, None),
            Err(GlmError::Weight { index: 5, .. })
        ));
        let loose = Glm {
            tolerance: 0.0,
            ..gaussian
        };
        assert_eq!(
            loose.fit(x.as_ref(), &y, None, None, None).unwrap_err(),
            GlmError::Tolerance
        );
    }
}
