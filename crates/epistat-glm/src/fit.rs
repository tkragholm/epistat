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
//! products do not depend on the number of threads, so neither does the fit.
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
//! The columns are tested by modified Gram-Schmidt with a second pass of
//! orthogonalisation on `U` at the working weights of the first iteration, in
//! order, on one thread. The second pass keeps the residual norm accurate to the
//! rounding of `U` itself, far below `tol`, so exactly aliased columns are
//! found. The set is fixed from then on, where R tests again in each iteration,
//! so a column whose aliasing depends on the weights as they change between
//! iterations is not treated as R does. The test is skipped when `ridge` is
//! positive, since the penalised system is not singular.
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

use crate::columns::{column_slices, contiguous_copy, linear_predictor};
use epistat_linalg::design::select_columns;
use epistat_linalg::{Attempt, DesignError, Factor, SolveError, weighted_gram, weighted_xtz};
use epistat_linalg::{Mat, MatRef};

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
    /// The Tweedie power lies in `(0, 1)` or is not finite.
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
}

/// A fitted model.
#[derive(Clone, Debug)]
pub struct Fit {
    /// The coefficients, `None` for an aliased column, as R reports `NA`.
    pub coefficients: Vec<Option<f64>>,
    /// The number of coefficients that are not aliased.
    pub rank: usize,
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
        let ones = vec![1.0; n];
        let zeros = vec![0.0; n];
        let weights = weights.unwrap_or(&ones);
        let offset = offset.unwrap_or(&zeros);
        check("y", n, y.len())?;
        check("weights", n, weights.len())?;
        check("offset", n, offset.len())?;
        if !(self.tolerance.is_finite() && self.tolerance > 0.0) {
            return Err(GlmError::Tolerance);
        }
        if let Some((index, &value)) = weights
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
        let deviance = |mu: &[f64]| -> f64 {
            (0..n)
                .map(|i| family.deviance(y[i], mu[i], weights[i]))
                .sum()
        };
        let copy = contiguous_copy(x);
        let columns = column_slices(x, &copy);
        // The linear predictor, the mean and the derivative of the mean.
        let predict = |beta: &[f64]| -> (Vec<f64>, Vec<f64>, Vec<f64>) {
            let eta = linear_predictor(&columns, beta, offset);
            let (mu, derivative) = eta
                .iter()
                .map(|&e| link.inverse_and_derivative(e))
                .unzip();
            (eta, mu, derivative)
        };
        let valid = |mu: &[f64]| mu.iter().all(|&m| family.valid_mean(m));

        // R starts from the family's means, mapped to the linear predictor and
        // back, unless coefficients are given.
        let (mut eta, mut mu, mut derivative) = start.map_or_else(
            || {
                let eta: Vec<f64> = (0..n)
                    .map(|i| link.link(family.start(y[i], weights[i])))
                    .collect();
                let (mu, derivative) = eta
                    .iter()
                    .map(|&e| link.inverse_and_derivative(e))
                    .unzip();
                (eta, mu, derivative)
            },
            predict,
        );
        let mut old_deviance = deviance(&mu);
        let mut previous: Option<Vec<f64>> = start.map(<[f64]>::to_vec);
        let mut beta = previous.clone().unwrap_or_else(|| vec![0.0; p]);
        let (mut converged, mut boundary, mut iterations) = (false, false, 0);
        let mut current_deviance = old_deviance;
        let mut last_weights = vec![0.0; n];
        let mut last_gram = Mat::<f64>::zeros(p, p);
        // The columns kept when some are aliased, and the design without the
        // others.
        let mut active: Option<(Vec<usize>, Mat<f64>)> = None;

        for iteration in 1..=self.max_iterations {
            iterations = iteration;
            let mut z = vec![0.0; n];
            let mut w2 = vec![0.0; n];
            for i in 0..n {
                let slope = derivative[i];
                if weights[i] > 0.0 && slope != 0.0 {
                    z[i] = eta[i] - offset[i] + (y[i] - mu[i]) / slope;
                    w2[i] = weights[i] * slope * slope / family.variance(mu[i]);
                }
            }
            if iteration == 1 && self.ridge == 0.0 {
                let keep = independent_columns(x, &w2, self.tolerance);
                if keep.iter().any(|&k| !k) {
                    let columns = (0..p).filter(|&j| keep[j]).collect();
                    active = Some((columns, select_columns(x, &keep)));
                }
            }
            let design = active.as_ref().map_or(x, |(_, reduced)| reduced.as_ref());
            let (solution, gram) = self.least_squares(design, &w2, &z)?;
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
            last_weights.clone_from(&w2);
            (eta, mu, derivative) = predict(&beta);
            current_deviance = deviance(&mu);
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
                    (eta, mu, derivative) = predict(&beta);
                    current_deviance = deviance(&mu);
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

        let used = weights.iter().filter(|&&w| w > 0.0).count();
        let columns: Vec<usize> = active
            .as_ref()
            .map_or_else(|| (0..p).collect(), |(columns, _)| columns.clone());
        let rank = columns.len();
        let dispersion = if family.fixed_dispersion() {
            Some(1.0)
        } else if used > rank {
            // As summary.glm: the working weights of the last iteration times
            // the working residuals at the final mean.
            let pearson: f64 = (0..n)
                .filter(|&i| last_weights[i] > 0.0)
                .map(|i| {
                    let residual = (y[i] - mu[i]) / derivative[i];
                    last_weights[i] * residual * residual
                })
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

    fn least_squares(
        &self,
        x: MatRef<'_, f64>,
        w2: &[f64],
        z: &[f64],
    ) -> Result<(Vec<f64>, Mat<f64>), GlmError> {
        let gram = self.information(x, w2)?;
        let rhs = weighted_xtz(x, w2, z)?;
        let solution = epistat_linalg::solve(gram.as_ref(), rhs.as_ref(), &Attempt::SYMMETRIC)?;
        Ok(((0..x.ncols()).map(|j| solution.x[(j, 0)]).collect(), gram))
    }
}

/// The columns of `sqrt(W) X` that R's `dqrdc2` keeps: true where the norm left
/// after projecting out the kept columns before it is at least `tol` times the
/// norm of the column. Modified Gram-Schmidt with a second pass, in column
/// order.
fn independent_columns(x: MatRef<'_, f64>, w2: &[f64], epsilon: f64) -> Vec<bool> {
    let tol = 1e-7_f64.min(epsilon / 1000.0);
    let n = x.nrows();
    let mut basis: Vec<Vec<f64>> = Vec::new();
    (0..x.ncols())
        .map(|j| {
            let mut u: Vec<f64> = (0..n).map(|i| w2[i].sqrt() * x[(i, j)]).collect();
            let original = norm(&u);
            for _ in 0..2 {
                for q in &basis {
                    let dot: f64 = q.iter().zip(&u).map(|(a, b)| a * b).sum();
                    for (v, a) in u.iter_mut().zip(q) {
                        *v -= dot * a;
                    }
                }
            }
            let left = norm(&u);
            // A column of zeros is aliased, as in dqrdc2, which compares with 1.
            if left < tol * if original > 0.0 { original } else { 1.0 } {
                return false;
            }
            for v in &mut u {
                *v /= left;
            }
            basis.push(u);
            true
        })
        .collect()
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
