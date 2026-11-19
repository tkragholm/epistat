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
//! # Dispersion and covariance
//!
//! The dispersion is 1 for the binomial and Poisson families, and for the others
//! the Pearson estimate
//!
//! ```text
//! φ = Σ w (y - μ)² / V(μ)  /  (n_w - p)
//! ```
//!
//! summed over rows with positive prior weight, with `n_w` the number of such
//! rows and `p` the number of coefficients. The covariance matrix is
//!
//! ```text
//! φ (X'WX + R)⁻¹
//! ```
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
//! the family is rejected with [`GlmError::Response`] before any iteration, and
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

mod family;
mod fit;

pub use family::{Family, Link};
pub use fit::{Fit, Glm, GlmError};
