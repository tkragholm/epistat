//! Negative binomial regression of type 2 with the shape `θ` estimated by
//! maximum likelihood, as `MASS::glm.nb` does.
//!
//! # Model
//!
//! A count `y` has mean `μ` and variance `μ + μ²/θ`, a Poisson with a
//! gamma-distributed multiplicative heterogeneity, and `g(μ) = x'β + o` for a
//! link `g` (log or identity) and an offset `o`. With prior weight `w`, treated
//! as a frequency, the log likelihood is
//!
//! ```text
//! l = Σ w [ lnΓ(θ+y) - lnΓ(θ) - lnΓ(y+1) + θ ln θ + y ln μ - (θ+y) ln(θ+μ) ]
//! ```
//!
//! and its derivative with respect to `θ` is, per row,
//!
//! ```text
//! s = w [ ψ(θ+y) - ψ(θ) + ln θ + 1 - ln(θ+μ) - (y+θ)/(μ+θ) ]
//! i = w [ ψ'(θ) - ψ'(θ+y) - 1/θ + 2/(μ+θ) - (y+θ)/(μ+θ)² ]      (i = -ds/dθ)
//! ```
//!
//! For a given `θ` the family is an exponential family with variance
//! `μ + μ²/θ`, so [`Glm`] fits `β` by the iteratively reweighted least
//! squares of R's `glm.fit`. For the log link the expected information is
//! block diagonal in `(β, θ)`, so the covariance of `β` given `θ` is the one
//! to report, and the uncertainty in `θ` costs nothing asymptotically \[1, 2\].
//!
//! # Algorithm
//!
//! Both rules start from the Poisson fit and alternate: fit `β` at `θ`, then
//! update `θ` at the new means, until the change is below the tolerance.
//!
//! [`ThetaRule::RParity`] is `glm.nb`. `θ` is updated by `theta.ml`: Newton's
//! method on the score from `θ0 = n / Σ w (y/μ - 1)²`, with `n = Σ w`, at
//! most `limit - 1` steps and until a step is below `eps = ε^(1/4)`, about
//! 1.2e-4. Each alternation fits `β` at the old `θ` from the old means, updates
//! `θ` at the old means, and takes the new means from the fit. The outer loop
//! stops when
//!
//! ```text
//! |l_old - l| / d1 + |θ_old - θ| <= tolerance,       d1 = sqrt(2 max(1, df))
//! ```
//!
//! with `df` the residual degrees of freedom of the first fit, after at most
//! `max_iterations` alternations. The `θ` that results is exact only to the
//! tolerance of `theta.ml`, about 1e-5 relative, because the likelihood is
//! flat in `θ`. The reported fit is the last one, conditional on the `θ` before
//! the final update, as `summary.negbin` reports it.
//!
//! [`ThetaRule::Tight`] solves the score for `θ` at fixed means to a relative
//! step of 1e-12, by Newton's method kept inside a bracket (a bisection on the
//! log scale where the step leaves it), and stops the alternation when `θ` and
//! the log likelihood have changed by at most `max(tolerance, 1e-10)` relative.
//! This is the maximum likelihood estimate. The floor of 1e-10 is the accuracy
//! to which `θ` can be located from a score that is a sum of terms rounded to
//! 1e-16, so a smaller tolerance cannot be met.
//!
//! # The identity link
//!
//! The identity link needs positive means at every iterate, and the
//! likelihood is not defined outside that region. Where the data keep the mean
//! well above 0, the default start (the Poisson fit of `y + 0.1`) works and
//! `MASS::glm.nb(link = identity)` agrees with this crate. With an offset the
//! first Poisson fit can leave the region, in R ("no valid set of coefficients
//! has been found") and here ([`GlmError::NoValidStart`]), and
//! [`NegBin::fit_from`] takes starting coefficients, as `glm.nb`'s `start`.
//!
//! # Large `θ` and the Poisson limit
//!
//! The terms `ψ(θ+y) - ψ(θ)` and `lnΓ(θ+y) - lnΓ(θ)` are differences of nearly
//! equal numbers for large `θ`. For integer `y` up to 50 they are computed as
//! `Σ_{j<y} 1/(θ+j)` and `Σ_{j<y} ln(θ+j)`, and the terms `ln θ - ln(θ+μ)` as
//! `-ln1p(μ/θ)`, which keeps the relative error of the score at about
//! `θ ε`. When the likelihood increases in `θ` up to `1e7`, the data are not
//! overdispersed, the estimate is not finite, and the fit is the Poisson fit
//! with [`NegBinFit::poisson_limit`] true and no `θ`, where R warns of an
//! "iteration limit reached" or an estimate "truncated at zero".
//!
//! # Likelihood ratio against the Poisson
//!
//! The Poisson is the boundary `1/θ = 0`, so the likelihood ratio statistic
//! `2 (l_NB - l_Poisson)` has asymptotically a 50:50 mixture of a point mass at
//! 0 and a chi-square with 1 degree of freedom \[3\], and the p-value is half
//! the chi-square tail. R does not report it.
//!
//! # References
//!
//! 1. Lawless JF. Negative binomial and mixed Poisson regression. Can J Stat.
//!    1987;15(3):209-25.
//! 2. Venables WN, Ripley BD. Modern Applied Statistics with S. 4th ed. New
//!    York: Springer; 2002. Section 7.4. The `MASS` functions `glm.nb` and
//!    `theta.ml`.
//! 3. Cameron AC, Trivedi PK. Regression Analysis of Count Data. 2nd ed.
//!    Cambridge: Cambridge University Press; 2013. Section 3.4.

use epistat_linalg::MatRef;
use epistat_stats::dist::chi_square::sf as chi_square_sf;
use epistat_stats::dist::gamma::{digamma, ln_gamma, trigamma};

use crate::{Family, Fit, Glm, GlmError, Link};

/// How `θ` is located.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThetaRule {
    /// The loops and stopping rules of `MASS::glm.nb` and `MASS::theta.ml`.
    RParity,
    /// The maximum likelihood estimate, to a relative 1e-10.
    Tight,
}

/// The controls of a negative binomial fit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NegBin {
    /// The link: [`Link::Log`] or [`Link::Identity`].
    pub link: Link,
    /// R's `epsilon`, for the fits of `β` and, under [`ThetaRule::RParity`],
    /// the alternation.
    pub tolerance: f64,
    /// R's `maxit`: iterations of each fit of `β`, steps of `theta.ml` and
    /// alternations.
    pub max_iterations: usize,
    /// A starting `θ`, so that the first fit is negative binomial instead of
    /// Poisson, as `init.theta`.
    pub init_theta: Option<f64>,
    /// How `θ` is located.
    pub theta_rule: ThetaRule,
}

impl NegBin {
    /// A log link, R's tolerance 1e-8 and 25 iterations, no starting `θ` and
    /// the tight rule.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            link: Link::Log,
            tolerance: 1e-8,
            max_iterations: 25,
            init_theta: None,
            theta_rule: ThetaRule::Tight,
        }
    }
}

impl Default for NegBin {
    fn default() -> Self {
        Self::new()
    }
}

/// A fitted negative binomial model.
#[derive(Clone, Debug)]
pub struct NegBinFit {
    /// The fit of the coefficients, with the dispersion fixed at 1 and the
    /// covariance given `θ`. When there is no `θ` it is the Poisson fit.
    pub glm: Fit,
    /// The shape `θ`, `None` when the likelihood is largest at infinity.
    pub theta: Option<f64>,
    /// The standard error of `θ` from the information at the estimate.
    pub theta_se: Option<f64>,
    /// The log likelihood at the estimate, the Poisson one when there is no `θ`.
    pub log_likelihood: f64,
    /// The number of alternations between `β` and `θ`.
    pub alternations: usize,
    /// Whether the alternation, the fits of `β` and the location of `θ` all
    /// converged.
    pub converged: bool,
    /// Whether the data show no overdispersion, so that the Poisson model is
    /// the maximum.
    pub poisson_limit: bool,
    /// The likelihood ratio statistic against the Poisson and its p-value,
    /// half the chi-square tail. `None` without a Poisson fit to compare with
    /// or when there is no `θ`.
    pub lr_vs_poisson: Option<(f64, f64)>,
}

const THETA_MAX: f64 = 1e7;
const THETA_MIN: f64 = 1e-10;
const INTEGER_LIMIT: f64 = 50.0;

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "y is a non-negative whole number no larger than 50 where it is cast"
)]
fn small_integer(y: f64) -> Option<usize> {
    ((0.0..=INTEGER_LIMIT).contains(&y) && y.fract() == 0.0).then_some(y as usize)
}

/// `ψ(θ+y) - ψ(θ)`.
fn psi_difference(theta: f64, y: f64) -> f64 {
    match small_integer(y) {
        Some(count) => (0..count).map(|j| 1.0 / (theta + j as f64)).sum(),
        None => digamma(theta + y) - digamma(theta),
    }
}

/// `ψ'(θ) - ψ'(θ+y)`.
fn trigamma_difference(theta: f64, y: f64) -> f64 {
    match small_integer(y) {
        Some(count) => (0..count)
            .map(|j| {
                let t = theta + j as f64;
                1.0 / (t * t)
            })
            .sum(),
        None => trigamma(theta) - trigamma(theta + y),
    }
}

/// `lnΓ(θ+y) - lnΓ(θ)`.
fn ln_gamma_difference(theta: f64, y: f64) -> f64 {
    match small_integer(y) {
        Some(count) => (0..count).map(|j| (theta + j as f64).ln()).sum(),
        None => ln_gamma(theta + y) - ln_gamma(theta),
    }
}

/// The negative binomial log likelihood with prior weights.
fn log_likelihood(theta: f64, y: &[f64], mu: &[f64], w: &[f64]) -> f64 {
    let mut total = 0.0;
    for i in 0..y.len() {
        if w[i] == 0.0 {
            continue;
        }
        let (yi, mi) = (y[i], mu[i]);
        let mut term =
            ln_gamma_difference(theta, yi) - ln_gamma(yi + 1.0) - theta * (mi / theta).ln_1p();
        if yi != 0.0 {
            term += yi * (mi / (theta + mi)).ln();
        }
        total += w[i] * term;
    }
    total
}

/// The Poisson log likelihood with prior weights.
fn poisson_log_likelihood(y: &[f64], mu: &[f64], w: &[f64]) -> f64 {
    let mut total = 0.0;
    for i in 0..y.len() {
        if w[i] == 0.0 {
            continue;
        }
        let mut term = -mu[i] - ln_gamma(y[i] + 1.0);
        if y[i] != 0.0 {
            term += y[i] * mu[i].ln();
        }
        total += w[i] * term;
    }
    total
}

/// The score and the information in `θ` at fixed means.
fn score_and_information(theta: f64, y: &[f64], mu: &[f64], w: &[f64]) -> (f64, f64) {
    let (mut score, mut information) = (0.0, 0.0);
    for i in 0..y.len() {
        if w[i] == 0.0 {
            continue;
        }
        let (yi, mi) = (y[i], mu[i]);
        let total = theta + mi;
        score += w[i] * (psi_difference(theta, yi) - (mi / theta).ln_1p() + (mi - yi) / total);
        information += w[i]
            * (trigamma_difference(theta, yi) - mi / (theta * total) + (mi - yi) / (total * total));
    }
    (score, information)
}

/// The moment start of `theta.ml`: `n / Σ w (y/μ - 1)²`.
fn moment_start(y: &[f64], mu: &[f64], w: &[f64], n: f64) -> f64 {
    let sum: f64 = (0..y.len())
        .map(|i| w[i] * (y[i] / mu[i] - 1.0) * (y[i] / mu[i] - 1.0))
        .sum();
    n / sum
}

/// What `MASS::theta.ml` returns.
struct ThetaMl {
    theta: f64,
    information: f64,
    limit_reached: bool,
}

/// `MASS::theta.ml`, step by step.
fn theta_ml(y: &[f64], mu: &[f64], w: &[f64], n: f64, limit: usize, eps: f64) -> ThetaMl {
    let mut theta = moment_start(y, mu, w, n);
    let mut iteration = 0;
    let mut delta = 1.0_f64;
    let mut information = f64::NAN;
    loop {
        iteration += 1;
        if !(iteration < limit && delta.abs() > eps) {
            break;
        }
        theta = theta.abs();
        let (score, info) = score_and_information(theta, y, mu, w);
        information = info;
        delta = score / info;
        theta += delta;
    }
    ThetaMl {
        theta,
        information,
        limit_reached: iteration == limit,
    }
}

/// Whether `theta.ml` found no estimate: it truncated at zero or went to
/// infinity, or it ran out of steps and the likelihood has no maximum below
/// the largest `θ` searched, where R warns.
fn no_estimate(result: &ThetaMl, y: &[f64], mu: &[f64], w: &[f64]) -> bool {
    if !(result.theta > 0.0 && result.theta.is_finite()) {
        return true;
    }
    result.limit_reached && matches!(theta_tight(y, mu, w, result.theta), Located::Limit)
}

enum Located {
    /// The estimate, the information there and whether the search converged.
    Finite(f64, f64, bool),
    /// The likelihood still increases at the largest `θ` searched.
    Limit,
}

/// The root of the score in `θ` at fixed means.
fn theta_tight(y: &[f64], mu: &[f64], w: &[f64], start: f64) -> Located {
    let mut theta = if start.is_finite() && start > 0.0 {
        start
    } else {
        1.0
    };
    let (mut low, mut high): (Option<f64>, Option<f64>) = (None, None);
    let mut information = f64::NAN;
    for _ in 0..80 {
        let (score, info) = score_and_information(theta, y, mu, w);
        information = info;
        if score > 0.0 {
            low = Some(theta);
        } else if score < 0.0 {
            high = Some(theta);
        } else {
            return Located::Finite(theta, info, true);
        }
        let newton = theta + score / info;
        let inside = newton.is_finite()
            && info > 0.0
            && newton > low.unwrap_or(0.0)
            && newton < high.unwrap_or(f64::INFINITY);
        let next = if inside {
            newton
        } else {
            match (low, high) {
                (Some(l), Some(h)) => (l * h).sqrt(),
                (Some(l), None) => l * 4.0,
                (None, Some(h)) => h / 4.0,
                (None, None) => theta,
            }
        };
        if next > THETA_MAX {
            return Located::Limit;
        }
        let next = next.max(THETA_MIN);
        let step = (next - theta).abs();
        if step <= 1e-12 * theta {
            let (_, info) = score_and_information(next, y, mu, w);
            return Located::Finite(next, info, true);
        }
        theta = next;
    }
    Located::Finite(theta, information, false)
}

impl NegBin {
    /// Fits the model to the counts `y` on the columns of `x`.
    ///
    /// `weights` are prior weights, which enter the likelihood as frequencies,
    /// so `θ` is estimated as if each row were repeated. `offset` is added to
    /// the linear predictor.
    ///
    /// # Errors
    ///
    /// [`GlmError::Link`] for a link other than log or identity,
    /// [`GlmError::Tolerance`] for a tolerance that is not positive and finite
    /// or no iterations, [`GlmError::Family`] for an `init_theta` that is not
    /// finite and positive, and the errors of [`Glm::fit`], such as
    /// [`GlmError::Response`] for a negative count.
    pub fn fit(
        &self,
        x: MatRef<'_, f64>,
        y: &[f64],
        weights: Option<&[f64]>,
        offset: Option<&[f64]>,
    ) -> Result<NegBinFit, GlmError> {
        self.fit_from(x, y, weights, offset, None)
    }

    /// Fits the model as [`fit`](Self::fit) does, with starting coefficients
    /// `start` (one per column of `x`) for the first fit, as the `start`
    /// argument of `glm.nb`. The identity link needs them when the default
    /// start, the fit of `y + 0.1`, gives a negative mean, as it can with an
    /// offset.
    ///
    /// # Errors
    ///
    /// The errors of [`fit`](Self::fit), and [`GlmError::Length`] when `start`
    /// does not have one entry per column, and [`GlmError::NoValidStart`] when
    /// the first fit leaves the domain of the link.
    pub fn fit_from(
        &self,
        x: MatRef<'_, f64>,
        y: &[f64],
        weights: Option<&[f64]>,
        offset: Option<&[f64]>,
        start: Option<&[f64]>,
    ) -> Result<NegBinFit, GlmError> {
        if !matches!(self.link, Link::Log | Link::Identity) {
            return Err(GlmError::Link);
        }
        if !(self.tolerance.is_finite() && self.tolerance > 0.0) || self.max_iterations == 0 {
            return Err(GlmError::Tolerance);
        }
        let n = y.len();
        let ones = vec![1.0; n];
        let w = weights.unwrap_or(&ones);
        let n_w: f64 = w.iter().sum();
        let glm = |family: Family| Glm {
            family,
            link: self.link,
            tolerance: self.tolerance,
            max_iterations: self.max_iterations,
            ridge: 0.0,
        };
        let start_of =
            |fit: &Fit| -> Vec<f64> { fit.coefficients.iter().map(|c| c.unwrap_or(0.0)).collect() };

        let first = match self.init_theta {
            None => glm(Family::Poisson).fit(x, y, weights, offset, start)?,
            Some(theta) => {
                glm(Family::NegativeBinomial { theta }).fit(x, y, weights, offset, start)?
            }
        };
        let poisson_ll = self
            .init_theta
            .is_none()
            .then(|| poisson_log_likelihood(y, &first.fitted, w));
        let poisson_limit = |first: Fit| NegBinFit {
            log_likelihood: poisson_log_likelihood(y, &first.fitted, w),
            glm: first,
            theta: None,
            theta_se: None,
            alternations: 0,
            converged: true,
            poisson_limit: true,
            lr_vs_poisson: None,
        };

        let mut mu = first.fitted.clone();
        let mut fit = first.clone();
        let mut converged = first.converged;
        let (theta, information, log_likelihood_value, alternations);
        match self.theta_rule {
            ThetaRule::RParity => {
                let eps = f64::EPSILON.powf(0.25);
                let initial = theta_ml(y, &mu, w, n_w, self.max_iterations, eps);
                if no_estimate(&initial, y, &mu, w) {
                    return Ok(poisson_limit(first));
                }
                let (mut th, mut info) = (initial.theta, initial.information);
                converged &= !initial.limit_reached;
                let d1 = (2.0 * (fit.residual_df as f64).max(1.0)).sqrt();
                let mut lm = log_likelihood(th, y, &mu, w);
                let mut lm0 = lm + 2.0 * d1;
                let mut delta = 1.0_f64;
                let mut iteration = 0;
                loop {
                    iteration += 1;
                    if iteration > self.max_iterations
                        || (lm0 - lm).abs() / d1 + delta.abs() <= self.tolerance
                    {
                        break;
                    }
                    fit = glm(Family::NegativeBinomial { theta: th }).fit(
                        x,
                        y,
                        weights,
                        offset,
                        Some(&start_of(&fit)),
                    )?;
                    converged &= fit.converged;
                    let previous = th;
                    let update = theta_ml(y, &mu, w, n_w, self.max_iterations, eps);
                    if no_estimate(&update, y, &mu, w) {
                        return Ok(poisson_limit(first));
                    }
                    th = update.theta;
                    info = update.information;
                    converged &= !update.limit_reached;
                    mu.clone_from(&fit.fitted);
                    delta = previous - th;
                    lm0 = lm;
                    lm = log_likelihood(th, y, &mu, w);
                }
                converged &= iteration <= self.max_iterations;
                theta = th;
                information = info;
                log_likelihood_value = lm;
                alternations = iteration - 1;
            }
            ThetaRule::Tight => {
                let tight_tolerance = self.tolerance.max(1e-10);
                let start = moment_start(y, &mu, w, n_w);
                let (mut th, mut info) = match theta_tight(y, &mu, w, start) {
                    Located::Limit => return Ok(poisson_limit(first)),
                    Located::Finite(th, info, ok) => {
                        converged &= ok;
                        (th, info)
                    }
                };
                let mut ll = log_likelihood(th, y, &mu, w);
                let mut count = 0;
                let mut done = false;
                while count < self.max_iterations {
                    count += 1;
                    fit = glm(Family::NegativeBinomial { theta: th }).fit(
                        x,
                        y,
                        weights,
                        offset,
                        Some(&start_of(&fit)),
                    )?;
                    converged &= fit.converged;
                    mu.clone_from(&fit.fitted);
                    let (new_th, new_info) = match theta_tight(y, &mu, w, th) {
                        Located::Limit => return Ok(poisson_limit(first)),
                        Located::Finite(t, i, ok) => {
                            converged &= ok;
                            (t, i)
                        }
                    };
                    let new_ll = log_likelihood(new_th, y, &mu, w);
                    let theta_change = (new_th - th).abs() / new_th;
                    let ll_change = (new_ll - ll).abs() / (new_ll.abs() + 0.1);
                    th = new_th;
                    info = new_info;
                    ll = new_ll;
                    if theta_change <= tight_tolerance && ll_change <= tight_tolerance {
                        done = true;
                        break;
                    }
                }
                converged &= done;
                theta = th;
                information = info;
                log_likelihood_value = ll;
                alternations = count;
            }
        }
        let lr_vs_poisson = poisson_ll.map(|p| {
            let statistic = (2.0 * (log_likelihood_value - p)).max(0.0);
            let p_value = if statistic > 0.0 {
                0.5 * chi_square_sf(statistic, 1.0)
            } else {
                1.0
            };
            (statistic, p_value)
        });
        Ok(NegBinFit {
            glm: fit,
            theta: Some(theta),
            theta_se: (information > 0.0).then(|| (1.0 / information).sqrt()),
            log_likelihood: log_likelihood_value,
            alternations,
            converged,
            poisson_limit: false,
            lr_vs_poisson,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_stable_differences_equal_the_direct_ones() {
        for theta in [0.3, 2.0, 40.0] {
            for y in [0.0, 1.0, 7.0, 50.0, 3.5, 80.0] {
                let direct = digamma(theta + y) - digamma(theta);
                assert!((psi_difference(theta, y) - direct).abs() < 1e-11 * direct.abs().max(1.0));
                let direct = ln_gamma(theta + y) - ln_gamma(theta);
                assert!(
                    (ln_gamma_difference(theta, y) - direct).abs() < 1e-11 * direct.abs().max(1.0)
                );
                let direct = trigamma(theta) - trigamma(theta + y);
                assert!(
                    (trigamma_difference(theta, y) - direct).abs() < 1e-11 * direct.abs().max(1.0)
                );
            }
        }
    }

    #[test]
    fn the_score_is_the_derivative_of_the_log_likelihood() {
        let y = [0.0, 1.0, 3.0, 8.0, 0.0, 2.0];
        let mu = [0.7, 1.2, 2.5, 6.0, 0.3, 2.2];
        let w = [1.0, 2.0, 1.0, 0.5, 1.0, 1.0];
        let theta = 1.7;
        let h = 1e-6;
        let numeric = (log_likelihood(theta + h, &y, &mu, &w)
            - log_likelihood(theta - h, &y, &mu, &w))
            / (2.0 * h);
        let (score, information) = score_and_information(theta, &y, &mu, &w);
        assert!((numeric - score).abs() < 1e-7, "{numeric} {score}");
        let (up, _) = score_and_information(theta + h, &y, &mu, &w);
        let (down, _) = score_and_information(theta - h, &y, &mu, &w);
        assert!((-(up - down) / (2.0 * h) - information).abs() < 1e-6);
    }

    #[test]
    fn a_response_without_overdispersion_is_the_poisson_limit() {
        // Counts that are less dispersed than the Poisson.
        let y: Vec<f64> = (0..60).map(|i| f64::from(3 + (i % 2))).collect();
        let x = epistat_linalg::Mat::from_fn(60, 1, |_, _| 1.0);
        let fit = NegBin::new().fit(x.as_ref(), &y, None, None).unwrap();
        assert!(fit.poisson_limit);
        assert!(fit.theta.is_none());
        let fit = NegBin {
            theta_rule: ThetaRule::RParity,
            ..NegBin::new()
        }
        .fit(x.as_ref(), &y, None, None)
        .unwrap();
        assert!(fit.poisson_limit);
    }

    #[test]
    fn invalid_arguments_are_errors() {
        let x = epistat_linalg::Mat::from_fn(3, 1, |_, _| 1.0);
        assert_eq!(
            NegBin {
                link: Link::Logit,
                ..NegBin::new()
            }
            .fit(x.as_ref(), &[1.0; 3], None, None)
            .unwrap_err(),
            GlmError::Link
        );
        assert!(matches!(
            NegBin::new().fit(x.as_ref(), &[1.0, -1.0, 2.0], None, None),
            Err(GlmError::Response { .. })
        ));
    }
}
