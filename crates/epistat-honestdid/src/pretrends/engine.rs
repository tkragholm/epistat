//! The engine of the pre-trends power.
//!
//! Event-study estimates `betahat` are approximately `N(beta, Sigma)`, with
//! `beta = tau + delta`: the treatment effects plus a violation of parallel
//! trends. The usual pre-test passes when no pre-period coefficient is
//! individually significant at level `alpha`. Roth \[1\] shows that this test
//! often has little power against violations that matter, and that the
//! estimates of those who pass are distorted. Given a hypothesised `delta`
//! over the coefficients, this module computes, as the R package `pretrends`
//! does:
//!
//! - the probability that the test passes and its power, [`PretrendAnalysis`];
//! - the means of the pre- and post-period coefficients conditional on
//!   passing, and the bias of the post-period ones;
//! - the likelihood ratio of the observed pre-period coefficients under
//!   `delta` against parallel trends, and the Bayes factor, the ratio of the
//!   probabilities of passing;
//! - the slope of a linear violation `delta_t = slope * (t - reference)`
//!   against which the test has a given power, [`slope_for_power`].
//!
//! # Algorithm
//!
//! Let `X` be the pre-period block, `X ~ N(mu, S)` with `mu = delta_pre` and
//! `S = Sigma_pre`, and `R = prod_j [-z s_j, z s_j]` the pass region, with
//! `s_j = sqrt(S_jj)` and `z = Phi^{-1}(1 - alpha / 2)`.
//!
//! ```text
//! pass      = P(X in R)
//! power     = 1 - pass
//! BF        = pass(delta) / pass(0)
//! LR        = N(x_obs; mu, S) / N(x_obs; 0, S)
//! ```
//!
//! The rectangle probability is computed by the transformation of Genz \[2\]:
//! with the Cholesky factor `C` of the correlation matrix and the variables in
//! the order that makes the early limits narrowest, the probability is the
//! integral over the unit cube of `prod_i (e_i - d_i)`, where `d_i, e_i` are
//! the normal probabilities of the limits shifted by the earlier, inverted
//! variables. The last variable is integrated exactly, so the cube has
//! dimension `p - 1`. The integral is evaluated by a randomly shifted
//! rank-1 Korobov lattice \[3\], with the points mapped by the tent
//! transform and, up to four integration variables, by the polynomial
//! `x^3 (10 - 15 x + 6 x^2)` with its weight, which makes the integrand
//! smooth and periodic. For each prime number
//! of points the generator is chosen from a fixed candidate list by the
//! weighted `P_2` criterion, and the shifts come from a stream with a fixed
//! seed, so the result is a deterministic function of the inputs. The error is
//! the standard error of the mean over the shifts. The number of points grows
//! through primes up to `2^18` until the error is below
//! [`PretrendConfig::abs_tol`]. One and zero dimensions are exact.
//!
//! The conditional mean of the pre block is Tallis's formula \[4\],
//!
//! ```text
//! E[X_k | X in R] = mu_k + (1 / pass) * sum_i S_ki
//!     ( f_i(-z s_i) P(X_{-i} in R_{-i} | X_i = -z s_i)
//!     - f_i( z s_i) P(X_{-i} in R_{-i} | X_i =  z s_i) ),
//! ```
//!
//! with `f_i` the `N(mu_i, S_ii)` density, so that `2p` rectangle
//! probabilities of dimension `p - 1` are needed. The post block follows from
//! the regression on the pre block,
//! `E[Y | X in R] = beta_post + Sigma_post,pre S^{-1} (E[X | X in R] - mu)`.
//!
//! The slope for a given power solves `power(slope * (t - reference)) =
//! target` on `[0, 8 max_j s_j]` by the Illinois method. The error of the
//! solution in power is the error of the evaluation.
//!
//! # References
//!
//! 1. Roth J. Pretest with caution: event-study estimates after testing for
//!    parallel trends. Am Econ Rev Insights 2022;4(3):305-22.
//!    doi:10.1257/aeri.20210236
//! 2. Genz A. Numerical computation of multivariate normal probabilities. J
//!    Comput Graph Stat 1992;1(2):141-9.
//! 3. Nuyens D, Cools R. Fast algorithms for component-by-component
//!    construction of rank-1 lattice rules. Math Comp 2006;75:903-20.
//! 4. Tallis GM. The moment generating function of the truncated
//!    multi-normal distribution. J R Stat Soc B 1961;23(1):223-9.

use epistat_core::{InputError, Level};
use epistat_rng::Stream;
use epistat_stats::dist::normal;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::error::Error;

/// Numerical settings of the rectangle probabilities.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PretrendConfig {
    /// Level of the test of each pre-period coefficient, which it passes
    /// inside `+- Phi^{-1}(1 - alpha / 2)` standard errors.
    pub level: Level,
    /// Critical value of the t-statistic of each pre-period coefficient. When
    /// `None` it is `Phi^{-1}(1 - alpha / 2)`; the R package uses 1.96.
    pub threshold: Option<f64>,
    /// The lattice grows until the standard error of a rectangle probability
    /// is at most this.
    pub abs_tol: f64,
    /// Number of random shifts.
    pub shifts: usize,
    /// Seed of the shifts.
    pub seed: u64,
}

#[cfg(test)]
impl PretrendConfig {
    /// The level `alpha = 0.05`, no threshold, tolerance `1e-7` and 12 shifts
    /// from `seed`.
    pub(crate) fn with_seed(seed: u64) -> Self {
        Self {
            level: Level::alpha(0.05).expect("0.05 is in (0, 1)"),
            threshold: None,
            abs_tol: 1e-7,
            shifts: 12,
            seed,
        }
    }
}

/// A value with the standard error of its numerical evaluation.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Evaluated {
    /// The value.
    pub value: f64,
    /// Standard error of the evaluation; 0 when it is exact.
    pub error: f64,
}

/// An event study with the pre-period coefficients first.
#[derive(Debug, Clone, PartialEq)]
pub struct PretrendInput {
    /// Coefficients, pre-treatment ones first. The reference period is not
    /// among them.
    pub betahat: Vec<f64>,
    /// Covariance matrix of `betahat`, one row per coefficient.
    pub covariance: Vec<Vec<f64>>,
    /// Relative time of each pre-period coefficient.
    pub pre_periods: Vec<f64>,
    /// Relative time of each post-period coefficient.
    pub post_periods: Vec<f64>,
}

impl PretrendInput {
    /// Number of pre-period coefficients.
    #[must_use]
    pub fn n_pre(&self) -> usize {
        self.pre_periods.len()
    }

    /// The violation `slope * (t - reference_period)` at every coefficient.
    #[must_use]
    pub fn linear_trend(&self, slope: f64, reference_period: f64) -> Vec<f64> {
        self.pre_periods
            .iter()
            .chain(&self.post_periods)
            .map(|t| slope * (t - reference_period))
            .collect()
    }

    pub(super) fn validate(&self) -> Result<(), Error> {
        use epistat_core::check;
        let k = self.pre_periods.len() + self.post_periods.len();
        check::non_empty("pre-treatment coefficients", self.pre_periods.len())?;
        check::same_length("betahat", k, self.betahat.len())?;
        check::same_length("covariance rows", k, self.covariance.len())?;
        for row in &self.covariance {
            check::same_length("covariance columns", k, row.len())?;
        }
        check::finite("betahat", &self.betahat)?;
        for (i, row) in self.covariance.iter().enumerate() {
            if let Some(j) = row.iter().position(|v| !v.is_finite()) {
                return Err(InputError::NotFinite {
                    what: "covariance",
                    row: i,
                    column: Some(j),
                }
                .into());
            }
        }
        for i in 0..k {
            for j in 0..i {
                let scale = self.covariance[i][i]
                    .abs()
                    .max(self.covariance[j][j].abs())
                    .max(f64::MIN_POSITIVE);
                if (self.covariance[i][j] - self.covariance[j][i]).abs() > 1e-8 * scale {
                    return Err(InputError::Domain {
                        what: "covariance",
                        value: self.covariance[i][j],
                        domain: "symmetric",
                        row: Some(i),
                    }
                    .into());
                }
            }
        }
        Ok(())
    }
}

/// The result of [`analyse_pretest`].
#[derive(Debug, Clone, PartialEq)]
pub struct PretrendAnalysis {
    /// Probability that the pre-test passes under the hypothesised violation.
    pub pass_probability: Evaluated,
    /// Probability that it passes under parallel trends.
    pub pass_probability_null: Evaluated,
    /// `1 - pass_probability`.
    pub power: Evaluated,
    /// `pass_probability / pass_probability_null`.
    pub bayes_factor: Evaluated,
    /// Likelihood of the observed pre-period coefficients under the
    /// hypothesised violation over that under parallel trends.
    pub likelihood_ratio: f64,
    /// Mean of each pre-period coefficient given that the test passes.
    pub mean_pre_after_pretest: Vec<Evaluated>,
    /// Mean of each post-period coefficient given that the test passes.
    pub mean_post_after_pretest: Vec<Evaluated>,
    /// Mean after passing minus the mean of the post-period coefficients
    /// before the test, which is the hypothesised value.
    pub post_bias: Vec<Evaluated>,
}

/// Power, conditional means, Bayes factor and likelihood ratio for the
/// hypothesised violation `delta`, one entry per coefficient of `input`.
///
/// The coefficients are taken to have mean `delta`, that is zero treatment
/// effect.
///
/// # Errors
///
/// An error if the input is malformed or the pre-period covariance is not
/// positive definite.
pub fn analyse_pretest(
    input: &PretrendInput,
    delta: &[f64],
    config: &PretrendConfig,
) -> Result<PretrendAnalysis, Error> {
    input.validate()?;
    validate_config(config)?;
    let k = input.betahat.len();
    epistat_core::check::same_length("violation", k, delta.len())?;
    epistat_core::check::finite("violation", delta)?;
    let p = input.n_pre();
    let cov_pre: Vec<Vec<f64>> = input.covariance[..p]
        .iter()
        .map(|r| r[..p].to_vec())
        .collect();
    let z = critical_value(config);
    let delta_pre = &delta[..p];
    let zero = vec![0.0; p];

    let pass = rect_prob(&box_limits(&cov_pre, z), delta_pre, &cov_pre, config)?;
    let pass0 = rect_prob(&box_limits(&cov_pre, z), &zero, &cov_pre, config)?;
    let moments = conditional_mean(delta_pre, &cov_pre, z, config)?;

    // Post block: regression on the pre block.
    let chol = cholesky(&cov_pre).ok_or(Error::NotPositiveDefinite {
        what: "pre-period block",
    })?;
    let shift: Vec<f64> = moments
        .iter()
        .zip(delta_pre)
        .map(|(m, d)| m.value - d)
        .collect();
    let shift_error: Vec<f64> = moments.iter().map(|m| m.error).collect();
    let weights = chol_solve(&chol, &shift);
    let abs_weights_err = chol_solve_abs_error(&cov_pre, &chol, &shift_error);
    let mut mean_post = Vec::new();
    let mut bias = Vec::new();
    for r in p..k {
        let cross = &input.covariance[r][..p];
        let adj: f64 = cross.iter().zip(&weights).map(|(c, w)| c * w).sum();
        let err: f64 = cross
            .iter()
            .zip(&abs_weights_err)
            .map(|(c, e)| c.abs() * e)
            .sum();
        mean_post.push(Evaluated {
            value: delta[r] + adj,
            error: err,
        });
        bias.push(Evaluated {
            value: adj,
            error: err,
        });
    }

    let x = &input.betahat[..p];
    let r_alt: Vec<f64> = x.iter().zip(delta_pre).map(|(a, b)| a - b).collect();
    let q_alt = quad_form(&chol, &r_alt);
    let q_null = quad_form(&chol, x);
    let likelihood_ratio = (-0.5 * (q_alt - q_null)).exp();

    let bf_value = pass.value / pass0.value;
    let bf_error = bf_value * (pass.error / pass.value + pass0.error / pass0.value);
    Ok(PretrendAnalysis {
        power: Evaluated {
            value: 1.0 - pass.value,
            error: pass.error,
        },
        pass_probability: pass,
        pass_probability_null: pass0,
        bayes_factor: Evaluated {
            value: bf_value,
            error: bf_error,
        },
        likelihood_ratio,
        mean_pre_after_pretest: moments,
        mean_post_after_pretest: mean_post,
        post_bias: bias,
    })
}

/// The probability that no pre-period coefficient is significant when the
/// pre-period coefficients have mean `delta_pre` and covariance `cov_pre`.
///
/// # Errors
///
/// An error if the inputs are malformed or `cov_pre` is not positive definite.
pub fn pretest_pass_probability(
    delta_pre: &[f64],
    cov_pre: &[Vec<f64>],
    config: &PretrendConfig,
) -> Result<Evaluated, Error> {
    validate_config(config)?;
    let p = delta_pre.len();
    epistat_core::check::non_empty("pre-period mean", p)?;
    epistat_core::check::same_length("pre-period covariance rows", p, cov_pre.len())?;
    for row in cov_pre {
        epistat_core::check::same_length("pre-period covariance columns", p, row.len())?;
    }
    let z = critical_value(config);
    rect_prob(&box_limits(cov_pre, z), delta_pre, cov_pre, config)
}

/// The size of the linear violation `slope * (t - reference_period)` against
/// which the pre-test has power `target_power`.
///
/// Returns the slope, positive, with the standard error of the power at that
/// slope. `input.pre_periods` give `t`; only the pre-period covariance is
/// used.
///
/// # Errors
///
/// An error if the input is malformed, `target_power` is not in `(0, 1)`, or
/// the power at `8 * max_j s_j` is below the target.
pub fn slope_for_power(
    input: &PretrendInput,
    reference_period: f64,
    target_power: f64,
    config: &PretrendConfig,
) -> Result<Evaluated, Error> {
    validate_config(config)?;
    if !(target_power > 0.0 && target_power < 1.0) {
        return Err(InputError::Domain {
            what: "power",
            value: target_power,
            domain: "(0, 1)",
            row: None,
        }
        .into());
    }
    let p = input.n_pre();
    epistat_core::check::non_empty("pre-treatment coefficients", p)?;
    if let Some(found) = std::iter::once(input.covariance.len())
        .chain(input.covariance.iter().take(p).map(Vec::len))
        .find(|&len| len < p)
    {
        return Err(InputError::Length {
            what: "covariance",
            expected: p,
            found,
        }
        .into());
    }
    let cov_pre: Vec<Vec<f64>> = input.covariance[..p]
        .iter()
        .map(|r| r[..p].to_vec())
        .collect();
    let rel: Vec<f64> = input
        .pre_periods
        .iter()
        .map(|t| t - reference_period)
        .collect();
    let power = |slope: f64| -> Result<Evaluated, Error> {
        let mean: Vec<f64> = rel.iter().map(|t| t * slope).collect();
        let pass = pretest_pass_probability(&mean, &cov_pre, config)?;
        Ok(Evaluated {
            value: 1.0 - pass.value,
            error: pass.error,
        })
    };
    let max_se = cov_pre
        .iter()
        .enumerate()
        .map(|(i, r)| r[i].sqrt())
        .fold(0.0, f64::max);
    let upper = 8.0 * max_se;
    let (mut lo, mut hi) = (0.0, upper);
    let mut f_lo = power(lo)?.value - target_power;
    let top = power(hi)?;
    let mut f_hi = top.value - target_power;
    if f_lo > 0.0 || f_hi < 0.0 {
        return Err(Error::NoSlope {
            target: target_power,
            upper,
            power_at_upper: top.value,
        });
    }
    let mut side = 0;
    let mut best = Evaluated {
        value: hi,
        error: top.error,
    };
    for _ in 0..200 {
        let x = (lo * f_hi - hi * f_lo) / (f_hi - f_lo);
        let fx = power(x)?;
        let f = fx.value - target_power;
        best = Evaluated {
            value: x,
            error: fx.error,
        };
        if f.abs() < 1e-12 || (hi - lo) < 1e-14 * upper {
            break;
        }
        if f < 0.0 {
            lo = x;
            f_lo = f;
            if side == -1 {
                f_hi /= 2.0;
            }
            side = -1;
        } else {
            hi = x;
            f_hi = f;
            if side == 1 {
                f_lo /= 2.0;
            }
            side = 1;
        }
    }
    Ok(best)
}

fn critical_value(config: &PretrendConfig) -> f64 {
    config
        .threshold
        .unwrap_or_else(|| normal::quantile(1.0 - config.level.alpha_value() / 2.0))
}

pub(super) fn validate_config(config: &PretrendConfig) -> Result<(), Error> {
    let domain = |what, value, domain| -> Error {
        InputError::Domain {
            what,
            value,
            domain,
            row: None,
        }
        .into()
    };
    if let Some(threshold) = config.threshold
        && !(threshold > 0.0 && threshold.is_finite())
    {
        return Err(domain("threshold", threshold, "positive and finite"));
    }
    if config.abs_tol <= 0.0 || config.abs_tol.is_nan() {
        return Err(domain("tolerance", config.abs_tol, "positive"));
    }
    if config.shifts < 2 {
        return Err(domain("shifts", config.shifts as f64, "at least 2"));
    }
    Ok(())
}

// ---------------------------------------------------------------- linear algebra

fn cholesky(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i][j];
            for m in 0..j {
                s -= l[i][m] * l[j][m];
            }
            if i == j {
                if s <= 0.0 || s.is_nan() {
                    return None;
                }
                l[i][i] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    Some(l)
}

fn chol_solve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = l.len();
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for m in 0..i {
            s -= l[i][m] * y[m];
        }
        y[i] = s / l[i][i];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for m in i + 1..n {
            s -= l[m][i] * x[m];
        }
        x[i] = s / l[i][i];
    }
    x
}

/// Bound on the error of `S^{-1} e` given componentwise errors `e`, by the
/// absolute values of the inverse's columns.
fn chol_solve_abs_error(cov: &[Vec<f64>], l: &[Vec<f64>], err: &[f64]) -> Vec<f64> {
    let n = cov.len();
    let mut out = vec![0.0; n];
    for (j, e) in err.iter().enumerate() {
        let mut unit = vec![0.0; n];
        unit[j] = 1.0;
        let col = chol_solve(l, &unit);
        for i in 0..n {
            out[i] += col[i].abs() * e;
        }
    }
    out
}

fn quad_form(l: &[Vec<f64>], r: &[f64]) -> f64 {
    let n = l.len();
    let mut y = vec![0.0; n];
    let mut q = 0.0;
    for i in 0..n {
        let mut s = r[i];
        for m in 0..i {
            s -= l[i][m] * y[m];
        }
        y[i] = s / l[i][i];
        q += y[i] * y[i];
    }
    q
}

// ---------------------------------------------------------------- rectangle probabilities

fn box_limits(cov: &[Vec<f64>], z: f64) -> Vec<(f64, f64)> {
    cov.iter()
        .enumerate()
        .map(|(i, r)| (-z * r[i].sqrt(), z * r[i].sqrt()))
        .collect()
}

/// `P(lower <= X <= upper)` for `X ~ N(mean, cov)`.
fn rect_prob(
    limits: &[(f64, f64)],
    mean: &[f64],
    cov: &[Vec<f64>],
    config: &PretrendConfig,
) -> Result<Evaluated, Error> {
    let p = mean.len();
    let sd: Vec<f64> = (0..p).map(|i| cov[i][i].sqrt()).collect();
    if sd.iter().any(|s| *s <= 0.0 || s.is_nan()) {
        return Err(Error::NotPositiveDefinite {
            what: "pre-period block",
        });
    }
    let corr: Vec<Vec<f64>> = (0..p)
        .map(|i| (0..p).map(|j| cov[i][j] / (sd[i] * sd[j])).collect())
        .collect();
    let a: Vec<f64> = (0..p).map(|i| (limits[i].0 - mean[i]) / sd[i]).collect();
    let b: Vec<f64> = (0..p).map(|i| (limits[i].1 - mean[i]) / sd[i]).collect();
    genz(&corr, &a, &b, config)
}

fn phi_diff(lo: f64, hi: f64) -> f64 {
    let d = if lo > 0.0 {
        normal::cdf(-lo) - normal::cdf(-hi)
    } else {
        normal::cdf(hi) - normal::cdf(lo)
    };
    d.max(0.0)
}

/// `Phi^{-1}` of the point at fraction `w` between `Phi(lo)` and `Phi(hi)`,
/// computed in the tail that keeps precision.
fn trunc_inv(lo: f64, hi: f64, w: f64, diff: f64) -> f64 {
    if lo > 0.0 {
        -normal::quantile(normal::cdf(-hi) + (1.0 - w) * diff)
    } else {
        normal::quantile(normal::cdf(lo) + w * diff)
    }
}

struct Prepared {
    c: Vec<Vec<f64>>,
    a: Vec<f64>,
    b: Vec<f64>,
}

fn reorder(corr: &[Vec<f64>], a0: &[f64], b0: &[f64]) -> Result<Prepared, Error> {
    let n = corr.len();
    let mut cov = corr.to_vec();
    let mut a = a0.to_vec();
    let mut b = b0.to_vec();
    let mut c = vec![vec![0.0; n]; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut best = i;
        let mut best_range = f64::INFINITY;
        for j in i..n {
            let mut s = 0.0;
            let mut var = cov[j][j];
            for m in 0..i {
                s += c[j][m] * y[m];
                var -= c[j][m] * c[j][m];
            }
            if var <= 1e-14 || var.is_nan() {
                continue;
            }
            let cjj = var.sqrt();
            let range = phi_diff((a[j] - s) / cjj, (b[j] - s) / cjj);
            if range < best_range {
                best_range = range;
                best = j;
            }
        }
        if !best_range.is_finite() {
            return Err(Error::NotPositiveDefinite {
                what: "pre-period block",
            });
        }
        if best != i {
            cov.swap(i, best);
            for row in &mut cov {
                row.swap(i, best);
            }
            a.swap(i, best);
            b.swap(i, best);
            c.swap(i, best);
        }
        let mut var = cov[i][i];
        let mut s = 0.0;
        for m in 0..i {
            var -= c[i][m] * c[i][m];
            s += c[i][m] * y[m];
        }
        let cii = var.sqrt();
        c[i][i] = cii;
        for r in i + 1..n {
            let mut v = cov[r][i];
            for m in 0..i {
                v -= c[r][m] * c[i][m];
            }
            c[r][i] = v / cii;
        }
        let lo = (a[i] - s) / cii;
        let hi = (b[i] - s) / cii;
        let range = phi_diff(lo, hi);
        y[i] = if range > 1e-300 {
            (normal::pdf(lo) - normal::pdf(hi)) / range
        } else if lo.is_finite() && hi.is_finite() {
            f64::midpoint(lo, hi)
        } else {
            0.0
        };
    }
    Ok(Prepared { c, a, b })
}

fn integrand(pr: &Prepared, w: &[f64]) -> f64 {
    let n = pr.a.len();
    let mut y = [0.0_f64; 64];
    let mut f = 1.0;
    for i in 0..n {
        let mut s = 0.0;
        for m in 0..i {
            s += pr.c[i][m] * y[m];
        }
        let lo = (pr.a[i] - s) / pr.c[i][i];
        let hi = (pr.b[i] - s) / pr.c[i][i];
        let diff = phi_diff(lo, hi);
        f *= diff;
        if i + 1 < n {
            if f == 0.0 {
                return 0.0;
            }
            y[i] = trunc_inv(lo, hi, w[i], diff);
        }
    }
    f
}

const PRIMES: [usize; 10] = [
    509, 1021, 2039, 4093, 8191, 16381, 32749, 65521, 131_071, 262_139,
];
const MAX_DIM: usize = 63;
/// Up to this many integration variables the lattice points are mapped
/// through a polynomial that makes the integrand smooth and periodic (weight
/// `30 x^2 (1-x)^2`); with more, the weights inflate the variance and the
/// tent map of the points is used.
const SMOOTH_MAX_DIM: usize = 4;

fn lattice_cache() -> &'static Mutex<HashMap<usize, usize>> {
    static CACHE: OnceLock<Mutex<HashMap<usize, usize>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Number of coordinates the generator is judged on.
const CRITERION_DIM: usize = 12;

/// Generating vector `(1, a, a^2, ...) mod n` of a Korobov lattice, with `a`
/// chosen among 24 fixed candidates by the weighted `P_2` criterion over the
/// first 12 coordinates with weights `1 / j^2`. The generator depends on `n`
/// only, so every dimension uses its prefix.
fn korobov(n: usize, dim: usize) -> Vec<usize> {
    let cached = lattice_cache().lock().ok().and_then(|g| g.get(&n).copied());
    let a = cached.unwrap_or_else(|| {
        let weights: Vec<f64> = (1..=CRITERION_DIM)
            .map(|j| 2.0 * std::f64::consts::PI.powi(2) / (j * j) as f64)
            .collect();
        let golden = 0.618_033_988_749_894_9;
        let mut best_a = 1;
        let mut best_crit = f64::INFINITY;
        for cand in 1..=24_usize {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "the product is in [0, n - 2)"
            )]
            let a = 1 + (((cand as f64 * golden).fract() * (n - 2) as f64) as usize).min(n - 3);
            let gens = powers(a, CRITERION_DIM, n);
            let mut sum = 0.0;
            for k in 0..n {
                let mut prod = 1.0;
                for (j, g) in gens.iter().enumerate() {
                    let x = ((k * g) % n) as f64 / n as f64;
                    prod *= 1.0 + weights[j] * (x * x - x + 1.0 / 6.0);
                }
                sum += prod;
            }
            let crit = sum / n as f64 - 1.0;
            if crit < best_crit {
                best_crit = crit;
                best_a = a;
            }
        }
        if let Ok(mut g) = lattice_cache().lock() {
            g.insert(n, best_a);
        }
        best_a
    });
    powers(a, dim, n)
}

fn powers(a: usize, dim: usize, n: usize) -> Vec<usize> {
    let mut v = Vec::with_capacity(dim);
    let mut g = 1;
    for _ in 0..dim {
        v.push(g);
        g = (g * a) % n;
    }
    v
}

fn genz(
    corr: &[Vec<f64>],
    a: &[f64],
    b: &[f64],
    config: &PretrendConfig,
) -> Result<Evaluated, Error> {
    let n = corr.len();
    if n > MAX_DIM {
        return Err(InputError::TooLarge {
            what: "pre-period coefficients",
            limit: MAX_DIM as u64,
            found: n as u64,
        }
        .into());
    }
    if n == 0 {
        return Ok(Evaluated {
            value: 1.0,
            error: 0.0,
        });
    }
    if n == 1 {
        return Ok(Evaluated {
            value: phi_diff(a[0], b[0]),
            error: 0.0,
        });
    }
    let pr = reorder(corr, a, b)?;
    let dim = n - 1;
    let mut stream = Stream::seeded(config.seed);
    let shifts: Vec<Vec<f64>> = (0..config.shifts)
        .map(|_| (0..dim).map(|_| stream.uniform()).collect())
        .collect();
    let mut result = Evaluated {
        value: f64::NAN,
        error: f64::INFINITY,
    };
    let mut w = vec![0.0; dim];
    let smooth = dim <= SMOOTH_MAX_DIM;
    for &np in &PRIMES {
        let gens = korobov(np, dim);
        let mut estimates = Vec::with_capacity(config.shifts);
        for shift in &shifts {
            let mut sum = 0.0;
            for k in 0..np {
                let mut weight = 1.0;
                for j in 0..dim {
                    let u = (((k * gens[j]) % np) as f64 / np as f64 + shift[j]).fract();
                    let x = (2.0 * u - 1.0).abs();
                    if smooth {
                        w[j] = x * x * x * (10.0 - 15.0 * x + 6.0 * x * x);
                        weight *= 30.0 * x * x * (1.0 - x) * (1.0 - x);
                    } else {
                        w[j] = x;
                    }
                }
                sum += weight * integrand(&pr, &w);
            }
            estimates.push(sum / np as f64);
        }
        let m = estimates.len() as f64;
        let mean = estimates.iter().sum::<f64>() / m;
        let var = estimates.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / (m - 1.0);
        result = Evaluated {
            value: mean,
            error: (var / m).sqrt(),
        };
        if result.error <= config.abs_tol {
            break;
        }
    }
    Ok(result)
}

// ---------------------------------------------------------------- conditional mean

/// Mean of `X ~ N(mean, cov)` given `X` in the significance box, by Tallis's
/// formula.
fn conditional_mean(
    mean: &[f64],
    cov: &[Vec<f64>],
    z: f64,
    config: &PretrendConfig,
) -> Result<Vec<Evaluated>, Error> {
    let p = mean.len();
    let limits = box_limits(cov, z);
    let pass = rect_prob(&limits, mean, cov, config)?;
    if pass.value <= 0.0 || pass.value.is_nan() {
        return Err(Error::PassProbabilityZero);
    }
    // g_i = f_i(a_i) P_i(a_i) - f_i(b_i) P_i(b_i), with its error bound.
    let mut g = vec![0.0; p];
    let mut g_err = vec![0.0; p];
    for i in 0..p {
        let s2 = cov[i][i];
        let others: Vec<usize> = (0..p).filter(|&j| j != i).collect();
        for (bound, sign) in [(limits[i].0, 1.0), (limits[i].1, -1.0)] {
            let dens = normal::pdf((bound - mean[i]) / s2.sqrt()) / s2.sqrt();
            let prob = if others.is_empty() {
                Evaluated {
                    value: 1.0,
                    error: 0.0,
                }
            } else {
                let cond_mean: Vec<f64> = others
                    .iter()
                    .map(|&j| mean[j] + cov[j][i] / s2 * (bound - mean[i]))
                    .collect();
                let cond_cov: Vec<Vec<f64>> = others
                    .iter()
                    .map(|&r| {
                        others
                            .iter()
                            .map(|&c| cov[r][c] - cov[r][i] * cov[c][i] / s2)
                            .collect()
                    })
                    .collect();
                let lim: Vec<(f64, f64)> = others.iter().map(|&j| limits[j]).collect();
                rect_prob(&lim, &cond_mean, &cond_cov, config)?
            };
            g[i] += sign * dens * prob.value;
            g_err[i] += dens * prob.error;
        }
    }
    Ok((0..p)
        .map(|k| {
            let tally: f64 = (0..p).map(|i| cov[k][i] * g[i]).sum();
            let tally_err: f64 = (0..p).map(|i| cov[k][i].abs() * g_err[i]).sum();
            let shift = tally / pass.value;
            let error = tally_err / pass.value + shift.abs() * pass.error / pass.value;
            Evaluated {
                value: mean[k] + shift,
                error,
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_dimension_is_exact() {
        let cov = vec![vec![4.0]];
        let p =
            pretest_pass_probability(&[1.0], &cov, &PretrendConfig::with_seed(20_260_125)).unwrap();
        let z = normal::quantile(0.975);
        let want = normal::cdf(z - 0.5) - normal::cdf(-z - 0.5);
        assert!((p.value - want).abs() < 1e-15);
        assert_eq!(p.error, 0.0);
    }

    #[test]
    fn independent_coordinates_factorise() {
        let cov = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 2.0, 0.0],
            vec![0.0, 0.0, 0.5],
        ];
        let mean = [0.3, -0.5, 0.2];
        let cfg = PretrendConfig::with_seed(20_260_125);
        let p = pretest_pass_probability(&mean, &cov, &cfg).unwrap();
        let z = normal::quantile(0.975);
        let want: f64 = (0..3)
            .map(|i| {
                let s = cov[i][i].sqrt();
                normal::cdf(z - mean[i] / s) - normal::cdf(-z - mean[i] / s)
            })
            .product();
        assert!(
            (p.value - want).abs() < 6.0 * p.error + 1e-12,
            "{} {} {}",
            p.value,
            want,
            p.error
        );
    }

    #[test]
    fn independent_conditional_mean_is_marginal_truncated_mean() {
        let cov = vec![vec![1.0, 0.0], vec![0.0, 2.0]];
        let mean = [0.7, -0.4];
        let z = normal::quantile(0.975);
        let m = conditional_mean(&mean, &cov, z, &PretrendConfig::with_seed(20_260_125)).unwrap();
        for i in 0..2 {
            let s = cov[i][i].sqrt();
            let (lo, hi) = ((-z * s - mean[i]) / s, (z * s - mean[i]) / s);
            let want = mean[i]
                + s * (normal::pdf(lo) - normal::pdf(hi)) / (normal::cdf(hi) - normal::cdf(lo));
            assert!((m[i].value - want).abs() < 6.0 * m[i].error + 1e-12);
        }
    }
}
