//! The parametric waiting time distribution and the duration of a
//! prescription.
//!
//! The waiting time distribution (WTD) of Hallas et al. \[1\] is the
//! distribution, over the persons who redeem a drug in an observation window,
//! of the time from the start of the window to their first redemption in it.
//! Prevalent users, who were in treatment when the window opened, give short
//! times; incident users, who start during the window, give times spread
//! evenly over it. The reverse WTD of Pottegård and Hallas \[2\] is the time
//! from the last redemption in the window to its end, and has the same form.
//! Støvring et al. \[3\] fit a parametric two-component model to it by maximum
//! likelihood and read off the duration of a prescription as a percentile of
//! the inter-arrival distribution: the time within which a share `k` of
//! prevalent users redeem their next prescription. This module does that.
//!
//! # Model
//!
//! For a stationary renewal process with inter-arrival distribution `F` and
//! mean `M`, the time from a random instant to the next redemption, and from
//! the last redemption to a random instant, have the forward recurrence
//! density `g(t) = (1 - F(t)) / M` \[3\]. With a share `gamma` of prevalent users
//! and a uniform density for the incident ones over a window of `delta` days,
//! the likelihood of one person with waiting time `t` is \[3\]
//!
//! ```text
//! L(t) = gamma * g(t) + (1 - gamma) / delta,        0 < t <= delta
//! ```
//!
//! For the two families of the paper,
//!
//! ```text
//! log-normal:  g(t) = Phi(-(log t - mu) / sigma) / M,   M = exp(mu + sigma^2 / 2)
//!              tau_k = exp(mu + sigma * Phi^-1(k))
//! Weibull:     g(t) = exp(-(beta t)^alpha) / M,         M = Gamma(1 + 1/alpha) / beta
//!              tau_k = (-log(1 - k))^(1/alpha) / beta
//! ```
//!
//! where `tau_k` is the `k` percentile of the inter-arrival distribution, the
//! duration. As in the paper, the likelihood does not account for censoring
//! at the end of the window: `g` is not renormalised to `(0, delta]`, which
//! is close to exact when `delta` is several times the mean `M`.
//!
//! # Algorithm
//!
//! The parameters are estimated on the scale the paper uses, `logit(gamma)`
//! with `mu` and `log(sigma)` or with `log(alpha)` and `log(beta)`, by BFGS
//! from `epistat-optim` on the mean negative log-likelihood with its analytic
//! gradient. With `D = gamma g + (1 - gamma) / delta`, `c = exp(-mu - sigma^2
//! / 2)`, `z = (log t - mu) / sigma` and `phi` the normal density,
//!
//! ```text
//! d l / d logit(gamma) = gamma (1 - gamma) (g - 1 / delta) / D
//! d l / d mu           = gamma (c phi(z) / sigma - g) / D
//! d l / d log(sigma)   = gamma (c phi(z) z - sigma^2 g) / D
//! d l / d log(alpha)   = gamma g (psi(1 + 1/alpha) / alpha - alpha (beta t)^alpha log(beta t)) / D
//! d l / d log(beta)    = gamma g (1 - alpha (beta t)^alpha) / D
//! ```
//!
//! The start is `gamma = 1/2`, and for the log-normal the mean and standard
//! deviation of `log t`, for the Weibull `alpha = 1` and `beta = 1 / mean(t)`.
//! BFGS ends where the cost stops changing in its last bits, which can leave
//! a gradient of about `1e-8`. Up to 20 Newton steps follow, with the Hessian
//! from central differences of the analytic gradient, and a step is kept only
//! when it lowers the gradient norm. The fit is converged when BFGS says so
//! or the final gradient norm is below `1e-10`.
//! The fit is serial and its result depends only on the times.
//!
//! [`waiting_times`] turns dispensing days into times. A dispensing on day `d`
//! of a window of days `[start, end)` is at the middle of its day, so the
//! forward time is `d - start + 1/2` and the reverse time `end - d - 1/2`, both
//! in `(0, delta)` with `delta = end - start`. The middle of the day keeps a
//! dispensing on the first day away from `t = 0`, where `log t` is undefined.
//!
//! # Not covered
//!
//! Standard errors of the estimates and of the duration are not computed,
//! and neither are the variants with covariates or with censoring at the end
//! of the window. The non-parametric WTD, the histogram of the times, is the
//! caller's to draw.
//!
//! # References
//!
//! 1. Hallas J, Gaist D, Bjerrum L. The waiting time distribution as a
//!    graphical approach to epidemiologic measures of drug utilization.
//!    Epidemiology. 1997;8(6):666-70.
//! 2. Pottegård A, Hallas J. Assigning exposure duration to single
//!    prescriptions by use of the waiting time distribution.
//!    Pharmacoepidemiol Drug Saf. 2013;22(8):803-9.
//! 3. Støvring H, Pottegård A, Hallas J. Determining prescription durations
//!    based on the parametric waiting time distribution. Pharmacoepidemiol
//!    Drug Saf. 2016;25(12):1451-9.

use epistat_optim::bfgs::Bfgs;
use epistat_stats::dist::{gamma, normal};

use crate::Error;
use crate::error::csr;
use epistat_core::InputError;

/// The family of the inter-arrival distribution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Family {
    /// Log-normal with log-mean `mu` and log-standard deviation `sigma`.
    LogNormal,
    /// Weibull with shape `alpha` and rate `beta`.
    Weibull,
}

/// Which waiting time [`waiting_times`] takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WaitingTime {
    /// From the start of the window to the first dispensing in it, the WTD
    /// of Hallas et al.
    Forward,
    /// From the last dispensing in the window to its end, the reverse WTD of
    /// Pottegård and Hallas.
    Reverse,
}

/// The name of [`WaitingTime`] before it was distinguished from
/// [`crate::residence::Direction`].
#[deprecated(since = "0.2.0", note = "use `wtd::WaitingTime`")]
pub type Direction = WaitingTime;

/// A fitted parametric WTD.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct WtdFit {
    /// The family of the inter-arrival distribution.
    pub family: Family,
    /// The share of prevalent users, `gamma`.
    pub prevalent: f64,
    /// `(mu, sigma)` for the log-normal, `(alpha, beta)` for the Weibull.
    pub parameters: (f64, f64),
    /// The mean inter-arrival time `M`, in days.
    pub mean_interarrival: f64,
    /// The maximised log-likelihood.
    pub log_likelihood: f64,
    /// The number of waiting times.
    pub n: usize,
    /// The window width `delta`, in days.
    pub window_days: f64,
    /// Whether BFGS met its gradient or cost tolerance.
    pub converged: bool,
    /// The parameters on the scale of estimation: `logit(gamma)` and then
    /// `mu, log(sigma)` or `log(alpha), log(beta)`.
    pub estimation_scale: [f64; 3],
    /// The gradient of the mean negative log-likelihood at the estimate, on
    /// the scale of estimation.
    pub gradient: [f64; 3],
}

impl WtdFit {
    /// The duration of a prescription: the `k` percentile of the
    /// inter-arrival distribution, the time within which a share `k` of
    /// prevalent users redeem again. Støvring et al. use `k = 0.8`. [`None`]
    /// unless `0 < k < 1`.
    #[must_use]
    pub fn duration(&self, k: f64) -> Option<f64> {
        if !(k > 0.0 && k < 1.0) {
            return None;
        }
        let (a, b) = self.parameters;
        Some(match self.family {
            Family::LogNormal => (a + b * normal::quantile(k)).exp(),
            Family::Weibull => (-(1.0 - k).ln()).powf(1.0 / a) / b,
        })
    }

    /// The forward recurrence density `g(t)` of the prevalent users.
    #[must_use]
    pub fn recurrence_density(&self, t: f64) -> f64 {
        let (a, b) = self.parameters;
        density(self.family, a, b, t).0
    }
}

/// `g(t)` and the two derivatives of `log`-scale terms the gradient needs:
/// for the log-normal `(g, d g / d mu, d g / d log sigma)`, for the Weibull
/// `(g, d g / d log alpha, d g / d log beta)`.
fn density(family: Family, a: f64, b: f64, t: f64) -> (f64, f64, f64) {
    match family {
        Family::LogNormal => {
            let (mu, sigma) = (a, b);
            let c = (-mu - 0.5 * sigma * sigma).exp();
            let z = (t.ln() - mu) / sigma;
            let g = c * normal::cdf(-z);
            let pz = normal::pdf(z);
            (g, c * pz / sigma - g, c * pz * z - sigma * sigma * g)
        }
        Family::Weibull => {
            let (alpha, beta) = (a, b);
            let bt = beta * t;
            let power = bt.powf(alpha);
            let g = beta * (-gamma::ln_gamma(1.0 + 1.0 / alpha) - power).exp();
            let d_alpha = g * (gamma::digamma(1.0 + 1.0 / alpha) / alpha - alpha * power * bt.ln());
            let d_beta = g * (1.0 - alpha * power);
            (g, d_alpha, d_beta)
        }
    }
}

/// The natural parameters from the scale of estimation.
fn natural(family: Family, x: &[f64]) -> (f64, f64, f64) {
    let prevalent = 1.0 / (1.0 + (-x[0]).exp());
    match family {
        Family::LogNormal => (prevalent, x[1], x[2].exp()),
        Family::Weibull => (prevalent, x[1].exp(), x[2].exp()),
    }
}

/// The mean negative log-likelihood and its gradient on the scale of
/// estimation, summed in the order of the times.
fn objective(family: Family, times: &[f64], delta: f64, x: &[f64]) -> (f64, Vec<f64>) {
    let (prevalent, a, b) = natural(family, x);
    let uniform = 1.0 / delta;
    let mut value = 0.0;
    let mut grad = [0.0; 3];
    for &t in times {
        let (g, d1, d2) = density(family, a, b, t);
        let mix = prevalent * g + (1.0 - prevalent) * uniform;
        value -= mix.ln();
        grad[0] -= prevalent * (1.0 - prevalent) * (g - uniform) / mix;
        grad[1] -= prevalent * d1 / mix;
        grad[2] -= prevalent * d2 / mix;
    }
    #[expect(clippy::cast_precision_loss, reason = "a count of persons")]
    let n = times.len() as f64;
    (value / n, grad.iter().map(|v| v / n).collect())
}

/// Fits the parametric WTD of Støvring et al. to waiting times in
/// `(0, window_days]`.
///
/// # Errors
///
/// [`Error::TooFew`] when there are fewer than three times,
/// [`Error::Input`] when the window is not positive and finite or a time is
/// outside `(0, window_days]` (with its row), and [`Error::Optim`] when the
/// optimiser cannot start.
pub fn fit_wtd(times: &[f64], window_days: f64, family: Family) -> Result<WtdFit, Error> {
    if !(window_days.is_finite() && window_days > 0.0) {
        return Err(InputError::Domain {
            what: "window_days",
            value: window_days,
            domain: "positive and finite",
            row: None,
        }
        .into());
    }
    if times.len() < 3 {
        return Err(Error::TooFew {
            what: "times",
            needed: 3,
            found: times.len(),
        });
    }
    if let Some(i) = times.iter().position(|&t| !(t > 0.0 && t <= window_days)) {
        return Err(if times[i].is_finite() {
            InputError::Domain {
                what: "times",
                value: times[i],
                domain: "(0, window_days]",
                row: Some(i),
            }
        } else {
            InputError::NotFinite {
                what: "times",
                row: i,
                column: None,
            }
        }
        .into());
    }
    #[expect(clippy::cast_precision_loss, reason = "a count of persons")]
    let n = times.len() as f64;
    let start = match family {
        Family::LogNormal => {
            let logs: Vec<f64> = times.iter().map(|t| t.ln()).collect();
            let mean = logs.iter().sum::<f64>() / n;
            let var = logs.iter().map(|l| (l - mean) * (l - mean)).sum::<f64>() / n;
            [0.0, mean, 0.5 * var.max(1e-4).ln()]
        }
        Family::Weibull => {
            let mean = times.iter().sum::<f64>() / n;
            [0.0, 0.0, -mean.ln()]
        }
    };
    let bfgs = Bfgs {
        max_iterations: 500,
        gradient_tolerance: 1e-10,
        // Stop on a step that leaves the cost unchanged, and refuse any step
        // that raises it; the gradient tolerance decides convergence.
        cost_tolerance: f64::MIN_POSITIVE,
        ..Bfgs::default()
    };
    let minimum = bfgs.minimize(&start, |x| objective(family, times, window_days, x), None)?;
    let (x, gradient) = polish(family, times, window_days, &minimum.x, &minimum.gradient);
    let converged = minimum.converged || norm(&gradient) < bfgs.gradient_tolerance;
    let (prevalent, a, b) = natural(family, &x);
    let mean_interarrival = match family {
        Family::LogNormal => (a + 0.5 * b * b).exp(),
        Family::Weibull => gamma::ln_gamma(1.0 + 1.0 / a).exp() / b,
    };
    Ok(WtdFit {
        family,
        prevalent,
        parameters: (a, b),
        mean_interarrival,
        log_likelihood: -minimum.value * n,
        n: times.len(),
        window_days,
        converged,
        estimation_scale: [x[0], x[1], x[2]],
        gradient,
    })
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|g| g * g).sum::<f64>().sqrt()
}

/// Newton steps from the BFGS point, with the Hessian from central
/// differences of the analytic gradient, each kept only when it lowers the
/// gradient norm.
fn polish(
    family: Family,
    times: &[f64],
    delta: f64,
    x0: &[f64],
    g0: &[f64],
) -> ([f64; 3], [f64; 3]) {
    let grad = |x: &[f64; 3]| -> [f64; 3] {
        let g = objective(family, times, delta, x).1;
        [g[0], g[1], g[2]]
    };
    let mut x = [x0[0], x0[1], x0[2]];
    let mut g = [g0[0], g0[1], g0[2]];
    for _ in 0..20 {
        let mut h = [[0.0; 3]; 3];
        for j in 0..3 {
            let step = 1e-5 * x[j].abs().max(1.0);
            let (mut up, mut down) = (x, x);
            up[j] += step;
            down[j] -= step;
            let (gu, gd) = (grad(&up), grad(&down));
            for i in 0..3 {
                h[i][j] = (gu[i] - gd[i]) / (2.0 * step);
            }
        }
        // Symmetrise, then solve h d = g by Cramer's rule.
        for i in 0..3 {
            for j in 0..i {
                let m = f64::midpoint(h[i][j], h[j][i]);
                h[i][j] = m;
                h[j][i] = m;
            }
        }
        let det = |m: &[[f64; 3]; 3]| {
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        };
        let d0 = det(&h);
        if !(d0.is_finite() && d0 != 0.0) {
            break;
        }
        let mut next = x;
        for j in 0..3 {
            let mut m = h;
            for (i, row) in m.iter_mut().enumerate() {
                row[j] = g[i];
            }
            next[j] = x[j] - det(&m) / d0;
        }
        let gn = grad(&next);
        if norm(&gn).partial_cmp(&norm(&g)) != Some(std::cmp::Ordering::Less) {
            break;
        }
        x = next;
        g = gn;
    }
    (x, g)
}

/// The waiting time of each person with a dispensing in the window of days
/// `[window_start, window_end)`, in person order: from the start to the first
/// dispensing, or from the last dispensing to the end. A dispensing on day
/// `d` is at the middle of its day, `d + 1/2`. Persons with no dispensing in
/// the window give no time.
///
/// `person_offsets` lays out `day` by person, as in
/// [`crate::exposure::Dispensings`]; the days of a person need not be sorted.
///
/// # Errors
///
/// [`Error::Input`] when the offsets are not a layout over the days
/// or the window is empty or inverted (`window_start` not below
/// `window_end`).
#[expect(
    clippy::cast_precision_loss,
    reason = "day differences of register data are far below 2^53"
)]
pub fn waiting_times(
    person_offsets: &[usize],
    day: &[i64],
    window_start: i64,
    window_end: i64,
    direction: WaitingTime,
) -> Result<Vec<f64>, Error> {
    csr("person_offsets", person_offsets, day.len())?;
    if window_end <= window_start {
        return Err(InputError::NotBelow {
            what: "window_start",
            value: window_start as f64,
            limit: window_end as f64,
        }
        .into());
    }
    let inside = |d: &&i64| (window_start..window_end).contains(*d);
    Ok(person_offsets
        .windows(2)
        .filter_map(|w| {
            let days = day[w[0]..w[1]].iter().filter(inside);
            match direction {
                WaitingTime::Forward => days.min().map(|&d| (d - window_start) as f64 + 0.5),
                WaitingTime::Reverse => days.max().map(|&d| (window_end - d) as f64 - 0.5),
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Central differences of the objective, step `h = 1e-5`, agree with the
    /// analytic gradient to the truncation error `O(h^2)` times the third
    /// derivative plus the rounding `eps / h` times the value, below `1e-7`
    /// for these values of order one.
    #[test]
    fn the_gradient_is_the_derivative_of_the_objective() {
        let times: Vec<f64> = (0..60).map(|i| 0.5 + f64::from(i) * 5.9).collect();
        for family in [Family::LogNormal, Family::Weibull] {
            let x = match family {
                Family::LogNormal => [0.3, 3.5, -0.4],
                Family::Weibull => [-0.2, 0.4, -4.0],
            };
            let (_, grad) = objective(family, &times, 365.0, &x);
            for j in 0..3 {
                let h = 1e-5;
                let (mut up, mut down) = (x, x);
                up[j] += h;
                down[j] -= h;
                let numeric = (objective(family, &times, 365.0, &up).0
                    - objective(family, &times, 365.0, &down).0)
                    / (2.0 * h);
                assert!(
                    (numeric - grad[j]).abs() < 1e-7,
                    "{family:?} {j}: {numeric} against {}",
                    grad[j]
                );
            }
        }
    }

    #[test]
    fn the_duration_is_the_percentile_of_the_inter_arrival_distribution() {
        // Støvring et al.: a log-normal with mean 90 and SD 30 days has the
        // 80th percentile 112.1 days.
        let sigma2 = (1.0_f64 + 30.0 * 30.0 / (90.0 * 90.0)).ln();
        let mu = 90.0_f64.ln() - 0.5 * sigma2;
        let fit = WtdFit {
            family: Family::LogNormal,
            prevalent: 0.5,
            parameters: (mu, sigma2.sqrt()),
            mean_interarrival: 90.0,
            log_likelihood: 0.0,
            n: 0,
            window_days: 365.0,
            converged: true,
            estimation_scale: [0.0; 3],
            gradient: [0.0; 3],
        };
        // The paper rounds mu and sigma^2 to 4.447 and 0.105 before the
        // percentile; unrounded it is 112.23, which the paper's one decimal
        // and its rounding of sigma^2 by up to 5e-4 (about 0.15 days here)
        // cover.
        assert!((fit.duration(0.8).unwrap() - 112.1).abs() < 0.15);
        assert_eq!(fit.duration(1.0), None);
        // A Weibull of shape 1 is the exponential, whose percentile is
        // -log(1 - k) / beta.
        let exp = WtdFit {
            family: Family::Weibull,
            parameters: (1.0, 1.0 / 90.0),
            ..fit
        };
        assert!((exp.duration(0.8).unwrap() - 90.0 * 5.0_f64.ln()).abs() < 1e-9);
    }

    #[test]
    fn waiting_times_take_the_middle_of_the_day() {
        let day = [10, 3, 40, 100, 5];
        let forward = waiting_times(&[0, 3, 4, 5], &day, 0, 50, WaitingTime::Forward).unwrap();
        assert_eq!(forward, [3.5, 5.5]);
        let reverse = waiting_times(&[0, 3, 4, 5], &day, 0, 50, WaitingTime::Reverse).unwrap();
        assert_eq!(reverse, [9.5, 44.5]);
        assert!(waiting_times(&[0, 2], &day, 0, 50, WaitingTime::Forward).is_err());
        assert!(waiting_times(&[0, 5], &day, 5, 5, WaitingTime::Forward).is_err());
    }

    #[test]
    fn refusals() {
        assert!(fit_wtd(&[1.0, 2.0], 10.0, Family::LogNormal).is_err());
        assert!(fit_wtd(&[1.0, 2.0, 11.0], 10.0, Family::LogNormal).is_err());
        assert!(fit_wtd(&[0.0, 2.0, 3.0], 10.0, Family::Weibull).is_err());
        assert!(fit_wtd(&[1.0, 2.0, 3.0], f64::NAN, Family::Weibull).is_err());
    }
}
