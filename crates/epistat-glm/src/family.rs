//! Error distributions and link functions, as R's `family` objects define them.
//!
//! A [`Family`] gives the variance function `V(μ)`, the unit deviance, the
//! starting mean and the valid responses. A [`Link`] gives the link `g(μ)`, its
//! inverse and the derivative `dμ/dη` of the inverse. Any family can be combined
//! with any link. The model is `g(μ) = η`, with `η` the linear predictor.
//!
//! # Families
//!
//! With prior weight `w`, response `y` and mean `μ`, the unit deviance below is
//! multiplied by `w` \[3\].
//!
//! ```text
//! family      V(μ)      unit deviance d(y, μ)                          start μ0
//! Gaussian    1         (y - μ)²                                       y
//! Binomial    μ(1-μ)    2 [y ln(y/μ) + (1-y) ln((1-y)/(1-μ))]          (w y + 0.5) / (w + 1)
//! Poisson     μ         2 [y ln(y/μ) - (y - μ)]                        y + 0.1
//! Gamma       μ²        -2 [ln(y/μ) - (y - μ)/μ]                      y
//! Tweedie(p)  μ^p       see below                                      y, or 0.1 if y = 0
//! NegBin(θ)   μ + μ²/θ  2 [y ln(max(1,y)/μ) - (y+θ) ln((y+θ)/(μ+θ))]  y, or 1/6 if y = 0
//! ```
//!
//! The negative binomial family of type 2 \[4\] has variance `μ + μ²/θ` for a
//! given `θ`, and its deviance is that of `MASS::negative.binomial`. `θ` is
//! estimated by [`crate::negbin`]. The quasi-Poisson and quasi-binomial
//! families have the variance and deviance of the Poisson and binomial
//! families, with the dispersion estimated by the Pearson statistic instead of
//! fixed at 1, so the standard errors are scaled by its square root and the
//! fit has no likelihood.
//!
//! A term `y ln(y/μ)` is 0 when `y = 0`. For the Poisson family with `y = 0` the
//! deviance is `2 μ`. In the gamma deviance a response of 0 is treated as
//! `y/μ = 1`.
//!
//! The binomial response is a proportion of successes in `[0, 1]`, and the prior
//! weight is the number of trials. The Tweedie family has variance `μ^p` \[1\],
//! defined for `p` outside `(0, 1)`, and a fit with `p` in `(0, 1)` is refused.
//! For `p` other than 0, 1 and 2 its unit deviance is
//!
//! ```text
//! d(y, μ) = 2 [ y^{2-p} / ((1-p)(2-p)) - y μ^{1-p} / (1-p) + μ^{2-p} / (2-p) ]
//! ```
//!
//! with the first term taken as 0 when `y = 0`. The powers 0, 1 and 2 use the
//! Gaussian, Poisson and gamma deviances. For `1 < p < 2` the distribution is a
//! compound Poisson-gamma law with a point mass at 0 \[1\] \[2\].
//!
//! # Valid responses and means
//!
//! ```text
//! family             valid y                        valid μ
//! Gaussian           finite                         finite
//! Binomial           0 <= y <= 1                    0 < μ < 1
//! Poisson            finite, y >= 0                 finite, μ > 0
//! Tweedie, p < 2     finite, y >= 0                 finite, μ > 0
//! Tweedie, p >= 2    finite, y > 0                  finite, μ > 0
//! Gamma              finite, y > 0                  finite, μ > 0
//! NegBin, QuasiPoisson   finite, y >= 0             finite, μ > 0
//! QuasiBinomial      0 <= y <= 1                    0 < μ < 1
//! ```
//!
//! An invalid response is an error before fitting. An invalid mean during
//! fitting triggers step halving. The binomial, Poisson and negative binomial
//! dispersion is fixed at 1, and that of the other families is estimated.
//!
//! # Links
//!
//! ```text
//! link       g(μ)          μ = g⁻¹(η)                dμ/dη
//! identity   μ             η                         1
//! log        ln μ          max(exp η, ε)             max(exp η, ε)
//! logit      ln(μ/(1-μ))   e / (1 + e)               exp η / (1 + exp η)²
//! cloglog    ln(-ln(1-μ))  1 - exp(-exp η)           exp(η - exp η)
//! probit     Φ⁻¹(μ)        Φ(η)                      φ(η)
//! ```
//!
//! with `Φ` and `φ` the standard normal distribution function and density of
//! `epistat-stats`. The probit inverse clamps `η` to `±T`, with
//! `T = -Φ⁻¹(ε)`, and its derivative is at least `ε`, as R's
//! `binomial("probit")` does.
//!
//! with `ε = f64::EPSILON`. For the logit inverse, `e = exp η` for
//! `|η| <= 30`, `e = ε` for `η < -30` and `e = 1/ε` for `η > 30`. The derivative
//! of the logit inverse is `ε` for `|η| > 30`. These are the bounds of R's
//! `binomial()$linkinv` and `mu.eta`, and they keep the mean off 0 and 1 and the
//! working weight off 0 when `η` is large.
//!
//! # References
//!
//! 1. Jørgensen B. Exponential dispersion models. J R Stat Soc Ser B.
//!    1987;49(2):127-62.
//! 2. Dunn PK, Smyth GK. Series evaluation of Tweedie exponential dispersion
//!    model densities. Stat Comput. 2005;15(4):267-80.
//! 3. McCullagh P, Nelder JA. Generalized Linear Models. 2nd ed. London: Chapman
//!    and Hall; 1989. Chapter 2.
//! 4. Venables WN, Ripley BD. Modern Applied Statistics with S. 4th ed. New
//!    York: Springer; 2002. Section 7.4.

#![expect(
    clippy::float_cmp,
    reason = "the special powers and zero responses are exact values"
)]

use epistat_core::InputError;

/// The link between the mean and the linear predictor.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Link {
    /// `eta = mu`.
    Identity,
    /// `eta = ln mu`.
    Log,
    /// `eta = ln(mu / (1 - mu))`.
    Logit,
    /// `eta = ln(-ln(1 - mu))`, with `mu` in `(0, 1)`.
    Cloglog,
    /// `eta = Phi^-1(mu)`, with `Phi` the standard normal distribution
    /// function and `mu` in `(0, 1)`.
    Probit,
}

/// The distribution of the response.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Family {
    /// Constant variance.
    Gaussian,
    /// A proportion of successes, with the number of trials as the prior
    /// weight.
    Binomial,
    /// Variance `mu`.
    Poisson,
    /// Variance `mu^2`.
    Gamma,
    /// Variance `mu^power`, for `power` outside `(0, 1)`. Powers 0, 1 and 2
    /// are the Gaussian, Poisson and gamma variances.
    Tweedie {
        /// The exponent of the variance function.
        power: f64,
    },
    /// Negative binomial of type 2: variance `mu + mu^2 / theta` for a given
    /// finite `theta > 0`. The dispersion is fixed at 1.
    NegativeBinomial {
        /// The shape `theta`.
        theta: f64,
    },
    /// The Poisson variance with the dispersion estimated.
    QuasiPoisson,
    /// The binomial variance with the dispersion estimated.
    QuasiBinomial,
}

const EPS: f64 = f64::EPSILON;
const THRESHOLD: f64 = 30.0;
const CLOGLOG_ETA_CAP: f64 = 700.0;

impl Link {
    /// The linear predictor `g(mu)` at the mean `mu`.
    #[must_use]
    pub fn link(self, mu: f64) -> f64 {
        match self {
            Self::Identity => mu,
            Self::Log => mu.ln(),
            Self::Logit => (mu / (1.0 - mu)).ln(),
            Self::Cloglog => (-(-mu).ln_1p()).ln(),
            Self::Probit => epistat_stats::dist::normal::quantile(mu),
        }
    }

    /// The mean `g^-1(eta)` at `eta`. The inverses keep the mean off 0 and
    /// 1 by machine epsilon, as R's do.
    #[must_use]
    pub fn inverse(self, eta: f64) -> f64 {
        match self {
            Self::Identity => eta,
            Self::Log => eta.exp().max(EPS),
            Self::Logit => {
                let e = if eta < -THRESHOLD {
                    EPS
                } else if eta > THRESHOLD {
                    1.0 / EPS
                } else {
                    eta.exp()
                };
                e / (1.0 + e)
            }
            Self::Cloglog => (-(-eta.exp()).exp_m1()).clamp(EPS, 1.0 - EPS),
            Self::Probit => {
                let bound = -epistat_stats::dist::normal::quantile(EPS);
                epistat_stats::dist::normal::cdf(eta.clamp(-bound, bound)).clamp(EPS, 1.0 - EPS)
            }
        }
    }

    /// The mean and `d mu / d eta` at `eta`, as [`Link::inverse`] and
    /// [`Link::derivative`] give them, with the exponential of the logit link
    /// computed once.
    #[must_use]
    pub fn inverse_and_derivative(self, eta: f64) -> (f64, f64) {
        match self {
            Self::Logit => {
                let e = if eta < -THRESHOLD {
                    EPS
                } else if eta > THRESHOLD {
                    1.0 / EPS
                } else {
                    eta.exp()
                };
                let derivative = if eta.abs() > THRESHOLD {
                    EPS
                } else {
                    e / ((1.0 + e) * (1.0 + e))
                };
                (e / (1.0 + e), derivative)
            }
            _ => (self.inverse(eta), self.derivative(eta)),
        }
    }

    /// `d mu / d eta` at `eta`.
    #[must_use]
    pub fn derivative(self, eta: f64) -> f64 {
        match self {
            Self::Identity => 1.0,
            Self::Log => eta.exp().max(EPS),
            Self::Logit => {
                if eta.abs() > THRESHOLD {
                    EPS
                } else {
                    let e = eta.exp();
                    e / ((1.0 + e) * (1.0 + e))
                }
            }
            Self::Cloglog => {
                let e = eta.min(CLOGLOG_ETA_CAP);
                (e - e.exp()).exp().max(EPS)
            }
            Self::Probit => epistat_stats::dist::normal::pdf(eta).max(EPS),
        }
    }
}

impl Family {
    pub(crate) fn variance(self, mu: f64) -> f64 {
        match self {
            Self::Gaussian => 1.0,
            Self::Binomial | Self::QuasiBinomial => mu * (1.0 - mu),
            Self::Poisson | Self::QuasiPoisson => mu,
            Self::Gamma => mu * mu,
            Self::Tweedie { power } => mu.powf(power),
            Self::NegativeBinomial { theta } => mu + mu * mu / theta,
        }
    }

    /// The unit deviance times the prior weight.
    pub(crate) fn deviance(self, y: f64, mu: f64, weight: f64) -> f64 {
        let y_log_y = |a: f64, b: f64| if a > 0.0 { a * (a / b).ln() } else { 0.0 };
        match self {
            Self::Gaussian => weight * (y - mu) * (y - mu),
            Self::Binomial | Self::QuasiBinomial => {
                2.0 * weight * (y_log_y(y, mu) + y_log_y(1.0 - y, 1.0 - mu))
            }
            Self::NegativeBinomial { theta } => {
                2.0 * weight
                    * (y * (y.max(1.0) / mu).ln() - (y + theta) * ((y + theta) / (mu + theta)).ln())
            }
            Self::Poisson | Self::QuasiPoisson => {
                if y > 0.0 {
                    2.0 * weight * (y * (y / mu).ln() - (y - mu))
                } else {
                    2.0 * weight * mu
                }
            }
            Self::Gamma => {
                let ratio = if y == 0.0 { 1.0 } else { y / mu };
                -2.0 * weight * (ratio.ln() - (y - mu) / mu)
            }
            Self::Tweedie { power } => {
                if power == 0.0 {
                    return Self::Gaussian.deviance(y, mu, weight);
                }
                if power == 1.0 {
                    return Self::Poisson.deviance(y, mu, weight);
                }
                if power == 2.0 {
                    return Self::Gamma.deviance(y, mu, weight);
                }
                let y_term = if y > 0.0 {
                    y.powf(2.0 - power) / ((1.0 - power) * (2.0 - power))
                } else {
                    0.0
                };
                2.0 * weight
                    * (y_term - y * mu.powf(1.0 - power) / (1.0 - power)
                        + mu.powf(2.0 - power) / (2.0 - power))
            }
        }
    }

    /// The starting mean R uses for an observation.
    pub(crate) fn start(self, y: f64, weight: f64) -> f64 {
        match self {
            Self::Gaussian | Self::Gamma => y,
            Self::Binomial | Self::QuasiBinomial => (weight * y + 0.5) / (weight + 1.0),
            Self::Poisson | Self::QuasiPoisson => y + 0.1,
            Self::NegativeBinomial { .. } => {
                if y == 0.0 {
                    1.0 / 6.0
                } else {
                    y
                }
            }
            Self::Tweedie { .. } => {
                if y == 0.0 {
                    y + 0.1
                } else {
                    y
                }
            }
        }
    }

    /// Whether `y` is a possible response.
    pub(crate) fn valid_response(self, y: f64) -> bool {
        match self {
            Self::Gaussian => y.is_finite(),
            Self::Binomial | Self::QuasiBinomial => (0.0..=1.0).contains(&y),
            Self::Poisson | Self::QuasiPoisson | Self::NegativeBinomial { .. } => {
                y.is_finite() && y >= 0.0
            }
            // Zero has probability mass only for powers between 1 and 2.
            Self::Tweedie { power } => y.is_finite() && (y > 0.0 || (y == 0.0 && power < 2.0)),
            Self::Gamma => y.is_finite() && y > 0.0,
        }
    }

    pub(crate) fn valid_mean(self, mu: f64) -> bool {
        match self {
            Self::Gaussian => mu.is_finite(),
            Self::Binomial | Self::QuasiBinomial => mu.is_finite() && mu > 0.0 && mu < 1.0,
            Self::Poisson
            | Self::QuasiPoisson
            | Self::NegativeBinomial { .. }
            | Self::Gamma
            | Self::Tweedie { .. } => mu.is_finite() && mu > 0.0,
        }
    }

    /// The family's parameter is in its range: a Tweedie power not in
    /// `(0, 1)`, where no Tweedie distribution exists, and a finite positive
    /// negative binomial `theta`; otherwise [`InputError::Domain`].
    pub(crate) fn check(self) -> Result<(), crate::Error> {
        let (what, value, domain, valid) = match self {
            Self::Tweedie { power } => (
                "Tweedie power",
                power,
                "finite and outside (0, 1)",
                power.is_finite() && !(power > 0.0 && power < 1.0),
            ),
            Self::NegativeBinomial { theta } => (
                "negative binomial theta",
                theta,
                "(0, inf)",
                theta.is_finite() && theta > 0.0,
            ),
            _ => return Ok(()),
        };
        if valid {
            Ok(())
        } else {
            Err(InputError::Domain {
                what,
                value,
                domain,
            }
            .into())
        }
    }

    /// Whether the dispersion is fixed at 1 rather than estimated.
    pub(crate) const fn fixed_dispersion(self) -> bool {
        matches!(
            self,
            Self::Binomial | Self::Poisson | Self::NegativeBinomial { .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_probit_inverse_undoes_the_link_and_stays_inside_the_unit_interval() {
        for mu in [1e-6, 0.05, 0.5, 0.9, 1.0 - 1e-9] {
            let back = Link::Probit.inverse(Link::Probit.link(mu));
            assert!((back - mu).abs() <= 1e-13 * mu.max(1e-3), "{mu}: {back}");
        }
        // The bound is the quantile at EPS, so the end values are EPS up to the
        // rounding of the quantile and the distribution function.
        assert!((Link::Probit.inverse(40.0) - (1.0 - EPS)).abs() <= EPS);
        assert!((Link::Probit.inverse(-40.0) - EPS).abs() <= 1e-14 * EPS);
        assert_eq!(Link::Probit.derivative(40.0), EPS);
        let h = 1e-6;
        let numeric = (Link::Probit.inverse(0.3 + h) - Link::Probit.inverse(0.3 - h)) / (2.0 * h);
        assert!((numeric - Link::Probit.derivative(0.3)).abs() < 1e-9);
    }

    #[test]
    fn the_cloglog_inverse_undoes_the_link_and_stays_inside_the_unit_interval() {
        for mu in [1e-6, 0.05, 0.5, 0.9, 1.0 - 1e-9] {
            let back = Link::Cloglog.inverse(Link::Cloglog.link(mu));
            assert!((back - mu).abs() <= 1e-14 * mu.max(1e-3), "{mu}: {back}");
        }
        assert_eq!(Link::Cloglog.inverse(800.0), 1.0 - EPS);
        assert_eq!(Link::Cloglog.inverse(-800.0), EPS);
        assert_eq!(Link::Cloglog.derivative(800.0), EPS);
        let h = 1e-6;
        let numeric = (Link::Cloglog.inverse(0.3 + h) - Link::Cloglog.inverse(0.3 - h)) / (2.0 * h);
        assert!((numeric - Link::Cloglog.derivative(0.3)).abs() < 1e-9);
    }

    #[test]
    fn the_mean_and_derivative_together_have_the_bits_of_the_two_calls() {
        let mut grid = vec![
            0.0,
            -0.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            THRESHOLD,
            -THRESHOLD,
            THRESHOLD + 1e-12,
            -THRESHOLD - 1e-12,
            CLOGLOG_ETA_CAP,
            800.0,
            -800.0,
        ];
        grid.extend((-400..400).map(|k| f64::from(k) * 0.173));
        for link in [
            Link::Identity,
            Link::Log,
            Link::Logit,
            Link::Cloglog,
            Link::Probit,
        ] {
            for &eta in &grid {
                let (mu, derivative) = link.inverse_and_derivative(eta);
                assert_eq!(mu.to_bits(), link.inverse(eta).to_bits(), "{link:?} {eta}");
                assert_eq!(
                    derivative.to_bits(),
                    link.derivative(eta).to_bits(),
                    "{link:?} {eta}"
                );
            }
        }
    }
}
