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
//! ```
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
//! ```
//!
//! An invalid response is an error before fitting. An invalid mean during
//! fitting triggers step halving. The binomial and Poisson dispersion is fixed
//! at 1, and that of the other families is estimated.
//!
//! # Links
//!
//! ```text
//! link       g(μ)          μ = g⁻¹(η)                dμ/dη
//! identity   μ             η                         1
//! log        ln μ          max(exp η, ε)             max(exp η, ε)
//! logit      ln(μ/(1-μ))   e / (1 + e)               exp η / (1 + exp η)²
//! ```
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

#![expect(
    clippy::float_cmp,
    reason = "the special powers and zero responses are exact values"
)]

/// The link between the mean and the linear predictor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Link {
    Identity,
    Log,
    Logit,
}

/// The distribution of the response.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Family {
    Gaussian,
    /// A proportion of successes, with the number of trials as the prior
    /// weight.
    Binomial,
    Poisson,
    Gamma,
    /// Variance `mu^power`, for `power` outside `(0, 1)`. Powers 0, 1 and 2
    /// are the Gaussian, Poisson and gamma variances.
    Tweedie {
        power: f64,
    },
}

const EPS: f64 = f64::EPSILON;
const THRESHOLD: f64 = 30.0;

impl Link {
    pub(crate) fn link(self, mu: f64) -> f64 {
        match self {
            Self::Identity => mu,
            Self::Log => mu.ln(),
            Self::Logit => (mu / (1.0 - mu)).ln(),
        }
    }

    /// The mean at `eta`. The log and logit inverses keep the mean off 0 and
    /// 1 by machine epsilon, as R's do.
    pub(crate) fn inverse(self, eta: f64) -> f64 {
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
        }
    }

    /// `d mu / d eta` at `eta`.
    pub(crate) fn derivative(self, eta: f64) -> f64 {
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
        }
    }
}

impl Family {
    pub(crate) fn variance(self, mu: f64) -> f64 {
        match self {
            Self::Gaussian => 1.0,
            Self::Binomial => mu * (1.0 - mu),
            Self::Poisson => mu,
            Self::Gamma => mu * mu,
            Self::Tweedie { power } => mu.powf(power),
        }
    }

    /// The unit deviance times the prior weight.
    pub(crate) fn deviance(self, y: f64, mu: f64, weight: f64) -> f64 {
        let y_log_y = |a: f64, b: f64| if a > 0.0 { a * (a / b).ln() } else { 0.0 };
        match self {
            Self::Gaussian => weight * (y - mu) * (y - mu),
            Self::Binomial => 2.0 * weight * (y_log_y(y, mu) + y_log_y(1.0 - y, 1.0 - mu)),
            Self::Poisson => {
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
            Self::Binomial => (weight * y + 0.5) / (weight + 1.0),
            Self::Poisson => y + 0.1,
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
            Self::Binomial => (0.0..=1.0).contains(&y),
            Self::Poisson => y.is_finite() && y >= 0.0,
            // Zero has probability mass only for powers between 1 and 2.
            Self::Tweedie { power } => y.is_finite() && (y > 0.0 || (y == 0.0 && power < 2.0)),
            Self::Gamma => y.is_finite() && y > 0.0,
        }
    }

    pub(crate) fn valid_mean(self, mu: f64) -> bool {
        match self {
            Self::Gaussian => mu.is_finite(),
            Self::Binomial => mu.is_finite() && mu > 0.0 && mu < 1.0,
            Self::Poisson | Self::Gamma | Self::Tweedie { .. } => mu.is_finite() && mu > 0.0,
        }
    }

    /// Whether the family's parameter is in its range: a Tweedie power not in
    /// `(0, 1)`, where no Tweedie distribution exists.
    pub(crate) fn valid(self) -> bool {
        match self {
            Self::Tweedie { power } => power.is_finite() && !(power > 0.0 && power < 1.0),
            _ => true,
        }
    }

    /// Whether the dispersion is fixed at 1 rather than estimated.
    pub(crate) const fn fixed_dispersion(self) -> bool {
        matches!(self, Self::Binomial | Self::Poisson)
    }
}
