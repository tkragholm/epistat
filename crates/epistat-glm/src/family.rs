//! Error distributions and link functions, as R's `family` objects define
//! them.

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
            Self::Poisson | Self::Tweedie { .. } => y.is_finite() && y >= 0.0,
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

    /// Whether the dispersion is fixed at 1 rather than estimated.
    pub(crate) const fn fixed_dispersion(self) -> bool {
        matches!(self, Self::Binomial | Self::Poisson)
    }
}
