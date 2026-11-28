//! Distributions as validated values, for parameters that come from data.
//!
//! Each type checks its parameters once, in its constructor, and refuses a
//! parameter outside the domain with [`Error::Input`] where the free
//! functions of the modules return NaN. Its methods then call those
//! functions, so a value is the same bits as the free function gives:
//! `StudentT::new(df)?.sf(t)` is [`t::sf`]`(t, df)`.
//!
//! | type | parameters | R |
//! |---|---|---|
//! | [`Normal`] | mean, standard deviation | `dnorm`, `pnorm`, `qnorm` |
//! | [`StudentT`] | degrees of freedom | `dt`, `pt`, `qt` |
//! | [`ChiSquared`] | degrees of freedom | `dchisq`, `pchisq`, `qchisq` |
//! | [`FisherF`] | two degrees of freedom | `df`, `pf`, `qf` |
//! | [`Gamma`] | shape, and a rate or a scale | `dgamma`, `pgamma`, `qgamma` |
//! | [`Beta`] | two shapes | `dbeta`, `pbeta`, `qbeta` |
//! | [`Poisson`] | mean | `dpois`, `ppois`, `qpois` |
//! | [`Binomial`] | trials, probability | `dbinom`, `pbinom`, `qbinom` |
//!
//! The methods are the same on every type: `pdf` (`pmf` for the two
//! discrete ones) and `ln_pdf`, `cdf` and `sf` for the lower and upper tail,
//! `ln_cdf` and `ln_sf` for their logs, `quantile` for the inverse of `cdf`
//! and `isf` for the inverse of `sf`. `isf(p)` is R's `q*(p, lower.tail =
//! FALSE)`: it takes the upper-tail probability itself, so a p-value below
//! `1e-16`, which `1 - p` would round to 1, still has its quantile:
//! `Normal::STANDARD.isf(1e-17)` is 8.49 where `quantile(1.0 - 1e-17)` is
//! infinite. The methods return NaN only for a probability outside `[0, 1]`
//! or a NaN argument.

use epistat_core::InputError;

use super::{beta, binomial, chi_square, f, gamma, normal, poisson, t};
use crate::Error;

/// `value` is finite.
fn finite(what: &'static str, value: f64) -> Result<(), Error> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(InputError::Domain {
            what,
            value,
            domain: "(-inf, inf)",
            row: None,
        }
        .into())
    }
}

/// `value` is finite and above 0.
fn positive(what: &'static str, value: f64) -> Result<(), Error> {
    epistat_core::check::positive(what, value).map_err(Error::from)
}

/// The normal distribution with a mean and a standard deviation.
///
/// ```
/// use epistat_stats::dist::Normal;
///
/// let iq = Normal::new(100.0, 15.0)?;
/// assert!((iq.cdf(130.0) - 0.977_249_868_051_820_8).abs() < 1e-15);
/// // The upper 2.5% point, and a tail too small for `quantile(1 - p)`.
/// assert!((Normal::STANDARD.isf(0.025) - 1.959_963_984_540_054).abs() < 1e-15);
/// assert!(Normal::STANDARD.isf(1e-17).is_finite());
/// assert!(Normal::new(0.0, -1.0).is_err());
/// # Ok::<(), epistat_stats::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Normal {
    mean: f64,
    sd: f64,
}

impl Normal {
    /// The standard normal distribution, mean 0 and standard deviation 1.
    pub const STANDARD: Self = Self { mean: 0.0, sd: 1.0 };

    /// The normal distribution with mean `mean` and standard deviation `sd`.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] when `mean` is not finite or `sd` is not finite and
    /// above 0.
    pub fn new(mean: f64, sd: f64) -> Result<Self, Error> {
        finite("mean", mean)?;
        positive("sd", sd)?;
        Ok(Self { mean, sd })
    }

    /// The mean.
    #[must_use]
    pub const fn mean(&self) -> f64 {
        self.mean
    }

    /// The standard deviation.
    #[must_use]
    pub const fn sd(&self) -> f64 {
        self.sd
    }

    fn z(&self, x: f64) -> f64 {
        (x - self.mean) / self.sd
    }

    /// The density at `x`, R's `dnorm(x, mean, sd)`.
    #[must_use]
    pub fn pdf(&self, x: f64) -> f64 {
        normal::pdf(self.z(x)) / self.sd
    }

    /// The log of the density, finite where the density underflows.
    #[must_use]
    pub fn ln_pdf(&self, x: f64) -> f64 {
        normal::ln_pdf(self.z(x)) - self.sd.ln()
    }

    /// `P(X <= x)`.
    #[must_use]
    pub fn cdf(&self, x: f64) -> f64 {
        normal::cdf(self.z(x))
    }

    /// `P(X > x)`, computed directly.
    #[must_use]
    pub fn sf(&self, x: f64) -> f64 {
        normal::sf(self.z(x))
    }

    /// `ln P(X <= x)`, finite far below where [`Normal::cdf`] underflows.
    #[must_use]
    pub fn ln_cdf(&self, x: f64) -> f64 {
        normal::log_cdf(self.z(x))
    }

    /// `ln P(X > x)`, finite far above where [`Normal::sf`] underflows.
    #[must_use]
    pub fn ln_sf(&self, x: f64) -> f64 {
        normal::ln_sf(self.z(x))
    }

    /// The `x` with `P(X <= x) = p`.
    #[must_use]
    pub fn quantile(&self, p: f64) -> f64 {
        self.mean + self.sd * normal::quantile(p)
    }

    /// The `x` with `P(X > x) = p`.
    #[must_use]
    pub fn isf(&self, p: f64) -> f64 {
        self.mean + self.sd * normal::isf(p)
    }
}

/// The Student t distribution.
///
/// ```
/// use epistat_stats::dist::StudentT;
///
/// // A 95% interval for a mean from 12 observations.
/// let t = StudentT::new(11.0)?;
/// let critical = t.isf(0.025);
/// assert!((critical - 2.200_985_160_091_638_8).abs() < 1e-13);
/// assert!((t.two_sided_p(critical) - 0.05).abs() < 1e-15);
/// assert!(StudentT::new(0.0).is_err());
/// # Ok::<(), epistat_stats::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StudentT {
    df: f64,
}

impl StudentT {
    /// The t distribution on `df` degrees of freedom.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] unless `df` is finite and above 0.
    pub fn new(df: f64) -> Result<Self, Error> {
        positive("df", df)?;
        Ok(Self { df })
    }

    /// The degrees of freedom.
    #[must_use]
    pub const fn df(&self) -> f64 {
        self.df
    }

    /// The density at `x`, R's `dt(x, df)`, from [`t::pdf`].
    #[must_use]
    pub fn pdf(&self, x: f64) -> f64 {
        t::pdf(x, self.df)
    }

    /// The log of the density, from the same terms as [`StudentT::pdf`].
    #[must_use]
    pub fn ln_pdf(&self, x: f64) -> f64 {
        t::ln_density(x, self.df)
    }

    /// `P(T <= x)`.
    #[must_use]
    pub fn cdf(&self, x: f64) -> f64 {
        t::cdf(x, self.df)
    }

    /// `P(T > x)`, computed directly above 0.
    #[must_use]
    pub fn sf(&self, x: f64) -> f64 {
        t::sf(x, self.df)
    }

    /// `ln P(T <= x)`, the log of [`StudentT::cdf`].
    #[must_use]
    pub fn ln_cdf(&self, x: f64) -> f64 {
        self.cdf(x).ln()
    }

    /// `ln P(T > x)`, the log of [`StudentT::sf`].
    #[must_use]
    pub fn ln_sf(&self, x: f64) -> f64 {
        self.sf(x).ln()
    }

    /// `P(|T| > |x|)`, the two-sided p-value of a t statistic.
    #[must_use]
    pub fn two_sided_p(&self, x: f64) -> f64 {
        t::two_sided_p(x, self.df)
    }

    /// The `x` with `P(T <= x) = p`.
    #[must_use]
    pub fn quantile(&self, p: f64) -> f64 {
        t::quantile(p, self.df)
    }

    /// The `x` with `P(T > x) = p`.
    #[must_use]
    pub fn isf(&self, p: f64) -> f64 {
        t::isf(p, self.df)
    }
}

/// The chi-square distribution.
///
/// ```
/// use epistat_stats::dist::ChiSquared;
///
/// let chi = ChiSquared::new(2.0)?;
/// assert!((chi.sf(3.21) - 0.200_889_554_910_232_1).abs() < 1e-15);
/// assert!((chi.isf(0.05) - 5.991_464_547_107_979).abs() < 1e-13);
/// # Ok::<(), epistat_stats::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChiSquared {
    df: f64,
}

impl ChiSquared {
    /// The chi-square distribution on `df` degrees of freedom.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] unless `df` is finite and above 0.
    pub fn new(df: f64) -> Result<Self, Error> {
        positive("df", df)?;
        Ok(Self { df })
    }

    /// The degrees of freedom.
    #[must_use]
    pub const fn df(&self) -> f64 {
        self.df
    }

    /// The density at `x`, R's `dchisq(x, df)`.
    #[must_use]
    pub fn pdf(&self, x: f64) -> f64 {
        chi_square::pdf(x, self.df)
    }

    /// The log of the density.
    #[must_use]
    pub fn ln_pdf(&self, x: f64) -> f64 {
        chi_square::ln_pdf(x, self.df)
    }

    /// `P(X <= x)`.
    #[must_use]
    pub fn cdf(&self, x: f64) -> f64 {
        chi_square::cdf(x, self.df)
    }

    /// `P(X > x)`, computed directly.
    #[must_use]
    pub fn sf(&self, x: f64) -> f64 {
        chi_square::sf(x, self.df)
    }

    /// `ln P(X <= x)`, the log of [`ChiSquared::cdf`].
    #[must_use]
    pub fn ln_cdf(&self, x: f64) -> f64 {
        self.cdf(x).ln()
    }

    /// `ln P(X > x)`, the log of [`ChiSquared::sf`].
    #[must_use]
    pub fn ln_sf(&self, x: f64) -> f64 {
        self.sf(x).ln()
    }

    /// The `x` with `P(X <= x) = p`.
    #[must_use]
    pub fn quantile(&self, p: f64) -> f64 {
        chi_square::quantile(p, self.df)
    }

    /// The `x` with `P(X > x) = p`.
    #[must_use]
    pub fn isf(&self, p: f64) -> f64 {
        chi_square::isf(p, self.df)
    }
}

/// The F distribution.
///
/// ```
/// use epistat_stats::dist::FisherF;
///
/// let f = FisherF::new(3.0, 20.0)?;
/// assert!((f.isf(0.05) - 3.098_391_212_140_779_5).abs() < 1e-12);
/// assert!((f.cdf(1.0) + f.sf(1.0) - 1.0).abs() < 1e-15);
/// # Ok::<(), epistat_stats::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FisherF {
    df1: f64,
    df2: f64,
}

impl FisherF {
    /// The F distribution on `df1` and `df2` degrees of freedom.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] unless both are finite and above 0.
    pub fn new(df1: f64, df2: f64) -> Result<Self, Error> {
        positive("df1", df1)?;
        positive("df2", df2)?;
        Ok(Self { df1, df2 })
    }

    /// The numerator degrees of freedom.
    #[must_use]
    pub const fn df1(&self) -> f64 {
        self.df1
    }

    /// The denominator degrees of freedom.
    #[must_use]
    pub const fn df2(&self) -> f64 {
        self.df2
    }

    /// The density at `x`, R's `df(x, df1, df2)`.
    #[must_use]
    pub fn pdf(&self, x: f64) -> f64 {
        f::pdf(x, self.df1, self.df2)
    }

    /// The log of the density.
    #[must_use]
    pub fn ln_pdf(&self, x: f64) -> f64 {
        f::ln_pdf(x, self.df1, self.df2)
    }

    /// `P(F <= x)`.
    #[must_use]
    pub fn cdf(&self, x: f64) -> f64 {
        f::cdf(x, self.df1, self.df2)
    }

    /// `P(F > x)`, computed directly.
    #[must_use]
    pub fn sf(&self, x: f64) -> f64 {
        f::sf(x, self.df1, self.df2)
    }

    /// `ln P(F <= x)`, the log of [`FisherF::cdf`].
    #[must_use]
    pub fn ln_cdf(&self, x: f64) -> f64 {
        self.cdf(x).ln()
    }

    /// `ln P(F > x)`, the log of [`FisherF::sf`].
    #[must_use]
    pub fn ln_sf(&self, x: f64) -> f64 {
        self.sf(x).ln()
    }

    /// The `x` with `P(F <= x) = p`.
    #[must_use]
    pub fn quantile(&self, p: f64) -> f64 {
        f::quantile(p, self.df1, self.df2)
    }

    /// The `x` with `P(F > x) = p`.
    #[must_use]
    pub fn isf(&self, p: f64) -> f64 {
        f::isf(p, self.df1, self.df2)
    }
}

/// The gamma distribution with a shape and a scale, made from a rate or a
/// scale as R's `pgamma(x, shape, rate)` and `pgamma(x, shape, scale = s)`
/// take them.
///
/// ```
/// use epistat_stats::dist::Gamma;
///
/// // Mean 2 / 0.5 = 4.
/// let g = Gamma::with_rate(2.0, 0.5)?;
/// assert_eq!(g.scale(), 2.0);
/// assert!((g.cdf(4.0) - 0.593_994_150_290_161_9).abs() < 1e-15);
/// assert_eq!(g.cdf(4.0), Gamma::with_scale(2.0, 2.0)?.cdf(4.0));
/// assert!(Gamma::with_rate(2.0, 0.0).is_err());
/// # Ok::<(), epistat_stats::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gamma {
    shape: f64,
    scale: f64,
}

impl Gamma {
    /// The gamma distribution with `shape` and `rate`, mean `shape / rate`.
    /// The scale is `1 / rate`, as in R.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] unless both are finite and above 0, and the scale
    /// `1 / rate` is finite.
    pub fn with_rate(shape: f64, rate: f64) -> Result<Self, Error> {
        positive("rate", rate)?;
        Self::with_scale(shape, 1.0 / rate)
    }

    /// The gamma distribution with `shape` and `scale`, mean
    /// `shape * scale`.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] unless both are finite and above 0.
    pub fn with_scale(shape: f64, scale: f64) -> Result<Self, Error> {
        positive("shape", shape)?;
        positive("scale", scale)?;
        Ok(Self { shape, scale })
    }

    /// The shape.
    #[must_use]
    pub const fn shape(&self) -> f64 {
        self.shape
    }

    /// The scale.
    #[must_use]
    pub const fn scale(&self) -> f64 {
        self.scale
    }

    /// The rate, `1 / scale`.
    #[must_use]
    pub fn rate(&self) -> f64 {
        1.0 / self.scale
    }

    /// The density at `x`, R's `dgamma(x, shape, scale = scale)`.
    #[must_use]
    pub fn pdf(&self, x: f64) -> f64 {
        gamma::pdf(x / self.scale, self.shape) / self.scale
    }

    /// The log of the density.
    #[must_use]
    pub fn ln_pdf(&self, x: f64) -> f64 {
        gamma::ln_pdf(x / self.scale, self.shape) - self.scale.ln()
    }

    /// `P(X <= x)`.
    #[must_use]
    pub fn cdf(&self, x: f64) -> f64 {
        gamma::cdf(x / self.scale, self.shape)
    }

    /// `P(X > x)`, computed directly.
    #[must_use]
    pub fn sf(&self, x: f64) -> f64 {
        gamma::sf(x / self.scale, self.shape)
    }

    /// `ln P(X <= x)`, the log of [`Gamma::cdf`].
    #[must_use]
    pub fn ln_cdf(&self, x: f64) -> f64 {
        self.cdf(x).ln()
    }

    /// `ln P(X > x)`, the log of [`Gamma::sf`].
    #[must_use]
    pub fn ln_sf(&self, x: f64) -> f64 {
        self.sf(x).ln()
    }

    /// The `x` with `P(X <= x) = p`.
    #[must_use]
    pub fn quantile(&self, p: f64) -> f64 {
        self.scale * gamma::quantile(p, self.shape)
    }

    /// The `x` with `P(X > x) = p`.
    #[must_use]
    pub fn isf(&self, p: f64) -> f64 {
        self.scale * gamma::isf(p, self.shape)
    }
}

/// The beta distribution.
///
/// ```
/// use epistat_stats::dist::Beta;
///
/// // A Jeffreys interval for 3 events in 20: Beta(3.5, 17.5).
/// let posterior = Beta::new(3.5, 17.5)?;
/// let (lower, upper) = (posterior.quantile(0.025), posterior.isf(0.025));
/// assert!(lower < 0.15 && 0.15 < upper);
/// assert!((posterior.cdf(lower) - 0.025).abs() < 1e-14);
/// # Ok::<(), epistat_stats::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Beta {
    a: f64,
    b: f64,
}

impl Beta {
    /// The beta distribution with shapes `a` and `b`, R's `shape1` and
    /// `shape2`.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] unless both are finite and above 0.
    pub fn new(a: f64, b: f64) -> Result<Self, Error> {
        positive("a", a)?;
        positive("b", b)?;
        Ok(Self { a, b })
    }

    /// The first shape.
    #[must_use]
    pub const fn a(&self) -> f64 {
        self.a
    }

    /// The second shape.
    #[must_use]
    pub const fn b(&self) -> f64 {
        self.b
    }

    /// The density at `x`, R's `dbeta(x, a, b)`.
    #[must_use]
    pub fn pdf(&self, x: f64) -> f64 {
        beta::pdf(x, self.a, self.b)
    }

    /// The log of the density.
    #[must_use]
    pub fn ln_pdf(&self, x: f64) -> f64 {
        beta::ln_pdf(x, self.a, self.b)
    }

    /// `P(X <= x)`.
    #[must_use]
    pub fn cdf(&self, x: f64) -> f64 {
        beta::cdf(x, self.a, self.b)
    }

    /// `P(X > x)`, computed directly.
    #[must_use]
    pub fn sf(&self, x: f64) -> f64 {
        beta::sf(x, self.a, self.b)
    }

    /// `ln P(X <= x)`, the log of [`Beta::cdf`].
    #[must_use]
    pub fn ln_cdf(&self, x: f64) -> f64 {
        self.cdf(x).ln()
    }

    /// `ln P(X > x)`, the log of [`Beta::sf`].
    #[must_use]
    pub fn ln_sf(&self, x: f64) -> f64 {
        self.sf(x).ln()
    }

    /// The `x` with `P(X <= x) = p`.
    #[must_use]
    pub fn quantile(&self, p: f64) -> f64 {
        beta::quantile(p, self.a, self.b)
    }

    /// The `x` with `P(X > x) = p`.
    #[must_use]
    pub fn isf(&self, p: f64) -> f64 {
        beta::isf(p, self.a, self.b)
    }
}

/// The Poisson distribution.
///
/// The variable is a count; `pmf` is 0 at a value that is not a
/// non-negative integer, and the tails take `x` down to an integer, as R
/// does. Quantiles are counts, returned as `f64`, infinite at `p = 1`.
///
/// ```
/// use epistat_stats::dist::Poisson;
///
/// let events = Poisson::new(3.5)?;
/// assert!((events.pmf(2.0) - 0.184_958_973_461_700_8).abs() < 1e-15);
/// assert_eq!(events.quantile(0.5), 3.0);
/// assert_eq!(events.quantile(events.cdf(5.0)), 5.0);
/// assert!(Poisson::new(-1.0).is_err());
/// # Ok::<(), epistat_stats::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Poisson {
    lambda: f64,
}

impl Poisson {
    /// The Poisson distribution with mean `lambda`.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] unless `lambda` is finite and at least 0.
    pub fn new(lambda: f64) -> Result<Self, Error> {
        if !(lambda.is_finite() && lambda >= 0.0) {
            return Err(InputError::Domain {
                what: "lambda",
                value: lambda,
                domain: "[0, inf)",
                row: None,
            }
            .into());
        }
        Ok(Self { lambda })
    }

    /// The mean.
    #[must_use]
    pub const fn lambda(&self) -> f64 {
        self.lambda
    }

    /// `P(X = x)`, R's `dpois(x, lambda)`.
    #[must_use]
    pub fn pmf(&self, x: f64) -> f64 {
        poisson::pmf(x, self.lambda)
    }

    /// `ln P(X = x)`, finite where [`Poisson::pmf`] underflows.
    #[must_use]
    pub fn ln_pmf(&self, x: f64) -> f64 {
        poisson::ln_pmf(x, self.lambda)
    }

    /// `P(X <= x)`.
    #[must_use]
    pub fn cdf(&self, x: f64) -> f64 {
        poisson::cdf(x, self.lambda)
    }

    /// `P(X > x)`, computed directly.
    #[must_use]
    pub fn sf(&self, x: f64) -> f64 {
        poisson::sf(x, self.lambda)
    }

    /// `ln P(X <= x)`, the log of [`Poisson::cdf`].
    #[must_use]
    pub fn ln_cdf(&self, x: f64) -> f64 {
        self.cdf(x).ln()
    }

    /// `ln P(X > x)`, the log of [`Poisson::sf`].
    #[must_use]
    pub fn ln_sf(&self, x: f64) -> f64 {
        self.sf(x).ln()
    }

    /// The smallest count `k` with `P(X <= k) >= p`, R's `qpois(p, lambda)`.
    #[must_use]
    pub fn quantile(&self, p: f64) -> f64 {
        poisson::quantile(p, self.lambda)
    }

    /// The smallest count `k` with `P(X > k) <= p`, R's
    /// `qpois(p, lambda, lower.tail = FALSE)`.
    #[must_use]
    pub fn isf(&self, p: f64) -> f64 {
        poisson::isf(p, self.lambda)
    }
}

/// The binomial distribution.
///
/// As for [`Poisson`], the variable is a count, here from 0 to the number of
/// trials.
///
/// ```
/// use epistat_stats::dist::Binomial;
///
/// let trials = Binomial::new(20, 0.15)?;
/// assert!((trials.cdf(2.0) - 0.404_896_278_007_435_6).abs() < 1e-14);
/// assert_eq!(trials.isf(0.05), 6.0);
/// assert!(Binomial::new(20, 1.5).is_err());
/// # Ok::<(), epistat_stats::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Binomial {
    trials: f64,
    p: f64,
}

/// The largest number of trials, `2^53`, below which every count is an exact
/// `f64`.
const MAX_TRIALS: u64 = 1 << 53;

impl Binomial {
    /// The binomial distribution with `trials` trials of probability `p`.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] when `p` is outside `[0, 1]` or `trials` is above
    /// `2^53`.
    pub fn new(trials: u64, p: f64) -> Result<Self, Error> {
        if trials > MAX_TRIALS {
            return Err(InputError::TooLarge {
                what: "trials",
                limit: MAX_TRIALS,
                found: trials,
            }
            .into());
        }
        epistat_core::check::probability("p", p)?;
        #[expect(
            clippy::cast_precision_loss,
            reason = "the trials are at most 2^53, which an f64 holds exactly"
        )]
        let trials = trials as f64;
        Ok(Self { trials, p })
    }

    /// The number of trials.
    #[must_use]
    pub const fn trials(&self) -> f64 {
        self.trials
    }

    /// The probability of each trial.
    #[must_use]
    pub const fn p(&self) -> f64 {
        self.p
    }

    /// `P(X = x)`, R's `dbinom(x, trials, p)`.
    #[must_use]
    pub fn pmf(&self, x: f64) -> f64 {
        binomial::pmf(x, self.trials, self.p)
    }

    /// `ln P(X = x)`, finite where [`Binomial::pmf`] underflows.
    #[must_use]
    pub fn ln_pmf(&self, x: f64) -> f64 {
        binomial::ln_pmf(x, self.trials, self.p)
    }

    /// `P(X <= x)`.
    #[must_use]
    pub fn cdf(&self, x: f64) -> f64 {
        binomial::cdf(x, self.trials, self.p)
    }

    /// `P(X > x)`, computed directly.
    #[must_use]
    pub fn sf(&self, x: f64) -> f64 {
        binomial::sf(x, self.trials, self.p)
    }

    /// `ln P(X <= x)`, the log of [`Binomial::cdf`].
    #[must_use]
    pub fn ln_cdf(&self, x: f64) -> f64 {
        self.cdf(x).ln()
    }

    /// `ln P(X > x)`, the log of [`Binomial::sf`].
    #[must_use]
    pub fn ln_sf(&self, x: f64) -> f64 {
        self.sf(x).ln()
    }

    /// The smallest count `k` with `P(X <= k) >= q`, R's
    /// `qbinom(q, trials, p)`.
    #[must_use]
    pub fn quantile(&self, q: f64) -> f64 {
        binomial::quantile(q, self.trials, self.p)
    }

    /// The smallest count `k` with `P(X > k) <= q`, R's
    /// `qbinom(q, trials, p, lower.tail = FALSE)`.
    #[must_use]
    pub fn isf(&self, q: f64) -> f64 {
        binomial::isf(q, self.trials, self.p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use epistat_core::Classify;

    #[test]
    fn methods_are_the_free_functions_to_the_bit() {
        let n = Normal::STANDARD;
        for x in [-40.0, -3.0, -0.2, 0.0, 1.5, 9.0] {
            assert_eq!(n.cdf(x).to_bits(), normal::cdf(x).to_bits());
            assert_eq!(n.sf(x).to_bits(), normal::sf(x).to_bits());
            assert_eq!(n.pdf(x).to_bits(), normal::pdf(x).to_bits());
        }
        for p in [1e-300, 0.025, 0.5, 0.9] {
            assert_eq!(n.quantile(p).to_bits(), normal::quantile(p).to_bits());
        }
        let st = StudentT::new(7.5).unwrap();
        let chi = ChiSquared::new(3.0).unwrap();
        let fd = FisherF::new(2.0, 11.0).unwrap();
        let g = Gamma::with_scale(2.5, 1.0).unwrap();
        let b = Beta::new(2.0, 3.5).unwrap();
        for x in [0.01, 0.3, 0.9, 2.5, 7.0] {
            assert_eq!(st.cdf(x).to_bits(), t::cdf(x, 7.5).to_bits());
            assert_eq!(st.sf(x).to_bits(), t::sf(x, 7.5).to_bits());
            assert_eq!(chi.sf(x).to_bits(), chi_square::sf(x, 3.0).to_bits());
            assert_eq!(fd.sf(x).to_bits(), f::sf(x, 2.0, 11.0).to_bits());
            assert_eq!(g.cdf(x).to_bits(), gamma::cdf(x, 2.5).to_bits());
            let u = x / 8.0;
            assert_eq!(b.cdf(u).to_bits(), beta::cdf(u, 2.0, 3.5).to_bits());
        }
        for p in [1e-10, 0.3, 0.975] {
            assert_eq!(st.quantile(p).to_bits(), t::quantile(p, 7.5).to_bits());
            assert_eq!(
                chi.quantile(p).to_bits(),
                chi_square::quantile(p, 3.0).to_bits()
            );
            assert_eq!(
                fd.quantile(p).to_bits(),
                f::quantile(p, 2.0, 11.0).to_bits()
            );
            assert_eq!(g.quantile(p).to_bits(), gamma::quantile(p, 2.5).to_bits());
            assert_eq!(
                b.quantile(p).to_bits(),
                beta::quantile(p, 2.0, 3.5).to_bits()
            );
        }
    }

    #[test]
    fn parameters_outside_the_domain_are_refused() {
        let refusals: [Result<(), Error>; 16] = [
            Normal::new(f64::NAN, 1.0).map(drop),
            Normal::new(0.0, 0.0).map(drop),
            Normal::new(0.0, f64::INFINITY).map(drop),
            StudentT::new(-1.0).map(drop),
            StudentT::new(f64::INFINITY).map(drop),
            ChiSquared::new(0.0).map(drop),
            FisherF::new(1.0, f64::NAN).map(drop),
            FisherF::new(0.0, 1.0).map(drop),
            Gamma::with_rate(1.0, -2.0).map(drop),
            Gamma::with_rate(1.0, 1e-320).map(drop),
            Gamma::with_scale(0.0, 1.0).map(drop),
            Beta::new(1.0, 0.0).map(drop),
            Poisson::new(-0.5).map(drop),
            Poisson::new(f64::INFINITY).map(drop),
            Binomial::new(10, -0.1).map(drop),
            Binomial::new(u64::MAX, 0.5).map(drop),
        ];
        for (i, refusal) in refusals.into_iter().enumerate() {
            let error = refusal.expect_err("a parameter outside the domain");
            assert_eq!(error.kind(), epistat_core::ErrorKind::Input, "case {i}");
        }
        assert_eq!(Normal::new(0.0, -1.0).unwrap_err().code(), "input.domain");
        assert_eq!(
            Binomial::new(u64::MAX, 0.5).unwrap_err().code(),
            "input.too_large"
        );
        assert!(Poisson::new(0.0).is_ok());
        assert!(Binomial::new(0, 1.0).is_ok());
    }

    #[test]
    fn the_upper_quantile_keeps_tails_that_one_minus_p_loses() {
        let p = 1e-17;
        assert_eq!(normal::quantile(1.0 - p), f64::INFINITY);
        let z = Normal::STANDARD.isf(p);
        assert!(z.is_finite() && (Normal::STANDARD.sf(z) / p - 1.0).abs() < 1e-12);
        let t = StudentT::new(4.0).unwrap();
        assert!((t.sf(t.isf(p)) / p - 1.0).abs() < 1e-11);
        let chi = ChiSquared::new(3.0).unwrap();
        assert!((chi.sf(chi.isf(p)) / p - 1.0).abs() < 1e-11);
        let g = Gamma::with_rate(2.0, 3.0).unwrap();
        assert!((g.sf(g.isf(p)) / p - 1.0).abs() < 1e-11);
        let fd = FisherF::new(4.0, 9.0).unwrap();
        assert!((fd.sf(fd.isf(p)) / p - 1.0).abs() < 1e-10);
        let b = Beta::new(2.0, 3.0).unwrap();
        let x = b.isf(1e-9);
        assert!((b.sf(x) / 1e-9 - 1.0).abs() < 1e-8);
    }

    #[test]
    fn probabilities_outside_zero_one_are_nan() {
        let n = Normal::STANDARD;
        assert!(n.quantile(1.5).is_nan() && n.isf(-0.1).is_nan());
        assert!(Poisson::new(2.0).unwrap().quantile(f64::NAN).is_nan());
        assert!(Binomial::new(5, 0.3).unwrap().isf(2.0).is_nan());
        assert_eq!(Poisson::new(2.0).unwrap().quantile(1.0), f64::INFINITY);
        assert_eq!(Poisson::new(2.0).unwrap().isf(0.0), f64::INFINITY);
        assert_eq!(Binomial::new(5, 0.3).unwrap().quantile(1.0), 5.0);
        assert_eq!(Binomial::new(5, 0.3).unwrap().isf(1.0), 0.0);
    }
}
