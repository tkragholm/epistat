//! The power of the pre-trends test and the distortion after passing it,
//! after Roth \[1\], as the R package `pretrends` 0.1.0 computes them.
//!
//! Event-study estimates `betahat` are approximately `N(beta, Sigma)`, with
//! `beta = tau + delta`: the treatment effects plus a violation of parallel
//! trends. The usual pre-test passes when no pre-period coefficient is
//! individually significant. Roth shows that this test often has little
//! power against violations that matter, and that the estimates of those who
//! pass are distorted. Given a hypothesised `delta` over the coefficients
//! and no treatment effect, [`power`] computes:
//!
//! - the probability that the test passes, its power, and the probability
//!   that it passes under parallel trends;
//! - the means of the pre- and post-period coefficients conditional on
//!   passing, and the bias of the post-period ones;
//! - the likelihood ratio of the observed pre-period coefficients under
//!   `delta` against parallel trends, and the Bayes factor, the ratio of the
//!   probabilities of passing;
//!
//! and, for [`Violation::Power`], first the slope of the linear violation
//! `delta_t = slope * (t - reference)` against which the test has the given
//! power, `pretrends::slope_for_power`.
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
//! smooth and periodic. For each prime number of points the generator is
//! chosen from a fixed candidate list by the weighted `P_2` criterion, and
//! the shifts come from the caller's seed ([`PretrendOptions::shifts`]), so
//! the result is a deterministic function of the inputs. The error is the
//! standard error of the mean over the shifts. The number of points grows
//! through primes up to `2^18` until the error is below
//! [`PretrendOptions::tolerance`]. One and zero dimensions are exact.
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

pub(crate) mod engine;

use epistat_core::{Bootstrap, EventStudy, InputError, Level};

use crate::error::Error;
pub use engine::Evaluated;
use engine::{PretrendConfig, PretrendInput};

/// The hypothesised violation of parallel trends.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum Violation {
    /// The violation at every coefficient, in the order of the event
    /// study's coefficients. R's `deltatrue`.
    Delta(Vec<f64>),
    /// The linear violation `slope * (t - reference)` at relative period `t`.
    LinearTrend {
        /// The slope, finite.
        slope: f64,
    },
    /// The linear violation against which the test has this power, in
    /// `(0, 1)`. R's `slope_for_power(targetPower = ..)`.
    Power(f64),
}

/// The critical value `z` of the pre-test: a pre-period coefficient is
/// significant when `|beta_j| > z s_j`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Critical {
    /// The two-sided normal quantile of the level, `z = Phi^{-1}(1 - alpha /
    /// 2)`, R's `alpha`.
    Level(Level),
    /// The critical value itself, finite and positive; `pretrends` uses 1.96.
    Value(f64),
}

impl Default for Critical {
    /// A confidence of 0.95, R's `alpha = 0.05`.
    fn default() -> Self {
        Self::Level(Level::default())
    }
}

/// The options of [`power`].
#[derive(Clone, Debug, PartialEq)]
pub struct PretrendOptions {
    /// The hypothesised violation.
    pub violation: Violation,
    /// The critical value of each pre-period coefficient's test. The default
    /// is the quantile of a confidence of 0.95.
    pub critical: Critical,
    /// The standard error of a rectangle probability the lattice grows to,
    /// positive. The default is `1e-7`.
    pub tolerance: f64,
    /// The random shifts of the quasi-Monte Carlo lattice, at least 2, and
    /// their seed: draws of the integration, not a bootstrap of the data.
    pub shifts: Bootstrap,
}

impl PretrendOptions {
    /// The options for `violation` with the lattice shifted by `shifts`, the
    /// other fields at their defaults. Twelve shifts give the error estimate
    /// eleven degrees of freedom.
    #[must_use]
    pub fn new(violation: Violation, shifts: Bootstrap) -> Self {
        Self {
            violation,
            critical: Critical::default(),
            tolerance: 1e-7,
            shifts,
        }
    }
}

/// The result of [`power`].
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct PretrendAnalysis {
    /// The violation the results are for, one entry per coefficient.
    pub delta: Vec<f64>,
    /// The slope of the linear violation, for [`Violation::Power`], with the
    /// error of the power at that slope.
    pub slope: Option<Evaluated>,
    /// Probability that the pre-test passes under the violation.
    pub pass_probability: Evaluated,
    /// Probability that it passes under parallel trends.
    pub pass_probability_null: Evaluated,
    /// `1 - pass_probability`.
    pub power: Evaluated,
    /// `pass_probability / pass_probability_null`.
    pub bayes_factor: Evaluated,
    /// Likelihood of the observed pre-period coefficients under the
    /// violation over that under parallel trends.
    pub likelihood_ratio: f64,
    /// Mean of each pre-period coefficient given that the test passes.
    pub mean_pre_after_pretest: Vec<Evaluated>,
    /// Mean of each post-period coefficient given that the test passes.
    pub mean_post_after_pretest: Vec<Evaluated>,
    /// The mean after passing minus the violation, the bias of each
    /// post-period coefficient after passing the pre-test.
    pub post_bias: Vec<Evaluated>,
}

/// The power of the pre-trends test of `study` against a violation, and the
/// distortion of the coefficients after passing it.
///
/// # Errors
///
/// [`Error::Input`] for no pre-treatment coefficient, an undefined or
/// asymmetric covariance, a violation of the wrong length or not finite, a
/// power outside `(0, 1)`, a threshold or tolerance that is not positive, or
/// fewer than two shifts; [`Error::NotPositiveDefinite`] when the
/// pre-treatment covariance is not positive definite; [`Error::NoSlope`]
/// when no slope up to eight standard errors has the requested power.
pub fn power(study: &EventStudy, options: &PretrendOptions) -> Result<PretrendAnalysis, Error> {
    let input = engine_input(study)?;
    let config = config(options)?;
    let reference = study.reference() as f64;
    let (delta, slope) = match &options.violation {
        Violation::Delta(delta) => {
            epistat_core::check::same_length("violation", study.coefficients().len(), delta.len())?;
            epistat_core::check::finite("violation", delta)?;
            (delta.clone(), None)
        }
        Violation::LinearTrend { slope } => {
            epistat_core::check::finite("slope", &[*slope])?;
            (input.linear_trend(*slope, reference), None)
        }
        Violation::Power(target) => {
            if !(*target > 0.0 && *target < 1.0) {
                return Err(InputError::Domain {
                    what: "power",
                    value: *target,
                    domain: "(0, 1)",
                    row: None,
                }
                .into());
            }
            let slope = engine::slope_for_power(&input, reference, *target, &config)?;
            (input.linear_trend(slope.value, reference), Some(slope))
        }
    };
    let analysis = engine::analyse_pretest(&input, &delta, &config)?;
    Ok(PretrendAnalysis {
        delta,
        slope,
        pass_probability: analysis.pass_probability,
        pass_probability_null: analysis.pass_probability_null,
        power: analysis.power,
        bayes_factor: analysis.bayes_factor,
        likelihood_ratio: analysis.likelihood_ratio,
        mean_pre_after_pretest: analysis.mean_pre_after_pretest,
        mean_post_after_pretest: analysis.mean_post_after_pretest,
        post_bias: analysis.post_bias,
    })
}

fn engine_input(study: &EventStudy) -> Result<PretrendInput, Error> {
    if study.n_pre() == 0 {
        return Err(InputError::Empty {
            what: "pre-treatment coefficients",
        }
        .into());
    }
    let k = study.coefficients().len();
    let mut covariance = Vec::with_capacity(k);
    for i in 0..k {
        let mut row = Vec::with_capacity(k);
        for j in 0..k {
            let value = study.covariance().get(i, j).filter(|v| v.is_finite());
            row.push(value.ok_or(InputError::NotFinite {
                what: "covariance",
                row: i * k + j,
                column: None,
            })?);
        }
        covariance.push(row);
    }
    let input = PretrendInput {
        betahat: study.coefficients().to_vec(),
        covariance,
        pre_periods: study.pre_periods().iter().map(|&t| t as f64).collect(),
        post_periods: study.post_periods().iter().map(|&t| t as f64).collect(),
    };
    // Lengths and finiteness hold by construction, so the engine's check can
    // only find the covariance asymmetric.
    input.validate()?;
    Ok(input)
}

fn config(options: &PretrendOptions) -> Result<PretrendConfig, Error> {
    let (level, threshold) = match options.critical {
        Critical::Level(level) => (level, None),
        Critical::Value(value) => (Level::default(), Some(value)),
    };
    let config = PretrendConfig {
        level,
        threshold,
        abs_tol: options.tolerance,
        shifts: options.shifts.replicates,
        seed: options.shifts.seed.value(),
    };
    engine::validate_config(&config)?;
    Ok(config)
}
