//! The sensitivity analysis of Rambachan and Roth \[1\], as R's `HonestDiD`
//! 0.2.8 computes it.
//!
//! An event study estimates `beta = (beta_pre, beta_post)` with covariance
//! `Sigma`. Under parallel trends the pre-treatment coefficients are zero
//! and `beta_post` estimates the effects. Rambachan and Roth let the
//! deviation `delta` from parallel trends be non-zero within a set `Delta`
//! and ask what can be said about a target `theta = l' tau_post` then:
//!
//! - [`Restriction::Smoothness`] is `Delta^SD(M)`: the slope of `delta`
//!   changes by at most `M` between adjacent periods,
//!   `|(delta_{t+1} - delta_t) - (delta_t - delta_{t-1})| <= M`.
//! - [`Restriction::RelativeMagnitude`] is `Delta^RM(Mbar)`: every change of
//!   `delta` between adjacent post-treatment periods is at most `Mbar` times
//!   the largest such change before treatment ([`RmBound::ParallelTrends`]),
//!   or every post-treatment deviation from a linear trend is at most `Mbar`
//!   times the largest pre-treatment one, `Delta^SDRM`
//!   ([`RmBound::LinearTrend`]).
//!
//! Either can be combined with a sign restriction on the post-treatment bias
//! ([`HonestOptions::bias`]) or a monotone `delta`
//! ([`HonestOptions::monotonicity`]), the `PB`, `NB`, `I` and `D` variants of
//! `HonestDiD`.
//!
//! [`summarize`] gives, for each value of `M` or `Mbar`, the identified set
//! of `theta` (the values consistent with `beta` and `Delta`, by linear
//! programs) or a robust confidence interval: the fixed-length interval
//! (`FLCI`) of Armstrong and Kolesár, the conditional test of Andrews, Roth and
//! Pakes inverted over a grid (`Conditional`), and its hybrids with an FLCI
//! (`C-F`) or least-favourable (`C-LF`) first stage, as
//! `createSensitivityResults` and `createSensitivityResults_relativeMagnitudes`
//! do. [`SensitivitySummary::breakdown`] is the largest value at which the
//! interval still excludes zero. [`original`] is the interval under exact
//! parallel trends, `createOriginalCSHonestDiD`.
//!
//! [`joint_path`] gives confidence regions for several targets at once,
//! one robust interval per post-treatment period or per direction, at a
//! pointwise level chosen so that they cover jointly at the requested level:
//! by Bonferroni, or by a max-`|t|` critical value simulated from the
//! covariance of the targets. These regions are epistat's; `HonestDiD` has no
//! joint regions, and the optimisation surface approximates, without
//! reproducing, the full set of directions.
//!
//! [`smoothness_bound_from_pretrends`] is the largest one-sided upper
//! confidence bound of the second differences of the pre-treatment
//! coefficients, a value of `M` the pre-trends support.
//!
//! The conic programs are solved by Clarabel, the linear programs by
//! epistat-optim's dense simplex. A program that ends `AlmostSolved`, or
//! needed a retry with tighter settings, is counted in
//! [`SensitivitySummary::solver`]. The least-favourable critical values and
//! the joint calibration are simulated from the caller's seeds, and the
//! results do not depend on the number of threads.
//!
//! # References
//!
//! 1. Rambachan A, Roth J. A more credible approach to parallel trends. Rev
//!    Econ Stud 2023;90(5):2555-91. doi:10.1093/restud/rdad018
//! 2. Andrews I, Roth J, Pakes A. Inference for linear conditional moment
//!    inequalities. Rev Econ Stud 2023;90(6):2763-91.
//!    doi:10.1093/restud/rdad004
//! 3. Armstrong TB, Kolesár M. Optimal inference in a class of regression
//!    models. Econometrica 2018;86(2):655-83. doi:10.3982/ECTA14434

use epistat_core::{Bootstrap, Estimate, EventStudy, InputError, Interval, Level};

use crate::error::Error;
pub use crate::sensitivity::SolverCounts;
use crate::sensitivity::{
    self as engine, HonestAssessment, HonestBiasDirection, HonestEventStudyInput,
    HonestJointPathConfig, HonestJointPathMethod, HonestMonotonicityDirection,
    HonestOptimizationSurfaceAdaptiveConfig, HonestOptimizationSurfaceAdaptiveRunConfig,
    HonestOptimizationSurfaceConfig, HonestRelativeMagnitudeBound, HonestSensitivity,
    LeastFavorableSimulation, RelativeMagnitudeConfidenceSetConfig, RelativeMagnitudeHybrid,
    SensitivitySummaryMethod, SmoothnessFlciConfig,
};
use crate::util::{InferenceConfig, usize_to_f64};

/// The linear combination `theta = l' tau_post` of the post-treatment
/// coefficients that the analysis is about.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum Target {
    /// `l`, one weight per post-treatment coefficient in the order of
    /// [`EventStudy::post_periods`], at least one of them non-zero. R's
    /// `l_vec`.
    Weights(Vec<f64>),
    /// The coefficient of one post-treatment period, by its relative period.
    Period(i64),
    /// The average of the post-treatment coefficients whose relative periods
    /// lie in `first..=last`.
    Average {
        /// The first relative period of the window.
        first: i64,
        /// The last relative period of the window.
        last: i64,
    },
}

impl Target {
    /// The weight of each post-treatment coefficient.
    fn weights(&self, study: &EventStudy) -> Result<Vec<f64>, Error> {
        let post = study.post_periods();
        let weights = match self {
            Self::Weights(weights) => {
                if weights.len() != post.len() {
                    return Err(InputError::Length {
                        what: "target weights",
                        expected: post.len(),
                        found: weights.len(),
                    }
                    .into());
                }
                epistat_core::check::finite("target weights", weights)?;
                // The tolerance of `HonestDiD`'s own check of an all-zero `l_vec`.
                if weights.iter().all(|w| w.abs() <= 1e-12) {
                    return Err(InputError::Domain {
                        what: "target weights",
                        value: 0.0,
                        domain: "at least one weight non-zero",
                        row: None,
                    }
                    .into());
                }
                weights.clone()
            }
            Self::Period(period) => {
                let Ok(index) = post.binary_search(period) else {
                    return Err(not_a_post_period(*period));
                };
                let mut weights = vec![0.0; post.len()];
                weights[index] = 1.0;
                weights
            }
            Self::Average { first, last } => {
                let covered: Vec<usize> = post
                    .iter()
                    .enumerate()
                    .filter_map(|(i, p)| (first..=last).contains(&p).then_some(i))
                    .collect();
                if covered.is_empty() {
                    return Err(not_a_post_period(*first));
                }
                let weight = 1.0 / usize_to_f64(covered.len());
                let mut weights = vec![0.0; post.len()];
                for i in covered {
                    weights[i] = weight;
                }
                weights
            }
        };
        Ok(weights)
    }
}

fn not_a_post_period(period: i64) -> Error {
    InputError::Domain {
        what: "target period",
        value: period as f64,
        domain: "a post-treatment period of the event study",
        row: None,
    }
    .into()
}

/// Which pre-treatment violation the relative-magnitude bound `Mbar`
/// multiplies.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RmBound {
    /// The largest change of the deviation from parallel trends between
    /// adjacent pre-treatment periods, `Delta^RM` (`HonestDiD`'s default).
    #[default]
    ParallelTrends,
    /// The largest deviation of the pre-treatment coefficients from a linear
    /// trend, their largest second difference, `Delta^SDRM`.
    LinearTrend,
}

/// The set of deviations from parallel trends the analysis allows, with the
/// values of its parameter to report.
///
/// An empty list of values asks for `HonestDiD`'s default grid: for `M`, ten
/// values from 0 to [`smoothness_bound_from_pretrends`] at level 0.05 (or to
/// the standard error of the one pre-treatment coefficient), for `Mbar` ten
/// values from 0 to 2.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum Restriction {
    /// `Delta^SD(M)`: the slope of the deviation changes by at most `M`
    /// between adjacent periods. R's `Mvec`.
    Smoothness {
        /// The values of `M`, each finite and non-negative.
        m: Vec<f64>,
    },
    /// `Delta^RM(Mbar)` or `Delta^SDRM(Mbar)`: post-treatment violations are
    /// at most `Mbar` times the largest pre-treatment one. R's `Mbarvec`.
    RelativeMagnitude {
        /// The values of `Mbar`, each finite and non-negative.
        mbar: Vec<f64>,
        /// Which pre-treatment violation `Mbar` multiplies.
        bound: RmBound,
    },
}

impl Restriction {
    fn values(&self) -> &[f64] {
        match self {
            Self::Smoothness { m } => m,
            Self::RelativeMagnitude { mbar, .. } => mbar,
        }
    }
}

/// A sign restriction on the post-treatment bias.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BiasDirection {
    /// The bias is non-negative (`PB`, R's `biasDirection = "positive"`).
    Positive,
    /// The bias is non-positive (`NB`, R's `biasDirection = "negative"`).
    Negative,
}

/// A monotone deviation from parallel trends.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Monotonicity {
    /// Non-decreasing in event time (`I`, R's `monotonicityDirection =
    /// "increasing"`).
    Increasing,
    /// Non-increasing in event time (`D`, R's `"decreasing"`).
    Decreasing,
}

/// How a row of the summary is computed.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// The identified set, without sampling error.
    IdentifiedSet,
    /// The fixed-length confidence interval (`FLCI`); under the smoothness
    /// restriction without a sign or monotonicity restriction only.
    Flci,
    /// The conditional test inverted over a grid, without a first stage
    /// (R's `"Conditional"`, `hybrid_flag = "ARP"`).
    Conditional,
    /// The conditional test with an FLCI first stage (`C-F`); under the
    /// smoothness restriction only.
    ConditionalFlci,
    /// The conditional test with a least-favourable first stage (`C-LF`).
    ConditionalLeastFavorable,
}

impl Method {
    /// The label `HonestDiD` prints: `"FLCI"`, `"Conditional"`, `"C-F"`,
    /// `"C-LF"`, or `"identified set"`.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::IdentifiedSet => "identified set",
            Self::Flci => "FLCI",
            Self::Conditional => "Conditional",
            Self::ConditionalFlci => "C-F",
            Self::ConditionalLeastFavorable => "C-LF",
        }
    }

    const fn engine(self) -> Option<SensitivitySummaryMethod> {
        match self {
            Self::IdentifiedSet => None,
            Self::Flci => Some(SensitivitySummaryMethod::Flci),
            Self::Conditional => Some(SensitivitySummaryMethod::Conditional),
            Self::ConditionalFlci => Some(SensitivitySummaryMethod::ConditionalFlci),
            Self::ConditionalLeastFavorable => {
                Some(SensitivitySummaryMethod::ConditionalLeastFavorable)
            }
        }
    }

    const fn from_engine(method: SensitivitySummaryMethod) -> Self {
        match method {
            SensitivitySummaryMethod::Flci => Self::Flci,
            SensitivitySummaryMethod::Conditional => Self::Conditional,
            SensitivitySummaryMethod::ConditionalFlci => Self::ConditionalFlci,
            SensitivitySummaryMethod::ConditionalLeastFavorable => Self::ConditionalLeastFavorable,
        }
    }
}

/// The options of [`summarize`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HonestOptions {
    /// How the rows are computed. `None` takes `HonestDiD`'s default: the
    /// FLCI under the smoothness restriction, `C-F` with a sign or
    /// monotonicity restriction added, and `C-LF` under the
    /// relative-magnitude restriction. R's `method`.
    pub method: Option<Method>,
    /// The confidence level of the intervals. R's `alpha`, as its
    /// complement; the default is 0.95.
    pub level: Level,
    /// A sign restriction on the post-treatment bias. R's `biasDirection`.
    pub bias: Option<BiasDirection>,
    /// A monotone deviation from parallel trends. R's
    /// `monotonicityDirection`. At most one of `bias` and `monotonicity`.
    pub monotonicity: Option<Monotonicity>,
    /// The draws of the least-favourable critical value and their seed, for
    /// the `C-LF` hybrid. `HonestDiD`'s R default is 1,000 draws from seed 0,
    /// so a caller who wants R's numbers passes `Bootstrap::new(1_000, 0)`.
    pub least_favorable: Bootstrap,
}

impl HonestOptions {
    /// The default method, the 95% level and no bias or monotonicity
    /// restriction, with the given least-favourable draws and seed.
    #[must_use]
    pub fn new(least_favorable: Bootstrap) -> Self {
        Self {
            method: None,
            level: Level::default(),
            bias: None,
            monotonicity: None,
            least_favorable,
        }
    }
}

/// The class of restrictions of a summary, by `HonestDiD`'s name for it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Delta {
    /// `DeltaSD`.
    Sd,
    /// `DeltaSDPB`.
    SdPositiveBias,
    /// `DeltaSDNB`.
    SdNegativeBias,
    /// `DeltaSDI`.
    SdIncreasing,
    /// `DeltaSDD`.
    SdDecreasing,
    /// `DeltaRM`.
    Rm,
    /// `DeltaRMPB`.
    RmPositiveBias,
    /// `DeltaRMNB`.
    RmNegativeBias,
    /// `DeltaRMI`.
    RmIncreasing,
    /// `DeltaRMD`.
    RmDecreasing,
    /// `DeltaSDRM`.
    SdRm,
    /// `DeltaSDRMPB`.
    SdRmPositiveBias,
    /// `DeltaSDRMNB`.
    SdRmNegativeBias,
    /// `DeltaSDRMI`.
    SdRmIncreasing,
    /// `DeltaSDRMD`.
    SdRmDecreasing,
}

impl Delta {
    /// `HonestDiD`'s name, such as `"DeltaSDPB"`.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Sd => "DeltaSD",
            Self::SdPositiveBias => "DeltaSDPB",
            Self::SdNegativeBias => "DeltaSDNB",
            Self::SdIncreasing => "DeltaSDI",
            Self::SdDecreasing => "DeltaSDD",
            Self::Rm => "DeltaRM",
            Self::RmPositiveBias => "DeltaRMPB",
            Self::RmNegativeBias => "DeltaRMNB",
            Self::RmIncreasing => "DeltaRMI",
            Self::RmDecreasing => "DeltaRMD",
            Self::SdRm => "DeltaSDRM",
            Self::SdRmPositiveBias => "DeltaSDRMPB",
            Self::SdRmNegativeBias => "DeltaSDRMNB",
            Self::SdRmIncreasing => "DeltaSDRMI",
            Self::SdRmDecreasing => "DeltaSDRMD",
        }
    }

    fn of(
        restriction: &Restriction,
        bias: Option<BiasDirection>,
        monotonicity: Option<Monotonicity>,
    ) -> Self {
        use BiasDirection::{Negative, Positive};
        use Monotonicity::{Decreasing, Increasing};
        let family = match restriction {
            Restriction::Smoothness { .. } => 0,
            Restriction::RelativeMagnitude {
                bound: RmBound::ParallelTrends,
                ..
            } => 1,
            Restriction::RelativeMagnitude {
                bound: RmBound::LinearTrend,
                ..
            } => 2,
        };
        let variant = match (bias, monotonicity) {
            (Some(Positive), _) => 1,
            (Some(Negative), _) => 2,
            (None, Some(Increasing)) => 3,
            (None, Some(Decreasing)) => 4,
            (None, None) => 0,
        };
        [
            [
                Self::Sd,
                Self::SdPositiveBias,
                Self::SdNegativeBias,
                Self::SdIncreasing,
                Self::SdDecreasing,
            ],
            [
                Self::Rm,
                Self::RmPositiveBias,
                Self::RmNegativeBias,
                Self::RmIncreasing,
                Self::RmDecreasing,
            ],
            [
                Self::SdRm,
                Self::SdRmPositiveBias,
                Self::SdRmNegativeBias,
                Self::SdRmIncreasing,
                Self::SdRmDecreasing,
            ],
        ][family][variant]
    }
}

/// The identified set of a target: the values consistent with the
/// coefficients and the restriction. Both ends are `None` when the set is
/// empty, as when a second difference of the pre-treatment coefficients
/// exceeds `M`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IdentifiedSet {
    /// The smallest value.
    pub lower: Option<f64>,
    /// The largest value.
    pub upper: Option<f64>,
}

impl IdentifiedSet {
    fn from_pair(lower: f64, upper: f64) -> Self {
        Self {
            lower: defined(lower),
            upper: defined(upper),
        }
    }
}

/// One row of a summary: the interval at one value of `M` or `Mbar`.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct SensitivityRow {
    /// The value of `M` or `Mbar`.
    pub value: f64,
    /// The robust confidence interval, or the identified set under
    /// [`Method::IdentifiedSet`]. An end is `None` where it is undefined
    /// (an empty identified set) and infinite where the set is unbounded.
    pub interval: Interval,
    /// The identified set, under the relative-magnitude restriction, where
    /// the interval is built around it.
    pub identified: Option<IdentifiedSet>,
    /// How the row was computed.
    pub method: Method,
    /// The class of restrictions.
    pub delta: Delta,
}

/// The result of [`summarize`]: one row per value of the restriction's
/// parameter, in the order given.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct SensitivitySummary {
    /// The rows.
    pub rows: Vec<SensitivityRow>,
    /// The programs that needed a retry or ended `AlmostSolved`.
    pub solver: SolverCounts,
}

impl SensitivitySummary {
    /// The breakdown value: the largest `M` or `Mbar` of the rows, taken in
    /// the order given, up to which every interval excludes zero. `None` when
    /// the first interval includes zero or has an undefined end. When every
    /// interval excludes zero it is the last value, a lower bound on the
    /// breakdown value of the whole restriction.
    #[must_use]
    pub fn breakdown(&self) -> Option<f64> {
        let mut breakdown = None;
        for row in &self.rows {
            match (row.interval.lower, row.interval.upper) {
                (Some(lower), Some(upper)) if lower > 0.0 || upper < 0.0 => {
                    breakdown = Some(row.value);
                }
                _ => break,
            }
        }
        breakdown
    }
}

/// `None` for NaN, the engine's mark of an undefined number.
fn defined(value: f64) -> Option<f64> {
    (!value.is_nan()).then_some(value)
}

fn interval(lower: f64, upper: f64, level: Level) -> Interval {
    Interval::new(defined(lower), defined(upper), level)
}

/// The event study in the engine's form.
fn engine_input(study: &EventStudy) -> Result<HonestEventStudyInput, Error> {
    let k = study.coefficients().len();
    let covariance = study.covariance();
    let mut rows = Vec::with_capacity(k);
    for i in 0..k {
        let mut row = Vec::with_capacity(k);
        for j in 0..k {
            let value = covariance.get(i, j).filter(|v| v.is_finite());
            row.push(value.ok_or(InputError::NotFinite {
                what: "covariance",
                row: i * k + j,
                column: None,
            })?);
        }
        rows.push(row);
    }
    let label = |period: i64| {
        i32::try_from(period).map_err(|_| InputError::Domain {
            what: "relative_periods",
            value: period as f64,
            domain: "a 32-bit integer",
            row: None,
        })
    };
    let pre_periods = study
        .pre_periods()
        .iter()
        .map(|&p| label(p))
        .collect::<Result<_, _>>()?;
    let post_periods = study
        .post_periods()
        .iter()
        .map(|&p| label(p))
        .collect::<Result<_, _>>()?;
    if study.n_post() == 0 {
        return Err(InputError::Empty {
            what: "post-treatment coefficients",
        }
        .into());
    }
    Ok(HonestEventStudyInput {
        betahat: study.coefficients().to_vec(),
        covariance: rows,
        pre_periods,
        post_periods,
    })
}

fn inference(level: Level) -> InferenceConfig {
    InferenceConfig {
        confidence_level: level.confidence_value(),
    }
}

fn simulation(bootstrap: Bootstrap, what: &'static str) -> Result<LeastFavorableSimulation, Error> {
    if bootstrap.replicates == 0 {
        return Err(InputError::Domain {
            what,
            value: 0.0,
            domain: "at least one draw",
            row: None,
        }
        .into());
    }
    Ok(LeastFavorableSimulation::new(
        bootstrap.replicates,
        bootstrap.seed,
    ))
}

fn check_values(restriction: &Restriction) -> Result<(), Error> {
    let what = match restriction {
        Restriction::Smoothness { .. } => "M",
        Restriction::RelativeMagnitude { .. } => "Mbar",
    };
    for &value in restriction.values() {
        if !value.is_finite() || value < 0.0 {
            return Err(InputError::Domain {
                what,
                value,
                domain: "finite and non-negative",
                row: None,
            }
            .into());
        }
    }
    Ok(())
}

const fn engine_bias(bias: Option<BiasDirection>) -> Option<HonestBiasDirection> {
    match bias {
        Some(BiasDirection::Positive) => Some(HonestBiasDirection::Positive),
        Some(BiasDirection::Negative) => Some(HonestBiasDirection::Negative),
        None => None,
    }
}

const fn engine_monotonicity(
    monotonicity: Option<Monotonicity>,
) -> Option<HonestMonotonicityDirection> {
    match monotonicity {
        Some(Monotonicity::Increasing) => Some(HonestMonotonicityDirection::Increasing),
        Some(Monotonicity::Decreasing) => Some(HonestMonotonicityDirection::Decreasing),
        None => None,
    }
}

/// The estimate of the target and its Wald interval under exact parallel
/// trends, from the post-treatment block of the covariance, as
/// `HonestDiD::constructOriginalCS`.
///
/// # Errors
///
/// [`Error::Input`] for a target that does not fit the event study, a
/// covariance entry that is undefined, or no post-treatment coefficient.
pub fn original(study: &EventStudy, target: &Target, level: Level) -> Result<Estimate, Error> {
    let input = engine_input(study)?;
    let weights = target.weights(study)?;
    let set = engine::compute_original_confidence_set(&input, &weights, inference(level))
        .map_err(Error::from)?;
    Ok(Estimate::point(set.estimate)
        .with_se(defined(set.se))
        .with_interval(Some(interval(set.ci.0, set.ci.1, level))))
}

/// The sensitivity of a target to violations of parallel trends: for each
/// value of the restriction's parameter, the identified set or a robust
/// confidence interval.
///
/// # Errors
///
/// [`Error::Input`] for a target that does not fit the event study, a
/// sensitivity value that is negative or not finite, an undefined covariance
/// entry, no draws, or too few pre-treatment coefficients for the default
/// grid; [`Error::Unsupported`] for a sign and a monotonicity restriction
/// together, or a method the restriction does not have (the FLCI and `C-F`
/// under the relative-magnitude restriction, the FLCI with a sign or
/// monotonicity restriction); [`Error::Solver`] when a program does not
/// solve, and the other numerical variants of [`Error`] when a critical
/// value cannot be found, a target is degenerate or the conditional test
/// accepts no value of its grid.
pub fn summarize(
    study: &EventStudy,
    target: &Target,
    restriction: &Restriction,
    options: &HonestOptions,
) -> Result<SensitivitySummary, Error> {
    let input = engine_input(study)?;
    let weights = target.weights(study)?;
    check_values(restriction)?;
    if options.bias.is_some() && options.monotonicity.is_some() {
        return Err(Error::Unsupported {
            what: "a sign and a monotonicity restriction together",
        });
    }
    let simulation = simulation(options.least_favorable, "least-favourable draws")?;
    let inference = inference(options.level);
    let bias = engine_bias(options.bias);
    let monotonicity = engine_monotonicity(options.monotonicity);
    let delta = Delta::of(restriction, options.bias, options.monotonicity);
    let values = restriction.values();
    let values = (!values.is_empty()).then_some(values);
    if options.method == Some(Method::IdentifiedSet) {
        return identified_sets(&input, &weights, restriction, options, delta);
    }
    let method = options.method.and_then(Method::engine);
    let summary = match restriction {
        Restriction::Smoothness { .. } => {
            if method == Some(SensitivitySummaryMethod::Flci)
                && (bias.is_some() || monotonicity.is_some())
            {
                return Err(Error::Unsupported {
                    what: "the FLCI with a sign or monotonicity restriction",
                });
            }
            if values.is_none() && input.num_pre_periods() == 0 {
                return Err(InputError::Empty {
                    what: "pre-treatment coefficients",
                }
                .into());
            }
            engine::summarize_smoothness_sensitivity_with_simulation(
                &input,
                inference,
                &weights,
                method,
                values,
                bias,
                monotonicity,
                simulation,
            )
        }
        Restriction::RelativeMagnitude { bound, .. } => {
            if matches!(
                method,
                Some(SensitivitySummaryMethod::Flci | SensitivitySummaryMethod::ConditionalFlci)
            ) {
                return Err(Error::Unsupported {
                    what: "the FLCI and C-F under the relative-magnitude restriction",
                });
            }
            engine::summarize_relative_magnitude_sensitivity_many_with_simulation(
                &input,
                inference,
                &[&weights],
                method,
                values,
                Some(engine_bound(*bound)),
                bias,
                monotonicity,
                simulation,
            )
            .map(|mut summaries| summaries.remove(0))
        }
    }
    .map_err(Error::from)?;
    Ok(SensitivitySummary {
        rows: summary
            .rows
            .into_iter()
            .map(|row| SensitivityRow {
                value: row.sensitivity_value,
                interval: interval(row.lb, row.ub, options.level),
                identified: row
                    .identified
                    .map(|set| IdentifiedSet::from_pair(set.lb, set.ub)),
                method: Method::from_engine(row.method),
                delta,
            })
            .collect(),
        solver: summary.solver,
    })
}

const fn engine_bound(bound: RmBound) -> HonestRelativeMagnitudeBound {
    match bound {
        RmBound::ParallelTrends => HonestRelativeMagnitudeBound::ParallelTrendsDeviation,
        RmBound::LinearTrend => HonestRelativeMagnitudeBound::LinearTrendDeviation,
    }
}

/// The rows of [`Method::IdentifiedSet`].
fn identified_sets(
    input: &HonestEventStudyInput,
    weights: &[f64],
    restriction: &Restriction,
    options: &HonestOptions,
    delta: Delta,
) -> Result<SensitivitySummary, Error> {
    let values = restriction.values();
    let values = if values.is_empty() {
        match restriction {
            Restriction::Smoothness { .. } => {
                engine::default_smoothness_m_values(input).map_err(Error::from)?
            }
            Restriction::RelativeMagnitude { .. } => {
                engine::default_relative_magnitude_mbar_values()
            }
        }
    } else {
        values.to_vec()
    };
    let counter = engine::SolverCounter::new();
    let bias = engine_bias(options.bias);
    let monotonicity = engine_monotonicity(options.monotonicity);
    let rows = values
        .into_iter()
        .map(|value| {
            let set = match restriction {
                Restriction::Smoothness { .. } => engine::smoothness_identified_set(
                    input,
                    weights,
                    value,
                    bias,
                    monotonicity,
                    &counter,
                ),
                Restriction::RelativeMagnitude { bound, .. } => {
                    engine::relative_magnitude_identified_set(
                        input,
                        weights,
                        value,
                        engine_bound(*bound),
                        bias,
                        monotonicity,
                        &counter,
                    )
                }
            }
            .map_err(Error::from)?;
            let identified = IdentifiedSet::from_pair(set.lb, set.ub);
            Ok(SensitivityRow {
                value,
                interval: Interval::new(identified.lower, identified.upper, options.level),
                identified: Some(identified),
                method: Method::IdentifiedSet,
                delta,
            })
        })
        .collect::<Result<_, Error>>()?;
    Ok(SensitivitySummary {
        rows,
        solver: counter.counts(),
    })
}

/// The largest one-sided upper confidence bound, at `level`, of the second
/// differences of the pre-treatment coefficients, `max_j (d_j + z_{1 - alpha}
/// se(d_j))`, and 0 when every bound is negative: a value of `M` that the
/// pre-trends support. With `Level::alpha(0.05)` it is the upper end of
/// `HonestDiD`'s default grid of `M`.
///
/// # Errors
///
/// [`Error::Input`] for fewer than two pre-treatment coefficients or an
/// undefined covariance entry.
pub fn smoothness_bound_from_pretrends(study: &EventStudy, level: Level) -> Result<f64, Error> {
    let input = engine_input(study)?;
    if input.num_pre_periods() < 2 {
        return Err(InputError::Length {
            what: "pre-treatment coefficients, at least",
            expected: 2,
            found: input.num_pre_periods(),
        }
        .into());
    }
    engine::estimate_smoothness_upper_bound_from_pretrends(&input, level.alpha_value())
        .map_err(Error::from)
}

/// How the pointwise level of a joint region is chosen.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Calibration {
    /// Each of the `k` intervals at level `1 - alpha / k`.
    Bonferroni,
    /// The pointwise level of a max-`|t|` critical value simulated from a
    /// normal distribution with the covariance of the `k` targets, with the
    /// given draws and seed.
    Gaussian(Bootstrap),
}

/// The targets of a joint region.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum Directions {
    /// The coefficient of each post-treatment period: the path.
    Periods,
    /// The given weights over the post-treatment coefficients, one target
    /// each.
    Weights(Vec<Vec<f64>>),
    /// The optimisation surface: the unit directions, their pairwise
    /// contrasts and random unit directions, the highest-variance ones
    /// refined by Nelder-Mead.
    Surface(Surface),
    /// The optimisation surface, with random directions added in batches
    /// until the pointwise level changes by less than a tolerance.
    AdaptiveSurface(Surface, Adaptive),
}

/// The directions of the optimisation surface.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Surface {
    /// Include the unit direction of each post-treatment period.
    pub basis: bool,
    /// Include the pairwise contrasts of the post-treatment periods.
    pub pairwise_contrasts: bool,
    /// The number of random unit directions and their seed.
    pub random: Bootstrap,
}

impl Surface {
    /// The unit directions and the pairwise contrasts, with `random` random
    /// unit directions drawn from `seed`.
    #[must_use]
    pub const fn new(random: usize, seed: u64) -> Self {
        Self {
            basis: true,
            pairwise_contrasts: true,
            random: Bootstrap::new(random, seed),
        }
    }
}

/// The growth of the adaptive optimisation surface.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Adaptive {
    /// Random directions added per iteration, positive.
    pub batch: usize,
    /// Random directions needed before the surface can count as converged.
    pub min_random: usize,
    /// The change of the pointwise level below which the surface has
    /// converged, finite and non-negative.
    pub tolerance: f64,
    /// Iterations at most, positive.
    pub max_iterations: usize,
}

impl Default for Adaptive {
    fn default() -> Self {
        let defaults = HonestOptimizationSurfaceAdaptiveConfig::default_for_production();
        Self {
            batch: defaults.random_batch_size,
            min_random: defaults.min_random_for_convergence,
            tolerance: defaults.pointwise_tolerance,
            max_iterations: defaults.max_iterations,
        }
    }
}

/// How each target's interval in a joint region is computed.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum JointMethod {
    /// The conditional interval of each target: `C-LF` under the
    /// relative-magnitude restriction (or `Conditional` with
    /// [`JointPathOptions::hybrid`] off), `C-F` with the options'
    /// least-favourable draws under the smoothness restriction.
    #[default]
    Conditional,
    /// The fixed-length interval of each target, for [`Directions::Periods`]
    /// and [`Directions::Weights`].
    Flci,
}

/// The options of [`joint_path`].
#[derive(Clone, Debug, PartialEq)]
pub struct JointPathOptions {
    /// The joint confidence level of the region; the default is 0.95.
    pub level: Level,
    /// The value each target is tested against.
    pub null: f64,
    /// The targets.
    pub directions: Directions,
    /// How each interval is computed.
    pub method: JointMethod,
    /// How the pointwise level is chosen; the default is Bonferroni.
    pub calibration: Calibration,
    /// The draws of the least-favourable critical value under the
    /// relative-magnitude restriction and their seed. `HonestDiD`'s R default
    /// is 1,000 draws from seed 0, so a caller who wants R's numbers passes
    /// `Bootstrap::new(1_000, 0)`.
    pub least_favorable: Bootstrap,
    /// The least-favourable first stage under the relative-magnitude
    /// restriction (`C-LF`); without it the conditional test alone. The
    /// default is on.
    pub hybrid: bool,
}

impl JointPathOptions {
    /// The 95% level, null 0, per-period targets, conditional intervals,
    /// Bonferroni calibration and the hybrid on, with the given
    /// least-favourable draws and seed.
    #[must_use]
    pub fn new(least_favorable: Bootstrap) -> Self {
        Self {
            level: Level::default(),
            null: 0.0,
            directions: Directions::Periods,
            method: JointMethod::Conditional,
            calibration: Calibration::Bonferroni,
            least_favorable,
            hybrid: true,
        }
    }
}

/// One target of a joint region.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct JointPathPoint {
    /// The relative period, for [`Directions::Periods`].
    pub period: Option<i64>,
    /// The weights of the target over the post-treatment coefficients.
    pub weights: Vec<f64>,
    /// The estimate under exact parallel trends, with its standard error and
    /// its interval at the pointwise level.
    pub original: Estimate,
    /// The robust interval at the pointwise level.
    pub robust: Interval,
    /// The identified set.
    pub identified: IdentifiedSet,
    /// Whether [`JointPathOptions::null`] lies outside the robust interval.
    pub robustly_significant: bool,
}

/// What the adaptive surface did.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfaceDiagnostics {
    /// The directions of the region.
    pub directions: usize,
    /// How many of them are random.
    pub random: usize,
    /// Iterations run.
    pub iterations: usize,
    /// Whether the pointwise level settled within the tolerance.
    pub converged: bool,
    /// The programs that needed a retry or ended `AlmostSolved`.
    pub solver: SolverCounts,
}

/// The result of [`joint_path`].
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct JointPathRegion {
    /// The joint level asked for.
    pub level: Level,
    /// The pointwise confidence level of each interval.
    pub pointwise_level: Level,
    /// The simulated max-`|t|` critical value, under
    /// [`Calibration::Gaussian`].
    pub critical_value: Option<f64>,
    /// The value of `M` or `Mbar`.
    pub value: f64,
    /// One point per target.
    pub points: Vec<JointPathPoint>,
    /// For the directions other than the periods.
    pub diagnostics: Option<SurfaceDiagnostics>,
}

/// A joint confidence region for several targets under a restriction at one
/// value of its parameter: one robust interval per target at a pointwise
/// level that makes them cover jointly at [`JointPathOptions::level`].
///
/// The restriction is the smoothness or the plain relative-magnitude one
/// ([`RmBound::ParallelTrends`]) with exactly one value.
///
/// # Errors
///
/// [`Error::Input`] for a restriction without exactly one value, a value that
/// is negative or not finite, directions of the wrong length or all zero, no
/// draws, or an undefined covariance entry; [`Error::Unsupported`] for the
/// linear-trend bound, or the FLCI on the optimisation surface;
/// [`Error::Solver`] when a program does not solve, and the other numerical
/// variants of [`Error`] when a critical value cannot be found or a direction
/// is degenerate.
pub fn joint_path(
    study: &EventStudy,
    restriction: &Restriction,
    options: &JointPathOptions,
) -> Result<JointPathRegion, Error> {
    let input = engine_input(study)?;
    check_values(restriction)?;
    let &[value] = restriction.values() else {
        return Err(InputError::Length {
            what: "values of the restriction",
            expected: 1,
            found: restriction.values().len(),
        }
        .into());
    };
    let sensitivity = match restriction {
        Restriction::Smoothness { .. } => HonestSensitivity::Smoothness(value),
        Restriction::RelativeMagnitude {
            bound: RmBound::ParallelTrends,
            ..
        } => HonestSensitivity::RelativeMagnitude(value),
        Restriction::RelativeMagnitude {
            bound: RmBound::LinearTrend,
            ..
        } => {
            return Err(Error::Unsupported {
                what: "a joint region under the linear-trend bound",
            });
        }
    };
    let inference = inference(options.level);
    let least_favorable = simulation(options.least_favorable, "least-favourable draws")?;
    let rm_config = RelativeMagnitudeConfidenceSetConfig {
        hybrid: if options.hybrid {
            RelativeMagnitudeHybrid::LeastFavorable
        } else {
            RelativeMagnitudeHybrid::ArpOnly
        },
        ..RelativeMagnitudeConfidenceSetConfig::from_inference(inference, least_favorable)
    };
    let joint_config = match options.calibration {
        Calibration::Bonferroni => HonestJointPathConfig::bonferroni(),
        Calibration::Gaussian(bootstrap) => {
            HonestJointPathConfig::gaussian(simulation(bootstrap, "calibration draws")?)
        }
    };
    let n_post = input.num_post_periods();
    let level = options.level;
    let critical = |method: HonestJointPathMethod, value: f64| {
        (method == HonestJointPathMethod::GaussianSimulated)
            .then_some(value)
            .and_then(defined)
    };
    let directions = match &options.directions {
        Directions::Weights(weights) => {
            for direction in weights {
                Target::Weights(direction.clone()).weights(study)?;
            }
            if weights.is_empty() {
                return Err(InputError::Empty { what: "directions" }.into());
            }
            Some(weights.clone())
        }
        Directions::Periods | Directions::Surface(_) | Directions::AdaptiveSurface(..) => None,
    };
    if options.method == JointMethod::Flci {
        let sets = directions.unwrap_or_else(|| unit_directions(n_post));
        let periods = matches!(options.directions, Directions::Periods);
        if !matches!(
            options.directions,
            Directions::Periods | Directions::Weights(_)
        ) {
            return Err(Error::Unsupported {
                what: "the FLCI on the optimisation surface",
            });
        }
        let result = match sensitivity {
            HonestSensitivity::RelativeMagnitude(mbar) => {
                let problem = engine::build_relative_magnitude_multi_flci_problem(
                    &input, &sets, mbar, inference,
                )
                .map_err(Error::from)?;
                engine::compute_relative_magnitude_multi_flci_with_config(
                    &problem,
                    rm_config,
                    joint_config,
                )
            }
            HonestSensitivity::Smoothness(m) => {
                let problem =
                    engine::build_smoothness_multi_flci_problem(&input, &sets, m, inference)
                        .map_err(Error::from)?;
                engine::compute_smoothness_multi_flci_with_config(
                    &problem,
                    SmoothnessFlciConfig::default_for_production(),
                    joint_config,
                )
            }
        }
        .map_err(Error::from)?;
        let pointwise = pointwise_level(result.pointwise_confidence_level)?;
        let points = result
            .points
            .into_iter()
            .enumerate()
            .map(|(i, point)| {
                let original =
                    engine::compute_original_confidence_set(&input, &point.post_weights, inference)
                        .map_err(Error::from)?;
                Ok(JointPathPoint {
                    period: periods.then(|| study.post_periods()[i]),
                    original: Estimate::point(original.estimate)
                        .with_se(defined(original.se))
                        .with_interval(Some(interval(
                            point.original_ci.0,
                            point.original_ci.1,
                            pointwise,
                        ))),
                    robust: interval(point.flci.0, point.flci.1, pointwise),
                    identified: IdentifiedSet::from_pair(
                        point.identified_set.0,
                        point.identified_set.1,
                    ),
                    robustly_significant: point.robustly_significant,
                    weights: point.post_weights,
                })
            })
            .collect::<Result<_, Error>>()?;
        return Ok(JointPathRegion {
            level,
            pointwise_level: pointwise,
            critical_value: critical(result.method, result.calibrated_max_t_critical_value),
            value,
            points,
            diagnostics: None,
        });
    }
    let region = match &options.directions {
        Directions::Periods => {
            let region = engine::assess_honest_event_study_joint_path_region_with_config(
                &input,
                sensitivity,
                inference,
                options.null,
                rm_config,
                joint_config,
            )
            .map_err(Error::from)?;
            let pointwise = pointwise_level(region.pointwise_confidence_level)?;
            let points = region
                .points
                .into_iter()
                .enumerate()
                .map(|(i, point)| {
                    let mut weights = vec![0.0; n_post];
                    weights[i] = 1.0;
                    joint_point(
                        Some(i64::from(point.post_period)),
                        weights,
                        &point.assessment,
                        pointwise,
                    )
                })
                .collect();
            return Ok(JointPathRegion {
                level,
                pointwise_level: pointwise,
                critical_value: critical(region.method, region.calibrated_max_t_critical_value),
                value,
                points,
                diagnostics: None,
            });
        }
        Directions::Weights(_) => engine::assess_honest_event_study_directional_region_with_config(
            &input,
            directions.as_deref().unwrap_or_default(),
            sensitivity,
            inference,
            options.null,
            rm_config,
            joint_config,
        ),
        Directions::Surface(surface) => {
            engine::assess_honest_event_study_optimization_surface_region_with_config(
                &input,
                sensitivity,
                inference,
                options.null,
                rm_config,
                joint_config,
                engine_surface(*surface),
            )
        }
        Directions::AdaptiveSurface(surface, adaptive) => {
            engine::assess_honest_event_study_optimization_surface_region_adaptive_with_config(
                &input,
                sensitivity,
                inference,
                options.null,
                HonestOptimizationSurfaceAdaptiveRunConfig {
                    relative_magnitude: rm_config,
                    joint: joint_config,
                    surface: engine_surface(*surface),
                    adaptive: HonestOptimizationSurfaceAdaptiveConfig {
                        random_batch_size: adaptive.batch,
                        min_random_for_convergence: adaptive.min_random,
                        pointwise_tolerance: adaptive.tolerance,
                        max_iterations: adaptive.max_iterations,
                    },
                },
            )
        }
    }
    .map_err(Error::from)?;
    let pointwise = pointwise_level(region.pointwise_confidence_level)?;
    let diagnostics = region.diagnostics;
    Ok(JointPathRegion {
        level,
        pointwise_level: pointwise,
        critical_value: critical(region.method, region.calibrated_max_t_critical_value),
        value,
        points: region
            .points
            .into_iter()
            .map(|point| joint_point(None, point.post_weights, &point.assessment, pointwise))
            .collect(),
        diagnostics: Some(SurfaceDiagnostics {
            directions: diagnostics.direction_count,
            random: diagnostics.random_direction_count,
            iterations: diagnostics.iterations_run,
            converged: diagnostics.converged,
            solver: SolverCounts {
                retried: diagnostics.solver_retries,
                almost_solved: diagnostics.solver_almost_solved,
            },
        }),
    })
}

fn unit_directions(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect()
}

const fn engine_surface(surface: Surface) -> HonestOptimizationSurfaceConfig {
    HonestOptimizationSurfaceConfig {
        include_basis: surface.basis,
        include_pairwise_contrasts: surface.pairwise_contrasts,
        random_unit_directions: surface.random.replicates,
        random_seed: surface.random.seed,
    }
}

fn pointwise_level(confidence: f64) -> Result<Level, Error> {
    Level::confidence(confidence).map_err(|_| Error::CriticalValue {
        what: "pointwise confidence level of the joint region",
    })
}

fn joint_point(
    period: Option<i64>,
    weights: Vec<f64>,
    assessment: &HonestAssessment,
    pointwise: Level,
) -> JointPathPoint {
    JointPathPoint {
        period,
        weights,
        original: Estimate::point(assessment.estimate)
            .with_se(defined(assessment.se))
            .with_interval(Some(interval(
                assessment.original_ci.0,
                assessment.original_ci.1,
                pointwise,
            ))),
        robust: interval(assessment.robust_ci.0, assessment.robust_ci.1, pointwise),
        identified: IdentifiedSet::from_pair(
            assessment.identified_set.0,
            assessment.identified_set.1,
        ),
        robustly_significant: assessment.robustly_significant,
    }
}
