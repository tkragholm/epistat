/// Whether a unit is in the treated or the control group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TreatmentGroup {
    /// Not treated.
    Control,
    /// Treated.
    Treated,
}

impl TreatmentGroup {
    /// Treated when `treated` is true, otherwise control.
    #[must_use]
    pub const fn from_bool(treated: bool) -> Self {
        if treated {
            Self::Treated
        } else {
            Self::Control
        }
    }

    /// Whether this is the treated group.
    #[must_use]
    pub const fn is_treated(self) -> bool {
        matches!(self, Self::Treated)
    }
}

/// Whether an observation is before or after treatment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimePeriod {
    /// Before treatment.
    Pre,
    /// After treatment.
    Post,
}

impl TimePeriod {
    /// Post when `post_period` is true, otherwise pre.
    #[must_use]
    pub const fn from_bool(post_period: bool) -> Self {
        if post_period { Self::Post } else { Self::Pre }
    }

    /// Whether this is the post-treatment period.
    #[must_use]
    pub const fn is_post(self) -> bool {
        matches!(self, Self::Post)
    }
}

/// One 2x2 `DiD` panel observation.
#[derive(bon::Builder, Debug, Clone, Copy, PartialEq)]
pub struct PanelObservation {
    /// Whether the unit is in the treated group.
    pub treated: bool,
    /// Whether the observation is from the post-treatment period.
    pub post_period: bool,
    /// Observed outcome value.
    pub outcome: f64,
    /// Sampling weight, 1.0 by default.
    #[builder(default = 1.0)]
    pub weight: f64,
}

impl PanelObservation {
    /// Builds an observation in the given cell with weight 1.0.
    #[must_use]
    pub const fn new(cell: DidCell, outcome: f64) -> Self {
        let (treated, post_period) = cell.flags();
        Self {
            treated,
            post_period,
            outcome,
            weight: 1.0,
        }
    }
}

/// Confidence level shared by the inference procedures of this crate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InferenceConfig {
    /// Confidence level as a probability strictly between 0 and 1, such as 0.95.
    /// The default is 0.95.
    pub confidence_level: f64,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
        }
    }
}

impl InferenceConfig {
    /// Builds a configuration with the given confidence level.
    #[must_use]
    pub const fn new(confidence_level: f64) -> Self {
        Self { confidence_level }
    }
}

/// Settings of the multiplier bootstrap behind bootstrap confidence intervals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BootstrapConfig {
    /// Number of bootstrap replications. The default is 999.
    pub reps: usize,
    /// Seed of the bootstrap random number generator. The default is 17,431.
    pub seed: u64,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            reps: 999,
            seed: 17_431,
        }
    }
}

/// Configuration of the 2x2 `DiD` estimator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DidConfig {
    /// Confidence level as a probability strictly between 0 and 1, such as 0.95.
    /// The default is 0.95.
    pub confidence_level: f64,
}

impl Default for DidConfig {
    fn default() -> Self {
        Self {
            confidence_level: InferenceConfig::default().confidence_level,
        }
    }
}

impl DidConfig {
    /// The confidence level as an [`InferenceConfig`].
    #[must_use]
    pub const fn inference(self) -> InferenceConfig {
        InferenceConfig {
            confidence_level: self.confidence_level,
        }
    }
}

/// 2x2 ATT estimate with a normal-approximation confidence interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DidEstimate {
    /// Estimated average treatment effect on the treated.
    pub att: f64,
    /// Standard error of the estimate.
    pub se: f64,
    /// Lower bound of the confidence interval.
    pub ci_low: f64,
    /// Upper bound of the confidence interval.
    pub ci_high: f64,
}

/// Canonical 2x2 cell identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DidCell {
    /// Treated units before treatment.
    TreatedPre,
    /// Treated units after treatment.
    TreatedPost,
    /// Control units before treatment.
    ControlPre,
    /// Control units after treatment.
    ControlPost,
}

impl DidCell {
    /// The cell for a treatment group and time period.
    #[must_use]
    pub const fn from_parts(group: TreatmentGroup, period: TimePeriod) -> Self {
        match (group, period) {
            (TreatmentGroup::Treated, TimePeriod::Pre) => Self::TreatedPre,
            (TreatmentGroup::Treated, TimePeriod::Post) => Self::TreatedPost,
            (TreatmentGroup::Control, TimePeriod::Pre) => Self::ControlPre,
            (TreatmentGroup::Control, TimePeriod::Post) => Self::ControlPost,
        }
    }

    /// Snake-case label of the cell, such as `treated_pre`.
    #[must_use]
    pub const fn as_label(self) -> &'static str {
        match self {
            Self::TreatedPre => "treated_pre",
            Self::TreatedPost => "treated_post",
            Self::ControlPre => "control_pre",
            Self::ControlPost => "control_post",
        }
    }

    /// The cell as `(treated, post_period)`.
    #[must_use]
    pub const fn flags(self) -> (bool, bool) {
        match self {
            Self::TreatedPre => (true, false),
            Self::TreatedPost => (true, true),
            Self::ControlPre => (false, false),
            Self::ControlPost => (false, true),
        }
    }

    /// Treatment group of the cell.
    #[must_use]
    pub const fn treatment_group(self) -> TreatmentGroup {
        match self {
            Self::TreatedPre | Self::TreatedPost => TreatmentGroup::Treated,
            Self::ControlPre | Self::ControlPost => TreatmentGroup::Control,
        }
    }

    /// Time period of the cell.
    #[must_use]
    pub const fn time_period(self) -> TimePeriod {
        match self {
            Self::TreatedPre | Self::ControlPre => TimePeriod::Pre,
            Self::TreatedPost | Self::ControlPost => TimePeriod::Post,
        }
    }
}

/// Summary statistics for one 2x2 cell.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CellSummary {
    /// The 2x2 cell concerned.
    pub cell: DidCell,
    /// Number of observations in the cell.
    pub observations: usize,
    /// Sum of the weights.
    pub weight_sum: f64,
    /// Effective sample size implied by the weights, `(sum w)^2 / sum w^2`. It
    /// equals `observations` when all weights are equal.
    pub effective_n: f64,
    /// Weighted mean outcome.
    pub mean_outcome: f64,
    /// Weighted variance of the outcome with divisor `sum w`, not `sum w - 1`.
    pub variance: f64,
}

/// Input diagnostics across all 2x2 cells.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DidInputSummary {
    /// Summary of the treated pre-period cell.
    pub treated_pre: CellSummary,
    /// Summary of the treated post-period cell.
    pub treated_post: CellSummary,
    /// Summary of the control pre-period cell.
    pub control_pre: CellSummary,
    /// Summary of the control post-period cell.
    pub control_post: CellSummary,
}

impl DidInputSummary {
    /// Total number of observations over the four cells.
    #[must_use]
    pub const fn total_observations(self) -> usize {
        self.treated_pre.observations
            + self.treated_post.observations
            + self.control_pre.observations
            + self.control_post.observations
    }

    /// Summary of the given cell.
    #[must_use]
    pub const fn cell(self, cell: DidCell) -> CellSummary {
        match cell {
            DidCell::TreatedPre => self.treated_pre,
            DidCell::TreatedPost => self.treated_post,
            DidCell::ControlPre => self.control_pre,
            DidCell::ControlPost => self.control_post,
        }
    }
}

/// One event-time estimate point for dynamic `DiD` aggregation.
#[derive(bon::Builder, Debug, Clone, Copy, PartialEq)]
pub struct EventTimePoint {
    /// Event time, the calendar period minus the first treated period.
    pub event_time: i32,
    /// Point estimate at this event time.
    pub estimate: f64,
    /// Standard error of the point estimate.
    pub se: f64,
    /// Weight of the point in the aggregate under
    /// [`EventTimeWeighting::ByWeight`], finite and positive, 1.0 by default.
    /// It is validated but not used under [`EventTimeWeighting::Equal`].
    #[builder(default = 1.0)]
    pub weight: f64,
}

impl EventTimePoint {
    /// Builds a point with weight 1.0.
    #[must_use]
    pub const fn new(event_time: i32, estimate: f64, se: f64) -> Self {
        Self {
            event_time,
            estimate,
            se,
            weight: 1.0,
        }
    }
}

/// How points at the same event time are weighted when they are aggregated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventTimeWeighting {
    /// Every point has the same weight, so the aggregate is the simple mean.
    Equal,
    /// Points are weighted by their `weight`, normalised to sum to 1.
    ByWeight,
}

/// Aggregated event-time estimate with confidence interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EventTimeEstimate {
    /// Event time, the calendar period minus the first treated period.
    pub event_time: i32,
    /// Weighted mean of the point estimates at this event time.
    pub estimate: f64,
    /// Standard error of the estimate, taking the points as independent.
    pub se: f64,
    /// Lower bound of the two-sided normal confidence interval.
    pub ci_low: f64,
    /// Upper bound of the two-sided normal confidence interval.
    pub ci_high: f64,
    /// Number of points aggregated.
    pub points: usize,
    /// Sum of the weights used, which is `points` under
    /// [`EventTimeWeighting::Equal`].
    pub total_weight: f64,
}

/// Errors returned by event-time aggregation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum EventTimeError {
    /// The confidence level is not strictly between 0 and 1.
    #[error("confidence_level must be finite and in (0, 1)")]
    InvalidConfidenceLevel,
    /// A point estimate is not finite.
    #[error("event-time point estimate must be finite")]
    InvalidPointEstimate,
    /// A standard error is not finite or is negative.
    #[error("event-time point standard error must be finite and non-negative")]
    InvalidPointSe,
    /// A weight is not finite and positive.
    #[error("event-time point weight must be finite and positive")]
    InvalidPointWeight,
}

/// Errors returned by `DiD` estimation.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum DidError {
    /// A 2x2 cell has no observations.
    #[error("missing observations for cell={cell:?}")]
    EmptyCell {
        /// The empty 2x2 cell.
        cell: DidCell,
    },
    /// A weight is not finite and positive.
    #[error("weights must be finite and positive, got {value}")]
    InvalidWeight {
        /// The offending value.
        value: f64,
    },
    /// An outcome is not finite.
    #[error("outcomes must be finite, got {value}")]
    InvalidOutcome {
        /// The offending value.
        value: f64,
    },
    /// The confidence level is not strictly between 0 and 1.
    #[error("confidence_level must be finite and in (0, 1), got {value}")]
    InvalidConfidenceLevel {
        /// The offending value.
        value: f64,
    },
}

/// One panel-level row for doubly robust `DiD` with outcome deltas.
#[derive(bon::Builder, Debug, Clone, PartialEq)]
pub struct DrDidObservation {
    /// Whether the unit is in the treated group.
    pub treated: bool,
    /// Change in outcome between the post and pre period.
    pub delta_outcome: f64,
    /// Sampling weight, 1.0 by default.
    #[builder(default = 1.0)]
    pub weight: f64,
    /// Covariate values, empty by default.
    #[builder(default)]
    pub covariates: Vec<f64>,
}

impl DrDidObservation {
    /// Builds an observation with weight 1.0 and no covariates.
    #[must_use]
    pub const fn new(group: TreatmentGroup, delta_outcome: f64) -> Self {
        Self {
            treated: group.is_treated(),
            delta_outcome,
            weight: 1.0,
            covariates: Vec::new(),
        }
    }
}

/// One repeated cross-section row for doubly robust `DiD`.
#[derive(bon::Builder, Debug, Clone, PartialEq)]
pub struct DrDidRepeatedObservation {
    /// Whether the unit is in the treated group.
    pub treated: bool,
    /// Whether the observation is from the post-treatment period.
    pub post_period: bool,
    /// Observed outcome value.
    pub outcome: f64,
    /// Sampling weight, 1.0 by default.
    #[builder(default = 1.0)]
    pub weight: f64,
    /// Covariate values, empty by default.
    #[builder(default)]
    pub covariates: Vec<f64>,
}

impl DrDidRepeatedObservation {
    /// Builds an observation in the given cell with weight 1.0 and no covariates.
    #[must_use]
    pub const fn new(cell: DidCell, outcome: f64) -> Self {
        let (treated, post_period) = cell.flags();
        Self {
            treated,
            post_period,
            outcome,
            weight: 1.0,
            covariates: Vec::new(),
        }
    }
}

/// Configuration for panel `DR-DiD` estimation.
#[derive(bon::Builder, Debug, Clone, Copy, PartialEq)]
pub struct DrDidConfig {
    /// Confidence level as a probability strictly between 0 and 1, such as 0.95.
    /// The default is 0.95.
    #[builder(default = 0.95)]
    pub confidence_level: f64,
    /// Propensity scores are clipped to `[propensity_clip, 1 - propensity_clip]`.
    /// It must lie in `(0, 0.5)`. The default is 1e-6.
    #[builder(default = 1e-6)]
    pub propensity_clip: f64,
    /// Ridge penalty added to the diagonal of the outcome regressions' normal
    /// equations, finite and non-negative. The default is 1e-8.
    #[builder(default = 1e-8)]
    pub ridge: f64,
    /// Maximum number of iterations of the propensity fit, positive. The default
    /// is 200.
    #[builder(default = 200_usize)]
    pub max_iter: usize,
    /// Convergence tolerance of the propensity fit, positive. The default is
    /// 1e-8.
    #[builder(default = 1e-8)]
    pub tol: f64,
    /// Number of replications of the multiplier bootstrap behind the confidence
    /// interval. With 0, the default, the interval is the normal interval
    /// `att +- z se` at `confidence_level`, as `DRDID` reports with its default
    /// `boot = FALSE` and `did::att_gt` with `bstrap = FALSE`. A positive count
    /// makes it the percentile interval of that many Rademacher multiplier
    /// replicates of the influence function.
    #[builder(default = 0_usize)]
    pub bootstrap_reps: usize,
    /// Seed of the multiplier bootstrap. The default is 17,431.
    #[builder(default = 17_431_u64)]
    pub bootstrap_seed: u64,
    /// Whether [`crate::estimate_drdid_repeated_cross_section`] computes as
    /// `epistat-did` 0.1 and `did_methods` 0.2 did, for reproducing results
    /// made with them: propensity score fitted without the sampling weights
    /// and clipped to `[propensity_clip, 1 - propensity_clip]`, no trimming,
    /// and an influence function without the estimation of the nuisance
    /// models. Off, the default, it is `DRDID::drdid_rc1`. The other
    /// estimators ignore it.
    #[builder(default = false)]
    pub legacy_repeated_cross_section: bool,
}

impl Default for DrDidConfig {
    fn default() -> Self {
        Self {
            confidence_level: InferenceConfig::default().confidence_level,
            propensity_clip: 1e-6,
            ridge: 1e-8,
            max_iter: 200,
            tol: 1e-8,
            bootstrap_reps: 0,
            bootstrap_seed: BootstrapConfig::default().seed,
            legacy_repeated_cross_section: false,
        }
    }
}

impl DrDidConfig {
    /// The confidence level as an [`InferenceConfig`].
    #[must_use]
    pub const fn inference(self) -> InferenceConfig {
        InferenceConfig {
            confidence_level: self.confidence_level,
        }
    }

    /// The bootstrap settings as a [`BootstrapConfig`].
    #[must_use]
    pub const fn bootstrap(self) -> BootstrapConfig {
        BootstrapConfig {
            reps: self.bootstrap_reps,
            seed: self.bootstrap_seed,
        }
    }
}

/// `DR-DiD` ATT estimate and uncertainty summary.
#[derive(Debug, Clone, PartialEq)]
pub struct DrDidEstimate {
    /// Estimated average treatment effect on the treated.
    pub att: f64,
    /// Standard error from the influence function, not accounting for
    /// clustering.
    pub se: f64,
    /// Lower end of the confidence interval: `att - z se` at the configured
    /// level, or the multiplier-bootstrap percentile when
    /// [`DrDidConfig::bootstrap_reps`] is positive. The improved estimators
    /// return `att - 1.96 * se` whatever the configuration.
    pub ci_low: f64,
    /// Upper end of the confidence interval, as `ci_low`.
    pub ci_high: f64,
    /// Number of treated units used.
    pub treated_n: usize,
    /// Number of comparison units used.
    pub control_n: usize,
    /// Sum of the weights of the observations used.
    pub total_weight: f64,
    /// Influence function of the estimate, one value per input observation.
    pub influence_function: Vec<f64>,
    /// Number of covariate columns dropped before fitting because the fit's own
    /// rows could not identify them: constant within an arm, or a combination
    /// of the columns before them. It is zero for a full-rank design. The
    /// estimate uses the remaining columns.
    pub design_columns_dropped: usize,
}

/// Configuration for `DiD_CC` estimators and the stationarity Hausman test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DidCcConfig {
    /// Settings of the doubly robust estimator, including the confidence level
    /// and the bootstrap settings.
    pub drdid: DrDidConfig,
    /// Significance level of the Hausman test, strictly between 0 and 1. The
    /// default is 0.05.
    pub hausman_alpha: f64,
    /// Whether the confidence intervals are multiplier-bootstrap percentile
    /// intervals, using `drdid.bootstrap_reps` and `drdid.bootstrap_seed`, or
    /// otherwise normal intervals from the standard error. The default is
    /// `false`. With `true`, `drdid.bootstrap_reps` must be positive.
    pub bootstrap_confidence_intervals: bool,
    /// Number of cross-fitting folds, at least 1. One means no cross-fitting.
    /// The default is 1.
    pub cross_fit_folds: usize,
    /// Seed of the cross-fitting fold assignment. The default is the default
    /// bootstrap seed.
    pub cross_fit_seed: u64,
}

impl Default for DidCcConfig {
    fn default() -> Self {
        Self {
            drdid: DrDidConfig::default(),
            hausman_alpha: 0.05,
            bootstrap_confidence_intervals: false,
            cross_fit_folds: 1,
            cross_fit_seed: BootstrapConfig::default().seed,
        }
    }
}

/// ATT estimate from the compositional-change `DiD_CC` family.
#[derive(Debug, Clone, PartialEq)]
pub struct DidCcEstimate {
    /// Estimated average treatment effect on the treated.
    pub att: f64,
    /// Standard error from the influence function, not accounting for
    /// clustering.
    pub se: f64,
    /// Lower end of the confidence interval, bootstrap or normal according to
    /// `DidCcConfig::bootstrap_confidence_intervals`.
    pub ci_low: f64,
    /// Upper end of the confidence interval, bootstrap or normal according to
    /// `DidCcConfig::bootstrap_confidence_intervals`.
    pub ci_high: f64,
    /// Number of treated observations in the post period.
    pub treated_post_n: usize,
    /// Number of treated observations in the pre period.
    pub treated_pre_n: usize,
    /// Number of control observations in the post period.
    pub control_post_n: usize,
    /// Number of control observations in the pre period.
    pub control_pre_n: usize,
    /// Sum of the weights of the observations used.
    pub total_weight: f64,
    /// Influence function of the estimate, one value per input observation.
    pub influence_function: Vec<f64>,
}

/// Hausman-style comparison between robust and stationarity-based `DiD_CC` ATT
/// estimators.
#[derive(Debug, Clone, PartialEq)]
pub struct DidCcHausmanTest {
    /// Composition-robust estimate.
    pub robust: DidCcEstimate,
    /// Estimate under the stationarity restriction.
    pub stationary: DidCcEstimate,
    /// Stationary estimate minus robust estimate.
    pub difference: f64,
    /// Standard error of the difference, from the difference of the two
    /// influence functions.
    pub difference_se: f64,
    /// Test statistic, `(difference / difference_se)^2`.
    pub statistic: f64,
    /// p-value of the statistic against a chi-square distribution with one
    /// degree of freedom.
    pub p_value: f64,
    /// Whether `p_value` is below `DidCcConfig::hausman_alpha`, so that the
    /// stationarity restriction is rejected.
    pub reject_null: bool,
}

/// Errors returned by `DiD_CC` estimation and testing.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum DidCcError {
    /// No observations were given.
    #[error("DiD_CC requires at least one observation")]
    EmptyInput,
    /// The sample has no treated units.
    #[error("DiD_CC sample has no treated units")]
    NoTreated,
    /// The sample has no control units.
    #[error("DiD_CC sample has no control units")]
    NoControl,
    /// The repeated cross-section sample is missing a 2x2 cell.
    #[error("DiD_CC repeated cross-section sample is missing cell={cell:?}")]
    MissingCell {
        /// The 2x2 cell concerned.
        cell: DidCell,
    },
    /// A weight is not finite and positive.
    #[error("weights must be finite and positive, got {value}")]
    InvalidWeight {
        /// The offending value.
        value: f64,
    },
    /// An outcome is not finite.
    #[error("outcomes must be finite, got {value}")]
    InvalidOutcome {
        /// The offending value.
        value: f64,
    },
    /// A covariate is not finite.
    #[error("covariates must be finite, got {value}")]
    InvalidCovariate {
        /// The offending value.
        value: f64,
    },
    /// The rows differ in their number of covariates.
    #[error(
        "all DiD_CC rows must have the same covariate count; expected {expected}, got {actual}"
    )]
    InconsistentCovariateCount {
        /// Number of covariates the rows should have.
        expected: usize,
        /// Number of covariates on the offending row.
        actual: usize,
    },
    /// The confidence level is not strictly between 0 and 1.
    #[error("confidence_level must be finite and in (0, 1)")]
    InvalidConfidenceLevel,
    /// The Hausman significance level is not strictly between 0 and 1.
    #[error("hausman_alpha must be finite and in (0, 1)")]
    InvalidHausmanAlpha,
    /// The configuration is invalid, with a description.
    #[error("invalid DiD_CC config: {0}")]
    InvalidConfig(String),
    /// The number of cross-fitting folds is zero.
    #[error("cross_fit_folds must be at least 1")]
    InvalidCrossFitFolds,
    /// A cell has fewer observations than cross-fitting folds.
    #[error(
        "cross-fitting with {folds} folds requires at least {folds} observations in cell={cell:?}"
    )]
    InsufficientCellForCrossFit {
        /// Number of cross-fitting folds requested.
        folds: usize,
        /// The cell with too few observations.
        cell: DidCell,
    },
    /// A linear system could not be solved.
    #[error("could not solve linear system")]
    SingularSystem,
    /// The variance of the Hausman difference is not positive.
    #[error("Hausman difference variance must be strictly positive")]
    DegenerateHausmanVariance,
}

/// Errors returned by panel `DR-DiD` estimation.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum DrDidError {
    /// No observations were given.
    #[error("DR-DiD requires at least one observation")]
    EmptyInput,
    /// The sample has no treated units.
    #[error("DR-DiD sample has no treated units")]
    NoTreated,
    /// The sample has no control units.
    #[error("DR-DiD sample has no control units")]
    NoControl,
    /// The repeated cross-section sample is missing a 2x2 cell.
    #[error("DR-DiD repeated cross-section sample is missing cell={cell:?}")]
    MissingCell {
        /// The 2x2 cell concerned.
        cell: DidCell,
    },
    /// A weight is not finite and positive.
    #[error("weights must be finite and positive, got {value}")]
    InvalidWeight {
        /// The offending value.
        value: f64,
    },
    /// A change in outcome is not finite.
    #[error("delta outcomes must be finite, got {value}")]
    InvalidOutcome {
        /// The offending value.
        value: f64,
    },
    /// A covariate is not finite.
    #[error("covariates must be finite, got {value}")]
    InvalidCovariate {
        /// The offending value.
        value: f64,
    },
    /// The rows differ in their number of covariates.
    #[error(
        "all DR-DiD rows must have the same covariate count; expected {expected}, got {actual}"
    )]
    InconsistentCovariateCount {
        /// Number of covariates the rows should have.
        expected: usize,
        /// Number of covariates on the offending row.
        actual: usize,
    },
    /// The configuration is invalid, with a description.
    #[error("invalid DR-DiD config: {0}")]
    InvalidConfig(String),
    /// A linear system could not be solved.
    #[error("could not solve linear system")]
    SingularSystem,
}

/// One row for staggered-adoption `ATT(g,t)` estimation.
#[derive(bon::Builder, Debug, Clone, Copy, PartialEq)]
pub struct AttGtObservation {
    /// Panel identifier, required by the panel route.
    pub unit_id: Option<i64>,
    /// Period in which the unit is first treated, `None` for never treated.
    pub first_treated_time: Option<i32>,
    /// Calendar period of the observation.
    pub time: i32,
    /// Observed outcome value.
    pub outcome: f64,
    /// Sampling weight, 1.0 by default.
    #[builder(default = 1.0)]
    pub weight: f64,
}

impl AttGtObservation {
    /// Builds an observation without a unit identifier and with weight 1.0.
    #[must_use]
    pub const fn new(first_treated_time: Option<i32>, time: i32, outcome: f64) -> Self {
        Self {
            unit_id: None,
            first_treated_time,
            time,
            outcome,
            weight: 1.0,
        }
    }

    /// Builds an observation carrying its panel identifier, with weight 1.0.
    #[must_use]
    pub const fn with_unit_id(
        unit_id: i64,
        first_treated_time: Option<i32>,
        time: i32,
        outcome: f64,
    ) -> Self {
        Self {
            unit_id: Some(unit_id),
            first_treated_time,
            time,
            outcome,
            weight: 1.0,
        }
    }
}

/// Comparison-group strategy for staggered `ATT(g,t)` estimators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonGroup {
    /// Units never treated in the sample.
    NeverTreated,
    /// Units not yet treated at the period concerned.
    NotYetTreated,
    /// Never-treated units tagged with the cell's own cohort, and no others.
    ///
    /// For a risk-set matched panel, where each never-treated unit was sampled
    /// to match a treated unit of one cohort and is observed on that cohort's
    /// calendar. Each never-treated row names its cohort in
    /// [`AttGtDrObservation::comparison_cohort`], and the row is a control for
    /// that cohort's cells only. Not-yet-treated units are never controls. The
    /// basic route has no cohort tag and refuses this mode.
    MatchedNeverTreated,
}

/// Base-period convention for staggered `ATT(g,t)` estimators.
///
/// On the covariate-adjusted routes "previous" means the previous period in
/// the sorted list of the sample's periods, and the cohort's base is the last
/// period before `g - anticipation`; with consecutive integer periods that is
/// `g - 1 - anticipation`. The route without covariates uses the arithmetic
/// `t - 1` and `g - 1 - anticipation`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasePeriod {
    /// Pre-treatment cells use the previous period as baseline, and
    /// post-treatment cells the cohort's base.
    Varying,
    /// Every cell uses the cohort's base as baseline.
    Universal,
}

/// How the panel routes treat units that are not observed in every period.
///
/// The repeated cross-section routes ignore it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnbalancedPanel {
    /// Drop every unit that is missing from some period, and report how many,
    /// as `did::att_gt(allow_unbalanced_panel = FALSE)`, R's default.
    Balance,
    /// Keep every row. When the panel is unbalanced, estimate each cell with
    /// the repeated cross-section estimator on the rows of its two periods and
    /// sum each unit's rows in the influence function, as
    /// `did::att_gt(allow_unbalanced_panel = TRUE)`; a balanced panel is
    /// estimated as a panel.
    RepeatedCrossSection,
    /// Keep every row, and estimate each cell as a panel on the units observed
    /// in both of its periods. No `did` counterpart. This suits a panel whose
    /// units are observed on windows of their own, such as matched sets, where
    /// balancing would drop most of them.
    PerCell,
}

/// Configuration for `ATT(g,t)` estimation.
#[derive(bon::Builder, Debug, Clone, Copy, PartialEq)]
pub struct AttGtConfig {
    /// Confidence level of the intervals. The default is 0.95.
    #[builder(default = InferenceConfig::default())]
    pub confidence_level: InferenceConfig,
    /// Which units serve as comparison for each cohort. The default is
    /// [`ComparisonGroup::NeverTreated`].
    #[builder(default = ComparisonGroup::NeverTreated)]
    pub comparison_group: ComparisonGroup,
    /// How the baseline period is chosen. The default is
    /// [`BasePeriod::Varying`].
    #[builder(default = BasePeriod::Varying)]
    pub base_period: BasePeriod,
    /// Number of periods before first treatment in which effects may already
    /// occur, non-negative. The baseline of post-treatment cells is the last
    /// period before `g - anticipation_periods`. The default is 0.
    #[builder(default = 0_i32)]
    pub anticipation_periods: i32,
    /// Whether cells that cannot be fitted are skipped and reported, instead of
    /// causing an error. The default is `true`.
    #[builder(default = true)]
    pub skip_incomplete_pairs: bool,
    /// How the panel routes treat an unbalanced panel. The default is
    /// [`UnbalancedPanel::Balance`], as in `did`.
    #[builder(default = UnbalancedPanel::Balance)]
    pub unbalanced_panel: UnbalancedPanel,
    /// Whether the covariate-adjusted routes refuse a never-treated
    /// comparison group with fewer observations per period than the number
    /// of covariates plus 5, as `did::att_gt` stops. Read only under
    /// [`ComparisonGroup::NeverTreated`]; it never changes an estimate. The
    /// default is `true`.
    #[builder(default = true)]
    pub refuse_small_never_treated: bool,
    /// Whether a cell whose largest fitted propensity score is 0.999 or more
    /// is left out, as `did::att_gt` sets it to NA, for the doubly robust and
    /// inverse probability weighted cells. The score is that of an unweighted
    /// logistic regression of the cell's arm on its design; without
    /// covariates it is the treated share. Every cell's largest score is
    /// reported in [`AttGtInfluenceOutput::overlap`] whatever this is. The
    /// default is `true`.
    #[builder(default = true)]
    pub refuse_poor_overlap: bool,
}

impl Default for AttGtConfig {
    fn default() -> Self {
        Self {
            confidence_level: InferenceConfig::default(),
            comparison_group: ComparisonGroup::NeverTreated,
            base_period: BasePeriod::Varying,
            anticipation_periods: 0,
            skip_incomplete_pairs: true,
            unbalanced_panel: UnbalancedPanel::Balance,
            refuse_small_never_treated: true,
            refuse_poor_overlap: true,
        }
    }
}

/// One identified group-time effect estimate.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AttGtEstimate {
    /// Cohort, the period in which the group is first treated.
    pub group: i32,
    /// Calendar period of the estimate.
    pub time: i32,
    /// Event time, the calendar period minus the first treated period.
    pub event_time: i32,
    /// Estimated average treatment effect on the treated.
    pub att: f64,
    /// Standard error of the estimate, not accounting for clustering. It is
    /// analytic for the no-covariate estimator and comes from the influence
    /// function for the doubly robust, outcome-regression and weighting
    /// estimators.
    pub se: f64,
    /// Lower bound of the confidence interval. It is a two-sided normal
    /// interval, except for the covariate-adjusted estimators when
    /// [`DrDidConfig::bootstrap_reps`] is positive, where it is a
    /// multiplier-bootstrap percentile interval.
    pub ci_low: f64,
    /// Upper bound of the confidence interval, as `ci_low`.
    pub ci_high: f64,
    /// Number of treated observations used. The no-covariate estimator counts
    /// those of the later period, the repeated cross-section DR estimators those
    /// of both periods, and the panel estimators count units.
    pub treated_n: usize,
    /// Number of comparison observations used, counted as `treated_n`.
    pub control_n: usize,
    /// Sum of the weights of the observations used.
    pub total_weight: f64,
}

/// An ATT(g,t) cell the estimator could not fit and left out, which it does
/// when [`AttGtConfig::skip_incomplete_pairs`] is set.
///
/// `reason` is the estimator's own word for it: `treated_panel` or
/// `control_panel` when the cell has no units on one side, `covariate_count`
/// when a unit's covariates do not match the cell's, `panel_fit` when the
/// doubly-robust fit did not converge, `influence_length` when it returned the
/// wrong shape. On the repeated-cross-section route the reasons are the
/// missing-pair kinds of that route.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SkippedCell {
    /// Cohort of the skipped cell.
    pub group: i32,
    /// Calendar period of the skipped cell.
    pub time: i32,
    /// Baseline period of the skipped cell.
    pub baseline_time: i32,
    /// Short label of why the cell was skipped.
    pub reason: &'static str,
}

/// An ATT(g,t) cell fitted on fewer covariates than it was given.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrunedCell {
    /// Cohort of the cell.
    pub group: i32,
    /// Calendar period of the cell.
    pub time: i32,
    /// Baseline period of the cell.
    pub baseline_time: i32,
    /// How many covariate columns the cell's own sample could not identify.
    pub columns_dropped: usize,
}

/// The sample an ATT(g,t) route estimated on, after the pre-processing of
/// `did::att_gt`, and what that pre-processing changed.
///
/// The covariate-adjusted routes apply `did`'s rules before forming any cell.
/// A cohort first treated after the last period plus the anticipation
/// periods is recoded as never treated. Without never-treated units, the
/// periods from `c - anticipation_periods` on are dropped, `c` the last
/// cohort, which then serves as comparison group: as not-yet-treated units
/// under [`ComparisonGroup::NotYetTreated`], recoded as never treated under
/// [`ComparisonGroup::NeverTreated`]. Units treated in the first period, or
/// within `anticipation_periods` of it, are dropped. On a panel route the
/// panel is then balanced or not as [`AttGtConfig::unbalanced_panel`] says.
/// Base periods are taken from the sorted list of the remaining periods, so
/// periods need not be consecutive.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AttGtSample {
    /// What each position of the influence vectors stands for: a unit id on
    /// the panel routes, ascending, and the index of an input row on the
    /// repeated cross-section routes.
    pub ids: Vec<i64>,
    /// The cohort of each position after recoding, `None` for never treated.
    pub groups: Vec<Option<i32>>,
    /// The sampling weight of each position: a row's own on the repeated
    /// cross-section routes; on the panel routes the weight of the unit's
    /// earliest row, or the mean over its rows when the unbalanced panel was
    /// estimated as repeated cross-sections, as `did::aggte` reads it.
    pub weights: Vec<f64>,
    /// The periods of the sample, ascending.
    pub periods: Vec<i32>,
    /// The cohorts whose cells were estimated, ascending.
    pub cohorts: Vec<i32>,
    /// Units (panel routes) or rows (repeated cross-section routes) whose
    /// cohort was after the last period and was recoded as never treated.
    pub recoded_never_treated: usize,
    /// The last cohort, when the sample had no never-treated units and it
    /// became the comparison group.
    pub last_cohort_as_comparison: Option<i32>,
    /// Rows dropped because their period was at or after that cohort's start
    /// minus the anticipation periods.
    pub dropped_late_rows: usize,
    /// Units (panel routes) or rows (repeated cross-section routes) treated in
    /// the first period or before it, given the anticipation periods, and
    /// dropped.
    pub dropped_treated_at_start: usize,
    /// Units dropped to balance the panel under [`UnbalancedPanel::Balance`].
    pub dropped_unbalanced: usize,
    /// Whether an unbalanced panel was estimated as repeated cross-sections,
    /// under [`UnbalancedPanel::RepeatedCrossSection`].
    pub unbalanced_as_repeated_cross_section: bool,
}

/// ATT(g,t) estimates with aligned influence-function vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct AttGtInfluenceOutput {
    /// ATT(g,t) estimates.
    pub estimates: Vec<AttGtEstimate>,
    /// One influence vector per estimate, indexed as [`AttGtSample::ids`].
    pub influence_functions: Vec<Vec<f64>>,
    /// The cells that were attempted and could not be fitted, so they are
    /// missing from `estimates`. Empty when every cell fitted, and always empty
    /// when `skip_incomplete_pairs` is off, because then the first failure is
    /// an error. It shows how many of the possible cells an aggregate of
    /// `estimates` is built on.
    pub skipped: Vec<SkippedCell>,
    /// The cells fitted on a design pruned to what their own sample could
    /// identify, with how many columns each lost. A cell's sample is the units
    /// observed at both of its periods, and a covariate that varies across the
    /// input can be constant, or a combination of the others, inside it.
    pub pruned: Vec<PrunedCell>,
    /// The sample the cells were estimated on, which the influence vectors
    /// are indexed by.
    pub sample: AttGtSample,
    /// The largest fitted propensity score of every cell that had both arms,
    /// fitted or not, in the order the cells were attempted.
    pub overlap: Vec<CellOverlap>,
}

/// The overlap diagnostic of one ATT(g,t) cell: the largest propensity score
/// of `did`'s overlap check. See [`AttGtConfig::refuse_poor_overlap`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CellOverlap {
    /// Cohort of the cell.
    pub group: i32,
    /// Calendar period of the cell.
    pub time: i32,
    /// Baseline period of the cell.
    pub baseline_time: i32,
    /// The largest fitted score over the cell's treated and comparison units
    /// or rows; without covariates the treated share. `None` when the
    /// logistic fit failed.
    pub max_propensity: Option<f64>,
}

/// Event-time aggregated ATT estimates with aligned influence-function vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct AttGtEventTimeInfluenceOutput {
    /// Event-time aggregated estimates.
    pub estimates: Vec<AttGtEventTimeEstimate>,
    /// Influence vectors aligned to the full input sample index.
    pub influence_functions: Vec<Vec<f64>>,
}

/// Errors returned by `ATT(g,t)` estimation.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum AttGtError {
    /// No observations were given.
    #[error("ATT(g,t) requires at least one observation")]
    EmptyInput,
    /// The confidence level is not strictly between 0 and 1.
    #[error("confidence_level must be finite and in (0, 1)")]
    InvalidConfidenceLevel,
    /// The number of anticipation periods is negative.
    #[error("anticipation_periods must be >= 0")]
    InvalidAnticipationPeriods,
    /// A weight is not finite and positive.
    #[error("weights must be finite and positive, got {value}")]
    InvalidWeight {
        /// The offending value.
        value: f64,
    },
    /// An outcome is not finite.
    #[error("outcomes must be finite, got {value}")]
    InvalidOutcome {
        /// The offending value.
        value: f64,
    },
    /// A covariate is not finite.
    #[error("covariates must be finite, got {value}")]
    InvalidCovariate {
        /// The offending value.
        value: f64,
    },
    /// The rows differ in their number of covariates.
    #[error(
        "all ATT(g,t) rows must have the same covariate count; expected {expected}, got {actual}"
    )]
    InconsistentCovariateCount {
        /// Number of covariates the rows should have.
        expected: usize,
        /// Number of covariates on the offending row.
        actual: usize,
    },
    /// The comparison group needs never-treated units and the sample has
    /// none: the matched comparison group on the covariate-adjusted routes,
    /// which otherwise compare with the last cohort as `did` does, and the
    /// never-treated group on the route without covariates.
    #[error("never-treated comparison group is required but missing")]
    MissingNeverTreatedGroup,
    /// The matched comparison group was requested on a route that cannot carry it.
    #[error(
        "the matched comparison group needs a comparison cohort on every never-treated \
         row, which only the doubly robust routes carry"
    )]
    MatchedComparisonUnsupported,
    /// The never-treated comparison group has too few observations, as
    /// `did::att_gt` refuses it.
    #[error(
        "the never-treated group is too small to serve as a reliable control \
         ({per_period} observations per period, fewer than {required}, the number of \
         covariates plus 5); try the not-yet-treated comparison group, which includes \
         not-yet-treated units as controls"
    )]
    SmallNeverTreatedGroup {
        /// Never-treated rows divided by the number of periods.
        per_period: f64,
        /// The number of covariates plus 5.
        required: usize,
    },
    /// A unit's first treatment period differs between its rows.
    #[error(
        "unit {unit_id} has more than one first treatment period; treatment must be \
         irreversible and its start the same on every row of the unit"
    )]
    InconsistentCohort {
        /// Identifier of the unit.
        unit_id: i64,
    },
    /// A row has no unit identifier on the panel route.
    #[error(
        "panel ATT(g,t) requires a unit_id on every row: a unit cannot be \
         differenced across periods unless it can be named"
    )]
    MissingUnitId,
    /// A unit has more than one row in one period of a cell.
    #[error(
        "unit {unit_id} has more than one row for a period in cell (group {group}, time \
         {time}); a panel unit is observed once per period, and repeating rows to express \
         a frequency weight silently collapses to one unit. Use the weight field."
    )]
    DuplicatePanelRow {
        /// Identifier of the unit.
        unit_id: i64,
        /// Cohort of the cell.
        group: i32,
        /// Calendar period of the cell.
        time: i32,
    },
    /// No ATT(g,t) cell could be estimated.
    #[error("no estimable ATT(g,t) pairs found")]
    NoEstimablePairs,
    /// A required 2x2 cell is missing for a group, period and baseline.
    #[error(
        "missing required cell for group={group}, time={time}, baseline={baseline_time}, cell={cell}"
    )]
    MissingCell {
        /// Cohort, the period in which the group is first treated.
        group: i32,
        /// Calendar period of the estimate.
        time: i32,
        /// Baseline period the effect is measured against.
        baseline_time: i32,
        /// Label of the missing 2x2 cell.
        cell: &'static str,
    },
    /// The estimator failed for a group and period.
    #[error("ATT(g,t) pair estimation failed for method={method}, group={group}, time={time}")]
    PairEstimationFailure {
        /// Name of the estimator.
        method: &'static str,
        /// Cohort, the period in which the group is first treated.
        group: i32,
        /// Calendar period of the estimate.
        time: i32,
    },
    /// The estimator returned an influence vector of the wrong length.
    #[error(
        "ATT(g,t) pair influence length mismatch for method={method}, group={group}, time={time}: expected {expected}, got {actual}"
    )]
    InfluenceLengthMismatch {
        /// Name of the estimator.
        method: &'static str,
        /// Cohort, the period in which the group is first treated.
        group: i32,
        /// Calendar period of the estimate.
        time: i32,
        /// Expected length of the influence vector.
        expected: usize,
        /// Length of the influence vector returned.
        actual: usize,
    },
}

/// One row for DR staggered-adoption `ATT(g,t)` estimation with covariates.
#[derive(bon::Builder, Debug, Clone, PartialEq)]
pub struct AttGtDrObservation {
    /// Panel identifier. Required by the panel route
    /// ([`estimate_att_gt_dr_panel`](crate::estimate_att_gt_dr_panel)), which
    /// differences a unit against itself; ignored by the repeated-cross-section
    /// route, where one row is one observation.
    pub unit_id: Option<i64>,
    /// Period in which the unit is first treated, `None` for never treated.
    pub first_treated_time: Option<i32>,
    /// Calendar period of the observation.
    pub time: i32,
    /// Observed outcome value.
    pub outcome: f64,
    /// Sampling weight, 1.0 by default.
    #[builder(default = 1.0)]
    pub weight: f64,
    /// Covariate values, empty by default.
    #[builder(default)]
    pub covariates: Vec<f64>,
    /// For a never-treated unit, the cohort it is a comparator for: the first
    /// treated time of the treated unit it was matched to. Read only under
    /// [`ComparisonGroup::MatchedNeverTreated`]; `None` there means the row is
    /// a control for no cell.
    pub comparison_cohort: Option<i32>,
}

impl AttGtDrObservation {
    /// Builds an observation without a unit identifier, with weight 1.0 and no covariates.
    #[must_use]
    pub const fn new(first_treated_time: Option<i32>, time: i32, outcome: f64) -> Self {
        Self {
            unit_id: None,
            first_treated_time,
            time,
            outcome,
            weight: 1.0,
            covariates: Vec::new(),
            comparison_cohort: None,
        }
    }

    /// The same row carrying its panel identifier.
    #[must_use]
    pub const fn with_unit_id(
        unit_id: i64,
        first_treated_time: Option<i32>,
        time: i32,
        outcome: f64,
    ) -> Self {
        Self {
            unit_id: Some(unit_id),
            first_treated_time,
            time,
            outcome,
            weight: 1.0,
            covariates: Vec::new(),
            comparison_cohort: None,
        }
    }
}

/// Unified configuration surface for event-study estimation and inference.
///
/// This groups settings for DR-DiD pair estimation, ATT(g,t) identification,
/// and subsequent aggregation/inference into a single hierarchical object.
#[derive(bon::Builder, Debug, Clone, PartialEq)]
pub struct EventStudyConfig {
    /// Confidence level as a probability, such as 0.95.
    #[builder(default = 0.95)]
    pub confidence_level: f64,
    /// Settings of the doubly robust pair estimator.
    #[builder(default)]
    pub drdid: DrDidConfig,
    /// Settings of the ATT(g,t) estimation.
    #[builder(default)]
    pub att_gt: AttGtConfig,
    /// Settings of the aggregation of ATT(g,t) estimates.
    #[builder(default)]
    pub aggregation: AttGtAggregationConfig,
}

impl Default for EventStudyConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            drdid: DrDidConfig::default(),
            att_gt: AttGtConfig::default(),
            aggregation: AttGtAggregationConfig::default(),
        }
    }
}

impl EventStudyConfig {
    /// Validate the configuration and ensure no internal conflicts.
    ///
    /// # Errors
    /// Returns an error string if sub-configs contradict top-level settings.
    pub fn validate(&self) -> Result<(), String> {
        // The sub-configurations must agree with the top-level settings.
        if (self.drdid.confidence_level - self.confidence_level).abs() > 1e-9 {
            return Err(format!(
                "Conflicting confidence_level: top-level {} vs drdid {}",
                self.confidence_level, self.drdid.confidence_level
            ));
        }
        if (self.att_gt.confidence_level.confidence_level - self.confidence_level).abs() > 1e-9 {
            return Err(format!(
                "Conflicting confidence_level: top-level {} vs att_gt {}",
                self.confidence_level, self.att_gt.confidence_level.confidence_level
            ));
        }
        if (self.aggregation.confidence_level.confidence_level - self.confidence_level).abs() > 1e-9
        {
            return Err(format!(
                "Conflicting confidence_level: top-level {} vs aggregation {}",
                self.confidence_level, self.aggregation.confidence_level.confidence_level
            ));
        }
        Ok(())
    }

    /// Synchronize all leaf configurations to match the top-level settings.
    #[must_use]
    pub const fn sync(mut self) -> Self {
        self.drdid.confidence_level = self.confidence_level;
        self.att_gt.confidence_level = self.inference();
        self.aggregation.confidence_level = self.inference();
        self
    }

    /// The confidence level as an [`InferenceConfig`].
    #[must_use]
    pub const fn inference(&self) -> InferenceConfig {
        InferenceConfig {
            confidence_level: self.confidence_level,
        }
    }
}

/// Human-readable summary of an estimation result for reporting.
#[derive(serde::Serialize, Debug, Clone, PartialEq)]
pub struct ReportingSummary {
    /// Schema version for downstream pipeline compatibility.
    pub schema_version: u32,
    /// Version of this crate that produced the result.
    pub crate_version: String,
    /// Hash of the configuration, if one was computed.
    pub config_hash: Option<String>,

    // Raw numeric fields
    /// Point estimate.
    pub estimate: f64,
    /// Standard error of the estimate.
    pub std_error: f64,
    /// Lower bound of the confidence interval.
    pub ci_lower: f64,
    /// Upper bound of the confidence interval.
    pub ci_upper: f64,
    /// p-value of the estimate, if computed.
    pub p_value: Option<f64>,

    // Context
    /// Name of the outcome.
    pub outcome: String,
    /// Name of the subgroup.
    pub subgroup: String,
    /// Name of the estimator.
    pub estimator: String,
    /// Number of observations.
    pub n_obs: usize,
    /// Number of clusters, if clustered.
    pub n_clusters: Option<usize>,
}

impl ReportingSummary {
    /// Display-formatted estimate string (4 decimal places).
    #[must_use]
    pub fn estimate_display(&self) -> String {
        format!("{:.4}", self.estimate)
    }

    /// Display-formatted CI string: "[lower, upper]".
    #[must_use]
    pub fn ci_display(&self) -> String {
        format!("[{:.4}, {:.4}]", self.ci_lower, self.ci_upper)
    }
}

/// Configuration for DR `ATT(g,t)` estimation.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AttGtDrConfig {
    /// Settings of the ATT(g,t) estimation.
    pub att_gt: AttGtConfig,
    /// Settings of the doubly robust pair estimator.
    pub drdid: DrDidConfig,
}

/// Weighting rule for ATT(g,t) aggregation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttGtAggregationWeighting {
    /// Every estimate has the same weight.
    Equal,
    /// Estimates are weighted by `treated_n`, the number of treated units. In
    /// the cohort, calendar and overall summaries of `ATT(g,t)` it is the
    /// cohort's share `P(G = g)` estimated from the unit panel's sampling
    /// weights, with its estimation in the influence function, as `did::aggte`.
    ByTreatedCount,
    /// Estimates are weighted by `total_weight`, the sum of the sampling weights
    /// of the observations used.
    ByTotalWeight,
}

/// Configuration for ATT(g,t) aggregation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttGtAggregationConfig {
    /// Confidence level of the intervals. The default is 0.95.
    pub confidence_level: InferenceConfig,
    /// Weighting rule of the aggregation. The default is
    /// [`AttGtAggregationWeighting::ByTreatedCount`].
    pub weighting: AttGtAggregationWeighting,
}

impl Default for AttGtAggregationConfig {
    fn default() -> Self {
        Self {
            confidence_level: InferenceConfig::default(),
            weighting: AttGtAggregationWeighting::ByTreatedCount,
        }
    }
}

/// Aggregated ATT result for a keyed dimension.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttGtAggregatedEstimate {
    /// Weighted mean of the ATT(g,t) estimates, with the weights of
    /// [`AttGtAggregationConfig::weighting`] normalised to sum to 1.
    pub estimate: f64,
    /// Standard error of the estimate. Where influence functions are given it
    /// comes from the combined influence function. Otherwise it combines the
    /// components' standard errors as if they were independent, which
    /// understates it when the components share units.
    pub se: f64,
    /// Lower bound of the two-sided normal confidence interval.
    pub ci_low: f64,
    /// Upper bound of the two-sided normal confidence interval.
    pub ci_high: f64,
    /// Number of ATT(g,t) estimates aggregated.
    pub components: usize,
    /// The sum of the components' `total_weight`, whatever the weighting.
    pub total_weight: f64,
}

/// Event-time aggregated ATT result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttGtEventTimeEstimate {
    /// Event time, the calendar period minus the first treated period.
    pub event_time: i32,
    /// Aggregated estimate for this event time.
    pub summary: AttGtAggregatedEstimate,
}

/// Cohort aggregated ATT result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttGtCohortEstimate {
    /// Cohort, the period in which the group is first treated.
    pub group: i32,
    /// Aggregated estimate for this cohort.
    pub summary: AttGtAggregatedEstimate,
}

/// Calendar-time aggregated ATT result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttGtCalendarEstimate {
    /// Calendar period of the estimate.
    pub time: i32,
    /// Aggregated estimate for this calendar period.
    pub summary: AttGtAggregatedEstimate,
}

/// Errors returned by ATT(g,t) aggregation helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AttGtAggregationError {
    /// No estimates were given.
    #[error("ATT(g,t) aggregation requires at least one estimate")]
    EmptyInput,
    /// The confidence level is not strictly between 0 and 1.
    #[error("confidence_level must be finite and in (0, 1)")]
    InvalidConfidenceLevel,
    /// An estimate is not finite.
    #[error("ATT(g,t) estimate value must be finite")]
    InvalidEstimate,
    /// A standard error is not finite or is negative.
    #[error("ATT(g,t) standard error must be finite and non-negative")]
    InvalidSe,
}

/// Configuration for simultaneous confidence bands over ATT(g,t) estimates.
///
/// Used by the multiplier-bootstrap bands built from influence functions and by
/// the Gaussian-max bands built from standard errors alone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttGtBandConfig {
    /// Joint confidence level of the bands. The default is 0.95.
    pub confidence_level: InferenceConfig,
    /// Number of replications of the draw of the maximum absolute t statistic,
    /// positive. The default is 999.
    pub reps: usize,
    /// Seed of the random draws. The default is 17,431.
    pub seed: u64,
}

impl Default for AttGtBandConfig {
    fn default() -> Self {
        Self {
            confidence_level: InferenceConfig::default(),
            reps: 999,
            seed: 17_431,
        }
    }
}

/// One row for continuous treatment `DiD` (CGBS 2024).
#[derive(bon::Builder, Debug, Clone, Copy, PartialEq)]
pub struct ContinuousObservation {
    /// Treatment dose (D). A dose of exactly 0 marks a control unit and a
    /// positive dose a treated unit.
    pub dose: f64,
    /// Change in outcome (ΔY) between the post and pre period.
    pub delta_outcome: f64,
    /// Sampling weight, 1.0 by default.
    #[builder(default = 1.0)]
    pub weight: f64,
}

impl ContinuousObservation {
    /// Builds an observation with weight 1.0.
    #[must_use]
    pub const fn new(dose: f64, delta_outcome: f64) -> Self {
        Self {
            dose,
            delta_outcome,
            weight: 1.0,
        }
    }
}

/// Result of an ACRT (Average Causal Response) Sieve estimation.
#[derive(Debug, Clone, PartialEq)]
pub struct ACRTResult {
    /// Unweighted mean over the treated units of the derivative of the fitted
    /// dose-response function at their dose.
    pub acrt_glob: f64,
    /// Sieve coefficients, one per basis function in the order of the basis.
    pub coefficients: Vec<f64>,
    /// Standard error of `acrt_glob` from its influence function.
    pub se: f64,
    /// Influence function for `acrt_glob`, one value per input observation,
    /// scaled so that the estimate moves by its mean. Units with a negative dose
    /// have zero influence.
    pub influence_function: Vec<f64>,
}

/// Errors returned by continuous `DiD` estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ContinuousDidError {
    /// No observations were given.
    #[error("continuous DiD requires at least one observation")]
    EmptyInput,
    /// No unit has a positive dose.
    #[error("no treated units found (dose > 0)")]
    NoTreatedUnits,
    /// No unit with dose 0 has a positive weight.
    #[error("no control units of positive weight found (dose = 0)")]
    NoControlUnits,
    /// A dose, outcome or weight is not finite, or a weight is negative.
    #[error("a dose, outcome or weight is not finite, or a weight is negative")]
    InvalidObservation,
    /// The basis matrix is singular.
    #[error("basis matrix is singular")]
    SingularBasis,
}

/// One row for Triple-Difference `DiD` (OVS 2025).
#[derive(bon::Builder, Debug, Clone, PartialEq)]
pub struct TripleDidObservation {
    /// Treatment indicator. The estimator does not read it: the subgroup is
    /// fixed by `group_s` and `partition_q`.
    pub treated: bool,
    /// Group indicator (S): whether the unit is in the group that is treated,
    /// such as an enabled region or cohort.
    pub group_s: bool,
    /// Partition indicator (Q): whether the unit is in the eligible partition,
    /// such as an eligible age group.
    pub partition_q: bool,
    /// Change in outcome (ΔY) between the post and pre period.
    pub delta_outcome: f64,
    /// Sampling weight, 1.0 by default.
    #[builder(default = 1.0)]
    pub weight: f64,
    /// Covariate values of the doubly robust estimation, empty by default.
    #[builder(default)]
    pub covariates: Vec<f64>,
}

/// Result of a Triple-Difference estimation.
#[derive(Debug, Clone, PartialEq)]
pub struct TripleDidResult {
    /// Estimated triple-difference effect, the ATT of the focal group `S = 1,
    /// Q = 1` against the three comparison groups.
    pub att_ddd: f64,
    /// Standard error from the influence function.
    pub se: f64,
    /// Lower end of the confidence interval, normal or bootstrap as
    /// [`DrDidEstimate::ci_low`].
    pub ci_low: f64,
    /// Upper end of the confidence interval, as `ci_low`.
    pub ci_high: f64,
    /// Influence function of the estimate, one value per input observation.
    pub influence_function: Vec<f64>,
}

/// Simultaneous-band result for one ATT(g,t) entry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttGtBandEstimate {
    /// Cohort, the period in which the group is first treated.
    pub group: i32,
    /// Calendar period of the estimate.
    pub time: i32,
    /// Event time, the calendar period minus the first treated period.
    pub event_time: i32,
    /// Estimated ATT(g,t).
    pub att: f64,
    /// Standard error the band is built on. It comes from the influence
    /// function where one is given, and is the estimate's own standard error
    /// for the bands without influence functions. It is `NaN` where
    /// [`att_gt_mboot_bands`](crate::att_gt_mboot_bands) found the draws
    /// degenerate.
    pub se: f64,
    /// Lower end of the simultaneous band, `att` minus the critical value times
    /// `se`.
    pub band_low: f64,
    /// Upper end of the simultaneous band, `att` plus the critical value times
    /// `se`.
    pub band_high: f64,
}

/// Errors returned by ATT(g,t) simultaneous-band helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AttGtBandError {
    /// No estimates were given.
    #[error("ATT(g,t) simultaneous bands require at least one estimate")]
    EmptyInput,
    /// The confidence level is not strictly between 0 and 1.
    #[error("confidence_level must be finite and in (0, 1)")]
    InvalidConfidenceLevel,
    /// The number of replications is zero.
    #[error("reps must be > 0")]
    InvalidReps,
    /// An estimate is not finite.
    #[error("ATT(g,t) estimate value must be finite")]
    InvalidEstimate,
    /// A standard error is not finite or is negative.
    #[error("ATT(g,t) standard error must be finite and non-negative")]
    InvalidSe,
    /// The number of influence vectors differs from the number of estimates.
    #[error("number of influence vectors must match estimate count")]
    InfluenceCountMismatch,
    /// An influence vector is empty.
    #[error("influence vectors must be non-empty")]
    EmptyInfluence,
    /// The influence vectors differ in length.
    #[error("all influence vectors must share the same sample length")]
    InconsistentInfluenceLength,
    /// An influence vector contains a non-finite value.
    #[error("influence vectors must contain only finite values")]
    InvalidInfluence,
    /// An influence vector has zero variance.
    #[error("influence vectors must have non-zero variance")]
    DegenerateInfluence,
}

/// Configuration for the baseline no-covariate `Efficient_DiD` panel path.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EfficientDidConfig {
    /// Confidence interval level.
    pub confidence_level: InferenceConfig,
}

/// One baseline contribution to an efficient `ATT(g,t)` estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EfficientBaselineWeight {
    /// Pre-treatment baseline period used by the underlying `DiD`.
    pub baseline_time: i32,
    /// Baseline-specific `ATT(g,t)` estimate.
    pub att: f64,
    /// Efficient aggregation weight attached to that baseline.
    pub weight: f64,
    /// Treated units with complete observations for this baseline pair.
    pub treated_n: usize,
    /// Never-treated units with complete observations for this baseline pair.
    pub control_n: usize,
}

/// Diagnostics for the baseline no-covariate `Efficient_DiD` weighting step.
#[derive(Debug, Clone, PartialEq)]
pub struct EfficientDidDiagnostics {
    /// Baseline-by-baseline covariance matrix of the influence-function inputs.
    pub baseline_covariance: Vec<Vec<f64>>,
    /// Raw precision-system solution before normalization.
    pub raw_precision_solution: Vec<f64>,
    /// Ridge penalty used to stabilize the covariance inversion, if any.
    pub ridge_penalty: Option<f64>,
}

/// Efficient baseline-weighted `ATT(g,t)` estimate for panel data.
#[derive(Debug, Clone, PartialEq)]
pub struct EfficientDidEstimate {
    /// Treatment cohort.
    pub group: i32,
    /// Post-treatment calendar time.
    pub time: i32,
    /// Event time `time - group`.
    pub event_time: i32,
    /// Efficiently weighted `ATT(g,t)`.
    pub att: f64,
    /// Standard error from the combined efficient influence function.
    pub se: f64,
    /// Lower bound of the two-sided normal confidence interval.
    pub ci_low: f64,
    /// Upper bound of the two-sided normal confidence interval.
    pub ci_high: f64,
    /// Number of treated units used in the panel comparison.
    pub treated_n: usize,
    /// Number of never-treated units used in the panel comparison.
    pub control_n: usize,
    /// Sum of the sampling weights of the observations used, those of the first
    /// baseline's comparison. For [`estimate_att_gt_efficient`](crate::estimate_att_gt_efficient)
    /// it is the sum of the weights of the cell's units, each counted once.
    pub total_weight: f64,
    /// Baseline-specific contributions and efficient weights.
    pub baseline_weights: Vec<EfficientBaselineWeight>,
    /// Combined influence function for the efficient `ATT(g,t)` estimate,
    /// scaled so that the standard error is `sqrt(sum psi_i^2) / n`. It has one
    /// entry per unit of the panel, in ascending order of identifier, for
    /// [`estimate_att_gt_efficient`](crate::estimate_att_gt_efficient), and one
    /// per input row for
    /// [`estimate_att_gt_efficient_dr`](crate::estimate_att_gt_efficient_dr).
    pub influence_function: Vec<f64>,
    /// Diagnostics for the efficient weighting step.
    pub diagnostics: EfficientDidDiagnostics,
}

/// Event-time aggregated efficient `DiD` estimate.
#[derive(Debug, Clone, PartialEq)]
pub struct EfficientDidEventTimeEstimate {
    /// Event time `e = t - g`.
    pub event_time: i32,
    /// Aggregated summary for this event time.
    pub summary: AttGtAggregatedEstimate,
}

/// Event-time aggregated efficient `DiD` estimates with aligned influence
/// vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct EfficientDidEventTimeInfluenceOutput {
    /// Event-time aggregated estimates.
    pub estimates: Vec<EfficientDidEventTimeEstimate>,
    /// Influence function of each aggregated estimate, one vector per estimate.
    pub influence_functions: Vec<Vec<f64>>,
}

/// Errors returned by the baseline no-covariate `Efficient_DiD` path.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum EfficientDidError {
    /// No observations were given.
    #[error("efficient DiD requires at least one observation")]
    EmptyInput,
    /// The confidence level is not strictly between 0 and 1.
    #[error("confidence_level must be finite and in (0, 1)")]
    InvalidConfidenceLevel,
    /// A row has no unit identifier.
    #[error("efficient DiD requires panel unit_id values on every row")]
    MissingUnitId,
    /// A weight is not finite and positive.
    #[error("weights must be finite and positive, got {value}")]
    InvalidWeight {
        /// The offending value.
        value: f64,
    },
    /// An outcome is not finite.
    #[error("outcomes must be finite, got {value}")]
    InvalidOutcome {
        /// The offending value.
        value: f64,
    },
    /// The sample has no never-treated units.
    #[error("never-treated comparison group is required but missing")]
    MissingNeverTreatedGroup,
    /// A treated cohort has no pre-treatment baseline periods.
    #[error("treated cohort {group} has no pre-treatment baseline periods")]
    NoPrePeriods {
        /// Treated cohort.
        group: i32,
    },
    /// A treated cohort has no post-treatment periods.
    #[error("treated cohort {group} has no post-treatment periods")]
    NoPostPeriods {
        /// Treated cohort.
        group: i32,
    },
    /// A cell has no treated units observed in every baseline.
    #[error("group {group} time {time} has no complete treated units across all baselines")]
    MissingTreatedPanel {
        /// Treated cohort.
        group: i32,
        /// Calendar period.
        time: i32,
    },
    /// A cell has no never-treated units observed in every baseline.
    #[error("group {group} time {time} has no complete never-treated units across all baselines")]
    MissingControlPanel {
        /// Treated cohort.
        group: i32,
        /// Calendar period.
        time: i32,
    },
    /// The baseline weighting system is singular.
    #[error("could not solve efficient baseline weighting system")]
    SingularWeightingSystem,
    /// A unit has two rows in the same period.
    #[error("unit {unit_id} has more than one row at time {time}")]
    DuplicateObservation {
        /// The unit.
        unit_id: i64,
        /// The period.
        time: i32,
    },
    /// A unit's rows give different first treatment periods.
    #[error("unit {unit_id} has rows with different first treatment periods")]
    TimeVaryingCohort {
        /// The unit.
        unit_id: i64,
    },
    /// A unit's rows give different sampling weights.
    #[error("unit {unit_id} has rows with different sampling weights")]
    TimeVaryingWeight {
        /// The unit.
        unit_id: i64,
    },
}
