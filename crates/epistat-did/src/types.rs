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
pub(crate) struct CellSummary {
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
pub(crate) struct DidInputSummary {
    /// Summary of the treated pre-period cell.
    pub treated_pre: CellSummary,
    /// Summary of the treated post-period cell.
    pub treated_post: CellSummary,
    /// Summary of the control pre-period cell.
    pub control_pre: CellSummary,
    /// Summary of the control post-period cell.
    pub control_post: CellSummary,
}

#[cfg(test)]
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
