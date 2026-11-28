/// Whether a unit is in the treated or the control group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum TreatmentGroup {
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
}

/// Whether an observation is before or after treatment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum TimePeriod {
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
}

/// Settings of the multiplier bootstrap behind bootstrap confidence intervals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BootstrapConfig {
    /// Number of bootstrap replications.
    pub reps: usize,
    /// Seed of the bootstrap random number generator.
    pub seed: u64,
}

/// Canonical 2x2 cell identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum DidCell {
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

/// One panel-level row for doubly robust `DiD` with outcome deltas.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DrDidObservation {
    /// Whether the unit is in the treated group.
    pub treated: bool,
    /// Change in outcome between the post and pre period.
    pub delta_outcome: f64,
    /// Sampling weight.
    pub weight: f64,
    /// Covariate values.
    pub covariates: Vec<f64>,
}

/// One repeated cross-section row for doubly robust `DiD`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DrDidRepeatedObservation {
    /// Whether the unit is in the treated group.
    pub treated: bool,
    /// Whether the observation is from the post-treatment period.
    pub post_period: bool,
    /// Observed outcome value.
    pub outcome: f64,
    /// Sampling weight.
    pub weight: f64,
    /// Covariate values.
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
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct DrDidConfig {
    /// Confidence level as a probability strictly between 0 and 1, such as 0.95.
    /// The default is 0.95.
    pub confidence_level: f64,
    /// Propensity scores are clipped to `[propensity_clip, 1 - propensity_clip]`.
    /// It must lie in `(0, 0.5)`. The default is 1e-6.
    pub propensity_clip: f64,
    /// Ridge penalty added to the diagonal of the outcome regressions' normal
    /// equations, finite and non-negative. The default is 1e-8.
    pub ridge: f64,
    /// Maximum number of iterations of the propensity fit, positive. The default
    /// is 200.
    pub max_iter: usize,
    /// Convergence tolerance of the propensity fit, positive. The default is
    /// 1e-8.
    pub tol: f64,
    /// Number of replications of the multiplier bootstrap behind the confidence
    /// interval. With 0, the default, the interval is the normal interval
    /// `att +- z se` at `confidence_level`, as `DRDID` reports with its default
    /// `boot = FALSE` and `did::att_gt` with `bstrap = FALSE`. A positive count
    /// makes it the percentile interval of that many Rademacher multiplier
    /// replicates of the influence function.
    pub bootstrap_reps: usize,
    /// Seed of the multiplier bootstrap, read only with replicates.
    pub bootstrap_seed: u64,
    /// Whether the traditional repeated cross-section estimator computes as
    /// `epistat-did` 0.1 and `did_methods` 0.2 did, for reproducing results
    /// made with them: propensity score fitted without the sampling weights
    /// and clipped to `[propensity_clip, 1 - propensity_clip]`, no trimming,
    /// and an influence function without the estimation of the nuisance
    /// models. Off, the default, it is `DRDID::drdid_rc1`. The other
    /// estimators ignore it.
    pub legacy_repeated_cross_section: bool,
}

impl Default for DrDidConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            propensity_clip: 1e-6,
            ridge: 1e-8,
            max_iter: 200,
            tol: 1e-8,
            bootstrap_reps: 0,
            bootstrap_seed: 0,
            legacy_repeated_cross_section: false,
        }
    }
}

impl DrDidConfig {
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
pub(crate) struct DrDidEstimate {
    /// Estimated average treatment effect on the treated.
    pub att: f64,
    /// Standard error from the influence function, not accounting for
    /// clustering.
    pub se: f64,
    /// Lower end of the confidence interval: `att - z se` at the configured
    /// level, or the multiplier-bootstrap percentile when
    /// [`DrDidConfig::bootstrap_reps`] is positive. The improved estimators
    /// always return the normal interval.
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
pub(crate) struct DidCcConfig {
    /// Settings of the doubly robust estimator, including the confidence level
    /// and the bootstrap settings.
    pub drdid: DrDidConfig,
    /// Whether the confidence intervals are multiplier-bootstrap percentile
    /// intervals, using `drdid.bootstrap_reps` and `drdid.bootstrap_seed`, or
    /// otherwise normal intervals from the standard error. The default is
    /// `false`. With `true`, `drdid.bootstrap_reps` must be positive.
    pub bootstrap_confidence_intervals: bool,
    /// Number of cross-fitting folds, at least 1. One means no cross-fitting.
    /// The default is 1.
    pub cross_fit_folds: usize,
    /// Seed of the cross-fitting fold assignment, read only with more than one
    /// fold.
    pub cross_fit_seed: u64,
    /// Whether a propensity logit that cannot be fitted on the covariates
    /// falls back to the intercept alone and then to the shares. Off, the
    /// default, such a fit is an error.
    pub propensity_fallback: bool,
}

impl Default for DidCcConfig {
    fn default() -> Self {
        Self {
            drdid: DrDidConfig::default(),
            bootstrap_confidence_intervals: false,
            cross_fit_folds: 1,
            cross_fit_seed: 0,
            propensity_fallback: false,
        }
    }
}

/// ATT estimate from the compositional-change `DiD_CC` family.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DidCcEstimate {
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
    /// The propensity models of the estimate that took a fallback, in fold
    /// order.
    pub propensity_fallbacks: Vec<crate::methods::did_cc::PropensityFallback>,
}

/// Hausman-style comparison between robust and stationarity-based `DiD_CC` ATT
/// estimators.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DidCcHausmanTest {
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
}

/// Errors returned by `DiD_CC` estimation and testing.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub(crate) enum DidCcError {
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
    /// The variance of the Hausman difference is not positive.
    #[error("Hausman difference variance must be strictly positive")]
    DegenerateHausmanVariance,
    /// A propensity logit could not be fitted on the covariates and the
    /// fallback is off.
    #[error("the propensity logit could not be fitted on the covariates")]
    PropensityFit {
        /// The cross-fitting fold of the fit, `None` for the fit on every row.
        fold: Option<usize>,
    },
}

/// Errors returned by panel `DR-DiD` estimation.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub(crate) enum DrDidError {
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
