use super::solver_tally::SolverCounts;
use crate::sensitivity::EngineError;
use crate::util::InferenceConfig;

/// Sensitivity restrictions for `HonestDiD` robustness analysis.
///
/// `Smoothness(M)` corresponds to the `ΔSD(M)` class, which bounds changes in
/// slope across adjacent event-time violations.
///
/// `RelativeMagnitude(Mbar)` corresponds to the `ΔRM(Mbar)` class, which bounds
/// post-treatment violations by a multiple of the largest pre-treatment
/// violation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HonestSensitivity {
    /// Smoothness restriction `ΔSD(M)`. The value is `M >= 0`, the bound on the
    /// change of slope between adjacent event times.
    Smoothness(f64),
    /// Relative-magnitude restriction `ΔRM(Mbar)`. The value is `Mbar >= 0`, the
    /// factor applied to the largest pre-treatment violation.
    RelativeMagnitude(f64),
}

/// Result of a scalar `HonestDiD` robustness calculation.
#[derive(Debug, Clone, PartialEq)]
#[cfg(test)]
pub struct HonestResult {
    /// Point estimate of the post-treatment coefficient, without any
    /// correction for violations of parallel trends.
    pub original_estimate: f64,
    /// Standard error of `original_estimate`.
    pub original_se: f64,
    /// Robust confidence interval as `(lower, upper)` at the requested
    /// confidence level.
    pub robust_ci: (f64, f64),
    /// Identified set as `(lower, upper)`: the range of values of the effect
    /// consistent with the restriction, before sampling error. Both bounds are
    /// NaN when the set is empty.
    pub identified_set: (f64, f64),
    /// Value of the sensitivity parameter: `M` for a smoothness restriction,
    /// `Mbar` for a relative-magnitude restriction.
    pub m: f64,
}

/// `HonestDiD` result for one post-treatment event-study period.
#[derive(Debug, Clone, PartialEq)]
#[cfg(test)]
pub struct HonestEventStudyPoint {
    /// Label of the post-treatment period, as listed in
    /// [`HonestEventStudyInput::post_periods`].
    pub post_period: i32,
    /// `HonestDiD` scalar result for the coefficient of that period.
    pub result: HonestResult,
}

/// Typed input for event-study `HonestDiD` analysis.
///
/// Coefficients are stored in pre-then-post order. The covariance matrix must
/// use the same ordering.
#[derive(Debug, Clone, PartialEq)]
pub struct HonestEventStudyInput {
    /// Event-study coefficient vector in pre-then-post order.
    pub betahat: Vec<f64>,
    /// Covariance matrix of `betahat`, square with one row per coefficient.
    pub covariance: Vec<Vec<f64>>,
    /// Event-time labels of the pre-treatment coefficients. Only their number
    /// enters the calculations, which split `betahat` after that many entries.
    pub pre_periods: Vec<i32>,
    /// Event-time labels of the post-treatment coefficients, in the order of the
    /// post part of `betahat`. Functionals and period arguments are matched
    /// against these labels.
    pub post_periods: Vec<i32>,
}

/// Original parallel-trends confidence set for a post-treatment linear
/// functional.
#[derive(Debug, Clone, PartialEq)]
pub struct HonestOriginalConfidenceSet {
    /// Point estimate of the linear functional, `post_weights` applied to the
    /// post-treatment coefficients.
    pub estimate: f64,
    /// Standard error of the functional, from the post-treatment block of the
    /// covariance matrix.
    pub se: f64,
    /// Two-sided Wald interval `(lower, upper)` for the functional at the
    /// requested confidence level.
    pub ci: (f64, f64),
}

/// Exact `ΔRM(Mbar)` identified set.
#[derive(Debug, Clone, PartialEq)]
pub struct HonestIdentifiedSet {
    /// Smallest value of the effect functional consistent with the restriction
    /// and the estimated pre- and post-treatment coefficients.
    pub lb: f64,
    /// Largest value of the effect functional consistent with the restriction
    /// and the estimated pre- and post-treatment coefficients.
    pub ub: f64,
}

/// Conditional confidence set under a relative-magnitude sensitivity
/// restriction.
#[derive(Debug, Clone, PartialEq)]
pub struct HonestConditionalConfidenceSet {
    /// Lower end of the confidence set.
    pub lb: f64,
    /// Upper end of the confidence set.
    pub ub: f64,
}

/// Hybrid mode for `DeltaRM` conditional confidence sets.
///
/// This matches the `hybrid_flag` argument used by the reference
/// `HonestDiD::computeConditionalCS_DeltaRM` implementation:
///
/// - `LeastFavorable` corresponds to `"LF"`
/// - `ArpOnly` corresponds to `"ARP"`
///
/// The least-favorable mode is the default of
/// [`RelativeMagnitudeConfidenceSetConfig::from_inference`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelativeMagnitudeHybrid {
    /// Least-favorable hybrid conditional confidence set (`"LF"` in R).
    LeastFavorable,
    /// ARP-only conditional confidence set (`"ARP"` in R).
    ArpOnly,
}

/// Hybrid mode for `DeltaSD` conditional confidence sets.
///
/// This matches the `hybrid_flag` argument used by the reference
/// `HonestDiD::computeConditionalCS_DeltaSD` implementation:
///
/// - `Flci` corresponds to `"FLCI"`
/// - `LeastFavorable` corresponds to `"LF"`
/// - `ArpOnly` corresponds to `"ARP"`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmoothnessHybrid {
    /// Conditional FLCI hybrid (`"FLCI"` in R).
    Flci,
    /// Least-favorable hybrid conditional confidence set (`"LF"` in R).
    LeastFavorable,
    /// ARP-only conditional confidence set (`"ARP"` in R).
    ArpOnly,
}

/// Sign restriction on the post-treatment bias, that is the deviation from
/// parallel trends, in a sensitivity analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HonestBiasDirection {
    /// The post-treatment bias is restricted to be non-negative (`PB`).
    Positive,
    /// The post-treatment bias is restricted to be non-positive (`NB`).
    Negative,
}

/// Monotonicity restriction on the deviation from parallel trends in a
/// sensitivity analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HonestMonotonicityDirection {
    /// The deviation from parallel trends is restricted to be non-decreasing in
    /// event time, across the pre- and post-treatment periods (`ΔI`). The
    /// deviation at the reference period is zero, so the pre-period deviations
    /// are at most 0 and the post-period deviations at least 0.
    Increasing,
    /// The deviation from parallel trends is restricted to be non-increasing in
    /// event time, across the pre- and post-treatment periods (`ΔD`).
    Decreasing,
}

/// Which pre-treatment violation the relative-magnitude bound `Mbar` is
/// multiplied by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HonestRelativeMagnitudeBound {
    /// Post-period changes in the deviation from parallel trends are bounded by
    /// `Mbar` times the largest such change between adjacent pre-periods
    /// (`ΔRM`). This is the default.
    ParallelTrendsDeviation,
    /// Post-period deviations from a linear trend are bounded by `Mbar` times
    /// the largest pre-period deviation from a linear trend, the largest second
    /// difference of the pre-period coefficients (`ΔSDRM`).
    LinearTrendDeviation,
}

/// Monte Carlo settings of the simulation behind a least-favorable critical
/// value, from the caller's draws and seed. `HonestDiD` uses 1,000 draws from
/// seed 0. More draws shrink the simulation error of the critical value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeastFavorableSimulation {
    /// Number of simulated draws.
    pub draws: usize,
    /// Seed of the simulation.
    pub seed: u64,
}

impl LeastFavorableSimulation {
    /// Settings with the given number of draws and seed.
    #[must_use]
    pub fn new(draws: usize, seed: u64) -> Self {
        Self { draws, seed }
    }

    /// Check that the number of draws is positive.
    ///
    /// # Errors
    /// Returns an error if `draws` is zero.
    pub fn validate(self) -> Result<(), EngineError> {
        if self.draws == 0 {
            return Err(EngineError::domain(
                "least-favourable draws",
                0.0,
                "[1, inf)",
            ));
        }
        Ok(())
    }
}

/// Configuration for `DeltaSD` conditional confidence-set construction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SmoothnessConfidenceSetConfig {
    /// Hybrid conditional confidence-set mode. The default is
    /// [`SmoothnessHybrid::Flci`].
    pub hybrid: SmoothnessHybrid,
    /// Size `κ` of the first-stage test of the hybrid, a probability in
    /// `[0, alpha]` where `alpha = 1 - confidence_level`. The default is
    /// `alpha / 10`.
    pub hybrid_kappa: f64,
    /// Whether the ARP test uses only the moments that involve a post-treatment
    /// coefficient. It has no effect when there is a single post period. The
    /// default is `true`.
    pub post_period_moments_only: bool,
    /// Number of equally spaced values of the target parameter at which the
    /// test is inverted to form the confidence set. At least 2. The default is
    /// 1,000.
    pub grid_points: usize,
}

impl SmoothnessConfidenceSetConfig {
    /// Build the default conditional-CS configuration for a target confidence
    /// level.
    #[must_use]
    pub fn from_inference(inference: InferenceConfig) -> Self {
        Self {
            hybrid: SmoothnessHybrid::Flci,
            hybrid_kappa: (1.0 - inference.confidence_level) / 10.0,
            post_period_moments_only: true,
            grid_points: 1_000,
        }
    }

    /// Validate the configuration against the requested confidence level.
    ///
    /// # Errors
    /// Returns an error if `hybrid_kappa` is not finite, outside `[0, alpha]`,
    /// or if the requested grid size is too small.
    pub fn validate(self, inference: InferenceConfig) -> Result<(), EngineError> {
        let alpha = 1.0 - inference.confidence_level;
        if !self.hybrid_kappa.is_finite() {
            return Err(EngineError::domain(
                "hybrid_kappa",
                self.hybrid_kappa,
                "[0, alpha]",
            ));
        }
        if !(0.0..=alpha).contains(&self.hybrid_kappa) {
            return Err(EngineError::domain(
                "hybrid_kappa",
                self.hybrid_kappa,
                "[0, alpha]",
            ));
        }
        if self.grid_points < 2 {
            return Err(EngineError::domain(
                "grid_points",
                self.grid_points as f64,
                "[2, inf)",
            ));
        }
        Ok(())
    }
}

/// Configuration for `DeltaRM` conditional confidence-set construction.
///
/// The reference `HonestDiD` implementation exposes this as the pair
/// `(hybrid_flag, hybrid_kappa)`. We keep the same semantics, but express them
/// as a typed Rust configuration object.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RelativeMagnitudeConfidenceSetConfig {
    /// Hybrid conditional confidence-set mode. The default is
    /// [`RelativeMagnitudeHybrid::LeastFavorable`].
    pub hybrid: RelativeMagnitudeHybrid,
    /// Size `κ` of the first-stage test of the hybrid, a probability in
    /// `[0, alpha]` where `alpha = 1 - confidence_level`. The default is
    /// `alpha / 10`.
    pub hybrid_kappa: f64,
    /// Number of simulated draws behind the least-favorable critical value,
    /// positive.
    pub least_favorable_draws: usize,
    /// Seed of the simulation behind the least-favorable critical value.
    pub least_favorable_seed: u64,
}

impl RelativeMagnitudeConfidenceSetConfig {
    /// Build the default conditional-CS configuration for a target confidence
    /// level, with the draws and seed of `simulation`.
    ///
    /// `hybrid_kappa` is `alpha / 10`, with `alpha = 1 - confidence_level`. The
    /// first stage selects the `1 - kappa` quantile of the simulated
    /// least-favorable distribution from the given number of draws.
    #[must_use]
    pub fn from_inference(
        inference: InferenceConfig,
        simulation: LeastFavorableSimulation,
    ) -> Self {
        Self {
            hybrid: RelativeMagnitudeHybrid::LeastFavorable,
            hybrid_kappa: (1.0 - inference.confidence_level) / 10.0,
            least_favorable_draws: simulation.draws,
            least_favorable_seed: simulation.seed,
        }
    }

    /// The draws and seed of the least-favorable simulation.
    #[must_use]
    pub const fn simulation(self) -> LeastFavorableSimulation {
        LeastFavorableSimulation {
            draws: self.least_favorable_draws,
            seed: self.least_favorable_seed,
        }
    }

    /// Validate the configuration against the requested confidence level.
    ///
    /// # Errors
    /// Returns an error if `hybrid_kappa` is not finite or is outside
    /// `[0, alpha]`, where `alpha = 1 - confidence_level`.
    pub fn validate(self, inference: InferenceConfig) -> Result<(), EngineError> {
        let alpha = 1.0 - inference.confidence_level;
        if !self.hybrid_kappa.is_finite() {
            return Err(EngineError::domain(
                "hybrid_kappa",
                self.hybrid_kappa,
                "[0, alpha]",
            ));
        }
        if self.least_favorable_draws == 0 {
            return Err(EngineError::domain(
                "least-favourable draws",
                0.0,
                "[1, inf)",
            ));
        }
        if !(0.0..=alpha).contains(&self.hybrid_kappa) {
            return Err(EngineError::domain(
                "hybrid_kappa",
                self.hybrid_kappa,
                "[0, alpha]",
            ));
        }
        Ok(())
    }
}

/// Study-facing assessment of whether a target functional remains
/// distinguishable from a null value under a given sensitivity restriction.
#[derive(Debug, Clone, PartialEq)]
pub struct HonestAssessment {
    /// Two-sided Wald interval `(lower, upper)` for the functional, valid under
    /// exact parallel trends, at the requested confidence level.
    pub original_ci: (f64, f64),
    /// Robust `HonestDiD` confidence interval `(lower, upper)` at the requested
    /// confidence level.
    pub robust_ci: (f64, f64),
    /// Identified set `(lower, upper)` of the functional under the restriction.
    /// Under a smoothness restriction both bounds are NaN when the set is
    /// empty, that is when a second difference of the pre-treatment
    /// coefficients exceeds `M`.
    pub identified_set: (f64, f64),
    /// Point estimate of the functional, `post_weights` applied to the
    /// post-treatment coefficients.
    pub estimate: f64,
    /// Standard error of the estimate.
    pub se: f64,
    /// Value of the functional that `robustly_significant` is tested against.
    pub null_value: f64,
    /// Value of the sensitivity parameter: `M` for a smoothness restriction,
    /// `Mbar` for a relative-magnitude restriction.
    pub sensitivity_value: f64,
    /// Whether `null_value` lies outside `robust_ci`.
    pub robustly_significant: bool,
}

/// One post-treatment point inside a conservative joint `HonestDiD` path region.
///
/// The interval is computed from a scalar `HonestDiD` assessment at an adjusted
/// confidence level so that the collection of intervals forms a family-wise
/// confidence region for the whole post-treatment path.
#[derive(Debug, Clone, PartialEq)]
pub struct HonestJointPathPoint {
    /// Label of the post-treatment period, as listed in
    /// [`HonestEventStudyInput::post_periods`].
    pub post_period: i32,
    /// Scalar assessment of this period's coefficient, run at the region's
    /// `pointwise_confidence_level` rather than at the requested level.
    pub assessment: HonestAssessment,
}

/// Method used to calibrate the simultaneous coverage of a set of post-treatment
/// intervals, one per period or per functional.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HonestJointPathMethod {
    /// Each interval is run at pointwise level `1 - alpha / k`, where `k` is the
    /// number of intervals. Ignores the covariance between them.
    Bonferroni,
    /// The pointwise level is set from a max-`|t|` critical value simulated from
    /// a multivariate normal with the covariance of the `k` functionals, so that
    /// the intervals cover jointly at the requested level.
    GaussianSimulated,
}

/// Configuration for simultaneous joint post-treatment path regions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HonestJointPathConfig {
    /// Method used to set the pointwise level so that the family of intervals
    /// has the requested joint coverage.
    pub method: HonestJointPathMethod,
    /// Number of simulated draws behind the max-`|t|` critical value, positive.
    /// Read only by [`HonestJointPathMethod::GaussianSimulated`].
    pub simulation_draws: usize,
    /// Seed of the simulation behind the max-`|t|` critical value. Read only
    /// by [`HonestJointPathMethod::GaussianSimulated`].
    pub simulation_seed: u64,
}

impl HonestJointPathConfig {
    /// Bonferroni calibration, which simulates nothing, so its draws and seed
    /// are zero and never read.
    #[must_use]
    pub const fn bonferroni() -> Self {
        Self {
            method: HonestJointPathMethod::Bonferroni,
            simulation_draws: 0,
            simulation_seed: 0,
        }
    }

    /// Gaussian-simulated calibration with the caller's draws and seed.
    #[must_use]
    pub const fn gaussian(simulation: LeastFavorableSimulation) -> Self {
        Self {
            method: HonestJointPathMethod::GaussianSimulated,
            simulation_draws: simulation.draws,
            simulation_seed: simulation.seed,
        }
    }

    /// Gaussian-simulated calibration with 20,000 draws and seed 42.
    #[must_use]
    #[cfg(test)]
    pub const fn default_for_production() -> Self {
        Self {
            method: HonestJointPathMethod::GaussianSimulated,
            simulation_draws: 20_000,
            simulation_seed: 42,
        }
    }
}

/// Simultaneous confidence region for the full post-treatment path.
///
/// This is implemented as a simultaneous rectangle over the post-treatment
/// coefficients. Depending on configuration, the rectangle is calibrated by a
/// conservative Bonferroni correction or by a tighter covariance-aware
/// simulated max-`t` critical value. It provides simultaneous coverage of the
/// whole post-treatment path at the requested confidence level, but it is
/// still not the tightest joint region obtainable from the full `HonestDiD`
/// optimization surface.
#[derive(Debug, Clone, PartialEq)]
pub struct HonestJointPathRegion {
    /// Requested joint confidence level, a probability such as 0.95.
    pub confidence_level: f64,
    /// Confidence level of each period's scalar interval, a probability at least
    /// `confidence_level`.
    pub pointwise_confidence_level: f64,
    /// Two-sided critical value of the max-`|t|` statistic across periods that
    /// corresponds to `pointwise_confidence_level`.
    pub calibrated_max_t_critical_value: f64,
    /// Method used to calibrate the simultaneous region.
    pub method: HonestJointPathMethod,
    /// One point per post-treatment period, in the order of
    /// [`HonestEventStudyInput::post_periods`].
    pub points: Vec<HonestJointPathPoint>,
}

/// One directional functional inside a simultaneous multi-functional
/// `HonestDiD` region.
///
/// The functional is represented by post-treatment weights `post_weights`, applied to
/// `τ_post`.
#[derive(Debug, Clone, PartialEq)]
pub struct HonestDirectionalFunctionalPoint {
    /// Weights of the functional on the post-treatment coefficients, one per
    /// post period in the order of [`HonestEventStudyInput::post_periods`].
    pub post_weights: Vec<f64>,
    /// Scalar assessment of the functional, run at the region's
    /// `pointwise_confidence_level` rather than at the requested level.
    pub assessment: HonestAssessment,
}

/// Simultaneous region over a finite set of post-treatment linear functionals.
///
/// This extends path-wise period intervals by allowing arbitrary direction sets
/// over `τ_post`. It is a practical approximation to the full optimization
/// surface: tighter than per-period rectangles when informative directions are
/// supplied, but still finite-direction rather than continuum-tight.
#[derive(Debug, Clone, PartialEq)]
pub struct HonestDirectionalRegion {
    /// Requested joint confidence level, a probability such as 0.95.
    pub confidence_level: f64,
    /// Confidence level of each functional's scalar interval, a probability at
    /// least `confidence_level`.
    pub pointwise_confidence_level: f64,
    /// Two-sided critical value of the max-`|t|` statistic across the
    /// functionals that corresponds to `pointwise_confidence_level`.
    pub calibrated_max_t_critical_value: f64,
    /// Method used to calibrate the simultaneous region.
    pub method: HonestJointPathMethod,
    /// One assessment per direction, in the order the directions were given or
    /// generated.
    pub points: Vec<HonestDirectionalFunctionalPoint>,
    /// Calibration and convergence diagnostics for this region.
    pub diagnostics: HonestDirectionalRegionDiagnostics,
}

/// Diagnostics emitted for directional/surface joint-region calibration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HonestDirectionalRegionDiagnostics {
    /// Number of distinct directions in the region, after duplicates are
    /// removed.
    pub direction_count: usize,
    /// Number of those directions that were drawn at random rather than being
    /// basis or pairwise-contrast directions.
    pub random_direction_count: usize,
    /// Number of direction sets built. It is 1 for a fixed direction set and
    /// counts the enrichment rounds for an adaptive one.
    pub iterations_run: usize,
    /// Whether the calibrated pointwise level stabilised within the tolerance
    /// before the random directions or iterations ran out. It is `true` for a
    /// fixed direction set.
    pub converged: bool,
    /// Conic programs that needed the retry with looser tolerances while the
    /// region was computed, adaptive iterations included. A retried program is
    /// solved to `1e-6` or `1e-7` instead of `1e-8`. The count covers every
    /// thread of the process that ran sensitivity computations meanwhile.
    pub solver_retries: usize,
    /// Conic programs whose final status was Clarabel's `AlmostSolved`, counted
    /// over the same span as `solver_retries`.
    pub solver_almost_solved: usize,
}

impl HonestDirectionalRegionDiagnostics {
    /// Diagnostics for a fixed direction set: one iteration, converged, and
    /// zero solver counts.
    #[must_use]
    pub const fn fixed(direction_count: usize, random_direction_count: usize) -> Self {
        Self {
            direction_count,
            random_direction_count,
            iterations_run: 1,
            converged: true,
            solver_retries: 0,
            solver_almost_solved: 0,
        }
    }

    /// The same diagnostics with `solver_retries` and `solver_almost_solved`
    /// taken from `counts`.
    #[must_use]
    pub const fn with_solver_counts(mut self, counts: SolverCounts) -> Self {
        self.solver_retries = counts.retried;
        self.solver_almost_solved = counts.almost_solved;
        self
    }

    /// Diagnostics for an adaptive direction set, with zero solver counts.
    #[must_use]
    pub const fn adaptive(
        direction_count: usize,
        random_direction_count: usize,
        iterations_run: usize,
        converged: bool,
    ) -> Self {
        Self {
            direction_count,
            random_direction_count,
            iterations_run,
            converged,
            solver_retries: 0,
            solver_almost_solved: 0,
        }
    }
}

/// One functional inside a simultaneous least-favorable confidence-interval
/// result.
#[derive(Debug, Clone, PartialEq)]
pub struct HonestMultiFlciPoint {
    /// Weights of the functional on the post-treatment coefficients, one per
    /// post period in the order of [`HonestEventStudyInput::post_periods`].
    pub post_weights: Vec<f64>,
    /// Least-favorable confidence interval `(lower, upper)` of the functional,
    /// run at the result's `pointwise_confidence_level` so that the intervals
    /// cover jointly at the requested level.
    pub flci: (f64, f64),
    /// Two-sided Wald interval `(lower, upper)` of the functional under exact
    /// parallel trends, at the requested confidence level.
    pub original_ci: (f64, f64),
    /// Identified set `(lower, upper)` of the functional under the restriction,
    /// NaN at both ends when it is empty.
    pub identified_set: (f64, f64),
    /// Value of the functional that `robustly_significant` is tested against.
    pub null_value: f64,
    /// Whether `null_value` lies outside `flci`.
    pub robustly_significant: bool,
}

/// Simultaneous least-favorable result over a functional set.
#[derive(Debug, Clone, PartialEq)]
pub struct HonestMultiFlciResult {
    /// Requested joint confidence level, a probability such as 0.95.
    pub confidence_level: f64,
    /// Confidence level of each functional's interval, a probability at least
    /// `confidence_level`.
    pub pointwise_confidence_level: f64,
    /// Two-sided critical value of the max-`|t|` statistic across the
    /// functionals that corresponds to `pointwise_confidence_level`.
    pub calibrated_max_t_critical_value: f64,
    /// Method used to calibrate the joint coverage.
    pub method: HonestJointPathMethod,
    /// One point per functional, in the order the functionals were given.
    pub points: Vec<HonestMultiFlciPoint>,
}

/// One functional of a simultaneous relative-magnitude interval result.
pub type RelativeMagnitudeMultiFlciPoint = HonestMultiFlciPoint;
/// Simultaneous interval result under a relative-magnitude restriction.
pub type RelativeMagnitudeMultiFlciResult = HonestMultiFlciResult;
/// One functional of a simultaneous smoothness interval result.
pub type SmoothnessMultiFlciPoint = HonestMultiFlciPoint;
/// Simultaneous interval result under a smoothness restriction.
pub type SmoothnessMultiFlciResult = HonestMultiFlciResult;

/// Direction-set construction for optimization-surface joint regions.
///
/// This controls which linear functionals are included when approximating the
/// full multi-period `HonestDiD` optimization surface via a finite directional
/// envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HonestOptimizationSurfaceConfig {
    /// Include the basis direction `e_t` of each post period, which selects a
    /// single period's coefficient.
    pub include_basis: bool,
    /// Include the contrast `(e_i - e_j) / sqrt(2)` of every pair of post
    /// periods `i < j`. Has no effect with a single post period.
    pub include_pairwise_contrasts: bool,
    /// Number of directions drawn at random, each normalised to unit length with
    /// the sign fixed so that its first non-zero weight is positive. Directions
    /// that duplicate an earlier one are dropped. In the adaptive variant this
    /// is the cap on random directions.
    pub random_unit_directions: usize,
    /// Seed of the random direction draw.
    pub random_seed: u64,
}

/// Adaptive enrichment settings for optimization-surface direction sets.
///
/// The adaptive routine adds random directions in batches, up to the
/// `random_unit_directions` of the surface configuration. It stops when the
/// calibrated pointwise confidence level changes by no more than
/// `pointwise_tolerance` between two rounds, or when a limit is reached.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HonestOptimizationSurfaceAdaptiveConfig {
    /// Number of random directions added per iteration, positive.
    pub random_batch_size: usize,
    /// Number of random directions the set must hold before a change in the
    /// pointwise level below the tolerance counts as convergence.
    pub min_random_for_convergence: usize,
    /// Largest absolute change in the calibrated pointwise confidence level,
    /// a probability, between two iterations that counts as convergence.
    /// Finite and non-negative.
    pub pointwise_tolerance: f64,
    /// Maximum number of enrichment iterations, positive.
    pub max_iterations: usize,
}

impl HonestOptimizationSurfaceAdaptiveConfig {
    /// Batches of 100, convergence checked from 250 random directions, tolerance
    /// 5e-4, at most 6 iterations.
    #[must_use]
    pub const fn default_for_production() -> Self {
        Self {
            random_batch_size: 100,
            min_random_for_convergence: 250,
            pointwise_tolerance: 5e-4,
            max_iterations: 6,
        }
    }

    /// Validate adaptive settings.
    ///
    /// # Errors
    /// Returns an error if the adaptive parameters are invalid.
    pub fn validate(self) -> Result<(), EngineError> {
        if self.random_batch_size == 0 {
            return Err(EngineError::domain("batch", 0.0, "[1, inf)"));
        }
        if self.max_iterations == 0 {
            return Err(EngineError::domain("max_iterations", 0.0, "[1, inf)"));
        }
        if !self.pointwise_tolerance.is_finite() || self.pointwise_tolerance < 0.0 {
            return Err(EngineError::domain(
                "tolerance",
                self.pointwise_tolerance,
                "finite and non-negative",
            ));
        }
        Ok(())
    }
}

/// Bundled configuration for the adaptive optimization-surface region API.
///
/// This groups the confidence-set, calibration and direction-set controls of
/// the adaptive region computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HonestOptimizationSurfaceAdaptiveRunConfig {
    /// `DeltaRM` conditional confidence-set construction settings.
    pub relative_magnitude: RelativeMagnitudeConfidenceSetConfig,
    /// Calibration of the joint coverage across the directions.
    pub joint: HonestJointPathConfig,
    /// Direction-set construction settings. Its `random_unit_directions` is the
    /// cap of the adaptive enrichment.
    pub surface: HonestOptimizationSurfaceConfig,
    /// Adaptive enrichment settings.
    pub adaptive: HonestOptimizationSurfaceAdaptiveConfig,
}

/// Which directions the directional region of the `HonestDiD` workflow covers.
#[derive(Debug, Clone, PartialEq)]
#[cfg(test)]
pub enum HonestWorkflowDirectionMode {
    /// One direction per post period, which is the joint path region reported
    /// as a directional region without a separate computation.
    Basis,
    /// A direction set generated from the configuration whose random part is
    /// grown adaptively.
    OptimizationSurfaceAdaptive {
        /// Direction-set construction settings. Its `random_unit_directions` caps
        /// the random directions.
        surface: HonestOptimizationSurfaceConfig,
        /// Settings of the adaptive enrichment.
        adaptive: HonestOptimizationSurfaceAdaptiveConfig,
    },
}

#[cfg(test)]
impl HonestWorkflowDirectionMode {
    /// Adaptive optimization-surface directions: the basis, the pairwise
    /// contrasts and up to 250 random directions drawn from `random_seed`,
    /// with the `default_for_production` adaptive settings.
    #[must_use]
    pub const fn adaptive_surface(random_seed: u64) -> Self {
        Self::OptimizationSurfaceAdaptive {
            surface: HonestOptimizationSurfaceConfig {
                include_basis: true,
                include_pairwise_contrasts: true,
                random_unit_directions: 250,
                random_seed,
            },
            adaptive: HonestOptimizationSurfaceAdaptiveConfig::default_for_production(),
        }
    }
}

/// Bundled configuration for the high-level `HonestDiD` workflow API.
#[derive(Debug, Clone, PartialEq)]
#[cfg(test)]
pub struct HonestWorkflowConfig {
    /// `DeltaRM` conditional confidence-set construction settings. Ignored under
    /// a smoothness restriction.
    pub relative_magnitude: RelativeMagnitudeConfidenceSetConfig,
    /// Calibration of the joint coverage of the path region, the directional
    /// region and the simultaneous intervals.
    pub joint: HonestJointPathConfig,
    /// Which directions the directional region covers.
    pub direction_mode: HonestWorkflowDirectionMode,
}

#[cfg(test)]
impl HonestWorkflowConfig {}

/// One functional assessment inside the high-level `HonestDiD` workflow
/// result.
#[derive(Debug, Clone, PartialEq)]
#[cfg(test)]
pub struct HonestFunctionalAssessmentPoint {
    /// The requested functional.
    pub functional: HonestPostFunctional,
    /// Scalar assessment of the functional at the requested confidence level.
    pub assessment: HonestAssessment,
}

/// Consolidated high-level `HonestDiD` workflow result.
#[derive(Debug, Clone, PartialEq)]
#[cfg(test)]
pub struct HonestStudyWorkflowResult {
    /// One scalar assessment per requested functional, in the order given.
    pub functional_assessments: Vec<HonestFunctionalAssessmentPoint>,
    /// Simultaneous region over the post-treatment coefficients, one interval
    /// per post period.
    pub joint_path_region: HonestJointPathRegion,
    /// Simultaneous region over the directions chosen by the workflow's
    /// direction mode.
    pub directional_region: HonestDirectionalRegion,
    /// Simultaneous least-favorable intervals for the requested functionals.
    /// The workflow always fills it.
    pub multi_flci: Option<HonestMultiFlciResult>,
}

/// Typed post-treatment linear functionals for event-study robustness
/// assessment.
///
/// Each functional is converted to a vector of weights over the post-treatment
/// coefficients `τ_post`, which is what the lower-level `HonestDiD` routines
/// take. Periods are matched against the labels in
/// [`HonestEventStudyInput::post_periods`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the library resolves its own targets; the tests use the other functionals"
    )
)]
pub enum HonestPostFunctional {
    /// The coefficient of the post period with this label. It is an error if the
    /// label is not in the input.
    Period(i32),
    /// Equal-weight average of the post periods whose labels lie in
    /// `start_period..=end_period`. Only periods present in the input count, so
    /// a window with gaps averages the periods it does contain. It is an error
    /// if `start_period` exceeds `end_period` or the window contains no post
    /// period.
    AverageWindow {
        /// Smallest label in the window, inclusive.
        start_period: i32,
        /// Largest label in the window, inclusive.
        end_period: i32,
    },
}

impl HonestEventStudyInput {
    /// Validate internal dimensions and numeric finiteness.
    ///
    /// # Errors
    /// Returns an error if the coefficient vector, covariance matrix, or period
    /// labels are inconsistent.
    pub fn validate(&self) -> Result<(), EngineError> {
        let total_periods = self.pre_periods.len() + self.post_periods.len();
        if self.betahat.len() != total_periods {
            return Err(EngineError::length(
                "event-study coefficients",
                total_periods,
                self.betahat.len(),
            ));
        }
        if let Some(found) = std::iter::once(self.covariance.len())
            .chain(self.covariance.iter().map(Vec::len))
            .find(|&len| len != total_periods)
        {
            return Err(EngineError::length(
                "event-study covariance",
                total_periods,
                found,
            ));
        }
        EngineError::check_finite("event-study coefficients", &self.betahat)?;
        for (row, values) in self.covariance.iter().enumerate() {
            if values.iter().any(|value| !value.is_finite()) {
                return Err(EngineError::not_finite("event-study covariance", row));
            }
        }
        Ok(())
    }

    /// Number of pre-treatment periods.
    #[must_use]
    pub const fn num_pre_periods(&self) -> usize {
        self.pre_periods.len()
    }

    /// Number of post-treatment periods.
    #[must_use]
    pub const fn num_post_periods(&self) -> usize {
        self.post_periods.len()
    }
}
