//! Group-time average treatment effects for staggered adoption, ATT(g,t), as
//! R's `did::att_gt` estimates them, and their aggregations, as `did::aggte`.
//!
//! [`estimate`] takes a [`StaggeredPanel`] and [`AttGtOptions`] and returns an
//! [`AttGt`]: one [`AttGtCell`] per cell attempted, with its estimate or the
//! reason it was skipped, the influence function of the estimated cells and
//! the sample after `did`'s pre-processing. [`AttGt::aggregate`] forms the
//! event-study, cohort, calendar and overall summaries, [`AttGt::bands`] the
//! simultaneous bands over the cells, [`AttGt::pretest`] the Wald pre-test,
//! and [`AttGt::event_study`] the event study that HonestDiD takes.
//!
//! # Algorithm
//!
//! For the cohort `g`, a comparison group `C` (never treated, not yet treated
//! by the later of the cell's two periods, or the never-treated units matched
//! to `g`) and the base period `b` (the last period before `g - anticipation`
//! under a universal base; under a varying base the same for a
//! post-treatment cell and the previous period for a pre-treatment one), the
//! doubly robust ATT(g,t) is
//!
//! ```text
//! ATT(g,t) = E[ (w_g - w_C(X)) (Y_t - Y_b - m_C(X)) ],
//! w_g      = G_g / E[G_g],
//! w_C(X)   = (p(X) C / (1 - p(X))) / E[p(X) C / (1 - p(X))],
//! ```
//!
//! with `p(X)` the logistic propensity of belonging to `g` among the units of
//! `g` and `C`, fitted by maximum likelihood as R's `glm`, and `m_C(X)` the
//! weighted least-squares regression of `Y_t - Y_b` on `X` among the units of
//! `C` \[2\]. Outcome regression drops the weights `w_C` and inverse
//! probability weighting drops `m_C`. On a repeated cross-section the outcome
//! is not differenced within a unit and the regressions are fitted by group
//! and period. The covariates are read at the earlier of the cell's two
//! periods, and the intercept is added by the estimator. Before any cell is
//! formed the panel is pre-processed as `did` 2.5.1 does it
//! ([`AttGtSample`]). The efficient estimator of Chen, Sant'Anna and Xie
//! \[3\] combines every pre-treatment baseline by generalised least squares
//! instead ([`Estimator::Efficient`]).
//!
//! # Inference
//!
//! Every cell has an influence function `psi_i` over the positions of the
//! sample, units on a panel and rows on a repeated cross-section, on the
//! per-unit scale of [`Influence`]: a cell fitted on `n_cell` of the `n`
//! positions has its values scaled by `n / n_cell`, as `did` does, so that
//! the cells line up for the aggregations. The standard error of a cell is
//! the two-period estimator's. With clusters in the panel it is
//! `sqrt(sum_c s_c^2)`, with `s_c` the sum of the centred `psi_i` in cluster
//! `c` divided by `n` and no factor `G / (G - 1)`, as `did` reports it with
//! `clustervars`, and its interval is the normal one. Sums run in a fixed
//! order, so the results do not depend on the number of threads.
//!
//! An aggregation is a weighted mean of the cells. By event time `e`,
//!
//! ```text
//! theta(e) = sum over g of P(G = g | G + e <= T) ATT(g, g + e),
//! ```
//!
//! and the cohort, calendar and overall summaries are formed as in `aggte`
//! \[1\]. The cohort shares are estimated, so their influence function enters
//! the variance of the aggregate, as `did:::wif` adds it. The module
//! [`aggte`] gives the details, the fixed-weight alternatives and the
//! multiplier bootstrap.
//!
//! # References
//!
//! 1. Callaway B, Sant'Anna PHC. Difference-in-differences with multiple time
//!    periods. J Econom 2021;225(2):200-30. doi:10.1016/j.jeconom.2020.12.001
//! 2. Sant'Anna PHC, Zhao J. Doubly robust difference-in-differences
//!    estimators. J Econom 2020;219(1):101-22.
//!    doi:10.1016/j.jeconom.2020.06.003
//! 3. Chen X, Sant'Anna PHC, Xie H. Efficient difference-in-differences and
//!    event study estimators. Working paper, 2025.
//!
//! The R references are `did` 2.5.1 (`att_gt`, `aggte`, `mboot`) and `DRDID`
//! 1.2.2.

use std::fmt;

use epistat_core::{Bootstrap, Covariance, Estimate, EventStudy, InputError, Level};
use epistat_inference::{ClusterRule, Clusters, Influence};

use crate::error::Error;
use crate::inference::did_clusters::standard_error;
use crate::inference::normal_interval;
use crate::methods::staggered::StaggeredPanel;
use crate::types::DrDidConfig;

mod aggregation;
pub mod aggte;
mod bands;
mod efficient;
mod overlap;
mod pair_estimators;
mod panel_pairs;
mod preprocess;
mod pretest;
mod rows;
#[cfg(test)]
mod tests;

pub use crate::methods::two_period::NuisanceOptions;
pub use aggte::{
    Aggte, AggteOptions, AggteOverall, AggtePoint, AggteType, AggteWeights, Band,
    EventStudyOptions, Inference,
};
pub use bands::{AttGtBand, AttGtBands};
pub use pretest::PretrendWald;

use rows::Rows;

/// Which units serve as comparison for each cohort, `did`'s `control_group`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum ComparisonGroup {
    /// Units never treated in the sample, `"nevertreated"`, `did`'s default.
    #[default]
    NeverTreated,
    /// Units not yet treated at the later of the cell's two periods plus the
    /// anticipation periods, `"notyettreated"`.
    NotYetTreated,
    /// Never-treated units matched to the cell's own cohort, and no others.
    ///
    /// For a risk-set matched panel, where each never-treated unit was sampled
    /// to match a treated unit of one cohort and is observed on that cohort's
    /// calendar. The panel names each never-treated unit's cohort in
    /// [`StaggeredPanelBuilder::comparison_cohort`](crate::StaggeredPanelBuilder::comparison_cohort),
    /// and the unit is a comparator for that cohort's cells only; a unit
    /// matched to none is a comparator for no cell. Not-yet-treated units are
    /// never comparators. No `did` counterpart.
    MatchedNeverTreated,
}

/// How the base period of a cell is chosen, `did`'s `base_period`.
///
/// "Previous" means the previous period in the sorted list of the sample's
/// periods, and the cohort's base is the last period before
/// `g - anticipation`; with consecutive integer periods that is
/// `g - 1 - anticipation`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum BasePeriod {
    /// Pre-treatment cells compare a period with the previous one, and
    /// post-treatment cells use the cohort's base, `"varying"`, `did`'s
    /// default.
    #[default]
    Varying,
    /// Every cell uses the cohort's base, `"universal"`: the fixed-baseline
    /// event study that HonestDiD takes.
    Universal,
}

/// How a panel whose units are not observed in every period is treated,
/// `did`'s `allow_unbalanced_panel`. A repeated cross-section ignores it.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum UnbalancedPanel {
    /// Drop every unit missing from some period and report how many, as
    /// `allow_unbalanced_panel = FALSE`, `did`'s default.
    #[default]
    Balance,
    /// Keep every row. When the panel is unbalanced, estimate each cell with
    /// the repeated cross-section estimator on the rows of its two periods and
    /// sum each unit's rows in the influence function, as
    /// `allow_unbalanced_panel = TRUE`; a balanced panel is estimated as a
    /// panel.
    RepeatedCrossSection,
    /// Keep every row, and estimate each cell as a panel on the units
    /// observed in both of its periods. No `did` counterpart. This suits a
    /// panel whose units are observed on windows of their own, such as
    /// matched sets, where balancing would drop most of them.
    PerCell,
}

/// The estimator of each cell, `did`'s `est_method`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Estimator {
    /// `"dr"`, `did`'s default: `DRDID::drdid_panel` on a panel and the
    /// locally efficient `DRDID::drdid_rc` on a repeated cross-section, which
    /// also fits the cells of an unbalanced panel under
    /// [`UnbalancedPanel::RepeatedCrossSection`].
    #[default]
    DoublyRobust,
    /// The traditional doubly robust estimator on a repeated cross-section,
    /// `DRDID::drdid_rc1`. Refused on a panel.
    TraditionalDoublyRobust,
    /// `"ipw"`: `DRDID::std_ipw_did_panel` or `std_ipw_did_rc`.
    InverseProbabilityWeighting,
    /// `"reg"`: `DRDID::reg_did_panel` or `reg_did_rc`.
    OutcomeRegression,
    /// The efficient ATT(g,t) of Chen, Sant'Anna and Xie under parallel trends
    /// in all periods, with never-treated comparators: every post-treatment
    /// cell combines all of its cohort's pre-treatment baselines by
    /// generalised least squares. On a panel it takes no covariates, no
    /// anticipation and only the never-treated comparison group, and uses
    /// every unit with its weight; on a repeated cross-section each baseline
    /// pair is the traditional doubly robust estimate, and a pre-treatment
    /// cell under a varying base has one baseline. Neither applies `did`'s
    /// pre-processing, and each cell reports its weights in
    /// [`AttGtCell::efficient`].
    Efficient,
}

/// Whether the rows are units followed over periods or independent
/// observations, `did`'s `panel`. A [`StaggeredPanel`] records it:
/// [`StaggeredPanel::builder`] builds a panel and
/// [`StaggeredPanel::cross_sections`] repeated cross-sections.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Design {
    /// Units observed over periods, `panel = TRUE`, `did`'s default. A cell
    /// differences each unit against itself, and the influence function is
    /// indexed by unit.
    #[default]
    Panel,
    /// Each row is its own observation, `panel = FALSE`. The influence
    /// function is indexed by row, so [`AttGtSample::units`] holds row
    /// numbers.
    RepeatedCrossSection,
}

/// The options of [`estimate`], with `did::att_gt`'s defaults, except that
/// the cells' intervals are analytic and pointwise. R's `att_gt` bootstraps
/// (`bstrap = TRUE`) and prints uniform bands (`cband = TRUE`), which need a
/// seed: [`AttGt::bands`] gives the bands over the cells, and
/// [`AggteOptions::inference`] with [`Inference::UniformBand`] those of an
/// aggregation, as R prints them.
///
/// A [`StaggeredPanel::cross_sections`] view is estimated as repeated
/// cross-sections, `panel = FALSE`; an event study for HonestDiD needs
/// [`BasePeriod::Universal`].
#[expect(
    clippy::struct_excessive_bools,
    reason = "four independent switches of `did::att_gt`, each named as R names it"
)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttGtOptions {
    /// The estimator of each cell, `est_method`.
    pub estimator: Estimator,
    /// Panel or repeated cross-section, `panel`. Repeated cross-sections
    /// built as a panel with invented unit codes are estimated as such when
    /// this says so; the view records its design instead.
    #[deprecated(
        since = "0.1.0",
        note = "build repeated cross-sections with StaggeredPanel::cross_sections, which \
                records the design; this field is removed in the next release"
    )]
    pub design: Design,
    /// The comparison group, `control_group`.
    pub comparison_group: ComparisonGroup,
    /// The base period, `base_period`.
    pub base_period: BasePeriod,
    /// Periods before first treatment in which effects may already occur,
    /// `anticipation`; not negative. The default is 0.
    pub anticipation: i64,
    /// How an unbalanced panel is treated, `allow_unbalanced_panel`.
    pub unbalanced: UnbalancedPanel,
    /// Whether a cell that cannot be fitted is reported in its
    /// [`AttGtCell::skipped`] rather than ending the call with
    /// [`Error::CellFailed`]. `did` sets such a cell to NA; the default is
    /// `true`.
    pub skip_failed_cells: bool,
    /// Whether a never-treated comparison group with fewer rows per period
    /// than the number of covariates plus 5 is refused, as `did::att_gt`
    /// stops. Read only under [`ComparisonGroup::NeverTreated`]; it never
    /// changes an estimate. The default is `true`.
    pub refuse_small_never_treated: bool,
    /// Whether a doubly robust or weighting cell whose largest fitted
    /// propensity score is 0.999 or more is skipped, as `did::att_gt` sets it
    /// to NA. The score is that of an unweighted logistic regression of the
    /// cell's arm on its design; without covariates it is the treated share.
    /// Every cell's largest score is reported in [`AttGtCell::overlap`]
    /// whatever this is. The default is `true`.
    pub refuse_poor_overlap: bool,
    /// The nuisance models of the two-period estimators.
    pub nuisance: NuisanceOptions,
    /// The confidence level of the cells' intervals, `1 - alp`.
    pub level: Level,
    /// A multiplier bootstrap for the cells' intervals, `DRDID`'s `boot =
    /// TRUE`: percentile intervals of Rademacher draws of each cell's
    /// influence function. `None`, the default, gives normal intervals. Not
    /// with clusters, whose intervals are analytic; [`AttGt::bands`] is the
    /// clustered bootstrap.
    pub bootstrap: Option<Bootstrap>,
    /// Whether the influence function, `n_units x n_cells` values, is kept.
    /// Without it [`AttGt::aggregate`], [`AttGt::bands`], [`AttGt::pretest`]
    /// and [`AttGt::event_study`] refuse with [`Error::NoInfluence`]; the
    /// cells' clustered standard errors are formed before it is dropped. The
    /// default is `true`.
    pub keep_influence: bool,
    #[doc(hidden)]
    pub _extensible: epistat_core::Extensible,
}

impl Default for AttGtOptions {
    #[expect(
        deprecated,
        reason = "the deprecated `design` keeps its default for one release"
    )]
    fn default() -> Self {
        Self {
            _extensible: epistat_core::Extensible::NEW,
            estimator: Estimator::default(),
            design: Design::default(),
            comparison_group: ComparisonGroup::default(),
            base_period: BasePeriod::default(),
            anticipation: 0,
            unbalanced: UnbalancedPanel::default(),
            skip_failed_cells: true,
            refuse_small_never_treated: true,
            refuse_poor_overlap: true,
            nuisance: NuisanceOptions::default(),
            level: Level::default(),
            bootstrap: None,
            keep_influence: true,
        }
    }
}

/// Why the two-period fit of a cell failed.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FitFailure {
    /// The cell had no rows.
    Empty,
    /// The cell had no treated unit.
    NoTreated,
    /// The cell had no comparator.
    NoComparison,
    /// The weights of the cell could not be normalised.
    InvalidWeight,
    /// An outcome of the cell was not finite.
    InvalidOutcome,
    /// A covariate of the cell was not finite.
    InvalidCovariate,
    /// A nuisance model's system was singular.
    Singular,
    /// Another failure of the two-period estimator.
    Other,
}

/// Why a cell was not estimated.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SkipReason {
    /// No treated unit observed at both of the cell's periods.
    NoTreated,
    /// No comparator observed at both of the cell's periods.
    NoComparison,
    /// On a repeated cross-section, no treated row at the base period.
    NoTreatedBase,
    /// On a repeated cross-section, no treated row at the later period.
    NoTreatedLater,
    /// On a repeated cross-section, no comparison row at the base period.
    NoComparisonBase,
    /// On a repeated cross-section, no comparison row at the later period.
    NoComparisonLater,
    /// The largest propensity score of `did`'s overlap check was 0.999 or
    /// more.
    PoorOverlap,
    /// The logistic fit of `did`'s overlap check failed.
    OverlapFitFailed,
    /// The two-period estimator failed.
    FitFailed(FitFailure),
    /// The fit gave an estimate, a standard error or an influence value that
    /// is not finite.
    NotFinite,
    /// The efficient estimator found no baseline pair with both arms.
    NoBaselines,
}

impl SkipReason {
    /// A stable code naming the reason, such as `"no_treated"` or
    /// `"fit.singular"`.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::NoTreated => "no_treated",
            Self::NoComparison => "no_comparison",
            Self::NoTreatedBase => "no_treated_base",
            Self::NoTreatedLater => "no_treated_later",
            Self::NoComparisonBase => "no_comparison_base",
            Self::NoComparisonLater => "no_comparison_later",
            Self::PoorOverlap => "poor_overlap",
            Self::OverlapFitFailed => "overlap_fit_failed",
            Self::FitFailed(failure) => match failure {
                FitFailure::Empty => "fit.empty",
                FitFailure::NoTreated => "fit.no_treated",
                FitFailure::NoComparison => "fit.no_comparison",
                FitFailure::InvalidWeight => "fit.invalid_weight",
                FitFailure::InvalidOutcome => "fit.invalid_outcome",
                FitFailure::InvalidCovariate => "fit.invalid_covariate",
                FitFailure::Singular => "fit.singular",
                FitFailure::Other => "fit.other",
            },
            Self::NotFinite => "not_finite",
            Self::NoBaselines => "no_baselines",
        }
    }
}

impl fmt::Display for SkipReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

/// The overlap diagnostic of a cell: the largest propensity score of `did`'s
/// overlap check. See [`AttGtOptions::refuse_poor_overlap`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Overlap {
    /// The largest fitted score over the cell's treated and comparison units
    /// or rows; without covariates the treated share. `None` when the
    /// logistic fit failed.
    pub max_propensity: Option<f64>,
}

/// One baseline of an efficient cell.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EfficientBaseline {
    /// The pre-treatment baseline period.
    pub base: i64,
    /// The cell's estimate against this baseline.
    pub estimate: f64,
    /// The weight of this baseline in the cell's estimate.
    pub weight: f64,
    /// Treated units or rows of this baseline's comparison.
    pub n_treated: usize,
    /// Comparison units or rows of this baseline's comparison.
    pub n_control: usize,
}

/// How an efficient cell combined its baselines.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct EfficientWeights {
    /// Each baseline with its estimate and weight, in increasing period.
    pub baselines: Vec<EfficientBaseline>,
    /// The covariance `Omega` of the baselines' influence functions,
    /// `sum_i psi_ib psi_ic / n`.
    pub covariance: Covariance,
    /// The solution `v` of `Omega v = 1` before it is normalised to sum to 1.
    pub precision_solution: Vec<f64>,
    /// The ridge added to `Omega` to solve it, `None` when none was needed.
    pub ridge: Option<f64>,
}

/// One cell `(g, t)` that [`estimate`] attempted.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct AttGtCell {
    /// The cohort, the period its units are first treated, `did`'s `group`.
    pub group: i64,
    /// The later period of the cell, `did`'s `t`.
    pub time: i64,
    /// The base period the cell is measured against; for an efficient cell
    /// the last of its baselines.
    pub base: i64,
    /// `time - group`.
    pub event_time: i64,
    /// The estimate with its standard error, interval at
    /// [`AttGtOptions::level`] and normal p-value; `None` for a skipped cell.
    pub estimate: Option<Estimate>,
    /// Treated units (panel) or rows (repeated cross-section) the cell used.
    pub n_treated: usize,
    /// Comparison units or rows the cell used.
    pub n_control: usize,
    /// The sum of the weights of the units or rows the cell used.
    pub total_weight: f64,
    /// Why the cell was not estimated, `None` for an estimated cell.
    pub skipped: Option<SkipReason>,
    /// Covariate columns the cell's own sample could not identify and its fit
    /// left out: constant within an arm, or a combination of the columns
    /// before them.
    pub columns_dropped: usize,
    /// The overlap diagnostic, `None` for a cell skipped before the check and
    /// for the efficient estimator.
    pub overlap: Option<Overlap>,
    /// The baseline weights of an efficient cell.
    pub efficient: Option<EfficientWeights>,
}

/// The sample the cells were estimated on, after the pre-processing of
/// `did::att_gt`, and what that pre-processing changed.
///
/// A cohort first treated after the last period plus the anticipation
/// periods is recoded as never treated. Without never-treated units, the
/// periods from `c - anticipation` on are dropped, `c` the last cohort, which
/// then serves as comparison group: as not-yet-treated units under
/// [`ComparisonGroup::NotYetTreated`], recoded as never treated under
/// [`ComparisonGroup::NeverTreated`]. Units treated in the first period, or
/// within the anticipation periods of it, are dropped. On a panel the panel is
/// then balanced or not as [`AttGtOptions::unbalanced`] says. Base periods
/// are taken from the sorted list of the remaining periods, so periods need
/// not be consecutive. The efficient estimator applies none of this.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AttGtSample {
    /// What each position of the influence function stands for, its
    /// [`Influence::units`]: a unit code on a panel, ascending, and a row
    /// number on a repeated cross-section.
    pub units: Vec<u32>,
    /// The cohort of each position after recoding, `None` for never treated.
    pub groups: Vec<Option<i64>>,
    /// The weight of each position: a row's own on a repeated cross-section;
    /// on a panel the unit's, or the mean over its rows when the unbalanced
    /// panel was estimated as repeated cross-sections, as `did::aggte` reads
    /// it.
    pub weights: Vec<f64>,
    /// The periods of the sample, ascending.
    pub periods: Vec<i64>,
    /// The cohorts whose cells were estimated, ascending.
    pub cohorts: Vec<i64>,
    /// Units (panel) or rows (repeated cross-section) whose cohort was after
    /// the last period and was recoded as never treated.
    pub recoded_never_treated: usize,
    /// The last cohort, when the sample had no never-treated units and it
    /// became the comparison group.
    pub last_cohort_as_comparison: Option<i64>,
    /// Rows dropped because their period was at or after that cohort's start
    /// minus the anticipation periods.
    pub dropped_late_rows: usize,
    /// Units (panel) or rows (repeated cross-section) treated in the first
    /// period or before it, given the anticipation periods, and dropped.
    pub dropped_treated_at_start: usize,
    /// Units dropped to balance the panel under [`UnbalancedPanel::Balance`].
    pub dropped_unbalanced: usize,
    /// Whether an unbalanced panel was estimated as repeated cross-sections,
    /// under [`UnbalancedPanel::RepeatedCrossSection`].
    pub unbalanced_as_repeated_cross_section: bool,
}

/// ATT(g,t): the cells, their influence function and the sample.
///
/// The fields are private so that the cells, the influence function and the
/// sample stay consistent for the methods that read them together.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct AttGt {
    cells: Vec<AttGtCell>,
    influence: Option<Influence>,
    sample: AttGtSample,
    clusters: Option<Vec<u32>>,
    anticipation: i64,
    base_period: BasePeriod,
}

impl AttGt {
    /// Every cell attempted, in the order of `did::att_gt`: by cohort, then
    /// by period.
    #[must_use]
    pub fn cells(&self) -> &[AttGtCell] {
        &self.cells
    }

    /// The influence function of the estimated cells: column `j` is the
    /// `j`-th cell whose [`AttGtCell::estimate`] is `Some`, and its units are
    /// [`AttGtSample::units`]. Its rule is `did`'s,
    /// [`ClusterRule::NoSmallSample`]. `None` when
    /// [`AttGtOptions::keep_influence`] was off.
    #[must_use]
    pub const fn influence(&self) -> Option<&Influence> {
        self.influence.as_ref()
    }

    /// The sample after the pre-processing.
    #[must_use]
    pub const fn sample(&self) -> &AttGtSample {
        &self.sample
    }

    /// The cluster code of each position of the influence function, from the
    /// panel's clusters; `None` without clusters.
    #[must_use]
    pub fn clusters(&self) -> Option<&[u32]> {
        self.clusters.as_deref()
    }

    /// The estimated cells with their estimates, in the order of the
    /// influence function's columns.
    fn estimated(&self) -> impl Iterator<Item = (&AttGtCell, &Estimate)> {
        self.cells
            .iter()
            .filter_map(|cell| cell.estimate.as_ref().map(|estimate| (cell, estimate)))
    }

    /// The influence function, or [`Error::NoInfluence`].
    fn kept_influence(&self) -> Result<&Influence, Error> {
        self.influence.as_ref().ok_or(Error::NoInfluence)
    }

    /// The aggregation of the cells that `options` describe, `did::aggte`;
    /// see [`aggte`]. The panel's clusters cluster its standard errors.
    ///
    /// # Errors
    ///
    /// [`Error::NoInfluence`] without the influence function,
    /// [`Error::NoPostTreatmentCells`] when no post-treatment cell survives
    /// the restrictions of a summary that needs one, [`Error::Input`] for an
    /// invalid restriction or bootstrap, and [`Error::Unsupported`] for a
    /// uniform band without a bootstrap.
    pub fn aggregate(&self, options: &AggteOptions) -> Result<Aggte, Error> {
        aggte::aggregate(self, options)
    }

    /// Simultaneous bands over the estimated cells from a multiplier
    /// bootstrap, as `did::mboot`, clustered by the panel's clusters.
    ///
    /// # Errors
    ///
    /// [`Error::NoInfluence`] without the influence function, and
    /// [`Error::Inference`] when no replicate gives a finite maximum.
    pub fn bands(&self, level: Level, bootstrap: Bootstrap) -> Result<AttGtBands, Error> {
        bands::bands(self, level, bootstrap)
    }

    /// The Wald pre-test of `did::att_gt`: that every pre-treatment cell with
    /// a standard error above `10 sqrt(eps)` is zero, with their covariance
    /// clustered by the panel's clusters.
    ///
    /// # Errors
    ///
    /// [`Error::NoInfluence`] without the influence function,
    /// [`Error::NoPreTreatmentCells`] when no pre-treatment cell has a
    /// standard error, [`Error::Singular`] when their covariance is singular
    /// and [`Error::NotFinite`] when the statistic is not finite.
    pub fn pretest(&self) -> Result<PretrendWald, Error> {
        pretest::pretest(self)
    }

    /// The event study of the dynamic aggregation restricted and weighted
    /// as `options` say, for HonestDiD; see [`Aggte::event_study`]. The fit
    /// must have used [`BasePeriod::Universal`].
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] for a fit under a varying base period, with the
    /// option to set, and otherwise as [`AttGt::aggregate`] and
    /// [`Aggte::event_study`].
    pub fn event_study(&self, options: &EventStudyOptions) -> Result<EventStudy, Error> {
        if self.base_period != BasePeriod::Universal {
            return Err(aggte::VARYING_BASE);
        }
        let dynamic = AggteOptions {
            aggregation: AggteType::Dynamic,
            weights: options.weights,
            balance_e: options.balance_e,
            min_e: options.min_e,
            max_e: options.max_e,
            ..AggteOptions::default()
        };
        self.aggregate(&dynamic)?.event_study()
    }
}

/// ATT(g,t) of the panel, `did::att_gt`.
///
/// The panel's weights are sampling weights, `weightsname`, and must be
/// positive; on a panel they must be the same on every row of a unit. Its
/// covariates are `xformla` without the intercept, which the estimator adds,
/// read at the earlier period of each cell. Its clusters, `clustervars`,
/// cluster the cells' standard errors and every aggregation, band and
/// pre-test. A matched design names each never-treated unit's cohort in the
/// panel's comparison cohort, read under
/// [`ComparisonGroup::MatchedNeverTreated`]. On a repeated cross-section each
/// row is its own observation and its unit code is not read.
///
/// The first cell that cannot be fitted ends the call unless
/// [`AttGtOptions::skip_failed_cells`] is set, which it is by default; a
/// skipped cell is reported with its reason.
///
/// ```
/// use epistat_did::StaggeredPanel;
/// use epistat_did::att_gt::{self, AggteOptions, AggteType, AttGtOptions};
///
/// // Twenty units over four periods: ten first treated in period 3, whose
/// // outcome rises by 2 from then on, and ten never treated.
/// let mut unit = Vec::new();
/// let mut period = Vec::new();
/// let mut cohort = Vec::new();
/// let mut y = Vec::new();
/// for u in 0..20_u32 {
///     for t in 1..=4_i64 {
///         unit.push(u);
///         period.push(t);
///         cohort.push((u < 10).then_some(3));
///         let effect = if u < 10 && t >= 3 { 2.0 } else { 0.0 };
///         y.push(f64::from(u % 4) + 0.5 * t as f64 + effect);
///     }
/// }
/// let panel = StaggeredPanel::builder(&unit, &period, &cohort, &y).build()?;
/// let fit = att_gt::estimate(&panel, &AttGtOptions::default())?;
/// let after = fit.cells().iter().find(|cell| cell.time == 4).unwrap();
/// assert!((after.estimate.unwrap().estimate - 2.0).abs() < 1e-12);
///
/// let dynamic = fit.aggregate(&AggteOptions {
///     aggregation: AggteType::Dynamic,
///     ..AggteOptions::default()
/// })?;
/// let overall = dynamic.overall.unwrap();
/// assert!((overall.estimate.estimate - 2.0).abs() < 1e-12);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// - [`Error::Input`] for an invalid option, a weight that is not positive,
///   a weight that changes within a unit on a panel, or a repeated
///   cross-section with more rows than `u32` can number.
/// - [`Error::Unsupported`] for the traditional doubly robust estimator on a
///   panel; the efficient estimator on a panel with covariates, another
///   comparison group or anticipation; and a cell bootstrap with clusters.
/// - [`Error::NoNeverTreated`] for a comparison group that needs
///   never-treated units the sample does not have, and
///   [`Error::SmallNeverTreatedGroup`] when `did` would refuse its size.
/// - [`Error::CellFailed`] for a cell that cannot be fitted when cells are not
///   skipped, and [`Error::NoEstimableCells`] when no cell is estimated.
/// - The efficient estimator's [`Error::NoPrePeriods`],
///   [`Error::NoPostPeriods`] and [`Error::Singular`].
pub fn estimate(panel: &StaggeredPanel<'_>, options: &AttGtOptions) -> Result<AttGt, Error> {
    let design = design(panel, options);
    validate(panel, options, design)?;
    let rows = Rows::new(panel);
    let config = Config::new(options);
    let output = match (options.estimator, design) {
        (Estimator::Efficient, Design::Panel) => {
            let unsupported = if rows.n_covariates() > 0 {
                Some("the efficient estimator on a panel with covariates")
            } else if options.comparison_group != ComparisonGroup::NeverTreated {
                Some(
                    "the efficient estimator on a panel with comparators other than the never treated",
                )
            } else if options.anticipation != 0 {
                Some("the efficient estimator on a panel with anticipation")
            } else {
                None
            };
            if let Some(what) = unsupported {
                return Err(Error::Unsupported { what });
            }
            efficient::estimate_panel(&rows, config.drdid.confidence_level)?
        }
        (Estimator::Efficient, Design::RepeatedCrossSection) => {
            efficient::estimate_repeated(&rows, config)?
        }
        (Estimator::TraditionalDoublyRobust, Design::Panel) => {
            return Err(Error::Unsupported {
                what: "the traditional doubly robust estimator on a panel",
            });
        }
        (estimator, Design::Panel) => {
            let estimator = match estimator {
                Estimator::InverseProbabilityWeighting => {
                    panel_pairs::Estimator::InverseProbabilityWeighting
                }
                Estimator::OutcomeRegression => panel_pairs::Estimator::OutcomeRegression,
                _ => panel_pairs::Estimator::DoublyRobust,
            };
            panel_pairs::estimate(&rows, config, estimator)?
        }
        (estimator, Design::RepeatedCrossSection) => {
            let estimator = match estimator {
                Estimator::TraditionalDoublyRobust => {
                    pair_estimators::Estimator::TraditionalDoublyRobust
                }
                Estimator::InverseProbabilityWeighting => {
                    pair_estimators::Estimator::InverseProbabilityWeighting
                }
                Estimator::OutcomeRegression => pair_estimators::Estimator::OutcomeRegression,
                _ => pair_estimators::Estimator::DoublyRobust,
            };
            pair_estimators::estimate(&rows, config, estimator)?
        }
    };
    assemble(panel, options, design, output)
}

/// The design of the estimate: repeated cross-sections when the view holds
/// them, or when the deprecated option asks for them on a panel.
fn design(panel: &StaggeredPanel<'_>, options: &AttGtOptions) -> Design {
    #[expect(
        deprecated,
        reason = "the deprecated `design` is honoured for one release"
    )]
    let asked = options.design;
    match panel.design() {
        Design::Panel => asked,
        _ => Design::RepeatedCrossSection,
    }
}

/// The checks of [`estimate`] that the panel's builder does not make.
fn validate(
    panel: &StaggeredPanel<'_>,
    options: &AttGtOptions,
    design: Design,
) -> Result<(), Error> {
    if options.anticipation < 0 {
        return Err(InputError::Domain {
            what: "anticipation",
            value: options.anticipation as f64,
            domain: "[0, inf)",
            row: None,
        }
        .into());
    }
    options.nuisance.validate()?;
    if let Some(bootstrap) = options.bootstrap {
        if bootstrap.replicates == 0 {
            return Err(domain("bootstrap replicates", 0.0, "[1, inf)"));
        }
        if panel.clusters().is_some() {
            return Err(Error::Unsupported {
                what: "a cell bootstrap with clusters; AttGt::bands is the clustered bootstrap",
            });
        }
    }
    if let Some(weights) = panel.weights()
        && let Some(&value) = weights.iter().find(|&&w| w <= 0.0)
    {
        return Err(domain("weights", value, "(0, inf)"));
    }
    match design {
        Design::Panel => {
            if let Some(weights) = panel.weights() {
                for (unit, rows) in panel.panel_key()?.units() {
                    let first = weights[rows.start].to_bits();
                    if weights[rows].iter().any(|w| w.to_bits() != first) {
                        return Err(InputError::NotConstant {
                            what: "weights",
                            unit,
                        }
                        .into());
                    }
                }
            }
        }
        Design::RepeatedCrossSection => {
            if u32::try_from(panel.len()).is_err() {
                return Err(InputError::TooLarge {
                    what: "rows of a repeated cross-section",
                    limit: u64::from(u32::MAX),
                    found: u64::try_from(panel.len()).unwrap_or(u64::MAX),
                }
                .into());
            }
        }
    }
    Ok(())
}

fn domain(what: &'static str, value: f64, domain: &'static str) -> Error {
    InputError::Domain {
        what,
        value,
        domain,
        row: None,
    }
    .into()
}

/// The cluster code of each position of the influence function.
fn position_clusters(
    panel: &StaggeredPanel<'_>,
    design: Design,
    units: &[u32],
) -> Option<Vec<u32>> {
    let clusters = panel.clusters()?;
    let (Design::Panel, Some(key)) = (design, panel.key()) else {
        return Some(units.iter().map(|&row| clusters[row as usize]).collect());
    };
    // `units` is an ascending subset of the panel's units, whose rows are
    // adjacent and in ascending unit order.
    let mut by_unit = Vec::with_capacity(units.len());
    let mut wanted = units.iter().peekable();
    for (unit, rows) in key.units() {
        if wanted.peek() == Some(&&unit) {
            by_unit.push(clusters[rows.start]);
            wanted.next();
        }
    }
    Some(by_unit)
}

/// The result from a route's output: each cell's estimate, clustered when the
/// panel has clusters, and the influence function.
fn assemble(
    panel: &StaggeredPanel<'_>,
    options: &AttGtOptions,
    design: Design,
    output: Output,
) -> Result<AttGt, Error> {
    let Output {
        attempts,
        influence,
        sample,
    } = output;
    let n = sample.units.len();
    let clusters = position_clusters(panel, design, &sample.units);
    let numbered = clusters.as_deref().map(Clusters::from_codes);
    let confidence = options.level.confidence_value();
    let mut column = 0;
    let cells = attempts
        .into_iter()
        .map(|attempt| {
            let (estimate, counts, columns_dropped) = match attempt.result {
                Ok(fit) => {
                    let (se, ci) = match &numbered {
                        Some(numbered) => {
                            let psi = &influence[column * n..(column + 1) * n];
                            let se = standard_error(psi, numbered);
                            (se, normal_interval(fit.att, se, confidence))
                        }
                        None => (fit.se, (fit.ci_low, fit.ci_high)),
                    };
                    column += 1;
                    (
                        Some(cell_estimate(fit.att, se, ci, options.level)),
                        (fit.treated_n, fit.control_n, fit.total_weight),
                        fit.columns_dropped,
                    )
                }
                Err(_) => (None, (0, 0, 0.0), 0),
            };
            AttGtCell {
                group: attempt.group,
                time: attempt.time,
                base: attempt.base,
                event_time: attempt.time - attempt.group,
                estimate,
                n_treated: counts.0,
                n_control: counts.1,
                total_weight: counts.2,
                skipped: attempt.result.err(),
                columns_dropped,
                overlap: attempt.overlap,
                efficient: attempt.efficient,
            }
        })
        .collect();
    let influence = if options.keep_influence {
        Some(Influence::new(
            influence,
            sample.units.clone(),
            column,
            ClusterRule::NoSmallSample,
        )?)
    } else {
        None
    };
    Ok(AttGt {
        cells,
        influence,
        sample,
        clusters,
        anticipation: options.anticipation,
        base_period: options.base_period,
    })
}

pub(super) use crate::inference::estimate_from as cell_estimate;

/// The settings every route reads, from [`AttGtOptions`].
#[derive(Debug, Clone, Copy)]
pub(super) struct Config {
    pub comparison_group: ComparisonGroup,
    pub base_period: BasePeriod,
    pub anticipation: i64,
    pub unbalanced: UnbalancedPanel,
    pub skip_failed_cells: bool,
    pub refuse_small_never_treated: bool,
    pub refuse_poor_overlap: bool,
    pub drdid: DrDidConfig,
}

impl Config {
    fn new(options: &AttGtOptions) -> Self {
        Self {
            comparison_group: options.comparison_group,
            base_period: options.base_period,
            anticipation: options.anticipation,
            unbalanced: options.unbalanced,
            skip_failed_cells: options.skip_failed_cells,
            refuse_small_never_treated: options.refuse_small_never_treated,
            refuse_poor_overlap: options.refuse_poor_overlap,
            drdid: options.nuisance.config(options.level, options.bootstrap),
        }
    }
}

#[cfg(test)]
impl Default for Config {
    fn default() -> Self {
        Self::new(&AttGtOptions::default())
    }
}

/// The numbers of a fitted cell, before clustering.
#[derive(Debug, Clone, Copy)]
pub(super) struct CellFit {
    pub att: f64,
    pub se: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    pub treated_n: usize,
    pub control_n: usize,
    pub total_weight: f64,
    pub columns_dropped: usize,
}

/// One cell a route attempted.
pub(super) struct Attempt {
    pub group: i64,
    pub time: i64,
    pub base: i64,
    pub result: Result<CellFit, SkipReason>,
    /// The overlap check, `None` for a cell skipped before it or an
    /// estimator without it.
    pub overlap: Option<Overlap>,
    pub efficient: Option<EfficientWeights>,
}

/// What a route returns: the cells in order, the influence function of the
/// fitted ones column after column, and the sample.
pub(super) struct Output {
    pub attempts: Vec<Attempt>,
    pub influence: Vec<f64>,
    pub sample: AttGtSample,
}

/// The attempt of cell `(group, time, base)`, with its influence vector
/// appended to `influence` when it was fitted.
///
/// # Errors
/// [`Error::CellFailed`] for a cell that could not be fitted when cells are
/// not skipped.
pub(super) fn record(
    (group, time, base): (i64, i64, i64),
    fitted: Result<(CellFit, Vec<f64>), SkipReason>,
    overlap: Option<Overlap>,
    influence: &mut Vec<f64>,
    config: Config,
) -> Result<Attempt, Error> {
    let result = match fitted {
        Ok((fit, psi)) => {
            influence.extend_from_slice(&psi);
            Ok(fit)
        }
        Err(reason) if config.skip_failed_cells => Err(reason),
        Err(reason) => {
            return Err(Error::CellFailed {
                group,
                time,
                base,
                reason,
            });
        }
    };
    Ok(Attempt {
        group,
        time,
        base,
        result,
        overlap,
        efficient: None,
    })
}

/// The output of a route, or [`Error::NoEstimableCells`] when no cell was
/// fitted.
pub(super) fn finish(
    attempts: Vec<Attempt>,
    influence: Vec<f64>,
    sample: AttGtSample,
) -> Result<Output, Error> {
    if attempts.iter().all(|attempt| attempt.result.is_err()) {
        return Err(Error::NoEstimableCells);
    }
    Ok(Output {
        attempts,
        influence,
        sample,
    })
}

/// Whether a unit of cohort `cohort` is a comparator for the cell of cohort
/// `group` whose later period is `pair_time`.
///
/// `comparison_cohort` is the cohort a never-treated unit was matched to, and
/// is read only under [`ComparisonGroup::MatchedNeverTreated`].
fn is_control_for_pair(
    cohort: Option<i64>,
    comparison_cohort: Option<i64>,
    group: i64,
    pair_time: i64,
    mode: ComparisonGroup,
    anticipation: i64,
) -> bool {
    match mode {
        ComparisonGroup::NeverTreated => cohort.is_none(),
        ComparisonGroup::NotYetTreated => cohort.is_none_or(|g| g > pair_time + anticipation),
        ComparisonGroup::MatchedNeverTreated => {
            cohort.is_none() && comparison_cohort == Some(group)
        }
    }
}
