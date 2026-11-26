//! The event-study estimator of de Chaisemartin and D'Haultfoeuille for
//! treatments that switch on and off and may take several values, as
//! `DIDmultiplegtDYN::did_multiplegt_dyn` computes it.
//!
//! # Algorithm
//!
//! Each group `g` has the baseline treatment `D_g1`, its first observed one,
//! and `F_g`, the first period in which its treatment differs from it
//! (`T + 1` if it never does). A group switches in when its mean treatment
//! from `F_g` to `T_g` is above `D_g1` and out when it is below. `T_g` is the
//! last period in which some group with the same baseline (and stratum) has
//! not yet switched, less one. The effect `l` periods after the first switch
//! compares the switchers' outcome change from `F_g - 1` to `F_g - 1 + l`
//! with that of the groups with the same baseline whose treatment has not
//! changed by `F_g - 1 + l`:
//!
//! ```text
//! DID_{+,l} = (1/N^1_l) sum_{g: S_g = 1, F_g - 1 + l <= T_g} N_{g,F_g-1+l}
//!             [ (Y_{g,F_g-1+l} - Y_{g,F_g-1})
//!               - sum_{g': D_g'1 = D_g1, F_g' > F_g-1+l} N_{g',F_g-1+l}
//!                 (Y_{g',F_g-1+l} - Y_{g',F_g-1}) / N^c_{F_g-1+l,D_g1} ],
//! ```
//!
//! `DID_{-,l}` the same for switchers out with the sign reversed, and
//! `DID_l = (N^1_l DID_{+,l} + N^0_l DID_{-,l}) / (N^1_l + N^0_l)`, where
//! `N_{g,t}` is the weight of a cell and `N^1_l`, `N^0_l` the weighted
//! numbers of switchers in and out. As in the package, every sum is written
//! as `DID_l = (1/G) sum_g U_{g,l}` over the `G` groups. Placebo `l` is the
//! same with the difference `Y_{g,F_g-1-l} - Y_{g,F_g-1}` for the same
//! switchers, among those observed then. The normalised effect divides
//! `DID_l` by the switchers' mean of `sum_{k=0}^{l-1} |D_{g,F_g+k} - D_g1|`.
//! The average total effect is
//!
//! ```text
//! delta = sum_l w_l DID_l / sum_l w_l delta^D_l,
//! ```
//!
//! with `w_l` the share of the switchers of effect `l` and `delta^D_l` the
//! switchers' mean of `|D_{g,F_g-1+l} - D_g1|`, formed by direction and
//! combined with the weight of each direction's denominator.
//!
//! With controls, every long difference of the outcome is first reduced by
//! `theta_d' (X_{g,t} - X_{g,t-l})`, where `theta_d` is the coefficient of
//! the weighted regression of first differences of the outcome on those of
//! the controls with fixed effects of period, baseline and stratum, among
//! the not-yet switchers with baseline `d`. With `trends_nonparam`, the
//! comparison groups are further restricted to the switcher's stratum.
//!
//! # Options
//!
//! - `trends_lin` allows a linear trend specific to each group. Groups that
//!   switch in the second period are dropped, the outcome and the controls
//!   become first differences and the first period goes. Effect `l` is then
//!   `sum_{k=1}^{l} DID_k` on the first differences, all `l` of them over
//!   the switchers for which every one can be estimated, and its influence
//!   function the same sum: for a switcher this is
//!   `Y_{F_g-1+l} - Y_{F_g-1} - l (Y_{F_g-1} - Y_{F_g-2})` less the same for
//!   its controls. Placebos alike. There is no average total effect.
//! - `same_switchers` keeps the switchers that have, at each of
//!   `F_g - 1 + k` for `k = 1..=L`, a long difference and a control with one,
//!   and `F_g - 1 + L <= T_g`. With `same_switchers_pl` the placebos further
//!   need, at each `F_g - 1 - k` for `k = 1..=L_pl`, the difference
//!   `Y_{F_g-1-k} - Y_{F_g-1}` and a control that has not switched by
//!   `F_g - 1 - k`.
//! - `less_conservative_se` demeans a switcher's long difference in the
//!   variance within its treatment path `(D_g1, F_g, D_{g,F_g}, ...,
//!   D_{g,F_g-1+l})` when another group shares the path, else within
//!   `(D_g1, F_g, D_{g,F_g})` when another group shares that, else within
//!   `(D_g1, F_g)`, and the small-sample factor counts the switchers' cells.
//!   The placebos keep the default cohorts, as in the package.
//! - `continuous` with order `p` compares groups within the stratum whatever
//!   their baseline, and adds the controls `1{t >= j} D_g1^k` for every
//!   period `j` after the first and `k = 1..=p`: without a change of
//!   treatment the outcome evolves as a polynomial in the baseline with
//!   coefficients specific to each period. The switchers' cohorts use the
//!   indicator of having switched times the direction of the switch, and
//!   the normalisation and the average total effect the treatment itself.
//! - [`DcdhResult::normalized_weights`] and [`DcdhResult::avg_time_periods`]
//!   describe the switchers' treatment changes. [`dcdh_design`],
//!   [`dcdh_first_switch_dates`], [`estimate_dcdh_heterogeneity`],
//!   [`estimate_dcdh_by`], [`estimate_dcdh_by_path`] and
//!   [`estimate_dcdh_bootstrap`] are the package's `design`,
//!   `date_first_switch`, `predict_het`, `by`, `by_path` and `bootstrap`.
//!
//! # Variance
//!
//! The variance is that of the package's influence-function estimator. Each
//! group's term `U^var_{g,l}` is `U_{g,l}` with every cell's long difference
//! demeaned within its cohort (switchers by baseline, first-switch date,
//! treatment at `F_g` and stratum; controls by baseline, period and
//! stratum, or the union of the two when a cohort has one member) and
//! scaled by `sqrt(n / (n - 1))` for a cohort of `n` groups or clusters.
//! With controls, the term of the estimated `theta_d` is subtracted. Then
//!
//! ```text
//! se(DID_l)^2 = (1/G^2) sum_c (sum_{g in c} U^var_{g,l})^2
//! ```
//!
//! over the clusters `c` (the groups without clustering), without centring.
//! The covariance of two estimates is the same sum of products, which gives
//! the joint tests that all effects, or all placebos, are zero and that the
//! effects are equal. The intervals are `estimate +- z se`.
//!
//! # The package's versions, and where this departs from them
//!
//! The rules are those of `DIDmultiplegtDYN` 2.4.0. Its tests run against
//! 2.2.0, the last release that runs without the R package polars, which is
//! not on CRAN. The two releases compute the same numbers on these options
//! except where 2.4.0 changed a rule, and there this module follows 2.4.0:
//!
//! - The variance term of `theta_d` scales the inverse of the controls' Gram
//!   matrix by the sum of `N_gt` over the not-yet switchers of baseline `d`
//!   that enter its regression. 2.2.0 sums over all groups and baselines,
//!   which changes the standard errors with controls and more than one
//!   baseline treatment.
//! - A cell whose cohort has one cluster is demeaned by 0 with factor 1;
//!   2.2.0 drops its variance term.
//! - The regression of the fitted values with period effects is weighted;
//!   2.2.0 drops the weights by accident.
//! - The equality test may cover a range of effects.
//!
//! It departs from both releases in these points:
//!
//! - The covariance matrix is `sum_c psi_c psi_c' / G^2`. The package's
//!   `coef$vcov` takes the first group's term for every group off the
//!   diagonal, so its covariances are wrong; its joint tests form them
//!   correctly, and agree with these.
//! - The weighted cell counts of a placebo combine the two directions by
//!   their maximum, as for the effects. 2.4.0 takes the switchers in only.
//! - The joint tests solve the covariance matrix and give `None` when it is
//!   singular. 2.2.0 uses `matlib::Ginv`, 2.4.0 returns NA when an
//!   eigenvalue is below `1e-10`.
//! - The input is one row per group and period; the package averages
//!   duplicate rows. The stratum and the cluster must be constant within a
//!   group. The package leaves the stratum missing in the cells its
//!   balancing adds, which changes `T_g` with `trends_nonparam` on an
//!   unbalanced panel; here those cells take their group's stratum.
//! - Controls that are collinear among the not-yet switchers of a baseline,
//!   or a baseline whose groups share one first-switch date, are an error;
//!   the package uses a generalised inverse, or fails.
//! - A period inside the panel that loses all its rows to the sample rules
//!   is an error.
//! - An estimate that is not finite, as a normalised effect with a zero
//!   denominator, is `None`.
//!
//! - With `trends_lin`, the first differences are between consecutive
//!   periods, so a period missing from a group's rows leaves the next one
//!   missing. The package differences consecutive rows, across the gap.
//! - With `trends_lin`, the unweighted number of switchers of an estimate
//!   is that of the pass that gives the estimate. The package reports the
//!   number from the last pass that touches it, which keeps only the
//!   switchers that allow more effects or placebos.
//! - With `continuous`, a missing treatment before the first switch is
//!   filled with the group's own baseline; the package fills it with the 0
//!   it puts in the baseline's place. The period controls are those of the
//!   periods after the first of the panel; the package numbers the periods
//!   from 2 by position but compares the number to the period, so when the
//!   panel does not start in the first period, as with `trends_lin`, it
//!   misses the last period and adds a constant. A period control that is
//!   zero among the not-yet switchers (one not-yet switcher, or all with
//!   one baseline, in that period) is left out, as the package's
//!   generalised inverse leaves it out; other collinear controls are an
//!   error.
//! - `predict_het` needs the outcome at `F_g - 2` to decide which dimensions
//!   of the cells vary only with `trends_lin`; the package always does
//!   (`!is.null(trends_lin)` is always true). A switcher alone in its cell
//!   adds nothing to the HC2 covariance, where the package gives NaN, and
//!   the intervals are at the configured level, where the package uses 95%.
//! - The normalised weights of an effect are over its switchers. With
//!   `switchers` in or out the package also counts the switchers of the
//!   other direction, and with `same_switchers` those that lack a horizon.
//! - `avg_time_periods` with `continuous` takes the treatment itself; the
//!   package takes its recoded indicator.
//! - `by_path` gives each path the switchers on it and the not-yet switchers
//!   of the same baseline from the input rows. The package matches the
//!   baseline on the first character of the path's text, and draws the rows
//!   from those left after its sample rules, which with `trends_lin` drops
//!   a second period. Paths with as many switchers are in increasing
//!   lexicographic order; 2.2.0 orders them by first appearance and 2.4.0
//!   by the text of the path.
//! - The bootstrap gives a group drawn `m` times `m` times its weights, as
//!   2.4.0 does and as 2.2.0's duplicated rows amount to; 2.2.0 fails with a
//!   cluster.
//!
//! `more_granular_demeaning` of 2.4.0 sets `less_conservative_se` and is
//! not a separate option. `reset` and `predict_het_hc2bm` are not
//! implemented.
//!
//! # Cost
//!
//! The work is linear in the cells of the balanced grid of groups and
//! periods for each effect and placebo, times the number of controls, with
//! one sort for the cluster counts. `same_switchers` adds a pass per effect
//! and placebo, `trends_lin` runs `L` passes of up to `L` effects, and
//! `avg_time_periods` counts controls over pairs of periods, `T^2` per
//! group. Nothing is parallel except the bootstrap's replicates, which do
//! not share draws, so the results do not depend on the number of
//! threads.
//!
//! # References
//!
//! 1. de Chaisemartin C, D'Haultfoeuille X. Difference-in-differences
//!    estimators of intertemporal treatment effects. Rev Econ Stat 2024.
//!    doi:10.1162/rest_a_01414
//! 2. de Chaisemartin C, D'Haultfoeuille X, Ciccia D, Vazquez-Bare G and
//!    others. R package `DIDmultiplegtDYN` 2.4.0, 2026, functions
//!    `did_multiplegt_main` and `did_multiplegt_dyn_core`.

#![expect(
    clippy::float_cmp,
    reason = "the indicators of the package are exactly 0, 1 or NaN, and R compares them and the treatment values exactly"
)]

mod bootstrap;
mod controls;
mod core;
mod design;
mod het;
mod panel;
mod subsets;
mod summaries;

pub use self::bootstrap::{DcdhBootstrap, DcdhBootstrapEstimate, estimate_dcdh_bootstrap};
pub use self::design::{
    DcdhDesign, DcdhPath, DcdhSwitchDate, DcdhSwitchDates, dcdh_design, dcdh_first_switch_dates,
};
pub use self::het::{DcdhHeterogeneity, estimate_dcdh_heterogeneity};
pub use self::subsets::{DcdhPathResult, DcdhSubgroup, estimate_dcdh_by, estimate_dcdh_by_path};

use epistat_linalg::Mat;

use crate::inference::{validate_confidence_level, z_score_for_confidence};

use self::controls::Controls;
use self::core::{DirectionTerms, Options, Terms, direction};
use self::panel::{Panel, prepare};
use self::summaries::{lag_weights, time_periods};

/// One cell of the panel: a group in a period.
#[derive(Debug, Clone, PartialEq)]
pub struct DcdhObservation {
    /// Group (unit) identifier.
    pub group: i64,
    /// Period. Periods are ranked, so only their order matters.
    pub time: i64,
    /// Outcome, `None` where missing.
    pub outcome: Option<f64>,
    /// Treatment, `None` where missing.
    pub treatment: Option<f64>,
    /// Sampling weight, finite and not negative; 1 for none.
    pub weight: f64,
    /// Control variables, the same number in every row.
    pub controls: Vec<f64>,
    /// Stratum within which comparisons are made, as `trends_nonparam`.
    /// Constant within a group; 0 for none.
    pub stratum: i64,
    /// Cluster, constant within a group. `None` in every row to cluster by
    /// group.
    pub cluster: Option<i64>,
}

/// Which switchers enter the effects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DcdhSwitchers {
    /// Switchers in and out (`switchers = ""`).
    #[default]
    Both,
    /// Switchers in only.
    In,
    /// Switchers out only.
    Out,
}

/// The test that effects are equal (`effects_equal`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DcdhEffectsEqual {
    /// No test.
    #[default]
    No,
    /// All effects.
    All,
    /// Effects `first..=last`, `last` cut to the number estimated.
    Range {
        /// First effect of the range, from 1.
        first: usize,
        /// Last effect of the range.
        last: usize,
    },
}

/// Options of [`estimate_dcdh`], named after `did_multiplegt_dyn`'s.
#[expect(
    clippy::struct_excessive_bools,
    reason = "the options are the R function's logical arguments"
)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DcdhConfig {
    /// Number of effects requested, at least 1.
    pub effects: usize,
    /// Number of placebos requested, at most `effects`.
    pub placebo: usize,
    /// Normalised effects.
    pub normalized: bool,
    /// Which switchers to use.
    pub switchers: DcdhSwitchers,
    /// Only never-switchers as controls.
    pub only_never_switchers: bool,
    /// Keep groups whose treatment goes both above and below its baseline.
    pub dont_drop_larger_lower: bool,
    /// Drop a group's outcomes from its first missing treatment, when that
    /// is before its first switch.
    pub drop_if_d_miss_before_first_switch: bool,
    /// The equality test.
    pub effects_equal: DcdhEffectsEqual,
    /// Confidence level of the intervals, in (0, 1).
    pub confidence_level: f64,
    /// Group-specific linear trends (`trends_lin`): effect `l` is the sum of
    /// effects `1..=l` on the outcome's first difference, each over the
    /// switchers for which all of them can be estimated. There is no
    /// average total effect.
    pub trends_lin: bool,
    /// A continuous baseline treatment (`continuous`): the polynomial order
    /// in the baseline of the outcome's evolution without a change in
    /// treatment, at least 1.
    pub continuous: Option<usize>,
    /// Only switchers for which every effect can be estimated
    /// (`same_switchers`).
    pub same_switchers: bool,
    /// With `same_switchers`, only switchers for which every placebo can
    /// also be estimated enter the placebos (`same_switchers_pl`).
    pub same_switchers_pl: bool,
    /// Switchers demeaned within their treatment paths in the variance
    /// (`less_conservative_se`; `more_granular_demeaning` of 2.4.0 sets the
    /// same option).
    pub less_conservative_se: bool,
    /// Compute [`DcdhResult::avg_time_periods`].
    pub avg_time_periods: bool,
}

impl Default for DcdhConfig {
    fn default() -> Self {
        Self {
            effects: 1,
            placebo: 0,
            normalized: false,
            switchers: DcdhSwitchers::Both,
            only_never_switchers: false,
            dont_drop_larger_lower: false,
            drop_if_d_miss_before_first_switch: false,
            effects_equal: DcdhEffectsEqual::No,
            confidence_level: 0.95,
            trends_lin: false,
            continuous: None,
            same_switchers: false,
            same_switchers_pl: false,
            less_conservative_se: false,
            avg_time_periods: false,
        }
    }
}

/// One estimate: an effect, a placebo or the average total effect, with the
/// columns of the package's results.
#[derive(Debug, Clone, PartialEq)]
pub struct DcdhEstimate {
    /// `l` for effect `l`, `-l` for placebo `l`, 0 for the average.
    pub relative_period: i64,
    /// The estimate; `None` when it has no switcher or no control.
    pub estimate: Option<f64>,
    /// Its standard error.
    pub se: Option<f64>,
    /// Its normal interval.
    pub ci: Option<(f64, f64)>,
    /// Cells used ("N").
    pub n_cells: usize,
    /// Switchers used ("Switchers").
    pub n_switchers: f64,
    /// Weighted cells ("N.w").
    pub n_cells_weighted: f64,
    /// Weighted switchers ("Switchers.w").
    pub n_switchers_weighted: f64,
    /// The influence function: one value per group of
    /// [`DcdhResult::groups`], scaled so that the variance is
    /// `sum_c (sum_{g in c} psi_g)^2 / G^2` over clusters `c`. Normalised
    /// like the estimate. `None` with the standard error.
    pub influence: Option<Vec<f64>>,
}

/// The result of [`estimate_dcdh`].
#[derive(Debug, Clone, PartialEq)]
pub struct DcdhResult {
    /// Effects `1..=L`, where `L` is the number requested or fewer when the
    /// data allow fewer.
    pub effects: Vec<DcdhEstimate>,
    /// Placebos `1..=L_pl`.
    pub placebos: Vec<DcdhEstimate>,
    /// The average total effect.
    pub average_total_effect: DcdhEstimate,
    /// Upper-tail probability of the Wald test that all effects are zero,
    /// with at least two effects that are all estimated.
    pub p_joint_effects: Option<f64>,
    /// The same for the placebos.
    pub p_joint_placebos: Option<f64>,
    /// The test that the effects are equal, when requested.
    pub p_equal_effects: Option<f64>,
    /// Covariance of the effects then the placebos, row-major, NaN where an
    /// estimate has no standard error.
    pub covariance: Vec<f64>,
    /// With `normalized`, the denominator of each effect.
    pub normalization: Vec<Option<f64>>,
    /// With `normalized`, the denominator of each placebo.
    pub placebo_normalization: Vec<Option<f64>>,
    /// Average number of periods over which an effect accumulates
    /// (`delta_D_avg_total`); `None` with `trends_lin`.
    pub delta_d_avg_total: Option<f64>,
    /// With `normalized`, the weights that normalised effect `l` puts on the
    /// effects of the current treatment and its lags (`normalized_weights`):
    /// entry `k` of row `l - 1` is the weight of lag `k`, for `k < l`.
    /// `None` for an effect without a normalisation.
    pub normalized_weights: Vec<Option<Vec<f64>>>,
    /// With [`DcdhConfig::avg_time_periods`], the average number of periods
    /// over which a dose of treatment affects the outcome.
    pub avg_time_periods: Option<DcdhAvgTimePeriods>,
    /// `G`, the number of groups that divides every estimate.
    pub n_groups: usize,
    /// The groups indexing the influence functions, increasing.
    pub groups: Vec<i64>,
    /// The cluster of each of those groups, when clustered.
    pub clusters: Option<Vec<i64>>,
}

/// The `avg_time_periods` output: over the switchers' changes in treatment
/// at `F_g + k`, the mean of `M_g - k`, the number of periods from `F_g + k`
/// to the last effect the switcher contributes to, weighted by the size of
/// the change.
#[derive(Debug, Clone, PartialEq)]
pub struct DcdhAvgTimePeriods {
    /// The weighted mean, `None` when no switcher changes treatment.
    pub average: Option<f64>,
    /// Its numerator.
    pub numerator: f64,
    /// Its denominator.
    pub denominator: f64,
    /// The weighted number of switchers contributing at each `k`, from 0.
    pub switchers: Vec<f64>,
}

/// Errors from [`estimate_dcdh`].
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum DcdhError {
    /// No observations.
    #[error("no observations")]
    Empty,
    /// An invalid option.
    #[error("invalid option: {0}")]
    InvalidConfig(&'static str),
    /// An invalid row.
    #[error("row {row}: {reason}")]
    InvalidRow {
        /// Index of the row.
        row: usize,
        /// What is wrong.
        reason: &'static str,
    },
    /// Two rows for one group and period.
    #[error("group {group} has two rows in period {time}")]
    DuplicateCell {
        /// The group.
        group: i64,
        /// The period.
        time: i64,
    },
    /// The stratum changes within a group.
    #[error("the stratum of group {group} is not constant")]
    VaryingStratum {
        /// The group.
        group: i64,
    },
    /// The cluster changes within a group.
    #[error("the cluster of group {group} is not constant")]
    ClusterNotNested {
        /// The group.
        group: i64,
    },
    /// After the sample rules no period has both switchers and controls
    /// (Design Restriction 1 of the paper fails).
    #[error("no treatment effect can be estimated")]
    NoEffect,
    /// After the sample rules a period inside the panel has no rows.
    #[error("a period inside the panel has no rows after the sample rules")]
    EmptyPeriod,
    /// A covariate of `predict_het` changes within a group.
    #[error("a covariate of group {group} is not constant")]
    VaryingCovariate {
        /// The group.
        group: i64,
    },
    /// The subgroup variable of `by` changes within a group.
    #[error("the subgroup of group {group} is not constant")]
    VaryingSubgroup {
        /// The group.
        group: i64,
    },
    /// The controls cannot be used for a baseline treatment.
    #[error("controls at baseline treatment {baseline}: {reason}")]
    Controls {
        /// The baseline treatment.
        baseline: f64,
        /// What is wrong.
        reason: &'static str,
    },
}

/// One estimate assembled from the two directions.
struct Combined {
    estimate: Option<f64>,
    /// Per grid group, `None` without a standard error.
    psi: Option<Vec<f64>>,
    n_switchers: f64,
    n_switchers_dw: f64,
    count: Vec<f64>,
    delta: Option<f64>,
}

fn live(t: Option<&Terms>) -> Option<&Terms> {
    t.filter(|t| t.n != 0.0)
}

type Side<'a> = (f64, f64, &'a [f64], &'a [f64]);

fn side(d: Option<&DirectionTerms>) -> Option<Side<'_>> {
    d.filter(|d| d.sum_n != 0.0).map(|d| {
        (
            d.ate_den,
            d.sum_n,
            d.ate_u.as_slice(),
            d.ate_u_var.as_slice(),
        )
    })
}

fn n_of(t: Option<&Terms>) -> f64 {
    t.map_or(0.0, |t| t.n)
}

fn estimable(switchers: DcdhSwitchers, n1: f64, n0: f64) -> bool {
    match switchers {
        DcdhSwitchers::Both => n1 != 0.0 || n0 != 0.0,
        DcdhSwitchers::In => n1 != 0.0,
        DcdhSwitchers::Out => n0 != 0.0,
    }
}

fn combine(
    panel: &Panel,
    plus: Option<&Terms>,
    minus: Option<&Terms>,
    config: &DcdhConfig,
) -> Combined {
    let gn = panel.n_grid_groups();
    let cells = gn * panel.nt;
    let (n1, n0) = (n_of(plus), n_of(minus));
    let (p, m) = (live(plus), live(minus));
    let (a, b) = (n1 / (n1 + n0), n0 / (n1 + n0));
    let mut total = 0.0;
    let mut psi = vec![0.0; gn];
    for g in 0..gn {
        let up = p.map_or(0.0, |t| t.u[g]);
        let um = m.map_or(0.0, |t| -t.u[g]);
        total += a * up + b * um;
        let vp = p.map_or(0.0, |t| t.u_var[g]);
        let vm = m.map_or(0.0, |t| -t.u_var[g]);
        psi[g] = vp * a + vm * b;
    }
    let mut count = vec![0.0; cells];
    for (c, slot) in count.iter_mut().enumerate() {
        let cp = p.map_or(0.0, |t| t.count[c]);
        let cm = m.map_or(0.0, |t| t.count[c]);
        *slot = if cp > cm { cp } else { cm };
    }
    let delta = config
        .normalized
        .then(|| a * p.map_or(0.0, |t| t.delta_norm) + b * m.map_or(0.0, |t| t.delta_norm));
    let mut estimate = total / panel.n_groups;
    let ok = estimable(config.switchers, n1, n0);
    if let Some(d) = delta {
        estimate /= d;
        for v in &mut psi {
            *v /= d;
        }
    }
    let n_dw = plus.map_or(0.0, |t| t.n_dw) + minus.map_or(0.0, |t| t.n_dw);
    Combined {
        estimate: (ok && estimate.is_finite()).then_some(estimate),
        psi: ok.then_some(psi),
        n_switchers: n1 + n0,
        n_switchers_dw: n_dw,
        count,
        delta,
    }
}

/// Sums of `psi` by cluster (by group without clusters), over `G`.
fn cluster_sums(panel: &Panel, psi: &[f64]) -> Vec<f64> {
    match &panel.cluster {
        None => psi.iter().map(|v| v / panel.n_groups).collect(),
        Some(cl) => {
            let mut s = vec![0.0; panel.n_cluster];
            for (g, v) in psi.iter().enumerate() {
                s[cl[g]] += v;
            }
            s.iter().map(|v| v / panel.n_groups).collect()
        }
    }
}

fn sum_sq(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn wald(estimates: &[f64], cov: &[Vec<f64>]) -> Option<f64> {
    let k = estimates.len();
    let m = Mat::from_fn(k, k, |i, j| cov[i][j]);
    epistat_inference::wald::wald(estimates, m.as_ref())
        .ok()
        .map(|w| w.p_value)
}

/// The joint test that `estimates` are zero, from their cluster sums.
fn joint_test(estimates: &[Option<f64>], sums: &[Option<Vec<f64>>]) -> Option<f64> {
    if estimates.len() < 2 {
        return None;
    }
    let b: Option<Vec<f64>> = estimates.iter().copied().collect();
    let s: Option<Vec<&Vec<f64>>> = sums.iter().map(Option::as_ref).collect();
    let (b, s) = (b?, s?);
    let cov: Vec<Vec<f64>> = s
        .iter()
        .map(|x| s.iter().map(|y| dot(x, y)).collect())
        .collect();
    wald(&b, &cov)
}

/// The test that `estimates` are equal: `D b = 0` with `D` the first `k - 1`
/// rows of the identity less `1/k`.
#[expect(
    clippy::needless_range_loop,
    reason = "the contrast matrix is indexed as in its formula"
)]
fn equality_test(estimates: &[Option<f64>], sums: &[Option<Vec<f64>>]) -> Option<f64> {
    let k = estimates.len();
    if k < 2 {
        return None;
    }
    let b: Vec<f64> = estimates.iter().copied().collect::<Option<_>>()?;
    let s: Vec<&Vec<f64>> = sums.iter().map(Option::as_ref).collect::<Option<_>>()?;
    let kf = k as f64;
    let d = |i: usize, j: usize| f64::from(u8::from(i == j)) - 1.0 / kf;
    let v: Vec<Vec<f64>> = s
        .iter()
        .map(|x| s.iter().map(|y| dot(x, y)).collect())
        .collect();
    let db: Vec<f64> = (0..k - 1)
        .map(|i| (0..k).map(|j| d(i, j) * b[j]).sum())
        .collect();
    let mut dvd = vec![vec![0.0; k - 1]; k - 1];
    for (i, row) in dvd.iter_mut().enumerate() {
        for (j, slot) in row.iter_mut().enumerate() {
            let mut acc = 0.0;
            for p in 0..k {
                for q in 0..k {
                    acc += d(i, p) * v[p][q] * d(j, q);
                }
            }
            *slot = acc;
        }
    }
    wald(&db, &dvd)
}

/// The `did_multiplegt_dyn` estimator of de Chaisemartin and
/// D'Haultfoeuille: event-study effects of the first treatment change for
/// treatments that switch on and off or take several values, their
/// placebos, the average total effect and the package's analytic inference.
/// See the module documentation for the estimator and the variance.
///
/// The sample rules are the package's: groups whose treatment goes both
/// above and below the baseline lose their rows from the period when both
/// have happened, baselines (by stratum) whose groups all switch at the
/// same date are dropped, periods without a control are dropped per
/// baseline, and missing treatments are filled by the package's
/// conventions.
///
/// # Errors
/// [`DcdhError`] for invalid input or options, or when no effect can be
/// estimated.
///
/// # Panics
/// Only when a number of periods or effects does not fit in `i64`.
pub fn estimate_dcdh(
    observations: &[DcdhObservation],
    config: &DcdhConfig,
) -> Result<DcdhResult, DcdhError> {
    validate(config)?;
    let panel = prepare(observations, config)?;
    let controls = if panel.controls.is_empty() {
        None
    } else {
        Some(Controls::new(&panel)?)
    };
    let counts = counts(&panel, config)?;
    let (inn, out) = directions(&panel, controls.as_ref(), config, &counts);
    Ok(assemble(
        observations,
        &panel,
        config,
        &counts,
        inn.as_ref(),
        out.as_ref(),
    ))
}

fn validate(config: &DcdhConfig) -> Result<(), DcdhError> {
    if config.effects == 0 {
        return Err(DcdhError::InvalidConfig("effects must be at least 1"));
    }
    if !validate_confidence_level(config.confidence_level) {
        return Err(DcdhError::InvalidConfig(
            "confidence_level must be in (0, 1)",
        ));
    }
    if let DcdhEffectsEqual::Range { first, last } = config.effects_equal
        && (first < 1 || last <= first)
    {
        return Err(DcdhError::InvalidConfig(
            "effects_equal needs 1 <= first < last",
        ));
    }
    if config.same_switchers_pl && !config.same_switchers {
        return Err(DcdhError::InvalidConfig(
            "same_switchers_pl needs same_switchers",
        ));
    }
    if config.continuous == Some(0) {
        return Err(DcdhError::InvalidConfig(
            "the polynomial order of continuous must be at least 1",
        ));
    }
    Ok(())
}

/// The numbers of effects and placebos each direction allows, and those
/// estimated.
struct Counts {
    l: usize,
    l_pl: usize,
    in_effects: usize,
    out_effects: usize,
    in_placebos: usize,
    out_placebos: usize,
}

fn counts(panel: &Panel, config: &DcdhConfig) -> Result<Counts, DcdhError> {
    let gn = panel.n_grid_groups();
    let max_of = |inc: f64, f: &dyn Fn(usize) -> Option<i64>| {
        (0..gn).filter(|&g| panel.s_g[g] == inc).filter_map(f).max()
    };
    // With `trends_lin` the panel starts in the second period, which leaves
    // one placebo fewer.
    let shift = i64::from(config.trends_lin);
    let in_effects = max_of(1.0, &|g| Some(panel.l_g[g])).unwrap_or(0).max(0);
    let out_effects = max_of(0.0, &|g| Some(panel.l_g[g])).unwrap_or(0).max(0);
    let pin = (max_of(1.0, &|g| panel.l_g_placebo[g]).unwrap_or(0).max(0) - shift).max(0);
    let pout = (max_of(0.0, &|g| panel.l_g_placebo[g]).unwrap_or(0).max(0) - shift).max(0);
    let to_usize = |v: i64| usize::try_from(v).expect("not negative");
    let (in_effects, out_effects, in_placebos, out_placebos) = (
        to_usize(in_effects),
        to_usize(out_effects),
        to_usize(pin),
        to_usize(pout),
    );
    let (avail, avail_pl) = match config.switchers {
        DcdhSwitchers::Both => (in_effects.max(out_effects), in_placebos.max(out_placebos)),
        DcdhSwitchers::In => (in_effects, in_placebos),
        DcdhSwitchers::Out => (out_effects, out_placebos),
    };
    if avail == 0 {
        return Err(DcdhError::NoEffect);
    }
    let l = avail.min(config.effects);
    Ok(Counts {
        l,
        l_pl: avail_pl.min(config.placebo).min(config.effects),
        in_effects,
        out_effects,
        in_placebos,
        out_placebos,
    })
}

/// The terms of the switchers in and of the switchers out.
fn directions(
    panel: &Panel,
    controls: Option<&Controls>,
    config: &DcdhConfig,
    k: &Counts,
) -> (Option<DirectionTerms>, Option<DirectionTerms>) {
    let (run_in, run_out) = match config.switchers {
        DcdhSwitchers::Both => (true, true),
        DcdhSwitchers::In => (true, false),
        DcdhSwitchers::Out => (false, true),
    };
    let options = |ss: Option<usize>, ss_pl: Option<usize>| Options {
        only_never_switchers: config.only_never_switchers,
        normalized: config.normalized,
        same_switchers: ss,
        same_switchers_pl: ss_pl,
        less_conservative_se: config.less_conservative_se,
        trends_lin: config.trends_lin,
    };
    let (l, l_pl) = (k.l, k.l_pl);
    let run = |go: bool, inc: f64, lmax: usize, lpmax: usize| -> Option<DirectionTerms> {
        if !go || lmax == 0 {
            return None;
        }
        if !config.trends_lin {
            let ss = config.same_switchers.then_some(l);
            let ss_pl = (config.same_switchers_pl && l_pl > 0).then_some(l_pl);
            return Some(direction(
                panel,
                controls,
                inc,
                lmax.min(l),
                l_pl.min(lpmax),
                &options(ss, ss_pl),
            ));
        }
        // `trends_lin`: effect `i` comes from a pass with `i` effects over the
        // switchers that allow all of them, placebo `i` from a pass with `i`
        // effects and `i` placebos over those that allow all of both.
        let mut effects = Vec::new();
        for i in 1..=lmax.min(l) {
            let mut d = direction(panel, controls, inc, i, 0, &options(Some(i), None));
            effects.push(d.effects.pop().expect("effect i"));
        }
        let mut placebos = Vec::new();
        for i in 1..=lpmax.min(l_pl) {
            let mut d = direction(panel, controls, inc, i, i, &options(Some(i), Some(i)));
            placebos.push(d.placebos.pop().expect("placebo i"));
        }
        Some(DirectionTerms {
            effects,
            placebos,
            sum_n: 0.0,
            ate_u: Vec::new(),
            ate_u_var: Vec::new(),
            ate_den: f64::NAN,
            delta_d_g: Vec::new(),
        })
    };
    (
        run(run_in, 1.0, k.in_effects, k.in_placebos),
        run(run_out, 0.0, k.out_effects, k.out_placebos),
    )
}

/// The results from the terms of the two directions.
#[expect(
    clippy::too_many_lines,
    reason = "the results block of did_multiplegt_main, in its order"
)]
fn assemble(
    observations: &[DcdhObservation],
    panel: &Panel,
    config: &DcdhConfig,
    k: &Counts,
    inn: Option<&DirectionTerms>,
    out: Option<&DirectionTerms>,
) -> DcdhResult {
    let gn = panel.n_grid_groups();
    let (l, l_pl) = (k.l, k.l_pl);
    let z = z_score_for_confidence(config.confidence_level);
    let finish = |c: Combined, period: i64| -> (DcdhEstimate, Option<Vec<f64>>, Vec<f64>) {
        let sums = c.psi.as_ref().map(|p| cluster_sums(panel, p));
        let se = sums.as_ref().map(|s| sum_sq(s).sqrt());
        let n_cells = c.count.iter().filter(|&&v| v > 0.0).count();
        let n_cells_weighted: f64 = c.count.iter().sum();
        let ci = c.estimate.zip(se).map(|(e, s)| (e - z * s, e + z * s));
        let influence = c.psi.as_ref().map(|p| expand(panel, p));
        (
            DcdhEstimate {
                relative_period: period,
                estimate: c.estimate,
                se: c.estimate.and(se),
                ci,
                n_cells,
                n_switchers: c.n_switchers_dw,
                n_cells_weighted,
                n_switchers_weighted: c.n_switchers,
                influence: c.estimate.and(influence),
            },
            c.estimate.and(sums),
            c.count,
        )
    };

    let mut effects = Vec::with_capacity(l);
    let mut effect_sums = Vec::with_capacity(l);
    let mut normalization = Vec::new();
    let mut max_count = vec![0.0; gn * panel.nt];
    let mut switchers_total = (0.0, 0.0);
    for i in 0..l {
        let c = combine(
            panel,
            inn.and_then(|d| d.effects.get(i)),
            out.and_then(|d| d.effects.get(i)),
            config,
        );
        if config.normalized {
            normalization.push(c.delta);
        }
        switchers_total.0 += c.n_switchers_dw;
        switchers_total.1 += c.n_switchers;
        let (e, s, count) = finish(c, i64::try_from(i + 1).expect("index"));
        for (m, v) in max_count.iter_mut().zip(&count) {
            if *v > *m {
                *m = *v;
            }
        }
        effects.push(e);
        effect_sums.push(s);
    }
    let mut placebos = Vec::with_capacity(l_pl);
    let mut placebo_sums = Vec::with_capacity(l_pl);
    let mut placebo_normalization = Vec::new();
    for i in 0..l_pl {
        let c = combine(
            panel,
            inn.and_then(|d| d.placebos.get(i)),
            out.and_then(|d| d.placebos.get(i)),
            config,
        );
        if config.normalized {
            placebo_normalization.push(c.delta);
        }
        let (e, s, _) = finish(c, -i64::try_from(i + 1).expect("index"));
        placebos.push(e);
        placebo_sums.push(s);
    }

    // The average total effect.
    let (sp, sm) = (side(inn), side(out));
    let w_plus = match config.switchers {
        DcdhSwitchers::In => 1.0,
        DcdhSwitchers::Out => 0.0,
        DcdhSwitchers::Both => {
            let a = sp.map_or(0.0, |s| s.0 * s.1);
            let b = sm.map_or(0.0, |s| s.0 * s.1);
            a / (a + b)
        }
    };
    let mut total = 0.0;
    let mut psi = vec![0.0; gn];
    for g in 0..gn {
        let up = sp.map_or(0.0, |s| s.2[g]);
        let um = sm.map_or(0.0, |s| -s.2[g]);
        total += w_plus * up + (1.0 - w_plus) * um;
        let vp = sp.map_or(0.0, |s| s.3[g]);
        let vm = sm.map_or(0.0, |s| -s.3[g]);
        psi[g] = w_plus * vp + (1.0 - w_plus) * vm;
    }
    let ate = total / panel.n_groups;
    let ate_ok = estimable(
        config.switchers,
        sp.map_or(0.0, |s| s.1),
        sm.map_or(0.0, |s| s.1),
    ) && ate.is_finite();
    let (average_total_effect, ate_sums, _) = finish(
        Combined {
            estimate: (ate.is_finite() && !config.trends_lin).then_some(ate),
            psi: (ate_ok && !config.trends_lin).then_some(psi),
            n_switchers: switchers_total.1,
            n_switchers_dw: switchers_total.0,
            count: max_count,
            delta: None,
        },
        0,
    );
    let _ = ate_sums;

    // Tests.
    let est = |v: &[DcdhEstimate]| v.iter().map(|e| e.estimate).collect::<Vec<_>>();
    let deltas_ok = !config.normalized
        || normalization
            .iter()
            .all(|d| d.is_some_and(|d| d != 0.0 && d.is_finite()));
    let p_joint_effects = if deltas_ok {
        joint_test(&est(&effects), &effect_sums)
    } else {
        None
    };
    let p_joint_placebos = joint_test(&est(&placebos), &placebo_sums);
    let p_equal_effects = match config.effects_equal {
        DcdhEffectsEqual::No => None,
        DcdhEffectsEqual::All => equality_test(&est(&effects), &effect_sums),
        DcdhEffectsEqual::Range { first, last } => {
            let last = last.min(l);
            if first > last {
                None
            } else {
                equality_test(
                    &est(&effects)[first - 1..last],
                    &effect_sums[first - 1..last],
                )
            }
        }
    };

    let all_sums: Vec<&Option<Vec<f64>>> = effect_sums.iter().chain(&placebo_sums).collect();
    let k = all_sums.len();
    let mut covariance = vec![f64::NAN; k * k];
    for (a, sa) in all_sums.iter().enumerate() {
        for (b, sb) in all_sums.iter().enumerate() {
            if let (Some(x), Some(y)) = (sa, sb) {
                covariance[a * k + b] = dot(x, y);
            }
        }
    }

    let delta_d_avg_total = if config.trends_lin {
        None
    } else {
        delta_d_avg(panel, inn, out, l)
    };
    let normalized_weights = if config.normalized {
        lag_weights(panel, inn, out, &normalization, &effects)
    } else {
        Vec::new()
    };
    let avg_time_periods = config
        .avg_time_periods
        .then(|| time_periods(panel, l, config));

    DcdhResult {
        effects,
        placebos,
        average_total_effect,
        p_joint_effects,
        p_joint_placebos,
        p_equal_effects,
        covariance,
        normalization,
        placebo_normalization,
        delta_d_avg_total,
        normalized_weights,
        avg_time_periods,
        n_groups: panel.counted_ids.len(),
        groups: panel.counted_ids.clone(),
        clusters: counted_clusters(observations, panel),
    }
}

/// A per-grid-group vector over all counted groups, 0 for a group without
/// rows in the grid.
fn expand(panel: &Panel, psi: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; panel.counted_ids.len()];
    let mut j = 0;
    for (g, id) in panel.group_ids.iter().enumerate() {
        while panel.counted_ids[j] != *id {
            j += 1;
        }
        out[j] = psi[g];
    }
    out
}

fn counted_clusters(observations: &[DcdhObservation], panel: &Panel) -> Option<Vec<i64>> {
    panel.cluster.as_ref()?;
    let map: std::collections::HashMap<i64, i64> = observations
        .iter()
        .filter_map(|o| o.cluster.map(|c| (o.group, c)))
        .collect();
    Some(panel.counted_ids.iter().map(|g| map[g]).collect())
}

/// `delta_D_avg_total`: over the switchers' cells `F_g - 1 + j`, the mean of
/// `min(L, T_g - F_g + 1) - j + 1` weighted by the change in treatment.
#[expect(
    clippy::needless_range_loop,
    reason = "a cell's index also gives its group"
)]
fn delta_d_avg(
    panel: &Panel,
    inn: Option<&DirectionTerms>,
    out: Option<&DirectionTerms>,
    l: usize,
) -> Option<f64> {
    let nt = panel.nt;
    let cells = panel.n_grid_groups() * nt;
    let lf = l as f64;
    let mut tag = vec![0_usize; cells];
    for d in [inn, out].into_iter().flatten() {
        for (j, e) in d.effects.iter().enumerate() {
            for c in 0..cells {
                if e.dist[c] == 1.0 {
                    tag[c] = j + 1;
                }
            }
        }
    }
    let value = |d: Option<&DirectionTerms>, j: usize, c: usize| {
        d.and_then(|d| d.effects.get(j))
            .filter(|e| e.n != 0.0)
            .map_or(0.0, |_| d.expect("present").delta_d_g[j][c])
    };
    let (mut num, mut den) = (0.0, 0.0);
    for c in 0..cells {
        if tag[c] == 0 {
            continue;
        }
        let j = tag[c] - 1;
        let plus = value(inn, j, c);
        let minus = value(out, j, c);
        let temp = if plus.is_nan() {
            f64::NAN
        } else if plus != 0.0 {
            plus
        } else {
            minus
        };
        if temp.is_nan() || temp == 0.0 {
            continue;
        }
        let g = c / nt;
        let m = lf.min((panel.t_g[g] - panel.f_g[g] + 1) as f64);
        num += temp * (m - (tag[c] as f64 - 1.0));
        den += temp;
    }
    let v = num / den;
    (!v.is_nan()).then_some(v)
}
