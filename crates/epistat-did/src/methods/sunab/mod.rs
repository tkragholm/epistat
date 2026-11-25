//! The interaction-weighted event study of Sun and Abraham \[1\], as
//! `fixest::feols(y ~ sunab(g, t) | id + t)` fits it and `fixest::aggregate`
//! summarises it \[2\].
//!
//! # Algorithm
//!
//! For unit `i` in period `t`, first treated in period `g_i`, the regression
//! is saturated in cohort and relative period:
//!
//! ```text
//! y_it = alpha_i + lambda_t + sum_{g in C} sum_{e not in R} delta_ge 1[g_i = g, t - g = e] + x_it' gamma + e_it
//! ```
//!
//! with `C` the treated cohorts and `R` the reference periods (by default
//! `{-1}`). The control cohorts carry no indicator: the never treated, the
//! cohorts with no row from their first treated period on (treated after the
//! panel, or not observed then), and the cohorts the caller names
//! ([`SunabControl::Cohorts`], `ref.c` of `sunab`). With
//! [`SunabControl::LastTreated`] the last treated cohort is the control and
//! the rows from its first treated period on are removed, Sun and Abraham's
//! choice when no unit is never treated \[1\]. Cohorts with no row before
//! their first treated period (always treated) are removed, as `sunab` means
//! to remove them. Binning ([`SunabBins`], `bin.rel` of `sunab`) relabels the
//! relative periods after the reference periods are removed, so a bin pools
//! the cells of its periods into one coefficient per cohort.
//!
//! `delta_ge` is the cohort-specific average treatment effect `CATT(g, e)`
//! under parallel trends and no anticipation: with the never treated as
//! controls and a balanced panel it equals the `ATT(g, g + e)` of Callaway and
//! Sant'Anna with the universal base period `g - 1`.
//!
//! The unit and period effects are absorbed by [`epistat_glm::absorb`] and the
//! regression is fitted by [`epistat_glm::feols`], so the cost is linear in
//! the rows at a fixed number of cells and no unit dummies are formed. The
//! covariance is clustered by unit, or by the caller's clusters, with the
//! small-sample correction of `fixest`.
//!
//! # Aggregation
//!
//! A summary averages the coefficients of a set `S` of cells by their shares
//! in the fit, `fixest`'s `aggregate`:
//!
//! ```text
//! n_ge = sum_{rows in cell (g, e)} w_it,     s_ge = n_ge / sum_{S} n,
//! theta = sum_S s_ge delta_ge,               Var(theta) = s' V s
//! ```
//!
//! with `V` the clustered covariance of the cells. The summaries are by
//! relative period (`S` the cells with label `e`, the interaction-weighted
//! estimator of \[1\]), by cohort and overall (`S` the cells with labels at or
//! after 0). The shares are taken as fixed, as `fixest` takes them, so the
//! variance leaves out their sampling error. Cells dropped as collinear do not
//! enter a summary.
//!
//! # Influence
//!
//! The influence function of a coefficient for unit `u` is
//! `psi_u = N_u e_k' B sum_{rows of u} w_it e_it x~_it`, with `N_u` the units in
//! the fit, `x~` the demeaned regressors and `B = (X~' W X~)^-1`, and that of a
//! summary is `sum_S s_ge psi_ge`. Summed within unit clusters by
//! [`did_cluster_sums`](crate::did_cluster_sums) and
//! [`did_covariance`](crate::did_covariance), they give the clustered
//! covariance without the small-sample factors.
//!
//! # Where this differs from `fixest` 0.14.2
//!
//! `sunab` classifies a cohort as always treated, never treated or treated
//! from its rows sorted by cohort, and skips the first row of every cohort
//! after the first: a cohort whose only row before treatment is that row is
//! taken as always treated, loses its indicators and becomes a control, so the
//! estimates depend on the order of the rows. It then sets to missing the rows
//! whose row numbers are the indices of the always-treated cohorts, not those
//! cohorts' rows, so the always treated stay in the fit as controls and other
//! rows are removed. A cohort first treated before the first period whose
//! value is not a period is taken as never treated. Here every row counts and
//! the always treated are removed.
//!
//! # References
//!
//! 1. Sun L, Abraham S. Estimating dynamic treatment effects in event studies
//!    with heterogeneous treatment effects. J Econom. 2021;225(2):175-99.
//!    doi:10.1016/j.jeconom.2020.09.006
//! 2. Bergé L. The R package `fixest` 0.14.2: `sunab`, `aggregate.fixest`.

use std::collections::{BTreeMap, BTreeSet};

use epistat_glm::{Absorb, AbsorbError, Feols, FixedEffects, Singletons, Ssc, Vcov};
use epistat_linalg::Mat;

use crate::methods::imputation::check_constant_within_units;
use crate::methods::staggered::StaggeredPanel;

/// The cohorts that serve as controls besides the never treated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SunabControl<'a> {
    /// The never treated and the cohorts without a row from their first
    /// treated period on, as `sunab` by default.
    NeverTreated,
    /// These cohorts as well, with all their rows (`ref.c`).
    Cohorts(&'a [i32]),
    /// The last treated cohort as well, with the rows from its first treated
    /// period on removed.
    LastTreated,
}

/// A relative period left out of the interactions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SunabReference {
    /// This relative period.
    Period(i32),
    /// The smallest relative period (`.F` of `sunab`).
    First,
    /// The largest relative period (`.L` of `sunab`).
    Last,
}

/// The binning of the relative periods (`bin.rel` of `sunab`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SunabBins {
    /// None.
    None,
    /// Periods below `lo` count as `lo` and periods above `hi` as `hi`,
    /// `bin.rel = list("lo" = min:lo, "hi" = hi:max)`.
    Clamp {
        /// The lowest label.
        lo: i32,
        /// The highest label.
        hi: i32,
    },
    /// Period `e` counts as `floor(e / width) * width`, `bin.rel = "bin::width"`.
    Width(i32),
}

impl SunabBins {
    fn label(self, e: i32) -> i32 {
        match self {
            Self::None => e,
            Self::Clamp { lo, hi } => e.clamp(lo, hi),
            Self::Width(w) => e.div_euclid(w) * w,
        }
    }
}

/// The reference period of `sunab` by default.
pub const SUNAB_DEFAULT_REFERENCES: &[SunabReference] = &[SunabReference::Period(-1)];

/// The controls of [`estimate_sunab`].
#[derive(Clone, Copy, Debug)]
pub struct SunabConfig<'a> {
    /// The control cohorts.
    pub control: SunabControl<'a>,
    /// The relative periods left out.
    pub references: &'a [SunabReference],
    /// The binning of the relative periods.
    pub bins: SunabBins,
    /// Cluster codes, one per row; the units when `None`.
    pub clusters: Option<&'a [u32]>,
    /// The demeaning.
    pub absorb: Absorb,
    /// The small-sample correction of the covariance.
    pub ssc: Ssc,
}

impl SunabConfig<'_> {
    /// The defaults of `sunab`: never-treated controls, reference period -1,
    /// no bins, unit clusters and `fixest`'s correction.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            control: SunabControl::NeverTreated,
            references: SUNAB_DEFAULT_REFERENCES,
            bins: SunabBins::None,
            clusters: None,
            absorb: Absorb::new(),
            ssc: Ssc::new(),
        }
    }
}

impl Default for SunabConfig<'_> {
    fn default() -> Self {
        Self::new()
    }
}

/// Why an interaction-weighted event study failed.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SunabError {
    /// An input has the wrong length.
    #[error("{what} has {found} entries where {expected} were expected")]
    Length {
        /// The input.
        what: &'static str,
        /// The length it needed.
        expected: usize,
        /// The length it had.
        found: usize,
    },
    /// A unit has rows with different first treated periods.
    #[error("unit {unit} has rows with different first treated periods")]
    Cohort {
        /// The unit.
        unit: u32,
    },
    /// No treated cohort has a row outside the reference periods.
    #[error("no row carries a cohort and relative period indicator")]
    NoTreatment,
    /// The bins are empty or their width is not positive.
    #[error("the bins need lo <= hi or a positive width")]
    Bins,
    /// A period or a relative period does not fit in 32 bits.
    #[error("the periods span more than 2^31 values")]
    Periods,
    /// The fit failed.
    #[error(transparent)]
    Fit(#[from] AbsorbError),
}

/// One cohort and relative period coefficient.
#[derive(Clone, Debug, PartialEq)]
pub struct SunabCell {
    /// The cohort, its first treated period.
    pub cohort: i32,
    /// The relative period, after binning.
    pub relative_period: i32,
    /// The coefficient, `None` when the indicator is collinear.
    pub estimate: Option<f64>,
    /// Its clustered standard error.
    pub std_error: Option<f64>,
    /// The sum of the prior weights of its rows in the fit, the `n_ge` of
    /// the shares.
    pub weight: f64,
}

/// What a summary averages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SunabTerm {
    /// The cells at this relative period.
    Period(i32),
    /// The cells of this cohort at relative periods from 0.
    Cohort(i32),
    /// Every cell at a relative period from 0.
    Att,
}

/// A summary of the cell coefficients.
#[derive(Clone, Debug, PartialEq)]
pub struct SunabEstimate {
    /// What it averages.
    pub term: SunabTerm,
    /// The share-weighted mean of its cells.
    pub estimate: f64,
    /// `sqrt(s' V s)`, `None` when the covariance is undefined.
    pub std_error: Option<f64>,
    /// The cells it averages.
    pub n_cells: usize,
}

/// An interaction-weighted event study.
#[derive(Clone, Debug)]
pub struct SunabResult {
    /// The cell coefficients, by relative period and then cohort, as
    /// `sunab` orders them.
    pub cells: Vec<SunabCell>,
    /// The summaries: by relative period, increasing, then by cohort,
    /// increasing, then overall. A summary without cells is left out.
    pub estimates: Vec<SunabEstimate>,
    /// The clustered covariance of the summaries, row-major, `None` when
    /// undefined.
    pub estimate_covariance: Vec<Option<f64>>,
    /// The influence function of each summary, one value per unit of
    /// [`SunabResult::units`].
    pub estimate_influence: Vec<Vec<f64>>,
    /// The coefficients of the covariates.
    pub covariate_coefficients: Vec<Option<f64>>,
    /// The clustered covariance of the cells and then the covariates,
    /// row-major, `None` for dropped terms.
    pub covariance: Vec<Option<f64>>,
    /// The `K` of the small-sample correction.
    pub df_k: usize,
    /// The number of clusters.
    pub n_clusters: usize,
    /// The influence function of each coefficient (cells, then covariates),
    /// one value per unit; `None` for dropped terms.
    pub influence: Vec<Option<Vec<f64>>>,
    /// The units in the fit, in increasing code.
    pub units: Vec<u32>,
    /// The rows in the fit.
    pub n_rows: usize,
    /// The rows removed as singletons.
    pub removed_singletons: usize,
    /// The rows removed before the fit: those of always-treated cohorts and,
    /// with [`SunabControl::LastTreated`], those from the last cohort's first
    /// treated period on.
    pub excluded_rows: usize,
}

/// The fixed-effect codes of the periods, from 0.
pub(crate) fn period_codes(time: &[i32]) -> Option<Vec<u32>> {
    let t_min = time.iter().copied().min().unwrap_or(0);
    time.iter()
        .map(|&t| u32::try_from(i64::from(t) - i64::from(t_min)).ok())
        .collect()
}

/// Whether each cohort has rows before and from its first treated period.
fn cohort_rows(
    panel: &StaggeredPanel<'_>,
    excluded: &[bool],
) -> Result<BTreeMap<i32, (bool, bool)>, SunabError> {
    let mut seen: BTreeMap<i32, (bool, bool)> = BTreeMap::new();
    for (i, g) in panel.first_treated.iter().enumerate() {
        if let Some(g) = *g
            && !excluded[i]
        {
            let e = panel.time[i].checked_sub(g).ok_or(SunabError::Periods)?;
            let entry = seen.entry(g).or_default();
            if e < 0 {
                entry.0 = true;
            } else {
                entry.1 = true;
            }
        }
    }
    Ok(seen)
}

/// Fits the interaction-weighted event study of `config` to `panel`. See the
/// module documentation.
///
/// # Errors
///
/// [`SunabError`] for inputs of the wrong length, cohorts that vary within a
/// unit, unusable bins, no indicator, periods out of range, and the errors of
/// the fit.
pub fn estimate_sunab(
    panel: &StaggeredPanel<'_>,
    config: &SunabConfig<'_>,
) -> Result<SunabResult, SunabError> {
    let n = panel.unit.len();
    for (what, found) in [
        ("time", panel.time.len()),
        ("first_treated", panel.first_treated.len()),
        ("outcome", panel.outcome.len()),
        ("weights", panel.weights.map_or(n, <[f64]>::len)),
        ("covariates", panel.covariates.map_or(n, |x| x.nrows())),
        ("clusters", config.clusters.map_or(n, <[u32]>::len)),
    ] {
        if found != n {
            return Err(SunabError::Length {
                what,
                expected: n,
                found,
            });
        }
    }
    match config.bins {
        SunabBins::Clamp { lo, hi } if lo > hi => return Err(SunabError::Bins),
        SunabBins::Width(w) if w <= 0 => return Err(SunabError::Bins),
        _ => {}
    }
    check_constant_within_units(panel.unit, panel.first_treated)
        .map_err(|unit| SunabError::Cohort { unit })?;
    let time = period_codes(panel.time).ok_or(SunabError::Periods)?;

    // The rows removed for the last-treated control, then the cohorts.
    let mut excluded = vec![false; n];
    let mut rows = cohort_rows(panel, &excluded)?;
    let named: &[i32] = match config.control {
        SunabControl::Cohorts(c) => c,
        _ => &[],
    };
    let mut cutoff = None;
    if config.control == SunabControl::LastTreated {
        let last = rows
            .iter()
            .filter(|&(_, &(pre, post))| pre && post)
            .map(|(&g, _)| g)
            .next_back()
            .ok_or(SunabError::NoTreatment)?;
        for (i, x) in excluded.iter_mut().enumerate() {
            *x = panel.time[i] >= last;
        }
        cutoff = Some(last);
        rows = cohort_rows(panel, &excluded)?;
    }
    let treated: BTreeSet<i32> = rows
        .iter()
        .filter(|&(g, &(pre, post))| pre && post && !named.contains(g))
        .map(|(&g, _)| g)
        .collect();
    let always: BTreeSet<i32> = rows
        .iter()
        .filter(|&(g, &(pre, post))| post && !pre && !named.contains(g))
        .map(|(&g, _)| g)
        .collect();
    for (i, x) in excluded.iter_mut().enumerate() {
        if panel.first_treated[i].is_some_and(|g| always.contains(&g)) {
            *x = true;
        }
    }

    // The reference periods. `.F` and `.L` of `sunab` run over the relative
    // periods of every cohort whose first treated period is a period, with
    // -1 for the others, before the always treated are removed.
    let periods: BTreeSet<i32> = panel.time.iter().copied().collect();
    let mut first = i32::MAX;
    let mut last = i32::MIN;
    for i in 0..n {
        if cutoff.is_some_and(|c| panel.time[i] >= c) {
            continue;
        }
        let e = match panel.first_treated[i] {
            Some(g) if periods.contains(&g) => panel.time[i] - g,
            _ => -1,
        };
        first = first.min(e);
        last = last.max(e);
    }
    let references: Vec<i32> = config
        .references
        .iter()
        .map(|r| match *r {
            SunabReference::Period(p) => p,
            SunabReference::First => first,
            SunabReference::Last => last,
        })
        .collect();

    // The cells and the cell of each row.
    let relative = |i: usize| -> Option<i32> {
        let g = panel.first_treated[i]?;
        if excluded[i] || !treated.contains(&g) {
            return None;
        }
        let e = panel.time[i] - g;
        (!references.contains(&e)).then(|| config.bins.label(e))
    };
    let cell_set: BTreeSet<(i32, i32)> = (0..n)
        .filter_map(|i| relative(i).map(|e| (e, panel.first_treated[i].unwrap_or(0))))
        .collect();
    if cell_set.is_empty() {
        return Err(SunabError::NoTreatment);
    }
    let cells: Vec<(i32, i32)> = cell_set.into_iter().collect();
    let cohorts: Vec<i32> = treated.iter().copied().collect();
    let (e_min, e_max) = (cells[0].0, cells[cells.len() - 1].0);
    let width = usize::try_from(i64::from(e_max) - i64::from(e_min) + 1)
        .map_err(|_| SunabError::Periods)?;
    let mut column_of = vec![u32::MAX; width * cohorts.len()];
    let slot = |e: i32, g: i32| -> usize {
        let a = usize::try_from(i64::from(e) - i64::from(e_min)).unwrap_or(usize::MAX);
        let b = cohorts.binary_search(&g).unwrap_or(usize::MAX);
        a * cohorts.len() + b
    };
    for (j, &(e, g)) in cells.iter().enumerate() {
        column_of[slot(e, g)] = u32::try_from(j).unwrap_or(u32::MAX);
    }
    let row_cell: Vec<u32> = (0..n)
        .map(|i| {
            relative(i).map_or(u32::MAX, |e| {
                column_of[slot(e, panel.first_treated[i].unwrap_or(0))]
            })
        })
        .collect();

    let k_cells = cells.len();
    let p_cov = panel.covariates.map_or(0, |x| x.ncols());
    let mut x = Mat::<f64>::zeros(n, k_cells + p_cov);
    for (i, &c) in row_cell.iter().enumerate() {
        if c != u32::MAX {
            x[(i, c as usize)] = 1.0;
        }
    }
    if let Some(cov) = panel.covariates {
        for j in 0..p_cov {
            let column = x.col_as_slice_mut(k_cells + j);
            for (i, v) in column.iter_mut().enumerate() {
                *v = cov[(i, j)];
            }
        }
    }
    let excluded_rows = excluded.iter().filter(|&&b| b).count();
    let fit_weights: Option<Vec<f64>> = (panel.weights.is_some() || excluded_rows > 0).then(|| {
        (0..n)
            .map(|i| {
                if excluded[i] {
                    0.0
                } else {
                    panel.weights.map_or(1.0, |w| w[i])
                }
            })
            .collect()
    });

    let fe = FixedEffects::new(
        &[panel.unit, &time],
        fit_weights.as_deref(),
        Singletons::Remove,
    )?;
    let fit = Feols {
        absorb: config.absorb,
    }
    .fit(panel.outcome, x.as_ref(), &fe)?;
    drop(x);
    let clusters = config.clusters.unwrap_or(panel.unit);
    let ways = [clusters];
    let cov = fit.vcov(Vcov::Cluster(&ways), config.ssc)?;
    let p = k_cells + p_cov;
    let se = |j: usize| cov.matrix[j * p + j].map(|v| v.max(0.0).sqrt());

    // The weight of each cell among the rows of the fit.
    let mut cell_weight = vec![0.0; k_cells];
    let fe_weights = fe.weights();
    for (k, &i) in fe.rows().iter().enumerate() {
        let c = row_cell[i];
        if c != u32::MAX {
            cell_weight[c as usize] += fe_weights.map_or(1.0, |w| w[k]);
        }
    }
    let cell_results: Vec<SunabCell> = cells
        .iter()
        .enumerate()
        .map(|(j, &(e, g))| SunabCell {
            cohort: g,
            relative_period: e,
            estimate: fit.coefficients[j],
            std_error: se(j),
            weight: cell_weight[j],
        })
        .collect();

    // Influence by unit.
    let unit_codes = fe.codes(0);
    let units = fe.labels(0).to_vec();
    let n_units = units.len();
    let sums = fit.influence_sums(unit_codes, n_units)?;
    let mut influence = vec![None; p];
    for (a, &j) in fit.kept_columns().iter().enumerate() {
        influence[j] = Some(
            (0..n_units)
                .map(|u| n_units as f64 * sums[(u, a)])
                .collect::<Vec<f64>>(),
        );
    }

    // The summaries.
    let kept = |j: usize| fit.coefficients[j].is_some();
    let mut groups: Vec<(SunabTerm, Vec<usize>)> = Vec::new();
    let labels: BTreeSet<i32> = cells.iter().map(|c| c.0).collect();
    for &e in &labels {
        groups.push((
            SunabTerm::Period(e),
            (0..k_cells)
                .filter(|&j| cells[j].0 == e && kept(j))
                .collect(),
        ));
    }
    for &g in &cohorts {
        groups.push((
            SunabTerm::Cohort(g),
            (0..k_cells)
                .filter(|&j| cells[j].1 == g && cells[j].0 >= 0 && kept(j))
                .collect(),
        ));
    }
    groups.push((
        SunabTerm::Att,
        (0..k_cells)
            .filter(|&j| cells[j].0 >= 0 && kept(j))
            .collect(),
    ));
    groups.retain(|(_, members)| !members.is_empty());
    let shares: Vec<Vec<(usize, f64)>> = groups
        .iter()
        .map(|(_, members)| {
            let total: f64 = members.iter().map(|&j| cell_weight[j]).sum();
            members
                .iter()
                .map(|&j| (j, cell_weight[j] / total))
                .collect()
        })
        .collect();
    let m = groups.len();
    let mut estimate_covariance = vec![None; m * m];
    for a in 0..m {
        for b in 0..m {
            let mut total = 0.0;
            let mut defined = true;
            for &(i, si) in &shares[a] {
                for &(j, sj) in &shares[b] {
                    match cov.matrix[i * p + j] {
                        Some(v) => total += si * sj * v,
                        None => defined = false,
                    }
                }
            }
            estimate_covariance[a * m + b] = defined.then_some(total);
        }
    }
    let estimates: Vec<SunabEstimate> = groups
        .iter()
        .zip(&shares)
        .enumerate()
        .map(|(a, ((term, members), s))| SunabEstimate {
            term: *term,
            estimate: s
                .iter()
                .map(|&(j, w)| w * fit.coefficients[j].unwrap_or(0.0))
                .sum(),
            std_error: estimate_covariance[a * m + a].map(|v| v.max(0.0).sqrt()),
            n_cells: members.len(),
        })
        .collect();
    let estimate_influence: Vec<Vec<f64>> = shares
        .iter()
        .map(|s| {
            let mut out = vec![0.0; n_units];
            for &(j, w) in s {
                if let Some(psi) = &influence[j] {
                    for (o, v) in out.iter_mut().zip(psi) {
                        *o += w * v;
                    }
                }
            }
            out
        })
        .collect();

    Ok(SunabResult {
        cells: cell_results,
        estimates,
        estimate_covariance,
        estimate_influence,
        covariate_coefficients: fit.coefficients[k_cells..].to_vec(),
        covariance: cov.matrix,
        df_k: cov.df_k,
        n_clusters: cov.min_clusters.unwrap_or(0),
        influence,
        units,
        n_rows: fe.n_rows(),
        removed_singletons: fe.removed_singletons(),
        excluded_rows,
    })
}
