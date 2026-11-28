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
//! covariance is clustered by the panel's clusters, or by unit when it has
//! none, with the small-sample correction of `fixest`. The intervals and
//! p-values of the cells and of the aggregates use the t distribution with
//! `G - 1` degrees of freedom, as `fixest` and its `aggregate` do.
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
//! summary is `sum_S s_ge psi_ge`. [`Influence::variance`] over the panel's
//! clusters gives the reported covariance without its factor
//! `(n - 1)/(n - K)`, under the
//! [`ClusterRule`](epistat_inference::ClusterRule) of the factor `G/(G - 1)`
//! of [`SunabOptions::ssc`].
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

use epistat_core::{Covariance, Estimate, EventStudy, Level};
use epistat_glm::{Absorb, Feols, FixedEffects, Singletons, Ssc, Vcov};
use epistat_inference::Influence;
use epistat_linalg::Mat;

use crate::error::Error;
use crate::methods::fe_output::{self, Reference};
use crate::methods::staggered::StaggeredPanel;

/// The cohorts that serve as controls besides the never treated.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum SunabControl {
    /// The never treated and the cohorts without a row from their first
    /// treated period on, as `sunab` by default.
    #[default]
    NeverTreated,
    /// These cohorts as well, with all their rows (`ref.c`).
    Cohorts(Vec<i64>),
    /// The last treated cohort as well, with the rows from its first treated
    /// period on removed.
    LastTreated,
}

/// A relative period left out of the interactions.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SunabReference {
    /// This relative period.
    Period(i64),
    /// The smallest relative period (`.F` of `sunab`).
    First,
    /// The largest relative period (`.L` of `sunab`).
    Last,
}

/// The binning of the relative periods (`bin.rel` of `sunab`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SunabBins {
    /// None.
    #[default]
    None,
    /// Periods below `lo` count as `lo` and periods above `hi` as `hi`,
    /// `bin.rel = list("lo" = min:lo, "hi" = hi:max)`.
    Clamp {
        /// The lowest label.
        lo: i64,
        /// The highest label.
        hi: i64,
    },
    /// Period `e` counts as `floor(e / width) * width`, `bin.rel = "bin::width"`.
    Width(i64),
}

impl SunabBins {
    fn label(self, e: i64) -> i64 {
        match self {
            Self::None => e,
            Self::Clamp { lo, hi } => e.clamp(lo, hi),
            Self::Width(w) => e.div_euclid(w) * w,
        }
    }
}

/// The options of [`fit`], with the defaults of `sunab`.
#[derive(Clone, Debug, PartialEq)]
pub struct SunabOptions {
    /// The control cohorts; by default the never treated.
    pub control: SunabControl,
    /// The relative periods left out (`ref.p`); by default -1.
    pub references: Vec<SunabReference>,
    /// The binning of the relative periods; by default none.
    pub bins: SunabBins,
    /// The demeaning, `fixef.tol` and `fixef.iter` of `feols`.
    pub absorb: Absorb,
    /// The small-sample correction of the covariance, `ssc` of `feols`.
    pub ssc: Ssc,
    /// The level of the intervals.
    pub level: Level,
    #[doc(hidden)]
    pub _extensible: epistat_core::Extensible,
}

impl Default for SunabOptions {
    fn default() -> Self {
        Self {
            _extensible: epistat_core::Extensible::NEW,
            control: SunabControl::NeverTreated,
            references: vec![SunabReference::Period(-1)],
            bins: SunabBins::None,
            absorb: Absorb::new(),
            ssc: Ssc::new(),
            level: Level::default(),
        }
    }
}

/// One cohort and relative period coefficient.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct SunabCell {
    /// The cohort, its first treated period.
    pub cohort: i64,
    /// The relative period, after binning.
    pub relative_period: i64,
    /// The coefficient with its clustered standard error, interval and
    /// p-value, `None` when the indicator is collinear.
    pub estimate: Option<Estimate>,
    /// The sum of the prior weights of its rows in the fit, the `n_ge` of
    /// the shares.
    pub weight: f64,
}

/// What an aggregate averages.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SunabTerm {
    /// The cells at this relative period.
    Period(i64),
    /// The cells of this cohort at relative periods from 0.
    Cohort(i64),
    /// Every cell at a relative period from 0.
    Att,
}

/// An aggregate of the cell coefficients, a row of `fixest::aggregate`.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct SunabAggregate {
    /// What it averages.
    pub term: SunabTerm,
    /// The share-weighted mean of its cells, with the standard error
    /// `sqrt(s' V s)` (`None` when the covariance is undefined), its interval
    /// and p-value.
    pub estimate: Estimate,
    /// The cells it averages.
    pub n_cells: usize,
}

/// An interaction-weighted event study.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct SunabFit {
    /// The cell coefficients, by relative period and then cohort, as
    /// `sunab` orders them.
    pub cells: Vec<SunabCell>,
    /// The aggregates: by relative period, increasing, then by cohort,
    /// increasing, then overall. An aggregate without cells is left out.
    pub aggregates: Vec<SunabAggregate>,
    /// The clustered covariance of the aggregates, `None` where undefined.
    pub aggregate_covariance: Covariance,
    /// The relative periods left out of the interactions, resolved and in
    /// increasing order.
    pub references: Vec<i64>,
    /// The influence function of each aggregate, one column each in their
    /// order with one value per unit in the fit in increasing code; `None`
    /// when there is no aggregate.
    pub aggregate_influence: Option<Influence>,
    /// The coefficients of the covariates, `None` when aliased.
    pub covariate_coefficients: Vec<Option<Estimate>>,
    /// The clustered covariance of the cells and then the covariates, `None`
    /// for dropped terms.
    pub covariance: Covariance,
    /// The `K` of the small-sample correction.
    pub df_k: usize,
    /// The number of clusters.
    pub n_clusters: usize,
    /// The influence functions of the coefficients that were estimated, the
    /// cells and then the covariates, one column each in that order, with
    /// one value per unit in the fit in increasing code; `None` when no
    /// coefficient was estimated. [`SunabFit::influence_of`] finds the column
    /// of a coefficient.
    pub influence: Option<Influence>,
    /// The units in the fit.
    pub n_units: usize,
    /// The rows in the fit.
    pub n_rows: usize,
    /// The rows removed as singletons.
    pub removed_singletons: usize,
    /// The rows removed before the fit: those of always-treated cohorts and,
    /// with [`SunabControl::LastTreated`], those from the last cohort's first
    /// treated period on.
    pub excluded_rows: usize,
}

impl SunabFit {
    /// The influence function of coefficient `j`, counting the cells and
    /// then the covariates; `None` for a dropped coefficient or one out of
    /// range.
    #[must_use]
    pub fn influence_of(&self, j: usize) -> Option<&[f64]> {
        let defined = self
            .cells
            .iter()
            .map(|c| c.estimate.is_some())
            .chain(self.covariate_coefficients.iter().map(Option::is_some));
        fe_output::column_of(self.influence.as_ref(), defined, j)
    }

    /// The event study of the aggregates by relative period, the
    /// interaction-weighted estimates of Sun and Abraham, with their block of
    /// [`SunabFit::aggregate_covariance`], the input of HonestDiD and of the
    /// pre-trend power. Its reference is the last relative period left out
    /// before 0.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] when no reference period lies before 0, and
    /// [`Error::Input`] when a binned period carries the reference's label.
    pub fn event_study(&self) -> Result<EventStudy, Error> {
        let Some(&reference) = self.references.iter().rev().find(|&&r| r < 0) else {
            return Err(Error::Unsupported {
                what: "an event study without a reference period before treatment",
            });
        };
        let (positions, periods): (Vec<usize>, Vec<i64>) = self
            .aggregates
            .iter()
            .enumerate()
            .filter_map(|(a, aggregate)| match aggregate.term {
                SunabTerm::Period(e) => Some((a, e)),
                _ => None,
            })
            .unzip();
        let estimates: Vec<Option<f64>> = positions
            .iter()
            .map(|&a| Some(self.aggregates[a].estimate.estimate))
            .collect();
        let values = positions
            .iter()
            .flat_map(|&a| {
                positions
                    .iter()
                    .map(move |&b| self.aggregate_covariance.get(a, b))
            })
            .collect();
        let covariance = Covariance::new(positions.len(), values)?;
        fe_output::event_study(&estimates, &periods, &covariance, reference)
    }
}

/// Whether each cohort has rows before and from its first treated period.
fn cohort_rows(
    panel: &StaggeredPanel<'_>,
    excluded: &[bool],
) -> Result<BTreeMap<i64, (bool, bool)>, Error> {
    let mut seen: BTreeMap<i64, (bool, bool)> = BTreeMap::new();
    for (i, g) in panel.first_treated().iter().enumerate() {
        if let Some(g) = *g
            && !excluded[i]
        {
            let e = panel.period()[i].checked_sub(g).ok_or(Error::Periods)?;
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

/// Fits the interaction-weighted event study of `options` to `panel`, with
/// the covariance clustered by the panel's clusters, or by unit when it has
/// none. See the module documentation.
///
/// # Errors
///
/// [`Error::Bins`] for unusable bins, [`Error::NoTreatment`] for no
/// indicator, [`Error::Periods`] for periods out of range, and [`Error::Glm`]
/// for the errors of the fit.
pub fn fit(panel: &StaggeredPanel<'_>, options: &SunabOptions) -> Result<SunabFit, Error> {
    let n = panel.len();
    match options.bins {
        SunabBins::Clamp { lo, hi } if lo > hi => return Err(Error::Bins),
        SunabBins::Width(w) if w <= 0 => return Err(Error::Bins),
        _ => {}
    }
    let time = panel.period_codes().ok_or(Error::Periods)?;

    // The rows removed for the last-treated control, then the cohorts.
    let mut excluded = vec![false; n];
    let mut rows = cohort_rows(panel, &excluded)?;
    let named: &[i64] = match &options.control {
        SunabControl::Cohorts(c) => c,
        _ => &[],
    };
    let mut cutoff = None;
    if options.control == SunabControl::LastTreated {
        let last = rows
            .iter()
            .filter(|&(_, &(pre, post))| pre && post)
            .map(|(&g, _)| g)
            .next_back()
            .ok_or(Error::NoTreatment)?;
        for (i, x) in excluded.iter_mut().enumerate() {
            *x = panel.period()[i] >= last;
        }
        cutoff = Some(last);
        rows = cohort_rows(panel, &excluded)?;
    }
    let treated: BTreeSet<i64> = rows
        .iter()
        .filter(|&(g, &(pre, post))| pre && post && !named.contains(g))
        .map(|(&g, _)| g)
        .collect();
    let always: BTreeSet<i64> = rows
        .iter()
        .filter(|&(g, &(pre, post))| post && !pre && !named.contains(g))
        .map(|(&g, _)| g)
        .collect();
    for (i, x) in excluded.iter_mut().enumerate() {
        if panel.first_treated()[i].is_some_and(|g| always.contains(&g)) {
            *x = true;
        }
    }

    // The reference periods. `.F` and `.L` of `sunab` run over the relative
    // periods of every cohort whose first treated period is a period, with
    // -1 for the others, before the always treated are removed.
    let periods: BTreeSet<i64> = panel.period().iter().copied().collect();
    let mut first = i64::MAX;
    let mut last = i64::MIN;
    for i in 0..n {
        if cutoff.is_some_and(|c| panel.period()[i] >= c) {
            continue;
        }
        let e = match panel.first_treated()[i] {
            Some(g) if periods.contains(&g) => panel.period()[i] - g,
            _ => -1,
        };
        first = first.min(e);
        last = last.max(e);
    }
    let references: Vec<i64> = options
        .references
        .iter()
        .map(|r| match *r {
            SunabReference::Period(p) => p,
            SunabReference::First => first,
            SunabReference::Last => last,
        })
        .collect();

    // The cells and the cell of each row.
    let relative = |i: usize| -> Option<i64> {
        let g = panel.first_treated()[i]?;
        if excluded[i] || !treated.contains(&g) {
            return None;
        }
        let e = panel.period()[i] - g;
        (!references.contains(&e)).then(|| options.bins.label(e))
    };
    let cell_set: BTreeSet<(i64, i64)> = (0..n)
        .filter_map(|i| relative(i).map(|e| (e, panel.first_treated()[i].unwrap_or(0))))
        .collect();
    if cell_set.is_empty() {
        return Err(Error::NoTreatment);
    }
    let cells: Vec<(i64, i64)> = cell_set.into_iter().collect();
    let cohorts: Vec<i64> = treated.iter().copied().collect();
    let (e_min, e_max) = (cells[0].0, cells[cells.len() - 1].0);
    let width = e_max
        .checked_sub(e_min)
        .and_then(|d| d.checked_add(1))
        .and_then(|w| usize::try_from(w).ok())
        .ok_or(Error::Periods)?;
    let mut column_of = vec![u32::MAX; width * cohorts.len()];
    let slot = |e: i64, g: i64| -> usize {
        let a = usize::try_from(e - e_min).unwrap_or(usize::MAX);
        let b = cohorts.binary_search(&g).unwrap_or(usize::MAX);
        a * cohorts.len() + b
    };
    for (j, &(e, g)) in cells.iter().enumerate() {
        column_of[slot(e, g)] = u32::try_from(j).unwrap_or(u32::MAX);
    }
    let row_cell: Vec<u32> = (0..n)
        .map(|i| {
            relative(i).map_or(u32::MAX, |e| {
                column_of[slot(e, panel.first_treated()[i].unwrap_or(0))]
            })
        })
        .collect();

    let k_cells = cells.len();
    let p_cov = panel.covariates().map_or(0, |x| x.ncols());
    let mut x = Mat::<f64>::zeros(n, k_cells + p_cov);
    for (i, &c) in row_cell.iter().enumerate() {
        if c != u32::MAX {
            x[(i, c as usize)] = 1.0;
        }
    }
    if let Some(cov) = panel.covariates() {
        for j in 0..p_cov {
            let column = x.col_as_slice_mut(k_cells + j);
            for (i, v) in column.iter_mut().enumerate() {
                *v = cov[(i, j)];
            }
        }
    }
    let excluded_rows = excluded.iter().filter(|&&b| b).count();
    let fit_weights: Option<Vec<f64>> =
        (panel.weights().is_some() || excluded_rows > 0).then(|| {
            (0..n)
                .map(|i| {
                    if excluded[i] {
                        0.0
                    } else {
                        panel.weights().map_or(1.0, |w| w[i])
                    }
                })
                .collect()
        });

    let fe = FixedEffects::new(
        &[panel.panel_unit()?, &time],
        fit_weights.as_deref(),
        Singletons::Remove,
    )?;
    let fit = Feols {
        absorb: options.absorb,
    }
    .fit_owned(x, panel.outcome(), &fe)?;
    let clusters = panel.clusters().unwrap_or(panel.panel_unit()?);
    let ways = [clusters];
    let cov = fit.vcov(&fe, Vcov::Cluster(&ways), options.ssc)?;
    let p = k_cells + p_cov;
    let se = |j: usize| cov.matrix.as_slice()[j * p + j].map(|v| v.max(0.0).sqrt());
    let reference = Reference::fixest_linear(&cov, fe.n_rows());
    let coefficient = |j: usize| -> Option<Estimate> {
        fit.coefficients[j].map(|b| fe_output::estimate(b, se(j), options.level, reference))
    };

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
            estimate: coefficient(j),
            weight: cell_weight[j],
        })
        .collect();

    // Influence by unit.
    let unit_codes = fe.codes(0);
    let units = fe.labels(0).to_vec();
    let n_units = units.len();
    let sums = fit.influence_sums(&fe, unit_codes, n_units)?;
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
    let labels: BTreeSet<i64> = cells.iter().map(|c| c.0).collect();
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
                    match cov.matrix.as_slice()[i * p + j] {
                        Some(v) => total += si * sj * v,
                        None => defined = false,
                    }
                }
            }
            estimate_covariance[a * m + b] = defined.then_some(total);
        }
    }
    let aggregates: Vec<SunabAggregate> = groups
        .iter()
        .zip(&shares)
        .enumerate()
        .map(|(a, ((term, members), s))| SunabAggregate {
            term: *term,
            estimate: fe_output::estimate(
                s.iter()
                    .map(|&(j, w)| w * fit.coefficients[j].unwrap_or(0.0))
                    .sum(),
                estimate_covariance[a * m + a].map(|v| v.max(0.0).sqrt()),
                options.level,
                reference,
            ),
            n_cells: members.len(),
        })
        .collect();
    let aggregate_influence: Vec<Option<Vec<f64>>> = shares
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
            Some(out)
        })
        .collect();
    let rule = fe_output::fixest_rule(options.ssc.g_adj);
    let covariate_coefficients = (k_cells..p).map(coefficient).collect();

    Ok(SunabFit {
        cells: cell_results,
        aggregates,
        aggregate_covariance: Covariance::new(m, estimate_covariance)?,
        references: references
            .iter()
            .copied()
            .collect::<BTreeSet<i64>>()
            .into_iter()
            .collect(),
        aggregate_influence: fe_output::influence(aggregate_influence, units.clone(), rule)?,
        covariate_coefficients,
        covariance: cov.matrix,
        df_k: cov.df_k,
        n_clusters: cov.min_clusters.unwrap_or(0),
        influence: fe_output::influence(influence, units, rule)?,
        n_units,
        n_rows: fe.n_rows(),
        removed_singletons: fe.removed_singletons(),
        excluded_rows,
    })
}
