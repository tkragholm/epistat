//! The ATT(g,t) fixtures of the parity tests are row records, as R's data
//! frames hold them; this turns them into a `StaggeredPanel` and reads the
//! fitted cells back in the shape the comparisons use.

use epistat_did::att_gt::{self, AttGt, AttGtOptions, Design};
use epistat_did::{Error, StaggeredPanel};
use epistat_linalg::MatRef;

/// One row of a fixture.
#[derive(Debug, Clone, PartialEq)]
pub struct Obs {
    /// The unit, read on a panel.
    pub unit: Option<i64>,
    /// The cohort, `None` for never treated.
    pub cohort: Option<i32>,
    pub time: i32,
    pub outcome: f64,
    pub weight: f64,
    pub covariates: Vec<f64>,
    /// The cohort a never-treated unit was matched to.
    pub matched: Option<i32>,
    /// The cluster of the row's unit.
    pub cluster: Option<u32>,
}

impl Obs {
    /// A row of `unit` with weight 1 and no covariates.
    pub const fn new(unit: Option<i64>, cohort: Option<i32>, time: i32, outcome: f64) -> Self {
        Self {
            unit,
            cohort,
            time,
            outcome,
            weight: 1.0,
            covariates: Vec::new(),
            matched: None,
            cluster: None,
        }
    }
}

/// The fit of `rows` as a panel, the rows sorted by unit and period.
///
/// # Panics
///
/// On a row without a unit, or a unit that is not a `u32`.
pub fn fit(rows: &[Obs], options: &AttGtOptions) -> Result<AttGt, Error> {
    fit_design(rows, Design::Panel, options)
}

/// The fit of `rows` under `design`: on a panel the rows sorted by unit and
/// period, on repeated cross-sections each row its own observation in the
/// order given.
///
/// # Panics
///
/// On a panel row without a unit, or a unit that is not a `u32`.
pub fn fit_design(rows: &[Obs], design: Design, options: &AttGtOptions) -> Result<AttGt, Error> {
    let mut order: Vec<usize> = (0..rows.len()).collect();
    let panel = design == Design::Panel;
    if panel {
        order.sort_by_key(|&i| (rows[i].unit.expect("a panel unit"), rows[i].time));
    }
    let unit: Vec<u32> = order
        .iter()
        .enumerate()
        .map(|(k, &i)| {
            let code = if panel {
                rows[i].unit.expect("a panel unit")
            } else {
                i64::try_from(k).expect("rows")
            };
            u32::try_from(code).expect("a unit code")
        })
        .collect();
    let period: Vec<i64> = order.iter().map(|&i| i64::from(rows[i].time)).collect();
    let cohort: Vec<Option<i64>> = order
        .iter()
        .map(|&i| rows[i].cohort.map(i64::from))
        .collect();
    let outcome: Vec<f64> = order.iter().map(|&i| rows[i].outcome).collect();
    let weight: Vec<f64> = order.iter().map(|&i| rows[i].weight).collect();
    let k = rows.first().map_or(0, |row| row.covariates.len());
    let covariates: Vec<f64> = (0..k)
        .flat_map(|j| order.iter().map(move |&i| rows[i].covariates[j]))
        .collect();
    let matched: Vec<Option<i64>> = order
        .iter()
        .map(|&i| rows[i].matched.map(i64::from))
        .collect();
    let cluster: Vec<u32> = order
        .iter()
        .map(|&i| rows[i].cluster.unwrap_or(0))
        .collect();

    let builder = if panel {
        StaggeredPanel::builder(&unit, &period, &cohort, &outcome)
    } else {
        StaggeredPanel::cross_sections(&period, &cohort, &outcome)
    };
    let builder = builder.weights(epistat_core::CaseWeights::new(&weight).expect("weights"));
    let builder = if k > 0 {
        builder.covariates(MatRef::from_column_major_slice(&covariates, rows.len(), k))
    } else {
        builder
    };
    let builder = if rows.iter().any(|row| row.matched.is_some()) {
        builder.comparison_cohort(&matched)
    } else {
        builder
    };
    let builder = if rows.iter().any(|row| row.cluster.is_some()) {
        builder.clusters(&cluster)
    } else {
        builder
    };
    att_gt::estimate(&builder.build()?, options)
}

/// An estimated cell, in the fields the comparisons read.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cell {
    pub group: i32,
    pub time: i32,
    pub event_time: i32,
    pub att: f64,
    pub se: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    pub treated_n: usize,
    pub control_n: usize,
    pub total_weight: f64,
}

fn small(value: i64) -> i32 {
    i32::try_from(value).expect("a period")
}

/// The estimated cells in order, with NaN for an undefined part.
pub fn cells(fit: &AttGt) -> Vec<Cell> {
    fit.cells()
        .iter()
        .filter_map(|cell| {
            let estimate = cell.estimate?;
            Some(Cell {
                group: small(cell.group),
                time: small(cell.time),
                event_time: small(cell.event_time),
                att: estimate.estimate,
                se: estimate.se.unwrap_or(f64::NAN),
                ci_low: estimate.lower().unwrap_or(f64::NAN),
                ci_high: estimate.upper().unwrap_or(f64::NAN),
                treated_n: cell.n_treated,
                control_n: cell.n_control,
                total_weight: cell.total_weight,
            })
        })
        .collect()
}

/// The influence function of each estimated cell.
pub fn influence(fit: &AttGt) -> Vec<Vec<f64>> {
    let influence = fit.influence().expect("influence");
    (0..influence.n_columns())
        .map(|j| influence.column(j).to_vec())
        .collect()
}

/// The units of the influence function as `i64`.
pub fn ids(fit: &AttGt) -> Vec<i64> {
    fit.sample().units.iter().map(|&u| i64::from(u)).collect()
}

/// A point of a summary, in the fields the comparisons read.
#[derive(Debug, Clone, PartialEq)]
pub struct Point {
    pub key: i32,
    pub att: f64,
    pub se: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    pub influence: Vec<f64>,
    pub components: usize,
    pub total_weight: f64,
}

/// A summary, with NaN for an undefined part and for a missing overall
/// effect.
#[derive(Debug, Clone, PartialEq)]
pub struct Summary {
    pub overall_att: f64,
    pub overall_se: f64,
    pub overall_ci_low: f64,
    pub overall_ci_high: f64,
    pub overall_influence: Vec<f64>,
    pub overall_components: usize,
    pub by_key: Vec<Point>,
    pub critical_value: f64,
    pub uniform_band: bool,
}

/// The summary `result` in the shape of the comparisons.
pub fn summary(result: &epistat_did::att_gt::Aggte) -> Summary {
    let influence = result.influence();
    let nan = |value: Option<f64>| value.unwrap_or(f64::NAN);
    let overall = result.overall;
    Summary {
        overall_att: overall.map_or(f64::NAN, |o| o.estimate.estimate),
        overall_se: nan(overall.and_then(|o| o.estimate.se)),
        overall_ci_low: nan(overall.and_then(|o| o.estimate.lower())),
        overall_ci_high: nan(overall.and_then(|o| o.estimate.upper())),
        overall_influence: if overall.is_some() {
            influence.column(result.points.len()).to_vec()
        } else {
            Vec::new()
        },
        overall_components: overall.map_or(0, |o| o.n_cells),
        by_key: result
            .points
            .iter()
            .enumerate()
            .map(|(j, point)| Point {
                key: small(point.key),
                att: point.estimate.estimate,
                se: nan(point.estimate.se),
                ci_low: nan(point.estimate.lower()),
                ci_high: nan(point.estimate.upper()),
                influence: influence.column(j).to_vec(),
                components: point.n_cells,
                total_weight: point.total_weight,
            })
            .collect(),
        critical_value: result.critical_value,
        uniform_band: result.uniform_band,
    }
}
