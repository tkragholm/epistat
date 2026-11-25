//! Long data to the panel matrices, as `synthdid`'s `panel.matrices`.

use std::collections::BTreeMap;

use super::{StaggeredSynthPanel, SynthDidError, SynthPanel};

/// One row of a long panel: a unit in a period.
#[derive(Debug, Clone, PartialEq)]
pub struct SynthObservation {
    /// The unit.
    pub unit: i64,
    /// The period.
    pub time: i32,
    /// The outcome.
    pub outcome: f64,
    /// Whether the unit is treated in this period.
    pub treated: bool,
    /// Time-varying covariates, the same number on every row.
    pub covariates: Vec<f64>,
}

/// The grid of a balanced panel, units and times in increasing order.
struct Grid {
    units: Vec<i64>,
    times: Vec<i32>,
    outcome: Vec<f64>,
    covariates: Vec<f64>,
    count: usize,
    treated: Vec<bool>,
}

fn grid(observations: &[SynthObservation]) -> Result<Grid, SynthDidError> {
    let first = observations
        .first()
        .ok_or_else(|| SynthDidError::Panel("no observations".into()))?;
    let count = first.covariates.len();
    let units: Vec<i64> = {
        let mut u: Vec<i64> = observations.iter().map(|o| o.unit).collect();
        u.sort_unstable();
        u.dedup();
        u
    };
    let times: Vec<i32> = {
        let mut t: Vec<i32> = observations.iter().map(|o| o.time).collect();
        t.sort_unstable();
        t.dedup();
        t
    };
    let unit_index: BTreeMap<i64, usize> = units.iter().enumerate().map(|(i, &u)| (u, i)).collect();
    let time_index: BTreeMap<i32, usize> = times.iter().enumerate().map(|(i, &t)| (t, i)).collect();
    let (n, periods) = (units.len(), times.len());
    if observations.len() != n * periods {
        return Err(SynthDidError::Panel(
            "the panel must be balanced: one observation per unit and period".into(),
        ));
    }
    let mut seen = vec![false; n * periods];
    let mut outcome = vec![0.0; n * periods];
    let mut covariates = vec![0.0; count * n * periods];
    let mut treated = vec![false; n * periods];
    for o in observations {
        if o.covariates.len() != count {
            return Err(SynthDidError::Dimension(
                "every row needs the same covariates",
            ));
        }
        let cell = unit_index[&o.unit] * periods + time_index[&o.time];
        if std::mem::replace(&mut seen[cell], true) {
            return Err(SynthDidError::Panel(format!(
                "unit {} has two observations in period {}",
                o.unit, o.time
            )));
        }
        if !o.outcome.is_finite() || o.covariates.iter().any(|v| !v.is_finite()) {
            return Err(SynthDidError::NotFinite);
        }
        outcome[cell] = o.outcome;
        treated[cell] = o.treated;
        for (k, &v) in o.covariates.iter().enumerate() {
            covariates[k * n * periods + cell] = v;
        }
    }
    Ok(Grid {
        units,
        times,
        outcome,
        covariates,
        count,
        treated,
    })
}

/// The rows of `grid` in the order `order`, for the outcome and each
/// covariate block.
fn reorder(grid: &Grid, order: &[usize]) -> (Vec<f64>, Vec<f64>) {
    let periods = grid.times.len();
    let n = grid.units.len();
    let mut outcome = Vec::with_capacity(grid.outcome.len());
    for &i in order {
        outcome.extend_from_slice(&grid.outcome[i * periods..(i + 1) * periods]);
    }
    let mut covariates = Vec::with_capacity(grid.covariates.len());
    for k in 0..grid.count {
        for &i in order {
            let start = (k * n + i) * periods;
            covariates.extend_from_slice(&grid.covariates[start..start + periods]);
        }
    }
    (outcome, covariates)
}

impl SynthPanel {
    /// The panel of a long data set with one adoption date, as
    /// `panel.matrices`: balanced, treatment absent in every unit before the
    /// first treated period and present in every treated unit after it. The
    /// controls come first, then the treated units, each by increasing
    /// unit identifier; [`SynthPanel::unit_ids`] gives the order.
    ///
    /// # Errors
    ///
    /// [`SynthDidError::Panel`] when the panel is not balanced, has no
    /// variation in treatment, or adoption is not simultaneous, and the errors
    /// of [`SynthPanel::new`].
    pub fn from_observations(observations: &[SynthObservation]) -> Result<Self, SynthDidError> {
        let grid = grid(observations)?;
        let (n, periods) = (grid.units.len(), grid.times.len());
        let row = |i: usize| &grid.treated[i * periods..(i + 1) * periods];
        let ever: Vec<bool> = (0..n).map(|i| row(i).iter().any(|&w| w)).collect();
        let first = (0..periods)
            .find(|&t| (0..n).any(|i| row(i)[t]))
            .ok_or_else(|| SynthDidError::Panel("there is no variation in treatment".into()))?;
        if ever.iter().all(|&e| e) {
            return Err(SynthDidError::Panel(
                "there is no variation in treatment".into(),
            ));
        }
        let simultaneous = (0..n).all(|i| {
            row(i)
                .iter()
                .enumerate()
                .all(|(t, &w)| w == (ever[i] && t >= first))
        });
        if !simultaneous {
            return Err(SynthDidError::Panel(
                "treatment adoption is not simultaneous".into(),
            ));
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| (ever[i], grid.units[i]));
        let (outcome, covariates) = reorder(&grid, &order);
        let controls = ever.iter().filter(|&&e| !e).count();
        let mut panel = Self::new(outcome, n, periods, controls, first)?;
        if grid.count > 0 {
            panel = panel.with_covariates(covariates, grid.count)?;
        }
        panel.unit_ids = order.iter().map(|&i| grid.units[i]).collect();
        panel.times = grid.times;
        Ok(panel)
    }
}

impl StaggeredSynthPanel {
    /// The panel of a long data set with absorbing treatment: each unit is
    /// untreated up to its adoption period and treated from then on, or never
    /// treated. Units keep the order of increasing identifier.
    ///
    /// # Errors
    ///
    /// [`SynthDidError::Panel`] when the panel is not balanced, a unit's
    /// treatment switches off, or a unit is treated in the first period, and
    /// the errors of [`StaggeredSynthPanel::new`].
    pub fn from_observations(observations: &[SynthObservation]) -> Result<Self, SynthDidError> {
        let grid = grid(observations)?;
        let (n, periods) = (grid.units.len(), grid.times.len());
        let mut adoption = Vec::with_capacity(n);
        for i in 0..n {
            let row = &grid.treated[i * periods..(i + 1) * periods];
            let first = row.iter().position(|&w| w);
            if let Some(a) = first {
                if row[a..].iter().any(|&w| !w) {
                    return Err(SynthDidError::Panel(format!(
                        "the treatment of unit {} is not absorbing",
                        grid.units[i]
                    )));
                }
                if a == 0 {
                    return Err(SynthDidError::Panel(format!(
                        "unit {} is treated in the first period",
                        grid.units[i]
                    )));
                }
            }
            adoption.push(first);
        }
        let order: Vec<usize> = (0..n).collect();
        let (outcome, covariates) = reorder(&grid, &order);
        let mut panel = Self::new(outcome, n, periods, adoption)?;
        if grid.count > 0 {
            panel = panel.with_covariates(covariates, grid.count)?;
        }
        panel.unit_ids = grid.units;
        panel.times = grid.times;
        Ok(panel)
    }
}
