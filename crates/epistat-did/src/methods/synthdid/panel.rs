//! The panel matrices of `synthdid` from a [`StaggeredPanel`], as
//! `panel.matrices` forms them from long data.

use epistat_inference::Clusters;

use crate::error::Error;
use crate::methods::staggered::StaggeredPanel;

use super::fit::Data;

/// A balanced panel as the fits read it: the outcomes by unit, in increasing
/// unit code, over the periods in increasing order, the covariates as blocks
/// of the same layout, the adoption column of each unit and the clusters of
/// the units.
pub(crate) struct Grid {
    outcome: Vec<f64>,
    covariates: Vec<f64>,
    count: usize,
    pub units: Vec<u32>,
    pub periods: Vec<i64>,
    pub adoption: Vec<Option<usize>>,
    pub clusters: Option<Clusters>,
}

impl Grid {
    /// The grid of `panel`.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] for prior weights or comparison cohorts,
    /// [`Error::Unbalanced`] unless every unit is observed in the same
    /// periods, [`Error::Adoption`] for a unit first treated in the first
    /// period or in a period outside the panel, [`Error::NoNeverTreated`]
    /// without a never-treated unit and [`Error::NoTreatment`] without a
    /// treated one.
    pub fn new(panel: &StaggeredPanel<'_>) -> Result<Self, Error> {
        if panel.weights().is_some() {
            return Err(Error::Unsupported {
                what: "prior weights in synthdid",
            });
        }
        if panel.comparison_cohort().is_some() {
            return Err(Error::Unsupported {
                what: "comparison cohorts in synthdid",
            });
        }
        let key = panel.key();
        let Some((_, first)) = key.units().next() else {
            return Err(Error::NoTreatment);
        };
        let periods: Vec<i64> = panel.period()[first].to_vec();
        let t = periods.len();
        let mut units = Vec::new();
        let mut adoption = Vec::new();
        let mut starts = Vec::new();
        for (unit, rows) in key.units() {
            if panel.period()[rows.clone()] != periods[..] {
                return Err(Error::Unbalanced);
            }
            let column = match panel.first_treated()[rows.start] {
                None => None,
                Some(g) => match periods.binary_search(&g) {
                    Ok(a) if a > 0 => Some(a),
                    _ => return Err(Error::Adoption { unit }),
                },
            };
            units.push(unit);
            adoption.push(column);
            starts.push(rows.start);
        }
        let never = adoption.iter().filter(|a| a.is_none()).count();
        if never == 0 {
            return Err(Error::NoNeverTreated);
        }
        if never == units.len() {
            return Err(Error::NoTreatment);
        }
        let n = units.len();
        let outcome: Vec<f64> = starts
            .iter()
            .flat_map(|&s| panel.outcome()[s..s + t].iter().copied())
            .collect();
        let (covariates, count) = panel.covariates().map_or((Vec::new(), 0), |x| {
            let count = x.ncols();
            let mut blocks = Vec::with_capacity(count * n * t);
            for k in 0..count {
                for &s in &starts {
                    blocks.extend((s..s + t).map(|i| x[(i, k)]));
                }
            }
            (blocks, count)
        });
        let clusters = panel.clusters().map(|c| {
            let codes: Vec<u32> = starts.iter().map(|&s| c[s]).collect();
            Clusters::from_codes(&codes)
        });
        Ok(Self {
            outcome,
            covariates,
            count,
            units,
            periods,
            adoption,
            clusters,
        })
    }

    /// The layout the fits read.
    pub fn data(&self) -> Data<'_> {
        Data {
            outcome: &self.outcome,
            covariates: &self.covariates,
            count: self.count,
            units: self.units.len(),
            periods: self.periods.len(),
        }
    }

    /// The positions of the never-treated units.
    pub fn controls(&self) -> Vec<usize> {
        (0..self.units.len())
            .filter(|&u| self.adoption[u].is_none())
            .collect()
    }

    /// The adoption columns with the positions of their units, in increasing
    /// order.
    pub fn cohorts(&self) -> Vec<(usize, Vec<usize>)> {
        let mut columns: Vec<usize> = self.adoption.iter().flatten().copied().collect();
        columns.sort_unstable();
        columns.dedup();
        columns
            .into_iter()
            .map(|a| {
                let units = (0..self.units.len())
                    .filter(|&u| self.adoption[u] == Some(a))
                    .collect();
                (a, units)
            })
            .collect()
    }

    /// The unit codes at `positions`.
    pub fn codes(&self, positions: &[usize]) -> Vec<u32> {
        positions.iter().map(|&u| self.units[u]).collect()
    }
}
