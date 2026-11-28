//! The columns of a [`StaggeredPanel`] as the ATT(g,t) routes read them, one
//! row at a time.

use epistat_linalg::MatRef;

use crate::methods::staggered::StaggeredPanel;

/// The rows of a validated panel.
#[derive(Clone, Copy)]
pub(super) struct Rows<'a> {
    pub unit: &'a [u32],
    pub period: &'a [i64],
    pub cohort: &'a [Option<i64>],
    pub outcome: &'a [f64],
    weights: Option<&'a [f64]>,
    covariates: Option<MatRef<'a, f64>>,
    comparison_cohort: Option<&'a [Option<i64>]>,
}

impl<'a> Rows<'a> {
    /// The rows of `panel`; repeated cross-sections have no unit codes, and
    /// their routes do not read them.
    pub(super) fn new(panel: &'a StaggeredPanel<'_>) -> Self {
        Self {
            unit: panel.unit().unwrap_or_default(),
            period: panel.period(),
            cohort: panel.first_treated(),
            outcome: panel.outcome(),
            weights: panel.weights(),
            covariates: panel.covariates(),
            comparison_cohort: panel.comparison_cohort(),
        }
    }

    pub(super) const fn len(&self) -> usize {
        self.period.len()
    }

    /// The weight of row `i`, 1 without weights.
    pub(super) fn weight(&self, i: usize) -> f64 {
        self.weights.map_or(1.0, |weights| weights[i])
    }

    /// The number of covariates, 0 without covariates.
    pub(super) fn n_covariates(&self) -> usize {
        self.covariates.map_or(0, |x| x.ncols())
    }

    /// Appends the covariates of row `i`, in column order.
    pub(super) fn extend_covariates(&self, i: usize, out: &mut Vec<f64>) {
        if let Some(x) = self.covariates {
            out.extend((0..x.ncols()).map(|j| x[(i, j)]));
        }
    }

    /// The covariates of row `i`, in column order.
    pub(super) fn covariates_of(&self, i: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n_covariates());
        self.extend_covariates(i, &mut out);
        out
    }

    /// The cohort the never-treated unit of row `i` was matched to, `None`
    /// without the column.
    pub(super) fn comparison_cohort(&self, i: usize) -> Option<i64> {
        self.comparison_cohort.and_then(|cohort| cohort[i])
    }
}
