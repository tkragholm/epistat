//! The panel of staggered adoption that the two-way fixed-effects, imputation,
//! two-stage and Goodman-Bacon estimators take.

use epistat_linalg::MatRef;

/// A panel with staggered adoption, one entry per row.
///
/// Each unit is first treated in one period, or never, and stays treated
/// from then on: row `i` is treated when `time[i] >= first_treated[i]`.
#[derive(Clone, Copy, Debug)]
pub struct StaggeredPanel<'a> {
    /// The unit of each row, any numbers.
    pub unit: &'a [u32],
    /// The period of each row.
    pub time: &'a [i32],
    /// The first treated period of the row's unit, `None` for never treated;
    /// the same on every row of a unit.
    pub first_treated: &'a [Option<i32>],
    /// The outcome.
    pub outcome: &'a [f64],
    /// Prior weights, 1 when `None`. Rows of weight 0 do not enter the
    /// two-way fixed-effects regression, nor, when untreated, the first stage
    /// of the imputation and two-stage estimators. The Goodman-Bacon
    /// decomposition takes none.
    pub weights: Option<&'a [f64]>,
    /// Time-varying covariates, one row per row, or `None`: regressors of the
    /// two-way fixed-effects regression, of the first stage of the
    /// imputation and two-stage estimators, and the controls of the
    /// Goodman-Bacon decomposition.
    pub covariates: Option<MatRef<'a, f64>>,
}
