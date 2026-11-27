//! The panel of staggered adoption, the data view of the estimators that
//! follow units over periods.

use epistat_core::{InputError, Panel, check};
use epistat_linalg::MatRef;

use crate::error::Error;

/// A validated panel with staggered adoption, one entry per row.
///
/// It holds one row per unit and period: the unit as a `u32` code, the
/// period as an `i64`, the period the unit is first treated (`None` for never
/// treated), the outcome, and optionally prior weights, covariates, cluster
/// codes and, for matched designs, the cohort a never-treated unit was
/// matched to. Its key is [`epistat_core::Panel`]: the rows are sorted by
/// unit and then period, with no pair repeated, so [`Panel::sort_order`]
/// gives the order for unsorted input. Labels become codes with
/// [`epistat_core::Codes::from_labels`], whose sorted order keeps the order
/// of the units that of their labels.
///
/// [`StaggeredPanel::builder`] takes the four required columns, the builder's
/// methods add the optional ones, and [`StaggeredPanelBuilder::build`] checks
/// every column once:
///
/// | column | check |
/// |---|---|
/// | `unit`, `period` | the same length, at least one row, sorted by unit then period, no repeated pair |
/// | every other column | one entry per row |
/// | `outcome` | finite |
/// | `weights` | finite and not negative; a zero weight is allowed and its meaning documented by each estimator |
/// | `covariates` | finite |
/// | `first_treated`, `clusters`, `comparison_cohort` | constant within a unit |
///
/// Each unit is first treated in one period, or never, and stays treated
/// from then on: row `i` is treated when `period[i] >= first_treated[i]`.
/// The estimators take the validated panel and do not repeat these checks.
///
/// ```
/// use epistat_core::Codes;
/// use epistat_did::StaggeredPanel;
///
/// // Two counties by their numbers, observed in 2004 and 2005; the first
/// // is treated from 2005.
/// let county = [1003_i64, 1003, 1001, 1001];
/// let year = [2004, 2005, 2004, 2005];
/// let first_treated = [Some(2005), Some(2005), None, None];
/// let y = [1.0, 2.0, 1.5, 1.6];
/// let state = [7, 7, 3, 3];
///
/// // Codes in sorted label order: 1001 is unit 0 and 1003 unit 1, so the
/// // rows of unit 0 come first.
/// let units = Codes::from_labels(&county)?;
/// let order = epistat_core::Panel::sort_order(units.codes(), &year)?;
/// let pick = |i: &u32| *i as usize;
/// let unit: Vec<u32> = order.iter().map(|i| units.codes()[pick(i)]).collect();
/// let period: Vec<i64> = order.iter().map(|i| year[pick(i)]).collect();
/// let cohort: Vec<Option<i64>> = order.iter().map(|i| first_treated[pick(i)]).collect();
/// let outcome: Vec<f64> = order.iter().map(|i| y[pick(i)]).collect();
/// let cluster: Vec<u32> = order.iter().map(|i| state[pick(i)]).collect();
///
/// let panel = StaggeredPanel::builder(&unit, &period, &cohort, &outcome)
///     .clusters(&cluster)
///     .build()?;
/// assert_eq!(panel.n_units(), 2);
/// assert_eq!(panel.first_treated()[0], None);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct StaggeredPanel<'a> {
    key: Panel<'a>,
    first_treated: &'a [Option<i64>],
    outcome: &'a [f64],
    weights: Option<&'a [f64]>,
    covariates: Option<MatRef<'a, f64>>,
    clusters: Option<&'a [u32]>,
    comparison_cohort: Option<&'a [Option<i64>]>,
}

/// The builder of a [`StaggeredPanel`].
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct StaggeredPanelBuilder<'a> {
    unit: &'a [u32],
    period: &'a [i64],
    first_treated: &'a [Option<i64>],
    outcome: &'a [f64],
    weights: Option<&'a [f64]>,
    covariates: Option<MatRef<'a, f64>>,
    clusters: Option<&'a [u32]>,
    comparison_cohort: Option<&'a [Option<i64>]>,
}

impl<'a> StaggeredPanel<'a> {
    /// The panel of the unit codes `unit`, the periods `period`, the first
    /// treated period of each row's unit (`None` for never treated) and the
    /// outcome.
    pub const fn builder(
        unit: &'a [u32],
        period: &'a [i64],
        first_treated: &'a [Option<i64>],
        outcome: &'a [f64],
    ) -> StaggeredPanelBuilder<'a> {
        StaggeredPanelBuilder {
            unit,
            period,
            first_treated,
            outcome,
            weights: None,
            covariates: None,
            clusters: None,
            comparison_cohort: None,
        }
    }

    /// The key: the unit code and period of each row.
    #[must_use]
    pub const fn key(&self) -> Panel<'a> {
        self.key
    }

    /// The unit code of each row.
    #[must_use]
    pub const fn unit(&self) -> &'a [u32] {
        self.key.unit()
    }

    /// The period of each row.
    #[must_use]
    pub const fn period(&self) -> &'a [i64] {
        self.key.period()
    }

    /// The first treated period of each row's unit, `None` for never
    /// treated.
    #[must_use]
    pub const fn first_treated(&self) -> &'a [Option<i64>] {
        self.first_treated
    }

    /// The outcome of each row.
    #[must_use]
    pub const fn outcome(&self) -> &'a [f64] {
        self.outcome
    }

    /// The prior weight of each row, `None` when every row weighs 1.
    #[must_use]
    pub const fn weights(&self) -> Option<&'a [f64]> {
        self.weights
    }

    /// The covariates, one row per row, `None` without covariates.
    #[must_use]
    pub const fn covariates(&self) -> Option<MatRef<'a, f64>> {
        self.covariates
    }

    /// The cluster code of each row, `None` when each unit is its own
    /// cluster.
    #[must_use]
    pub const fn clusters(&self) -> Option<&'a [u32]> {
        self.clusters
    }

    /// For each row of a never-treated unit, the cohort the unit was matched
    /// to as a comparator, `None` for a unit matched to none; `None` for the
    /// whole column when the design is not matched.
    #[must_use]
    pub const fn comparison_cohort(&self) -> Option<&'a [Option<i64>]> {
        self.comparison_cohort
    }

    /// The number of rows.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.key.len()
    }

    /// Whether the panel has no rows, which the builder refuses, so always
    /// false.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.key.is_empty()
    }

    /// The number of distinct units.
    #[must_use]
    pub fn n_units(&self) -> usize {
        self.key.n_units()
    }

    /// The periods as fixed-effect codes `period - min(period)`, `None` when
    /// they span more than 2^32 values.
    pub(crate) fn period_codes(&self) -> Option<Vec<u32>> {
        let period = self.period();
        let t_min = period.iter().copied().min().unwrap_or(0);
        period
            .iter()
            .map(|&t| u32::try_from(t.checked_sub(t_min)?).ok())
            .collect()
    }
}

impl<'a> StaggeredPanelBuilder<'a> {
    /// Prior weights, one per row.
    pub const fn weights(mut self, weights: &'a [f64]) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Covariates, one row per row.
    pub const fn covariates(mut self, covariates: MatRef<'a, f64>) -> Self {
        self.covariates = Some(covariates);
        self
    }

    /// Cluster codes, one per row and constant within a unit.
    pub const fn clusters(mut self, clusters: &'a [u32]) -> Self {
        self.clusters = Some(clusters);
        self
    }

    /// The cohort each never-treated unit was matched to as a comparator,
    /// one entry per row and constant within a unit.
    pub const fn comparison_cohort(mut self, comparison_cohort: &'a [Option<i64>]) -> Self {
        self.comparison_cohort = Some(comparison_cohort);
        self
    }

    /// The validated panel.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] for the first check of [`StaggeredPanel`] that
    /// fails: [`InputError::Length`], [`InputError::Empty`],
    /// [`InputError::Unsorted`] or [`InputError::Duplicate`] for the key,
    /// [`InputError::Length`] for another column of the wrong length,
    /// [`InputError::NotFinite`] for an outcome, weight or covariate that is
    /// not finite, [`InputError::Domain`] for a negative weight and
    /// [`InputError::NotConstant`] for a column that changes within a unit.
    pub fn build(self) -> Result<StaggeredPanel<'a>, Error> {
        let key = Panel::new(self.unit, self.period)?;
        let n = key.len();
        check::same_length("first_treated", n, self.first_treated.len())?;
        check::same_length("outcome", n, self.outcome.len())?;
        check::finite("outcome", self.outcome)?;
        if let Some(weights) = self.weights {
            check::same_length("weights", n, weights.len())?;
            check::finite("weights", weights)?;
            if let Some(&value) = weights.iter().find(|&&w| w < 0.0) {
                return Err(InputError::Domain {
                    what: "weights",
                    value,
                    domain: "[0, inf)",
                    row: None,
                }
                .into());
            }
        }
        if let Some(x) = self.covariates {
            check::same_length("covariates", n, x.nrows())?;
            for j in 0..x.ncols() {
                if let Some(row) = (0..n).find(|&i| !x[(i, j)].is_finite()) {
                    return Err(InputError::NotFinite {
                        what: "covariates",
                        row,
                        column: None,
                    }
                    .into());
                }
            }
        }
        constant_by_unit(key, "first_treated", self.first_treated)?;
        if let Some(clusters) = self.clusters {
            check::same_length("clusters", n, clusters.len())?;
            constant_by_unit(key, "clusters", clusters)?;
        }
        if let Some(cohort) = self.comparison_cohort {
            check::same_length("comparison_cohort", n, cohort.len())?;
            constant_by_unit(key, "comparison_cohort", cohort)?;
        }
        Ok(StaggeredPanel {
            key,
            first_treated: self.first_treated,
            outcome: self.outcome,
            weights: self.weights,
            covariates: self.covariates,
            clusters: self.clusters,
            comparison_cohort: self.comparison_cohort,
        })
    }
}

/// `value` is the same on every row of a unit. The rows of a unit are
/// adjacent in a validated key, so each run is compared with its first row.
fn constant_by_unit<T: PartialEq>(
    key: Panel<'_>,
    what: &'static str,
    value: &[T],
) -> Result<(), InputError> {
    for (unit, rows) in key.units() {
        let first = &value[rows.start];
        if value[rows].iter().any(|v| v != first) {
            return Err(InputError::NotConstant { what, unit });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use epistat_core::Classify;
    use epistat_linalg::Mat;

    const UNIT: [u32; 6] = [0, 0, 0, 2, 2, 2];
    const PERIOD: [i64; 6] = [2000, 2001, 2002, 2000, 2001, 2002];
    const COHORT: [Option<i64>; 6] = [Some(2001), Some(2001), Some(2001), None, None, None];
    const Y: [f64; 6] = [1.0, 2.0, 3.0, 1.5, 1.5, 2.5];

    fn input(result: Result<StaggeredPanel<'_>, Error>) -> InputError {
        match result {
            Err(Error::Input(error)) => error,
            other => panic!("expected an input error, got {other:?}"),
        }
    }

    #[test]
    fn a_valid_panel_keeps_its_columns() {
        let w = [1.0, 1.0, 1.0, 0.0, 2.0, 2.0];
        let x = Mat::from_fn(6, 1, |i, _| f64::from(u32::try_from(i).unwrap()));
        let clusters = [7, 7, 7, 3, 3, 3];
        let matched = [None, None, None, Some(2001), Some(2001), Some(2001)];
        let panel = StaggeredPanel::builder(&UNIT, &PERIOD, &COHORT, &Y)
            .weights(&w)
            .covariates(x.as_ref())
            .clusters(&clusters)
            .comparison_cohort(&matched)
            .build()
            .unwrap();
        assert_eq!(panel.len(), 6);
        assert!(!panel.is_empty());
        assert_eq!(panel.n_units(), 2);
        assert_eq!(panel.unit(), &UNIT);
        assert_eq!(panel.period(), &PERIOD);
        assert_eq!(panel.first_treated(), &COHORT);
        assert_eq!(panel.outcome(), &Y);
        assert_eq!(panel.weights(), Some(&w[..]));
        assert_eq!(panel.covariates().map(|x| x.ncols()), Some(1));
        assert_eq!(panel.clusters(), Some(&clusters[..]));
        assert_eq!(panel.comparison_cohort(), Some(&matched[..]));
    }

    #[test]
    fn the_key_must_be_sorted_and_unique() {
        let period = [2001, 2000, 2002, 2000, 2001, 2002];
        let error = input(StaggeredPanel::builder(&UNIT, &period, &COHORT, &Y).build());
        assert_eq!(
            error,
            InputError::Unsorted {
                what: "panel",
                row: 1
            }
        );
        let period = [2000, 2000, 2002, 2000, 2001, 2002];
        let error = input(StaggeredPanel::builder(&UNIT, &period, &COHORT, &Y).build());
        assert_eq!(
            error,
            InputError::Duplicate {
                what: "panel",
                row: 1
            }
        );
        let error = input(StaggeredPanel::builder(&[], &[], &[], &[]).build());
        assert_eq!(error.code(), "input.empty");
    }

    #[test]
    fn columns_of_the_wrong_length_are_refused() {
        let error = input(StaggeredPanel::builder(&UNIT, &PERIOD, &COHORT[..5], &Y).build());
        assert_eq!(
            error,
            InputError::Length {
                what: "first_treated",
                expected: 6,
                found: 5
            }
        );
        let error = input(
            StaggeredPanel::builder(&UNIT, &PERIOD, &COHORT, &Y)
                .clusters(&[0, 0])
                .build(),
        );
        assert_eq!(error.code(), "input.length");
        let x = Mat::<f64>::zeros(5, 1);
        let error = input(
            StaggeredPanel::builder(&UNIT, &PERIOD, &COHORT, &Y)
                .covariates(x.as_ref())
                .build(),
        );
        assert_eq!(error.code(), "input.length");
    }

    #[test]
    fn values_outside_their_domain_are_refused() {
        let mut y = Y;
        y[4] = f64::NAN;
        let error = input(StaggeredPanel::builder(&UNIT, &PERIOD, &COHORT, &y).build());
        assert_eq!(
            error,
            InputError::NotFinite {
                what: "outcome",
                row: 4,
                column: None
            }
        );
        let w = [1.0, 1.0, -1.0, 1.0, 1.0, 1.0];
        let error = input(
            StaggeredPanel::builder(&UNIT, &PERIOD, &COHORT, &Y)
                .weights(&w)
                .build(),
        );
        assert_eq!(error.code(), "input.domain");
        let x = Mat::from_fn(
            6,
            2,
            |i, j| if i == 3 && j == 1 { f64::INFINITY } else { 0.0 },
        );
        let error = input(
            StaggeredPanel::builder(&UNIT, &PERIOD, &COHORT, &Y)
                .covariates(x.as_ref())
                .build(),
        );
        assert_eq!(
            error,
            InputError::NotFinite {
                what: "covariates",
                row: 3,
                column: None
            }
        );
    }

    #[test]
    fn bad_columns_are_input_errors() {
        use epistat_testkit::{BadInput, assert_rejects_bad_input};
        let reversed: Vec<i64> = PERIOD.iter().rev().copied().collect();
        assert_rejects_bad_input("StaggeredPanel::build", |bad| {
            let y = bad.column(&Y);
            let built = match bad {
                BadInput::Unsorted => {
                    StaggeredPanel::builder(&UNIT, &reversed, &COHORT, &Y).build()
                }
                BadInput::Empty => StaggeredPanel::builder(&[], &[], &[], &y).build(),
                _ => StaggeredPanel::builder(&UNIT, &PERIOD, &COHORT, &y).build(),
            };
            Some(built.map(|panel| panel.len()))
        });
    }

    #[test]
    fn unit_level_columns_must_be_constant_within_a_unit() {
        let cohort = [Some(2001), Some(2002), Some(2001), None, None, None];
        let error = input(StaggeredPanel::builder(&UNIT, &PERIOD, &cohort, &Y).build());
        assert_eq!(
            error,
            InputError::NotConstant {
                what: "first_treated",
                unit: 0
            }
        );
        let clusters = [1, 1, 1, 4, 5, 4];
        let error = input(
            StaggeredPanel::builder(&UNIT, &PERIOD, &COHORT, &Y)
                .clusters(&clusters)
                .build(),
        );
        assert_eq!(
            error,
            InputError::NotConstant {
                what: "clusters",
                unit: 2
            }
        );
    }
}
